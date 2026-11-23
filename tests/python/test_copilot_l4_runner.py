"""Tests for Colab L4 Qwen3-4B BF16 LoRA SFT -> Z3-GRPO runner, config flags, and live eval artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vertibench_tr_en.validate_vertibench import (
    MAX_JACCARD_THRESHOLD,
    _load_jsonl,
    parse_compact_dsl_py,
    validate_hv_telemetry,
)
from ml.copilot import train_sft_grpo
from ml.copilot.grpo_z3_rewards import score_single_completion
from ml.copilot.train_sft_grpo import (
    GrpoPolicyConfig,
    build_training_plan_summary,
    sanitize_grpo_metadata,
    summarize_reward_history,
)
from scripts.colab.run_l4_qwen3_z3_grpo import (
    HELDOUT_PROVENANCE,
    _candidate_branches_for_row,
    _gold_branch_key,
    build_all_verified_splits,
    format_zeroshot_prompt,
)

ARTIFACT_JSON = REPO_ROOT / "artifacts" / "v3.1.0" / "l4_qwen3_4b_z3_grpo_eval.json"


def test_train_sft_grpo_l4_cli_flags_and_dry_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = train_sft_grpo.main(
        [
            "--model",
            "Qwen/Qwen3-4B",
            "--train-jsonl",
            str(tmp_path / "train.jsonl"),
            "--output-dir",
            str(tmp_path / "ckpt"),
            "--batch-size",
            "2",
            "--no-vllm",
            "--vllm-gpu-util",
            "0.30",
            "--sft-epochs",
            "2",
            "--max-grpo-steps",
            "120",
            "--dry-run",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["model_id"] == "Qwen/Qwen3-4B"
    assert plan["target_gpu_tier"] == "L4_24GB"
    assert plan["stage1_sft"]["per_device_batch_size"] == 2
    assert plan["stage1_sft"]["effective_batch_size"] == 32
    assert plan["stage2_grpo"]["use_vllm"] is False
    assert plan["stage2_grpo"]["vllm_gpu_memory_utilization"] == 0.30
    assert plan["stage2_grpo"]["z3_reward_pool_start_method"] == "spawn"
    assert plan["stage2_grpo"]["beta"] == 0.04

    with pytest.raises(ValueError, match="vllm_gpu_memory_utilization"):
        GrpoPolicyConfig(vllm_gpu_memory_utilization=0.95)
    with pytest.raises(ValueError, match="batch_size"):
        GrpoPolicyConfig(batch_size=0)
    with pytest.raises(ValueError, match="banned"):
        GrpoPolicyConfig(model_id="CohereForAI/aya-expanse-8b")

    opt_vllm = GrpoPolicyConfig(use_vllm=True, vllm_gpu_memory_utilization=0.30)
    plan_v = build_training_plan_summary(opt_vllm, tmp_path / "t.jsonl", tmp_path / "o")
    assert plan_v["stage2_grpo"]["use_vllm"] is True


def test_l4_runner_heldout_paraphrase_splits_low_jaccard_and_z3_sat(tmp_path: Path) -> None:
    meta = build_all_verified_splits(tmp_path / "data", scale_per_bucket=6)
    assert meta["passed_anti_contamination"] is True
    assert meta["seed_overlap"] == 0
    assert meta["max_char_5gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert meta["max_token_3gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert meta["eval_provenance"] == HELDOUT_PROVENANCE

    for split_name in ("train", "calibration", "test"):
        rows = _load_jsonl(tmp_path / "data" / f"{split_name}.jsonl")
        assert len(rows) > 0
        for row in rows:
            assert validate_hv_telemetry(row["rta_snapshot"])
            assert parse_compact_dsl_py(row["canonical_dsl"]) is not None
            branch_keys = {k for k, _ in _candidate_branches_for_row(row)}
            assert _gold_branch_key(row) in branch_keys
            if split_name != "train":
                assert row["origin_provenance"] == HELDOUT_PROVENANCE
        p = format_zeroshot_prompt(rows[0], is_qwen3=True)
        assert "<think>\n\n</think>" in p and "ACT=DIVERT_SLZ" in p

    assert summarize_reward_history([]) == (0.0, 0.0)
    assert summarize_reward_history([0.5]) == (0.5, 0.5)
    assert summarize_reward_history([0.1] * 10 + [0.9] * 10) == (0.1, 0.9)

    # Verify Arrow None-padded gold_intent never crashes score_single_completion on clarification rollout
    nom_row = next(r for r in _load_jsonl(tmp_path / "data" / "train.jsonl") if r["bucket"] == "NOMINAL")
    arrow_padded_meta = sanitize_grpo_metadata(
        {
            "rta_snapshot": nom_row["rta_snapshot"],
            "bucket": nom_row["bucket"],
            "gold_intent": {**nom_row["gold_intent"], "top2_sat_pads": None, "requires_abstention": None},
            "expected_lang": nom_row["lang"],
        }
    )
    slz_ids = [s["id"] for s in nom_row["rta_snapshot"]["ranked_slzs"][:2]]
    clar_out = f"ACT=REQUEST_CLARIFICATION;A={slz_ids[0]};B={slz_ids[1]};RSN=AMBIG\nLNG=TR;MOD=CLARIFY"
    score = score_single_completion(clar_out, arrow_padded_meta)
    assert 0.0 <= score <= 1.0


def test_l4_qwen3_eval_artifact_schema_honesty_and_integrity() -> None:
    assert ARTIFACT_JSON.is_file(), f"Missing live L4 evaluation artifact: {ARTIFACT_JSON}"
    raw = ARTIFACT_JSON.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")

    report = json.loads(raw.decode("utf-8"))
    assert report["schema_version"] == "vertirta.l4_qwen3_z3_grpo_eval/v3.1.0"
    assert report["evidence_status"] == "LIVE_COLAB_L4_EXECUTED_REAL_WEIGHTS"
    assert "PENDING_HUMAN_NATIVE_TR_V1" in report["eval_split_provenance"]

    hw = report["hardware_and_software"]
    assert "L4" in hw["gpu_name"].upper()
    assert 15.0 <= float(hw["gpu_vram_total_gb"]) <= 30.0
    assert 0.5 < float(hw["peak_vram_gb"]) < 24.0

    ds = report["dataset"]
    assert ds["split_sizes"] == {"train": 1800, "calibration": 200, "test": 560}
    assert ds["seed_overlap"] == 0
    assert ds["max_char_5gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert ds["max_token_3gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert ds["passed_anti_contamination"] is True

    evals = report["evaluations"]
    for key in ("qwen3_4b_zero_shot", "qwen3_4b_sft", "qwen3_4b_sft_z3_grpo"):
        assert key in evals, f"Missing evaluation stage {key}"
        ov = evals[key]["overall"]
        assert ov["n_cases"] == 560
        assert 0.0 <= ov["grammar_validity_pct"] <= 100.0
        assert 0.0 <= ov["ast_stl_accuracy_pct"] <= 100.0
        assert 0.0 <= ov["unsafe_rejection_pct"] <= 100.0
        assert {"TR", "EN", "MIX"}.issubset(evals[key]["by_lang"].keys())
        assert {"NOMINAL", "FAULT", "ADVERSARIAL", "AMBIGUOUS_ASR"}.issubset(
            evals[key]["by_bucket"].keys()
        )

    # SFT+Z3-GRPO must genuinely outperform zero-shot Qwen3-4B on VertiBench AST/STL accuracy
    assert (
        evals["qwen3_4b_sft_z3_grpo"]["overall"]["ast_stl_accuracy_pct"]
        > evals["qwen3_4b_zero_shot"]["overall"]["ast_stl_accuracy_pct"]
    )

    conf = report["conformal_calibration"]
    assert conf["n_cal"] == 200 and conf["n_test"] == 560
    assert conf["order_statistic_k"] == 191 and abs(conf["q_level"] - 0.955) < 1e-6
    assert 0.0 <= conf["empirical_test_coverage_pct"] <= 100.0
    assert len(conf["empirical_coverage_clopper_pearson_95ci"]) == 2
    assert len(conf["finite_sample_beta_coverage_95ci_for_n_cal"]) == 2
    assert "N_cal>=750" in conf["calibration_caveat"]

    art = report["trained_adapter_artifact"]
    assert art["bytes"] > 1_000_000
    assert len(art["sha256"]) == 64

    # Verify self-consistent report SHA-256 digest
    digest_stored = report["report_sha256"]
    report_copy = {k: v for k, v in report.items() if k != "report_sha256"}
    digest_calc = hashlib.sha256(
        json.dumps(report_copy, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert digest_stored == digest_calc
