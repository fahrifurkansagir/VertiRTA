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
from ml.copilot.calibrate_conformal import (
    compute_ece,
    compute_nll,
    evaluate_conformal_sets,
    fit_and_calibrate_conformal,
    fit_temperature_scaling_nll,
    softmax_with_temperature,
)
from ml.copilot.grpo_z3_rewards import score_single_completion
from ml.copilot.train_sft_grpo import (
    GrpoPolicyConfig,
    build_training_plan_summary,
    sanitize_grpo_metadata,
    summarize_reward_history,
)
from scripts.colab.run_l4_qwen3_z3_grpo import (
    HELDOUT_PROVENANCE,
    VertiRtaGbnfLogitsProcessor,
    _candidate_branches_for_row,
    _gold_branch_key,
    apply_z3_fail_closed_gate,
    build_all_verified_splits,
    build_vertirta_gbnf_dfa,
    format_zeroshot_prompt,
)

ARTIFACT_JSON = REPO_ROOT / "artifacts" / "v3.1.0" / "l4_qwen3_4b_z3_grpo_eval.json"
ARTIFACT_V351_JSON = REPO_ROOT / "artifacts" / "v3.5.1" / "l4_qwen3_4b_v2_calibrated_eval.json"


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


def test_posthoc_nll_temperature_scaling_tightens_conformal_sets_zero_leakage() -> None:
    import random

    rng = random.Random(2026)
    cal_scores: list[list[float]] = []
    cal_golds: list[int] = []
    test_scores: list[list[float]] = []
    test_golds: list[int] = []

    def gen_row(idx: int) -> tuple[list[float], int]:
        gold = idx % 10
        # Flat uncalibrated branch log-likelihoods (mimicking gamma=0.5 raw scores):
        # 94% clear top-1 rows + 6% genuinely ambiguous 2-way ties (so the 95.5th
        # percentile lands on a 2-way tie where flat tau=1.0 leaks mass into 8 wrong branches).
        row = [-1.05 + rng.uniform(-0.15, 0.15) for _ in range(10)]
        if idx % 16 == 0:
            alt = (gold + 1) % 10
            row[alt] = -0.25
            row[gold] = -0.35
        else:
            row[gold] = -0.20 + rng.uniform(-0.08, 0.08)
        return row, gold

    for i in range(200):
        r, g = gen_row(i)
        cal_scores.append(r)
        cal_golds.append(g)
    for i in range(200, 760):
        r, g = gen_row(i)
        test_scores.append(r)
        test_golds.append(g)

    assert abs(sum(softmax_with_temperature(cal_scores[0], tau=0.25)) - 1.0) < 1e-9
    ts = fit_temperature_scaling_nll(cal_scores, cal_golds)
    assert ts["tau_opt"] < 0.8
    assert ts["nll_after"] < ts["nll_before"]
    assert ts["ece_after"] < ts["ece_before"]
    assert compute_nll(cal_scores, cal_golds, ts["tau_opt"]) <= compute_nll(cal_scores, cal_golds, 1.0)
    assert compute_ece(
        [softmax_with_temperature(r, ts["tau_opt"]) for r in cal_scores], cal_golds
    ) <= compute_ece([softmax_with_temperature(r, 1.0) for r in cal_scores], cal_golds)

    bundle = fit_and_calibrate_conformal(cal_scores, cal_golds, alpha=0.05)
    assert bundle["strict_split_cal_audit"] is not None
    assert bundle["strict_split_cal_audit"]["n_tau_fit"] == 100
    assert bundle["strict_split_cal_audit"]["n_qhat_cal"] == 100

    uncal_test = evaluate_conformal_sets(
        test_scores, test_golds, tau=1.0, q_hat=bundle["uncalibrated"]["q_hat"]
    )
    cal_test = evaluate_conformal_sets(
        test_scores, test_golds, tau=bundle["calibrated"]["tau"], q_hat=bundle["calibrated"]["q_hat"]
    )
    assert cal_test["empirical_coverage_pct"] >= 94.0
    assert cal_test["mean_prediction_set_size"] < 1.8
    assert cal_test["mean_prediction_set_size"] < uncal_test["mean_prediction_set_size"]
    assert cal_test["singleton_exec_pct"] >= 65.0
    assert cal_test["nll"] < uncal_test["nll"]


def test_paraphrase_augmented_train_split_gbnf_dfa_and_z3_fail_closed_gate(tmp_path: Path) -> None:
    meta = build_all_verified_splits(
        tmp_path / "data_aug", scale_per_bucket=10, paraphrase_augment_train=True
    )
    assert meta["train_paraphrase_augmented"] is True
    assert meta["passed_anti_contamination"] is True
    assert meta["seed_overlap"] == 0
    assert meta["max_char_5gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert meta["max_token_3gram_jaccard"] < MAX_JACCARD_THRESHOLD

    train_rows = _load_jsonl(tmp_path / "data_aug" / "train.jsonl")
    aug_count = sum(1 for r in train_rows if "train_paraphrase_aug_v2" in r.get("asr_corruption_tags", []))
    assert aug_count >= int(0.70 * len(train_rows))

    dfa = build_vertirta_gbnf_dfa()
    assert not dfa.accepts("<think>leak</think>\nACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF")
    assert not dfa.accepts("ACT=HOLD_LOITER;DUR=999;AGL=60\nLNG=TR;MOD=BRIEF")
    for split_name in ("train", "calibration", "test"):
        for row in _load_jsonl(tmp_path / "data_aug" / f"{split_name}.jsonl"):
            assert dfa.accepts(row["canonical_dsl"])

    # Mock ASCII + multi-char tokenizer verifying token-level GBNF processor masks prose/<think>
    class _MockTok:
        def __init__(self) -> None:
            chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_=;,\n")
            multi = ["ACT=", "HOLD_LOITER", ";DUR=", ";AGL=", "\nLNG=", ";MOD=", "BRIEF", "<think>", "hello "]
            self._id_to_tok = chars + multi + ["<|im_end|>"]
            self.vocab_size = len(self._id_to_tok)
            self.eos_token_id = self.vocab_size - 1
            self.all_special_ids = [self.eos_token_id]

        def decode(
            self,
            ids: list[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join(self._id_to_tok[i] for i in ids)

    tok = _MockTok()
    proc = VertiRtaGbnfLogitsProcessor(tok, dfa=dfa)
    think_id = tok._id_to_tok.index("<think>")
    prose_id = tok._id_to_tok.index("hello ")
    act_id = tok._id_to_tok.index("ACT=")
    assert act_id in proc.allowed_ids[dfa.start_state]
    assert think_id not in proc.allowed_ids[dfa.start_state]
    assert prose_id not in proc.allowed_ids[dfa.start_state]

    # Verify Z3 fail-closed gate repairs invalid/UNSAT proposals to 100% Z3-SAT
    test_rows = _load_jsonl(tmp_path / "data_aug" / "test.jsonl")
    bad_preds = [{"completion": "UNSAFE INVALID PROSE", "latency_ms": 10.0} for _ in test_rows]
    gated, stats = apply_z3_fail_closed_gate(test_rows, bad_preds)
    assert stats["raw_z3_constraint_satisfaction_pct"] == 0.0
    assert stats["gated_z3_constraint_satisfaction_pct"] == 100.0
    assert stats["z3_fail_closed_interventions"] == len(test_rows)
    for pred in gated:
        assert parse_compact_dsl_py(pred["completion"]) is not None


def test_l4_qwen3_v351_calibrated_eval_artifact_schema_honesty_and_integrity() -> None:
    assert ARTIFACT_V351_JSON.is_file(), f"Missing live v3.5.1 L4 artifact: {ARTIFACT_V351_JSON}"
    raw = ARTIFACT_V351_JSON.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")

    report = json.loads(raw.decode("utf-8"))
    assert report["schema_version"] == "vertirta.l4_qwen3_z3_grpo_eval/v3.5.1"
    assert report["evidence_status"] == "LIVE_COLAB_L4_EXECUTED_REAL_WEIGHTS"
    assert "PENDING_HUMAN_NATIVE_TR_V1" in report["eval_split_provenance"]

    hw = report["hardware_and_software"]
    assert "L4" in hw["gpu_name"].upper()
    assert 15.0 <= float(hw["gpu_vram_total_gb"]) <= 30.0
    assert 0.5 < float(hw["peak_vram_gb"]) < 24.0

    ds = report["dataset"]
    assert ds["split_sizes"] == {"train": 1800, "calibration": 200, "test": 560}
    assert ds["train_paraphrase_augmented"] is True
    assert ds["seed_overlap"] == 0
    assert ds["max_char_5gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert ds["max_token_3gram_jaccard"] < MAX_JACCARD_THRESHOLD
    assert ds["passed_anti_contamination"] is True

    tcfg = report["training_config"]
    assert tcfg["sft_epochs"] == 2
    assert tcfg["grpo_max_steps"] == 120
    assert tcfg["train_paraphrase_augmented"] is True

    tm = report["training_metrics"]
    assert tm["grpo_reward_batches"] == 120
    assert tm["grpo_mean_reward_last10"] >= tm["grpo_mean_reward_first10"]

    evals = report["evaluations"]
    for key in (
        "qwen3_4b_zero_shot",
        "qwen3_4b_zero_shot_gbnf_z3",
        "qwen3_4b_sft",
        "qwen3_4b_sft_z3_grpo",
        "qwen3_4b_sft_z3_grpo_gbnf_only",
        "qwen3_4b_sft_z3_grpo_gbnf_z3_conformal",
    ):
        assert key in evals, f"Missing evaluation stage {key}"
        assert evals[key]["overall"]["n_cases"] == 560

    # Paraphrase + 120-step Z3-GRPO beats SFT and v3.1.0; GBNF+Z3 hits 100% grammar & Z3 sat
    assert (
        evals["qwen3_4b_sft_z3_grpo"]["overall"]["ast_stl_accuracy_pct"]
        >= evals["qwen3_4b_sft"]["overall"]["ast_stl_accuracy_pct"]
    )
    assert evals["qwen3_4b_sft_z3_grpo_gbnf_z3_conformal"]["overall"]["grammar_validity_pct"] == 100.0
    assert (
        evals["qwen3_4b_sft_z3_grpo_gbnf_z3_conformal"]["overall"]["ast_stl_accuracy_pct"]
        > evals["qwen3_4b_sft_z3_grpo"]["overall"]["ast_stl_accuracy_pct"]
    )

    cmp = report["gbnf_z3_fail_closed_comparison"]["qwen3_4b_sft_z3_grpo"]
    assert cmp["gbnf_plus_z3_fail_closed"]["z3_constraint_satisfaction_pct"] == 100.0
    assert (
        cmp["gbnf_plus_z3_fail_closed"]["z3_constraint_satisfaction_pct"]
        >= cmp["unconstrained"]["z3_constraint_satisfaction_pct"]
    )

    conf = report["conformal_calibration"]
    assert conf["n_cal"] == 200 and conf["n_test"] == 560
    assert conf["empirical_test_coverage_pct"] >= 95.0
    assert conf["temperature_calibrated_test"]["nll"] < conf["uncalibrated_baseline_tau_1"]["nll"]
    assert conf["temperature_calibrated_test"]["ece"] < conf["uncalibrated_baseline_tau_1"]["ece"]
    assert (
        conf["temperature_calibrated_test"]["mean_prediction_set_size"]
        < conf["uncalibrated_baseline_tau_1"]["mean_prediction_set_size"]
    )
    assert (
        conf["temperature_calibrated_test"]["singleton_exec_pct"]
        > conf["uncalibrated_baseline_tau_1"]["singleton_exec_pct"]
    )
    assert conf["strict_split_cal_audit"]["n_tau_fit"] == 100
    assert conf["strict_split_cal_audit"]["test_eval"]["empirical_coverage_pct"] >= 95.0

    art = report["trained_adapter_artifact"]
    assert art["bytes"] > 1_000_000
    assert len(art["sha256"]) == 64

    digest_stored = report["report_sha256"]
    report_copy = {k: v for k, v in report.items() if k != "report_sha256"}
    digest_calc = hashlib.sha256(
        json.dumps(report_copy, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert digest_stored == digest_calc
