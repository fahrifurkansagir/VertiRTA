# SPDX-License-Identifier: Apache-2.0
"""TDD tests for VertiBench-v2, Mondrian Conformal calibration, and Wave-1 G4 job smoke paths."""

from __future__ import annotations

import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from benchmarks.vertibench_v2.build_vertibench_v2 import build_vertibench_v2_splits
from benchmarks.vertibench_v2.schema import HONEST_PROVENANCE_TAG_V2
from benchmarks.vertibench_v2.validate_vertibench_v2 import validate_vertibench_v2
from ml.copilot.calibrate_conformal import (
    bootstrap_resplit_mondrian_coverage,
    cv_select_gamma_tau_calibration_only,
    evaluate_mondrian_conformal,
    fit_mondrian_conformal_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_STRATA = ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")


def test_vertibench_v2_on_disk_and_fresh_build_pass_all_anti_contamination_gates(tmp_path: Path) -> None:
    # 1) Validate tracked benchmarks/vertibench_v2/ splits
    on_disk = validate_vertibench_v2(REPO_ROOT / "benchmarks/vertibench_v2")
    assert on_disk.passed, f"On-disk VertiBench-v2 errors: {on_disk.errors}"
    assert on_disk.max_char_5gram_jaccard < 0.15
    assert on_disk.max_token_3gram_jaccard < 0.15
    assert on_disk.seed_overlap == 0
    assert on_disk.cal_lang_counts == {"TR": 800, "EN": 800, "MIX": 400}
    assert on_disk.test_lang_counts == {"TR": 500, "EN": 500, "MIX": 200}
    assert all(c == 500 for c in on_disk.cal_snr_counts.values())
    assert on_disk.all_fleet20_west_of_29_1895 is True
    assert on_disk.vertibench_v1_unmodified is True

    # 2) Verify fresh deterministic rebuild matches SHA-256 bit-for-bit
    rebuilt = build_vertibench_v2_splits(out_dir=tmp_path, seed=20261124)
    assert rebuilt["provenance_tag"] == HONEST_PROVENANCE_TAG_V2
    disk_manifest = json.loads((REPO_ROOT / "benchmarks/vertibench_v2/manifest.json").read_text(encoding="utf-8"))
    assert rebuilt["file_sha256"] == disk_manifest["file_sha256"]


def test_mondrian_conformal_cv_and_100_resplit_bootstrap_enforces_alpha_0_05() -> None:
    rng = random.Random(42)
    lp_rows, tc_rows, probs, golds, groups = [], [], [], [], []
    for g_idx, g in enumerate(CANONICAL_STRATA):
        for _ in range(300):
            p_gold = max(0.30, min(0.99, 0.94 - 0.11 * g_idx + rng.uniform(-0.07, 0.04)))
            rem = (1.0 - p_gold) / 3.0
            row = [p_gold, rem, rem, rem]
            probs.append(row)
            lp_rows.append([-1.2, -5.0 + 0.7 * g_idx, -5.4 + 0.7 * g_idx, -5.8 + 0.7 * g_idx])
            tc_rows.append([18, 18, 16, 12])
            golds.append(0)
            groups.append(g)

    cv = cv_select_gamma_tau_calibration_only(lp_rows, tc_rows, golds, gammas=(0.5, 0.75, 1.0), n_folds=5)
    assert cv["zero_test_leakage"] is True
    assert cv["best_gamma"] in (0.5, 0.75, 1.0)
    assert 0.05 <= cv["best_tau"] <= 5.0

    fit = fit_mondrian_conformal_groups(probs, golds, groups, alpha=0.05)
    assert fit["alpha"] == pytest.approx(0.05, abs=1e-9)
    assert fit["group_q_hat"]["S3_Garbled"] > fit["group_q_hat"]["S0_Clean"]

    ev = evaluate_mondrian_conformal(probs, golds, groups, fit["group_q_hat"])
    assert ev["overall_coverage_pct"] >= 94.5

    boot = bootstrap_resplit_mondrian_coverage(probs, golds, groups, alpha=0.05, n_resplits=100, seed=7)
    assert boot["n_resplits"] == 100
    for g in CANONICAL_STRATA:
        assert boot["group_metrics"][g]["coverage_mean"] >= 0.93


def test_g4_qwen3_v3_and_whisper_bvi_dry_run_and_smoke_cpu(tmp_path: Path) -> None:
    # 1) Qwen3-4B v3 --dry-run & --smoke-cpu
    q_dry = tmp_path / "qwen3_dry.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py",
            "--dry-run",
            "--out-json",
            str(q_dry),
            "--blob-dir",
            str(tmp_path / "blobs"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert json.loads(q_dry.read_text(encoding="utf-8"))["mode"] == "DRY_RUN"

    q_json = tmp_path / "qwen3_smoke.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py",
            "--smoke-cpu",
            "--out-json",
            str(q_json),
            "--blob-dir",
            str(tmp_path / "blobs"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    qd = json.loads(q_json.read_text(encoding="utf-8"))
    assert qd["mode"] == "SMOKE_CPU_ONLY"
    assert qd["eval_split_provenance"] == HONEST_PROVENANCE_TAG_V2
    assert qd["conformal_language_conditional"]["TR"]["n_cal"] >= 750
    assert qd["conformal_language_conditional"]["EN"]["n_cal"] >= 750
    assert "MIX" in qd["conformal_language_conditional"]
    assert all(qd["conformal_snr_mondrian"][s]["n_cal"] >= 250 for s in CANONICAL_STRATA)
    assert qd["bootstrap_resplit_audit"]["n_resplits"] == 100
    assert qd["liveness_soc_lt_5pct_override_verified"] is True
    assert qd["mayday_priority_wm_015_verified"] is True
    assert qd["oom_guard_config"]["gradient_checkpointing"] is True
    assert qd["oom_guard_config"]["completion_only_sliced_log_softmax"] is True
    assert qd["oom_guard_config"]["z3_pool_start_method"] == "spawn"

    # 2) Whisper BVI LoRA --dry-run & --smoke-cpu
    w_dry = tmp_path / "whisper_dry.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/colab/g4_jobs/job_whisper_bvi_lora.py",
            "--dry-run",
            "--out-json",
            str(w_dry),
            "--blob-dir",
            str(tmp_path / "blobs"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    assert json.loads(w_dry.read_text(encoding="utf-8"))["mode"] == "DRY_RUN"

    w_json = tmp_path / "whisper_smoke.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/colab/g4_jobs/job_whisper_bvi_lora.py",
            "--smoke-cpu",
            "--out-json",
            str(w_json),
            "--blob-dir",
            str(tmp_path / "blobs"),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    wd = json.loads(w_json.read_text(encoding="utf-8"))
    assert wd["mode"] == "SMOKE_CPU_ONLY"
    assert wd["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert wd["whisper_inference_locks"]["translate"] is False
    assert wd["whisper_inference_locks"]["no_context"] is True
    assert wd["whisper_inference_locks"]["min_audio_ctx"] == 256
    assert 112.0 <= wd["coaxial_bvi_dsp_audit"]["max_rpm_bpf_hz"] <= 117.0
    assert wd["coaxial_bvi_dsp_audit"]["max_snr_error_db"] < 0.5
    assert set(wd["snr_strata_metrics"].keys()) == set(CANONICAL_STRATA)
    # Scoped license_audit must list strictly actually ingested sources (zero uningested corpus claims)
    assert set(wd["license_audit"]["train_source_names"]) == {
        "FLEURS-CC-BY-4.0",
        "VertiRTA-BVI-DSP-Apache2",
    }


def test_qwen3_v3_language_matched_branches_sft_masking_z3_enrichment_and_whisper_wer_cer() -> None:
    from benchmarks.vertibench_tr_en.validate_vertibench import parse_compact_dsl_py
    from ml.copilot.grpo_z3_rewards import verify_z3_static_corridor
    from ml.copilot.train_sft_grpo import QWEN3_EMPTY_THINK_SUFFIX, format_copilot_prompt
    from scripts.colab.g4_jobs.job_qwen3_v3_vertibench_v2 import (
        build_language_matched_candidate_branches,
        enrich_vertibench_v2_row_for_z3,
        gold_branch_key_v2,
        mask_sft_prompt_labels,
    )
    from scripts.colab.g4_jobs.job_whisper_bvi_lora import compute_wer_cer

    cal_lines = (REPO_ROOT / "benchmarks/vertibench_v2/cal.jsonl").read_text(encoding="utf-8").splitlines()
    # Sample across TR, EN, MIX and all 4 buckets
    rows = [json.loads(ln) for ln in cal_lines[::50]]
    langs_seen = set()
    for raw in rows:
        enriched = enrich_vertibench_v2_row_for_z3(raw)
        lang = enriched["lang"]
        langs_seen.add(lang)

        # 1) Canonical Qwen3 empty think suffix in prompt
        prompt = format_copilot_prompt(enriched)
        assert prompt.endswith(QWEN3_EMPTY_THINK_SUFFIX)

        # 2) Language-matched candidate branches (ZERO hardcoded LNG=TR on EN or MIX!)
        branches = build_language_matched_candidate_branches(enriched)
        for _, dsl_txt in branches:
            assert f"LNG={lang};" in dsl_txt, f"Language mismatch for {lang}: {dsl_txt}"
            if lang != "TR":
                assert "LNG=TR;" not in dsl_txt
        g_key = gold_branch_key_v2(enriched)
        branch_keys = [k for k, _ in branches]
        assert g_key in branch_keys, f"Gold branch {g_key} missing from {branch_keys}"

        # 3) Real Microsoft Z3 satisfiability of enriched canonical_dsl
        ast = parse_compact_dsl_py(enriched["canonical_dsl"])
        assert ast is not None
        z3_res = verify_z3_static_corridor(ast, enriched["rta_snapshot"])
        assert z3_res.is_sat or (ast.action == "REJECT_UNSAFE" and z3_res.fallback_is_sat)

    assert langs_seen == {"TR", "EN", "MIX"}

    # 4) Prompt tokens + padding masked to -100 while completion tokens stay supervised
    masked = mask_sft_prompt_labels([10, 11, 12, 20, 21, 99, 99], prompt_len=3, pad_token_id=99)
    assert masked == [-100, -100, -100, 20, 21, -100, -100]

    # 5) Real WER / CER metric accuracy (incl. Turkish Unicode normalization)
    w0, c0 = compute_wer_cer(["İrtifa 80 metre, Kartal Sahil."], ["irtifa 80 metre kartal sahil"])
    assert w0 == 0.0 and c0 == 0.0
    w1, c1 = compute_wer_cer(["divert to kartal sahil"], ["divert to dragos sahil"])
    assert w1 == pytest.approx(0.25, abs=1e-4)
    assert c1 > 0.0


def test_g4_live_path_contracts_gbnf_signature_ast_key_use_cache_and_whisper_labels() -> None:
    import inspect
    from types import SimpleNamespace

    from benchmarks.vertibench_tr_en.eval_vertibench import evaluate_predictions
    from scripts.colab.g4_jobs.job_whisper_bvi_lora import build_whisper_label_ids
    from scripts.colab.run_l4_qwen3_z3_grpo import (
        VertiRtaGbnfLogitsProcessor,
        build_vertirta_gbnf_dfa,
    )

    # (1) Exact GBNF DFA builder & LogitsProcessor signature (tokenizer, dfa)
    dfa = build_vertirta_gbnf_dfa()
    assert dfa.start_state >= 0
    sig = list(inspect.signature(VertiRtaGbnfLogitsProcessor.__init__).parameters.keys())
    assert sig[:3] == ["self", "tokenizer", "dfa"]

    # (2) Exact evaluate_predictions key ("ast_stl_accuracy_pct") & KV cache / 512 prompt checks
    qwen_src = (REPO_ROOT / "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py").read_text(encoding="utf-8")
    assert "VertiRtaGbnfDfa.build" not in qwen_src
    assert "from_tokenizer" not in qwen_src
    assert "ast_stl_correctness_pct" not in qwen_src
    assert '"ast_stl_accuracy_pct"' in qwen_src
    assert "model.config.use_cache = True" in qwen_src
    assert '"max_prompt_length": 512' in qwen_src
    assert "ast_stl_accuracy_pct" in inspect.getsource(evaluate_predictions)

    # (4) Whisper LoRA labels must NEVER duplicate <|startoftranscript|> or <|notimestamps|>
    class _MockTok:
        eos_token_id = 50257

        def convert_tokens_to_ids(self, tok: str) -> int:
            return {"<|startoftranscript|>": 50258, "<|notimestamps|>": 50363}[tok]

        def __call__(self, text: str, add_special_tokens: bool = True) -> Any:
            assert add_special_tokens is False, "Must disable special tokens to avoid duplicate <|notimestamps|>"
            return SimpleNamespace(input_ids=[101, 102, 103])

    class _MockProcessor:
        tokenizer = _MockTok()

        def get_decoder_prompt_ids(self, *, language: str, task: str, no_timestamps: bool = True) -> list[tuple[int, int]]:
            assert language in ("turkish", "english") and task == "transcribe" and no_timestamps is True
            return [(1, 50268), (2, 50359), (3, 50363)]

    ids = build_whisper_label_ids(_MockProcessor(), "Kartal Sahil", "turkish")
    assert ids == [50258, 50268, 50359, 50363, 101, 102, 103, 50257]
    assert ids.count(50258) == 1
    assert ids.count(50363) == 1

    # (5) Real sliced encoder benchmark + direct HF parquet API fallback in job_whisper_bvi_lora.py
    whisper_src = (REPO_ROOT / "scripts/colab/g4_jobs/job_whisper_bvi_lora.py").read_text(encoding="utf-8")
    assert "256.0 / 1500.0" not in whisper_src
    assert "embed_positions.weight[:seq_t]" in whisper_src
    assert "https://huggingface.co/api/datasets/google/fleurs/parquet/" in whisper_src


