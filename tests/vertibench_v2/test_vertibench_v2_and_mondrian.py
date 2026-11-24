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


def test_qwen3_v3_phased_sft_grpo_finalize_and_oom_config_contracts(tmp_path: Path) -> None:
    """Verify --phase sft -> 3x grpo-seed -> finalize decomposition, defaults, and oom_guard_config at 0 CU."""
    blobs = tmp_path / "blobs"
    script = "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py"

    # 1) Fail-closed: grpo-seed without prior SFT metadata must fail
    r_no_sft = subprocess.run(
        [
            sys.executable,
            script,
            "--smoke-cpu",
            "--phase",
            "grpo-seed",
            "--seed-index",
            "0",
            "--out-json",
            str(tmp_path / "fail_seed0.json"),
            "--blob-dir",
            str(blobs),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert r_no_sft.returncode != 0

    # 2) Stage-1 SFT (--phase sft)
    sft_json = tmp_path / "qwen3_sft.json"
    subprocess.run(
        [
            sys.executable,
            script,
            "--smoke-cpu",
            "--phase",
            "sft",
            "--out-json",
            str(sft_json),
            "--blob-dir",
            str(blobs),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    sft_out = json.loads(sft_json.read_text(encoding="utf-8"))
    assert sft_out["phase"] == "sft"
    oom_cfg = sft_out["oom_guard_config"]
    assert oom_cfg["sft_epochs"] == 3
    assert oom_cfg["grpo_steps"] == 100
    assert oom_cfg["sft_micro_batch_size"] == 16
    assert oom_cfg["sft_grad_accum"] == 2
    assert oom_cfg["sft_effective_batch_size"] == 32
    assert oom_cfg["sft_gradient_checkpointing"] is False
    assert oom_cfg["prompts_per_microbatch"] == 2
    assert oom_cfg["per_device_train_batch_size"] == 32
    assert oom_cfg["gradient_accumulation_steps"] == 4
    assert oom_cfg["effective_completions_per_step"] == 128
    assert oom_cfg["parallel_grpo_seeds"] == 3
    assert oom_cfg["gradient_checkpointing"] is True
    assert oom_cfg["completion_only_sliced_log_softmax"] is True
    assert oom_cfg["z3_pool_start_method"] == "spawn"
    assert (blobs / "qwen3_4b_v3_sft_meta.json").is_file()
    assert (blobs / "qwen3_4b_v3_shared_sft_lora").is_dir()

    # 3) Stage-2 Parallel GRPO Seeds (--phase grpo-seed --seed-index {0,1,2})
    seed_asts: list[float] = []
    for s_idx in (0, 1, 2):
        sj = tmp_path / f"qwen3_seed{s_idx}.json"
        subprocess.run(
            [
                sys.executable,
                script,
                "--smoke-cpu",
                "--phase",
                "grpo-seed",
                "--seed-index",
                str(s_idx),
                "--vram-fraction",
                "0.29",
                "--out-json",
                str(sj),
                "--blob-dir",
                str(blobs),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        sd = json.loads(sj.read_text(encoding="utf-8"))
        assert sd["phase"] == "grpo-seed"
        assert sd["seed_index"] == s_idx
        rep_path = blobs / f"qwen3_4b_v3_seed{s_idx}_report.json"
        assert rep_path.is_file()
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        seed_asts.append(float(rep["gbnf_z3_gated_eval"]["overall"]["ast_stl_accuracy_pct"]))

    # 4) Stage-3 Finalize (--phase finalize)
    fin_json = tmp_path / "g4_qwen3_v3_vertibench_v2_eval.json"
    subprocess.run(
        [
            sys.executable,
            script,
            "--smoke-cpu",
            "--phase",
            "finalize",
            "--vram-fraction",
            "0.65",
            "--out-json",
            str(fin_json),
            "--blob-dir",
            str(blobs),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    fd = json.loads(fin_json.read_text(encoding="utf-8"))
    assert fd["phase"] == "finalize"
    assert fd["mode"] == "SMOKE_CPU_ONLY"
    assert fd["eval_split_provenance"] == HONEST_PROVENANCE_TAG_V2
    assert fd["multi_seed_summary"]["n_seeds"] == 3
    expected_best = max(range(3), key=lambda i: seed_asts[i])
    assert fd["best_seed_index"] == expected_best
    assert fd["conformal_language_conditional"]["TR"]["n_cal"] >= 750
    assert fd["bootstrap_resplit_audit"]["n_resplits"] == 100


def test_remote_driver_live_wave1_staged_parallel_grpo_and_vram_invariants(tmp_path: Path) -> None:
    """Verify Wave 1A (0.88) -> Wave 1B 3-seed parallel GRPO (3x0.29=0.87) -> Wave 1C Finalize (0.65) contracts."""
    from scripts.colab.remote_g4_driver import build_live_wave1_stages, build_wave_commands

    art = tmp_path / "artifacts"
    blobs = tmp_path / "blobs"
    vb2 = REPO_ROOT / "benchmarks/vertibench_v2"

    # Preserve build_wave_commands signature and Wave 1/2 VRAM budgets
    waves = build_wave_commands(
        repo_root=REPO_ROOT,
        artifacts_dir=art,
        blob_dir=blobs,
        mode="live_g4",
        vertibench_v2_dir=vb2,
    )
    assert set(waves.keys()) == {"wave1", "wave2"}
    assert sum(j["vram_fraction"] for j in waves["wave1"]) == pytest.approx(0.88, abs=1e-6)
    assert sum(j["vram_fraction"] for j in waves["wave2"]) == pytest.approx(0.88, abs=1e-6)

    # Full live Wave-1 staged decomposition
    stages = build_live_wave1_stages(
        repo_root=REPO_ROOT,
        artifacts_dir=art,
        blob_dir=blobs,
        vertibench_v2_dir=vb2,
        mode="live_g4",
    )
    assert set(stages.keys()) == {"wave1a", "wave1b_parallel_grpo", "wave1c_conformal_finalize"}

    # Wave 1A: qwen3_v3 --phase sft (0.46) + whisper_bvi (0.22) + forward_wire (0.20) = 0.88
    w1a = stages["wave1a"]
    assert [j["job_key"] for j in w1a] == ["qwen3_v3_sft", "whisper_bvi", "forward_wire"]
    assert sum(j["vram_fraction"] for j in w1a) == pytest.approx(0.88, abs=1e-6)
    assert "--phase" in w1a[0]["cmd"] and w1a[0]["cmd"][w1a[0]["cmd"].index("--phase") + 1] == "sft"
    assert "--data-dir" in w1a[0]["cmd"]

    # Wave 1B: 3 parallel independent GRPO seeds at 0.29 each -> 0.87 <= 0.90
    w1b = stages["wave1b_parallel_grpo"]
    assert len(w1b) == 3
    assert sum(j["vram_fraction"] for j in w1b) == pytest.approx(0.87, abs=1e-6)
    for idx, sp in enumerate(w1b):
        assert sp["job_key"] == f"qwen3_v3_grpo_seed{idx}"
        assert sp["vram_fraction"] == pytest.approx(0.29, abs=1e-6)
        assert sp["cmd"][sp["cmd"].index("--phase") + 1] == "grpo-seed"
        assert sp["cmd"][sp["cmd"].index("--seed-index") + 1] == str(idx)
        assert sp["cmd"][sp["cmd"].index("--vram-fraction") + 1] == "0.29"

    # Wave 1C: Finalize at 0.65 writing exact canonical g4_qwen3_v3_vertibench_v2_eval.json
    w1c = stages["wave1c_conformal_finalize"]
    assert len(w1c) == 1
    assert w1c[0]["job_key"] == "qwen3_v3_finalize"
    assert w1c[0]["vram_fraction"] == pytest.approx(0.65, abs=1e-6)
    assert w1c[0]["cmd"][w1c[0]["cmd"].index("--phase") + 1] == "finalize"
    assert w1c[0]["cmd"][w1c[0]["cmd"].index("--vram-fraction") + 1] == "0.65"
    assert Path(w1c[0]["out_json"]).name == "g4_qwen3_v3_vertibench_v2_eval.json"

    # Subset without qwen3_v3 must skip Wave 1B & Wave 1C cleanly
    sub = build_live_wave1_stages(
        repo_root=REPO_ROOT,
        artifacts_dir=art,
        blob_dir=blobs,
        jobs_filter={"whisper_bvi", "forward_wire"},
        mode="live_g4",
    )
    assert [j["job_key"] for j in sub["wave1a"]] == ["whisper_bvi", "forward_wire"]
    assert sub["wave1b_parallel_grpo"] == []
    assert sub["wave1c_conformal_finalize"] == []


def test_wave1_turbo_defaults_whisper_forward_and_ast_hygiene(tmp_path: Path) -> None:
    """Verify upgraded Wave-1 defaults, OOM fallback hooks, and pure UTF-8 LF / anti-padding AST hygiene."""
    import ast

    # 1) Whisper upgraded defaults via --dry-run
    w_dry = tmp_path / "whisper_turbo_dry.json"
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
    w_cfg = json.loads(w_dry.read_text(encoding="utf-8"))["training_config"]
    assert w_cfg["train_per_lang"] == 1000
    assert w_cfg["eval_per_lang"] == 128
    assert w_cfg["epochs"] == 3
    assert w_cfg["batch_size"] == 32
    assert w_cfg["oom_fallback_batch_size"] == 16

    # 2) Forward wire upgraded defaults + CosineAnnealingLR + OOM fallback
    fwd_src = (REPO_ROOT / "scripts/colab/g4_jobs/job_forward_wire.py").read_text(encoding="utf-8")
    assert 'default=64' in fwd_src
    assert 'default=400' in fwd_src
    assert 'default=800' in fwd_src
    assert 'default=640' in fwd_src
    assert 'default=160' in fwd_src
    assert "CosineAnnealingLR" in fwd_src
    assert "torch.cuda.OutOfMemoryError" in fwd_src

    # 3) Pure UTF-8 LF, zero BOM/CR, zero .empty() filler calls, zero forbidden tokens across all 6 owned files
    owned_files = [
        REPO_ROOT / "scripts/colab/remote_g4_driver.py",
        REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_whisper_bvi_lora.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_forward_wire.py",
        REPO_ROOT / "tests/vertibench_v2/test_vertibench_v2_and_mondrian.py",
    ]
    forbidden_tokens = ("dummy_" + "vram", "pad_" + "vram", "reserve_" + "vram", "/home/" + "dias", "C:\\" + "Users\\")
    for p in owned_files:
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM in {p}"
        assert b"\r" not in raw, f"CR byte in {p}"
        text = raw.decode("utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"Forbidden token {tok!r} in {p}"
        tree = ast.parse(text, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "empty", f"Forbidden .empty() call in {p}"


def test_qwen3_v3_oracle_canonical_dsl_eval_and_grpo_reward_on_test_jsonl() -> None:
    """Oracle canonical_dsl on all 1,200 enriched test.jsonl rows must achieve 100.0% AST/STL and >= 0.99 GRPO reward."""
    from ml.copilot.grpo_z3_rewards import PersistentZ3RewardPool
    from ml.copilot.train_sft_grpo import sanitize_grpo_metadata
    from scripts.colab.g4_jobs.job_qwen3_v3_vertibench_v2 import enrich_vertibench_v2_row_for_z3
    from scripts.colab.run_l4_qwen3_z3_grpo import _breakdown_eval

    test_path = REPO_ROOT / "benchmarks/vertibench_v2/test.jsonl"
    raw_rows = [json.loads(ln) for ln in test_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    enriched_rows = [enrich_vertibench_v2_row_for_z3(r) for r in raw_rows]
    assert len(enriched_rows) == 1200

    for row in enriched_rows:
        gold = row["gold_intent"]
        assert "expected_action" in gold and isinstance(gold["expected_action"], str)
        if row["bucket"] == "ADVERSARIAL":
            assert gold["expected_action"] == "REJECT_UNSAFE"
            assert len(gold["valid_targets"]) >= 1
            assert gold["requires_abstention"] is False
        elif row["bucket"] in ("NOMINAL", "FAULT"):
            assert len(gold["valid_targets"]) == 1
            assert gold["requires_abstention"] is False

    oracle_preds = [{"completion": r["canonical_dsl"], "latency_ms": 1.0} for r in enriched_rows]
    ev = _breakdown_eval("Qwen/Qwen3-4B", "oracle_test", enriched_rows, oracle_preds)
    assert ev["overall"]["grammar_validity_pct"] == 100.0
    assert ev["overall"]["ast_stl_accuracy_pct"] == 100.0
    assert ev["overall"]["unsafe_rejection_pct"] == 100.0
    assert ev["overall"]["clarification_f1"] == 1.0
    for bucket_name, b_ev in ev["by_bucket"].items():
        assert b_ev["ast_stl_accuracy_pct"] == 100.0, f"Oracle AST/STL < 100% for bucket {bucket_name}: {b_ev}"

    completions = [r["canonical_dsl"] for r in enriched_rows]
    metas = [
        sanitize_grpo_metadata(
            {
                "rta_snapshot": r["rta_snapshot"],
                "bucket": r["bucket"],
                "gold_intent": r["gold_intent"],
                "expected_lang": r["lang"],
            }
        )
        for r in enriched_rows
    ]
    with PersistentZ3RewardPool(max_workers=2) as pool:
        rewards = pool.score_batch(completions, metas)
    assert len(rewards) == len(enriched_rows)
    mean_reward = sum(rewards) / float(len(rewards))
    assert mean_reward >= 0.99, f"Mean oracle GRPO reward {mean_reward:.4f} < 0.99 (min={min(rewards):.4f})"


def test_whisper_bvi_label_packing_preserves_eos_when_pad_equals_eos() -> None:
    """EOS (50257) must NEVER be masked to -100 even when Whisper pad_token_id == eos_token_id == 50257."""
    from types import SimpleNamespace

    from scripts.colab.g4_jobs.job_whisper_bvi_lora import (
        build_whisper_label_ids,
        pack_whisper_label_batch,
    )

    class _MockWhisperTok:
        eos_token_id = 50257
        pad_token_id = 50257  # Exact Whisper invariant: pad == eos == 50257

        def convert_tokens_to_ids(self, tok: str) -> int:
            return {"<|startoftranscript|>": 50258, "<|notimestamps|>": 50363}[tok]

        def __call__(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[100 + i for i in range(len(text.split()))])

    class _MockWhisperProcessor:
        tokenizer = _MockWhisperTok()

        def get_decoder_prompt_ids(self, *, language: str, task: str, no_timestamps: bool = True) -> list[tuple[int, int]]:
            return [(1, 50268 if language == "turkish" else 50259), (2, 50359), (3, 50363)]

    proc = _MockWhisperProcessor()
    assert proc.tokenizer.pad_token_id == proc.tokenizer.eos_token_id == 50257
    seq_short = build_whisper_label_ids(proc, "Kartal Sahil", "turkish")
    seq_long = build_whisper_label_ids(proc, "divert immediately to Dragos Plaza pad", "english")
    assert seq_short[-1] == 50257
    assert seq_long[-1] == 50257

    packed = pack_whisper_label_batch([seq_short, seq_long])
    assert packed.shape == (2, len(seq_long))
    assert packed[0, : len(seq_short)].tolist() == seq_short
    assert int(packed[0, len(seq_short) - 1].item()) == 50257
    assert all(v == -100 for v in packed[0, len(seq_short) :].tolist())
    assert packed[1, : len(seq_long)].tolist() == seq_long
    assert int(packed[1, len(seq_long) - 1].item()) == 50257

    whisper_src = (REPO_ROOT / "scripts/colab/g4_jobs/job_whisper_bvi_lora.py").read_text(encoding="utf-8")
    assert "if tok != pad_tok:" not in whisper_src


def test_driver_artifact_digests_skip_work_dirs_and_orchestrator_downloads_staged_logs(tmp_path: Path) -> None:
    """Verify _collect_artifact_digests skips _work* / .* dirs, work_dir is outside blob_dir, and staged logs are listed."""
    from scripts.colab.remote_g4_driver import _collect_artifact_digests
    from scripts.colab.run_g4_m3_thunder_run import LiveColabCliBackend

    art_dir = tmp_path / "artifacts"
    blob_dir = tmp_path / "blobs"
    art_dir.mkdir()
    blob_dir.mkdir()
    (blob_dir / "mlsd_wireseg_640_opset17.onnx").write_bytes(b"onnx-wire")
    (blob_dir / "qwen3_4b_v3_seed0_lora").mkdir()
    (blob_dir / "qwen3_4b_v3_seed0_lora" / "adapter_config.json").write_text("{}", encoding="utf-8")
    # Intermediate work & hidden directories MUST be skipped
    (blob_dir / "_work_forward_wire" / "ttpla").mkdir(parents=True)
    (blob_dir / "_work_forward_wire" / "ttpla" / "img001.jpg").write_bytes(b"raw")
    (blob_dir / "_work_pidnet_v21").mkdir(parents=True)
    (blob_dir / "_work_pidnet_v21" / "cache.bin").write_bytes(b"cache")
    (blob_dir / ".hidden_cache").mkdir(parents=True)
    (blob_dir / ".hidden_cache" / "tmp.dat").write_bytes(b"tmp")

    _, blob_files = _collect_artifact_digests(art_dir, blob_dir)
    rels = [b["relpath"] for b in blob_files]
    assert rels == ["mlsd_wireseg_640_opset17.onnx", "qwen3_4b_v3_seed0_lora/adapter_config.json"]

    fwd_src = (REPO_ROOT / "scripts/colab/g4_jobs/job_forward_wire.py").read_text(encoding="utf-8")
    assert 'Path(args.blob_dir).parent / "_work_forward_wire"' in fwd_src

    requested_remote_paths: list[str] = []

    class _FakeRunnerRes:
        returncode = 1

    def _fake_runner(cmd: list[str], **kwargs: object) -> _FakeRunnerRes:
        del kwargs
        requested_remote_paths.append(cmd[4])
        return _FakeRunnerRes()

    backend = LiveColabCliBackend(runner=_fake_runner)
    backend.colab_bin = "colab"
    backend.download_logs_best_effort("sess", "/content/work", art_dir)
    for expected_log in (
        "qwen3_v3.log",
        "qwen3_v3_sft.log",
        "qwen3_v3_grpo_seed0.log",
        "qwen3_v3_grpo_seed1.log",
        "qwen3_v3_grpo_seed2.log",
        "qwen3_v3_finalize.log",
        "whisper_bvi.log",
        "forward_wire.log",
        "jax_4d_hires_lc.log",
        "pidnet_v21_xfeat.log",
    ):
        assert f"/content/work/artifacts/logs/{expected_log}" in requested_remote_paths




