# SPDX-License-Identifier: Apache-2.0
"""Full 5-Job M3 (v4.0.0) End-to-End 0-CU Smoke Suite (Wave 1 -> Barrier -> Wave 2 -> TU116 Gate).

Executes ALL 5 G4 jobs (Qwen3-4B v3, Whisper BVI LoRA, Forward Wire/D-FINE-N, JAX 4D HiRes/LC,
PIDNet-S v2.1/XFeat) through `remote_g4_driver.py --smoke-cpu`, `run_g4_m3_thunder_run.py --dry-run`,
and `bench_gtx1650s_m3_timesliced.py --smoke-cpu` locally at 0.00 CU, verifying:
  - Exact Wave-1 (0.46 + 0.22 + 0.20 = 0.88) -> Drain Barrier -> Wave-2 (0.58 XLA + 0.30 PyTorch = 0.88)
  - All 5 job JSON contracts, SHA-256 blob parity, zero NaN/Inf floats across every emitted artifact
  - Zero banned licenses, zero dummy VRAM padding, pure UTF-8 LF, zero personal path leaks
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.colab.g4_jobs.common import CANONICAL_SNR_STRATA, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[2]
ALL_FIVE_JOBS: tuple[str, ...] = (
    "qwen3_v3",
    "whisper_bvi",
    "forward_wire",
    "jax_4d_hires_lc",
    "pidnet_v21_xfeat",
)


def _assert_no_nan_or_inf(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float {obj!r} at {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan_or_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_no_nan_or_inf(item, f"{path}[{idx}]")


def test_all_five_g4_jobs_pure_lf_no_personal_paths_no_dummy_padding() -> None:
    paths = [
        REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py",
        REPO_ROOT / "scripts/colab/remote_g4_driver.py",
        REPO_ROOT / "scripts/colab/g4_jobs/__init__.py",
        REPO_ROOT / "scripts/colab/g4_jobs/common.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_whisper_bvi_lora.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_forward_wire.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_jax_4d_hires_and_lc.py",
        REPO_ROOT / "scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py",
        REPO_ROOT / "scripts/bench_gtx1650s_m3_timesliced.py",
    ]
    forbidden_tokens = ("dummy_vram", "pad_vram", "reserve_vram", "/home/dias", "C:\\Users\\")
    for p in paths:
        assert p.is_file(), f"Missing required G4 Thunder module: {p}"
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"BOM found in {p}"
        assert b"\r" not in raw, f"CR found in {p}"
        text = raw.decode("utf-8")
        for tok in forbidden_tokens:
            assert tok not in text, f"Forbidden token {tok!r} found in {p}"
        tree = ast.parse(text, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "empty", f"Forbidden .empty() filler call in {p}"


def test_orchestrator_dry_run_lists_all_five_jobs_with_exact_wave_caps() -> None:
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/colab/run_g4_m3_thunder_run.py"), "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    dry = json.loads(res.stdout)
    _assert_no_nan_or_inf(dry)
    assert dry["mode"] == "DRY_RUN"
    assert dry["cu_consumed"] == 0.0
    assert dry["wave1_vram_sum"] == pytest.approx(0.88, abs=1e-4)
    assert dry["wave2_vram_sum"] == pytest.approx(0.88, abs=1e-4)

    w1_keys = {j["job_key"] for j in dry["waves"]["wave1"]}
    w2_keys = {j["job_key"] for j in dry["waves"]["wave2"]}
    assert w1_keys == {"qwen3_v3", "whisper_bvi", "forward_wire"}
    assert w2_keys == {"jax_4d_hires_lc", "pidnet_v21_xfeat"}

    jax_entry = next(j for j in dry["waves"]["wave2"] if j["job_key"] == "jax_4d_hires_lc")
    assert jax_entry["env_overrides"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert float(jax_entry["env_overrides"]["XLA_PYTHON_CLIENT_MEM_FRACTION"]) <= 0.58


def test_full_five_job_remote_driver_smoke_cpu_and_tu116_edge_gate(tmp_path: Path) -> None:
    """Execute ALL 5 G4 jobs through remote_g4_driver --smoke-cpu + TU116 edge bench at 0.00 CU."""
    art_dir = tmp_path / "artifacts"
    blob_dir = tmp_path / "blobs"

    # 1) Run ALL 5 jobs: Wave 1 (3 parallel) -> Drain Barrier -> Wave 2 (2 parallel)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/colab/remote_g4_driver.py"),
            "--smoke-cpu",
            "--artifacts-dir",
            str(art_dir),
            "--blob-dir",
            str(blob_dir),
        ],
        check=True,
    )

    bundle = json.loads((art_dir / "remote_bundle_manifest.json").read_text(encoding="utf-8"))
    _assert_no_nan_or_inf(bundle)
    assert bundle["state"] == "DONE"
    assert bundle["drain_barrier"]["drained"] is True
    job_keys = {r["job_key"] for r in bundle["job_results"]}
    assert job_keys == set(ALL_FIVE_JOBS)
    assert all(r["returncode"] == 0 for r in bundle["job_results"])

    # Verify SHA-256 parity of EVERY JSON and blob artifact listed in remote_bundle_manifest.json
    for jf in bundle["json_artifacts"]:
        p = art_dir / jf["filename"]
        assert p.is_file(), f"Missing JSON artifact: {p}"
        assert sha256_file(p) == jf["sha256"]
        _assert_no_nan_or_inf(json.loads(p.read_text(encoding="utf-8")), jf["filename"])
    for bf in bundle["blob_artifacts"]:
        p = blob_dir / bf["relpath"]
        assert p.is_file(), f"Missing blob artifact: {p}"
        assert sha256_file(p) == bf["sha256"]

    # 2) Verify Job 1 (Qwen3-4B v3 on VertiBench-v2)
    qwen = json.loads((art_dir / "g4_qwen3_v3_vertibench_v2_eval.json").read_text(encoding="utf-8"))
    assert qwen["mode"] == "SMOKE_CPU_ONLY"
    assert qwen["eval_split_provenance"] == "PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2"
    assert qwen["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert qwen["oom_guard_config"]["prompts_per_microbatch"] <= 2
    assert qwen["oom_guard_config"]["gradient_checkpointing"] is True
    assert qwen["oom_guard_config"]["completion_only_sliced_log_softmax"] is True
    assert qwen["liveness_soc_lt_5pct_override_verified"] is True
    assert qwen["mayday_priority_wm_015_verified"] is True
    assert qwen["conformal_language_conditional"]["TR"]["n_cal"] >= 750
    assert qwen["conformal_language_conditional"]["EN"]["n_cal"] >= 750
    assert "MIX" in qwen["conformal_language_conditional"]
    assert all(qwen["conformal_snr_mondrian"][s]["n_cal"] >= 250 for s in CANONICAL_SNR_STRATA)
    assert qwen["bootstrap_resplit_audit"]["n_resplits"] == 100

    # 3) Verify Job 2 (Whisper BVI LoRA)
    whisp = json.loads((art_dir / "g4_whisper_bvi_lora_eval.json").read_text(encoding="utf-8"))
    assert whisp["mode"] == "SMOKE_CPU_ONLY"
    assert whisp["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert whisp["whisper_inference_locks"]["translate"] is False
    assert whisp["whisper_inference_locks"]["min_audio_ctx"] == 256
    assert 112.0 <= whisp["coaxial_bvi_dsp_audit"]["max_rpm_bpf_hz"] <= 117.0
    assert set(whisp["snr_strata_metrics"].keys()) == set(CANONICAL_SNR_STRATA)

    # 4) Verify Job 3 (Forward Wire + D-FINE-N)
    fwd = json.loads((art_dir / "g4_forward_wire_metrics.json").read_text(encoding="utf-8"))
    assert fwd["mode"] == "SMOKE_CPU_ONLY"
    assert fwd["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert fwd["license_audit"]["train_source_names"] == ["VertiRTA_Procedural_Catenary_Horizon"]
    assert fwd["heldout_validation"]["val_samples"] >= 1
    assert fwd["split_stack_contract"]["round_trip_err_px"] < 0.5
    assert fwd["onnx_exports"]["mlsd_wireseg"]["onnx_checker_passed"] is True
    assert fwd["onnx_exports"]["dfine_n"]["onnx_checker_passed"] is True

    # 5) Verify Job 4 (JAX 4D HiRes 101x71 & Lift+Cruise 8+1)
    jax_m = json.loads((art_dir / "g4_jax_4d_hires_and_lc_manifest.json").read_text(encoding="utf-8"))
    assert jax_m["mode"] == "SMOKE_CPU_ONLY"
    assert jax_m["execution_mode"] == "sequential_per_regime_oom_safe"
    assert jax_m["lift_cruise_cd_r"] < 0.001
    assert jax_m["xla_mem_fraction"] <= 0.58
    assert jax_m["vmap_batch_size"] == 2
    assert jax_m["smoke_ratio_2p5_audit"]["passed"] is True
    assert jax_m["artifacts"]["x8_240_vrtalut1"]["size_bytes"] == 11313864
    assert jax_m["artifacts"]["x8_240_vrtalut1"]["magic"] == "VRTALUT1"
    assert jax_m["artifacts"]["x8_240_vrtalut2_hires"]["magic"] == "VRTALUT2"
    assert jax_m["artifacts"]["lift_cruise_8p1_vrtalut1"]["size_bytes"] == 11313864
    assert jax_m["artifacts"]["x8_240_cel2d"]["size_bytes"] == 11230
    assert jax_m["x8_soundness_passed"] is True
    assert jax_m["lc_soundness_passed"] is True
    assert jax_m["soundness_passed"] is True

    # 6) Verify Job 5 (PIDNet-S v2.1 + XFeat/MegaLoc Kartal 0.5m Reloc)
    pid = json.loads((art_dir / "g4_pidnet_v21_xfeat_metrics.json").read_text(encoding="utf-8"))
    assert pid["mode"] == "SMOKE_CPU_ONLY"
    assert pid["license_audit"]["zero_sharealike_verified"] is True
    assert pid["skyscenes_nested_tar_filter_active"] is True
    assert pid["onnx_exports"]["pidnet_slz_v21"]["deploy_parameters"] == 8003817
    assert pid["onnx_exports"]["xfeat_512"]["onnx_checker_passed"] is True
    assert pid["kartal_reloc_eval"]["cep50_m"] >= 0.0
    assert pid["kartal_reloc_eval"]["matcher_backend"]

    # 7) Run TU116 Time-Sliced Edge Benchmark in --smoke-cpu mode over all 4 ONNX exports
    bench_json = art_dir / "local_gtx1650s_m3_timesliced_proof.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/bench_gtx1650s_m3_timesliced.py"),
            "--smoke-cpu",
            "--blob-dir",
            str(blob_dir),
            "--artifacts-dir",
            str(art_dir),
            "--out-json",
            str(bench_json),
        ],
        check=True,
    )
    bench = json.loads(bench_json.read_text(encoding="utf-8"))
    _assert_no_nan_or_inf(bench)
    assert bench["gates"]["onnx_sha256_all_verified"] is True
    assert bench["gates"]["empirical_fp32_vs_fp16_logged"] is True
    assert bench["gates"]["all_trained_outputs_finite_and_nondegenerate"] is True
