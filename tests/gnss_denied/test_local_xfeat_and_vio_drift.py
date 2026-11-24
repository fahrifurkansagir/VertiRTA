# SPDX-License-Identifier: Apache-2.0
"""Verification tests for Live Physical GTX 1650 SUPER XFeat Proof & 6-DOF VIO Drift Audit (E3)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vio_drift_audit.run_vio_drift_audit import (  # noqa: E402
    LTBX_ARP_LAT_DEG,
    LTBX_ARP_LON_DEG,
    run_full_vio_drift_audit,
)

LIVE_PROOF_PATH = REPO_ROOT / "artifacts" / "v4.0.0" / "local_gtx1650s_xfeat_live_proof.json"
VIO_REPORT_PATH = REPO_ROOT / "benchmarks" / "vio_drift_audit" / "vio_drift_audit_report.json"
BENCH_SCRIPT = REPO_ROOT / "scripts" / "bench_local_gtx1650s_xfeat_live.py"


def _assert_no_nan_inf(obj: Any) -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float found: {obj}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nan_inf(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_nan_inf(v)


def test_bench_local_xfeat_smoke_cpu_runs_cleanly(tmp_path: Path) -> None:
    """CPU-only CI smoke test verifies the full XFeat + VIO drift pipeline with 0 GPU and 0 CU."""
    out_json = tmp_path / "smoke_xfeat.json"
    vio_json = tmp_path / "smoke_vio.json"
    cache_dir = tmp_path / "cache"
    proc = subprocess.run(
        [
            sys.executable,
            str(BENCH_SCRIPT),
            "--smoke-cpu",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_json),
            "--vio-out",
            str(vio_json),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    assert out_json.is_file()
    assert vio_json.is_file()

    data = json.loads(out_json.read_text(encoding="utf-8"))
    _assert_no_nan_inf(data)
    assert data["execution_mode"] == "CPU_SMOKE"
    assert data["colab_cu_spent"] == 0.0
    assert data["relocalization_summary"]["total_pairs_evaluated"] >= 16
    assert data["vio_drift_audit_summary"]["all_gates_passed"] is True
    assert data["vio_drift_audit_summary"]["total_ctr_breaches_all_regimes"] == 0


def test_vio_drift_audit_all_three_regimes_and_geodesy_locks(tmp_path: Path) -> None:
    """Verify 6-DOF Basalt-aligned Stereo-IMU VIO + Marmara water-mask + 1 Hz XFeat + 3sigma+dh/dt CTR."""
    assert LTBX_ARP_LAT_DEG == pytest.approx(40.990556, abs=1e-6)
    assert LTBX_ARP_LON_DEG == pytest.approx(29.216111, abs=1e-6)
    assert abs(LTBX_ARP_LAT_DEG - 40.99889) > 0.005  # Reject stale DR-10 LTBX draft typo

    proof_arg = LIVE_PROOF_PATH if LIVE_PROOF_PATH.is_file() else None
    out_path = tmp_path / "vio_audit.json"
    report = run_full_vio_drift_audit(xfeat_proof_json=proof_arg, out_json=out_path)
    _assert_no_nan_inf(report)

    assert report["airspace_lock"]["basalt_gnss_feedback_allowed"] is False
    assert report["airspace_lock"]["stereo_baseline_m"] == pytest.approx(0.80)
    assert report["all_gates_passed"] is True

    reg_a = report["regimes"]["regime_a_xfeat_1hz_bounded_cruise"]
    assert reg_a["max_3sigma_m"] < 25.0
    assert reg_a["ltfj_breaches"] == 0
    assert reg_a["ltbx_breaches"] == 0
    assert reg_a["water_features_purged"] > 0

    reg_b = report["regimes"]["regime_b_unbounded_vio_marmara_abort"]
    assert reg_b["caution_entered"] is True
    assert reg_b["visual_slz_abort_latched"] is True
    assert 65.0 <= reg_b["max_1sigma_m"] < 80.0
    assert reg_b["ltfj_breaches"] == 0
    assert reg_b["ltbx_breaches"] == 0
    assert reg_b["max_ctr_r_dot_infl_mps"] > 0.0

    reg_c = report["regimes"]["regime_c_slow_ramp_spoof_and_water_glint"]
    assert reg_c["spoof_latched"] is True
    assert reg_c["spoof_ttd_s"] <= 1.4
    assert reg_c["post_rollback_slip_m"] <= 0.85
    assert reg_c["water_glint_false_alarms"] == 0
    assert reg_c["ltfj_breaches"] == 0
    assert reg_c["ltbx_breaches"] == 0


def test_live_gtx1650s_xfeat_proof_artifact_verified() -> None:
    """Assert committed physical GTX 1650 SUPER XFeat + VIO proof artifact meets all E3 locks."""
    assert LIVE_PROOF_PATH.is_file(), f"Missing live GTX 1650 SUPER proof artifact: {LIVE_PROOF_PATH}"
    assert VIO_REPORT_PATH.is_file(), f"Missing VIO drift audit artifact: {VIO_REPORT_PATH}"

    raw_text = LIVE_PROOF_PATH.read_text(encoding="utf-8")
    assert "\r" not in raw_text
    assert not re.search(r"/home/[a-z0-9_-]+|C:\\Users\\[A-Za-z0-9_-]+", raw_text)

    data = json.loads(raw_text)
    _assert_no_nan_inf(data)

    # Hardware & Zero-CU Provenance
    assert data["execution_mode"] == "LIVE_PHYSICAL_GTX1650S_CUDA"
    assert "GTX 1650 SUPER" in data["gpu"]["name"]
    assert data["colab_cu_spent"] == 0.0
    assert data["model"]["license"] == "Apache-2.0"
    assert re.fullmatch(r"[0-9a-f]{64}", str(data["model"]["weights_sha256"]))
    assert data["license_audit"]["zero_sharealike_verified"] is True

    # Real permissive aerial orthophoto dataset count & >= 100 real pairs evaluated
    prov = data["dataset_provenance"]
    real_tiles = int(prov.get("flair1", {}).get("tiles_loaded", 0)) + int(
        prov.get("openearthmap_ccby4", {}).get("tiles_loaded", 0)
    )
    assert real_tiles >= 16
    reloc = data["relocalization_summary"]
    assert reloc["total_pairs_evaluated"] >= 100

    # Empirical FP32 vs FP16 latency + < 300 MiB TU116 CUDA peak VRAM lock
    lv = data["latency_and_vram"]
    assert lv["override_6_empirical_check_done"] is True
    for mode_key in ("fp32", "fp16_autocast"):
        assert lv[mode_key]["p50_ms"] > 0.0
        assert lv[mode_key]["p99_ms"] > 0.0
        assert lv[mode_key]["cuda_peak_allocated_mib"] < 300.0
        assert lv[mode_key]["cuda_peak_reserved_mib"] < 300.0
        assert lv[mode_key]["within_300mib_vram_budget"] is True

    # Real geometric accuracy & water-mask benefit
    overall = reloc["overall_water_masked"]
    assert overall["ransac_3gate_pass_rate"] >= 0.70
    assert overall["cep50_m"] < 2.0
    assert overall["ce95_m"] < 5.0

    # Chained VIO drift audit gates all PASS with 0 CTR breaches
    vio_sum = data["vio_drift_audit_summary"]
    assert vio_sum["all_gates_passed"] is True
    assert vio_sum["regime_a_max_3sigma_m"] < 25.0
    assert 65.0 <= vio_sum["regime_b_abort_1sigma_m"] < 80.0
    assert vio_sum["regime_c_spoof_ttd_s"] <= 1.4
    assert vio_sum["regime_c_water_glint_far"] == 0
    assert vio_sum["total_ctr_breaches_all_regimes"] == 0
