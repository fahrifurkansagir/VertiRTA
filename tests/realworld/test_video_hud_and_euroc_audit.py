# SPDX-License-Identifier: Apache-2.0
"""Tests for real-world aerial video HUD demo and quarantined EuRoC MAV GNSS-denied audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_gtx1650s_m3_timesliced import select_platform_cuda_provider_options
from scripts.ci.check_public_hygiene import scan_public_hygiene
from scripts.realworld.run_euroc_mav_audit import (
    main as euroc_main,
    validate_euroc_quarantine,
)
from scripts.realworld.run_video_hud_demo import main as video_hud_main

LIVE_VIDEO_HUD_PROOF = REPO_ROOT / "artifacts" / "v4.0.0" / "realworld_video_hud_gtx1650s_proof.json"
LIVE_EUROC_AUDIT_PROOF = REPO_ROOT / "artifacts" / "v4.0.0" / "realworld_euroc_mav_gnss_audit.json"


def _verify_canonical_sha256(path: Path) -> dict:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM detected in {path}"
    assert b"\r" not in raw, f"CR/CRLF line ending detected in {path}"
    assert len(raw) < 1_000_000, f"Artifact exceeds 1 MB limit: {len(raw)} bytes"

    data = json.loads(raw.decode("utf-8"))
    assert "report_sha256" in data, f"Missing report_sha256 in {path}"
    expected = data["report_sha256"]
    body = {k: v for k, v in data.items() if k != "report_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    assert actual == expected, f"SHA-256 mismatch in {path}: actual={actual} != embedded={expected}"
    # Verify zero NaN/Inf across entire structure
    json.dumps(data, allow_nan=False)
    return data


def test_video_hud_smoke_cpu_offline_and_mp4(tmp_path: Path) -> None:
    import importlib.util

    has_cv2 = importlib.util.find_spec("cv2") is not None
    out_json = tmp_path / "smoke_video_hud.json"
    out_mp4 = tmp_path / "smoke_video_hud.mp4"

    rc = video_hud_main(
        [
            "--smoke-cpu",
            "--max-frames",
            "6",
            "--out-mp4",
            str(out_mp4),
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    data = _verify_canonical_sha256(out_json)
    assert data["schema_version"] == "vertirta.realworld_video_hud_gtx1650s_proof/v4.0.0"
    assert data["execution_mode"] == "SMOKE_CPU_ONLY"
    assert data["colab_cu_spent"] == 0.0
    assert data["frame_metrics_summary"]["real_frames_evaluated"] == 6
    if has_cv2:
        assert out_mp4.is_file() and out_mp4.stat().st_size > 0
        assert data["hud_render"]["rendered"] is True
        assert data["hud_render"]["size_bytes"] == out_mp4.stat().st_size
        assert len(data["hud_render"]["sha256"]) == 64
    else:
        assert data["hud_render"]["rendered"] is False
        assert data["models"]["optical_tracker_backend"] == "numpy_smoke_tracker_no_cv2"

    caps = data["frame_metrics_summary"]["cbvf_speed_cap_distribution"]
    assert caps["v_max_15_0_mps_frames"] + caps["v_max_22_0_mps_frames"] == 6
    assert data["gates"]["all_outputs_finite"] is True
    assert data["gates"]["zero_nan_inf"] is True
    assert data["gates"]["license_permissive_verified"] is True
    assert data["gates"]["hud_mp4_rendered_if_requested"] is True


def test_euroc_audit_smoke_cpu_offline(tmp_path: Path) -> None:
    out_json = tmp_path / "smoke_euroc_audit.json"
    rc = euroc_main(["--smoke-cpu", "--out-json", str(out_json)])
    assert rc == 0
    assert out_json.is_file()

    data = _verify_canonical_sha256(out_json)
    assert data["schema_version"] == "vertirta.realworld_euroc_mav_gnss_audit/v4.0.0"
    assert data["execution_mode"] == "SMOKE_CPU_ONLY"
    assert data["colab_cu_spent"] == 0.0
    assert data["all_gates_passed"] is True
    assert data["license_audit"]["quarantined_zero_shot_eval_only"] is True
    assert data["license_audit"]["train_source_names"] == []

    reg_a = data["regimes"]["regime_a_clean_flight"]
    assert reg_a["false_alarms"] == 0
    assert reg_a["final_hypothesis"] == "H0_NOMINAL"

    reg_b = data["regimes"]["regime_b_slow_ramp_spoof"]
    assert reg_b["detected"] is True
    assert reg_b["detected_hypothesis"] == "H3_SLOW_RAMP_SPOOF"
    assert reg_b["ttd_s"] is not None and reg_b["ttd_s"] <= 1.4
    assert reg_b["rollback_executed"] is True
    assert reg_b["ctr_inflation"]["monotonic_3sigma_growth"] is True
    assert reg_b["ctr_inflation"]["ctr_breaches"] == 0

    reg_c = data["regimes"]["regime_c_jam_and_step"]
    assert reg_c["rf_jamming_c1"]["detected_hypothesis"] == "H1_RF_JAMMING"
    assert reg_c["rf_jamming_c1"]["passed"] is True
    assert reg_c["step_meaconing_c2"]["detected_hypothesis"] == "H2_STEP_MEACONING"
    assert reg_c["step_meaconing_c2"]["passed"] is True
    assert reg_c["gate_deterministic_discrimination"] is True


def test_euroc_license_quarantine_fail_closed() -> None:
    ok_audit = validate_euroc_quarantine(
        sequence="V1_01_easy",
        quarantined_zero_shot_eval_only=True,
        as_train_source=False,
    )
    assert ok_audit["quarantined_zero_shot_eval_only"] is True
    assert ok_audit["train_source_names"] == []
    assert ok_audit["eval_only_source_names"] == ["EuRoC_MAV_V1_01_easy"]

    with pytest.raises(ValueError, match="quarantined_zero_shot_eval_only=True"):
        validate_euroc_quarantine(
            sequence="V1_01_easy",
            quarantined_zero_shot_eval_only=False,
            as_train_source=False,
        )

    with pytest.raises(ValueError, match="Non-permissive license"):
        validate_euroc_quarantine(
            sequence="V1_01_easy",
            quarantined_zero_shot_eval_only=True,
            as_train_source=True,
        )


def test_platform_cuda_provider_options_contract() -> None:
    win_opts = select_platform_cuda_provider_options(system_name="Windows", enable_cuda_graph=True)
    assert win_opts["cudnn_conv_algo_search"] == "EXHAUSTIVE"
    assert win_opts["cudnn_conv_use_max_workspace"] == "1"
    assert win_opts["enable_cuda_graph"] == "1"

    lin_opts = select_platform_cuda_provider_options(system_name="Linux", enable_cuda_graph=True)
    assert lin_opts["cudnn_conv_algo_search"] == "HEURISTIC"
    assert lin_opts["cudnn_conv_use_max_workspace"] == "0"
    assert lin_opts["enable_cuda_graph"] == "1"


def test_committed_live_video_hud_gtx1650s_proof_artifact() -> None:
    assert LIVE_VIDEO_HUD_PROOF.is_file(), f"Missing live proof: {LIVE_VIDEO_HUD_PROOF}"
    data = _verify_canonical_sha256(LIVE_VIDEO_HUD_PROOF)

    assert data["schema_version"] == "vertirta.realworld_video_hud_gtx1650s_proof/v4.0.0"
    assert data["execution_mode"] == "LIVE_PHYSICAL_GTX1650S_CUDA"
    assert data["colab_cu_spent"] == 0.0
    assert "GTX 1650 SUPER" in str(data["gpu"]["name"])
    assert "CUDAExecutionProvider" in data["models"]["pidnet_slz"]["active_providers"]
    assert data["models"]["pidnet_slz"]["sha256"] and len(data["models"]["pidnet_slz"]["sha256"]) == 64

    summary = data["frame_metrics_summary"]
    assert summary["real_frames_evaluated"] >= 30
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps_mean"):
        val = float(summary["network_latency_ms"][key])
        assert math.isfinite(val) and val > 0.0
        val_e2e = float(summary["e2e_frame_latency_ms"][key])
        assert math.isfinite(val_e2e) and val_e2e > 0.0

    gates = data["gates"]
    assert gates["real_frames_ge_30"] is True
    assert gates["all_outputs_finite"] is True
    assert gates["zero_nan_inf"] is True
    assert gates["cuda_ep_active"] is True
    assert gates["license_permissive_verified"] is True
    assert gates["hud_mp4_rendered_if_requested"] is True

    hygiene = scan_public_hygiene(
        REPO_ROOT,
        tracked_files=["artifacts/v4.0.0/realworld_video_hud_gtx1650s_proof.json"],
    )
    assert hygiene["passed"] is True, f"Public hygiene violation: {hygiene['violations']}"


def test_committed_live_euroc_mav_gnss_audit_artifact() -> None:
    assert LIVE_EUROC_AUDIT_PROOF.is_file(), f"Missing live EuRoC audit: {LIVE_EUROC_AUDIT_PROOF}"
    data = _verify_canonical_sha256(LIVE_EUROC_AUDIT_PROOF)

    assert data["schema_version"] == "vertirta.realworld_euroc_mav_gnss_audit/v4.0.0"
    assert data["execution_mode"] == "LIVE_REAL_EUROC_MAV_ZERO_SHOT"
    assert data["colab_cu_spent"] == 0.0
    assert data["license_audit"]["quarantined_zero_shot_eval_only"] is True
    assert data["license_audit"]["train_source_names"] == []
    assert data["dataset_provenance"]["license"] == "CC BY-SA 4.0"
    assert data["dataset_provenance"]["imu_csv_sha256"] and len(data["dataset_provenance"]["imu_csv_sha256"]) == 64
    assert data["dataset_provenance"]["gt_csv_sha256"] and len(data["dataset_provenance"]["gt_csv_sha256"]) == 64

    tel = data["real_flight_telemetry"]
    assert tel["imu_csv_rows"] >= 20_000
    assert tel["gt_csv_rows"] >= 20_000
    assert tel["flight_duration_s"] >= 100.0
    assert tel["imu_measured_rate_hz"] >= 190.0

    reg_a = data["regimes"]["regime_a_clean_flight"]
    assert reg_a["false_alarms"] == 0
    assert reg_a["gate_zero_false_alarms"] is True

    reg_b = data["regimes"]["regime_b_slow_ramp_spoof"]
    assert reg_b["detected_hypothesis"] == "H3_SLOW_RAMP_SPOOF"
    assert reg_b["ttd_s"] is not None and reg_b["ttd_s"] <= 1.4
    assert reg_b["rollback_executed"] is True
    assert reg_b["ctr_inflation"]["monotonic_3sigma_growth"] is True
    assert reg_b["ctr_inflation"]["ctr_breaches"] == 0

    reg_c = data["regimes"]["regime_c_jam_and_step"]
    assert reg_c["rf_jamming_c1"]["detected_hypothesis"] == "H1_RF_JAMMING"
    assert reg_c["step_meaconing_c2"]["detected_hypothesis"] == "H2_STEP_MEACONING"
    assert reg_c["gate_deterministic_discrimination"] is True
    assert data["all_gates_passed"] is True

    hygiene = scan_public_hygiene(
        REPO_ROOT,
        tracked_files=["artifacts/v4.0.0/realworld_euroc_mav_gnss_audit.json"],
    )
    assert hygiene["passed"] is True, f"Public hygiene violation: {hygiene['violations']}"
