# SPDX-License-Identifier: Apache-2.0
"""Tests for the 4-scene real-world multi-video aerial perception stress test."""

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

from scripts.ci.check_public_hygiene import scan_public_hygiene
from scripts.realworld.run_multivideo_stress_test import (
    MULTIVIDEO_SCENES,
    describe_camera_angle_plain,
    main as multivideo_main,
)
from scripts.realworld.run_real_drone_video_temporal_hud import (
    validate_depth_model_license,
    validate_permissive_license,
)

LIVE_MULTIVIDEO_PROOF = REPO_ROOT / "artifacts" / "v4.0.0" / "realworld_multivideo_stress_proof.json"


def _verify_canonical_sha256(path: Path) -> dict:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM detected in {path}"
    assert b"\r" not in raw, f"CR/CRLF detected in {path}"
    assert len(raw) < 1_000_000, f"Artifact exceeds 1 MB limit: {len(raw)} bytes"

    data = json.loads(raw.decode("utf-8"))
    assert "report_sha256" in data, f"Missing report_sha256 in {path}"
    expected = data["report_sha256"]
    body = {k: v for k, v in data.items() if k != "report_sha256"}
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    assert actual == expected, f"SHA-256 mismatch in {path}: actual={actual} != embedded={expected}"
    json.dumps(data, allow_nan=False)
    return data


def test_multivideo_scene_registry_strictly_permissive_and_plain_labels() -> None:
    assert len(MULTIVIDEO_SCENES) == 4
    for sc in MULTIVIDEO_SCENES:
        prov = sc["provenance"]
        assert validate_permissive_license(str(prov["license"])) is True
        assert prov["zero_sharealike_verified"] is True
        assert prov["zero_noncommercial_verified"] is True
        assert str(prov["download_url"]).startswith("https://")
    assert "Top-Down" in describe_camera_angle_plain(84.0)
    assert "Angled Down" in describe_camera_angle_plain(50.0)
    assert "Forward View" in describe_camera_angle_plain(16.0)
    assert "Top-Down" in describe_camera_angle_plain(float("nan"))


def test_multivideo_smoke_cpu_offline_zero_network(tmp_path: Path) -> None:
    out_json = tmp_path / "smoke_multivideo.json"
    demo_dir = tmp_path / "demos"
    rc = multivideo_main(
        [
            "--smoke-cpu",
            "--max-frames",
            "6",
            "--out-demo-dir",
            str(demo_dir),
            "--out-json",
            str(out_json),
            "--no-videos",
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    data = _verify_canonical_sha256(out_json)
    assert data["schema_version"] == "vertirta.realworld_multivideo_stress_proof/v4.0.0"
    assert data["execution_mode"] == "SMOKE_CPU_ONLY"
    assert data["colab_cu_spent"] == 0.0
    assert len(data["scenes"]) == 4

    by_id = {s["scene_id"]: s for s in data["scenes"]}
    assert by_id["scene_1_oblique_urban"]["camera_geometry_and_3d_veto"]["safe_pixels_vetoed_to_hazard_frac_mean"] >= 0.20
    assert by_id["scene_2_topdown_nadir"]["camera_geometry_and_3d_veto"]["estimated_camera_pitch_deg_mean"] >= 75.0
    assert by_id["scene_2_topdown_nadir"]["camera_geometry_and_3d_veto"]["safe_pixels_vetoed_to_hazard_frac_mean"] < 0.05
    assert by_id["scene_4_forward_horizon"]["forward_detectors_and_domain_gate"]["frames_in_domain"] == 6

    gates = data["gates"]
    assert gates["all_4_scenes_evaluated"] is True
    assert gates["all_outputs_finite"] is True
    assert gates["zero_nan_inf"] is True
    assert gates["all_licenses_permissive"] is True
    assert gates["oblique_urban_facade_veto_ge_20pct"] is True
    assert gates["topdown_nadir_no_false_collapse"] is True
    assert gates["forward_domain_gate_exercised_both_states"] is True
    assert gates["temporal_flicker_reduced_all_scenes"] is True


def test_committed_live_gtx1650s_multivideo_stress_proof() -> None:
    assert LIVE_MULTIVIDEO_PROOF.is_file(), f"Missing committed multi-video proof: {LIVE_MULTIVIDEO_PROOF}"
    data = _verify_canonical_sha256(LIVE_MULTIVIDEO_PROOF)

    assert data["schema_version"] == "vertirta.realworld_multivideo_stress_proof/v4.0.0"
    assert data["release_version"] == "v1.0.0"
    assert data["execution_mode"] == "LIVE_PHYSICAL_GTX1650S_CUDA"
    assert data["colab_cu_spent"] == 0.0
    assert "GTX 1650 SUPER" in str(data["gpu"]["name"])

    # License audit
    lic = data["license_audit"]
    assert lic["all_videos_permissive_verified"] is True
    assert lic["zero_sharealike_verified"] is True
    assert lic["zero_noncommercial_verified"] is True
    assert lic["depth_model_permissive_gate_passed"] is True
    assert validate_depth_model_license(str(lic["depth_model_id"]), str(lic["depth_model_license"])) is True

    # Locked local GTX ONNX models
    models = data["models"]
    assert models["pidnet_slz_v21_local_gtx"]["filename"] == "pidnet_slz_v21_local_gtx_opset17.onnx"
    assert "CUDAExecutionProvider" in models["pidnet_slz_v21_local_gtx"]["active_providers"]
    assert models["depth_anything_v2_small"]["license"] == "Apache-2.0"
    assert "CUDAExecutionProvider" in models["depth_anything_v2_small"]["active_providers"]
    assert models["mlsd_wireseg_v11"]["active"] is True
    assert models["dfine_n_v11"]["active"] is True

    # All 4 real scenes verified
    assert len(data["scenes"]) == 4
    for sc in data["scenes"]:
        prov = sc["video_provenance"]
        assert validate_permissive_license(str(prov["license"])) is True
        assert prov["sha256"] and len(prov["sha256"]) == 64
        assert sc["temporal_stability"]["frames_evaluated"] >= 120
        assert sc["honest_failure_and_limitation_notes"]
        hw = sc["hardware_performance"]
        assert math.isfinite(float(hw["pidnet_fps_mean"])) and float(hw["pidnet_fps_mean"]) > 0.0
        assert math.isfinite(float(hw["depth_3d_fps_mean"])) and float(hw["depth_3d_fps_mean"]) > 0.0
        assert sc["visual_outputs"]["h264_mp4_rendered"] is True

    # Cross-scene gates
    gates = data["gates"]
    assert gates["all_4_scenes_evaluated"] is True
    assert gates["all_outputs_finite"] is True
    assert gates["zero_nan_inf"] is True
    assert gates["all_licenses_permissive"] is True
    assert gates["cuda_ep_active"] is True
    assert gates["depth_3d_cuda_ep_active"] is True
    assert gates["oblique_urban_facade_veto_ge_20pct"] is True
    assert gates["topdown_nadir_no_false_collapse"] is True
    assert gates["forward_domain_gate_exercised_both_states"] is True
    assert gates["temporal_flicker_reduced_all_scenes"] is True
    assert gates["demo_outputs_rendered_if_requested"] is True

    # Combined visual outputs rendered cleanly
    vis = data["combined_visual_outputs"]
    assert vis["highlight_reel_mp4"]["rendered"] is True
    assert vis["diagnostic_contact_sheet_png"]["rendered"] is True
    assert vis["plain_english_overlay_standard"] is True

    # Public hygiene gate on all owned files
    hygiene = scan_public_hygiene(
        REPO_ROOT,
        tracked_files=[
            "scripts/realworld/run_multivideo_stress_test.py",
            "tests/realworld/test_multivideo_stress_test.py",
            "artifacts/v4.0.0/realworld_multivideo_stress_proof.json",
        ],
    )
    assert hygiene["passed"] is True, f"Public hygiene violation: {hygiene['violations']}"
