# SPDX-License-Identifier: Apache-2.0
"""Tests for real moving drone temporal-stability stabilizer, CPU smoke, and GTX 1650 SUPER proof."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ci.check_public_hygiene import scan_public_hygiene
from scripts.realworld.run_real_drone_video_temporal_hud import (
    TemporalSlzStabilizer,
    _compute_binary_iou,
    main as temporal_hud_main,
    validate_permissive_license,
)

LIVE_PROOF_JSON = REPO_ROOT / "artifacts" / "v4.0.0" / "realworld_drone_video_temporal_proof.json"
README_PREVIEW_GIF = REPO_ROOT / "docs" / "assets" / "vertirta_real_drone_hud_preview.gif"


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


def _logits_for_safe_prob(p_safe: float, h: int = 32, w: int = 32) -> np.ndarray:
    """Construct 5-class logits whose softmax(T=1) yields exact total safe probability p_safe."""
    p = float(np.clip(p_safe, 1e-4, 1.0 - 1e-4))
    # Safe classes [0, 1] split p equally; hazard classes [2, 3, 4] split (1-p) equally.
    logits = np.zeros((5, h, w), dtype=np.float32)
    logits[0] = math.log(p / 2.0)
    logits[1] = math.log(p / 2.0)
    logits[2] = math.log((1.0 - p) / 3.0)
    logits[3] = math.log((1.0 - p) / 3.0)
    logits[4] = math.log((1.0 - p) / 3.0)
    return logits


def test_temporal_stabilizer_reduces_flicker_on_oscillating_logits() -> None:
    rng = np.random.default_rng(12345)
    h, w = 64, 64
    base = np.zeros((5, h, w), dtype=np.float32)
    # Left half marginally safe, right half marginally hazard, boundary near zero margin
    base[0, :, :32] = 0.65
    base[2, :, :32] = 0.40
    base[0, :, 32:] = 0.40
    base[2, :, 32:] = 0.65

    stab = TemporalSlzStabilizer(
        ema_alpha=0.25,
        p_enter_safe=0.60,
        p_exit_safe=0.40,
        motion_reset_px=35.0,
        temperature=1.0,
        min_slz_area_px=16,
        warp_with_homography=False,
    )

    prev_raw: np.ndarray | None = None
    prev_stab: np.ndarray | None = None
    raw_flickers: list[float] = []
    stab_flickers: list[float] = []
    raw_ious: list[float] = []
    stab_ious: list[float] = []

    for step in range(25):
        sign = 1.0 if step % 2 == 0 else -1.0
        noise = rng.normal(0.0, 0.75, size=(5, h, w)).astype(np.float32)
        logits = base + noise
        logits[0] += np.float32(0.35 * sign)
        logits[2] -= np.float32(0.35 * sign)

        raw_cls = np.argmax(logits, axis=0)
        raw_safe = (raw_cls == 0) | (raw_cls == 1)
        out = stab.update(logits, mean_flow_px=1.5)
        assert out.outputs_finite is True
        assert np.all(np.isfinite(out.ema_probs))

        if prev_raw is not None and prev_stab is not None:
            raw_flickers.append(float(np.mean(raw_safe != prev_raw) * 100.0))
            stab_flickers.append(float(np.mean(out.safe_mask != prev_stab) * 100.0))
            raw_ious.append(_compute_binary_iou(raw_safe, prev_raw))
            stab_ious.append(_compute_binary_iou(out.safe_mask, prev_stab))
        prev_raw = raw_safe
        prev_stab = out.safe_mask

    assert float(np.mean(stab_flickers)) < float(np.mean(raw_flickers)) * 0.45
    assert float(np.mean(stab_ious)) > float(np.mean(raw_ious))


def test_temporal_stabilizer_hysteresis_lock_and_motion_reset() -> None:
    stab = TemporalSlzStabilizer(
        ema_alpha=1.0,  # Pure hysteresis test without EMA lag
        p_enter_safe=0.65,
        p_exit_safe=0.40,
        motion_reset_px=30.0,
        temperature=1.0,
        min_slz_area_px=16,
        warp_with_homography=False,
    )

    # Start Hazard (p_safe=0.30 < 0.65)
    out0 = stab.update(_logits_for_safe_prob(0.30), mean_flow_px=0.0)
    assert not bool(out0.safe_mask.any())
    assert out0.motion_reset_triggered is False

    # Deadband rise (p_safe=0.55 < 0.65) must stay Hazard
    out1 = stab.update(_logits_for_safe_prob(0.55), mean_flow_px=2.0)
    assert not bool(out1.safe_mask.any())

    # Cross enter threshold (p_safe=0.72 >= 0.65) -> enters Safe
    out2 = stab.update(_logits_for_safe_prob(0.72), mean_flow_px=2.0)
    assert bool(out2.safe_mask.all())

    # Deadband dip (p_safe=0.48 >= 0.40) must remain locked Safe
    out3 = stab.update(_logits_for_safe_prob(0.48), mean_flow_px=2.0)
    assert bool(out3.safe_mask.all())

    # Drop below exit threshold (p_safe=0.32 < 0.40) -> exits Safe to Hazard
    out4 = stab.update(_logits_for_safe_prob(0.32), mean_flow_px=2.0)
    assert not bool(out4.safe_mask.any())

    # Now test motion-aware hard reset on a smoothed stabilizer (ema_alpha=0.20)
    stab_ema = TemporalSlzStabilizer(
        ema_alpha=0.20,
        p_enter_safe=0.62,
        p_exit_safe=0.42,
        motion_reset_px=30.0,
        temperature=1.0,
        min_slz_area_px=16,
        warp_with_homography=False,
    )
    a0 = stab_ema.update(_logits_for_safe_prob(0.90), mean_flow_px=0.0)
    assert bool(a0.safe_mask.all())
    # Small flow with sudden hazard frame -> EMA preserves Safe on single transient
    a1 = stab_ema.update(_logits_for_safe_prob(0.10), mean_flow_px=2.0)
    assert a1.motion_reset_triggered is False
    assert bool(a1.safe_mask.all())
    # Large homography displacement (>= 30 px) triggers immediate motion reset -> accepts new Hazard frame
    a2 = stab_ema.update(_logits_for_safe_prob(0.10), mean_flow_px=42.5)
    assert a2.motion_reset_triggered is True
    assert a2.effective_alpha == pytest.approx(1.0)
    assert stab_ema.reset_count == 1
    assert not bool(a2.safe_mask.any())


def test_temporal_stabilizer_handles_nonfinite_inputs_cleanly() -> None:
    stab = TemporalSlzStabilizer(warp_with_homography=False, min_slz_area_px=16)
    logits = np.zeros((5, 32, 32), dtype=np.float32)
    logits[0, 0:4, 0:4] = np.nan
    logits[1, 4:8, 4:8] = np.inf
    logits[2, 8:12, 8:12] = -np.inf
    out = stab.update(logits, mean_flow_px=float("nan"))
    assert out.outputs_finite is True
    assert np.all(np.isfinite(out.ema_probs))


def test_permissive_license_gate_fail_closed() -> None:
    for ok_lic in (
        "Public Domain (PD-USGov-NASA)",
        "CC0",
        "CC BY 4.0",
        "CC BY 3.0",
        "Apache-2.0",
    ):
        assert validate_permissive_license(ok_lic) is True

    for bad_lic in (
        "CC BY-SA 4.0",
        "CC BY-NC 4.0",
        "CC BY-NC-SA 3.0",
        "GPL-3.0",
        "All Rights Reserved",
        "",
    ):
        with pytest.raises(ValueError):
            validate_permissive_license(bad_lic)


def test_smoke_cpu_offline_zero_network(tmp_path: Path) -> None:
    out_json = tmp_path / "smoke_temporal_proof.json"
    rc = temporal_hud_main(
        [
            "--smoke-cpu",
            "--max-frames",
            "10",
            "--out-json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    data = _verify_canonical_sha256(out_json)
    assert data["schema_version"] == "vertirta.realworld_drone_video_temporal_proof/v4.0.0"
    assert data["execution_mode"] == "SMOKE_CPU_ONLY"
    assert data["colab_cu_spent"] == 0.0
    comp = data["temporal_stability_comparison"]
    assert comp["frames_evaluated"] == 10
    assert comp["safe_hazard_flicker_pixel_rate_pct"]["stabilized_mean_pct"] < comp["safe_hazard_flicker_pixel_rate_pct"]["raw_mean_pct"]
    assert data["gates"]["all_outputs_finite"] is True
    assert data["gates"]["zero_nan_inf"] is True
    assert data["gates"]["permissive_license_verified"] is True
    assert data["gates"]["temporal_flicker_reduced"] is True


def test_committed_live_gtx1650s_drone_temporal_proof_and_gif() -> None:
    assert LIVE_PROOF_JSON.is_file(), f"Missing live proof JSON: {LIVE_PROOF_JSON}"
    data = _verify_canonical_sha256(LIVE_PROOF_JSON)

    assert data["schema_version"] == "vertirta.realworld_drone_video_temporal_proof/v4.0.0"
    assert data["execution_mode"] == "LIVE_PHYSICAL_GTX1650S_CUDA"
    assert data["colab_cu_spent"] == 0.0
    assert "GTX 1650 SUPER" in str(data["gpu"]["name"])
    assert "CUDAExecutionProvider" in data["models"]["pidnet_slz_v21_local_gtx"]["active_providers"]
    assert data["models"]["pidnet_slz_v21_local_gtx"]["iobinding_enabled"] is True
    assert data["models"]["pidnet_slz_v21_local_gtx"]["filename"] == "pidnet_slz_v21_local_gtx_opset17.onnx"
    assert data["ood_and_bias_calibration"]["safe_logit_bias_delta"] == pytest.approx(-0.271367, abs=1e-5)

    # Permissive provenance lock
    prov = data["video_provenance"]
    assert prov["download_url"] and str(prov["download_url"]).startswith("https://")
    assert prov["sha256"] and len(prov["sha256"]) == 64
    assert data["license_audit"]["zero_sharealike_verified"] is True
    assert data["license_audit"]["zero_noncommercial_verified"] is True
    assert validate_permissive_license(str(data["license_audit"]["license"])) is True

    # Temporal stability metrics
    comp = data["temporal_stability_comparison"]
    assert comp["frames_evaluated"] >= 150
    flk = comp["safe_hazard_flicker_pixel_rate_pct"]
    assert flk["stabilized_mean_pct"] < flk["raw_mean_pct"]
    assert flk["relative_reduction_pct"] > 0.0
    iou = comp["safe_mask_iou_t_t1"]
    assert iou["stabilized_mean"] >= iou["raw_mean"]

    # Hardware performance finiteness
    hw = data["hardware_performance"]
    for key in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "fps_mean"):
        assert math.isfinite(float(hw["pidnet_cuda_latency_ms"][key])) and float(hw["pidnet_cuda_latency_ms"][key]) > 0.0
        assert math.isfinite(float(hw["full_stack_e2e_latency_ms"][key])) and float(hw["full_stack_e2e_latency_ms"][key]) > 0.0

    # Gates
    gates = data["gates"]
    assert gates["frames_evaluated_sufficient"] is True
    assert gates["all_outputs_finite"] is True
    assert gates["zero_nan_inf"] is True
    assert gates["cuda_ep_active"] is True
    assert gates["permissive_license_verified"] is True
    assert gates["temporal_flicker_reduced"] is True
    assert gates["temporal_safe_iou_improved"] is True
    assert gates["gif_under_900kb_if_rendered"] is True

    # Committed README preview GIF budget & SHA-256 check
    assert README_PREVIEW_GIF.is_file(), f"Missing README preview GIF: {README_PREVIEW_GIF}"
    gif_bytes = README_PREVIEW_GIF.read_bytes()
    assert gif_bytes[:6] in (b"GIF89a", b"GIF87a"), "Invalid GIF header"
    assert len(gif_bytes) < 900_000, f"Preview GIF exceeds 900 KB budget: {len(gif_bytes)} bytes"
    gif_sha = hashlib.sha256(gif_bytes).hexdigest()
    assert data["visual_renders"]["readme_preview_gif"]["rendered"] is True
    assert data["visual_renders"]["readme_preview_gif"]["sha256"] == gif_sha

    # Public hygiene gate on all newly created repo files
    hygiene = scan_public_hygiene(
        REPO_ROOT,
        tracked_files=[
            "scripts/realworld/run_real_drone_video_temporal_hud.py",
            "tests/realworld/test_real_drone_video_temporal_hud.py",
            "artifacts/v4.0.0/realworld_drone_video_temporal_proof.json",
        ],
    )
    assert hygiene["passed"] is True, f"Public hygiene violation: {hygiene['violations']}"
