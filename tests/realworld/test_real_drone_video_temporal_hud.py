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
    DEFAULT_DAV2_MODEL_ID,
    DEFAULT_GIF_MAX_BYTES,
    SLZ_CLASS_BGR,
    SLZ_CLASS_RGB,
    NonRigidSurfaceMotionTracker,
    TemporalSlzStabilizer,
    _compute_binary_iou,
    apply_3d_normal_and_ood_veto,
    compute_3d_surface_normals_from_disparity,
    evaluate_forward_domain_gate,
    main as temporal_hud_main,
    validate_depth_model_license,
    validate_permissive_license,
)
from scripts.train_local_gtx1650s_pidnet_v21_recovery import (
    SLZ_CLASS_RGB as GRID_SLZ_CLASS_RGB,
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


def test_depth_model_apache2_allowlist_fail_closed() -> None:
    assert validate_depth_model_license(DEFAULT_DAV2_MODEL_ID, "Apache-2.0") is True
    for bad_id, bad_lic in (
        ("depth-anything/Depth-Anything-V2-Base-hf", "CC-BY-NC-4.0"),
        ("depth-anything/Depth-Anything-V2-Large-hf", "Apache-2.0"),
        ("depth-anything/Depth-Anything-V2-Small-hf", "CC-BY-NC-4.0"),
        ("depth-anything/Depth-Anything-V2-Small-hf", "GPL-3.0"),
    ):
        with pytest.raises(ValueError):
            validate_depth_model_license(bad_id, bad_lic)


def test_3d_surface_normal_and_ood_veto_rejects_vertical_facade_and_preserves_flat_roof() -> None:
    # Construct oblique aerial scene: horizontal plane has disparity increasing downward (+v),
    # while a vertical building facade in [48:96, 48:96] slopes away moving downward (dZ/dv > 0).
    h, w = 128, 128
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    disp = np.broadcast_to(0.15 + 0.70 * yy, (h, w)).copy()
    disp[48:96, 48:96] = 0.65 - 0.35 * np.linspace(0.0, 1.0, 48, dtype=np.float32)[:, None]

    n3d = compute_3d_surface_normals_from_disparity(disp, blur_ksize=5, blur_sigma=1.0)
    assert n3d["outputs_finite"] is True
    assert 25.0 <= n3d["camera_pitch_deg"] <= 85.0
    # Interior of flat horizontal roof/ground must NOT be steep
    assert float(n3d["steep_soft_mask"][10:36, 10:118].mean()) < 0.05
    # Interior of vertical building facade MUST be flagged steep (> 90%)
    assert float(n3d["steep_soft_mask"][56:88, 56:88].mean()) > 0.90

    # Raw 2D logits falsely predict Safe_Flat_Ground (0) across flat ground + vertical facade,
    # and Hazard_Water_Vegetation (4) in a separate tree block [100:120, 10:40].
    logits = np.full((5, h, w), -2.0, dtype=np.float32)
    logits[0, :, :] = 3.0
    logits[0, 100:120, 10:40] = -2.0
    logits[4, 100:120, 10:40] = 4.0

    ood_mask = np.zeros((h, w), dtype=bool)
    ood_mask[16:30, 16:30] = True  # OOD anomaly sitting on flat roof

    veto = apply_3d_normal_and_ood_veto(
        logits,
        safe_logit_bias_delta=-0.271367,
        steep_soft_mask=n3d["steep_soft_mask"],
        steep_hard_mask=n3d["steep_hard_mask"],
        ood_mask=ood_mask,
        min_slz_area_px=16,
    )
    # Vertical facade & OOD anomaly flipped fail-closed to Hazard_Obstacle_Structure (2)
    assert bool((veto["vetoed_raw_pred"][56:88, 56:88] == 2).all())
    assert bool((veto["vetoed_raw_pred"][16:30, 16:30] == 2).all())
    # Flat horizontal area outside OOD remains Safe (0)
    assert bool((veto["vetoed_raw_pred"][12:34, 60:115] == 0).all())
    # Existing Water/Veg hazard subclass (4) preserved
    assert bool((veto["vetoed_raw_pred"][102:118, 12:38] == 4).all())

    # TemporalSlzStabilizer hard_veto_mask overrides locked Safe immediately (zero hysteresis lag)
    stab = TemporalSlzStabilizer(
        ema_alpha=0.20, p_enter_safe=0.60, p_exit_safe=0.40, min_slz_area_px=16, warp_with_homography=False
    )
    out0 = stab.update(_logits_for_safe_prob(0.92, h=32, w=32), mean_flow_px=0.0)
    assert bool(out0.safe_mask.all())
    hard_wall = np.zeros((32, 32), dtype=bool)
    hard_wall[8:24, 8:24] = True
    out1 = stab.update(_logits_for_safe_prob(0.92, h=32, w=32), mean_flow_px=1.0, hard_veto_mask=hard_wall)
    assert not bool(out1.safe_mask[8:24, 8:24].any())
    assert bool(out1.safe_mask[:6, :6].all())


def test_forward_domain_gate_and_unified_palette_parity() -> None:
    assert evaluate_forward_domain_gate(51.0, max_forward_pitch_deg=25.0)["in_domain"] is False
    assert evaluate_forward_domain_gate(10.0, max_forward_pitch_deg=25.0)["in_domain"] is True
    assert np.array_equal(SLZ_CLASS_RGB, GRID_SLZ_CLASS_RGB), "HUD and validation grid palettes diverged"
    assert np.array_equal(SLZ_CLASS_BGR, SLZ_CLASS_RGB[:, ::-1]), "BGR/RGB palette mismatch"


def test_non_rigid_surface_motion_tracker_blocks_churning_surf_and_preserves_static_roof() -> None:
    """First-principles rigid-surface motion veto must preserve static concrete and reject churning surf."""
    h, w = 128, 128
    rng = np.random.default_rng(20261127)
    base_static = np.full((h, w), 140.0, dtype=np.float32)
    # Add mild rigid texture across the whole frame (static concrete)
    base_static += rng.normal(0.0, 2.0, (h, w)).astype(np.float32)
    base_static = np.clip(base_static, 0.0, 255.0)

    tracker = NonRigidSurfaceMotionTracker(
        ema_alpha=0.25, tau_nonrigid_luma=7.5, warmup_steps=4, blur_ksize=11, blur_sigma=2.5
    )
    eye_h = np.eye(3, dtype=np.float64)

    # 1) Static rigid surface with small camera noise -> 0% non-rigid veto
    for _ in range(8):
        frame = np.clip(base_static + rng.normal(0.0, 1.0, (h, w)), 0.0, 255.0).astype(np.uint8)
        res = tracker.update(frame, homography=eye_h)
        assert res["outputs_finite"] is True
    assert res["non_rigid_frac"] == pytest.approx(0.0, abs=1e-6)

    # 2) Churning ocean surf / dynamic patch on left half [16:112, 8:56], static roof on right half [16:112, 72:120]
    tracker.reset()
    for step in range(12):
        frame = base_static.copy()
        surf_patch = rng.uniform(110.0, 250.0, size=(96, 48)).astype(np.float32)
        if step % 2 == 1:
            surf_patch = 255.0 - 0.5 * surf_patch
        frame[16:112, 8:56] = surf_patch
        res = tracker.update(np.clip(frame, 0.0, 255.0).astype(np.uint8), homography=eye_h)
        assert res["outputs_finite"] is True

    nr_mask = res["non_rigid_mask"]
    assert float(nr_mask[24:104, 16:48].mean()) > 0.90, "Churning surf region must be vetoed as non-rigid"
    assert float(nr_mask[24:104, 80:112].mean()) < 0.02, "Static concrete roof must remain preserved"

    # Verify apply_3d_normal_and_ood_veto routes non_rigid_mask pixels from Safe -> Hazard (class 2)
    logits = np.full((5, h, w), -2.5, dtype=np.float32)
    logits[1, :, :] = 3.5  # Raw 2D falsely marks everything Safe_Flat_Roof (1)
    veto = apply_3d_normal_and_ood_veto(
        logits,
        safe_logit_bias_delta=-0.271367,
        steep_soft_mask=np.zeros((h, w), dtype=bool),
        steep_hard_mask=np.zeros((h, w), dtype=bool),
        ood_mask=np.zeros((h, w), dtype=bool),
        non_rigid_mask=nr_mask,
        min_slz_area_px=16,
    )
    assert veto["safe_vetoed_by_nonrigid_frac"] > 0.25
    assert bool((veto["vetoed_raw_pred"][24:104, 16:48] == 2).mean() > 0.90)
    assert bool((veto["vetoed_raw_pred"][24:104, 80:112] == 1).mean() > 0.98)

    # 3) Non-finite input must fail closed (100% non-rigid veto, outputs finite)
    nan_frame = base_static.copy()
    nan_frame[0, 0] = np.nan
    fail_res = tracker.update(nan_frame, homography=eye_h)
    assert fail_res["outputs_finite"] is True
    assert fail_res["non_rigid_frac"] == pytest.approx(1.0, abs=1e-6)


def test_3d_normal_veto_pure_nadir_flat_roof_zero_false_veto() -> None:
    """Pure top-down (90 deg nadir) flat roof must yield pitch ~90 deg and ~0% false veto."""
    h, w = 128, 128
    disp_const = np.full((h, w), 0.75, dtype=np.float32)
    n3d = compute_3d_surface_normals_from_disparity(disp_const, blur_ksize=5, blur_sigma=1.0)
    assert n3d["outputs_finite"] is True
    assert n3d["disp_input_all_finite"] is True
    assert n3d["camera_pitch_deg"] >= 88.0
    assert float(n3d["u_up"][2]) <= -0.99
    assert n3d["steep_soft_frac"] == pytest.approx(0.0, abs=1e-6)
    assert n3d["steep_hard_frac"] == pytest.approx(0.0, abs=1e-6)

    # Verify flat roof Safe_Flat_Roof (class 1) is NOT falsely vetoed
    logits = np.full((5, h, w), -2.5, dtype=np.float32)
    logits[1, :, :] = 3.5
    veto = apply_3d_normal_and_ood_veto(
        logits,
        safe_logit_bias_delta=-0.271367,
        steep_soft_mask=n3d["steep_soft_mask"],
        steep_hard_mask=n3d["steep_hard_mask"],
        ood_mask=np.zeros((h, w), dtype=bool),
        min_slz_area_px=16,
    )
    assert veto["safe_vetoed_by_3d_frac"] == pytest.approx(0.0, abs=1e-6)
    assert veto["vetoed_raw_safe_frac"] == pytest.approx(1.0, abs=1e-6)

    # Also verify pure-nadir flat roof with realistic random sensor noise stays ~90 deg & ~0% false veto
    rng = np.random.default_rng(20261127)
    disp_noisy = (0.75 + rng.normal(0.0, 0.005, (h, w))).astype(np.float32)
    n3d_j = compute_3d_surface_normals_from_disparity(disp_noisy, blur_ksize=9, blur_sigma=2.0)
    assert n3d_j["outputs_finite"] is True
    assert n3d_j["camera_pitch_deg"] >= 75.0
    assert n3d_j["steep_soft_frac"] < 0.05


def test_3d_normal_veto_nadir_roof_plus_vertical_wall_dropoff() -> None:
    """Pure-nadir flat roof elevated above flat ground preserves both flats while vetoing drop-off walls."""
    h, w = 128, 128
    disp = np.full((h, w), 0.15, dtype=np.float32)
    disp[28:100, 28:100] = 0.85  # Elevated flat rooftop with vertical perimeter drop-off

    n3d = compute_3d_surface_normals_from_disparity(disp, blur_ksize=5, blur_sigma=1.0)
    assert n3d["outputs_finite"] is True
    assert n3d["camera_pitch_deg"] >= 82.0
    # Flat roof interior and flat ground interior must remain safe (~0% false veto)
    assert float(n3d["steep_soft_mask"][38:90, 38:90].mean()) < 0.02
    assert float(n3d["steep_soft_mask"][4:18, 4:124].mean()) < 0.02
    # Vertical perimeter drop-off walls MUST be vetoed as steep hazards
    top_edge = float(n3d["steep_soft_mask"][26:30, 38:90].mean())
    bot_edge = float(n3d["steep_soft_mask"][98:102, 38:90].mean())
    left_edge = float(n3d["steep_soft_mask"][38:90, 26:30].mean())
    right_edge = float(n3d["steep_soft_mask"][38:90, 98:102].mean())
    assert min(top_edge, bot_edge, left_edge, right_edge) > 0.70


def test_3d_normal_veto_45deg_oblique_building_facade() -> None:
    """Exact 45-deg oblique geometry recovers ~45 deg camera pitch and vetoes vertical building facade."""
    h, w = 128, 128
    focal_px = 420.0
    f_eff = focal_px * (w / 512.0)
    _, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # Exact 45-deg horizontal plane: P . (0, -cos45, -sin45) = const => Z(v) proportional to 1 / ((v-h/2)/f + 1)
    inv_z_plane = np.clip((v - h * 0.5) / f_eff + 1.0, 0.25, 2.5)
    z_plane = 1.0 / inv_z_plane
    # Map z_plane to disparity in [0, 1] consistent with z_far_ratio=3.5
    inv_z_norm = (1.0 / z_plane - 1.0 / float(z_plane.max())) / max(1e-5, (1.0 / float(z_plane.min()) - 1.0 / float(z_plane.max())))
    disp = np.clip(inv_z_norm, 0.0, 1.0).astype(np.float32)
    # Inject a front-facing vertical building wall whose depth increases going down (+v)
    disp[44:92, 40:88] = 0.72 - 0.42 * np.linspace(0.0, 1.0, 48, dtype=np.float32)[:, None]

    n3d = compute_3d_surface_normals_from_disparity(disp, blur_ksize=5, blur_sigma=1.0, focal_px=focal_px)
    assert n3d["outputs_finite"] is True
    assert 30.0 <= n3d["camera_pitch_deg"] <= 60.0
    assert float(n3d["steep_soft_mask"][10:34, 10:118].mean()) < 0.05
    assert float(n3d["steep_soft_mask"][52:84, 48:80].mean()) > 0.90


def test_3d_normal_veto_forward_horizon_low_pitch_and_sky_immunity() -> None:
    """Forward-horizon aerial scene (sky top + receding ground + vertical tower) opens forward gate <= 25 deg."""
    h, w = 128, 128
    disp = np.zeros((h, w), dtype=np.float32)
    # Top 42% is constant far sky (disp=0.0); bottom 58% is flat receding ground
    horizon_y = 54
    ground_ramp = np.linspace(0.02, 1.0, h - horizon_y, dtype=np.float32)[:, None]
    disp[horizon_y:, :] = ground_ramp
    # Foreground vertical structure / tower facing camera (constant near depth)
    disp[62:118, 54:74] = 0.88

    n3d = compute_3d_surface_normals_from_disparity(disp, blur_ksize=5, blur_sigma=1.0)
    assert n3d["outputs_finite"] is True
    assert n3d["camera_pitch_deg"] <= 25.0
    assert evaluate_forward_domain_gate(n3d["camera_pitch_deg"], max_forward_pitch_deg=25.0)["in_domain"] is True
    # Flat ground beside tower must NOT be falsely vetoed
    assert float(n3d["steep_soft_mask"][72:118, 8:42].mean()) < 0.10
    # Foreground vertical tower facing forward camera MUST be vetoed as steep obstacle
    assert float(n3d["steep_soft_mask"][70:110, 58:70].mean()) > 0.85


def test_3d_normal_veto_nan_inf_disparity_fail_closed() -> None:
    """Non-finite (NaN / Inf) disparity pixels must fail closed to steep hazard with 100% finite outputs."""
    h, w = 64, 64
    # Case 1: Partial NaN / +/-Inf corruption on a flat nadir roof
    disp_partial = np.full((h, w), 0.6, dtype=np.float32)
    disp_partial[16:32, 16:32] = np.nan
    disp_partial[32:44, 32:44] = np.inf
    disp_partial[44:52, 44:52] = -np.inf
    bad_mask = ~np.isfinite(disp_partial)

    n3d_p = compute_3d_surface_normals_from_disparity(disp_partial, blur_ksize=5, blur_sigma=1.0)
    assert n3d_p["outputs_finite"] is True
    assert n3d_p["disp_input_all_finite"] is False
    assert bool(n3d_p["steep_soft_mask"][bad_mask].all())
    assert bool(n3d_p["steep_hard_mask"][bad_mask].all())
    assert float(n3d_p["cos_up"][bad_mask].max()) == pytest.approx(0.0, abs=1e-6)
    assert float(n3d_p["world_tilt_deg"][bad_mask].min()) == pytest.approx(90.0, abs=1e-4)
    # Valid flat region remains safe
    assert float(n3d_p["steep_soft_mask"][0:12, 0:64].mean()) < 0.02

    # Case 2: Total NaN disparity corruption -> 100% fail-closed vetoed to Hazard_Obstacle_Structure (2)
    disp_all_nan = np.full((h, w), np.nan, dtype=np.float32)
    n3d_all = compute_3d_surface_normals_from_disparity(disp_all_nan, blur_ksize=5, blur_sigma=1.0)
    assert n3d_all["outputs_finite"] is True
    assert n3d_all["disp_input_all_finite"] is False
    assert n3d_all["steep_soft_frac"] == pytest.approx(1.0)
    assert n3d_all["steep_hard_frac"] == pytest.approx(1.0)

    logits = np.full((5, h, w), -2.0, dtype=np.float32)
    logits[0, :, :] = 4.0  # Raw 2D claims 100% safe ground
    veto = apply_3d_normal_and_ood_veto(
        logits,
        safe_logit_bias_delta=-0.271367,
        steep_soft_mask=n3d_all["steep_soft_mask"],
        steep_hard_mask=n3d_all["steep_hard_mask"],
        ood_mask=np.zeros((h, w), dtype=bool),
        min_slz_area_px=16,
    )
    assert veto["vetoed_raw_safe_frac"] == pytest.approx(0.0)
    assert bool((veto["vetoed_raw_pred"] == 2).all())


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
    assert data["release_version"] == "v1.0.0"
    assert data["execution_mode"] == "SMOKE_CPU_ONLY"
    assert data["colab_cu_spent"] == 0.0
    assert data["surface_normal_3d_veto"]["enabled"] is True
    assert data["surface_normal_3d_veto"]["safe_pixels_vetoed_to_hazard_frac_mean"] > 0.0
    assert data["forward_detector_domain_gate"]["enabled"] is True
    comp = data["temporal_stability_comparison"]
    assert comp["frames_evaluated"] == 10
    assert comp["safe_hazard_flicker_pixel_rate_pct"]["stabilized_mean_pct"] < comp["safe_hazard_flicker_pixel_rate_pct"]["raw_mean_pct"]
    assert data["gates"]["all_outputs_finite"] is True
    assert data["gates"]["zero_nan_inf"] is True
    assert data["gates"]["permissive_license_verified"] is True
    assert data["gates"]["depth_model_apache2_verified"] is True
    assert data["gates"]["surface_normal_3d_veto_active"] is True
    assert data["gates"]["temporal_flicker_reduced"] is True


def test_committed_live_gtx1650s_drone_temporal_proof_and_gif() -> None:
    assert LIVE_PROOF_JSON.is_file(), f"Missing live proof JSON: {LIVE_PROOF_JSON}"
    data = _verify_canonical_sha256(LIVE_PROOF_JSON)

    assert data["schema_version"] == "vertirta.realworld_drone_video_temporal_proof/v4.0.0"
    assert data["release_version"] == "v1.0.0"
    assert data["execution_mode"] == "LIVE_PHYSICAL_GTX1650S_CUDA"
    assert data["colab_cu_spent"] == 0.0
    assert "GTX 1650 SUPER" in str(data["gpu"]["name"])
    assert "CUDAExecutionProvider" in data["models"]["pidnet_slz_v21_local_gtx"]["active_providers"]
    assert data["models"]["pidnet_slz_v21_local_gtx"]["iobinding_enabled"] is True
    assert data["models"]["pidnet_slz_v21_local_gtx"]["filename"] == "pidnet_slz_v21_local_gtx_opset17.onnx"
    assert data["models"]["depth_anything_v2_small"]["license"] == "Apache-2.0"
    assert "CUDAExecutionProvider" in data["models"]["depth_anything_v2_small"]["active_providers"]
    assert data["ood_and_bias_calibration"]["safe_logit_bias_delta"] == pytest.approx(-0.271367, abs=1e-5)

    # Permissive provenance lock
    prov = data["video_provenance"]
    assert prov["download_url"] and str(prov["download_url"]).startswith("https://")
    assert prov["sha256"] and len(prov["sha256"]) == 64
    assert data["license_audit"]["zero_sharealike_verified"] is True
    assert data["license_audit"]["zero_noncommercial_verified"] is True
    assert data["license_audit"]["depth_model_permissive_gate_passed"] is True
    assert validate_permissive_license(str(data["license_audit"]["license"])) is True
    assert validate_depth_model_license(
        str(data["license_audit"]["depth_model_id"]),
        str(data["license_audit"]["depth_model_license"]),
    ) is True

    # 3D Surface-Normal + OOD veto & forward pitch domain gate proof
    veto = data["surface_normal_3d_veto"]
    assert veto["enabled"] is True
    assert veto["ood_fail_closed_veto_enabled"] is True
    assert veto["safe_pixels_vetoed_to_hazard_frac_mean"] >= 0.15
    assert veto["stabilized_vetoed_safe_area_frac_mean"] < veto["raw_2d_safe_area_frac_mean"]
    fgate = data["forward_detector_domain_gate"]
    assert fgate["enabled"] is True
    assert fgate["raw_forward_detections_logged_unfiltered"] is True
    assert fgate["frames_gated_out_of_domain"] > 0

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
        assert math.isfinite(float(hw["depth_3d_cuda_latency_ms"][key])) and float(hw["depth_3d_cuda_latency_ms"][key]) > 0.0
        assert math.isfinite(float(hw["full_stack_e2e_latency_ms"][key])) and float(hw["full_stack_e2e_latency_ms"][key]) > 0.0

    # Gates
    gates = data["gates"]
    assert gates["frames_evaluated_sufficient"] is True
    assert gates["all_outputs_finite"] is True
    assert gates["zero_nan_inf"] is True
    assert gates["cuda_ep_active"] is True
    assert gates["depth_3d_cuda_ep_active"] is True
    assert gates["permissive_license_verified"] is True
    assert gates["depth_model_apache2_verified"] is True
    assert gates["surface_normal_3d_veto_active"] is True
    assert gates["temporal_flicker_reduced"] is True
    assert gates["temporal_safe_iou_improved"] is True
    assert gates["gif_under_budget_if_rendered"] is True
    assert gates["gif_duration_and_frames_verified"] is True

    # Committed README preview GIF budget, ground-truth duration/frame count & SHA-256 check
    assert README_PREVIEW_GIF.is_file(), f"Missing README preview GIF: {README_PREVIEW_GIF}"
    gif_bytes = README_PREVIEW_GIF.read_bytes()
    assert gif_bytes[:6] in (b"GIF89a", b"GIF87a"), "Invalid GIF header"
    assert len(gif_bytes) <= DEFAULT_GIF_MAX_BYTES, (
        f"Preview GIF exceeds {DEFAULT_GIF_MAX_BYTES} B budget: {len(gif_bytes)} bytes"
    )
    gif_sha = hashlib.sha256(gif_bytes).hexdigest()
    gif_meta = data["visual_renders"]["readme_preview_gif"]
    assert gif_meta["rendered"] is True
    assert gif_meta["sha256"] == gif_sha
    assert int(gif_meta["frame_count"]) >= 24
    assert float(gif_meta["actual_duration_s"]) >= 3.5

    from PIL import Image

    with Image.open(README_PREVIEW_GIF) as im:
        disk_frames = int(getattr(im, "n_frames", 1))
        disk_ms = sum(int(im.seek(fi) or im.info.get("duration", 100)) for fi in range(disk_frames))
    disk_dur_s = round(disk_ms / 1000.0, 3)
    assert disk_frames >= 24, f"GIF frame count too low: {disk_frames}"
    assert disk_dur_s >= 3.5, f"GIF actual duration too short: {disk_dur_s}s"
    assert disk_frames == int(gif_meta["frame_count"])
    assert disk_dur_s == pytest.approx(float(gif_meta["actual_duration_s"]), abs=0.05)

    # Public hygiene gate on all owned repo files
    hygiene = scan_public_hygiene(
        REPO_ROOT,
        tracked_files=[
            "scripts/realworld/run_real_drone_video_temporal_hud.py",
            "tests/realworld/test_real_drone_video_temporal_hud.py",
            "artifacts/v4.0.0/realworld_drone_video_temporal_proof.json",
        ],
    )
    assert hygiene["passed"] is True, f"Public hygiene violation: {hygiene['violations']}"
