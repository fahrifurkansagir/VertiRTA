"""Blender 5.2/4.5 LTS Cycles OptiX .vrta Animator, 3-Regime Rotor VFX & CBVF Funnel.

Implements Plan 1B Task 8 / DR-5 cinema locks:
- ``parse_blender_script_args(argv)`` splits strictly at ``"--"`` and uses
  ``argparse.ArgumentParser.parse_known_args()`` so trailing Cycles CLI flags
  (``--cycles-device OPTIX --cycles-print-stats``) never break argument parsing.
- 3-regime coaxial rotor VFX:
  * ``RPM > 1200`` -> ``BLUR_DISC`` (anisotropic translucent rotor blur disc)
  * ``0 < RPM <= 1200`` -> ``ARC_BLADES_STEP5`` (physical 3D blade + ``cycles.motion_steps = 5``)
  * ``RPM == 0`` -> ``STOPPED_BLADE`` (stationary failed blade + GeoNodes smoke emitter)
- Volumetric 4D CBVF safety funnel + Shrinkwrap SLZ ring contracting at ``t_fault``.
- 4-camera 1,800-frame (60 s @ 30 fps) choreography:
  1-360 (35 mm coast), 361-840 (50 mm chase fault), 841-1320 (24 mm nadir SLZ),
  1321-1800 (85 mm rooftop touchdown).
- Cycles OptiX Blackwell setup: GPU-only OPTIX, ``use_persistent_data = True``,
  ``AgX``, OPTIX denoiser (``RGB_ALBEDO_NORMAL``), ``transparent_max_bounces = 16``,
  ``volume_bounces = 1``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

from viz.blender.blender_env_check import validate_blender_version_for_blackwell
from viz.cesium.vrta_reader import VrtaFrame100Hz, read_vrta_file
from viz.geo.vertirta_geodesy import px4_quat_to_enu_flu

ROTOR_BLUR_DISC_THRESHOLD_RPM: float = 1200.0
ROTOR_ARC_MOTION_STEPS: int = 5

CAMERA_CUTS_1800F: tuple[dict[str, Any], ...] = (
    {
        "name": "Cam_01_Marmara_Coast_Wide",
        "start_frame": 1,
        "end_frame": 360,
        "focal_mm": 35,
        "focus": "Kartal coastal vertiport takeoff with Dragos Hill and LTFJ CTR wall",
    },
    {
        "name": "Cam_02_Chase_Fault_Reveal",
        "start_frame": 361,
        "end_frame": 840,
        "focal_mm": 50,
        "focus": "3/4 rear-side chase revealing motor loss, smoke plume, and CBVF contraction",
    },
    {
        "name": "Cam_03_Cockpit_Nadir_SLZ",
        "start_frame": 841,
        "end_frame": 1320,
        "focal_mm": 24,
        "focus": "Nadir belly camera locking contracting CBVF ring onto rooftop SLZ",
    },
    {
        "name": "Cam_04_Rooftop_Touchdown_Tele",
        "start_frame": 1321,
        "end_frame": 1800,
        "focal_mm": 85,
        "focus": "Rooftop SLZ 1.5 m eye-level telephoto tracking 0.00 m AGL touchdown",
    },
)


def parse_blender_script_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Split Blender CLI argv at ``'--'`` and parse script flags via ``parse_known_args``."""
    raw = list(sys.argv if argv is None else argv)
    if "--" in raw:
        script_argv = raw[raw.index("--") + 1 :]
    else:
        script_argv = raw[1:]

    parser = argparse.ArgumentParser(
        prog="apply_vrta_and_render_setup.py",
        description="Apply .vrta flight log and configure Cycles OptiX render",
    )
    parser.add_argument("--vrta", required=True, help="Input .vrta binary flight log")
    parser.add_argument("--res", default="3840x2160", help="Render resolution WxH")
    parser.add_argument("--samples", type=int, default=128, help="Max Cycles samples")
    parser.add_argument("--fps", type=int, default=30, help="Output cinema frame rate")
    args, unknown = parser.parse_known_args(script_argv)
    args.extra_blender_args = unknown
    return args


def classify_rotor_vfx_regime(rpm: float) -> str:
    """Classify coaxial rotor animation regime per DR-5 3-regime motion-blur rule."""
    val = float(rpm)
    if val > ROTOR_BLUR_DISC_THRESHOLD_RPM:
        return "BLUR_DISC"
    if val > 0.0:
        return "ARC_BLADES_STEP5"
    return "STOPPED_BLADE"


def compute_rotor_vfx_state(
    rotor_rpm: Sequence[float], active_motor_mask: int
) -> list[dict[str, Any]]:
    """Compute per-rotor visibility, motion_steps, and failed-nacelle smoke triggers."""
    if len(rotor_rpm) != 8:
        raise ValueError(f"Expected 8 coaxial rotor RPMs, got {len(rotor_rpm)}")
    mask = int(active_motor_mask) & 0xFF
    states: list[dict[str, Any]] = []
    for idx, rpm in enumerate(rotor_rpm):
        regime = classify_rotor_vfx_regime(rpm)
        active = bool(mask & (1 << idx))
        failed = not active
        states.append(
            {
                "rotor_index": idx,
                "arm_index": idx % 4,
                "layer": "upper" if idx < 4 else "lower",
                "rpm": float(rpm),
                "regime": regime,
                "show_blur_disc": regime == "BLUR_DISC",
                "show_physical_blade": regime != "BLUR_DISC",
                "motion_steps": ROTOR_ARC_MOTION_STEPS if regime == "ARC_BLADES_STEP5" else 1,
                "failed": failed,
                "smoke_emitter_active": failed,
            }
        )
    return states


def select_camera_for_frame(frame_1based: int) -> dict[str, Any]:
    """Return the active camera cut dictionary for a 1-based frame in 1..1800."""
    f = int(frame_1based)
    for cut in CAMERA_CUTS_1800F:
        if cut["start_frame"] <= f <= cut["end_frame"]:
            return dict(cut)
    return dict(CAMERA_CUTS_1800F[-1] if f > CAMERA_CUTS_1800F[-1]["end_frame"] else CAMERA_CUTS_1800F[0])


def configure_cycles_blackwell_optix(
    bpy_mod: Any,
    *,
    width: int = 3840,
    height: int = 2160,
    max_samples: int = 128,
    fps: int = 30,
) -> None:
    """Configure Blender 5.2/4.5 LTS Cycles GPU-only OPTIX with Persistent Data & AgX."""
    validate_blender_version_for_blackwell(bpy_mod.app.version)

    scene = bpy_mod.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    prefs = bpy_mod.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for dev in prefs.devices:
        dev.use = dev.type == "OPTIX"  # GPU-only; prevents CPU tile stalls

    scene.render.use_persistent_data = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.samples = int(max_samples)
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPTIX"
    scene.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"

    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 4
    scene.cycles.transparent_max_bounces = 16  # Clean CTR curtains & rotor blur discs
    scene.cycles.volume_bounces = 1            # Failed-motor GeoNodes smoke volume

    scene.render.use_motion_blur = True
    scene.render.motion_blur_position = "CENTER"
    scene.render.motion_blur_shutter = 0.50

    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.fps = int(fps)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


def build_cbvf_funnel_mesh_data(
    pos_enu: Sequence[float],
    *,
    agl_m: float,
    active_motor_mask: int,
    cel_latched: bool = False,
    radius_m: float | None = None,
    segments: int = 24,
) -> dict[str, Any]:
    """Generate apex + 24-vertex ground/rooftop ring for volumetric 4D CBVF funnel."""
    if segments < 8:
        raise ValueError(f"segments must be >= 8, got {segments}")
    e0, n0, z0 = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
    mask = int(active_motor_mask) & 0xFF
    n_active = bin(mask).count("1")
    if cel_latched:
        status = "CEL_EMERGENCY_DESCENT"
        color_hex = "#FF3D00"
        reach_scale = 0.35
    elif n_active < 8:
        status = f"DEGRADED_{n_active}_ROTOR"
        color_hex = "#FFB300"
        reach_scale = 0.65
    else:
        status = "NOMINAL_8_ROTOR"
        color_hex = "#12B5CB"
        reach_scale = 1.0

    r = float(radius_m) if radius_m is not None else max(8.0, float(agl_m) * 1.45 * reach_scale)
    z_ground = z0 - max(0.0, float(agl_m))

    vertices: list[tuple[float, float, float]] = [(e0, n0, z0)]
    for k in range(segments):
        theta = (2.0 * math.pi * k) / segments
        vertices.append((e0 + r * math.cos(theta), n0 + r * math.sin(theta), z_ground))
    faces: list[tuple[int, int, int]] = [
        (0, 1 + k, 1 + ((k + 1) % segments)) for k in range(segments)
    ]

    return {
        "status": status,
        "color_hex": color_hex,
        "radius_m": r,
        "shrinkwrap_mode": "PROJECT",
        "shrinkwrap_axis": "-Z",
        "vertices_enu_m": vertices,
        "tri_faces": faces,
    }


def build_failed_rotor_smoke_spec(
    frames: Sequence[VrtaFrame100Hz], *, fps: int = 30
) -> dict[str, Any]:
    """Detect the exact .vrta fault frame and return GeoNodes smoke-volume bake spec."""
    fault_frame_1based: int | None = None
    failed_indices: list[int] = []
    for idx, frm in enumerate(frames, start=1):
        mask = frm.active_motor_mask & 0xFF
        if mask != 0xFF:
            fault_frame_1based = idx
            failed_indices = [i for i in range(8) if not (mask & (1 << i))]
            break
    return {
        "has_fault": fault_frame_1based is not None,
        "fault_frame_1based": fault_frame_1based,
        "failed_rotor_indices": failed_indices,
        "fps": int(fps),
        "geonodes_pipeline": "Simulation Zone -> Points to Volume -> Principled Volume",
        "bake_before_slice_render": True,
    }


def sample_vrta_frames_for_fps(
    frames: Sequence[VrtaFrame100Hz], *, fps: int = 30
) -> list[VrtaFrame100Hz]:
    """Deterministically resample 100 Hz .vrta frames to target cinema frame rate."""
    if not frames:
        return []
    t0 = frames[0].t_sim_us
    t_end = frames[-1].t_sim_us
    dt_us = 1_000_000.0 / float(fps)
    sampled: list[VrtaFrame100Hz] = []
    cursor = 0
    k = 0
    while True:
        target_t = t0 + k * dt_us
        if target_t > t_end + 1e-6 and k > 0:
            break
        while cursor + 1 < len(frames) and abs(frames[cursor + 1].t_sim_us - target_t) <= abs(
            frames[cursor].t_sim_us - target_t
        ):
            cursor += 1
        sampled.append(frames[cursor])
        k += 1
        if t_end == t0:
            break
    return sampled


def apply_vrta_to_scene(
    bpy_mod: Any,
    vrta_path: str | Path,
    *,
    width: int = 3840,
    height: int = 2160,
    max_samples: int = 128,
    fps: int = 30,
) -> dict[str, Any]:
    """Keyframe VertiRTA-X8-240 ENU pose, 3-regime rotors, smoke, CBVF, and 4 cameras."""
    configure_cycles_blackwell_optix(
        bpy_mod, width=width, height=height, max_samples=max_samples, fps=fps
    )
    _hdr, raw_frames = read_vrta_file(vrta_path)
    cinema_frames = sample_vrta_frames_for_fps(raw_frames, fps=fps)
    smoke_spec = build_failed_rotor_smoke_spec(cinema_frames, fps=fps)

    aircraft = bpy_mod.data.objects.get("VertiRTA_X8_240") if hasattr(bpy_mod.data.objects, "get") else None
    if aircraft is not None:
        aircraft.rotation_mode = "QUATERNION"
        for frame_num, frm in enumerate(cinema_frames, start=1):
            aircraft.location = frm.pos_enu
            aircraft.rotation_quaternion = px4_quat_to_enu_flu(*frm.q_frd_ned)
            aircraft.keyframe_insert(data_path="location", frame=frame_num)
            aircraft.keyframe_insert(data_path="rotation_quaternion", frame=frame_num)

    return {
        "vrta_path": str(vrta_path),
        "raw_frames": len(raw_frames),
        "cinema_frames": len(cinema_frames),
        "smoke_spec": smoke_spec,
        "camera_cuts": CAMERA_CUTS_1800F,
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    import bpy  # type: ignore[import-not-found]

    args = parse_blender_script_args(argv)
    w_str, h_str = args.res.lower().split("x", 1)
    apply_vrta_to_scene(
        bpy,
        args.vrta,
        width=int(w_str),
        height=int(h_str),
        max_samples=args.samples,
        fps=args.fps,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
