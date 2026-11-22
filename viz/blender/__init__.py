"""VertiRTA Colab G4 Headless Blender 5.2 LTS Cycles OptiX 4K .vrta Cinema Pipeline."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "BLACKWELL_MIN_BLENDER_VERSION": ("viz.blender.blender_env_check", "BLACKWELL_MIN_BLENDER_VERSION"),
    "BLENDER_52_MIN_DRIVER": ("viz.blender.blender_env_check", "BLENDER_52_MIN_DRIVER"),
    "BLENDER_45_MIN_DRIVER": ("viz.blender.blender_env_check", "BLENDER_45_MIN_DRIVER"),
    "build_colab_install_commands": ("viz.blender.blender_env_check", "build_colab_install_commands"),
    "parse_nvidia_driver_major": ("viz.blender.blender_env_check", "parse_nvidia_driver_major"),
    "select_blender_release": ("viz.blender.blender_env_check", "select_blender_release"),
    "validate_blender_version_for_blackwell": (
        "viz.blender.blender_env_check",
        "validate_blender_version_for_blackwell",
    ),
    "MIX_RGBA_A_IDX": ("viz.blender.build_istanbul_scene", "MIX_RGBA_A_IDX"),
    "MIX_RGBA_B_IDX": ("viz.blender.build_istanbul_scene", "MIX_RGBA_B_IDX"),
    "MIX_RGBA_OUT_IDX": ("viz.blender.build_istanbul_scene", "MIX_RGBA_OUT_IDX"),
    "build_ctr_curtain_material": ("viz.blender.build_istanbul_scene", "build_ctr_curtain_material"),
    "build_ctr_curtain_object": ("viz.blender.build_istanbul_scene", "build_ctr_curtain_object"),
    "build_marmara_ocean_shader": ("viz.blender.build_istanbul_scene", "build_marmara_ocean_shader"),
    "build_nishita_istanbul_sky": ("viz.blender.build_istanbul_scene", "build_nishita_istanbul_sky"),
    "build_terracotta_roof_shader": ("viz.blender.build_istanbul_scene", "build_terracotta_roof_shader"),
    "build_turkish_rooftop_concrete_slz_shader": (
        "viz.blender.build_istanbul_scene",
        "build_turkish_rooftop_concrete_slz_shader",
    ),
    "create_material_with_nodes": ("viz.blender.build_istanbul_scene", "create_material_with_nodes"),
    "validate_offline_scene_tos_clean": (
        "viz.blender.build_istanbul_scene",
        "validate_offline_scene_tos_clean",
    ),
    "CAMERA_CUTS_1800F": ("viz.blender.apply_vrta_and_render_setup", "CAMERA_CUTS_1800F"),
    "apply_vrta_to_scene": ("viz.blender.apply_vrta_and_render_setup", "apply_vrta_to_scene"),
    "build_cbvf_funnel_mesh_data": (
        "viz.blender.apply_vrta_and_render_setup",
        "build_cbvf_funnel_mesh_data",
    ),
    "build_failed_rotor_smoke_spec": (
        "viz.blender.apply_vrta_and_render_setup",
        "build_failed_rotor_smoke_spec",
    ),
    "classify_rotor_vfx_regime": (
        "viz.blender.apply_vrta_and_render_setup",
        "classify_rotor_vfx_regime",
    ),
    "compute_rotor_vfx_state": ("viz.blender.apply_vrta_and_render_setup", "compute_rotor_vfx_state"),
    "configure_cycles_blackwell_optix": (
        "viz.blender.apply_vrta_and_render_setup",
        "configure_cycles_blackwell_optix",
    ),
    "parse_blender_script_args": (
        "viz.blender.apply_vrta_and_render_setup",
        "parse_blender_script_args",
    ),
    "sample_vrta_frames_for_fps": (
        "viz.blender.apply_vrta_and_render_setup",
        "sample_vrta_frames_for_fps",
    ),
    "select_camera_for_frame": ("viz.blender.apply_vrta_and_render_setup", "select_camera_for_frame"),
    "assert_vram_within_budget": ("viz.blender.render_slices", "assert_vram_within_budget"),
    "build_blender_slice_cmd": ("viz.blender.render_slices", "build_blender_slice_cmd"),
    "build_preflight_3frame_cmd": ("viz.blender.render_slices", "build_preflight_3frame_cmd"),
    "build_slice_rsync_cmd": ("viz.blender.render_slices", "build_slice_rsync_cmd"),
    "compute_frame_slices": ("viz.blender.render_slices", "compute_frame_slices"),
    "parse_cycles_peak_vram_gb": ("viz.blender.render_slices", "parse_cycles_peak_vram_gb"),
    "build_ffmpeg_nvenc_cmd": ("viz.blender.mux_hud_ffmpeg", "build_ffmpeg_nvenc_cmd"),
    "format_hud_telemetry_payload": ("viz.blender.mux_hud_ffmpeg", "format_hud_telemetry_payload"),
    "render_hud_frame_rgba": ("viz.blender.mux_hud_ffmpeg", "render_hud_frame_rgba"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val
