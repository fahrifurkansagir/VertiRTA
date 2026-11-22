"""VertiRTA offline 4D fault-degraded CBVF and 2D CEL reachability LUT pipeline."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_REGIMES": ("reachability.cbvf_4d_dynamics", "DEFAULT_REGIMES"),
    "HAS_HJ_REACHABILITY": ("reachability.cbvf_4d_dynamics", "HAS_HJ_REACHABILITY"),
    "HAS_JAX": ("reachability.cbvf_4d_dynamics", "HAS_JAX"),
    "RegimeSpec": ("reachability.cbvf_4d_dynamics", "RegimeSpec"),
    "VertiRta4DDynamics": ("reachability.cbvf_4d_dynamics", "VertiRta4DDynamics"),
    "build_regimes_from_config": ("reachability.cbvf_4d_dynamics", "build_regimes_from_config"),
    "compute_all_regime_v_caps": ("reachability.cbvf_4d_dynamics", "compute_all_regime_v_caps"),
    "compute_degraded_v_cap": ("reachability.cbvf_4d_dynamics", "compute_degraded_v_cap"),
    "conservative_smax_4": ("reachability.cbvf_4d_dynamics", "conservative_smax_4"),
    "load_vehicle_params": ("reachability.cbvf_4d_dynamics", "load_vehicle_params"),
    "LUT_EXPORT_SHAPE": ("reachability.solve_cbvf_luts", "LUT_EXPORT_SHAPE"),
    "LUT_FINE_SHAPE": ("reachability.solve_cbvf_luts", "LUT_FINE_SHAPE"),
    "LUT_HEADER_BYTES": ("reachability.solve_cbvf_luts", "LUT_HEADER_BYTES"),
    "LUT_MAGIC": ("reachability.solve_cbvf_luts", "LUT_MAGIC"),
    "LUT_PAYLOAD_BYTES": ("reachability.solve_cbvf_luts", "LUT_PAYLOAD_BYTES"),
    "LUT_TOTAL_BYTES": ("reachability.solve_cbvf_luts", "LUT_TOTAL_BYTES"),
    "VrtaLutHeader": ("reachability.solve_cbvf_luts", "VrtaLutHeader"),
    "eval_cbvf_4d": ("reachability.solve_cbvf_luts", "eval_cbvf_4d"),
    "load_vrta_lut": ("reachability.solve_cbvf_luts", "load_vrta_lut"),
    "solve_and_export_lut": ("reachability.solve_cbvf_luts", "solve_and_export_lut"),
    "CEL2D_EXPORT_SHAPE": ("reachability.solve_cel_2d_lut", "CEL2D_EXPORT_SHAPE"),
    "CEL2D_HEADER_BYTES": ("reachability.solve_cel_2d_lut", "CEL2D_HEADER_BYTES"),
    "CEL2D_MAGIC": ("reachability.solve_cel_2d_lut", "CEL2D_MAGIC"),
    "CEL2D_PAYLOAD_BYTES": ("reachability.solve_cel_2d_lut", "CEL2D_PAYLOAD_BYTES"),
    "CEL2D_TOTAL_BYTES": ("reachability.solve_cel_2d_lut", "CEL2D_TOTAL_BYTES"),
    "CelTierSpec": ("reachability.solve_cel_2d_lut", "CelTierSpec"),
    "DEFAULT_CEL_TIERS": ("reachability.solve_cel_2d_lut", "DEFAULT_CEL_TIERS"),
    "VrtaCel2DHeader": ("reachability.solve_cel_2d_lut", "VrtaCel2DHeader"),
    "analytic_cel_z_req": ("reachability.solve_cel_2d_lut", "analytic_cel_z_req"),
    "eval_cel_2d": ("reachability.solve_cel_2d_lut", "eval_cel_2d"),
    "find_cel_zero_level_z": ("reachability.solve_cel_2d_lut", "find_cel_zero_level_z"),
    "load_cel_2d_lut": ("reachability.solve_cel_2d_lut", "load_cel_2d_lut"),
    "solve_and_export_cel_2d": ("reachability.solve_cel_2d_lut", "solve_and_export_cel_2d"),
    "verify_all_soundness": ("reachability.verify_lut_soundness", "verify_all_soundness"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_name, attr_name = _EXPORTS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr_name)
    raise AttributeError(f"module 'reachability' has no attribute {name!r}")
