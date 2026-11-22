"""Physically valid Marmara coastal/urban wind shear & MIL-F-8785C Dryden turbulence model.

Overrides the DR-4 inverted urban shear erratum: urban Exposure B mean wind speed MUST remain
strictly below open-water coastal Exposure D fetch at all altitudes while mechanical turbulence
intensity increases over built-up Dragos/Kartal/Cevizli terrain.
"""

from __future__ import annotations

import math


METERS_PER_FOOT = 0.3048
ASCE7_KZ_COASTAL_D = 1.15
ASCE7_KZ_URBAN_B = 0.70
URBAN_SPEED_REDUCTION_FACTOR = math.sqrt(ASCE7_KZ_URBAN_B / ASCE7_KZ_COASTAL_D)
COASTAL_SHEAR_ALPHA = 0.14
URBAN_SHEAR_ALPHA = 0.22
URBAN_TURBULENCE_BOOST = 1.35


def compute_dryden_parameters(w20_mps: float, z_agl_m: float) -> dict[str, float]:
    """Compute MIL-F-8785C low-altitude Dryden scale lengths and RMS gust intensities."""
    if w20_mps < 0.0:
        raise ValueError(f"w20_mps must be >= 0.0, got {w20_mps}")
    if z_agl_m <= 0.0:
        raise ValueError(f"z_agl_m must be > 0.0, got {z_agl_m}")

    h_ft = min(z_agl_m / METERS_PER_FOOT, 1000.0)
    denom = 0.177 + 0.000823 * h_ft
    l_w_m = z_agl_m
    l_uv_m = z_agl_m / (denom**1.2)
    sigma_w_mps = 0.1 * w20_mps
    sigma_uv_mps = sigma_w_mps / (denom**0.4)

    return {
        "L_u_m": l_uv_m,
        "L_v_m": l_uv_m,
        "L_w_m": l_w_m,
        "sigma_u_mps": sigma_uv_mps,
        "sigma_v_mps": sigma_uv_mps,
        "sigma_w_mps": sigma_w_mps,
    }


def compute_wind_profile_mps(
    w20_mps: float,
    z_agl_m: float,
    exposure: str,
) -> tuple[float, float, float]:
    """Return (mean_wind_mps, sigma_w_mps, sigma_u_mps) for coastal_d or urban_b exposure.

    Enforces ASCE-7 Exposure B reduction so urban mean wind NEVER exceeds open-water coastal_d
    fetch (preventing the DR-4 inverted U_urban = W_20 * (z/6)^0.30 erratum).
    """
    if exposure not in {"coastal_d", "urban_b"}:
        raise ValueError(f"Unsupported exposure '{exposure}'; expected 'coastal_d' or 'urban_b'")

    dryden = compute_dryden_parameters(w20_mps=w20_mps, z_agl_m=z_agl_m)
    u_coastal = w20_mps * ((z_agl_m / 6.0) ** COASTAL_SHEAR_ALPHA)

    if exposure == "coastal_d":
        return (u_coastal, dryden["sigma_w_mps"], dryden["sigma_u_mps"])

    u_urban_raw = URBAN_SPEED_REDUCTION_FACTOR * w20_mps * ((z_agl_m / 10.0) ** URBAN_SHEAR_ALPHA)
    # Physical monotonicity guard across the entire 0..300 m atmospheric boundary layer
    u_urban = min(u_urban_raw, 0.95 * u_coastal)
    return (
        u_urban,
        dryden["sigma_w_mps"] * URBAN_TURBULENCE_BOOST,
        dryden["sigma_u_mps"] * URBAN_TURBULENCE_BOOST,
    )
