#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Deterministic zero-OpenFOAM CI fixture generator emitting a compact `< 80 KiB` `VRTAWND1` LUT.

Generates a physically coherent synthetic coastal-canyon bluff-body wake across 48 inflow states
(16 azimuths x 3 speeds: 6, 12, 18 m/s) over Dragos-Maltepe-Cevizli-west Kartal strictly west of
29.1895 deg E (`max_lon_deg = 29.1892`), compresses via SVD POD (R=8) + multiquadric RBF + LODO
`eps_rom_p95`, and writes a byte-valid `77,824 B` (`< 80 KiB`) `VRTAWND1` binary into a caller
temp path in < 0.3 s — zero OpenFOAM in GitHub Actions and zero multi-MB blobs in git.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.cfd_offline_gpl.openfoam_case_gen import (
    KARTAL_DATUM_ALT_AMSL_M,
    KARTAL_DATUM_EGM2008_N_M,
    KARTAL_DATUM_LAT_DEG,
    KARTAL_DATUM_LON_DEG,
    RANS_DIRECTIONS_DEG,
    RANS_SPEEDS_MPS,
    roughness_z0_for_direction,
)
from tools.cfd_offline_gpl.pod_rbf_compressor import GridSpec, compress_and_pack_vrtawnd1


def _synth_coastal_wake_grid(
    spec: GridSpec,
    inflow_dirs_rad: np.ndarray,
    inflow_speeds_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize physically realistic ABL + bluff-body coastal canyon downdraft snapshots."""
    xs = spec.origin_enu[0] + np.arange(spec.nx, dtype=np.float64) * spec.spacing_m[0]
    ys = spec.origin_enu[1] + np.arange(spec.ny, dtype=np.float64) * spec.spacing_m[1]
    zs = spec.origin_enu[2] + np.arange(spec.nz, dtype=np.float64) * spec.spacing_m[2]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")

    m_snaps = int(inflow_dirs_rad.shape[0])
    vel = np.zeros((m_snaps, spec.nx, spec.ny, spec.nz, 3), dtype=np.float64)

    # Synthetic bluff coastal building cluster centered near Kartal/Dragos shoreline (west of datum)
    bldg_x, bldg_y, bldg_h = -40.0, 20.0, 38.0
    dx_b = gx - bldg_x
    dy_b = gy - bldg_y
    r2_b = dx_b * dx_b + dy_b * dy_b

    for k in range(m_snaps):
        ang = float(inflow_dirs_rad[k])
        u_ref = float(inflow_speeds_mps[k])
        z0 = roughness_z0_for_direction(math.degrees(ang))
        # Log-law ABL vertical wind profile
        z_eff = np.maximum(gz, 1.5)
        log_prof = np.log((z_eff + z0) / z0) / math.log((10.0 + z0) / z0)
        u_mag = u_ref * np.clip(log_prof, 0.35, 1.65)

        fx = math.cos(ang)
        fy = math.sin(ang)

        # Downwind wake deficit + lee-side helical downdraft behind coastal high-rise ridge
        s_along = dx_b * fx + dy_b * fy
        s_cross = -dx_b * fy + dy_b * fx
        in_wake = np.exp(-0.5 * ((s_cross / 75.0) ** 2)) * np.exp(
            -0.5 * ((np.maximum(s_along - 30.0, 0.0) / 140.0) ** 2)
        )
        wake_step = 1.0 / (1.0 + np.exp(-s_along / 20.0))
        roof_band = np.exp(-0.5 * (((gz - bldg_h) / 22.0) ** 2))

        deficit = 0.42 * in_wake * wake_step * roof_band
        # Add subtle higher-order directional harmonic so LODO residual eps_rom_p95 is non-trivial
        harmonic = 0.08 * math.sin(5.0 * ang + 0.3) * np.exp(-r2_b / (220.0**2)) * roof_band
        u_e = u_mag * (1.0 - deficit) * fx + u_ref * harmonic * (-fy)
        u_n = u_mag * (1.0 - deficit) * fy + u_ref * harmonic * fx
        u_u = -0.26 * u_ref * in_wake * wake_step * roof_band + 0.07 * u_ref * math.cos(
            4.0 * ang
        ) * np.exp(-r2_b / (180.0**2)) * roof_band

        vel[k, ..., 0] = u_e
        vel[k, ..., 1] = u_n
        vel[k, ..., 2] = u_u

    # Anisotropic turbulence intensity & integral length scale
    roof_shear = np.exp(-0.5 * (((gz - bldg_h) / 26.0) ** 2)) * np.exp(-r2_b / (350.0**2))
    sig = np.zeros((spec.nx, spec.ny, spec.nz, 3), dtype=np.float64)
    sig[..., 0] = 0.65 + 1.45 * roof_shear
    sig[..., 1] = 0.55 + 1.25 * roof_shear
    sig[..., 2] = 0.40 + 1.10 * roof_shear
    l_turb = np.clip(18.0 + 0.35 * gz - 8.0 * roof_shear, 8.0, 140.0)
    return vel, sig, l_turb


def generate_fixture_vrtawnd1(out_path: Path | str) -> dict[str, Any]:
    """Generate a compact `< 80 KiB` byte-valid `VRTAWND1` fixture into `out_path`."""
    dirs_rad = []
    speeds = []
    for d_deg in RANS_DIRECTIONS_DEG:
        for spd in RANS_SPEEDS_MPS:
            dirs_rad.append(math.radians(d_deg))
            speeds.append(float(spd))
    inflow_dirs = np.asarray(dirs_rad, dtype=np.float64)
    inflow_speeds = np.asarray(speeds, dtype=np.float64)

    # Compact 8x8x8 Tier-0 corridor (strictly west of +100 m E -> lon <= 29.1892 < 29.1895)
    tier0 = GridSpec(
        nx=8,
        ny=8,
        nz=8,
        origin_enu=(-2700.0, -400.0, 0.0),
        spacing_m=(400.0, 600.0, 25.0),
    )
    # Compact 8x8x8 Tier-1 rooftop SLZ nest centered on west Kartal (-80..+60 m E)
    tier1 = GridSpec(
        nx=8,
        ny=8,
        nz=8,
        origin_enu=(-80.0, -80.0, 0.0),
        spacing_m=(20.0, 20.0, 12.0),
    )

    t0_vel, t0_sig, t0_lt = _synth_coastal_wake_grid(tier0, inflow_dirs, inflow_speeds)
    t1_vel, t1_sig, t1_lt = _synth_coastal_wake_grid(tier1, inflow_dirs, inflow_speeds)

    return compress_and_pack_vrtawnd1(
        out_path,
        tier0_spec=tier0,
        tier1_spec=tier1,
        inflow_dirs_rad=inflow_dirs,
        inflow_speeds_mps=inflow_speeds,
        tier0_snapshots_enu=t0_vel,
        tier1_snapshots_enu=t1_vel,
        tier0_sigma_enu=t0_sig,
        tier1_sigma_enu=t1_sig,
        tier0_lturb_m=t0_lt,
        tier1_lturb_m=t1_lt,
        datum_lat_deg=KARTAL_DATUM_LAT_DEG,
        datum_lon_deg=KARTAL_DATUM_LON_DEG,
        datum_alt_hae_m=KARTAL_DATUM_ALT_AMSL_M + KARTAL_DATUM_EGM2008_N_M,
        max_lon_deg=29.1892,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate compact deterministic VRTAWND1 CI fixture.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    meta = generate_fixture_vrtawnd1(args.out)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
