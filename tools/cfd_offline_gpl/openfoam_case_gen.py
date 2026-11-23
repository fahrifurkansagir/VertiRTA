#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""COST 732 / AIJ-compliant offline OpenFOAM v2412/v2606 coastal wake case generator.

Strictly air-gapped GPL-3.0 offline data generator for the Dragos-Maltepe-Cevizli-west Kartal
coastal corridor strictly WEST of the LTFJ CTR J2 boundary (`lon_deg < 29.1895 deg E`).

Hard forensic errata enforced:
  - Rejects any SLZ or domain boundary with `lon_deg >= 29.1895 deg E` (catching DR-09's
    erroneous 29.2057 deg E ISTMarina placement +1.14 km inside LTFJ CTR).
  - Uses `slip` lateral/top boundary conditions (NEVER `symmetryPlane`, avoiding OpenFOAM GL#3226).
  - Targets offline `openfoam-dev:apptainer` execution (NEVER `dl.openfoam.com` apt in CI, GL#3236).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

KARTAL_DATUM_LAT_DEG = 40.8860
KARTAL_DATUM_LON_DEG = 29.1880
KARTAL_DATUM_ALT_AMSL_M = 12.0
KARTAL_DATUM_EGM2008_N_M = 36.7
MAX_ALLOWED_LON_DEG = 29.1895

RANS_DIRECTIONS_DEG: tuple[float, ...] = tuple(i * 22.5 for i in range(16))
RANS_SPEEDS_MPS: tuple[float, ...] = (6.0, 12.0, 18.0)
IDDES_DOMINANT_REGIMES: tuple[tuple[str, float, float, float], ...] = (
    ("poyraz_ne_14mps", 45.0, 14.0, 0.50),
    ("yildiz_n_12mps", 0.0, 12.0, 0.50),
    ("lodos_sw_16mps", 225.0, 16.0, 0.002),
    ("lodos_ssw_18mps", 202.5, 18.0, 0.002),
    ("sea_breeze_s_8mps", 180.0, 8.0, 0.002),
    ("coastal_wnw_10mps", 292.5, 10.0, 0.10),
)


def _meters_per_deg_lon(lat_deg: float) -> float:
    return 111_319.49079327358 * math.cos(math.radians(lat_deg))


def _load_airspace_max_lon(airspace_yaml: Path | None = None) -> float:
    candidate = airspace_yaml or (
        Path(__file__).resolve().parents[2] / "config" / "istanbul_ltfj_airspace.yaml"
    )
    if candidate.is_file():
        cfg = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        val = cfg.get("odd_bounds", {}).get("max_lon_near_j2_deg")
        if val is not None:
            return float(val)
    return MAX_ALLOWED_LON_DEG


def roughness_z0_for_direction(wind_from_deg: float) -> float:
    """Return aerodynamic roughness z0 [m]: onshore SW Lodos sea fetch (0.002 m) vs NE urban (0.50 m)."""
    ang = wind_from_deg % 360.0
    if 150.0 <= ang <= 260.0:
        return 0.002
    if 260.0 < ang <= 320.0:
        return 0.10
    return 0.50


def validate_coastal_longitude_bound(
    slz_lat_deg: float,
    slz_lon_deg: float,
    east_extent_m: float,
    *,
    max_lon_deg: float = MAX_ALLOWED_LON_DEG,
) -> float:
    """Hard-reject any SLZ or eastern CFD domain vertex at or east of `max_lon_deg` (29.1895 E)."""
    if slz_lon_deg >= max_lon_deg:
        raise ValueError(
            f"SLZ longitude {slz_lon_deg:.6f} deg E breaches LTFJ CTR western buffer "
            f"({max_lon_deg:.4f} deg E); ISTMarina (29.2057 E) is strictly forbidden."
        )
    domain_east_lon = slz_lon_deg + east_extent_m / _meters_per_deg_lon(slz_lat_deg)
    if domain_east_lon >= max_lon_deg:
        raise ValueError(
            f"CFD eastern domain boundary {domain_east_lon:.6f} deg E breaches LTFJ CTR "
            f"buffer ({max_lon_deg:.4f} deg E). Reduce east_extent_m or shift domain west."
        )
    return domain_east_lon


def generate_coastal_openfoam_case(
    case_dir: Path | str,
    *,
    slz_lat_deg: float = KARTAL_DATUM_LAT_DEG,
    slz_lon_deg: float = KARTAL_DATUM_LON_DEG,
    wind_from_deg: float = 225.0,
    u_ref_mps: float = 12.0,
    z_ref_m: float = 10.0,
    mode: str = "rans",
    east_extent_m: float = 80.0,
    west_extent_m: float = 3200.0,
    south_extent_m: float = 600.0,
    north_extent_m: float = 4200.0,
    top_height_m: float = 360.0,
    airspace_yaml: Path | None = None,
) -> Path:
    """Write a COST 732 / AIJ OpenFOAM v2412/v2606 RANS or IDDES case strictly west of 29.1895 E."""
    if mode not in ("rans", "iddes"):
        raise ValueError(f"Unsupported OpenFOAM mode '{mode}'; expected 'rans' or 'iddes'.")

    max_lon = _load_airspace_max_lon(airspace_yaml)
    domain_east_lon = validate_coastal_longitude_bound(
        slz_lat_deg, slz_lon_deg, east_extent_m, max_lon_deg=max_lon
    )

    root = Path(case_dir)
    for sub in ("0", "constant", "system"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    z0_m = roughness_z0_for_direction(wind_from_deg)
    # Meteorological wind_from_deg -> flow direction vector (going toward) in ENU
    flow_az_rad = math.radians((wind_from_deg + 180.0) % 360.0)
    flow_x = math.sin(flow_az_rad)
    flow_y = math.cos(flow_az_rad)

    solver = "simpleFoam" if mode == "rans" else "pimpleFoam"
    turb_model = "kOmegaSST" if mode == "rans" else "kOmegaSSTIDDES"
    sim_type = "RAS" if mode == "rans" else "LES"

    u_file = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// COST 732 / AIJ coastal ABL velocity; GL#3226 lock: lateral/top use `slip` only.
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform ({u_ref_mps * flow_x:.4f} {u_ref_mps * flow_y:.4f} 0);

boundaryField
{{
    inlet
    {{
        type            atmBoundaryLayerInletVelocity;
        flowDir         ({flow_x:.6f} {flow_y:.6f} 0);
        zDir            (0 0 1);
        Uref            {u_ref_mps:.4f};
        Zref            {z_ref_m:.2f};
        z0              uniform {z0_m:.4f};
        zGround         uniform 0.0;
        value           $internalField;
    }}
    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           $internalField;
    }}
    top
    {{
        type            slip;
    }}
    lateral_west
    {{
        type            slip;
    }}
    lateral_east
    {{
        type            slip;
    }}
    ground
    {{
        type            noSlip;
    }}
    buildings
    {{
        type            noSlip;
    }}
}}
"""
    (root / "0" / "U").write_text(u_file, encoding="utf-8", newline="\n")

    nut_file = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0.0;

boundaryField
{{
    inlet           {{ type calculated; value uniform 0; }}
    outlet          {{ type calculated; value uniform 0; }}
    top             {{ type slip; }}
    lateral_west    {{ type slip; }}
    lateral_east    {{ type slip; }}
    ground
    {{
        type            atmNutkWallFunction;
        z0              uniform {z0_m:.4f};
        value           uniform 0;
    }}
    buildings
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
}}
"""
    (root / "0" / "nut").write_text(nut_file, encoding="utf-8", newline="\n")

    p_file = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;

boundaryField
{
    inlet           { type zeroGradient; }
    outlet          { type fixedValue; value uniform 0; }
    top             { type slip; }
    lateral_west    { type slip; }
    lateral_east    { type slip; }
    ground          { type zeroGradient; }
    buildings       { type zeroGradient; }
}
"""
    (root / "0" / "p").write_text(p_file, encoding="utf-8", newline="\n")

    turb_file = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
simulationType  {sim_type};
{sim_type}
{{
    {"RASModel" if sim_type == "RAS" else "LESModel"}    {turb_model};
    turbulence      on;
    printCoeffs     on;
}}
"""
    (root / "constant" / "turbulenceProperties").write_text(
        turb_file, encoding="utf-8", newline="\n"
    )

    control_dict = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
application     {solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {2000 if mode == "rans" else 180};
deltaT          {1 if mode == "rans" else 0.05};
writeControl    timeStep;
writeInterval   {500 if mode == "rans" else 200};
writeFormat     binary;
writePrecision  7;
"""
    (root / "system" / "controlDict").write_text(control_dict, encoding="utf-8", newline="\n")

    block_mesh = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// Anchored to Kartal datum ({slz_lat_deg:.6f} N, {slz_lon_deg:.6f} E), max_lon={domain_east_lon:.6f} E < {max_lon:.4f} E
scale 1;
vertices
(
    ({-west_extent_m:.1f} {-south_extent_m:.1f} 0.0)
    ({east_extent_m:.1f} {-south_extent_m:.1f} 0.0)
    ({east_extent_m:.1f} {north_extent_m:.1f} 0.0)
    ({-west_extent_m:.1f} {north_extent_m:.1f} 0.0)
    ({-west_extent_m:.1f} {-south_extent_m:.1f} {top_height_m:.1f})
    ({east_extent_m:.1f} {-south_extent_m:.1f} {top_height_m:.1f})
    ({east_extent_m:.1f} {north_extent_m:.1f} {top_height_m:.1f})
    ({-west_extent_m:.1f} {north_extent_m:.1f} {top_height_m:.1f})
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (96 48 28) simpleGrading (1 1 4)
);
"""
    (root / "system" / "blockMeshDict").write_text(block_mesh, encoding="utf-8", newline="\n")

    schedule: dict[str, Any] = {
        "schema_version": "vertirta.openfoam_schedule/v1.0.0",
        "license": "GPL-3.0-or-later (offline case generator; numeric output decoupled per GPLv3 Sec 0)",
        "mode": mode,
        "solver": solver,
        "turbulence_model": turb_model,
        "datum": {
            "lat_deg": slz_lat_deg,
            "lon_deg": slz_lon_deg,
            "alt_amsl_m": KARTAL_DATUM_ALT_AMSL_M,
            "egm2008_n_m": KARTAL_DATUM_EGM2008_N_M,
        },
        "domain_east_lon_deg": round(domain_east_lon, 6),
        "ltfj_max_lon_limit_deg": max_lon,
        "z0_m": z0_m,
        "rans_snapshot_count": len(RANS_DIRECTIONS_DEG) * len(RANS_SPEEDS_MPS),
        "rans_directions_deg": list(RANS_DIRECTIONS_DEG),
        "rans_speeds_mps": list(RANS_SPEEDS_MPS),
        "iddes_regimes": [
            {"name": n, "wind_from_deg": d, "u_ref_mps": u, "z0_m": z0}
            for n, d, u, z0 in IDDES_DOMINANT_REGIMES
        ],
    }
    (root / "snapshot_schedule.json").write_text(
        json.dumps(schedule, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate offline GPL-3.0 OpenFOAM coastal case.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--slz-lat-deg", type=float, default=KARTAL_DATUM_LAT_DEG)
    parser.add_argument("--slz-lon-deg", type=float, default=KARTAL_DATUM_LON_DEG)
    parser.add_argument("--wind-from-deg", type=float, default=225.0)
    parser.add_argument("--u-ref-mps", type=float, default=12.0)
    parser.add_argument("--mode", choices=("rans", "iddes"), default="rans")
    args = parser.parse_args(argv)
    generate_coastal_openfoam_case(
        args.out_dir,
        slz_lat_deg=args.slz_lat_deg,
        slz_lon_deg=args.slz_lon_deg,
        wind_from_deg=args.wind_from_deg,
        u_ref_mps=args.u_ref_mps,
        mode=args.mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
