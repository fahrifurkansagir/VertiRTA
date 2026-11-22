# SPDX-License-Identifier: Apache-2.0
"""Stratified 8,000-frame Maltepe-Dragos-Kartal Gazebo Harmonic EGL dataset schedule generator.

Enforces DR-7 & Spec v1.2 locks:
  - Kartal Coastal Vertiport ENU datum: 40.8860 N, 29.1880 E, 12.0 m AMSL (NEVER 40.8880 N)
  - Strictly Copernicus DEM GLO-30 + 12 procedural coastal blocks (uses_osm_odbl=False ALWAYS)
  - Exact AGL strata: 35% low [15, 30) m, 40% mid [30, 65) m, 25% high [65, 120] m
  - Exact 15% OOD synthetic rooftop outlier injection (label=254)
  - Gimbal off-nadir pitch in [0, 35] deg, quadratic depth noise sigma_Z = 0.002 * Z^2
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.slz_perception.data.license_gate import DatasetEntry, verify_training_manifest

LOCKED_DATUM_LAT = 40.8860
LOCKED_DATUM_LON = 29.1880
LOCKED_DATUM_ALT_AMSL_M = 12.0

# 12 Copernicus GLO-30 procedural coastal block centers in local Kartal ENU (E <= 0, west of LTFJ J2)
PROCEDURAL_COASTAL_BLOCKS_ENU: tuple[tuple[float, float, float], ...] = (
    (-60.0, 40.0, 18.4),
    (-140.0, 85.0, 22.4),
    (-240.0, 130.0, 16.4),
    (-380.0, 210.0, 21.5),
    (-560.0, 320.0, 24.4),
    (-780.0, 480.0, 15.4),
    (-1050.0, 690.0, 28.4),
    (-1380.0, 940.0, 19.4),
    (-1760.0, 1240.0, 23.4),
    (-2200.0, 1620.0, 17.4),
    (-2740.0, 2180.0, 21.4),
    (-3320.0, 2780.0, 26.4),
)


@dataclass(frozen=True)
class FrameSpec:
    """Single deterministic frame specification for Colab-G4 Gazebo Harmonic EGL capture."""

    frame_id: int
    stratum: str
    datum_lat: float
    datum_lon: float
    datum_alt_amsl_m: float
    uses_osm_odbl: bool
    coastal_block_id: int
    pos_enu_m: tuple[float, float, float]
    agl_m: float
    pitch_off_nadir_deg: float
    gimbal_pitch_down_rad: float
    yaw_deg: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    fog_visibility_m: float
    inject_ood_254: bool
    inject_dynamic_40: bool
    depth_noise_coeff: float
    depth_dropout_prob: float
    motion_blur_px: float


def load_locked_istanbul_datum(
    config_path: Path | None = None,
) -> tuple[float, float, float]:
    """Read Kartal Coastal Vertiport datum from config/istanbul_ltfj_airspace.yaml and assert lock."""
    if config_path is None:
        candidate = Path(__file__).resolve().parents[3] / "config" / "istanbul_ltfj_airspace.yaml"
        config_path = candidate if candidate.is_file() else None

    if config_path is not None and config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        datum = cfg["datum"]
        lat = float(datum["lat_deg"])
        lon = float(datum["lon_deg"])
        alt = float(datum["alt_amsl_m"])
        if (
            abs(lat - LOCKED_DATUM_LAT) > 1e-6
            or abs(lon - LOCKED_DATUM_LON) > 1e-6
            or abs(alt - LOCKED_DATUM_ALT_AMSL_M) > 1e-6
        ):
            raise ValueError(
                f"Airspace datum mismatch in {config_path}: got ({lat}, {lon}, {alt}), "
                f"expected ({LOCKED_DATUM_LAT}, {LOCKED_DATUM_LON}, {LOCKED_DATUM_ALT_AMSL_M})"
            )
        return lat, lon, alt

    return LOCKED_DATUM_LAT, LOCKED_DATUM_LON, LOCKED_DATUM_ALT_AMSL_M


def generate_sampling_schedule(
    n_frames: int = 8000,
    seed: int = 42,
    *,
    airspace_config: Path | None = None,
) -> list[FrameSpec]:
    """Generate a stratified, zero-OSM-ODbL sampling schedule for Gazebo-Istanbul 8K."""
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")

    datum_lat, datum_lon, datum_alt = load_locked_istanbul_datum(airspace_config)
    rng = np.random.default_rng(seed)

    n_low = int(round(0.35 * n_frames))
    n_mid = int(round(0.40 * n_frames))
    n_high = n_frames - n_low - n_mid
    n_ood = int(round(0.15 * n_frames))
    n_dyn = int(round(0.25 * n_frames))

    strata = (
        ["low_15_30m"] * n_low
        + ["mid_30_65m"] * n_mid
        + ["high_65_120m"] * n_high
    )
    rng.shuffle(strata)

    ood_flags = np.zeros(n_frames, dtype=bool)
    ood_flags[:n_ood] = True
    rng.shuffle(ood_flags)

    dyn_flags = np.zeros(n_frames, dtype=bool)
    dyn_flags[:n_dyn] = True
    rng.shuffle(dyn_flags)

    # Balanced coverage across all 12 Copernicus GLO-30 procedural blocks
    block_ids = np.arange(n_frames, dtype=np.int64) % len(PROCEDURAL_COASTAL_BLOCKS_ENU) + 1
    rng.shuffle(block_ids)

    schedule: list[FrameSpec] = []
    for idx in range(n_frames):
        stratum = strata[idx]
        if stratum == "low_15_30m":
            agl_m = float(rng.uniform(15.0, 30.0 - 1e-4))
            pitch_off_nadir = float(rng.uniform(10.0, 35.0))
        elif stratum == "mid_30_65m":
            agl_m = float(rng.uniform(30.0, 65.0 - 1e-4))
            pitch_off_nadir = float(rng.uniform(0.0, 30.0))
        else:
            agl_m = float(rng.uniform(65.0, 120.0))
            pitch_off_nadir = float(rng.uniform(0.0, 20.0))

        block_id = int(block_ids[idx])
        bx, by, bz = PROCEDURAL_COASTAL_BLOCKS_ENU[block_id - 1]
        lateral_r = float(rng.uniform(0.0, 22.0))
        lateral_az = float(rng.uniform(0.0, 2.0 * math.pi))
        pos_e = round(min(bx + lateral_r * math.cos(lateral_az), -5.0), 3)
        pos_n = round(by + lateral_r * math.sin(lateral_az), 3)
        pos_u = round(bz + agl_m, 3)

        # Off-nadir angle 0 deg => pitch_down = pi/2 (nadir); 35 deg => pitch_down = 55 deg
        pitch_down_rad = round(math.radians(90.0 - pitch_off_nadir), 6)

        schedule.append(
            FrameSpec(
                frame_id=idx,
                stratum=stratum,
                datum_lat=datum_lat,
                datum_lon=datum_lon,
                datum_alt_amsl_m=datum_alt,
                uses_osm_odbl=False,
                coastal_block_id=block_id,
                pos_enu_m=(pos_e, pos_n, pos_u),
                agl_m=round(agl_m, 4),
                pitch_off_nadir_deg=round(pitch_off_nadir, 4),
                gimbal_pitch_down_rad=pitch_down_rad,
                yaw_deg=round(float(rng.uniform(0.0, 360.0)), 4),
                sun_azimuth_deg=round(float(rng.uniform(0.0, 360.0)), 4),
                sun_elevation_deg=round(float(rng.uniform(12.0, 72.0)), 4),
                fog_visibility_m=round(float(rng.uniform(80.0, 5000.0)), 2),
                inject_ood_254=bool(ood_flags[idx]),
                inject_dynamic_40=bool(dyn_flags[idx]),
                depth_noise_coeff=0.002,
                depth_dropout_prob=0.02,
                motion_blur_px=round(float(rng.uniform(1.0, 5.0)), 3),
            )
        )

    return schedule


def apply_quadratic_depth_noise(
    depth_m: np.ndarray,
    rng: np.random.Generator,
    *,
    coeff: float = 0.002,
    dropout_prob: float = 0.02,
    min_m: float = 0.5,
    max_m: float = 150.0,
) -> np.ndarray:
    """Apply DR-7 quadratic stereo depth noise sigma_Z = coeff * Z^2 and random dropout."""
    z = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(z) & (z >= min_m) & (z <= max_m)
    sigma = np.float32(coeff) * (z * z)
    noise = rng.normal(loc=0.0, scale=1.0, size=z.shape).astype(np.float32) * sigma
    noisy = np.clip(z + noise, np.float32(min_m), np.float32(max_m))
    dropout = rng.random(size=z.shape) < dropout_prob
    out = np.where(valid & (~dropout), noisy, np.float32(np.nan))
    return out.astype(np.float32, copy=False)


def export_schedule_manifest(schedule: list[FrameSpec], out_json: Path | str) -> Path:
    """Validate Gazebo_Istanbul_8K license lock and write the stratified schedule to JSON."""
    if not schedule:
        raise ValueError("Cannot export an empty sampling schedule.")
    if any(f.uses_osm_odbl for f in schedule):
        raise ValueError("Schedule contains uses_osm_odbl=True; forbidden for training weights.")

    verify_training_manifest(
        [
            DatasetEntry(
                name="Gazebo_Istanbul_8K",
                split="train",
                license="Apache-2.0",
                uses_osm_odbl=False,
                notes="Copernicus GLO-30 + 12 procedural coastal blocks; zero OSM ODbL.",
            )
        ]
    )

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": "vertirta_istanbul_8k_schedule_v1.0",
        "license": "Apache-2.0",
        "uses_osm_odbl": False,
        "datum": {
            "lat_deg": schedule[0].datum_lat,
            "lon_deg": schedule[0].datum_lon,
            "alt_amsl_m": schedule[0].datum_alt_amsl_m,
        },
        "n_frames": len(schedule),
        "frames": [asdict(f) for f in schedule],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate VertiRTA Gazebo-Istanbul 8K stratified capture schedule."
    )
    parser.add_argument("--n-frames", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/istanbul_8k_schedule.json"),
    )
    args = parser.parse_args()
    sched = generate_sampling_schedule(n_frames=args.n_frames, seed=args.seed)
    export_schedule_manifest(sched, args.out)


if __name__ == "__main__":
    main()
