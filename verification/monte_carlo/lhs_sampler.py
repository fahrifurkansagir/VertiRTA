"""Deterministic Stratified Latin Hypercube (LHS) sampler for VertiRTA Verification Pyramid."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from scipy.stats import qmc
import yaml

from verification.monte_carlo.wind_model import compute_wind_profile_mps


# 8-rotor coaxial X8 PX4 topology: upper rotors = {0,1,2,3} (Arms 0..3), lower rotors = {4,5,6,7}
SAME_ARM_PAIRS: tuple[tuple[int, int], ...] = ((0, 4), (1, 5), (2, 6), (3, 7))
CROSS_ARM_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(8) for j in range(i + 1, 8) if (i % 4) != (j % 4)
)
IMU_FAULT_MODES: tuple[str, ...] = ("FREEZE", "BIAS_RAMP", "STEP_HARDOVER", "NOISE_BURST")

TIER2_STRATA_COUNTS: dict[str, int] = {
    "S1": 100,
    "S2": 250,
    "S3": 250,
    "S4": 150,
    "S5": 150,
    "S6": 100,
}

TIER_MULTIPLIERS: dict[str, int] = {
    "tier2_1k": 1,
    "tier1_10k": 10,
}


def _lerp(u: float, lo: float, hi: float) -> float:
    return lo + u * (hi - lo)


def generate_stratified_manifest(
    odd_path: Path,
    tier: str = "tier2_1k",
    seed: int = 20261122,
) -> list[dict[str, Any]]:
    """Generate a deterministic, balanced LHS scenario manifest honoring ODD.yaml bounds."""
    if tier not in TIER_MULTIPLIERS:
        raise ValueError(f"Unsupported tier '{tier}'; expected one of {sorted(TIER_MULTIPLIERS)}")

    odd = yaml.safe_load(odd_path.read_text(encoding="utf-8"))
    od = odd["operational_domain"]
    bbox = od["scenery_elements"]["corridor_bounding_box"]
    lat_lo, lat_hi = float(bbox["lat_deg"][0]), float(bbox["lat_deg"][1])
    lon_lo, lon_hi = float(bbox["lon_deg"][0]), float(bbox["lon_deg"][1])
    if lon_hi >= 29.1900:
        raise ValueError(f"ODD corridor lon upper bound {lon_hi} violates LTFJ J2 west lock (< 29.1900)")

    wind_lo, wind_hi = map(float, od["environmental_elements"]["mean_wind_mps"]["range"])
    gust_lo, gust_hi = map(float, od["environmental_elements"]["turbulence_gust_mps"]["peak_gust_range_mps"])
    soc_lo, soc_hi = map(float, od["operating_parameters"]["battery_soc_pct"]["dispatch_range"])
    vel_lo, vel_hi = map(float, od["dynamic_elements"]["airspeed_horizontal_mps"])

    mult = TIER_MULTIPLIERS[tier]
    manifest: list[dict[str, Any]] = []
    run_id = 0

    for s_idx, (stratum, base_n) in enumerate(TIER2_STRATA_COUNTS.items()):
        n_stratum = base_n * mult
        sampler = qmc.LatinHypercube(d=6, seed=seed + 1009 * (s_idx + 1))
        samples = sampler.random(n=n_stratum)

        for local_i in range(n_stratum):
            u_lat, u_lon, u_w, u_g, u_soc, u_vel = map(float, samples[local_i])
            exposure = "coastal_d" if (local_i % 2 == 0) else "urban_b"

            if stratum == "S6":
                # Compound cliff-discovery probe: elevated wind, low SoC, eastern corridor edge
                mean_wind = _lerp(u_w, 10.0, 18.0)
                peak_gust = _lerp(u_g, 5.0, gust_hi)
                init_soc = _lerp(u_soc, 18.0, soc_lo)
                start_lon = _lerp(u_lon, max(lon_lo, 29.1650), lon_hi)
            else:
                mean_wind = _lerp(u_w, wind_lo, wind_hi)
                peak_gust = _lerp(u_g, gust_lo, gust_hi)
                init_soc = _lerp(u_soc, soc_lo, soc_hi)
                start_lon = _lerp(u_lon, lon_lo, lon_hi)

            start_lat = _lerp(u_lat, lat_lo, lat_hi)
            cruise_speed = _lerp(u_vel, vel_lo, vel_hi)
            u_60, sw_60, su_60 = compute_wind_profile_mps(mean_wind, 60.0, exposure)

            failed_motors: list[int] = []
            imu_fault: dict[str, Any] | None = None
            if stratum == "S1":
                fault_topology = "FLT_NONE"
                outcome_class = "NOMINAL"
            elif stratum == "S2":
                fault_topology = "FLT_1M"
                outcome_class = "CSFL"
                failed_motors = [local_i % 8]
            elif stratum == "S3":
                fault_topology = "FLT_2M_CROSS"
                outcome_class = "CSFL"
                failed_motors = list(CROSS_ARM_PAIRS[local_i % len(CROSS_ARM_PAIRS)])
            elif stratum == "S4":
                fault_topology = "FLT_2M_SAME_ARM"
                outcome_class = "CEL"
                failed_motors = list(SAME_ARM_PAIRS[local_i % len(SAME_ARM_PAIRS)])
            elif stratum == "S5":
                fault_topology = "FLT_IMU_1OF3"
                outcome_class = "CSFL"
                imu_fault = {
                    "channel": local_i % 3,
                    "mode": IMU_FAULT_MODES[local_i % len(IMU_FAULT_MODES)],
                }
            else:
                # S6 cycles across 1M, Cross-2M, and Same-Arm-2M keeping CEL/CSFL labels honest
                mod = local_i % 3
                if mod == 0:
                    fault_topology = "FLT_1M"
                    outcome_class = "CSFL"
                    failed_motors = [local_i % 8]
                elif mod == 1:
                    fault_topology = "FLT_2M_CROSS"
                    outcome_class = "CSFL"
                    failed_motors = list(CROSS_ARM_PAIRS[local_i % len(CROSS_ARM_PAIRS)])
                else:
                    fault_topology = "FLT_2M_SAME_ARM"
                    outcome_class = "CEL"
                    failed_motors = list(SAME_ARM_PAIRS[local_i % len(SAME_ARM_PAIRS)])

            manifest.append(
                {
                    "run_id": run_id,
                    "tier": tier,
                    "stratum": stratum,
                    "outcome_class": outcome_class,
                    "fault_topology": fault_topology,
                    "failed_motors": failed_motors,
                    "imu_fault": imu_fault,
                    "seed": seed + run_id,
                    "exposure": exposure,
                    "start_lat_deg": round(start_lat, 6),
                    "start_lon_deg": round(start_lon, 6),
                    "mean_wind_mps": round(mean_wind, 4),
                    "peak_gust_mps": round(peak_gust, 4),
                    "wind_at_60m_mps": round(u_60, 4),
                    "dryden_sigma_w_60m_mps": round(sw_60, 4),
                    "dryden_sigma_u_60m_mps": round(su_60, 4),
                    "initial_soc_pct": round(init_soc, 3),
                    "cruise_speed_mps": round(cruise_speed, 3),
                }
            )
            run_id += 1

    return manifest
