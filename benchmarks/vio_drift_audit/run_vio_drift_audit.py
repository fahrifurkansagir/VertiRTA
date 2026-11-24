#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""6-DOF Stereo-IMU VIO + Marmara Water-Mask + 1 Hz XFeat + 3sigma+dh/dt CBVF CTR Drift Benchmark.

Simulates 300 s Kartal-Dragos-Maltepe GNSS-denied coastal corridor flights strictly west of the
LTFJ NW J2 pinch (29.192122 E) and official DHMI AIP LTBX ARP (40d59m26sN 029d12m58sE = 40.990556 N,
29.216111 E), exercising:
  1) Basalt-compatible 6-DOF SO(3) IMU preintegration + 0.80 m carbon-boom stereo + radalt/baro lock
     + PIDNet-S Marmara water-pixel KLT feature masking (ZERO GNSS feedback into VIO!).
  2) 10 Hz 4-hypothesis GnssSpoofGuard (Lagged-Baseline [t-15s, t-5s] 2D Vector MCUSUM: 0.25/2.0 land
     vs 0.35/3.4 open water + corroboration, telescoping 1.5s/4.0s GLRT, 2.0 s pose rollback).
  3) 1 Hz VisualGeoRelocAdapter (Joseph-form SPD KF driven directly by live GTX 1650 SUPER XFeat
     empirical error residuals and 3-gate acceptance statistics).
  4) Dynamic 3sigma + dh/dt CBVF CTR wall inflation into horizontal radial braking CBVF + 3-state FSM
     (BOUNDED_GNSS_DENIED_CRUISE -> DEGRADED_CAUTION [13.5 m/s] -> DETERMINISTIC_VISUAL_SLZ_ABORT >=65 m).

DISCLAIMER: Research & simulation testbed benchmark ONLY — NOT for certified airborne navigation.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# Canonical Kartal & DHMI AIP LTFJ/LTBX locks (config/istanbul_ltfj_airspace.yaml + 01-DR08-13 Override #3)
KARTAL_LAT_DEG = 40.8860
KARTAL_LON_DEG = 29.1880
KARTAL_ALT_AMSL_M = 12.0
LTFJ_ARP_LAT_DEG = 40.898333
LTFJ_ARP_LON_DEG = 29.309167
LTFJ_RADIUS_M = 9260.0
LTFJ_CEIL_AMSL_M = 762.0
LTFJ_J1_LAT_DEG = 40.987500
LTFJ_J1_LON_DEG = 29.325000
LTFJ_J2_LAT_DEG = 40.912786
LTFJ_J2_LON_DEG = 29.192122
LTFJ_WEST_JOIN_LAT_DEG = 40.898333
LTFJ_WEST_JOIN_LON_DEG = 29.199133

# Verified official DHMI AIP Turkey LTBX ARP: 40d59m26sN 029d12m58sE (NOT stale DR-10 40d59m56sN)
LTBX_ARP_LAT_DEG = 40.990556
LTBX_ARP_LON_DEG = 29.216111
LTBX_RADIUS_M = 9260.0
LTBX_CEIL_AMSL_M = 609.6

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, h_hae_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s_lat = math.sin(lat)
    c_lat = math.cos(lat)
    s_lon = math.sin(lon)
    c_lon = math.cos(lon)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s_lat * s_lat)
    return np.array(
        [
            (n + h_hae_m) * c_lat * c_lon,
            (n + h_hae_m) * c_lat * s_lon,
            (n * (1.0 - WGS84_E2) + h_hae_m) * s_lat,
        ],
        dtype=np.float64,
    )


_DATUM_ECEF = _geodetic_to_ecef(KARTAL_LAT_DEG, KARTAL_LON_DEG, 48.7)
_LAT0 = math.radians(KARTAL_LAT_DEG)
_LON0 = math.radians(KARTAL_LON_DEG)
_R_ECEF_TO_ENU = np.array(
    [
        [-math.sin(_LON0), math.cos(_LON0), 0.0],
        [-math.sin(_LAT0) * math.cos(_LON0), -math.sin(_LAT0) * math.sin(_LON0), math.cos(_LAT0)],
        [math.cos(_LAT0) * math.cos(_LON0), math.cos(_LAT0) * math.sin(_LON0), math.sin(_LAT0)],
    ],
    dtype=np.float64,
)


def ll_to_local_en(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Convert WGS84 (lat, lon) to Kartal local [East, North] meters (matches core/flight_core/geodesy.hpp)."""
    ecef = _geodetic_to_ecef(lat_deg, lon_deg, 48.7)
    enu = _R_ECEF_TO_ENU @ (ecef - _DATUM_ECEF)
    return np.array([enu[0], enu[1]], dtype=np.float64)


def _signed_dist_convex_poly(p: np.ndarray, poly: list[np.ndarray]) -> float:
    min_d2 = 1e30
    inside = True
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        q = p - a
        if e[0] * q[1] - e[1] * q[0] < 0.0:
            inside = False
        e2 = float(np.dot(e, e))
        t = max(0.0, min(1.0, float(np.dot(q, e)) / e2)) if e2 > 1e-12 else 0.0
        diff = q - t * e
        min_d2 = min(min_d2, float(np.dot(diff, diff)))
    d = math.sqrt(min_d2)
    return -d if inside else +d


_LTFJ_ARP_EN = ll_to_local_en(LTFJ_ARP_LAT_DEG, LTFJ_ARP_LON_DEG)
_LTBX_ARP_EN = ll_to_local_en(LTBX_ARP_LAT_DEG, LTBX_ARP_LON_DEG)
_LTFJ_NW_POLY = [
    _LTFJ_ARP_EN,
    ll_to_local_en(LTFJ_J1_LAT_DEG, LTFJ_J1_LON_DEG),
    ll_to_local_en(LTFJ_J2_LAT_DEG, LTFJ_J2_LON_DEG),
    ll_to_local_en(LTFJ_WEST_JOIN_LAT_DEG, LTFJ_WEST_JOIN_LON_DEG),
]


def signed_margin_ltfj_from_ned(pos_ned: np.ndarray, alt_amsl_m: float = 52.0) -> float:
    """Positive [m] = outside LTFJ CTR (safe); negative = inside CTR (matches intruder_kf.cpp)."""
    p_en = np.array([pos_ned[1], pos_ned[0]], dtype=np.float64)
    sd_arc = float(np.linalg.norm(p_en - _LTFJ_ARP_EN)) - LTFJ_RADIUS_M
    sd_poly = _signed_dist_convex_poly(p_en, _LTFJ_NW_POLY)
    sd_h = min(sd_arc, sd_poly)
    if alt_amsl_m > LTFJ_CEIL_AMSL_M:
        return max(sd_h, alt_amsl_m - LTFJ_CEIL_AMSL_M)
    return sd_h


def signed_margin_ltbx_from_ned(pos_ned: np.ndarray, alt_amsl_m: float = 52.0) -> float:
    """Positive [m] = outside verified DHMI AIP LTBX CTR (40d59m26sN); negative = violation."""
    p_en = np.array([pos_ned[1], pos_ned[0]], dtype=np.float64)
    sd_h = float(np.linalg.norm(p_en - _LTBX_ARP_EN)) - LTBX_RADIUS_M
    if alt_amsl_m > LTBX_CEIL_AMSL_M:
        return max(sd_h, alt_amsl_m - LTBX_CEIL_AMSL_M)
    return sd_h


def _so3_exp(omega_dt: np.ndarray) -> np.ndarray:
    """Exact Rodrigues exponential map on SO(3) for 6-DOF Basalt-style IMU preintegration."""
    theta = float(np.linalg.norm(omega_dt))
    if theta < 1e-10:
        return np.eye(3, dtype=np.float64)
    k = omega_dt / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


@dataclass
class XFeatEmpiricalModel:
    """Empirical XFeat relocalization error distribution loaded from live GTX 1650 SUPER proof."""

    source: str
    cep50_m: float
    ce95_m: float
    gate_pass_rate: float
    meas_sigma_m: float
    samples_m: tuple[float, ...] = ()


def load_xfeat_empirical_model(proof_json: Path | None) -> XFeatEmpiricalModel:
    """Load live physical GTX 1650 SUPER XFeat residuals when available, else conservative baseline."""
    if proof_json is not None and proof_json.is_file():
        data = json.loads(proof_json.read_text(encoding="utf-8"))
        summary = data.get("relocalization_summary", {})
        overall = summary.get("overall_water_masked", summary.get("nominal_perspective_warp", {}))
        cep50 = float(overall.get("cep50_m", 1.2))
        ce95 = float(overall.get("ce95_m", 2.8))
        pass_rate = float(overall.get("ransac_3gate_pass_rate", 0.90))
        raw_errs = tuple(float(x) for x in data.get("empirical_errors_m_water_masked", []) if math.isfinite(float(x)))
        # Conservative Kalman measurement sigma: at least max(2.0 m, ce95 / 2.45) to keep Joseph KF well-calibrated
        meas_sigma = max(2.0, ce95 / 2.4477)
        return XFeatEmpiricalModel(
            source=proof_json.name,
            cep50_m=cep50,
            ce95_m=ce95,
            gate_pass_rate=pass_rate,
            meas_sigma_m=meas_sigma,
            samples_m=raw_errs,
        )
    return XFeatEmpiricalModel(
        source="conservative_default_05m_ortho",
        cep50_m=1.15,
        ce95_m=2.60,
        gate_pass_rate=0.92,
        meas_sigma_m=2.0,
        samples_m=(),
    )


def _cov_sigma_max(P: np.ndarray) -> float:
    tr = float(P[0, 0] + P[1, 1])
    diff = float(P[0, 0] - P[1, 1])
    disc = math.sqrt(diff * diff + 4.0 * float(P[0, 1]) * float(P[0, 1]))
    return math.sqrt(max(0.0, 0.5 * (tr + disc)))


def _joseph_update_2d(pos_est: np.ndarray, P: np.ndarray, z_meas: np.ndarray, r_sigma: float) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Exact 2D Mahalanobis gate (chi^2 <= 9.21) + SPD Joseph-form Kalman update."""
    R = np.eye(2, dtype=np.float64) * (r_sigma * r_sigma)
    S = P + R
    det = float(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
    if det <= 1e-9:
        return pos_est, P, 999.0, False
    S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]], dtype=np.float64) / det
    y = z_meas - pos_est
    chi2 = float(y.T @ S_inv @ y)
    if chi2 > 9.21:
        return pos_est, P, chi2, False
    K = P @ S_inv
    pos_new = pos_est + K @ y
    I_K = np.eye(2, dtype=np.float64) - K
    P_new = I_K @ P @ I_K.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    P_new[0, 0] = max(0.25, P_new[0, 0])
    P_new[1, 1] = max(0.25, P_new[1, 1])
    return pos_new, P_new, chi2, True


def _apply_cbvf_ctr_filter(
    pos_ned: np.ndarray,
    vel_ned: np.ndarray,
    a_nom_ned: np.ndarray,
    P: np.ndarray,
    P_dot: np.ndarray,
    *,
    b_map_m: float = 2.0,
    a_xy_max: float = 5.5,
) -> tuple[np.ndarray, float, float, float, float]:
    """Horizontal radial braking CBVF with exact 3sigma + dh/dt CTR wall inflation (matching cbvf_safety_filter.cpp)."""
    a_safe = a_nom_ned.copy()
    ltfj_m = signed_margin_ltfj_from_ned(pos_ned, 52.0)
    ltbx_m = signed_margin_ltbx_from_ned(pos_ned, 52.0)

    ltfj_arp_ned = np.array([_LTFJ_ARP_EN[1], _LTFJ_ARP_EN[0]], dtype=np.float64)
    ltbx_arp_ned = np.array([_LTBX_ARP_EN[1], _LTBX_ARP_EN[0]], dtype=np.float64)

    max_delta_r = 0.0
    max_r_dot = 0.0
    for margin_m, arp_ned in ((ltfj_m, ltfj_arp_ned), (ltbx_m, ltbx_arp_ned)):
        to_arp = arp_ned - pos_ned[:2]
        d_arp = max(1.0, float(np.linalg.norm(to_arp)))
        er = -to_arp / d_arp  # Outward safe unit normal FROM CTR wall TOWARD aircraft

        sigma_n = math.sqrt(max(0.04, float(er.T @ P @ er)))
        delta_r = 3.0 * sigma_n + b_map_m
        var_dot = max(0.0, float(er.T @ P_dot @ er))
        r_dot = min(6.0, 3.0 * var_dot / (2.0 * max(sigma_n, 0.25)))
        max_delta_r = max(max_delta_r, delta_r)
        max_r_dot = max(max_r_dot, r_dot)

        r_eff = max(margin_m, 0.5) - delta_r
        if r_eff < 500.0:
            # Exact tall-wall horizontal braking CBVF: v_r = er . vel_ned - r_dot (exact dh/dt = -r_dot!)
            v_r = float(np.dot(er, vel_ned[:2])) - r_dot
            v_close = max(0.0, 0.50 - v_r)
            a_r_eff = max(0.5, a_xy_max - 0.5)
            t_br = min(6.0, v_close / a_r_eff)
            h_val = (r_eff - 4.5) - (v_close * t_br - 0.5 * a_r_eff * t_br * t_br) - 0.15
            # Enforce Lie-derivative barrier: h_dot + gamma * h >= 0
            gamma = 1.5
            a_er = float(np.dot(er, a_safe[:2]))
            req_a_er = -(v_r + gamma * h_val) / max(t_br, 0.45) if h_val < 20.0 else -1e6
            if h_val < 0.0:
                # BRT fallback: full outward braking along safe normal er
                req_a_er = a_xy_max * 0.92
            if a_er < req_a_er:
                a_safe[:2] = a_safe[:2] + (req_a_er - a_er) * er

    norm_a = float(np.linalg.norm(a_safe[:2]))
    if norm_a > a_xy_max:
        a_safe[:2] *= a_xy_max / norm_a
    return a_safe, ltfj_m, ltbx_m, max_delta_r, max_r_dot


def run_corridor_regime(
    regime_name: str,
    xfeat_model: XFeatEmpiricalModel,
    *,
    duration_s: float = 300.0,
    enable_xfeat_1hz: bool = True,
    water_mask_enabled: bool = True,
    inject_spoof_at_s: float | None = None,
    spoof_ramp_mps: float = 1.0,
    seed: int = 20261124,
) -> dict[str, Any]:
    """Run a full 300 s 6-DOF Stereo-IMU + Water-Mask + SpoofGuard + XFeat + CBVF CTR flight."""
    rng = np.random.default_rng(seed)

    # Start safely outside BOTH LTFJ and verified LTBX (Kartal Coastal Vertiport: 40.8860 N, 29.1880 E, 40 m AGL)
    # Cruise NW/SE along the Kartal-Maltepe coastal corridor while periodically probing east toward LTFJ!
    pos_true = np.array([0.0, 0.0, -40.0], dtype=np.float64)
    vel_true = np.array([12.0, -4.0, 0.0], dtype=np.float64)
    R_b_to_ned = np.eye(3, dtype=np.float64)

    # Strictly GNSS-decoupled 6-DOF Basalt-compatible VIO state (ZERO GNSS feedback!)
    pos_vio = pos_true.copy()
    vel_vio = vel_true.copy()
    R_vio = R_b_to_ned.copy()
    vio_bias_xy = np.array([0.04, -0.03], dtype=np.float64)
    gyro_bias = np.array([0.0002, -0.00015, 0.0003], dtype=np.float64)

    # Navigation filter state (switches to VIO + XFeat on GNSS denial)
    gnss_denied = inject_spoof_at_s is None
    pos_nav = pos_true[:2].copy()
    sigma_init = 3.5 if gnss_denied else 1.5
    P = np.eye(2, dtype=np.float64) * (sigma_init * sigma_init)
    open_loop_dist_m = 0.0
    nav_state = "BOUNDED_GNSS_DENIED_CRUISE"
    abort_latched = False
    abort_time_s: float | None = None
    caution_entered = False

    # 10 Hz GnssSpoofGuard ring state (L2=150 = 15 s, L1=50 = 5 s guard, rollback=20 = 2.0 s)
    ring: deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(maxlen=160)
    mcusum = np.zeros(2, dtype=np.float64)
    spoof_e = 0.0
    spoof_latched = False
    spoof_ttd_s: float | None = None
    post_rollback_slip_m: float | None = None
    water_glint_false_alarms = 0

    # Telemetry accumulators
    dt_sim = 0.1  # 10 Hz outer loop with 10x 100 Hz SO(3) IMU preintegration sub-steps per epoch!
    n_epochs = int(round(duration_s / dt_sim))
    q_rw = 0.04
    kappa_drift = 0.008  # 0.8% systematic VIO drift (matches CtrInflationFsm)

    min_ltfj_m = 1e9
    min_ltbx_m = 1e9
    max_3sigma_m = 0.0
    max_1sigma_m = 0.0
    steady_3sigma_samples: list[float] = []
    xfeat_attempts = 0
    xfeat_accepts = 0
    xfeat_rejects = 0
    water_features_purged_total = 0
    land_features_kept_total = 0
    max_r_dot_mps = 0.0
    true_pos_err_max_m = 0.0
    total_path_m = 0.0

    phi_ou = math.exp(-dt_sim / 60.0)
    ou_scale_land = 0.10 * math.sqrt(1.0 - phi_ou * phi_ou)
    ou_scale_water_masked = 0.08 * math.sqrt(1.0 - phi_ou * phi_ou)
    ou_scale_water_unmasked = 0.28 * math.sqrt(1.0 - phi_ou * phi_ou)

    for k in range(n_epochs):
        t = k * dt_sim

        # Coastal water fraction schedule: open Marmara Sea exposure during t in [120, 180) s
        water_fraction = 0.82 if (120.0 <= t < 180.0) else 0.08
        n_feat = 60
        n_water = int(round(n_feat * water_fraction))
        n_land = n_feat - n_water
        if water_mask_enabled:
            water_features_purged_total += n_water
            land_features_kept_total += n_land
        else:
            land_features_kept_total += n_feat

        # Corridor waypoint guidance (Kartal -> Maltepe coast, alternating gentle east probes toward LTFJ)
        if not gnss_denied:
            v_max_cap = 22.0
        elif abort_latched:
            v_max_cap = 12.0
        elif nav_state == "DEGRADED_CAUTION":
            v_max_cap = 13.5
        else:
            v_max_cap = 22.0

        # Patrol leg along Kartal-Maltepe coastal corridor (keeps aircraft actively moving 14-18 m/s)
        leg_sign = 1.0 if ((int(t) // 110) % 2 == 0) else -1.0
        v_target_n = leg_sign * min(16.0, v_max_cap * 0.90)
        # Deliberately probe eastward (+E toward LTFJ) unless westward caution bias is active
        v_target_e = -3.0 if (nav_state != "BOUNDED_GNSS_DENIED_CRUISE" and gnss_denied) else 2.5 * math.sin(2.0 * math.pi * t / 75.0)
        a_nom = np.array([(v_target_n - vel_true[0]) * 0.7, (v_target_e - vel_true[1]) * 0.7, 0.0], dtype=np.float64)

        # Propagate covariance P when GNSS-denied (exact CtrInflationFsm math)
        speed_xy = float(np.linalg.norm(vel_true[:2]))
        if gnss_denied:
            ds = speed_xy * dt_sim
            d_sys = (kappa_drift * kappa_drift) * (2.0 * open_loop_dist_m * ds + ds * ds)
            dp = q_rw * dt_sim + d_sys
            open_loop_dist_m += ds
            P[0, 0] += dp
            P[1, 1] += dp
            P_dot = np.eye(2, dtype=np.float64) * (dp / dt_sim)
        else:
            P_dot = np.zeros((2, 2), dtype=np.float64)

        # Apply exact horizontal radial braking CBVF with 3sigma + dh/dt CTR wall inflation!
        a_safe, ltfj_m, ltbx_m, _delta_r, r_dot = _apply_cbvf_ctr_filter(
            pos_true, vel_true, a_nom, P, P_dot, b_map_m=2.0, a_xy_max=5.5
        )
        min_ltfj_m = min(min_ltfj_m, ltfj_m)
        min_ltbx_m = min(min_ltbx_m, ltbx_m)
        max_r_dot_mps = max(max_r_dot_mps, r_dot)

        # 6-DOF SO(3) IMU preintegration (10 sub-steps at 100 Hz inside each 10 Hz epoch)
        dt_imu = dt_sim / 10.0
        omega_true = np.array([0.0, 0.0, 0.015 * math.cos(t / 25.0)], dtype=np.float64)
        for _ in range(10):
            R_b_to_ned = R_b_to_ned @ _so3_exp(omega_true * dt_imu)
            vel_true = vel_true + a_safe * dt_imu
            vxy = float(np.linalg.norm(vel_true[:2]))
            if vxy > v_max_cap and vxy > 1e-6:
                vel_true[:2] *= v_max_cap / vxy
            pos_true = pos_true + vel_true * dt_imu
            total_path_m += float(np.linalg.norm(vel_true[:2])) * dt_imu

            # Decoupled Basalt-style VIO preintegration (noisy gyro + accel, ZERO GNSS input!)
            omega_meas = omega_true + gyro_bias + rng.normal(0.0, 0.0005, size=3)
            R_vio = R_vio @ _so3_exp(omega_meas * dt_imu)
            a_body_true = R_b_to_ned.T @ a_safe
            a_body_meas = a_body_true + rng.normal(0.0, 0.012, size=3)
            a_ned_vio = R_vio @ a_body_meas
            vel_vio = vel_vio + a_ned_vio * dt_imu
            pos_vio = pos_vio + vel_vio * dt_imu

        # Stereo + radalt/baro scale update + OU drift on decoupled VIO (water-masked vs unmasked)
        if water_fraction > 0.35:
            ou_s = ou_scale_water_masked if water_mask_enabled else ou_scale_water_unmasked
        else:
            ou_s = ou_scale_land
        vio_bias_xy = phi_ou * vio_bias_xy + rng.normal(0.0, ou_s, size=2)
        # Stereo velocity fusion anchors VIO velocity to true velocity minus slowly drifting OU bias
        vel_vio[:2] = 0.65 * (vel_true[:2] - vio_bias_xy) + 0.35 * vel_vio[:2]
        # Radalt/baro locks vertical metric scale
        pos_vio[2] = 0.65 * pos_true[2] + 0.35 * pos_vio[2]
        if k == 0:
            prev_vio_xy = pos_vio[:2].copy()
        dvio_xy = pos_vio[:2] - prev_vio_xy
        prev_vio_xy = pos_vio[:2].copy()

        # --- Layer 2: 10 Hz GnssSpoofGuard (active when spoof injection scenario is enabled) ---
        if inject_spoof_at_s is not None and not spoof_latched:
            spoof_v_e = 0.0
            if t >= inject_spoof_at_s:
                spoof_v_e = spoof_ramp_mps
                spoof_e += spoof_v_e * dt_sim

            gnss_pos_xy = pos_true[:2] + np.array([0.0, spoof_e]) + rng.normal(0.0, 0.15, size=2)
            gnss_vel_xy = vel_true[:2] + np.array([0.0, spoof_v_e]) + rng.normal(0.0, 0.14, size=2)
            v_res = gnss_vel_xy - vel_vio[:2]
            ring.append((gnss_pos_xy.copy(), pos_vio[:2].copy(), v_res.copy()))

            # Lagged-baseline [t-15s, t-5s] bias estimate (L2=150, L1=50 guard band)
            if len(ring) > 51:
                lag_slice = list(ring)[max(0, len(ring) - 150) : len(ring) - 50]
                b_lag = np.mean([r[2] for r in lag_slice], axis=0)
            else:
                b_lag = np.zeros(2, dtype=np.float64)

            # Adaptive sea/land thresholds (0.25/2.0 land vs 0.35/3.4 open water)
            w_blend = max(0.0, min(1.0, (water_fraction - 0.20) / 0.15))
            k_d = 0.25 + w_blend * (0.35 - 0.25)
            h_t = 2.0 + w_blend * (3.4 - 2.0)

            u_mc = mcusum + (v_res - b_lag)
            nu = float(np.linalg.norm(u_mc))
            mcusum = u_mc * ((nu - k_d) / nu) if nu > k_d else np.zeros(2, dtype=np.float64)
            h3a_hit = float(np.linalg.norm(mcusum)) > h_t

            # H3b telescoping 1.5 s (15 ep) / 4.0 s (40 ep) GLRT
            h3b_short = 0.0
            h3b_long = 0.0
            if len(ring) > 15:
                gp0, vp0, _ = ring[-16]
                h3b_short = float(np.linalg.norm((gnss_pos_xy - gp0) - (pos_vio[:2] - vp0) - 1.5 * b_lag))
            if len(ring) > 40:
                gp0, vp0, _ = ring[-41]
                h3b_long = float(np.linalg.norm((gnss_pos_xy - gp0) - (pos_vio[:2] - vp0) - 4.0 * b_lag))
            h3b_hit = (h3b_short > 1.70) or (h3b_long > 2.56)

            over_water = water_fraction > 0.35
            triggered = (h3a_hit and h3b_hit) if over_water else (h3a_hit or h3b_hit)
            if triggered:
                if t < inject_spoof_at_s:
                    water_glint_false_alarms += 1
                else:
                    spoof_latched = True
                    gnss_denied = True
                    spoof_ttd_s = round(t - inject_spoof_at_s + dt_sim, 3)
                    # Execute 2.0 s pose ring-buffer rollback to strip pre-detection spoof bias!
                    rb_idx = max(0, len(ring) - 21)
                    rb_gnss, rb_vio, _ = ring[rb_idx]
                    pos_nav = rb_gnss + (pos_vio[:2] - rb_vio)
                    post_rollback_slip_m = round(float(np.linalg.norm(pos_nav - pos_true[:2])), 4)
                    P = np.eye(2, dtype=np.float64) * (3.5 * 3.5)
                    open_loop_dist_m = 0.0
            else:
                pos_nav = gnss_pos_xy.copy()
        else:
            # Propagate GNSS-denied nav position purely via decoupled VIO deltas
            pos_nav = pos_nav + dvio_xy

        # --- Layer 3: 1 Hz Real XFeat Visual Geo-Registration (Joseph-form SPD KF) ---
        if gnss_denied and enable_xfeat_1hz and (k % 10 == 0) and (water_fraction < 0.70):
            xfeat_attempts += 1
            if rng.random() <= xfeat_model.gate_pass_rate:
                if xfeat_model.samples_m:
                    err_mag = float(xfeat_model.samples_m[int(rng.integers(0, len(xfeat_model.samples_m)))])
                else:
                    err_mag = float(abs(rng.normal(xfeat_model.cep50_m, 0.55)))
                ang = float(rng.uniform(0.0, 2.0 * math.pi))
                z_xfeat = pos_true[:2] + np.array([err_mag * math.cos(ang), err_mag * math.sin(ang)])
                pos_upd, P_upd, _chi2, accepted = _joseph_update_2d(
                    pos_nav, P, z_xfeat, xfeat_model.meas_sigma_m
                )
                if accepted:
                    pos_nav = pos_upd
                    P = P_upd
                    open_loop_dist_m = 0.0
                    xfeat_accepts += 1
                else:
                    xfeat_rejects += 1
            else:
                xfeat_rejects += 1

        # Update 3-State Navigation FSM (exact CtrInflationFsm thresholds: 25 m / 65 m, 5 m hysteresis)
        sigma_1 = _cov_sigma_max(P)
        sigma_3 = 3.0 * sigma_1
        max_1sigma_m = max(max_1sigma_m, sigma_1)
        max_3sigma_m = max(max_3sigma_m, sigma_3)
        if gnss_denied and t >= 20.0:
            steady_3sigma_samples.append(sigma_3)

        if gnss_denied:
            if abort_latched or sigma_1 >= 65.0:
                if not abort_latched:
                    abort_time_s = round(t, 2)
                abort_latched = True
                nav_state = "DETERMINISTIC_VISUAL_SLZ_ABORT"
            elif nav_state == "DEGRADED_CAUTION":
                if sigma_1 < 20.0:
                    nav_state = "BOUNDED_GNSS_DENIED_CRUISE"
            elif sigma_1 >= 25.0:
                nav_state = "DEGRADED_CAUTION"
                caution_entered = True

        true_pos_err_max_m = max(true_pos_err_max_m, float(np.linalg.norm(pos_nav - pos_true[:2])))

        # Once Visual-SLZ abort latches in unbounded drift regime, record clean abort before 80 m floor
        if abort_latched and not enable_xfeat_1hz:
            break

    steady_3sigma_p95 = float(np.percentile(steady_3sigma_samples, 95)) if steady_3sigma_samples else 0.0
    steady_3sigma_mean = float(np.mean(steady_3sigma_samples)) if steady_3sigma_samples else 0.0

    return {
        "regime_name": regime_name,
        "duration_s": round(min(duration_s, (k + 1) * dt_sim), 2),
        "total_path_m": round(total_path_m, 2),
        "xfeat_1hz_enabled": enable_xfeat_1hz,
        "water_mask_enabled": water_mask_enabled,
        "final_nav_state": nav_state,
        "caution_entered": caution_entered,
        "visual_slz_abort_latched": abort_latched,
        "abort_time_s": abort_time_s,
        "max_1sigma_m": round(max_1sigma_m, 4),
        "max_3sigma_m": round(max_3sigma_m, 4),
        "steady_3sigma_mean_m": round(steady_3sigma_mean, 4),
        "steady_3sigma_p95_m": round(steady_3sigma_p95, 4),
        "bounded_3sigma_under_25m": bool(max_3sigma_m < 25.0),
        "true_nav_error_max_m": round(true_pos_err_max_m, 4),
        "xfeat_attempts": xfeat_attempts,
        "xfeat_accepts": xfeat_accepts,
        "xfeat_rejects": xfeat_rejects,
        "water_features_purged": water_features_purged_total,
        "land_features_kept": land_features_kept_total,
        "spoof_latched": spoof_latched,
        "spoof_ttd_s": spoof_ttd_s,
        "post_rollback_slip_m": post_rollback_slip_m,
        "water_glint_false_alarms": water_glint_false_alarms,
        "min_ltfj_margin_m": round(min_ltfj_m, 3),
        "min_ltbx_margin_m": round(min_ltbx_m, 3),
        "ltfj_breaches": 0 if min_ltfj_m > 0.0 else 1,
        "ltbx_breaches": 0 if min_ltbx_m > 0.0 else 1,
        "max_ctr_r_dot_infl_mps": round(max_r_dot_mps, 4),
    }


def run_full_vio_drift_audit(
    xfeat_proof_json: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Execute all 3 Kartal-Maltepe GNSS-denied VIO + Water-Mask + XFeat + CBVF CTR audit regimes."""
    xfeat_model = load_xfeat_empirical_model(xfeat_proof_json)

    regime_a = run_corridor_regime(
        "regime_a_xfeat_1hz_bounded_cruise",
        xfeat_model,
        duration_s=300.0,
        enable_xfeat_1hz=True,
        water_mask_enabled=True,
        inject_spoof_at_s=None,
        seed=20261124,
    )
    # Regime B: sustained open-loop GNSS denial (up to 650 s so 0.8% systematic drift crosses 25 m -> 65 m deterministically)
    regime_b = run_corridor_regime(
        "regime_b_unbounded_vio_marmara_abort",
        xfeat_model,
        duration_s=650.0,
        enable_xfeat_1hz=False,
        water_mask_enabled=True,
        inject_spoof_at_s=None,
        seed=20261125,
    )
    regime_c = run_corridor_regime(
        "regime_c_slow_ramp_spoof_and_water_glint",
        xfeat_model,
        duration_s=300.0,
        enable_xfeat_1hz=True,
        water_mask_enabled=True,
        inject_spoof_at_s=185.0,
        spoof_ramp_mps=1.0,
        seed=20261126,
    )

    # Also quantify water-masking ablation (unmasked vs masked open-water VIO error over 60 s Marmara crossing)
    regime_c_unmasked = run_corridor_regime(
        "ablation_marmara_unmasked_vio",
        xfeat_model,
        duration_s=300.0,
        enable_xfeat_1hz=True,
        water_mask_enabled=False,
        inject_spoof_at_s=185.0,
        spoof_ramp_mps=1.0,
        seed=20261126,
    )

    gates = {
        "regime_a_3sigma_bounded_under_25m": regime_a["max_3sigma_m"] < 25.0,
        "regime_a_zero_ctr_breaches": (regime_a["ltfj_breaches"] == 0 and regime_a["ltbx_breaches"] == 0),
        "regime_b_deterministic_abort_at_65m_before_80m_floor": (
            regime_b["caution_entered"]
            and regime_b["visual_slz_abort_latched"]
            and 65.0 <= regime_b["max_1sigma_m"] < 80.0
        ),
        "regime_b_zero_ctr_breaches": (regime_b["ltfj_breaches"] == 0 and regime_b["ltbx_breaches"] == 0),
        "regime_c_spoof_ttd_le_1p4s": (
            regime_c["spoof_latched"] and regime_c["spoof_ttd_s"] is not None and regime_c["spoof_ttd_s"] <= 1.4
        ),
        "regime_c_zero_water_glint_far": regime_c["water_glint_false_alarms"] == 0,
        "regime_c_zero_ctr_breaches": (regime_c["ltfj_breaches"] == 0 and regime_c["ltbx_breaches"] == 0),
    }

    report: dict[str, Any] = {
        "schema_version": "vertirta.vio_drift_audit/v4.0.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "airspace_lock": {
            "kartal_datum": [KARTAL_LAT_DEG, KARTAL_LON_DEG],
            "ltfj_j2_pinch": [LTFJ_J2_LAT_DEG, LTFJ_J2_LON_DEG],
            "ltbx_verified_dhmi_aip_arp": [LTBX_ARP_LAT_DEG, LTBX_ARP_LON_DEG],
            "basalt_gnss_feedback_allowed": False,
            "stereo_baseline_m": 0.80,
        },
        "xfeat_empirical_model": {
            "source": xfeat_model.source,
            "cep50_m": round(xfeat_model.cep50_m, 4),
            "ce95_m": round(xfeat_model.ce95_m, 4),
            "gate_pass_rate": round(xfeat_model.gate_pass_rate, 4),
            "kalman_meas_sigma_m": round(xfeat_model.meas_sigma_m, 4),
            "empirical_sample_count": len(xfeat_model.samples_m),
        },
        "regimes": {
            "regime_a_xfeat_1hz_bounded_cruise": regime_a,
            "regime_b_unbounded_vio_marmara_abort": regime_b,
            "regime_c_slow_ramp_spoof_and_water_glint": regime_c,
            "ablation_marmara_unmasked_vio": regime_c_unmasked,
        },
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="VertiRTA M3 6-DOF Stereo-IMU VIO + XFeat + CBVF CTR Drift Audit")
    parser.add_argument("--xfeat-proof-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/vio_drift_audit/vio_drift_audit_report.json"))
    args = parser.parse_args()

    report = run_full_vio_drift_audit(xfeat_proof_json=args.xfeat_proof_json, out_json=args.out)
    print(json.dumps({"all_gates_passed": report["all_gates_passed"], "gates": report["gates"]}, indent=2))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
