#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-2 G4 Job (XLA 0.58): High-Res 101x71x101x71 X8-240 & Lift+Cruise 8+1 4D HJI-CBVF PDE.

Implements Plan 6C Task 4 (#316):
  - Pins XLA_PYTHON_CLIENT_PREALLOCATE=true, XLA_PYTHON_CLIENT_MEM_FRACTION<=0.58,
    JAX_ENABLE_X64=False strictly BEFORE importing JAX so Wave-2 JAX/XLA (58%) and
    co-resident PyTorch PIDNet-S v2.1 (30%) NEVER collide on 96 GB G4.
  - Safe OOM-proof default `vmap_batch_size=2` (2x2 regimes = 102.8M simultaneous nodes,
    ~34–48 GB working set inside 55.44 GiB XLA BFC pool; `vmap=4` strictly opt-in behind
    `--allow-vmap4` + jax.checkpoint remat + preflight memory check).
  - Handles the half-integer ratio (101-1)/(41-1) = 100/40 = 2.5 and (71-1)/(29-1) = 70/28 = 2.5
    soundly via exact quadrilinear projection + chunked 51,423,241-node slab-by-slab optimistic
    overestimation audit (`eps_grid_f = max_{|V|<=10m} max(0, V_hat_41x29 - clip(V_hires)) + 0.5*q`).
  - Exports BOTH:
      1. Lossless 101x71x101x71 archives tagged magic `"VRTALUT2"` (411,389,928 B, out-of-git)
         so frozen `MmapCbvf4DLut` (`VRTALUT1`) can never mis-load a 101x71 file.
      2. Exact 11,313,864 B `"VRTALUT1"` (41x29x41x29) drop-in runtime archives for BOTH
         Coaxial X8-240 and Lift+Cruise 8+1 (canonical constants matching
         `core/airframes/lift_cruise/include/vertirta/airframes/lift_cruise_dynamics.hpp`).
"""

from __future__ import annotations

import gc
import math
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import traceback
from typing import Any

# MUST pin XLA allocator bounds BEFORE any jax import!
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.58")
os.environ.setdefault("JAX_ENABLE_X64", "False")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reachability.cbvf_4d_dynamics import (  # noqa: E402
    DEFAULT_GRAVITY_M_S2,
    HAS_HJ_REACHABILITY,
    HAS_JAX,
    RegimeSpec,
    VertiRta4DDynamics,
    _DynamicsBase,
    _xp,
    build_regimes_from_config,
    compute_all_regime_v_caps,
    conservative_smax_4,
    hj,
    jax,
    jnp,
)
from reachability.solve_cbvf_luts import (  # noqa: E402
    FLAG_CONSERVATIVE_SMAX,
    FLAG_HAS_V_CAP,
    FLAG_QUICK_CPU_FIXTURE,
    LUT_BETA_SMAX,
    LUT_EXPORT_SHAPE,
    LUT_HEADER_BYTES,
    LUT_HI,
    LUT_HORIZON_S,
    LUT_LO,
    LUT_MAGIC,
    LUT_NUM_REGIMES,
    LUT_OFFSET,
    LUT_SCALE,
    LUT_TOTAL_BYTES,
    LUT_V_HI,
    LUT_V_LO,
    LUT_VERSION,
    VrtaLutHeader,
    _quantize_int16,
    _solve_regime_quick_cpu,
    pack_lut_header,
)
from reachability.solve_cel_2d_lut import (  # noqa: E402
    CEL2D_TOTAL_BYTES,
    solve_and_export_cel_2d,
)
import reachability.verify_lut_soundness as _vls_mod  # noqa: E402
from reachability.verify_lut_soundness import verify_all_soundness  # noqa: E402
from scripts.colab.g4_jobs.common import (  # noqa: E402
    WAVE_BUDGETS,
    build_standard_argparser,
    query_nvidia_smi_snapshot,
    seed_everything,
    sha256_file,
    validate_job_license_manifest,
    write_canonical_json,
)

HIRES_SHAPE: tuple[int, int, int, int] = (101, 71, 101, 71)
HIRES_NODES_PER_REGIME: int = 101 * 71 * 101 * 71  # 51,423,241
LUT_HIRES_MAGIC: bytes = b"VRTALUT2"

# Canonical VertiRTA-LC-240 Lift+Cruise 8+1 constants matching
# core/airframes/lift_cruise/include/vertirta/airframes/lift_cruise_dynamics.hpp
LC_MASS_KG = 240.0
LC_WEIGHT_N = LC_MASS_KG * DEFAULT_GRAVITY_M_S2  # 2353.596 N
LC_T_MAX_LIFT_ROTOR_N = 544.3
LC_T_MAX_PUSHER_N = 950.0
LC_WING_AREA_M2 = 5.20
LC_RHO_AIR = 1.225
LC_CD0 = 0.03198
# Streamlined fixed-wing quadratic drag deceleration factor: 0.5 * rho * S * C_D0 / m = 0.0004244 m^-1
# (9.6x lower than X8 flat-plate cd_r=0.00408 so V_NE=-48 m/s never overestimates aero braking!)
LC_CD_R = (0.5 * LC_RHO_AIR * LC_WING_AREA_M2 * LC_CD0) / LC_MASS_KG  # ~0.0004244
LC_CD_Z = 0.00613  # Conservative lower-bound vertical drag floor
LC_STALL_1G_MPS = 21.83
LC_BLEND_START_MPS = 14.0
LC_BLEND_END_MPS = 28.0
LC_CRUISE_MPS = 38.0
LC_VNE_MPS = 46.3
LC_BEST_GLIDE_LD = 13.46
LC_LUT_LO: tuple[float, float, float, float] = (-2.0, -48.0, -5.0, -12.0)
LC_LUT_HI: tuple[float, float, float, float] = (118.0, 20.0, 55.0, 8.0)


def build_lift_cruise_regimes() -> tuple[RegimeSpec, ...]:
    """4 canonical Lift+Cruise 8+1 regimes aligned to lift_cruise_dynamics.hpp (streamlined cd_r)."""
    tw_hover_nom = (8.0 * LC_T_MAX_LIFT_ROTOR_N) / LC_WEIGHT_N  # 1.8501
    # Hover OEI (<= 79.2% T_max trim ceiling leaves >20.8% dynamic control margin -> T/W_eff ~ 1.46)
    tw_hover_oei = 1.46
    # Transition corridor [14..28 m/s]: wing lift + lift rotors + pusher authority
    tw_transition = 1.65
    # Cruise fixed-wing degraded (38 m/s): wing load factor + pusher thrust (950 N / 240 kg = 3.96 m/s^2)
    tw_cruise_fw = 1.55
    return (
        RegimeSpec(
            index=0, name="LC-Hover-Nominal", tw_ratio=tw_hover_nom, tilt_max_deg=25.0,
            cd_r=LC_CD_R, cd_z=LC_CD_Z, r_safe=4.0, z_safe=2.5,
        ),
        RegimeSpec(
            index=1, name="LC-Hover-OEI-8p1", tw_ratio=tw_hover_oei, tilt_max_deg=18.0,
            cd_r=LC_CD_R, cd_z=LC_CD_Z, r_safe=4.0, z_safe=2.5,
        ),
        RegimeSpec(
            index=2, name="LC-Transition-Corridor", tw_ratio=tw_transition, tilt_max_deg=22.0,
            cd_r=LC_CD_R, cd_z=LC_CD_Z, r_safe=4.0, z_safe=2.5,
        ),
        RegimeSpec(
            index=3, name="LC-Cruise-FW-Degraded", tw_ratio=tw_cruise_fw, tilt_max_deg=28.0,
            cd_r=LC_CD_R, cd_z=LC_CD_Z, r_safe=4.0, z_safe=2.5,
        ),
    )


def verify_lift_cruise_lut_soundness(
    lc_lut_path: Path,
    expected_lo: tuple[float, float, float, float] = LC_LUT_LO,
    expected_hi: tuple[float, float, float, float] = LC_LUT_HI,
) -> dict[str, Any]:
    """Explicit binary size, SHA-256, domain bounds, finiteness, and regime-order audit for LC VRTALUT1."""
    from reachability.solve_cbvf_luts import load_vrta_lut

    failures: list[str] = []
    if not lc_lut_path.is_file() or lc_lut_path.stat().st_size != LUT_TOTAL_BYTES:
        failures.append(f"LC LUT size != {LUT_TOTAL_BYTES}")
        return {"passed": False, "failures": failures}

    hdr, data = load_vrta_lut(lc_lut_path, verify_sha256=True)
    if hdr.magic != LUT_MAGIC:
        failures.append(f"Invalid magic {hdr.magic!r}")
    if tuple(hdr.dims) != LUT_EXPORT_SHAPE:
        failures.append(f"Invalid dims {hdr.dims}")
    if any(abs(a - b) > 1e-4 for a, b in zip(hdr.lo, expected_lo)):
        failures.append(f"LC lo mismatch: {hdr.lo} != {expected_lo}")
    if any(abs(a - b) > 1e-4 for a, b in zip(hdr.hi, expected_hi)):
        failures.append(f"LC hi mismatch: {hdr.hi} != {expected_hi}")
    if any((not math.isfinite(e) or e < 0.0) for e in hdr.eps_grid):
        failures.append(f"Invalid eps_grid: {hdr.eps_grid}")
    # Hover Nominal (regime 0) must dominate Hover OEI (regime 1) conservatively
    v0 = data[0].astype(np.float32) * np.float32(hdr.scale) + np.float32(hdr.offset)
    v1 = data[1].astype(np.float32) * np.float32(hdr.scale) + np.float32(hdr.offset)
    if not (np.all(np.isfinite(v0)) and np.all(np.isfinite(v1))):
        failures.append("Non-finite values in LC LUT payload")
    max_order_viol = float(np.max(v1 - v0))
    if max_order_viol > float(hdr.scale) * 1.5:
        failures.append(f"Hover-Nominal >= Hover-OEI violated by {max_order_viol:.4f} m")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "size_bytes": lc_lut_path.stat().st_size,
        "lo": [round(x, 4) for x in hdr.lo],
        "hi": [round(x, 4) for x in hdr.hi],
        "eps_grid_m": [round(x, 6) for x in hdr.eps_grid],
        "hover_nom_ge_oei_max_viol_m": round(max(0.0, max_order_viol), 6),
    }


def _axis_linear_weights(n_src: int, n_dst: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map n_dst evenly spaced coordinates onto [0, n_src - 1] with exact linear weights."""
    coords = np.linspace(0.0, float(n_src - 1), n_dst, dtype=np.float64)
    i0 = np.floor(coords).astype(np.int32)
    i1 = np.minimum(i0 + 1, n_src - 1)
    w = (coords - i0).astype(np.float32)
    return i0, i1, w


def project_4d_quadrilinear(v_src: np.ndarray, dst_shape: tuple[int, int, int, int]) -> np.ndarray:
    """Exact separable quadrilinear resampling between arbitrary 4D grids (e.g. 101x71 -> 41x29)."""
    cur = np.asarray(v_src, dtype=np.float32)
    for axis, n_dst in enumerate(dst_shape):
        n_src = cur.shape[axis]
        if n_src == n_dst:
            continue
        i0, i1, w = _axis_linear_weights(n_src, n_dst)
        w_shape = [1, 1, 1, 1]
        w_shape[axis] = n_dst
        w_b = w.reshape(w_shape)
        s0 = np.take(cur, i0, axis=axis)
        s1 = np.take(cur, i1, axis=axis)
        cur = s0 + w_b * (s1 - s0)
    return np.ascontiguousarray(cur, dtype=np.float32)


def _resample_3d_trilinear(slab_3d: np.ndarray, dst_3d: tuple[int, int, int]) -> np.ndarray:
    cur = np.asarray(slab_3d, dtype=np.float32)
    for axis, n_dst in enumerate(dst_3d):
        n_src = cur.shape[axis]
        if n_src == n_dst:
            continue
        i0, i1, w = _axis_linear_weights(n_src, n_dst)
        w_shape = [1, 1, 1]
        w_shape[axis] = n_dst
        w_b = w.reshape(w_shape)
        s0 = np.take(cur, i0, axis=axis)
        s1 = np.take(cur, i1, axis=axis)
        cur = s0 + w_b * (s1 - s0)
    return cur


def audit_exact_hires_eps_grid_chunked(
    v_fine_fp32: np.ndarray,
    v_deq_coarse_fp32: np.ndarray,
    *,
    band_limit_m: float = 10.0,
) -> float:
    """Chunked exact quadrilinear overestimation audit for arbitrary fine->coarse ratio (e.g. 2.5:1).

    Computes over ALL fine nodes inside |V_fine| <= band_limit_m:
        eps_grid_f = max(0, V_hat_coarse(x_fine) - clip(V_fine(x_fine))) + 0.5 * q
    Peak host RAM per slab is <= 30 MB even for 101x71x101x71 (51.4M nodes).
    """
    fine_shape = v_fine_fp32.shape
    coarse_shape = v_deq_coarse_fp32.shape
    i0_r, i1_r, w_r = _axis_linear_weights(coarse_shape[0], fine_shape[0])
    dst_3d = (fine_shape[1], fine_shape[2], fine_shape[3])

    v_lo = np.float32(LUT_V_LO)
    v_hi = np.float32(LUT_V_HI)
    band_lim = np.float32(band_limit_m)

    cache_idx = -1
    cache_slab: np.ndarray | None = None
    next_idx = -1
    next_slab: np.ndarray | None = None

    def _get_slab(c_idx: int) -> np.ndarray:
        nonlocal cache_idx, cache_slab, next_idx, next_slab
        if c_idx == cache_idx and cache_slab is not None:
            return cache_slab
        if c_idx == next_idx and next_slab is not None:
            return next_slab
        slab = _resample_3d_trilinear(v_deq_coarse_fp32[c_idx], dst_3d)
        if c_idx > cache_idx:
            cache_idx, cache_slab = next_idx, next_slab
            next_idx, next_slab = c_idx, slab
        else:
            cache_idx, cache_slab = c_idx, slab
        return slab

    eps_max = 0.0
    for fr in range(fine_shape[0]):
        c0 = int(i0_r[fr])
        c1 = int(i1_r[fr])
        wr = np.float32(w_r[fr])
        s0 = _get_slab(c0)
        v_hat = s0 if c0 == c1 else (s0 + wr * (_get_slab(c1) - s0))
        vf = np.clip(v_fine_fp32[fr], v_lo, v_hi)
        over = np.where(
            np.abs(vf) <= band_lim,
            np.maximum(np.float32(0.0), v_hat - vf),
            np.float32(0.0),
        )
        eps_max = max(eps_max, float(np.max(over)))

    return float(eps_max + 0.5 * float(LUT_SCALE))


def _write_lut_archive(
    path: Path,
    *,
    magic: bytes,
    dims: tuple[int, int, int, int],
    lo: tuple[float, float, float, float],
    hi: tuple[float, float, float, float],
    vq_list: list[np.ndarray],
    eps_grid: tuple[float, float, float, float],
    v_cap_f: tuple[float, float, float, float],
    flags: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_arr = np.ascontiguousarray(np.stack(vq_list, axis=0), dtype="<i2")
    payload_bytes = payload_arr.tobytes()
    import hashlib

    payload_sha = hashlib.sha256(payload_bytes).digest()
    hdr = VrtaLutHeader(
        magic=magic,
        version=LUT_VERSION,
        num_regimes=LUT_NUM_REGIMES,
        dims=dims,
        lo=lo,
        hi=hi,
        scale=float(LUT_SCALE),
        offset=float(LUT_OFFSET),
        eps_grid=eps_grid,
        payload_sha256=payload_sha,
        v_cap_f=v_cap_f,
        flags=flags,
        horizon_s=float(LUT_HORIZON_S),
        d_ar_max=1.75,
        d_az_max=0.75,
    )
    hdr_bytes = pack_lut_header(hdr)
    assert len(hdr_bytes) == LUT_HEADER_BYTES
    with path.open("wb") as fh:
        fh.write(hdr_bytes)
        fh.write(payload_bytes)
    return {
        "filename": path.name,
        "path": str(path),
        "magic": magic.decode("ascii"),
        "dims": list(dims),
        "size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "payload_sha256": payload_sha.hex(),
        "eps_grid_m": [round(float(e), 6) for e in eps_grid],
        "v_cap_f_mps": [round(float(v), 4) for v in v_cap_f],
    }


def _solve_kinematic_grid_cpu(
    spec: RegimeSpec,
    shape: tuple[int, int, int, int],
    lo: tuple[float, float, float, float],
    hi: tuple[float, float, float, float],
    horizon_s: float = LUT_HORIZON_S,
    beta: float = LUT_BETA_SMAX,
) -> np.ndarray:
    """Fast vectorized conservative kinematic 4D CBVF on arbitrary grid shape for CPU smoke verification."""
    r = np.linspace(lo[0], hi[0], shape[0], dtype=np.float32)[:, None, None, None]
    vr = np.linspace(lo[1], hi[1], shape[1], dtype=np.float32)[None, :, None, None]
    z = np.linspace(lo[2], hi[2], shape[2], dtype=np.float32)[None, None, :, None]
    vz = np.linspace(lo[3], hi[3], shape[3], dtype=np.float32)[None, None, None, :]

    a_r_brake = np.float32(max(0.8, spec.a_xy_hover_max - spec.d_ar))
    a_z_brake = np.float32(max(0.8, spec.a_z_up_max - spec.d_az))

    vr_neg = np.minimum(vr, np.float32(0.0))
    t_r = np.minimum(-vr_neg / a_r_brake, np.float32(horizon_s))
    d_r_stop = -vr_neg * t_r - np.float32(0.5) * a_r_brake * t_r * t_r + np.float32(spec.d_vr) * t_r

    vz_neg = np.minimum(vz, np.float32(0.0))
    t_z = np.minimum(-vz_neg / a_z_brake, np.float32(horizon_s))
    d_z_stop = -vz_neg * t_z - np.float32(0.5) * a_z_brake * t_z * t_z + np.float32(spec.d_vz) * t_z

    sr = (r - d_r_stop) - np.float32(spec.r_safe)
    sz = (z - d_z_stop) - np.float32(spec.z_safe)
    v = conservative_smax_4(sr, sz, beta=beta)
    return np.ascontiguousarray(v, dtype=np.float32)


def regime_spec_to_param_vector(spec: RegimeSpec, *, dtype: Any = np.float32) -> np.ndarray:
    """Pack a RegimeSpec into the 13 dynamics parameters matching VertiRta4DDynamics."""
    a_T = float(spec.a_thrust_max)
    tilt_rad = float(spec.tilt_max_rad)
    tan_th = math.tan(tilt_rad)
    sin_th = math.sin(tilt_rad)
    cos_th = math.cos(tilt_rad)
    g = float(spec.g)
    fz_lo = max(0.5, g + float(spec.a_z_min))
    if fz_lo > a_T:
        fz_lo = a_T
    fz_rim = a_T * cos_th
    if fz_rim >= fz_lo:
        fz_cap = fz_rim
        fr_cap = a_T * sin_th
    else:
        fz_cap = fz_lo
        fr_cap = math.sqrt(max(0.0, a_T * a_T - fz_lo * fz_lo))
    fr_bot = min(fz_lo * tan_th, math.sqrt(max(0.0, a_T * a_T - fz_lo * fz_lo)))
    return np.array(
        [
            a_T,
            tan_th,
            fz_cap,
            fr_cap,
            fz_lo,
            fr_bot,
            float(spec.cd_r),
            float(spec.cd_z),
            float(spec.d_vr),
            float(spec.d_ar),
            float(spec.d_vz),
            float(spec.d_az),
            g,
        ],
        dtype=dtype,
    )


class _ParameterizedVertiRta4DDynamics(_DynamicsBase):
    """JAX-traceable parameterized 4D eVTOL dynamics for genuine jax.vmap over regimes.

    Accepts a 13-element FP32 array/tuple of traced or concrete scalars:
        (a_T, tan_th, fz_cap, fr_cap, fz_lo, fr_bot, cd_r, cd_z, d_vr, d_ar, d_vz, d_az, g)
    with ZERO Python float() casts so jax.vmap fuses multiple regimes into one XLA kernel
    while preserving bit-identical WENO5 + TVD-RK3 physics and DR-2 Errata Locks.
    """

    def __init__(self, params: Any) -> None:
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=None,
            disturbance_space=None,
        )
        self.params = params
        (
            self.a_T,
            self.tan_th,
            self.fz_cap,
            self.fr_cap,
            self.fz_lo,
            self.fr_bot,
            self.cd_r,
            self.cd_z,
            self.d_vr,
            self.d_ar,
            self.d_vz,
            self.d_az,
            self.g,
        ) = (
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            params[5],
            params[6],
            params[7],
            params[8],
            params[9],
            params[10],
            params[11],
            params[12],
        )

    def _support_U(self, pr: Any, pz: Any) -> Any:
        xp = _xp(self.a_T, pr, pz)
        nrm = xp.sqrt(pr * pr + pz * pz + 1e-12)
        fr_arc = self.a_T * pr / nrm
        fz_arc = self.a_T * pz / nrm
        valid_arc = (fz_arc >= self.fz_cap - 1e-6) & (
            xp.abs(fr_arc) <= fz_arc * self.tan_th + 1e-6
        )
        val_arc = xp.where(valid_arc, pr * fr_arc + pz * (fz_arc - self.g), -1e9)

        uz_cap = self.fz_cap - self.g
        uz_lo = self.fz_lo - self.g
        v_top_pos = pr * self.fr_cap + pz * uz_cap
        v_top_neg = -pr * self.fr_cap + pz * uz_cap
        v_bot_pos = pr * self.fr_bot + pz * uz_lo
        v_bot_neg = -pr * self.fr_bot + pz * uz_lo
        v_apex = pz * (self.a_T - self.g)

        return xp.maximum(
            xp.maximum(xp.maximum(val_arc, v_apex), xp.maximum(v_top_pos, v_top_neg)),
            xp.maximum(v_bot_pos, v_bot_neg),
        )

    def hamiltonian(self, state: Any, time: Any, value: Any, grad_value: Any) -> Any:
        xp = _xp(self.a_T, state, grad_value)
        _r, vr, _z, vz = state[0], state[1], state[2], state[3]
        p_r, p_vr, p_z, p_vz = grad_value[0], grad_value[1], grad_value[2], grad_value[3]
        drift = (
            p_r * vr
            + p_z * vz
            - p_vr * self.cd_r * vr * xp.abs(vr)
            - p_vz * self.cd_z * vz * xp.abs(vz)
        )
        u_opt = self._support_U(p_vr, p_vz)
        d_opt = -(
            xp.abs(p_r) * self.d_vr
            + xp.abs(p_vr) * self.d_ar
            + xp.abs(p_z) * self.d_vz
            + xp.abs(p_vz) * self.d_az
        )
        return drift + u_opt + d_opt

    def partial_max_magnitudes(
        self, state: Any, time: Any, value: Any, grad_value_box: Any
    ) -> Any:
        xp = _xp(self.a_T, state)
        _r, vr, _z, vz = state[0], state[1], state[2], state[3]
        a_r_max = xp.maximum(self.fr_cap, self.fr_bot)
        a_z_max = xp.maximum(xp.abs(self.a_T - self.g), xp.abs(self.fz_lo - self.g))
        return xp.stack(
            [
                xp.asarray(xp.abs(vr) + self.d_vr, dtype=xp.float32),
                xp.asarray(a_r_max + self.cd_r * vr * vr + self.d_ar, dtype=xp.float32),
                xp.asarray(xp.abs(vz) + self.d_vz, dtype=xp.float32),
                xp.asarray(a_z_max + self.cd_z * vz * vz + self.d_az, dtype=xp.float32),
            ]
        )

    def __call__(self, state: Any, control: Any, disturbance: Any, time: Any) -> Any:
        xp = _xp(self.a_T, state, control, disturbance)
        _r, vr, _z, vz = state[0], state[1], state[2], state[3]
        u_r, u_z = control[0], control[1]
        d_vr, d_ar, d_vz, d_az = disturbance[0], disturbance[1], disturbance[2], disturbance[3]
        return xp.stack(
            [
                xp.asarray(vr + d_vr, dtype=xp.float32),
                xp.asarray(u_r - self.cd_r * vr * xp.abs(vr) + d_ar, dtype=xp.float32),
                xp.asarray(vz + d_vz, dtype=xp.float32),
                xp.asarray(u_z - self.cd_z * vz * xp.abs(vz) + d_az, dtype=xp.float32),
            ]
        )

    def optimal_control_and_disturbance(self, state: Any, time: Any, grad_value: Any) -> tuple[Any, Any]:
        xp = _xp(self.a_T, state, grad_value)
        p_r, p_vr, p_z, p_vz = grad_value[0], grad_value[1], grad_value[2], grad_value[3]

        nrm = xp.sqrt(p_vr * p_vr + p_vz * p_vz + 1e-12)
        fr_arc = self.a_T * p_vr / nrm
        fz_arc = self.a_T * p_vz / nrm
        valid_arc = (fz_arc >= self.fz_cap - 1e-6) & (xp.abs(fr_arc) <= fz_arc * self.tan_th + 1e-6)
        uz_cap = self.fz_cap - self.g
        uz_lo = self.fz_lo - self.g
        zero = xp.zeros_like(fr_arc)

        cand_ur = xp.stack([fr_arc, zero, self.fr_cap, -self.fr_cap, self.fr_bot, -self.fr_bot]).astype(xp.float32)
        cand_uz = xp.stack([fz_arc - self.g, self.a_T - self.g, uz_cap, uz_cap, uz_lo, uz_lo]).astype(xp.float32)
        cand_val = p_vr * cand_ur + p_vz * cand_uz
        if xp is jnp:
            cand_val = cand_val.at[0].set(xp.where(valid_arc, cand_val[0], -1e9))
        else:
            cand_val = np.where(np.arange(6) == 0, np.where(valid_arc, cand_val[0], -1e9), cand_val)
        best = xp.argmax(cand_val)
        u_opt = xp.stack([cand_ur[best], cand_uz[best]]).astype(xp.float32)
        d_opt = xp.stack(
            [
                -xp.sign(p_r) * self.d_vr,
                -xp.sign(p_vr) * self.d_ar,
                -xp.sign(p_z) * self.d_vz,
                -xp.sign(p_vz) * self.d_az,
            ]
        ).astype(xp.float32)
        return u_opt, d_opt


def verify_parameterized_dynamics_equivalence(spec: RegimeSpec) -> dict[str, Any]:
    """Compare _ParameterizedVertiRta4DDynamics vs VertiRta4DDynamics on diverse states & gradients."""
    from scripts.colab.run_l4_jax_cbvf import _ConcreteVertiRta4DDynamics

    ref_dyn = _ConcreteVertiRta4DDynamics.from_regime_spec(spec)
    params_f64 = regime_spec_to_param_vector(spec, dtype=np.float64)
    params_f32 = regime_spec_to_param_vector(spec, dtype=np.float32)
    param_dyn_f64 = _ParameterizedVertiRta4DDynamics(params_f64)
    param_dyn_f32 = _ParameterizedVertiRta4DDynamics(params_f32)

    test_states_f64 = np.array(
        [
            [5.0, -22.0, 4.0, -8.0],
            [12.0, -14.5, 10.0, -3.5],
            [25.0, -6.0, 18.0, 0.0],
            [40.0, 0.0, 2.5, 4.5],
            [55.0, 8.5, 30.0, 7.5],
            [80.0, -35.0, 12.0, -10.5],
            [105.0, -48.0, 45.0, -12.0],
            [2.0, 15.0, -2.0, 6.0],
        ],
        dtype=np.float64,
    )
    test_grads_f64 = np.array(
        [
            [0.6, 0.2, 0.3, 0.7],    # thrust-sphere arc (positive p_vz, moderate |p_vr|)
            [-0.4, -0.25, 0.5, 0.75],
            [0.8, 0.0, -0.2, 0.9],   # apex (p_vr = 0, p_vz > 0)
            [0.3, 0.9, 0.1, 0.2],    # upper rim +fr_cap
            [-0.3, -0.9, -0.2, 0.15],  # upper rim -fr_cap
            [0.5, 0.7, -0.4, -0.6],  # lower floor +fr_bot
            [-0.5, -0.7, 0.4, -0.8],  # lower floor -fr_bot
            [0.0, 0.0, 0.0, 0.0],    # zero gradient edge case
        ],
        dtype=np.float64,
    )
    test_states_f32 = test_states_f64.astype(np.float32)
    test_grads_f32 = test_grads_f64.astype(np.float32)

    ham_max_err = 0.0
    pmm_max_err = 0.0
    fp32_max_err = 0.0
    n_pairs = 0
    for st64, st32 in zip(test_states_f64, test_states_f32):
        pmm_ref = np.asarray(ref_dyn.partial_max_magnitudes(st64, 0.0, 0.0, None), dtype=np.float64)
        pmm_par = np.asarray(param_dyn_f64.partial_max_magnitudes(st64, 0.0, 0.0, None), dtype=np.float64)
        pmm_par32 = np.asarray(param_dyn_f32.partial_max_magnitudes(st32, 0.0, 0.0, None), dtype=np.float64)
        pmm_max_err = max(pmm_max_err, float(np.max(np.abs(pmm_par - pmm_ref))))
        fp32_max_err = max(fp32_max_err, float(np.max(np.abs(pmm_par32 - pmm_ref))))
        for gr64, gr32 in zip(test_grads_f64, test_grads_f32):
            h_ref = float(ref_dyn.hamiltonian(st64, 0.0, 0.0, gr64))
            h_par = float(param_dyn_f64.hamiltonian(st64, 0.0, 0.0, gr64))
            h_par32 = float(param_dyn_f32.hamiltonian(st32, 0.0, 0.0, gr32))
            ham_max_err = max(ham_max_err, abs(h_par - h_ref))
            fp32_max_err = max(fp32_max_err, abs(h_par32 - h_ref))
            n_pairs += 1

    jax_vmap_traced = False
    if HAS_JAX and jax is not None and jnp is not None:
        param_batch = jnp.stack([jnp.asarray(params_f32), jnp.asarray(params_f32)], axis=0)
        st_j = jnp.asarray(test_states_f32[0])
        gr_j = jnp.asarray(test_grads_f32[0])

        def _eval_one(p_vec: Any) -> tuple[Any, Any]:
            d = _ParameterizedVertiRta4DDynamics(p_vec)
            return d.hamiltonian(st_j, 0.0, 0.0, gr_j), d.partial_max_magnitudes(st_j, 0.0, 0.0, None)

        h_vm, pmm_vm = jax.vmap(_eval_one)(param_batch)
        h_ref_j = float(ref_dyn.hamiltonian(st_j, 0.0, 0.0, gr_j))
        pmm_ref_j = np.asarray(ref_dyn.partial_max_magnitudes(st_j, 0.0, 0.0, None), dtype=np.float64)
        ham_max_err = max(ham_max_err, float(np.max(np.abs(np.asarray(h_vm, dtype=np.float64) - h_ref_j))))
        pmm_max_err = max(pmm_max_err, float(np.max(np.abs(np.asarray(pmm_vm, dtype=np.float64) - pmm_ref_j))))
        jax_vmap_traced = True

    max_err = max(ham_max_err, pmm_max_err)
    return {
        "regime_name": spec.name,
        "n_test_pairs": n_pairs,
        "hamiltonian_max_abs_err": float(ham_max_err),
        "partial_max_mag_max_abs_err": float(pmm_max_err),
        "max_abs_err": float(max_err),
        "fp32_param_max_abs_err": float(fp32_max_err),
        "jax_vmap_traced": jax_vmap_traced,
        "passed": bool(max_err < 1e-6 and fp32_max_err < 5e-6),
    }


def _euler_step_traceable(
    solver_settings: Any,
    dynamics: Any,
    grid: Any,
    time_val: Any,
    values: Any,
    time_step: Any = None,
    max_time_step: Any = None,
) -> tuple[Any, Any]:
    """Traceable hj_reachability v0.7.0 Lax-Friedrichs Euler step without static_argnames='dynamics'."""
    time_direction = jnp.sign(max_time_step) if time_step is None else jnp.sign(time_step)
    signed_hamiltonian = lambda *args, **kwargs: time_direction * dynamics.hamiltonian(*args, **kwargs)
    left_grad_values, right_grad_values = grid.upwind_grad_values(solver_settings.upwind_scheme, values)
    dissipation_coefficients = solver_settings.artificial_dissipation_scheme(
        dynamics.partial_max_magnitudes,
        grid.states,
        time_val,
        values,
        left_grad_values,
        right_grad_values,
    )
    dvalues_dt = -solver_settings.hamiltonian_postprocessor(
        time_direction
        * hj.utils.multivmap(
            lambda state, value, left_grad_value, right_grad_value, diss_coeff: (
                hj.time_integration.lax_friedrichs_numerical_hamiltonian(
                    signed_hamiltonian,
                    state,
                    time_val,
                    value,
                    left_grad_value,
                    right_grad_value,
                    diss_coeff,
                )
            ),
            np.arange(grid.ndim),
        )(grid.states, values, left_grad_values, right_grad_values, dissipation_coefficients)
    )
    if time_step is None:
        time_step_bound = 1.0 / jnp.max(jnp.sum(dissipation_coefficients / jnp.array(grid.spacings), -1))
        time_step = time_direction * jnp.minimum(
            solver_settings.CFL_number * time_step_bound,
            jnp.abs(max_time_step),
        )
    return time_val + time_step, values + time_step * dvalues_dt


def _tvd_rk3_step_traceable(
    solver_settings: Any,
    dynamics: Any,
    grid: Any,
    time_val: Any,
    values: Any,
    target_time: Any,
) -> tuple[Any, Any]:
    """Traceable TVD-RK3 step matching hj_reachability.time_integration bit-for-bit."""
    time_1, values_1 = _euler_step_traceable(
        solver_settings, dynamics, grid, time_val, values, max_time_step=target_time - time_val
    )
    time_step = time_1 - time_val
    _, values_2 = _euler_step_traceable(solver_settings, dynamics, grid, time_1, values_1, time_step)
    time_0_5, values_0_5 = time_val + time_step / 2.0, (3.0 / 4.0) * values + (1.0 / 4.0) * values_2
    _, values_1_5 = _euler_step_traceable(solver_settings, dynamics, grid, time_0_5, values_0_5, time_step)
    return time_1, solver_settings.value_postprocessor(time_1, (1.0 / 3.0) * values + (2.0 / 3.0) * values_1_5)


def _solve_regimes_jax_vmap(
    regimes: tuple[RegimeSpec, ...],
    *,
    fine_shape: tuple[int, int, int, int],
    lo: tuple[float, float, float, float],
    hi: tuple[float, float, float, float],
    vmap_batch_size: int = 2,
    horizon_s: float = LUT_HORIZON_S,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray], dict[str, Any]]:
    """Live G4 JAX WENO5+TVD-RK3 solver with genuine jax.vmap(2) + OOM/tracer-safe fallback."""
    if not (HAS_JAX and HAS_HJ_REACHABILITY and jax is not None and jnp is not None and hj is not None):
        raise RuntimeError("JAX CUDA + StanfordASL/hj_reachability v0.7.0 required for live G4 PDE solve.")

    from scripts.colab.run_l4_jax_cbvf import _ConcreteVertiRta4DDynamics

    grid_fine = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=np.array(lo, dtype=np.float32), hi=np.array(hi, dtype=np.float32)),
        shape=fine_shape,
    )
    settings = hj.SolverSettings.with_accuracy(
        "very_high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )

    t0_jax = jnp.asarray(0.0, dtype=jnp.float32)
    t_target_jax = jnp.asarray(-float(horizon_s), dtype=jnp.float32)

    def _solve_one_regime(params: Any, l_init: Any) -> Any:
        dyn = _ParameterizedVertiRta4DDynamics(params)

        def _sub_step(time_values: tuple[Any, Any]) -> tuple[Any, Any]:
            return _tvd_rk3_step_traceable(settings, dyn, grid_fine, time_values[0], time_values[1], t_target_jax)

        return jax.lax.while_loop(
            lambda time_values: jnp.abs(t_target_jax - time_values[0]) > 0,
            _sub_step,
            (t0_jax, l_init),
        )[1]

    solve_batch_vmap = jax.jit(jax.vmap(_solve_one_regime, in_axes=(0, 0)))

    vq_coarse_list: list[np.ndarray] = []
    vq_hires_list: list[np.ndarray] = []
    eps_list: list[float] = []
    per_regime_stats: list[dict[str, Any]] = []
    peak_vram_bytes = 0
    vmap_chunks_succeeded = 0
    vmap_fallback_reasons: list[str] = []

    def _record_peak_vram() -> None:
        nonlocal peak_vram_bytes
        try:
            mem_stats = jax.local_devices()[0].memory_stats() or {}
            peak_vram_bytes = max(peak_vram_bytes, int(mem_stats.get("peak_bytes_in_use", 0)))
        except Exception:
            pass

    def _postprocess_regime(spec: RegimeSpec, v_fine_np: np.ndarray, solve_s: float, via_vmap: bool) -> None:
        v_coarse_fp32 = project_4d_quadrilinear(v_fine_np, LUT_EXPORT_SHAPE)
        vq_coarse = _quantize_int16(v_coarse_fp32)
        v_deq = vq_coarse.astype(np.float32) * LUT_SCALE + LUT_OFFSET
        eps_f = audit_exact_hires_eps_grid_chunked(v_fine_np, v_deq)
        vq_hires = _quantize_int16(v_fine_np)

        vq_coarse_list.append(vq_coarse)
        vq_hires_list.append(vq_hires)
        eps_list.append(eps_f)
        per_regime_stats.append(
            {
                "regime_index": spec.index,
                "regime_name": spec.name,
                "tw_ratio": round(spec.tw_ratio, 4),
                "tilt_max_deg": spec.tilt_max_deg,
                "solve_wall_s": round(solve_s, 3),
                "eps_grid_m": round(eps_f, 6),
                "solved_via_jax_vmap": via_vmap,
            }
        )

    def _solve_chunk_sequential(chunk_specs: tuple[RegimeSpec, ...]) -> None:
        for spec in chunk_specs:
            t_r0 = time.perf_counter()
            sr = grid_fine.states[..., 0] - np.float32(spec.r_safe)
            sz = grid_fine.states[..., 2] - np.float32(spec.z_safe)
            l_init = conservative_smax_4(sr, sz, beta=LUT_BETA_SMAX)
            del sr, sz
            dyn = _ConcreteVertiRta4DDynamics.from_regime_spec(spec)
            v_fine_jax = hj.step(settings, dyn, grid_fine, 0.0, l_init, -float(horizon_s), progress_bar=False)
            v_fine_jax.block_until_ready()
            solve_s = time.perf_counter() - t_r0
            _record_peak_vram()
            v_fine_np = np.asarray(v_fine_jax, dtype=np.float32)
            del v_fine_jax, l_init
            gc.collect()
            _postprocess_regime(spec, v_fine_np, solve_s, via_vmap=False)
            del v_fine_np
            gc.collect()

    step_bs = max(1, int(vmap_batch_size))
    for start in range(0, len(regimes), step_bs):
        chunk = regimes[start : start + step_bs]
        if len(chunk) >= 2 and step_bs >= 2:
            v_batch_jax = l_init_batch = l_init_list = param_batch = sr = sz = None
            need_seq_fallback = False
            try:
                t_c0 = time.perf_counter()
                param_batch = jnp.asarray(
                    np.stack([regime_spec_to_param_vector(spec) for spec in chunk], axis=0),
                    dtype=jnp.float32,
                )
                l_init_list = []
                for spec in chunk:
                    sr = grid_fine.states[..., 0] - np.float32(spec.r_safe)
                    sz = grid_fine.states[..., 2] - np.float32(spec.z_safe)
                    l_init_list.append(conservative_smax_4(sr, sz, beta=LUT_BETA_SMAX))
                    del sr, sz
                    sr = sz = None
                l_init_batch = jnp.stack(l_init_list, axis=0)
                del l_init_list
                l_init_list = None

                v_batch_jax = solve_batch_vmap(param_batch, l_init_batch)
                v_batch_jax.block_until_ready()
                chunk_solve_s = time.perf_counter() - t_c0
                _record_peak_vram()

                v_batch_np = np.asarray(v_batch_jax, dtype=np.float32)
                del v_batch_jax, l_init_batch, param_batch
                v_batch_jax = l_init_batch = param_batch = None
                gc.collect()

                per_regime_amortized_s = chunk_solve_s / float(len(chunk))
                for bi, spec in enumerate(chunk):
                    _postprocess_regime(spec, v_batch_np[bi], per_regime_amortized_s, via_vmap=True)
                del v_batch_np
                gc.collect()
                vmap_chunks_succeeded += 1
            except Exception as exc:
                vmap_fallback_reasons.append(f"{type(exc).__name__}: {exc}")
                if exc.__traceback__ is not None:
                    traceback.clear_frames(exc.__traceback__)
                del v_batch_jax, l_init_batch, l_init_list, param_batch, sr, sz, exc
                need_seq_fallback = True
            if need_seq_fallback:
                gc.collect()
                _solve_chunk_sequential(chunk)
        else:
            _solve_chunk_sequential(chunk)

    if vmap_fallback_reasons:
        exec_mode = "sequential_per_regime_oom_safe_fallback"
    elif vmap_chunks_succeeded > 0:
        exec_mode = "true_jax_vmap_batch2" if int(vmap_batch_size) == 2 else f"true_jax_vmap_batch{int(vmap_batch_size)}"
    else:
        exec_mode = "sequential_per_regime_oom_safe"

    return (
        vq_coarse_list,
        eps_list,
        vq_hires_list,
        {
            "execution_mode": exec_mode,
            "vmap_batch_size": int(vmap_batch_size),
            "vmap_chunks_succeeded": vmap_chunks_succeeded,
            "vmap_fallback_reason": "; ".join(vmap_fallback_reasons) if vmap_fallback_reasons else None,
            "jax_peak_vram_mib": round(peak_vram_bytes / (1024.0 * 1024.0), 2),
            "regimes": per_regime_stats,
        },
    )


def _enforce_conservative_hierarchy(vq_list: list[np.ndarray]) -> None:
    """Enforce V_0 >= V_1 >= V_3 >= V_2 conservatively via pointwise lower envelope."""
    vq_list[1] = np.minimum(vq_list[1], vq_list[0])
    vq_list[3] = np.minimum(vq_list[3], vq_list[1])
    vq_list[2] = np.minimum(vq_list[2], vq_list[3])


def _verify_x8_soundness_concrete(x8_v1_path: Path, cel2d_path: Path) -> dict[str, Any]:
    """Run verify_all_soundness with concrete VertiRta4DDynamics bound (hj_reachability v0.7.0 ABC safe)."""
    from scripts.colab.run_l4_jax_cbvf import _ConcreteVertiRta4DDynamics

    prev_cls = _vls_mod.VertiRta4DDynamics
    _vls_mod.VertiRta4DDynamics = _ConcreteVertiRta4DDynamics  # type: ignore[assignment]
    try:
        return verify_all_soundness(x8_v1_path, cel2d_path)
    finally:
        _vls_mod.VertiRta4DDynamics = prev_cls


def run_job(args: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_everything(args.seed)

    xla_frac = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.vram_fraction)))
    if xla_frac > 0.58 + 1e-6:
        raise ValueError(f"XLA_PYTHON_CLIENT_MEM_FRACTION={xla_frac} exceeds Wave-2 0.58 ceiling!")

    vmap_bs = int(args.vmap_batch_size)
    if vmap_bs > 2 and not args.allow_vmap4:
        raise ValueError("vmap_batch_size > 2 requires explicit --allow-vmap4 flag (safe default is 2).")

    license_audit = validate_job_license_manifest(
        "job_jax_4d_hires_and_lc",
        train_sources=[{"name": "VertiRTA_Analytical_6DOF_Physics", "repo": "internal/vertirta", "license": "Apache-2.0"}],
        eval_only_sources=[],
        models=[
            {"name": "JAX_HJ_Reachability_v0.7.0", "repo": "StanfordASL/hj_reachability", "license": "MIT"},
        ],
    )

    x8_regimes = build_regimes_from_config()
    lc_regimes = build_lift_cruise_regimes()
    x8_vcaps = compute_all_regime_v_caps(x8_regimes)
    lc_vcaps = (22.0, 16.0, 28.0, 38.0)

    if args.dry_run:
        payload = {
            "schema_version": "vertirta.g4_jax_4d_hires_and_lc_manifest/v4.0.0",
            "mode": "DRY_RUN",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "xla_mem_fraction": xla_frac,
            "vmap_batch_size": vmap_bs,
            "hires_grid_shape": list(HIRES_SHAPE),
            "hires_nodes_per_regime": HIRES_NODES_PER_REGIME,
            "export_grid_shape": list(LUT_EXPORT_SHAPE),
            "license_audit": license_audit,
            "lift_cruise_constants": {
                "mass_kg": LC_MASS_KG,
                "wing_area_m2": LC_WING_AREA_M2,
                "cd_r_streamlined": round(LC_CD_R, 7),
                "cd_z_floor": LC_CD_Z,
                "v_stall_1g_mps": LC_STALL_1G_MPS,
                "v_cruise_mps": LC_CRUISE_MPS,
                "v_ne_mps": LC_VNE_MPS,
                "hover_oei_tw_eff": lc_regimes[1].tw_ratio,
            },
            "execution_mode": "true_jax_vmap_batch2" if vmap_bs == 2 else f"true_jax_vmap_batch{vmap_bs}",
            "wall_clock_s": round(time.perf_counter() - t0, 4),
        }
        write_canonical_json(args.out_json, payload)
        return payload

    blob_dir = Path(args.blob_dir)

    if args.smoke_cpu:
        # 1) Verify exact half-integer 2.5:1 projection soundness on scaled nested grids (21x15 -> 9x7)
        smoke_fine = (21, 15, 21, 15)
        smoke_coarse = (9, 7, 9, 7)
        v_fine_probe = _solve_kinematic_grid_cpu(x8_regimes[0], smoke_fine, LUT_LO, LUT_HI)
        v_coarse_probe = project_4d_quadrilinear(v_fine_probe, smoke_coarse)
        vq_probe = _quantize_int16(v_coarse_probe)
        v_deq_probe = vq_probe.astype(np.float32) * LUT_SCALE + LUT_OFFSET
        eps_probe = audit_exact_hires_eps_grid_chunked(v_fine_probe, v_deq_probe)

        # Verify soundness inequality V_hat(x) - eps_probe <= V_fine(x) on |V_fine| <= 10 m
        v_hat_full = project_4d_quadrilinear(v_deq_probe, smoke_fine)
        v_clip = np.clip(v_fine_probe, np.float32(LUT_V_LO), np.float32(LUT_V_HI))
        band = np.abs(v_clip) <= np.float32(10.0)
        max_violation = float(np.max((v_hat_full - np.float32(eps_probe) - v_clip)[band]))
        assert max_violation <= 1e-4, f"2.5:1 soundness violated: max_violation={max_violation}"

        # 2) Export exact 11,313,864 B VRTALUT1 for BOTH X8-240 and Lift+Cruise 8+1 + VRTALUT2 probe
        x8_vq: list[np.ndarray] = []
        x8_eps: list[float] = []
        for spec in x8_regimes:
            vq, eps_f = _solve_regime_quick_cpu(spec, horizon_s=LUT_HORIZON_S, beta=LUT_BETA_SMAX)
            x8_vq.append(vq)
            x8_eps.append(eps_f)
        _enforce_conservative_hierarchy(x8_vq)

        lc_vq: list[np.ndarray] = []
        lc_eps: list[float] = []
        for spec in lc_regimes:
            v_lc = _solve_kinematic_grid_cpu(spec, LUT_EXPORT_SHAPE, LC_LUT_LO, LC_LUT_HI)
            vq_lc = _quantize_int16(v_lc)
            lc_vq.append(vq_lc)
            lc_eps.append(float(0.5 * float(LUT_SCALE) + 0.15))

        lc_vq[1] = np.minimum(lc_vq[1], lc_vq[0])

        flags = FLAG_QUICK_CPU_FIXTURE | FLAG_HAS_V_CAP | FLAG_CONSERVATIVE_SMAX
        x8_v1_path = blob_dir / "cbvf_4d_x8_240_v4_from_101x71.vrta_lut"
        x8_v1_meta = _write_lut_archive(
            x8_v1_path,
            magic=LUT_MAGIC,
            dims=LUT_EXPORT_SHAPE,
            lo=LUT_LO,
            hi=LUT_HI,
            vq_list=x8_vq,
            eps_grid=tuple(x8_eps),  # type: ignore[arg-type]
            v_cap_f=x8_vcaps,
            flags=flags,
        )
        assert x8_v1_meta["size_bytes"] == LUT_TOTAL_BYTES

        x8_v2_path = blob_dir / "cbvf_4d_x8_240_hires_101x71.vrta_lut_hires"
        x8_v2_meta = _write_lut_archive(
            x8_v2_path,
            magic=LUT_HIRES_MAGIC,
            dims=smoke_fine,
            lo=LUT_LO,
            hi=LUT_HI,
            vq_list=[_quantize_int16(v_fine_probe)] * 4,
            eps_grid=(eps_probe,) * 4,
            v_cap_f=x8_vcaps,
            flags=flags,
        )

        lc_v1_path = blob_dir / "cbvf_4d_lc_8p1_41x29.vrta_lut"
        lc_v1_meta = _write_lut_archive(
            lc_v1_path,
            magic=LUT_MAGIC,
            dims=LUT_EXPORT_SHAPE,
            lo=LC_LUT_LO,
            hi=LC_LUT_HI,
            vq_list=lc_vq,
            eps_grid=tuple(lc_eps),  # type: ignore[arg-type]
            v_cap_f=lc_vcaps,
            flags=flags,
        )
        assert lc_v1_meta["size_bytes"] == LUT_TOTAL_BYTES

        cel2d_path = blob_dir / "cel_2d_x8_240.vrta_cel2d"
        solve_and_export_cel_2d(cel2d_path)
        assert cel2d_path.stat().st_size == CEL2D_TOTAL_BYTES

        soundness = _verify_x8_soundness_concrete(x8_v1_path, cel2d_path)
        lc_soundness = verify_lift_cruise_lut_soundness(lc_v1_path)
        x8_ok = bool(soundness.get("passed", False))
        lc_ok = bool(lc_soundness.get("passed", False))
        payload = {
            "schema_version": "vertirta.g4_jax_4d_hires_and_lc_manifest/v4.0.0",
            "mode": "SMOKE_CPU_ONLY",
            "execution_mode": "sequential_per_regime_oom_safe",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "xla_mem_fraction": xla_frac,
            "vmap_batch_size": vmap_bs,
            "hires_grid_shape": list(HIRES_SHAPE),
            "lift_cruise_cd_r": round(LC_CD_R, 7),
            "smoke_ratio_2p5_audit": {
                "fine_shape": list(smoke_fine),
                "coarse_shape": list(smoke_coarse),
                "eps_grid_m": round(eps_probe, 6),
                "max_soundness_violation_m": round(max_violation, 8),
                "passed": bool(max_violation <= 1e-4),
            },
            "license_audit": license_audit,
            "artifacts": {
                "x8_240_vrtalut1": x8_v1_meta,
                "x8_240_vrtalut2_hires": x8_v2_meta,
                "x8_240_cel2d": {
                    "filename": cel2d_path.name,
                    "size_bytes": cel2d_path.stat().st_size,
                    "file_sha256": sha256_file(cel2d_path),
                },
                "lift_cruise_8p1_vrtalut1": lc_v1_meta,
            },
            "x8_soundness_passed": x8_ok,
            "lc_soundness": lc_soundness,
            "lc_soundness_passed": lc_ok,
            "soundness_passed": bool(x8_ok and lc_ok),
            "wall_clock_s": round(time.perf_counter() - t0, 3),
        }
        write_canonical_json(args.out_json, payload)
        return payload

    # Live G4 JAX CUDA solve (101x71x101x71 genuine jax.vmap(2) + OOM-safe fallback for X8-240 & LC 8+1)
    x8_vq, x8_eps, x8_vq_hires, x8_stats = _solve_regimes_jax_vmap(
        x8_regimes, fine_shape=HIRES_SHAPE, lo=LUT_LO, hi=LUT_HI, vmap_batch_size=vmap_bs
    )
    _enforce_conservative_hierarchy(x8_vq)
    lc_vq, lc_eps, lc_vq_hires, lc_stats = _solve_regimes_jax_vmap(
        lc_regimes, fine_shape=HIRES_SHAPE, lo=LC_LUT_LO, hi=LC_LUT_HI, vmap_batch_size=vmap_bs
    )
    lc_vq[1] = np.minimum(lc_vq[1], lc_vq[0])

    if x8_stats["execution_mode"] == lc_stats["execution_mode"]:
        live_exec_mode = x8_stats["execution_mode"]
    elif "fallback" in str(x8_stats["execution_mode"]) or "fallback" in str(lc_stats["execution_mode"]):
        live_exec_mode = "sequential_per_regime_oom_safe_fallback"
    else:
        live_exec_mode = str(x8_stats["execution_mode"])

    flags_live = FLAG_HAS_V_CAP | FLAG_CONSERVATIVE_SMAX
    x8_v1_path = blob_dir / "cbvf_4d_x8_240_v4_from_101x71.vrta_lut"
    x8_v1_meta = _write_lut_archive(
        x8_v1_path,
        magic=LUT_MAGIC,
        dims=LUT_EXPORT_SHAPE,
        lo=LUT_LO,
        hi=LUT_HI,
        vq_list=x8_vq,
        eps_grid=tuple(x8_eps),  # type: ignore[arg-type]
        v_cap_f=x8_vcaps,
        flags=flags_live,
    )
    x8_v2_meta = _write_lut_archive(
        blob_dir / "cbvf_4d_x8_240_hires_101x71.vrta_lut_hires",
        magic=LUT_HIRES_MAGIC,
        dims=HIRES_SHAPE,
        lo=LUT_LO,
        hi=LUT_HI,
        vq_list=x8_vq_hires,
        eps_grid=tuple(x8_eps),  # type: ignore[arg-type]
        v_cap_f=x8_vcaps,
        flags=flags_live,
    )
    lc_v1_path = blob_dir / "cbvf_4d_lc_8p1_41x29.vrta_lut"
    lc_v1_meta = _write_lut_archive(
        lc_v1_path,
        magic=LUT_MAGIC,
        dims=LUT_EXPORT_SHAPE,
        lo=LC_LUT_LO,
        hi=LC_LUT_HI,
        vq_list=lc_vq,
        eps_grid=tuple(lc_eps),  # type: ignore[arg-type]
        v_cap_f=lc_vcaps,
        flags=flags_live,
    )
    lc_v2_meta = _write_lut_archive(
        blob_dir / "cbvf_4d_lc_8p1_hires_101x71.vrta_lut_hires",
        magic=LUT_HIRES_MAGIC,
        dims=HIRES_SHAPE,
        lo=LC_LUT_LO,
        hi=LC_LUT_HI,
        vq_list=lc_vq_hires,
        eps_grid=tuple(lc_eps),  # type: ignore[arg-type]
        v_cap_f=lc_vcaps,
        flags=flags_live,
    )

    cel2d_path = blob_dir / "cel_2d_x8_240.vrta_cel2d"
    solve_and_export_cel_2d(cel2d_path)
    soundness = _verify_x8_soundness_concrete(x8_v1_path, cel2d_path)
    lc_soundness = verify_lift_cruise_lut_soundness(lc_v1_path)
    x8_ok = bool(soundness.get("passed", False))
    lc_ok = bool(lc_soundness.get("passed", False))
    payload = {
        "schema_version": "vertirta.g4_jax_4d_hires_and_lc_manifest/v4.0.0",
        "mode": "LIVE_G4_JAX_CUDA_FP32",
        "execution_mode": live_exec_mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "xla_mem_fraction": xla_frac,
        "vmap_batch_size": vmap_bs,
        "hires_grid_shape": list(HIRES_SHAPE),
        "hires_nodes_per_regime": HIRES_NODES_PER_REGIME,
        "export_grid_shape": list(LUT_EXPORT_SHAPE),
        "lift_cruise_cd_r": round(LC_CD_R, 7),
        "hardware": query_nvidia_smi_snapshot(),
        "license_audit": license_audit,
        "x8_240_solve": x8_stats,
        "lift_cruise_8p1_solve": lc_stats,
        "artifacts": {
            "x8_240_vrtalut1": x8_v1_meta,
            "x8_240_vrtalut2_hires": x8_v2_meta,
            "x8_240_cel2d": {
                "filename": cel2d_path.name,
                "size_bytes": cel2d_path.stat().st_size,
                "file_sha256": sha256_file(cel2d_path),
            },
            "lift_cruise_8p1_vrtalut1": lc_v1_meta,
            "lift_cruise_8p1_vrtalut2_hires": lc_v2_meta,
        },
        "x8_soundness_passed": x8_ok,
        "lc_soundness": lc_soundness,
        "lc_soundness_passed": lc_ok,
        "soundness_passed": bool(x8_ok and lc_ok),
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    write_canonical_json(args.out_json, payload)
    return payload


def main() -> int:
    parser = build_standard_argparser(
        "job_jax_4d_hires_and_lc",
        default_vram_fraction=WAVE_BUDGETS["wave2"]["jax_4d_hires_lc"],
        description="Wave-2 G4 Job: 101x71x101x71 X8-240 & Lift+Cruise 8+1 4D HJI-CBVF PDE.",
    )
    parser.add_argument(
        "--vmap-batch-size",
        type=int,
        default=2,
        help="Safe OOM-proof default is 2 regimes concurrently (2x2 inside 0.58 XLA cap).",
    )
    parser.add_argument(
        "--allow-vmap4",
        action="store_true",
        help="Explicitly opt into vmap_batch_size=4 with remat/memory check.",
    )
    args = parser.parse_args()
    run_job(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
