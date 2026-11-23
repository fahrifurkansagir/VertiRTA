#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline SVD POD-RBF compressor emitting pure numeric `VRTAWND1` Morton-brick LUT binaries.

Binary ABI contract (byte-exact with `core/wind_acoustics/include/vertirta/wind_acoustics/vrta_wind_format.hpp`):
  - Header: exactly `12,288 B` (`VRTAWND1` magic, version=1, R=8 modes, 48 multiquadric RBF centers,
    datum + max_lon_deg < 29.1895, quantization scales, Tier-0/Tier-1 GridDescs, RBF weights,
    and FIPS 180-4 SHA-256 over the payload bytes).
  - Payload: contiguous array of `64 B` cache-aligned `VrtaWndNode64` (`32 x int16_t`) stored in
    `4x4x4` Morton-swizzled bricks (`tier0` followed by `tier1`):
      * ch[0..2]  : mean velocity u0 (E, N, U) [m/s]
      * ch[3..26] : 8 spatial POD modes x 3 velocity components (mode m at 3 + 3*m + c)
      * ch[27..29]: anisotropic turbulence RMS sigma_u, sigma_v, sigma_w [m/s]
      * ch[30]    : calibrated LODO residual eps_rom_p95 [m/s]
      * ch[31]    : integral turbulence length scale L_turb [m]
  - Full-scale Dragos-Maltepe-Kartal size:
      Tier-0 (96 x 48 x 28 = 129,024) + Tier-1 (64 x 64 x 24 = 98,304) = 227,328 nodes
      -> 12,288 + 227,328 * 64 = 14,561,280 B (13.887 MiB < 16 MiB).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VRTAWND1_MAGIC = b"VRTAWND1"
VRTAWND1_VERSION = 1
VRTAWND1_HEADER_BYTES = 12288
VRTAWND1_NODE_BYTES = 64
VRTAWND1_NUM_MODES = 8
VRTAWND1_NUM_CENTERS = 48

FULL_SCALE_TIER0_DIMS = (96, 48, 28)
FULL_SCALE_TIER1_DIMS = (64, 64, 24)
FULL_SCALE_TOTAL_NODES = (
    FULL_SCALE_TIER0_DIMS[0] * FULL_SCALE_TIER0_DIMS[1] * FULL_SCALE_TIER0_DIMS[2]
    + FULL_SCALE_TIER1_DIMS[0] * FULL_SCALE_TIER1_DIMS[1] * FULL_SCALE_TIER1_DIMS[2]
)
FULL_SCALE_SIZE_BYTES = VRTAWND1_HEADER_BYTES + FULL_SCALE_TOTAL_NODES * VRTAWND1_NODE_BYTES

MAX_ALLOWED_LON_DEG = 29.1895


@dataclass(frozen=True)
class GridSpec:
    """Regular ENU grid specification with dimensions divisible by 4 for 4x4x4 Morton bricks."""

    nx: int
    ny: int
    nz: int
    origin_enu: tuple[float, float, float]
    spacing_m: tuple[float, float, float]

    def validate(self) -> None:
        if self.nx <= 0 or self.ny <= 0 or self.nz <= 0:
            raise ValueError(f"Grid dimensions must be positive: {(self.nx, self.ny, self.nz)}")
        if (self.nx % 4) != 0 or (self.ny % 4) != 0 or (self.nz % 4) != 0:
            raise ValueError(
                f"Grid dimensions must be multiples of 4 for 4x4x4 Morton bricks: "
                f"{(self.nx, self.ny, self.nz)}"
            )

    @property
    def node_count(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def brick_count(self) -> int:
        return (self.nx // 4) * (self.ny // 4) * (self.nz // 4)


def morton_encode_4x4x4(lx: int, ly: int, lz: int) -> int:
    """Interleave 2-bit local coordinates (lx, ly, lz in [0..3]) into Morton index [0..63]."""
    x0 = lx & 1
    x1 = (lx >> 1) & 1
    y0 = ly & 1
    y1 = (ly >> 1) & 1
    z0 = lz & 1
    z1 = (lz >> 1) & 1
    return x0 | (y0 << 1) | (z0 << 2) | (x1 << 3) | (y1 << 4) | (z1 << 5)


def morton_swizzle_index(ix: int, iy: int, iz: int, nx: int, ny: int) -> int:
    """Compute flat node index inside a 4x4x4 Morton-swizzled brick grid."""
    bx, lx = divmod(ix, 4)
    by, ly = divmod(iy, 4)
    bz, lz = divmod(iz, 4)
    nbx = nx // 4
    nby = ny // 4
    brick_idx = (bz * nby + by) * nbx + bx
    return brick_idx * 64 + morton_encode_4x4x4(lx, ly, lz)


def _fit_multiquadric_rbf(
    inflow_ux: np.ndarray,
    inflow_uy: np.ndarray,
    modal_coeffs: np.ndarray,
    shape_c: float = 4.0,
    ridge_lambda: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit regularized multiquadric RBF + affine polynomial in continuous (ux, uy) wind space.

    Args:
        inflow_ux: (M,) u_ref * cos(dir_rad)
        inflow_uy: (M,) u_ref * sin(dir_rad)
        modal_coeffs: (M, R) snapshot modal amplitudes

    Returns:
        (rbf_weights (M, R), poly_weights (R, 3)) where poly_weights multiplies [1, ux, uy].
    """
    m_pts = int(inflow_ux.shape[0])
    dx = inflow_ux[:, None] - inflow_ux[None, :]
    dy = inflow_uy[:, None] - inflow_uy[None, :]
    phi = np.sqrt(dx * dx + dy * dy + shape_c * shape_c)
    p_mat = np.column_stack([np.ones(m_pts, dtype=np.float64), inflow_ux, inflow_uy])

    # Saddle-point regularized RBF system: [[Phi + lam*I, P], [P^T, 0]] [w; p] = [a; 0]
    a_sys = np.zeros((m_pts + 3, m_pts + 3), dtype=np.float64)
    a_sys[:m_pts, :m_pts] = phi + ridge_lambda * np.eye(m_pts, dtype=np.float64)
    a_sys[:m_pts, m_pts:] = p_mat
    a_sys[m_pts:, :m_pts] = p_mat.T

    rhs = np.zeros((m_pts + 3, modal_coeffs.shape[1]), dtype=np.float64)
    rhs[:m_pts, :] = modal_coeffs

    sol = np.linalg.solve(a_sys, rhs)
    rbf_w = sol[:m_pts, :]  # (M, R)
    poly_w = sol[m_pts:, :].T  # (R, 3)
    return rbf_w, poly_w


def _eval_multiquadric_rbf(
    ux: float,
    uy: float,
    centers_ux: np.ndarray,
    centers_uy: np.ndarray,
    rbf_w: np.ndarray,
    poly_w: np.ndarray,
    shape_c: float,
) -> np.ndarray:
    dx = ux - centers_ux
    dy = uy - centers_uy
    phi = np.sqrt(dx * dx + dy * dy + shape_c * shape_c)
    poly_vec = np.array([1.0, ux, uy], dtype=np.float64)
    return phi @ rbf_w + poly_w @ poly_vec


def _quantize_i16(val: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0.0:
        return np.zeros_like(val, dtype=np.int16)
    q = np.rint(val / scale)
    return np.clip(q, -32767.0, 32767.0).astype("<i2")


def _pack_grid_desc(grid: GridSpec, node_offset: int) -> bytes:
    return struct.pack(
        "<4I3d3d2Q",
        grid.nx,
        grid.ny,
        grid.nz,
        grid.brick_count,
        float(grid.origin_enu[0]),
        float(grid.origin_enu[1]),
        float(grid.origin_enu[2]),
        float(grid.spacing_m[0]),
        float(grid.spacing_m[1]),
        float(grid.spacing_m[2]),
        int(node_offset),
        int(grid.node_count),
    )


def compress_and_pack_vrtawnd1(
    out_path: Path | str,
    *,
    tier0_spec: GridSpec,
    tier1_spec: GridSpec,
    inflow_dirs_rad: np.ndarray,
    inflow_speeds_mps: np.ndarray,
    tier0_snapshots_enu: np.ndarray,
    tier1_snapshots_enu: np.ndarray,
    tier0_sigma_enu: np.ndarray,
    tier1_sigma_enu: np.ndarray,
    tier0_lturb_m: np.ndarray,
    tier1_lturb_m: np.ndarray,
    datum_lat_deg: float = 40.8860,
    datum_lon_deg: float = 29.1880,
    datum_alt_hae_m: float = 48.7,
    max_lon_deg: float = 29.1892,
    rbf_shape_c: float = 4.0,
) -> dict[str, Any]:
    """Run SVD POD (R=8) + 48-center multiquadric RBF + LODO p95 residual and pack `VRTAWND1`."""
    tier0_spec.validate()
    tier1_spec.validate()
    if max_lon_deg >= MAX_ALLOWED_LON_DEG:
        raise ValueError(
            f"Refusing to pack VRTAWND1 with max_lon_deg={max_lon_deg:.6f} >= {MAX_ALLOWED_LON_DEG}"
        )

    m_snaps = int(inflow_dirs_rad.shape[0])
    if m_snaps != VRTAWND1_NUM_CENTERS:
        raise ValueError(
            f"VRTAWND1 requires exactly {VRTAWND1_NUM_CENTERS} inflow snapshots, got {m_snaps}"
        )

    # Flatten (..., nx, ny, nz[, C]) -> (..., N_nodes[, C]) in 4x4x4 Morton brick order
    def _to_morton_order(field: np.ndarray, spec: GridSpec, *, has_c: bool) -> np.ndarray:
        lead = field.shape[:-4] if has_c else field.shape[:-3]
        c_dim = field.shape[-1] if has_c else 1
        reshaped = field.reshape(*lead, spec.nx, spec.ny, spec.nz, c_dim)
        out = np.empty((*lead, spec.node_count, c_dim), dtype=np.float64)
        for iz in range(spec.nz):
            for iy in range(spec.ny):
                for ix in range(spec.nx):
                    m_idx = morton_swizzle_index(ix, iy, iz, spec.nx, spec.ny)
                    out[..., m_idx, :] = reshaped[..., ix, iy, iz, :]
        return out if has_c else out[..., 0]

    t0_vel = _to_morton_order(tier0_snapshots_enu, tier0_spec, has_c=True)  # (M, N0, 3)
    t1_vel = _to_morton_order(tier1_snapshots_enu, tier1_spec, has_c=True)  # (M, N1, 3)
    t0_sig = _to_morton_order(tier0_sigma_enu, tier0_spec, has_c=True)  # (N0, 3)
    t1_sig = _to_morton_order(tier1_sigma_enu, tier1_spec, has_c=True)  # (N1, 3)
    t0_lt = _to_morton_order(tier0_lturb_m, tier0_spec, has_c=False)  # (N0,)
    t1_lt = _to_morton_order(tier1_lturb_m, tier1_spec, has_c=False)  # (N1,)

    vel_all = np.concatenate([t0_vel, t1_vel], axis=1)  # (M, N_tot, 3)
    sig_all = np.concatenate([t0_sig, t1_sig], axis=0)  # (N_tot, 3)
    lt_all = np.concatenate([t0_lt, t1_lt], axis=0)  # (N_tot,)
    n_tot = vel_all.shape[1]

    u0 = vel_all.mean(axis=0)  # (N_tot, 3)
    fluct = (vel_all - u0[None, :, :]).reshape(m_snaps, n_tot * 3)  # (M, 3*N_tot)

    # Economy SVD: fluct = U @ diag(S) @ Vt
    u_svd, s_svd, vt_svd = np.linalg.svd(fluct, full_matrices=False)
    r_modes = VRTAWND1_NUM_MODES
    # Spatial modes phi_modes: (R, N_tot, 3), modal amplitudes a_snaps: (M, R)
    phi_modes = vt_svd[:r_modes, :].reshape(r_modes, n_tot, 3)
    a_snaps = u_svd[:, :r_modes] * s_svd[None, :r_modes]

    # Normalize each spatial mode to unit peak magnitude so int16 dynamic range is maximized
    mode_scales = np.zeros(r_modes, dtype=np.float64)
    for m_idx in range(r_modes):
        peak = float(np.max(np.abs(phi_modes[m_idx])))
        if peak < 1e-12:
            peak = 1.0
        phi_modes[m_idx] /= peak
        a_snaps[:, m_idx] *= peak
        mode_scales[m_idx] = 1.0 / 32767.0

    # Fit 48-center multiquadric RBF in continuous (ux, uy) inflow velocity space
    centers_ux = inflow_speeds_mps * np.cos(inflow_dirs_rad)
    centers_uy = inflow_speeds_mps * np.sin(inflow_dirs_rad)
    rbf_w, poly_w = _fit_multiquadric_rbf(
        centers_ux, centers_uy, a_snaps, shape_c=rbf_shape_c
    )

    # Calibrate spatial LODO residual eps_rom_p95 [m/s] across unique inflow directions
    uniq_dirs = np.unique(np.round(inflow_dirs_rad, 6))
    err_accum = np.zeros((m_snaps, n_tot), dtype=np.float64)
    for d_val in uniq_dirs:
        hold_mask = np.abs(inflow_dirs_rad - d_val) < 1e-5
        train_mask = ~hold_mask
        if int(train_mask.sum()) < 6:
            continue
        w_sub, p_sub = _fit_multiquadric_rbf(
            centers_ux[train_mask],
            centers_uy[train_mask],
            a_snaps[train_mask],
            shape_c=rbf_shape_c,
        )
        for s_idx in np.nonzero(hold_mask)[0]:
            a_hat = _eval_multiquadric_rbf(
                float(centers_ux[s_idx]),
                float(centers_uy[s_idx]),
                centers_ux[train_mask],
                centers_uy[train_mask],
                w_sub,
                p_sub,
                rbf_shape_c,
            )
            u_hat = u0 + np.tensordot(a_hat, phi_modes, axes=([0], [0]))
            err_vec = vel_all[s_idx] - u_hat
            err_accum[s_idx] = np.linalg.norm(err_vec, axis=-1)

    # Per-node 95th-percentile LODO error + modest conservative floor (0.15 m/s)
    eps_rom_p95 = np.maximum(np.percentile(err_accum, 95, axis=0), 0.15)

    # Quantize every node to exact 32 x int16_t = 64 B (`VrtaWndNode64`)
    vel_scale = max(float(np.max(np.abs(u0))), 1.0) / 32767.0
    sigma_scale = max(float(np.max(np.abs(sig_all))), 1.0) / 32767.0
    eps_scale = max(float(np.max(eps_rom_p95)), 1.0) / 32767.0
    lturb_scale = max(float(np.max(lt_all)), 1.0) / 32767.0

    nodes_i16 = np.zeros((n_tot, 32), dtype="<i2")
    nodes_i16[:, 0:3] = _quantize_i16(u0, vel_scale)
    for m_idx in range(r_modes):
        nodes_i16[:, 3 + 3 * m_idx : 6 + 3 * m_idx] = _quantize_i16(
            phi_modes[m_idx], mode_scales[m_idx]
        )
    nodes_i16[:, 27:30] = _quantize_i16(sig_all, sigma_scale)
    nodes_i16[:, 30] = _quantize_i16(eps_rom_p95, eps_scale)
    nodes_i16[:, 31] = _quantize_i16(lt_all, lturb_scale)

    payload_bytes_raw = nodes_i16.tobytes()
    assert len(payload_bytes_raw) == n_tot * VRTAWND1_NODE_BYTES
    payload_sha256_bytes = hashlib.sha256(payload_bytes_raw).digest()

    # Pack exact 12,288 B header matching C++ VrtaWndHeader byte-for-byte
    hdr_parts: list[bytes] = []
    hdr_parts.append(
        struct.pack(
            "<8s4I4dd8d4d",
            VRTAWND1_MAGIC,
            VRTAWND1_VERSION,
            0x03,  # bit0: has_tier1, bit1: morton_4x4x4
            VRTAWND1_NUM_MODES,
            VRTAWND1_NUM_CENTERS,
            float(datum_lat_deg),
            float(datum_lon_deg),
            float(datum_alt_hae_m),
            float(max_lon_deg),
            float(vel_scale),
            *[float(x) for x in mode_scales],
            float(sigma_scale),
            float(eps_scale),
            float(lturb_scale),
            float(rbf_shape_c),
        )
    )
    hdr_parts.append(_pack_grid_desc(tier0_spec, node_offset=0))
    hdr_parts.append(_pack_grid_desc(tier1_spec, node_offset=tier0_spec.node_count))
    hdr_parts.append(struct.pack("<48d", *[float(x) for x in centers_ux]))
    hdr_parts.append(struct.pack("<48d", *[float(x) for x in centers_uy]))
    hdr_parts.append(struct.pack("<384d", *[float(x) for x in rbf_w.reshape(-1)]))
    hdr_parts.append(struct.pack("<24d", *[float(x) for x in poly_w.reshape(-1)]))
    hdr_parts.append(struct.pack("<2Q", int(n_tot), int(len(payload_bytes_raw))))
    hdr_parts.append(payload_sha256_bytes)

    header_body = b"".join(hdr_parts)
    if len(header_body) != 4400:
        raise RuntimeError(f"Internal ABI mismatch: header body is {len(header_body)} B != 4400 B")
    header_full = header_body + bytes(VRTAWND1_HEADER_BYTES - len(header_body))
    assert len(header_full) == VRTAWND1_HEADER_BYTES

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(header_full + payload_bytes_raw)

    return {
        "file": str(out_file),
        "total_bytes": len(header_full) + len(payload_bytes_raw),
        "header_bytes": VRTAWND1_HEADER_BYTES,
        "node_bytes": VRTAWND1_NODE_BYTES,
        "total_nodes": int(n_tot),
        "tier0_dims": [tier0_spec.nx, tier0_spec.ny, tier0_spec.nz],
        "tier1_dims": [tier1_spec.nx, tier1_spec.ny, tier1_spec.nz],
        "full_scale_size_bytes": FULL_SCALE_SIZE_BYTES,
        "max_lon_deg": float(max_lon_deg),
        "eps_rom_p95_min_mps": round(float(eps_rom_p95.min()), 4),
        "eps_rom_p95_max_mps": round(float(eps_rom_p95.max()), 4),
        "payload_sha256": payload_sha256_bytes.hex(),
    }
