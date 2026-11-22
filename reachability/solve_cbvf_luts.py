"""Offline 4D Fault-Degraded CBVF LUT Solver & .vrta_lut Binary Exporter.

Supports two execution modes:
  1. Full Colab G4 Blackwell JAX mode (`--mode jax`, requires `jax` + `hj_reachability` v0.7.0):
     Solves the exact 4D HJI-VI on the 2:1 nested fine grid 81 x 57 x 81 x 57 FP32
     (WENO5 + TVD-RK3, backwards_reachable_tube, conservative_smax_4 terminal set),
     decimates by exact stride-2 to 41 x 29 x 41 x 29 int16 (q = 0.992 mm), and
     audits exact maximum optimistic overestimation eps_grid_f over all 21.3M nodes.
  2. Deterministic Local CPU Fixture mode (`--quick-cpu-fixture`, pure NumPy):
     Evaluates the closed-form kinematic conservative 4D CBVF approximation on the
     exact 41 x 29 x 41 x 29 grid across all 4 CSFL regimes, audits exact 2:1 nested
     81 x 57 x 81 x 57 fine-grid eps_grid_f via chunked exact multilinear upsampling,
     and exports a bit-for-bit reproducible 11,313,864 B `.vrta_lut` fixture.

Canonical `.vrta_lut` Binary Layout (Little-Endian, Total = 11,313,864 B):
  Header (`4,096 B` page-aligned, naturally 4-byte aligned, C++20 POD-compatible):
      0..7   (8 B): char     magic[8]           = "VRTALUT1"
      8..11  (4 B): uint32_t version            = 1
     12..15  (4 B): uint32_t num_regimes        = 4
     16..31 (16 B): uint32_t dims[4]            = {41, 29, 41, 29}
     32..47 (16 B): float    lo[4]              = {-2.0, -24.0, -5.0, -12.0}
     48..63 (16 B): float    hi[4]              = {58.0,  16.0, 55.0,   8.0}
     64..67  (4 B): float    scale              = 65.0 / 65534.0  (~0.00099185 m)
     68..71  (4 B): float    offset             = 7.5             (m)
     72..87 (16 B): float    eps_grid[4]        = per-regime conservative margin [m]
     88..119(32 B): uint8_t  payload_sha256[32] = SHA-256 of raw int16 payload
    120..135(16 B): float    v_cap_f[4]         = degraded cruise speed caps [m/s]
    136..139 (4 B): uint32_t flags              = bit0: quick_cpu_fixture,
                                                  bit1: has_v_cap,
                                                  bit2: conservative_smax
    140..143 (4 B): float    horizon_s          = 6.0
    144..147 (4 B): float    d_ar_max           = 1.75
    148..151 (4 B): float    d_az_max           = 0.75
    152..4095:      uint8_t  reserved[3944]     = 0x00
  Payload (`11,309,768 B` = 4 * 41 * 29 * 41 * 29 * 2 B):
    C-contiguous `<i2` (little-endian int16), innermost axis v_z (N3 = 29).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from reachability.cbvf_4d_dynamics import (
    HAS_HJ_REACHABILITY,
    HAS_JAX,
    RegimeSpec,
    VertiRta4DDynamics,
    build_regimes_from_config,
    compute_all_regime_v_caps,
    conservative_smax_4,
    hj,
    jax,
    jnp,
    load_vehicle_params,
)

LUT_MAGIC = b"VRTALUT1"
LUT_VERSION = 1
LUT_HEADER_BYTES = 4096
LUT_NUM_REGIMES = 4
LUT_EXPORT_SHAPE = (41, 29, 41, 29)
LUT_FINE_SHAPE = (81, 57, 81, 57)
LUT_ELEMENTS_PER_REGIME = 41 * 29 * 41 * 29  # 1,413,721
LUT_PAYLOAD_ELEMENTS = LUT_NUM_REGIMES * LUT_ELEMENTS_PER_REGIME  # 5,654,884
LUT_PAYLOAD_BYTES = LUT_PAYLOAD_ELEMENTS * 2  # 11,309,768 B
LUT_TOTAL_BYTES = LUT_HEADER_BYTES + LUT_PAYLOAD_BYTES  # 11,313,864 B exact

LUT_LO = (-2.0, -24.0, -5.0, -12.0)
LUT_HI = (58.0, 16.0, 55.0, 8.0)
LUT_V_LO = -25.0
LUT_V_HI = 40.0
LUT_SCALE = np.float32((LUT_V_HI - LUT_V_LO) / 65534.0)
LUT_OFFSET = np.float32(0.5 * (LUT_V_HI + LUT_V_LO))
LUT_HORIZON_S = 6.0
LUT_BETA_SMAX = 4.0

FLAG_QUICK_CPU_FIXTURE = 1 << 0
FLAG_HAS_V_CAP = 1 << 1
FLAG_CONSERVATIVE_SMAX = 1 << 2

LUT_HEADER_STRUCT = struct.Struct("<8sII4I4f4f2f4f32s4fI3f")
assert LUT_HEADER_STRUCT.size == 152


@dataclass(frozen=True)
class VrtaLutHeader:
    """Parsed `.vrta_lut` 4096 B header."""

    magic: bytes
    version: int
    num_regimes: int
    dims: tuple[int, int, int, int]
    lo: tuple[float, float, float, float]
    hi: tuple[float, float, float, float]
    scale: float
    offset: float
    eps_grid: tuple[float, float, float, float]
    payload_sha256: bytes
    v_cap_f: tuple[float, float, float, float]
    flags: int
    horizon_s: float
    d_ar_max: float
    d_az_max: float

    @property
    def is_quick_fixture(self) -> bool:
        return bool(self.flags & FLAG_QUICK_CPU_FIXTURE)

    @property
    def dx(self) -> tuple[float, float, float, float]:
        return tuple(
            (self.hi[d] - self.lo[d]) / (self.dims[d] - 1) for d in range(4)
        )  # type: ignore[return-value]


def pack_lut_header(hdr: VrtaLutHeader) -> bytes:
    """Serialize VrtaLutHeader into exact 4,096 B zero-padded little-endian bytes."""
    raw = LUT_HEADER_STRUCT.pack(
        hdr.magic,
        hdr.version,
        hdr.num_regimes,
        *hdr.dims,
        *hdr.lo,
        *hdr.hi,
        hdr.scale,
        hdr.offset,
        *hdr.eps_grid,
        hdr.payload_sha256,
        *hdr.v_cap_f,
        hdr.flags,
        hdr.horizon_s,
        hdr.d_ar_max,
        hdr.d_az_max,
    )
    return raw.ljust(LUT_HEADER_BYTES, b"\x00")


def unpack_lut_header(buf: bytes) -> VrtaLutHeader:
    """Parse and validate a 4,096 B `.vrta_lut` header."""
    if len(buf) < LUT_HEADER_BYTES:
        raise ValueError(f"Header buffer too small: {len(buf)} < {LUT_HEADER_BYTES}")
    fields = LUT_HEADER_STRUCT.unpack_from(buf, 0)
    magic = fields[0]
    if magic != LUT_MAGIC:
        raise ValueError(f"Invalid .vrta_lut magic: {magic!r} != {LUT_MAGIC!r}")
    version = int(fields[1])
    if version != LUT_VERSION:
        raise ValueError(f"Unsupported .vrta_lut version: {version}")
    num_regimes = int(fields[2])
    dims = (int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6]))
    lo = (float(fields[7]), float(fields[8]), float(fields[9]), float(fields[10]))
    hi = (float(fields[11]), float(fields[12]), float(fields[13]), float(fields[14]))
    scale = float(fields[15])
    offset = float(fields[16])
    eps_grid = (float(fields[17]), float(fields[18]), float(fields[19]), float(fields[20]))
    payload_sha256 = bytes(fields[21])
    v_cap_f = (float(fields[22]), float(fields[23]), float(fields[24]), float(fields[25]))
    flags = int(fields[26])
    horizon_s = float(fields[27])
    d_ar_max = float(fields[28])
    d_az_max = float(fields[29])
    return VrtaLutHeader(
        magic=magic,
        version=version,
        num_regimes=num_regimes,
        dims=dims,
        lo=lo,
        hi=hi,
        scale=scale,
        offset=offset,
        eps_grid=eps_grid,
        payload_sha256=payload_sha256,
        v_cap_f=v_cap_f,
        flags=flags,
        horizon_s=horizon_s,
        d_ar_max=d_ar_max,
        d_az_max=d_az_max,
    )


def load_vrta_lut(
    path: Path | str, *, verify_sha256: bool = True
) -> tuple[VrtaLutHeader, np.ndarray]:
    """Load `.vrta_lut`, verifying exact size, header fields, and payload SHA-256."""
    raw = Path(path).read_bytes()
    if len(raw) != LUT_TOTAL_BYTES:
        raise ValueError(
            f"Invalid .vrta_lut file size: {len(raw)} B != expected {LUT_TOTAL_BYTES} B"
        )
    hdr = unpack_lut_header(raw[:LUT_HEADER_BYTES])
    if hdr.num_regimes != LUT_NUM_REGIMES or hdr.dims != LUT_EXPORT_SHAPE:
        raise ValueError(
            f"Unexpected .vrta_lut geometry: regimes={hdr.num_regimes}, dims={hdr.dims}"
        )
    payload = raw[LUT_HEADER_BYTES:]
    if verify_sha256:
        digest = hashlib.sha256(payload).digest()
        if digest != hdr.payload_sha256:
            raise ValueError(
                f"SHA-256 mismatch in {path}: "
                f"computed {digest.hex()} != header {hdr.payload_sha256.hex()}"
            )
    arr = np.frombuffer(payload, dtype="<i2").reshape((hdr.num_regimes, *hdr.dims))
    return hdr, arr


def eval_cbvf_4d(
    hdr: VrtaLutHeader,
    data: np.ndarray,
    regime_idx: int,
    r: float,
    vr: float,
    z: float,
    vz: float,
) -> tuple[float, tuple[float, float, float, float]]:
    """Reference 16-corner quadrilinear evaluation matching C++20 MmapCbvf4DLut::eval.

    Returns (conservative_value_m, (dV/dr, dV/dvr, dV/dz, dV/dvz)) where
    conservative_value_m = V_hat(x) - eps_grid[regime_idx].
    Raises ValueError on OOB regime_idx or non-finite query inputs (DR-2 Erratum #4).
    """
    if not (0 <= regime_idx < hdr.num_regimes):
        raise ValueError(f"regime_idx {regime_idx} out of bounds [0, {hdr.num_regimes - 1}]")
    q = (float(r), float(vr), float(z), float(vz))
    if not all(math.isfinite(x) for x in q):
        raise ValueError(f"Non-finite CBVF query state: {q}")

    inv_dx = [(hdr.dims[d] - 1) / (hdr.hi[d] - hdr.lo[d]) for d in range(4)]
    idx = [0, 0, 0, 0]
    frac = [0.0, 0.0, 0.0, 0.0]
    for d in range(4):
        u = (q[d] - hdr.lo[d]) * inv_dx[d]
        uc = min(max(u, 0.0), float(hdr.dims[d] - 1) - 1e-9)
        i_d = int(uc)
        idx[d] = i_d
        frac[d] = uc - float(i_d)

    i0, i1, i2, i3 = idx
    scale = float(hdr.scale)
    offset = float(hdr.offset)
    slab = data[regime_idx, i0 : i0 + 2, i1 : i1 + 2, i2 : i2 + 2, i3 : i3 + 2]
    c = slab.astype(np.float64) * scale + offset  # shape (2, 2, 2, 2)

    f0, f1, f2, f3 = frac
    o0, o1, o2, o3 = 1.0 - f0, 1.0 - f1, 1.0 - f2, 1.0 - f3

    # Axis 3 (vz)
    a = c[..., 0] * o3 + c[..., 1] * f3
    da3 = c[..., 1] - c[..., 0]
    # Axis 2 (z)
    b = a[..., 0] * o2 + a[..., 1] * f2
    db2 = a[..., 1] - a[..., 0]
    db3 = da3[..., 0] * o2 + da3[..., 1] * f2
    # Axis 1 (vr)
    g = b[..., 0] * o1 + b[..., 1] * f1
    dg1 = b[..., 1] - b[..., 0]
    dg2 = db2[..., 0] * o1 + db2[..., 1] * f1
    dg3 = db3[..., 0] * o1 + db3[..., 1] * f1
    # Axis 0 (r)
    val = float(g[0] * o0 + g[1] * f0) - float(hdr.eps_grid[regime_idx])
    grad = (
        float((g[1] - g[0]) * inv_dx[0]),
        float((dg1[0] * o0 + dg1[1] * f0) * inv_dx[1]),
        float((dg2[0] * o0 + dg2[1] * f0) * inv_dx[2]),
        float((dg3[0] * o0 + dg3[1] * f0) * inv_dx[3]),
    )
    return val, grad


def _quantize_int16(
    v_fp32: np.ndarray,
    v_lo: float = LUT_V_LO,
    v_hi: float = LUT_V_HI,
    scale: np.float32 = LUT_SCALE,
    offset: np.float32 = LUT_OFFSET,
) -> np.ndarray:
    """Quantize FP32 value grid onto [-32767, +32767] int16 (q = 0.992 mm)."""
    vc = np.clip(v_fp32, np.float32(v_lo), np.float32(v_hi))
    q = np.rint((vc - offset) / scale)
    return np.clip(q, -32767, 32767).astype("<i2")


def _kinematic_braking_1d(
    pos: np.ndarray,
    vel: np.ndarray,
    margin: float,
    a_eff: float,
    d_v: float,
    horizon_s: float,
) -> np.ndarray:
    """Closed-form worst-case 1D braking signed distance over [0, horizon_s]."""
    v_closing = np.maximum(np.float32(0.0), np.float32(d_v) - vel)
    t_brake = np.minimum(np.float32(horizon_s), v_closing / np.float32(a_eff))
    d_stop = v_closing * t_brake - np.float32(0.5 * a_eff) * (t_brake * t_brake)
    return (pos - np.float32(margin)) - d_stop


def _upsample_3d_2x(slab: np.ndarray) -> np.ndarray:
    """Exact 2:1 nested trilinear upsampling (29, 41, 29) -> (57, 81, 57) in FP32."""
    n1, n2, n3 = slab.shape
    # Axis 2: 29 -> 57
    u3 = np.empty((n1, n2, 2 * n3 - 1), dtype=np.float32)
    u3[:, :, 0::2] = slab
    u3[:, :, 1::2] = np.float32(0.5) * (slab[:, :, :-1] + slab[:, :, 1:])
    # Axis 1: 41 -> 81
    u2 = np.empty((n1, 2 * n2 - 1, 2 * n3 - 1), dtype=np.float32)
    u2[:, 0::2, :] = u3
    u2[:, 1::2, :] = np.float32(0.5) * (u3[:, :-1, :] + u3[:, 1:, :])
    # Axis 0: 29 -> 57
    u1 = np.empty((2 * n1 - 1, 2 * n2 - 1, 2 * n3 - 1), dtype=np.float32)
    u1[0::2, :, :] = u2
    u1[1::2, :, :] = np.float32(0.5) * (u2[:-1, :, :] + u2[1:, :, :])
    return u1


def _solve_regime_quick_cpu(
    spec: RegimeSpec,
    *,
    horizon_s: float = LUT_HORIZON_S,
    beta: float = LUT_BETA_SMAX,
) -> tuple[np.ndarray, float]:
    """Fast, physically sound pure-NumPy 4D kinematic CBVF + exact 21.3M-node eps_grid audit."""
    a_r_eff = max(0.5, spec.a_xy_hover_max - spec.d_ar)
    a_z_eff = max(0.5, spec.a_z_up_max - spec.d_az)

    r_exp = np.linspace(LUT_LO[0], LUT_HI[0], LUT_EXPORT_SHAPE[0], dtype=np.float32)
    vr_exp = np.linspace(LUT_LO[1], LUT_HI[1], LUT_EXPORT_SHAPE[1], dtype=np.float32)
    z_exp = np.linspace(LUT_LO[2], LUT_HI[2], LUT_EXPORT_SHAPE[2], dtype=np.float32)
    vz_exp = np.linspace(LUT_LO[3], LUT_HI[3], LUT_EXPORT_SHAPE[3], dtype=np.float32)

    sr_exp = _kinematic_braking_1d(
        r_exp[:, None], vr_exp[None, :], spec.r_safe, a_r_eff, spec.d_vr, horizon_s
    )  # (41, 29)
    sz_exp = _kinematic_braking_1d(
        z_exp[:, None], vz_exp[None, :], spec.z_safe, a_z_eff, spec.d_vz, horizon_s
    )  # (41, 29)

    v_exp = conservative_smax_4(
        sr_exp[:, :, None, None], sz_exp[None, None, :, :], beta=beta
    ).astype(np.float32)
    vq = _quantize_int16(v_exp)
    v_deq = vq.astype(np.float32) * LUT_SCALE + LUT_OFFSET

    # Exact 2:1 nested fine-grid audit across ALL 81 x 57 x 81 x 57 = 21,316,689 nodes
    # Sweeping coarse r slabs keeps peak RAM ~3 MB while remaining 100% exact.
    r_fine = np.linspace(LUT_LO[0], LUT_HI[0], LUT_FINE_SHAPE[0], dtype=np.float32)
    vr_fine = np.linspace(LUT_LO[1], LUT_HI[1], LUT_FINE_SHAPE[1], dtype=np.float32)
    z_fine = np.linspace(LUT_LO[2], LUT_HI[2], LUT_FINE_SHAPE[2], dtype=np.float32)
    vz_fine = np.linspace(LUT_LO[3], LUT_HI[3], LUT_FINE_SHAPE[3], dtype=np.float32)

    sz_fine = _kinematic_braking_1d(
        z_fine[:, None], vz_fine[None, :], spec.z_safe, a_z_eff, spec.d_vz, horizon_s
    )  # (81, 57)

    eps_max = 0.0
    slab_lo = _upsample_3d_2x(v_deq[0])
    v_lo_f32 = np.float32(LUT_V_LO)
    v_hi_f32 = np.float32(LUT_V_HI)
    band_lim = np.float32(10.0)

    def _audit_slice(ir: int, v_hat_slice: np.ndarray) -> float:
        sr_row = _kinematic_braking_1d(
            r_fine[ir : ir + 1, None],
            vr_fine[None, :],
            spec.r_safe,
            a_r_eff,
            spec.d_vr,
            horizon_s,
        )[0]  # (57,)
        v_fine_slice = conservative_smax_4(
            sr_row[:, None, None], sz_fine[None, :, :], beta=beta
        )
        v_fine_c = np.clip(v_fine_slice, v_lo_f32, v_hi_f32)
        over = np.where(
            np.abs(v_fine_c) <= band_lim,
            np.maximum(np.float32(0.0), v_hat_slice - v_fine_c),
            np.float32(0.0),
        )
        return float(np.max(over))

    for cr in range(LUT_EXPORT_SHAPE[0] - 1):
        eps_max = max(eps_max, _audit_slice(2 * cr, slab_lo))
        slab_hi = _upsample_3d_2x(v_deq[cr + 1])
        slab_mid = np.float32(0.5) * (slab_lo + slab_hi)
        eps_max = max(eps_max, _audit_slice(2 * cr + 1, slab_mid))
        slab_lo = slab_hi
    eps_max = max(eps_max, _audit_slice(LUT_FINE_SHAPE[0] - 1, slab_lo))

    eps_grid_f = eps_max + 0.5 * float(LUT_SCALE)
    return vq, float(eps_grid_f)


def _solve_regime_jax_hj(
    spec: RegimeSpec,
    *,
    fine_shape: tuple[int, int, int, int] = LUT_FINE_SHAPE,
    horizon_s: float = LUT_HORIZON_S,
    beta: float = LUT_BETA_SMAX,
) -> tuple[np.ndarray, float]:
    """Full Colab G4 Blackwell JAX hj_reachability v0.7.0 4D HJI-VI solver."""
    if not (HAS_JAX and HAS_HJ_REACHABILITY and jax is not None and jnp is not None and hj is not None):
        raise RuntimeError(
            "JAX + StanfordASL/hj_reachability v0.7.0 required for --mode jax. "
            "Use --quick-cpu-fixture for local CPU fixture generation."
        )
    if any((f - 1) != 2 * (e - 1) for f, e in zip(fine_shape, LUT_EXPORT_SHAPE)):
        raise ValueError(
            f"fine_shape {fine_shape} must be exact 2:1 nested over {LUT_EXPORT_SHAPE}"
        )

    lo = np.array(LUT_LO, dtype=np.float32)
    hi = np.array(LUT_HI, dtype=np.float32)
    grid_fine = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=lo, hi=hi),
        shape=fine_shape,
    )
    grid_exp = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=lo, hi=hi),
        shape=LUT_EXPORT_SHAPE,
    )
    sr = grid_fine.states[..., 0] - np.float32(spec.r_safe)
    sz = grid_fine.states[..., 2] - np.float32(spec.z_safe)
    l_init = conservative_smax_4(sr, sz, beta=beta)

    settings = hj.SolverSettings.with_accuracy(
        "very_high",  # WENO5 + TVD-RK3 (CFL = 0.75)
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )
    dyn = VertiRta4DDynamics.from_regime_spec(spec)
    v_fine = hj.step(settings, dyn, grid_fine, 0.0, l_init, -float(horizon_s))
    v_exp = v_fine[::2, ::2, ::2, ::2]  # Exact 2:1 nested stride-2 decimation

    vc = jnp.clip(v_exp, LUT_V_LO, LUT_V_HI)
    vq_jax = jnp.clip(jnp.rint((vc - LUT_OFFSET) / LUT_SCALE), -32767, 32767).astype(jnp.int16)
    v_deq = vq_jax.astype(jnp.float32) * LUT_SCALE + LUT_OFFSET

    interp_batch = jax.jit(jax.vmap(grid_exp.interpolate, in_axes=(None, 0)))
    fine_pts = grid_fine.states.reshape(-1, 4)
    v_hat_fine = interp_batch(v_deq, fine_pts).reshape(fine_shape)
    v_fine_c = jnp.clip(v_fine, LUT_V_LO, LUT_V_HI)
    band = jnp.abs(v_fine_c) <= 10.0
    eps_max = float(jnp.max(jnp.where(band, jnp.maximum(0.0, v_hat_fine - v_fine_c), 0.0)))
    eps_grid_f = eps_max + 0.5 * float(LUT_SCALE)
    return np.asarray(vq_jax, dtype="<i2"), float(eps_grid_f)


def solve_and_export_lut(
    out_path: Path | str,
    *,
    quick_cpu_fixture: bool = False,
    config_path: Path | str | None = None,
    fine_shape: tuple[int, int, int, int] = LUT_FINE_SHAPE,
    horizon_s: float = LUT_HORIZON_S,
) -> VrtaLutHeader:
    """Solve all 4 CSFL regimes and export exact 11,313,864 B `.vrta_lut` binary."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    params = load_vehicle_params(config_path)
    regimes = build_regimes_from_config(config_path)
    v_cap_f = compute_all_regime_v_caps(
        regimes,
        r_max=LUT_HI[0],
        horizon_s=horizon_s,
        v_max=params["v_max_mps"],
    )

    payload = bytearray()
    eps_grid_list: list[float] = []
    for spec in regimes:
        if quick_cpu_fixture:
            vq, eps_f = _solve_regime_quick_cpu(spec, horizon_s=horizon_s)
        else:
            vq, eps_f = _solve_regime_jax_hj(
                spec, fine_shape=fine_shape, horizon_s=horizon_s
            )
        if vq.shape != LUT_EXPORT_SHAPE:
            raise RuntimeError(f"Unexpected regime grid shape {vq.shape} != {LUT_EXPORT_SHAPE}")
        eps_grid_list.append(eps_f)
        payload.extend(np.ascontiguousarray(vq, dtype="<i2").tobytes(order="C"))

    if len(payload) != LUT_PAYLOAD_BYTES:
        raise RuntimeError(
            f"Internal error: payload {len(payload)} B != {LUT_PAYLOAD_BYTES} B"
        )

    sha256 = hashlib.sha256(payload).digest()
    flags = FLAG_HAS_V_CAP | FLAG_CONSERVATIVE_SMAX
    if quick_cpu_fixture:
        flags |= FLAG_QUICK_CPU_FIXTURE

    hdr = VrtaLutHeader(
        magic=LUT_MAGIC,
        version=LUT_VERSION,
        num_regimes=LUT_NUM_REGIMES,
        dims=LUT_EXPORT_SHAPE,
        lo=LUT_LO,
        hi=LUT_HI,
        scale=float(LUT_SCALE),
        offset=float(LUT_OFFSET),
        eps_grid=tuple(eps_grid_list),  # type: ignore[arg-type]
        payload_sha256=sha256,
        v_cap_f=v_cap_f,
        flags=flags,
        horizon_s=float(horizon_s),
        d_ar_max=float(regimes[0].d_ar),
        d_az_max=float(regimes[0].d_az),
    )
    hdr_bytes = pack_lut_header(hdr)
    assert len(hdr_bytes) == LUT_HEADER_BYTES

    with out.open("wb") as f:
        f.write(hdr_bytes)
        f.write(payload)

    assert out.stat().st_size == LUT_TOTAL_BYTES
    return hdr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VertiRTA 4D Fault-Degraded CBVF LUT Solver & Exporter (.vrta_lut)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/quick_test.vrta_lut"),
        help="Output .vrta_lut file path (exact 11,313,864 B).",
    )
    parser.add_argument(
        "--quick-cpu-fixture",
        action="store_true",
        help="Use fast deterministic pure-NumPy kinematic CBVF + exact 21.3M-node eps_grid audit.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config/vehicle_x8_240.yaml.",
    )
    parser.add_argument(
        "--horizon-s",
        type=float,
        default=LUT_HORIZON_S,
        help="Backward reachability time horizon in seconds (default: 6.0).",
    )
    args = parser.parse_args(argv)

    hdr = solve_and_export_lut(
        args.out,
        quick_cpu_fixture=args.quick_cpu_fixture,
        config_path=args.config,
        horizon_s=args.horizon_s,
    )
    print(
        f"[solve_cbvf_luts] Exported {args.out} ({args.out.stat().st_size:,} B, "
        f"mode={'quick_cpu_fixture' if hdr.is_quick_fixture else 'jax_hj'}, "
        f"payload_sha256={hdr.payload_sha256.hex()})"
    )
    print(f"  eps_grid [m]  : {[round(e, 5) for e in hdr.eps_grid]}")
    print(f"  v_cap_f [m/s] : {[round(v, 3) for v in hdr.v_cap_f]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
