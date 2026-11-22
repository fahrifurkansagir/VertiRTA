"""Offline 2D CEL Vertical Braking LUT Solver & `.vrta_cel2d` Binary Exporter.

Solves the 2D double-integrator HJI-VI for SAME_ARM_DUAL_LOSS Controlled Emergency
Landing (CEL) across all 3 voltage-sag tiers (T/W in {1.25, 1.20, 1.15}, DR-2 §1.3):

    0 = min( l_2D(z, v_z) - V_2D,
             D_t V_2D + dV/dz * v_z + max_{a_z in [a_z_min, a_z_max]} dV/dv_z * a_z
             - |dV/dv_z| * d_z_max )

with unsafe terminal set F_2D = { z <= 0 and v_z < -v_td_max } (v_td_max = 2.0 m/s),
domain z in [-2.0, 58.0] m, v_z in [-12.0, 4.0] m/s, solved via 5th-order WENO5 +
Lax-Friedrichs + TVD-RK3 on a 2:1 nested fine grid (81 x 57) (or JAX hj_reachability
v0.7.0 when available and requested) and stride-2 decimated to 41 x 29 int16 per tier.

Canonical `.vrta_cel2d` Binary Layout (Little-Endian, Total = 11,230 B exact):
  Header (`4,096 B` page-aligned, naturally 4-byte aligned, C++20 POD-compatible):
      0..7   (8 B): char     magic[8]           = "VRTACEL2"
      8..11  (4 B): uint32_t version            = 1
     12..15  (4 B): uint32_t num_tiers          = 3
     16..23  (8 B): uint32_t dims[2]            = {41, 29}
     24..31  (8 B): float    lo[2]              = {-2.0, -12.0}
     32..39  (8 B): float    hi[2]              = {58.0,   4.0}
     40..43  (4 B): float    scale              = 100.0 / 65534.0 (~0.001526 m)
     44..47  (4 B): float    offset             = 10.0            (m)
     48..59 (12 B): float    tw_tiers[3]        = {1.25, 1.20, 1.15}
     60..71 (12 B): float    v_term_thresh[3]   = {358.4, 348.0, 338.0}
     72..83 (12 B): float    a_net_tiers[3]     = {1.9516625, 1.46133, 0.9709975}
     84..95 (12 B): float    eps_grid[3]        = per-tier conservative margin [m]
     96..127(32 B): uint8_t  payload_sha256[32] = SHA-256 of raw int16 payload
    128..131 (4 B): float    v_td_max           = 2.0
    132..135 (4 B): float    d_z_max            = 0.50
    136..4095:      uint8_t  reserved[3960]     = 0x00
  Payload (`7,134 B` = 6.967 KiB = 3 * 41 * 29 * 2 B):
    C-contiguous `<i2` (little-endian int16), innermost axis v_z (N1 = 29).
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
    DEFAULT_A_Z_MIN,
    DEFAULT_GRAVITY_M_S2,
    load_vehicle_params,
)

CEL2D_MAGIC = b"VRTACEL2"
CEL2D_VERSION = 1
CEL2D_HEADER_BYTES = 4096
CEL2D_NUM_TIERS = 3
CEL2D_EXPORT_SHAPE = (41, 29)
CEL2D_FINE_SHAPE = (81, 57)  # Exact 2:1 nested over (41, 29) matching 4D LUT nesting
CEL2D_ELEMENTS_PER_TIER = 41 * 29  # 1,189
CEL2D_PAYLOAD_ELEMENTS = CEL2D_NUM_TIERS * CEL2D_ELEMENTS_PER_TIER  # 3,567
CEL2D_PAYLOAD_BYTES = CEL2D_PAYLOAD_ELEMENTS * 2  # 7,134 B (6.967 KiB)
CEL2D_TOTAL_BYTES = CEL2D_HEADER_BYTES + CEL2D_PAYLOAD_BYTES  # 11,230 B exact

CEL2D_LO = (-2.0, -12.0)  # (z_lo_m, vz_lo_mps)
CEL2D_HI = (58.0, 4.0)  # (z_hi_m, vz_hi_mps)
CEL2D_V_LO = -40.0
CEL2D_V_HI = 60.0
CEL2D_SCALE = np.float32((CEL2D_V_HI - CEL2D_V_LO) / 65534.0)
CEL2D_OFFSET = np.float32(0.5 * (CEL2D_V_HI + CEL2D_V_LO))
CEL2D_V_TD_MAX = 2.0
CEL2D_D_Z_MAX = 0.50
CEL2D_TAU_S = 1.0

CEL2D_HEADER_STRUCT = struct.Struct("<8sII2I2f2f2f3f3f3f3f32s2f")
assert CEL2D_HEADER_STRUCT.size == 136


@dataclass(frozen=True)
class CelTierSpec:
    """Voltage-sag vertical braking tier for SAME_ARM_DUAL_LOSS CEL (DR-2 Table 2)."""

    index: int
    name: str
    tw_ratio: float
    v_term_min_v: float
    g: float = DEFAULT_GRAVITY_M_S2
    d_z_max: float = CEL2D_D_Z_MAX
    a_z_min: float = DEFAULT_A_Z_MIN
    v_td_max: float = CEL2D_V_TD_MAX

    @property
    def a_z_max(self) -> float:
        return (self.tw_ratio - 1.0) * self.g

    @property
    def a_net(self) -> float:
        return self.a_z_max - self.d_z_max


DEFAULT_CEL_TIERS: tuple[CelTierSpec, ...] = (
    CelTierSpec(index=0, name="Nominal-Same-Arm", tw_ratio=1.25, v_term_min_v=358.4),
    CelTierSpec(index=1, name="Moderate-Sag", tw_ratio=1.20, v_term_min_v=348.0),
    CelTierSpec(index=2, name="Severe-Sag", tw_ratio=1.15, v_term_min_v=338.0),
)


@dataclass(frozen=True)
class VrtaCel2DHeader:
    """Parsed `.vrta_cel2d` 4096 B header."""

    magic: bytes
    version: int
    num_tiers: int
    dims: tuple[int, int]
    lo: tuple[float, float]
    hi: tuple[float, float]
    scale: float
    offset: float
    tw_tiers: tuple[float, float, float]
    v_term_thresh: tuple[float, float, float]
    a_net_tiers: tuple[float, float, float]
    eps_grid: tuple[float, float, float]
    payload_sha256: bytes
    v_td_max: float
    d_z_max: float


def pack_cel2d_header(hdr: VrtaCel2DHeader) -> bytes:
    """Serialize VrtaCel2DHeader into exact 4,096 B zero-padded little-endian bytes."""
    raw = CEL2D_HEADER_STRUCT.pack(
        hdr.magic,
        hdr.version,
        hdr.num_tiers,
        *hdr.dims,
        *hdr.lo,
        *hdr.hi,
        hdr.scale,
        hdr.offset,
        *hdr.tw_tiers,
        *hdr.v_term_thresh,
        *hdr.a_net_tiers,
        *hdr.eps_grid,
        hdr.payload_sha256,
        hdr.v_td_max,
        hdr.d_z_max,
    )
    return raw.ljust(CEL2D_HEADER_BYTES, b"\x00")


def unpack_cel2d_header(buf: bytes) -> VrtaCel2DHeader:
    """Parse and validate a 4,096 B `.vrta_cel2d` header."""
    if len(buf) < CEL2D_HEADER_BYTES:
        raise ValueError(f"Header buffer too small: {len(buf)} < {CEL2D_HEADER_BYTES}")
    f = CEL2D_HEADER_STRUCT.unpack_from(buf, 0)
    magic = f[0]
    if magic != CEL2D_MAGIC:
        raise ValueError(f"Invalid .vrta_cel2d magic: {magic!r} != {CEL2D_MAGIC!r}")
    version = int(f[1])
    if version != CEL2D_VERSION:
        raise ValueError(f"Unsupported .vrta_cel2d version: {version}")
    return VrtaCel2DHeader(
        magic=magic,
        version=version,
        num_tiers=int(f[2]),
        dims=(int(f[3]), int(f[4])),
        lo=(float(f[5]), float(f[6])),
        hi=(float(f[7]), float(f[8])),
        scale=float(f[9]),
        offset=float(f[10]),
        tw_tiers=(float(f[11]), float(f[12]), float(f[13])),
        v_term_thresh=(float(f[14]), float(f[15]), float(f[16])),
        a_net_tiers=(float(f[17]), float(f[18]), float(f[19])),
        eps_grid=(float(f[20]), float(f[21]), float(f[22])),
        payload_sha256=bytes(f[23]),
        v_td_max=float(f[24]),
        d_z_max=float(f[25]),
    )


def load_cel_2d_lut(
    path: Path | str, *, verify_sha256: bool = True
) -> tuple[VrtaCel2DHeader, np.ndarray]:
    """Load `.vrta_cel2d`, verifying exact 11,230 B size, header, and SHA-256."""
    raw = Path(path).read_bytes()
    if len(raw) != CEL2D_TOTAL_BYTES:
        raise ValueError(
            f"Invalid .vrta_cel2d file size: {len(raw)} B != expected {CEL2D_TOTAL_BYTES} B"
        )
    hdr = unpack_cel2d_header(raw[:CEL2D_HEADER_BYTES])
    if hdr.num_tiers != CEL2D_NUM_TIERS or hdr.dims != CEL2D_EXPORT_SHAPE:
        raise ValueError(
            f"Unexpected .vrta_cel2d geometry: tiers={hdr.num_tiers}, dims={hdr.dims}"
        )
    payload = raw[CEL2D_HEADER_BYTES:]
    if verify_sha256:
        digest = hashlib.sha256(payload).digest()
        if digest != hdr.payload_sha256:
            raise ValueError(
                f"SHA-256 mismatch in {path}: "
                f"computed {digest.hex()} != header {hdr.payload_sha256.hex()}"
            )
    arr = np.frombuffer(payload, dtype="<i2").reshape((hdr.num_tiers, *hdr.dims))
    return hdr, arr


def analytic_cel_z_req(
    vz: float, a_net: float, v_td_max: float = CEL2D_V_TD_MAX
) -> float:
    """Analytic minimum AGL braking altitude z_req = (v_z^2 - v_td_max^2) / (2 * a_net)."""
    if vz >= -v_td_max:
        return 0.0
    return float((vz * vz - v_td_max * v_td_max) / (2.0 * a_net))


def select_cel_tier(hdr: VrtaCel2DHeader, v_term_v: float) -> int:
    """Map terminal battery voltage V_term to sag tier index {0, 1, 2}."""
    if v_term_v >= hdr.v_term_thresh[0]:
        return 0
    if v_term_v >= hdr.v_term_thresh[1]:
        return 1
    return 2


def eval_cel_2d(
    hdr: VrtaCel2DHeader,
    data: np.ndarray,
    tier_idx: int,
    z_agl: float,
    vz_up: float,
    *,
    subtract_eps: bool = True,
) -> tuple[float, tuple[float, float]]:
    """Bilinear lookup of V_2D(z, v_z) and analytic gradient (dV/dz, dV/dv_z)."""
    if not (0 <= tier_idx < hdr.num_tiers):
        raise ValueError(f"tier_idx {tier_idx} out of bounds [0, {hdr.num_tiers - 1}]")
    if not (math.isfinite(z_agl) and math.isfinite(vz_up)):
        raise ValueError(f"Non-finite CEL 2D query: z={z_agl}, vz={vz_up}")

    inv_dz = (hdr.dims[0] - 1) / (hdr.hi[0] - hdr.lo[0])
    inv_dvz = (hdr.dims[1] - 1) / (hdr.hi[1] - hdr.lo[1])

    uz = min(max((z_agl - hdr.lo[0]) * inv_dz, 0.0), float(hdr.dims[0] - 1) - 1e-9)
    uv = min(max((vz_up - hdr.lo[1]) * inv_dvz, 0.0), float(hdr.dims[1] - 1) - 1e-9)
    iz, iv = int(uz), int(uv)
    fz, fv = uz - float(iz), uv - float(iv)
    oz, ov = 1.0 - fz, 1.0 - fv

    scale = float(hdr.scale)
    offset = float(hdr.offset)
    c00 = float(data[tier_idx, iz, iv]) * scale + offset
    c01 = float(data[tier_idx, iz, iv + 1]) * scale + offset
    c10 = float(data[tier_idx, iz + 1, iv]) * scale + offset
    c11 = float(data[tier_idx, iz + 1, iv + 1]) * scale + offset

    a0 = c00 * ov + c01 * fv
    a1 = c10 * ov + c11 * fv
    val = a0 * oz + a1 * fz
    if subtract_eps:
        val -= float(hdr.eps_grid[tier_idx])
    dv_dz = (a1 - a0) * inv_dz
    dv_dvz = ((c01 - c00) * oz + (c11 - c10) * fz) * inv_dvz
    return float(val), (float(dv_dz), float(dv_dvz))


def find_cel_zero_level_z(
    hdr: VrtaCel2DHeader, data: np.ndarray, tier_idx: int, vz_up: float
) -> float:
    """Locate the zero-level braking altitude z where V_2D(z, vz_up) == 0 via bisection."""
    z_lo, z_hi = hdr.lo[0], hdr.hi[0]
    v_lo, _ = eval_cel_2d(hdr, data, tier_idx, z_lo, vz_up, subtract_eps=True)
    v_hi, _ = eval_cel_2d(hdr, data, tier_idx, z_hi, vz_up, subtract_eps=True)
    if v_lo >= 0.0:
        return z_lo
    if v_hi <= 0.0:
        return z_hi
    a, b = z_lo, z_hi
    for _ in range(48):
        mid = 0.5 * (a + b)
        vm, _ = eval_cel_2d(hdr, data, tier_idx, mid, vz_up, subtract_eps=True)
        if vm < 0.0:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def _pad_linear_extrap(arr: np.ndarray, pad: int, axis: int) -> np.ndarray:
    """Linearly extrapolate `pad` ghost cells at both ends along `axis`."""
    arr = np.moveaxis(arr, axis, 0)
    n = arr.shape[0]
    out = np.empty((n + 2 * pad, *arr.shape[1:]), dtype=arr.dtype)
    out[pad : pad + n] = arr
    slope_lo = arr[1] - arr[0]
    slope_hi = arr[-1] - arr[-2]
    for k in range(1, pad + 1):
        out[pad - k] = arr[0] - k * slope_lo
        out[pad + n - 1 + k] = arr[-1] + k * slope_hi
    return np.moveaxis(out, 0, axis)


def _weno5_derivatives_1d(u: np.ndarray, dx: float, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute left-biased (D-) and right-biased (D+) WENO5 spatial derivatives."""
    padded = _pad_linear_extrap(u, pad=3, axis=axis)
    p = np.moveaxis(padded, axis, 0)
    # First differences d[i] = (p[i+1] - p[i]) / dx, length N + 5
    d = (p[1:] - p[:-1]) / dx
    n = u.shape[axis]
    eps = 1e-6

    # Left-biased D- uses stencils ending at index i (d indices: i, i+1, i+2, i+3, i+4)
    v1 = d[0:n]
    v2 = d[1 : n + 1]
    v3 = d[2 : n + 2]
    v4 = d[3 : n + 3]
    v5 = d[4 : n + 4]

    s1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2 + 0.25 * (v1 - 4.0 * v2 + 3.0 * v3) ** 2
    s2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
    s3 = (13.0 / 12.0) * (v3 - 2.0 * v4 + v5) ** 2 + 0.25 * (3.0 * v3 - 4.0 * v4 + v5) ** 2
    a1 = 0.1 / (eps + s1) ** 2
    a2 = 0.6 / (eps + s2) ** 2
    a3 = 0.3 / (eps + s3) ** 2
    a_sum = a1 + a2 + a3
    w1, w2, w3 = a1 / a_sum, a2 / a_sum, a3 / a_sum
    dm = (
        w1 * ((1.0 / 3.0) * v1 - (7.0 / 6.0) * v2 + (11.0 / 6.0) * v3)
        + w2 * ((-1.0 / 6.0) * v2 + (5.0 / 6.0) * v3 + (1.0 / 3.0) * v4)
        + w3 * ((1.0 / 3.0) * v3 + (5.0 / 6.0) * v4 - (1.0 / 6.0) * v5)
    )

    # Right-biased D+ uses reversed stencil starting at i+1 (d indices: i+5, i+4, i+3, i+2, i+1)
    u1 = d[5 : n + 5]
    u2 = d[4 : n + 4]
    u3 = d[3 : n + 3]
    u4 = d[2 : n + 2]
    u5 = d[1 : n + 1]

    t1 = (13.0 / 12.0) * (u1 - 2.0 * u2 + u3) ** 2 + 0.25 * (u1 - 4.0 * u2 + 3.0 * u3) ** 2
    t2 = (13.0 / 12.0) * (u2 - 2.0 * u3 + u4) ** 2 + 0.25 * (u2 - u4) ** 2
    t3 = (13.0 / 12.0) * (u3 - 2.0 * u4 + u5) ** 2 + 0.25 * (3.0 * u3 - 4.0 * u4 + u5) ** 2
    b1 = 0.1 / (eps + t1) ** 2
    b2 = 0.6 / (eps + t2) ** 2
    b3 = 0.3 / (eps + t3) ** 2
    b_sum = b1 + b2 + b3
    r1, r2, r3 = b1 / b_sum, b2 / b_sum, b3 / b_sum
    dp = (
        r1 * ((1.0 / 3.0) * u1 - (7.0 / 6.0) * u2 + (11.0 / 6.0) * u3)
        + r2 * ((-1.0 / 6.0) * u2 + (5.0 / 6.0) * u3 + (1.0 / 3.0) * u4)
        + r3 * ((1.0 / 3.0) * u3 + (5.0 / 6.0) * u4 - (1.0 / 6.0) * u5)
    )

    return np.moveaxis(dm, 0, axis), np.moveaxis(dp, 0, axis)


def _solve_cel_tier_hji_vi_pde(
    tier: CelTierSpec,
    *,
    tau_s: float | None = None,
    fine_shape: tuple[int, int] = CEL2D_FINE_SHAPE,
    horizon_s: float | None = None,
) -> tuple[np.ndarray, float]:
    """Solve 2D CEL HJI-VI PDE via WENO5 + Godunov upwind + TVD-RK2 on fine_shape.

    Uses the slope-1 viability terminal function
        l_2D(z, v_z) = z + tau_s * max(0, v_z + v_td_max)
    with shared tau_s = v_td_max / min_tier(a_net), which guarantees:
      - dV/dz == 1.0 everywhere (zero slope kink across V == 0),
      - H_2D(z, v_z, grad V) == 0 on v_z < -v_td_max and >= 0 on v_z >= -v_td_max,
      - exact zero-level set z == (v_z^2 - v_td_max^2) / (2 * a_net) for v_z < -v_td_max,
      - global tier monotonicity V_tier0 >= V_tier1 >= V_tier2.

    Stride-decimates cleanly to (41, 29) int16 and audits exact fine-grid eps_grid.
    """
    nz_f, nv_f = fine_shape
    nz_e, nv_e = CEL2D_EXPORT_SHAPE
    if (nz_f - 1) % (nz_e - 1) != 0 or (nv_f - 1) % (nv_e - 1) != 0:
        raise ValueError(f"fine_shape {fine_shape} must nest cleanly over {CEL2D_EXPORT_SHAPE}")
    stride_z = (nz_f - 1) // (nz_e - 1)
    stride_v = (nv_f - 1) // (nv_e - 1)

    z_fine = np.linspace(CEL2D_LO[0], CEL2D_HI[0], nz_f, dtype=np.float64)
    vz_fine = np.linspace(CEL2D_LO[1], CEL2D_HI[1], nv_f, dtype=np.float64)
    dz = float(z_fine[1] - z_fine[0])
    dvz = float(vz_fine[1] - vz_fine[0])

    tau_val = float(tau_s) if tau_s is not None else float(tier.v_td_max / max(tier.a_net, 0.25))
    Z, VZ = np.meshgrid(z_fine, vz_fine, indexing="ij")
    l_init = Z + tau_val * np.maximum(0.0, VZ + tier.v_td_max)

    a_up_net = tier.a_net  # > 0 upward braking against downward gust
    a_dn_net = tier.a_z_min + tier.d_z_max  # < 0 downward accel against upward gust

    # Required horizon so the slowest-braking state (v_z = -12 -> -2) reaches fixed point
    t_req = (abs(CEL2D_LO[1]) - tier.v_td_max) / max(a_up_net, 0.25) + 0.5
    t_horizon = float(horizon_s) if horizon_s is not None else t_req

    # Separable Godunov upwind CFL: z-speed is |v_z| <= 12, active upward braking is a_up_net
    # (downward branch a_dn_net is only active if D-_v < 0; include full bound for strict CFL)
    max_rate = float(np.max(np.abs(VZ)) / dz + max(abs(a_up_net), abs(a_dn_net)) / dvz)
    dt_cfl = 0.80 / max_rate
    n_steps = max(1, int(math.ceil(t_horizon / dt_cfl)))
    dt = t_horizon / n_steps
    vz_pos = np.maximum(VZ, 0.0)
    vz_neg = np.minimum(VZ, 0.0)

    def _rhs(v_grid: np.ndarray) -> np.ndarray:
        dz_m, dz_p = _weno5_derivatives_1d(v_grid, dz, axis=0)
        dv_m, dv_p = _weno5_derivatives_1d(v_grid, dvz, axis=1)
        # Exact monotone Godunov upwind Hamiltonian for separable H_2D:
        #   H_z(pz) = vz * pz  -> vz+ * D+_z + vz- * D-_z
        #   H_v(pv) = max(a_up_net * pv, a_dn_net * pv)
        #           -> max(a_up_net * max(D+_v, 0), a_dn_net * min(D-_v, 0))
        hz = vz_pos * dz_p + vz_neg * dz_m
        hv = np.maximum(a_up_net * np.maximum(dv_p, 0.0), a_dn_net * np.minimum(dv_m, 0.0))
        return hz + hv

    V = l_init.copy()
    for _ in range(n_steps):
        # TVD-RK2 (Heun) with backwards_reachable_tube min(..., l_init) projection
        v1 = np.minimum(l_init, V + dt * _rhs(V))
        V = np.minimum(l_init, 0.5 * V + 0.5 * (v1 + dt * _rhs(v1)))

    # Exact stride decimation to 41 x 29 int16
    v_exp = V[::stride_z, ::stride_v].astype(np.float32)
    vc = np.clip(v_exp, np.float32(CEL2D_V_LO), np.float32(CEL2D_V_HI))
    vq = np.clip(np.rint((vc - CEL2D_OFFSET) / CEL2D_SCALE), -32767, 32767).astype("<i2")
    v_deq = vq.astype(np.float64) * float(CEL2D_SCALE) + float(CEL2D_OFFSET)

    # Exact bilinear interpolation of quantized export LUT back onto fine_shape for eps_grid audit
    z_exp = np.linspace(CEL2D_LO[0], CEL2D_HI[0], nz_e, dtype=np.float64)
    vz_exp = np.linspace(CEL2D_LO[1], CEL2D_HI[1], nv_e, dtype=np.float64)
    iz_idx = np.clip(np.searchsorted(z_exp, z_fine, side="right") - 1, 0, nz_e - 2)
    iv_idx = np.clip(np.searchsorted(vz_exp, vz_fine, side="right") - 1, 0, nv_e - 2)
    fz = (z_fine - z_exp[iz_idx]) / (z_exp[iz_idx + 1] - z_exp[iz_idx])
    fv = (vz_fine - vz_exp[iv_idx]) / (vz_exp[iv_idx + 1] - vz_exp[iv_idx])

    c00 = v_deq[np.ix_(iz_idx, iv_idx)]
    c01 = v_deq[np.ix_(iz_idx, iv_idx + 1)]
    c10 = v_deq[np.ix_(iz_idx + 1, iv_idx)]
    c11 = v_deq[np.ix_(iz_idx + 1, iv_idx + 1)]
    v_hat_fine = (
        (c00 * (1.0 - fv)[None, :] + c01 * fv[None, :]) * (1.0 - fz)[:, None]
        + (c10 * (1.0 - fv)[None, :] + c11 * fv[None, :]) * fz[:, None]
    )

    v_fine_c = np.clip(V, CEL2D_V_LO, CEL2D_V_HI)
    band = np.abs(v_fine_c) <= 10.0
    eps_max = float(np.max(np.where(band, np.maximum(0.0, v_hat_fine - v_fine_c), 0.0)))
    eps_grid_tier = eps_max + 0.5 * float(CEL2D_SCALE)
    return vq, float(eps_grid_tier)


def build_cel_tiers_from_config(
    config_path: Path | str | None = None,
) -> tuple[CelTierSpec, ...]:
    """Build the 3 CEL sag tiers aligned with config/vehicle_x8_240.yaml gravity."""
    params = load_vehicle_params(config_path)
    g = params["gravity_m_s2"]
    v_min = params["v_min_tw112_v"]
    return (
        CelTierSpec(index=0, name="Nominal-Same-Arm", tw_ratio=1.25, v_term_min_v=358.4, g=g),
        CelTierSpec(index=1, name="Moderate-Sag", tw_ratio=1.20, v_term_min_v=348.0, g=g),
        CelTierSpec(index=2, name="Severe-Sag", tw_ratio=1.15, v_term_min_v=v_min, g=g),
    )


def solve_and_export_cel_2d(
    out_path: Path | str,
    *,
    config_path: Path | str | None = None,
    fine_shape: tuple[int, int] = CEL2D_FINE_SHAPE,
) -> VrtaCel2DHeader:
    """Solve all 3 CEL sag tiers and export exact 11,230 B `.vrta_cel2d` binary."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tiers = build_cel_tiers_from_config(config_path)
    shared_tau_s = float(CEL2D_V_TD_MAX / min(t.a_net for t in tiers))

    payload = bytearray()
    eps_list: list[float] = []
    for tier in tiers:
        vq, eps_tier = _solve_cel_tier_hji_vi_pde(
            tier, tau_s=shared_tau_s, fine_shape=fine_shape
        )
        if vq.shape != CEL2D_EXPORT_SHAPE:
            raise RuntimeError(f"Unexpected CEL grid shape {vq.shape} != {CEL2D_EXPORT_SHAPE}")
        eps_list.append(eps_tier)
        payload.extend(np.ascontiguousarray(vq, dtype="<i2").tobytes(order="C"))

    if len(payload) != CEL2D_PAYLOAD_BYTES:
        raise RuntimeError(
            f"Internal error: CEL payload {len(payload)} B != {CEL2D_PAYLOAD_BYTES} B"
        )

    sha256 = hashlib.sha256(payload).digest()
    hdr = VrtaCel2DHeader(
        magic=CEL2D_MAGIC,
        version=CEL2D_VERSION,
        num_tiers=CEL2D_NUM_TIERS,
        dims=CEL2D_EXPORT_SHAPE,
        lo=CEL2D_LO,
        hi=CEL2D_HI,
        scale=float(CEL2D_SCALE),
        offset=float(CEL2D_OFFSET),
        tw_tiers=tuple(t.tw_ratio for t in tiers),  # type: ignore[arg-type]
        v_term_thresh=tuple(t.v_term_min_v for t in tiers),  # type: ignore[arg-type]
        a_net_tiers=tuple(t.a_net for t in tiers),  # type: ignore[arg-type]
        eps_grid=tuple(eps_list),  # type: ignore[arg-type]
        payload_sha256=sha256,
        v_td_max=float(CEL2D_V_TD_MAX),
        d_z_max=float(CEL2D_D_Z_MAX),
    )
    hdr_bytes = pack_cel2d_header(hdr)
    assert len(hdr_bytes) == CEL2D_HEADER_BYTES

    with out.open("wb") as f:
        f.write(hdr_bytes)
        f.write(payload)

    assert out.stat().st_size == CEL2D_TOTAL_BYTES
    return hdr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VertiRTA 2D CEL Vertical Braking LUT Solver & Exporter (.vrta_cel2d)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/quick_test.vrta_cel2d"),
        help="Output .vrta_cel2d file path (exact 11,230 B).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to config/vehicle_x8_240.yaml.",
    )
    args = parser.parse_args(argv)

    hdr = solve_and_export_cel_2d(args.out, config_path=args.config)
    print(
        f"[solve_cel_2d_lut] Exported {args.out} ({args.out.stat().st_size:,} B, "
        f"payload_sha256={hdr.payload_sha256.hex()})"
    )
    print(f"  a_net [m/s^2] : {[round(a, 4) for a in hdr.a_net_tiers]}")
    print(f"  eps_grid [m]  : {[round(e, 5) for e in hdr.eps_grid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
