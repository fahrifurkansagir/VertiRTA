"""4D Fault-Degraded CBVF Relative eVTOL Dynamics & DR-2 Errata Locks.

Implements the coupled thrust-sphere & tilt-cone 4D relative eVTOL dynamics
x = [r, v_r, z, v_z]^T for StanfordASL/hj_reachability v0.7.0 (Colab G4 Blackwell
FP32) AND pure-NumPy CPU testing without requiring a GPU or JAX install.

Enforces DR-2 Forensic Errata Locks:
  - Erratum #1: partial_max_magnitudes horizontal acceleration bound matches
    _support_U (a_T * sin(tilt_max) ~= 14.06 m/s^2 for Nominal, NOT hover-only
    g * tan(tilt_max) ~= 6.87 m/s^2) so Lax-Friedrichs CFL dissipation is sound.
  - Erratum #2: conservative_smax_4 subtracts ln(2) / beta from log-sum-exp so
    smax_cons(sr, sz) <= max(sr, sz) everywhere (0.0 at (0, 0), not +0.173 m).
  - Erratum #3: compute_degraded_v_cap computes per-regime cruise speed cap
    v_cap_f so worst-case braking under d_ar=1.75 m/s^2 fits inside the BRT
    horizon (T=6.0 s) and radial grid margin (r_max - R_safe = 55.0 m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional fallback if PyYAML absent
    yaml = None  # type: ignore[assignment]

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)  # Native FP32 path for Blackwell
    HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    HAS_JAX = False

try:
    import hj_reachability as hj

    HAS_HJ_REACHABILITY = True
    _DynamicsBase = hj.Dynamics
except ImportError:
    hj = None  # type: ignore[assignment]
    HAS_HJ_REACHABILITY = False

    class _DynamicsBase:  # type: ignore[no-redef]
        """Lightweight CPU stand-in matching hj_reachability.Dynamics signature."""

        def __init__(
            self,
            control_mode: str = "max",
            disturbance_mode: str = "min",
            control_space: Any = None,
            disturbance_space: Any = None,
        ) -> None:
            self.control_mode = control_mode
            self.disturbance_mode = disturbance_mode
            self.control_space = control_space
            self.disturbance_space = disturbance_space


DEFAULT_VEHICLE_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "vehicle_x8_240.yaml"
)

# Frozen DR-2 §1.1 & Spec v1.2 vehicle / aerodynamic / disturbance constants
DEFAULT_MASS_KG = 240.0
DEFAULT_GRAVITY_M_S2 = 9.80665
DEFAULT_TW_NOMINAL = 2.50
DEFAULT_V_MAX_MPS = 22.0
DEFAULT_R_SAFE_M = 3.0
DEFAULT_Z_SAFE_M = 2.0
DEFAULT_A_Z_MIN = -6.0
DEFAULT_D_VR = 0.35
DEFAULT_D_VZ = 0.25
DEFAULT_D_AR = 1.75
DEFAULT_D_AZ = 0.75
DEFAULT_CD_R = 0.00408
DEFAULT_CD_Z = 0.00613


def _xp(*args: Any) -> Any:
    """Select jax.numpy when any argument is a JAX array/tracer, else numpy."""
    if HAS_JAX and jax is not None and jnp is not None:
        jax_array_cls = getattr(jax, "Array", ())
        jax_tracer_cls = getattr(getattr(jax, "core", None), "Tracer", ())
        for arg in args:
            if arg is None:
                continue
            if (jax_array_cls and isinstance(arg, jax_array_cls)) or (
                jax_tracer_cls and isinstance(arg, jax_tracer_cls)
            ):
                return jnp
            if type(arg).__module__.startswith("jax"):
                return jnp
    return np


def load_vehicle_params(config_path: Path | str | None = None) -> dict[str, float]:
    """Load frozen SI vehicle parameters from config/vehicle_x8_240.yaml."""
    path = Path(config_path) if config_path is not None else DEFAULT_VEHICLE_CONFIG_PATH
    params: dict[str, float] = {
        "mass_kg": DEFAULT_MASS_KG,
        "gravity_m_s2": DEFAULT_GRAVITY_M_S2,
        "tw_nominal": DEFAULT_TW_NOMINAL,
        "v_max_mps": DEFAULT_V_MAX_MPS,
        "v_min_tw112_v": 338.0,
        "ke_bus_vs_rad": 1.36,
    }
    if yaml is not None and path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            if "mass_kg" in raw:
                params["mass_kg"] = float(raw["mass_kg"])
            if "gravity_m_s2" in raw:
                params["gravity_m_s2"] = float(raw["gravity_m_s2"])
            if "tw_nominal" in raw:
                params["tw_nominal"] = float(raw["tw_nominal"])
            limits = raw.get("limits", {})
            if isinstance(limits, dict) and "v_max_mps" in limits:
                params["v_max_mps"] = float(limits["v_max_mps"])
            battery = raw.get("battery", {})
            if isinstance(battery, dict) and "v_min_tw112_v" in battery:
                params["v_min_tw112_v"] = float(battery["v_min_tw112_v"])
            prop = raw.get("propulsion", {})
            if isinstance(prop, dict) and "ke_bus_vs_rad" in prop:
                params["ke_bus_vs_rad"] = float(prop["ke_bus_vs_rad"])
    return params


@dataclass(frozen=True)
class RegimeSpec:
    """Offline CSFL fault-degraded flight envelope specification (DR-2 Table 1)."""

    index: int
    name: str
    tw_ratio: float
    tilt_max_deg: float
    a_z_min: float = DEFAULT_A_Z_MIN
    g: float = DEFAULT_GRAVITY_M_S2
    d_vr: float = DEFAULT_D_VR
    d_vz: float = DEFAULT_D_VZ
    d_ar: float = DEFAULT_D_AR
    d_az: float = DEFAULT_D_AZ
    cd_r: float = DEFAULT_CD_R
    cd_z: float = DEFAULT_CD_Z
    r_safe: float = DEFAULT_R_SAFE_M
    z_safe: float = DEFAULT_Z_SAFE_M

    @property
    def a_thrust_max(self) -> float:
        return self.tw_ratio * self.g

    @property
    def tilt_max_rad(self) -> float:
        return math.radians(self.tilt_max_deg)

    @property
    def a_xy_hover_max(self) -> float:
        """Level-flight horizontal acceleration limit at hover thrust F_z = g."""
        return min(
            self.g * math.tan(self.tilt_max_rad),
            self.a_thrust_max * math.sin(self.tilt_max_rad),
        )

    @property
    def a_z_up_max(self) -> float:
        """Maximum vertical upward acceleration at zero tilt."""
        return self.a_thrust_max - self.g


DEFAULT_REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec(index=0, name="Nominal", tw_ratio=2.50, tilt_max_deg=35.0),
    RegimeSpec(index=1, name="1M", tw_ratio=2.10, tilt_max_deg=28.0),
    RegimeSpec(index=2, name="2M-Cross", tw_ratio=1.80, tilt_max_deg=22.0),
    RegimeSpec(index=3, name="Low-Vbat", tw_ratio=1.85, tilt_max_deg=22.0),
)

_REGIME_BY_NAME: dict[str, RegimeSpec] = {
    spec.name.lower(): spec for spec in DEFAULT_REGIMES
}


def build_regimes_from_config(
    config_path: Path | str | None = None,
) -> tuple[RegimeSpec, ...]:
    """Build the 4 CSFL RegimeSpecs synchronized to config/vehicle_x8_240.yaml."""
    params = load_vehicle_params(config_path)
    g = params["gravity_m_s2"]
    tw_scale = params["tw_nominal"] / DEFAULT_TW_NOMINAL
    return tuple(
        RegimeSpec(
            index=spec.index,
            name=spec.name,
            tw_ratio=spec.tw_ratio * tw_scale,
            tilt_max_deg=spec.tilt_max_deg,
            a_z_min=spec.a_z_min,
            g=g,
            d_vr=spec.d_vr,
            d_vz=spec.d_vz,
            d_ar=spec.d_ar,
            d_az=spec.d_az,
            cd_r=spec.cd_r,
            cd_z=spec.cd_z,
            r_safe=spec.r_safe,
            z_safe=spec.z_safe,
        )
        for spec in DEFAULT_REGIMES
    )


def resolve_regime_spec(
    regime: str | int | RegimeSpec,
    regimes: tuple[RegimeSpec, ...] = DEFAULT_REGIMES,
) -> RegimeSpec:
    """Resolve a regime by name, index (0..3), or RegimeSpec instance."""
    if isinstance(regime, RegimeSpec):
        return regime
    if isinstance(regime, int):
        if 0 <= regime < len(regimes):
            return regimes[regime]
        raise ValueError(f"Invalid regime index {regime}; expected 0..{len(regimes) - 1}")
    key = regime.strip().lower()
    for spec in regimes:
        if spec.name.lower() == key:
            return spec
    if key in _REGIME_BY_NAME:
        return _REGIME_BY_NAME[key]
    raise ValueError(f"Unknown CSFL regime '{regime}'")


def conservative_smax_4(sr: Any, sz: Any, beta: float = 4.0) -> Any:
    """Conservative smooth maximum for cylindrical safety signed distance (DR-2 Erratum #2).

    Raw log-sum-exp smax_beta(sr, sz) overestimates max(sr, sz) by up to +ln(2)/beta
    (+0.173 m at sr == sz == 0 for beta = 4.0). Subtracting ln(2)/beta guarantees:

        max(sr, sz) - ln(2)/beta <= conservative_smax_4(sr, sz) <= max(sr, sz)

    with exact equality at the cylinder ridge sr == sz (0.0 at (0.0, 0.0)).
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}")
    xp = _xp(sr, sz)
    if xp is np and isinstance(sr, (int, float)) and isinstance(sz, (int, float)):
        m = max(float(sr), float(sz))
        diff = abs(float(sr) - float(sz))
        return m + (math.log1p(math.exp(-beta * diff)) - math.log(2.0)) / beta

    dtype = xp.result_type(sr, sz, xp.float32)
    sr_a = xp.asarray(sr, dtype=dtype)
    sz_a = xp.asarray(sz, dtype=dtype)
    m = xp.maximum(sr_a, sz_a)
    diff = xp.abs(sr_a - sz_a)
    beta_t = xp.asarray(beta, dtype=dtype)
    ln2_t = xp.asarray(math.log(2.0), dtype=dtype)
    return m + (xp.log1p(xp.exp(-beta_t * diff)) - ln2_t) / beta_t


def compute_degraded_v_cap(
    regime: str | int | RegimeSpec,
    d_ar: float = DEFAULT_D_AR,
    r_max: float = 58.0,
    horizon_s: float = 6.0,
    *,
    d_vr: float = DEFAULT_D_VR,
    r_safe: float = DEFAULT_R_SAFE_M,
    v_max: float = DEFAULT_V_MAX_MPS,
    g: float | None = None,
    r_vis: float | None = None,
) -> float:
    """Compute degraded cruise speed cap v_cap_f inside the 4D BRT (DR-2 Erratum #3).

    In 2M-Cross / Low-Vbat (22 deg max tilt -> a_xy,max = 3.96 m/s^2), braking from
    22 m/s under worst-case d_ar = 1.75 m/s^2 takes 9.95 s / 109.5 m (> T=6.0 s and
    r_max=58 m). This function returns the highest cruise speed whose worst-case
    level-flight stopping maneuver completes strictly within both the BRT horizon
    (horizon_s) and radial grid margin (r_max - r_safe, plus optional r_vis - r_safe).
    """
    spec = resolve_regime_spec(regime)
    g_val = spec.g if g is None else float(g)
    a_xy_max = min(
        g_val * math.tan(spec.tilt_max_rad),
        spec.tw_ratio * g_val * math.sin(spec.tilt_max_rad),
    )
    a_eff = a_xy_max - float(d_ar)
    if a_eff <= 0.0:
        return 0.0

    # 1) Time-horizon cap: (v + d_vr) / a_eff <= horizon_s
    v_cap_time = max(0.0, a_eff * float(horizon_s) - float(d_vr))

    # 2) Radial grid braking-distance cap: v^2/(2*a_eff) + d_vr*(v/a_eff) <= r_avail
    r_avail = max(0.0, float(r_max) - float(r_safe))
    v_cap_dist = max(0.0, math.sqrt(d_vr * d_vr + 2.0 * a_eff * r_avail) - float(d_vr))

    v_cap = min(float(v_max), v_cap_time, v_cap_dist)

    if r_vis is not None:
        r_vis_avail = max(0.0, float(r_vis) - float(r_safe))
        v_cap_vis = max(
            0.0, math.sqrt(d_vr * d_vr + 2.0 * a_eff * r_vis_avail) - float(d_vr)
        )
        v_cap = min(v_cap, v_cap_vis)

    return float(v_cap)


def compute_all_regime_v_caps(
    regimes: tuple[RegimeSpec, ...] = DEFAULT_REGIMES,
    *,
    r_max: float = 58.0,
    horizon_s: float = 6.0,
    v_max: float = DEFAULT_V_MAX_MPS,
    r_vis: float | None = None,
) -> tuple[float, float, float, float]:
    """Compute v_cap_f[4] tuple for all 4 CSFL regimes."""
    caps = tuple(
        compute_degraded_v_cap(
            spec,
            d_ar=spec.d_ar,
            r_max=r_max,
            horizon_s=horizon_s,
            d_vr=spec.d_vr,
            r_safe=spec.r_safe,
            v_max=v_max,
            g=spec.g,
            r_vis=r_vis,
        )
        for spec in regimes
    )
    assert len(caps) == 4
    return caps  # type: ignore[return-value]


class VertiRta4DDynamics(_DynamicsBase):
    """Coupled thrust-sphere & tilt-cone 4D relative eVTOL dynamics (r, vr, z, vz).

    Control set:
        U_f = { (u_r, u_z) : sqrt(u_r^2 + (u_z + g)^2) <= a_T,
                             |u_r| <= (u_z + g) * tan(tilt_max),
                             u_z >= max(0.5 - g, a_z_min) }
    Disturbance box:
        D = { |d_vr| <= d_vr, |d_ar| <= d_ar, |d_vz| <= d_vz, |d_az| <= d_az }
    """

    def __init__(
        self,
        a_thrust_max: float,
        tilt_max_rad: float,
        a_z_min: float = DEFAULT_A_Z_MIN,
        d_vr: float = DEFAULT_D_VR,
        d_vz: float = DEFAULT_D_VZ,
        d_ar: float = DEFAULT_D_AR,
        d_az: float = DEFAULT_D_AZ,
        cd_r: float = DEFAULT_CD_R,
        cd_z: float = DEFAULT_CD_Z,
        g: float = DEFAULT_GRAVITY_M_S2,
    ) -> None:
        super().__init__(
            control_mode="max",
            disturbance_mode="min",
            control_space=None,
            disturbance_space=None,
        )
        if a_thrust_max <= 0.0:
            raise ValueError(f"a_thrust_max must be positive, got {a_thrust_max}")
        if not (0.0 < tilt_max_rad < 0.5 * math.pi):
            raise ValueError(f"tilt_max_rad must lie in (0, pi/2), got {tilt_max_rad}")

        self.a_T = float(a_thrust_max)
        self.tilt_max_rad = float(tilt_max_rad)
        self.tan_th = math.tan(self.tilt_max_rad)
        self.sin_th = math.sin(self.tilt_max_rad)
        self.cos_th = math.cos(self.tilt_max_rad)
        self.a_z_min = float(a_z_min)
        self.d_vr = float(d_vr)
        self.d_vz = float(d_vz)
        self.d_ar = float(d_ar)
        self.d_az = float(d_az)
        self.cd_r = float(cd_r)
        self.cd_z = float(cd_z)
        self.g = float(g)

        # Lower vertical specific-force floor & thrust-sphere rim coordinates
        self.fz_lo = max(0.5, self.g + self.a_z_min)
        if self.fz_lo > self.a_T:
            self.fz_lo = self.a_T
        fz_rim = self.a_T * self.cos_th
        if fz_rim >= self.fz_lo:
            self.fz_cap = fz_rim
            self.fr_cap = self.a_T * self.sin_th
        else:
            # Extreme tilt fallback where cone rim drops below Fz_lo
            self.fz_cap = self.fz_lo
            self.fr_cap = math.sqrt(max(0.0, self.a_T * self.a_T - self.fz_lo * self.fz_lo))
        self.fr_bot = min(
            self.fz_lo * self.tan_th,
            math.sqrt(max(0.0, self.a_T * self.a_T - self.fz_lo * self.fz_lo)),
        )

    @classmethod
    def from_regime_spec(cls, spec: RegimeSpec) -> VertiRta4DDynamics:
        """Construct VertiRta4DDynamics from a RegimeSpec."""
        return cls(
            a_thrust_max=spec.a_thrust_max,
            tilt_max_rad=spec.tilt_max_rad,
            a_z_min=spec.a_z_min,
            d_vr=spec.d_vr,
            d_vz=spec.d_vz,
            d_ar=spec.d_ar,
            d_az=spec.d_az,
            cd_r=spec.cd_r,
            cd_z=spec.cd_z,
            g=spec.g,
        )

    def max_achievable_ur(self) -> float:
        """Exact max |u_r| attainable in _support_U (a_T * sin(tilt_max) for Nominal = 14.062 m/s^2)."""
        return float(max(self.fr_cap, self.fr_bot))

    def max_achievable_uz_mag(self) -> float:
        """Exact max |u_z| attainable in _support_U."""
        return float(max(abs(self.a_T - self.g), abs(self.fz_lo - self.g)))

    def _support_U(self, pr: Any, pz: Any) -> Any:
        """Closed-form support function max_{(ur, uz) in U_f} (pr * ur + pz * uz)."""
        xp = _xp(pr, pz)
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
        """HJI-VI Hamiltonian H_f(x, p) = max_{u in U_f} min_{d in D} p^T f_f(x, u, d)."""
        xp = _xp(state, grad_value)
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
        """Lax-Friedrichs artificial dissipation bounds |dH/dp_i| (DR-2 Erratum #1 FIX).

        Uses max_achievable_ur() == a_T * sin(tilt_max) (14.062 m/s^2 for Nominal)
        so Lax-Friedrichs numerical Hamiltonian is strictly monotone and CFL-sound
        against the exact thrust-sphere/tilt-cone support function _support_U.
        """
        xp = _xp(state)
        _r, vr, _z, vz = state[0], state[1], state[2], state[3]
        a_r_max = self.max_achievable_ur()
        a_z_max = self.max_achievable_uz_mag()
        return xp.array(
            [
                xp.abs(vr) + self.d_vr,
                a_r_max + self.cd_r * vr * vr + self.d_ar,
                xp.abs(vz) + self.d_vz,
                a_z_max + self.cd_z * vz * vz + self.d_az,
            ],
            dtype=xp.float32,
        )
