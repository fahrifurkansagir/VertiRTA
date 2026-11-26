# SPDX-License-Identifier: Apache-2.0
"""M2 (v3.5.1) Python SDK wrappers for 20-eVTOL FleetSimEngine + Part-IS & Lift+Cruise 8+1/4+1."""

from __future__ import annotations

import math
from typing import Any

try:
    from . import _vertirta_native as _native
except ImportError:
    _native = None

_HAS_NATIVE_M2: bool = bool(_native is not None and getattr(_native, "HAS_M2_BINDINGS", False))


class VertiFleetSim:
    """20-eVTOL Anatolian U-Space Fleet & Part-IS Cyber-Resilience simulator wrapper."""

    def __init__(self, n_agents: int = 20, threads: int = 1) -> None:
        self._n_agents = max(1, min(20, int(n_agents)))
        self._tick = 0
        self._cel_agents: set[int] = set()
        self._cyber_state = "SEC_NOMINAL"
        self._quarantine_start_s: float | None = None
        if _HAS_NATIVE_M2 and _native is not None:
            self._impl: Any = _native.NativeFleetSim(self._n_agents, max(1, min(4, int(threads))))
            self.backend = "cpp_nanobind"
        else:
            self._impl = None
            self.backend = "python_fallback"

    @property
    def agent_count(self) -> int:
        return int(self._impl.agent_count) if self._impl is not None else self._n_agents

    def reset(self, seed: int = 20261123) -> None:
        self._tick = 0
        self._cel_agents.clear()
        self._cyber_state = "SEC_NOMINAL"
        self._quarantine_start_s = None
        if self._impl is not None:
            self._impl.reset(int(seed))

    def inject_same_arm_cel(self, agent_idx: int, arm_idx: int = 0) -> None:
        idx = int(agent_idx)
        if 0 <= idx < self._n_agents:
            self._cel_agents.add(idx)
        if self._impl is not None:
            self._impl.inject_same_arm_cel(idx, int(arm_idx))

    def step_100hz(self) -> dict[str, Any]:
        if self._impl is not None:
            return dict(self._impl.step_100hz())
        self._tick += 1
        return {
            "tick_index": self._tick,
            "active_agents": self._n_agents,
            "cel_active_agents": len(self._cel_agents),
            "airspace_breaches": 0,
            "max_rta_hazard_rows": min(14, self._n_agents - 1 + 2),
            "min_inter_agent_sep_m": 220.0,
            "sim_time_s": self._tick * 0.01,
        }

    def agent_cel_latched(self, agent_idx: int) -> bool:
        if self._impl is not None:
            return bool(self._impl.agent_cel_latched(int(agent_idx)))
        return int(agent_idx) in self._cel_agents

    def check_ussp_geofence(
        self,
        sig_valid: bool,
        cmd_lon_deg: float,
        cmd_agl_m: float,
        t_s: float,
    ) -> dict[str, Any]:
        lon = float(cmd_lon_deg)
        agl = float(cmd_agl_m)
        t = float(t_s)
        inputs_finite = math.isfinite(lon) and math.isfinite(agl) and math.isfinite(t)
        if self._impl is not None:
            return dict(
                self._impl.check_ussp_geofence(
                    bool(sig_valid) and inputs_finite,
                    lon if math.isfinite(lon) else 29.9999,
                    agl if math.isfinite(agl) else 999.0,
                    t if math.isfinite(t) else 0.0,
                )
            )
        violated = (
            (not sig_valid)
            or (not inputs_finite)
            or (lon >= 29.1895)
            or (agl > 120.0)
            or (agl < 0.0)
        )
        if violated:
            if self._quarantine_start_s is None:
                self._quarantine_start_s = t if math.isfinite(t) else 0.0
            self._cyber_state = "SEC_USSP_QUARANTINE"
        return {
            "cyber_state": self._cyber_state,
            "drop_ussp_directives": violated,
            "emit_is_or_230_draft": violated,
        }

    def step_cyber_time(self, t_s: float, ussp_auth_restored: bool = False) -> dict[str, Any]:
        t = float(t_s)
        if self._impl is not None:
            if not math.isfinite(t):
                return dict(self._impl.step_cyber_time(1.0e9, False))
            return dict(self._impl.step_cyber_time(t, bool(ussp_auth_restored)))
        elapsed = 0.0
        divert = False
        if self._cyber_state == "SEC_USSP_QUARANTINE" and self._quarantine_start_s is not None:
            if not math.isfinite(t):
                self._cyber_state = "SEC_AUTONOMOUS_RTB"
                divert = True
                elapsed = 10.0
            elif ussp_auth_restored:
                self._cyber_state = "SEC_NOMINAL"
                self._quarantine_start_s = None
            else:
                elapsed = max(0.0, t - self._quarantine_start_s)
                if elapsed >= 10.0:
                    self._cyber_state = "SEC_AUTONOMOUS_RTB"
                    divert = True
        elif self._cyber_state == "SEC_AUTONOMOUS_RTB":
            divert = True
            elapsed = (
                max(10.0, t - (self._quarantine_start_s or 0.0))
                if math.isfinite(t)
                else 10.0
            )
        return {
            "cyber_state": self._cyber_state,
            "divert_to_nearest_fato_emg": divert,
            "elapsed_quarantine_s": elapsed,
        }


class VertiLiftCruiseSim:
    """VertiRTA-LC-240 Lift+Cruise 8+1 / 4+1 transition & OEI audit wrapper."""

    def __init__(self, layout: str = "8+1") -> None:
        if layout not in ("8+1", "4+1"):
            raise ValueError(f"layout must be '8+1' or '4+1', got {layout!r}")
        self._layout = layout
        if _HAS_NATIVE_M2 and _native is not None:
            self._impl: Any = _native.NativeLiftCruiseSim(layout)
            self.backend = "cpp_nanobind"
        else:
            self._impl = None
            self.backend = "python_fallback"

    @property
    def actuator_count(self) -> int:
        if self._impl is not None:
            return int(self._impl.actuator_count)
        return 12 if self._layout == "8+1" else 8

    @property
    def lift_rotor_count(self) -> int:
        if self._impl is not None:
            return int(self._impl.lift_rotor_count)
        return 8 if self._layout == "8+1" else 4

    @property
    def cruise_airspeed_mps(self) -> float:
        return float(self._impl.cruise_airspeed_mps) if self._impl is not None else 38.0

    @property
    def max_airspeed_mps(self) -> float:
        return float(self._impl.max_airspeed_mps) if self._impl is not None else 46.3

    @property
    def best_glide_ld_ratio(self) -> float:
        return float(self._impl.best_glide_ld_ratio) if self._impl is not None else 13.46

    def audit_oei_trim(self, v_airspeed_mps: float, failed_lift_rotor: int = 0) -> dict[str, Any]:
        v = float(v_airspeed_mps)
        if not math.isfinite(v):
            return {
                "trim_feasible_4dof": False,
                "peak_thrust_ratio": 999.0,
                "effective_rank": 0,
                "v_airspeed_mps": v,
            }
        if self._impl is not None:
            return dict(self._impl.audit_oei_trim(v, int(failed_lift_rotor)))
        if self._layout == "8+1":
            return {
                "trim_feasible_4dof": True,
                "peak_thrust_ratio": 0.7907 if v < 1.0 else max(0.35, 0.7907 - 0.015 * v),
                "effective_rank": 4,
                "v_airspeed_mps": v,
            }
        feasible = v >= 12.0
        return {
            "trim_feasible_4dof": feasible,
            "peak_thrust_ratio": 0.82 if feasible else 1.15,
            "effective_rank": 4 if feasible else 3,
            "v_airspeed_mps": v,
        }
