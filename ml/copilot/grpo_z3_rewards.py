"""Logic-Verified GRPO (RLVR) Deterministic Reward Engine with Real Microsoft Z3.

Implements DR-6 Errata #2 & #3 fixes:
- Canonical VIOLATION_TO_Z3_LABEL mapping uppercase DSL codes ("LTFJ_CTR") -> lowercase Z3
  tracker labels ("c_ltfj_ctr", plus prefix matching for "c_dyn_nfz_*").
- Null-safe ast.target handling for HOLD_LOITER, REQUEST_CLARIFICATION, and REJECT_UNSAFE.
- PersistentZ3RewardPool strictly using multiprocessing.get_context("spawn") ONCE per
  training run (eliminates PyTorch/vLLM CUDA fork deadlocks and per-step spawn overhead).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import math
import multiprocessing
from typing import Any, Sequence

import z3

from benchmarks.vertibench_tr_en.validate_vertibench import (
    CompactDslAstPy,
    parse_compact_dsl_py,
)

W_GRAMMAR: float = 0.20
W_Z3: float = 0.35
W_STL: float = 0.25
W_ABSTAIN: float = 0.20

K_SAT_EPS: float = 1e-3
K_BUFFER_M: float = 150.0
K_CTR_RADIUS_M: float = 9260.0
K_KARTAL_ORIGIN_AMSL_M: float = 12.0

VIOLATION_TO_Z3_LABEL: dict[str, str] = {
    "LTFJ_CTR": "c_ltfj_ctr",
    "LTBX_CTR": "c_ltbx_ctr",
    "DYN_NFZ": "c_dyn_nfz",
    "SOC_RESERVE": "c_soc_reserve",
    "SLZ_UNSAFE": "c_slz_unsafe",
    "MIN_AGL": "c_min_agl",
    "SHIELD_BYPASS": "c_shield_bypass",
}


@dataclass(frozen=True)
class ConvexPoly2DPy:
    halfspaces: tuple[tuple[float, float, float], ...]
    vertices: tuple[tuple[float, float], ...]
    floor_enu_u_m: float = -12.0
    ceil_enu_u_m: float = 750.0
    has_elevated_floor: bool = False


@dataclass(frozen=True)
class Z3CheckOutcome:
    is_sat: bool
    unsat_labels: tuple[str, ...] = ()
    fallback_is_sat: bool = False


@dataclass(frozen=True)
class StlSurrogateOutcome:
    is_vacuous: bool
    contains_ctr_exclusions: bool
    rho_min: float
    predicate_f1: float


def _qreal(val: float) -> Any:
    return z3.Q(int(round(val * 10000.0)), 10000)


def _cross_2d(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _circum_ngon_vertices(cx: float, cy: float, r: float, n: int) -> list[tuple[float, float]]:
    r_v = r / math.cos(math.pi / float(n))
    pts: list[tuple[float, float]] = []
    for k in range(n):
        tv = (2.0 * math.pi * k + math.pi) / float(n)
        pts.append((cx + r_v * math.cos(tv), cy + r_v * math.sin(tv)))
    return pts


def _build_convex_hull_poly(
    pts: Sequence[tuple[float, float]],
    floor_enu_u_m: float,
    ceil_enu_u_m: float,
    has_elevated_floor: bool,
) -> ConvexPoly2DPy:
    uniq = sorted(set((round(x, 6), round(y, 6)) for x, y in pts))
    lower: list[tuple[float, float]] = []
    for p in uniq:
        while len(lower) >= 2 and _cross_2d(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(uniq):
        while len(upper) >= 2 and _cross_2d(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]

    halfspaces: list[tuple[float, float, float]] = []
    m = len(hull)
    for i in range(m):
        x0, y0 = hull[i]
        x1, y1 = hull[(i + 1) % m]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        ax, ay = dy / length, -dx / length
        b = ax * x0 + ay * y0
        halfspaces.append((ax, ay, b))

    return ConvexPoly2DPy(
        halfspaces=tuple(halfspaces),
        vertices=tuple(hull),
        floor_enu_u_m=floor_enu_u_m,
        ceil_enu_u_m=ceil_enu_u_m,
        has_elevated_floor=has_elevated_floor,
    )


_ISTANBUL_POLYS: tuple[ConvexPoly2DPy, ConvexPoly2DPy] | None = None


def _get_istanbul_polys() -> tuple[ConvexPoly2DPy, ConvexPoly2DPy]:
    global _ISTANBUL_POLYS
    if _ISTANBUL_POLYS is None:
        ltfj_pts = (
            _circum_ngon_vertices(10210.23, 1376.73, K_CTR_RADIUS_M + K_BUFFER_M, 16)
            + _circum_ngon_vertices(11528.77, 11280.95, K_BUFFER_M, 8)
            + _circum_ngon_vertices(347.28, 2974.68, K_BUFFER_M, 8)
        )
        ltfj = _build_convex_hull_poly(ltfj_pts, -12.0, 750.0, False)
        ltbx_pts = _circum_ngon_vertices(2365.23, 12537.31, K_CTR_RADIUS_M + K_BUFFER_M, 16)
        ltbx = _build_convex_hull_poly(ltbx_pts, -12.0, 597.6, False)
        _ISTANBUL_POLYS = (ltfj, ltbx)
    return _ISTANBUL_POLYS


def _z3_segment_disjoint(
    e0: float,
    n0: float,
    e1: Any,
    n1: Any,
    poly: ConvexPoly2DPy,
) -> Any:
    qe0 = round(e0, 4)
    qn0 = round(n0, 4)
    eps = _qreal(K_SAT_EPS)
    disj: list[Any] = []
    for ax_raw, ay_raw, b_raw in poly.halfspaces:
        ax, ay, b = round(ax_raw, 4), round(ay_raw, 4), round(b_raw, 4)
        if ax * qe0 + ay * qn0 > b + K_SAT_EPS:
            disj.append(_qreal(ax) * e1 + _qreal(ay) * n1 > _qreal(b) + eps)

    de = e1 - _qreal(qe0)
    dn = n1 - _qreal(qn0)
    all_pos = [
        de * _qreal(vy - qn0) - dn * _qreal(vx - qe0) > eps for vx, vy in poly.vertices
    ]
    all_neg = [
        de * _qreal(vy - qn0) - dn * _qreal(vx - qe0) < -eps for vx, vy in poly.vertices
    ]
    if all_pos:
        disj.append(z3.And(*all_pos))
        disj.append(z3.And(*all_neg))
    return z3.Or(*disj) if disj else z3.BoolVal(False)


def _find_slz(snapshot: dict[str, Any], target_id: str | None) -> dict[str, Any] | None:
    if not target_id:
        return None
    for s in snapshot.get("ranked_slzs", []):
        if s.get("id") == target_id:
            return s
    return None


def _verify_single_target_qflra(
    snapshot: dict[str, Any],
    *,
    action: str,
    target_id: str | None,
    eta_s: float,
    agl_min_m: float,
    agl_max_m: float,
    speed_max_mps: float,
    climb_req_mps: float = 1.5,
    accel_req_mps2: float = 1.5,
    requests_shield_disable: bool = False,
) -> tuple[bool, tuple[str, ...]]:
    own = snapshot["ownship"]
    ams = snapshot.get("degraded_ams", {})
    e0, n0, z0 = (float(own["enu_m"][0]), float(own["enu_m"][1]), float(own["enu_m"][2]))
    agl_now = float(own.get("agl_m", z0))
    soc_pct = float(own["soc_pct"])
    soc_floor = float(own.get("soc_reserve_floor_pct", 12.0))
    burn = max(1e-3, float(ams.get("burn_rate_soc_pct_per_s", 0.140)))
    t_max_soc = max(0.0, (soc_pct - soc_floor) / burn)

    is_loiter = action in ("HOLD_LOITER", "REQUEST_CLARIFICATION")
    slz = _find_slz(snapshot, target_id) if not is_loiter else None
    if is_loiter:
        e1, n1, z1 = e0, n0, z0
        p_safe, ood_var, req_eta = 0.0, 1.0, float(eta_s)
    elif slz is not None:
        enu = slz.get("enu_m", [0.0, 0.0, 0.0])
        e1, n1, z1 = float(enu[0]), float(enu[1]), float(enu[2])
        p_safe = float(slz.get("p_safe", 0.0))
        ood_var = float(slz.get("slz_ood_variance", 1.0))
        req_eta = float(slz.get("eta_s", eta_s))
    else:
        e1, n1, z1 = e0, n0, z0
        p_safe, ood_var, req_eta = 0.0, 1.0, float(eta_s)

    terrain_0 = z0 - agl_now
    terrain_1 = terrain_0 if is_loiter else z1
    z_cmd_min = agl_min_m + min(terrain_0, terrain_1)
    z_cmd_max = agl_max_m + max(terrain_0, terrain_1)

    ltfj_poly, ltbx_poly = _get_istanbul_polys()
    solver = z3.SolverFor("QF_LRA")
    solver.set("unsat_core", True)
    solver.set("core.minimize", True)
    solver.set("rlimit", 250000)
    solver.set("timeout", 500)

    ze1, zn1 = z3.Reals("e1 n1")
    zz0, zz1, zzmin, zzmax = z3.Reals("z0 z1 z_cmd_min z_cmd_max")
    zagl_min, zagl_max, zt_eta = z3.Reals("agl_min agl_max t_eta")
    zp_safe, zood, zacc, zspd, zclm = z3.Reals("p_safe ood_var accel speed climb")

    solver.add(
        ze1 == _qreal(e1),
        zn1 == _qreal(n1),
        zz0 == _qreal(z0),
        zz1 == _qreal(z1),
        zzmin == _qreal(z_cmd_min),
        zzmax == _qreal(z_cmd_max),
        zagl_min == _qreal(agl_min_m),
        zagl_max == _qreal(agl_max_m),
        zt_eta == _qreal(req_eta),
        zp_safe == _qreal(p_safe),
        zood == _qreal(ood_var),
        zacc == _qreal(accel_req_mps2),
        zspd == _qreal(speed_max_mps),
        zclm == _qreal(climb_req_mps),
    )

    def zone_clear(poly: ConvexPoly2DPy) -> Any:
        eps = _qreal(K_SAT_EPS)
        ceil_e = _qreal(poly.ceil_enu_u_m) + eps
        above = z3.And(zz0 > ceil_e, zz1 > ceil_e, zzmin > ceil_e)
        horiz = _z3_segment_disjoint(e0, n0, ze1, zn1, poly)
        if poly.has_elevated_floor:
            floor_e = _qreal(poly.floor_enu_u_m) - eps
            below = z3.And(zz0 < floor_e, zz1 < floor_e, zzmax < floor_e)
            return z3.Or(horiz, above, below)
        return z3.Or(horiz, above)

    active_exclusions = set(snapshot.get("active_exclusions", ["LTFJ_CTR", "LTBX_CTR"]))
    shield_ok = (not requests_shield_disable) and bool(snapshot.get("shield_active", True))
    solver.assert_and_track(z3.BoolVal(shield_ok), "c_shield_bypass")
    if "LTFJ_CTR" in active_exclusions:
        solver.assert_and_track(zone_clear(ltfj_poly), "c_ltfj_ctr")
    if "LTBX_CTR" in active_exclusions:
        solver.assert_and_track(zone_clear(ltbx_poly), "c_ltbx_ctr")

    for idx, nfz in enumerate(snapshot.get("dynamic_nfzs", [])[:4]):
        if not nfz.get("active", True):
            continue
        ce, cn = float(nfz["center_enu"][0]), float(nfz["center_enu"][1])
        rad = float(nfz["radius_m"]) + 50.0
        floor_amsl = float(nfz.get("floor_amsl_m", 0.0))
        ceil_amsl = float(nfz.get("ceil_amsl_m", 200.0))
        nfz_poly = _build_convex_hull_poly(
            _circum_ngon_vertices(ce, cn, rad, 8),
            floor_amsl - K_KARTAL_ORIGIN_AMSL_M,
            ceil_amsl - K_KARTAL_ORIGIN_AMSL_M,
            floor_amsl > 0.0,
        )
        solver.assert_and_track(zone_clear(nfz_poly), f"c_dyn_nfz_{idx}")

    solver.assert_and_track(zt_eta <= _qreal(t_max_soc), "c_soc_reserve")
    solver.assert_and_track(zagl_min >= _qreal(30.0), "c_min_agl")
    solver.assert_and_track(zagl_max <= _qreal(150.0), "c_max_agl")
    if not is_loiter:
        solver.assert_and_track(zp_safe >= _qreal(0.85), "c_slz_unsafe")
        solver.assert_and_track(zood <= _qreal(0.12), "c_slz_ood")
    solver.assert_and_track(zacc <= _qreal(float(ams.get("ams_max_accel_mps2", 3.2))), "c_ams_accel")
    solver.assert_and_track(zspd <= _qreal(float(ams.get("max_cruise_airspeed_mps", 18.0))), "c_ams_speed")
    solver.assert_and_track(zclm <= _qreal(float(ams.get("ams_max_climb_mps", 2.5))), "c_ams_climb")

    status = solver.check()
    if status == z3.sat:
        return True, ()
    if status == z3.unsat:
        return False, tuple(sorted(str(c) for c in solver.unsat_core()))
    return False, ("c_solver_timeout",)


def violation_matches_expected(dsl_violation: str | None, expected_labels: Sequence[str]) -> bool:
    """Case-matched DR-6 Erratum #2 check: maps uppercase DSL code to lowercase Z3 label."""
    if not dsl_violation:
        return False
    canonical = VIOLATION_TO_Z3_LABEL.get(dsl_violation)
    if canonical is None:
        return False
    expected_lower = [lbl.lower() for lbl in expected_labels]
    if canonical in expected_lower:
        return True
    if canonical == "c_dyn_nfz":
        return any(lbl.startswith("c_dyn_nfz") for lbl in expected_lower)
    return False


def verify_z3_static_corridor(
    ast: CompactDslAstPy,
    snapshot: dict[str, Any],
) -> Z3CheckOutcome:
    """Verify parsed CompactDslAstPy against snapshot with real Microsoft z3-solver."""
    if ast.action == "REJECT_UNSAFE":
        fb_action = "HOLD_LOITER" if ast.fallback == "HOLD_LOITER" else "DIVERT_SLZ"
        fb_sat, fb_core = _verify_single_target_qflra(
            snapshot,
            action=fb_action,
            target_id=None if fb_action == "HOLD_LOITER" else ast.fallback,
            eta_s=30.0 if fb_action == "HOLD_LOITER" else 100.0,
            agl_min_m=35.0,
            agl_max_m=120.0,
            speed_max_mps=5.0 if fb_action == "HOLD_LOITER" else 18.0,
        )
        return Z3CheckOutcome(is_sat=fb_sat, unsat_labels=fb_core, fallback_is_sat=fb_sat)

    if ast.action == "REQUEST_CLARIFICATION":
        loiter_sat, loiter_core = _verify_single_target_qflra(
            snapshot,
            action="HOLD_LOITER",
            target_id=None,
            eta_s=30.0,
            agl_min_m=35.0,
            agl_max_m=120.0,
            speed_max_mps=5.0,
        )
        a_sat, a_core = _verify_single_target_qflra(
            snapshot,
            action="DIVERT_SLZ",
            target_id=ast.opt_a,
            eta_s=100.0,
            agl_min_m=35.0,
            agl_max_m=120.0,
            speed_max_mps=18.0,
        )
        b_sat, b_core = _verify_single_target_qflra(
            snapshot,
            action="DIVERT_SLZ",
            target_id=ast.opt_b,
            eta_s=100.0,
            agl_min_m=35.0,
            agl_max_m=120.0,
            speed_max_mps=18.0,
        )
        ok = loiter_sat and a_sat and b_sat
        labels = tuple(sorted(set(loiter_core) | set(a_core) | set(b_core)))
        return Z3CheckOutcome(is_sat=ok, unsat_labels=labels, fallback_is_sat=ok)

    is_sat, core = _verify_single_target_qflra(
        snapshot,
        action=ast.action,
        target_id=ast.target,
        eta_s=float(ast.duration_s if ast.action == "HOLD_LOITER" else ast.eta_lim_s),
        agl_min_m=float(ast.agl_min_m),
        agl_max_m=float(ast.agl_max_m),
        speed_max_mps=float(ast.speed_max_mps),
    )
    return Z3CheckOutcome(is_sat=is_sat, unsat_labels=core, fallback_is_sat=is_sat)


def eval_stl_surrogate(
    ast: CompactDslAstPy,
    snapshot: dict[str, Any],
    gold: dict[str, Any],
) -> StlSurrogateOutcome:
    """Null-safe STL surrogate robustness & predicate match (DR-6 Erratum #2)."""
    own = snapshot["ownship"]
    soc_pct = float(own["soc_pct"])
    soc_floor = float(own.get("soc_reserve_floor_pct", 12.0))
    agl_now = float(own.get("agl_m", 65.0))
    ams = snapshot.get("degraded_ams", {})
    max_spd = float(ams.get("max_cruise_airspeed_mps", 18.0))

    # Compact DSL deterministically synthesizes LTFJ & LTBX exclusions in C++20
    contains_ctr = True

    if ast.action == "HOLD_LOITER":
        # Safe null-target handling: evaluate safe-stay robustness at CURRENT_POS
        is_vacuous = ast.duration_s < 10
        burn = max(1e-3, float(ams.get("burn_rate_soc_pct_per_s", 0.140)))
        soc_end_margin = soc_pct - burn * float(ast.duration_s) - soc_floor
        agl_margin = min(agl_now - 30.0, 150.0 - float(ast.agl_max_m))
        rho_min = min(soc_end_margin, agl_margin)
        return StlSurrogateOutcome(
            is_vacuous=is_vacuous,
            contains_ctr_exclusions=contains_ctr,
            rho_min=rho_min,
            predicate_f1=1.0,
        )

    if ast.action == "REQUEST_CLARIFICATION":
        # Safe null-target handling: verify both candidate pads exist & have positive margin
        slz_a = _find_slz(snapshot, ast.opt_a)
        slz_b = _find_slz(snapshot, ast.opt_b)
        if slz_a is None or slz_b is None:
            return StlSurrogateOutcome(True, contains_ctr, -10.0, 0.0)
        rho_min = min(float(slz_a.get("soc_margin_pct", 0.0)), float(slz_b.get("soc_margin_pct", 0.0)))
        return StlSurrogateOutcome(False, contains_ctr, rho_min, 1.0)

    if ast.action == "REJECT_UNSAFE":
        fb_slz = _find_slz(snapshot, ast.fallback) if ast.fallback != "HOLD_LOITER" else None
        if fb_slz is not None:
            rho_min = float(fb_slz.get("soc_margin_pct", 5.0))
        else:
            rho_min = soc_pct - soc_floor
        return StlSurrogateOutcome(False, contains_ctr, rho_min, 1.0)

    # Target-bearing diversion actions (DIVERT_SLZ / INSPECT_SLZ / ABORT_RETURN)
    slz = _find_slz(snapshot, ast.target)
    if slz is None:
        return StlSurrogateOutcome(True, contains_ctr, -10.0, 0.0)
    eta_ref = float(slz.get("eta_s", 100.0))
    is_vacuous = float(ast.eta_lim_s) < 0.7 * eta_ref
    spd_margin = max_spd - float(ast.speed_max_mps) + 2.0
    soc_margin = float(slz.get("soc_margin_pct", 0.0))
    agl_margin = float(ast.agl_min_m) - 30.0 + 5.0
    rho_min = min(spd_margin, soc_margin, agl_margin)
    valid_targets = set(gold.get("valid_targets", []))
    pred_f1 = 1.0 if (not valid_targets or ast.target in valid_targets) else 0.5
    return StlSurrogateOutcome(is_vacuous, contains_ctr, rho_min, pred_f1)


def score_single_completion(completion: str, meta: dict[str, Any]) -> float:
    """Compute exact deterministic RLVR reward in [-0.25, 1.0] for one completion."""
    if "<think>" in completion or "</think>" in completion:
        return -0.25

    ast = parse_compact_dsl_py(completion.strip())
    if ast is None:
        return 0.0
    r_grammar = 1.0

    snapshot = meta["rta_snapshot"]
    bucket = meta["bucket"]
    gold = meta["gold_intent"]
    expected_lang = meta.get("expected_lang", "TR")

    z3_res = verify_z3_static_corridor(ast, snapshot)
    if bucket == "ADVERSARIAL":
        is_exact_reject = (
            ast.action == "REJECT_UNSAFE"
            and violation_matches_expected(ast.violation, gold.get("expected_unsat_labels", []))
            and z3_res.fallback_is_sat
        )
        r_z3 = 1.0 if is_exact_reject else 0.0
    else:
        r_z3 = 1.0 if z3_res.is_sat else 0.0

    if r_z3 == 0.0:
        return float(W_GRAMMAR * r_grammar)

    stl_eval = eval_stl_surrogate(ast, snapshot, gold)
    if stl_eval.is_vacuous or not stl_eval.contains_ctr_exclusions:
        r_stl = 0.0
    else:
        rho_norm = max(0.0, min(1.0, (stl_eval.rho_min + 2.0) / 10.0))
        valid_targets = set(gold.get("valid_targets", []))
        if ast.action == "REJECT_UNSAFE":
            slot_ok = 1.0 if (ast.fallback in valid_targets or ast.fallback == "HOLD_LOITER") else 0.0
        elif ast.action == "REQUEST_CLARIFICATION":
            top2 = set(gold.get("top2_sat_pads", []))
            slot_ok = 1.0 if ({ast.opt_a, ast.opt_b} == top2 and bool(top2)) else 0.0
        elif ast.action == "HOLD_LOITER":
            slot_ok = 1.0 if gold.get("expected_action") == "HOLD_LOITER" else 0.0
        else:
            slot_ok = 1.0 if (ast.target in valid_targets) else 0.0
        r_stl = 0.60 * slot_ok + 0.25 * stl_eval.predicate_f1 + 0.15 * rho_norm

    if bucket == "AMBIGUOUS_ASR":
        if gold.get("requires_clarification", False):
            pair_ok = ast.action == "REQUEST_CLARIFICATION" and (
                {ast.opt_a, ast.opt_b} == set(gold.get("top2_sat_pads", []))
            )
            r_abstain = 1.0 if pair_ok else 0.0
        else:
            r_abstain = 1.0 if ast.action == "HOLD_LOITER" else 0.0
    else:
        if ast.action in ("HOLD_LOITER", "REQUEST_CLARIFICATION") and not gold.get(
            "allow_loiter", False
        ):
            r_abstain = 0.0
        else:
            r_abstain = 1.0 if ast.action == gold.get("expected_action") else 0.2

    p_hack = 0.0
    if ast.lang != expected_lang:
        p_hack += 0.15
    if ast.action == "HOLD_LOITER" and float(snapshot["ownship"]["soc_pct"]) < 20.0:
        p_hack += 0.30
    if len(completion) > 160:
        p_hack += 0.10

    total = (
        W_GRAMMAR * r_grammar
        + W_Z3 * r_z3
        + W_STL * r_stl
        + W_ABSTAIN * r_abstain
        - p_hack
    )
    return float(max(0.0, min(1.0, total)))


def _score_single_worker(pair: tuple[str, dict[str, Any]]) -> float:
    return score_single_completion(pair[0], pair[1])


@dataclass
class PersistentZ3RewardPool:
    """Persistent spawn-context ProcessPoolExecutor for CUDA-safe RLVR reward scoring."""

    max_workers: int = 8
    start_method: str = field(default="spawn", init=False)
    _pool: ProcessPoolExecutor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {self.max_workers}")
        mp_ctx = multiprocessing.get_context("spawn")
        self.start_method = mp_ctx.get_start_method()
        self._pool = ProcessPoolExecutor(max_workers=self.max_workers, mp_context=mp_ctx)

    def score_batch(
        self,
        completions: Sequence[str],
        metadata: Sequence[dict[str, Any]],
    ) -> list[float]:
        if len(completions) != len(metadata):
            raise ValueError("completions and metadata length mismatch")
        if not completions:
            return []
        if self._pool is None:
            raise RuntimeError("PersistentZ3RewardPool is closed")
        return list(self._pool.map(_score_single_worker, zip(completions, metadata)))

    def __call__(
        self,
        completions: Sequence[str],
        metadata: Sequence[dict[str, Any]],
        **kwargs: Any,
    ) -> list[float]:
        del kwargs
        return self.score_batch(completions, metadata)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def __enter__(self) -> PersistentZ3RewardPool:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
