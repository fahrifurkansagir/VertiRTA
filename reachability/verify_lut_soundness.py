"""Soundness & Binary Integrity Verifier for VertiRTA 4D CBVF & 2D CEL LUTs.

Verifies:
  1. Exact binary file sizes:
       .vrta_lut   == 11,313,864 B (4,096 B VRTALUT1 header + 11,309,768 B payload)
       .vrta_cel2d ==     11,230 B (4,096 B VRTACEL2 header +      7,134 B payload)
  2. Header magic, version, geometry, and payload SHA-256 digests.
  3. All 3 DR-2 Critical Errata Locks:
       - Erratum #1: Lax-Friedrichs CFL bound partial_max_magnitudes matches _support_U
                     (|dH/dp_i| <= partial_max_magnitudes[i]; Nominal max |u_r| >= 14.06 m/s^2).
       - Erratum #2: conservative_smax_4(sr, sz, beta=4.0) <= max(sr, sz) (0.0 at (0,0)).
       - Erratum #3: v_cap_f[2] (2M-Cross) & v_cap_f[3] (Low-Vbat) < 16.5 m/s in header.
  4. 4D CBVF monotonicity across fault regimes, finite gradients, safe/unsafe sanity.
  5. 2D CEL zero-level braking curve within 0.35 m of z_req = (v_z^2 - 4) / (2 * a_net)
     across all 3 voltage-sag tiers (T/W in {1.25, 1.20, 1.15}).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running directly as `python reachability/verify_lut_soundness.py --quick-fixture`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reachability.cbvf_4d_dynamics import (
    DEFAULT_REGIMES,
    VertiRta4DDynamics,
    compute_degraded_v_cap,
    conservative_smax_4,
)
from reachability.solve_cbvf_luts import (
    LUT_TOTAL_BYTES,
    eval_cbvf_4d,
    load_vrta_lut,
    solve_and_export_lut,
)
from reachability.solve_cel_2d_lut import (
    CEL2D_TOTAL_BYTES,
    DEFAULT_CEL_TIERS,
    analytic_cel_z_req,
    find_cel_zero_level_z,
    load_cel_2d_lut,
    solve_and_export_cel_2d,
)

DEFAULT_QUICK_LUT = _REPO_ROOT / "tests" / "fixtures" / "quick_test.vrta_lut"
DEFAULT_QUICK_CEL2D = _REPO_ROOT / "tests" / "fixtures" / "quick_test.vrta_cel2d"


def verify_all_soundness(
    lut_path: Path | str,
    cel2d_path: Path | str,
    *,
    cel_curve_tol_m: float = 0.35,
) -> dict[str, Any]:
    """Run all binary, errata, and physical soundness gates on .vrta_lut & .vrta_cel2d."""
    lut_p = Path(lut_path)
    cel_p = Path(cel2d_path)
    failures: list[str] = []

    # 1. Binary sizes & SHA-256 integrity
    lut_bytes = lut_p.stat().st_size if lut_p.is_file() else -1
    cel_bytes = cel_p.stat().st_size if cel_p.is_file() else -1
    if lut_bytes != LUT_TOTAL_BYTES:
        failures.append(f".vrta_lut size {lut_bytes} B != {LUT_TOTAL_BYTES} B")
    if cel_bytes != CEL2D_TOTAL_BYTES:
        failures.append(f".vrta_cel2d size {cel_bytes} B != {CEL2D_TOTAL_BYTES} B")

    lut_hdr, lut_data = load_vrta_lut(lut_p, verify_sha256=True)
    cel_hdr, cel_data = load_cel_2d_lut(cel_p, verify_sha256=True)

    # 2. Erratum #1: CFL partial_max_magnitudes matches _support_U
    nom_dyn = VertiRta4DDynamics.from_regime_spec(DEFAULT_REGIMES[0])
    nom_ur_max = nom_dyn.max_achievable_ur()
    pm_nom = np.asarray(
        nom_dyn.partial_max_magnitudes(
            np.array([10.0, 0.0, 10.0, 0.0], dtype=np.float32), 0.0, 0.0, None
        ),
        dtype=np.float64,
    )
    if nom_ur_max < 14.06 or float(pm_nom[1]) < nom_ur_max + nom_dyn.d_ar - 1e-4:
        failures.append(
            f"Erratum #1 CFL mismatch: max_ur={nom_ur_max:.4f}, pm[1]={pm_nom[1]:.4f}"
        )

    max_cfl_ratio = 0.0
    eps = 1e-4
    probe_state = np.array([15.0, -12.0, 12.0, -5.0], dtype=np.float32)
    for spec in DEFAULT_REGIMES:
        dyn = VertiRta4DDynamics.from_regime_spec(spec)
        pm = np.asarray(dyn.partial_max_magnitudes(probe_state, 0.0, 0.0, None), dtype=np.float64)
        for theta in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
            p = np.array([math.cos(theta), math.sin(theta), math.sin(2 * theta), math.cos(2 * theta)])
            for d in range(4):
                dp = np.zeros(4)
                dp[d] = eps
                dh = abs(
                    float(dyn.hamiltonian(probe_state, 0.0, 0.0, p + dp))
                    - float(dyn.hamiltonian(probe_state, 0.0, 0.0, p - dp))
                ) / (2.0 * eps)
                ratio = dh / max(float(pm[d]), 1e-9)
                max_cfl_ratio = max(max_cfl_ratio, ratio)
                if dh > float(pm[d]) + 1e-3:
                    failures.append(
                        f"Erratum #1 CFL violated in {spec.name} dim {d}: |dH/dp|={dh:.4f} > {pm[d]:.4f}"
                    )

    # 3. Erratum #2: conservative_smax_4 <= max(sr, sz)
    smax_zero = float(conservative_smax_4(0.0, 0.0, beta=4.0))
    if smax_zero > 1e-6:
        failures.append(f"Erratum #2 optimistic smax_4(0,0)={smax_zero:.6f} > 0")
    for sr, sz in [(0.0, 0.0), (1.0, 1.0), (-2.0, 0.5), (3.5, -1.2)]:
        if float(conservative_smax_4(sr, sz, beta=4.0)) > max(sr, sz) + 1e-6:
            failures.append(f"Erratum #2 violated at ({sr}, {sz})")

    # 4. Erratum #3: degraded cruise speed cap v_cap_f embedded in .vrta_lut header
    v_cap_2m_calc = compute_degraded_v_cap("2M-Cross", d_ar=1.75, r_max=58.0, horizon_s=6.0)
    if not (0.0 < lut_hdr.v_cap_f[2] < 16.5 and 0.0 < lut_hdr.v_cap_f[3] < 16.5):
        failures.append(
            f"Erratum #3 v_cap_f violated in header: 2M-Cross={lut_hdr.v_cap_f[2]:.3f}, "
            f"Low-Vbat={lut_hdr.v_cap_f[3]:.3f}"
        )
    if abs(lut_hdr.v_cap_f[2] - v_cap_2m_calc) > 1e-3:
        failures.append(
            f"Header v_cap_f[2]={lut_hdr.v_cap_f[2]:.4f} != computed {v_cap_2m_calc:.4f}"
        )

    # 5. 4D LUT sanity & regime monotonicity
    if not all(math.isfinite(e) and e >= 0.5 * lut_hdr.scale for e in lut_hdr.eps_grid):
        failures.append(f"Invalid eps_grid in .vrta_lut header: {lut_hdr.eps_grid}")
    for r_hi, r_lo in [(0, 1), (1, 3), (3, 2)]:
        min_diff = int((lut_data[r_hi].astype(np.int32) - lut_data[r_lo].astype(np.int32)).min())
        if min_diff < -1:
            failures.append(f"4D LUT monotonicity failure: regime {r_hi} < regime {r_lo} (min_diff={min_diff})")

    val_safe, grad_safe = eval_cbvf_4d(lut_hdr, lut_data, 0, 15.0, -5.0, 20.0, -1.0)
    val_unsafe, _ = eval_cbvf_4d(lut_hdr, lut_data, 0, 0.5, -10.0, 0.5, -6.0)
    if not (math.isfinite(val_safe) and val_safe > 0.0 and all(math.isfinite(g) for g in grad_safe)):
        failures.append(f"4D LUT safe query failed: val={val_safe}, grad={grad_safe}")
    if not (math.isfinite(val_unsafe) and val_unsafe < -0.25):
        failures.append(f"4D LUT unsafe query failed: val={val_unsafe}")

    # 6. 2D CEL LUT braking curve match within 0.35 m across all 3 sag tiers
    cel_max_err = 0.0
    cel_tier_errors: dict[str, float] = {}
    for tier_idx, tier in enumerate(DEFAULT_CEL_TIERS):
        tier_max = 0.0
        for vz_test in (-4.0, -6.0, -8.0, -10.0):
            z_lut = find_cel_zero_level_z(cel_hdr, cel_data, tier_idx, vz_test)
            z_ref = analytic_cel_z_req(vz_test, tier.a_net, v_td_max=cel_hdr.v_td_max)
            err = abs(z_lut - z_ref)
            tier_max = max(tier_max, err)
            cel_max_err = max(cel_max_err, err)
            if err > cel_curve_tol_m:
                failures.append(
                    f"2D CEL tier {tier.name} vz={vz_test}: "
                    f"LUT z0={z_lut:.3f} m vs analytic {z_ref:.3f} m (err={err:.3f} m > {cel_curve_tol_m} m)"
                )
        cel_tier_errors[tier.name] = tier_max

    for t_hi, t_lo in [(0, 1), (1, 2)]:
        min_diff = int((cel_data[t_hi].astype(np.int32) - cel_data[t_lo].astype(np.int32)).min())
        if min_diff < -2:
            failures.append(f"2D CEL tier monotonicity failure: tier {t_hi} < tier {t_lo} ({min_diff})")

    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "lut_bytes": lut_bytes,
        "lut_payload_sha256": lut_hdr.payload_sha256.hex(),
        "lut_eps_grid": lut_hdr.eps_grid,
        "lut_v_cap_f": lut_hdr.v_cap_f,
        "cel2d_bytes": cel_bytes,
        "cel2d_payload_sha256": cel_hdr.payload_sha256.hex(),
        "cel2d_eps_grid": cel_hdr.eps_grid,
        "cel2d_max_curve_err_m": cel_max_err,
        "cel2d_tier_errors_m": cel_tier_errors,
        "erratum1_nom_max_ur": nom_ur_max,
        "erratum1_max_cfl_ratio": max_cfl_ratio,
        "erratum2_smax_zero": smax_zero,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify VertiRTA 4D CBVF (.vrta_lut) & 2D CEL (.vrta_cel2d) LUT Soundness"
    )
    parser.add_argument(
        "--quick-fixture",
        action="store_true",
        help="Verify tests/fixtures/quick_test.{vrta_lut,vrta_cel2d} (generate if missing).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of quick fixtures before verification.",
    )
    parser.add_argument("--lut", type=Path, default=None, help="Path to .vrta_lut file.")
    parser.add_argument("--cel2d", type=Path, default=None, help="Path to .vrta_cel2d file.")
    args = parser.parse_args(argv)

    if args.quick_fixture:
        lut_path = args.lut or DEFAULT_QUICK_LUT
        cel_path = args.cel2d or DEFAULT_QUICK_CEL2D
        if args.regenerate or not lut_path.is_file():
            solve_and_export_lut(lut_path, quick_cpu_fixture=True)
        if args.regenerate or not cel_path.is_file():
            solve_and_export_cel_2d(cel_path)
    else:
        if args.lut is None or args.cel2d is None:
            parser.error("Provide --quick-fixture or both --lut and --cel2d")
        lut_path, cel_path = args.lut, args.cel2d

    report = verify_all_soundness(lut_path, cel_path)
    print(f"[verify_lut_soundness] .vrta_lut   : {lut_path} ({report['lut_bytes']:,} B)")
    print(f"  payload_sha256 : {report['lut_payload_sha256']}")
    print(f"  eps_grid [m]   : {[round(e, 5) for e in report['lut_eps_grid']]}")
    print(f"  v_cap_f [m/s]  : {[round(v, 3) for v in report['lut_v_cap_f']]}")
    print(f"[verify_lut_soundness] .vrta_cel2d : {cel_path} ({report['cel2d_bytes']:,} B)")
    print(f"  payload_sha256 : {report['cel2d_payload_sha256']}")
    print(f"  eps_grid [m]   : {[round(e, 5) for e in report['cel2d_eps_grid']]}")
    print(
        f"  max curve err  : {report['cel2d_max_curve_err_m']:.4f} m "
        f"(limit <= 0.35 m; per-tier: { {k: round(v, 4) for k, v in report['cel2d_tier_errors_m'].items()} })"
    )
    print(
        f"[verify_lut_soundness] Errata Locks: "
        f"E1 nom_ur={report['erratum1_nom_max_ur']:.3f} m/s^2 (max |dH/dp|/pm={report['erratum1_max_cfl_ratio']:.4f} <= 1.0), "
        f"E2 smax_4(0,0)={report['erratum2_smax_zero']:.6f}, "
        f"E3 2M-Cross v_cap={report['lut_v_cap_f'][2]:.3f} m/s < 16.5"
    )
    if not report["passed"]:
        print("FAILED SOUNDNESS CHECKS:")
        for item in report["failures"]:
            print(f"  - {item}")
        return 1
    print("[verify_lut_soundness] ALL SOUNDNESS & BINARY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
