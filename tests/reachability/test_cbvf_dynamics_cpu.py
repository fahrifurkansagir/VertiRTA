"""CPU unit & soundness tests for VertiRTA 4D CBVF and 2D CEL LUT pipeline.

Pins Plan 1A Task 6 and DR-2 Critical Errata #1, #2, #3:
  - Erratum #1: Matched _support_U / partial_max_magnitudes Lax-Friedrichs CFL
                (Nominal max achievable |u_r| = a_T * sin(35 deg) ~= 14.06 m/s^2,
                NOT hover-only g * tan(35 deg) ~= 6.87 m/s^2).
  - Erratum #2: Conservative smooth max smax_4(sr, sz) - ln(2)/beta <= max(sr, sz)
                (eliminates +0.173 m optimistic corner bias at sr == sz == 0).
  - Erratum #3: Degraded cruise speed cap v_cap_f embedded in .vrta_lut header
                (2M-Cross & Low-Vbat < 16.5 m/s so braking fits inside T=6.0 s & r_max=58 m).
  - Exact binary layouts:
      .vrta_lut   == 11,313,864 B (4,096 B VRTALUT1 header + 11,309,768 B int16 payload)
      .vrta_cel2d ==     11,230 B (4,096 B VRTACEL2 header +      7,134 B int16 payload)
  - 2D CEL vertical braking LUT matches z_req = (v_z^2 - 4) / (2 * a_net) within 0.35 m
    across all 3 voltage-sag tiers (T/W in {1.25, 1.20, 1.15}).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

try:
    import jax.numpy as jnp
except ImportError:  # Pure-NumPy CPU fallback when JAX is not installed locally
    jnp = np  # type: ignore[assignment]

from reachability.cbvf_4d_dynamics import (
    DEFAULT_REGIMES,
    VertiRta4DDynamics,
    compute_degraded_v_cap,
    conservative_smax_4,
    load_vehicle_params,
)
from reachability.solve_cbvf_luts import (
    LUT_HEADER_BYTES,
    LUT_MAGIC,
    LUT_PAYLOAD_BYTES,
    LUT_TOTAL_BYTES,
    eval_cbvf_4d,
    load_vrta_lut,
    solve_and_export_lut,
)
from reachability.solve_cel_2d_lut import (
    CEL2D_HEADER_BYTES,
    CEL2D_MAGIC,
    CEL2D_PAYLOAD_BYTES,
    CEL2D_TOTAL_BYTES,
    DEFAULT_CEL_TIERS,
    analytic_cel_z_req,
    eval_cel_2d,
    find_cel_zero_level_z,
    load_cel_2d_lut,
    solve_and_export_cel_2d,
)
from reachability.verify_lut_soundness import verify_all_soundness


def test_cfl_bounds_conservative_smax_2m_vcap_and_cel_2d(tmp_path: Path) -> None:
    """Plan 1A Task 6 canonical pinning test for DR-2 Errata #1/#2/#3 and 2D CEL."""
    dyn = VertiRta4DDynamics(
        a_thrust_max=2.50 * 9.80665,
        tilt_max_rad=float(np.deg2rad(35.0)),
    )
    pm = dyn.partial_max_magnitudes(jnp.array([10.0, 0.0, 10.0, 0.0]), 0.0, 0.0, None)
    # Erratum #1: matches a_T * sin(35 deg) ~= 14.062 m/s^2 + d_ar (1.75 m/s^2)!
    assert dyn.max_achievable_ur() >= 14.06
    assert float(pm[1]) >= dyn.max_achievable_ur() + dyn.d_ar - 1e-4

    # Erratum #2: conservative smax_4 subtracts ln(2)/beta so never optimistic
    for sr, sz in [(0.0, 0.0), (1.0, 1.0), (-2.0, 0.5)]:
        assert float(conservative_smax_4(sr, sz, beta=4.0)) <= max(sr, sz) + 1e-6

    # Erratum #3: 2M-Cross degraded speed cap strictly below 16.5 m/s
    v_cap_2m = compute_degraded_v_cap("2M-Cross", d_ar=1.75, r_max=58.0, horizon_s=6.0)
    assert 0.0 < v_cap_2m < 16.5

    # Verify 2D CEL LUT (41x29 x 3 sag tiers) matches z_req=(vz^2-4)/(2*a_net) within 0.35 m
    cel_path = tmp_path / "test_cel.vrta_cel2d"
    solve_and_export_cel_2d(cel_path)
    assert cel_path.stat().st_size == CEL2D_TOTAL_BYTES == 11_230
    hdr, data = load_cel_2d_lut(cel_path)
    assert hdr.magic == CEL2D_MAGIC
    assert data.shape == (3, 41, 29)
    for tier_idx, tier in enumerate(DEFAULT_CEL_TIERS):
        for vz_test in (-4.0, -6.0, -8.0, -10.0):
            z_lut = find_cel_zero_level_z(hdr, data, tier_idx, vz_test)
            z_ref = analytic_cel_z_req(vz_test, tier.a_net, v_td_max=2.0)
            assert abs(z_lut - z_ref) <= 0.35, (
                f"Tier {tier.name} vz={vz_test}: LUT z0={z_lut:.3f} m vs analytic {z_ref:.3f} m"
            )


def test_erratum1_cfl_dominates_hamiltonian_partials_all_regimes() -> None:
    """Verify |dH/dp_i| <= partial_max_magnitudes[i] across all 4 regimes & angles."""
    eps = 1e-4
    angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    test_states = [
        np.array([10.0, 0.0, 10.0, 0.0], dtype=np.float32),
        np.array([25.0, -20.0, 15.0, -8.0], dtype=np.float32),
        np.array([5.0, 12.0, 30.0, 6.0], dtype=np.float32),
    ]
    for spec in DEFAULT_REGIMES:
        dyn = VertiRta4DDynamics.from_regime_spec(spec)
        expected_ur = spec.a_thrust_max * math.sin(spec.tilt_max_rad)
        assert math.isclose(dyn.max_achievable_ur(), expected_ur, rel_tol=1e-6)
        for state in test_states:
            pm = np.asarray(dyn.partial_max_magnitudes(state, 0.0, 0.0, None), dtype=np.float64)
            for theta in angles:
                p = np.array(
                    [math.cos(theta), math.sin(theta), math.sin(2.0 * theta), math.cos(2.0 * theta)],
                    dtype=np.float64,
                )
                for dim in range(4):
                    dp = np.zeros(4, dtype=np.float64)
                    dp[dim] = eps
                    h_plus = float(dyn.hamiltonian(state, 0.0, 0.0, p + dp))
                    h_minus = float(dyn.hamiltonian(state, 0.0, 0.0, p - dp))
                    dh_dpi = abs((h_plus - h_minus) / (2.0 * eps))
                    assert dh_dpi <= pm[dim] + 1e-3, (
                        f"CFL violation in {spec.name} dim={dim}: |dH/dp|={dh_dpi:.4f} > pm={pm[dim]:.4f}"
                    )


def test_erratum2_conservative_smax_bounds_and_corner_zero() -> None:
    """Verify conservative_smax_4(0, 0) == 0.0 (not +0.173 m) and bounded below by max - ln(2)/beta."""
    beta = 4.0
    ln2_over_beta = math.log(2.0) / beta
    assert abs(float(conservative_smax_4(0.0, 0.0, beta=beta))) < 1e-7

    sr_grid = np.linspace(-5.0, 5.0, 41, dtype=np.float32)
    sz_grid = np.linspace(-5.0, 5.0, 41, dtype=np.float32)
    SR, SZ = np.meshgrid(sr_grid, sz_grid, indexing="ij")
    out = np.asarray(conservative_smax_4(SR, SZ, beta=beta))
    exact_max = np.maximum(SR, SZ)
    assert np.all(out <= exact_max + 1e-6)
    assert np.all(out >= exact_max - ln2_over_beta - 1e-5)
    # On diagonal sr == sz, conservative_smax_4 equals sr exactly
    diag = np.asarray(conservative_smax_4(sr_grid, sr_grid, beta=beta))
    assert np.allclose(diag, sr_grid, atol=1e-6)


def test_erratum3_degraded_v_cap_physics_and_monotonicity() -> None:
    """Verify v_cap_f ensures braking fits in horizon_s=6.0 s and r_max-R_safe=55.0 m."""
    v_nom = compute_degraded_v_cap("Nominal", d_ar=1.75, r_max=58.0, horizon_s=6.0)
    v_1m = compute_degraded_v_cap("1M", d_ar=1.75, r_max=58.0, horizon_s=6.0)
    v_low = compute_degraded_v_cap("Low-Vbat", d_ar=1.75, r_max=58.0, horizon_s=6.0)
    v_2m = compute_degraded_v_cap("2M-Cross", d_ar=1.75, r_max=58.0, horizon_s=6.0)

    assert math.isclose(v_nom, 22.0, abs_tol=1e-6)
    assert v_nom >= v_1m >= v_low >= v_2m > 0.0
    assert v_2m < 16.5
    assert v_low < 16.5

    # Verify stopping time & distance for 2M-Cross under d_ar=1.75 m/s^2, d_vr=0.35 m/s
    a_eff_2m = 9.80665 * math.tan(math.radians(22.0)) - 1.75
    t_stop = (v_2m + 0.35) / a_eff_2m
    d_stop = (v_2m * v_2m) / (2.0 * a_eff_2m) + 0.35 * (v_2m / a_eff_2m)
    assert t_stop <= 6.0 + 1e-6
    assert d_stop <= (58.0 - 3.0) + 1e-6


def test_vrta_lut_quick_fixture_binary_layout_sha256_and_eval(tmp_path: Path) -> None:
    """Verify exact 11,313,864 B layout, header fields, SHA-256 tamper rejection, and gradients."""
    out1 = tmp_path / "lut1.vrta_lut"
    out2 = tmp_path / "lut2.vrta_lut"
    solve_and_export_lut(out1, quick_cpu_fixture=True)
    solve_and_export_lut(out2, quick_cpu_fixture=True)

    # Exact byte size & bit-for-bit determinism
    assert out1.stat().st_size == LUT_TOTAL_BYTES == 11_313_864
    assert LUT_HEADER_BYTES == 4096
    assert LUT_PAYLOAD_BYTES == 11_309_768
    b1, b2 = out1.read_bytes(), out2.read_bytes()
    assert hashlib.sha256(b1).hexdigest() == hashlib.sha256(b2).hexdigest()

    hdr, data = load_vrta_lut(out1)
    assert hdr.magic == LUT_MAGIC
    assert hdr.version == 1
    assert hdr.num_regimes == 4
    assert hdr.dims == (41, 29, 41, 29)
    assert data.shape == (4, 41, 29, 41, 29)
    assert data.dtype == np.int16
    assert all(math.isfinite(e) and e >= 0.5 * hdr.scale for e in hdr.eps_grid)
    assert hdr.v_cap_f[0] == pytest.approx(22.0, abs=1e-4)
    assert hdr.v_cap_f[2] < 16.5
    assert hdr.v_cap_f[3] < 16.5

    # Regime ordering: Nominal >= 1M >= Low-Vbat >= 2M-Cross (quantized values)
    for r_hi, r_lo in [(0, 1), (1, 3), (3, 2)]:
        diff = data[r_hi].astype(np.int32) - data[r_lo].astype(np.int32)
        assert int(diff.min()) >= -1

    # Safe far-field vs deep unsafe & analytic gradient check vs finite differences
    val_safe, grad_safe = eval_cbvf_4d(hdr, data, 0, 15.0, -5.0, 20.0, -1.0)
    assert math.isfinite(val_safe) and val_safe > 5.0
    assert all(math.isfinite(g) for g in grad_safe)

    val_unsafe, _ = eval_cbvf_4d(hdr, data, 0, 0.5, -10.0, 0.5, -6.0)
    assert val_unsafe < -0.25

    # Overhead-intruder climb-braking sign check: approaching in z (vz=-5) gives dV/dvz > 0
    _, grad_ovh = eval_cbvf_4d(hdr, data, 0, 0.5, 0.0, 8.0, -5.0)
    assert grad_ovh[2] > 0.0  # dV/dz > 0
    assert grad_ovh[3] > 0.0  # dV/dvz > 0

    # Analytic quadrilinear gradient matches central finite differences inside a cell
    q = (12.4, -7.3, 9.2, -3.4)
    _, g_ana = eval_cbvf_4d(hdr, data, 1, *q)
    h = 1e-4
    for d in range(4):
        qp = list(q)
        qm = list(q)
        qp[d] += h
        qm[d] -= h
        vp, _ = eval_cbvf_4d(hdr, data, 1, *qp)
        vm, _ = eval_cbvf_4d(hdr, data, 1, *qm)
        assert math.isclose(g_ana[d], (vp - vm) / (2.0 * h), rel_tol=1e-4, abs_tol=1e-4)

    # Corrupt payload byte must fail SHA-256 verification
    bad_path = tmp_path / "corrupt.vrta_lut"
    corrupt = bytearray(b1)
    corrupt[LUT_HEADER_BYTES + 1234] ^= 0xFF
    bad_path.write_bytes(corrupt)
    with pytest.raises(ValueError, match="SHA-256"):
        load_vrta_lut(bad_path)


def test_cel_2d_lut_tamper_and_monotonicity(tmp_path: Path) -> None:
    """Verify exact 11,230 B .vrta_cel2d layout, sag monotonicity, and SHA-256 check."""
    cel_path = tmp_path / "cel.vrta_cel2d"
    solve_and_export_cel_2d(cel_path)
    assert cel_path.stat().st_size == CEL2D_TOTAL_BYTES == 11_230
    assert CEL2D_HEADER_BYTES == 4096
    assert CEL2D_PAYLOAD_BYTES == 7134

    hdr, data = load_cel_2d_lut(cel_path)
    assert hdr.magic == CEL2D_MAGIC
    assert hdr.num_tiers == 3
    assert hdr.dims == (41, 29)
    assert hdr.v_td_max == pytest.approx(2.0, abs=1e-5)
    assert hdr.d_z_max == pytest.approx(0.5, abs=1e-5)

    # Higher T/W tier must have >= safety value than lower T/W tier everywhere
    assert int((data[0].astype(np.int32) - data[1].astype(np.int32)).min()) >= -2
    assert int((data[1].astype(np.int32) - data[2].astype(np.int32)).min()) >= -2

    # Safe hover at 20 m AGL, vz=0 vs unsafe fast descent at 5 m AGL, vz=-8 m/s
    v_safe, g_safe = eval_cel_2d(hdr, data, 0, 20.0, 0.0)
    v_unsafe, g_unsafe = eval_cel_2d(hdr, data, 2, 5.0, -8.0)
    assert v_safe > 5.0
    assert v_unsafe < -5.0
    assert g_unsafe[0] > 0.0  # higher z increases margin
    assert g_unsafe[1] > 0.0  # less negative vz increases margin

    # Corrupt byte must fail SHA-256 check
    bad = bytearray(cel_path.read_bytes())
    bad[CEL2D_HEADER_BYTES + 42] ^= 0x55
    bad_path = tmp_path / "corrupt.vrta_cel2d"
    bad_path.write_bytes(bad)
    with pytest.raises(ValueError, match="SHA-256"):
        load_cel_2d_lut(bad_path)


def test_vehicle_yaml_contract_and_soundness_gate(tmp_path: Path) -> None:
    """Verify config/vehicle_x8_240.yaml loading and end-to-end soundness verifier."""
    params = load_vehicle_params()
    assert params["mass_kg"] == pytest.approx(240.0)
    assert params["gravity_m_s2"] == pytest.approx(9.80665)
    assert params["tw_nominal"] == pytest.approx(2.5)
    assert params["v_max_mps"] == pytest.approx(22.0)

    lut_path = tmp_path / "s.vrta_lut"
    cel_path = tmp_path / "s.vrta_cel2d"
    solve_and_export_lut(lut_path, quick_cpu_fixture=True)
    solve_and_export_cel_2d(cel_path)
    report = verify_all_soundness(lut_path, cel_path)
    assert report["passed"] is True
    assert report["lut_bytes"] == 11_313_864
    assert report["cel2d_bytes"] == 11_230
    assert report["cel2d_max_curve_err_m"] <= 0.35


def test_committed_quick_fixtures_present_and_sound() -> None:
    """Verify generated fixtures under tests/fixtures/ match exact byte sizes & soundness."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    lut_fix = repo_root / "tests" / "fixtures" / "quick_test.vrta_lut"
    cel_fix = repo_root / "tests" / "fixtures" / "quick_test.vrta_cel2d"
    assert lut_fix.is_file(), f"Missing fixture {lut_fix}"
    assert cel_fix.is_file(), f"Missing fixture {cel_fix}"
    assert lut_fix.stat().st_size == 11_313_864
    assert cel_fix.stat().st_size == 11_230

    report = verify_all_soundness(lut_fix, cel_fix)
    assert report["passed"] is True
    assert report["cel2d_max_curve_err_m"] <= 0.35
    assert report["erratum1_nom_max_ur"] >= 14.06
    assert report["erratum2_smax_zero"] <= 1e-6
    assert report["lut_v_cap_f"][2] < 16.5
