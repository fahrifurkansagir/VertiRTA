"""TDD verification tests for Plan 1B Task 4: LHS Sampler, Wind Model, Atlas Stats & Cliff Ledger."""

from __future__ import annotations

import json
import math
from pathlib import Path
import pytest

from verification.monte_carlo.atlas_stats import (
    clopper_pearson_and_wilson,
    min_zero_failure_trials,
    summarize_atlas_pools,
)
from verification.monte_carlo.cliff_ledger import (
    validate_cliff_ledger,
    write_failure_cliff_ledger,
)
from verification.monte_carlo.lhs_sampler import generate_stratified_manifest
from verification.monte_carlo.wind_model import (
    compute_dryden_parameters,
    compute_wind_profile_mps,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ODD_PATH = REPO_ROOT / "verification/ODD.yaml"


def test_wind_model_urban_never_exceeds_coastal_and_dryden_scales() -> None:
    for w20 in (5.1, 10.3, 12.9, 18.0):
        for z_agl in (15.0, 30.0, 60.0, 120.0):
            u_coast, sw_c, su_c = compute_wind_profile_mps(w20, z_agl, "coastal_d")
            u_urban, sw_u, su_u = compute_wind_profile_mps(w20, z_agl, "urban_b")
            # DR-4 erratum lock: urban mean wind MUST remain strictly below open-water coastal fetch
            assert u_urban < u_coast, f"Inverted wind shear at w20={w20}, z={z_agl}: {u_urban} >= {u_coast}"
            assert sw_u >= sw_c and su_u >= su_c

    d30 = compute_dryden_parameters(w20_mps=12.9, z_agl_m=30.0)
    assert math.isclose(d30["L_w_m"], 30.0, rel_tol=1e-3)
    assert math.isclose(d30["L_u_m"], 152.5, rel_tol=1e-2)
    assert math.isclose(d30["sigma_w_mps"], 1.29, rel_tol=1e-3)

    d60 = compute_dryden_parameters(w20_mps=12.9, z_agl_m=60.0)
    assert math.isclose(d60["L_w_m"], 60.0, rel_tol=1e-3)
    assert math.isclose(d60["L_u_m"], 219.7, rel_tol=1e-2)

    with pytest.raises(ValueError, match="exposure"):
        compute_wind_profile_mps(10.0, 30.0, "inverted_urban")
    with pytest.raises(ValueError, match="z_agl_m"):
        compute_wind_profile_mps(10.0, 0.0, "coastal_d")


def test_lhs_sampler_deterministic_and_balanced_strata() -> None:
    m1 = generate_stratified_manifest(ODD_PATH, tier="tier2_1k", seed=20261122)
    m2 = generate_stratified_manifest(ODD_PATH, tier="tier2_1k", seed=20261122)
    assert m1 == m2
    assert len(m1) == 1000

    counts: dict[str, int] = {}
    for row in m1:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
        assert row["start_lon_deg"] < 29.1900
        assert 0.0 <= row["mean_wind_mps"] <= 12.0 or row["stratum"] == "S6"
    assert counts == {"S1": 100, "S2": 250, "S3": 250, "S4": 150, "S5": 150, "S6": 100}

    s4_rows = [r for r in m1 if r["stratum"] == "S4"]
    s4_pairs = {tuple(r["failed_motors"]) for r in s4_rows}
    assert all(r["outcome_class"] == "CEL" for r in s4_rows)
    assert s4_pairs == {(0, 4), (1, 5), (2, 6), (3, 7)}

    s2_rows = [r for r in m1 if r["stratum"] == "S2"]
    assert all(r["outcome_class"] == "CSFL" for r in s2_rows)
    assert len({r["failed_motors"][0] for r in s2_rows}) == 8

    s3_rows = [r for r in m1 if r["stratum"] == "S3"]
    s3_pairs = {tuple(r["failed_motors"]) for r in s3_rows}
    assert all(r["outcome_class"] == "CSFL" for r in s3_rows)
    assert len(s3_pairs) == 24
    assert (0, 5) in s3_pairs and (0, 4) not in s3_pairs

    m10k = generate_stratified_manifest(ODD_PATH, tier="tier1_10k", seed=20261122)
    assert len(m10k) == 10000


def test_clopper_pearson_wilson_and_log1p_exact_bounds() -> None:
    s1000 = clopper_pearson_and_wilson(k=0, n=1000, alpha=0.05)
    assert math.isclose(s1000["cp_one_sided_upper"], 0.00299125, rel_tol=1e-4)
    assert math.isclose(s1000["cp_two_sided"][0], 0.0, abs_tol=1e-12)
    assert math.isclose(s1000["cp_two_sided"][1], 0.003682, rel_tol=1e-3)
    assert math.isclose(s1000["wilson_two_sided"][1], 0.003827, rel_tol=1e-3)

    s650 = clopper_pearson_and_wilson(k=0, n=650, alpha=0.05)
    assert math.isclose(s650["cp_one_sided_upper"], 0.0045979, rel_tol=1e-4)
    assert math.isclose(s650["cp_two_sided"][1], 0.005659, rel_tol=1e-3)
    assert math.isclose(s650["wilson_two_sided"][1], 0.005875, rel_tol=1e-3)

    s150 = clopper_pearson_and_wilson(k=0, n=150, alpha=0.05)
    assert math.isclose(s150["cp_one_sided_upper"], 0.019773, rel_tol=1e-3)

    s1_1000 = clopper_pearson_and_wilson(k=1, n=1000, alpha=0.05)
    assert math.isclose(s1_1000["cp_one_sided_upper"], 0.004735, rel_tol=1e-3)

    s5_250 = clopper_pearson_and_wilson(k=5, n=250, alpha=0.05)
    assert math.isclose(s5_250["cp_one_sided_upper"], 0.041590, rel_tol=1e-3)

    # Exact log1p zero-failure horizon proof for 10^-9 / FH (avoids naive log(1-1e-9) = 2,995,732,357 cancellation)
    assert min_zero_failure_trials(p_target=1e-9, alpha=0.05) == 2_995_732_273


def test_csfl_cel_denominators_strictly_isolated() -> None:
    manifest = generate_stratified_manifest(ODD_PATH, tier="tier2_1k", seed=20261122)
    rows = [
        {
            "run_id": r["run_id"],
            "stratum": r["stratum"],
            "outcome_class": r["outcome_class"],
            "passed": True,
        }
        for r in manifest
    ]
    summary = summarize_atlas_pools(rows, alpha=0.05)
    assert summary["csfl_pool"]["strata"] == ["S2", "S3", "S5"]
    assert summary["csfl_pool"]["n"] == 650
    assert summary["cel_pool"]["strata"] == ["S4"]
    assert summary["cel_pool"]["n"] == 150
    assert "S4" not in summary["csfl_pool"]["strata"]

    # Denominator contamination lock: tagging S4 as CSFL MUST raise ValueError
    poisoned = [dict(r) for r in rows]
    for r in poisoned:
        if r["stratum"] == "S4":
            r["outcome_class"] = "CSFL"
            break
    with pytest.raises(ValueError, match="denominator"):
        summarize_atlas_pools(poisoned, alpha=0.05)


def test_cliff_ledger_schema_and_vrta_links(tmp_path: Path) -> None:
    sample_results = [
        {
            "run_id": 12,
            "stratum": "S6",
            "outcome_class": "CEL",
            "seed": 99012,
            "passed": False,
            "initial_state": {
                "mean_wind_mps": 12.0,
                "peak_gust_mps": 8.0,
                "soc_pct": 21.5,
                "airspeed_mps": 19.0,
                "dist_ltfj_m": 1040.0,
            },
            "fault_topology": "FLT_2M_SAME_ARM",
            "fault_inject_time_s": 14.2,
            "violated_invariant": "TOUCHDOWN_SINK_RATE_GT_2P5",
            "root_cause_class": "CONTROL_AUTHORITY_CLIFF",
            "vrta_path": "artifacts/vrta/run_0012_s6.vrta",
        },
        {
            "run_id": 13,
            "stratum": "S2",
            "outcome_class": "CSFL",
            "seed": 99013,
            "passed": True,
            "vrta_path": "artifacts/vrta/run_0013_s2.vrta",
        },
    ]
    out_file = tmp_path / "failure_cliff_ledger.json"
    ledger = write_failure_cliff_ledger(sample_results, out_file)
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded == ledger
    assert ledger["schema_version"] == "vertirta.cliff_ledger/v1.0.0"
    assert ledger["total_runs"] == 2 and ledger["total_failures"] == 1
    assert ledger["entries"][0]["vrta_path"].endswith(".vrta")
    validate_cliff_ledger(ledger)

    # Missing .vrta link MUST fail schema validation
    broken = json.loads(out_file.read_text(encoding="utf-8"))
    broken["entries"][0]["vrta_path"] = "artifacts/logs/no_trace.txt"
    with pytest.raises(ValueError, match="\\.vrta"):
        validate_cliff_ledger(broken)
