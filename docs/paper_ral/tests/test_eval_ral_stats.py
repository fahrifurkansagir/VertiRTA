"""Unit tests for dual-mode eval_ral_stats.py (paired per-seed mode & conservative Fréchet-Hoeffding mode)."""

from __future__ import annotations

import json
from pathlib import Path

from docs.paper_ral.scripts.eval_ral_stats import (
    compute_iqm_bootstrap,
    compute_mcnemar_holm_bonferroni,
    evaluate_paired_runs,
    evaluate_v1_atlas,
    render_ral_tables_tex,
)


REPO = Path(__file__).resolve().parents[3]


def test_paired_per_seed_mode_exact_mcnemar_and_iqm(tmp_path: Path) -> None:
    """Verify evaluate_paired_runs computes exact (b, c), Holm-Bonferroni, and real bootstrap IQM."""
    n = 120
    runs_primary = [
        {
            "run_id": i,
            "stratum": "S3" if i < 90 else "S4",
            "outcome_class": "CSFL" if i < 90 else "CEL",
            "passed": i >= 3,  # 87/90 CSFL, 30/30 CEL
            "min_cbvf_margin_m": 1.5 + 0.01 * i,
            "touchdown_vz_mps": 0.65 + 0.002 * i,
        }
        for i in range(n)
    ]
    runs_hocbf = [
        {
            "run_id": i,
            "stratum": "S3" if i < 90 else "S4",
            "outcome_class": "CSFL" if i < 90 else "CEL",
            "passed": (i >= 45) if i < 90 else True,  # 45/90 CSFL, 30/30 CEL
            "min_cbvf_margin_m": 0.4 + 0.01 * i,
            "touchdown_vz_mps": 0.85 + 0.002 * i,
        }
        for i in range(n)
    ]
    runs_stock = [
        {
            "run_id": i,
            "stratum": "S3" if i < 90 else "S4",
            "outcome_class": "CSFL" if i < 90 else "CEL",
            "passed": (i >= 3) if i < 90 else False,  # 87/90 CSFL, 0/30 CEL
            "min_cbvf_margin_m": 1.5 + 0.01 * i,
            "touchdown_vz_mps": 0.65 + 0.002 * i,
        }
        for i in range(n)
    ]

    report = evaluate_paired_runs(
        primary_runs=runs_primary,
        baseline_runs={"hocbf__indi_wls": runs_hocbf, "hj_cbvf__stock_ca": runs_stock},
        alpha=0.05,
        n_resamples=500,
        seed=42,
    )
    assert report["method"] == "paired_per_seed_exact"
    assert report["iqm_status"] == "COMPUTED_FROM_PER_RUN_VECTORS"
    assert report["csfl_n"] == 90 and report["cel_n"] == 30
    by_name = {p["name"]: p for p in report["holm_bonferroni_pairs"]}
    assert by_name["csfl:hj_cbvf__indi_wls_vs_hocbf__indi_wls"]["b"] == 42
    assert by_name["csfl:hj_cbvf__indi_wls_vs_hocbf__indi_wls"]["c"] == 0
    assert by_name["csfl:hj_cbvf__indi_wls_vs_hocbf__indi_wls"]["significant_holm"] is True
    assert by_name["cel:hj_cbvf__indi_wls_vs_hj_cbvf__stock_ca"]["b"] == 30
    assert by_name["cel:hj_cbvf__indi_wls_vs_hj_cbvf__stock_ca"]["c"] == 0
    assert by_name["cel:hj_cbvf__indi_wls_vs_hj_cbvf__stock_ca"]["significant_holm"] is True

    iqm_margin = report["iqm_metrics"]["hj_cbvf__indi_wls"]["min_cbvf_margin_m"]
    assert iqm_margin["ci_low"] <= iqm_margin["iqm"] <= iqm_margin["ci_high"]
    assert iqm_margin["n"] == 120

    out_tex = render_ral_tables_tex(report, tmp_path / "paired_tables.tex")
    tex = out_tex.read_text(encoding="utf-8")
    assert "TBD" not in tex and "paired" in tex.lower()


def test_holm_bonferroni_step_down_monotonicity_and_zero_fabrication_v1_atlas() -> None:
    pairs = compute_mcnemar_holm_bonferroni(
        [
            {"name": "strong", "b": 80, "c": 2},
            {"name": "borderline", "b": 10, "c": 3},
            {"name": "tie", "b": 10, "c": 10},
        ],
        alpha=0.05,
    )
    by_name = {p["name"]: p for p in pairs}
    assert by_name["strong"]["significant_holm"] is True
    assert by_name["tie"]["significant_holm"] is False
    assert by_name["tie"]["p_holm"] == 1.0
    assert by_name["strong"]["p_holm"] <= by_name["borderline"]["p_holm"] <= by_name["tie"]["p_holm"]

    iqm = compute_iqm_bootstrap([10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1000.0], n_resamples=1000, seed=7)
    # Trimmed 25%-75% IQM MUST resist the 1000.0 outlier
    assert 4.0 <= iqm["iqm"] <= 5.5

    atlas_path = REPO / "artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json"
    cm7_path = REPO / "artifacts/v1.0.0/benchmarks/tier1_structural.json"
    summary = evaluate_v1_atlas(atlas_path, structural_path=cm7_path)
    assert summary["method"] == "frechet_hoeffding_conservative_bound"
    assert summary["iqm_status"] == "PENDING_PER_RUN_EXPORT"
    assert summary["iqm_metrics"] == {}
    assert summary["cortex_m7"]["flash_bytes"] == 28886
    assert summary["cortex_m7"]["ram_bytes"] == 0
    assert summary["cortex_m7"]["max_dag_bytes"] == 2960
    assert json.dumps(summary)  # JSON-serializable cleanly
