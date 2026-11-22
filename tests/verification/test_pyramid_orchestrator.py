"""TDD verification tests for Plan 1B Task 5: 2-Tier Verification Pyramid & Survival Atlas Plots."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest

from verification.monte_carlo.cliff_ledger import validate_cliff_ledger
from verification.monte_carlo.lhs_sampler import generate_stratified_manifest
from verification.monte_carlo.plot_survival_atlas import (
    plot_ab_cdf,
    plot_survival_heatmap,
)
from verification.monte_carlo.run_tier1_6dof_atlas import (
    run_tier1_atlas,
    subsample_manifest_stratified,
)
from verification.monte_carlo.run_tier2_gazebo_atlas import (
    FORBIDDEN_PORTS,
    build_gz_hot_reset_cmd,
    build_worker_env,
    partition_worker_manifests,
    summarize_tier2_results,
)
from verification.monte_carlo.run_tier3_lidar_subset import (
    build_egl_lidar_worker_cmd,
    select_tier3_50_scenarios,
    summarize_tier3_results,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ODD_PATH = REPO_ROOT / "verification/ODD.yaml"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_hot_reset_uses_model_only_and_worker_env_isolates_ports() -> None:
    joined = " ".join(build_gz_hot_reset_cmd(world_name="kartal_coast", worker_id=2))
    assert "/world/kartal_coast/control" in joined
    assert "reset: {model_only: true}" in joined
    assert "all: true" not in joined and "time_only: true" not in joined

    seen_ports: set[int] = set()
    for k in range(4):
        env = build_worker_env(k)
        assert env["ROS_DOMAIN_ID"] == str(10 + k)
        assert env["GZ_PARTITION"] == f"vertirta_w{k}"
        assert env["VERTIRTA_MODEL_NAME"] == f"vertirta_x8_240_{k}"
        port = int(env["VERTIRTA_PX4_UDP_PORT"])
        assert port == 8900 + k and port not in FORBIDDEN_PORTS
        seen_ports.add(port)
    assert len(seen_ports) == 4


def test_worker_partition_and_tier3_50_selection_and_egl_cmd(tmp_path: Path) -> None:
    strata = ["S1", "S2", "S3", "S4", "S5", "S6"]
    rows = [
        {
            "run_id": i,
            "stratum": "S6" if i < 25 else strata[i % len(strata)],
            "outcome_class": "CEL" if (i >= 25 and strata[i % len(strata)] == "S4") else "CSFL",
            "min_cbvf_margin_m": float(i) * 0.1 - 1.0,
            "passed": i >= 10,
        }
        for i in range(120)
    ]
    shards = partition_worker_manifests(rows, num_workers=4)
    assert len(shards) == 4 and sum(len(s) for s in shards) == 120
    assert {r["run_id"] for s in shards for r in s} == set(range(120))

    subset = select_tier3_50_scenarios(rows)
    assert len(subset) == 50
    subset_ids = {r["run_id"] for r in subset}
    assert len(subset_ids) == 50
    # All 25 lowest-margin / failed boundary cases MUST be included
    assert set(range(25)).issubset(subset_ids)

    egl_cmd = " ".join(build_egl_lidar_worker_cmd(world_name="kartal_coast", worker_id=1))
    assert "--headless-rendering" in egl_cmd and "kartal_coast.sdf" in egl_cmd

    t3 = summarize_tier3_results(subset, esdf_p99_ms=1.45, out_path=tmp_path / "tier3.json")
    assert t3["total_runs"] == 50 and t3["esdf_gate_passed"] is True
    t3_slow = summarize_tier3_results(subset, esdf_p99_ms=2.35, out_path=tmp_path / "tier3_slow.json")
    assert t3_slow["esdf_gate_passed"] is False


def test_tier1_proportional_subsample_and_cli_ab_integration(tmp_path: Path) -> None:
    full_1k = generate_stratified_manifest(ODD_PATH, tier="tier2_1k", seed=20261122)
    sub_100 = subsample_manifest_stratified(full_1k, max_runs=100)
    assert len(sub_100) == 100
    counts: dict[str, int] = {}
    for row in sub_100:
        counts[row["stratum"]] = counts.get(row["stratum"], 0) + 1
    assert counts == {"S1": 10, "S2": 25, "S3": 25, "S4": 15, "S5": 15, "S6": 10}

    # Regression lock: 1 <= max_runs < len(active_strata) MUST terminate cleanly with exact length k
    for k in (1, 3, 5):
        sub_k = subsample_manifest_stratified(full_1k, max_runs=k)
        assert len(sub_k) == k
        assert len({r["run_id"] for r in sub_k}) == k
    assert {r["stratum"] for r in subsample_manifest_stratified(full_1k, max_runs=3)} == {"S2", "S3", "S4"}

    # Mock vertirta_6dof_batch_cli honoring exact Wave-3 CLI & JSON schema
    mock_cli = tmp_path / "mock_6dof_batch.py"
    mock_cli.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--manifest', required=True)\n"
        "p.add_argument('--out', required=True)\n"
        "p.add_argument('--lut', default='')\n"
        "p.add_argument('--cel-lut', default='')\n"
        "p.add_argument('--cbf-mode', default='hj_cbvf')\n"
        "p.add_argument('--alloc-mode', default='indi_wls')\n"
        "p.add_argument('--vrta-dir', default='')\n"
        "p.add_argument('--max-runs', type=int, default=0)\n"
        "a = p.parse_args()\n"
        "rows = json.loads(Path(a.manifest).read_text(encoding='utf-8'))\n"
        "if a.max_runs > 0: rows = rows[:a.max_runs]\n"
        "out_runs = []\n"
        "for r in rows:\n"
        "    is_s6_fail = (r['stratum'] == 'S6' and r['mean_wind_mps'] > 16.0)\n"
        "    is_hocbf_fail = (a.cbf_mode == 'hocbf' and r['stratum'] == 'S3' and r['run_id'] % 5 == 0)\n"
        "    is_stock_fail = (a.alloc_mode == 'stock_ca' and r['stratum'] == 'S4')\n"
        "    passed = not (is_s6_fail or is_hocbf_fail or is_stock_fail)\n"
        "    inv = None if passed else ('CTR_WALL_BREACH' if is_hocbf_fail else 'TOUCHDOWN_SINK_RATE_GT_2P5')\n"
        "    vrta = None if passed else f\"{a.vrta_dir}/run_{r['run_id']:05d}.vrta\"\n"
        "    out_runs.append({\n"
        "        'run_id': r['run_id'], 'stratum': r['stratum'], 'outcome_class': r['outcome_class'],\n"
        "        'fault_topology': r['fault_topology'], 'seed': r['seed'], 'passed': passed,\n"
        "        'violated_invariant': inv, 'min_cbvf_margin_m': 1.8 if a.cbf_mode == 'hj_cbvf' else 0.4,\n"
        "        'touchdown_vz_mps': 0.7 if passed else 4.2, 'max_tilt_deg': 12.0,\n"
        "        'min_v_term_v': 368.0, 'cel_latched': r['stratum'] == 'S4',\n"
        "        'mean_yaw_rate_rad_s': 2.2 if r['stratum'] == 'S4' else 0.05, 'vrta_path': vrta,\n"
        "    })\n"
        "Path(a.out).write_text(json.dumps({'schema_version':'vertirta.tier1_batch/v1.0.0','cbf_mode':a.cbf_mode,'alloc_mode':a.alloc_mode,'runs':out_runs}), encoding='utf-8')\n",
        encoding="utf-8",
        newline="\n",
    )

    out_dir = tmp_path / "atlas_out"
    report = run_tier1_atlas(
        odd_path=ODD_PATH,
        out_dir=out_dir,
        batch_cmd=[sys.executable, str(mock_cli)],
        tier="tier2_1k",
        max_runs=100,
        seed=20261122,
        run_ab=True,
    )
    assert report["summary"]["csfl_pool"]["n"] == 65
    assert report["summary"]["cel_pool"]["n"] == 15
    assert "ab_comparison" in report
    assert set(report["ab_comparison"].keys()) == {"hj_cbvf__indi_wls", "hocbf__indi_wls", "hj_cbvf__stock_ca"}
    # Stock CA MUST show worse CEL survival than INDI+WLS on S4
    assert report["ab_comparison"]["hj_cbvf__indi_wls"]["cel_pass_rate"] > report["ab_comparison"]["hj_cbvf__stock_ca"]["cel_pass_rate"]
    ledger = json.loads((out_dir / "failure_cliff_ledger.json").read_text(encoding="utf-8"))
    validate_cliff_ledger(ledger)

    # Also verify Tier-2 summary helper produces isolated Sn5 report
    t2_summary, t2_ledger = summarize_tier2_results(report["primary_runs"], out_dir=tmp_path / "t2")
    assert t2_summary["csfl_pool"]["n"] == 65 and t2_summary["cel_pool"]["n"] == 15
    validate_cliff_ledger(t2_ledger)


def test_plot_survival_atlas_generates_valid_pngs(tmp_path: Path) -> None:
    runs = [
        {
            "run_id": i,
            "stratum": ["S2", "S3", "S4", "S5", "S6"][i % 5],
            "mean_wind_mps": 2.0 + (i % 15),
            "initial_soc_pct": 25.0 + 3.0 * (i % 20),
            "min_cbvf_margin_m": 0.5 + 0.05 * i,
            "touchdown_vz_mps": 0.7 if i % 7 else 2.9,
            "passed": (i % 7 != 0),
        }
        for i in range(60)
    ]
    hm_path = plot_survival_heatmap(runs, tmp_path / "survival_heatmap.png")
    cdf_path = plot_ab_cdf(
        {
            "hj_cbvf__indi_wls": runs,
            "hocbf__indi_wls": [{**r, "min_cbvf_margin_m": r["min_cbvf_margin_m"] - 0.8} for r in runs],
        },
        tmp_path / "ab_cdf.png",
    )
    assert hm_path.read_bytes()[:8] == PNG_MAGIC and hm_path.stat().st_size > 2000
    assert cdf_path.read_bytes()[:8] == PNG_MAGIC and cdf_path.stat().st_size > 2000
