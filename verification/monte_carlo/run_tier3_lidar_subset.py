"""Tier-3 Verification Pyramid Runner: 50-Run Headless EGL gpu_lidar Full-Sensor Subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from verification.monte_carlo.atlas_stats import clopper_pearson_and_wilson


ESDF_P99_BUDGET_MS = 2.0
TIER3_TARGET_COUNT = 50
TIER3_BOUNDARY_COUNT = 25


def _margin_sort_key(row: dict[str, Any]) -> tuple[int, float, int]:
    failed_priority = 0 if not bool(row.get("passed", True)) else 1
    margin = float(row.get("min_cbvf_margin_m", row.get("margin_m", 0.0)))
    return (failed_priority, margin, int(row["run_id"]))


def select_tier3_50_scenarios(tier2_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select exact 50 Tier-3 scenarios: 25 lowest-margin/failed + 25 stratified representative."""
    if len(tier2_results) < TIER3_TARGET_COUNT:
        raise ValueError(
            f"Need at least {TIER3_TARGET_COUNT} Tier-2 results to select Tier-3 subset, got {len(tier2_results)}"
        )

    ranked = sorted(tier2_results, key=_margin_sort_key)
    selected: list[dict[str, Any]] = list(ranked[:TIER3_BOUNDARY_COUNT])
    selected_ids = {int(r["run_id"]) for r in selected}

    # Group remaining candidates by stratum and round-robin for balanced coverage
    remaining_by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(tier2_results, key=lambda r: int(r["run_id"])):
        if int(row["run_id"]) in selected_ids:
            continue
        remaining_by_stratum.setdefault(str(row.get("stratum", "S1")), []).append(row)

    strata_order = sorted(remaining_by_stratum.keys())
    cursors = {s: 0 for s in strata_order}
    while len(selected) < TIER3_TARGET_COUNT:
        progressed = False
        for s in strata_order:
            if len(selected) >= TIER3_TARGET_COUNT:
                break
            pool = remaining_by_stratum[s]
            if cursors[s] < len(pool):
                cand = pool[cursors[s]]
                cursors[s] += 1
                selected.append(cand)
                selected_ids.add(int(cand["run_id"]))
                progressed = True
        if not progressed:
            break

    return selected


def build_egl_lidar_worker_cmd(world_name: str = "kartal_coast", worker_id: int = 0) -> list[str]:
    """Build headless EGL Gazebo Harmonic gpu_lidar worker invocation (2 workers, 10-13 min)."""
    if worker_id < 0 or worker_id >= 8:
        raise ValueError(f"worker_id must be in [0, 7], got {worker_id}")
    return [
        "env",
        f"ROS_DOMAIN_ID={20 + worker_id}",
        f"GZ_PARTITION=vertirta_egl_w{worker_id}",
        "DISPLAY=",
        "gz",
        "sim",
        "-s",
        "-r",
        "--headless-rendering",
        f"sim/worlds/{world_name}.sdf",
    ]


def summarize_tier3_results(
    results: list[dict[str, Any]],
    esdf_p99_ms: float,
    out_path: Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Emit Sn6 tier3_50_lidar_cliff_ledger.json verifying ROG-Map ESDF p99 < 2.0 ms."""
    n = len(results)
    k = sum(1 for r in results if not bool(r.get("passed", True)))
    stats = clopper_pearson_and_wilson(k=k, n=n, alpha=alpha)
    esdf_ok = esdf_p99_ms < ESDF_P99_BUDGET_MS

    report: dict[str, Any] = {
        "schema_version": "vertirta.tier3_lidar_subset/v1.0.0",
        "total_runs": n,
        "failures": k,
        "stats": stats,
        "esdf_p99_ms": esdf_p99_ms,
        "esdf_p99_budget_ms": ESDF_P99_BUDGET_MS,
        "esdf_gate_passed": esdf_ok,
        "passed": (n == TIER3_TARGET_COUNT) and esdf_ok,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and prepare Tier-3 50-run EGL gpu_lidar subset.")
    parser.add_argument("--tier2-runs", type=Path, required=True)
    parser.add_argument("--world", default="kartal_coast")
    parser.add_argument("--out-dir", type=Path, default=Path("build/atlas_tier3"))
    args = parser.parse_args()

    payload = json.loads(args.tier2_runs.read_text(encoding="utf-8"))
    rows = payload["runs"] if isinstance(payload, dict) and "runs" in payload else payload
    subset = select_tier3_50_scenarios(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tier3_50_manifest.json").write_text(
        json.dumps(subset, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    cmds = [build_egl_lidar_worker_cmd(args.world, k) for k in range(2)]
    (args.out_dir / "tier3_egl_commands.json").write_text(
        json.dumps(cmds, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"selected_runs": len(subset), "egl_workers": len(cmds)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
