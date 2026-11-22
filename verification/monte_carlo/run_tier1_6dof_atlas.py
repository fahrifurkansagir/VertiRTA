"""Tier-1 Verification Pyramid Runner: 10k C++20 6-DOF Stratified Survival Atlas + A/B Ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from verification.monte_carlo.atlas_stats import summarize_atlas_pools
from verification.monte_carlo.cliff_ledger import (
    ALLOWED_ROOT_CAUSE_CLASSES,
    write_failure_cliff_ledger,
)
from verification.monte_carlo.lhs_sampler import (
    TIER2_STRATA_COUNTS,
    generate_stratified_manifest,
)


DEFAULT_CLI_CANDIDATES = (
    Path("build/release-linux-x64/tools/cli/vertirta_6dof_batch_cli"),
    Path("build/dev-linux-x64/tools/cli/vertirta_6dof_batch_cli"),
    Path("build/tools/cli/vertirta_6dof_batch_cli"),
)


def subsample_manifest_stratified(
    manifest: list[dict[str, Any]],
    max_runs: int,
) -> list[dict[str, Any]]:
    """Proportionally subsample S1..S6 so CSFL (S2+S3+S5) & CEL (S4) pools stay populated."""
    if max_runs <= 0 or max_runs >= len(manifest):
        return list(manifest)

    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        by_stratum.setdefault(str(row["stratum"]), []).append(row)

    total_weight = sum(TIER2_STRATA_COUNTS.values())
    order = list(TIER2_STRATA_COUNTS.keys())
    alloc: dict[str, int] = {}
    for s in order:
        avail = len(by_stratum.get(s, []))
        if avail == 0:
            alloc[s] = 0
            continue
        target = max(1, (max_runs * TIER2_STRATA_COUNTS[s]) // total_weight)
        alloc[s] = min(avail, target)

    # Adjust rounding remainder deterministically across strata
    while sum(alloc.values()) < max_runs:
        progressed = False
        for s in order:
            if sum(alloc.values()) >= max_runs:
                break
            if alloc[s] < len(by_stratum.get(s, [])):
                alloc[s] += 1
                progressed = True
        if not progressed:
            break
    trim_zero_order = sorted(order, key=lambda s: (TIER2_STRATA_COUNTS[s], -order.index(s)))
    while sum(alloc.values()) > max_runs:
        progressed = False
        for s in reversed(order):
            if sum(alloc.values()) <= max_runs:
                break
            if alloc[s] > 1:
                alloc[s] -= 1
                progressed = True
        if not progressed:
            # Fewer requested runs than active strata: trim smallest-weight strata to 0
            for s in trim_zero_order:
                if sum(alloc.values()) <= max_runs:
                    break
                if alloc[s] > 0:
                    alloc[s] -= 1
                    progressed = True
        if not progressed:
            break

    selected: list[dict[str, Any]] = []
    for s in order:
        rows_s = by_stratum.get(s, [])
        k = alloc[s]
        if k <= 0:
            continue
        if k >= len(rows_s):
            selected.extend(rows_s)
        else:
            # Evenly stride across the LHS stratum to preserve full parameter-space coverage
            stride = len(rows_s) / k
            selected.extend(rows_s[int(i * stride)] for i in range(k))
    return selected


def _classify_root_cause(violated_invariant: str | None) -> str:
    if not violated_invariant:
        return "CONTROL_AUTHORITY_CLIFF"
    upper = violated_invariant.upper()
    if any(tok in upper for tok in ("CTR", "LTFJ", "LTBX", "GEOFENCE", "BARRIER")):
        return "CTR_BARRIER_VIOLATION"
    if any(tok in upper for tok in ("SOC", "VOLT", "BATT", "ENERGY", "RESERVE")):
        return "ENERGY_RESERVE_EXHAUSTION"
    if any(tok in upper for tok in ("TMR", "IMU")):
        return "TMR_ISOLATION_TIMEOUT"
    if "QP" in upper:
        return "QP_INFEASIBLE"
    if "TIMEOUT" in upper:
        return "SIM_TIMEOUT"
    return "CONTROL_AUTHORITY_CLIFF"


def _enrich_runs_for_assurance(
    raw_runs: list[dict[str, Any]],
    manifest_by_id: dict[int, dict[str, Any]],
    vrta_dir: Path,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for r in raw_runs:
        rid = int(r["run_id"])
        m = manifest_by_id.get(rid, {})
        passed = bool(r["passed"])
        violated = r.get("violated_invariant")
        if not passed and not violated:
            violated = "UNSPECIFIED_INVARIANT"
        vrta_path = r.get("vrta_path")
        if not passed and (not vrta_path or not str(vrta_path).endswith(".vrta")):
            vrta_path = (vrta_dir / f"run_{rid:05d}.vrta").as_posix()
        root_cause = r.get("root_cause_class")
        if root_cause not in ALLOWED_ROOT_CAUSE_CLASSES:
            root_cause = _classify_root_cause(violated)

        row = dict(r)
        row["mean_wind_mps"] = m.get("mean_wind_mps", 0.0)
        row["initial_soc_pct"] = m.get("initial_soc_pct", 80.0)
        row["initial_state"] = {
            "mean_wind_mps": m.get("mean_wind_mps", 0.0),
            "peak_gust_mps": m.get("peak_gust_mps", 0.0),
            "initial_soc_pct": m.get("initial_soc_pct", 80.0),
            "cruise_speed_mps": m.get("cruise_speed_mps", 0.0),
            "start_lat_deg": m.get("start_lat_deg", 40.8860),
            "start_lon_deg": m.get("start_lon_deg", 29.1880),
            "exposure": m.get("exposure", "coastal_d"),
        }
        row["violated_invariant"] = violated
        row["root_cause_class"] = root_cause
        row["vrta_path"] = vrta_path
        enriched.append(row)
    return enriched


def _invoke_batch_cli(
    batch_cmd: list[str],
    manifest_path: Path,
    out_json: Path,
    vrta_dir: Path,
    cbf_mode: str,
    alloc_mode: str,
    lut_path: Path | None,
    cel_lut_path: Path | None,
) -> dict[str, Any]:
    vrta_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *batch_cmd,
        "--manifest",
        str(manifest_path),
        "--out",
        str(out_json),
        "--cbf-mode",
        cbf_mode,
        "--alloc-mode",
        alloc_mode,
        "--vrta-dir",
        str(vrta_dir),
    ]
    if lut_path is not None and lut_path.exists():
        cmd.extend(["--lut", str(lut_path)])
    if cel_lut_path is not None and cel_lut_path.exists():
        cmd.extend(["--cel-lut", str(cel_lut_path)])
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text(encoding="utf-8"))


def _ab_metrics_from_runs(runs: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    csfl = summary["csfl_pool"]
    cel = summary["cel_pool"]
    margins = [float(r.get("min_cbvf_margin_m", 0.0)) for r in runs]
    vz = [abs(float(r.get("touchdown_vz_mps", 0.0))) for r in runs]
    return {
        "total_runs": len(runs),
        "csfl_pass_rate": round(1.0 - csfl["p_hat"], 6),
        "csfl_cp_one_sided_upper": csfl["cp_one_sided_upper"],
        "cel_pass_rate": round(1.0 - cel["p_hat"], 6),
        "cel_cp_one_sided_upper": cel["cp_one_sided_upper"],
        "mean_min_cbvf_margin_m": round(sum(margins) / max(len(margins), 1), 4),
        "mean_touchdown_vz_mps": round(sum(vz) / max(len(vz), 1), 4),
    }


def run_tier1_atlas(
    odd_path: Path,
    out_dir: Path,
    batch_cmd: list[str],
    tier: str = "tier1_10k",
    max_runs: int = 0,
    seed: int = 20261122,
    run_ab: bool = False,
    lut_path: Path | None = None,
    cel_lut_path: Path | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute Tier-1 6-DOF Atlas (+ optional A/B ablations), writing Sn4 summary & cliff ledger."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_stratified_manifest(odd_path, tier=tier, seed=seed)
    if max_runs > 0:
        manifest = subsample_manifest_stratified(manifest, max_runs=max_runs)

    manifest_path = out_dir / f"manifest_{tier}_{len(manifest)}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest_by_id = {int(r["run_id"]): r for r in manifest}

    configs = [("hj_cbvf", "indi_wls")]
    if run_ab:
        configs.extend([("hocbf", "indi_wls"), ("hj_cbvf", "stock_ca")])

    ab_comparison: dict[str, dict[str, Any]] = {}
    ab_runs: dict[str, list[dict[str, Any]]] = {}
    primary_runs: list[dict[str, Any]] = []
    primary_summary: dict[str, Any] = {}

    for cbf_mode, alloc_mode in configs:
        tag = f"{cbf_mode}__{alloc_mode}"
        raw_out = out_dir / f"raw_{tag}.json"
        vrta_dir = out_dir / "vrta" / tag
        raw_payload = _invoke_batch_cli(
            batch_cmd=batch_cmd,
            manifest_path=manifest_path,
            out_json=raw_out,
            vrta_dir=vrta_dir,
            cbf_mode=cbf_mode,
            alloc_mode=alloc_mode,
            lut_path=lut_path,
            cel_lut_path=cel_lut_path,
        )
        runs = _enrich_runs_for_assurance(raw_payload["runs"], manifest_by_id, vrta_dir)
        summary = summarize_atlas_pools(runs, alpha=alpha)
        ab_comparison[tag] = _ab_metrics_from_runs(runs, summary)
        ab_runs[tag] = runs
        if tag == "hj_cbvf__indi_wls":
            primary_runs = runs
            primary_summary = summary

    ledger = write_failure_cliff_ledger(primary_runs, out_dir / "failure_cliff_ledger.json")

    report: dict[str, Any] = {
        "schema_version": "vertirta.tier1_atlas_report/v1.0.0",
        "tier": tier,
        "seed": seed,
        "runs_executed": len(primary_runs),
        "summary": primary_summary,
        "cliff_ledger_sha256": ledger["sha256"],
        "total_failures": ledger["total_failures"],
        "ab_comparison": ab_comparison,
        "primary_runs": primary_runs,
        "ab_runs": ab_runs,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps({k: v for k, v in report.items() if k not in {"primary_runs", "ab_runs"}}, sort_keys=True).encode("utf-8")
    ).hexdigest()

    sn4_path = out_dir / "tier1_10k_clopper_pearson.json"
    sn4_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def _discover_batch_cli(explicit_bin: Path | None) -> Path:
    if explicit_bin is not None:
        return explicit_bin
    for cand in DEFAULT_CLI_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "vertirta_6dof_batch_cli not found; pass --bin <path_to_vertirta_6dof_batch_cli>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tier-1 6-DOF Stratified Survival Envelope Atlas.")
    parser.add_argument("--odd", type=Path, default=Path("verification/ODD.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("build/atlas_tier1"))
    parser.add_argument("--bin", dest="batch_bin", type=Path, default=None)
    parser.add_argument("--tier", choices=("tier1_10k", "tier2_1k"), default="tier1_10k")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--lut", type=Path, default=Path("tests/fixtures/quick_test.vrta_lut"))
    parser.add_argument("--cel-lut", type=Path, default=Path("tests/fixtures/quick_test.vrta_cel2d"))
    parser.add_argument("--run-ab", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cli_bin = _discover_batch_cli(args.batch_bin)
    report = run_tier1_atlas(
        odd_path=args.odd,
        out_dir=args.out_dir,
        batch_cmd=[str(cli_bin)],
        tier=args.tier,
        max_runs=args.max_runs,
        seed=args.seed,
        run_ab=args.run_ab,
        lut_path=args.lut if args.lut.exists() else None,
        cel_lut_path=args.cel_lut if args.cel_lut.exists() else None,
    )
    if args.plot:
        from verification.monte_carlo.plot_survival_atlas import plot_ab_cdf, plot_survival_heatmap

        plot_survival_heatmap(report["primary_runs"], args.out_dir / "survival_heatmap.png")
        plot_ab_cdf(report["ab_runs"], args.out_dir / "ab_cdf.png")

    print(
        json.dumps(
            {
                "runs_executed": report["runs_executed"],
                "total_failures": report["total_failures"],
                "csfl_pool": report["summary"]["csfl_pool"],
                "cel_pool": report["summary"]["cel_pool"],
                "ab_comparison": report["ab_comparison"],
                "sha256": report["sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
