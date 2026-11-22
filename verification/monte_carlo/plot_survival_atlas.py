"""Headless Agg Matplotlib Survival Envelope Heatmaps & A/B Ablation CDF Plotter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_survival_heatmap(runs: list[dict[str, Any]], out_png: Path) -> Path:
    """Render Fault-Stratum x Wind-Speed survival rate heatmap + Wind x SoC scatter map."""
    if not runs:
        raise ValueError("Cannot plot survival heatmap from empty runs")

    strata = ["S2", "S3", "S4", "S5", "S6"]
    wind_edges = np.array([0.0, 4.0, 8.0, 12.0, 15.0, 18.5])
    grid = np.full((len(strata), len(wind_edges) - 1), np.nan, dtype=float)

    for r_idx, s in enumerate(strata):
        s_runs = [r for r in runs if str(r.get("stratum")) == s]
        for c_idx in range(len(wind_edges) - 1):
            w_lo, w_hi = wind_edges[c_idx], wind_edges[c_idx + 1]
            bucket = [
                r for r in s_runs if w_lo <= float(r.get("mean_wind_mps", 0.0)) < w_hi
            ]
            if bucket:
                grid[r_idx, c_idx] = 100.0 * sum(1 for r in bucket if r.get("passed")) / len(bucket)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    im = ax1.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=100.0, aspect="auto")
    ax1.set_yticks(range(len(strata)))
    ax1.set_yticklabels(["S2 (1M CSFL)", "S3 (2M-X CSFL)", "S4 (Same-Arm CEL)", "S5 (3-IMU TMR)", "S6 (Cliff)"])
    ax1.set_xticks(range(len(wind_edges) - 1))
    ax1.set_xticklabels(["0-4", "4-8", "8-12", "12-15", "15-18"])
    ax1.set_xlabel("Mean Wind Speed W_20 [m/s]")
    ax1.set_title("VertiRTA Stratified Survival Rate [%]")
    fig.colorbar(im, ax=ax1, label="Survival [%]")

    winds = np.array([float(r.get("mean_wind_mps", 0.0)) for r in runs])
    socs = np.array([float(r.get("initial_soc_pct", 80.0)) for r in runs])
    passed_mask = np.array([bool(r.get("passed", False)) for r in runs])
    ax2.scatter(winds[passed_mask], socs[passed_mask], c="#1B9E77", s=16, alpha=0.7, label="Pass")
    if np.any(~passed_mask):
        ax2.scatter(winds[~passed_mask], socs[~passed_mask], c="#D95F02", marker="x", s=28, label="Fail / Cliff")
    ax2.set_xlabel("Mean Wind Speed W_20 [m/s]")
    ax2.set_ylabel("Initial Battery SoC [%]")
    ax2.set_title("Wind x Battery SoC Survival Boundary")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def plot_ab_cdf(ab_runs: dict[str, list[dict[str, Any]]], out_png: Path) -> Path:
    """Render empirical CDFs of min CBVF margin and touchdown sink speed across A/B configurations."""
    if not ab_runs:
        raise ValueError("Cannot plot A/B CDF from empty ab_runs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    for label, rows in sorted(ab_runs.items()):
        if not rows:
            continue
        margins = np.sort([float(r.get("min_cbvf_margin_m", 0.0)) for r in rows])
        vz = np.sort([abs(float(r.get("touchdown_vz_mps", 0.0))) for r in rows])
        cdf = np.arange(1, len(rows) + 1) / len(rows)
        ax1.plot(margins, cdf, linewidth=2.0, label=label)
        ax2.plot(vz, cdf, linewidth=2.0, label=label)

    ax1.axvline(0.0, color="#D93025", linestyle="--", linewidth=1.2, label="Barrier V = 0")
    ax1.set_xlabel("Min 4D CBVF Margin [m]")
    ax1.set_ylabel("Empirical CDF")
    ax1.set_title("A/B Safety Margin CDF (HJ-CBVF vs HOCBF)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axvline(2.5, color="#D93025", linestyle="--", linewidth=1.2, label="CEL Limit 2.5 m/s")
    ax2.set_xlabel("Touchdown Sink Rate |v_z| [m/s]")
    ax2.set_ylabel("Empirical CDF")
    ax2.set_title("A/B Touchdown Sink Rate CDF (INDI+WLS vs Stock CA)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Survival Envelope Heatmap & A/B CDFs.")
    parser.add_argument("--atlas-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.atlas_json.read_text(encoding="utf-8"))
    primary_runs = payload.get("primary_runs", payload.get("runs", []))
    ab_runs = payload.get("ab_runs", {"primary": primary_runs})

    hm = plot_survival_heatmap(primary_runs, args.out_dir / "survival_heatmap.png")
    cdf = plot_ab_cdf(ab_runs, args.out_dir / "ab_cdf.png")
    print(json.dumps({"heatmap_png": str(hm), "ab_cdf_png": str(cdf)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
