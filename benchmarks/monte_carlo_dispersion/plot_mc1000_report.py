# SPDX-License-Identifier: Apache-2.0
"""Publication-grade 2-panel VertiRTA MC-1000 dispersion & failure-boundary visualizer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse
import numpy as np


def _draw_cov_ellipse(
    ax: plt.Axes,
    mean_x: float,
    mean_y: float,
    cov_xx: float,
    cov_xy: float,
    cov_yy: float,
    chi2_val: float,
    **kwargs: Any,
) -> None:
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    width = 2.0 * math.sqrt(chi2_val * eigvals[0])
    height = 2.0 * math.sqrt(chi2_val * eigvals[1])
    angle_deg = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    ell = Ellipse((mean_x, mean_y), width=width, height=height, angle=angle_deg, **kwargs)
    ax.add_patch(ell)


def render_mc1000_figure(report_path: Path, out_png: Path) -> None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    samples = np.array(data["touchdown_samples"], dtype=float)
    # Schema: [airframe_id, regime_id, dx_m (North), dy_m (East), td_vz_mps, min_v, max_tilt, safe, breach]

    x8_mask = samples[:, 0] == 0
    lc_mask = samples[:, 0] == 1
    x8_stats = data["by_airframe"]["VertiRTA-X8-240"]
    lc_stats = data["by_airframe"]["VertiRTA-LC-240-8p1"]
    overall = data["overall"]
    sweep = data["failure_boundary_sweep"]
    fb = sweep["summary"]

    # Styling: crisp dark aerospace engineering telemetry theme
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.titlesize": 10.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.6,
            "axes.labelcolor": "#E2E8F0",
            "axes.titlecolor": "#F8FAFC",
            "axes.edgecolor": "#475569",
            "axes.linewidth": 0.9,
            "xtick.color": "#CBD5E1",
            "ytick.color": "#CBD5E1",
            "text.color": "#F1F5F9",
        }
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.0, 6.25),
        dpi=150,
        gridspec_kw={"width_ratios": [1.06, 1.0], "wspace": 0.30},
    )
    bg_fig = "#0B1120"
    bg_ax = "#111A2E"
    fig.patch.set_facecolor(bg_fig)
    ax1.set_facecolor(bg_ax)
    ax2.set_facecolor(bg_ax)

    # ------------------------------------------------------------------
    # LEFT PANEL: 1,000-Flight Touchdown Dispersion (East vs North)
    # ------------------------------------------------------------------
    x8_e = samples[x8_mask, 3]
    x8_n = samples[x8_mask, 2]
    lc_e = samples[lc_mask, 3]
    lc_n = samples[lc_mask, 2]

    # Separate nominal/CSFL/GNSS/Low-SoC vs CEL/Dual-Kill for visual clarity
    x8_cel = x8_mask & (samples[:, 1] == 2)
    x8_non = x8_mask & (samples[:, 1] != 2)
    lc_dual = lc_mask & (samples[:, 1] == 2)
    lc_non = lc_mask & (samples[:, 1] != 2)

    ax1.scatter(
        samples[x8_non, 3],
        samples[x8_non, 2],
        s=15,
        alpha=0.55,
        color="#38BDF8",
        edgecolors="none",
        label=f"X8-240 Non-CEL (n={int(x8_non.sum())})",
        zorder=3,
    )
    ax1.scatter(
        samples[x8_cel, 3],
        samples[x8_cel, 2],
        s=22,
        alpha=0.78,
        marker="^",
        color="#F59E0B",
        edgecolors="none",
        label=f"X8-240 Same-Arm CEL Spin (n={int(x8_cel.sum())})",
        zorder=4,
    )
    ax1.scatter(
        samples[lc_non, 3],
        samples[lc_non, 2],
        s=15,
        alpha=0.55,
        color="#34D399",
        edgecolors="none",
        label=f"LC-240 8+1 Non-Dual (n={int(lc_non.sum())})",
        zorder=3,
    )
    ax1.scatter(
        samples[lc_dual, 3],
        samples[lc_dual, 2],
        s=20,
        alpha=0.78,
        marker="s",
        color="#A78BFA",
        edgecolors="none",
        label=f"LC-240 8+1 Dual-Lift Divert (n={int(lc_dual.sum())})",
        zorder=4,
    )

    # Highlight any failed operational flights if present
    fail_mask = samples[:, 7] < 0.5
    if np.any(fail_mask):
        ax1.scatter(
            samples[fail_mask, 3],
            samples[fail_mask, 2],
            s=42,
            marker="x",
            linewidths=1.6,
            color="#FB7185",
            label=f"Envelope Exceedance (n={int(fail_mask.sum())})",
            zorder=6,
        )

    # Empirical CEP50 & CEP95 circles + 95% covariance ellipses (East=x, North=y)
    # Note: cov_xx is North^2, cov_yy is East^2, cov_xy is North*East
    ax1.add_patch(
        Circle(
            (0.0, 0.0),
            x8_stats["cep50_m"],
            fill=False,
            linestyle="--",
            linewidth=1.35,
            edgecolor="#38BDF8",
            zorder=5,
        )
    )
    ax1.add_patch(
        Circle(
            (0.0, 0.0),
            x8_stats["cep95_m"],
            fill=False,
            linestyle="-",
            linewidth=1.55,
            edgecolor="#38BDF8",
            zorder=5,
        )
    )
    ax1.add_patch(
        Circle(
            (0.0, 0.0),
            lc_stats["cep50_m"],
            fill=False,
            linestyle="--",
            linewidth=1.35,
            edgecolor="#34D399",
            zorder=5,
        )
    )
    ax1.add_patch(
        Circle(
            (0.0, 0.0),
            lc_stats["cep95_m"],
            fill=False,
            linestyle="-",
            linewidth=1.55,
            edgecolor="#34D399",
            zorder=5,
        )
    )

    # 95% 2D Gaussian covariance ellipses (chi2_2(0.95) = 5.9915) in (East, North)
    _draw_cov_ellipse(
        ax1,
        x8_stats["mean_dy_m"],
        x8_stats["mean_dx_m"],
        x8_stats["cov_yy_m2"],
        x8_stats["cov_xy_m2"],
        x8_stats["cov_xx_m2"],
        5.9915,
        facecolor="#38BDF8",
        alpha=0.06,
        edgecolor="#7DD3FC",
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )
    _draw_cov_ellipse(
        ax1,
        lc_stats["mean_dy_m"],
        lc_stats["mean_dx_m"],
        lc_stats["cov_yy_m2"],
        lc_stats["cov_xy_m2"],
        lc_stats["cov_xx_m2"],
        5.9915,
        facecolor="#34D399",
        alpha=0.06,
        edgecolor="#6EE7B7",
        linestyle=":",
        linewidth=1.0,
        zorder=2,
    )

    ax1.axhline(0.0, color="#475569", linewidth=0.75, linestyle="-", zorder=1)
    ax1.axvline(0.0, color="#475569", linewidth=0.75, linestyle="-", zorder=1)
    ax1.scatter([0.0], [0.0], s=55, marker="+", color="#F8FAFC", linewidths=1.5, zorder=7)

    # Symmetric axis limits framing 99% of points + CEP95 cleanly
    r_all = np.hypot(samples[:, 2], samples[:, 3])
    lim = max(12.0, float(np.percentile(r_all, 98.5)) * 1.22, x8_stats["cep95_m"] * 1.28, lc_stats["cep95_m"] * 1.28)
    lim = math.ceil(lim / 5.0) * 5.0
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, color="#1E293B", linewidth=0.7, linestyle="--")
    ax1.set_xlabel("East Touchdown Offset from SLZ Aimpoint [m]")
    ax1.set_ylabel("North Touchdown Offset from SLZ Aimpoint [m]")
    ax1.set_title("A. 1,000-Flight SLZ Touchdown Dispersion (X8 vs LC 8+1)", pad=10)

    # Zoomed inset (±3.2 m) in upper-right corner showing the 900-flight precision landing core!
    ax_in = ax1.inset_axes([0.64, 0.64, 0.335, 0.335])
    ax_in.set_facecolor("#0B1324")
    ax_in.scatter(
        samples[x8_non, 3],
        samples[x8_non, 2],
        s=7,
        alpha=0.55,
        color="#38BDF8",
        edgecolors="none",
        zorder=3,
    )
    ax_in.scatter(
        samples[lc_non, 3],
        samples[lc_non, 2],
        s=7,
        alpha=0.55,
        color="#34D399",
        edgecolors="none",
        zorder=3,
    )
    ax_in.scatter(
        samples[lc_dual, 3],
        samples[lc_dual, 2],
        s=9,
        alpha=0.75,
        marker="s",
        color="#A78BFA",
        edgecolors="none",
        zorder=4,
    )
    ax_in.add_patch(
        Circle((0.0, 0.0), lc_stats["cep50_m"], fill=False, linestyle="--", linewidth=1.1, edgecolor="#34D399", zorder=5)
    )
    ax_in.add_patch(
        Circle((0.0, 0.0), lc_stats["cep95_m"], fill=False, linestyle="-", linewidth=1.2, edgecolor="#34D399", zorder=5)
    )
    ax_in.axhline(0.0, color="#334155", linewidth=0.6, zorder=1)
    ax_in.axvline(0.0, color="#334155", linewidth=0.6, zorder=1)
    ax_in.set_xlim(-3.2, 3.2)
    ax_in.set_ylim(-3.2, 3.2)
    ax_in.set_aspect("equal", adjustable="box")
    ax_in.set_xticks([-2, 0, 2])
    ax_in.set_yticks([-2, 0, 2])
    ax_in.tick_params(labelsize=6.8, colors="#CBD5E1", pad=2)
    ax_in.set_title("Core Zoom (±3.2 m)", fontsize=7.6, color="#E2E8F0", pad=3)
    for spine in ax_in.spines.values():
        spine.set_edgecolor("#64748B")
        spine.set_linewidth(0.85)

    # Clean non-overlapping summary box in bottom-left
    breach_ub = overall["airspace_breach_ci95"]["clopper_pearson_95_one_sided_hi"] * 100.0
    rec_lo = overall["recovery_ci95"]["clopper_pearson_95_lo"] * 100.0
    rec_rate = overall["recovery_ci95"]["rate"] * 100.0
    summary_txt = (
        f"Overall Safe: {overall['safe_recoveries']}/{overall['flights']} ({rec_rate:.1f}%, 95% CP >= {rec_lo:.2f}%)\n"
        f"Airspace Breaches: {overall['airspace_breaches']}/{overall['flights']} (95% 1-sided <= {breach_ub:.2f}%)\n"
        f"X8-240:  CEP50={x8_stats['cep50_m']:.2f}m | CEP95={x8_stats['cep95_m']:.2f}m | vz99={x8_stats['vz_p99_mps']:.2f} m/s\n"
        f"LC-240:  CEP50={lc_stats['cep50_m']:.2f}m | CEP95={lc_stats['cep95_m']:.2f}m | vz99={lc_stats['vz_p99_mps']:.2f} m/s"
    )
    ax1.text(
        0.025,
        0.025,
        summary_txt,
        transform=ax1.transAxes,
        fontsize=7.9,
        family="DejaVu Sans Mono",
        color="#E2E8F0",
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#0F172A", edgecolor="#334155", alpha=0.92),
        zorder=8,
    )
    leg = ax1.legend(
        loc="upper left",
        fontsize=7.8,
        frameon=True,
        facecolor="#0F172A",
        edgecolor="#334155",
        framealpha=0.92,
        markerscale=1.05,
    )
    for txt in leg.get_texts():
        txt.set_color("#E2E8F0")

    # ------------------------------------------------------------------
    # RIGHT PANEL: 6x6 Honest Physical Failure-Boundary Heatmap
    # ------------------------------------------------------------------
    mass_pcts = sweep["mass_overload_pcts"]
    wind_vals = sweep["wind_gust_mps"]
    n_m = len(mass_pcts)
    n_w = len(wind_vals)
    grid_surv = np.zeros((n_m, n_w), dtype=float)
    grid_vz = np.zeros((n_m, n_w), dtype=float)
    grid_safe_cnt = np.zeros((n_m, n_w), dtype=int)
    grid_tot_cnt = np.zeros((n_m, n_w), dtype=int)

    for c in sweep["grid_cells"]:
        mi = int(c["mass_idx"])
        wi = int(c["wind_idx"])
        grid_surv[mi, wi] = float(c["survival_rate"]) * 100.0
        grid_vz[mi, wi] = float(c["mean_td_vz_mps"])
        grid_safe_cnt[mi, wi] = int(c["safe_recoveries"])
        grid_tot_cnt[mi, wi] = int(c["flights"])

    cmap = LinearSegmentedColormap.from_list(
        "vertirta_env",
        [(0.0, "#B91C1C"), (0.35, "#DC2626"), (0.60, "#D97706"), (0.85, "#65A30D"), (1.0, "#059669")],
    )
    im = ax2.imshow(
        grid_surv,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=100.0,
    )

    for mi in range(n_m):
        for wi in range(n_w):
            surv = grid_surv[mi, wi]
            vz = grid_vz[mi, wi]
            sc = grid_safe_cnt[mi, wi]
            tc = grid_tot_cnt[mi, wi]
            txt_col = "#FFFFFF"
            ax2.text(
                wi,
                mi,
                f"{surv:.0f}%\n({sc}/{tc})\nvz={vz:.1f}",
                ha="center",
                va="center",
                fontsize=7.6,
                fontweight="bold" if surv in (0.0, 100.0) else "normal",
                color=txt_col,
            )

    # Crisp cell grid separators
    ax2.set_xticks(np.arange(n_w))
    ax2.set_xticklabels([f"{w:.0f}" for w in wind_vals])
    ax2.set_yticks(np.arange(n_m))
    ax2.set_yticklabels([f"+{m:.0f}%\n({240.0 * (1.0 + m / 100.0):.0f} kg)" for m in mass_pcts])
    ax2.set_xticks(np.arange(-0.5, n_w, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n_m, 1), minor=True)
    ax2.grid(which="minor", color="#0B1120", linestyle="-", linewidth=1.6)
    ax2.tick_params(which="minor", bottom=False, left=False)

    ax2.set_xlabel("Mean Headwind + Turbulent Gust Intensity [m/s]")
    ax2.set_ylabel("Mass Overload Relative to 240 kg MTOW")
    ax2.set_title("B. Dual-Fault Envelope Breakdown (6×6 × 5 = 180 Flights)", pad=10)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
    cbar.set_label("Safe Recovery Rate [%]", color="#E2E8F0", fontsize=9.0)
    cbar.ax.yaxis.set_tick_params(color="#CBD5E1")
    plt.setp(cbar.ax.get_yticklabels(), color="#CBD5E1")
    cbar.outline.set_edgecolor("#475569")

    fig.suptitle(
        f"VertiRTA Native C++20 Monte Carlo Dispersion & Thrust-Saturation Failure Boundary "
        f"(Seed {data['master_seed']}, N={data['operational_flights_total']}+{data['stress_flights_total']} Flights)",
        fontsize=12.2,
        fontweight="bold",
        color="#F8FAFC",
        y=0.972,
    )
    fig.text(
        0.5,
        0.014,
        f"SHA-256: {data['report_sha256'][:24]}...  |  "
        f"Stress Envelope: {fb['green_cells_100pct']} green (100%) / {fb['transition_cells_partial']} transition / "
        f"{fb['red_cells_0pct']} red (0%)  |  "
        f"First breakdown: +{fb['first_breakdown_mass_overload_pct']:.0f}% mass @ {fb['first_breakdown_wind_mps']:.0f} m/s",
        ha="center",
        fontsize=8.2,
        color="#94A3B8",
    )

    fig.subplots_adjust(left=0.068, right=0.932, top=0.875, bottom=0.125)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=bg_fig)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot VertiRTA MC-1000 dispersion & failure-boundary PNG.")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json"),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("docs/assets/vertirta_mc1000_envelope_dispersion.png"),
    )
    args = parser.parse_args()
    render_mc1000_figure(args.report_json, args.out_png)
    size_kb = args.out_png.stat().st_size / 1024.0
    print(f"Wrote {args.out_png} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
