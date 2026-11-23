#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate 3 double-anonymous publication-grade PDF figures for IEEE RA-L Paper 1.

Strictly reads live measured JSON artifacts from artifacts/v1.0.0/ and artifacts/v3.1.0/
(plus verified M2 v3.5.0 test metrics) and writes compact vector PDFs to
docs/paper_ral/figures/. Strips all author/host metadata for double-anonymous review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "paper_ral" / "figures"

# Colorblind-safe Okabe-Ito palette + clean IEEE publication styling
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_YELLOW = "#E69F00"
C_PURPLE = "#CC79A7"
C_GREY = "#555555"

ANONYMOUS_PDF_META: dict[str, str] = {
    "Author": "Anonymous Authors",
    "Creator": "Anonymous Reproducible Figure Pipeline",
    "Producer": "matplotlib",
}


def _configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 7.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 200,
        }
    )


def _draw_box(
    ax: Any,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    *,
    facecolor: str,
    edgecolor: str,
    linestyle: str = "-",
    title_color: str = "#111111",
) -> None:
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.05,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=6.1,
        fontweight="bold",
        color=title_color,
    )
    ax.text(
        x + w / 2.0,
        y + h * 0.27,
        subtitle,
        ha="center",
        va="center",
        fontsize=5.2,
        color="#333333",
    )


def _arrow(
    ax: Any,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str = "",
    *,
    color: str = "#222222",
    lx_off: float = 0.0,
    ly_off: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(x0, y0),
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.05,
            "shrinkA": 1.0,
            "shrinkB": 1.0,
        },
    )
    if label:
        mx = (x0 + x1) / 2.0 + lx_off
        my = (y0 + y1) / 2.0 + ly_off
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=5.5,
            fontweight="bold",
            color=color,
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.95},
        )


def build_fig1_rta_architecture(out_path: Path) -> Path:
    """Figure 1: ASTM F3269-21 3-Tier Simplex RTA & Dual-FCC Bare-Metal Architecture."""
    fig, ax = plt.subplots(figsize=(7.1, 2.95))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Zone backdrops with generous header clearance
    z_untrusted = mpatches.FancyBboxPatch(
        (0.01, 0.72), 0.98, 0.26, boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor="#FFF8EC", edgecolor=C_YELLOW, linewidth=0.95, linestyle="--",
    )
    z_comp = mpatches.FancyBboxPatch(
        (0.01, 0.365), 0.98, 0.285, boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor="#EEF5FB", edgecolor=C_BLUE, linewidth=1.0,
    )
    z_core = mpatches.FancyBboxPatch(
        (0.01, 0.015), 0.98, 0.28, boxstyle="round,pad=0.008,rounding_size=0.018",
        facecolor="#ECF8F3", edgecolor=C_GREEN, linewidth=1.05,
    )
    for z in (z_untrusted, z_comp, z_core):
        ax.add_patch(z)

    ax.text(0.025, 0.963, "Untrusted Mission & Tactical U-Space Layer (Outside Deterministic RTA Boundary)",
            fontsize=6.1, fontweight="bold", color="#8A5A00", va="top")
    ax.text(0.025, 0.635, "100 Hz Companion RTA Layer (C++20, Zero-Copy mmap, 4-Thread Lock-Free SPSC/Seqlock)",
            fontsize=6.1, fontweight="bold", color="#004C78", va="top")
    ax.text(0.025, 0.280, "400 Hz Bare-Metal Flight Core (C++17 Cortex-M7: 28,886 B Flash, 0 B RAM, 2,960 B DAG, 0 Heap)",
            fontsize=6.1, fontweight="bold", color="#00664B", va="top")

    # Top tier boxes
    _draw_box(ax, 0.03, 0.745, 0.43, 0.145, "Nominal Guidance / Complex Function (CF)",
              "Reference accel a_ref, mission trajectory, vertiport slots",
              facecolor="white", edgecolor=C_YELLOW, linestyle="--")
    _draw_box(ax, 0.50, 0.745, 0.47, 0.145, "M2 Multi-eVTOL U-Space & Part-IS Preemption",
              "20-agent SoA (0.151 ms/tick), QF_IDL (1.81 ms), <=14 slots, <500 ms P0",
              facecolor="white", edgecolor=C_YELLOW, linestyle="--")

    # Middle tier boxes
    _draw_box(ax, 0.03, 0.395, 0.27, 0.175, "4D HJI-CBVF & 2D CEL LUTs",
              "GPU 81x57x81x57 WENO5->41x29\n11.31 MB mmap, SHA-256, eps_grid",
              facecolor="white", edgecolor=C_BLUE)
    _draw_box(ax, 0.34, 0.395, 0.31, 0.175, "HJI-CBVF-QP Safety Filter",
              "StaticDaqpSolver<5,32> (5,888 B)\nK<=16 rows, u*_HJ fallback, CTR walls",
              facecolor="white", edgecolor=C_BLUE)
    _draw_box(ax, 0.69, 0.395, 0.28, 0.175, "Robocentric ESDF & Airspace",
              "3.094 MiB ESDF, 64 WireCapsules,\nLTFJ/LTBX CTR (<29.1895 E), 16 KF",
              facecolor="white", edgecolor=C_BLUE)

    # Bottom tier boxes
    _draw_box(ax, 0.03, 0.042, 0.21, 0.175, "3-IMU TMR & Motor FDI",
              "Chi2(24)>72.23 (<=20 ms)\n1M / 2M-Cross vs Same-Arm",
              facecolor="white", edgecolor=C_GREEN)
    _draw_box(ax, 0.27, 0.042, 0.22, 0.175, "ASTM F3269-21 Arbiter",
              "Priority: CEL > CSFL > CBVF > CF\nIrreversible <=2.5 ms CEL latch",
              facecolor="white", edgecolor=C_ORANGE)
    _draw_box(ax, 0.52, 0.042, 0.22, 0.175, "f64 INDI-WLS & Option B1b",
              "8 Hz sync-LPF, +17.6% unblank\n|Omega_z|=2.2 rad/s (+10.6 deg)",
              facecolor="white", edgecolor=C_GREEN)
    _draw_box(ax, 0.77, 0.042, 0.20, 0.175, "Dual-FCC Cyphal & VRTA",
              "Renode 12.50 ms SIL failover\n256 B CRC-32C append-only log",
              facecolor="white", edgecolor=C_GREEN)

    # Clean orthogonal/vertical arrows routed strictly outside header text
    _arrow(ax, 0.42, 0.733, 0.42, 0.582, "a_ref", color="#8A5A00", lx_off=0.04, ly_off=0.025)
    _arrow(ax, 0.83, 0.733, 0.83, 0.582, "<=14 hazards", color="#8A5A00", lx_off=0.065, ly_off=0.025)
    _arrow(ax, 0.312, 0.482, 0.328, 0.482, "", color=C_BLUE)
    _arrow(ax, 0.678, 0.482, 0.662, 0.482, "", color=C_BLUE)
    _arrow(ax, 0.38, 0.383, 0.38, 0.229, "100 Hz SPSC", color=C_BLUE, lx_off=0.068, ly_off=0.025)
    _arrow(ax, 0.252, 0.130, 0.258, 0.130, "", color=C_GREEN)
    _arrow(ax, 0.502, 0.130, 0.508, 0.130, "", color=C_ORANGE)
    _arrow(ax, 0.752, 0.130, 0.758, 0.130, "", color=C_GREEN)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=ANONYMOUS_PDF_META)
    plt.close(fig)
    return out_path


def build_fig2_wind_cliff_and_paired_survival(stats: dict[str, Any], out_path: Path) -> Path:
    """Figure 2: Paired 30k CSFL/CEL Survival, Wind Sweep, and Speed-Gust Authority Cliff."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.50))

    # Panel (a): Isolated CSFL vs Same-Arm CEL survival rates with CP95 lower bounds
    ax0 = axes[0]
    ab = stats["ab_summary"]
    cfgs = ["hj_cbvf__indi_wls", "hocbf__indi_wls", "hj_cbvf__stock_ca"]
    labels = ["HJ-CBVF\n+INDI-WLS\n(Ours)", "HOCBF\n+INDI-WLS", "HJ-CBVF\n+Stock CA"]
    csfl_vals = [ab[c]["csfl_pass_pct"] for c in cfgs]
    csfl_err = [ab[c]["csfl_cp95_upper_pct"] - (100.0 - ab[c]["csfl_pass_pct"]) for c in cfgs]
    cel_vals = [ab[c]["cel_pass_pct"] for c in cfgs]
    cel_err = [ab[c]["cel_cp95_upper_pct"] - (100.0 - ab[c]["cel_pass_pct"]) for c in cfgs]

    x = np.arange(len(cfgs))
    w = 0.34
    b1 = ax0.bar(x - w / 2, csfl_vals, w, color=C_BLUE, label="CSFL (n=6,500)")
    b2 = ax0.bar(x + w / 2, cel_vals, w, color=C_GREEN, label="CEL (n=1,500)")
    ax0.errorbar(
        x - w / 2, csfl_vals, yerr=[np.clip(csfl_err, 0.0, None), np.zeros_like(csfl_err)],
        fmt="none", ecolor="#111111", elinewidth=0.8, capsize=1.8,
    )
    ax0.errorbar(
        x + w / 2, cel_vals, yerr=[np.clip(cel_err, 0.0, None), np.zeros_like(cel_err)],
        fmt="none", ecolor="#111111", elinewidth=0.8, capsize=1.8,
    )
    for bar, offset in [(b1[0], 1.5), (b2[0], 5.5), (b1[1], 2.0), (b2[1], 2.0), (b1[2], 2.0), (b2[2], 2.0)]:
        h = bar.get_height()
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0, h + offset,
            f"{h:.1f}%", ha="center", va="bottom", fontsize=5.1,
        )
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=5.8)
    ax0.set_ylim(0, 134)
    ax0.set_ylabel("Survival Rate [%]")
    ax0.set_title("(a) Paired 30k CSFL vs. CEL")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, fontsize=5.1, columnspacing=0.6)

    # Panel (b): Coastal wind-gust sweep (0..22 m/s, 200 runs/bin)
    ax1 = axes[1]
    gust = stats["gust_failure_cliff_sweep"]
    winds = [g["mean_wind_mps"] for g in gust]
    pass_ours = [g["hj_cbvf__indi_wls_pass_pct"] for g in gust]
    pass_hocbf = [g["hocbf__indi_wls_pass_pct"] for g in gust]
    margins = [g["hj_cbvf__indi_wls_mean_margin_m"] for g in gust]

    l1 = ax1.plot(winds, pass_ours, "o-", color=C_BLUE, markersize=3.0, label="HJ-CBVF Pass [%]")
    l2 = ax1.plot(winds, pass_hocbf, "s--", color=C_ORANGE, markersize=2.8, label="HOCBF Pass [%]")
    ax1.set_ylim(0, 108)
    ax1.set_xlabel("Mean Coastal Wind [m/s]")
    ax1.set_ylabel("CSFL Pass Rate [%]")
    ax1.set_title("(b) Wind Sweep (200 runs/bin)")
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax1_r = ax1.twinx()
    l3 = ax1_r.plot(winds, margins, "^:", color=C_GREEN, markersize=3.0, label="Mean Margin [m]")
    ax1_r.set_ylim(0.35, 1.25)
    ax1_r.set_ylabel("Mean CBVF Margin [m]", color=C_GREEN)
    ax1_r.tick_params(axis="y", colors=C_GREEN, labelsize=6.0)
    lines_all = l1 + l2 + l3
    ax1.legend(
        lines_all, [ln.get_label() for ln in lines_all],
        loc="lower left", frameon=True, framealpha=0.95, fontsize=5.2,
    )

    # Panel (c): Cross-arm S3 approach-speed authority cliff under 18 m/s wind
    ax2 = axes[2]
    cliff = stats["speed_gust_authority_cliff_sweep"]
    speeds = [c["cruise_speed_mps"] for c in cliff]
    c_ours = [c["hj_cbvf__indi_wls_pass_pct"] for c in cliff]
    c_hocbf = [c["hocbf__indi_wls_pass_pct"] for c in cliff]

    ax2.plot(speeds, c_ours, "o-", color=C_BLUE, markersize=3.2, label="HJ-CBVF+INDI-WLS")
    ax2.plot(speeds, c_hocbf, "s--", color=C_ORANGE, markersize=3.0, label="Analytical HOCBF")
    ax2.axvspan(10.0, 16.0, color=C_BLUE, alpha=0.11, label="+6 m/s Envelope")
    ax2.axvline(12.92, color=C_GREY, linestyle=":", linewidth=0.9)
    ax2.text(13.1, 22.0, "$v_{\\mathrm{cap}}$=12.9", fontsize=5.5, color=C_GREY)
    ax2.set_ylim(-5, 138)
    ax2.set_xlim(7.5, 22.5)
    ax2.set_xlabel("Approach Speed [m/s] (18 m/s Wind)")
    ax2.set_ylabel("S3 Survival Rate [%]")
    ax2.set_title("(c) S3 Speed-Gust Cliff")
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=5.1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35, w_pad=0.75)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=ANONYMOUS_PDF_META)
    plt.close(fig)
    return out_path


def build_fig3_renode_esbmc_latency_profile(
    stats: dict[str, Any],
    bench_local_t3: dict[str, Any],
    bench_local_t4: dict[str, Any],
    bench_ci_x86: dict[str, Any],
    bench_ci_arm64: dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 3: Cross-ISA Latency Profile, Cortex-M7 Budget, and Formal/SIL/M2 Gates."""
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.50))

    # Panel (a): Per-tick RTA execution latency (log scale, us)
    ax0 = axes[0]
    runners = ["x86 Loc.\nCPUTIME", "x86 Loc.\nMONO", "x86 CI\nMONO", "ARM64 CI\nCobalt"]
    p50 = [
        bench_local_t3["p50_us"],
        bench_local_t4["metrics"]["p50_us"],
        bench_ci_x86["metrics"]["p50_us"],
        bench_ci_arm64["metrics"]["p50_us"],
    ]
    p99 = [
        bench_local_t3["p99_us"],
        bench_local_t4["metrics"]["p99_us"],
        bench_ci_x86["metrics"]["p99_us"],
        bench_ci_arm64["metrics"]["p99_us"],
    ]
    p999 = [
        bench_local_t3["p999_us"],
        bench_local_t4["metrics"]["p999_us"],
        bench_ci_x86["metrics"]["p999_us"],
        bench_ci_arm64["metrics"]["p999_us"],
    ]
    x = np.arange(len(runners))
    w = 0.25
    ax0.bar(x - w, p50, w, color=C_BLUE, label="p50")
    ax0.bar(x, p99, w, color=C_GREEN, label="p99")
    ax0.bar(x + w, p999, w, color=C_ORANGE, label="p99.9")
    ax0.axhline(10000.0, color="#AA0000", linestyle="--", linewidth=0.9, label="100 Hz Cap")
    ax0.set_yscale("log")
    ax0.set_ylim(0.2, 45000.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(runners, fontsize=5.7)
    ax0.set_ylabel("Per-Tick Latency [$\\mu$s, log]")
    ax0.set_title("(a) Cross-ISA Tick Latency")
    ax0.grid(axis="y", linestyle=":", alpha=0.5)
    ax0.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.95, fontsize=5.1, columnspacing=0.6)

    # Panel (b): Cortex-M7 bare-metal resource utilization vs hard budgets
    ax1 = axes[1]
    cm7 = stats["cortex_m7"]
    res_labels = ["Flash\n(.text+.data)", "Static RAM\n(Library)", "Max Stack\nFrame", "Stack Call\nGraph DAG", "Heap\nSymbols"]
    used_pct = [
        100.0 * cm7["flash_bytes"] / cm7["max_flash_bytes"],
        100.0 * cm7["ram_bytes"] / cm7["max_ram_bytes"],
        100.0 * cm7["max_frame_bytes"] / 4096.0,
        100.0 * cm7["max_dag_bytes"] / 16384.0,
        0.0,
    ]
    raw_ann = [
        f"{cm7['flash_bytes']:,} B",
        f"{cm7['ram_bytes']} B",
        f"{cm7['max_frame_bytes']:,} B",
        f"{cm7['max_dag_bytes']:,} B",
        "0 syms",
    ]
    y = np.arange(len(res_labels))
    bars = ax1.barh(y, used_pct, height=0.58, color=[C_BLUE, C_GREEN, C_YELLOW, C_PURPLE, C_GREEN])
    ax1.axvline(100.0, color="#AA0000", linestyle="--", linewidth=0.9)
    for idx, bar in enumerate(bars):
        val = bar.get_width()
        ax1.text(val + 2.5, bar.get_y() + bar.get_height() / 2.0,
                 f"{raw_ann[idx]} ({val:.1f}%)", va="center", fontsize=5.3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(res_labels, fontsize=5.8)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 114)
    ax1.set_xlabel("Budget Utilization [%]")
    ax1.set_title("(b) Cortex-M7 Footprint")
    ax1.grid(axis="x", linestyle=":", alpha=0.5)

    # Panel (c): Live SIL, Formal, and M2 Multi-Aircraft Timing Margins (% of Gate)
    ax2 = axes[2]
    live = stats["live_ci_verification"]
    gate_labels = [
        "Renode SIL\nFailover",
        "M2 20-eVTOL\nFleet Tick",
        "M2 Fleet\nQF_IDL",
        "ESBMC v8.7\n1,905 Props",
    ]
    ratios = [
        100.0 * live["renode_failover_ms"] / 20.0,
        100.0 * 0.1507 / 1.0,
        100.0 * 1.81 / 3.0,
        100.0 * live["esbmc_properties_verified"] / float(live["esbmc_properties_verified"] + live["esbmc_properties_failed"]),
    ]
    colors = [C_BLUE, C_GREEN, C_YELLOW, C_PURPLE]
    ann = [
        f"{live['renode_failover_ms']:.1f}/20 ms",
        "0.15/1.0 ms",
        "1.81/3.0 ms",
        f"{live['esbmc_properties_verified']:,} PASS",
    ]
    y2 = np.arange(len(gate_labels))
    bars2 = ax2.barh(y2, ratios, height=0.55, color=colors)
    ax2.axvline(100.0, color="#AA0000", linestyle="--", linewidth=0.9)
    for idx, bar in enumerate(bars2):
        val = bar.get_width()
        if idx in (0, 2, 3):
            ax2.text(val / 2.0, bar.get_y() + bar.get_height() / 2.0, ann[idx],
                     va="center", ha="center", fontsize=5.5, fontweight="bold", color="white")
        else:
            ax2.text(val + 3.0, bar.get_y() + bar.get_height() / 2.0, ann[idx],
                     va="center", ha="left", fontsize=5.5, color="#111111")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(gate_labels, fontsize=5.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 114)
    ax2.set_xlabel("Gate Fraction / Pass [%]")
    ax2.set_title("(c) SIL, Formal & M2 Gates")
    ax2.grid(axis="x", linestyle=":", alpha=0.5)
    ax2.text(
        0.5, -0.25, "Physical STM32H753 DWT->CYCCNT HWM = Pending",
        transform=ax2.transAxes, ha="center", va="top", fontsize=5.1, style="italic", color=C_GREY,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35, w_pad=0.75)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=ANONYMOUS_PDF_META)
    plt.close(fig)
    return out_path


def generate_all_ral_figures(repo_root: Path = REPO_ROOT, out_dir: Path = DEFAULT_OUT_DIR) -> list[Path]:
    _configure_ieee_style()
    stats = json.loads((repo_root / "artifacts/v3.1.0/paired_30k_ral_stats.json").read_text(encoding="utf-8"))
    t3_local = json.loads((repo_root / "artifacts/v1.0.0/benchmarks/tier3_thread_cputime_x86_64.json").read_text(encoding="utf-8"))
    t4_local = json.loads((repo_root / "artifacts/v1.0.0/benchmarks/tier4_wallclock_x86_64.json").read_text(encoding="utf-8"))
    t4_ci_x86 = json.loads((repo_root / "artifacts/v1.0.0/benchmarks/cloud_ci/tier4_wallclock_x86_64.json").read_text(encoding="utf-8"))
    t4_ci_arm = json.loads((repo_root / "artifacts/v1.0.0/benchmarks/cloud_ci/tier4_wallclock_arm64.json").read_text(encoding="utf-8"))

    f1 = build_fig1_rta_architecture(out_dir / "fig1_rta_architecture.pdf")
    f2 = build_fig2_wind_cliff_and_paired_survival(stats, out_dir / "fig2_wind_cliff_and_paired_survival.pdf")
    f3 = build_fig3_renode_esbmc_latency_profile(
        stats, t3_local, t4_local, t4_ci_x86, t4_ci_arm, out_dir / "fig3_renode_esbmc_latency_profile.pdf",
    )
    return [f1, f2, f3]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate double-anonymous IEEE RA-L Paper 1 PDF figures.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    paths = generate_all_ral_figures(args.repo_root, args.out_dir)
    for p in paths:
        print(f"Generated {p.as_posix()} ({p.stat().st_size} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
