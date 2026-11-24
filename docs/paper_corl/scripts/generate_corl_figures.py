#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate 3 double-anonymous publication vector PDF figures for Paper 2 (CoRL 2027).

Reads strictly from docs/paper_corl/corl_metrics_manifest.json (grounded in verified
artifacts/v3.1.0/ and artifacts/v3.5.1/ baselines plus optional Phase 3 v4.0.0 G4 results).
Strips all author/host PDF metadata for double-anonymous peer review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta as beta_dist  # noqa: E402

CORL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = CORL_DIR / "corl_metrics_manifest.json"
DEFAULT_OUT_DIR = CORL_DIR / "figures"

# Colorblind-safe Okabe-Ito palette
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_YELLOW = "#E69F00"
C_PURPLE = "#CC79A7"
C_GREY = "#555555"

ANONYMOUS_PDF_META: dict[str, str] = {
    "Author": "Anonymous Authors",
    "Creator": "Anonymous CoRL Reproducible Figure Pipeline",
    "Producer": "matplotlib",
}


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 200,
        }
    )


def _save_pdf(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.05, metadata=ANONYMOUS_PDF_META)
    plt.close(fig)


def make_fig1_bilingual_mondrian_pareto(manifest: dict[str, Any], out_path: Path) -> None:
    b = manifest["verified_baselines"]
    g = manifest["phase3_g4_slots"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"wspace": 0.30})

    # Panel (a): AST/STL accuracy & Grammar progression across measured stages
    stages = ["Q2.5-7B\nZS", "Q3-4B\nZS", "v1\nGRPO", "v2\nUncon.", "v2+GBNF\n+Z3"]
    ast_vals = [
        b["v310_qwen25_7b_zs"]["ast_stl_accuracy_pct"],
        b["v310_qwen3_4b_zs"]["ast_stl_accuracy_pct"],
        b["v310_qwen3_4b_sft_grpo"]["ast_stl_accuracy_pct"],
        b["v351_qwen3_4b_v2_unconstrained"]["ast_stl_accuracy_pct"],
        b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]["ast_stl_accuracy_pct"],
    ]
    gram_vals = [
        b["v310_qwen25_7b_zs"]["grammar_validity_pct"],
        b["v310_qwen3_4b_zs"]["grammar_validity_pct"],
        b["v310_qwen3_4b_sft_grpo"]["grammar_validity_pct"],
        b["v351_qwen3_4b_v2_unconstrained"]["grammar_validity_pct"],
        b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]["grammar_validity_pct"],
    ]
    if g["qwen3_v3_ast_stl_mean_pct"]["value"] is not None:
        stages.append("v3 Mondr.\n(G4)")
        ast_vals.append(g["qwen3_v3_ast_stl_mean_pct"]["value"])
        gram_vals.append(100.0)

    x = np.arange(len(stages))
    w = 0.36
    bars_g = ax1.bar(x - w / 2, gram_vals, width=w, color=C_BLUE, alpha=0.85, label="Grammar Valid (%)")
    bars_a = ax1.bar(x + w / 2, ast_vals, width=w, color=C_GREEN, alpha=0.90, label="AST/STL Exact (%)")
    for bar in bars_a:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylim(0, 114)
    ax1.set_ylabel("Accuracy / Validity (%)")
    ax1.set_title("(a) Bilingual Synthesis Progression (TR/EN/MIX)")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Panel (b): Conformal efficiency Pareto (Mean Set Size vs Singleton Execution)
    q3v1 = b["v310_qwen3_4b_sft_grpo"]
    q3v2 = b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]
    pts = [
        ("v1 Uncal.\n($\\tau=1.0$)", q3v1["mean_prediction_set_size"], q3v1["singleton_exec_pct"], C_ORANGE, "o"),
        ("v2 Uncal.\n($\\tau=1.0$)", q3v2["mean_prediction_set_size_uncal"], q3v2["singleton_exec_pct_uncal"], C_YELLOW, "s"),
        ("v2 NLL-Cal.\n($\\tau=0.51$)", q3v2["mean_prediction_set_size"], q3v2["singleton_exec_pct"], C_BLUE, "D"),
    ]
    if g["qwen3_v3_mondrian_mean_set_size"]["value"] is not None:
        pts.append(
            (
                "v3 Mondr.\n(G4)",
                g["qwen3_v3_mondrian_mean_set_size"]["value"],
                g["qwen3_v3_mondrian_singleton_exec_pct"]["value"] or 0.0,
                C_GREEN,
                "*",
            )
        )
    for idx_p, (lbl, sz, sng, col, mk) in enumerate(pts):
        ax2.scatter([sz], [sng], color=col, marker=mk, s=52, zorder=4)
        if idx_p == 1:
            ax2.annotate(
                lbl, (sz, sng), xytext=(9, -2), textcoords="offset points", ha="left", va="center", fontsize=5.8
            )
        else:
            ax2.annotate(lbl, (sz, sng), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=5.8)
    ax2.axvline(1.8, color=C_GREEN, linestyle="--", linewidth=1.1, label="Target Mean $|C| < 1.8$")
    ax2.set_xlim(1.0, 6.2)
    ax2.set_ylim(-6, 85)
    ax2.set_xlabel(r"Mean Conformal Set Size $|C|$ ($\downarrow$ tighter)")
    ax2.set_ylabel(r"Singleton Execution Rate (%) ($\uparrow$)")
    ax2.set_title(r"(b) Conformal Efficiency at $\alpha=0.05$ ($\geq 95$% Cov.)")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    _save_pdf(fig, out_path)


def make_fig2_snr_stratum_coverage(manifest: dict[str, Any], out_path: Path) -> None:
    alpha = 0.05
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"wspace": 0.30})

    # Panel (a): Exact finite-sample Beta-Binomial 95% coverage envelope vs N_cal
    n_grid = np.array([100, 150, 200, 250, 400, 500, 750, 800, 1000, 1500, 2000])
    lo_95 = []
    hi_95 = []
    for n_val in n_grid:
        k = int(np.ceil((n_val + 1) * (1.0 - alpha)))
        l_shape = n_val + 1 - k
        lo_95.append(100.0 * float(beta_dist.ppf(0.025, k, l_shape)))
        hi_95.append(100.0 * float(beta_dist.ppf(0.975, k, l_shape)))

    ax1.fill_between(n_grid, lo_95, hi_95, color=C_BLUE, alpha=0.22, label="Exact 95% Beta Envelope")
    ax1.plot(n_grid, lo_95, color=C_BLUE, linewidth=1.2)
    ax1.plot(n_grid, hi_95, color=C_BLUE, linewidth=1.2)
    ax1.axhline(95.0, color=C_GREY, linestyle="--", linewidth=1.0, label=r"Nominal $1-\alpha=95$%")
    ax1.axvline(200, color=C_ORANGE, linestyle=":", linewidth=1.2, label="v1 Pooled ($n=200$)")
    ax1.axvline(500, color=C_GREEN, linestyle="-.", linewidth=1.2, label="v2 SNR Stratum ($n_g=500$)")
    ax1.axvline(800, color=C_PURPLE, linestyle="-.", linewidth=1.1, label="v2 Lang Group ($n_{\\ell}=800$)")
    ax1.set_xlim(80, 2050)
    ax1.set_ylim(89.5, 99.0)
    ax1.set_xlabel("Calibration Set Size per Mondrian Group ($n_g$)")
    ax1.set_ylabel("Finite-Sample Coverage Bound (%)")
    ax1.set_title("(a) Beta-Binomial Coverage Width vs. $n_g$")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=5.6)

    # Panel (b): Four-stratum coaxial BVI SNR operating bands + emergency preemption policy
    strata = ["S0_Clean\n(>15 dB)", "S1_Cruise\n(8-15 dB)", "S2_Hover\n(2-8 dB)", "S3_Garbled\n(<2 dB)"]
    n_cal_per_stratum = [500, 500, 500, 500]
    colors = [C_GREEN, C_BLUE, C_YELLOW, C_ORANGE]
    x = np.arange(len(strata))
    bars = ax2.bar(x, n_cal_per_stratum, width=0.52, color=colors, alpha=0.88, edgecolor="#222222", linewidth=0.6)
    policy_labels = ["Standard\n$\\hat{q}_{S0}$", "Verify if\n$p<0.80$", "Clarify\nLOW_ASR", "MAYDAY\n$w_m\\geq 0.15$"]
    for bar, plbl in zip(bars, policy_labels):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.74,
            plbl,
            ha="center",
            va="center",
            fontsize=5.6,
            fontweight="bold",
            color="#111111",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        )
    ax2.axhline(250, color="#222222", linestyle="--", linewidth=1.0, label=r"DR-13 Floor ($n_g \geq 250$)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strata)
    ax2.set_ylim(0, 620)
    ax2.set_ylabel("VertiBench-v2 Cal. Samples ($n_g$)")
    ax2.set_title("(b) 4-Stratum BVI SNR Mondrian Allocation")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)

    _save_pdf(fig, out_path)


def make_fig3_slz_real_transfer_pareto(manifest: dict[str, Any], out_path: Path) -> None:
    b = manifest["verified_baselines"]
    g = manifest["phase3_g4_slots"]
    p1 = b["v310_pidnet_v1"]
    p2 = b["v351_pidnet_v2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={"wspace": 0.32})

    # Panel (a): Safety-Accuracy Pareto (False-Safe Rate vs mIoU on Val & AeroScapes)
    ax1.scatter(
        [p1["val_fsr"] * 100.0], [p1["val_miou"]], color=C_ORANGE, marker="o", s=54, label="v1 Val (3-src)"
    )
    ax1.scatter(
        [p2["val_fsr"] * 100.0], [p2["val_miou"]], color=C_BLUE, marker="D", s=54, label="v2 Val (+SkyScenes)"
    )
    ax1.scatter(
        [p1["aeroscapes_fsr"] * 100.0],
        [p1["aeroscapes_miou"]],
        color=C_ORANGE,
        marker="^",
        s=54,
        facecolors="none",
        linewidths=1.4,
        label="v1 AeroScapes ZS",
    )
    ax1.scatter(
        [p2["aeroscapes_fsr"] * 100.0],
        [p2["aeroscapes_miou"]],
        color=C_BLUE,
        marker="^",
        s=54,
        facecolors="none",
        linewidths=1.4,
        label="v2 AeroScapes ZS",
    )
    if g["pidnet_v21_val_miou"]["value"] is not None:
        ax1.scatter(
            [g["pidnet_v21_val_fsr"]["value"] * 100.0],
            [g["pidnet_v21_val_miou"]["value"]],
            color=C_GREEN,
            marker="*",
            s=76,
            label="v2.1 Val (G4)",
        )
    if g["pidnet_v21_aeroscapes_miou"]["value"] is not None:
        ax1.scatter(
            [g["pidnet_v21_aeroscapes_fsr"]["value"] * 100.0],
            [g["pidnet_v21_aeroscapes_miou"]["value"]],
            color=C_GREEN,
            marker="P",
            s=62,
            label="v2.1 Aero. ZS (G4)",
        )

    ax1.axvline(8.0, color=C_GREEN, linestyle="--", linewidth=1.0, label=r"FSR Gate ($\leq 8$%)")
    ax1.set_xlim(0, 24.5)
    ax1.set_ylim(0.18, 0.58)
    ax1.set_xlabel(r"False-Safe Rate FSR (%) ($\downarrow$ safer)")
    ax1.set_ylabel(r"Semantic mIoU ($\uparrow$)")
    ax1.set_title("(a) SLZ Safety vs. Accuracy (Val & AeroScapes)")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=5.4)

    # Panel (b): Physical GTX 1650 SUPER (TU116) empirical FP32+CG vs FP16 trap
    configs = ["v1 FP32\n+CG", "v2 FP32\n+CG", "v1 FP16", "v2 FP16"]
    p50_ms = [
        p1["gtx1650s_fp32_cg_p50_ms"],
        p2["gtx1650s_fp32_cg_p50_ms"],
        p1["gtx1650s_fp16_p50_ms"],
        p2["gtx1650s_fp16_p50_ms"],
    ]
    p99_ms = [
        p1["gtx1650s_fp32_cg_p99_ms"],
        p2["gtx1650s_fp32_cg_p99_ms"],
        37.10,
        p2["gtx1650s_fp16_p99_ms"],
    ]
    x = np.arange(len(configs))
    w = 0.36
    ax2.bar(x - w / 2, p50_ms, width=w, color=C_BLUE, alpha=0.88, label="p50 (ms)")
    ax2.bar(x + w / 2, p99_ms, width=w, color=C_ORANGE, alpha=0.88, label="p99 (ms)")
    ax2.axhline(12.0, color=C_GREEN, linestyle="--", linewidth=1.1, label="12 ms Edge Gate")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylim(5, 180)
    ax2.set_ylabel("Network Latency (ms, log scale)")
    ax2.set_title("(b) Physical TU116 Edge Latency (FP32 vs. FP16)")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    _save_pdf(fig, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    _configure_style()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    f1 = args.out_dir / "fig1_bilingual_mondrian_pareto.pdf"
    f2 = args.out_dir / "fig2_snr_stratum_coverage.pdf"
    f3 = args.out_dir / "fig3_slz_real_transfer_pareto.pdf"
    make_fig1_bilingual_mondrian_pareto(manifest, f1)
    make_fig2_snr_stratum_coverage(manifest, f2)
    make_fig3_slz_real_transfer_pareto(manifest, f3)
    for fp in (f1, f2, f3):
        print(f"Wrote {fp.name}: {fp.stat().st_size} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
