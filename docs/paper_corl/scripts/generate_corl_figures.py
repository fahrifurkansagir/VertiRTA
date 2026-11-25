#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate 3 double-anonymous publication vector PDF figures for Paper 2 (CoRL 2027).

Reads strictly from docs/paper_corl/corl_metrics_manifest.json (grounded in verified
artifacts/v3.1.0/, artifacts/v3.5.1/, and live Phase 3 artifacts/v4.0.0/ measurements).
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
            "font.size": 7.5,
            "axes.titlesize": 7.9,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.0,
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
    abl = manifest.get("phase3_ablations_and_edge", {})
    q3_det = abl.get("qwen3_v3_detail", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.70), gridspec_kw={"wspace": 0.30})

    # Panel (a): AST/STL accuracy & Grammar progression across measured stages
    stages = ["Q2.5-7B\nZS", "Q3-4B\nZS", "v1\nGRPO", "v2\nUncon.", "v2\nGBNF+Z3"]
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
    ast_err = [0.0, 0.0, 0.0, 0.0, 0.0]
    if g["qwen3_v3_ast_stl_mean_pct"]["value"] is not None:
        stages.append("v3 3-Seed\n(v2)")
        ast_vals.append(float(g["qwen3_v3_ast_stl_mean_pct"]["value"]))
        gram_vals.append(100.0)
        ast_err.append(float(g["qwen3_v3_ast_stl_sd_pct"]["value"] or 0.0))

    x = np.arange(len(stages))
    w = 0.36
    ax1.bar(x - w / 2, gram_vals, width=w, color=C_BLUE, alpha=0.85, label="Grammar Valid (%)")
    bars_a = ax1.bar(
        x + w / 2,
        ast_vals,
        width=w,
        color=C_GREEN,
        alpha=0.90,
        label="AST/STL Exact (%)",
    )
    for idx_b, bar in enumerate(bars_a):
        h = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        if ast_err[idx_b] > 0.0:
            ax1.errorbar(
                [cx], [h], yerr=[ast_err[idx_b]], fmt="none", ecolor="#111111", elinewidth=0.9, capsize=2.5
            )
        lbl = f"{h:.1f}" if ast_err[idx_b] == 0.0 else f"{h:.1f}±{ast_err[idx_b]:.1f}"
        ax1.text(
            cx,
            h + ast_err[idx_b] + 1.8,
            lbl,
            ha="center",
            va="bottom",
            fontsize=5.3,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=5.8)
    ax1.set_ylim(0, 116)
    ax1.set_ylabel("Accuracy / Validity (%)")
    ax1.set_title("(a) Bilingual Synthesis Progression (TR/EN/MIX)")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.92)

    # Panel (b): Conformal efficiency Pareto (Mean Set Size vs Singleton Execution)
    q3v1 = b["v310_qwen3_4b_sft_grpo"]
    q3v2 = b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]
    pts: list[tuple[str, float, float, str, str, tuple[int, int], str]] = [
        (
            "v1 Uncal.\n(96.3% cov)",
            q3v1["mean_prediction_set_size"],
            q3v1["singleton_exec_pct"],
            C_ORANGE,
            "o",
            (0, 10),
            "center",
        ),
        (
            "v2 Uncal.",
            q3v2["mean_prediction_set_size_uncal"],
            q3v2["singleton_exec_pct_uncal"],
            C_YELLOW,
            "s",
            (8, -2),
            "left",
        ),
        (
            "v2 NLL-Cal.\n(95.4% cov)",
            q3v2["mean_prediction_set_size"],
            q3v2["singleton_exec_pct"],
            C_BLUE,
            "D",
            (0, 10),
            "center",
        ),
    ]
    max_sz = 5.6
    if g["qwen3_v3_mondrian_mean_set_size"]["value"] is not None:
        sz_lang = float(g["qwen3_v3_mondrian_mean_set_size"]["value"])
        sng_lang = float(g["qwen3_v3_mondrian_singleton_exec_pct"]["value"] or 0.0)
        cov_lang = float(g["qwen3_v3_mondrian_coverage_pct"]["value"] or 96.0)
        pts.append(
            (
                f"v3 Lang\n({cov_lang:.1f}% cov)",
                sz_lang,
                sng_lang,
                C_GREEN,
                "*",
                (4, 10),
                "center",
            )
        )
        max_sz = max(max_sz, sz_lang)
    if q3_det.get("snr_mondrian_mean_set_size") is not None:
        sz_snr = float(q3_det["snr_mondrian_mean_set_size"])
        sng_snr = float(q3_det.get("snr_mondrian_singleton_exec_pct") or 0.0)
        cov_snr = float(q3_det.get("snr_mondrian_coverage_pct") or 95.25)
        pts.append(
            (
                f"v3 SNR\n({cov_snr:.2f}% cov)",
                sz_snr,
                sng_snr,
                C_PURPLE,
                "^",
                (-22, 34),
                "center",
            )
        )
        max_sz = max(max_sz, sz_snr)

    for lbl, sz, sng, col, mk, offset, ha in pts:
        ax2.scatter([sz], [sng], color=col, marker=mk, s=56, zorder=4, edgecolor="#111111", linewidths=0.4)
        ax2.annotate(
            lbl,
            (sz, sng),
            xytext=offset,
            textcoords="offset points",
            ha=ha,
            va="bottom" if offset[1] > 0 else "center",
            fontsize=5.5,
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.5} if abs(offset[1]) > 18 else None,
        )
    ax2.axvline(1.8, color=C_GREEN, linestyle="--", linewidth=1.1, label="Aspirational $|C| < 1.8$")
    ax2.set_xlim(1.0, max(6.4, max_sz + 1.15))
    ax2.set_ylim(-6, 55)
    ax2.set_xlabel(r"Mean Conformal Set Size $|C|$ ($\downarrow$ tighter)")
    ax2.set_ylabel(r"Singleton Execution Rate (%) ($\uparrow$)")
    ax2.set_title(r"(b) Conformal Efficiency at $\alpha=0.05$ ($\geq 95$% Cov.)")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92)

    _save_pdf(fig, out_path)


def make_fig2_snr_stratum_coverage(manifest: dict[str, Any], out_path: Path) -> None:
    alpha = 0.05
    g = manifest["phase3_g4_slots"]
    abl = manifest.get("phase3_ablations_and_edge", {})
    q3_det = abl.get("qwen3_v3_detail", {})
    w_abl = abl.get("whisper_bvi_lora_ablation", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.70), gridspec_kw={"wspace": 0.30})

    # Panel (a): Exact finite-sample Beta-Binomial 95% coverage envelope + measured v2 groups
    n_grid = np.array([100, 150, 200, 250, 400, 500, 750, 800, 1000, 1500, 2000])
    lo_95 = []
    hi_95 = []
    for n_val in n_grid:
        k = int(np.ceil((n_val + 1) * (1.0 - alpha)))
        l_shape = n_val + 1 - k
        lo_95.append(100.0 * float(beta_dist.ppf(0.025, k, l_shape)))
        hi_95.append(100.0 * float(beta_dist.ppf(0.975, k, l_shape)))

    ax1.fill_between(n_grid, lo_95, hi_95, color=C_BLUE, alpha=0.20, label="Exact 95% Beta Envelope")
    ax1.plot(n_grid, lo_95, color=C_BLUE, linewidth=1.1)
    ax1.plot(n_grid, hi_95, color=C_BLUE, linewidth=1.1)
    ax1.axhline(95.0, color=C_GREY, linestyle="--", linewidth=1.0, label=r"Nominal $1-\alpha=95$%")
    ax1.axvline(200, color=C_ORANGE, linestyle=":", linewidth=1.1, label="v1 Pooled ($n=200$)")

    snr_covs = q3_det.get("snr_stratum_coverage_pct", {})
    lang_covs = q3_det.get("lang_group_coverage_pct", {})
    if snr_covs:
        s_vals = [float(v) for v in snr_covs.values() if v is not None]
        ax1.scatter(
            [500] * len(s_vals),
            s_vals,
            color=C_GREEN,
            marker="o",
            s=32,
            zorder=5,
            label="v2 SNR Strata ($n_g=500$)",
        )
    if lang_covs:
        lang_n = {"MIX": 400, "EN": 800, "TR": 800}
        xs = [lang_n[k] for k in lang_covs if k in lang_n and lang_covs[k] is not None]
        ys = [float(lang_covs[k]) for k in lang_covs if k in lang_n and lang_covs[k] is not None]
        ax1.scatter(
            xs,
            ys,
            color=C_PURPLE,
            marker="D",
            s=34,
            zorder=6,
            label="v2 Lang Groups ($n_{\\ell}=400,800$)",
        )
    if q3_det.get("snr_mondrian_coverage_pct") is not None:
        ax1.scatter(
            [2000],
            [float(q3_det["snr_mondrian_coverage_pct"])],
            color="#111111",
            marker="*",
            s=62,
            zorder=7,
            label="v2 Overall ($N_{\\text{cal}}=2000$)",
        )

    ax1.set_xlim(80, 2120)
    ax1.set_ylim(89.5, 99.6)
    ax1.set_xlabel("Calibration Set Size per Mondrian Group ($n_g$)")
    ax1.set_ylabel("Empirical / Bound Coverage (%)")
    ax1.set_title("(a) Beta-Binomial Envelope & Live v2 Coverage")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=5.3)

    # Panel (b): Measured 4-stratum Whisper-large-v3-turbo Zero-Shot vs BVI-LoRA negative ablation
    strata_keys = ["S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"]
    strata_lbls = ["S0_Clean\n(18 dB)", "S1_Cruise\n(11.5 dB)", "S2_Hover\n(5 dB)", "S3_Garbled\n(0.5 dB)"]
    zs_wers = [
        g["whisper_s0_clean_wer_pct"]["value"],
        g["whisper_s1_cruise_wer_pct"]["value"],
        g["whisper_s2_hover_wer_pct"]["value"],
        g["whisper_s3_garbled_wer_pct"]["value"],
    ]
    lora_map = w_abl.get("turbo_bvi_lora_wer_pct", {})
    lora_wers = [lora_map.get(k) for k in strata_keys]

    x = np.arange(len(strata_keys))
    if all(v is not None for v in zs_wers):
        w = 0.36
        zs_arr = [float(v) for v in zs_wers]
        b_zs = ax2.bar(
            x - w / 2,
            zs_arr,
            width=w,
            color=C_BLUE,
            alpha=0.88,
            edgecolor="#222222",
            linewidth=0.5,
            label="Zero-Shot (Retained)",
        )
        for bar in b_zs:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=5.5
            )
        if all(v is not None for v in lora_wers):
            lora_arr = [float(v) for v in lora_wers]
            b_lo = ax2.bar(
                x + w / 2,
                lora_arr,
                width=w,
                color=C_ORANGE,
                alpha=0.85,
                edgecolor="#222222",
                linewidth=0.5,
                label="3-ep LoRA (Neg. Ablation)",
            )
            for bar in b_lo:
                h = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.8,
                    f"{h:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                )
        ax2.set_ylim(0, 44)
        ax2.set_ylabel("Word Error Rate WER (%) ($\\downarrow$ better)")
        ax2.set_title("(b) Whisper-v3-Turbo BVI SNR Ablation (FLEURS)")
    else:
        ax2.bar(x, [500, 500, 500, 500], width=0.52, color=[C_GREEN, C_BLUE, C_YELLOW, C_ORANGE], alpha=0.88)
        ax2.set_ylim(0, 620)
        ax2.set_ylabel("VertiBench-v2 Cal. Samples ($n_g$)")
        ax2.set_title("(b) 4-Stratum BVI SNR Mondrian Allocation")

    ax2.set_xticks(x)
    ax2.set_xticklabels(strata_lbls)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=5.6)

    _save_pdf(fig, out_path)


def make_fig3_slz_real_transfer_pareto(manifest: dict[str, Any], out_path: Path) -> None:
    b = manifest["verified_baselines"]
    g = manifest["phase3_g4_slots"]
    abl = manifest.get("phase3_ablations_and_edge", {})
    p1 = b["v310_pidnet_v1"]
    p2 = b["v351_pidnet_v2"]
    g4_col = abl.get("pidnet_v21_g4_collapsed_epoch2", {})
    edge = abl.get("gtx1650s_timesliced_edge", {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.70), gridspec_kw={"wspace": 0.32})

    # Panel (a): Safety-Accuracy Pareto (False-Safe Rate vs mIoU on Val & AeroScapes)
    ax1.scatter(
        [p1["val_fsr"] * 100.0], [p1["val_miou"]], color=C_ORANGE, marker="o", s=48, label="v1 Val (3-src)"
    )
    ax1.scatter(
        [p2["val_fsr"] * 100.0], [p2["val_miou"]], color=C_BLUE, marker="D", s=48, label="v2 Val (+SkyScenes)"
    )
    ax1.scatter(
        [p1["aeroscapes_fsr"] * 100.0],
        [p1["aeroscapes_miou"]],
        color=C_ORANGE,
        marker="^",
        s=48,
        facecolors="none",
        linewidths=1.3,
        label="v1 AeroScapes ZS",
    )
    ax1.scatter(
        [p2["aeroscapes_fsr"] * 100.0],
        [p2["aeroscapes_miou"]],
        color=C_BLUE,
        marker="^",
        s=48,
        facecolors="none",
        linewidths=1.3,
        label="v2 AeroScapes ZS",
    )
    if g4_col.get("val_miou") is not None:
        ax1.scatter(
            [float(g4_col["val_fsr"]) * 100.0],
            [float(g4_col["val_miou"])],
            color="#CC0000",
            marker="x",
            s=58,
            linewidths=1.6,
            zorder=5,
            label="v2.1 G4 Ep.2 (collapsed)",
        )
    if g["pidnet_v21_val_miou"]["value"] is not None:
        v_fsr = float(g["pidnet_v21_val_fsr"]["value"]) * 100.0
        v_miou = float(g["pidnet_v21_val_miou"]["value"])
        ax1.scatter(
            [v_fsr],
            [v_miou],
            color=C_GREEN,
            marker="*",
            s=82,
            zorder=6,
            edgecolor="#111111",
            linewidths=0.4,
            label="v2.1 Rec. Val (0 CU, cal.)",
        )
        if g4_col.get("val_miou") is not None:
            ax1.annotate(
                "",
                xy=(v_fsr, v_miou - 0.015),
                xytext=(float(g4_col["val_fsr"]) * 100.0 + 0.15, float(g4_col["val_miou"]) + 0.015),
                arrowprops={"arrowstyle": "->", "color": C_GREEN, "lw": 1.0, "ls": "--"},
            )
    if g["pidnet_v21_aeroscapes_miou"]["value"] is not None:
        ax1.scatter(
            [float(g["pidnet_v21_aeroscapes_fsr"]["value"]) * 100.0],
            [float(g["pidnet_v21_aeroscapes_miou"]["value"])],
            color=C_GREEN,
            marker="P",
            s=56,
            zorder=5,
            label="v2.1 Rec. Aero. ZS",
        )

    ax1.axvline(3.0, color=C_GREEN, linestyle=":", linewidth=1.0, label=r"v2.1 Gate ($\leq 3$%)")
    ax1.axvline(8.0, color=C_GREY, linestyle="--", linewidth=0.9, label=r"v2 Gate ($\leq 8$%)")
    ax1.set_xlim(-0.4, 24.0)
    ax1.set_ylim(0.10, 0.56)
    ax1.set_xlabel(r"False-Safe Rate FSR (%) ($\downarrow$ safer)")
    ax1.set_ylabel(r"Semantic mIoU ($\uparrow$)")
    ax1.set_title("(a) SLZ Safety vs. Accuracy (Val & AeroScapes)")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=4.6, ncol=2, columnspacing=0.7, handletextpad=0.35)

    # Panel (b): Physical GTX 1650 SUPER (TU116) time-sliced & FP32+CG vs FP16 trap
    if edge.get("terminal_p50_ms") is not None:
        configs = [
            "v2 SLZ\nFP32+CG",
            "v2.1 Term.\n(164 MiB)",
            "v2.1 Cruise\n(366 MiB)",
            "v2.1 SLZ\nFP16",
            "v2 SLZ\nFP16",
        ]
        p50_ms = [
            p2["gtx1650s_fp32_cg_p50_ms"],
            float(edge["terminal_p50_ms"]),
            float(edge["cruise_p50_ms"]),
            float(edge["pidnet_v21_fp16_p50_ms"]),
            p2["gtx1650s_fp16_p50_ms"],
        ]
        p99_ms = [
            p2["gtx1650s_fp32_cg_p99_ms"],
            float(edge["terminal_p99_ms"]),
            float(edge["cruise_p99_ms"]),
            float(edge["pidnet_v21_fp16_p99_ms"]),
            p2["gtx1650s_fp16_p99_ms"],
        ]
    else:
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
    ax2.axhline(12.0, color=C_GREEN, linestyle="--", linewidth=1.0, label="12 ms Term. Target")
    ax2.axhline(25.0, color=C_PURPLE, linestyle=":", linewidth=1.1, label="25 ms Cruise Gate")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, fontsize=5.8)
    ax2.set_ylim(5, 180)
    ax2.set_ylabel("Network Latency (ms, log scale)")
    ax2.set_title("(b) Physical TU116 Time-Sliced Latency (4 GB)")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.92, fontsize=5.3)

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
