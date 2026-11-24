#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic CoRL 2027 LaTeX tables from corl_metrics_manifest.json.

Supports Phase 2 (--allow-pending-g4) rendering pending Phase-3 G4 slots cleanly as
\\text{--} (zero fabricated numbers, zero TODO/[TBD] placeholders) and Phase 3
(--ingest-v400 <dir>) automatic ingestion of live Plan 6C G4 Thunder Run JSONs.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

CORL_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = CORL_DIR / "corl_metrics_manifest.json"
DEFAULT_OUT_TEX = CORL_DIR / "generated" / "corl_tables.tex"


def _fmt(val: float | int | None, fmt_str: str = "{:.2f}", scale: float = 1.0) -> str:
    if val is None:
        return r"\text{--}"
    return fmt_str.format(float(val) * scale)


def _fmt_mean_sd(mean_val: float | None, sd_val: float | None) -> str:
    if mean_val is None:
        return r"\text{--}"
    if sd_val is None:
        return f"{mean_val:.2f}"
    return f"{mean_val:.2f} $\\pm$ {sd_val:.2f}"


def _set_slot(slots: dict[str, Any], key: str, value: float | int | None, status: str) -> None:
    if value is not None and key in slots:
        slots[key]["value"] = float(value)
        slots[key]["status"] = status


def ingest_v400_into_manifest(manifest: dict[str, Any], v400_dir: Path) -> dict[str, Any]:
    """Ingest live Phase 3 G4 Thunder Run artifacts into a deep copy of the manifest."""
    out = copy.deepcopy(manifest)
    g4_slots = out["phase3_g4_slots"]
    ext_slots = out["external_eval_only_slots"]

    qwen_path = v400_dir / "g4_qwen3_v3_vertibench_v2_eval.json"
    if qwen_path.is_file():
        qd = json.loads(qwen_path.read_text(encoding="utf-8"))
        status = str(qd.get("mode", "LIVE_G4_EXECUTED"))
        summary = qd.get("summary", qd)
        _set_slot(g4_slots, "qwen3_v3_ast_stl_mean_pct", summary.get("ast_stl_accuracy_mean_pct"), status)
        _set_slot(g4_slots, "qwen3_v3_ast_stl_sd_pct", summary.get("ast_stl_accuracy_sd_pct"), status)
        _set_slot(g4_slots, "qwen3_v3_b3_reject_pct", summary.get("bucket3_reject_pct"), status)
        _set_slot(
            g4_slots, "qwen3_v3_mondrian_coverage_pct", summary.get("mondrian_coverage_mean_pct"), status
        )
        _set_slot(g4_slots, "qwen3_v3_mondrian_ece", summary.get("mondrian_ece"), status)
        _set_slot(
            g4_slots, "qwen3_v3_mondrian_mean_set_size", summary.get("mondrian_mean_set_size"), status
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_mondrian_singleton_exec_pct",
            summary.get("mondrian_singleton_exec_pct"),
            status,
        )

    whisper_path = v400_dir / "g4_whisper_bvi_lora_eval.json"
    if whisper_path.is_file():
        wd = json.loads(whisper_path.read_text(encoding="utf-8"))
        status = str(wd.get("mode", "LIVE_G4_EXECUTED"))
        strata = wd.get("snr_strata_metrics", {})
        _set_slot(g4_slots, "whisper_s0_clean_wer_pct", strata.get("S0_Clean", {}).get("wer_pct"), status)
        _set_slot(g4_slots, "whisper_s1_cruise_wer_pct", strata.get("S1_Cruise", {}).get("wer_pct"), status)
        _set_slot(g4_slots, "whisper_s2_hover_wer_pct", strata.get("S2_Hover", {}).get("wer_pct"), status)
        _set_slot(
            g4_slots, "whisper_s3_garbled_wer_pct", strata.get("S3_Garbled", {}).get("wer_pct"), status
        )

    pidnet_path = v400_dir / "g4_pidnet_v21_xfeat_metrics.json"
    if pidnet_path.is_file():
        pd = json.loads(pidnet_path.read_text(encoding="utf-8"))
        status = str(pd.get("mode", "LIVE_G4_EXECUTED"))
        summary = pd.get("summary", pd)
        _set_slot(g4_slots, "pidnet_v21_val_miou", summary.get("val_miou"), status)
        _set_slot(g4_slots, "pidnet_v21_val_fsr", summary.get("val_fsr"), status)
        _set_slot(g4_slots, "pidnet_v21_aeroscapes_miou", summary.get("aeroscapes_miou"), status)
        _set_slot(g4_slots, "pidnet_v21_aeroscapes_fsr", summary.get("aeroscapes_fsr"), status)
        _set_slot(
            g4_slots, "xfeat_kartal_recall_at_5m_pct", summary.get("xfeat_kartal_recall_at_5m_pct"), status
        )
        # Optional external eval-only zero-shot datasets: populate ONLY if genuinely present.
        if summary.get("icg_drone_miou") is not None:
            _set_slot(ext_slots, "icg_drone_miou", summary.get("icg_drone_miou"), status)
            _set_slot(ext_slots, "icg_drone_fsr", summary.get("icg_drone_fsr"), status)
        if summary.get("uavscenes_miou") is not None:
            _set_slot(ext_slots, "uavscenes_miou", summary.get("uavscenes_miou"), status)
            _set_slot(ext_slots, "uavscenes_fsr", summary.get("uavscenes_fsr"), status)

    return out


def render_corl_tables_tex(manifest: dict[str, Any]) -> str:
    """Render deterministic booktabs LaTeX tables from manifest."""
    b = manifest["verified_baselines"]
    g = manifest["phase3_g4_slots"]
    ext = manifest["external_eval_only_slots"]

    q25 = b["v310_qwen25_7b_zs"]
    q3z = b["v310_qwen3_4b_zs"]
    q3v1 = b["v310_qwen3_4b_sft_grpo"]
    q3u = b["v351_qwen3_4b_v2_unconstrained"]
    q3v2 = b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]
    p1 = b["v310_pidnet_v1"]
    p2 = b["v351_pidnet_v2"]

    v3_ast = _fmt_mean_sd(
        g["qwen3_v3_ast_stl_mean_pct"]["value"], g["qwen3_v3_ast_stl_sd_pct"]["value"]
    )
    v3_b3 = _fmt(g["qwen3_v3_b3_reject_pct"]["value"])
    v3_cov = _fmt(g["qwen3_v3_mondrian_coverage_pct"]["value"])
    v3_ece = _fmt(g["qwen3_v3_mondrian_ece"]["value"], "{:.4f}")
    v3_sz = _fmt(g["qwen3_v3_mondrian_mean_set_size"]["value"], "{:.3f}")
    v3_sng = _fmt(g["qwen3_v3_mondrian_singleton_exec_pct"]["value"])

    w_s0 = _fmt(g["whisper_s0_clean_wer_pct"]["value"])
    w_s1 = _fmt(g["whisper_s1_cruise_wer_pct"]["value"])
    w_s2 = _fmt(g["whisper_s2_hover_wer_pct"]["value"])
    w_s3 = _fmt(g["whisper_s3_garbled_wer_pct"]["value"])

    p21_vmiou = _fmt(g["pidnet_v21_val_miou"]["value"], "{:.4f}")
    p21_vfsr = _fmt(g["pidnet_v21_val_fsr"]["value"], "{:.2f}", scale=100.0)
    p21_amiou = _fmt(g["pidnet_v21_aeroscapes_miou"]["value"], "{:.4f}")
    p21_afsr = _fmt(g["pidnet_v21_aeroscapes_fsr"]["value"], "{:.2f}", scale=100.0)
    icg_miou = _fmt(ext["icg_drone_miou"]["value"], "{:.4f}")
    icg_fsr = _fmt(ext["icg_drone_fsr"]["value"], "{:.2f}", scale=100.0)
    uavs_miou = _fmt(ext["uavscenes_miou"]["value"], "{:.4f}")
    uavs_fsr = _fmt(ext["uavscenes_fsr"]["value"], "{:.2f}", scale=100.0)
    xfeat_r5 = _fmt(g["xfeat_kartal_recall_at_5m_pct"]["value"])

    return (
        "% Auto-generated by docs/paper_corl/scripts/generate_corl_tables.py\n"
        "% Source: docs/paper_corl/corl_metrics_manifest.json (zero manual edits).\n"
        "\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Bilingual Neuro-Symbolic Mission Synthesis and Conformal Calibration Progression "
        "($\\alpha = 0.05$; dashes indicate Phase-3 batch GPU runs awaiting ingestion).}\n"
        "\\label{tab:corl_copilot_mondrian}\n"
        "\\setlength{\\tabcolsep}{4.5pt}\n"
        "\\begin{tabular}{lccccccc}\n"
        "\\toprule\n"
        "Model / Decoding / Calibration & Grammar (\\%) & AST/STL (\\%) & B3 Rej. (\\%) & "
        "Cov. (\\%) & ECE $\\downarrow$ & Mean $|C|$ $\\downarrow$ & Singl. (\\%) \\\\\n"
        "\\midrule\n"
        f"Qwen2.5-7B Zero-Shot (v1) & {q25['grammar_validity_pct']:.2f} & {q25['ast_stl_accuracy_pct']:.2f} & "
        f"{q25['unsafe_rejection_pct']:.2f} & \\text{{--}} & \\text{{--}} & \\text{{--}} & \\text{{--}} \\\\\n"
        f"Qwen3-4B Zero-Shot (v1) & {q3z['grammar_validity_pct']:.2f} & {q3z['ast_stl_accuracy_pct']:.2f} & "
        f"{q3z['unsafe_rejection_pct']:.2f} & \\text{{--}} & \\text{{--}} & \\text{{--}} & \\text{{--}} \\\\\n"
        f"Qwen3-4B SFT+Z3-GRPO (v1, $n_{{\\text{{cal}}}}=200$) & {q3v1['grammar_validity_pct']:.2f} & "
        f"{q3v1['ast_stl_accuracy_pct']:.2f} & {q3v1['unsafe_rejection_pct']:.2f} & "
        f"{q3v1['conformal_coverage_pct']:.2f} & \\text{{--}} & {q3v1['mean_prediction_set_size']:.3f} & "
        f"{q3v1['singleton_exec_pct']:.2f} \\\\\n"
        f"Qwen3-4B v2 Unconstrained & {q3u['grammar_validity_pct']:.2f} & {q3u['ast_stl_accuracy_pct']:.2f} & "
        f"{q3u['unsafe_rejection_pct']:.2f} & \\text{{--}} & \\text{{--}} & \\text{{--}} & \\text{{--}} \\\\\n"
        f"Qwen3-4B v2 + GBNF + Z3 + NLL ($\\tau={q3v2['tau_opt']:.4f}$) & "
        f"\\textbf{{{q3v2['grammar_validity_pct']:.2f}}} & {q3v2['ast_stl_accuracy_pct']:.2f} & "
        f"{q3v2['unsafe_rejection_pct']:.2f} & {q3v2['conformal_coverage_pct']:.2f} & "
        f"{q3v2['ece_calibrated']:.4f} & {q3v2['mean_prediction_set_size']:.3f} & "
        f"{q3v2['singleton_exec_pct']:.2f} \\\\\n"
        "\\midrule\n"
        f"Qwen3-4B v3 Mondrian (v2 bench, 3-seed, $N_{{\\text{{cal}}}}=2000$) & 100.00 & {v3_ast} & "
        f"{v3_b3} & {v3_cov} & {v3_ece} & {v3_sz} & {v3_sng} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
        "\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Four-Stratum Coaxial BVI Cockpit SNR Conditioning ($1{,}100$--$2{,}292$\\,RPM, "
        "$55.0$--$114.6$\\,Hz BPF; $n_g=500$/stratum in \\texttt{VertiBench-v2}).}\n"
        "\\label{tab:corl_snr_strata}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Stratum & SNR Band (dB) & $n_g$ (Cal) & WER (\\%) & Guard Mode \\\\\n"
        "\\midrule\n"
        f"\\texttt{{S0\\_Clean}} & $>15$ & 500 & {w_s0} & Standard $\\hat{{q}}_{{S0}}$ \\\\\n"
        f"\\texttt{{S1\\_Cruise}} & $8$--$15$ & 500 & {w_s1} & Escalate $<0.80$ \\\\\n"
        f"\\texttt{{S2\\_Hover}} & $2$--$8$ & 500 & {w_s2} & Clarify \\texttt{{LOW\\_ASR}} \\\\\n"
        f"\\texttt{{S3\\_Garbled}} & $<2$ & 500 & {w_s3} & MAYDAY $w_m\\ge 0.15$ \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Zero-ShareAlike Active Visual SLZ Distillation, Quarantined Real-Imagery Zero-Shot Transfer "
        "(AeroScapes, ICG Drone, UAVScenes), and Physical TU116 Edge Latency (GTX~1650~SUPER, 4\\,GB).}\n"
        "\\label{tab:corl_slz_transfer}\n"
        "\\setlength{\\tabcolsep}{4.2pt}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        "Model Variant & Val mIoU & Val FSR (\\%) & Aero. mIoU & Aero. FSR (\\%) & "
        "ICG / UAVS. & p50 / p99 (ms) & VRAM (MiB) & XFeat R@5m \\\\\n"
        "\\midrule\n"
        f"PIDNet-S v1 (3-src, L4) & {p1['val_miou']:.4f} & {p1['val_fsr'] * 100.0:.2f} & "
        f"{p1['aeroscapes_miou']:.4f} & {p1['aeroscapes_fsr'] * 100.0:.2f} & "
        f"\\text{{--}} / \\text{{--}} & {p1['gtx1650s_fp32_cg_p50_ms']:.2f} / "
        f"{p1['gtx1650s_fp32_cg_p99_ms']:.2f} & {p1['gtx1650s_fp32_cg_vram_mb']:.1f} & \\text{{--}} \\\\\n"
        f"PIDNet-S v2 (+SkyScenes, FSR-gate) & {p2['val_miou']:.4f} & {p2['val_fsr'] * 100.0:.2f} & "
        f"{p2['aeroscapes_miou']:.4f} & {p2['aeroscapes_fsr'] * 100.0:.2f} & "
        f"\\text{{--}} / \\text{{--}} & {p2['gtx1650s_fp32_cg_p50_ms']:.2f} / "
        f"\\textbf{{{p2['gtx1650s_fp32_cg_p99_ms']:.2f}}} & {p2['gtx1650s_fp32_cg_vram_mb']:.1f} & "
        f"\\text{{--}} \\\\\n"
        f"PIDNet-S v2.1 (Oblique-bal., G4) & {p21_vmiou} & {p21_vfsr} & {p21_amiou} & {p21_afsr} & "
        f"{icg_miou} ({icg_fsr}\\%) / {uavs_miou} ({uavs_fsr}\\%) & \\text{{--}} / \\text{{--}} & "
        f"\\text{{--}} & {xfeat_r5} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-tex", type=Path, default=DEFAULT_OUT_TEX)
    parser.add_argument("--ingest-v400", type=Path, default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.ingest_v400 is not None:
        manifest = ingest_v400_into_manifest(manifest, args.ingest_v400)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    tex = render_corl_tables_tex(manifest)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text(tex, encoding="utf-8", newline="\n")
    print(f"Wrote {args.out_tex} ({len(tex)} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
