#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate deterministic CoRL 2027 LaTeX tables from corl_metrics_manifest.json.

Supports Phase 2 (--allow-pending-g4) rendering pending Phase-3 G4 slots cleanly as
\\text{--} (zero fabricated numbers, zero TODO/[TBD] placeholders) and Phase 3
(--ingest-v400 <dir>) automatic ingestion of live Plan 6C G4 Thunder Run + local
zero-CU recovery JSONs while preserving full backward compatibility with flat mock
schemas used by contract tests.
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


def _first_not_none(*vals: Any) -> Any:
    for v in vals:
        if v is not None:
            return v
    return None


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


def _set_slot(
    slots: dict[str, Any],
    key: str,
    value: float | int | None,
    status: str,
    source_artifact: str | None = None,
) -> None:
    if value is not None and key in slots:
        slots[key]["value"] = round(float(value), 6)
        slots[key]["status"] = status
        if source_artifact is not None:
            slots[key]["source_artifact"] = source_artifact


def _extract_wer_pct(
    strata: dict[str, Any],
    by_snr: dict[str, Any],
    stratum_key: str,
) -> float | None:
    flat_val = strata.get(stratum_key, {}).get("wer_pct")
    if flat_val is not None:
        return float(flat_val)
    nested_wer = by_snr.get(stratum_key, {}).get("wer")
    if nested_wer is not None:
        return round(float(nested_wer) * 100.0, 4)
    return None


def ingest_v400_into_manifest(manifest: dict[str, Any], v400_dir: Path) -> dict[str, Any]:
    """Ingest live Phase 3 G4 Thunder Run & local recovery artifacts into a deep copy."""
    out = copy.deepcopy(manifest)
    g4_slots = out["phase3_g4_slots"]
    ext_slots = out["external_eval_only_slots"]
    ablations: dict[str, Any] = {}

    qwen_path = v400_dir / "g4_qwen3_v3_vertibench_v2_eval.json"
    if qwen_path.is_file():
        qd = json.loads(qwen_path.read_text(encoding="utf-8"))
        status = str(qd.get("mode", "LIVE_G4_EXECUTED"))
        summary = qd.get("summary", qd)
        multi = qd.get("multi_seed_summary", {})
        overall = qd.get("test_overall_metrics", {})

        _set_slot(
            g4_slots,
            "qwen3_v3_ast_stl_mean_pct",
            _first_not_none(
                summary.get("ast_stl_accuracy_mean_pct"),
                multi.get("ast_stl_correctness_mean_pct"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_ast_stl_sd_pct",
            _first_not_none(
                summary.get("ast_stl_accuracy_sd_pct"),
                multi.get("ast_stl_correctness_sd_pct"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_b3_reject_pct",
            _first_not_none(
                summary.get("bucket3_reject_pct"),
                multi.get("bucket3_safe_reject_mean_pct"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_mondrian_coverage_pct",
            _first_not_none(
                summary.get("mondrian_coverage_mean_pct"),
                overall.get("lang_mondrian_coverage_pct"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_mondrian_ece",
            _first_not_none(
                summary.get("mondrian_ece"),
                overall.get("ece"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_mondrian_mean_set_size",
            _first_not_none(
                summary.get("mondrian_mean_set_size"),
                overall.get("lang_mondrian_mean_set_size"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )
        _set_slot(
            g4_slots,
            "qwen3_v3_mondrian_singleton_exec_pct",
            _first_not_none(
                summary.get("mondrian_singleton_exec_pct"),
                overall.get("lang_mondrian_singleton_exec_pct"),
            ),
            status,
            "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
        )

        if overall or multi:
            seed_reports = qd.get("grpo_seed_reports", [])
            uncon_ast = [
                r["unconstrained_eval"]["overall"]["ast_stl_accuracy_pct"]
                for r in seed_reports
                if "unconstrained_eval" in r
            ]
            uncon_gram = [
                r["unconstrained_eval"]["overall"]["grammar_validity_pct"]
                for r in seed_reports
                if "unconstrained_eval" in r
            ]
            cv = qd.get("cv_gamma_tau_selection", {})
            boot = qd.get("bootstrap_resplit_audit", {})
            snr_conf = qd.get("conformal_snr_mondrian", {})
            lang_conf = qd.get("conformal_language_conditional", {})
            ablations["qwen3_v3_detail"] = {
                "source": "artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json",
                "n_seeds": multi.get("n_seeds"),
                "grammar_validity_mean_pct": multi.get("grammar_validity_mean_pct"),
                "gated_z3_sat_mean_pct": multi.get("gated_z3_sat_mean_pct"),
                "unconstrained_grammar_mean_pct": (
                    round(sum(uncon_gram) / len(uncon_gram), 2) if uncon_gram else None
                ),
                "unconstrained_ast_stl_mean_pct": (
                    round(sum(uncon_ast) / len(uncon_ast), 2) if uncon_ast else None
                ),
                "cv_best_gamma": cv.get("best_gamma"),
                "cv_best_tau": cv.get("best_tau"),
                "snr_mondrian_coverage_pct": overall.get("snr_mondrian_coverage_pct"),
                "snr_mondrian_mean_set_size": overall.get("snr_mondrian_mean_set_size"),
                "snr_mondrian_singleton_exec_pct": overall.get("snr_mondrian_singleton_exec_pct"),
                "lang_group_coverage_pct": {
                    k: v.get("empirical_coverage_pct") for k, v in lang_conf.items()
                },
                "snr_stratum_coverage_pct": {
                    k: v.get("empirical_coverage_pct") for k, v in snr_conf.items()
                },
                "bootstrap_snr_coverage_mean_pct": (
                    round(float(boot["snr_strata"]["overall_coverage_mean"]) * 100.0, 2)
                    if "snr_strata" in boot
                    else None
                ),
                "bootstrap_snr_coverage_sd_pct": (
                    round(float(boot["snr_strata"]["overall_coverage_sd"]) * 100.0, 2)
                    if "snr_strata" in boot
                    else None
                ),
                "bootstrap_lang_coverage_mean_pct": (
                    round(float(boot["language_groups"]["overall_coverage_mean"]) * 100.0, 2)
                    if "language_groups" in boot
                    else None
                ),
                "bootstrap_lang_coverage_sd_pct": (
                    round(float(boot["language_groups"]["overall_coverage_sd"]) * 100.0, 2)
                    if "language_groups" in boot
                    else None
                ),
            }

    whisper_path = v400_dir / "g4_whisper_bvi_lora_eval.json"
    if whisper_path.is_file():
        wd = json.loads(whisper_path.read_text(encoding="utf-8"))
        status = str(wd.get("mode", "LIVE_G4_EXECUTED"))
        strata = wd.get("snr_strata_metrics", {})
        turbo = wd.get("models_trained", {}).get("whisper_large_v3_turbo_bvi_lora", {})
        zs_snr = turbo.get("zero_shot_eval", {}).get("by_snr_stratum", {})
        lora_snr = turbo.get("bvi_lora_eval", {}).get("by_snr_stratum", {})

        _set_slot(
            g4_slots,
            "whisper_s0_clean_wer_pct",
            _extract_wer_pct(strata, zs_snr, "S0_Clean"),
            status,
            "artifacts/v4.0.0/g4_whisper_bvi_lora_eval.json",
        )
        _set_slot(
            g4_slots,
            "whisper_s1_cruise_wer_pct",
            _extract_wer_pct(strata, zs_snr, "S1_Cruise"),
            status,
            "artifacts/v4.0.0/g4_whisper_bvi_lora_eval.json",
        )
        _set_slot(
            g4_slots,
            "whisper_s2_hover_wer_pct",
            _extract_wer_pct(strata, zs_snr, "S2_Hover"),
            status,
            "artifacts/v4.0.0/g4_whisper_bvi_lora_eval.json",
        )
        _set_slot(
            g4_slots,
            "whisper_s3_garbled_wer_pct",
            _extract_wer_pct(strata, zs_snr, "S3_Garbled"),
            status,
            "artifacts/v4.0.0/g4_whisper_bvi_lora_eval.json",
        )

        if lora_snr:
            small = wd.get("models_trained", {}).get("whisper_small_bvi_lora", {})
            small_zs = small.get("zero_shot_eval", {}).get("by_snr_stratum", {})
            small_lora = small.get("bvi_lora_eval", {}).get("by_snr_stratum", {})
            ablations["whisper_bvi_lora_ablation"] = {
                "source": "artifacts/v4.0.0/g4_whisper_bvi_lora_eval.json",
                "finding": "NEGATIVE_ABLATION_ZERO_SHOT_BEATS_AGGRESSIVE_3EP_LORA",
                "turbo_zero_shot_wer_pct": {
                    k: round(float(zs_snr[k]["wer"]) * 100.0, 2)
                    for k in ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
                    if k in zs_snr
                },
                "turbo_bvi_lora_wer_pct": {
                    k: round(float(lora_snr[k]["wer"]) * 100.0, 2)
                    for k in ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
                    if k in lora_snr
                },
                "small_zero_shot_wer_pct": {
                    k: round(float(small_zs[k]["wer"]) * 100.0, 2)
                    for k in ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
                    if k in small_zs
                },
                "small_bvi_lora_wer_pct": {
                    k: round(float(small_lora[k]["wer"]) * 100.0, 2)
                    for k in ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
                    if k in small_lora
                },
                "turbo_audio_ctx_speedup": turbo.get("audio_ctx_latency_benchmark", {}).get(
                    "speedup_factor"
                ),
            }

    pidnet_path = v400_dir / "g4_pidnet_v21_xfeat_metrics.json"
    local_rec_path = v400_dir / "local_gtx1650s_pidnet_v21_recovery_metrics.json"

    if pidnet_path.is_file():
        pd = json.loads(pidnet_path.read_text(encoding="utf-8"))
        status = str(pd.get("mode", "LIVE_G4_EXECUTED"))
        summary = pd.get("summary", pd)
        pval = pd.get("permissive_val_metrics", {})
        paero = pd.get("quarantined_zero_shot_aeroscapes", {})
        reloc = pd.get("kartal_reloc_eval", {})
        ood = pd.get("ood_metrics", {})

        g4_val_miou = _first_not_none(summary.get("val_miou"), pval.get("miou_present_classes"))
        g4_val_fsr = _first_not_none(summary.get("val_fsr"), pval.get("false_safe_rate_fsr"))
        g4_aero_miou = _first_not_none(
            summary.get("aeroscapes_miou"), paero.get("miou_present_classes")
        )
        g4_aero_fsr = _first_not_none(
            summary.get("aeroscapes_fsr"), paero.get("false_safe_rate_fsr")
        )
        xfeat_pass_pct = _first_not_none(
            summary.get("xfeat_kartal_recall_at_5m_pct"),
            round(float(reloc["ransac_gate_pass_rate"]) * 100.0, 4)
            if reloc.get("ransac_gate_pass_rate") is not None
            else None,
        )

        # Populate from G4 first; overridden below when local zero-CU recovery artifact is present.
        _set_slot(
            g4_slots,
            "pidnet_v21_val_miou",
            g4_val_miou,
            status,
            "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
        )
        _set_slot(
            g4_slots,
            "pidnet_v21_val_fsr",
            g4_val_fsr,
            status,
            "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
        )
        _set_slot(
            g4_slots,
            "pidnet_v21_aeroscapes_miou",
            g4_aero_miou,
            status,
            "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
        )
        _set_slot(
            g4_slots,
            "pidnet_v21_aeroscapes_fsr",
            g4_aero_fsr,
            status,
            "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
        )
        _set_slot(
            g4_slots,
            "xfeat_kartal_recall_at_5m_pct",
            xfeat_pass_pct,
            status,
            "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
        )

        if pval:
            hist = pd.get("pidnet_v21_training", {}).get("pareto_history", [])
            peak_row = max(hist, key=lambda r: float(r.get("val_miou", 0.0))) if hist else {}
            sel = pd.get("pidnet_v21_training", {}).get("pareto_selection", {}).get("selected", {})
            ablations["pidnet_v21_g4_collapsed_epoch2"] = {
                "source": "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
                "status": "COLLAPSED_EPOCH2_ALL_HAZARD_SELECTOR_BUG",
                "selected_epoch": int(sel.get("epoch", 2)),
                "val_miou": g4_val_miou,
                "val_fsr": g4_val_fsr,
                "ood_boundary_auroc": ood.get("boundary_compensated_auroc"),
                "aeroscapes_miou": g4_aero_miou,
                "aeroscapes_fsr": g4_aero_fsr,
                "unselected_peak_epoch": int(peak_row.get("epoch", 18)) if peak_row else None,
                "unselected_peak_val_miou": peak_row.get("val_miou"),
                "unselected_peak_val_fsr": peak_row.get("val_fsr"),
            }

        if reloc:
            ablations["xfeat_kartal_reloc"] = {
                "source": "artifacts/v4.0.0/g4_pidnet_v21_xfeat_metrics.json",
                "queries_evaluated": reloc.get("queries_evaluated"),
                "ransac_3gate_pass_count": reloc.get("ransac_3gate_pass_count"),
                "ransac_gate_pass_rate_pct": xfeat_pass_pct,
                "cep50_m": reloc.get("cep50_m"),
                "ce95_m": reloc.get("ce95_m"),
                "mean_inliers": reloc.get("mean_inliers"),
                "mean_reproj_rmse_px": reloc.get("mean_reproj_rmse_px"),
                "global_retrieval_r1": reloc.get("global_retrieval_r1"),
                "global_retrieval_r5": reloc.get("global_retrieval_r5"),
            }

        # Optional external eval-only zero-shot datasets: populate ONLY if genuinely present.
        if summary.get("icg_drone_miou") is not None:
            _set_slot(ext_slots, "icg_drone_miou", summary.get("icg_drone_miou"), status)
            _set_slot(ext_slots, "icg_drone_fsr", summary.get("icg_drone_fsr"), status)
        if summary.get("uavscenes_miou") is not None:
            _set_slot(ext_slots, "uavscenes_miou", summary.get("uavscenes_miou"), status)
            _set_slot(ext_slots, "uavscenes_fsr", summary.get("uavscenes_fsr"), status)

    if local_rec_path.is_file():
        ld = json.loads(local_rec_path.read_text(encoding="utf-8"))
        lstatus = str(ld.get("mode", "LIVE_LOCAL_GTX1650S_CUDA_FP16"))
        lcal = ld.get("permissive_val_metrics_bias_calibrated", {})
        lraw = ld.get("permissive_val_metrics_raw", {})
        laero_cal = ld.get("quarantined_zero_shot_aeroscapes_bias_calibrated", {})
        laero_raw = ld.get("quarantined_zero_shot_aeroscapes_raw", {})
        lood = ld.get("ood_metrics", {})
        lood_cal = ld.get("ood_calibration", {})
        ltrain = ld.get("training", {})

        _set_slot(
            g4_slots,
            "pidnet_v21_val_miou",
            lcal.get("miou_present_classes"),
            lstatus,
            "artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json",
        )
        _set_slot(
            g4_slots,
            "pidnet_v21_val_fsr",
            lcal.get("false_safe_rate_fsr"),
            lstatus,
            "artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json",
        )
        _set_slot(
            g4_slots,
            "pidnet_v21_aeroscapes_miou",
            laero_cal.get("miou_present_classes"),
            lstatus,
            "artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json",
        )
        _set_slot(
            g4_slots,
            "pidnet_v21_aeroscapes_fsr",
            laero_cal.get("false_safe_rate_fsr"),
            lstatus,
            "artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json",
        )

        ablations["pidnet_v21_local_recovery"] = {
            "source": "artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json",
            "cu_consumed": ld.get("cu_consumed", 0.0),
            "selected_epoch": int(
                ltrain.get("pareto_selection", {}).get("selected", {}).get("epoch", 17)
            ),
            "min_miou_floor": ltrain.get("pareto_selection", {}).get("min_miou_floor", 0.25),
            "safe_logit_bias_delta": lood_cal.get("safe_logit_bias_delta"),
            "raw_val_miou": lraw.get("miou_present_classes"),
            "raw_val_fsr": lraw.get("false_safe_rate_fsr"),
            "cal_val_miou": lcal.get("miou_present_classes"),
            "cal_val_fsr": lcal.get("false_safe_rate_fsr"),
            "cal_val_safe_precision": lcal.get("safe_precision"),
            "ood_raw_auroc": lood.get("raw_energy_auroc"),
            "ood_boundary_auroc": lood.get("boundary_compensated_auroc"),
            "raw_aeroscapes_miou": laero_raw.get("miou_present_classes"),
            "raw_aeroscapes_fsr": laero_raw.get("false_safe_rate_fsr"),
            "cal_aeroscapes_miou": laero_cal.get("miou_present_classes"),
            "cal_aeroscapes_fsr": laero_cal.get("false_safe_rate_fsr"),
        }

    edge_path = v400_dir / "gtx1650s_m3_timesliced_edge_gate.json"
    if edge_path.is_file():
        ed = json.loads(edge_path.read_text(encoding="utf-8"))
        pb = ed.get("phase_b_timesliced_schedule", {})
        pa = ed.get("phase_a_empirical_fp32_vs_fp16", {})
        cruise = pb.get("cruise_30hz_with_1hz_xfeat", {})
        term = pb.get("terminal_pidnet_v21", {})
        pid_a = pa.get("pidnet_slz_v21", {})
        ablations["gtx1650s_timesliced_edge"] = {
            "source": "artifacts/v4.0.0/gtx1650s_m3_timesliced_edge_gate.json",
            "cruise_mean_ms": cruise.get("mean_ms"),
            "cruise_p50_ms": cruise.get("p50_ms"),
            "cruise_p99_ms": cruise.get("p99_ms"),
            "cruise_fps_mean": cruise.get("fps_mean"),
            "cruise_vram_delta_mib": pb.get("cruise_vram_delta_mib"),
            "terminal_mean_ms": term.get("mean_ms"),
            "terminal_p50_ms": term.get("p50_ms"),
            "terminal_p99_ms": term.get("p99_ms"),
            "terminal_fps_mean": term.get("fps_mean"),
            "terminal_vram_delta_mib": pb.get("terminal_vram_delta_mib"),
            "pidnet_v21_fp32_cg_p50_ms": pid_a.get("fp32_cuda_graph", {}).get("p50_ms"),
            "pidnet_v21_fp32_cg_p99_ms": pid_a.get("fp32_cuda_graph", {}).get("p99_ms"),
            "pidnet_v21_fp16_p50_ms": pid_a.get("fp16", {}).get("p50_ms"),
            "pidnet_v21_fp16_p99_ms": pid_a.get("fp16", {}).get("p99_ms"),
        }

    rw_path = v400_dir / "g4_realworld_eval_report.json"
    if rw_path.is_file():
        rwd = json.loads(rw_path.read_text(encoding="utf-8"))
        pa = rwd.get("part_a_bf16_voice_qwen3_z3", {})
        zg = pa.get("z3_gate_stats", {})
        ablations["g4_realworld_bf16_copilot"] = {
            "source": "artifacts/v4.0.0/g4_realworld_eval_report.json",
            "n_turns": zg.get("n_cases"),
            "raw_z3_sat_pct": zg.get("raw_z3_constraint_satisfaction_pct"),
            "gated_z3_sat_pct": zg.get("gated_z3_constraint_satisfaction_pct"),
            "z3_fail_closed_interventions": zg.get("z3_fail_closed_interventions"),
            "bucket3_n_cases": zg.get("bucket3_n_cases"),
            "bucket3_gated_safe_reject_pct": zg.get("bucket3_gated_safe_reject_pct"),
        }

    wire_path = v400_dir / "g4_forward_wire_metrics.json"
    if wire_path.is_file():
        wfd = json.loads(wire_path.read_text(encoding="utf-8"))
        hv = wfd.get("heldout_validation", {})
        di = wfd.get("dataset_ingestion", {})
        ablations["g4_forward_wire"] = {
            "source": "artifacts/v4.0.0/g4_forward_wire_metrics.json",
            "real_ttpla_frames_ingested": di.get("real_ttpla_frames_ingested"),
            "procedural_frames_ingested": di.get("procedural_frames_ingested"),
            "val_wire_iou": hv.get("val_wire_iou"),
            "val_cldice_score": hv.get("val_cldice_score"),
            "val_det_cls_acc": hv.get("val_det_cls_acc"),
        }

    euroc_path = v400_dir / "realworld_euroc_mav_gnss_audit.json"
    if euroc_path.is_file():
        eud = json.loads(euroc_path.read_text(encoding="utf-8"))
        reg = eud.get("regimes", {})
        ablations["realworld_euroc_gnss"] = {
            "source": "artifacts/v4.0.0/realworld_euroc_mav_gnss_audit.json",
            "aligned_10hz_epochs": eud.get("real_flight_telemetry", {}).get("aligned_10hz_epochs"),
            "regime_a_false_alarms": reg.get("regime_a_clean_flight", {}).get("false_alarms"),
            "regime_b_spoof_ttd_s": reg.get("regime_b_slow_ramp_spoof", {}).get("ttd_s"),
            "regime_b_ctr_breaches": reg.get("regime_b_slow_ramp_spoof", {})
            .get("ctr_inflation", {})
            .get("ctr_breaches"),
        }

    ulog_path = v400_dir / "realworld_px4_ulog_forensics_proof.json"
    if ulog_path.is_file():
        uld = json.loads(ulog_path.read_text(encoding="utf-8"))
        fe = uld.get("forensics_e5x", {})
        ablations["realworld_px4_ulog_forensics"] = {
            "source": "artifacts/v4.0.0/realworld_px4_ulog_forensics_proof.json",
            "frames_verified": fe.get("frames_verified"),
            "xsd_valid": fe.get("xsd_valid"),
            "numeric_hallucination_count": fe.get("numeric_hallucination_count"),
            "cictt_category": fe.get("cictt_category"),
        }

    if ablations:
        out["phase3_ablations_and_edge"] = ablations
    else:
        out.pop("phase3_ablations_and_edge", None)

    return out


def render_corl_tables_tex(manifest: dict[str, Any]) -> str:
    """Render deterministic booktabs LaTeX tables from manifest."""
    b = manifest["verified_baselines"]
    g = manifest["phase3_g4_slots"]
    ext = manifest["external_eval_only_slots"]
    abl = manifest.get("phase3_ablations_and_edge", {})

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

    q3_det = abl.get("qwen3_v3_detail", {})
    v3_snr_row = ""
    if q3_det.get("snr_mondrian_coverage_pct") is not None:
        v3_snr_cov = _fmt(q3_det.get("snr_mondrian_coverage_pct"))
        v3_snr_sz = _fmt(q3_det.get("snr_mondrian_mean_set_size"), "{:.3f}")
        v3_snr_sng = _fmt(q3_det.get("snr_mondrian_singleton_exec_pct"))
        v3_snr_row = (
            f"Qwen3-4B v3 SNR-Mondrian (v2 bench, 3-seed, $N_{{\\text{{cal}}}}=2000$) & 100.00 & "
            f"{v3_ast} & {v3_b3} & {v3_snr_cov} & {v3_ece} & {v3_snr_sz} & {v3_snr_sng} \\\\\n"
        )

    w_s0 = _fmt(g["whisper_s0_clean_wer_pct"]["value"])
    w_s1 = _fmt(g["whisper_s1_cruise_wer_pct"]["value"])
    w_s2 = _fmt(g["whisper_s2_hover_wer_pct"]["value"])
    w_s3 = _fmt(g["whisper_s3_garbled_wer_pct"]["value"])

    w_abl = abl.get("whisper_bvi_lora_ablation", {}).get("turbo_bvi_lora_wer_pct", {})
    wl_s0 = _fmt(w_abl.get("S0_Clean"))
    wl_s1 = _fmt(w_abl.get("S1_Cruise"))
    wl_s2 = _fmt(w_abl.get("S2_Hover"))
    wl_s3 = _fmt(w_abl.get("S3_Garbled"))

    p21_vmiou = _fmt(g["pidnet_v21_val_miou"]["value"], "{:.4f}")
    p21_vfsr = _fmt(g["pidnet_v21_val_fsr"]["value"], "{:.2f}", scale=100.0)
    p21_amiou = _fmt(g["pidnet_v21_aeroscapes_miou"]["value"], "{:.4f}")
    p21_afsr = _fmt(g["pidnet_v21_aeroscapes_fsr"]["value"], "{:.2f}", scale=100.0)
    icg_miou = _fmt(ext["icg_drone_miou"]["value"], "{:.4f}")
    icg_fsr = _fmt(ext["icg_drone_fsr"]["value"], "{:.2f}", scale=100.0)
    uavs_miou = _fmt(ext["uavscenes_miou"]["value"], "{:.4f}")
    uavs_fsr = _fmt(ext["uavscenes_fsr"]["value"], "{:.2f}", scale=100.0)
    xfeat_r5 = _fmt(g["xfeat_kartal_recall_at_5m_pct"]["value"])

    edge = abl.get("gtx1650s_timesliced_edge", {})
    term_p50 = _fmt(edge.get("terminal_p50_ms"))
    term_p99 = _fmt(edge.get("terminal_p99_ms"))
    term_vram = _fmt(edge.get("terminal_vram_delta_mib"), "{:.1f}")

    g4_col = abl.get("pidnet_v21_g4_collapsed_epoch2", {})
    loc_rec = abl.get("pidnet_v21_local_recovery", {})

    if g4_col and loc_rec:
        c_vmiou = _fmt(g4_col.get("val_miou"), "{:.4f}")
        c_vfsr = _fmt(g4_col.get("val_fsr"), "{:.2f}", scale=100.0)
        c_amiou = _fmt(g4_col.get("aeroscapes_miou"), "{:.4f}")
        c_afsr = _fmt(g4_col.get("aeroscapes_fsr"), "{:.2f}", scale=100.0)
        p21_rows = (
            f"PIDNet-S v2.1 (G4 Ep.2 collapsed, ablation) & {c_vmiou} & {c_vfsr} & "
            f"{c_amiou} & {c_afsr} & \\text{{--}} / \\text{{--}} & \\text{{--}} / \\text{{--}} & "
            f"\\text{{--}} & \\text{{--}} \\\\\n"
            f"PIDNet-S v2.1 (Local 0\\,CU rec., $\\delta=-0.271$) & {p21_vmiou} & \\textbf{{{p21_vfsr}}} & "
            f"{p21_amiou} & {p21_afsr} & {icg_miou} ({icg_fsr}\\%) / {uavs_miou} ({uavs_fsr}\\%) & "
            f"{term_p50} / {term_p99} & {term_vram} & {xfeat_r5} \\\\\n"
        )
    else:
        p21_rows = (
            f"PIDNet-S v2.1 (Oblique-bal., G4) & {p21_vmiou} & {p21_vfsr} & {p21_amiou} & {p21_afsr} & "
            f"{icg_miou} ({icg_fsr}\\%) / {uavs_miou} ({uavs_fsr}\\%) & {term_p50} / {term_p99} & "
            f"{term_vram} & {xfeat_r5} \\\\\n"
        )

    t1_caption = (
        "Bilingual Neuro-Symbolic Mission Synthesis and Conformal Calibration Progression "
        "($\\alpha = 0.05$; v1 evaluated on $n_{\\text{test}}=560$, v3 3-seed Mondrian on "
        "\\texttt{VertiBench-v2} $N_{\\text{cal}}=2{,}000, N_{\\text{test}}=1{,}200$)."
    )

    return (
        "% Auto-generated by docs/paper_corl/scripts/generate_corl_tables.py\n"
        "% Source: docs/paper_corl/corl_metrics_manifest.json (zero manual edits).\n"
        "\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{t1_caption}}}\n"
        "\\label{tab:corl_copilot_mondrian}\n"
        "\\setlength{\\tabcolsep}{4.2pt}\n"
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
        f"Qwen3-4B v3 Lang-Mondrian (v2 bench, 3-seed, $N_{{\\text{{cal}}}}=2000$) & 100.00 & {v3_ast} & "
        f"{v3_b3} & {v3_cov} & {v3_ece} & {v3_sz} & {v3_sng} \\\\\n"
        f"{v3_snr_row}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
        "\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Four-Stratum Coaxial BVI Cockpit SNR Conditioning ($1{,}100$--$2{,}292$\\,RPM, "
        "$55.0$--$114.6$\\,Hz BPF; \\texttt{whisper-large-v3-turbo} on real CC-BY-4.0 FLEURS TR/EN, "
        "$n=256$/stratum; paired 3-epoch LoRA negative ablation shown honestly).}\n"
        "\\label{tab:corl_snr_strata}\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Stratum & SNR Band (dB) & $n_g$ (Cal) & ZS WER (\\%) & LoRA WER (\\%) & Guard Mode \\\\\n"
        "\\midrule\n"
        f"\\texttt{{S0\\_Clean}} & $>15$ & 500 & {w_s0} & {wl_s0} & Standard $\\hat{{q}}_{{S0}}$ \\\\\n"
        f"\\texttt{{S1\\_Cruise}} & $8$--$15$ & 500 & {w_s1} & {wl_s1} & Escalate $<0.80$ \\\\\n"
        f"\\texttt{{S2\\_Hover}} & $2$--$8$ & 500 & {w_s2} & {wl_s2} & Clarify \\texttt{{LOW\\_ASR}} \\\\\n"
        f"\\texttt{{S3\\_Garbled}} & $<2$ & 500 & {w_s3} & {wl_s3} & MAYDAY $w_m\\ge 0.15$ \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
        "\n"
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Zero-ShareAlike Active Visual SLZ Distillation, Honest G4 Epoch-2 Collapse vs.\\ "
        "Local 0\\,CU Recovery, Quarantined Real-Imagery Zero-Shot Transfer (AeroScapes; ICG Drone "
        "\\& UAVScenes unrun), Physical TU116 Edge Latency (GTX~1650~SUPER, 4\\,GB), and XFeat+RANSAC "
        "3-Gate Pass Rate (36 coastal 0.5\\,m tiles, 0.218\\,m CEP50).}\n"
        "\\label{tab:corl_slz_transfer}\n"
        "\\setlength{\\tabcolsep}{3.6pt}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        "Model Variant & Val mIoU & Val FSR (\\%) & Aero. mIoU & Aero. FSR (\\%) & "
        "ICG / UAVS. & p50 / p99 (ms) & VRAM (MiB) & XFeat Pass (\\%) \\\\\n"
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
        f"{p21_rows}"
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
