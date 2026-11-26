#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Zero-drift, double-anonymous & anti-fabrication verifier for Paper 2 (CoRL 2027)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from docs.paper_corl.scripts.generate_corl_tables import render_corl_tables_tex

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORL_DIR = REPO_ROOT / "docs" / "paper_corl"

FORBIDDEN_TOKENS = (
    "github.com/",
    "mailto:",
    "TODO",
    "[TBD]",
    "TBD",
    "UAVid",
    "uavid",
    "piper1-gpl",
    "UWB-ATCC",
    "ATCO2",
    "ESC-50",
    "DEMAND",
    "fahrifurkansagir",
)


def _check_close(
    errors: list[str], label: str, manifest_val: float, source_val: float, tol: float = 1e-4
) -> None:
    if abs(float(manifest_val) - float(source_val)) > tol:
        errors.append(f"Drift on {label}: manifest={manifest_val} vs source={source_val}")


def _verify_v400_slots_against_sources(errors: list[str], manifest: dict[str, Any]) -> None:
    """Cross-check populated Phase-3 v4.0.0 slots against their live source JSONs."""
    g = manifest["phase3_g4_slots"]

    qwen_p = REPO_ROOT / "artifacts" / "v4.0.0" / "g4_qwen3_v3_vertibench_v2_eval.json"
    if g["qwen3_v3_ast_stl_mean_pct"]["value"] is not None and qwen_p.is_file():
        qd = json.loads(qwen_p.read_text(encoding="utf-8"))
        ms = qd["multi_seed_summary"]
        ov = qd["test_overall_metrics"]
        _check_close(
            errors,
            "qwen3_v3_ast_stl_mean_pct",
            g["qwen3_v3_ast_stl_mean_pct"]["value"],
            ms["ast_stl_correctness_mean_pct"],
        )
        _check_close(
            errors,
            "qwen3_v3_ast_stl_sd_pct",
            g["qwen3_v3_ast_stl_sd_pct"]["value"],
            ms["ast_stl_correctness_sd_pct"],
        )
        _check_close(
            errors,
            "qwen3_v3_b3_reject_pct",
            g["qwen3_v3_b3_reject_pct"]["value"],
            ms["bucket3_safe_reject_mean_pct"],
        )
        _check_close(
            errors,
            "qwen3_v3_mondrian_coverage_pct",
            g["qwen3_v3_mondrian_coverage_pct"]["value"],
            ov["lang_mondrian_coverage_pct"],
        )
        _check_close(
            errors,
            "qwen3_v3_mondrian_ece",
            g["qwen3_v3_mondrian_ece"]["value"],
            ov["ece"],
        )
        _check_close(
            errors,
            "qwen3_v3_mondrian_mean_set_size",
            g["qwen3_v3_mondrian_mean_set_size"]["value"],
            ov["lang_mondrian_mean_set_size"],
        )
        _check_close(
            errors,
            "qwen3_v3_mondrian_singleton_exec_pct",
            g["qwen3_v3_mondrian_singleton_exec_pct"]["value"],
            ov["lang_mondrian_singleton_exec_pct"],
        )

    whisper_p = REPO_ROOT / "artifacts" / "v4.0.0" / "g4_whisper_bvi_lora_eval.json"
    if g["whisper_s0_clean_wer_pct"]["value"] is not None and whisper_p.is_file():
        wd = json.loads(whisper_p.read_text(encoding="utf-8"))
        zs = wd["models_trained"]["whisper_large_v3_turbo_bvi_lora"]["zero_shot_eval"][
            "by_snr_stratum"
        ]
        for slot_k, str_k in (
            ("whisper_s0_clean_wer_pct", "S0_Clean"),
            ("whisper_s1_cruise_wer_pct", "S1_Cruise"),
            ("whisper_s2_hover_wer_pct", "S2_Hover"),
            ("whisper_s3_garbled_wer_pct", "S3_Garbled"),
        ):
            _check_close(errors, slot_k, g[slot_k]["value"], float(zs[str_k]["wer"]) * 100.0)

    local_rec_p = REPO_ROOT / "artifacts" / "v4.0.0" / "local_gtx1650s_pidnet_v21_recovery_metrics.json"
    if g["pidnet_v21_val_miou"]["value"] is not None and local_rec_p.is_file():
        ld = json.loads(local_rec_p.read_text(encoding="utf-8"))
        _check_close(
            errors,
            "pidnet_v21_val_miou",
            g["pidnet_v21_val_miou"]["value"],
            ld["permissive_val_metrics_bias_calibrated"]["miou_present_classes"],
        )
        _check_close(
            errors,
            "pidnet_v21_val_fsr",
            g["pidnet_v21_val_fsr"]["value"],
            ld["permissive_val_metrics_bias_calibrated"]["false_safe_rate_fsr"],
        )
        _check_close(
            errors,
            "pidnet_v21_aeroscapes_miou",
            g["pidnet_v21_aeroscapes_miou"]["value"],
            ld["quarantined_zero_shot_aeroscapes_bias_calibrated"]["miou_present_classes"],
        )
        _check_close(
            errors,
            "pidnet_v21_aeroscapes_fsr",
            g["pidnet_v21_aeroscapes_fsr"]["value"],
            ld["quarantined_zero_shot_aeroscapes_bias_calibrated"]["false_safe_rate_fsr"],
        )

    xfeat_p = REPO_ROOT / "artifacts" / "v4.0.0" / "g4_pidnet_v21_xfeat_metrics.json"
    if g["xfeat_kartal_recall_at_5m_pct"]["value"] is not None and xfeat_p.is_file():
        xd = json.loads(xfeat_p.read_text(encoding="utf-8"))
        _check_close(
            errors,
            "xfeat_kartal_recall_at_5m_pct",
            g["xfeat_kartal_recall_at_5m_pct"]["value"],
            float(xd["kartal_reloc_eval"]["ransac_gate_pass_rate"]) * 100.0,
        )

    ext = manifest.get("external_eval_only_slots", {})
    uavs_p = REPO_ROOT / "artifacts" / "v4.0.0" / "realworld_uavscenes_zeroshot_eval.json"
    if ext.get("uavscenes_miou", {}).get("value") is not None:
        if not uavs_p.is_file():
            errors.append("uavscenes_miou populated in manifest but realworld_uavscenes_zeroshot_eval.json missing")
        else:
            ud = json.loads(uavs_p.read_text(encoding="utf-8"))
            ucal = ud["quarantined_zero_shot_uavscenes_bias_calibrated"]
            uq = ud["quarantine_policy"]
            if uq.get("used_in_training") is not False or uq.get("used_in_calibration") is not False:
                errors.append("UAVScenes quarantine violation: used_in_training/used_in_calibration must be false")
            if uq.get("raw_pixels_deleted") is not True:
                errors.append("UAVScenes quarantine violation: raw_pixels_deleted must be true")
            _check_close(
                errors,
                "uavscenes_miou",
                ext["uavscenes_miou"]["value"],
                ucal["miou_present_classes"],
            )
            _check_close(
                errors,
                "uavscenes_fsr",
                ext["uavscenes_fsr"]["value"],
                ucal["false_safe_rate_fsr"],
            )

    abl = manifest.get("phase3_ablations_and_edge", {})
    loc_q_rec = abl.get("qwen3_v3_local_conformal_recovery")
    loc_q_p = REPO_ROOT / "artifacts" / "v4.0.0" / "local_gtx1650s_qwen3_v3_conformal_raps.json"
    if loc_q_rec is not None:
        if not loc_q_p.is_file():
            errors.append(
                "qwen3_v3_local_conformal_recovery populated in manifest but local_gtx1650s_qwen3_v3_conformal_raps.json missing"
            )
        else:
            lq = json.loads(loc_q_p.read_text(encoding="utf-8"))
            if lq.get("zero_test_leakage_verified") is not True or lq.get("quantization_loss_zero") is not True:
                errors.append("Local Qwen3 conformal recovery must preserve zero test leakage and zero quantization loss")
            lq_sum = lq["summary"]
            lq_lac = lq["paired_lac_on_fixed_scores"]
            _check_close(errors, "local_qwen3.lang_raps_cov", loc_q_rec["lang_raps_coverage_pct"], lq_sum["lang_mondrian_raps_coverage_pct"])
            _check_close(errors, "local_qwen3.lang_raps_ece", loc_q_rec["lang_raps_ece"], lq_sum["lang_mondrian_raps_ece"])
            _check_close(errors, "local_qwen3.lang_raps_sz", loc_q_rec["lang_raps_mean_set_size"], lq_sum["lang_mondrian_raps_mean_set_size"])
            _check_close(errors, "local_qwen3.snr_raps_sz", loc_q_rec["snr_raps_mean_set_size"], lq_sum["snr_mondrian_raps_mean_set_size"])
            _check_close(errors, "local_qwen3.lang_lac_sz", loc_q_rec["lang_lac_mean_set_size"], lq_lac["lang_mondrian_lac_test"]["overall_mean_set_size"])
            _check_close(errors, "local_qwen3.snr_lac_sz", loc_q_rec["snr_lac_mean_set_size"], lq_lac["snr_mondrian_lac_test"]["overall_mean_set_size"])

    edge_rec = abl.get("gtx1650s_timesliced_edge")
    edge_p = REPO_ROOT / "artifacts" / "v4.0.0" / "gtx1650s_m3_timesliced_edge_gate.json"
    if edge_rec is not None and edge_p.is_file():
        ed = json.loads(edge_p.read_text(encoding="utf-8"))
        pb = ed["phase_b_timesliced_schedule"]
        _check_close(errors, "edge.cruise_p99_ms", edge_rec["cruise_p99_ms"], pb["cruise_30hz_with_1hz_xfeat"]["p99_ms"])
        _check_close(errors, "edge.terminal_p99_ms", edge_rec["terminal_p99_ms"], pb["terminal_pidnet_v21"]["p99_ms"])
        _check_close(errors, "edge.cruise_vram", edge_rec["cruise_vram_delta_mib"], pb["cruise_vram_delta_mib"])
        _check_close(errors, "edge.terminal_vram", edge_rec["terminal_vram_delta_mib"], pb["terminal_vram_delta_mib"])


def verify_corl_manuscript(
    corl_dir: Path = DEFAULT_CORL_DIR,
    *,
    allow_pending_g4: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    manifest_path = corl_dir / "corl_metrics_manifest.json"
    tex_path = corl_dir / "vertirta_corl_manuscript.tex"
    bib_path = corl_dir / "references_corl.bib"
    gen_tex_path = corl_dir / "generated" / "corl_tables.tex"

    for p in (manifest_path, tex_path, bib_path, gen_tex_path):
        if not p.is_file():
            errors.append(f"Missing required Paper 2 artifact: {p}")
    if errors:
        return {"passed": False, "errors": errors, "pending_g4_slot_count": -1}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    gen_tex = gen_tex_path.read_text(encoding="utf-8")

    # 1) Cross-check EVERY verified v3.1.0 & v3.5.1 baseline value directly against source JSONs.
    b = manifest["verified_baselines"]

    v31_q_path = REPO_ROOT / b["v310_qwen3_4b_sft_grpo"]["source"]
    v31_q = json.loads(v31_q_path.read_text(encoding="utf-8"))
    _check_close(
        errors,
        "v310_qwen25_7b_zs.grammar_validity_pct",
        b["v310_qwen25_7b_zs"]["grammar_validity_pct"],
        v31_q["evaluations"]["qwen25_7b_zero_shot"]["overall"]["grammar_validity_pct"],
    )
    _check_close(
        errors,
        "v310_qwen3_4b_zs.ast_stl_accuracy_pct",
        b["v310_qwen3_4b_zs"]["ast_stl_accuracy_pct"],
        v31_q["evaluations"]["qwen3_4b_zero_shot"]["overall"]["ast_stl_accuracy_pct"],
    )
    _check_close(
        errors,
        "v310_qwen3_4b_sft_grpo.ast_stl_accuracy_pct",
        b["v310_qwen3_4b_sft_grpo"]["ast_stl_accuracy_pct"],
        v31_q["evaluations"]["qwen3_4b_sft_z3_grpo"]["overall"]["ast_stl_accuracy_pct"],
    )
    _check_close(
        errors,
        "v310_qwen3_4b_sft_grpo.mean_prediction_set_size",
        b["v310_qwen3_4b_sft_grpo"]["mean_prediction_set_size"],
        v31_q["conformal_calibration"]["mean_prediction_set_size"],
    )

    v351_q_path = REPO_ROOT / b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]["source"]
    v351_q = json.loads(v351_q_path.read_text(encoding="utf-8"))
    _check_close(
        errors,
        "v351_qwen3_4b_v2.ast_stl_accuracy_pct",
        b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]["ast_stl_accuracy_pct"],
        v351_q["gbnf_z3_fail_closed_comparison"]["qwen3_4b_sft_z3_grpo"]["gbnf_plus_z3_fail_closed"][
            "ast_stl_accuracy_pct"
        ],
    )
    _check_close(
        errors,
        "v351_qwen3_4b_v2.ece_calibrated",
        b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]["ece_calibrated"],
        v351_q["conformal_calibration"]["temperature_calibrated_test"]["ece"],
    )
    _check_close(
        errors,
        "v351_qwen3_4b_v2.mean_prediction_set_size",
        b["v351_qwen3_4b_v2_gbnf_z3_calibrated"]["mean_prediction_set_size"],
        v351_q["conformal_calibration"]["temperature_calibrated_test"]["mean_prediction_set_size"],
    )

    v31_p = json.loads((REPO_ROOT / b["v310_pidnet_v1"]["source"]).read_text(encoding="utf-8"))
    v31_e = json.loads((REPO_ROOT / b["v310_pidnet_v1"]["edge_source"]).read_text(encoding="utf-8"))
    _check_close(
        errors,
        "v310_pidnet_v1.val_miou",
        b["v310_pidnet_v1"]["val_miou"],
        v31_p["evaluation"]["permissive_validation"]["miou_present_classes"],
    )
    _check_close(
        errors,
        "v310_pidnet_v1.aeroscapes_fsr",
        b["v310_pidnet_v1"]["aeroscapes_fsr"],
        v31_p["evaluation"]["quarantined_zero_shot_aeroscapes"]["false_safe_rate_fsr"],
    )
    _check_close(
        errors,
        "v310_pidnet_v1.gtx1650s_p99",
        b["v310_pidnet_v1"]["gtx1650s_fp32_cg_p99_ms"],
        v31_e["benchmarks"]["ort_cuda_fp32_cuda_graph"]["network_iobinding"]["p99_ms"],
    )

    v351_p = json.loads((REPO_ROOT / b["v351_pidnet_v2"]["source"]).read_text(encoding="utf-8"))
    v351_e = json.loads((REPO_ROOT / b["v351_pidnet_v2"]["edge_source"]).read_text(encoding="utf-8"))
    _check_close(
        errors,
        "v351_pidnet_v2.val_miou",
        b["v351_pidnet_v2"]["val_miou"],
        v351_p["evaluation"]["permissive_validation"]["miou_present_classes"],
    )
    _check_close(
        errors,
        "v351_pidnet_v2.val_fsr",
        b["v351_pidnet_v2"]["val_fsr"],
        v351_p["evaluation"]["permissive_validation"]["false_safe_rate_fsr"],
    )
    _check_close(
        errors,
        "v351_pidnet_v2.aeroscapes_fsr",
        b["v351_pidnet_v2"]["aeroscapes_fsr"],
        v351_p["evaluation"]["quarantined_zero_shot_aeroscapes"]["false_safe_rate_fsr"],
    )
    _check_close(
        errors,
        "v351_pidnet_v2.gtx1650s_p99",
        b["v351_pidnet_v2"]["gtx1650s_fp32_cg_p99_ms"],
        v351_e["benchmarks"]["ort_cuda_fp32_cuda_graph"]["network_iobinding"]["p99_ms"],
    )

    # 1b) Cross-check populated Phase-3 v4.0.0 slots against live v4.0.0 source JSONs
    _verify_v400_slots_against_sources(errors, manifest)

    # 2) Verify generated/corl_tables.tex matches manifest byte-for-byte
    expected_tex = render_corl_tables_tex(manifest)
    if gen_tex != expected_tex:
        errors.append("generated/corl_tables.tex drifted from corl_metrics_manifest.json")

    # 3) Check pending Phase-3 G4 slots
    pending_g4 = [
        k
        for k, slot in manifest["phase3_g4_slots"].items()
        if slot.get("status") == "PENDING_PHASE3_G4_THUNDER_RUN" or slot.get("value") is None
    ]
    if pending_g4 and not allow_pending_g4:
        errors.append(
            f"{len(pending_g4)} Phase-3 G4 slots remain PENDING_PHASE3_G4_THUNDER_RUN "
            "(run with --allow-pending-g4 in Phase 2)"
        )

    # 4) Double-anonymity, zero banned tokens, zero certification claims, honest provenance
    for label, text in (
        ("vertirta_corl_manuscript.tex", tex),
        ("references_corl.bib", bib),
        ("generated/corl_tables.tex", gen_tex),
        ("corl_metrics_manifest.json", manifest_path.read_text(encoding="utf-8")),
    ):
        for tok in FORBIDDEN_TOKENS:
            if tok in text:
                errors.append(f"Forbidden token '{tok}' found in {label}")

    if re.search(r"we (achieve|demonstrate|are) (DO-178C|SC-VTOL|SORA).*certified", tex, re.IGNORECASE):
        errors.append("Forbidden airworthiness certification claim in manuscript")
    if re.search(r"human[- ]native (Turkish|English|bilingual) benchmark", tex, re.IGNORECASE):
        errors.append("Premature human-native benchmark claim in manuscript")
    if manifest["provenance"]["vertibench_v2_tag"] != (
        "PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2"
    ):
        errors.append("Unexpected VertiBench-v2 provenance tag in manifest")

    # 5) BibTeX closure and balanced \begin{} / \end{} environments
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cited = {
        k.strip()
        for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex)
        for k in grp.split(",")
        if k.strip()
    }
    missing_bib = sorted(cited - bib_keys)
    if missing_bib:
        errors.append(f"Unresolved BibTeX citations: {missing_bib}")

    begins = re.findall(r"\\begin\{([^}]+)\}", tex)
    ends = re.findall(r"\\end\{([^}]+)\}", tex)
    if len(begins) != len(ends):
        errors.append(f"Unbalanced LaTeX environments: {len(begins)} begins vs {len(ends)} ends")

    # 6) Vector PDF figure presence & leak audit
    for fig_name in (
        "fig1_bilingual_mondrian_pareto.pdf",
        "fig2_snr_stratum_coverage.pdf",
        "fig3_slz_real_transfer_pareto.pdf",
    ):
        fp = corl_dir / "figures" / fig_name
        if not fp.is_file():
            errors.append(f"Missing PDF figure: {fp}")
        else:
            raw = fp.read_bytes()
            if not raw.startswith(b"%PDF-"):
                errors.append(f"Invalid PDF header in {fig_name}")
            for leak in (b"github.com", b"fahrifurkansagir", b"UAVid", b"uavid"):
                if leak in raw:
                    errors.append(f"Leak {leak!r} inside {fig_name}")

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "pending_g4_slot_count": len(pending_g4),
        "external_eval_pending_count": sum(
            1
            for s in manifest["external_eval_only_slots"].values()
            if s.get("status") == "PENDING_EVAL_ONLY_REAL_IMAGERY"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corl-dir", type=Path, default=DEFAULT_CORL_DIR)
    parser.add_argument("--allow-pending-g4", action="store_true")
    args = parser.parse_args()

    res = verify_corl_manuscript(args.corl_dir, allow_pending_g4=args.allow_pending_g4)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
