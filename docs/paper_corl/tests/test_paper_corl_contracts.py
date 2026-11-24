# SPDX-License-Identifier: Apache-2.0
"""TDD & anti-drift contract tests for Paper 2 (CoRL 2027 — Plan 6B Task 6)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from docs.paper_corl.scripts.generate_corl_tables import (
    ingest_v400_into_manifest,
    render_corl_tables_tex,
)
from docs.paper_corl.scripts.verify_corl_numbers import verify_corl_manuscript
from scripts.ci.check_public_hygiene import scan_public_hygiene

REPO = Path(__file__).resolve().parents[3]
CORL_DIR = REPO / "docs" / "paper_corl"


def test_paper_corl_double_anonymous_bans_uavid_and_enforces_honest_provenance() -> None:
    tex_path = CORL_DIR / "vertirta_corl_manuscript.tex"
    bib_path = CORL_DIR / "references_corl.bib"
    gen_tex_path = CORL_DIR / "generated" / "corl_tables.tex"
    manifest_path = CORL_DIR / "corl_metrics_manifest.json"

    tex = tex_path.read_text(encoding="utf-8")
    bib = bib_path.read_text(encoding="utf-8")
    gen_tex = gen_tex_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert (
        scan_public_hygiene(
            REPO,
            tracked_files=[
                "docs/paper_corl/vertirta_corl_manuscript.tex",
                "docs/paper_corl/references_corl.bib",
                "docs/paper_corl/generated/corl_tables.tex",
                "docs/paper_corl/corl_metrics_manifest.json",
            ],
        )["passed"]
        is True
    )
    assert "Anonymous Author" in tex
    assert "\\input{generated/corl_tables.tex}" in tex

    # Hard double-anonymity + UAVid ban + audio license ban + anti-placeholder firewall
    for forbidden in (
        "github.com/",
        "mailto:",
        "TODO",
        "[TBD]",
        "UAVid",
        "uavid",
        "piper1-gpl",
        "UWB-ATCC",
        "ATCO2",
        "fahrifurkansagir",
    ):
        assert forbidden not in tex, f"Forbidden leak or banned corpus in Paper 2 .tex: {forbidden}"
        assert forbidden not in bib, f"Forbidden leak or banned corpus in Paper 2 .bib: {forbidden}"
        assert forbidden not in gen_tex, f"Forbidden token in generated tables: {forbidden}"
        assert forbidden not in manifest_path.read_text(encoding="utf-8"), (
            f"Forbidden token in manifest: {forbidden}"
        )

    # Zero certification claims & honest VertiBench-v2 provenance framing
    assert not re.search(
        r"we (achieve|demonstrate|are) (DO-178C|SC-VTOL|SORA).*certified", tex, re.IGNORECASE
    )
    assert not re.search(
        r"human[- ]native (Turkish|English|bilingual) benchmark", tex, re.IGNORECASE
    )
    assert (
        manifest["provenance"]["vertibench_v2_tag"]
        == "PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2"
    )

    # All real-imagery SLZ datasets strictly from AeroScapes / ICG / UAVScenes (eval-only)
    for ds in ("AeroScapes", "ICG", "UAVScenes"):
        assert ds in tex

    # BibTeX closure check
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cited = {
        k.strip()
        for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex)
        for k in grp.split(",")
        if k.strip()
    }
    assert cited and cited.issubset(bib_keys), f"Unresolved BibTeX keys: {cited - bib_keys}"


def test_verify_corl_numbers_and_figures_pass_zero_errors_in_phase2() -> None:
    res = verify_corl_manuscript(CORL_DIR, allow_pending_g4=True)
    assert res["passed"] is True, f"Verification errors: {res['errors']}"
    assert res["errors"] == []

    # Strict mode (without --allow-pending-g4) MUST reject pending Phase-3 G4 slots
    strict_res = verify_corl_manuscript(CORL_DIR, allow_pending_g4=False)
    if res["pending_g4_slot_count"] > 0:
        assert strict_res["passed"] is False

    # Verify all 3 double-anonymous vector PDF figures exist, are wired in tex, and leak-free
    tex = (CORL_DIR / "vertirta_corl_manuscript.tex").read_text(encoding="utf-8")
    for fig_name in (
        "fig1_bilingual_mondrian_pareto.pdf",
        "fig2_snr_stratum_coverage.pdf",
        "fig3_slz_real_transfer_pareto.pdf",
    ):
        fig_path = CORL_DIR / "figures" / fig_name
        assert fig_path.is_file(), f"Missing Paper 2 PDF figure: {fig_path}"
        raw_pdf = fig_path.read_bytes()
        assert raw_pdf.startswith(b"%PDF-")
        assert 5_000 < len(raw_pdf) < 250_000
        assert f"figures/{fig_name}" in tex
        for leak in (b"github.com", b"fahrifurkansagir", b"UAVid", b"uavid"):
            assert leak not in raw_pdf, f"Leak {leak!r} in {fig_name}"


def test_phase3_v400_ingestion_populates_slots_without_fabricating_unrun_external_slz(
    tmp_path: Path,
) -> None:
    manifest = json.loads((CORL_DIR / "corl_metrics_manifest.json").read_text(encoding="utf-8"))
    v400_dir = tmp_path / "v4.0.0"
    v400_dir.mkdir()

    (v400_dir / "g4_qwen3_v3_vertibench_v2_eval.json").write_text(
        json.dumps(
            {
                "mode": "LIVE_G4_EXECUTED",
                "ast_stl_accuracy_mean_pct": 95.4,
                "ast_stl_accuracy_sd_pct": 0.6,
                "bucket3_reject_pct": 94.2,
                "mondrian_coverage_mean_pct": 95.8,
                "mondrian_ece": 0.062,
                "mondrian_mean_set_size": 1.48,
                "mondrian_singleton_exec_pct": 71.5,
            }
        ),
        encoding="utf-8",
    )
    (v400_dir / "g4_whisper_bvi_lora_eval.json").write_text(
        json.dumps(
            {
                "mode": "LIVE_G4_EXECUTED",
                "snr_strata_metrics": {
                    "S0_Clean": {"wer_pct": 4.2, "ser_pct": 1.1},
                    "S1_Cruise": {"wer_pct": 7.5, "ser_pct": 2.4},
                    "S2_Hover": {"wer_pct": 12.8, "ser_pct": 4.1},
                    "S3_Garbled": {"wer_pct": 21.6, "ser_pct": 7.9},
                },
            }
        ),
        encoding="utf-8",
    )
    (v400_dir / "g4_pidnet_v21_xfeat_metrics.json").write_text(
        json.dumps(
            {
                "mode": "LIVE_G4_EXECUTED",
                "val_miou": 0.512,
                "val_fsr": 0.028,
                "aeroscapes_miou": 0.341,
                "aeroscapes_fsr": 0.024,
                "xfeat_kartal_recall_at_5m_pct": 93.4,
            }
        ),
        encoding="utf-8",
    )

    updated = ingest_v400_into_manifest(manifest, v400_dir)
    for key, slot in updated["phase3_g4_slots"].items():
        assert slot["status"] == "LIVE_G4_EXECUTED", f"Slot {key} not populated"
        assert slot["value"] is not None
    # Unrun external datasets stay honestly PENDING_EVAL_ONLY_REAL_IMAGERY (zero fabrication!)
    assert updated["external_eval_only_slots"]["icg_drone_miou"]["status"] == (
        "PENDING_EVAL_ONLY_REAL_IMAGERY"
    )
    assert updated["external_eval_only_slots"]["uavscenes_miou"]["status"] == (
        "PENDING_EVAL_ONLY_REAL_IMAGERY"
    )

    tex_out = render_corl_tables_tex(updated)
    assert "95.4" in tex_out and "1.48" in tex_out and "0.341" in tex_out
    assert "TODO" not in tex_out and "[TBD]" not in tex_out
