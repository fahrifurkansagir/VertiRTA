"""TDD tests for Plan 4B Task 6: standards_registry v2.0, DCO 1.1 governance, and IEEE RA-L Paper 1."""

from __future__ import annotations

import json
from pathlib import Path
import re
import yaml


REPO = Path(__file__).resolve().parents[3]


def test_standards_registry_v2_adds_all_seven_2025_2026_instruments() -> None:
    reg = yaml.safe_load((REPO / "docs/standards_registry.yaml").read_text(encoding="utf-8"))
    assert reg["registry_version"] == "vertirta.standards/v2.0.0"
    assert reg["compliance_claim"] == "NONE_RESEARCH_TESTBED_ONLY"
    ids = {s["id"] for s in reg["standards"]}
    required_new = {
        "REG-EU-2022-1645-PART-IS",
        "REG-EU-2023-203-PART-IS",
        "CIR-EU-2026-1821",
        "CDR-EU-2020-2034-ERCS",
        "EUROCAE-ED-155A",
        "SHGM-SHY-OLAY",
        "SHGM-SHT-21-EK6",
    }
    assert required_new.issubset(ids)
    # Preserve all 15 v1.0.0 standards without deletion
    assert len(reg["standards"]) >= 22
    by_id = {s["id"]: s for s in reg["standards"]}
    assert "NIS2" in by_id["REG-EU-2023-203-PART-IS"]["role_in_vertirta"]


def test_contributing_locks_dco11_permanent_vrta_and_dr08_13_bans() -> None:
    txt = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for token in (
        "DCO 1.1",
        "No Corporate CLA",
        "Permanent Apache-2.0",
        "VRTA0001",
        "OpenVINS",
        "VINS-Fusion",
        "BlenderProc",
        "piper1-gpl",
        "UAVid",
        "libsodium >= 1.0.21",
        "Fast DDS >= 2.14.6",
        "sherpa-onnx >= 2.0.0",
        "Basalt",
        "XFeat",
    ):
        assert token in txt, f"Missing governance/firewall lock in CONTRIBUTING.md: {token}"


def test_paper_ral_is_double_anonymous_honest_and_computes_mcnemar_and_iqm() -> None:
    from docs.paper.scripts.verify_paper_numbers import verify_paper_honesty_and_numbers
    from docs.paper_ral.scripts.eval_ral_stats import (
        compute_iqm_bootstrap,
        compute_mcnemar_holm_bonferroni,
        evaluate_v1_atlas,
    )
    from scripts.ci.check_public_hygiene import scan_public_hygiene

    tex_path = REPO / "docs/paper_ral/vertirta_ral_manuscript.tex"
    tex = tex_path.read_text(encoding="utf-8")

    # Strict double-anonymous check: zero repo URLs, personal paths/usernames, placeholders, or Pillar 2/3 leaks
    assert scan_public_hygiene(REPO, tracked_files=["docs/paper_ral/vertirta_ral_manuscript.tex"])["passed"] is True
    for leak in ("github.com/", "mailto:", "TODO", "[TBD]", "Qwen3", "PIDNet", "VertiBench"):
        assert leak not in tex, f"Double-anonymous or Paper-1 scope violation in vertirta_ral_manuscript.tex: {leak}"
    assert "Anonymous Authors" in tex
    assert "\\input{generated/ral_tables.tex}" in tex
    assert not re.search(r"we (achieve|demonstrate|are) (DO-178C|SC-VTOL|SORA).*certified", tex, re.IGNORECASE)

    # Verify BibTeX citations & balanced LaTeX environments via existing paper verifier (no manifest drift check)
    ral_dir = REPO / "docs/paper_ral"
    bib_keys = set(
        re.findall(
            r"@\w+\s*\{\s*([^,\s]+)\s*,",
            (ral_dir / "references_ral.bib").read_text(encoding="utf-8"),
        )
    )
    cited = {
        k.strip()
        for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex)
        for k in grp.split(",")
        if k.strip()
    }
    assert cited and cited.issubset(bib_keys), f"Missing BibTeX keys: {cited - bib_keys}"
    # Run honesty/placeholder/environment check on docs/paper_ral/ (non-existent manifest skips v1/v2/v3 tables check)
    assert verify_paper_honesty_and_numbers(ral_dir, ral_dir / "_no_manifest.json") == []

    # Exact paired McNemar + Holm-Bonferroni sanity check (per-seed mode)
    res = compute_mcnemar_holm_bonferroni([{"name": "A_vs_B", "b": 120, "c": 5}], alpha=0.05)
    assert res[0]["p_exact"] < 1e-10 and res[0]["significant_holm"] is True

    iqm = compute_iqm_bootstrap([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], n_resamples=1000, seed=42)
    assert iqm["ci_low"] <= iqm["iqm"] <= iqm["ci_high"]

    # Summary conservative-bound mode on aggregate tier1_10k_clopper_pearson.json (ZERO fake per-seed synthesis!)
    report = evaluate_v1_atlas(REPO / "artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json")
    assert report["csfl_n"] == 6500 and report["cel_n"] == 1500
    assert report["method"] == "frechet_hoeffding_conservative_bound"
    assert report["iqm_status"] == "PENDING_PER_RUN_EXPORT"
    assert report["holm_bonferroni_pairs"][0]["significant_holm"] is True

    # Generated artifacts on disk match live paired v3.1.0 computation
    gen_json = json.loads((ral_dir / "generated/ral_stats_summary.json").read_text(encoding="utf-8"))
    gen_tex = (ral_dir / "generated/ral_tables.tex").read_text(encoding="utf-8")
    assert gen_json["method"] == "paired_per_seed_exact"
    assert gen_json["iqm_status"] == "COMPUTED_FROM_PER_RUN_VECTORS"
    assert "95.98" in gen_tex and "44.85" in gen_tex and "100.00" in gen_tex and "28{,}886" in gen_tex
    assert "TBD" not in gen_tex and "TODO" not in gen_tex
