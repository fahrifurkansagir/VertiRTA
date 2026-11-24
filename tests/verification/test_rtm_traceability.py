# SPDX-License-Identifier: Apache-2.0
"""TDD & anti-drift tests for E5: R01..R48 Machine-Auditable RTM & LaTeX/Figure Audit."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from docs.paper_ral.scripts.audit_latex_and_figures import audit_all_papers
from scripts.ci.verify_rtm_traceability import (
    audit_rtm_dict,
    verify_rtm_file,
)

REPO = Path(__file__).resolve().parents[2]
RTM_PATH = REPO / "docs" / "assurance" / "rtm_v4_traceability.yaml"


def test_rtm_v4_covers_all_48_requirements_with_zero_broken_paths_or_symbols() -> None:
    report = verify_rtm_file(RTM_PATH, REPO)
    assert report["passed"] is True, f"RTM violations: {report['violations']}"
    assert report["requirements_audited"] == 48
    assert report["violations"] == []
    assert report["paths_verified"] >= 120
    assert report["symbols_verified"] >= 60
    assert report["status_breakdown"]["TESTBED_VERIFIED"] >= 40
    assert report["status_breakdown"]["CERTIFICATION_UNDEVELOPED"] >= 2

    required_standards = {
        "ASTM-F3269-21",
        "SC-VTOL-02",
        "SC-VTOL-MOC-5",
        "REG-EU-2022-1645-PART-IS",
        "REG-EU-2023-203-PART-IS",
        "CIR-EU-2026-1821",
        "CDR-EU-2020-2034-ERCS",
        "EUROCAE-ED-155A",
        "SHGM-SHY-OLAY",
        "SHGM-SHT-21-EK6",
        "STM32H7-ES0392-REV16",
    }
    assert required_standards.issubset(set(report["covered_standards"]))


def test_rtm_verifier_fails_closed_on_broken_path_missing_symbol_or_fake_certified_claim() -> None:
    base = yaml.safe_load(RTM_PATH.read_text(encoding="utf-8"))

    # 1) Missing file path must fail closed
    bad_path = copy.deepcopy(base)
    bad_path["requirements"][0]["source_files"].append("core/flight_core/does_not_exist.hpp")
    res_path = audit_rtm_dict(bad_path, REPO)
    assert res_path["passed"] is False
    assert any("does_not_exist.hpp" in v for v in res_path["violations"])

    # 2) Nonexistent C++ symbol must fail closed
    bad_sym = copy.deepcopy(base)
    bad_sym["requirements"][0]["cpp_or_py_symbols"].append("FakePhantomController999")
    res_sym = audit_rtm_dict(bad_sym, REPO)
    assert res_sym["passed"] is False
    assert any("FakePhantomController999" in v for v in res_sym["violations"])

    # 3) Dishonest 'CERTIFIED' verification status must fail closed
    bad_status = copy.deepcopy(base)
    bad_status["requirements"][0]["verification_status"] = "CERTIFIED"
    res_status = audit_rtm_dict(bad_status, REPO)
    assert res_status["passed"] is False
    assert any("verification_status" in v for v in res_status["violations"])


def test_paper_ral_and_omnibus_latex_bibtex_and_figure_audit_pass_zero_errors() -> None:
    report = audit_all_papers(REPO)
    assert report["passed"] is True, f"Paper audit errors: {report}"
    for key in ("paper_ral", "paper_omnibus"):
        sub = report["papers"][key]
        assert sub["passed"] is True, f"{key} errors: {sub['errors']}"
        assert sub["undefined_citations"] == []
        assert sub["orphan_bib_entries"] == []
        assert sub["undefined_refs"] == []
        assert sub["duplicate_labels"] == []
        assert sub["missing_includes"] == []
        assert sub["missing_figures"] == []
        assert sub["anonymity_or_honesty_violations"] == []
    assert report["papers"]["paper_ral"]["cited_count"] == 18
    assert report["papers"]["paper_omnibus"]["cited_count"] == 23
