"""TDD verification tests for Plan 1B Task 3: Aviation Assurance Docs & CI-Linked GSN v3."""

from __future__ import annotations

from pathlib import Path
import copy
import pytest
import yaml

from verification.gsn.build_gsn import (
    build_gsn_mermaid,
    inject_gsn_into_safety_case,
    validate_odd_yaml,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_standards_registry_pins_2026_dates_and_draft_statuses() -> None:
    reg = yaml.safe_load((REPO_ROOT / "docs/standards_registry.yaml").read_text(encoding="utf-8"))
    by_id = {entry["id"]: entry for entry in reg["standards"]}
    assert by_id["SC-VTOL-02"]["date"] == "2024-06-10"
    assert by_id["SC-VTOL-MOC-5"]["date"] == "2026-11-05"
    assert by_id["SC-VTOL-MOC-5"]["status"] == "FINAL"
    assert by_id["ED-2026-010-R"]["date"] == "2026-11-16"
    assert by_id["ED-2026-010-R"]["sora_20_retirement_date"] == "2027-01-01"
    assert by_id["SHGM-SHT-IHA"]["date"] == "2026-07-30"
    assert by_id["ASTM-F3269-21"]["status"] == "ACTIVE"
    assert by_id["EASA-AI-CP-ISS03"]["status"] == "PROPOSED_DRAFT"
    assert by_id["ED-324-ARP6983"]["status"] == "DRAFT"
    assert by_id["SAE-ARP4761A"]["date"] == "2023-12-20"
    assert by_id["SCSC-141C-GSN-V3"]["status"] == "ACTIVE"


def test_odd_yaml_enforces_j2_west_boundary_and_locked_spec() -> None:
    odd = yaml.safe_load((REPO_ROOT / "verification/ODD.yaml").read_text(encoding="utf-8"))
    report = validate_odd_yaml(odd)
    assert report["passed"] is True
    assert odd["compliance_claim"] == "NONE_RESEARCH_TESTBED_ONLY"

    scenery = odd["operational_domain"]["scenery_elements"]
    assert scenery["corridor_bounding_box"]["lon_deg"][1] < 29.1900
    assert scenery["egm2008_undulation_m"] == 36.7
    assert scenery["datum_origin"]["lat_deg"] == 40.8860
    assert scenery["datum_origin"]["lon_deg"] == 29.1880
    assert scenery["datum_origin"]["alt_amsl_m"] == 12.0

    slz = odd["operational_domain"]["slz_requirements"]
    assert (slz["r_slz_min_m"], slz["slope_max_deg"], slz["roughness_sigma_z_max_m"]) == (
        5.0,
        8.0,
        0.15,
    )
    assert odd["operational_domain"]["operating_parameters"]["map_staleness_inflate_ms"] == 250
    assert odd["operational_domain"]["regulatory_stage_split"]["stage_a_subscale_unmanned"]["passengers"] == 0
    assert all(c["actuation_authority"] == "ZERO" for c in odd["ai_constituent_odds"])

    # Erratum lock: longitude >= 29.1900 intrudes on LTFJ NW join J2 (29.19212°E) -> MUST fail validation
    bad_odd = copy.deepcopy(odd)
    bad_odd["operational_domain"]["scenery_elements"]["corridor_bounding_box"]["lon_deg"][1] = 29.2100
    bad_report = validate_odd_yaml(bad_odd)
    assert bad_report["passed"] is False
    assert any("J2" in err or "29.1900" in err for err in bad_report["errors"])


def test_gsn_builder_generates_mermaid_and_rejects_developed_g5_g6(tmp_path: Path) -> None:
    gsn = yaml.safe_load(
        (REPO_ROOT / "verification/gsn/safety_case_gsn.yaml").read_text(encoding="utf-8")
    )
    hashes = {f"Sn{i}": f"sha256_{i:02d}" for i in range(1, 11)}
    mermaid = build_gsn_mermaid(gsn, artifact_hashes=hashes, git_sha="deadbeef01")
    assert "G5 [UNDEVELOPED]" in mermaid
    assert "G6 [UNDEVELOPED]" in mermaid
    assert "sha256_07" in mermaid
    for idx in range(1, 11):
        assert f"Sn{idx}" in mermaid

    # Hard lock: G5 and G6 MUST remain undeveloped: true
    for forbidden_id in ("G5", "G6"):
        mutated = copy.deepcopy(gsn)
        for node in mutated["nodes"]:
            if node["id"] == forbidden_id:
                node["undeveloped"] = False
        with pytest.raises(ValueError, match="undeveloped"):
            build_gsn_mermaid(mutated, artifact_hashes=hashes, git_sha="deadbeef01")

    # Verify deterministic injection into SAFETY_CASE.md markers
    target = tmp_path / "SAFETY_CASE.md"
    target.write_text(
        "# Safety Case\n<!-- GSN:BEGIN -->\nSENTINEL_STALE_GSN_CONTENT\n<!-- GSN:END -->\n",
        encoding="utf-8",
    )
    inject_gsn_into_safety_case(target, mermaid)
    updated = target.read_text(encoding="utf-8")
    assert "G5 [UNDEVELOPED]" in updated and "SENTINEL_STALE_GSN_CONTENT" not in updated


def test_safety_case_fta_and_math_derivations_locks() -> None:
    sc = (REPO_ROOT / "docs/SAFETY_CASE.md").read_text(encoding="utf-8")
    for token in (
        "NOT a certified airborne product",
        "SC-VTOL",
        "MOC-5",
        "5 Nov 2026",
        "ASTM F3269-21",
        "2026/010/R",
        "16 Nov 2026",
        "SHT-İHA",
        "30 Jul 2026",
        "M3",
        "İHATTYS",
        "LATCHED_CEL",
        "LATCHED_CSFL",
        "<!-- GSN:BEGIN -->",
        "G5 [UNDEVELOPED]",
        "G6 [UNDEVELOPED]",
    ):
        assert token in sc, f"Missing required lock token in SAFETY_CASE.md: {token}"

    fta = (REPO_ROOT / "docs/FHA_FMEA_TRACEABILITY.md").read_text(encoding="utf-8")
    for idx in range(1, 11):
        assert f"FHA-{idx:02d}" in fta
    assert "3 * lambda_fcc^2 * T" in fta
    assert "3.00e-8" in fta
    assert "1.83e-5" in fta
    assert "lambda_fcc^3 * T^2" not in fta

    math_doc = (REPO_ROOT / "docs/MATH_DERIVATIONS.md").read_text(encoding="utf-8")
    assert len(math_doc.splitlines()) <= 165, "MATH_DERIVATIONS.md must remain a crisp 2-page cold-derivation lock"
    for token in ("60.0%", "441.5", "+17.6%", "2.2 rad/s", "10.6", "log1p", "3 * lambda_fcc^2 * T"):
        assert token in math_doc, f"Missing math lock token in MATH_DERIVATIONS.md: {token}"
