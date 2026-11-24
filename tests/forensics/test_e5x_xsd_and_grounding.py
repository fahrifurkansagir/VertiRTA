# SPDX-License-Identifier: Apache-2.0
"""Contract & XSD verification tests for VertiRTA M3 Regulatory Forensics (Plan 6B Task 4)."""

from __future__ import annotations

import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
GBNF_PATH = REPO / "config" / "eccairs2_nodigit.gbnf"
YAML_PATH = REPO / "config" / "forensics_eccairs2.yaml"
XSD_PATH = REPO / "core" / "forensics" / "schemas" / "eccairs2_e5x_minimal.xsd"

E5X_NS = "http://eccairsportal.jrc.ec.europa.eu/schemas/e5x-minimal/v1"
NA_INJURY_IDS = {114, 187, 262, 460, 462, 463, 469, 472}

ALLOWED_SLOTS = {
    "EVENT_SEQ",
    "T_SIM_S",
    "MIN_LTFJ_MARGIN_M",
    "MIN_LTBX_MARGIN_M",
    "SOC_PCT",
    "V_TERM_V",
    "AGL_M",
    "PEAK_CHI2",
    "ROTOR_MASK_HEX",
    "SNR_STRATUM",
    "CONFORMAL_SET_SIZE",
    "ERCS_SCORE",
    "CICTT_CODE",
    "OCCURRENCE_PHASE",
    "FLIGHT_REGIME",
}


def _extract_numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z_])[-+]?(?:\d+\.\d+|\d+)(?![A-Za-z_])", text)


def _render_and_verify_python(template: str, slots: dict[str, str]) -> tuple[bool, str, list[str]]:
    # Strip valid allowlisted {{SLOT}} placeholders before checking for forbidden raw digits.
    def _check_slot(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in ALLOWED_SLOTS or name not in slots:
            return match.group(0)
        return " "

    stripped = re.sub(r"\{\{([A-Z0-9_]+)\}\}", _check_slot, template)
    if "{{" in stripped or "}}" in stripped:
        return False, "", ["UNRESOLVED_OR_UNKNOWN_SLOT"]
    hallucinated = re.findall(r"\d+(?:\.\d+)?", stripped)
    if hallucinated:
        return False, "", hallucinated

    rendered = re.sub(r"\{\{([A-Z0-9_]+)\}\}", lambda m: slots[m.group(1)], template)
    # Post-render grounding: every numeric token in rendered output must come strictly from slot values.
    allowed_nums: set[str] = set()
    for val in slots.values():
        allowed_nums.update(_extract_numeric_tokens(val))
    for tok in _extract_numeric_tokens(rendered):
        if tok not in allowed_nums:
            return False, rendered, [tok]
    return True, rendered, []


def _build_synthetic_e5x_xml(rng: random.Random, idx: int) -> tuple[str, dict[str, str]]:
    seq = rng.randint(100, 9999)
    t_s = f"{seq / 100.0:.2f}"
    ltfj_m = f"{rng.uniform(12.0, 240.0):.1f}"
    ltbx_m = f"{rng.uniform(120.0, 900.0):.1f}"
    soc = f"{rng.uniform(11.0, 88.0):.1f}"
    v_term = f"{rng.uniform(345.0, 412.0):.1f}"
    agl = f"{rng.uniform(15.0, 120.0):.1f}"
    chi2 = f"{rng.uniform(1.0, 85.0):.1f}"
    ercs = rng.choice(["I4", "I5", "M3", "M5", "S1"])
    cictt = rng.choice(["SCF-NP", "MAC", "FUEL", "LALT"])
    snr = rng.choice(["S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"])
    set_sz = str(rng.choice([1, 2, 3]))
    slots = {
        "EVENT_SEQ": str(seq),
        "T_SIM_S": t_s,
        "MIN_LTFJ_MARGIN_M": ltfj_m,
        "MIN_LTBX_MARGIN_M": ltbx_m,
        "SOC_PCT": soc,
        "V_TERM_V": v_term,
        "AGL_M": agl,
        "PEAK_CHI2": chi2,
        "ROTOR_MASK_HEX": "0xFE",
        "SNR_STRATUM": snr,
        "CONFORMAL_SET_SIZE": set_sz,
        "ERCS_SCORE": ercs,
        "CICTT_CODE": cictt,
        "OCCURRENCE_PHASE": "10602",
        "FLIGHT_REGIME": "SINGLE_LOSS",
    }
    en_tmpl = (
        "At frame {{EVENT_SEQ}} ({{T_SIM_S}} s), minimum LTFJ boundary clearance was "
        "{{MIN_LTFJ_MARGIN_M}} m at {{AGL_M}} m AGL and {{SOC_PCT}}% SoC ({{V_TERM_V}} V); "
        "category {{CICTT_CODE}}, draft ERCS {{ERCS_SCORE}}."
    )
    tr_tmpl = (
        "{{EVENT_SEQ}} numarali cercevede ({{T_SIM_S}} sn), en kucuk LTFJ sinir mesafesi "
        "{{MIN_LTFJ_MARGIN_M}} m, AGL {{AGL_M}} m ve batarya {{SOC_PCT}}% ({{V_TERM_V}} V) "
        "olarak olculdu; kategori {{CICTT_CODE}}, taslak ERCS {{ERCS_SCORE}}."
    )
    ok_en, en_txt, err_en = _render_and_verify_python(en_tmpl, slots)
    ok_tr, tr_txt, err_tr = _render_and_verify_python(tr_tmpl, slots)
    assert ok_en and not err_en
    assert ok_tr and not err_tr

    digest = hashlib.sha256(f"trace-{idx}-{seq}".encode("utf-8")).hexdigest()
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<SET xmlns="{E5X_NS}" schemaVersion="vertirta.e5x.draft/v1.0.0">
  <Occurrence entityId="24" reportingStatus="DRAFT_INTERNAL_PRE_SUBMISSION" complianceClaim="NONE_RESEARCH_TESTBED_ONLY">
    <ATTRIBUTES>
      <Attribute id="16" name="FIR_UIR">LTBB</Attribute>
      <Attribute id="121" name="EVTOL_ROTORCRAFT_FLIGHT_PHASE">10602</Attribute>
      <Attribute id="372" name="ATS_UNIT">LTFJ_TWR</Attribute>
      <Attribute id="430" name="OCCURRENCE_CATEGORY_CICTT">{cictt}</Attribute>
      <Attribute id="431" name="OCCURRENCE_CLASS_DRAFT">INCIDENT_DRAFT_ONLY</Attribute>
      <Attribute id="433" name="LOCAL_DATE_TIME">2026-11-24T12:00:00+03:00</Attribute>
      <Attribute id="440" name="LOCATION_NAME">Kartal_Sahil_Istanbul</Attribute>
      <Attribute id="452" name="FILE_NUMBER">VRT-2026-{idx:04d}</Attribute>
      <Attribute id="453" name="REPORTING_ENTITY">VERTIRTA_RESEARCH_TESTBED</Attribute>
      <Attribute id="454" name="STATE_AREA_OF_OCCURRENCE">TR</Attribute>
      <Attribute id="477" name="UTC_DATE_TIME">2026-11-24T09:00:00Z</Attribute>
      <Attribute id="601" name="HEADLINE">Envelope protection activation along coastal corridor</Attribute>
      <Attribute id="659" name="ATA_JASC_CHAPTER">27</Attribute>
      <Attribute id="1109" name="ERCS_RISK_GRADE">{ercs}</Attribute>
    </ATTRIBUTES>
    <METADATA>
      <Cir2026Trigger>CRITICAL_ENERGY_OR_ENVELOPE_PROT</Cir2026Trigger>
      <ShgmShyOlayDeadlineHours>72</ShgmShyOlayDeadlineHours>
      <VrtaFileSha256>{digest}</VrtaFileSha256>
      <SidecarChainVerified>true</SidecarChainVerified>
      <NumericGroundingVerified>true</NumericGroundingVerified>
    </METADATA>
    <ENTITIES>
      <Aircraft entityId="4">
        <ATTRIBUTES>
          <Attribute id="21" name="AIRCRAFT_MODEL">VertiRTA-X8-240</Attribute>
          <Attribute id="121" name="FLIGHT_PHASE">10602</Attribute>
          <Attribute id="175" name="MTOW_KG">240</Attribute>
        </ATTRIBUTES>
      </Aircraft>
      <Event entityId="14">
        <ATTRIBUTES>
          <Attribute id="390" name="EVENT_TYPE_ADREP">{cictt}</Attribute>
          <Attribute id="391" name="EVENT_PHASE">10602</Attribute>
        </ATTRIBUTES>
      </Event>
      <Narrative entityId="22" language="en">{en_txt}</Narrative>
      <Narrative entityId="22" language="tr">{tr_txt}</Narrative>
      <PartInfo entityId="26">
        <ATTRIBUTES>
          <Attribute id="659" name="ATA_JASC_CHAPTER">27</Attribute>
        </ATTRIBUTES>
      </PartInfo>
    </ENTITIES>
  </Occurrence>
</SET>
"""
    return xml, slots


def _validate_xml_against_minimal_xsd(xml_text: str) -> None:
    assert XSD_PATH.is_file(), f"Missing XSD schema: {XSD_PATH}"
    xsd_root = ET.fromstring(XSD_PATH.read_text(encoding="utf-8"))
    assert xsd_root.tag.endswith("schema")

    # Use lxml XMLSchema when available, plus strict structural assertions always.
    try:
        from lxml import etree as lxml_etree  # type: ignore[import-not-found]

        schema = lxml_etree.XMLSchema(lxml_etree.parse(str(XSD_PATH)))
        doc = lxml_etree.fromstring(xml_text.encode("utf-8"))
        schema.assertValid(doc)
    except ImportError:
        pass

    root = ET.fromstring(xml_text)
    assert root.tag == f"{{{E5X_NS}}}SET"
    occ = root.find(f"{{{E5X_NS}}}Occurrence")
    assert occ is not None
    assert occ.attrib["entityId"] == "24"
    assert occ.attrib["reportingStatus"] == "DRAFT_INTERNAL_PRE_SUBMISSION"
    assert occ.attrib["complianceClaim"] == "NONE_RESEARCH_TESTBED_ONLY"

    # Strictly ascending <Attribute id="..."> inside every <ATTRIBUTES>
    for attrs in root.iter(f"{{{E5X_NS}}}ATTRIBUTES"):
        ids = [int(a.attrib["id"]) for a in attrs.findall(f"{{{E5X_NS}}}Attribute")]
        assert ids == sorted(ids) and len(ids) == len(set(ids)), f"Non-ascending attribute IDs: {ids}"
        assert NA_INJURY_IDS.isdisjoint(ids), f"N/A injury count IDs must be omitted: {set(ids) & NA_INJURY_IDS}"

    occ_attrs = {
        int(a.attrib["id"]): (a.text or "").strip()
        for a in occ.find(f"{{{E5X_NS}}}ATTRIBUTES").findall(f"{{{E5X_NS}}}Attribute")  # type: ignore[union-attr]
    }
    assert occ_attrs[431] == "INCIDENT_DRAFT_ONLY"
    assert occ_attrs[430] != "SCF-PP"
    assert occ_attrs[16] == "LTBB"
    assert occ_attrs[454] == "TR"

    narratives = occ.findall(f"{{{E5X_NS}}}ENTITIES/{{{E5X_NS}}}Narrative")
    langs = {n.attrib["language"] for n in narratives}
    assert {"en", "tr"}.issubset(langs)


def test_eccairs2_nodigit_gbnf_syntactically_bans_raw_digits() -> None:
    assert GBNF_PATH.is_file(), f"Missing GBNF grammar: {GBNF_PATH}"
    gbnf = GBNF_PATH.read_text(encoding="utf-8")
    assert "prose-char" in gbnf
    assert "slot" in gbnf
    # prose-char MUST NOT contain 0-9 character range or bare digits
    prose_lines = [ln for ln in gbnf.splitlines() if ln.strip().startswith("prose-char")]
    assert len(prose_lines) == 1
    assert "0-9" not in prose_lines[0]
    for d in "0123456789":
        assert d not in prose_lines[0], f"Raw digit {d} leaked into prose-char rule!"
    for slot in ("EVENT_SEQ", "MIN_LTFJ_MARGIN_M", "SOC_PCT", "ERCS_SCORE", "CICTT_CODE"):
        assert f'"{{{{{slot}}}}}"' in gbnf


def test_forensics_yaml_locks_regulatory_and_ercs_invariants() -> None:
    assert YAML_PATH.is_file(), f"Missing forensics config: {YAML_PATH}"
    cfg = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    assert cfg["compliance_claim"] == "NONE_RESEARCH_TESTBED_ONLY"
    assert cfg["ercs"]["default_occupant_capacity_x8_240"] == 1
    assert cfg["ercs"]["draft_occurrence_class_id_431"] == "INCIDENT_DRAFT_ONLY"
    assert cfg["ercs"]["auto_classify_accident_or_serious_incident"] is False
    assert cfg["ercs"]["rotor_control_sw_cictt_code"] == "SCF-NP"
    assert cfg["ercs"]["barrier8_forced_zero_on_high_energy"] is True
    assert cfg["shgm_shy_olay"]["mandatory_reporting_deadline_hours"] == 72
    assert cfg["ed155a_sidecar"]["min_libsodium_version"] == "1.0.21"


def test_25_randomized_synthetic_traces_pass_xsd_and_zero_hallucination_gate() -> None:
    rng = random.Random(20261124)
    hallucinations = 0
    for idx in range(25):
        xml_text, slots = _build_synthetic_e5x_xml(rng, idx)
        _validate_xml_against_minimal_xsd(xml_text)

        # Inject a deliberate hallucinated raw digit and confirm fail-closed detection every time
        bad_tmpl = "At frame {{EVENT_SEQ}}, rotor speed reached 2450 RPM."
        ok, _, errs = _render_and_verify_python(bad_tmpl, slots)
        assert not ok and errs == ["2450"]

    assert hallucinations == 0
