# SPDX-License-Identifier: Apache-2.0
"""Tests for PX4 ULog forensics replayer & Istanbul live METAR/wind/ADS-B corridor bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.realworld.fetch_istanbul_live_feed import (  # noqa: E402
    build_smoke_fixtures,
    run_istanbul_corridor_audit,
)
from scripts.realworld.run_px4_ulog_forensics import (  # noqa: E402
    parse_px4_ulog,
    validate_e5x_xml_against_xsd,
    verify_sidecar_chain_vertigcs_compatible,
    write_ed155a_sidecar_jsonl,
    write_synthetic_smoke_ulog,
)


def _recompute_report_sha256(report: dict[str, object]) -> str:
    body = {k: v for k, v in report.items() if k != "report_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def test_px4_ulog_forensics_smoke_cpu_zero_network_xsd_and_sidecar_chain(tmp_path: Path) -> None:
    """Zero-network `--smoke-cpu` must parse ULog v1, verify ED-155A chain, and pass W3C XSD."""
    cache_dir = tmp_path / "ulog_cache"
    out_json = tmp_path / "px4_smoke_proof.json"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/realworld/run_px4_ulog_forensics.py",
            "--smoke-cpu",
            "--cache-dir",
            str(cache_dir),
            "--out-json",
            str(out_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    assert out_json.is_file()

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["mode"] == "SMOKE_CPU_ONLY"
    assert rep["colab_cu_consumed"] == 0.0
    assert rep["all_gates_passed"] is True
    assert rep["report_sha256"] == _recompute_report_sha256(rep)

    # Pure-Python ULog v1 telemetry extraction
    src = rep["source_ulog"]
    assert src["zero_gpl_parser"] == "pure_python_ulog_v1_apache2"
    assert src["ulog_version"] in (0, 1)
    assert src["flight_duration_s"] > 10.0
    assert src["extracted_telemetry_rows"]["vehicle_local_position"] == 120
    assert src["extracted_telemetry_rows"]["vehicle_attitude"] == 120
    assert src["extracted_telemetry_rows"]["battery_status"] == 120
    assert src["extracted_telemetry_rows"]["actuator_outputs"] == 120

    # Sidecar SHA-256 chain & tamper rejection
    sc = rep["ed155a_sidecar"]
    assert sc["vertigcs_chain_verified"] is True
    assert sc["tamper_rejection_self_test_passed"] is True
    assert sc["event_count"] >= 2

    # Statutory ECCAIRS2 E5X + W3C XSD verification
    e5x = rep["forensics_e5x"]
    assert e5x["ok"] is True
    assert e5x["xsd_valid"] is True
    assert e5x["xsd_errors"] == []
    assert e5x["cictt_category"] == "SCF-NP"
    assert e5x["effective_barrier8_score"] == 0
    assert e5x["occurrence_class_id_431"] == "DRAFT_INCIDENT_ONLY"
    assert e5x["auto_classified_accident_or_serious_incident"] is False
    assert e5x["numeric_hallucination_count"] == 0
    assert e5x["strictly_ascending_attribute_ids"] is True
    assert e5x["na_injury_ids_omitted"] is True
    assert e5x["narrative_languages"] == ["en", "tr"]

    # Independently re-validate emitted XML & sidecar files on disk in tmp_path
    e5x_file = cache_dir / "outputs" / e5x["e5x_xml_filename"]
    sidecar_file = cache_dir / "outputs" / sc["filename"]
    assert e5x_file.is_file()
    assert sidecar_file.is_file()
    xsd_check = validate_e5x_xml_against_xsd(e5x_file)
    assert xsd_check["xsd_valid"] is True
    chain_check = verify_sidecar_chain_vertigcs_compatible(
        sidecar_file.read_text(encoding="utf-8"),
        public_key_hex=sc["public_key_hex"],
    )
    assert chain_check["valid"] is True


def test_px4_ulog_parser_and_sidecar_tamper_detection_unit(tmp_path: Path) -> None:
    """Verify ULog parser round-trip and fail-closed detection on sidecar payload/chain tampering."""
    ulog_file = write_synthetic_smoke_ulog(tmp_path / "unit.ulg", n_samples=25)
    parsed = parse_px4_ulog(ulog_file)
    assert parsed.version == 1
    assert parsed.info_messages["sys_name"] == "PX4"
    assert len(parsed.datasets[("vehicle_local_position", 0)].rows) == 25
    assert len(parsed.logged_strings) == 1

    sc_path = tmp_path / "unit.vrta_events.jsonl"
    pub_hex, _ = write_ed155a_sidecar_jsonl(
        sc_path,
        [
            {
                "seq": 0,
                "t_sim_us": 0,
                "kind": "RTA_ENVELOPE_PROTECTION",
                "snr_stratum": "S0_Clean",
                "conformal_set_size": 1,
                "asr_transcript_utf8": "Event zero",
                "gsn_node_id": "G1.1",
            },
            {
                "seq": 10,
                "t_sim_us": 100_000,
                "kind": "MOTOR_FDI_FAULT",
                "snr_stratum": "S1_Cruise",
                "conformal_set_size": 1,
                "asr_transcript_utf8": "Event ten",
                "gsn_node_id": "G2.1",
            },
        ],
    )
    raw_sc = sc_path.read_text(encoding="utf-8")
    assert verify_sidecar_chain_vertigcs_compatible(raw_sc, public_key_hex=pub_hex)["valid"] is True

    # Single character tamper must fail closed
    tampered = raw_sc.replace("Event ten", "Event 999")
    assert verify_sidecar_chain_vertigcs_compatible(tampered, public_key_hex=pub_hex)["valid"] is False

    # Wrong HMAC key must fail closed
    wrong_key = "1" * 64
    assert verify_sidecar_chain_vertigcs_compatible(raw_sc, public_key_hex=wrong_key)["valid"] is False


def test_istanbul_live_feed_smoke_cpu_zero_network_and_well_clear_fail_closed(tmp_path: Path) -> None:
    """Zero-network `--smoke-cpu` verifies LTFJ J2, official DHMI LTBX, CBVF wind, and ADS-B separation."""
    out_json = tmp_path / "istanbul_smoke_audit.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/realworld/fetch_istanbul_live_feed.py",
            "--smoke-cpu",
            "--out-json",
            str(out_json),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.returncode == 0
    assert out_json.is_file()

    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert rep["mode"] == "SMOKE_CPU_ONLY"
    assert rep["colab_cu_consumed"] == 0.0
    assert rep["all_gates_passed"] is True
    assert rep["report_sha256"] == _recompute_report_sha256(rep)

    # Airspace boundaries: LTFJ J2 & official DHMI AIP LTBX (40d59'26"N = 40.990556 N)
    air = rep["airspace_boundary_audit"]
    assert air["ltbx_official_dhmi_arp_lat_lon"][0] == pytest.approx(40.990556, abs=1e-4)
    assert air["ltbx_yaml_to_code_arp_parity_m"] < 1.0
    assert air["min_corridor_ltfj_clearance_m"] > 1000.0
    assert air["min_corridor_ltbx_clearance_m"] > 1500.0
    assert air["all_waypoints_clear_of_ltfj_and_ltbx"] is True

    # Weather + CBVF wind/energy envelope
    assert "LTFJ" in rep["live_weather"]["metar"]
    assert rep["live_weather"]["metar"]["LTFJ"]["raw_metar"].startswith("METAR LTFJ")
    env = rep["cbvf_wind_energy_envelope"]
    assert env["cbvf_gust_capped_airspeed_mps"] <= env["nominal_v_max_mps"]
    assert env["soc_margin_above_reserve_pct"] > 20.0
    assert env["wind_envelope_verdict"] == "GO_NOMINAL"

    # ADS-B traffic audit
    adsb = rep["live_adsb_traffic_audit"]
    assert adsb["total_aircraft_with_position"] == 4
    assert adsb["airborne_aircraft_count"] == 3
    assert adsb["ground_aircraft_count"] == 1
    assert adsb["well_clear_violation_count"] == 0

    # Fail-closed check: inject a low-altitude airborne intruder right over Kartal Vertiport
    fixtures = build_smoke_fixtures()
    fixtures["adsb"]["ac"].append(
        {
            "hex": "dead01",
            "flight": "INTRUDER",
            "t": "H135",
            "lat": 40.8862,
            "lon": 29.1878,
            "alt_baro": 300,
            "alt_geom": 300,
            "gs": 95.0,
        }
    )
    statuses = {
        "metar": {"ok": True},
        "open_meteo": {"ok": True},
        "adsb": {"ok": True},
    }
    intruder_rep = run_istanbul_corridor_audit(fixtures, statuses, mode="SMOKE_INTRUDER_TEST")
    assert intruder_rep["live_adsb_traffic_audit"]["well_clear_violation_count"] == 1
    assert intruder_rep["safety_gates"]["adsb_well_clear_zero_violations"] is False
    assert intruder_rep["all_gates_passed"] is False


def test_realworld_live_proof_artifacts_sealed_utf8_lf_and_zero_leaked_paths() -> None:
    """Verify tracked v4.0.0 real-world proof JSONs and owned files meet schema, SHA-256, and hygiene gates."""
    px4_art = REPO_ROOT / "artifacts/v4.0.0/realworld_px4_ulog_forensics_proof.json"
    ist_art = REPO_ROOT / "artifacts/v4.0.0/realworld_istanbul_live_feed_audit.json"
    assert px4_art.is_file(), "Missing realworld_px4_ulog_forensics_proof.json"
    assert ist_art.is_file(), "Missing realworld_istanbul_live_feed_audit.json"

    px4_rep = json.loads(px4_art.read_text(encoding="utf-8"))
    assert px4_rep["mode"] == "LIVE_REALWORLD_PX4_ULOG"
    assert px4_rep["colab_cu_consumed"] == 0.0
    assert px4_rep["all_gates_passed"] is True
    assert px4_rep["forensics_e5x"]["xsd_valid"] is True
    assert px4_rep["forensics_e5x"]["cictt_category"] == "SCF-NP"
    assert px4_rep["forensics_e5x"]["effective_barrier8_score"] == 0
    assert px4_rep["forensics_e5x"]["occurrence_class_id_431"] == "DRAFT_INCIDENT_ONLY"
    assert px4_rep["forensics_e5x"]["numeric_hallucination_count"] == 0
    assert px4_rep["ed155a_sidecar"]["vertigcs_chain_verified"] is True
    assert px4_rep["source_ulog"]["total_ulog_messages"] > 1000
    assert px4_rep["report_sha256"] == _recompute_report_sha256(px4_rep)

    ist_rep = json.loads(ist_art.read_text(encoding="utf-8"))
    assert ist_rep["mode"] == "LIVE_REALWORLD_ISTANBUL_FEED"
    assert ist_rep["colab_cu_consumed"] == 0.0
    assert ist_rep["all_gates_passed"] is True
    assert ist_rep["live_weather"]["metar"]["LTFJ"]["raw_metar"].startswith("METAR LTFJ")
    assert ist_rep["live_adsb_traffic_audit"]["total_aircraft_with_position"] >= 1
    assert ist_rep["airspace_boundary_audit"]["min_corridor_ltfj_clearance_m"] > 0.0
    assert ist_rep["airspace_boundary_audit"]["min_corridor_ltbx_clearance_m"] > 0.0
    assert ist_rep["report_sha256"] == _recompute_report_sha256(ist_rep)

    owned_files = (
        REPO_ROOT / "scripts/realworld/run_px4_ulog_forensics.py",
        REPO_ROOT / "scripts/realworld/fetch_istanbul_live_feed.py",
        REPO_ROOT / "tests/realworld/test_px4_ulog_and_istanbul_live_feed.py",
        px4_art,
        ist_art,
    )
    forbidden = ("/home/" + "dias", "C:\\" + "Users\\", "Obsidian" + "Vault")
    for p in owned_files:
        raw = p.read_bytes()
        assert len(raw) < 1_000_000, f"{p.name} exceeds 1 MB cap"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM in {p.name}"
        assert b"\r" not in raw, f"CR byte in {p.name}"
        text = raw.decode("utf-8")
        for tok in forbidden:
            assert tok not in text, f"Leaked path token {tok!r} in {p.name}"
