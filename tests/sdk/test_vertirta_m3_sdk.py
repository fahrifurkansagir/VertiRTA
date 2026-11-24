# SPDX-License-Identifier: Apache-2.0
"""Unit & integration tests for VertiRTA M3 nanobind Python SDK bindings and CLI commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]
SDK_DIR = REPO / "sdk" / "python"
if str(SDK_DIR) not in sys.path:
    sys.path.append(str(SDK_DIR))

from vertirta import (  # noqa: E402
    VertiForensicsEngine,
    VertiForwardHazard,
    VertiGnssDenied,
    VertiMondrianVoiceGate,
)
from vertirta.cli import app  # noqa: E402


def test_m3_forward_hazard_gnss_voice_and_forensics_sdk_bindings(tmp_path: Path) -> None:
    fh = VertiForwardHazard()
    gnss = VertiGnssDenied()
    voice = VertiMondrianVoiceGate()
    forensics = VertiForensicsEngine()

    if os.environ.get("VERTIRTA_REQUIRE_NATIVE") == "1":
        assert fh.backend == "cpp_nanobind"
        assert gnss.backend == "cpp_nanobind"
        assert voice.backend == "cpp_nanobind"
        assert forensics.backend == "cpp_nanobind"

    # 1. Forward Hazard radar-degraded grazing fallback -> <= 15.0 m/s
    fh.reset()
    fh.push_pose_200hz(1_000_000_000, (0.0, 0.0, -45.0), (14.0, 0.0, 0.0))
    fh_res = fh.step(
        1_030_000_000,
        radar_healthy=False,
        wire_lines=[(120.0, 160.0, 1120.0, 165.0, 0.91)],
    )
    assert fh_res["ok"] is True
    assert fh_res["recommended_v_max_mps"] <= 15.0

    # 2. GNSS Spoof Guard 1.5 m/s slow-ramp spoof over land -> latches H3 in <= 1.4 s + rollback
    gnss.reset(2.0)
    dt_s = 0.1
    pos_e = -500.0
    spoof_e = 0.0
    ttd_s = None
    for i in range(190):
        pos_e += 12.0 * dt_s
        v_sp = 1.5 if i >= 160 else 0.0
        spoof_e += v_sp * dt_s
        st = gnss.step_spoof_guard_10hz(
            stamp_ns=(i + 1) * 100_000_000,
            gnss_pos_ned=(200.0, pos_e + spoof_e, -45.0),
            gnss_vel_ned=(0.0, 12.0 + v_sp, 0.0),
            vio_pos_ned=(200.0, pos_e, -45.0),
            vio_vel_ned=(0.0, 12.0, 0.0),
            water_fraction=0.05,
        )
        if st["latched"] and ttd_s is None and i >= 160:
            ttd_s = (i - 160 + 1) * dt_s
            assert st["hypothesis"] == "H3_SLOW_RAMP_SPOOF"
            assert st["rollback_executed"] is True
            assert st["gnss_fused_allowed"] is False
            break
    assert ttd_s is not None and ttd_s <= 1.401

    ctr = gnss.propagate_and_evaluate_ctr(10.0, 12.0, (200.0, pos_e, -45.0), 57.0)
    assert ctr["three_sigma_pos_m"] > 0.0
    assert ctr["ltfj_delta_r_m"] >= ctr["three_sigma_pos_m"]

    # 3. Voice Mondrian Gate: MAYDAY in S3_Garbled (w_m >= 0.15) + < 5% SoC liveness commit
    mayday = voice.evaluate(
        snr_db=-3.0,
        lang="TR",
        candidates=[
            ("DIVERT_SLZ:Kartal_Sahil", 0.44, True),
            ("HOLD_LOITER", 0.36, True),
        ],
        mayday_acoustic_weight=0.22,
    )
    assert mayday["stratum"] == "S3_Garbled"
    assert mayday["decision"] == "EMERGENCY_PREEMPT"

    liveness = voice.evaluate(
        snr_db=10.0,
        lang="EN",
        candidates=[
            ("DIVERT_SLZ:Kartal_Sahil", 0.48, True),
            ("HOLD_LOITER", 0.42, True),
        ],
        soc_pct=15.0,
        soc_reserve_floor_pct=12.0,
    )
    assert liveness["decision"] == "LIVENESS_EMERGENCY_COMMIT"

    # 4. Forensics ERCS CDR (EU) 2020/2034 statutory locks
    ercs = forensics.score_ercs(
        trigger="CRITICAL_ENERGY_OR_ENVELOPE_PROT",
        fault_domain="ROTOR_ACTUATOR_OR_INVERTER",
        key_risk_area="LOC_I_HIGH_ENERGY",
        occupant_capacity=1,
        barriers=[1, 2, 2, 1, 2, 1, 1, 3],
    )
    assert ercs["cictt_category"] == "SCF-NP"
    assert ercs["effective_barrier8_score"] == 0
    assert ercs["occurrence_class_id_431"] == "DRAFT_INCIDENT_ONLY"
    assert ercs["auto_classified_accident_or_serious_incident"] is False

    # Verify CLI forensics-export produces valid XML matching eccairs2_e5x_minimal.xsd
    runner = CliRunner()
    vrta_file = tmp_path / "m3_flight.vrta"
    e5x_file = tmp_path / "draft.e5x.xml"
    r_sim = runner.invoke(app, ["sim", "--duration", "0.25", "--fault", "same_arm_dual", "--out", str(vrta_file)])
    assert r_sim.exit_code == 0, r_sim.stdout

    r_exp = runner.invoke(app, ["forensics-export", str(vrta_file), "--out-e5x", str(e5x_file)])
    assert r_exp.exit_code == 0, r_exp.stdout
    exp_json = json.loads(r_exp.stdout)
    assert exp_json["ok"] is True
    assert exp_json["cictt_category"] == "SCF-NP"
    assert exp_json["effective_barrier8_score"] == 0
    assert exp_json["occurrence_class_id_431"] == "DRAFT_INCIDENT_ONLY"
    assert e5x_file.is_file()

    # Validate emitted XML against real W3C XSD schema if lxml is available (and ElementTree always)
    root = ET.fromstring(e5x_file.read_text(encoding="utf-8"))
    assert root.tag.endswith("SET")
    xsd_path = REPO / "core" / "forensics" / "schemas" / "eccairs2_e5x_minimal.xsd"
    try:
        from lxml import etree  # type: ignore[import-untyped]

        schema = etree.XMLSchema(etree.parse(str(xsd_path)))
        doc = etree.parse(str(e5x_file))
        assert schema.validate(doc), str(schema.error_log)
    except ImportError:
        pass


def test_m3_cli_gnss_spoof_sim_and_m3_demo(tmp_path: Path) -> None:
    runner = CliRunner()

    r_gnss = runner.invoke(app, ["gnss-spoof-sim", "--ramp-mps", "1.5", "--water-fraction", "0.05"])
    assert r_gnss.exit_code == 0, r_gnss.stdout
    gnss_data = json.loads(r_gnss.stdout)
    assert gnss_data["guard"]["hypothesis"] == "H3_SLOW_RAMP_SPOOF"
    assert gnss_data["ttd_s"] is not None and gnss_data["ttd_s"] <= 1.401

    demo_dir = tmp_path / "m3_demo"
    r_demo = runner.invoke(app, ["m3-demo", "--out-dir", str(demo_dir)])
    assert r_demo.exit_code == 0, r_demo.stdout
    demo_data = json.loads(r_demo.stdout)
    assert demo_data["forward_hazard"]["recommended_v_max_mps"] <= 15.0
    assert demo_data["voice_mayday"]["decision"] == "EMERGENCY_PREEMPT"
    assert demo_data["forensics_e5x"]["ok"] is True
    assert Path(demo_data["files"]["e5x"]).is_file()
