# SPDX-License-Identifier: Apache-2.0
"""Contract tests for PX4 v1.16 internal wq:rate_ctrl Lift+Cruise overlay (Plan 5A Task 2)."""

from __future__ import annotations

from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
OVERLAY_DIR = REPO / "core" / "airframes" / "lift_cruise" / "px4_v116_overlay"


def test_px4_v116_overlay_locks_internal_wq_rate_ctrl_and_forbids_offboard_400hz() -> None:
    params_path = OVERLAY_DIR / "vertirta_lc240_8p1_params.yaml"
    preflight_path = OVERLAY_DIR / "standard_vtol_preflight_check.yaml"
    header_path = OVERLAY_DIR / "vtol_rate_ctrl_overlay.hpp"

    for p in (params_path, preflight_path, header_path):
        assert p.is_file(), f"Missing PX4 v1.16 overlay artifact: {p}"
        raw = p.read_bytes()
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")

    cfg = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    assert cfg["px4_version"] == "v1.16"
    assert cfg["airframe_layout"] == "8+1_lift_cruise"
    assert cfg["rate_loop_placement"] == "px4_internal_wq_rate_ctrl"
    assert cfg["offboard_direct_actuator_400hz_forbidden"] is True

    p = cfg["params"]
    assert p["UXRCE_DDS_SYNCT"] == 1
    assert p["VT_ARSP_BLEND"] == 14.0
    assert p["VT_ARSP_TRANS"] == 28.0
    assert p["FW_AIRSPD_MIN"] == 24.0
    assert p["VT_PSHER_SLEW"] == 0.33
    assert p["DSHOT_BIDIR_EN"] == 1
    assert "EKF2_GPS_DELAY" in p

    topics = cfg["uorb_topics"]
    assert topics["gps"] == "vehicle_gps_position"
    assert topics["esc_rpm"] == "esc_status"
    assert topics["vtol_status"] == "vtol_vehicle_status"

    pf = yaml.safe_load(preflight_path.read_text(encoding="utf-8"))
    assert pf["require_vtol_att_control_active"] is True
    assert pf["require_bidirectional_dshot_esc_status"] is True
    assert pf["forbid_offboard_direct_actuator_400hz"] is True

    hdr = header_path.read_text(encoding="utf-8")
    assert "wq:rate_ctrl" in hdr
    assert "UXRCE_DDS_SYNCT = 1" in hdr or "kUxrceDdsSynct = 1" in hdr
    assert "kOffboardDirectActuator400HzAllowed = false" in hdr
