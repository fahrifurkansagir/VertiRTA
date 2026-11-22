"""Contract tests for PX4 v1.16 VertiRTA overlay (Plan 1A Task 10 + DR-1/DR-2 locks)."""

from __future__ import annotations

import re
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PX4_OVERLAY = REPO_ROOT / "px4_overlay"

UORB_TYPE_SIZES: dict[str, tuple[int, int]] = {
    "bool": (1, 1),
    "int8": (1, 1),
    "uint8": (1, 1),
    "char": (1, 1),
    "int16": (2, 2),
    "uint16": (2, 2),
    "int32": (4, 4),
    "uint32": (4, 4),
    "float32": (4, 4),
    "int64": (8, 8),
    "uint64": (8, 8),
    "float64": (8, 8),
}


def parse_uorb_fields(msg_text: str) -> list[tuple[str, str, int, int]]:
    """Return list of (base_type, name, count, total_bytes) for payload fields."""
    fields: list[tuple[str, str, int, int]] = []
    for raw_line in msg_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or "=" in line:
            continue
        parts = line.split()
        assert len(parts) >= 2, f"Malformed uORB field line: {raw_line!r}"
        type_token, field_name = parts[0], parts[1]
        m = re.fullmatch(r"([a-z0-9_]+)(?:\[(\d+)\])?", type_token)
        assert m is not None, f"Unsupported uORB type token: {type_token!r}"
        base_type = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        assert base_type in UORB_TYPE_SIZES, f"Unknown uORB scalar type: {base_type!r}"
        elem_size, _ = UORB_TYPE_SIZES[base_type]
        fields.append((base_type, field_name, count, elem_size * count))
    return fields


def sum_uorb_field_bytes(msg_text: str) -> int:
    return sum(total_bytes for _, _, _, total_bytes in parse_uorb_fields(msg_text))


def check_natural_alignment_no_padding(msg_text: str) -> int:
    """Verify declaration-order layout has zero implicit C struct padding."""
    offset = 0
    for base_type, field_name, _, total_bytes in parse_uorb_fields(msg_text):
        _, align = UORB_TYPE_SIZES[base_type]
        assert offset % align == 0, (
            f"Implicit padding before field {field_name!r} ({base_type}): "
            f"offset={offset} not aligned to {align}"
        )
        offset += total_bytes
    assert offset % 8 == 0, f"Total struct size {offset} is not 8-byte aligned"
    return offset


ROS2_FIELD_NAME_RE = re.compile(r"^[a-z](?!.*__)[a-z0-9_]*[a-z0-9]$|^[a-z]$")


def assert_rosidl_compliant_field_names(msg_text: str) -> None:
    """Enforce ROS 2 Jazzy rosidl_parser field naming (no leading/trailing '_' or '__')."""
    for _, field_name, _, _ in parse_uorb_fields(msg_text):
        assert ROS2_FIELD_NAME_RE.fullmatch(field_name), (
            f"Field name {field_name!r} violates ROS 2 rosidl_parser naming rules"
        )


def test_envelope_msg_is_exact_72_bytes_packed_and_aligned():
    msg_path = PX4_OVERLAY / "msg" / "VertirtaCapabilityEnvelope.msg"
    msg = msg_path.read_text(encoding="utf-8")

    assert "float32 chi2_residual" in msg
    assert "uint32 reserved_pad" in msg
    assert sum_uorb_field_bytes(msg) == 72
    assert check_natural_alignment_no_padding(msg) == 72
    assert_rosidl_compliant_field_names(msg)

    field_names = {name for _, name, _, _ in parse_uorb_fields(msg)}
    required = {
        "timestamp",
        "timestamp_sample",
        "fault_bitmask",
        "tmr_state",
        "imu_health_mask",
        "flight_regime",
        "t_max",
        "a_xy_max",
        "tilt_max",
        "vz_sink_max",
        "v_term",
        "i_pack",
        "soc",
        "p_hover_deg",
        "spin_rate_ref",
        "spin_rate_meas",
        "chi2_residual",
        "fdi_latency_ms",
        "reserved_align",
        "reserved_pad",
    }
    missing = required - field_names
    assert not missing, f"Missing fields in VertirtaCapabilityEnvelope.msg: {missing}"


def test_rta_setpoint_msg_is_aligned_and_complete():
    msg_path = PX4_OVERLAY / "msg" / "VertirtaRtaSetpoint.msg"
    msg = msg_path.read_text(encoding="utf-8")

    size = check_natural_alignment_no_padding(msg)
    assert size == sum_uorb_field_bytes(msg)
    assert size == 56
    assert_rosidl_compliant_field_names(msg)

    field_names = {name for _, name, _, _ in parse_uorb_fields(msg)}
    for required in (
        "timestamp",
        "timestamp_sample",
        "accel_ned",
        "thrust_des_n",
        "yaw_sp",
        "yawspeed_sp",
        "vz_limit_mps",
        "control_mode",
        "cel_latched",
        "valid",
    ):
        assert required in field_names


def test_dds_topics_yaml_exports_envelope_and_subscribes_setpoint():
    cfg_path = PX4_OVERLAY / "config" / "dds_topics_vertirta.yaml"
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    pubs = {entry["topic"]: entry for entry in data["publications"]}
    subs = {entry["topic"]: entry for entry in data["subscriptions"]}

    assert "/fmu/out/vertirta_capability_envelope" in pubs
    env_pub = pubs["/fmu/out/vertirta_capability_envelope"]
    assert env_pub["type"] == "px4_msgs::msg::VertirtaCapabilityEnvelope"
    assert env_pub["rate_limit"] == 100

    assert "/fmu/in/vertirta_rta_setpoint" in subs
    sp_sub = subs["/fmu/in/vertirta_rta_setpoint"]
    assert sp_sub["type"] == "px4_msgs::msg::VertirtaRtaSetpoint"


def test_airframe_locks_cel_params_and_cleanly_switches_indi_vs_stock():
    af = (PX4_OVERLAY / "airframes" / "12050_gz_vertirta_x8_240").read_text(encoding="utf-8")
    rc = (PX4_OVERLAY / "airframes" / "rc.vertirta_apps").read_text(encoding="utf-8")

    required_tokens = [
        "CA_ROTOR_COUNT 8",
        "EKF2_MULTI_IMU 3",
        "SENS_IMU_MODE 0",
        "IMU_INTEG_RATE 400",
        "SYS_FAILURE_EN 1",
        "FD_FAIL_R 0",
        "FD_FAIL_P 0",
        "CA_FAILURE_MODE 1",
        "SIM_GZ_EC_MAX 240",
        "VERTIRTA_USE_INDI",
        "rc.vertirta_apps",
        "kartal_coast",
        "vertirta_x8_240",
        "UXRCE_DDS_SYNCT 1",
        "UXRCE_DDS_PRT 8888",
    ]
    for token in required_tokens:
        assert token in af, f"Missing required airframe lock token: {token!r}"

    for idx in range(8):
        assert f"CA_ROTOR{idx}_CT" in af, f"Missing CA_ROTOR{idx}_CT in airframe"
        assert f"SIM_GZ_EC_MAX{idx + 1} 240" in af, f"Missing SIM_GZ_EC_MAX{idx + 1} 240"

    # rc.vertirta_apps must start mc_indi_wls_control and NEVER start stock MC controllers
    active_rc_lines = [
        line.strip()
        for line in rc.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any("mc_indi_wls_control start" in line for line in active_rc_lines)
    for forbidden in (
        "mc_pos_control start",
        "mc_att_control start",
        "mc_rate_control start",
        "control_allocator start",
    ):
        assert not any(forbidden in line for line in active_rc_lines), (
            f"rc.vertirta_apps must exclude stock module {forbidden!r}"
        )


def test_gz_bridge_patch_has_all_four_parts_and_valid_unified_diff_hunks():
    patch_text = (PX4_OVERLAY / "patches" / "px4_v116_gz_bridge_multi_imu.patch").read_text(
        encoding="utf-8"
    )

    # Validate all 4 DR-1 patch components
    assert "_px4_gyros" in patch_text and "_px4_accels" in patch_text
    assert "PX4Gyroscope" in patch_text and "PX4Accelerometer" in patch_text
    assert "DeviceId" in patch_text or "device_id" in patch_text
    assert "imu_sensor_" in patch_text
    assert "air_pressure_sensor_" in patch_text
    assert "sleep_count_limit" in patch_text and "5000" in patch_text
    assert "_failure_injection_sub" in patch_text
    assert "GZMixingInterfaceESC::updateOutputs" in patch_text
    assert "VEHICLE_CMD_INJECT_FAILURE" in patch_text

    # Strict unified-diff hunk count verification
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    lines = patch_text.splitlines()
    hunk_count = 0
    i = 0
    while i < len(lines):
        m = hunk_re.match(lines[i])
        if not m:
            i += 1
            continue
        hunk_count += 1
        expected_old = int(m.group(2)) if m.group(2) is not None else 1
        expected_new = int(m.group(4)) if m.group(4) is not None else 1
        old_seen = 0
        new_seen = 0
        i += 1
        while i < len(lines) and not lines[i].startswith(("@@ ", "diff --git ", "--- ", "+++ ")):
            ln = lines[i]
            if ln.startswith("-"):
                old_seen += 1
            elif ln.startswith("+"):
                new_seen += 1
            elif ln.startswith(" "):
                old_seen += 1
                new_seen += 1
            elif ln.startswith("\\ No newline"):
                pass
            else:
                raise AssertionError(f"Unexpected diff line inside hunk {hunk_count}: {ln!r}")
            i += 1
        assert old_seen == expected_old, (
            f"Hunk #{hunk_count} old line count mismatch: expected {expected_old}, got {old_seen}"
        )
        assert new_seen == expected_new, (
            f"Hunk #{hunk_count} new line count mismatch: expected {expected_new}, got {new_seen}"
        )

    assert hunk_count >= 4, f"Expected >= 4 unified-diff hunks across GZBridge/ESC, got {hunk_count}"


def test_mc_indi_wls_control_module_contracts_and_freestanding_hygiene():
    mod_dir = PX4_OVERLAY / "modules" / "mc_indi_wls_control"
    cmake = (mod_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    hpp = (mod_dir / "mc_indi_wls_control.hpp").read_text(encoding="utf-8")
    cpp = (mod_dir / "mc_indi_wls_control.cpp").read_text(encoding="utf-8")

    assert "px4_add_module" in cmake
    assert "mc_indi_wls_control" in cmake
    assert "vertirta_flight_core" in cmake

    combined = hpp + "\n" + cpp
    assert "wq_configurations::rate_ctrl" in combined
    assert "TmrVoter" in combined
    assert "IndiWlsController" in combined
    assert "MotorFdiObserver" in combined
    assert "TheveninBattery1RC" in combined
    assert "vertirta_capability_envelope" in combined
    assert "vertirta_rta_setpoint" in combined
    assert "actuator_motors" in combined
    assert "chi2_residual" in combined

    # Freestanding C++17 NuttX hygiene: no <algorithm> or std::clamp
    assert "<algorithm>" not in combined
    assert "std::clamp" not in combined
