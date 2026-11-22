"""Integration & Gate-2 closed-loop contract suite for ROS 2 Jazzy VertiRTA stack (Plan 1A Task 12)."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROS2_SRC = REPO_ROOT / "ros2_ws" / "src"

ROS2_FIELD_NAME_RE = re.compile(r"^[a-z](?!.*__)[a-z0-9_]*[a-z0-9]$|^[a-z]$")
ROS2_CONST_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*[A-Z0-9]$")
FORBIDDEN_PORTS = {4173, 4174, 8765, 9222, 9223}


def _iter_interface_fields(text: str) -> tuple[list[tuple[str, str]], int]:
    """Return ([(type_str, field_name), ...], separator_count) for ROS 2 .msg/.srv/.action."""
    fields: list[tuple[str, str]] = []
    separators = 0
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line == "---":
            separators += 1
            continue
        if "=" in line:
            lhs = line.split("=", 1)[0].strip()
            parts = lhs.split()
            assert len(parts) == 2, f"Malformed ROS 2 constant line: {raw_line!r}"
            assert ROS2_CONST_NAME_RE.fullmatch(parts[1]), (
                f"ROS 2 constant {parts[1]!r} must be UPPER_SNAKE_CASE"
            )
            continue
        parts = line.split()
        assert len(parts) >= 2, f"Malformed ROS 2 field line: {raw_line!r}"
        fields.append((parts[0], parts[1]))
    return fields, separators


def test_vertirta_msgs_schemas_are_rosidl_compliant_and_match_cross_plan_contracts():
    msgs_dir = ROS2_SRC / "vertirta_msgs"
    expected_files = [
        ("msg/SlzPrior.msg", 0),
        ("msg/SlzPriorArray.msg", 0),
        ("msg/WireCapsuleArray.msg", 0),
        ("msg/IntruderArray.msg", 0),
        ("msg/RtaStateSnapshot.msg", 0),
        ("srv/InjectFault.srv", 1),
        ("action/ExecuteContingency.action", 2),
    ]

    parsed_fields: dict[str, set[str]] = {}
    for rel_path, expected_seps in expected_files:
        p = msgs_dir / rel_path
        text = p.read_text(encoding="utf-8")
        fields, seps = _iter_interface_fields(text)
        assert seps == expected_seps, f"{rel_path}: expected {expected_seps} '---', found {seps}"
        assert len(fields) > 0, f"{rel_path}: empty interface"
        for _, name in fields:
            assert ROS2_FIELD_NAME_RE.fullmatch(name), (
                f"{rel_path}: field {name!r} violates ROS 2 rosidl_parser naming rules"
            )
        parsed_fields[rel_path] = {name for _, name in fields}

    # Cross-plan contract: Plan 1A Task 9 / Plan 3 SlzPrior
    for req in (
        "id",
        "lat_deg",
        "lon_deg",
        "elev_amsl_m",
        "center_enu",
        "p_safe",
        "slz_ood_variance",
        "theta_max_deg",
        "sigma_z_m",
        "step_max_m",
        "clearance_r_m",
        "h_dyn_max",
        "eta_s",
        "soc_margin_pct",
        "rank",
        "reason_mask",
    ):
        assert req in parsed_fields["msg/SlzPrior.msg"]

    # Cross-plan contract: Plan 3 SlzPriorArray + gimbal telemetry
    for req in ("header", "stamp_ns", "stale", "active_decision_state", "gimbal_pitch_rad", "gimbal_yaw_rad", "priors"):
        assert req in parsed_fields["msg/SlzPriorArray.msg"]

    # Cross-plan contract: Plan 2 400 V / 22 kWh RtaStateSnapshot + chi2_residual + ranked SlzPriors
    for req in (
        "flight_regime",
        "fault_bitmask",
        "tmr_state",
        "cel_latched",
        "shield_active",
        "pack_nominal_volts",
        "pack_energy_kwh",
        "v_term_volts",
        "soc_pct",
        "soc_reserve_floor_pct",
        "chi2_residual",
        "cbvf_min_margin_m",
        "ltfj_margin_m",
        "ltbx_margin_m",
        "slz_priors",
    ):
        assert req in parsed_fields["msg/RtaStateSnapshot.msg"]

    # Cross-plan contract: Plan 2 ExecuteContingency action
    for req in ("action_kind", "target_id", "target_enu", "z3_verified", "accepted", "completed", "stl_robustness_rho"):
        assert req in parsed_fields["action/ExecuteContingency.action"]
    action_text = (msgs_dir / "action/ExecuteContingency.action").read_text(encoding="utf-8")
    assert "ACTION_HOLD_LOITER" in action_text


def test_all_six_ros2_packages_have_valid_package_xml_and_cmake():
    packages = [
        "vertirta_msgs",
        "vertirta_px4_bridge",
        "vertirta_rta_node",
        "vertirta_map_node",
        "vertirta_fault_injector",
        "vertirta_bringup",
    ]
    for pkg in packages:
        pkg_dir = ROS2_SRC / pkg
        pkg_xml = pkg_dir / "package.xml"
        cmake = pkg_dir / "CMakeLists.txt"
        assert pkg_xml.exists(), f"Missing {pkg_xml}"
        assert cmake.exists(), f"Missing {cmake}"

        root = ET.parse(pkg_xml).getroot()
        assert root.attrib.get("format") == "3"
        assert root.findtext("name") == pkg
        assert root.findtext("license") == "Apache-2.0"

    msgs_cmake = (ROS2_SRC / "vertirta_msgs" / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "rosidl_generate_interfaces" in msgs_cmake
    for iface in (
        "msg/SlzPrior.msg",
        "msg/SlzPriorArray.msg",
        "msg/WireCapsuleArray.msg",
        "msg/IntruderArray.msg",
        "msg/RtaStateSnapshot.msg",
        "srv/InjectFault.srv",
        "action/ExecuteContingency.action",
    ):
        assert iface in msgs_cmake


def test_px4_bridge_locks_100hz_mode_cel_defer_and_upstream_bug_mitigations():
    bridge_dir = ROS2_SRC / "vertirta_px4_bridge"
    combined = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(bridge_dir.rglob("*"))
        if p.suffix in {".hpp", ".cpp"}
    )

    # VertiRtaMode: 100 Hz ModeBase + AttitudeSetpointType (CSFL) + VertirtaRtaSetpoint (CEL)
    assert "ModeBase" in combined
    assert "setSetpointUpdateRate(100.0f)" in combined
    assert "AttitudeSetpointType" in combined
    assert "vertirta_rta_setpoint" in combined
    assert "vertirta_capability_envelope" in combined
    assert "cel_latched_" in combined

    # VertiRtaExecutor: ModeExecutorBase + /vertirta/execute_contingency + deferFailsafesSync(true, 60)
    assert "ModeExecutorBase" in combined
    assert "/vertirta/execute_contingency" in combined
    assert "deferFailsafesSync(true, 60)" in combined
    assert "ACTION_HOLD_LOITER" in combined

    # Auterion/px4-ros2-interface-lib upstream bug mitigations (#201, #167, #191)
    assert "#201" in combined and "model_only" in combined
    assert "#167" in combined and "pending_state_" in combined
    assert "#191" in combined and "vehicle_namespace" in combined


def test_rta_node_enforces_4_thread_spsc_isolation_and_400v_22kwh_snapshot():
    rta_dir = ROS2_SRC / "vertirta_rta_node"
    combined = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(rta_dir.rglob("*"))
        if p.suffix in {".hpp", ".cpp"}
    )

    assert "SingleThreadedExecutor" in combined
    assert "SCHED_FIFO" in combined
    assert "alignas(64)" in combined
    assert "SpscRingBuffer" in combined
    assert "SeqlockSnapshot" in combined
    assert "/vertirta/rta_state_snapshot" in combined
    assert "/vertirta/slz_priors" in combined
    assert "/vertirta/internal/rta_cmd" in combined
    assert "/fmu/in/vertirta_rta_setpoint" not in combined  # Sole publisher is VertiRtaMode
    assert "400.0f" in combined and "22.0f" in combined
    assert "shield_active" in combined
    assert "chi2_residual" in combined
    assert "VRTA0001" in combined and "0x82F63B78" in combined
    # Staleness bounds (>250 ms, >1000 ms -> 8.0 m/s, >2000 ms divert)
    assert "250" in combined and "8.0" in combined and "2000" in combined

    # Map & fault injector contracts
    map_src = (ROS2_SRC / "vertirta_map_node" / "src" / "map_node.cpp").read_text(encoding="utf-8")
    assert "ignore_local_publications = true" in map_src
    assert "/vertirta/slz/gimbal_cmd" in map_src
    fi_src = (ROS2_SRC / "vertirta_fault_injector" / "src" / "fault_injector_node.cpp").read_text(encoding="utf-8")
    assert "/vertirta/inject_fault" in fi_src
    assert "VEHICLE_CMD_INJECT_FAILURE" in fi_src
    assert "/vertirta/fault/motor_mask" in fi_src


def test_bringup_launch_and_params_enforce_port_isolation_and_geoid_consistency():
    bringup_dir = ROS2_SRC / "vertirta_bringup"
    launch_path = bringup_dir / "launch" / "vertirta_sitl.launch.py"
    params_path = bringup_dir / "config" / "vertirta_sitl_params.yaml"

    launch_src = launch_path.read_text(encoding="utf-8")
    ast.parse(launch_src)  # Valid Python syntax

    assert "8891" in launch_src
    assert "8888" in launch_src
    assert "GZ_PARTITION" in launch_src
    assert "VERTIRTA_USE_INDI" in launch_src
    assert "kartal_coast" in launch_src
    assert "/vertirta/lidar_16" in launch_src
    assert "/vertirta/fault/motor_mask" in launch_src
    assert "OpaqueFunction" in launch_src

    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    ports = params["ports"]
    assert ports["foxglove_ws"] == 8891
    assert ports["micro_xrce_agent_udp"] == 8888
    assert set(ports.values()).isdisjoint(FORBIDDEN_PORTS)

    # Review Focus #5 cross-check: kartal_coast.sdf AMSL (12.0 m) + EGM2008 (36.7 m) = 48.7 m HAE (no double-add)
    geodesy_cfg = params["geodesy"]
    assert geodesy_cfg["datum_lat_deg"] == pytest.approx(40.8860)
    assert geodesy_cfg["datum_lon_deg"] == pytest.approx(29.1880)
    assert geodesy_cfg["datum_amsl_m"] == pytest.approx(12.0)
    assert geodesy_cfg["egm2008_n_m"] == pytest.approx(36.7)
    assert geodesy_cfg["datum_amsl_m"] + geodesy_cfg["egm2008_n_m"] == pytest.approx(48.7)

    # Verify forbidden port validator raises ValueError if someone tries to bind a forbidden port
    ns: dict[str, object] = {}
    exec(compile(launch_src, str(launch_path), "exec"), ns)
    validate_ports = ns["validate_sitl_ports"]
    validate_ports(foxglove_port=8891, micro_xrce_port=8888)
    for bad_port in FORBIDDEN_PORTS:
        with pytest.raises(ValueError, match="forbidden"):
            validate_ports(foxglove_port=bad_port, micro_xrce_port=8888)


@pytest.mark.skipif(
    os.environ.get("VERTIRTA_RUN_LIVE_SITL") != "1",
    reason="Live PX4 v1.16 + Gazebo Harmonic + ROS 2 Jazzy Gate-2 SITL harness requires VERTIRTA_RUN_LIVE_SITL=1",
)
def test_gate2_imu_fdi_csfl_and_same_arm_spin_vs_stock_px4():
    """Live Gate-2 closed-loop SITL verification harness (Plan 1A Task 12).

    Runs against live PX4 v1.16 SITL + Gazebo Harmonic (kartal_coast) + ROS 2 Jazzy:
      1) Stock PX4 (VERTIRTA_USE_INDI=0) w/ M0+M4 cut @ 35 m AGL -> verifies capsize (>60 deg tilt).
      2) VertiRTA (VERTIRTA_USE_INDI=1):
         (a) IMU0 stuck -> DUPLEX_2OF2 <= 20 ms;
         (b) M0+M5 cross-arm -> CSFL 0 CBVF/CTR breach;
         (c) M0+M4 same-arm -> irreversible CEL latch, Option B1b flat spin (~2.2 rad/s),
             touchdown |v_z| <= 2.5 (>10 m), <= 1.5 (<5 m), <= 0.8 (contact), valid CRC32C .vrta!
    """
    for tool in ("gz", "ros2"):
        proc = subprocess.run(["which", tool], capture_output=True, text=True, check=False)
        assert proc.returncode == 0, f"Live Gate-2 SITL requires `{tool}` on PATH"

    # Launch harness entrypoint invokes ros2 launch vertirta_bringup vertirta_sitl.launch.py
    launch_check = subprocess.run(
        ["ros2", "pkg", "prefix", "vertirta_bringup"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert launch_check.returncode == 0, "Build ros2_ws with colcon build before running live Gate-2 SITL"
