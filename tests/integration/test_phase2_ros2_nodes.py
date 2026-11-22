"""Contract & hygiene tests for Phase-2 ROS 2 nodes (Plan 2 Task 8 & Plan 3 Task 6)."""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROS2_SRC = REPO_ROOT / "ros2_ws" / "src"
MISSION_DIR = ROS2_SRC / "vertirta_mission_node"
PERCEPTION_DIR = ROS2_SRC / "vertirta_perception_node"
BENCH_SCRIPT = REPO_ROOT / "scripts" / "benchmark_ort_gtx1650s.sh"

FORBIDDEN_PORTS = {4173, 4174, 8765, 9222, 9223}


def test_phase2_ros2_packages_have_valid_package_xml_and_cmake():
    for pkg_name, pkg_dir, req_deps in (
        (
            "vertirta_mission_node",
            MISSION_DIR,
            {"rclcpp", "rclcpp_action", "std_msgs", "px4_msgs", "vertirta_msgs"},
        ),
        (
            "vertirta_perception_node",
            PERCEPTION_DIR,
            {"rclcpp", "geometry_msgs", "sensor_msgs", "visualization_msgs", "px4_msgs", "vertirta_msgs"},
        ),
    ):
        pkg_xml = pkg_dir / "package.xml"
        cmake = pkg_dir / "CMakeLists.txt"
        assert pkg_xml.exists(), f"Missing {pkg_xml}"
        assert cmake.exists(), f"Missing {cmake}"

        root = ET.parse(pkg_xml).getroot()
        assert root.attrib.get("format") == "3"
        assert root.findtext("name") == pkg_name
        assert root.findtext("license") == "Apache-2.0"
        found_deps = {el.text.strip() for el in root.findall("depend") if el.text}
        missing = req_deps - found_deps
        assert not missing, f"{pkg_name}/package.xml missing depends: {missing}"

        cmake_txt = cmake.read_text(encoding="utf-8")
        assert "ament_cmake" in cmake_txt
        assert "ament_package" in cmake_txt


def test_mission_node_locks_isolation_topics_action_and_z3_gate():
    core_hpp = (MISSION_DIR / "include" / "vertirta_mission_node" / "mission_core.hpp").read_text(
        encoding="utf-8"
    )
    node_cpp = (MISSION_DIR / "src" / "mission_node.cpp").read_text(encoding="utf-8")
    combined = core_hpp + "\n" + node_cpp

    for token in (
        "VERTIRTA_WITH_ROS2",
        "std::jthread",
        "/vertirta/rta_state_snapshot",
        "/fmu/out/vertirta_capability_envelope",
        "/vertirta/execute_contingency",
        "/vertirta/copilot_utterance",
        "/vertirta/copilot_response",
        "/vertirta/stl_robustness",
        "verify_mission_qflra",
        "resolve_unsat_and_repair",
        "evaluate_conformal_gate",
        "StlMonitorEngine",
        "z3_verified",
        "15.0",
    ):
        assert token in combined, f"Missing required mission node contract token: {token!r}"


def test_perception_node_locks_topics_config_staleness_and_22kwh_vehicle_contract():
    hpp = (PERCEPTION_DIR / "include" / "vertirta_perception_node" / "perception_node.hpp").read_text(
        encoding="utf-8"
    )
    cpp = (PERCEPTION_DIR / "src" / "perception_node.cpp").read_text(encoding="utf-8")
    cfg_path = PERCEPTION_DIR / "config" / "slz_perception.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    combined = hpp + "\n" + cpp

    for token in (
        "VERTIRTA_WITH_ROS2",
        "SlzOrtRunner",
        "SlzBeliefGrid",
        "VisualSlzPriorProvider",
        "SlzReasonBits",
        "/vertirta/slz_priors",
        "/vertirta/slz/gimbal_cmd",
        "/fmu/out/vertirta_capability_envelope",
        "InjectMockFrame",
        "CheckStaleness",
        "LatestSnapshot",
    ):
        assert token in combined, f"Missing required perception node contract token: {token!r}"

    slz_cfg = cfg["vertirta_perception_node"]["ros__parameters"]
    assert slz_cfg["r_slz_min_m"] == pytest.approx(5.0)
    assert slz_cfg["theta_commit_deg"] == pytest.approx(8.0)
    assert slz_cfg["sigma_z_commit_m"] == pytest.approx(0.15)
    assert slz_cfg["step_max_m"] == pytest.approx(0.25)
    assert slz_cfg["staleness_warn_ms"] == 250
    assert slz_cfg["pack_energy_kwh"] == pytest.approx(22.0)
    assert slz_cfg["rotor_radius_m"] == pytest.approx(0.75)
    assert slz_cfg["foxglove_ws_port"] == 8891
    assert slz_cfg["foxglove_ws_port"] not in FORBIDDEN_PORTS


def test_benchmark_ort_gtx1650s_script_syntax_ports_and_lf():
    raw = BENCH_SCRIPT.read_bytes()
    assert b"\r" not in raw, "benchmark_ort_gtx1650s.sh must use pure LF line endings"
    text = raw.decode("utf-8")

    for token in ("GTX 1650 SUPER", "TU116", "FP32", "FP16", "300", "8891", "QT_QPA_PLATFORM=xcb"):
        assert token in text, f"Missing required benchmark script lock: {token!r}"
    for bad in ("4173", "4174", "8765", "9222", "9223"):
        # Forbidden ports may appear ONLY inside the forbidden-port rejection guard, never as defaults
        assert f"PORT={bad}" not in text and f"port:={bad}" not in text

    bash = shutil.which("bash")
    if bash is not None:
        proc = subprocess.run(
            [bash, "-n", BENCH_SCRIPT.relative_to(REPO_ROOT).as_posix()],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"bash -n failed: {proc.stderr}"


def test_all_phase2_ros2_node_files_are_pure_lf():
    bad_files: list[str] = []
    for root_dir in (MISSION_DIR, PERCEPTION_DIR):
        assert root_dir.exists()
        for p in root_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in str(p):
                if b"\r" in p.read_bytes():
                    bad_files.append(str(p.relative_to(REPO_ROOT)))
    assert not bad_files, f"Files with CRLF detected: {bad_files}"
