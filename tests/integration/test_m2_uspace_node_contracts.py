# SPDX-License-Identifier: Apache-2.0
"""Contract verification for ros2_ws/src/vertirta_uspace_node (Plan 5B Task 5)."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

REPO = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO / "ros2_ws" / "src" / "vertirta_uspace_node"
FORBIDDEN_PORTS = ("4173", "4174", "8765", "9222", "9223")


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def test_uspace_node_files_exist_pure_lf_no_bom_and_zero_forbidden_ports() -> None:
    expected = [
        PKG_ROOT / "CMakeLists.txt",
        PKG_ROOT / "package.xml",
        PKG_ROOT / "include" / "vertirta_uspace_node" / "uspace_core.hpp",
        PKG_ROOT / "src" / "uspace_node.cpp",
        PKG_ROOT / "config" / "uspace_corridor.yaml",
        PKG_ROOT / "config" / "fastdds_shm_mc.xml",
        PKG_ROOT / "config" / "fastdds_part_is_udp.xml",
        PKG_ROOT / "test" / "CMakeLists.txt",
        PKG_ROOT / "test" / "test_uspace_node.cpp",
    ]
    for path in expected:
        assert path.is_file(), f"Missing file: {path}"
        raw = path.read_bytes()
        assert b"\r" not in raw, f"CRLF detected in {path}"
        assert not raw.startswith(b"\xef\xbb\xbf"), f"UTF-8 BOM detected in {path}"
        text = raw.decode("utf-8")
        for port in FORBIDDEN_PORTS:
            assert port not in text, f"Forbidden port {port} found in {path}"


def test_package_xml_and_corridor_yaml_enforce_canonical_locks() -> None:
    pkg = ET.fromstring((PKG_ROOT / "package.xml").read_text(encoding="utf-8"))
    assert pkg.findtext("name") == "vertirta_uspace_node"
    assert pkg.findtext("license") == "Apache-2.0"

    cfg = yaml.safe_load((PKG_ROOT / "config" / "uspace_corridor.yaml").read_text(encoding="utf-8"))
    params = cfg["vertirta_uspace_node"]["ros__parameters"]
    assert params["datum"]["lat_deg"] == 40.8860
    assert params["datum"]["lon_deg"] == 29.1880
    assert params["datum"]["egm2008_n0_m"] == 36.7
    assert params["airspace"]["eastern_geofence_lon_deg"] == 29.1895
    assert params["airspace"]["cruise_ceiling_agl_m"] == 120.0
    assert len(params["vertiports"]) == 4
    for vp in params["vertiports"]:
        assert vp["lon_deg"] < 29.1895
        assert vp["fato_emg"] is True
        assert vp["chilled_stands_15c"] == 4
    assert params["astm_f3548_21"]["priority_0_cel_astm"] == 100
    assert params["astm_f3411_22a_drip"]["message_pack_bytes"] == 228
    assert params["part_is_security"]["min_fastdds_version_v3"] == "3.4.1"
    assert params["part_is_security"]["min_libsodium_version"] == "1.0.21"


def test_fastdds_dual_profiles_separate_shm_mc_from_udp_part_is_security() -> None:
    shm_text = (PKG_ROOT / "config" / "fastdds_shm_mc.xml").read_text(encoding="utf-8")
    udp_text = (PKG_ROOT / "config" / "fastdds_part_is_udp.xml").read_text(encoding="utf-8")

    shm_root = ET.fromstring(shm_text)
    udp_root = ET.fromstring(udp_text)

    shm_types = [el.text for el in shm_root.iter() if _strip_ns(el.tag) == "type"]
    udp_types = [el.text for el in udp_root.iter() if _strip_ns(el.tag) == "type"]
    assert "SHM" in shm_types
    assert "UDPv4" in udp_types
    assert "SHM" not in udp_types, "Fast DDS Security profile MUST NOT enable SHM transport"

    # Verify builtin transports disabled in UDP Part-IS profile (preventing implicit SHM)
    builtin_flags = [el.text for el in udp_root.iter() if _strip_ns(el.tag) == "useBuiltinTransports"]
    assert builtin_flags == ["false"]

    # Verify all 4 OMG DDS Security plugins configured in Part-IS profile
    prop_values = {el.text for el in udp_root.iter() if _strip_ns(el.tag) == "value"}
    for required_plugin in (
        "builtin.PKI-DH",
        "builtin.Access-Permissions",
        "builtin.AES-GCM-GMAC",
        "builtin.DDS_LogTopic",
    ):
        assert required_plugin in prop_values, f"Missing DDS Security plugin: {required_plugin}"

    # Verify Priority-0 preemption endpoint pins BEST_EFFORT (never RELIABLE)
    p0_writers = [
        el
        for el in udp_root.iter()
        if _strip_ns(el.tag) == "data_writer"
        and el.attrib.get("profile_name") == "/vertirta/uspace/p0_preempt"
    ]
    assert len(p0_writers) == 1
    rel_kinds = [el.text for el in p0_writers[0].iter() if _strip_ns(el.tag) == "kind"]
    assert "BEST_EFFORT" in rel_kinds
    assert "RELIABLE" not in rel_kinds
    assert "3.4.1" in udp_text
