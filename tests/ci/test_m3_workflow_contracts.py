# SPDX-License-Identifier: Apache-2.0
"""Contract tests for .github/workflows/m3_perception_voice_forensics.yml & extended Zero-GPL gate."""

from __future__ import annotations

from pathlib import Path
import re

import yaml

from scripts.ci.check_zero_gpl_symbols import (
    SCHEMA_VERSION,
    find_gpl_ldd_dependencies,
    find_gpl_nm_symbols,
    scan_source_roots_for_gpl_includes,
)

REPO = Path(__file__).resolve().parents[2]
EXPECTED_LIBSODIUM_1022_SHA256 = (
    "adbdd8f16149e81ac6078a03aca6fc03b592b89ef7b5ed83841c086191be3349"
)


def test_m3_workflow_locks_4_jobs_libsodium_1022_z3_sanitizers_zero_gpl_and_0cu_dryrun() -> None:
    wf_path = REPO / ".github" / "workflows" / "m3_perception_voice_forensics.yml"
    assert wf_path.is_file(), f"Missing workflow: {wf_path}"
    raw = wf_path.read_bytes()
    assert b"\r" not in raw, "CRLF detected in m3_perception_voice_forensics.yml"
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM detected in m3_perception_voice_forensics.yml"

    text = raw.decode("utf-8")
    wf = yaml.safe_load(text)
    jobs = wf["jobs"]
    assert {
        "m3_cpp_matrix",
        "m3_sanitizers_gpl_and_size_locks",
        "m3_python_and_dryrun_contracts",
        "m3_vertigcs_build_and_wasm_lock",
    }.issubset(jobs.keys())

    assert wf["env"]["LIBSODIUM_SHA256"] == EXPECTED_LIBSODIUM_1022_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", wf["env"]["LIBSODIUM_SHA256"]) is not None

    for required in (
        "ubuntu-24.04",
        "ubuntu-24.04-arm",
        "download.libsodium.org/libsodium/releases/libsodium-1.0.22.tar.gz",
        "~/.local/vertirta-libsodium-1.0.22",
        "VERTIRTA_REQUIRE_LIBSODIUM=ON",
        "VERTIRTA_REQUIRE_Z3=ON",
        "libz3-dev",
        "tsan_libz3.supp",
        "check_zero_gpl_symbols.py",
        "check_arch_isolation.py",
        "check_bare_metal_size.py",
        "check_public_hygiene.py",
        "28886",
        "test_forward_data_license_gate.py",
        "test_conformal_raps_calibration.py",
        "test_e5x_xsd_and_grounding.py",
        "test_m3_voice_node_contracts.py",
        "test_paper_corl_contracts.py",
        "test_g4_thunder_contracts.py",
        "test_ingest_g4_bundle.py",
        "test_m3_workflow_contracts.py",
        "tests/realworld/",
        "benchmarks/monte_carlo_dispersion/**",
        "tools/mavlink_rta_bridge/**",
        "test_monte_carlo_dispersion",
        "test_mavlink_v2_rta_bridge",
        "Setup Node.js 22",
        'node-version: "22"',
        "--with z3-solver",
        "--with pillow",
        "run_g4_m3_thunder_run.py --dry-run",
        "vite build --base /VertiRTA/",
        "maplibre-gl-worker-",
    ):
        assert required in text, f"Missing required M3 CI lock: {required!r}"

    # Ban vulnerable apt libsodium 1.0.18, root /opt cache, and live Colab provisioning in CI
    assert "libsodium-dev" not in text
    assert "-stable.tar.gz" not in text
    assert "/opt/vertirta-libsodium" not in text
    assert "colab new" not in text
    assert "--live-g4" not in text


def test_root_cmakelists_wires_all_m3_libraries_and_tests_under_companion_guard() -> None:
    cmake_path = REPO / "CMakeLists.txt"
    raw = cmake_path.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")

    for subdir in (
        "core/forward_hazard",
        "sim/gazebo_plugins/mmwave_77ghz",
        "core/gnss_denied",
        "core/voice",
        "core/forensics",
        "tools/mavlink_rta_bridge",
        "benchmarks/monte_carlo_dispersion",
        "tests/forward_hazard",
        "tests/gnss_denied",
        "tests/voice",
        "tests/forensics",
    ):
        assert subdir in text, f"Root CMakeLists.txt missing M3 subdir: {subdir}"


def test_extended_zero_gpl_gate_catches_all_m3_vio_vision_and_voice_gpl_tokens(tmp_path: Path) -> None:
    assert SCHEMA_VERSION == "vertirta.zero_gpl_gate/v1.0.0"

    # Voice & VIO/vision runtime shared libs
    for ldd_line in (
        "libespeak-ng.so.1 => /usr/lib/libespeak-ng.so.1\n",
        "libpiper_phonemize.so => /usr/lib/libpiper_phonemize.so\n",
        "libpiper1_gpl.so => /usr/lib/libpiper1_gpl.so\n",
        "libov_msckf.so => /usr/lib/libov_msckf.so\n",
        "libORB_SLAM3.so => /usr/lib/libORB_SLAM3.so\n",
    ):
        assert find_gpl_ldd_dependencies(ldd_line) != [], f"Missed ldd ban: {ldd_line!r}"

    # Symbols (`nm`) covering BOTH Plan 6A VIO/vision AND Plan 6B Voice GPL bans
    for sym_line in (
        "U espeak_ng_Synthesize\n",
        "U piper_phonemize_text\n",
        "U piper1_gpl_init\n",
        "U openvins_init\n",
        "U vins_fusion_step\n",
        "U orb_slam3_track\n",
        "U hybvio_update\n",
        "U sqrtvins_propagate\n",
        "U blenderproc_render\n",
        "U dinov2_salad_embed\n",
        "U ultralytics_predict\n",
        "U sample4geo_match\n",
    ):
        assert find_gpl_nm_symbols(sym_line) != [], f"Missed nm ban: {sym_line!r}"

    # Permissive Apache-2.0 / MIT / BSD-3 symbols must remain clean
    for ok_sym in (
        "0000000000000000 T basalt_vio_step\n",
        "0000000000000000 T whisper_full\n",
        "0000000000000000 T SherpaOnnxCreateOfflineTts\n",
        "0000000000000000 T xfeat_match\n",
        "0000000000000000 T megaloc_retrieve\n",
    ):
        assert find_gpl_nm_symbols(ok_sym) == [], f"False positive on permissive symbol: {ok_sym!r}"

    # Synthetic bad include injection caught across both vision and voice
    bad = tmp_path / "bad_m3.cpp"
    bad.write_text('#include <espeak-ng/speak_lib.h>\n#include "openvins/state.h"\n', encoding="utf-8")
    hits = scan_source_roots_for_gpl_includes([bad])
    assert len(hits) == 2

    # All M3 C++ source roots MUST pass with 0 violations
    m3_roots = [
        REPO / "core" / "forward_hazard",
        REPO / "core" / "gnss_denied",
        REPO / "sim" / "gazebo_plugins" / "mmwave_77ghz",
        REPO / "core" / "voice",
        REPO / "core" / "forensics",
        REPO / "benchmarks" / "monte_carlo_dispersion",
        REPO / "tools" / "mavlink_rta_bridge",
        REPO / "ros2_ws" / "src" / "vertirta_forward_hazard_node",
        REPO / "ros2_ws" / "src" / "vertirta_gnss_denied_node",
        REPO / "ros2_ws" / "src" / "vertirta_voice_node",
    ]
    assert scan_source_roots_for_gpl_includes(m3_roots) == []
