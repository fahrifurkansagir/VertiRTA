# SPDX-License-Identifier: Apache-2.0
"""Contract tests for .github/workflows/m2_uspace_and_physics.yml (Plan 5B Task 7)."""

from __future__ import annotations

from pathlib import Path
import re
import yaml

REPO = Path(__file__).resolve().parents[2]
EXPECTED_LIBSODIUM_1022_SHA256 = (
    "adbdd8f16149e81ac6078a03aca6fc03b592b89ef7b5ed83841c086191be3349"
)


def test_m2_workflow_runs_cross_isa_pins_libsodium_ge_1021_and_enforces_zero_gpl() -> None:
    wf_path = REPO / ".github" / "workflows" / "m2_uspace_and_physics.yml"
    assert wf_path.is_file(), f"Missing workflow: {wf_path}"
    raw = wf_path.read_bytes()
    assert b"\r" not in raw, "CRLF detected in m2_uspace_and_physics.yml"
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM detected in m2_uspace_and_physics.yml"

    text = raw.decode("utf-8")
    wf = yaml.safe_load(text)
    jobs = wf["jobs"]
    assert {
        "m2_cpp_matrix",
        "m2_sanitizers_and_gpl_gate",
        "m2_uspace_node_and_vertigcs",
    }.issubset(jobs.keys())

    # Pin real immutable libsodium-1.0.22.tar.gz SHA-256 (verified live against download.libsodium.org & GitHub)
    assert wf["env"]["LIBSODIUM_SHA256"] == EXPECTED_LIBSODIUM_1022_SHA256
    assert re.fullmatch(r"[0-9a-f]{64}", wf["env"]["LIBSODIUM_SHA256"]) is not None

    for required in (
        "ubuntu-24.04",
        "ubuntu-24.04-arm",
        "download.libsodium.org/libsodium/releases/libsodium-1.0.22.tar.gz",
        "VERTIRTA_REQUIRE_LIBSODIUM=ON",
        "VERTIRTA_REQUIRE_Z3=ON",
        "VERTIRTA_EMIT_FLEET_SMT2_BIN",
        "VERTIRTA_VERIFY_VRTAWND1_BIN",
        "libz3-dev",
        "check_zero_gpl_symbols.py",
        "--source-roots",
        "test_fleet_z3_smt2_crosscheck.py",
        "test_m2_uspace_node_contracts.py",
        "test_m2_workflow_contracts.py",
        "test_zero_gpl_gate.py",
        "libvertirta_flight_core.wasm",
        "check_arch_isolation.py",
        "check_public_hygiene.py",
        "28886",
    ):
        assert required in text, f"Missing required M2 CI lock: {required!r}"

    # Ban Ubuntu 24.04 vulnerable apt libsodium 1.0.18, moving -stable tarballs, and root-owned /opt cache paths
    assert "apt-get install -y libsodium-dev" not in text
    assert "libsodium-dev" not in text
    assert "-stable.tar.gz" not in text
    assert "/opt/vertirta-libsodium" not in text
    assert "sudo make install" not in text
    assert "~/.local/vertirta-libsodium-1.0.22" in text
    assert "openfoam" not in text.lower() or "tools/cfd_offline_gpl/**" in text
    # Never install OpenFOAM in GitHub Actions CI
    assert "apt-get install" not in text or "openfoam" not in text.split("apt-get install")[1].splitlines()[0].lower()


def test_root_cmakelists_wires_all_four_m2_libraries_and_tests_under_companion_guard() -> None:
    cmake_path = REPO / "CMakeLists.txt"
    raw = cmake_path.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8")

    for subdir in (
        "core/airframes/lift_cruise",
        "core/wind_acoustics",
        "core/uspace_fleet",
        "core/fleet_z3",
        "tests/airframes",
        "tests/wind_acoustics",
        "tests/uspace_fleet",
        "tests/fleet_z3",
    ):
        assert subdir in text, f"Root CMakeLists.txt missing M2 subdir: {subdir}"
