# SPDX-License-Identifier: Apache-2.0
"""Contract tests for OpenCyphal v1.0 DSDL, Nunavut C11 codegen, and o1heap isolation."""

from pathlib import Path
import shutil
import subprocess

REPO_ROOT = Path(__file__).resolve().parents[2]
CYPHAL_DIR = REPO_ROOT / "firmware" / "renode_cyphal"
FLIGHT_CORE_DIR = REPO_ROOT / "core" / "flight_core"


def test_safety_envelope_dsdl_single_frame_canfd_and_nunavut_c11_only():
    dsdl = (CYPHAL_DIR / "dsdl" / "vertirta" / "flight" / "SafetyEnvelope.1.0.dsdl").read_text(
        encoding="utf-8"
    )
    assert "1024" in dsdl
    assert "7509" in dsdl
    assert "float32[4] wrench_min_frd" in dsdl
    assert "float32[4] wrench_max_frd" in dsdl
    assert "uint32 crc32c" in dsdl
    assert "@assert _offset_ == {384}" in dsdl
    assert "@assert _offset_ <= {504}" in dsdl
    assert "@sealed" in dsdl

    codegen_path = CYPHAL_DIR / "scripts" / "codegen_dsdl.sh"
    raw_sh = codegen_path.read_bytes()
    assert b"\r" not in raw_sh, "codegen_dsdl.sh must use pure LF"
    sh = raw_sh.decode("utf-8")
    assert "--target-language c" in sh
    assert "--language-standard c11" in sh
    assert "--target-language cpp" not in sh

    if shutil.which("bash") is not None:
        # Relative POSIX path works across Linux bash, Windows Git Bash, and Windows WSL bash.exe.
        rel_script = codegen_path.relative_to(REPO_ROOT).as_posix()
        syntax = subprocess.run(
            ["bash", "-n", rel_script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert syntax.returncode == 0, f"bash -n failed: {syntax.stderr}"
        verify = subprocess.run(
            ["bash", rel_script, "--verify-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert verify.returncode == 0, f"codegen_dsdl.sh --verify-only failed: {verify.stderr}"

    codec_h = (CYPHAL_DIR / "include" / "vertirta_cyphal" / "safety_envelope_codec.h").read_text(
        encoding="utf-8"
    )
    for token in (
        "VERTIRTA_SAFETY_ENVELOPE_SUBJECT_ID 1024u",
        "VERTIRTA_HEARTBEAT_SUBJECT_ID 7509u",
        "VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES 48u",
        "VERTIRTA_CANFD_WIRE_TIME_48B_TENTHS_US 1606u",
        "VERTIRTA_CANFD_WIRE_TIME_64B_TENTHS_US 1926u",
        "VERTIRTA_FAILOVER_MODELED_TOTAL_US 13750u",
        "VERTIRTA_FAILOVER_HARD_CEILING_US 20000u",
    ):
        assert token in codec_h


def test_o1heap_arena_remains_strictly_outside_frozen_flight_core():
    o1_hdr = (CYPHAL_DIR / "include" / "vertirta_cyphal" / "o1heap_arena.hpp").read_text(
        encoding="utf-8"
    )
    assert "32768u" in o1_hdr
    assert "30095u" in o1_hdr

    for path in FLIGHT_CORE_DIR.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "o1heap" not in text.lower(), f"o1heap leaked into frozen core/flight_core: {path}"
