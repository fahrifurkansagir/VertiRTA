# SPDX-License-Identifier: Apache-2.0
"""Live Cortex-M7 cross-compile and ELF forensic audit for firmware/stm32h7_hitl."""

from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HITL_DIR = REPO_ROOT / "firmware" / "stm32h7_hitl"
FROZEN_LIB = REPO_ROOT / "build" / "arm-cortex-m7" / "core" / "flight_core" / "libvertirta_flight_core.a"


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def _run_m7_audit_shell(script_body: str) -> subprocess.CompletedProcess[bytes]:
    payload = script_body.replace("\r\n", "\n").encode("utf-8")
    if sys.platform == "win32" and shutil.which("arm-none-eabi-g++") is None:
        if shutil.which("wsl") is None:
            pytest.skip("Neither native arm-none-eabi-g++ nor WSL2 is available")
        return subprocess.run(
            ["wsl", "--", "bash", "-s"],
            input=payload,
            capture_output=True,
            check=False,
        )
    if shutil.which("arm-none-eabi-g++") is None:
        pytest.skip("arm-none-eabi-g++ not found on PATH")
    return subprocess.run(
        ["bash", "-s"],
        input=payload,
        capture_output=True,
        check=False,
    )


def test_hitl_harness_errata_and_honest_hwm_framing():
    regs_hdr = (HITL_DIR / "include" / "vertirta_bsp" / "stm32h7_regs.hpp").read_text(encoding="utf-8")
    dwt_src = (HITL_DIR / "src" / "dwt_profiler.cpp").read_text(encoding="utf-8")
    main_src = (HITL_DIR / "src" / "main_hitl.cpp").read_text(encoding="utf-8")
    readme = (HITL_DIR / "README.md").read_text(encoding="utf-8")

    # Superscalar Cortex-M7 DWT serialization + Erratum 2.1.6 FPSCR[4:0] sticky polling.
    assert "dsb()" in dwt_src and "isb()" in dwt_src
    assert "vmrs" in dwt_src and "vmsr" in dwt_src
    assert "kFpscrExceptionMask" in dwt_src and ("0x1Fu" in regs_hdr or "0x0000001Fu" in regs_hdr)
    assert "poll_and_clear_fpscr_flags" in main_src
    assert "EMPIRICAL_HWM_NOT_WCET_PROOF" in main_src
    assert "VERTIRTA_ENABLE_TINYUSB" in main_src
    assert "NUCLEO-H743ZI2" in readme
    assert "Pixhawk 6X" in readme and "0x08000000" in readme


def test_stm32h7_hitl_elf_builds_links_and_audits_cleanly():
    assert FROZEN_LIB.is_file(), f"Frozen Cortex-M7 archive missing: {FROZEN_LIB}"

    use_wsl = sys.platform == "win32" and shutil.which("arm-none-eabi-g++") is None
    src_dir = _to_wsl_path(HITL_DIR) if use_wsl else HITL_DIR.as_posix()
    lib_path = _to_wsl_path(FROZEN_LIB) if use_wsl else FROZEN_LIB.as_posix()

    script = f"""set -euo pipefail
BUILD=/tmp/vertirta_pytest_hitl_$$
trap 'rm -rf "$BUILD"' EXIT
cmake -S "{src_dir}" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVERTIRTA_FLIGHT_CORE_LIB="{lib_path}" >/dev/null
cmake --build "$BUILD" >/dev/null
ELF="$BUILD/vertirta_stm32h7_hitl.elf"
echo "===SECTIONS==="
arm-none-eabi-readelf -S -W "$ELF"
echo "===NM==="
arm-none-eabi-nm -n "$ELF"
echo "===OBJDUMP_BAD==="
arm-none-eabi-objdump -d "$ELF" | grep -iE "e51ff004|04 f0 1f e5|bfcsel" || echo "CLEAN_VENEERS"
"""
    proc = _run_m7_audit_shell(script)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, f"HITL ELF build/audit failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    sections_part = stdout.split("===SECTIONS===")[1].split("===NM===")[0]
    nm_part = stdout.split("===NM===")[1].split("===OBJDUMP_BAD===")[0]
    objdump_part = stdout.split("===OBJDUMP_BAD===")[1]

    def parse_section(name: str) -> tuple[int, int]:
        m = re.search(rf"\s+{re.escape(name)}\s+\S+\s+([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s", sections_part)
        assert m is not None, f"Section {name} missing from readelf -S output"
        return int(m.group(1), 16), int(m.group(2), 16)

    null_vma, null_sz = parse_section(".itcm_null_guard")
    vec_vma, vec_sz = parse_section(".itcm_isr_vector")
    itcm_vma, itcm_sz = parse_section(".itcm_text")
    guard_vma, guard_sz = parse_section(".dtcm_stack_guard")
    ro_vma, ro_sz = parse_section(".dtcm_rodata")
    axi_vma, axi_sz = parse_section(".axi_sram")

    assert null_vma == 0x00000000 and null_sz == 1024
    assert vec_vma == 0x00000400 and vec_sz == 1024
    assert itcm_vma == 0x00000800 and itcm_sz > 0 and (itcm_vma + itcm_sz) <= 65536
    assert guard_vma == 0x20000000 and guard_sz == 256
    assert ro_vma == 0x20000100 and ro_sz > 0
    assert axi_vma == 0x24000000 and axi_sz == 40000

    # Zero forbidden heap symbols referenced in the linked bare-metal ELF.
    for forbidden in (" malloc", " calloc", " realloc", " free", " _sbrk", " __wrap_malloc"):
        for line in nm_part.splitlines():
            assert not line.endswith(forbidden), f"Forbidden heap symbol found: {line}"

    assert "CLEAN_VENEERS" in objdump_part
