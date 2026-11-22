# SPDX-License-Identifier: Apache-2.0
"""Contract tests and live dual-FCC Cortex-M7 ELF audits for Renode v1.17.0 failover suite."""

from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CYPHAL_DIR = REPO_ROOT / "firmware" / "renode_cyphal"
RENODE_DIR = CYPHAL_DIR / "renode"
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


def test_renode_dual_fcc_overlays_quantum_and_three_fault_scenarios():
    repl = (RENODE_DIR / "stm32h753_avionics.repl").read_text(encoding="utf-8")
    resc = (RENODE_DIR / "dual_fcc_failover.resc").read_text(encoding="utf-8")
    robot = (RENODE_DIR / "failover_suite.robot").read_text(encoding="utf-8")
    pwr = (RENODE_DIR / "STM32H7_PWR.cs").read_text(encoding="utf-8")
    fdcan = (CYPHAL_DIR / "src" / "fdcan_stm32h7.cpp").read_text(encoding="utf-8")

    # Renode STM32H753 fixes: renode#1009 SysTick 480 MHz, renode#649/#857 PWR D3CR 0x18 VOSRDY,
    # renode#598 zero SocketCANBridge, tight 25 us global quantum.
    assert "480000000" in repl or "480_000_000" in repl
    assert "STM32H7_PWR" in repl and "0x20030450" in repl
    assert 'SetGlobalQuantum "0.000025"' in resc
    assert "STM32H7_PWR.cs" in resc
    assert "CreateCANHub" in resc
    # Ensure SocketCANBridge is never instantiated in .resc (only mentioned in comments if at all).
    non_comment_resc = "\n".join(
        ln for ln in resc.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "SocketCANBridge" not in non_comment_resc
    assert "0x18" in pwr and "VOSRDY" in pwr

    # Robot Framework v1.17.0 strict virtual-time timeout=0.020, CANTester, 3 orthogonal faults,
    # and honest SIL-only (NOT hardware WCET) framing.
    assert "defaultPauseEmulation=true" in robot
    assert "timeout=0.020" in robot
    assert "${PRIMARY_ELF}" in robot and "${SECONDARY_ELF}" in robot
    assert "$primary_elf=@${PRIMARY_ELF}" in robot and "$secondary_elf=@${SECONDARY_ELF}" in robot
    assert "SetPendingSynchronousFault" in robot
    assert "connector Disconnect fcc_primary.fdcan1 canHub" in robot
    assert "CORRUPT_CRC_FRAME" in robot
    assert "CANTester" in robot or "Create CAN Tester" in robot
    assert "NOT hardware WCET" in robot and "Functional & Protocol SIL" in robot

    # ES0392 FDCAN errata 2.24.2-2.24.5 enforcement (both EFBI=0 and DAR=0 explicitly cleared).
    assert "rev_y_fdcan_errata_flag" in fdcan
    assert "cccr &= ~(kCccrEfbi | kCccrDar)" in fdcan
    assert "TFQM=0" in fdcan


def test_renode_cyphal_tree_is_pure_lf():
    files = [p for p in CYPHAL_DIR.rglob("*") if p.is_file()]
    assert len(files) >= 12
    for path in files:
        assert b"\r" not in path.read_bytes(), f"CRLF detected in {path}"


def test_dual_fcc_bare_metal_elfs_build_link_and_audit_cleanly():
    assert FROZEN_LIB.is_file(), f"Frozen Cortex-M7 archive missing: {FROZEN_LIB}"

    use_wsl = sys.platform == "win32" and shutil.which("arm-none-eabi-g++") is None
    src_dir = _to_wsl_path(CYPHAL_DIR) if use_wsl else CYPHAL_DIR.as_posix()
    lib_path = _to_wsl_path(FROZEN_LIB) if use_wsl else FROZEN_LIB.as_posix()

    script = f"""set -euo pipefail
BUILD=/tmp/vertirta_pytest_cyphal_$$
trap 'rm -rf "$BUILD"' EXIT
cmake -S "{src_dir}" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVERTIRTA_CYPHAL_BARE_METAL=ON \
  -DVERTIRTA_FLIGHT_CORE_LIB="{lib_path}" >/dev/null
cmake --build "$BUILD" >/dev/null
for ELF in vertirta_fcc_primary.elf vertirta_fcc_secondary.elf vertirta_cyphal_node.elf; do
  echo "===ELF:$ELF==="
  arm-none-eabi-readelf -S -W "$BUILD/$ELF"
  echo "===NM:$ELF==="
  arm-none-eabi-nm -n "$BUILD/$ELF"
  echo "===VENEER:$ELF==="
  arm-none-eabi-objdump -d "$BUILD/$ELF" | grep -iE "e51ff004|04 f0 1f e5|bfcsel" || echo "CLEAN_VENEERS"
done
"""
    proc = _run_m7_audit_shell(script)
    stdout = proc.stdout.decode("utf-8", errors="replace")
    stderr = proc.stderr.decode("utf-8", errors="replace")
    assert proc.returncode == 0, f"Dual-FCC ELF build/audit failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    for elf_name in ("vertirta_fcc_primary.elf", "vertirta_fcc_secondary.elf", "vertirta_cyphal_node.elf"):
        sec_block = stdout.split(f"===ELF:{elf_name}===")[1].split(f"===NM:{elf_name}===")[0]
        nm_block = stdout.split(f"===NM:{elf_name}===")[1].split(f"===VENEER:{elf_name}===")[0]
        ven_block = stdout.split(f"===VENEER:{elf_name}===")[1].split("===ELF:")[0]

        def parse_sec(name: str) -> tuple[int, int]:
            m = re.search(rf"\s+{re.escape(name)}\s+\S+\s+([0-9a-fA-F]+)\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)\s", sec_block)
            assert m is not None, f"{elf_name}: missing section {name}"
            return int(m.group(1), 16), int(m.group(2), 16)

        vec_vma, vec_sz = parse_sec(".itcm_isr_vector")
        itcm_vma, itcm_sz = parse_sec(".itcm_text")
        bss_vma, bss_sz = parse_sec(".bss")

        assert vec_vma == 0x00000400 and vec_sz == 1024
        assert itcm_vma == 0x00000800 and (itcm_vma + itcm_sz) <= 65536
        # DTCM .bss contains the 32 KiB o1heap arena and leaves >= 16 KB stack in 128 KB DTCM.
        assert bss_vma >= 0x20000100 and bss_sz >= 32768 and (bss_vma + bss_sz) <= (0x20020000 - 16384)

        for forbidden in (" malloc", " calloc", " realloc", " free", " _sbrk"):
            for line in nm_block.splitlines():
                assert not line.endswith(forbidden), f"{elf_name}: forbidden heap symbol {line}"
        assert "CLEAN_VENEERS" in ven_block
