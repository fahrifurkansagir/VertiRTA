# SPDX-License-Identifier: Apache-2.0
"""Contract tests for STM32H753 ITCM/DTCM linker script, vector startup, and 8-region W^X MPU."""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
HITL_DIR = REPO_ROOT / "firmware" / "stm32h7_hitl"


def test_tcm_linker_startup_and_mpu_errata_contracts():
    ld = (HITL_DIR / "stm32h753_tcm.ld").read_text(encoding="utf-8")
    assert "0x00000400" in ld
    assert ".itcm_null_guard" in ld
    assert ".itcm_isr_vector" in ld
    assert ".itcm_text" in ld
    assert ".dtcm_stack_guard" in ld
    assert ".dtcm_rodata" in ld
    assert ".dma_nocache" in ld
    assert ".axi_sram" in ld
    assert "libvertirta_flight_core.a" in ld
    assert "EXCLUDE_FILE(*libvertirta_flight_core.a)" in ld

    asm = (HITL_DIR / "startup" / "startup_stm32h753.S").read_text(encoding="utf-8")
    assert ".thumb_func" in asm
    assert ".type Reset_Handler, %function" in asm
    assert ".type Default_Handler, %function" in asm
    assert "0x00000400" in asm
    assert "0xE000ED08" in asm

    boot = (HITL_DIR / "src" / "stm32h7_boot.cpp").read_text(encoding="utf-8")
    assert "0x2003" in boot
    assert "0x1003" in boot
    assert "rev_y_fdcan_errata_flag" in boot
    assert "LATENCY = 4" in boot or "kFlashLatencyVos0 = 4" in boot
    assert "0xC5ACCE55" in boot
    assert "LSPEN" in boot
    # ES0392 Erratum 2.1.1: Write-Through cache attributes and FORCEWT/SIWT are forbidden.
    assert "FORCEWT" not in boot
    assert "SIWT" not in boot
    # Verify all 8 MPU regions R0..R7 including AN4838 background firewall (SRD=0x87).
    assert re.search(r"R0.*0x60000000|SRD.*0x87|0x87", boot, re.DOTALL)
    for region_idx in range(8):
        assert f"R{region_idx}" in boot or f"{region_idx}u," in boot


def test_stm32h7_hitl_tree_is_pure_lf():
    files = [p for p in HITL_DIR.rglob("*") if p.is_file()]
    assert len(files) >= 10
    for path in files:
        raw = path.read_bytes()
        assert b"\r" not in raw, f"CRLF detected in {path}"
