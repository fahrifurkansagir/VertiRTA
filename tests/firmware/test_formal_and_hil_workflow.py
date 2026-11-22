# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Plan 4A Task 7: Headless GitHub Actions formal & bare-metal HIL workflow."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_formal_and_hil_workflow_pins_and_x86_64_esbmc_guard() -> None:
    wf_path = Path(".github/workflows/formal_and_hil.yml")
    assert wf_path.is_file(), "Missing .github/workflows/formal_and_hil.yml"
    raw_bytes = wf_path.read_bytes()
    assert b"\r" not in raw_bytes, "formal_and_hil.yml must use pure LF"
    raw = raw_bytes.decode("utf-8")

    wf = yaml.safe_load(raw)
    assert isinstance(wf, dict) and "jobs" in wf
    jobs = wf["jobs"]
    expected_jobs = {
        "stm32h7-bare-metal-elf",
        "renode-dual-fcc-failover",
        "esbmc-formal-x86_64",
        "misra-aligned-static-profile",
    }
    assert expected_jobs <= set(jobs), f"Missing required CI jobs: {expected_jobs - set(jobs)}"

    # ESBMC --32 MUST run strictly on ubuntu-24.04 x86_64 (ARM64 esbmc-linux-armv8 fails --32)
    assert jobs["esbmc-formal-x86_64"]["runs-on"] == "ubuntu-24.04"
    esbmc_job_text = yaml.safe_dump(jobs["esbmc-formal-x86_64"])
    assert "ubuntu-24.04-arm" not in esbmc_job_text
    assert "run_esbmc_suite.sh" in esbmc_job_text
    assert "8.7" in esbmc_job_text

    # Renode v1.17.0 matrix covers both x86_64 and ARM64 using portable .NET tarballs
    assert "ubuntu-24.04-arm" in raw
    assert "1.17.0" in raw
    assert "14.2.Rel1" in raw
    assert "linux-portable" in raw and "linux-arm64-portable" in raw
    assert "antmicro/renode-test-action" not in raw, "Use pinned portable .NET tarballs, not source builds"

    # Mandatory structural, veneer, isolation, cross-lane wiring, and public hygiene CI gates
    for required in (
        "check_bare_metal_size.py",
        "check_no_heap_symbols.py",
        "check_stack_usage.py",
        "check_arch_isolation.py",
        "check_public_hygiene.py",
        "arm-none-eabi-objdump",
        "e51ff004|04 f0 1f e5|bfcsel",
        "-DVERTIRTA_FLIGHT_CORE_LIB=",
        "--variable PRIMARY_ELF:",
        "--variable SECONDARY_ELF:",
        "run_static_profile.sh",
        "--strict-tools",
    ):
        assert required in raw, f"Missing mandatory CI gate '{required}' in formal_and_hil.yml"
    assert "VERTIRTA_FLIGHT_CORE_ARCHIVE" not in raw, "Must use Lane A1 cache var VERTIRTA_FLIGHT_CORE_LIB"
