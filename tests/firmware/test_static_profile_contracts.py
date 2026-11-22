# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Plan 4A Task 6: MISRA-aligned C++17 static profile & 5-class gap register."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


def test_misra_aligned_profile_llvm22_cppcheck_and_5class_gap_register() -> None:
    root = Path("verification/formal/misra_profile")

    tidy_path = root / ".clang-tidy"
    assert tidy_path.is_file(), "Missing .clang-tidy profile"
    tidy_bytes = tidy_path.read_bytes()
    assert b"\r" not in tidy_bytes, ".clang-tidy must use pure LF"
    tidy = yaml.safe_load(tidy_bytes.decode("utf-8"))
    checks = tidy["Checks"] if isinstance(tidy["Checks"], str) else ",".join(tidy["Checks"])
    assert "clang-analyzer-*" in checks, "LLVM 22 removed clang-analyzer-* from defaults; must opt in explicitly"
    assert "bugprone-*" in checks
    assert "cert-*" in checks
    assert "hicpp-*" not in checks, "LLVM 22 2026 RFC deprecated hicpp-* module"

    supp_path = root / "cppcheck_suppressions.txt"
    assert supp_path.is_file(), "Missing cppcheck_suppressions.txt"
    assert b"\r" not in supp_path.read_bytes(), "cppcheck_suppressions.txt must use pure LF"

    sh_path = root / "run_static_profile.sh"
    assert sh_path.is_file(), "Missing run_static_profile.sh"
    sh_bytes = sh_path.read_bytes()
    assert b"\r" not in sh_bytes, "run_static_profile.sh must use pure LF"
    sh = sh_bytes.decode("utf-8")
    assert "--platform=unix32" in sh and "--funsigned-char" in sh
    assert "unix32-unsigned" not in sh, "Cppcheck 2.19+ removed unix32-unsigned"
    assert "--strict-tools" in sh and "--check-tools" in sh
    assert "check_bare_metal_size.py" in sh and "check_no_heap_symbols.py" in sh
    assert "check_stack_usage.py" in sh and "check_arch_isolation.py" in sh
    assert subprocess.run(["bash", "-n", sh_path.as_posix()], check=False).returncode == 0

    gap_path = root / "MISRA_GAP_REGISTER.md"
    assert gap_path.is_file(), "Missing MISRA_GAP_REGISTER.md"
    gap_bytes = gap_path.read_bytes()
    assert b"\r" not in gap_bytes, "MISRA_GAP_REGISTER.md must use pure LF"
    gap = gap_bytes.decode("utf-8")
    for token in (
        "Dir-0.3.1",
        "Dir-0.3.2",
        "Rule-4.1.3",
        "Rule-6.2.1",
        "Rule-7.0.5",
        "Rule-8.2.10",
        "Rule-11.6.2",
        "Compliance:2020",
        "NOT full MISRA",
    ):
        assert token in gap, f"Missing mandatory gap-register token '{token}'"
