# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Plan 4A Task 5: ESBMC v8.7 C++17 & C99 formal verification harnesses."""

from __future__ import annotations

import subprocess
from pathlib import Path

HARNESSES: tuple[str, ...] = (
    "verify_tmr_voter.cpp",
    "verify_indi_wls.cpp",
    "verify_motor_fdi.cpp",
    "verify_coax_envelope.cpp",
    "verify_daqp_bounds.c",
    "verify_cbvf_filter.cpp",
)


def test_esbmc_harnesses_cli_flags_and_honest_do333_framing() -> None:
    root = Path("verification/formal/esbmc")
    compat = (root / "esbmc_compat.h").read_text(encoding="utf-8")
    assert "VERTIRTA_ESBMC_ANALYSIS" in compat and "VERTIRTA_IS_ESBMC" in compat

    for harness in HARNESSES:
        hp = root / harness
        assert hp.is_file(), f"Missing ESBMC harness: {harness}"
        raw = hp.read_bytes()
        assert b"\r" not in raw, f"CRLF detected in {hp.as_posix()}"
        src = raw.decode("utf-8")
        assert "__ESBMC_assert" in src or "VERTIRTA_FORMAL_ASSERT" in src
        assert "#include" in src

    sh_path = root / "run_esbmc_suite.sh"
    assert sh_path.is_file(), "Missing run_esbmc_suite.sh"
    sh_bytes = sh_path.read_bytes()
    assert b"\r" not in sh_bytes, "run_esbmc_suite.sh must use pure LF line endings"
    sh = sh_bytes.decode("utf-8")

    for required in (
        "--32",
        "--floatbv",
        "--nan-check",
        "--overflow-check",
        "--unsigned-overflow-check",
        "--memory-leak-check",
        "--struct-fields-check",
        "--bitwuzla",
        "x86_64",
        "--unwind 5",
        "--unwind 14",
        "--unwind 10",
        "--unwind 27",
        "--unwind 12",
        "--interval-analysis",
        "--scalarize-structs",
        "--enforce-all-contracts",
        "Supplemental Engineering Verification",
        "FM.A-7",
        "DO-330",
        "VERTIRTA_ESBMC_ANALYSIS",
        "daqp_qp.cpp",
        "daqp_auxiliary_norec.c",
        "factorization.c",
    ):
        assert required in sh, f"Missing required ESBMC contract token '{required}' in run_esbmc_suite.sh"

    assert "--no-unwinding-assertions" not in sh, (
        "FORBIDDEN: --no-unwinding-assertions silently yields vacuous ESBMC proofs"
    )
    assert subprocess.run(["bash", "-n", sh_path.as_posix()], check=False).returncode == 0


def test_esbmc_harnesses_cover_required_kernel_invariants() -> None:
    root = Path("verification/formal/esbmc")

    tmr = (root / "verify_tmr_voter.cpp").read_text(encoding="utf-8")
    assert "TmrVoter" in tmr and "tmr_voter.hpp" in tmr

    indi = (root / "verify_indi_wls.cpp").read_text(encoding="utf-8")
    assert "wls_alloc_8x4_f64" in indi and "build_B" in indi and "1e-12" in indi

    fdi = (root / "verify_motor_fdi.cpp").read_text(encoding="utf-8")
    assert "MotorFdiObserver" in fdi and "SAME_ARM_ACTIVE_SPIN" in fdi

    coax = (root / "verify_coax_envelope.cpp").read_text(encoding="utf-8")
    assert "two_arm_tw_at_vterm" in coax and "TheveninBattery1RC" in coax

    daqp = (root / "verify_daqp_bounds.c").read_text(encoding="utf-8")
    assert "iter_limit" in daqp and "25" in daqp and "16" in daqp

    cbvf = (root / "verify_cbvf_filter.cpp").read_text(encoding="utf-8")
    assert "CbvfSafetyFilter" in cbvf and "MmapCbvf4DLut" in cbvf and "is_cel_latched" in cbvf
