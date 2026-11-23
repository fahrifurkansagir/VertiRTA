# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the automated ldd/nm + header-scan Zero-GPL runtime CI gate."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.ci.check_zero_gpl_symbols import (
    SCHEMA_VERSION,
    audit_targets,
    find_gpl_ldd_dependencies,
    find_gpl_nm_symbols,
    scan_source_roots_for_gpl_includes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_zero_gpl_gate_catches_openfoam_and_gpl_injections_and_passes_clean_core(
    tmp_path: Path,
) -> None:
    # 1) Shared library (`ldd`) catches OpenFOAM & GPL/LGPL runtime dependencies
    assert find_gpl_ldd_dependencies("libOpenFOAM.so => /opt/openfoam2412/lib/libOpenFOAM.so\n") != []
    assert find_gpl_ldd_dependencies("libfiniteVolume.so => /opt/openfoam/libfiniteVolume.so\n") != []
    assert find_gpl_ldd_dependencies("libespeak-ng.so.1 => /usr/lib/libespeak-ng.so.1\n") != []
    assert find_gpl_ldd_dependencies("libgmp.so.10 => /usr/lib/x86_64-linux-gnu/libgmp.so.10\n") != []
    assert find_gpl_ldd_dependencies("libfftw3.so.3 => /usr/lib/libfftw3.so.3\n") != []
    assert (
        find_gpl_ldd_dependencies(
            "linux-vdso.so.1 (0x00007fff)\nlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6\n"
            "libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6\n"
        )
        == []
    )

    # 2) Symbol table (`nm`) catches Foam::, cholmod, gsl, fftw, gmp, espeak symbols
    assert find_gpl_nm_symbols("                 U _ZN4Foam6fvMeshC1Ev\n") != []
    assert find_gpl_nm_symbols("                 U cholmod_start\n") != []
    assert find_gpl_nm_symbols("                 U gsl_matrix_alloc\n") != []
    assert find_gpl_nm_symbols("                 U __gmpz_init\n") != []
    assert find_gpl_nm_symbols("0000000000000000 T vertirta_wind_query_enu\n") == []

    # 3) Clean Apache-2.0 source tree under core/ MUST have 0 GPL/OpenFOAM includes
    assert scan_source_roots_for_gpl_includes([REPO_ROOT / "core"]) == []

    # 4) Synthetic bad header injection in tmp_path MUST be caught
    bad_dir = tmp_path / "bad_core"
    bad_dir.mkdir()
    (bad_dir / "leak.hpp").write_text('#include "fvCFD.H"\n', encoding="utf-8")
    bad_hits = scan_source_roots_for_gpl_includes([bad_dir])
    assert len(bad_hits) == 1
    assert bad_hits[0]["rule"] == "OPENFOAM_FVCFD_HDR"

    # 5) End-to-end audit_targets JSON report schema & deterministic SHA-256
    out_json = tmp_path / "zero_gpl_report.json"
    rep = audit_targets([], source_roots=[REPO_ROOT / "core"], out_json=out_json)
    assert rep["schema_version"] == SCHEMA_VERSION
    assert rep["passed"] is True
    assert rep["violations"] == []
    assert len(rep["sha256"]) == 64
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["passed"] is True and loaded["sha256"] == rep["sha256"]

    # Missing target must fail closed
    rep_missing = audit_targets([tmp_path / "nonexistent.a"])
    assert rep_missing["passed"] is False
    assert rep_missing["violations"][0]["kind"] == "MISSING_TARGET"
