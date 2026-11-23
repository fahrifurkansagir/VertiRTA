"""Cross-verification of C++20 Two-Tier Fleet Z3 SMT-LIB2 dumps (QF_RDL & QF_LRA)
against real Microsoft Python z3-solver (mirrors tests/copilot/test_z3_smt2_crosscheck.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import pytest
import z3

REPO = Path(__file__).resolve().parents[2]


def _find_emit_fleet_smt2_bin() -> tuple[list[str], bool]:
    env_bin = os.environ.get("VERTIRTA_EMIT_FLEET_SMT2_BIN")
    if env_bin:
        if os.name == "nt" and env_bin.startswith("/"):
            return (["wsl", "-d", "Ubuntu", "--", env_bin], True)
        return ([env_bin], True)
    for preset in ("dev-linux-x64", "release-linux-x64"):
        cand = REPO / "build" / preset / "tests" / "fleet_z3" / "emit_fleet_smt2_cases"
        if cand.exists():
            if os.name == "nt":
                wsl_path = "/mnt/" + cand.drive[0].lower() + cand.as_posix()[2:]
                return (["wsl", "-d", "Ubuntu", "--", wsl_path], True)
            return ([str(cand)], True)
    return ([], False)


def test_cpp_emitted_fleet_smt2_matches_real_microsoft_z3() -> None:
    cmd, found = _find_emit_fleet_smt2_bin()
    if not found:
        pytest.skip("emit_fleet_smt2_cases not built; set VERTIRTA_EMIT_FLEET_SMT2_BIN to run")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert len(rows) >= 6  # Covers QF_RDL SAT/UNSAT + QF_LRA Art. 10(8)/(9)/(10) + <60 m FATO mutex + LTFJ wake

    logics_seen: set[str] = set()
    sat_seen = 0
    unsat_seen = 0

    for row in rows:
        logic = row["logic"]
        smt2 = row["smt2"]
        assert logic in ("QF_RDL", "QF_LRA")
        assert f"(set-logic {logic})" in smt2
        logics_seen.add(logic)

        # Verify raw SMT-LIB2 parses cleanly via BOTH z3.parse_smt2_string & solver.from_string
        solver = z3.SolverFor(logic)
        solver.set("unsat_core", True)
        solver.set("core.minimize", True)
        m = re.search(r"\(check-sat-assuming \(([^)]+)\)\)", smt2)
        if m is not None:
            trackers = m.group(1).split()
            body = smt2[: m.start()]
            z3.parse_smt2_string(body)
            solver.from_string(body)
            res = solver.check([z3.Bool(t) for t in trackers])
        else:
            body = smt2.split("(check-sat)")[0]
            z3.parse_smt2_string(body)
            solver.from_string(body)
            res = solver.check()

        assert res in (z3.sat, z3.unsat), f"{row['name']} unexpected Z3 result: {res}"
        assert (res == z3.sat) == bool(row["cpp_is_sat"]), f"{row['name']} SAT/UNSAT mismatch"
        if res == z3.sat:
            sat_seen += 1
        else:
            unsat_seen += 1
            assert row.get("cpp_unsat_core"), f"{row['name']} UNSAT must report non-empty unsat_core"
            z3_core = {str(s) for s in solver.unsat_core()}
            assert z3_core.issubset(set(row["cpp_unsat_core"])), (
                f"{row['name']}: Z3 core {z3_core} not subset of C++ core {row['cpp_unsat_core']}"
            )

    assert logics_seen == {"QF_RDL", "QF_LRA"}
    assert sat_seen >= 4
    assert unsat_seen >= 4
