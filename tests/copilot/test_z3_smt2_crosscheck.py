"""Cross-verification of C++20 QF_LRA verifier SMT-LIB2 dumps against real Microsoft Z3,
plus unit tests for KnowNo Split Conformal Python calibration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import pytest
import z3

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.copilot.calibrate_conformal import calibrate_from_jsonl, compute_split_conformal_qhat


def _find_emit_smt2_binary() -> tuple[list[str], bool]:
    """Locate emit_smt2_cases binary from env override or standard build presets."""
    env_bin = os.environ.get("VERTIRTA_EMIT_SMT2_BIN")
    if env_bin:
        if env_bin.startswith("/"):
            return (["wsl", "-d", "Ubuntu", "--", env_bin], True)
        return ([env_bin], True)

    candidates = [
        REPO_ROOT / "build" / "dev-linux-x64" / "tests" / "copilot" / "emit_smt2_cases",
        REPO_ROOT / "build" / "release-linux-x64" / "tests" / "copilot" / "emit_smt2_cases",
    ]
    for cand in candidates:
        if cand.exists():
            if os.name == "nt":
                wsl_path = "/mnt/" + cand.drive[0].lower() + cand.as_posix()[2:]
                return (["wsl", "-d", "Ubuntu", "--", wsl_path], True)
            return ([str(cand)], True)
    return ([], False)


def _verify_smt2_with_real_z3(smt2_text: str) -> tuple[bool, list[str], list[str]]:
    """Parse SMT-LIB2 (with explicit check-sat-assuming) in real Microsoft Z3."""
    assert "(set-logic QF_LRA)" in smt2_text
    assert "(check-sat-assuming (" in smt2_text
    assert "(get-unsat-core)" in smt2_text

    match = re.search(r"\(check-sat-assuming \(([^)]+)\)\)", smt2_text)
    assert match is not None, "Missing (check-sat-assuming (...)) clause"
    trackers = match.group(1).split()
    body = smt2_text[: match.start()]

    # Verify raw SMT-LIB2 declarations & assertions parse cleanly in Z3
    solver = z3.SolverFor("QF_LRA")
    solver.set("unsat_core", True)
    solver.set("core.minimize", True)
    solver.from_string(body)

    tracker_syms = {name: z3.Bool(name) for name in trackers}
    result = solver.check([tracker_syms[name] for name in trackers])
    assert result in (z3.sat, z3.unsat), f"Unexpected Z3 status: {result}"

    is_sat = result == z3.sat
    min_core = sorted(str(sym) for sym in solver.unsat_core()) if not is_sat else []
    all_violated = sorted(
        name for name in trackers if solver.check(tracker_syms[name]) == z3.unsat
    )
    return is_sat, min_core, all_violated


def test_emitted_smt2_matches_real_microsoft_z3() -> None:
    cmd, found = _find_emit_smt2_binary()
    if not found:
        pytest.skip("emit_smt2_cases binary not built; set VERTIRTA_EMIT_SMT2_BIN to run")

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert len(rows) == 10

    for row in rows:
        z3_sat, z3_core, z3_all_violated = _verify_smt2_with_real_z3(row["smt2"])
        cpp_sat = bool(row["cpp_is_sat"])
        cpp_core = sorted(row["cpp_unsat_core"])
        assert z3_sat == cpp_sat, f"{row['name']}: Z3 sat={z3_sat} != C++ sat={cpp_sat}"
        assert z3_all_violated == cpp_core, (
            f"{row['name']}: Z3 violated={z3_all_violated} != C++ core={cpp_core}"
        )
        for lbl in z3_core:
            assert lbl in cpp_core, f"{row['name']}: Z3 min core {lbl} not in {cpp_core}"


def test_split_conformal_calibration_191_of_200() -> None:
    scores = [(i + 1) / 200.0 for i in range(200)]
    q_hat, rank_k, q_level = compute_split_conformal_qhat(scores, alpha=0.05)
    assert rank_k == 191
    assert abs(q_level - 0.955) < 1e-9
    assert abs(q_hat - 0.955) < 1e-9

    with tempfile.TemporaryDirectory() as tmp:
        cal_path = Path(tmp) / "calibration.jsonl"
        out_path = Path(tmp) / "conformal_calibration.json"
        cal_path.write_text(
            "\n".join(json.dumps({"gold_prob": 1.0 - s}) for s in scores) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        payload = calibrate_from_jsonl(cal_path, out_path, alpha=0.05, expected_n_cal=200)
        assert payload["n_cal"] == 200
        assert payload["order_statistic_k"] == 191
        assert abs(payload["q_level"] - 0.955) < 1e-6
        assert abs(payload["p_min_threshold"] - 0.045) < 1e-6
        assert out_path.exists()
