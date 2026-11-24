#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Machine-auditable bidirectional verifier for VertiRTA R01..R48 Requirements Traceability Matrix.

Checks every requirement in docs/assurance/rtm_v4_traceability.yaml against:
  1) docs/standards_registry.yaml (plus silicon errata STM32H7-ES0392-REV16),
  2) live repository source files, tests, formal/SIL harnesses, and artifact JSONs on disk,
  3) exact C++/Python symbol occurrence inside declared source files, and
  4) strict non-certification honesty locks (TESTBED_VERIFIED vs CERTIFICATION_UNDEVELOPED only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTM_PATH = REPO_ROOT / "docs" / "assurance" / "rtm_v4_traceability.yaml"
STANDARDS_REGISTRY_PATH = REPO_ROOT / "docs" / "standards_registry.yaml"

ALLOWED_STATUSES = frozenset({"TESTBED_VERIFIED", "CERTIFICATION_UNDEVELOPED"})
EXTRA_HARDWARE_STANDARDS = frozenset({"STM32H7-ES0392-REV16", "ASTM-F3548-21", "ASTM-F3411-22A"})

REQUIRED_STANDARD_COVERAGE = frozenset(
    {
        "ASTM-F3269-21",
        "SC-VTOL-02",
        "SC-VTOL-MOC-5",
        "REG-EU-2022-1645-PART-IS",
        "REG-EU-2023-203-PART-IS",
        "CIR-EU-2026-1821",
        "CDR-EU-2020-2034-ERCS",
        "EUROCAE-ED-155A",
        "SHGM-SHY-OLAY",
        "SHGM-SHT-21-EK6",
        "STM32H7-ES0392-REV16",
    }
)


def _load_valid_standards(repo_root: Path) -> set[str]:
    reg_path = repo_root / "docs" / "standards_registry.yaml"
    reg = yaml.safe_load(reg_path.read_text(encoding="utf-8"))
    ids = {str(s["id"]) for s in reg.get("standards", [])}
    ids.update(EXTRA_HARDWARE_STANDARDS)
    return ids


def audit_rtm_dict(rtm: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    violations: list[str] = []
    valid_standards = _load_valid_standards(repo_root)

    if rtm.get("compliance_claim") != "NONE_RESEARCH_TESTBED_ONLY":
        violations.append(
            f"Top-level compliance_claim must be 'NONE_RESEARCH_TESTBED_ONLY', got {rtm.get('compliance_claim')!r}"
        )

    reqs = rtm.get("requirements")
    if not isinstance(reqs, list) or len(reqs) != 48:
        violations.append(f"Expected exactly 48 requirements (R01..R48), found {len(reqs) if isinstance(reqs, list) else 'non-list'}")
        return {
            "schema_version": "vertirta.rtm_audit/v1.0.0",
            "passed": False,
            "requirements_audited": 0,
            "paths_verified": 0,
            "symbols_verified": 0,
            "status_breakdown": {},
            "covered_standards": [],
            "violations": violations,
            "sha256": "",
        }

    expected_ids = [f"R{i:02d}" for i in range(1, 49)]
    seen_ids: list[str] = []
    status_counts: dict[str, int] = {"TESTBED_VERIFIED": 0, "CERTIFICATION_UNDEVELOPED": 0}
    covered_standards: set[str] = set()
    paths_verified = 0
    symbols_verified = 0
    file_text_cache: dict[Path, str] = {}

    def _read_cached(p: Path) -> str:
        if p not in file_text_cache:
            file_text_cache[p] = p.read_text(encoding="utf-8", errors="replace")
        return file_text_cache[p]

    for idx, req in enumerate(reqs):
        rid = str(req.get("id", f"idx_{idx}"))
        seen_ids.append(rid)

        std_id = str(req.get("standard_id", ""))
        if std_id not in valid_standards:
            violations.append(f"{rid}: unknown standard_id '{std_id}'")
        else:
            covered_standards.add(std_id)

        status = str(req.get("verification_status", ""))
        if status not in ALLOWED_STATUSES:
            violations.append(
                f"{rid}: invalid verification_status '{status}' (allowed: {sorted(ALLOWED_STATUSES)})"
            )
        else:
            status_counts[status] += 1

        if not str(req.get("clause", "")).strip() or not str(req.get("title", "")).strip():
            violations.append(f"{rid}: missing non-empty clause or title")
        if not str(req.get("honest_scope_note", "")).strip():
            violations.append(f"{rid}: missing honest_scope_note")

        # Verify all referenced paths exist on disk
        src_files = list(req.get("source_files") or [])
        test_files = list(req.get("verification_tests") or [])
        harness_files = list(req.get("formal_or_sil_harnesses") or [])
        artifact_files = list(req.get("evidence_artifacts") or [])

        if not src_files:
            violations.append(f"{rid}: source_files must be non-empty")
        if status == "TESTBED_VERIFIED" and not test_files and not harness_files:
            violations.append(f"{rid}: TESTBED_VERIFIED requirement must cite >= 1 test or harness")

        resolved_sources: list[Path] = []
        for field_name, path_list in (
            ("source_files", src_files),
            ("verification_tests", test_files),
            ("formal_or_sil_harnesses", harness_files),
            ("evidence_artifacts", artifact_files),
        ):
            for rel in path_list:
                full = repo_root / str(rel)
                if not full.is_file():
                    violations.append(f"{rid}: broken path in {field_name}: '{rel}'")
                else:
                    paths_verified += 1
                    if field_name == "source_files":
                        resolved_sources.append(full)

        # Verify every declared C++/Python symbol actually appears in >= 1 listed source_file
        symbols = list(req.get("cpp_or_py_symbols") or [])
        if status == "TESTBED_VERIFIED" and not symbols:
            violations.append(f"{rid}: TESTBED_VERIFIED requirement must declare >= 1 symbol")
        for sym in symbols:
            sym_s = str(sym).strip()
            pattern = re.compile(r"\b" + re.escape(sym_s) + r"\b")
            found = any(pattern.search(_read_cached(sf)) is not None for sf in resolved_sources)
            if not found:
                violations.append(
                    f"{rid}: symbol '{sym_s}' not found in declared source_files {src_files}"
                )
            else:
                symbols_verified += 1

    if seen_ids != expected_ids:
        violations.append(f"Requirement IDs must be contiguous R01..R48; got mismatch against expected order")

    missing_stds = sorted(REQUIRED_STANDARD_COVERAGE - covered_standards)
    if missing_stds:
        violations.append(f"Missing mandatory standard coverage: {missing_stds}")

    if status_counts["CERTIFICATION_UNDEVELOPED"] < 2:
        violations.append(
            "RTM must explicitly retain >= 2 CERTIFICATION_UNDEVELOPED rows (physical STM32H7 HITL & G5/G6 certification)"
        )

    payload: dict[str, Any] = {
        "schema_version": "vertirta.rtm_audit/v1.0.0",
        "passed": len(violations) == 0,
        "requirements_audited": len(reqs),
        "paths_verified": paths_verified,
        "symbols_verified": symbols_verified,
        "status_breakdown": status_counts,
        "covered_standards": sorted(covered_standards),
        "violations": violations,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["sha256"] = digest
    return payload


def verify_rtm_file(rtm_path: Path = DEFAULT_RTM_PATH, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    rtm = yaml.safe_load(rtm_path.read_text(encoding="utf-8"))
    return audit_rtm_dict(rtm, repo_root=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtm", type=Path, default=DEFAULT_RTM_PATH)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    report = verify_rtm_file(args.rtm, args.repo_root)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
