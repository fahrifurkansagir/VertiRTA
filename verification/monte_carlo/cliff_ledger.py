"""Zero-cherry-picking failure-cliff ledger linking every failed Atlas run to its .vrta trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CLIFF_LEDGER_SCHEMA_VERSION = "vertirta.cliff_ledger/v1.0.0"

ALLOWED_ROOT_CAUSE_CLASSES = frozenset(
    {
        "CONTROL_AUTHORITY_CLIFF",
        "ENERGY_RESERVE_EXHAUSTION",
        "CTR_BARRIER_VIOLATION",
        "TMR_ISOLATION_TIMEOUT",
        "QP_INFEASIBLE",
        "SIM_TIMEOUT",
    }
)

REQUIRED_ENTRY_FIELDS = (
    "run_id",
    "stratum",
    "outcome_class",
    "seed",
    "initial_state",
    "fault_topology",
    "violated_invariant",
    "root_cause_class",
    "vrta_path",
)


def validate_cliff_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Validate failure_cliff_ledger.json schema and .vrta replay traceability."""
    if ledger.get("schema_version") != CLIFF_LEDGER_SCHEMA_VERSION:
        raise ValueError(f"Expected schema_version '{CLIFF_LEDGER_SCHEMA_VERSION}'")

    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries must be a list")
    if ledger.get("total_failures") != len(entries):
        raise ValueError("total_failures does not match len(entries)")

    for idx, entry in enumerate(entries):
        for field in REQUIRED_ENTRY_FIELDS:
            if field not in entry or entry[field] is None:
                raise ValueError(f"Ledger entry {idx} missing required field '{field}'")
        if not isinstance(entry["initial_state"], dict) or not entry["initial_state"]:
            raise ValueError(f"Ledger entry {idx} initial_state must be a non-empty dict")
        if entry["root_cause_class"] not in ALLOWED_ROOT_CAUSE_CLASSES:
            raise ValueError(
                f"Ledger entry {idx} invalid root_cause_class '{entry['root_cause_class']}'"
            )
        vrta_path = str(entry["vrta_path"])
        if not vrta_path.endswith(".vrta"):
            raise ValueError(
                f"Ledger entry {idx} vrta_path '{vrta_path}' must link a replayable .vrta trace"
            )

    return ledger


def write_failure_cliff_ledger(
    results: list[dict[str, Any]],
    out_path: Path,
) -> dict[str, Any]:
    """Record every failed Atlas run with seed, state, violated invariant, and .vrta trace."""
    entries: list[dict[str, Any]] = []
    for row in results:
        if bool(row.get("passed", False)):
            continue
        entries.append(
            {
                "run_id": row["run_id"],
                "stratum": row["stratum"],
                "outcome_class": row["outcome_class"],
                "seed": row["seed"],
                "initial_state": row["initial_state"],
                "fault_topology": row["fault_topology"],
                "fault_inject_time_s": row.get("fault_inject_time_s"),
                "violated_invariant": row["violated_invariant"],
                "root_cause_class": row["root_cause_class"],
                "vrta_path": row["vrta_path"],
            }
        )

    ledger: dict[str, Any] = {
        "schema_version": CLIFF_LEDGER_SCHEMA_VERSION,
        "total_runs": len(results),
        "total_failures": len(entries),
        "entries": entries,
    }
    validate_cliff_ledger(ledger)
    ledger["sha256"] = hashlib.sha256(
        json.dumps(ledger, sort_keys=True).encode("utf-8")
    ).hexdigest()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8", newline="\n")
    return ledger
