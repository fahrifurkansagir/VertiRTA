"""Tier-1 CI Gate: Ban dynamic heap allocation symbols in libvertirta_flight_core archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


FORBIDDEN_HEAP_PATTERN = re.compile(
    r"\b("
    r"malloc|calloc|realloc|free|aligned_alloc|posix_memalign|"
    r"_Znwm|_Znam|_Znwj|_Znaj|"
    r"_ZnwmSt11align_val_t|_ZnamSt11align_val_t|"
    r"_ZnwjSt11align_val_t|_ZnajSt11align_val_t|"
    r"_ZnwmRKSt9nothrow_t|_ZnamRKSt9nothrow_t|"
    r"_ZnwjRKSt9nothrow_t|_ZnajRKSt9nothrow_t|"
    r"_ZnwmSt11align_val_tRKSt9nothrow_t|_ZnamSt11align_val_tRKSt9nothrow_t"
    r")\b"
)


def find_forbidden_heap_symbols(nm_stdout: str) -> list[str]:
    """Extract forbidden undefined dynamic-allocation symbols from `nm -u` output."""
    offenders: list[str] = []
    seen: set[str] = set()
    for raw_line in nm_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        # `nm -u` lines are typically `U <sym>` or `<addr> U <sym>`; ignore archive headers (`foo.o:`)
        if line.endswith(":"):
            continue
        if len(parts) >= 2 and parts[-2] == "U":
            sym = parts[-1]
        elif len(parts) == 1:
            sym = parts[0]
        else:
            continue
        sym_base = sym.split("@")[0]
        if FORBIDDEN_HEAP_PATTERN.fullmatch(sym_base) and sym_base not in seen:
            seen.add(sym_base)
            offenders.append(sym_base)
    return offenders


def audit_archive(
    archive_path: Path,
    nm_bin: str = "nm",
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Run `nm -u` on a static archive or ELF and fail if any heap symbol is referenced."""
    proc = subprocess.run(
        [nm_bin, "-u", str(archive_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    offenders = find_forbidden_heap_symbols(proc.stdout)
    report: dict[str, Any] = {
        "schema_version": "vertirta.tier1_noheap/v1.0.0",
        "archive": str(archive_path),
        "nm_bin": nm_bin,
        "forbidden_symbols": offenders,
        "passed": len(offenders) == 0,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify zero dynamic heap symbols via nm -u.")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--nm-bin", default="nm")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = audit_archive(args.archive, nm_bin=args.nm_bin, out_json=args.out)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
