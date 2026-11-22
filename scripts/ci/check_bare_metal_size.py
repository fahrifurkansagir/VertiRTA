"""Tier-1 CI Gate: Bare-metal ARM Cortex-M7 Flash (<256 KB) & RAM (<128 KB) footprint check."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def evaluate_berkeley_size(
    size_stdout: str,
    max_flash_bytes: int = 262144,
    max_ram_bytes: int = 131072,
) -> dict[str, Any]:
    """Sum Berkeley `size` output across ELF or static archive members and enforce budgets."""
    total_text = 0
    total_data = 0
    total_bss = 0
    members_parsed = 0

    for raw_line in size_stdout.splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 4:
            continue
        if parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
            total_text += int(parts[0])
            total_data += int(parts[1])
            total_bss += int(parts[2])
            members_parsed += 1

    if members_parsed == 0:
        raise ValueError("Could not parse any numeric rows from Berkeley size output")

    flash_bytes = total_text + total_data
    ram_bytes = total_data + total_bss
    passed = (flash_bytes < max_flash_bytes) and (ram_bytes < max_ram_bytes)

    report: dict[str, Any] = {
        "schema_version": "vertirta.tier1_bare_metal_size/v1.0.0",
        "members_parsed": members_parsed,
        "text_bytes": total_text,
        "data_bytes": total_data,
        "bss_bytes": total_bss,
        "flash_bytes": flash_bytes,
        "ram_bytes": ram_bytes,
        "max_flash_bytes": max_flash_bytes,
        "max_ram_bytes": max_ram_bytes,
        "passed": passed,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce Cortex-M7 Flash/RAM budget via arm-none-eabi-size.")
    parser.add_argument("--target", type=Path, required=True, help="Path to .a archive or .elf binary")
    parser.add_argument("--size-bin", default="arm-none-eabi-size")
    parser.add_argument("--max-flash-bytes", type=int, default=262144)
    parser.add_argument("--max-ram-bytes", type=int, default=131072)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    proc = subprocess.run(
        [args.size_bin, "--format=berkeley", str(args.target)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = evaluate_berkeley_size(
        proc.stdout,
        max_flash_bytes=args.max_flash_bytes,
        max_ram_bytes=args.max_ram_bytes,
    )
    report["target"] = str(args.target)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
