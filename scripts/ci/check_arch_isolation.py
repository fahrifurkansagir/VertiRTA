"""GSN Sn1 Static Isolation Gate: Guard core/flight_core & core/rta_companion against AI/ROS/Python/PROJ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


FORBIDDEN_INCLUDE_RE = re.compile(
    r'^\s*#\s*include\s*[<"]([^>"]*(?:'
    r'llama|z3|onnxruntime|NvInfer|tensorrt|torch|tensorflow|opencv|'
    r'Python\.h|pybind11|rclcpp|rosidl|proj\.h|proj_api\.h|gdal'
    r')[^>"]*)[>"]',
    re.IGNORECASE,
)

FORBIDDEN_CMAKE_RE = re.compile(
    r"\b(rclcpp|rosidl|onnxruntime|tensorrt|llama|z3|PROJ|Python3?|pybind11)\b"
)

SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx"})


def scan_forbidden_isolation(
    repo_root: Path,
    scan_dirs: tuple[str, ...] = ("core/flight_core", "core/rta_companion"),
) -> dict[str, Any]:
    """Scan deterministic C++ cores for forbidden AI/LLM/Python/ROS/PROJ dependencies."""
    violations: list[dict[str, Any]] = []
    files_scanned = 0

    for rel_dir in scan_dirs:
        base = repo_root / rel_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(repo_root).as_posix()
            if path.suffix.lower() in SOURCE_SUFFIXES:
                files_scanned += 1
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                ):
                    m = FORBIDDEN_INCLUDE_RE.search(line)
                    if m:
                        violations.append(
                            {
                                "file": rel_path,
                                "line": lineno,
                                "kind": "FORBIDDEN_INCLUDE",
                                "match": m.group(1),
                            }
                        )
            elif path.name == "CMakeLists.txt":
                files_scanned += 1
                for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
                ):
                    stripped = line.split("#", 1)[0]
                    m = FORBIDDEN_CMAKE_RE.search(stripped)
                    if m:
                        violations.append(
                            {
                                "file": rel_path,
                                "line": lineno,
                                "kind": "FORBIDDEN_CMAKE_LINK",
                                "match": m.group(1),
                            }
                        )

    report: dict[str, Any] = {
        "schema_version": "vertirta.arch_isolation/v1.0.0",
        "scan_dirs": list(scan_dirs),
        "files_scanned": files_scanned,
        "violations": violations,
        "passed": len(violations) == 0,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify ASTM F3269 core isolation from AI/ROS/Python/PROJ.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = scan_forbidden_isolation(args.repo_root)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
