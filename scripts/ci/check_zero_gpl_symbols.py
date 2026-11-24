#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CI Zero-GPL Runtime Symbol & Header Isolation Gate for VertiRTA.

Verifies that Apache-2.0 runtime libraries and binaries never link against or include
GPL/LGPL/OpenFOAM dependencies (`libOpenFOAM`, `libfiniteVolume`, `Foam::`, `fvCFD.H`,
`libgsl`, `libfftw3`, `libespeak`, `libreadline`, `libgmp`, `libmpfr`, `libcholmod`,
`libsuitesparse`, `libpiper_phonemize`).

Stable CLI contract (shared by Plan 5A & Plan 5B Wave 2 CI):
  python3 scripts/ci/check_zero_gpl_symbols.py \
      --targets <path1> [<path2> ...] \
      [--source-roots core/] \
      [--nm-bin nm] \
      [--ldd-bin ldd] \
      [--out report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "vertirta.zero_gpl_gate/v1.0.0"

BANNED_LDD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OPENFOAM_CORE", re.compile(r"\blibOpenFOAM\b", re.IGNORECASE)),
    ("OPENFOAM_FV", re.compile(r"\blibfiniteVolume\b", re.IGNORECASE)),
    ("OPENFOAM_TURB", re.compile(r"\blibturbulenceModels\b", re.IGNORECASE)),
    ("OPENFOAM_INCOMP", re.compile(r"\blibincompressible\w*", re.IGNORECASE)),
    ("OPENFOAM_FVMESH", re.compile(r"\blibfvMesh\b", re.IGNORECASE)),
    ("GPL_GSL", re.compile(r"\blibgsl(?:cblas)?\.so", re.IGNORECASE)),
    ("GPL_FFTW3", re.compile(r"\blibfftw3\w*\.so", re.IGNORECASE)),
    ("GPL_ESPEAK", re.compile(r"\blibespeak(?:-ng)?\.so", re.IGNORECASE)),
    ("GPL_READLINE", re.compile(r"\blibreadline\.so", re.IGNORECASE)),
    ("LGPL_GMP", re.compile(r"\blibgmp(?:xx)?\.so", re.IGNORECASE)),
    ("LGPL_MPFR", re.compile(r"\blibmpfr\.so", re.IGNORECASE)),
    ("GPL_CHOLMOD", re.compile(r"\blibcholmod\.so", re.IGNORECASE)),
    ("GPL_SUITESPARSE", re.compile(r"\blibsuitesparse\w*\.so", re.IGNORECASE)),
    ("GPL_PIPER_PHONEMIZE", re.compile(r"\blibpiper_phonemize\.so", re.IGNORECASE)),
    ("GPL_PIPER1_GPL", re.compile(r"\blibpiper1[_-]?gpl\w*\.so", re.IGNORECASE)),
    ("GPL_OPENVINS", re.compile(r"\blib(?:ov_msckf|ov_core|openvins)\w*\.so", re.IGNORECASE)),
    ("GPL_VINS_FUSION", re.compile(r"\blibvins[_-]?fusion\w*\.so", re.IGNORECASE)),
    ("GPL_ORB_SLAM", re.compile(r"\blibORB_SLAM\d*\w*\.so", re.IGNORECASE)),
    ("GPL_HYBVIO", re.compile(r"\blibhybvio\w*\.so", re.IGNORECASE)),
    ("GPL_SQRTVINS", re.compile(r"\blibsqrtvins\w*\.so", re.IGNORECASE)),
    ("GPL_BLENDERPROC", re.compile(r"\blibblenderproc\w*\.so", re.IGNORECASE)),
    ("GPL_ULTRALYTICS", re.compile(r"\blibultralytics\w*\.so", re.IGNORECASE)),
)

BANNED_NM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OPENFOAM_SYM", re.compile(r"(?:_ZN\d*Foam|_ZNK\d*Foam|\bFoam::)")),
    ("OPENFOAM_FVMESH_SYM", re.compile(r"\bfvMesh\b|\bfvCFD\b")),
    ("GPL_GSL_SYM", re.compile(r"(?<![A-Za-z0-9_])gsl_[a-z0-9_]+")),
    ("GPL_FFTW3_SYM", re.compile(r"(?<![A-Za-z0-9_])fftw[flq]?_[a-z0-9_]+")),
    ("GPL_ESPEAK_SYM", re.compile(r"(?<![A-Za-z0-9_])(?:espeak_[A-Za-z0-9_]+|espeak_ng_[A-Za-z0-9_]+|libespeak)")),
    ("GPL_READLINE_SYM", re.compile(r"(?<![A-Za-z0-9_])(?:rl_readline|add_history|readline)(?:@|$|\s)")),
    ("LGPL_GMP_SYM", re.compile(r"(?<![A-Za-z0-9_])__gmp[a-z0-9_]+")),
    ("LGPL_MPFR_SYM", re.compile(r"(?<![A-Za-z0-9_])mpfr_[a-z0-9_]+")),
    ("GPL_CHOLMOD_SYM", re.compile(r"(?<![A-Za-z0-9_])cholmod_[a-z0-9_]+")),
    ("GPL_SUITESPARSE_SYM", re.compile(r"(?<![A-Za-z0-9_])SuiteSparse_[A-Za-z0-9_]+")),
    ("GPL_PIPER_PHONEMIZE_SYM", re.compile(r"piper_phonemize")),
    ("GPL_PIPER1_GPL_SYM", re.compile(r"piper1_gpl", re.IGNORECASE)),
    ("GPL_OPENVINS_SYM", re.compile(r"(?:openvins|ov_msckf|ov_core)", re.IGNORECASE)),
    ("GPL_VINS_FUSION_SYM", re.compile(r"vins_fusion", re.IGNORECASE)),
    ("GPL_ORB_SLAM_SYM", re.compile(r"orb_slam", re.IGNORECASE)),
    ("GPL_HYBVIO_SYM", re.compile(r"hybvio", re.IGNORECASE)),
    ("GPL_SQRTVINS_SYM", re.compile(r"sqrtvins", re.IGNORECASE)),
    ("GPL_BLENDERPROC_SYM", re.compile(r"blenderproc", re.IGNORECASE)),
    ("GPL_DINOV2_SALAD_SYM", re.compile(r"dinov2_salad", re.IGNORECASE)),
    ("GPL_ULTRALYTICS_SYM", re.compile(r"ultralytics", re.IGNORECASE)),
    ("GPL_SAMPLE4GEO_SYM", re.compile(r"sample4geo", re.IGNORECASE)),
)

BANNED_INCLUDE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OPENFOAM_FVCFD_HDR", re.compile(r'#\s*include\s*[<"].*\bfvCFD\.H[>"]')),
    ("OPENFOAM_FVMESH_HDR", re.compile(r'#\s*include\s*[<"].*\bfvMesh\.H[>"]')),
    ("OPENFOAM_DIR_HDR", re.compile(r'#\s*include\s*[<"](?:OpenFOAM|finiteVolume|turbulenceModels)/')),
    ("GPL_GSL_HDR", re.compile(r'#\s*include\s*[<"]gsl/')),
    ("GPL_FFTW3_HDR", re.compile(r'#\s*include\s*[<"]fftw3\.h[>"]')),
    ("GPL_ESPEAK_HDR", re.compile(r'#\s*include\s*[<"].*(?:espeak[-_]ng|libespeak|espeak/)', re.IGNORECASE)),
    ("GPL_READLINE_HDR", re.compile(r'#\s*include\s*[<"]readline/')),
    ("LGPL_GMP_HDR", re.compile(r'#\s*include\s*[<"]gmp(?:xx)?\.h[>"]')),
    ("LGPL_MPFR_HDR", re.compile(r'#\s*include\s*[<"]mpfr\.h[>"]')),
    ("GPL_CHOLMOD_HDR", re.compile(r'#\s*include\s*[<"](?:suitesparse/)?cholmod\.h[>"]')),
    ("GPL_PIPER_PHONEMIZE_HDR", re.compile(r'#\s*include\s*[<"].*piper_phonemize', re.IGNORECASE)),
    ("GPL_PIPER1_GPL_HDR", re.compile(r'#\s*include\s*[<"].*piper1[_-]?gpl', re.IGNORECASE)),
    ("GPL_OPENVINS_HDR", re.compile(r'#\s*include\s*[<"].*(?:openvins|ov_msckf|ov_core)/', re.IGNORECASE)),
    ("GPL_VINS_FUSION_HDR", re.compile(r'#\s*include\s*[<"].*vins[_-]?fusion', re.IGNORECASE)),
    ("GPL_ORB_SLAM_HDR", re.compile(r'#\s*include\s*[<"].*orb[_-]?slam', re.IGNORECASE)),
    ("GPL_HYBVIO_HDR", re.compile(r'#\s*include\s*[<"].*hybvio', re.IGNORECASE)),
    ("GPL_SQRTVINS_HDR", re.compile(r'#\s*include\s*[<"].*sqrtvins', re.IGNORECASE)),
    ("GPL_BLENDERPROC_HDR", re.compile(r'#\s*include\s*[<"].*blenderproc', re.IGNORECASE)),
    ("GPL_DINOV2_SALAD_HDR", re.compile(r'#\s*include\s*[<"].*dinov2[_-]?salad', re.IGNORECASE)),
    ("GPL_ULTRALYTICS_HDR", re.compile(r'#\s*include\s*[<"].*ultralytics', re.IGNORECASE)),
    ("GPL_SAMPLE4GEO_HDR", re.compile(r'#\s*include\s*[<"].*sample4geo', re.IGNORECASE)),
)

SOURCE_EXTENSIONS: frozenset[str] = frozenset({".h", ".hh", ".hpp", ".hxx", ".c", ".cc", ".cpp", ".cxx"})


def find_gpl_ldd_dependencies(ldd_stdout: str) -> list[dict[str, str]]:
    """Scan `ldd` stdout lines for banned GPL/LGPL/OpenFOAM shared libraries."""
    findings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in ldd_stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for rule, pat in BANNED_LDD_PATTERNS:
            m = pat.search(line)
            if m:
                key = (rule, m.group(0))
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        {
                            "kind": "BANNED_LDD_DEPENDENCY",
                            "rule": rule,
                            "match": m.group(0),
                            "line": line,
                        }
                    )
    return findings


def find_gpl_nm_symbols(nm_stdout: str) -> list[dict[str, str]]:
    """Scan `nm` stdout lines (defined & undefined symbols) for banned GPL/OpenFOAM symbols."""
    findings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_line in nm_stdout.splitlines():
        line = raw_line.strip()
        if not line or line.endswith(":"):
            continue
        for rule, pat in BANNED_NM_PATTERNS:
            m = pat.search(line)
            if m:
                sym = line.split()[-1]
                key = (rule, sym)
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        {
                            "kind": "BANNED_NM_SYMBOL",
                            "rule": rule,
                            "match": sym,
                            "line": line,
                        }
                    )
    return findings


def scan_source_roots_for_gpl_includes(source_roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Scan C/C++ headers and sources under `source_roots` for banned GPL/OpenFOAM #includes."""
    violations: list[dict[str, Any]] = []
    for root in source_roots:
        rpath = Path(root)
        if not rpath.exists():
            continue
        files = [rpath] if rpath.is_file() else sorted(rpath.rglob("*"))
        for fpath in files:
            if not fpath.is_file() or fpath.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            text = fpath.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for rule, pat in BANNED_INCLUDE_PATTERNS:
                    m = pat.search(line)
                    if m:
                        violations.append(
                            {
                                "kind": "BANNED_SOURCE_INCLUDE",
                                "rule": rule,
                                "file": fpath.as_posix(),
                                "line": lineno,
                                "match": m.group(0).strip(),
                            }
                        )
    return violations


def _is_static_archive(path: Path) -> bool:
    if path.suffix.lower() == ".a":
        return True
    try:
        return path.read_bytes()[:8] == b"!<arch>\n"
    except OSError:
        return False


def audit_targets(
    targets: Iterable[Path],
    *,
    source_roots: Iterable[Path] = (),
    nm_bin: str = "nm",
    ldd_bin: str = "ldd",
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Audit binary targets (`nm` + `ldd` on ELFs/shared libs) and source trees for GPL leakage."""
    violations: list[dict[str, Any]] = []
    scanned_targets: list[str] = []

    for tgt in targets:
        tpath = Path(tgt)
        if not tpath.is_file():
            violations.append(
                {
                    "kind": "MISSING_TARGET",
                    "rule": "TARGET_MUST_EXIST",
                    "target": tpath.as_posix(),
                    "match": "file not found",
                }
            )
            continue

        scanned_targets.append(tpath.as_posix())
        is_archive = _is_static_archive(tpath)

        # 1) Symbol table scan via `nm` (works on static .a, .so, and ELF executables)
        nm_proc = subprocess.run(
            [nm_bin, str(tpath)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if nm_proc.returncode != 0 and not nm_proc.stdout:
            # Fallback to dynamic symbol table (-D) on stripped shared objects/ELFs
            nm_proc = subprocess.run(
                [nm_bin, "-D", str(tpath)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        for finding in find_gpl_nm_symbols(nm_proc.stdout):
            violations.append({"target": tpath.as_posix(), **finding})

        # 2) Shared library dependency scan via `ldd` on linked ELFs / shared objects (skip .a)
        if not is_archive:
            ldd_proc = subprocess.run(
                [ldd_bin, str(tpath)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            combined_ldd = (ldd_proc.stdout or "") + "\n" + (ldd_proc.stderr or "")
            for finding in find_gpl_ldd_dependencies(combined_ldd):
                violations.append({"target": tpath.as_posix(), **finding})

    # 3) Source header include scan
    src_roots_list = [Path(r) for r in source_roots]
    violations.extend(scan_source_roots_for_gpl_includes(src_roots_list))

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "nm_bin": nm_bin,
        "ldd_bin": ldd_bin,
        "targets_scanned": scanned_targets,
        "source_roots_scanned": [p.as_posix() for p in src_roots_list],
        "violations": violations,
        "passed": len(violations) == 0,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify Apache-2.0 runtime targets & headers contain zero GPL/OpenFOAM symbols."
    )
    parser.add_argument("--targets", nargs="*", type=Path, default=[])
    parser.add_argument("--source-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--nm-bin", default="nm")
    parser.add_argument("--ldd-bin", default="ldd")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = audit_targets(
        args.targets,
        source_roots=args.source_roots,
        nm_bin=args.nm_bin,
        ldd_bin=args.ldd_bin,
        out_json=args.out,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
