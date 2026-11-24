#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publication-grade LaTeX, BibTeX, cross-reference & figure audit for Paper 1 (RA-L) and Omnibus report."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_figure_path(paper_dir: Path, raw_ref: str) -> Path | None:
    cand = paper_dir / raw_ref
    if cand.is_file():
        return cand
    for ext in (".pdf", ".png", ".eps", ".jpg"):
        alt = paper_dir / f"{raw_ref}{ext}"
        if alt.is_file():
            return alt
    return None


def _audit_single_paper(
    repo_root: Path,
    paper_dir: Path,
    tex_name: str,
    bib_name: str,
    *,
    double_anonymous: bool,
    extra_forbidden: tuple[str, ...] = (),
) -> dict[str, Any]:
    errors: list[str] = []
    tex_path = paper_dir / tex_name
    bib_path = paper_dir / bib_name

    if not tex_path.is_file():
        errors.append(f"Missing TeX file: {tex_path}")
    if not bib_path.is_file():
        errors.append(f"Missing BibTeX file: {bib_path}")
    if errors:
        return {"passed": False, "errors": errors}

    tex_raw = tex_path.read_text(encoding="utf-8")
    bib_raw = bib_path.read_text(encoding="utf-8")

    # Resolve \input{...} files recursively (1 level)
    missing_includes: list[str] = []
    combined_tex = tex_raw
    for inc in re.findall(r"\\input\{([^}]+)\}", tex_raw):
        inc_path = paper_dir / (inc if inc.endswith(".tex") else f"{inc}.tex")
        if not inc_path.is_file():
            missing_includes.append(inc)
        else:
            combined_tex += "\n" + inc_path.read_text(encoding="utf-8")
    if missing_includes:
        errors.append(f"Missing \\input targets: {missing_includes}")

    # Strip LaTeX line comments before checking citations/labels/refs/environments
    stripped_lines = [re.sub(r"(?<!\\)%.*$", "", ln) for ln in combined_tex.splitlines()]
    clean_tex = "\n".join(stripped_lines)

    # 1) BibTeX citation parity (0 undefined \cite, 0 uncited orphan .bib entries)
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_raw))
    cited_keys = {
        k.strip()
        for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", clean_tex)
        for k in grp.split(",")
        if k.strip()
    }
    undefined_citations = sorted(cited_keys - bib_keys)
    orphan_bib_entries = sorted(bib_keys - cited_keys)
    if undefined_citations:
        errors.append(f"Undefined BibTeX citations: {undefined_citations}")
    if orphan_bib_entries:
        errors.append(f"Uncited orphan BibTeX entries: {orphan_bib_entries}")

    # 2) Label & \ref consistency (0 undefined refs, 0 duplicate labels)
    all_labels = re.findall(r"\\label\{([^}]+)\}", clean_tex)
    seen_labels: set[str] = set()
    duplicate_labels: list[str] = []
    for lbl in all_labels:
        if lbl in seen_labels:
            duplicate_labels.append(lbl)
        seen_labels.add(lbl)
    all_refs = set(re.findall(r"\\(?:eq|auto|page|c|C)?ref\{([^}]+)\}", clean_tex))
    undefined_refs = sorted(all_refs - seen_labels)
    unreferenced_labels = sorted(seen_labels - all_refs)
    if duplicate_labels:
        errors.append(f"Duplicate \\label definitions: {duplicate_labels}")
    if undefined_refs:
        errors.append(f"Undefined \\ref targets: {undefined_refs}")

    # 3) Environment balance
    begins = re.findall(r"\\begin\{([^}]+)\}", clean_tex)
    ends = re.findall(r"\\end\{([^}]+)\}", clean_tex)
    if begins != ends:
        # Check exact multiset / stack nesting
        stack: list[str] = []
        env_tokens = re.findall(r"\\(begin|end)\{([^}]+)\}", clean_tex)
        for kind, name in env_tokens:
            if kind == "begin":
                stack.append(name)
            else:
                if not stack or stack[-1] != name:
                    errors.append(f"Mismatched LaTeX environment: \\end{{{name}}} (stack={stack[-3:]})")
                    break
                stack.pop()
        if stack:
            errors.append(f"Unclosed LaTeX environments: {stack}")

    # 4) Figure existence, header integrity & double-anonymous metadata scan
    fig_refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", clean_tex)
    missing_figures: list[str] = []
    figures_audited: list[dict[str, Any]] = []
    anonymity_violations: list[str] = []
    for fref in fig_refs:
        fpath = _resolve_figure_path(paper_dir, fref)
        if fpath is None:
            missing_figures.append(fref)
            continue
        raw_bytes = fpath.read_bytes()
        valid_magic = raw_bytes.startswith(b"%PDF-") or raw_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        if not valid_magic:
            errors.append(f"Invalid figure binary magic: {fpath.relative_to(repo_root)}")
        if double_anonymous:
            for leak in (b"github.com", b"fahrifurkansagir", b"mailto:") + tuple(
                s.encode("utf-8") for s in extra_forbidden
            ):
                if leak in raw_bytes:
                    anonymity_violations.append(f"Figure leak {leak!r} in {fref}")
        figures_audited.append(
            {
                "ref": fref,
                "path": fpath.relative_to(repo_root).as_posix(),
                "bytes": len(raw_bytes),
                "valid_magic": valid_magic,
            }
        )
    if missing_figures:
        errors.append(f"Missing figures: {missing_figures}")

    # 5) Anonymity, anti-placeholder, and non-certification honesty scan
    common_bans = ("TODO", "[TBD]")
    if double_anonymous:
        common_bans = common_bans + ("github.com/", "mailto:", "fahrifurkansagir") + extra_forbidden
    for tok in common_bans:
        if tok in clean_tex or tok in bib_raw:
            anonymity_violations.append(f"Forbidden token '{tok}' found in {tex_name}/{bib_name}")
    if re.search(
        r"we (achieve|demonstrate|are) (DO-178C|SC-VTOL|SORA).*certified", clean_tex, re.IGNORECASE
    ):
        anonymity_violations.append("Forbidden airworthiness certification claim found")
    if anonymity_violations:
        errors.extend(anonymity_violations)

    # 6) Optional pdflatex compile test if installed on host/WSL2
    pdflatex_bin = shutil.which("pdflatex")
    pdflatex_status = "SKIPPED_NOT_INSTALLED"
    if pdflatex_bin:
        with tempfile.TemporaryDirectory(prefix="vertirta_latex_") as tmpdir:
            proc = subprocess.run(
                [pdflatex_bin, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", tmpdir, tex_name],
                cwd=paper_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                pdflatex_status = "COMPILED_OK"
            else:
                pdflatex_status = f"COMPILE_FAILED_RC_{proc.returncode}"
                errors.append(f"pdflatex failed on {tex_name}: rc={proc.returncode}")

    return {
        "passed": len(errors) == 0,
        "tex_file": tex_path.relative_to(repo_root).as_posix(),
        "bib_file": bib_path.relative_to(repo_root).as_posix(),
        "double_anonymous": double_anonymous,
        "cited_count": len(cited_keys),
        "bib_entry_count": len(bib_keys),
        "undefined_citations": undefined_citations,
        "orphan_bib_entries": orphan_bib_entries,
        "label_count": len(seen_labels),
        "ref_count": len(all_refs),
        "undefined_refs": undefined_refs,
        "duplicate_labels": duplicate_labels,
        "unreferenced_labels_info": unreferenced_labels,
        "missing_includes": missing_includes,
        "missing_figures": missing_figures,
        "figures_audited": figures_audited,
        "anonymity_or_honesty_violations": anonymity_violations,
        "pdflatex_status": pdflatex_status,
        "errors": errors,
    }


def audit_all_papers(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    paper_ral = _audit_single_paper(
        repo_root,
        repo_root / "docs" / "paper_ral",
        "vertirta_ral_manuscript.tex",
        "references_ral.bib",
        double_anonymous=True,
        extra_forbidden=("Qwen3", "PIDNet", "VertiBench"),
    )
    paper_omnibus = _audit_single_paper(
        repo_root,
        repo_root / "docs" / "paper",
        "vertirta_tech_report.tex",
        "references.bib",
        double_anonymous=False,
    )
    return {
        "schema_version": "vertirta.latex_figure_audit/v1.0.0",
        "passed": bool(paper_ral["passed"] and paper_omnibus["passed"]),
        "papers": {
            "paper_ral": paper_ral,
            "paper_omnibus": paper_omnibus,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    res = audit_all_papers(args.repo_root)
    print(json.dumps(res, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
