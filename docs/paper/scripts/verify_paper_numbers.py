"""Honesty, anti-placeholder, BibTeX, and numeric-consistency linter for docs/paper/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


PLACEHOLDER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[TBD]/TBD", re.compile(r"(?:\[\s*TBD\s*\]|\bTBD\b)", re.IGNORECASE)),
    ("[Ölçülecek]", re.compile(r"\[\s*(?:Ölçülecek|Olculecek|OLCULECEK)\s*\]", re.IGNORECASE)),
    ("TODO/FIXME", re.compile(r"\b(?:TODO|FIXME)\b")),
)

NEGATION_RE = re.compile(
    r"\b(?:not|never|cannot|can't|does\s+not|do\s+not|neither|nor|no|zero|without|"
    r"non-certif\w*|non-complian\w*|disclaim\w*|impossib\w*|prohibit\w*|insufficient|"
    r"refut\w*|exclud\w*|undeveloped|outside|rather\s+than)\b",
    re.IGNORECASE,
)

MC_1E9_RE = re.compile(r"10\s*\^\s*\{\s*-9\s*\}|10\^-9", re.IGNORECASE)
MC_CLAIM_VERB_RE = re.compile(
    r"\b(?:prove[sd]?|proving|demonstrate[sd]?|establishes|satisfies|achieves|compliant|compliance|certified)\b",
    re.IGNORECASE,
)

REG_STANDARD_RE = re.compile(
    r"\b(?:DO-178C|DO-254|SC-VTOL(?:-02)?|SORA\s*2\.5|SORA|EASA|SHGM|Part-21|ARP4754[AB]?|ARP4761A?)\b",
    re.IGNORECASE,
)
CERT_CLAIM_RE = re.compile(
    r"\b(?:certified|type-certified|airworthy|compliant|full\s+compliance|achieves\s+compliance|"
    r"meets\s+certification|passes\s+certification|certification\s+credit)\b",
    re.IGNORECASE,
)

CITE_RE = re.compile(r"\\cite[a-z]*\{([^}]+)\}")
BIB_KEY_RE = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,")
ENV_RE = re.compile(r"\\(begin|end)\{([^}]+)\}")


def _strip_latex_comments(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"(?<!\\)%.*$", "", line)
        out_lines.append(cleaned)
    return "\n".join(out_lines)


def _iter_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", flat) if s.strip()]


def _collect_numeric_leaves(obj: Any, prefix: str = "") -> list[tuple[str, int | float]]:
    items: list[tuple[str, int | float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            items.extend(_collect_numeric_leaves(v, sub))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        items.append((prefix, obj))
    return items


def _number_candidates(val: int | float) -> list[str]:
    if isinstance(val, int):
        cands = [str(val)]
        if abs(val) >= 10_000:
            cands.append(f"{val:,}")
            cands.append(f"{val:,}".replace(",", "{,}"))
        return cands
    cands = [f"{val:g}", f"{val:.1f}", f"{val:.2f}", f"{val:.3f}", f"{val:.4f}"]
    trimmed = f"{val:.6f}".rstrip("0")
    if trimmed.endswith("."):
        trimmed += "0"
    cands.append(trimmed)
    return list(dict.fromkeys(cands))


def _check_latex_environments(path: Path, body: str) -> list[str]:
    errors: list[str] = []
    stack: list[str] = []
    for m in ENV_RE.finditer(body):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                errors.append(
                    f"{path.name}: mismatched \\end{{{env}}} (open stack: {stack[-3:]})"
                )
            else:
                stack.pop()
    if stack:
        errors.append(f"{path.name}: unclosed LaTeX environments: {stack}")
    return errors


def verify_paper_honesty_and_numbers(tex_dir: Path, manifest_path: Path) -> list[str]:
    """Audit all .tex files in tex_dir for honesty violations, placeholders, BibTeX, and number drift."""
    errors: list[str] = []
    tex_files = sorted(tex_dir.rglob("*.tex"))
    if not tex_files:
        return [f"No .tex files found under {tex_dir}"]

    # 1. Placeholder, 10^-9 MC claim, false-certification, and LaTeX balance audit
    bib_path = tex_dir / "references.bib"
    bib_keys: set[str] = set()
    if bib_path.exists():
        bib_keys = set(BIB_KEY_RE.findall(bib_path.read_text(encoding="utf-8")))

    for tex_path in tex_files:
        raw = tex_path.read_text(encoding="utf-8")
        body = _strip_latex_comments(raw)

        for label, pat in PLACEHOLDER_PATTERNS:
            if pat.search(body):
                errors.append(f"{tex_path.name}: forbidden placeholder {label} found")

        for sent in _iter_sentences(body):
            has_neg = bool(NEGATION_RE.search(sent))
            if MC_1E9_RE.search(sent) and MC_CLAIM_VERB_RE.search(sent) and not has_neg:
                errors.append(
                    f"{tex_path.name}: forbidden affirmative 10^{{-9}}/FH claim without negation: '{sent[:120]}'"
                )
            if REG_STANDARD_RE.search(sent) and CERT_CLAIM_RE.search(sent) and not has_neg:
                errors.append(
                    f"{tex_path.name}: forbidden affirmative certified/compliance claim: '{sent[:120]}'"
                )

        errors.extend(_check_latex_environments(tex_path, body))

        if bib_keys:
            for m in CITE_RE.finditer(body):
                for key in (k.strip() for k in m.group(1).split(",")):
                    if key and key not in bib_keys:
                        errors.append(f"{tex_path.name}: undefined BibTeX citation key '{key}'")

    # 2. Verify numeric leaves of manifest_path appear verbatim in generated tables.tex
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tables_candidates = [tex_dir / "generated" / "tables.tex", tex_dir / "tables.tex"]
        tables_path = next((p for p in tables_candidates if p.exists()), None)
        if tables_path is None:
            errors.append(f"Missing generated tables.tex under {tex_dir}")
        else:
            tables_body = _strip_latex_comments(tables_path.read_text(encoding="utf-8"))
            tables_norm = tables_body.replace("{,}", "").replace(",", "")
            for key, val in _collect_numeric_leaves(manifest):
                cands = _number_candidates(val)
                if not any((c in tables_body) or (c.replace("{,}", "").replace(",", "") in tables_norm) for c in cands):
                    errors.append(
                        f"{tables_path.name}: manifest metric '{key}'={val} missing from generated tables"
                    )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify VertiRTA LaTeX paper honesty and numbers.")
    parser.add_argument("--tex-dir", type=Path, default=Path("docs/paper"))
    parser.add_argument("--manifest", type=Path, default=Path("docs/paper/metrics_manifest.json"))
    args = parser.parse_args()
    errs = verify_paper_honesty_and_numbers(args.tex_dir, args.manifest)
    if errs:
        for e in errs:
            print(f"ERROR: {e}")
        return 1
    print("VERIFY_PAPER_HONESTY_AND_NUMBERS: PASS (0 errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
