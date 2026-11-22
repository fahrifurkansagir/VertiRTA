"""Public GitHub Release Hygiene Gate: bans personal paths, secret tokens, .agents/, and CRLF .sh."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any


BINARY_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".vrta",
        ".vrta_lut",
        ".vrta_cel2d",
        ".onnx",
        ".bin",
        ".gz",
        ".zst",
        ".zip",
        ".tar",
        ".mp4",
        ".mkv",
        ".pyc",
        ".o",
        ".a",
        ".so",
        ".exe",
        ".gguf",
    }
)

# Patterns assembled via concatenation so this script never triggers on itself.
PERSONAL_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("WINDOWS_USER_PATH", re.compile(r"[A-Za-z]:[\\/]" + r"Users[\\/]", re.IGNORECASE)),
    ("MACOS_USER_PATH", re.compile(r"(?<![A-Za-z0-9_])/" + r"Users/[A-Za-z0-9._-]+/")),
    ("PRIVATE_VAULT_REF", re.compile(r"Obsidian" + r"Vault", re.IGNORECASE)),
    ("LOCAL_USERNAME_REF", re.compile(r"(?<![A-Za-z0-9_])" + r"one" + r"ra(?![A-Za-z0-9_])", re.IGNORECASE)),
)

SECRET_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "GITHUB_TOKEN",
        re.compile(r"\b(?:ghp" + r"_|gho" + r"_|ghu" + r"_|ghs" + r"_|github_pat" + r"_)[A-Za-z0-9_]{16,}\b"),
    ),
    ("GOOGLE_API_KEY", re.compile(r"\bAI" + r"za[0-9A-Za-z_-]{16,}\b")),
    ("OPENAI_OR_ANTHROPIC_KEY", re.compile(r"\bsk" + r"-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{16,}\b")),
    ("HF_TOKEN", re.compile(r"\bhf" + r"_[A-Za-z0-9]{16,}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAK" + r"IA[0-9A-Z]{16}\b")),
    ("PRIVATE_KEY_PEM", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE " + r"KEY-----")),
)


def list_git_tracked_files(repo_root: Path) -> list[str]:
    """Return all tracked files from `git ls-files -z` as POSIX-relative paths."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    raw = proc.stdout.decode("utf-8", errors="replace")
    return [entry for entry in raw.split("\x00") if entry]


def _redact_match(text: str) -> str:
    if len(text) <= 10:
        return text[:4] + "***"
    return text[:6] + "..." + text[-4:]


def scan_public_hygiene(
    repo_root: Path,
    tracked_files: list[str] | None = None,
) -> dict[str, Any]:
    """Scan tracked repository files for leaked personal paths, secrets, .agents/, and CRLF .sh."""
    rel_paths = list_git_tracked_files(repo_root) if tracked_files is None else list(tracked_files)
    violations: list[dict[str, Any]] = []
    files_scanned = 0

    for rel in sorted(rel_paths):
        norm_rel = rel.replace("\\", "/")
        if norm_rel == ".agents" or norm_rel.startswith(".agents/"):
            violations.append(
                {
                    "file": norm_rel,
                    "line": 0,
                    "kind": "TRACKED_AGENTS_DIR",
                    "rule": "FORBIDDEN_AGENTS_TRACKING",
                    "match": norm_rel,
                }
            )
            continue

        path = repo_root / norm_rel
        if not path.is_file():
            continue

        files_scanned += 1
        raw_bytes = path.read_bytes()

        if norm_rel.endswith(".sh") and b"\r" in raw_bytes:
            violations.append(
                {
                    "file": norm_rel,
                    "line": 0,
                    "kind": "CRLF_IN_SHELL_SCRIPT",
                    "rule": "SHELL_SCRIPT_MUST_BE_LF",
                    "match": "CR byte present",
                }
            )

        if path.suffix.lower() in BINARY_SUFFIXES or b"\x00" in raw_bytes[:8192]:
            continue

        text = raw_bytes.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule_name, pat in PERSONAL_PATH_PATTERNS:
                m = pat.search(line)
                if m:
                    violations.append(
                        {
                            "file": norm_rel,
                            "line": lineno,
                            "kind": "PERSONAL_PATH_OR_USER",
                            "rule": rule_name,
                            "match": m.group(0),
                        }
                    )
            for rule_name, pat in SECRET_TOKEN_PATTERNS:
                m = pat.search(line)
                if m:
                    violations.append(
                        {
                            "file": norm_rel,
                            "line": lineno,
                            "kind": "HARDCODED_SECRET_TOKEN",
                            "rule": rule_name,
                            "match": _redact_match(m.group(0)),
                        }
                    )

    report: dict[str, Any] = {
        "schema_version": "vertirta.public_hygiene/v1.0.0",
        "files_scanned": files_scanned,
        "violations": violations,
        "passed": len(violations) == 0,
    }
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify tracked repository files contain zero leaked paths, tokens, .agents/, or CRLF .sh."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = scan_public_hygiene(args.repo_root)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
