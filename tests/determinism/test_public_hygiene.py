"""TDD unit tests for scripts/ci/check_public_hygiene.py (public GitHub release hygiene gate)."""

from __future__ import annotations

from pathlib import Path

from scripts.ci.check_public_hygiene import scan_public_hygiene


def test_clean_repo_passes_public_hygiene(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# VertiRTA\nClean Apache-2.0 repo.\n", encoding="utf-8", newline="\n")
    (tmp_path / "run.sh").write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8", newline="\n")
    report = scan_public_hygiene(tmp_path, tracked_files=["README.md", "run.sh"])
    assert report["passed"] is True
    assert report["violations"] == []
    assert report["files_scanned"] == 2


def test_detects_personal_paths_tokens_agents_dir_and_crlf_sh(tmp_path: Path) -> None:
    win_leak = "C:\\" + "Users\\" + "one" + "ra\\secret.txt"
    mac_leak = "/" + "Users/jdoe/workspace"
    vault_leak = "Sync with " + "Obsidian" + "Vault now"
    tok_gh = "ghp" + "_" + "A1b2C3d4E5f6G7h8I9j0K1l2"
    tok_g = "AI" + "zaSyD1234567890abcdefGHIJKLMN"
    tok_oai = "sk" + "-proj-1234567890abcdefGHIJKLMN"
    tok_hf = "hf" + "_abcdef1234567890ABCDEF1234"

    (tmp_path / "leaks.md").write_text(
        "\n".join([win_leak, mac_leak, vault_leak, tok_gh, tok_g, tok_oai, tok_hf]) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / "bad.sh").write_bytes(b"#!/usr/bin/env bash\r\necho crlf\r\n")
    agents_file = tmp_path / ".agents" / "rules" / "lock.md"
    agents_file.parent.mkdir(parents=True)
    agents_file.write_text("# private rule\n", encoding="utf-8", newline="\n")

    report = scan_public_hygiene(
        tmp_path,
        tracked_files=["leaks.md", "bad.sh", ".agents/rules/lock.md"],
    )
    assert report["passed"] is False
    kinds = {v["kind"] for v in report["violations"]}
    assert "TRACKED_AGENTS_DIR" in kinds
    assert "CRLF_IN_SHELL_SCRIPT" in kinds
    assert "PERSONAL_PATH_OR_USER" in kinds
    assert "HARDCODED_SECRET_TOKEN" in kinds
    # Verify all 4 token families caught
    token_hits = [v for v in report["violations"] if v["kind"] == "HARDCODED_SECRET_TOKEN"]
    assert len(token_hits) == 4
    # Verify personal path + vault + mac path caught
    path_hits = [v for v in report["violations"] if v["kind"] == "PERSONAL_PATH_OR_USER"]
    assert len(path_hits) >= 3


def test_live_repository_passes_public_hygiene() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    report = scan_public_hygiene(repo_root)
    assert report["files_scanned"] > 100
    assert report["passed"] is True, f"Public hygiene violations: {report['violations']}"

