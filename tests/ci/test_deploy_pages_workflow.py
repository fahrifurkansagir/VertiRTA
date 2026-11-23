"""Contract tests for zero-install GitHub Pages VertiGCS WASM deployment (v3.1.0 Track 1)."""

from __future__ import annotations

from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO / ".github/workflows/deploy_pages.yml"


def test_deploy_pages_workflow_builds_wasm_sample_vrta_and_publishes_pages() -> None:
    raw = WORKFLOW_PATH.read_bytes()
    assert b"\r" not in raw, "deploy_pages.yml must use pure LF"
    assert not raw.startswith(b"\xef\xbb\xbf"), "deploy_pages.yml must not have UTF-8 BOM"
    text = raw.decode("utf-8")
    wf = yaml.safe_load(text)

    # Required GitHub Pages permissions & non-canceling concurrency
    perms = wf.get("permissions", {})
    assert perms.get("pages") == "write"
    assert perms.get("id-token") == "write"
    assert wf.get("concurrency", {}).get("group") == "pages"

    jobs = wf["jobs"]
    assert {"build", "deploy"}.issubset(jobs.keys())
    assert jobs["build"]["runs-on"] == "ubuntu-24.04"
    assert jobs["deploy"]["runs-on"] == "ubuntu-24.04"
    assert jobs["deploy"]["needs"] == "build"
    assert jobs["deploy"]["environment"]["name"] == "github-pages"

    # Required build, sample .vrta, WASM cap, Vite subpath, and official Pages actions
    for required in (
        "mymindstorm/setup-emsdk",
        "wasm/build_wasm.sh",
        "153600",
        "vertirta_replay_cli",
        "corridor_faults.vrta",
        "307200",
        "vite build --base /VertiRTA/",
        "actions/configure-pages@v5",
        "actions/upload-pages-artifact@v3",
        "actions/deploy-pages@v4",
    ):
        assert required in text, f"Missing required Pages deployment lock: {required!r}"

    # Zero pthreads/embind or forbidden workspace port bindings
    for banned in ("-pthread", "USE_PTHREADS", "--bind", "4173:", "4174:", "8765:", "9222:", "9223:"):
        assert banned not in text, f"Forbidden token {banned!r} in deploy_pages.yml"


def test_vertigcs_gitignore_excludes_ci_generated_wasm_and_sample_vrta() -> None:
    gi = (REPO / "apps/vertigcs/.gitignore").read_text(encoding="utf-8")
    for entry in ("node_modules/", "dist/", "public/wasm/", "public/data/corridor_faults.vrta"):
        assert entry in gi, f"apps/vertigcs/.gitignore must ignore {entry!r}"
