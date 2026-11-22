"""Contract tests for .github/workflows/product_and_wasm.yml (Plan 4B Task 7)."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_product_and_wasm_workflow_covers_wasm_150kb_vertigcs_sdk_and_docker() -> None:
    wf_path = REPO / ".github/workflows/product_and_wasm.yml"
    raw = wf_path.read_bytes()
    assert b"\r" not in raw, "Workflow must use pure LF"
    assert not raw.startswith(b"\xef\xbb\xbf"), "Workflow must not contain UTF-8 BOM"

    text = raw.decode("utf-8")
    wf = yaml.safe_load(text)
    jobs = wf["jobs"]
    expected_jobs = {
        "wasm_flight_core",
        "vertigcs_frontend",
        "python_sdk_and_cmake_install",
        "docker_and_governance",
    }
    assert expected_jobs.issubset(jobs.keys())

    for name, job in jobs.items():
        assert job["runs-on"] == "ubuntu-24.04", f"{name} must run on ubuntu-24.04"

    assert "153600" in text and "build_wasm.sh" in text
    assert "npm test" in text and "npm run build" in text
    assert "tests/cmake" in text and "test_vertirta_sdk.py" in text
    assert "docker compose -f docker/docker-compose.yml config" in text
    assert "check_public_hygiene.py" in text
    assert "check_arch_isolation.py" in text

    # Zero overlap with Plan 4A formal/HIL jobs
    assert "esbmc" not in text.lower() and "renode" not in text.lower()

    # Hard ban threading/COI enablement in WASM/productization workflow
    forbidden_tokens = ("-p" + "thread", "coi" + "-serviceworker", "USE_PTHREADS")
    for tok in forbidden_tokens:
        assert tok not in text, f"Forbidden WASM threading/COI token {tok!r} in workflow"
