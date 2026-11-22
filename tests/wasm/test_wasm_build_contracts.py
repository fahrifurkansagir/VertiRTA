"""Contract tests for single-threaded flat-C Emscripten WASM build (Plan 4B Task 1)."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]


def test_wasm_build_bans_pthreads_embind_and_enforces_150kb_cap() -> None:
    script_path = REPO / "wasm/build_wasm.sh"
    script = script_path.read_bytes()
    assert b"\r" not in script, "wasm/build_wasm.sh must use pure LF"
    text = script.decode("utf-8")

    for required in (
        "-sENVIRONMENT=worker",
        "-sMODULARIZE=1",
        "-sEXPORT_ES6=1",
        "-sEXPORTED_FUNCTIONS=@",
        "-fno-rtti",
        "-fno-exceptions",
        "-ffp-contract=off",
        "-fno-fast-math",
        "153600",
    ):
        assert required in text, f"Missing required WASM flag/cap: {required}"

    for banned in ("-pthread", "USE_PTHREADS", "coi-serviceworker", "--bind", "embind", "-ffast-math"):
        assert banned not in text, f"Forbidden WASM token present: {banned}"

    exports_path = REPO / "wasm/exported_funcs.txt"
    assert b"\r" not in exports_path.read_bytes()
    exports = exports_path.read_text(encoding="utf-8")
    for fn in (
        "_vertirta_init",
        "_vertirta_reset",
        "_vertirta_step_batch",
        "_vertirta_inject_fault",
        "_vertirta_get_vrta_ptr",
        "_vertirta_get_vrta_size",
        "_vertirta_parse_vrta_frame",
        "_vertirta_egm2008_n_m",
        "_vertirta_eval_cbvf_margin",
    ):
        assert fn in exports, f"Missing exported flat-C function: {fn}"

    bash = shutil.which("bash")
    if bash is not None:
        proc = subprocess.run(
            [bash, "-n", script_path.relative_to(REPO).as_posix()],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"bash -n failed: {proc.stderr}"
