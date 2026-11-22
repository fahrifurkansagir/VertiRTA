"""TDD verification suite for Plan 1B Tasks 1 & 2: 4-Tier Determinism CI & Embedded Gates."""

from __future__ import annotations

from pathlib import Path
import struct
import pytest
import yaml

from scripts.ci.check_arch_isolation import scan_forbidden_isolation
from scripts.ci.check_bare_metal_size import evaluate_berkeley_size
from scripts.ci.check_no_heap_symbols import find_forbidden_heap_symbols
from scripts.ci.check_stack_usage import parse_su_and_ci_dag
from scripts.ci.generate_benchmarks_md import render_benchmarks_markdown
from scripts.ci.verify_determinism import (
    VRTA_FRAME_FMT,
    VRTA_HEADER_FMT,
    VRTA_MAGIC,
    classify_steal_taint,
    compare_cross_isa_records,
    load_vrta_frames,
    parse_work_units_log,
    verify_same_isa_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_no_heap_symbols_catches_aligned_new_and_non_inline_tu() -> None:
    sample_nm = "\n".join(
        [
            "         U _Znwm",
            "         U _Znam",
            "         U _Znwj",
            "         U _Znaj",
            "         U _ZnwmSt11align_val_t",
            "         U _ZnamSt11align_val_t",
            "         U malloc",
            "         U calloc",
            "         U realloc",
            "         U free",
            "         U aligned_alloc",
            "         U posix_memalign",
            "         U sqrt",
        ]
    )
    offenders = set(find_forbidden_heap_symbols(sample_nm))
    expected = {
        "_Znwm",
        "_Znam",
        "_Znwj",
        "_Znaj",
        "_ZnwmSt11align_val_t",
        "_ZnamSt11align_val_t",
        "malloc",
        "calloc",
        "realloc",
        "free",
        "aligned_alloc",
        "posix_memalign",
    }
    assert expected == offenders
    assert "sqrt" not in offenders

    assert not (REPO_ROOT / "core/flight_core/include/vertirta/flight_core_no_heap.hpp").exists()
    trap_src = (REPO_ROOT / "tests/determinism/no_heap_link_trap.cpp").read_text(encoding="utf-8")
    assert "inline void* operator new" not in trap_src
    assert "std::align_val_t" in trap_src
    assert "vertirta_forbidden_heap_allocation_link_trap" in trap_src


def test_stack_usage_parses_su_ci_dag_and_rejects_dynamic_recursion_or_missing(tmp_path: Path) -> None:
    # Valid 2-level DAG including x86-64 GCC bounded stack realignment (`dynamic,bounded`)
    (tmp_path / "ok.su").write_text(
        "a.cpp:1:5:main\t256\tstatic\na.cpp:2:5:step\t512\tdynamic,bounded\n",
        encoding="utf-8",
    )
    (tmp_path / "ok.ci").write_text(
        'graph: { title: "a.cpp"\n'
        'node: { title: "main" label: "main\\n256 bytes (static)" }\n'
        'node: { title: "step" label: "step\\n512 bytes (dynamic,bounded)" }\n'
        'edge: { sourcename: "main" targetname: "step" }\n}\n',
        encoding="utf-8",
    )
    res = parse_su_and_ci_dag(tmp_path, max_frame_bytes=4096, max_dag_bytes=16384)
    assert res["passed"] is True
    assert res["max_frame_bytes"] == 512
    assert res["max_dag_bytes"] == 768
    assert res["bounded_realigned_frames"] == 1

    # Unbounded dynamic stack frame (VLA / alloca) MUST fail
    dyn_dir = tmp_path / "dyn"
    dyn_dir.mkdir()
    (dyn_dir / "bad.su").write_text("b.cpp:10:6:vla_fn\t512\tdynamic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unbounded dynamic"):
        parse_su_and_ci_dag(dyn_dir, max_frame_bytes=4096, max_dag_bytes=16384)

    # Recursion cycle MUST fail
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()
    (rec_dir / "r.su").write_text("r.cpp:1:5:f\t64\tstatic\nr.cpp:2:5:g\t64\tstatic\n", encoding="utf-8")
    (rec_dir / "r.ci").write_text(
        'graph: {\nnode: { title: "f" }\nnode: { title: "g" }\n'
        'edge: { sourcename: "f" targetname: "g" }\n'
        'edge: { sourcename: "g" targetname: "f" }\n}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="recursion"):
        parse_su_and_ci_dag(rec_dir, max_frame_bytes=4096, max_dag_bytes=16384)

    # Missing .su files (e.g. LTO erasure) MUST fail
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match=r"\.su"):
        parse_su_and_ci_dag(empty_dir, max_frame_bytes=4096, max_dag_bytes=16384)


def test_bare_metal_size_sums_archive_members_and_enforces_256k_128k() -> None:
    archive_out = (
        "   text    data     bss     dec     hex filename\n"
        "   2400      16     128    2544     9f0 frames.cpp.obj (ex libvertirta_flight_core.a)\n"
        "   2220      32     256    2508     9cc tmr_voter.cpp.obj (ex libvertirta_flight_core.a)\n"
    )
    ok = evaluate_berkeley_size(archive_out, max_flash_bytes=262144, max_ram_bytes=131072)
    assert ok["passed"] is True
    assert ok["flash_bytes"] == 2400 + 16 + 2220 + 32
    assert ok["ram_bytes"] == 16 + 128 + 32 + 256

    oversized = (
        "   text    data     bss     dec     hex filename\n"
        " 260000    8192    1024  269216   41ba0 vertirta_flight_core.elf\n"
    )
    assert evaluate_berkeley_size(oversized, max_flash_bytes=262144, max_ram_bytes=131072)["passed"] is False


def test_arch_isolation_guards_flight_core_and_rta_companion_against_ai_ros_proj(tmp_path: Path) -> None:
    # Real repo flight_core MUST pass cleanly
    real_res = scan_forbidden_isolation(REPO_ROOT)
    assert real_res["passed"] is True and real_res["violations"] == []

    # Synthetic violations in flight_core AND rta_companion MUST both be caught
    fc = tmp_path / "core/flight_core/src"
    rc = tmp_path / "core/rta_companion/src"
    fc.mkdir(parents=True)
    rc.mkdir(parents=True)
    (fc / "bad_proj.cpp").write_text('#include <proj.h>\n', encoding="utf-8")
    (rc / "bad_ai.cpp").write_text('#include "onnxruntime_cxx_api.h"\n#include <rclcpp/rclcpp.hpp>\n', encoding="utf-8")
    bad_res = scan_forbidden_isolation(tmp_path)
    assert bad_res["passed"] is False
    assert len(bad_res["violations"]) == 3


def test_verify_determinism_same_isa_cross_isa_work_units_and_steal_taint(tmp_path: Path) -> None:
    f1 = tmp_path / "run1.bin"
    f2 = tmp_path / "run2.bin"
    payload = struct.pack("<4d", 1.0, 2.0, 3.0, 4.0)
    f1.write_bytes(payload)
    f2.write_bytes(payload)
    same = verify_same_isa_sha256(f1, f2)
    assert same["passed"] is True and same["sha256_run1"] == same["sha256_run2"]

    assert compare_cross_isa_records(
        [{"mode": 2, "tmr_state": 0, "u": [1.0, -2.0]}],
        [{"mode": 2, "tmr_state": 0, "u": [1.0 + 5e-13, -2.0]}],
        max_abs_tol=1e-12,
    )["passed"] is True
    assert compare_cross_isa_records(
        [{"mode": 2, "tmr_state": 0, "u": [1.0]}],
        [{"mode": 3, "tmr_state": 0, "u": [1.0]}],
        max_abs_tol=1e-12,
    )["passed"] is False

    perf_ok = parse_work_units_log("perf_stat", "   450,000,000      instructions:u\n", 1000, 1200000)
    assert perf_ok["passed"] is True and perf_ok["ir_per_tick"] == 450000
    cg_fail = parse_work_units_log("valgrind_callgrind_fallback", "1,300,000,000  PROGRAM TOTALS\n", 1000, 1200000)
    assert cg_fail["passed"] is False
    cg_v322 = parse_work_units_log(
        "valgrind_callgrind_fallback",
        "723,332,439 (100.0%)  PROGRAM TOTALS\n",
        1000,
        1200000,
    )
    assert cg_v322["passed"] is True and cg_v322["ir_per_tick"] == 723332

    assert classify_steal_taint(1.01) == "NOISY_NEIGHBOR_TAINTED"
    assert classify_steal_taint(0.45) == "CLEAN"


def test_vrta_binary_128b_header_and_256b_frame_cross_isa_unpack(tmp_path: Path) -> None:
    assert struct.calcsize(VRTA_HEADER_FMT) == 128
    assert struct.calcsize(VRTA_FRAME_FMT) == 256

    hdr = struct.pack(
        VRTA_HEADER_FMT,
        VRTA_MAGIC,
        1,
        0,
        40.8860,
        29.1880,
        12.0,
        36.7,
        b"\x11" * 32,
        b"0" * 40,
        b"\x00" * 4,
        0x12345678,
    )

    def _make_frame(pos_e: float, flight_regime: int = 1, cel_latched: int = 0) -> bytes:
        floats_31 = [0.25 * i for i in range(31)]
        return struct.pack(
            VRTA_FRAME_FMT,
            7,
            70000,
            pos_e,
            -250.5,
            60.0,
            *floats_31,
            0xFB,           # active_motor_mask (byte 164)
            flight_regime,  # flight_regime (byte 165)
            1,              # tmr_state (byte 166)
            cel_latched,    # cel_latched (byte 167)
            0x00000003,     # flags (bytes 168..171)
            b"\x00" * 80,
            0xAABBCCDD,
        )

    p_arm = tmp_path / "arm64.vrta"
    p_x86_ok = tmp_path / "x86_ok.vrta"
    p_x86_drift = tmp_path / "x86_drift.vrta"
    p_x86_cel = tmp_path / "x86_cel.vrta"
    p_bad_magic = tmp_path / "bad_magic.vrta"

    p_arm.write_bytes(hdr + _make_frame(120.0))
    p_x86_ok.write_bytes(hdr + _make_frame(120.0 + 5e-13))
    p_x86_drift.write_bytes(hdr + _make_frame(120.0 + 5e-12))
    p_x86_cel.write_bytes(hdr + _make_frame(120.0, cel_latched=1))
    p_bad_magic.write_bytes(b"BADMAGIC" + hdr[8:] + _make_frame(120.0))

    res_ok = compare_cross_isa_records(load_vrta_frames(p_arm), load_vrta_frames(p_x86_ok), max_abs_tol=1e-12)
    assert res_ok["passed"] is True and res_ok["mode_divergences"] == 0 and res_ok["max_abs_diff"] < 1e-12

    res_drift = compare_cross_isa_records(load_vrta_frames(p_arm), load_vrta_frames(p_x86_drift), max_abs_tol=1e-12)
    assert res_drift["passed"] is False and res_drift["mode_divergences"] == 0

    res_cel = compare_cross_isa_records(load_vrta_frames(p_arm), load_vrta_frames(p_x86_cel), max_abs_tol=1e-12)
    assert res_cel["passed"] is False and res_cel["mode_divergences"] == 1

    with pytest.raises(ValueError, match="magic"):
        load_vrta_frames(p_bad_magic)


def test_generate_benchmarks_md_embeds_hypervisor_wcet_caveat() -> None:
    md = render_benchmarks_markdown(
        tier1={"flash_bytes": 4620, "ram_bytes": 0, "max_frame_bytes": 256, "max_dag_bytes": 768, "passed": True},
        tier2={"same_isa_0ulp": True, "max_abs_diff": 4.2e-13, "mode_divergences": 0, "passed": True},
        tier3_by_arch={"arm64": {"pmu_mode": "valgrind_callgrind_fallback", "ir_per_tick": 82000, "passed": True}},
        tier4_by_arch={"arm64": {"p50_us": 14.2, "p99_us": 28.5, "p999_us": 62.0, "steal_pct": 0.2, "taint": "CLEAN"}},
    )
    for token in (
        "ubuntu-24.04-arm",
        "Cobalt 100",
        "Neoverse N2",
        "NOISY_NEIGHBOR_TAINTED",
        "NOT",
        "WCET",
        "4620",
    ):
        assert token in md


def test_determinism_workflow_yaml_pins_runners_and_isolates_malloc_wrap() -> None:
    wf_path = REPO_ROOT / ".github/workflows/determinism.yml"
    wf_txt = wf_path.read_text(encoding="utf-8")
    assert "\n| true" not in wf_txt and "\n| sudo" not in wf_txt
    wf = yaml.safe_load(wf_txt)

    assert wf["jobs"]["tier1_embedded_structural"]["runs-on"] == "ubuntu-24.04-arm"
    runners = {m["runner"] for m in wf["jobs"]["tier2_3_4_matrix"]["strategy"]["matrix"]["include"]}
    assert runners == {"ubuntu-24.04", "ubuntu-24.04-arm"}
    assert "--wrap=malloc" not in str(wf["jobs"]["tier2_3_4_matrix"])

    cmake_txt = (REPO_ROOT / "tests/determinism/CMakeLists.txt").read_text(encoding="utf-8")
    assert "if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)" in cmake_txt
    assert "target_link_options(vertirta_flight_core_noheap_test PRIVATE" in cmake_txt
    assert "add_link_options(" not in cmake_txt
