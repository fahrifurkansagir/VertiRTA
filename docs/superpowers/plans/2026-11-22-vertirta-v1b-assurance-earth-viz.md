# VertiRTA v1.0.0 — Plan 1B: Assurance, Determinism CI, 2-Tier Verification Pyramid & Photorealistic Earth Viz Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the complete `v1.0.0` 4-tier ARM64/x86_64 determinism CI, non-compliant-disclaimed aviation assurance & CI-linked GSN v3 Living Safety Case, 2-tier stratified Survival Envelope Atlas, and ToS-clean photorealistic Istanbul Earth visualization (`CesiumJS :5180` + Colab G4 `Blender 5.2 LTS` Cycles OptiX) with zero file overlap against Plan 1A.

**Architecture:** Plan 1B consumes Plan 1A's `vertirta::flight_core` / `vertirta::rta_companion` CMake aliases, `tools/cli/` binaries (`vertirta_replay_cli`, `vertirta_bench_cli`, `vertirta_6dof_batch_cli`), canonical `.vrta` binary frames (`VrtaFileHeader` `128 B` `magic="VRTA0001"` + `VrtaFrame100Hz` `256 B` LE POD + `crc32c`), and PX4/Gazebo stack strictly via read-only CLI/target interfaces. It adds four decoupled layers: (1) standalone CMake/CI determinism & Cortex-M7 bare-metal gates, (2) MLEAP `ODD.yaml` + ARP4761A FHA/FTA + automated SCSC-141C GSN v3 builder, (3) a 10k/1k/50 Verification Pyramid with strict CSFL/CEL denominator isolation, and (4) a legally firewalled dual-tier Istanbul Earth visualization pipeline.

**Tech Stack:** GitHub Actions (`ubuntu-24.04`, `ubuntu-24.04-arm` Azure Cobalt 100 / Armv9-A Neoverse N2), `g++-14`, `arm-none-eabi-g++`, `perf`/`valgrind --tool=callgrind`, Python 3.11+ (`pytest`, `scipy.stats`, `pyyaml`, `pyproj`, `rasterio`, `shapely`, `duckdb`, `matplotlib`, `websockets`, `Pillow`), `GDAL`/`osmium-tool`, CesiumJS, Blender 5.2 LTS (`bpy` Cycles OptiX, fallback 4.5 LTS), `ffmpeg` (`hevc_nvenc`).

**Spec:** `docs/superpowers/specs/2026-11-22-vertirta-tier-s-design.md` (v1.2 FROZEN) & `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (7/7 Forensic Errata Overrides).

## Global Constraints

- **License & Legal Firewall:** Apache 2.0; ZERO `CC BY-NC-SA` assets (EOX Sentinel-2 Cloudless & FABDEM hard-banned), ZERO offline scraping of Bing/Esri/Google/Cesium ion tiles, ZERO OSM `ODbL` data committed to git (download scripts + local `viz/geo/data/.gitignore` only).
- **Vehicle, Datum & ODD Locks:** `VertiRTA-X8-240` (`240 kg`, `T/W=2.5`, `400 V / 22 kWh`, `R_SLZ >= 5.0 m`, `theta <= 8 deg`, `sigma_z <= 0.15 m`, `age > 250 ms` staleness). Kartal Coastal Vertiport datum: `40.8860 N, 29.1880 E, 12.0 m AMSL`, `EGM2008 N0 = +36.7 m`, `R_gauss = 6375036.971 m`. `ODD.yaml` strictly `< 29.1900 E` near `LTFJ` NW join `J2 (40.91279 N, 29.19212 E)`.
- **Regulatory Honesty Lock:** Zero certification claims (explicit Non-Compliance Disclaimer in `SAFETY_CASE.md` & `BENCHMARKS.md`); `SC-VTOL-02` + `MOC-5 Iss. 2 FINAL` (5 Nov 2026) + `ASTM F3269-21` + `SHGM SHT-İHA` (eff. 30 Jul 2026, M3/İHATTYS) primary drivers; `SORA 2.5` (`ED Dec 2026/010/R` retiring SORA 2.0 on 1 Jan 2027) restricted to Stage-A 25 kg unmanned subscale; Triplex 2oo3 FCC FTA = `3 * lambda_fcc^2 * T = 3.00e-8/FH` (NOT `lambda^3 T^2`); CSFL (`n=650`) vs CEL (`n=150`) denominators strictly isolated; `G5`/`G6` permanently `undeveloped: true`.
- **Determinism & Embedded Locks:** Pinned `ubuntu-24.04` + `ubuntu-24.04-arm` (`g++-14`, `-fno-fast-math -ffp-contract=off`); standalone TU non-inline `operator new` trap (NO `inline` in headers!); `-Wl,--wrap=malloc` isolated from `vertirta_replay_cli`; `-fstack-usage -fno-lto -fcallgraph-info=su,da`; Cortex-M7 `< 256 KB` Flash / `< 128 KB` RAM on `libvertirta_flight_core` only; Same-ISA `0-ULP` SHA-256 + Cross-ISA `< 1e-12` + `0` discrete mode divergence; `steal > 1.0%` flagged `NOISY_NEIGHBOR_TAINTED`.
- **Ports, VRAM & Geospatial/Blender Locks:** CesiumJS `:5180` (`< 1.8 GB` VRAM on GTX 1650 SUPER), Foxglove `:8891`, `MicroXRCEAgent` `:8888`; forbidden `4173, 4174, 8765, 9222, 9223`. Warp rasters strictly to Kartal local topocentric CRS (`+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84 +units=m +no_defs`, NEVER `EPSG:32635`) with `-dstnodata -9999` (NEVER `0.0`). Blender 5.2 LTS (`driver >= 575`, fallback 4.5 LTS), `ShaderNodeMix` RGBA indices `[6]/[7]/[2]`, `argparse.parse_known_args()`, `hevc_nvenc -preset p5`.

## Review Focus

1. **Silent `N0 = 0.0 m` EGM2008 PROJ Grid Fallback:** Missing `us_nga_egm08_25.tif` silently buries terrain/aircraft `36.7 m` underground — pinned in Task 6 (`test_geodesy_rejects_missing_egm2008_grid`).
2. **Blender `ShaderNodeMix` Hidden Float Sockets:** String lookup `inputs["A"/"B"]` / `outputs["Result"]` on `RGBA` mix nodes binds hidden float sockets, turning procedural PBR roofs flat gray — pinned in Task 8 (`test_pbr_mix_nodes_use_rgba_indices_6_7_2`).
3. **Inverted Urban Wind Shear & Pooled CSFL/CEL Denominators:** `W_20*(z/6)^0.30` yields urban winds faster than coastal fetch, and pooling `S4` (`n=150`) into CSFL (`n=650`) corrupts CIs — pinned in Task 4 (`test_urban_wind_shear_never_exceeds_coastal`, `test_csfl_cel_denominators_strictly_isolated`).
4. **Aligned C++17 `operator new` Leak, LTO `.su` Erasure & `replay_cli` Wrap Breakage:** Narrow `nm` regex misses `_ZnwmSt11align_val_t`/`_Znwj`, `-flto` drops `.su`, and wrapping `malloc` breaks `vertirta_replay_cli` glibc I/O — pinned in Task 1 (`test_nm_symbol_ban_catches_aligned_new`, `test_stack_dag_rejects_dynamic_or_missing_su`) & Task 2 (`test_steal_taint_and_workflow_isolation`).
5. **GSN `G5`/`G6` Premature Development & Frozen CesiumJS `requestRenderMode`:** Clearing `undeveloped: true` falsifies certification status, and omitting `scene.requestRender()` per WS frame freezes `:5180` — pinned in Task 3 (`test_gsn_builder_fails_if_g5_or_g6_developed`) & Task 7 (`test_cesium_config_triggers_request_render_and_uses_lat_lon_hae`).

---

## File Structure & Cross-Plan Interfaces (ZERO Overlap with Plan 1A)

- **Consumes (Plan 1A):** `core/flight_core/` (`vertirta::flight_core`) & `core/rta_companion/` (`vertirta::rta_companion`), `tools/cli/{vertirta_replay_cli,vertirta_bench_cli,vertirta_6dof_batch_cli}`, canonical `.vrta` (`VrtaFileHeader` `128 B` `magic="VRTA0001"` + `VrtaFrame100Hz` `256 B` LE POD + `crc32c`), `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`, headless PX4/Gazebo `vertirta_x8_240_${K}` on `sim/worlds/kartal_coast.sdf` (`world_name="kartal_coast"`).
- **Produces (Plan 1B):**
  - Determinism CI: `.github/workflows/determinism.yml`, `cmake/arm-none-eabi-cortex-m7.cmake`, `tests/determinism/{CMakeLists.txt,no_heap_link_trap.cpp,cm7_entry_stub.cpp}`, `scripts/ci/{check_stack_usage.py,check_embedded_footprint.py,check_arch_isolation.py,compare_vrta_cross_isa.py,parse_work_units.py,run_monotonic_with_steal.py}`, `docs/BENCHMARKS.md`.
  - Assurance & GSN: `docs/{standards_registry.yaml,SAFETY_CASE.md,FHA_FMEA_TRACEABILITY.md,MATH_DERIVATIONS.md}`, `verification/ODD.yaml`, `verification/gsn/{vertirta_safety_case.yaml,build_gsn.py}`.
  - Verification Pyramid: `verification/monte_carlo/{atlas_sampler.py,atlas_stats.py,cliff_ledger.py,run_tier1_6dof_atlas.py,run_tier2_gazebo_atlas.py,run_tier3_lidar_subset.py,plot_survival_atlas.py}`.
  - Real-World Istanbul Viz: `viz/geo/{vertirta_geodesy.py,airspace_polygons.py,build_istanbul_corridor.py,data/.gitignore}`, `viz/cesium/{vrta_ws_server.py,index.html,cesium_live_ops.js,config.example.json}`, `viz/blender/{setup_colab_blender.sh,procedural_coastal_pbr.py,apply_vrta_and_render.py,render_hud_and_mux.py,run_150f_slices.sh}`.
  - Test suites: `tests/ci/*`, `tests/verification/*`, `tests/viz/*`.

---

### Task 1: Tier 1 Structural Determinism Gate (No-Heap Link TU, Static Stack DAG, AI Isolation & Cortex-M7 Footprint)

**Files:**
- Create: `tests/determinism/CMakeLists.txt`, `tests/determinism/no_heap_link_trap.cpp`, `tests/determinism/cm7_entry_stub.cpp`, `cmake/arm-none-eabi-cortex-m7.cmake`
- Create: `scripts/ci/check_stack_usage.py`, `scripts/ci/check_embedded_footprint.py`, `scripts/ci/check_arch_isolation.py`
- Test: `tests/ci/test_tier1_structural.py`

**Interfaces:**
- Consumes: Plan 1A `core/flight_core/` (`vertirta::flight_core` static archive).
- Produces: Standalone CMake project `tests/determinism/` building `vertirta_flight_core_noheap_test` & `vertirta_flight_core_cm7.elf`; CLI validators emitting `tier1_structural.json` (`Sn7`) and `arch_isolation_report.json` (`Sn1`).

- [ ] **Step 1: Write failing tests in `tests/ci/test_tier1_structural.py`**
```python
from pathlib import Path
import pytest
from scripts.ci.check_stack_usage import parse_su_and_ci_dag, check_nm_forbidden_symbols
from scripts.ci.check_embedded_footprint import evaluate_berkeley_size
from scripts.ci.check_arch_isolation import scan_forbidden_ai_includes

def test_nm_symbol_ban_catches_aligned_new():
    offenders = check_nm_forbidden_symbols("         U _ZnwmSt11align_val_t\n         U _Znwj\n")
    assert {"_ZnwmSt11align_val_t", "_Znwj"}.issubset(offenders)
    assert not Path("core/flight_core/include/vertirta/flight_core_no_heap.hpp").exists()
    assert "inline void* operator new" not in Path("tests/determinism/no_heap_link_trap.cpp").read_text()

def test_stack_dag_rejects_dynamic_or_missing_su(tmp_path: Path):
    (tmp_path / "a.su").write_text("a.cpp:10:6:foo\t512\tdynamic\n")
    with pytest.raises(ValueError, match="dynamic"):
        parse_su_and_ci_dag(tmp_path, max_frame_bytes=4096, max_dag_bytes=16384)
    (tmp_path / "a.su").unlink()
    with pytest.raises(FileNotFoundError, match=r"\.su"):
        parse_su_and_ci_dag(tmp_path, max_frame_bytes=4096, max_dag_bytes=16384)

def test_cortex_m7_footprint_and_ai_isolation(tmp_path: Path):
    ok = evaluate_berkeley_size("   text    data     bss     dec\n 180000    4096   32768  216864", 262144, 131072)
    assert ok["passed"] is True and ok["flash_bytes"] == 184096 and ok["ram_bytes"] == 36864
    assert evaluate_berkeley_size("   text    data     bss     dec\n 260000    8192    1024  269216", 262144, 131072)["passed"] is False
    (tmp_path / "bad.cpp").write_text('#include "onnxruntime_cxx_api.h"\n')
    assert len(scan_forbidden_ai_includes(tmp_path)) == 1
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/ci/test_tier1_structural.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement Tier-1 CMake/C++ link traps and Python CI validators**
  - `tests/determinism/no_heap_link_trap.cpp`: Standalone TU (zero `inline`!) defining global `operator new`/`new[]` (normal, `std::nothrow_t`, C++17 `std::align_val_t`) + `__wrap_{malloc,calloc,realloc,free}` calling unresolved `extern "C" void vertirta_forbidden_heap_allocation_link_trap();`.
  - `tests/determinism/CMakeLists.txt` & `cm7_entry_stub.cpp` & `cmake/arm-none-eabi-cortex-m7.cmake`: Standalone CMake harness (`-DVERTIRTA_CORE_DIR=../../core/flight_core`, zero edit to Plan 1A root `CMakeLists.txt`) compiling `vertirta_flight_core_noheap_test` (with `-fno-lto -fstack-usage -fcallgraph-info=su,da -Wstack-usage=4096 -Wvla -Werror -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`) and `vertirta_flight_core_cm7.elf` (`-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-lto -fstack-usage -fcallgraph-info=su,da -Wstack-usage=4096 -Wvla -Werror -Wl,--gc-sections`).
  - `scripts/ci/check_stack_usage.py`: `parse_su_and_ci_dag(build_dir: Path, max_frame_bytes: int, max_dag_bytes: int) -> dict` (parses `.su` + `.ci` VCG callgraph; fails on `dynamic`/`bounded`, cycles, or missing `.su`) & `check_nm_forbidden_symbols(nm_stdout: str) -> list[str]` matching `\b(malloc|calloc|realloc|free|_Znwm|_Znam|_Znwj|_Znaj|_ZnwmSt11align_val_t|_ZnamSt11align_val_t|_ZnwjSt11align_val_t|_ZnajSt11align_val_t)\b`.
  - `scripts/ci/check_embedded_footprint.py`: `evaluate_berkeley_size(size_stdout: str, max_flash_bytes: int, max_ram_bytes: int) -> dict` (`flash = text+data < 262144`, `ram = data+bss < 131072`) + SHA-256 JSON output (`Sn7`).
  - `scripts/ci/check_arch_isolation.py`: `scan_forbidden_ai_includes(core_dir: Path) -> list[str]` banning `llama`, `z3`, `onnxruntime`, `tensorrt`, `rclcpp` across both `core/flight_core/` and `core/rta_companion/`, emitting `arch_isolation_report.json` (`Sn1`).
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/ci/test_tier1_structural.py -v` -> PASS (`3 passed`).
- [ ] **Step 5: Commit** — `git add cmake/arm-none-eabi-cortex-m7.cmake tests/determinism/ scripts/ci/check_stack_usage.py scripts/ci/check_embedded_footprint.py scripts/ci/check_arch_isolation.py tests/ci/test_tier1_structural.py && git commit -m "ci: add Tier-1 zero-heap link TU, static stack DAG, AI isolation, and Cortex-M7 footprint gates"`

---

### Task 2: Tiers 2–4 Determinism Verifiers, `BENCHMARKS.md` & `.github/workflows/determinism.yml`

**Files:**
- Create: `scripts/ci/compare_vrta_cross_isa.py`, `scripts/ci/parse_work_units.py`, `scripts/ci/run_monotonic_with_steal.py`, `docs/BENCHMARKS.md`, `.github/workflows/determinism.yml`
- Test: `tests/ci/test_tiers234_determinism.py`

**Interfaces:**
- Consumes: Plan 1A `vertirta_replay_cli`, `vertirta_bench_cli`, golden `.vrta` binary (`128 B` header + `256 B` frames); Task 1 Tier-1 validators.
- Produces: CI JSON artifacts `tier2_cross_isa.json` (`Sn8`), `tier3_work_<arch>.json` (`Sn9`), `tier4_wallclock_<arch>.json` (`Sn10`), and `.github/workflows/determinism.yml`.

- [ ] **Step 1: Write failing tests in `tests/ci/test_tiers234_determinism.py`**
```python
from pathlib import Path
import yaml
from scripts.ci.compare_vrta_cross_isa import compare_replay_records
from scripts.ci.parse_work_units import parse_work_log
from scripts.ci.run_monotonic_with_steal import classify_steal_taint

def test_cross_isa_tolerance_and_zero_mode_divergence():
    assert compare_replay_records([{"mode": 2, "u": [1.0]}], [{"mode": 2, "u": [1.0 + 5e-13]}], 1e-12)["passed"] is True
    assert compare_replay_records([{"mode": 2, "u": [1.0]}], [{"mode": 3, "u": [1.0]}], 1e-12)["passed"] is False

def test_work_unit_parser_handles_perf_and_callgrind_fallback():
    assert parse_work_log("perf_stat", "  450,000,000 instructions:u\n", 1000, 1200000)["passed"] is True
    assert parse_work_log("valgrind_callgrind_fallback", "1,300,000,000  PROGRAM TOTALS\n", 1000, 1200000)["passed"] is False

def test_steal_taint_and_workflow_isolation():
    assert classify_steal_taint(1.01) == "NOISY_NEIGHBOR_TAINTED" and classify_steal_taint(0.45) == "CLEAN"
    wf_txt = Path(".github/workflows/determinism.yml").read_text()
    wf = yaml.safe_load(wf_txt)
    runners = {m["runner"] for m in wf["jobs"]["tier2_3_4_matrix"]["strategy"]["matrix"]["include"]}
    assert runners == {"ubuntu-24.04", "ubuntu-24.04-arm"}
    assert "VERTIRTA_WRAP_MALLOC" not in str(wf["jobs"]["tier2_3_4_matrix"])
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/ci/test_tiers234_determinism.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement Tiers 2–4 scripts, `docs/BENCHMARKS.md` caveat, and `.github/workflows/determinism.yml`**
  - `scripts/ci/compare_vrta_cross_isa.py`: `compare_replay_records(arm64_frames: list[dict], x86_frames: list[dict], max_abs_tol: float = 1e-12) -> dict` + CLI supporting Same-ISA golden SHA-256 (`0-ULP`) and Cross-ISA (`<1e-12`, `0` mode divergence) on `.vrta` files.
  - `scripts/ci/parse_work_units.py`: `parse_work_log(mode: str, log_text: str, ticks: int, max_ir_per_tick: int) -> dict` parsing `perf stat` (`instructions:u`) or automatic `callgrind_annotate` (`PROGRAM TOTALS`) fallback when Azure Hyper-V masks vPMU.
  - `scripts/ci/run_monotonic_with_steal.py`: `classify_steal_taint(steal_pct: float) -> str` (`> 1.0` -> `"NOISY_NEIGHBOR_TAINTED"`) + CLI recording `/proc/stat` `steal` % and `ru_nivcsw` around `vertirta_bench_cli --clock monotonic_raw --ticks 20000`.
  - `docs/BENCHMARKS.md`: Leads with exact `Execution Environment & WCET Scope Caveat` (virtualized Azure Cobalt 100 Neoverse N2 & x86_64 runners; NOT bare-metal RTOS WCET; Tier 1–4 summary table).
  - `.github/workflows/determinism.yml`: Valid YAML (zero broken `||` newlines!), pinned `g++-14`, `ubuntu-24.04` + `ubuntu-24.04-arm`, `-ffp-contract=off -fno-fast-math -fno-associative-math -fno-reciprocal-math -frounding-math`; isolates `-Wl,--wrap=malloc` strictly inside `tier1_embedded_structural` (`tests/determinism/`) and NEVER passes it to `vertirta_replay_cli` / `vertirta_bench_cli` in `tier2_3_4_matrix`.
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/ci/test_tiers234_determinism.py -v` -> PASS (`3 passed`).
- [ ] **Step 5: Commit** — `git add scripts/ci/ docs/BENCHMARKS.md .github/workflows/determinism.yml tests/ci/test_tiers234_determinism.py && git commit -m "ci: add 4-tier ARM64/x86_64 determinism workflow and WCET caveat benchmarks doc"`

---

### Task 3: Aviation Assurance Docs (`ODD.yaml`, `SAFETY_CASE.md`, Corrected ARP4761A FHA/FTA, `MATH_DERIVATIONS.md`) & CI-Linked GSN v3 Generator

**Files:**
- Create: `docs/standards_registry.yaml`, `verification/ODD.yaml`, `docs/SAFETY_CASE.md`, `docs/FHA_FMEA_TRACEABILITY.md`, `docs/MATH_DERIVATIONS.md`, `verification/gsn/vertirta_safety_case.yaml`, `verification/gsn/build_gsn.py`
- Test: `tests/verification/test_assurance_gsn.py`

**Interfaces:**
- Consumes: CI evidence JSONs `Sn1..Sn10` (Tasks 1, 2, 4, 5) and `config/istanbul_ltfj_airspace.yaml` (Plan 1A).
- Produces: Validated `verification/ODD.yaml` (consumed by Tasks 4–5), `odd_part_iam_validation.json` (`Sn3`), and auto-rendered GSN v3 Mermaid block injected into `docs/SAFETY_CASE.md`.

- [ ] **Step 1: Write failing tests in `tests/verification/test_assurance_gsn.py`**
```python
from pathlib import Path
import yaml, pytest
from verification.gsn.build_gsn import validate_odd_yaml, build_gsn_mermaid

def test_odd_yaml_enforces_j2_west_boundary_and_locked_spec():
    odd = yaml.safe_load(Path("verification/ODD.yaml").read_text())
    assert validate_odd_yaml(odd)["passed"] is True
    assert odd["operational_domain"]["scenery_elements"]["corridor_bounding_box"]["lon_deg"][1] < 29.1900
    assert odd["operational_domain"]["scenery_elements"]["egm2008_undulation_m"] == 36.7
    slz = odd["operational_domain"]["slz_requirements"]
    assert (slz["r_slz_min_m"], slz["slope_max_deg"], slz["roughness_sigma_z_max_m"]) == (5.0, 8.0, 0.15)
    assert odd["operational_domain"]["operating_parameters"]["map_staleness_inflate_ms"] == 250

def test_gsn_builder_fails_if_g5_or_g6_developed():
    gsn = yaml.safe_load(Path("verification/gsn/vertirta_safety_case.yaml").read_text())
    mermaid = build_gsn_mermaid(gsn, artifact_hashes={"Sn7": "abc123"}, git_sha="deadbeef")
    assert "G5 [UNDEVELOPED]" in mermaid and "G6 [UNDEVELOPED]" in mermaid and "abc123" in mermaid
    for n in gsn["nodes"]:
        if n["id"] in {"G5", "G6"}:
            n["undeveloped"] = False
            with pytest.raises(ValueError, match="undeveloped"):
                build_gsn_mermaid(gsn, artifact_hashes={}, git_sha="deadbeef")
            n["undeveloped"] = True

def test_safety_case_citations_and_corrected_2oo3_fta():
    sc = Path("docs/SAFETY_CASE.md").read_text()
    for token in ("NOT a certified airborne product", "MOC-5", "5 Nov 2026", "ASTM F3269-21", "2026/010/R", "SHT-İHA", "30 Jul 2026"):
        assert token in sc
    fta = Path("docs/FHA_FMEA_TRACEABILITY.md").read_text()
    assert "3 * lambda_fcc^2 * T" in fta and "3.00e-8" in fta and "lambda_fcc^3 * T^2" not in fta
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/verification/test_assurance_gsn.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Author locked assurance documents and `verification/gsn/build_gsn.py`**
  - `docs/standards_registry.yaml`: Authoritative registry pinning `SC-VTOL-02` (10 Jun 2024), `MOC-4 Iss. 2` (11 Jul 2025), `MOC-5 Iss. 2 FINAL` (5 Nov 2026), `Reg. (EU) 2024/1111` Part-IAM (1 May 2025), `SORA 2.5` (`ED Dec 2025/018/R` + `ED Dec 2026/010/R` 16 Nov 2026 deleting SORA 2.0 on 1 Jan 2027), `SHGM SHT-İHA` (30 Jul 2026, M3/İHATTYS), `ASTM F3269-21`, `EASA AI CP Iss. 02` / `Proposed Iss. 03` (Jun 2026 Draft), `MLEAP` (May 2024), `ED-324/ARP6983` (Draft), `SAE ARP4761A/ED-135` (20 Dec 2023), `SCSC-141C GSN v3`.
  - `verification/ODD.yaml`: MLEAP OD + AI Constituent ODDs (`AI-CONST-DR6-COPILOT` L1B/DAL E, `AI-CONST-DR7-SLZ-PERCEPTION` L1A/2A), `lon_deg: [29.1050, 29.1895]` (`< 29.1900 E` west of `J2`), `egm2008_undulation_m: 36.7`, `r_slz_min_m: 5.0`, `slope_max_deg: 8.0`, `roughness_sigma_z_max_m: 0.15`, `map_staleness_inflate_ms: 250`, Stage-A (25 kg SORA 2.5) vs Stage-B (240 kg SC-VTOL-02) split.
  - `docs/SAFETY_CASE.md`: Exact 5-point `Non-Compliance & Legal Scope Disclaimer`, `ASTM F3269-21` 3-Tier Arbiter (`LATCHED_CEL > LATCHED_CSFL > Hysteresis CBVF > CF`, `T_dwell=1.0 s`, `Delta_hyst=0.05`), SHGM `SHT-İHA` M3/İHATTYS hook, and `<!-- GSN:BEGIN --> ... <!-- GSN:END -->` target block.
  - `docs/FHA_FMEA_TRACEABILITY.md`: 10-row ARP4761A mini-FHA (`FHA-01..10`), Requirement Traceability Matrix (`REQ-SAF-01..12`), and illustrative hardware FTA cut-sets with **corrected 2oo3 Triplex FCC `3 * lambda_fcc^2 * T = 3.00e-8/FH`** (explaining `1oo3` COM-MON or `lambda_fcc <= 1.83e-5/FH`).
  - `docs/MATH_DERIVATIONS.md`: 2-page cold-derivation whiteboard sheet covering all 5 pillars (including corrected WLS 1-motor offloading peak `60.0% / 441.5 N / 1.84x` & upper-rotor-only `+17.6%` wake relief).
  - `verification/gsn/vertirta_safety_case.yaml` & `build_gsn.py`: `validate_odd_yaml(odd: dict) -> dict` (`Sn3`) and `build_gsn_mermaid(gsn: dict, artifact_hashes: dict[str, str], git_sha: str) -> str` raising `ValueError` if `G5` or `G6` is not `undeveloped: true`.
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/verification/test_assurance_gsn.py -v` -> PASS (`3 passed`).
- [ ] **Step 5: Commit** — `git add docs/standards_registry.yaml docs/SAFETY_CASE.md docs/FHA_FMEA_TRACEABILITY.md docs/MATH_DERIVATIONS.md verification/ODD.yaml verification/gsn/ tests/verification/test_assurance_gsn.py && git commit -m "docs: add ODD.yaml, disclaimed safety case, corrected ARP4761A FTA, and GSN v3 builder"`

---

### Task 4: Stratified LHS Atlas Sampler, Non-Inverted Wind Guard, Isolated CSFL/CEL Stats & Failure-Cliff Ledger

**Files:**
- Create: `verification/monte_carlo/atlas_sampler.py`, `verification/monte_carlo/atlas_stats.py`, `verification/monte_carlo/cliff_ledger.py`
- Test: `tests/verification/test_atlas_stats_and_sampler.py`

**Interfaces:**
- Consumes: `verification/ODD.yaml` (Task 3).
- Produces: `generate_stratified_manifest(odd_path: Path, tier: str, seed: int) -> list[dict]`, `compute_wind_profile_mps(w20_mps: float, z_agl_m: float, exposure: str) -> tuple[float, float, float]`, `clopper_pearson_and_wilson(k: int, n: int, alpha: float = 0.05) -> dict`, `summarize_atlas_pools(results: list[dict], alpha: float = 0.05) -> dict`, `write_failure_cliff_ledger(results: list[dict], out_path: Path) -> dict`.

- [ ] **Step 1: Write failing tests in `tests/verification/test_atlas_stats_and_sampler.py`**
```python
import math, pytest
from pathlib import Path
from verification.monte_carlo.atlas_sampler import generate_stratified_manifest, compute_wind_profile_mps
from verification.monte_carlo.atlas_stats import clopper_pearson_and_wilson, summarize_atlas_pools
from verification.monte_carlo.cliff_ledger import write_failure_cliff_ledger

def test_clopper_pearson_exact_values():
    assert math.isclose(clopper_pearson_and_wilson(0, 1000)["cp_one_sided_upper"], 0.00299125, rel_tol=1e-4)
    s650 = clopper_pearson_and_wilson(0, 650)
    assert math.isclose(s650["cp_one_sided_upper"], 0.0045979, rel_tol=1e-4)
    assert math.isclose(s650["wilson_two_sided"][1], 0.005875, rel_tol=1e-3)

def test_urban_wind_shear_never_exceeds_coastal():
    u_coast, sw_c, _ = compute_wind_profile_mps(12.9, 60.0, "coastal_d")
    u_urban, sw_u, _ = compute_wind_profile_mps(12.9, 60.0, "urban_b")
    assert u_urban < u_coast and sw_u >= sw_c

def test_csfl_cel_denominators_strictly_isolated_and_cliff_ledger(tmp_path: Path):
    manifest = generate_stratified_manifest(Path("verification/ODD.yaml"), "tier2_1k", seed=20261122)
    assert len(manifest) == 1000
    rows = [{"run_id": i, "stratum": r["stratum"], "outcome_class": r["outcome_class"], "passed": i != 0,
             "seed": r["seed"], "violated_invariant": "SINK_RATE" if i == 0 else None, "vrta_path": f"r{i}.vrta"}
            for i, r in enumerate(manifest)]
    summary = summarize_atlas_pools(rows)
    assert summary["csfl_pool"]["n"] == 650 and summary["cel_pool"]["n"] == 150
    ledger = write_failure_cliff_ledger(rows, tmp_path / "failure_cliff_ledger.json")
    assert ledger["total_failures"] == 1 and ledger["entries"][0]["vrta_path"] == "r0.vrta"
    rows[0]["stratum"], rows[0]["outcome_class"] = "S4", "CSFL"
    with pytest.raises(ValueError, match="denominator"):
        summarize_atlas_pools(rows)
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/verification/test_atlas_stats_and_sampler.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement LHS sampler, wind guard, statistical engine, and cliff ledger**
  - `atlas_sampler.py`: Deterministic `scipy.stats.qmc.LatinHypercube(seed)` allocating `S1 (100 Nominal)`, `S2 (250 1M / 8 rotors)`, `S3 (250 Cross-Arm 2M / 24 pairs)`, `S4 (150 Same-Arm 2M / 4 booms, CEL)`, `S5 (150 3-IMU TMR)`, `S6 (100 Compound Cliff)` for `tier2_1k` (`10x` for `tier1_10k`). `compute_wind_profile_mps` applies ASCE-7 Exposure D coastal `W_20 * (z/6)^0.14` and physically valid urban Exposure B `min(U_coastal, sqrt(0.70/1.15) * W_20 * (z/10)^0.22)` (permanently fixing `DR-4`'s inverted urban shear!) with `1.35x` mechanical `sigma_w`.
  - `atlas_stats.py`: `clopper_pearson_and_wilson(k, n, alpha=0.05)` (`scipy.stats.beta.ppf` + Wilson score) & `summarize_atlas_pools` enforcing strict `csfl_pool` (`S2+S3+S5`, `n=650`) vs `cel_pool` (`S4`, `n=150`) isolation.
  - `cliff_ledger.py`: `write_failure_cliff_ledger(results, out_path)` logging every failed run's seed, initial state, fault topology, violated invariant, and `.vrta` trace path (zero cherry-picking).
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/verification/test_atlas_stats_and_sampler.py -v` -> PASS (`3 passed`).
- [ ] **Step 5: Commit** — `git add verification/monte_carlo/atlas_sampler.py verification/monte_carlo/atlas_stats.py verification/monte_carlo/cliff_ledger.py tests/verification/test_atlas_stats_and_sampler.py && git commit -m "feat(verification): add LHS Atlas sampler, non-inverted wind guard, and isolated CSFL/CEL stats"`

---

### Task 5: 2-Tier Verification Pyramid Orchestrator (10k 6-DOF + 1,000 Hot-Reset PX4/Gazebo + 50 EGL `gpu_lidar`)

**Files:**
- Create: `verification/monte_carlo/run_tier1_6dof_atlas.py`, `verification/monte_carlo/run_tier2_gazebo_atlas.py`, `verification/monte_carlo/run_tier3_lidar_subset.py`, `verification/monte_carlo/plot_survival_atlas.py`
- Test: `tests/verification/test_pyramid_orchestrator.py`

**Interfaces:**
- Consumes: Task 4 (`atlas_sampler`, `atlas_stats`, `cliff_ledger`), Plan 1A (`vertirta_6dof_batch_cli`, headless PX4 v1.16 airframe `12050_gz_vertirta_x8_240` + Gazebo Harmonic `vertirta_x8_240_${K}` on `world_name="kartal_coast"`).
- Produces: `tier1_10k_clopper_pearson.json` (`Sn4`), `tier2_1k_stratified_atlas.json` (`Sn5`), `tier3_50_lidar_cliff_ledger.json` (`Sn6`), A/B tables (*HOCBF vs HJ-CBVF*, *Stock PX4 vs INDI+WLS*), and Atlas heatmaps.

- [ ] **Step 1: Write failing tests in `tests/verification/test_pyramid_orchestrator.py`**
```python
from verification.monte_carlo.run_tier2_gazebo_atlas import build_gz_hot_reset_cmd, partition_worker_manifests
from verification.monte_carlo.run_tier3_lidar_subset import select_tier3_50_scenarios

def test_hot_reset_uses_model_only_keeping_monotonic_clock():
    joined = " ".join(build_gz_hot_reset_cmd(world_name="kartal_coast", worker_id=2))
    assert "reset: {model_only: true}" in joined and "all: true" not in joined and "time_only: true" not in joined

def test_worker_partition_and_tier3_selection():
    rows = [{"run_id": i, "stratum": "S6" if i < 30 else "S2", "margin_m": float(i), "passed": i >= 10} for i in range(100)]
    shards = partition_worker_manifests(rows, num_workers=4)
    assert len(shards) == 4 and sum(len(s) for s in shards) == 100
    subset = select_tier3_50_scenarios(rows)
    assert len(subset) == 50 and {r["run_id"] for r in rows[:25]}.issubset({x["run_id"] for x in subset})
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/verification/test_pyramid_orchestrator.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement Tier-1, Tier-2 hot-reset, Tier-3 EGL `gpu_lidar`, and heatmap runners**
  - `run_tier1_6dof_atlas.py`: Executes `10,000` runs via Plan 1A's `vertirta_6dof_batch_cli` (`<90 s`) plus `--cbf-mode hocbf|hj_cbvf` and `--alloc-mode stock_ca|indi_wls` A/B sweeps -> `tier1_10k_clopper_pearson.json` (`Sn4`).
  - `run_tier2_gazebo_atlas.py`: `build_gz_hot_reset_cmd(world_name, worker_id)` emitting `gz service -s /world/<world>/control --reqtype gz.msgs.WorldControl --reptype gz.msgs.Boolean --timeout 5000 --req 'reset: {model_only: true}'` (keeps sim-clock monotonic for `UXRCE_DDS_SYNCT=1` & avoids `px4-ros2-interface-lib #201/#191`); runs 4 isolated workers (`ROS_DOMAIN_ID=10+k`, `GZ_PARTITION=vertirta_w${k}`, ~45 min) -> `tier2_1k_stratified_atlas.json` (`Sn5`) + `failure_cliff_ledger.json`.
  - `run_tier3_lidar_subset.py`: `select_tier3_50_scenarios(tier2_results)` picks 25 lowest-margin/failed + 25 stratified runs; executes 2 headless EGL `--headless-rendering` `gpu_lidar` workers validating `ROG-Map` ESDF `< 2 ms` -> `tier3_50_lidar_cliff_ledger.json` (`Sn6`).
  - `plot_survival_atlas.py`: Renders survival boundary heatmaps (`fault × wind × SoC`) and A/B CDF plots.
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/verification/test_pyramid_orchestrator.py -v` -> PASS (`2 passed`).
- [ ] **Step 5: Commit** — `git add verification/monte_carlo/ tests/verification/test_pyramid_orchestrator.py && git commit -m "feat(verification): add 2-tier 10k/1k hot-reset + 50-run EGL LiDAR pyramid orchestrator"`

---

### Task 6: ToS-Clean Istanbul Geospatial Builder & Curvature-Compensated Geodesy Bridge (`viz/geo/`)

**Files:**
- Create: `viz/geo/vertirta_geodesy.py`, `viz/geo/airspace_polygons.py`, `viz/geo/build_istanbul_corridor.py`, `viz/geo/data/.gitignore`
- Test: `tests/viz/test_geodesy_and_corridor.py`

**Interfaces:**
- Consumes: `config/istanbul_ltfj_airspace.yaml` (Plan 1A), local `PROJ` `us_nga_egm08_25.tif`.
- Produces: `VertiRTAGeodesyBridge` (`flat_enu_to_cesium_wgs84`, `px4_quat_to_enu_flu`), `build_ltfj_ltbx_geometries()` (exact polygon+arc CTRs), and offline Kartal `tmerc` assets (`kartal_15km_egm2008.tif`, `kartal_15km_s2_l2a.tif`, `kartal_15km_buildings.parquet` in git-ignored `viz/geo/data/`).

- [ ] **Step 1: Write failing tests in `tests/viz/test_geodesy_and_corridor.py`**
```python
import math, pytest
from viz.geo.vertirta_geodesy import VertiRTAGeodesyBridge, assert_valid_istanbul_geoid
from viz.geo.airspace_polygons import build_ltfj_ltbx_geometries, signed_distance_to_ltfj_m
from viz.geo.build_istanbul_corridor import build_gdalwarp_cmd, validate_asset_url_tos

def test_geodesy_rejects_missing_egm2008_grid():
    assert_valid_istanbul_geoid(36.68)
    with pytest.raises(RuntimeError, match="EGM2008"):
        assert_valid_istanbul_geoid(0.0)

def test_curvature_compensated_enu_eliminates_8p82m_corner_float():
    p = VertiRTAGeodesyBridge(fixed_geoid_n_m=36.68).flat_enu_to_cesium_wgs84(7500.0, 7500.0, 0.0)
    assert math.isclose(p["alt_hae_m"], 48.68, abs_tol=0.02) and 8.80 < p["curvature_drop_m"] < 8.85

def test_ltfj_polygon_arc_j2_and_gdalwarp_local_tmerc_nodata():
    geom = build_ltfj_ltbx_geometries()
    assert geom["ltfj"]["ceiling_hae_m"] == pytest.approx(798.58, abs=0.2)
    assert geom["ltbx"]["ceiling_hae_m"] == pytest.approx(646.35, abs=0.2)
    assert signed_distance_to_ltfj_m(40.91546, 29.17299) == pytest.approx(1635.0, abs=40.0)  # SLZ-07 to J2
    cmd = " ".join(build_gdalwarp_cmd("in.tif", "out.tif"))
    assert "+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84" in cmd
    assert "EPSG:32635" not in cmd and "-dstnodata -9999" in cmd and "-dstnodata 0" not in cmd
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        validate_asset_url_tos("https://tiles.maps.eox.at/s2cloudless-2024")
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/viz/test_geodesy_and_corridor.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement geodesy bridge, exact `LTFJ`/`LTBX` polygons, and ToS-clean corridor builder**
  - `vertirta_geodesy.py`: `assert_valid_istanbul_geoid(n_m)` enforcing `20.0 < n_m < 50.0`; `VertiRTAGeodesyBridge` (`pyproj` `EPSG:9518->4979`, `R_GAUSS_KARTAL=6375036.971`, `u_tan = Z_flat - (E^2+N^2)/(2*R_gauss)`, `h_hae = (12.0+Z_flat) + N_local`) + `px4_quat_to_enu_flu`.
  - `airspace_polygons.py`: Exact DHMİ `LTFJ` 5 NM arc (`40.89833 N, 29.30917 E`) + NW straight join to `J2 (40.91279 N, 29.19212 E)` (`798.58 m HAE`) & `LTBX` (`40.99889 N, 29.21611 E`, `646.35 m HAE`), exporting GeoJSON & ENU curtains.
  - `build_istanbul_corridor.py` & `viz/geo/data/.gitignore`: `validate_asset_url_tos(url)` banning `eox.at`, `fabdem`, `bing`, `arcgisonline`, `tile.googleapis.com`; `build_gdalwarp_cmd(src, dst)` warping raw AWS Copernicus GLO-30 (`N40_00_E029_00`) + AWS `sentinel-cogs` L2A (`35TPF`) strictly to Kartal local `tmerc` (`-te -7500 -7500 7500 7500 -tr 10 10 -dstnodata -9999`), plus DuckDB Overture + Geofabrik OSM download/conflation commands (zero ODbL committed; `viz/geo/data/.gitignore` ignores `*` except `!.gitignore`).
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/viz/test_geodesy_and_corridor.py -v` -> PASS (`3 passed`).
- [ ] **Step 5: Commit** — `git add viz/geo/ tests/viz/test_geodesy_and_corridor.py && git commit -m "feat(viz): add ToS-clean Kartal tmerc corridor builder, exact LTFJ/LTBX polygons, and geodesy bridge"`

---

### Task 7: Live Tactical Earth Console (`viz/cesium/`, Port `:5180`, GTX 1650 SUPER `< 1.8 GB` VRAM)

**Files:**
- Create: `viz/cesium/vrta_ws_server.py`, `viz/cesium/index.html`, `viz/cesium/cesium_live_ops.js`, `viz/cesium/config.example.json`
- Test: `tests/viz/test_cesium_ws_and_config.py`

**Interfaces:**
- Consumes: `.vrta` frames (Plan 1A), `VertiRTAGeodesyBridge` & `airspace_polygons.py` (Task 6).
- Produces: HTTP + WebSocket server on `127.0.0.1:5180` (`vrta.ws.v1.2`, clean `footprint_lat_lon_hae` `[lat, lon, hae]` order) and `< 1.8 GB` VRAM CesiumJS live tactical console.

- [ ] **Step 1: Write failing tests in `tests/viz/test_cesium_ws_and_config.py`**
```python
from pathlib import Path
from viz.cesium.vrta_ws_server import encode_ws_telemetry_frame, compute_touchdown_blend_hae

def test_ws_schema_uses_clean_lat_lon_hae_and_touchdown_blend():
    pkt = encode_ws_telemetry_frame(
        seq=10, sim_time_s=1.0, lat_deg=40.886, lon_deg=29.188, alt_amsl_m=12.0,
        geoid_n_m=36.68, agl_m=0.0, q_enu_flu=(1.0, 0.0, 0.0, 0.0),
        active_motor_mask=251, cbvf_ring_lat_lon_hae=[(40.886, 29.188, 48.68)],
    )
    lat, lon, hae = pkt["cbvf_overlay"]["footprint_lat_lon_hae"][0]
    assert pkt["schema_ver"] == "vrta.ws.v1.2" and 40.0 < lat < 42.0 and 28.0 < lon < 30.0 and hae > 45.0
    assert compute_touchdown_blend_hae(48.68, 2.4, 0.0) == 51.08
    assert abs(compute_touchdown_blend_hae(100.0, 2.4, 60.0) - 100.0) < 1e-4

def test_cesium_config_triggers_request_render_and_uses_lat_lon_hae():
    js = Path("viz/cesium/cesium_live_ops.js").read_text()
    for token in ("cacheBytes = 256 * 1024 * 1024", "msaaSamples: 2", "requestRenderMode: true",
                  "viewer.scene.requestRender()", "sampleHeightMostDetailed", "footprint_lat_lon_hae",
                  "basemapProvider", "cesium-credits"):
        assert token in js
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/viz/test_cesium_ws_and_config.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement `:5180` `.vrta` WebSocket streamer and GTX 1650 SUPER CesiumJS console**
  - `vrta_ws_server.py`: `encode_ws_telemetry_frame(...) -> dict` (`footprint_lat_lon_hae` `[lat, lon, hae]`) & `compute_touchdown_blend_hae(h_wgs84, delta_z_surf, agl_m)` (`h_wgs84 + delta_z_surf * exp(-(agl_m/15.0)^2)`); streams `30 Hz` pose / `5 Hz` CBVF on `ws://127.0.0.1:5180/telemetry`.
  - `cesium_live_ops.js`, `index.html`, `config.example.json`: Swappable `basemapProvider` (`esri_key` / `ion_bing` / `azure_maps`, zero hardcoded tokens), GTX 1650 SUPER `< 1.8 GB` VRAM locks (`shadows: false`, `msaaSamples: 2`, `requestRenderMode: true`, `maximumRenderTimeChange: Infinity`, `tileCacheSize = 64`, `osmBuildings.cacheBytes = 256 * 1024 * 1024`, `maximumCacheOverflowBytes = 128 * 1024 * 1024`, `maximumScreenSpaceError = 24`), `viewer.scene.requestRender()` inside `ws.onmessage`, `scene.sampleHeightMostDetailed` Gaussian touchdown blend, PBR coaxial X8 glTF + `NodeTransformationProperty` rotors, `LTFJ`/`LTBX` `WallGraphics` (`depthTestAgainstTerrain = true`), and `#cesium-credits`.
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/viz/test_cesium_ws_and_config.py -v` -> PASS (`2 passed`).
- [ ] **Step 5: Commit** — `git add viz/cesium/ tests/viz/test_cesium_ws_and_config.py && git commit -m "feat(viz): add CesiumJS :5180 live Earth console hardened for GTX 1650 SUPER 4 GB VRAM"`

---

### Task 8: Colab G4 Headless Blender 5.2 LTS Cycles OptiX 4K `.vrta` Cinema & `hevc_nvenc` Synced HUD (`viz/blender/`)

**Files:**
- Create: `viz/blender/setup_colab_blender.sh`, `viz/blender/procedural_coastal_pbr.py`, `viz/blender/apply_vrta_and_render.py`, `viz/blender/render_hud_and_mux.py`, `viz/blender/run_150f_slices.sh`
- Test: `tests/viz/test_blender_pipeline.py`

**Interfaces:**
- Consumes: `.vrta` binary log (Plan 1A), Task 6 ToS-clean Kartal `tmerc` DEM/Sentinel-2/OSM/Overture & `LTFJ`/`LTBX` curtains.
- Produces: Deterministic 4K/1080p 30 fps `VertiRTA_X8_240_Istanbul_4K30_FlightReplay.mp4` (`hevc_nvenc -preset p5`, synced RGBA HUD, 3-regime coaxial rotor VFX, failed-rotor smoke, volumetric 4D CBVF funnel).

- [ ] **Step 1: Write failing tests in `tests/viz/test_blender_pipeline.py`**
```python
from pathlib import Path
from viz.blender.procedural_coastal_pbr import MIX_RGBA_A_IDX, MIX_RGBA_B_IDX, MIX_RGBA_OUT_IDX
from viz.blender.apply_vrta_and_render import parse_blender_script_args, classify_rotor_vfx_regime
from viz.blender.render_hud_and_mux import build_ffmpeg_nvenc_cmd

def test_pbr_mix_nodes_use_rgba_indices_6_7_2():
    assert (MIX_RGBA_A_IDX, MIX_RGBA_B_IDX, MIX_RGBA_OUT_IDX) == (6, 7, 2)
    src = Path("viz/blender/procedural_coastal_pbr.py").read_text()
    assert 'mix_col.inputs["A"]' not in src and 'mix_col.outputs["Result"]' not in src
    assert "inputs[MIX_RGBA_A_IDX]" in src and "outputs[MIX_RGBA_OUT_IDX]" in src

def test_blender_cli_arg_parsing_and_3_regime_rotor_vfx():
    args = parse_blender_script_args(["blender", "-b", "--", "--cycles-device", "OPTIX", "--vrta", "a.vrta", "--samples", "128"])
    assert args.vrta == "a.vrta" and args.samples == 128
    assert (classify_rotor_vfx_regime(2400.0), classify_rotor_vfx_regime(450.0), classify_rotor_vfx_regime(0.0)) == (
        "BLUR_DISC", "ARC_BLADES_STEP5", "STOPPED_BLADE"
    )

def test_ffmpeg_uses_nvenc_p5_and_colab_script_selects_52_lts_or_45_fallback():
    cmd = " ".join(build_ffmpeg_nvenc_cmd("frames/frame_%04d.png", "hud/hud_%04d.png", "out.mp4"))
    assert "-c:v hevc_nvenc" in cmd and "-preset p5" in cmd and "-preset slow" not in cmd
    setup_sh = Path("viz/blender/setup_colab_blender.sh").read_text()
    assert "5.2" in setup_sh and "4.5" in setup_sh and "575" in setup_sh
```
- [ ] **Step 2: Run test to verify it fails** — Run `pytest tests/viz/test_blender_pipeline.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement Colab Blender 5.2 LTS setup, PBR shaders, `.vrta` animator, HUD compositor, and 150-frame slice driver**
  - `setup_colab_blender.sh`: Queries `nvidia-smi` driver major version; installs **Blender 5.2 LTS** (`>= 575`) or automatic **Blender 4.5 LTS** fallback (`535–574`), asserting `OPTIX` Blackwell device visibility.
  - `procedural_coastal_pbr.py`: Exports `MIX_RGBA_A_IDX = 6`, `MIX_RGBA_B_IDX = 7`, `MIX_RGBA_OUT_IDX = 2` (NEVER string `"A"/"B"/"Result"`); builds procedural Kartal concrete rooftop SLZ, terracotta roofs, Marmara Sea specular water, Nishita Istanbul sky (`40.886 N, 29.188 E`), and non-shadow-casting (`visible_shadow=False`, `visible_diffuse=False`) holographic `LTFJ`/`LTBX` CTR curtains.
  - `apply_vrta_and_render.py`: `parse_blender_script_args(argv)` splitting at `--` via `argparse.parse_known_args()`, `classify_rotor_vfx_regime(rpm)` (`>1200 -> BLUR_DISC`, `0 < rpm <= 1200 -> ARC_BLADES_STEP5` with `cycles.motion_steps=5`, `==0 -> STOPPED_BLADE`), OptiX + `use_persistent_data=True` + AgX, `.vrta` keyframing, failed-rotor smoke volume bake, dynamic CBVF funnel + `Shrinkwrap` SLZ ring, and 4-camera 1,800-frame cuts (`1-360` coast, `361-840` chase fault, `841-1320` nadir SLZ, `1321-1800` 85 mm touchdown).
  - `render_hud_and_mux.py`: `Pillow` synced RGBA HUD (`V`, `AGL`, `SoC`, `V_term`, `TMR`, `||u_safe - u_nom||`, 8-rotor mask, `LTFJ`/`LTBX` margins) + `build_ffmpeg_nvenc_cmd` (`hevc_nvenc -preset p5 -rc vbr -cq 18 -tag:v hvc1 -movflags +faststart`).
  - `run_150f_slices.sh`: 3-frame `< 18 GB` VRAM `--cycles-print-stats` preflight + resumable `150`-frame slices (`seq 1 150 1800`) syncing each completed slice to persistent Drive storage.
- [ ] **Step 4: Run test to verify it passes** — Run `pytest tests/viz/test_blender_pipeline.py -v` -> PASS (`3 passed`).
- [ ] **Step 5: Commit** — `git add viz/blender/ tests/viz/test_blender_pipeline.py && git commit -m "feat(viz): add Colab G4 Blender 5.2 LTS Cycles OptiX .vrta 4K cinema and NVENC HUD pipeline"`

---

## Gate 3 Release Verification Checklist (`v1.0.0`)

Execute `pytest tests/ci tests/verification tests/viz -v` (all green), verify `.github/workflows/determinism.yml` passes Tiers 1–4 on `ubuntu-24.04` + `ubuntu-24.04-arm`, run the 2-tier Verification Pyramid (`10k` + `1k` hot-reset + `50` EGL `gpu_lidar`) and `verification/gsn/build_gsn.py` to stamp `Sn1..Sn10` SHA-256 hashes into `docs/SAFETY_CASE.md`, verify `0.00 m AGL` touchdown alignment across Gazebo / CesiumJS `:5180` / Blender 5.2 LTS, and publish the 60 s ToS-clean 4K `.vrta` flight-test cinema.
