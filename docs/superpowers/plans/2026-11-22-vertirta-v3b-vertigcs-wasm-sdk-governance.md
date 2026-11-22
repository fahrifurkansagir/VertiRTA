# VertiRTA v3.0.0 — Plan 4B: Unified VertiGCS (MapLibre + Deck.gl + WASM), nanobind Python SDK, Hardened Docker, Governance & IEEE RA-L Paper 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the `M1 (v3.0.0)` productization, developer-onboarding, and governance stack with zero file overlap against Plan 4A (`firmware/`, `verification/formal/`, `.github/workflows/formal_and_hil.yml`) and zero regression to frozen `v2.0.0` invariants (`28,886 B Flash / 0 B RAM` bare-metal Cortex-M7 `libvertirta_flight_core.a`, `256 B VRTA0001` binary frames, `0` ASTM F3269 isolation violations):
1. A `< 150 KB` single-threaded Emscripten flat-C WebAssembly build of `libvertirta_flight_core` (`wasm/`) running inside a Web Worker with `0` `-pthread`/`coi-serviceworker`/`embind` dependencies.
2. A unified TypeScript/Vite **`VertiGCS`** console (`apps/vertigcs/`) built on **MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4 `MapView (z >= 8)`** behind an `IGeoViewportAdapter` (preserving `viz/cesium/` intact), serving both a zero-install static GitHub Pages `.vrta` replay & live fault-injection sandbox AND the `:5180`/`:8891` live WebSocket operations console.
3. A turnkey **`pip install vertirta`** SDK (`sdk/python/` + `pyproject.toml`) using `nanobind` + `scikit-build-core` + `cibuildwheel v4` (`cp312-abi3`), Gymnasium v1.1+ (`VertiRTA-CoaxialX8-v0` & multi-threaded C++ `VertiVecEnv`), Typer CLI, and relocatable CMake `FILE_SET HEADERS` (`find_package(VertiRTA)`).
4. A hardened multi-stage non-root read-only `docker/` deployment stack (`:5180` / `:8891` / `:8888`).
5. Upgraded governance (`docs/standards_registry.yaml v2.0.0`, `CONTRIBUTING.md` DCO 1.1 + permanent Apache-2.0 `.vrta` commitment + `DR-08..DR-13` license firewall), double-anonymous **Paper 1 (`docs/paper_ral/`)** with paired McNemar/Holm-Bonferroni + `rliable` IQM evaluation on live `artifacts/v1.0.0/tier1_atlas/`, and `.github/workflows/product_and_wasm.yml`.

**Architecture:** Plan 4B wraps the frozen C++17 `vertirta::flight_core` static archive (`core/flight_core/`) strictly from the outside via thin ABI layers (`wasm/vertirta_wasm_abi.cpp` for Emscripten flat-C exports and `sdk/python/src/vertirta_bindings.cpp` for `nanobind`). `apps/vertigcs/` decouples telemetry ingress behind `FlightTransport` (WASM Web Worker + drag-and-drop `.vrta` in Sandbox Mode; binary/JSON WebSocket on `:8891`/`:5180` in Live Mode) and decouples 3D map rendering behind `IGeoViewportAdapter` (`MapLibreDeckAdapter` default; `CesiumBridgeAdapter` preserving `viz/cesium/`). Governance, relocatable CMake, Docker, and Paper 1 statistical harnesses consume existing `artifacts/v1.0.0/` and `config/*.yaml` read-only.

**Tech Stack:**
- **Host Windows (verified local):** `node v24.19.0`, `npm 11.17.0`, `uv 0.13.5`, `Python 3.11+` (`pytest`, `pyyaml`, `scipy`, `numpy`, `typer`, `gymnasium`).
- **WSL2 Ubuntu (verified local):** `gcc/g++ 15.2.0`, `cmake 4.2.3`, `ninja 1.13.2`, `arm-none-eabi-gcc 14.2.1`, `Catch2 v3.7.1`.
- **Portable / CI (`ubuntu-24.04` x86_64):** `emsdk` (`emcc` / LLVM 20+, single-threaded Worker ABI), `nanobind`, `scikit-build-core`, `cibuildwheel v4`, Vite 6 + TypeScript 5.7 + Vitest + MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4 + `pmtiles`, Docker multi-stage (`debian:bookworm-slim`, `nginxinc/nginx-unprivileged`).

**Spec & Locks:**
- `docs/superpowers/specs/2026-11-22-vertirta-v3-v4-expansion-design.md` (`v3.0 -> v4.0` Master Design Spec)
- `docs/superpowers/specs/2026-11-22-vertirta-tier-s-design.md` (`v1.2` FROZEN baseline)
- `docs/research/01-DR08-13-FORENSIC-ERRATA-AND-LOCKS.md` (Canonical Override Table & License/CVE/Part-IS Firewall)
- `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-01..DR-07` baseline locks)

---

## Global Constraints

- **Frozen `flight_core` & `.vrta` Invariant (ZERO Drift):** NEVER edit `core/flight_core/src/**` or `core/flight_core/include/**`. Bare-metal `arm-cortex-m7` archive MUST remain exact `28,886 B Flash / 0 B RAM` (`0` heap symbols), `VrtaFileHeader` exact `128 B` (`magic="VRTA0001"`), `VrtaFrame100Hz` exact `256 B` (`crc32c` offset `252`), and `scripts/ci/check_arch_isolation.py` MUST report `0` violations.
- **Canonical Vehicle & Geodesy Overrides (`01-DR08-13`):** `VertiRTA-X8-240` (`240 kg`, `L=1.35 m`, `R=0.75 m / D=1.50 m`, `I=[85,95,140]`, `112S12P` `68 kg` / `21.77 kWh` / `403.2 V`, `omega_max=240 rad/s`, `v_max=22.0 m/s`, Option B1b flat spin `|Omega_z|=2.2 rad/s`). Kartal ENU datum: `(40.8860 N, 29.1880 E, 12.0 m AMSL, EGM2008 N0 = +36.7 m)`, exact `LTFJ` NW `J2 (40.912786 N, 29.192122 E)` polygon+arc (`max_lon_near_j2 = 29.1895 E`), Kartal Dr. Lutfi Kirdar Sehir Hastanesi active helipad separation cylinder, and zero outside-ODD examples (NEVER use Bosphorus bridge or `40.8880 N` / `N=37.35 m` in CLI/UI defaults).
- **WASM & Frontend Hard Bans:** Single-threaded Emscripten `-sENVIRONMENT=worker` + flat `extern "C"` ABI via `-sEXPORTED_FUNCTIONS=@exported_funcs.txt` (`< 150 KB` `.wasm`, `-ffp-contract=off -fno-fast-math`). **HARD BAN** `-pthread`, `SharedArrayBuffer`, `coi-serviceworker`, and `--bind`/`embind`. Lock `MapLibre GL JS v6 + @deck.gl/maplibre v9.4` to `MapView` at `z >= 8` (restrict `GlobeView` to `z < 6` to neutralize GlobeView `TextLayer`/`IconLayer` culling defect); zero API tokens required; Copernicus GLO-30 attribution (`DOI 10.5270/ESA-c5d3d65`, EGM2008 -> WGS84 HAE); keep `viz/cesium/` intact behind `IGeoViewportAdapter`.
- **Port Isolation & Container Hardening:** Dedicated ports `:5180` (`VertiGCS` HTTP), `:8891` (WebSocket telemetry / Foxglove bridge), `:8888/udp` (`MicroXRCEAgent`). **HARD BAN** `4173, 4174, 8765, 9222, 9223` across frontend, SDK CLI, Docker, and tests. Containers MUST be multi-stage, non-root (`USER 10001:10001`), `read_only: true`, `cap_drop: [ALL]`, `no-new-privileges:true`, with `/healthz` probes.
- **License, Governance & Publication Honesty:** Permanent **Apache-2.0 + DCO 1.1** (NO corporate CLA) on `flight_core`, `.vrta`, and single-operator `VertiGCS`. Enforce full `DR-08..DR-13` license firewall (zero GPL/LGPL/AGPL/NC/ShareAlike/UAVid). **Paper 1 (`docs/paper_ral/`)** MUST be strictly **double-anonymous** (zero author names, affiliations, repo URLs, or personal paths), strictly Pillar 1 + Cortex-M7/ESBMC + U-Space scope (zero LLM/VLM), zero certification claims, and `PENDING_*` tags for unexecuted physical hardware runs (never `[TBD]` or fabricated numbers).
- **Hygiene & Line Endings:** Pure UTF-8 (no BOM) and pure **LF (`\n`)** on every created/edited file (`scripts/ci/check_public_hygiene.py` MUST pass `0` hits).

---

## Review Focus

1. **Emscripten Bloat & GitHub Pages Reload Regression:** Enabling `-pthread` drags in `SharedArrayBuffer`/`coi-serviceworker` (breaking first-load GitHub Pages UX) and linking `embind` adds RTTI/`+11 KB` gzip bloat — pinned in Task 1 (`test_wasm_abi_host.cpp`, `test_wasm_build_contracts.py`) and Task 7 (`product_and_wasm.yml` `< 150 KB` gate).
2. **`@deck.gl/maplibre` GlobeView Culling & Silent Geoid Drift:** Rendering `TextLayer`/`IconLayer` in `GlobeView` at urban zoom clips labels, and omitting EGM2008 `N0 = +36.7 m` shifts terrain/aircraft vertically by `36.7 m` — pinned in Task 2 (`viewport_and_geodesy.test.ts`).
3. **CMake `FILE_SET HEADERS` Breaking Bare-Metal `arm-cortex-m7`:** Touching `core/flight_core/CMakeLists.txt` or root `CMakeLists.txt` carelessly could alter `arm-cortex-m7` flags or break cross-compile `install()` — pinned in Task 4 (`tests/cmake/` standalone consumer + mandatory `arm-cortex-m7` `28,886 B / 0 B RAM` re-verification).
4. **Python `cibuildwheel` Free-Threaded `cp314t` Failure & Non-Deterministic `Gymnasium` Seeding:** `cibuildwheel v3.1+` enables `cp314t-*` by default (incompatible with `cp312-abi3`), and resetting RNG on `seed=None` breaks Gymnasium v1.1+ contracts — pinned in Task 4 (`test_vertirta_sdk.py`).
5. **Double-Anonymous Deanonymization Leak & `standards_registry.yaml` Backwards Compatibility:** Leaving a GitHub URL/username in `docs/paper_ral/` desk-rejects IEEE RA-L, and altering existing `v1.0.0` IDs in `docs/standards_registry.yaml` breaks `tests/verification/test_assurance_gsn.py` — pinned in Task 6 (`test_paper_ral_and_governance.py`).

---

## File Structure & Cross-Plan Interfaces (ZERO Overlap with Plan 4A)

- **Owned Strictly by Plan 4A (`@flight-stack-auditor`, DO NOT TOUCH in Plan 4B):**
  - `firmware/stm32h7_hitl/**`, `firmware/renode_cyphal/**`, `verification/formal/**`, `.github/workflows/formal_and_hil.yml`.
- **Consumed Read-Only by Plan 4B:**
  - `core/flight_core/{include,src}/**` (`vertirta::flight_core`, `CoaxialPlant6Dof`, `IndiWlsController`, `TmrVoter`, `MotorFdiObserver`, `VrtaFileHeader`, `VrtaFrame100Hz`, `crc32c`), `config/{vehicle_x8_240,istanbul_ltfj_airspace}.yaml`, `viz/cesium/**`, `viz/geo/{airspace_polygons,vertirta_geodesy}.py`, `artifacts/v1.0.0/tier1_atlas/{tier1_10k_clopper_pearson,failure_cliff_ledger}.json`.
- **Created / Modified Strictly by Plan 4B:**
  - **Task 1 (WASM ABI):** `wasm/{CMakeLists.txt,vertirta_wasm_abi.hpp,vertirta_wasm_abi.cpp,exported_funcs.txt,build_wasm.sh}`, `tests/wasm/{CMakeLists.txt,test_wasm_abi_host.cpp,test_wasm_build_contracts.py}`.
  - **Task 2 (`VertiGCS` Engine):** `apps/vertigcs/{package.json,tsconfig.json,vite.config.ts,vitest.config.ts,index.html}`, `apps/vertigcs/src/{types/telemetry.ts,geo/istanbul_datum.ts,vrta/vrta_parser.ts,transport/flight_transport.ts,worker/flight_sim.worker.ts,adapters/geo_viewport_adapter.ts,adapters/maplibre_deck_adapter.ts,adapters/cesium_bridge_adapter.ts}`, `apps/vertigcs/tests/{vrta_parser.test.ts,viewport_and_geodesy.test.ts}`.
  - **Task 3 (`VertiGCS` Dual-Mode UI):** `apps/vertigcs/src/{main.ts,ui/hud_store.ts,ui/fault_panel.ts,ui/telemetry_hud.ts,ui/replay_scrubber.ts,styles/vertigcs.css}`, `apps/vertigcs/public/data/istanbul_corridor_airspace.geojson`, `apps/vertigcs/tests/dual_mode_app.test.ts`.
  - **Task 4 (Relocatable CMake + `nanobind` Python SDK):** `CMakeLists.txt` (root, `VERTIRTA_INSTALL` + `3.25`), `CMakePresets.json` (`3.25.0`), `core/flight_core/CMakeLists.txt` (`FILE_SET HEADERS` metadata only), `cmake/VertiRTAConfig.cmake.in`, `tests/cmake/{CMakeLists.txt,consumer_smoke.cpp}`, `pyproject.toml`, `sdk/python/{CMakeLists.txt,src/vertirta_bindings.cpp,vertirta/__init__.py,vertirta/_version.py,vertirta/env.py,vertirta/vec_env.py,vertirta/cli.py}`, `tests/sdk/test_vertirta_sdk.py`.
  - **Task 5 (Hardened Docker Stack):** `docker/{Dockerfile.sim,Dockerfile.vertigcs,nginx.vertigcs.conf,docker-compose.yml,README.md}`, `viz/cesium/ws_server.py` (3-line `/healthz` `200 OK` route), `tests/integration/test_docker_stack_contracts.py`.
  - **Task 6 (Governance + Paper 1 RA-L):** `docs/standards_registry.yaml` (`v2.0.0`), `CONTRIBUTING.md`, `docs/paper_ral/{vertirta_ral_manuscript.tex,references_ral.bib,scripts/__init__.py,scripts/eval_ral_stats.py,generated/ral_stats_summary.json,generated/ral_tables.tex,tests/__init__.py,tests/test_paper_ral_and_governance.py}`.
  - **Task 7 (Productization & WASM CI):** `.github/workflows/product_and_wasm.yml`, `tests/ci/test_product_and_wasm_workflow.py`.

---

### Task 1: Single-Threaded Emscripten Flat-C WASM Core (`wasm/`) & IEEE-754 Parity Gate (`< 150 KB`)

**Files:**
- Create: `wasm/CMakeLists.txt`, `wasm/vertirta_wasm_abi.hpp`, `wasm/vertirta_wasm_abi.cpp`, `wasm/exported_funcs.txt`, `wasm/build_wasm.sh`
- Create: `tests/wasm/CMakeLists.txt`, `tests/wasm/test_wasm_abi_host.cpp`, `tests/wasm/test_wasm_build_contracts.py`

**Interfaces:**
- Consumes: `core/flight_core/` (`vertirta::flight_core` static archive — `CoaxialPlant6Dof`, `IndiWlsController`, `TmrVoter`, `MotorFdiObserver`, `VrtaFileHeader`, `VrtaFrame100Hz`, `crc32c`).
- Produces: Flat `extern "C"` C ABI compiled identically on host `g++` (for immediate WSL2 Catch2 + ASan/UBSan parity tests) and `emcc` (`libvertirta_flight_core.wasm`, `< 150 KB`, `-sENVIRONMENT=worker`, zero `-pthread`/`embind`), plus `< 1 KB` `5'x5'` Istanbul EGM2008 bilinear geoid lookup anchored at `N0 = +36.7 m` at Kartal datum.

- [ ] **Step 1: Write failing host parity test (`tests/wasm/test_wasm_abi_host.cpp`) and build-contract test (`tests/wasm/test_wasm_build_contracts.py`)**
```cpp
// tests/wasm/test_wasm_abi_host.cpp
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vertirta/core/vrta_recorder.hpp"
#include "vertirta_wasm_abi.hpp"

TEST_CASE("WASM flat-C ABI runs deterministic 400 Hz control + fault injection with zero heap") {
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 36.7) == 0);
  REQUIRE(vertirta_get_vrta_size() == 256U);

  // Step 400 ticks (1.0 s) nominal hover
  REQUIRE(vertirta_step_batch(400U, 0.0, 0.0, 0.0) == 0);
  const auto* frame_ptr = reinterpret_cast<const vertirta::core::VrtaFrame100Hz*>(vertirta_get_vrta_ptr());
  REQUIRE(frame_ptr != nullptr);
  REQUIRE(vertirta::core::verify_vrta_frame(*frame_ptr));
  REQUIRE(frame_ptr->active_motor_mask == 0xFFU);
  REQUIRE(frame_ptr->cel_latched == 0U);

  // Inject same-arm dual-motor loss (Arm 0: rotors 0 & 4) -> Option B1b flat spin CEL latch
  REQUIRE(vertirta_inject_fault(VERTIRTA_FAULT_SAME_ARM_DUAL, 0U) == 0);
  REQUIRE(vertirta_step_batch(400U, 3.0, -1.5, 0.0) == 0);
  REQUIRE(vertirta::core::verify_vrta_frame(*frame_ptr));
  REQUIRE(frame_ptr->cel_latched == 1U);
  REQUIRE(std::isfinite(frame_ptr->v_term_v));
  REQUIRE(frame_ptr->v_term_v > 300.0f);

  // Geoid lookup anchored to Kartal EGM2008 +36.7 m
  REQUIRE_THAT(vertirta_egm2008_n_m(40.8860, 29.1880), Catch::Matchers::WithinAbs(36.7, 0.05));
}

TEST_CASE("WASM ABI .vrta parser validates Castagnoli CRC32C and rejects corrupted frames") {
  REQUIRE(vertirta_init(40.8860, 29.1880, 12.0, 36.7) == 0);
  REQUIRE(vertirta_step_batch(4U, 0.0, 0.0, 0.0) == 0);
  uint8_t buf[256]{};
  std::memcpy(buf, vertirta_get_vrta_ptr(), sizeof(buf));
  REQUIRE(vertirta_parse_vrta_frame(buf, sizeof(buf)) == 0);
  buf[42] ^= 0x5AU;  // Flip one payload bit
  REQUIRE(vertirta_parse_vrta_frame(buf, sizeof(buf)) != 0);
}
```
```python
# tests/wasm/test_wasm_build_contracts.py
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def test_wasm_build_bans_pthreads_embind_and_enforces_150kb_cap():
    script = (REPO / "wasm/build_wasm.sh").read_bytes()
    assert b"\r" not in script, "wasm/build_wasm.sh must be pure LF"
    text = script.decode("utf-8")
    for required in ("-sENVIRONMENT=worker", "-sEXPORTED_FUNCTIONS=@", "-fno-rtti", "-fno-exceptions", "-ffp-contract=off", "153600"):
        assert required in text, f"Missing required WASM flag/cap: {required}"
    for banned in ("-pthread", "USE_PTHREADS", "coi-serviceworker", "--bind", "embind", "-ffast-math"):
        assert banned not in text, f"Forbidden WASM token present: {banned}"
    exports = (REPO / "wasm/exported_funcs.txt").read_text(encoding="utf-8")
    for fn in ("_vertirta_init", "_vertirta_reset", "_vertirta_step_batch", "_vertirta_inject_fault", "_vertirta_get_vrta_ptr", "_vertirta_parse_vrta_frame", "_vertirta_egm2008_n_m"):
        assert fn in exports
```

- [ ] **Step 2: Run tests to verify failure** — Run `uv run --python 3.11 --with pytest pytest tests/wasm/test_wasm_build_contracts.py -v` -> FAIL (`FileNotFoundError`).

- [ ] **Step 3: Implement `wasm/` flat-C ABI, static arena, `< 1 KB` Istanbul EGM2008 table, and Emscripten build script**
  - `wasm/vertirta_wasm_abi.hpp`: Pure `extern "C"` header defining fault enums (`VERTIRTA_FAULT_NONE=0`, `VERTIRTA_FAULT_SINGLE_MOTOR=1`, `VERTIRTA_FAULT_SAME_ARM_DUAL=2`, `VERTIRTA_FAULT_IMU_BIAS=3`, `VERTIRTA_FAULT_WIND_GUST=4`) and functions: `vertirta_init`, `vertirta_reset`, `vertirta_step_batch`, `vertirta_inject_fault`, `vertirta_get_vrta_ptr`, `vertirta_get_vrta_size`, `vertirta_parse_vrta_frame`, `vertirta_egm2008_n_m`, `vertirta_eval_cbvf_margin`.
  - `wasm/vertirta_wasm_abi.cpp`: Pre-allocates a static contiguous arena (`alignas(8)` single-instance `CoaxialPlant6Dof`, `IndiWlsController`, `TmrVoter`, `MotorFdiObserver`, `VrtaFrame100Hz` output buffer + `256 B` decode scratch buffer — zero `new`/`malloc`!), links `vertirta::flight_core`, computes exact `LTFJ`/`LTBX` margins and Option B1b `2.2 rad/s` flat-spin recovery on same-arm loss, finalizes every frame with `finalize_vrta_frame()` (`crc32c`), and embeds a `13x8` (`104`-node, `416 B`) `float32` `5'x5'` Istanbul EGM2008 bilinear undulation patch anchored to `+36.7 m` at Kartal `(40.8860 N, 29.1880 E)`.
  - `wasm/exported_funcs.txt`: Response file listing exact C-prefixed exported symbols (`_vertirta_init`, `_vertirta_reset`, `_vertirta_step_batch`, `_vertirta_inject_fault`, `_vertirta_get_vrta_ptr`, `_vertirta_get_vrta_size`, `_vertirta_parse_vrta_frame`, `_vertirta_egm2008_n_m`, `_vertirta_eval_cbvf_margin`).
  - `wasm/CMakeLists.txt`, `tests/wasm/CMakeLists.txt` & `wasm/build_wasm.sh`: `tests/wasm/CMakeLists.txt` builds standalone directly against `core/flight_core` (zero edit to root `CMakeLists.txt` needed before Task 4, matching `tests/determinism/`), compiling `vertirta_wasm_abi` + `test_wasm_abi_host` natively on host `g++` under ASan/UBSan; `wasm/build_wasm.sh` builds the `emcc` ES6 Worker bundle (`apps/vertigcs/public/wasm/libvertirta_flight_core.{mjs,wasm}`), asserting `stat -c%s libvertirta_flight_core.wasm <= 153600` (`< 150 KiB`), zero `SharedArrayBuffer`, and pure strict IEEE-754 flags (`-ffp-contract=off -fno-fast-math`).

- [ ] **Step 4: Run tests to verify pass** — Compile and run `tests/wasm/test_wasm_abi_host.cpp` under WSL2 ASan/UBSan (`-Wall -Wextra -Wpedantic -Werror`) -> `All tests passed`; run `uv run --python 3.11 --with pytest pytest tests/wasm/test_wasm_build_contracts.py -v` -> PASS; run `bash -n wasm/build_wasm.sh` -> exit `0` (and if `emcc` is available, execute `bash wasm/build_wasm.sh` and confirm `.wasm < 150 KB`).

- [ ] **Step 5: Commit** — `git add wasm/ tests/wasm/ && git commit -s -m "wasm: add single-threaded flat-C flight_core WASM ABI and <150 KB parity gate"`

---

### Task 2: Unified `VertiGCS` Core Engine (`apps/vertigcs/` — MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4 + `IGeoViewportAdapter` + Web Worker)

**Files:**
- Create: `apps/vertigcs/package.json`, `apps/vertigcs/tsconfig.json`, `apps/vertigcs/vite.config.ts`, `apps/vertigcs/vitest.config.ts`, `apps/vertigcs/index.html`
- Create: `apps/vertigcs/src/types/telemetry.ts`, `apps/vertigcs/src/geo/istanbul_datum.ts`, `apps/vertigcs/src/vrta/vrta_parser.ts`, `apps/vertigcs/src/transport/flight_transport.ts`, `apps/vertigcs/src/worker/flight_sim.worker.ts`
- Create: `apps/vertigcs/src/adapters/geo_viewport_adapter.ts`, `apps/vertigcs/src/adapters/maplibre_deck_adapter.ts`, `apps/vertigcs/src/adapters/cesium_bridge_adapter.ts`
- Create: `apps/vertigcs/tests/vrta_parser.test.ts`, `apps/vertigcs/tests/viewport_and_geodesy.test.ts`

**Interfaces:**
- Consumes: Task 1 WASM Worker ABI (`libvertirta_flight_core.mjs`/`.wasm`), canonical `VRTA0001` binary format (`128 B` header + `256 B` frames), and `vrta.ws.v1.2` JSON schema from `viz/cesium/ws_server.py`.
- Produces: Zero-token MapLibre v6 + `@deck.gl/maplibre` v9.4 interleaved WebGL2 viewport (`MapView` locked for `z >= 8`), `IGeoViewportAdapter` abstraction preserving `viz/cesium/`, pure TypeScript `.vrta` binary decoder with CRC-32C verification, and transferable `ArrayBuffer` Web Worker transport.

- [ ] **Step 1: Write failing Vitest unit tests (`apps/vertigcs/tests/vrta_parser.test.ts` & `viewport_and_geodesy.test.ts`)**
```ts
// apps/vertigcs/tests/vrta_parser.test.ts
import { describe, expect, it } from "vitest";
import { crc32c, parseVrtaHeader, parseVrtaFrame, VRTA_HEADER_SIZE, VRTA_FRAME_SIZE } from "../src/vrta/vrta_parser";

describe("VRTA0001 binary parser & Castagnoli CRC-32C", () => {
  it("matches RFC 3720 CRC-32C check value 0xE3069283", () => {
    const bytes = new TextEncoder().encode("123456789");
    expect(crc32c(bytes)).toBe(0xe3069283);
  });

  it("decodes canonical 128 B header and 256 B frame and rejects corrupted CRC", () => {
    expect(VRTA_HEADER_SIZE).toBe(128);
    expect(VRTA_FRAME_SIZE).toBe(256);

    const hdrBuf = new ArrayBuffer(128);
    const hv = new DataView(hdrBuf);
    const magic = new TextEncoder().encode("VRTA0001");
    new Uint8Array(hdrBuf, 0, 8).set(magic);
    hv.setUint32(8, 1, true);
    hv.setFloat64(16, 40.8860, true);
    hv.setFloat64(24, 29.1880, true);
    hv.setFloat64(32, 12.0, true);
    hv.setFloat64(40, 36.7, true);
    hv.setUint32(124, crc32c(new Uint8Array(hdrBuf, 0, 124)), true);

    const hdr = parseVrtaHeader(hdrBuf);
    expect(hdr.datumLatDeg).toBeCloseTo(40.8860, 6);
    expect(hdr.egm2008NM).toBeCloseTo(36.7, 4);

    const fBuf = new ArrayBuffer(256);
    const fv = new DataView(fBuf);
    fv.setBigUint64(0, 42n, true);
    fv.setFloat64(16, -150.0, true);
    fv.setFloat32(136, 398.5, true);
    fv.setFloat32(140, 79.2, true);
    fv.setUint8(164, 0xee);
    fv.setUint8(167, 1);
    fv.setUint32(252, crc32c(new Uint8Array(fBuf, 0, 252)), true);

    const frame = parseVrtaFrame(fBuf);
    expect(frame.seq).toBe(42);
    expect(frame.vTermV).toBeCloseTo(398.5, 2);
    expect(frame.celLatched).toBe(true);

    new Uint8Array(fBuf)[16] ^= 0xff;
    expect(() => parseVrtaFrame(fBuf)).toThrow(/CRC32C/i);
  });
});
```
```ts
// apps/vertigcs/tests/viewport_and_geodesy.test.ts
import { describe, expect, it } from "vitest";
import { KARTAL_DATUM, egm2008UndulationM, mslToWgs84HaeM, computeTouchdownBlendHae } from "../src/geo/istanbul_datum";
import { selectProjectionMode, validateLivePort } from "../src/adapters/maplibre_deck_adapter";

describe("Istanbul geodesy, GlobeView culling guard, and port isolation", () => {
  it("anchors Kartal datum to 40.8860 N, 29.1880 E, +36.7 m EGM2008 and Gaussian touchdown blend", () => {
    expect(KARTAL_DATUM.latDeg).toBe(40.8860);
    expect(KARTAL_DATUM.lonDeg).toBe(29.1880);
    expect(egm2008UndulationM(40.8860, 29.1880)).toBeCloseTo(36.7, 2);
    expect(mslToWgs84HaeM(12.0, 40.8860, 29.1880)).toBeCloseTo(48.7, 2);
    expect(computeTouchdownBlendHae(48.7, -1.2, 0.0)).toBeCloseTo(47.5, 4);
    expect(computeTouchdownBlendHae(48.7, -1.2, 60.0)).toBeCloseTo(48.7, 4);
  });

  it("enforces MapView for z >= 8 (neutralizing @deck.gl/maplibre GlobeView culling) and rejects forbidden ports", () => {
    expect(selectProjectionMode(12.5)).toBe("mercator");
    expect(selectProjectionMode(8.0)).toBe("mercator");
    expect(selectProjectionMode(4.5)).toBe("globe");
    expect(validateLivePort(5180)).toBe(5180);
    expect(validateLivePort(8891)).toBe(8891);
    for (const bad of [4173, 4174, 8765, 9222, 9223]) {
      expect(() => validateLivePort(bad)).toThrow(/forbidden/i);
    }
  });
});
```

- [ ] **Step 2: Run tests to verify failure** — Run `npm --prefix apps/vertigcs test` -> FAIL (`package.json` / modules missing).

- [ ] **Step 3: Implement `VertiGCS` package scaffold, `.vrta` parser, geodesy, Web Worker, and viewport adapters**
  - `apps/vertigcs/package.json` & `vite.config.ts`: Pins verified live npm releases `maplibre-gl` (`^6.19.0`), `@deck.gl/core` + `@deck.gl/layers` + `@deck.gl/geo-layers` + `@deck.gl/maplibre` (`^9.4.1`, with `resolve.conditions: ["visgl:webgl-only", "import", "module", "browser", "default"]` stripping WGSL shaders), `pmtiles`, Vite, TypeScript, and Vitest; binds dev server strictly to port `5180` (`strictPort: true`).
  - `apps/vertigcs/src/geo/istanbul_datum.ts`: Exports frozen `KARTAL_DATUM` (`40.8860 N, 29.1880 E, 12.0 m AMSL, N0 = 36.7 m`, `R_gauss = 6375036.971 m`), `< 1 KB` `13x8` `5'x5'` Istanbul EGM2008 bilinear patch, `flatEnuToWgs84()`, `mslToWgs84HaeM()`, `computeTouchdownBlendHae(hWgs84, deltaZSurf, agl, sigma=15.0)`, and Copernicus GLO-30 attribution string (`DOI 10.5270/ESA-c5d3d65`).
  - `apps/vertigcs/src/vrta/vrta_parser.ts`: Zero-dependency little-endian `DataView` parser for `128 B VrtaFileHeader` (`VRTA0001`) + `256 B VrtaFrame100Hz` + reflected Castagnoli `crc32c` (`0x82F63B78`) + streaming bulk file parser for drag-and-drop `.vrta` logs.
  - `apps/vertigcs/src/worker/flight_sim.worker.ts` & `src/transport/flight_transport.ts`: Single-threaded Web Worker loading `public/wasm/libvertirta_flight_core.mjs` (falling back cleanly to a deterministic 240 kg coaxial kinematic stub when `.wasm` has not yet been built in offline npm unit tests) and ping-ponging pre-allocated `256 B` / `5,120 B` `ArrayBuffer`s via `postMessage(buf, [buf])` (zero `SharedArrayBuffer`!). `FlightTransport` unifies `SandboxTransport` (WASM Worker + `.vrta` file replay) and `LiveWsTransport` (`ws://127.0.0.1:8891` or `:5180/telemetry`, validating non-forbidden ports).
  - `apps/vertigcs/src/adapters/{geo_viewport_adapter,maplibre_deck_adapter,cesium_bridge_adapter}.ts`: Defines engine-neutral `IGeoViewportAdapter`; implements `MapLibreDeckAdapter` (`interleaved: true` WebGL2, `selectProjectionMode(zoom)` locking `mercator` for `z >= 8`, zero-token `pmtiles://` + offline dark vector grid fallback, zero-copy `Float32Array` aircraft & translucent 3D CBVF reachability tube layers with scoped `updateTriggers`, `LTFJ`/`LTBX`/hospital helipad exclusion polygons); implements `CesiumBridgeAdapter` forwarding to existing `viz/cesium/` `:5180` enterprise globe mode without duplicating Cesium code.

- [ ] **Step 4: Run tests to verify pass** — Run `npm --prefix apps/vertigcs install && npm --prefix apps/vertigcs test` -> PASS (`vrta_parser.test.ts` + `viewport_and_geodesy.test.ts`).

- [ ] **Step 5: Commit** — `git add apps/vertigcs/ && git commit -s -m "vertigcs: add MapLibre v6 + Deck.gl v9.4 engine, VRTA0001 parser, geodesy, and Worker transport"`

---

### Task 3: `VertiGCS` Dual-Mode UI (GitHub Pages `.vrta`/Fault-Injection Sandbox + `:5180`/`:8891` Live Ops Console)

**Files:**
- Create: `apps/vertigcs/src/main.ts`, `apps/vertigcs/src/ui/hud_store.ts`, `apps/vertigcs/src/ui/fault_panel.ts`, `apps/vertigcs/src/ui/telemetry_hud.ts`, `apps/vertigcs/src/ui/replay_scrubber.ts`, `apps/vertigcs/src/styles/vertigcs.css`
- Create: `apps/vertigcs/public/data/istanbul_corridor_airspace.geojson`
- Create: `apps/vertigcs/tests/dual_mode_app.test.ts`

**Interfaces:**
- Consumes: Task 2 `FlightTransport`, `MapLibreDeckAdapter`, `parseVrtaHeader`/`parseVrtaFrame`, `KARTAL_DATUM`, and frozen `config/istanbul_ltfj_airspace.yaml` geometries (`LTFJ` NW `J2` polygon+arc, `LTBX`, Kartal/Dragos/Maltepe/Cevizli pads, Kartal Dr. Lutfi Kirdar Sehir Hastanesi active helipad).
- Produces: Production static Vite bundle (`apps/vertigcs/dist/`) runnable zero-install on GitHub Pages (Sandbox + drag-and-drop `.vrta` replay) AND live on `:5180`/`:8891` (`?mode=live&ws=ws://127.0.0.1:8891`).

- [ ] **Step 1: Write failing dual-mode UI & airspace GeoJSON tests (`apps/vertigcs/tests/dual_mode_app.test.ts`)**
```ts
// apps/vertigcs/tests/dual_mode_app.test.ts
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { HudRingStore } from "../src/ui/hud_store";
import { parseLaunchMode } from "../src/main";

describe("VertiGCS dual-mode controller, ring-buffer HUD, and frozen Istanbul corridor GeoJSON", () => {
  it("resolves sandbox mode by default and enforces port isolation in live mode", () => {
    expect(parseLaunchMode("")).toEqual({ mode: "sandbox", wsUrl: "ws://127.0.0.1:8891" });
    expect(parseLaunchMode("?mode=live&ws=ws://127.0.0.1:5180/telemetry").mode).toBe("live");
    expect(() => parseLaunchMode("?mode=live&ws=ws://127.0.0.1:4173/telemetry")).toThrow(/forbidden/i);
  });

  it("bounds ring-buffer telemetry history to fixed capacity without unbounded allocations", () => {
    const store = new HudRingStore(64);
    for (let i = 0; i < 200; i++) {
      store.push({ seq: i, vTermV: 400 - i * 0.05, socPct: 80 - i * 0.01, cbvfMinV: 2.5, aglM: 35 });
    }
    expect(store.length).toBe(64);
    expect(store.latest()?.seq).toBe(199);
  });

  it("bundles exact LTFJ NW J2 pinch, LTBX, Kartal/Dragos/Maltepe/Cevizli pads, and Hospital helipad", () => {
    const raw = readFileSync(resolve(__dirname, "../public/data/istanbul_corridor_airspace.geojson"), "utf-8");
    const fc = JSON.parse(raw);
    const ids = new Set(fc.features.map((f: { properties: { id: string } }) => f.properties.id));
    for (const req of ["LTFJ_CTR", "LTBX_CTR", "SLZ_KARTAL", "SLZ_DRAGOS", "SLZ_MALTEPE", "SLZ_CEVIZLI", "HELIPAD_KARTAL_HOSPITAL"]) {
      expect(ids.has(req)).toBe(true);
    }
    expect(raw).not.toContain("29.2057"); // ISTMarina inside LTFJ CTR hard-banned
  });
});
```

- [ ] **Step 2: Run test to verify failure** — Run `npm --prefix apps/vertigcs test` -> FAIL (`hud_store` / `main` / GeoJSON missing).

- [ ] **Step 3: Implement imperative 60 fps HUD, Fault-Injection Panel, `.vrta` Scrubber, and Corridor GeoJSON**
  - `apps/vertigcs/src/ui/hud_store.ts`: Fixed-capacity ring buffer (`default=300` samples = 5 s @ 60 Hz) avoiding GC churn on 4 GB VRAM laptops; tracks 8-rotor thrust bars, 400 V / 22 kWh `V_term` / `SoC`, `CBVF h(x)`, `STL rho(t)`, `LTFJ`/`LTBX` margins, TMR state, CEL latch status, and Z3 UNSAT core diagnostics.
  - `apps/vertigcs/src/ui/{telemetry_hud,fault_panel,replay_scrubber}.ts`: Pure imperative DOM updates scheduled on `requestAnimationFrame` (zero React VDOM overhead!); `fault_panel.ts` provides one-click fault injection buttons (`Rotor #3 Kill`, `Same-Arm Dual Kill (CEL 2.2 rad/s)`, `IMU #1 TMR Fault`, `12 m/s Coastal Gust`, `Reset Nominal`) sending structured commands to the WASM Worker (or WS fault endpoint); `replay_scrubber.ts` handles drag-and-drop `.vrta` upload, play/pause, timeline scrub, and `1x/2x/5x` speed.
  - `apps/vertigcs/public/data/istanbul_corridor_airspace.geojson`: Generated directly from frozen `config/istanbul_ltfj_airspace.yaml` (`LTFJ` 5 NM arc + NW `J1/J2` pinch at `29.192122 E`, `LTBX` 5 NM arc, Kartal/Dragos/Maltepe pads + Cevizli coastal spoke west of `29.188 E` + Kartal Dr. Lutfi Kirdar Sehir Hastanesi active helipad Tier-2 cylinder — strictly excluding ISTMarina `29.2057 E`).
  - `apps/vertigcs/src/main.ts` & `styles/vertigcs.css`: Parses URL query params (`mode=sandbox|live`, `ws=...`, enforcing `validateLivePort`), caps canvas `devicePixelRatio` to `Math.min(window.devicePixelRatio || 1, 1.5)` (matching Design Spec 4 GB GTX 1650 SUPER guardrail), displays mandatory Copernicus GLO-30 attribution (`DOI 10.5270/ESA-c5d3d65`) and non-certification research disclaimer banner.

- [ ] **Step 4: Run tests & production Vite build** — Run `npm --prefix apps/vertigcs test` -> PASS; run `npm --prefix apps/vertigcs run build` -> verify `dist/` succeeds cleanly and contains **zero** occurrences of `coi-serviceworker`, `SharedArrayBuffer`, or forbidden ports (`4173/4174/8765/9222/9223`).

- [ ] **Step 5: Commit** — `git add apps/vertigcs/ && git commit -s -m "vertigcs: add dual-mode sandbox/live UI, fault-injection panel, .vrta scrubber, and Istanbul corridor GeoJSON"`

---

### Task 4: Relocatable Modern CMake (`FILE_SET HEADERS` + `VertiRTAConfig.cmake`) & `pip install vertirta` (`nanobind` + `Gymnasium v1.1+` + Typer CLI)

**Files:**
- Modify: `CMakeLists.txt` (bump `cmake_minimum_required(VERSION 3.25)`, add `VERTIRTA_INSTALL` option & `VertiRTAConfig.cmake` export guarded by `NOT CMAKE_CROSSCOMPILING`), `CMakePresets.json` (`3.25.0`), `core/flight_core/CMakeLists.txt` (declare `FILE_SET HEADERS` metadata only — zero source or compiler flag changes!)
- Create: `cmake/VertiRTAConfig.cmake.in`, `tests/cmake/CMakeLists.txt`, `tests/cmake/consumer_smoke.cpp`
- Create: `pyproject.toml`, `sdk/python/CMakeLists.txt`, `sdk/python/src/vertirta_bindings.cpp`
- Create: `sdk/python/vertirta/{__init__.py,_version.py,env.py,vec_env.py,cli.py}`
- Create: `tests/sdk/test_vertirta_sdk.py`

**Interfaces:**
- Consumes: `vertirta::flight_core` (`CoaxialPlant6Dof`, `IndiWlsController`, `TmrVoter`, `MotorFdiObserver`, `VrtaFileHeader`, `VrtaFrame100Hz`), `viz/cesium/vrta_reader.py`.
- Produces: Relocatable `find_package(VertiRTA REQUIRED COMPONENTS flight_core)` installation verified by `tests/cmake/`; `pip install vertirta` (`nanobind` + `scikit-build-core` + `cibuildwheel v4` `cp312-abi3` skipping `cp31?t-*`), Gymnasium v1.1+ `VertiRTA-CoaxialX8-v0` & multi-threaded C++ `VertiVecEnv`, and flat Typer CLI (`vertirta sim|inspect|atlas|verify-stl|gcs`).

- [ ] **Step 1: Write failing SDK, Gymnasium v1.1+, CLI, and packaging tests (`tests/sdk/test_vertirta_sdk.py`)**
```python
# tests/sdk/test_vertirta_sdk.py
from pathlib import Path
import tomllib
import numpy as np
import pytest
from typer.testing import CliRunner

REPO = Path(__file__).resolve().parents[2]

def test_pyproject_locks_nanobind_abi3_and_skips_free_threaded_cp314t():
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert "scikit-build-core" in " ".join(cfg["build-system"]["requires"])
    assert "nanobind" in " ".join(cfg["build-system"]["requires"])
    assert cfg["tool"]["scikit-build"]["wheel"]["py-api"] == "cp312"
    cibw = cfg["tool"]["cibuildwheel"]
    assert "cp31?t-*" in cibw["skip"]
    assert cibw["build-frontend"] == "uv"

def test_gymnasium_single_and_vec_env_determinism_and_autoreset_modes():
    import gymnasium as gym
    from vertirta.env import VertiRtaCoaxialX8Env
    from vertirta.vec_env import VertiVecEnv

    env1 = VertiRtaCoaxialX8Env(fault_mode="none")
    env2 = VertiRtaCoaxialX8Env(fault_mode="none")
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(obs1, obs2)
    # Gymnasium v1.1+ contract: reset(seed=None) MUST NOT re-seed RNG to constant
    obs_a, _ = env1.reset(seed=None)
    assert obs_a.shape == obs1.shape

    vec = VertiVecEnv(num_envs=4, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    assert vec.metadata["autoreset_mode"] == gym.vector.AutoresetMode.SAME_STEP
    vobs, _ = vec.reset(seed=7)
    assert vobs.shape == (4, env1.observation_space.shape[0])
    next_obs, rew, term, trunc, info = vec.step(np.zeros((4, 4), dtype=np.float32))
    assert next_obs.shape == vobs.shape and rew.shape == (4,)

def test_vertirta_cli_subcommands_and_port_guard(tmp_path: Path):
    from vertirta.cli import app
    runner = CliRunner()
    out_vrta = tmp_path / "smoke.vrta"
    res = runner.invoke(app, ["sim", "--airframe", "x8", "--duration", "0.5", "--out", str(out_vrta)])
    assert res.exit_code == 0 and out_vrta.exists()
    res_insp = runner.invoke(app, ["inspect", str(out_vrta), "--summary"])
    assert res_insp.exit_code == 0 and "VRTA0001" in res_insp.stdout
    res_bad_port = runner.invoke(app, ["gcs", "--port", "4173", "--dry-run"])
    assert res_bad_port.exit_code != 0
    # Verify no out-of-ODD Bosphorus examples remain in CLI help
    res_help = runner.invoke(app, ["verify-stl", "--help"])
    assert "Boğaz" not in res_help.stdout and "Kartal" in res_help.stdout
```

- [ ] **Step 2: Run test to verify failure** — Run `PYTHONPATH=sdk/python:. uv run --python 3.11 --with pytest --with numpy --with gymnasium --with typer pytest tests/sdk/test_vertirta_sdk.py -v` -> FAIL (`FileNotFoundError` / `ModuleNotFoundError`).

- [ ] **Step 3: Implement relocatable CMake `FILE_SET HEADERS`, `nanobind` C++20 bindings, Gymnasium wrappers, and Typer CLI**
  - `CMakeLists.txt`, `CMakePresets.json`, `core/flight_core/CMakeLists.txt`, `cmake/VertiRTAConfig.cmake.in`:
    - Bump minimum CMake to `3.25`; add `target_sources(vertirta_flight_core PUBLIC FILE_SET HEADERS BASE_DIRS include FILES ...)` (zero changes to `src/*.cpp`, `include/**`, or compiler flags!).
    - Export `VertiRTA::flight_core` cleanly under `if(VERTIRTA_INSTALL AND NOT CMAKE_CROSSCOMPILING)` with `CMakePackageConfigHelpers` (`SameMajorVersion`), guarding optional companion/Z3 components behind component checks (`check_required_components(VertiRTA)`).
    - Add standalone `tests/cmake/{CMakeLists.txt,consumer_smoke.cpp}` verifying `find_package(VertiRTA REQUIRED COMPONENTS flight_core)` links cleanly in an external build dir.
  - `pyproject.toml` & `sdk/python/CMakeLists.txt` & `sdk/python/src/vertirta_bindings.cpp`:
    - `scikit-build-core` + `nanobind` (`STABLE_ABI` on `cp312`, skipping `cp31?t-*`, `manylinux_2_28` `x86_64` + native `ubuntu-24.04-arm` `aarch64` + Windows `delvewheel`, `build-frontend = "uv"`).
    - Exposes zero-copy `nb::ndarray` views of `CoaxialPlant6Dof` batch stepping (`step_batch` releasing GIL via `nb::gil_scoped_release` across a fixed worker thread pool), `IndiWlsController`, `VrtaFrame100Hz` serialization, and `.pyi` stub generation hook.
  - `sdk/python/vertirta/{__init__,_version,env,vec_env,cli}.py`:
    - `VertiRtaCoaxialX8Env` (`gymnasium.Env`, registered as `VertiRTA-CoaxialX8-v0` with fault schedules & RTA mode selector; uses compiled `_vertirta_native` extension when built and clean pure-Python fallback contract for editable-only smoke tests while reporting `backend="cpp_nanobind"` vs `"python_fallback"` honestly!).
    - `VertiVecEnv` (`gymnasium.vector.VectorEnv`) implementing all 3 Gymnasium v1.1+ `AutoresetMode` modes (`NEXT_STEP`, `SAME_STEP`, `DISABLED` with `reset_mask`) and independent per-env RNG seeding.
    - `cli.py`: 5 flat Typer subcommands (`sim`, `inspect`, `atlas`, `verify-stl` with Kartal/LTFJ examples, `gcs --port 5180 --ws-port 8891` enforcing port isolation).

- [ ] **Step 4: Run tests & regression gates**
  - In WSL2: configure/build/install `release-linux-x64` into `/tmp/vertirta_install_test`, configure/build/run `tests/cmake/` standalone consumer -> PASS (then `rm -rf /tmp/vertirta_install_test`).
  - Re-verify frozen bare-metal gate: `cmake --preset arm-cortex-m7 && cmake --build --preset arm-cortex-m7` + `check_bare_metal_size.py` (`28,886 B Flash / 0 B RAM` EXACT!) + `check_no_heap_symbols.py` (`0` symbols) + `check_arch_isolation.py` (`0` violations).
  - Run `PYTHONPATH=sdk/python:. uv run --python 3.11 --with pytest --with numpy --with gymnasium --with typer pytest tests/sdk/test_vertirta_sdk.py -v` -> PASS (and build native `nanobind` wheel in WSL2 via `uv build --wheel` / `pip wheel` verifying `backend == "cpp_nanobind"`).

- [ ] **Step 5: Commit** — `git add CMakeLists.txt CMakePresets.json core/flight_core/CMakeLists.txt cmake/VertiRTAConfig.cmake.in tests/cmake/ pyproject.toml sdk/python/ tests/sdk/ && git commit -s -m "sdk: add relocatable FILE_SET HEADERS CMake export and nanobind Gymnasium/CLI Python package"`

---

### Task 5: Hardened Multi-Stage Non-Root `docker/` Stack (`:5180` / `:8891` / `:8888`)

**Files:**
- Create: `docker/Dockerfile.sim`, `docker/Dockerfile.vertigcs`, `docker/nginx.vertigcs.conf`, `docker/docker-compose.yml`, `docker/README.md`
- Modify: `viz/cesium/ws_server.py` (add 3-line `/healthz` `200 OK` JSON route `{"status":"ok"}` alongside existing `:5180` routes — zero regression to `tests/viz/`)
- Create: `tests/integration/test_docker_stack_contracts.py`

**Interfaces:**
- Consumes: Root CMake C++20 targets (`vertirta_replay_cli`, `vertirta_6dof_batch_cli`), full `viz/{__init__.py,cesium/,geo/}` package + read-only `config/istanbul_ltfj_airspace.yaml`, `wasm/build_wasm.sh`, and `apps/vertigcs/` Vite production build.
- Produces: Turnkey hardened `docker compose` stack (`core-sim` default on `:5180` & `:8891`, optional `sitl-px4` profile exposing `:8888/udp`), live `/healthz` `200 OK` on both `vertirta-sim` and `vertigcs-web`, zero build toolchains in runtime stages, zero forbidden ports.

- [ ] **Step 1: Write failing Docker security, runtime dependency, `/healthz`, and port-isolation contract test (`tests/integration/test_docker_stack_contracts.py`)**
```python
# tests/integration/test_docker_stack_contracts.py
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
FORBIDDEN_PORTS = {"4173", "4174", "8765", "9222", "9223"}

def test_docker_compose_enforces_nonroot_readonly_healthchecks_and_port_isolation():
    compose_path = REPO / "docker/docker-compose.yml"
    raw = compose_path.read_bytes()
    assert b"\r" not in raw
    doc = yaml.safe_load(raw.decode("utf-8"))
    services = doc["services"]
    assert {"vertirta-sim", "vertigcs-web"}.issubset(services.keys())

    for name, svc in services.items():
        assert svc.get("read_only") is True, f"{name} must set read_only: true"
        assert "ALL" in svc.get("cap_drop", []), f"{name} must drop ALL capabilities"
        assert "no-new-privileges:true" in svc.get("security_opt", []), f"{name} must set no-new-privileges"
        assert svc.get("healthcheck"), f"{name} must define a healthcheck"
        for p in svc.get("ports", []):
            p_str = str(p)
            for bad in FORBIDDEN_PORTS:
                assert f"{bad}:" not in p_str and f":{bad}" not in p_str, f"Forbidden port {bad} in {name}"

    assert services["vertigcs-web"]["depends_on"]["vertirta-sim"]["condition"] == "service_healthy"

def test_dockerfiles_are_multistage_nonroot_copy_full_viz_and_expose_healthz():
    for df in ("docker/Dockerfile.sim", "docker/Dockerfile.vertigcs"):
        raw = (REPO / df).read_bytes()
        assert b"\r" not in raw
        text = raw.decode("utf-8")
        assert text.count("FROM ") >= 2, f"{df} must be multi-stage"
        assert "USER 10001:10001" in text or "USER 101" in text, f"{df} must switch to explicit non-root USER"
    sim_df = (REPO / "docker/Dockerfile.sim").read_text(encoding="utf-8")
    for req in ("viz/geo", "viz/cesium", "viz/__init__.py", "config/istanbul_ltfj_airspace.yaml"):
        assert req in sim_df, f"Dockerfile.sim missing runtime import dependency: {req}"
    ws_src = (REPO / "viz/cesium/ws_server.py").read_text(encoding="utf-8")
    assert '"/healthz"' in ws_src, "ws_server.py must serve 200 OK on /healthz for Docker healthcheck"
```

- [ ] **Step 2: Run test to verify failure** — Run `uv run --python 3.11 --with pytest --with pyyaml pytest tests/integration/test_docker_stack_contracts.py -v` -> FAIL (`FileNotFoundError`).

- [ ] **Step 3: Implement multi-stage Dockerfiles, `/healthz` route, unprivileged Nginx config, and `docker-compose.yml`**
  - `viz/cesium/ws_server.py`: Add 3-line `elif path == "/healthz": body = b'{"status":"ok"}\n'; ctype = "application/json; charset=utf-8"` inside `_handle_client()` (verify existing `tests/viz/` stays 100% GREEN!).
  - `docker/Dockerfile.sim`: Stage 1 (`debian:bookworm-slim` builder) builds stripped `vertirta_replay_cli` / `vertirta_6dof_batch_cli` + minimal Python runtime (`pyyaml`, `pyproj`, `shapely` only where imported by `viz.geo`); Stage 2 (`debian:bookworm-slim`) copies ONLY stripped binaries + runtime libs + full `viz/__init__.py`, `viz/cesium/`, `viz/geo/` + read-only `config/istanbul_ltfj_airspace.yaml`, creates `USER 10001:10001`, exposes `:8891` serving `/telemetry` and `/healthz` (`200 OK`).
  - `docker/Dockerfile.vertigcs` & `docker/nginx.vertigcs.conf`: Stage 1 (`emscripten/emsdk` builds `< 150 KB` `.wasm`), Stage 2 (`node:22-bookworm-slim` runs `npm ci && npm run build`), Stage 3 (`nginxinc/nginx-unprivileged:1.27-alpine`) serves static `dist/` + `.pmtiles` (`Accept-Ranges: bytes`, zero `gzip` on `.pmtiles` HTTP 206 Range requests!) on `:5180` with `/healthz` endpoint under non-root `USER 101`.
  - `docker/docker-compose.yml`: Defines `vertirta-sim` (`:8891`, `healthcheck` probing `http://127.0.0.1:8891/healthz`) and `vertigcs-web` (`:5180`, `depends_on: {vertirta-sim: {condition: service_healthy}}`) + optional `sitl-px4` profile (`:8888/udp`); enforces `read_only: true`, `cap_drop: [ALL]`, `security_opt: ["no-new-privileges:true"]`, `tmpfs` mounts, `:ro` volume mounts for `.pmtiles`/`.vrta`, and explicit `mem_limit`/`cpus` caps.

- [ ] **Step 4: Run tests to verify pass** — Run `uv run --python 3.11 --with pytest --with pyyaml pytest tests/integration/test_docker_stack_contracts.py tests/viz/ -v` -> PASS (and if `docker` daemon/CLI is available, run `docker compose -f docker/docker-compose.yml config --quiet`).

- [ ] **Step 5: Commit** — `git add docker/ viz/cesium/ws_server.py tests/integration/test_docker_stack_contracts.py && git commit -s -m "docker: add hardened multi-stage non-root VertiGCS stack, full viz runtime deps, and /healthz"`

---

### Task 6: Governance Upgrade (`standards_registry.yaml v2.0.0`, `CONTRIBUTING.md` DCO 1.1) & Double-Anonymous IEEE RA-L Paper 1 (`docs/paper_ral/`)

**Files:**
- Modify: `docs/standards_registry.yaml` (bump `vertirta.standards/v1.0.0` -> `v2.0.0`, appending all 7 new 2025–2026 instruments while preserving all 15 existing `v1.0.0` entries verbatim!)
- Modify: `CONTRIBUTING.md` (add explicit DCO 1.1 No-CLA policy, permanent Apache-2.0 commitment for `libvertirta_flight_core`/`.vrta`/`VertiGCS`, and `DR-08..DR-13` license/CVE firewall pins)
- Create: `docs/paper_ral/vertirta_ral_manuscript.tex`, `docs/paper_ral/references_ral.bib`
- Create: `docs/paper_ral/scripts/__init__.py`, `docs/paper_ral/scripts/eval_ral_stats.py`
- Create: `docs/paper_ral/generated/ral_stats_summary.json`, `docs/paper_ral/generated/ral_tables.tex`
- Create: `docs/paper_ral/tests/__init__.py`, `docs/paper_ral/tests/test_paper_ral_and_governance.py`

**Interfaces:**
- Consumes: Live `artifacts/v1.0.0/tier1_atlas/{tier1_10k_clopper_pearson.json,failure_cliff_ledger.json}` and `artifacts/v1.0.0/benchmarks/bare_metal_size_cortex_m7.json`.
- Produces: Backwards-compatible `docs/standards_registry.yaml v2.0.0`, DCO 1.1 governance locks in `CONTRIBUTING.md`, reproducible paired McNemar + Holm-Bonferroni + `rliable` IQM bootstrap evaluation script, and double-anonymous 8-page IEEE RA-L manuscript (`docs/paper_ral/`).

- [ ] **Step 1: Write failing governance, double-anonymity, and statistical evaluation tests (`docs/paper_ral/tests/test_paper_ral_and_governance.py`)**
```python
# docs/paper_ral/tests/test_paper_ral_and_governance.py
from pathlib import Path
import re
import yaml

REPO = Path(__file__).resolve().parents[3]

def test_standards_registry_v2_adds_all_seven_2025_2026_instruments():
    reg = yaml.safe_load((REPO / "docs/standards_registry.yaml").read_text(encoding="utf-8"))
    assert reg["registry_version"] == "vertirta.standards/v2.0.0"
    assert reg["compliance_claim"] == "NONE_RESEARCH_TESTBED_ONLY"
    ids = {s["id"] for s in reg["standards"]}
    required_new = {
        "REG-EU-2022-1645-PART-IS",
        "REG-EU-2023-203-PART-IS",
        "CIR-EU-2026-1821",
        "CDR-EU-2020-2034-ERCS",
        "EUROCAE-ED-155A",
        "SHGM-SHY-OLAY",
        "SHGM-SHT-21-EK6",
    }
    assert required_new.issubset(ids)

def test_contributing_locks_dco11_permanent_vrta_and_dr08_13_bans():
    txt = (REPO / "CONTRIBUTING.md").read_text(encoding="utf-8")
    for token in ("DCO 1.1", "No Corporate CLA", "Permanent Apache-2.0", "VRTA0001", "OpenVINS", "VINS-Fusion", "BlenderProc", "piper1-gpl", "UAVid", "libsodium >= 1.0.21", "sherpa-onnx >= 2.0.0"):
        assert token in txt, f"Missing governance/firewall lock in CONTRIBUTING.md: {token}"

def test_paper_ral_is_double_anonymous_honest_and_computes_mcnemar_and_iqm():
    from docs.paper_ral.scripts.eval_ral_stats import compute_mcnemar_holm_bonferroni, compute_iqm_bootstrap, evaluate_v1_atlas
    from scripts.ci.check_public_hygiene import scan_public_hygiene

    tex = (REPO / "docs/paper_ral/vertirta_ral_manuscript.tex").read_text(encoding="utf-8")
    # Strict double-anonymous check: zero repo URLs, personal paths/usernames, placeholders, or Pillar 2/3 leaks
    assert scan_public_hygiene(REPO, tracked_files=["docs/paper_ral/vertirta_ral_manuscript.tex"])["passed"] is True
    for leak in ("github.com/", "mailto:", "TODO", "[TBD]", "Qwen3", "PIDNet"):
        assert leak not in tex, f"Double-anonymous or Paper-1 scope violation in vertirta_ral_manuscript.tex: {leak}"
    assert "Anonymous Authors" in tex
    assert not re.search(r"we (achieve|demonstrate|are) (DO-178C|SC-VTOL|SORA).*certified", tex, re.IGNORECASE)

    # Exact paired McNemar + Holm-Bonferroni sanity check (per-seed mode)
    res = compute_mcnemar_holm_bonferroni([{"name": "A_vs_B", "b": 120, "c": 5}], alpha=0.05)
    assert res[0]["p_exact"] < 1e-10 and res[0]["significant_holm"] is True

    iqm = compute_iqm_bootstrap([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], n_resamples=1000, seed=42)
    assert iqm["ci_low"] <= iqm["iqm"] <= iqm["ci_high"]

    # Summary conservative-bound mode on aggregate tier1_10k_clopper_pearson.json (ZERO fake per-seed synthesis!)
    report = evaluate_v1_atlas(REPO / "artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json")
    assert report["csfl_n"] == 6500 and report["cel_n"] == 1500
    assert report["method"] == "frechet_hoeffding_conservative_bound"
    assert report["iqm_status"] == "PENDING_PER_RUN_EXPORT"
    assert report["holm_bonferroni_pairs"][0]["significant_holm"] is True
```

- [ ] **Step 2: Run test to verify failure** — Run `PYTHONPATH=. uv run --python 3.11 --with pytest --with pyyaml --with scipy --with numpy pytest docs/paper_ral/tests/test_paper_ral_and_governance.py -v` -> FAIL (`AssertionError` / `ModuleNotFoundError`).

- [ ] **Step 3: Upgrade `standards_registry.yaml` & `CONTRIBUTING.md` and build `docs/paper_ral/` + dual-mode statistical script**
  - `docs/standards_registry.yaml`: Bump `registry_version: "vertirta.standards/v2.0.0"`; preserve all 15 existing `v1.0.0` standards unchanged; append the 7 newly surfaced instruments (`REG-EU-2022-1645-PART-IS` 2025-10-16, `REG-EU-2023-203-PART-IS` 2026-02-22 explicitly noting `NIS2 != Part-IS`, `CIR-EU-2026-1821` 2026-07-27, `CDR-EU-2020-2034-ERCS`, `EUROCAE-ED-155A` 2025-03-01, `SHGM-SHY-OLAY` 2026-09-18 / eff. 2027-03-18, `SHGM-SHT-21-EK6` eff. 2026-11-26).
  - `CONTRIBUTING.md`: Add **DCO 1.1 (No Corporate CLA)** section, **Permanent Apache-2.0 Commitment** for `libvertirta_flight_core`, `.vrta` (`VRTA0001`), and single-operator `VertiGCS`, and expand the License & CVE Firewall section with all `DR-08..DR-13` bans and required version pins (`libsodium >= 1.0.21`, `Fast DDS >= 2.14.6 / >= 3.4.1`, `sherpa-onnx >= 2.0.0`, `Basalt`, `XFeat+LighterGlue`).
  - `docs/paper_ral/scripts/eval_ral_stats.py`: Implements strict zero-fabrication dual-mode statistical evaluation:
    1. **Paired per-seed mode (`evaluate_paired_runs()`):** Ingests real `vertirta_6dof_batch_cli` per-run `runs[]` JSON outputs across identical seeds (`seed=20261122`), computing exact empirical discordant counts `(b, c)`, exact two-sided McNemar binomial p-values + Holm-Bonferroni step-down adjusted p-values (`compute_mcnemar_holm_bonferroni()`), and true `rliable`-compatible 25%–75% Interquartile Mean + 1,000-resample stratified percentile 95% bootstrap CIs (`compute_iqm_bootstrap()`).
    2. **Summary conservative-bound mode (`evaluate_v1_atlas()`):** When ingesting aggregate-only `artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json` (which stores pass rates and means without per-seed vectors), computes exact Fréchet–Hoeffding worst-case discordant counts (`c_max = min(fail_A, pass_B)`, `b = c_max + (pass_A - pass_B)`), explicitly tags `"method": "frechet_hoeffding_conservative_bound"` and `"iqm_status": "PENDING_PER_RUN_EXPORT"`, and NEVER synthesizes fake per-seed IQM arrays; emits `docs/paper_ral/generated/{ral_stats_summary.json,ral_tables.tex}` with honest `Pending (Physical STM32H7 HITL)` / `Pending (M2 U-Space)` cells.
  - `docs/paper_ral/{vertirta_ral_manuscript.tex,references_ral.bib}`: Strictly **double-anonymous** IEEE RA-L 8-page manuscript (`\author{Anonymous Authors}`) scoped strictly to Pillar 1 (Coaxial X8 INDI-WLS Option B1b flat spin + 4D HJI-CBVF reachability + 28,886 B Cortex-M7/ESBMC verification + U-Space preemption interface, zero LLM/VLM scope), `\input{generated/ral_tables.tex}`, SOTA baseline matrix (Smeur, `px4_mpc`, Nan RA-L 2022, `safe-control-gym`, Choi CBVF-QP, `hj_reachability`, `DAQP`, `ESBMC`, `rliable`), and upfront non-certification disclaimer.

- [ ] **Step 4: Run tests to verify pass** — Run `PYTHONPATH=. uv run --python 3.11 --with pytest --with pyyaml --with scipy --with numpy pytest docs/paper_ral/tests/test_paper_ral_and_governance.py tests/verification/test_assurance_gsn.py -v` -> verify BOTH new Paper 1/governance tests AND existing `v1.0` `test_assurance_gsn.py` PASS 100%!

- [ ] **Step 5: Commit** — `git add docs/standards_registry.yaml CONTRIBUTING.md docs/paper_ral/ && git commit -s -m "docs: upgrade standards registry to v2.0.0, lock DCO 1.1, and add double-anonymous IEEE RA-L Paper 1 scaffold"`

---

### Task 7: Productization, WASM & SDK CI Pipeline (`.github/workflows/product_and_wasm.yml`)

**Files:**
- Create: `.github/workflows/product_and_wasm.yml`
- Create: `tests/ci/test_product_and_wasm_workflow.py`

**Interfaces:**
- Consumes: Task 1 (`wasm/build_wasm.sh`, `tests/wasm/`), Task 2–3 (`apps/vertigcs/`), Task 4 (`pyproject.toml`, `tests/cmake/`, `tests/sdk/`), Task 5 (`docker/docker-compose.yml`), Task 6 (`docs/paper_ral/tests/`), and existing `scripts/ci/{check_arch_isolation.py,check_public_hygiene.py}`.
- Produces: Self-contained GitHub Actions workflow `.github/workflows/product_and_wasm.yml` on `ubuntu-24.04` (zero overlap with Plan 4A `.github/workflows/formal_and_hil.yml` or existing `determinism.yml` / `vertibench_ci.yml`).

- [ ] **Step 1: Write failing CI workflow contract test (`tests/ci/test_product_and_wasm_workflow.py`)**
```python
# tests/ci/test_product_and_wasm_workflow.py
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]

def test_product_and_wasm_workflow_covers_wasm_150kb_vertigcs_sdk_and_docker():
    wf_path = REPO / ".github/workflows/product_and_wasm.yml"
    raw = wf_path.read_bytes()
    assert b"\r" not in raw, "Workflow must use pure LF"
    wf = yaml.safe_load(raw.decode("utf-8"))
    jobs = wf["jobs"]
    assert {"wasm_flight_core", "vertigcs_frontend", "python_sdk_and_cmake_install", "docker_and_governance"}.issubset(jobs.keys())
    for name, job in jobs.items():
        assert job["runs-on"] == "ubuntu-24.04", f"{name} must run on ubuntu-24.04"
    text = raw.decode("utf-8")
    assert "153600" in text and "build_wasm.sh" in text
    assert "npm test" in text and "npm run build" in text
    assert "tests/cmake" in text and "test_vertirta_sdk.py" in text
    assert "docker compose -f docker/docker-compose.yml config" in text
    assert "check_public_hygiene.py" in text
    # Zero overlap with Plan 4A formal/HIL jobs
    assert "esbmc" not in text.lower() and "renode" not in text.lower()
```

- [ ] **Step 2: Run test to verify failure** — Run `uv run --python 3.11 --with pytest --with pyyaml pytest tests/ci/test_product_and_wasm_workflow.py -v` -> FAIL (`FileNotFoundError`).

- [ ] **Step 3: Implement `.github/workflows/product_and_wasm.yml`**
  - `wasm_flight_core` (`ubuntu-24.04`): installs pinned `emsdk` (`mymindstorm/setup-emsdk`), runs `bash wasm/build_wasm.sh` enforcing `<= 153600 B` (`< 150 KiB`), greps `libvertirta_flight_core.{mjs,wasm}` for zero `SharedArrayBuffer`/`coi-serviceworker`/`embind`, compiles & runs `tests/wasm/test_wasm_abi_host.cpp` under ASan/UBSan, and runs `pytest tests/wasm/test_wasm_build_contracts.py`.
  - `vertigcs_frontend` (`ubuntu-24.04`): Node 22 (`actions/setup-node`), `npm ci` + `npm test` (Vitest) + `npm run build` (Vite) in `apps/vertigcs/`, verifies `dist/` has zero `coi-serviceworker`/`SharedArrayBuffer`/forbidden ports, uploads static GitHub Pages artifact.
  - `python_sdk_and_cmake_install` (`ubuntu-24.04`): builds & installs relocatable `VertiRTA` into `/tmp/vertirta_install`, builds & executes standalone `tests/cmake/` consumer (`find_package(VertiRTA REQUIRED COMPONENTS flight_core)`), builds `nanobind` wheel via `uv`, and runs `pytest tests/sdk/test_vertirta_sdk.py -v`.
  - `docker_and_governance` (`ubuntu-24.04`): runs `docker compose -f docker/docker-compose.yml config --quiet`, runs `pytest tests/integration/test_docker_stack_contracts.py docs/paper_ral/tests/ tests/ci/test_product_and_wasm_workflow.py -v`, and runs `scripts/ci/check_arch_isolation.py` + `scripts/ci/check_public_hygiene.py`.

- [ ] **Step 4: Run full Plan 4B local verification gate**
  - Run `uv run --python 3.11 --with pytest --with pyyaml --with scipy --with numpy --with gymnasium --with typer pytest tests/wasm/test_wasm_build_contracts.py tests/sdk/test_vertirta_sdk.py tests/integration/test_docker_stack_contracts.py docs/paper_ral/tests/ tests/ci/test_product_and_wasm_workflow.py tests/verification/test_assurance_gsn.py -v` -> 100% PASS.
  - Run `npm --prefix apps/vertigcs test && npm --prefix apps/vertigcs run build` -> 100% PASS.
  - Run `python scripts/ci/check_arch_isolation.py --repo-root .` (`0` violations) and `python scripts/ci/check_public_hygiene.py --repo-root .` (`0` hits).
  - Verify `arm-cortex-m7` preset remains exact `28,886 B Flash / 0 B RAM`, `0` heap symbols.

- [ ] **Step 5: Commit** — `git add .github/workflows/product_and_wasm.yml tests/ci/test_product_and_wasm_workflow.py && git commit -s -m "ci: add productization, WASM <150 KB, VertiGCS, SDK, and Docker verification workflow"`
