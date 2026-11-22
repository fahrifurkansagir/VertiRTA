# VertiRTA — Phase 3 (`v3.0.0 -> v4.0.0`) Expansion & Productization Design Specification

- **Status:** **FROZEN CANONICAL SPEC (`M1 v3.0.0` -> `M2 v3.5.0` -> `M3 v4.0.0`)**
- **Date:** `2026-11-22`
- **Baseline Commit:** `27e45cf` (`main`; tags `v1.0.0`, `v2.0.0-alpha`, `v2.0.0`, plus `M0` release-hardening wave)
- **Companion Audit Locks:**
  - `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-01..DR-07` — 31 errata locked)
  - `docs/research/01-DR08-13-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-08..DR-13` — 8 canonical overrides & full `v3.0–v4.0` license/CVE/Part-IS firewall)
  - `docs/superpowers/specs/2026-11-22-vertirta-tier-s-design.md` (`v1.2 FROZEN` — Pillars 1–3 baseline)
- **Governance & License:** Permanent **`Apache-2.0`** + **`DCO 1.1`** (`Signed-off-by:`, strictly NO corporate CLA) on `libvertirta_flight_core`, `.vrta` format/parsers, `VertiRTA-SDK`, and single-operator `VertiGCS`. Zero `GPL`/`LGPL`/`AGPL`/`CC-BY-NC`/`ShareAlike` contamination in any runtime binary, shipped weight, or bundled training dataset.
- **Non-Certification Scope Lock:** `VertiRTA` is an open-source research, simulation, and software-verification testbed ONLY. It makes **ZERO** airworthiness certification or regulatory compliance claims under `DO-178C`, `DO-254`, `DO-330`, `DO-333`, `EASA SC-VTOL-02`, `Part-IAM`, `Part-IS`, `SORA 2.5`, or `SHGM SHT-İHA / SHY-OLAY`.

---

## 1. Ground-Truth Baseline, Toolchain Matrix & Hard Non-Regression Invariants

### 1.1 Shipped `v2.0.0` + `M0` Measured Ground Truth
Every `v3.0–v4.0` module bolts cleanly onto our verified `v2.0.0` foundation without breaking or inflating a single existing invariant:
- **Bare-Metal Flight Core (`libvertirta_flight_core.a`):** `28,886 B` Flash (`11.0%` of `256 KiB`), `0 B` static library RAM (`< 128 KiB`), `1,496 B` max static frame (`<= 4,096 B`, `77` fns, `0` dynamic), `2,960 B` max call-graph DAG (`<= 16,384 B`, `98` nodes, `0` recursion), `0` forbidden heap symbols (`nm` & `arm-none-eabi-nm` + `-Wl,--wrap=malloc/calloc/realloc`).
- **`ASTM F3269-21` Safety Isolation:** `37` core files (`core/{flight_core,rta_companion}`), `0` violations (`scripts/ci/check_arch_isolation.py`); `16/16` CTest (`dev`/`release`), `15/15` ASan/UBSan, `1M+` 4-thread SPSC/Seqlock TSan ops (`0` races), `test_mission_isolation` TSan `0` races.
- **Live 30,000-Sim Tier-1 6-DOF Survival Atlas (`artifacts/v1.0.0/tier1_atlas/`):** `HJ-CBVF+INDI-WLS` **`96.82%` CSFL** (`6,293/6,500`, `k=207`, CP95 upper `3.57%`) vs `HOCBF` `44.85%`; **`100.00%` CEL** (`1,500/1,500`, `k=0`, CP95 upper `0.20%`) vs `Stock CA` `0.00%`; all `244` cliffs logged transparently.
- **Physical Local GPU (`NVIDIA GTX 1650 SUPER`, `TU116`, `4 GB` VRAM):** `artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json` measured `126.0 MiB` VRAM, `8.06 ms` p50 / `9.06 ms` p99 (`122.96 FPS`) on `ORT-CUDA FP32 + CUDA Graph`, while `FP16` regressed **`3.2×` slower (`28.67 ms`)** (`8.00M` deploy params, architecture latency/VRAM benchmark only).
- **Confirmed Local & CI Toolchains:**
  - *Host Windows:* `node v24.19.0`, `npm 11.17.0`, `uv 0.13.5`, `Python 3.14.7` (plus `uv --python 3.11`).
  - *WSL2 Ubuntu:* `gcc/g++ 15.2.0`, `cmake 4.2.3`, `ninja 1.13.2`, **`arm-none-eabi-gcc 14.2.1`** (exact `DR-08` compiler match).
  - *GitHub Actions CI (`ubuntu-24.04` x86_64 & `ubuntu-24.04-arm`):* `emsdk` (`emcc`), `esbmc v8.7` (**strictly x86_64** — ARM64 fails `--32`), `renode v1.17.0` portable `.NET`, `clang-tidy 22`, `cppcheck 2.22.0`, `cibuildwheel v4`, Apptainer offline `OpenFOAM`.

### 1.2 Seven Hard Non-Regression Invariants (`v2.0 -> v4.0`)

| # | Invariant | Non-Negotiable Rule Across `M1`, `M2`, `M3` |
| --- | --- | --- |
| **I1** | **`flight_core` Zero-Bloat Freeze** | **0 bytes added** to `libvertirta_flight_core.a` (`28,886 B` Flash / `0 B` RAM / `0` heap / `<= 2,960 B` Cortex-M7 DAG). All BSP/Cyphal `o1heap` arenas (`32 KiB` DTCM), WASM ABI shims, fleet schedulers, and lift+cruise dynamics live strictly in external targets (`firmware/`, `wasm/`, `core/uspace_fleet/`, `core/airframes/`). |
| **I2** | **`.vrta` Wire Freeze + Sidecar** | `VRTA0001` (`128 B` header + `256 B` 100 Hz Castagnoli CRC-32C frames) stays bit-frozen and strictly **append-only** (`EUROCAE ED-155A` power-loss safety — never requires an end-of-file index). Voice/ASR, Z3 UNSAT cores, GSN events, chained SHA-256, and Ed25519 signatures write strictly to a companion append-only **`.vrta_events.jsonl`** sidecar linked by `(seq, t_sim_us)`. |
| **I3** | **Canonical Vehicle Override (`vehicle_x8_240.yaml`)** | `m = 240.0 kg`, `R = 0.75 m` (`D = 1.50 m`), `L = 1.35 m`, `R_veh = 2.10 m`, `z/D = 0.233`, `eta_coax = 0.85`, `I = diag(85, 95, 140)`, `tau_up/down = 85/120 ms`, `omega_max = 240 rad/s (2,292 RPM)`, `v_max = 22.0 m/s (42.8 kt)`, `1` pax, Option B1b flat spin `|Omega_z| = 2.2 rad/s`, **`112S12P` (`68 kg`, `21.77 kWh`, `403.2 V`, `R0 = 0.1363 Ω`, `R1 = 0.0840 Ω`)**. Overrides ALL `DR-09/10/11/12` parameter drifts (`D=1.65 m`, `112S11P 98 kg`, `10–20°` tilted spin, `1,850 RPM`, `4–6 pax`). `90 kt` cruise applies solely to `M2` `Lift+Cruise`. |
| **I4** | **Canonical Istanbul Geodesy (`istanbul_ltfj_airspace.yaml`)** | Kartal Coastal Vertiport ENU datum **`(40.8860° N, 29.1880° E, 12.0 m AMSL, EGM2008 N = +36.7 m)`**, exact `LTFJ` NW `J2` polygon+arc (`max_lon_near_j2 = 29.1895° E`). Rejects `DR-09` İSTMarina (`29.2057° E`, `+1.14 km` INSIDE `LTFJ`), `DR-11` `N = 37.35 m` / `29.1880° E` slicing Kartal, `DR-12` `40.8880° N`, and `DR-13` Bosphorus CLI demo. Adds Cevizli west of `29.188° E` + **Kartal Dr. Lütfi Kırdar Şehir Hastanesi active helipad** Tier-2 hard separation cylinder. *(Verify `LTBX` ARP `40°59'26"N` vs `40°59'56"N` against DHMİ AIP PDF prior to `M2`.)* |
| **I5** | **PX4 `v1.16` Internal Rate Loop & RTA Slot Cap** | 400 Hz INDI-WLS (`X8` and `Lift+Cruise`) runs **PX4-internal** (`wq:rate_ctrl`, `UXRCE_DDS_SYNCT=1`, `vehicle_gps_position`, `EKF2_GPS_DELAY`) — never offboard over uXRCE-DDS. `CbvfSafetyFilter` (`StaticDaqpSolver<5,32>`, `K <= 16` QP rows) stays untouched; `core/uspace_fleet/` feeds Top-10 CBF-ORCA + 2 FATO + 2 CTR (`<= 14 <= 15` slots). |
| **I6** | **TU116 Empirical Gate & Zero-GPL Runtime Gate** | Every neural inference pipeline targeting local `GTX 1650 SUPER` (`TU116`, `< 300 MiB` VRAM, locked `512×512` nadir) MUST run an empirical **FP32-vs-FP16 latency/VRAM benchmark** (never assume `2×` FP16 speedup). Every runtime C++/WASM/Python binary MUST pass an automated **`ldd`/`nm` zero-GPL symbol gate** (strictly air-gapping offline GPL tools `viz/blender/` and `tools/cfd_offline_gpl/`). |
| **I7** | **Port Isolation & Public Hygiene** | Dedicated ports: **`:5180`** (`VertiGCS` / CesiumJS), **`:8891`** (WebSocket / Foxglove), **`:8888`** (`MicroXRCEAgent`). **FORBIDDEN:** `4173, 4174, 8765, 9222, 9223`. Every commit passes `scripts/ci/check_public_hygiene.py`, `check_arch_isolation.py`, and pure LF (`\n`). |

---

## 2. Unified Three-Release Expansion Architecture & Module Tree

### 2.1 Dependency & Isolation Topology (`M1` -> `M2` -> `M3`)

```text
                       [ UNTRUSTED / ADVISORY / PRODUCT LAYER ]
  M1: apps/vertigcs (MapLibre v6 + Deck.gl v9.4 + <150KB WASM Worker; IGeoViewportAdapter -> viz/cesium)
  M1: sdk/python (nanobind cp312-abi3 + Gymnasium VertiVecEnv + Typer CLI + docker/)
  M3: core/voice (whisper.cpp + sherpa-onnx>=2.0 + 4-stratum Mondrian KnowNo) -> core/copilot (Z3 + STL)
  M3: core/forensics (.vrta + .vrta_events.jsonl -> E5X/ERCS + zero-digit GBNF)
  M3: core/forward_hazard (45 deg split-stack + 77GHz radar) + core/gnss_denied (Basalt + MCUSUM + XFeat)
                                        |  (Lock-free SPSC / Async Action / Read-Only Sidecar)
                                        v
                   [ M2 FLEET & FULL-ENVELOPE PHYSICS LAYER (C++20) ]
  M2: core/uspace_fleet (1-2 Hero PX4 + 19 SoA agents, Top-10 CBF-ORCA, Part-IS 40B Header + Mode B)
  M2: core/fleet_z3 (QF_RDL STN <3ms + bounded |N_local|<=6 QF_LRA <25ms) + vertirta_uspace_node
  M2: core/airframes/lift_cruise (IAirframeDynamics 8+1) + core/wind_acoustics (mmap VRTAWND1 + dBA)
                                        |  (<=14 hazard slots, reference setpoints, eps_rom_p95 / 3sigma CTR)
                                        v
             [ FROZEN ASTM F3269-21 DETERMINISTIC RTA & FLIGHT CORE (v1.0) ]
  libvertirta_rta_companion (100 Hz HJI-CBVF-QP StaticDaqpSolver<5,32>, ROG-Map, WireCapsules, IntruderKf)
  libvertirta_flight_core   (400 Hz C++17: 28,886 B Flash, 0 B RAM, 0 heap, 2,960 B DAG, 256 B VRTA0001)
          |                                          |
          v (M1 Silicon HITL & Cyphal Bus)           v (M1 Supplemental Formal Verification)
  firmware/stm32h7_hitl + firmware/renode_cyphal     verification/formal (ESBMC v8.7 x86_64 + MISRA-Aligned)
```

### 2.2 Canonical Expansion Directory Ownership Map

| Directory / Path | Milestone | License Tier | Runtime Coupling to `flight_core` |
| --- | :---: | --- | --- |
| `firmware/stm32h7_hitl/` | `M1 (v3.0)` | `Apache-2.0` + `Apache-2.0 CMSIS-Core` | Links frozen `libvertirta_flight_core.a` (read-only) |
| `firmware/renode_cyphal/` | `M1 (v3.0)` | `Apache-2.0` + `MIT` (`libcanard`/`o1heap`) | BSP wraps `32 KiB` `o1heap` DTCM outside `flight_core` |
| `verification/formal/` | `M1 (v3.0)` | `Apache-2.0` (CI invokes `ESBMC v8.7`) | Verification harnesses only; zero runtime linkage |
| `wasm/vertirta_wasm_abi.cpp` | `M1 (v3.0)` | `Apache-2.0` | Thin flat `extern "C"` wrapper over `flight_core` |
| `apps/vertigcs/` | `M1 (v3.0)` | `Apache-2.0` (Permanent single-op UI) | Consumes `<150 KB` `.wasm` Worker & `:8891` WS |
| `sdk/python/` + `docker/` | `M1 (v3.0)` | `Apache-2.0` (`nanobind` BSD-3) | Binds `flight_core` + `rta_companion` via `nanobind` |
| `docs/paper_ral/` + `docs/standards_registry.yaml` (`v2.0`) | `M1 (v3.0)` | `Apache-2.0` | Reads frozen `artifacts/` manifests only |
| `core/uspace_fleet/` + `core/fleet_z3/` + `ros2_ws/src/vertirta_uspace_node/` | `M2 (v3.5)` | `Apache-2.0` (`MIT` Z3, `ISC` libsodium) | Feeds `<= 14` slots into `CbvfSafetyFilter` |
| `core/airframes/lift_cruise/` | `M2 (v3.5)` | `Apache-2.0` | Parallel `IAirframeDynamics` target; `0 B` in X8 core |
| `tools/cfd_offline_gpl/` -> `core/wind_acoustics/` | `M2 (v3.5)` | `GPL-3.0` **OFFLINE ONLY** -> `Apache-2.0` runtime | Air-gapped `VRTAWND1` numeric binary only (`0` GPL syms) |
| `core/forward_hazard/` + `core/gnss_denied/` | `M3 (v4.0)` | `Apache-2.0` + `BSD-3` (`Basalt`) + `MIT` (`MegaLoc`) | Implements `IForwardHazardProvider` + `3σ` CTR inflation |
| `core/voice/` + `core/forensics/` + `benchmarks/vertibench_v2/` | `M3 (v4.0)` | `Apache-2.0` + `MIT` (`whisper.cpp`) + `Apache-2.0` (`sherpa-onnx >= 2.0`) | Async advisory + `.vrta_events.jsonl` sidecar only |
| `docs/paper_corl/` | `M3 (v4.0)` | `Apache-2.0` | Double-anonymous CoRL 2027 (zero `UAVid`) |

---

## 3. Milestone `M1 (v3.0.0)` — Productization, Silicon HITL & Formal Verification Core

`M1` converts `VertiRTA` from a verified Linux/WSL2 research repo into a **zero-install interactive browser product**, **`pip install vertirta` C++20 SDK**, **executed Cortex-M7 bare-metal ELF + dual-FCC CAN-FD SIL**, and **supplemental ESBMC formal proof suite**, executed in two strictly zero-overlap implementation plans (`Plan 4A` & `Plan 4B`).

### 3.1 Bare-Metal STM32H753 HITL, Renode OpenCyphal CAN-FD & Formal Verification (`Plan 4A`)
1. **STM32H753 Bare-Metal Target (`firmware/stm32h7_hitl/`):**
   - **Memory Map (`stm32h753_tcm.ld`):** Zero-wait-state `64 KiB` ITCM (`0x00000000`, bottom `1 KiB` `0x00000000–0x000003FF` reserved as unmapped null-pointer trap; vector table relocated to `VTOR = 0x00000400`) + `128 KiB` DTCM (`0x20000000`, deterministic stack + static `flight_core` arenas) + AXI SRAM (`0x24000000`, DMA/Cyphal descriptors).
   - **Mandatory `ES0392 Rev 16` Silicon Errata Locks:**
     - Bootloader checks **`DBGMCU->IDCODE == 0x2003`** (Silicon **Rev V**) before enabling `480 MHz VOS0`; on Rev Y (`0x1003`), refuses `480 MHz VOS0` (bench diagnostic fallback `400 MHz VOS1 / LATENCY = 2`) and latches `rev_y_fdcan_errata_flag = true`, blocking safety-critical FDCAN arming (zero silent **Erratum `2.24.2`** message-RAM corruption).
     - Configures 480 MHz `VOS0` with **`FLASH_ACR.LATENCY = 4, WRHIGHFREQ = 0b10`** (explicitly overriding PX4 `fmu-v6x` `board.h`'s stale `LATENCY = 2`).
     - Configures MPU regions `R0..R7`: bottom `1 KiB` ITCM `AP_NONE / XN=1`, **forbids Write-Through cacheability** (**Erratum `2.1.1`** — use Write-Back or Device/Strongly-Ordered), disables FPU lazy stacking (**`FPCCR.LSPEN = 0`**), unlocks **`DWT->LAR = 0xC5ACCE55`** before reading `DWT->CYCCNT` (with `__DSB()` barrier), and polls **`FPSCR[4:0]`** sticky FP exception bits per tick (**Erratum `2.1.6`**).
   - **Honest Timing Scope:** `DWT->CYCCNT` hardware measurements on physical `NUCLEO-H743ZI2` / `STM32H753` constitute **Empirical High-Water Mark (HWM)** timing ONLY — NEVER certified analytical WCET under `DO-178C`.
   - **Local Verification:** Compiles and links cleanly into a bare-metal `.elf` right now using WSL2 **`arm-none-eabi-gcc 14.2.1`** (`-Os -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard`), proving `libvertirta_flight_core.a` remains `28,886 B / 0 B RAM`.
2. **Dual-FCC OpenCyphal CAN-FD SIL (`firmware/renode_cyphal/`):**
   - Clean-license bus stack: **MIT** `libcanard v3.2.x/v4.x` + **MIT** `o1heap v3.0.0` (`32 KiB` DTCM pool strictly inside BSP wrapper, zero heap inside `flight_core`) + **Apache-2.0** `CMSIS-Core` + **Nunavut C11** DSDL codegen (never experimental C++).
   - Publishes OpenCyphal heartbeat (**Subject-ID `7509`**) and `48 B` (`<= 504 b` single CAN-FD frame) lockstep flight-state telemetry (**Subject-ID `1024`**).
   - Renode **`v1.17.0`** dual-STM32H753 `.repl`/`.resc` harness configured with `SetGlobalQuantum "0.000025"` (`25 us` deterministic quantum) verifies active-FCC fault kill -> standby-FCC takeover in **`< 20 ms`** in CI (labeled strictly **Functional SIL**, never cycle-accurate WCET).
3. **Supplemental Bounded Model Checking & MISRA-Aligned Profile (`verification/formal/`):**
   - **`ESBMC v8.7`** harnesses (`--32 --floatbv --nan-check --overflow-check --bounds-check --pointer-check --k-induction`) covering `tmr_voter`, `indi_wls`, `motor_fdi`, `coax_envelope`, `daqp_qp`, and `cbvf_filter`, executed strictly on **`ubuntu-24.04` x86_64** CI (ARM64 fails `--32`). Explicitly framed as **Supplemental Engineering Verification** (zero `DO-330` TQL-5 TQSP claim; documents Clang front-end vs GCC `14.2.1` target-code discrepancy `FM.A-7[9]`).
   - Static analysis profile combining `clang-tidy 22` + OSS `cppcheck 2.22.0 --platform=unix32 --funsigned-char` labeled strictly **"MISRA-Aligned High-Integrity C++17 Profile"** accompanied by an explicit **5-class gap register** (never claims OSS `MISRA C++:2023` compliance).
   - Dedicated workflow: **`.github/workflows/formal_and_hil.yml`** (`ubuntu-24.04` x86_64).

### 3.2 Unified `VertiGCS`, `<150 KB` WASM, `VertiRTA-SDK`, Docker & Governance (`Plan 4B`)
1. **Single-Threaded Emscripten WASM ABI (`wasm/vertirta_wasm_abi.cpp`):**
   - Compiles `libvertirta_flight_core` single-threaded (`-sENVIRONMENT=worker`, `-Os`, `-fno-exceptions -fno-rtti -ffp-contract=off -fno-fast-math`), exporting a flat `extern "C"` ABI (~8–12 functions: `vertirta_init`, `vertirta_reset`, `vertirta_step_batch`, `vertirta_inject_fault`, `vertirta_get_vrta_ptr`, `vertirta_eval_cbvf`, `vertirta_parse_vrta_frame`) over a static contiguous linear-memory arena.
   - **HARD BANS:** `-pthread` / `SharedArrayBuffer`, `coi-serviceworker` (breaks GitHub Pages first-paint & shift-reload), and `embind` (RTTI bloat + copy overhead). Hard size gate: **`< 150 KB` `.wasm`** verified in CI.
   - Telemetry transfers between the Web Worker and UI thread via zero-copy ping-pong `postMessage(buf, [buf])` (`256 B`/tick single-aircraft; `5.00 KiB`/tick at 20 aircraft × 60 fps).
2. **Unified `VertiGCS` Web Frontend (`apps/vertigcs/`):**
   - Single Vite + TypeScript frontend built on **MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4** (interleaved WebGL2, `visgl:webgl-only` bundle stripping).
   - **Mandatory Engine Invariants:** Lock `MapView` (Mercator) at operational zoom levels **`z >= 8`** (permitting `GlobeView` strictly at `z < 6` to avoid `@deck.gl/maplibre`'s GlobeView `TextLayer`/icon culling defect); compute explicit terrain-relative AGL/MSL inside C++ WASM using an embedded `< 1 KiB` `5'×5'` Istanbul EGM2008 grid anchored to **`N = +36.7 m`** at Kartal (`h_W84 = H_MSL + N`); isolate map rendering behind **`IGeoViewportAdapter`** keeping existing **`viz/cesium/`** 100% intact for enterprise 3D globe / `> 30 km` BVLOS deployments.
   - **Dual Mode via `FlightTransport`:** (a) **Zero-Install GitHub Pages Sandbox** (drag-and-drop `.vrta` replay + live 1M / 2M-Cross / Same-Arm CEL / `LTFJ` fault injection at 60 fps, zero API tokens, self-hosted `.pmtiles` + Copernicus GLO-30 `DOI 10.5270/ESA-c5d3d65` abstraction); (b) **Live Fleet Ops Mode** (`:5180` UI + `:8891` binary WebSocket telemetry & UNSAT-core inspector).
   - **4 GB VRAM Browser Guardrails:** Fixed pre-allocated `Float32Array`/`Uint8Array` ring buffers, zero React state reconciliation inside the 60 Hz animation loop, `devicePixelRatio` clamped `<= 1.5`, scoped Deck.gl `updateTriggers`, and bounded primitive pooling.
3. **`VertiRTA-SDK` (`sdk/python/`), Relocatable CMake & Hardened Docker (`docker/`):**
   - **`pip install vertirta`:** `nanobind` + `scikit-build-core` + `cibuildwheel v4` emitting `manylinux_2_28` (`x86_64`, native `ubuntu-24.04-arm` `aarch64`) and Windows `cp312-abi3` wheels (explicitly skipping free-threaded `cp31?t-*`), generating `.pyi` stubs and zero-copy `nb::ndarray` buffers.
   - **Gymnasium v1.1+ & CLI:** Registers `VertiRTA-CoaxialX8-v0` (`gymnasium.Env`) and multi-threaded C++ `VertiVecEnv` (`gymnasium.vector.VectorEnv` releasing GIL via `nb::gil_scoped_release`, supporting `AutoresetMode.{NEXT_STEP,SAME_STEP,DISABLED}`), plus Typer CLI (`vertirta sim|inspect|atlas|verify-stl|gcs`, using Kartal ODD examples — NEVER Bosphorus).
   - **Relocatable CMake (`VertiRTAConfig.cmake`):** Modern CMake 3.25+ `FILE_SET HEADERS` exporting `VertiRTA::flight_core` cleanly without pulling Z3/Python unless requested via `COMPONENTS`.
   - **Hardened `docker/` Stack:** Digest-pinned multi-stage `Dockerfile` + `docker-compose.yml` (`vertirta-sim`, `vertirta-copilot`, `vertigcs-web` on `nginx-unprivileged :5180`), non-root `USER 10001:10001`, `read_only: true`, `cap_drop: [ALL]`, `/healthz` probes, and strict port isolation (`:5180/:8891/:8888`).
4. **Governance, Standards Registry `v2.0.0` & Paper 1 Scaffold (`docs/paper_ral/`):**
   - Bump `docs/standards_registry.yaml` to **`v2.0.0`** adding all 7 newly audited 2025–2026 instruments: `Part-IS (EU) 2022/1645` (16 Oct 2025), `Part-IS (EU) 2023/203` (22 Feb 2026) + `ED 2025/013–015/R` + `(EU) 2025/2293`, `CIR (EU) 2026/1821` (27 Jul 2026), `CDR (EU) 2020/2034` ERCS, `EUROCAE ED-155A` (Mar 2025), `SHGM SHY-OLAY` (18 Sep 2026, eff. 18 Mar 2027), and `SHGM SHT-21 Ek-6` (eff. 26 Nov 2026).
   - Update `CONTRIBUTING.md` with permanent Apache-2.0 + `DCO 1.1` (no CLA) lock on `flight_core`, `.vrta`, and single-operator `VertiGCS`.
   - Create **`docs/paper_ral/`**: Double-anonymous 8-page IEEE RA-L manuscript scaffold (targeting IROS 2027 transfer window closing **30 Apr 2027** or ICRA 2028; 0 LLM/VLM scope; keeping `docs/paper/` intact as the arXiv/Zenodo omnibus) + reproducible statistical script executing **exact paired McNemar's test + Holm-Bonferroni correction** (`alpha_eff = 0.00833`) and **`rliable` IQM + 95% bootstrap CIs** directly over `artifacts/v1.0.0/tier1_atlas/`!
   - Dedicated workflow: **`.github/workflows/product_and_wasm.yml`** (`ubuntu-24.04` CI for WASM `<150 KB`, `apps/vertigcs` Vite + Vitest, `nanobind` wheel + pytest smoke).

---

## 4. Milestone `M2 (v3.5.0)` — Multi-eVTOL U-Space Fleet & Full-Envelope Physics

### 4.1 Multi-eVTOL U-Space Fleet (`5–20` Hybrid Aircraft), `Part-IS` Cybersecurity & Thermal Vertiports
- **Dual-Fidelity Hybrid Fleet Engine (`core/uspace_fleet/`):** Co-simulates `1–2` Hero PX4 v1.16 + Gazebo Harmonic SITL aircraft alongside `4–19` native Structure-of-Arrays (SoA) C++20 agents reusing `CoaxialPlant6Dof`, `IndiWlsController`, and `CbvfSafetyFilter` (`0.37–0.57 ms/tick` at 100 Hz; `0 B` added to `flight_core`).
- **Anatolian Coastal Vertiport Network & Liquid-Chilled Turnaround:**
  - Four coastal sites strictly west of `29.1895° E` (`Maltepe`, `Dragos`, `Cevizli`, `Kartal`), each with harmonized `8.0 m` TLOF / `12.0 m` FATO / `24.2 m` Safety Area, EASA OFV Type 1 (`12.5%` seaward slope), dedicated zero-dwell **`FATO-EMG`** (`>= 60 m` spacing), `60/75/90 s` wind-conditioned SISO wake spacing, and segregated bidirectional cruise lanes (`70 m` / `105 m` AGL beneath `120 m` ceiling & `457 m` TMA floor).
  - **Electro-Thermal Scaling Lock:** Scales `DR-11`'s pad-side `15°C` active liquid battery chiller (`260 W/K`, 4 chargers/hub, `40/45/50°C` C-rate derating) to our frozen **`68 kg` `112S12P` pack** (`C_th ≈ 70.7 kJ/K`), stabilizing 20 aircraft at `> 60 flights/hr` without multi-cycle thermal ratchet (`25%/20%` U-Space dispatch policy strictly above `10%/12%` airborne reserve).
- **`ASTM F3548-21`, `F3411-22a` / `RFC 9575` & `< 500 ms` Preemption (`ros2_ws/src/vertirta_uspace_node/`):**
  - Local InterUSS-aligned DSS/USSP implementing exact bijective priority translation (**Internal `Priority-0` CEL/CSFL <-> ASTM `priority = 100` `Contingent`**, omitting OVN key for zero-block DSS write; `Priority-1` Revenue <-> `10`; `Priority-2` Repositioning <-> `5`), handling S2 Level-13 DAR false positives via local ENU 4D intersection and tiling NOTAM polygons `< 100 km²` (InterUSS `#1549`).
  - Broadcast & Network Remote ID (`opendroneid-core-c` + `RFC 9575` 5-page Extended DRIP Wrapper = `228 B` in single BLE 5 / Wi-Fi Beacon pack at 1 Hz, pinned to authoritative IANA DRIP SAM codepoints).
  - **Decentralized 3D Ellipsoidal CBF-ORCA + RHR Curl:** Tracks all `19` peers (`R_z = 36 m` downwash ellipsoid) and injects **Top-10 peers + 2 FATO + 2 CTR rows (`<= 14 <= 15` slots)** into each aircraft's frozen `CbvfSafetyFilter`. Priority-0 emergency alerts bypass Fast DDS `RELIABLE` (`177.6 ms` spike) via **`BEST_EFFORT 3×` UDP burst**, achieving **`< 500 ms`** corridor clearing & FATO vacation (`~132 ms` V2V / `~291 ms` USSP p99.9 budget).
- **`Part-IS (EU) 2023/203` Cybersecurity & Replay-Gap Patch (`NIS2 != Part-IS`):**
  - Patches OMG DDS Security 1.2's `AES-GCM-GMAC` monotonic `IV_s` omission by prepending an encrypted **`40 B VertiRtaSecureHeader`** (`16 B` RFC 9374 DET + `20 B` monotonic nonce/TAI/epoch + `4 B` framing) validated by an `O(1)` **`ReplayWindow128`** (`RFC 6479`).
  - **Mode B Crypto Profile:** `AES-256-GCM` + `ReplayWindow128` on 20 Hz V2V kinematics; `Ed25519` on 1 Hz RID and all P0 alerts/OIV mutations (`~1.47%` Cortex-A72 single-core budget); TLS 1.3 mTLS + `RFC 9421` HTTP signatures on REST/WSS. Separates SHM Monte Carlo vs UDP-Crypto DDS profiles (Fast DDS Security forbids SHM).
  - **Mandatory CVE Pins:** **`libsodium >= 1.0.21`** (Dec 2025 Ed25519 subgroup fix) and **`Fast DDS >= 2.14.6` / `>= 3.4.1`** (`CVE-2024-28231`, `CVE-2025-62799`, `CVE-2026-22590`).
  - **Deterministic Cyber Fail-Safe FSM (`IS.I.OR.220`):** `SEC_NOMINAL -> SEC_SUSPECT_PEER` (demote spoofed peer to non-cooperative obstacle `beta_ij = 1.0`, strip P0 rights) or `SEC_USSP_QUARANTINE -> SEC_AUTONOMOUS_RTB` (**`RTO <= 10 s`** to onboard-verified `FATO-EMG` + trigger `IS.I.OR.230` 72h draft occurrence export).
- **Two-Tier Strategic Fleet Z3 (`core/fleet_z3/`):**
  - Permanently bans global fleet STL-MILP (`2^{156,000}` branch-and-bound blowup). Uses **Tier-1 `QF_RDL` STN fast path** (`<= 160` event times, `0` Booleans, `< 3 ms`) and **Tier-2 separate-context bounded local `QF_LRA`** (`|N_local| <= 6`, `try-for 18 ms`, `< 25 ms` p99) with minimal UNSAT-core repair, preventing `smt.arith.solver=3` cross-logic collisions and strictly respecting Regulation `(EU) 2021/664` Art. 10(8)/(9)/(10) preemption rules.

### 4.2 Second Airframe (`Lift+Cruise 8+1`), Air-Gapped OpenFOAM->POD Wakes & Coaxial BVI Acoustics
- **Multi-Airframe Generality (`core/airframes/lift_cruise/`):** Introduces `IAirframeDynamics` and an **`8+1` Lift+Cruise transition eVTOL** (`<= 79.2% T_max` under hover OEI, backward-compatible `4+1` config) with quintic Hermite transition blending and Härkegård WLS running **PX4-internal** (`wq:rate_ctrl`) via a separate PX4 `standard_vtol` overlay (`90 kt` cruise applies exclusively here).
- **GPL-Isolated Urban Canyon CFD -> `mmap` `VRTAWND1` (`tools/cfd_offline_gpl/` -> `core/wind_acoustics/`):**
  - Offline **`OpenFOAM v2412/v2606` (GPL-3.0)** executes strictly in Apptainer CI (`slip` BCs avoiding GL#3226; air-gapped from `dl.openfoam.com` GL#3236) over Dragos–Maltepe–west Kartal strictly west of `29.1895° E`, compressing URANS canyon wakes into a **`13.89 MiB` `VRTAWND1`** POD-RBF lookup table (`< 1.1 us` runtime lookup + `eps_rom_p95` ROM error injected as conservative CBVF disturbance inflation).
  - Enforced by an automated `ldd`/`nm` **zero-GPL symbol gate** proving `core/wind_acoustics/` links zero OpenFOAM libraries.
- **Coaxial BVI / BPM / ISO 9613-1/2:2024 Acoustic Soft Routing (`core/wind_acoustics/`):** Recalibrates coaxial BVI + NASA RP-1218 BPM + Amiet–Lowson + ISO 9613 atmospheric absorption hemisphere tables strictly to frozen **`D = 1.50 m (R = 0.75 m, 1,450 RPM hover)`** (`< 1 ms` 180-node 4D soft-cost planner), encoding **Kartal Dr. Lütfi Kırdar Şehir Hastanesi** as BOTH a hard Tier-2 helipad separation cylinder and a soft acoustic penalty.

---

## 5. Milestone `M3 (v4.0.0)` — GNSS-Denied Autonomy, Forward Hazards, Voice ATC, Forensics & CoRL 2027

### 5.1 Forward Thin-Obstacle Vision + 77 GHz 4D mmWave Radar (`core/forward_hazard/`)
- Implements Plan 3's `IForwardHazardProvider` port using a decoupled **`45°` narrow-FOV IMU-stabilized forward camera** (`1280×320 -> 640×640` horizon split-stack, leaving nadir `512×512` SLZ camera untouched) running Apache-2.0 **`M-LSD-WireSeg` (`clDice` loss)** + **`D-FINE-N`** fused with a custom Gazebo `gz::sensors::Sensor` **`MmWave77GHzSensor`** (modeling Fresnel glint, Bragg backscatter, Doppler radial separation of static catenary wires/cranes vs moving birds/drones, and `-45 dBsm` drop-off past `20°` skew).
- 200 Hz `SE(3)` ring buffer fuses vision+radar into frozen `WireCapsuleTable` (`<= 64`, vehicle radius **`R_veh = 2.10 m`**) and `IntruderTrackerKf` (`<= 16`), enforcing a **`<= 15 m/s` radar-degraded CBVF speed cap**.
- Time-sliced by flight phase (Forward-Hazard in cruise, Nadir SLZ in terminal approach) inside a single-process CUDA context staying **`< 300 MiB`** on `GTX 1650 SUPER` (empirically FP32-vs-FP16 gated).
- **Clean Forward Training Data:** **`TTPLA` (`Apache-2.0`, strictly decoupled forward `45°` wire head ONLY — still banned from `512×512` nadir SLZ)** + `Mendeley TEİAŞ` (`CC BY 4.0`) + `Purdue UAV` (`BSD-3`) + Apache-2.0 Gazebo OGRE2 procedural catenary/crane synthesis.

### 5.2 GNSS-Denied Anti-Spoofing Navigation (`core/gnss_denied/`)
- **Layer 1 (Slow-Ramp Spoof/Jam Detector):** 4-hypothesis **Lagged-Baseline 2D MCUSUM** (`0.25/2.0` land, `0.35/3.4` water thresholds; `0.2–1.4 s` P95 TTD on `0.5–2.0 m/s` slow-ramp spoofing where SNR looks normal; `2.0 s` state rollback) cleanly separating `GNSS_SPOOFED` vs `GNSS_JAMMED`.
- **Layer 2 (Permissive VIO):** GNSS-decoupled **`Basalt` (`BSD-3`)** consuming a `0.80 m` belly stereo baseline + radar altimeter/baro + `PIDNet-S` Marmara water mask (suppressing false sea-surface features). *(Optional `OKVIS2` BSD-3 fallback strictly requires `USE_NN=OFF` and GPL `SuiteSparse/CHOLMOD` stripped; `OpenVINS`/`VINS-Fusion`/`HybVIO`/`ORB-SLAM3` permanently banned.)*
- **Layer 3 (0.5–1 Hz Visual Geo-Registration & Dynamic `3σ` CBVF Inflation):** Matches nadir imagery against a local `0.5–1.0 m` ortho sim MBTiles cache (or licensed GCP; acknowledging `10 m` Sentinel-2 gives `~45 m` CEP50 gross-blunder check only) via **`XFeat + LighterGlue` (`Apache-2.0`)** + **`MegaLoc` (`MIT`)**.
- **Safety Coupling:** Injects dynamic position-uncertainty inflation **`3 * sqrt(n^T P n) + b_map`** plus time-derivative `∂h/∂t` into `CbvfSafetyFilter` near `LTFJ`/`LTBX` CTR walls (`3σ < 25 m` nominal cruise -> `25–65 m` degraded `12–15 m/s` -> `>= 65 m` immediate visual-SLZ abort).

### 5.3 Cockpit-Noise Offline TR/EN Voice ATC, `ECCAIRS2` Forensics, `VertiBench-v2` & CoRL 2027
- **Dual-Tier Offline ASR & Zero-GPL TTS (`core/voice/`):**
  - *ASR (`MIT`):* `Silero VAD v6.2.0` + `whisper.cpp` dual tier (**`whisper-small` Q8_0** `~0.40 s` tactical fast path + **`whisper-large-v3-turbo` Q8_0** `audio_ctx=256–384`, `translate=false`, `no_context=true`, `carry_initial_prompt=true` `~1.16 s` verification path), fine-tuned with LoRA strictly on clean permissive audio (`CV 27.0` CC0 + `ISSAI TSC` MIT/CC-BY + `MUSAN` + `SLR28` + filtered `FSD50K` + Apache-2.0 coaxial BVI DSP up to `2,292 RPM` / `115 Hz` BPF).
  - *TTS (`< 250 ms` TTFA, Zero GPL):* Hard-pinned **`sherpa-onnx >= 2.0.0`** (PR `#4055` removes GPL-3.0 `espeak-ng`/`piper-phonemize`) running `en_US-ljspeech-medium` (PD) + clean Turkish **`canberkkkkkk/ema-lightning` (`Apache-2.0`)** / **`ahakan/okur-tts` (`Apache-2.0 + CC BY 4.0`)** / clean-room trained `vits-piper-tr_TR-vertirta` (**ALL prebuilt `tr_TR` Piper voices & `piper1-gpl` hard-banned**).
- **SNR-Stratified Mondrian Conformal Gate & `VertiBench-v2` (`benchmarks/vertibench_v2/`):**
  - Preserves `v2.0` `VertiBench-v1` (`200/560`) frozen as baseline; builds **`VertiBench-v2`** (`VertiBench-Voice-TR/EN` + `VertiBench-Fleet20`) resolving `DR-13`'s Beta-inversion under-coverage proof with **language-group-conditional `N_cal >= 750–1,000` per language (`1,500–2,000` total)**, **`N_test >= 500–1,000/lang`**, and **4-stratum SNR Mondrian calibration (`S0..S3`, `n_g >= 250/stratum`)**, locking `alpha = 0.05`, MAYDAY/PAN-PAN asymmetric priority (`w_m >= 0.15`), and the `< 5.0%` SoC `LIVENESS_EMERGENCY_COMMIT` override.
- **Deterministic `ECCAIRS2` / `SHY-OLAY` Forensic Reporter (`core/forensics/`):**
  - Consumes append-only `.vrta` + `.vrta_events.jsonl` + GSN v3 digests to generate XSD-valid **`.e5x` XML** aligned with `Regulation (EU) 376/2014`, **`CIR (EU) 2026/1821`** (FATO/U-Space/envelope/cyber/CMU triggers), **`SHGM SHY-OLAY`** (72h ADREP), and **`CDR (EU) 2020/2034` ERCS** 8-barrier scoring:
    - Rotor/flight-control/SW/datalink faults map strictly to CICTT **`SCF-NP`** (not `SCF-PP`).
    - ERCS Barrier `#8` (Low-Energy) = **`0 (N/A)`** on high-energy airborne collision (`MAC`) / upset (`LOC-I`).
    - Emits **draft/`Incident` ONLY** (`Entity 24 / ID 431` — NEVER auto-classifies `Accident` or `Serious Incident`, which trigger statutory SIA investigations).
    - Narrative summarization uses a **zero-digit GBNF grammar (`eccairs2_nodigit.gbnf`)** whose factual slots are populated deterministically from `.vrta` and verified by `verify_numeric_grounding()` (**0.0% numeric hallucination**).
- **Paper 2 (`docs/paper_corl/` — CoRL 2027, May 2027 Deadline):** Double-anonymous 8-page manuscript covering Bilingual Z3-GRPO (`5` seeds) + Language/SNR-Conditional Conformal (`100` resplits) + Active Visual SLZ (real-imagery `AeroScapes`/`ICG`/`UAVScenes` transfer, **zero `UAVid`**), referencing Paper 1's RTA cage neutrally.

---

## 6. Master License Firewall, CVE Pins & Commercial Open-Core Governance

### 6.1 Permanent Hard-Ban & Clean-Stack Matrix (`v2.0 -> v4.0`)

| Domain | **HARD-BANNED (CI Reject)** | **REQUIRED CLEAN PIN** |
| --- | --- | --- |
| **VIO / Visual Loc** | `OpenVINS`, `VINS-Fusion`, `HybVIO`, `ORB-SLAM3`, `DINOv2-SALAD` (`GPL-3.0`); `sqrtVINS` (`LGPL-3.0`); `Sample4Geo` (unlicensed); `EfficientLoFTR` (custom) | `Basalt` (`BSD-3`), `XFeat + LighterGlue` (`Apache-2.0`), `MegaLoc` (`MIT`) |
| **Vision / Synth Data** | `BlenderProc` (`GPL-3.0` in runtime/train); `UAVid`, `RescueNet`, `LoveDA`, `DroneDeploy`, `FlyAwareV2`, `Mid-Air`, `SegFormer`, `DINOv3`, `DAv2-B/L/G`, `Ultralytics` (`AGPL`), `Deep Hough`, `InsPLAD`, `DDOS`, `Drone-vs-Bird` | `FLAIR-1` (`etalab-2.0`), `OEM 41/100`, `FloodNet` (`CDLA-P`), `SkyScenes` (`MIT`), `TTPLA` (`Apache-2.0`, forward `45°` ONLY), `Mendeley TEİAŞ` (`CC BY 4.0`), `Purdue UAV` (`BSD-3`), Gazebo OGRE2 (`Apache-2.0`); Eval-only: `AeroScapes`, `ICG`, `UAVScenes` |
| **Voice ASR / TTS** | `OHF-Voice/piper1-gpl` (`>=1.7.0`, `GPL-3.0`), **ALL prebuilt `tr_TR` Piper voices** (`dfki`, `fahrettin`, `fettah`), `en_US-lessac`/`ryan`/`hfc`, `mms-tts-*`, `XTTS-v2`, `OmniVoice`, `UWB-ATCC`, `ATCO2`, `ESC-50`, `DEMAND`, unfiltered `FSD50K` | **`sherpa-onnx >= 2.0.0`** (PR `#4055` zero-`espeak-ng`), `whisper.cpp` + `Silero VAD` (`MIT`), `en_US-ljspeech-medium` (`PD`), `ema-lightning` (`Apache-2.0`) / `okur-tts` (`Apache-2.0+CC-BY`) / clean-room TR VITS; `CV 27.0` (`CC0`) + `ISSAI TSC` (`MIT/CC-BY`) + `MUSAN`/`SLR28` |
| **CFD / Crypto / Cyphal** | Unpinned `libsodium <= 1.0.20`, `Fast DDS < 2.14.6 / < 3.4.1`; OpenFOAM linked into runtime; `embind` / `-pthread` / `coi-serviceworker` in WASM | Offline air-gapped `OpenFOAM` (`ldd/nm` 0-GPL gate); **`libsodium >= 1.0.21`**, **`Fast DDS >= 2.14.6 / 3.4.1`**; MIT `libcanard` + `o1heap v3.0.0` + Nunavut C11 |

### 6.2 Open-Core Governance & Commercial Boundary (`DCO 1.1`, Zero-CLA)
Replicating the proven Auterion/PX4, PickNik/MoveIt 2, and Foxglove-MCAP boundaries while avoiding Foxglove Studio's closed-source backlash:
- **Permanently `Apache-2.0` (`DCO 1.1`):** `libvertirta_flight_core`, `libvertirta_rta_companion`, `256 B` `.vrta` format/parsers/Parquet exporter, single-operator `VertiGCS` + GitHub Pages WASM sandbox, `pip install vertirta` Gymnasium SDK, `VertiBench-TR/EN` (`v1`/`v2`), and starter `ESBMC`/Renode harnesses.
- **Future Commercial Enterprise Wedges (Non-Discretionary 2025–2026 Regulatory Value, Built Strictly Atop Stable C++/gRPC/ABI Boundaries):** (1) `VertiRTA CertPack` (`DO-178C`/`DO-333`/`MOC-5`/`SHT-21 Ek-6` cert dossiers & MC/DC kits), (2) `VertiRTA Shield` (`Part-IS` HSM/PKI `.vrta` signing, SSO/RBAC multi-operator Fleet Ops, 72h reporting), (3) `VertiRTA Studio` (OEM custom airframe ID & multi-GPU HJI-CBVF cluster), (4) `VertiRTA Airspace` (managed InterUSS `F3548-21` / `SHGM İHATTYS` connectors).

---

## 7. Objective Release Acceptance Criteria (`M1` -> `M2` -> `M3`)

| Release | Binary Machine-Verifiable Exit Gates (ALL Mandatory Before Tagging) | Honest Scope Classification |
| --- | --- | --- |
| **`M1 (v3.0.0)`** | 1. `libvertirta_flight_core.a` unchanged at `28,886 B` Flash / `0 B` RAM / `0` heap / `<= 2,960 B` DAG; `check_arch_isolation` `0` violations.<br>2. `firmware/stm32h7_hitl/` links clean bare-metal ELF on `arm-none-eabi-gcc 14.2.1` (Rev V `0x2003`, `LATENCY=4`, `VTOR=0x00000400`, `LSPEN=0`, `DWT->LAR`).<br>3. `firmware/renode_cyphal/` CI proves `< 20 ms` dual-STM32H753 CAN-FD failover on Subject `1024` (`48 B`) using `o1heap v3.0.0`.<br>4. `verification/formal/` passes `ESBMC v8.7 --32` on `ubuntu-24.04` x86_64 + MISRA-Aligned `clang-tidy`/`cppcheck` + 5-class gap register.<br>5. Single-threaded Emscripten `.wasm` `< 150 KB` (`0` `-pthread`/`coi-serviceworker`/`embind`); `apps/vertigcs/` Vite build + Vitest PASS (`MapView z>=8`, `IGeoViewportAdapter` keeping `viz/cesium/` intact).<br>6. `sdk/python/` `nanobind` wheel + `VertiVecEnv` Gymnasium v1.1+ `check_env` + `VertiRTAConfig.cmake` consumer test + non-root `docker compose` config validation PASS.<br>7. `docs/standards_registry.yaml v2.0.0`, `CONTRIBUTING.md` DCO 1.1, and `docs/paper_ral/` (with live paired McNemar + `rliable` IQM output & `verify_paper_numbers` `0` errors) PASS. | Renode = **Functional SIL**; physical `DWT->CYCCNT` = **Empirical HWM**; ESBMC = **Supplemental Formal Verification** (no DO-330 TQSP); static linter = **MISRA-Aligned** (not MISRA C++:2023 certified). |
| **`M2 (v3.5.0)`** | 1. `core/uspace_fleet/` executes 20-aircraft hybrid fleet (`<= 0.6 ms/tick`, `0` separation violations, `< 500 ms` P0 emergency preemption, `68 kg` liquid-chilled thermal turnaround `<= 1.74 min` hold).<br>2. `Part-IS` test suite verifies `40 B VertiRtaSecureHeader` + `ReplayWindow128` rejects replayed/spoofed DDS packets (`RTO <= 10 s`), pins `libsodium >= 1.0.21` & `Fast DDS >= 3.4.1`.<br>3. `core/fleet_z3/` solves 20-aircraft conflicts in `< 25 ms` p99 (`QF_RDL` `< 3 ms` + local `|N_local| <= 6` `QF_LRA`).<br>4. `core/airframes/lift_cruise/` passes 8+1 OEI hover (`<= 79.2% T_max`) & transition CTest.<br>5. `core/wind_acoustics/` loads `VRTAWND1` (`< 1.1 us`), computes `D=1.50 m` BVI/BPM `dBA` (`< 1 ms`), respects Kartal Hospital helipad cylinder, and passes **`ldd`/`nm` 0-GPL runtime gate**. | Simulation & protocol conformance testbed; zero certified USSP / Part-IS / SC-VTOL operational approval claimed. |
| **`M3 (v4.0.0)`** | 1. `core/forward_hazard/` passes Gazebo `MmWave77GHzSensor` + `45°` wire fusion CTest (`R_veh = 2.10 m`, `<= 15 m/s` degraded cap, empirical TU116 `< 300 MiB` VRAM gate).<br>2. `core/gnss_denied/` (`Basalt` BSD-3 + Lagged 2D MCUSUM + `XFeat+LighterGlue`) detects `0.5–2.0 m/s` slow-ramp spoofing in `<= 1.4 s` P95 and enforces `3σ` CBVF CTR wall inflation with `0` `LTFJ`/`LTBX` breaches.<br>3. `core/voice/` (`whisper.cpp` + `sherpa-onnx >= 2.0.0`, `< 250 ms` TTFA, `0` GPL/NC voices) + `VertiBench-v2` (`N_cal >= 750/lang`, 4-stratum SNR Mondrian, `< 5%` SoC override) pass all schema & leak gates.<br>4. `core/forensics/` emits XSD-valid `.e5x` + ERCS (`SCF-NP`, Barrier `#8=0`, draft-only `ID 431`, `0.0%` numeric hallucination) from `.vrta` + `.vrta_events.jsonl`.<br>5. `docs/paper_corl/` passes `verify_paper_numbers` (`0` errors, zero `UAVid`). | Research perception, voice, and draft occurrence-reporting pipeline; zero airworthiness or SIA investigative authority claimed. |
