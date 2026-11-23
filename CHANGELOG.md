# Changelog

All notable verified releases of **VertiRTA** are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and Semantic Versioning.

> **Scope Reminder:** Every release below is an open-source research, simulation, and software-verification testbed — NOT a certified airborne product.

## [3.5.1] - 2026-11-23 (Phase B Real-World Hardening & Colab L4 ML Leap)

### Added
- **Colab `L4` `PIDNet-S v2` Zero-ShareAlike Distillation & Physical `GTX 1650 SUPER` `< 12 ms` Proof (`artifacts/v3.5.1/{l4_pidnet_v2_distill_metrics,local_gtx1650s_pidnet_v2_proof}.json`):**
  - Fixed `SkyScenes` stem parsing + nested-tar filter (`276` frames ingested: `207` train / `69` val, `49.4%` vertical facades remapped -> Class 2) alongside `FLAIR-1` (`250`), `OpenEarthMap-41` (`500`), and `FloodNet` (`350`), plus dynamic 5-class energy margins (`m_in=-6.0, m_out=-2.5`) and asymmetric FSR-gated Stage-2 selection (`val_FSR <= 0.08`).
  - Permissive val (`65M` px): **`val_mIoU = 0.4848`** (`+10.84 pp` vs `v3.1.0` `0.3764`), **`val_FSR = 4.03% <= 8.0%`** (`5.24×` cut vs `21.11%`), `safe_precision = 90.30%`, `OOD AUROC = 0.6487`, calibrated `T_prob = 1.0839`, `tau_ood = -2.6106`; quarantined zero-shot `AeroScapes` (`350` oblique frames): **`FSR = 1.95%`** (`9.46×` cut vs `18.44%`), `safe_precision = 82.59%`, Class-2 obstacle IoU `208×` gain (`3.6e-5 -> 7.5e-3`, conservative FSR penalty traded oblique ground recall `mIoU 0.3395 -> 0.2387`).
  - Physical Windows `GTX 1650 SUPER (TU116)` `FP32 + CUDA Graph + EXHAUSTIVE + max_ws=1 + kSameAsRequested` on `pidnet_slz_v2_opset17.onnx` (`30.584 MB`, SHA-256 verified `d6584a83...227c`): **`p50 = 8.887 ms, p95 = 11.022 ms, p99 = 11.695 ms < 12.0 ms`**, `VRAM delta = 155.0 MiB < 300 MiB`, **`all_local_gates_passed: true`** (WSL2 desktop-load `p99 = 20.196 ms` logged transparently).
  - Production C++20 `core/slz_bayes/` & `vertirta_perception_node` hardening: Brown-Conrady distortion, `ood_calibration.json` loader, `|t_rgb - t_depth| <= 2 ms` sync gate, real `ProcessRgbdFrame()` callback (removed silent `InjectMockFrame()` production bypass), and live `/vertirta/slz_markers` publishing.
- **Colab `L4` `Qwen3-4B v2` Paraphrase-SFT + 120-Step Z3-GRPO + Live O(1) Token-Level GBNF DFA + Post-Hoc NLL Calibration (`artifacts/v3.5.1/l4_qwen3_4b_v2_calibrated_eval.json`):**
  - Strictly disjoint train-only TR/EN/MIX paraphrase augmentation (`seed_overlap=0`, `max_char_5gram_jaccard=0.1397 < 0.15`), 2-epoch SFT -> 120-step Z3-GRPO (`reward 0.9710 -> 0.9869`), and exact O(1) `VertiRtaGbnfDfa` (`1,810` states) + Microsoft `z3-solver` fail-closed gate: **Grammar `100.0%`**, **AST/STL `91.96%`** (`+19.82 pp` vs `v3.1.0`; `TR = 96.93%`, `MIX = 100.0%`, `FAULT = 100.0%`, `AMBIGUOUS_ASR = 100.0%`), **Bucket-3 Reject `75.00%`** (`TR = 100.0%`), **Z3 Sat `100.0%`** (`+3.33 ms` GBNF overhead).
  - Zero-leakage post-hoc NLL temperature scaling (`tau = 0.5066` fitted strictly on `n_cal = 200`): Test ECE **`0.3322 -> 0.1501` (`2.21×` cut)**, Mean Set Size **`5.438 -> 3.125` (`-42.5%`)**, Empirical Coverage **`95.36%`** (`534/560`), Singleton Exec **`0.0% -> 22.86%`** (`24.64%` under 50/50 split-cal).
- **`VertiGCS`, ROS 2, Docker & SDK Hardening:** Bundled offline MapLibre GL JS v6 CSP module worker (eliminating Pages `HTTP 503`), 1-click replay-to-fault exit+inject, `360 px` zero-overflow sidebar, compact collapsible 20-eVTOL Fleet telemetry table (Vitest **`12/12` PASS**), bidirectional RFC 6455 `/telemetry` WebSocket + reconnect, `ubuntu:24.04` `g++-14` / `emsdk:4.0.4` Docker parity, real `CbvfSafetyFilter` + DEM terrain AGL in `vertirta_rta_node`, `100 ms` `rta_cmd` deadman fallback in `vertirta_px4_bridge`, and `FleetSimEngine` / `LiftCruiseDynamics` Python SDK bindings.
- **STM32H753 Silicon, Wind Cache & Lift+Cruise Sensor Hardening:** `PLL1Q = 80 MHz` (`DIVQ1EN`/`DIVQ1=/12`) for FDCAN1 `1/5 Mbps`, fail-closed `wait_init_bit(false)` timeout, `#ifdef VERTIRTA_RENODE_SIL` guard keeping physical HITL strict `rev_id == 0x2003u`, cached 180-node acoustic hemisphere (`|ΔRPM|/RPM > 2%` or `|ΔV| > 2 m/s`) + best-effort `mlock()` in `MmapVrtaWindLut`, and noisy IMU/Pitot-failover/DShot-staleness `TransitionSensorFrame` in `TransitionIndiWlsController`.
- **Paper 1 RA-L Figures & Governance:** 3 double-anonymous publication-grade vector PDFs (`docs/paper_ral/figures/fig1..fig3*.pdf`, `generate_ral_figures.py`), `CITATION.cff` (`CFF 1.2.0`), `CHANGELOG.md`, `[project.urls]` in `pyproject.toml`, and 28-day AIRAC metadata + closed official DHMİ AIP `LTBX` ARP (`40°59'26"N, 029°12'58"E`) in `config/istanbul_ltfj_airspace.yaml`.

### Changed
- Synchronized IEEE RA-L Paper 1 (`docs/paper_ral/`): corrected 30k paired atlas wall-clock (`50.50 s -> 57.86 s`), retired `Pending (M2 U-Space)` with live `0.151 ms/tick; 1.81 ms QF_IDL` results, and embedded all 3 vector figures.
- Refreshed `docs/SAFETY_CASE.md` (`v3.5.0` `bf8adb1` supplemental evidence table `Sn11–Sn14`, preserving `G5`/`G6` `[UNDEVELOPED]`) and synced `docs/paper/metrics_manifest.json` + regenerated `docs/paper/generated/tables.tex`.
- Bumped package & archival metadata in `pyproject.toml` and `.zenodo.json` to `3.5.1`.

## [3.5.0] - 2026-11-23 (`bf8adb1`, 6/6 Live Cloud CI GREEN)

### Added
- **Multi-Airframe `Lift+Cruise 8+1` (`core/airframes/lift_cruise/`):** Zero-heap `IAirframeDynamics`, `8+1` hover OEI (`79.07% <= 79.2% T_max`) & `4+1` compat, 400 Hz quintic Hermite `f64` INDI-WLS (`0.1977 m` transition altitude error, `0.98 us` WLS p99), and PX4 `v1.16` internal `wq:rate_ctrl` overlay.
- **Air-Gapped Urban Wind & Aeroacoustics (`tools/cfd_offline_gpl/`, `core/wind_acoustics/`, `scripts/ci/check_zero_gpl_symbols.py`):** Offline `GPL-3.0` OpenFOAM POD-RBF (`< 29.1895° E`, rejecting İSTMarina `29.2057° E`) emitting pure numeric `VRTAWND1`, Apache-2.0 `mmap` reader (`p99 = 0.272 us`, SHA-256 tamper reject), 180-node coaxial BVI/BPM/ISO 9613-1/2:2024 `dBA` (`p99 = 0.0738 ms`), 64 Istanbul receptors + Kartal Dr. Lütfi Kırdar Hospital Tier-2 hard helipad cylinder (`500`-run stress `0` breaches), and automated `ldd`/`nm` `0`-GPL gate (`20` targets, `0` violations).
- **20-eVTOL U-Space, `Part-IS (EU) 2023/203` & Two-Tier Fleet Z3 (`core/uspace_fleet/`, `core/fleet_z3/`, `ros2_ws/src/vertirta_uspace_node/`, `apps/vertigcs/src/fleet/`):** SoA 20-agent `FleetSimEngine` (`0.1507 ms/tick`), `15°C` chilled `68 kg 112S12P` 4-stand vertiports (`C_th = 70.72 kJ/K`), Top-10 3D CBF-ORCA + RHR curl (`<= 14` RTA slots, `< 500 ms` P0 CEL), `40 B` `VertiRtaSecureHeader` + `ReplayWindow128` + immutable `libsodium-1.0.22` + `RTO <= 10 s` FSM, ASTM `F3548-21` / `F3411-22a` / `RFC 9575` `228 B` DRIP, separate-context two-tier Z3 (`QF_IDL` `1.81 ms` + local `|N_local| <= 6` `QF_LRA < 25 ms`, `100%` SMT-LIB2 parity), interactive `VertiGCS` 20-eVTOL Fleet & 1-click `Part-IS` Cyber-Attack simulator (Vitest `10/10`), and dual-arch `.github/workflows/m2_uspace_and_physics.yml` CI (`25/25` root CTests, `0 B` added to `28,886 B` `flight_core` or `38,470 B` `.wasm`).

## [3.1.0] - 2026-11-23 (`6869842`, 5/5 Live Cloud CI GREEN)

### Added
- **Live 1-Click GitHub Pages `VertiGCS` WASM Console (`deploy_pages.yml`):** Published `https://fahrifurkansagir.github.io/VertiRTA/` serving `38,470 B` single-threaded `.wasm` Worker with zero API tokens.
- **Empirical Evidence Closure (`artifacts/v3.1.0/`, `19.35 CU` Colab `T4`/`L4`):**
  - Full `81×57×81×57` WENO5/TVD-RK3 4D JAX `HJI-CBVF` PDE LUT (`11.31 MB`, `soundness_passed: true`) + exact paired 30,000-sim RA-L statistics (`95.98%` CSFL `(3324, 0)` / `100.00%` CEL `(1500, 0)`, `rliable` IQM margin `2.886 m`, retiring `PENDING_PER_RUN_EXPORT`).
  - Real Colab `L4` `Qwen/Qwen3-4B` BF16 LoRA SFT→Z3-GRPO (`74.29%` SFT / `72.14%` GRPO AST, `96.25%` `KnowNo` Split Conformal coverage, retiring `PENDING_COLAB`).
  - Real Colab `L4` zero-ShareAlike `PIDNet-S` distillation (`val_mIoU 0.3764`, `OOD AUROC 0.6984`, quarantined zero-shot `AeroScapes 0.3395`) verified live on physical Windows `GTX 1650 SUPER` (`8.67 ms` p50 / `111.12 FPS` / `126.0 MiB` VRAM, WDDM `p99 = 14.20 ms` logged transparently, retiring `UNTRAINED_WEIGHTS`).

## [3.0.0] - 2026-11-23 (`dc28e37`, 4/4 Live Cloud CI GREEN)

### Added
- Bare-metal `STM32H753` ITCM/DTCM HITL firmware (`firmware/stm32h7_hitl/`, `ES0392 Rev 16` errata locks, `36,624 B` ITCM / `43,708 B` Flash / `0` veneers).
- Dual-FCC Renode `v1.17.0` OpenCyphal CAN-FD Functional SIL (`firmware/renode_cyphal/`, MIT `libcanard` + `o1heap v3.0.0`, `3/3` PASS, `12.50 ms` failover, `4` CRC rejects).
- Six live `ESBMC v8.7.0 --32` supplemental formal verification harnesses (`1,905 / 1,905` properties `VERIFICATION SUCCESSFUL`) + live `Cppcheck 2.22.0` / `clang-tidy` MISRA-aligned profile + 5-class gap register (`verification/formal/`, `.github/workflows/formal_and_hil.yml`).
- Single-threaded `< 150 KiB` Emscripten Worker WASM ABI (`wasm/`), unified MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4 `VertiGCS` (`apps/vertigcs/`, `IGeoViewportAdapter` preserving `viz/cesium/`), relocatable CMake `FILE_SET HEADERS` (`cmake/VertiRTAConfig.cmake.in`), `nanobind` `cp312-abi3` `pip install vertirta` Gymnasium/CLI SDK (`sdk/python/`), hardened non-root Docker stack (`docker/`), and `.github/workflows/product_and_wasm.yml`.
- Governance `v2.0.0` (`docs/standards_registry.yaml` +7 2025–2026 EU/SHGM/ED-155A instruments, `CONTRIBUTING.md` `DCO 1.1` No-CLA + permanent Apache-2.0 lock) and double-anonymous IEEE RA-L Paper 1 (`docs/paper_ral/`).

## [2.0.0] - 2026-11-22

### Added
- **Pillar 3 Active Visual SLZ (`core/slz_bayes/`, `ml/slz_perception/`, `ros2_ws/src/vertirta_perception_node/`):** Zero-ShareAlike 5-source license gate + `OpenEarthMap-41` allowlist, static-PAPPM `PIDNet-S` Opset-17 ONNX export, `1.5625 MiB` cell-local 2.5D Bayesian SLZ grid with OOD variance floor, 5-state `ActiveInspectionPolicy`, Colab kinematic EGL rig, and manifest-driven IEEE/arXiv technical report (`docs/paper/`) + `.zenodo.json`.
- **M0 Release Hardening:** `CONTRIBUTING.md`, `SECURITY.md`, `scripts/ci/check_public_hygiene.py`, and turnkey GPU notebooks (`notebooks/`).

## [2.0.0-alpha] - 2026-11-22

### Added
- **Pillar 2 Verified Bilingual TR/EN Contingency Co-Pilot (`core/stl/`, `core/copilot/`, `benchmarks/vertibench_tr_en/`, `ros2_ws/src/vertirta_mission_node/`):** `<= 30`-token GBNF grammar, zero-heap quantitative STL monitor (`FixedMonotonicWedge`), `< 12 ms` Z3 `QF_LRA` corridor verifier + UNSAT-core repair, `KnowNo` Split Conformal gate (`q_lvl = 0.955`), and Apache-2.0 `VertiBench-TR/EN` `< 0.15` Jaccard ARM64 CI (`.github/workflows/vertibench_ci.yml`).

## [1.0.0] - 2026-11-22 (`ef9be75`)

### Added
- **Pillar 1 Bare-Metal Flight Core & RTA Companion (`core/flight_core/`, `core/rta_companion/`):** Freestanding C++17 `400 Hz` core (`28,886 B` Cortex-M7 Flash, `0 B` static RAM, `1,496 B` max frame, `2,960 B` stack DAG, `0` heap) with 3-IMU TMR voter, coaxial-unblanked `f64` INDI+WLS (`StaticDaqpSolver`), Motor FDI, Option B1b active flat-spin CEL, and `256 B` CRC-32C `.vrta` flight recorder; `100 Hz` C++20 companion with `mmap` 4D `HJI-CBVF` / 2D `CEL` LUTs, `3.094 MiB` robocentric ESDF, wire capsules, intruder CV-KF, and real DHMİ AIP `LTFJ`/`LTBX` hard CTR walls.
- Live 30,000-sim Tier-1 6-DOF Survival Atlas + failure cliff ledger (`artifacts/v1.0.0/`), CI-hashed `SCSC-141C` GSN v3 Living Safety Case (`docs/SAFETY_CASE.md`), PX4 `v1.16` / ROS 2 Jazzy / Gazebo Harmonic overlays, dual-layer Istanbul CesiumJS/Blender visualization (`viz/`), and 4-tier ARM64/x86_64 determinism CI (`.github/workflows/determinism.yml`).
