# Changelog

All notable releases of **VertiRTA** are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-11-25

First stable public release of **VertiRTA** — deterministic eVTOL flight control, ASTM F3269-21 run-time assurance (RTA), bare-metal formal/SIL verification, multi-aircraft U-Space & Part-IS security, zero-GPL MAVLink v2 offboard shield, and real-world perception/forensics pipelines.

### Added

#### Flight Core, Formal Verification & Silicon SIL
- **Freestanding C++17 Flight Core (`core/flight_core/`):** 400 Hz deterministic core compiling to **28,886 B** Cortex-M7 Flash, **0 B** library static RAM, **1,496 B** max frame, **2,960 B** static call-graph DAG, **0** heap symbols, and **0** recursion — including 3-IMU TMR voting, synchronized-LPF INDI + 8×4 WLS (`StaticDaqpSolver`), dual-channel Motor FDI, Option B1b flat-spin CEL, and 256 B Castagnoli CRC-32C `.vrta` flight recording.
- **Bare-Metal STM32H753 Firmware & Dual-FCC OpenCyphal CAN-FD SIL (`firmware/`):** ITCM/DTCM linker layout (`ES0392 Rev 16`, `PLL1Q = 80 MHz` FDCAN1, `0` veneers) and Renode v1.17.0 dual-FCC failover verification (**3/3 PASS**, **12.50 ms** takeover, 4 CRC rejects).
- **Supplemental Formal Verification (`verification/formal/`):** Six direct-source ESBMC v8.7.0 (`--32`) bounded model-checking harnesses passing **1,905 / 1,905** properties, plus Cppcheck 2.22.0 / clang-tidy and living GSN v3 safety case ([`docs/SAFETY_CASE.md`](docs/SAFETY_CASE.md), `Sn1–Sn19`).

#### Multi-Airframe GNC, 4D Reachability & Monte Carlo Campaigns
- **Dual-Airframe Dynamics (`core/airframes/lift_cruise/`, `core/rta_companion/`):** `VertiRTA-X8-240` coaxial X8 and `Lift+Cruise 8+1` transition dynamics with 400 Hz quintic Hermite INDI-WLS (`0.198 m` transition altitude error, `0.98 us` WLS p99) and 100 Hz `mmap` 4D HJI-CBVF-QP safety filtering + 3.094 MiB robocentric ESDF + Istanbul `LTFJ`/`LTBX` hard CTR walls.
- **102.85M-Node Dual-Airframe 4D JAX PDE (`reachability/`, `artifacts/v4.0.0/g4_jax_4d_hires_and_lc_manifest.json`):** `101×71×101×71` grids across both airframes with **0** soundness violations, exported to `11.31 MB` runtime `VRTALUT1` tables; paired 30,000-run atlas achieving **95.98%** CSFL / **100.0%** CEL survival.
- **1,000 + 180 Native C++20 Monte Carlo Dispersion & Failure Boundary (`benchmarks/monte_carlo_dispersion/`):** Deterministic LHS dispersion campaign (**0 / 1,000** airspace breaches, **200/200** GNSS spoof detection, `LC-240 8+1` **500/500** `CEP95 = 1.97 m`, `X8-240` `429/500` exposing ±5 cm CG-shift CEL sensitivity) plus 6×6 mass-overload × gust thrust-saturation sweep (`+14%` @ `10 m/s` transition, `>= +21%` saturation wall).

#### MAVLink v2 Shield, U-Space Fleet & Part-IS Security
- **Zero-GPL Native C++20 MAVLink v2 UDP/Offboard RTA Shield (`tools/mavlink_rta_bridge/`):** Clean-room Apache-2.0 `0xFD` codec + X.25 CRC-16/MCRF4XX (`8` `common.xml` `CRC_EXTRA` messages) wired directly into `CbvfSafetyFilter`, Istanbul NFZ walls, 500 ms deadman hover, and airborne disarm/flight-termination denial. Live 2,500-packet POSIX UDP loopback proof: **0** drops, **0** unsafe leaks, **1,181/1,181** clamped, **240/240** denied, **239/239** CRC rejected, **p50 = 18.0 us / p99 = 124.4 us**.
- **20-eVTOL U-Space Fleet, Part-IS Security & Two-Tier Fleet Z3 (`core/uspace_fleet/`, `core/fleet_z3/`, `core/wind_acoustics/`):** SoA 20-aircraft `FleetSimEngine` (`0.151 ms/tick`), `15°C` chilled vertiports, 40 B `VertiRtaSecureHeader` + `ReplayWindow128` + pinned `libsodium 1.0.22` + `RTO <= 10 s` FSM, ASTM F3548-21 / F3411-22a DRIP, separate-context `QF_IDL`/`QF_LRA` Fleet Z3, air-gapped `VRTAWND1` wind LUT (`0` GPL runtime symbols), and BVI aeroacoustics.

#### Real-World Perception, GNSS-Denied Navigation, Co-Pilot, Voice & Forensics
- **Moving NASA Goddard Drone Video Temporal SLZ HUD (`scripts/realworld/run_real_drone_video_temporal_hud.py`):** 240 live frames (`PD-USGov-NASA`) on GTX 1650 SUPER — causal homography EMA + hysteresis reduces safe/hazard flicker from **2.19% to 0.96% (-56.25%)** and SLZ centroid jitter from **1.22 to 0.46 px (-62.15%)** at **77.4 FPS**.
- **Zero-ShareAlike `PIDNet-S v2.1` SLZ Perception (`ml/slz_perception/`, `core/slz_bayes/`):** Local GTX 1650 SUPER FP16 recovery (`pidnet_slz_v21_local_gtx_opset17.onnx`, `32.07 MB`): **Val mIoU = 0.4669**, **FSR = 2.81%** (≤ 3.0%), **OOD AUROC = 0.7554**, quarantined zero-shot AeroScapes `0.2468`.
- **Forward-Wire Vision, GNSS-Denied `EuRoC MAV` & `XFeat` (`core/forward_hazard/`, `core/gnss_denied/`):** `M-LSD-WireSeg` + `D-FINE-N` (`clDice 0.9644` procedural), ETH Zurich `EuRoC MAV V1_01_easy` (**7/7 PASS**, `0.1 s` spoof TTD, `0` false alarms), Kartal `XFeat` **36/36** RANSAC (`CEP50 0.218 m`), and live Istanbul METAR/Open-Meteo/ADS-B verification.
- **Bilingual Co-Pilot, Conformal RAPS, Zero-GPL Voice ATC & Forensics (`core/copilot/`, `core/stl/`, `core/voice/`, `core/forensics/`):** 3-seed `Qwen3-4B` Z3-GRPO on `VertiBench-v2` (**100%** grammar & Z3 SAT, **82.83 ± 0.71%** AST-STL, **95.25%** SNR-Mondrian coverage), APS/RAPS conformal calibration upgrade + diagnostic (`ml/copilot/calibrate_conformal.py`), Zero-GPL `sherpa-onnx v2.0.3` TTS (**71.7 ms** TTFA), and real `PX4 sample.ulg` → W3C XSD-valid `E5X` regulatory draft export.

#### WebAssembly Console, SDK, Packaging & Governance
- **Live `VertiGCS` WebAssembly Console (`wasm/`, `apps/vertigcs/`):** `38,470 B` single-threaded `.wasm` Worker + offline bundled MapLibre/Deck.gl 3D console deployed on GitHub Pages (`17/17` Vitest).
- **Relocatable CMake, Python SDK & Hardened Docker (`cmake/`, `sdk/python/`, `docker/`):** `find_package(VertiRTA 1.0 CONFIG REQUIRED COMPONENTS flight_core)`, `nanobind` `cp312-abi3` Gymnasium SDK (`6/6` native tests), and non-root Docker Compose stack.
- **Open-Source Governance & CI (`7/7` Workflows Green):** Apache-2.0 + mandatory Section 4(d) [`NOTICE`](NOTICE) authorship attribution (`Fahri Furkan Sagir`), DCO 1.1 (`git commit -s`), [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`SECURITY.md`](SECURITY.md), structured Issue/PR templates, [`docs/SYSTEM_CARD.md`](docs/SYSTEM_CARD.md), [`docs/research/CROSS_BENCHMARK_AUDIT_v1.md`](docs/research/CROSS_BENCHMARK_AUDIT_v1.md), and `36/36` CTest + Zero-GPL runtime symbol gates across x86_64 and ARM64.
