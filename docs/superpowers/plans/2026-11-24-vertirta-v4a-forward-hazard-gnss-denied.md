# VertiRTA M3 (`v4.0.0`) Plan 6A — Forward Wire/Intruder Radar-Vision Fusion, 77 GHz Gazebo Plugin, GNSS-Denied VIO/Geo-Reloc & Dynamic 3σ CBVF CTR Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NON-CERTIFICATION RESEARCH DISCLAIMER:** VertiRTA is an academic and simulation research testbed for autonomous eVTOL runtime assurance, perception fusion, and GNSS-denied navigation. It is **NOT** a certified airborne product under EASA SC-VTOL, EASA CS-ETSO, DO-178C, DO-254, DO-200B, or SHGM SHT-İHA. All Istanbul `LTFJ`/`LTBX` airspace geometry is a simulation reference snapshot only.

**Goal:** Implement the complete M3 (`v4.0.0`) Track A autonomous perception & GNSS-denied navigation stack across `core/forward_hazard/`, `sim/gazebo_plugins/mmwave_77ghz/`, `core/gnss_denied/`, `ros2_ws/src/vertirta_forward_hazard_node/`, `ros2_ws/src/vertirta_gnss_denied_node/`, `apps/vertigcs/` M3 telemetry panel, root `CMakeLists.txt` Wave-2 wiring, and `.github/workflows/m3_perception_voice_forensics.yml` — with zero modifications to frozen `core/flight_core/` (`28,886 B` Cortex-M7 lock), frozen `core/rta_companion/` (`CbvfSafetyFilter`), or frozen `wasm/` (`38,470 B` browser lock), and zero GPL/LGPL/NC/ShareAlike contamination.

**Architecture:**
1. **Decoupled Forward Thin-Wire & Intruder Pipeline (`core/forward_hazard/`):** Strictly decoupled from the locked `512×512` `80°` nadir SLZ camera (`fx = 305.09 px`). Consumes an IMU-stabilized `1280×720` `45°` HFOV narrow-FOV forward camera cropped to a `1280×320` horizon ROI and split-stacked into a `640×640` tensor (`f_net = 1545.1 px`, 16 px horizontal overlap or explicit coordinate channels), feeding permissive `M-LSD-WireSeg` (with `clDice` topology loss) + `D-FINE-N` (`Apache-2.0`) through an ONNX Runtime C++20 adapter (`enable_fp16: false` default on `GTX 1650 SUPER TU116` per `01-DR08-13` Override `#6`, empirically gated in Plan 6C). Fuses 30 Hz vision interpretation planes (`chi^2 <= 9.21` Mahalanobis gating) and 15–20 Hz 4D 77 GHz mmWave radar returns inside a `200 Hz` `2.0 s` `SE(3)` SLERP pose ring buffer into a 6-state wire-span EKF (with `> 3°` SVD grazing-angle parallax fallback and conservative horizon standoff capsules), emitting piecewise catenary capsules into frozen `WireCapsuleTable` (`<= 64` slots, `sizeof(WireCapsule) <= 48 B`, canonical `R_veh = 2.10 m`) and dynamic tracks into frozen `IntruderTrackerKf` (`<= 16` tracks), while clamping `CapabilityEnvelope::v_max_mps <= 15.0 m/s` (`35.8 m` emergency stop) whenever radar is degraded over urban clutter.
2. **Physics-Faithful 77 GHz mmWave Gazebo Harmonic Plugin (`sim/gazebo_plugins/mmwave_77ghz/`):** Custom Apache-2.0 C++20 pure-physics core (`MmWave77GHzPhysics`) wrapped behind `#ifdef VERTIRTA_HAS_GZ_SENSORS` into `gz::sensors::Sensor` (`MmWave77GHzSensor`) and `gz::sim::System` (`MmWave77GHzSystem`, `<sensor type="custom" gz:type="mmwave_77ghz">`). Models 77 GHz (`lambda = 3.89 mm`) Fresnel normal-incidence wire glint (`+1.96 to +5.94 dBsm` at `100 m`, `1/R^3` decay), helical stranded ACSR Bragg lobes (`7.0°–14.1°`, vanishing `> 15°`), the `-45 dBsm` cliff past `20°` skew/grazing incidence (and zero Bragg lobes on smooth crane hoist ropes), Swerling I/III fluctuation (`17.0 / 13.0 dB` vs `8.5 dB`), Tier-1B 4-chip cascaded MIMO (`AWR2243`/`ARS540`) and Tier-2 active-PA profiles, and ego-motion-compensated Doppler radial separation (`v_res = v_r + v_ego · u_hat`) with off-boresight azimuth leakage widening the `3σ` static gate from `±0.64 m/s` to `±1.01 m/s` at `20 m/s` / `30°`, cross-checking static-gate returns against `D-FINE-N` to catch hovering drones/birds.
3. **GNSS-Denied Navigation, Spoofing Defense & Dynamic 3σ+`dh/dt` CBVF CTR Safety (`core/gnss_denied/`):**
   - **Layer 1 (Permissive Decoupled VIO):** `Basalt` (`BSD-3-Clause`) decoupled VIO adapter (`0.80 m` rigid carbon-boom belly stereo baseline + nadir radar altimeter/baro metric scale lock + `PIDNet-S` Marmara water-pixel mask purging wave-glint KLT features; optional `OKVIS2` BSD-3 only with `USE_NN=OFF` & GPL `SuiteSparse/CHOLMOD` stripped). Strictly zero GNSS feedback into VIO to prevent pre-detection spoofing contamination.
   - **Layer 2 (10 Hz 4-Hypothesis `H0..H3` Anti-Spoofing/Jamming Guard):** `H1` RF jamming monitor (`TTD < 0.2 s`, C/N0, AGC, HDOP `> 2.5` on PX4 `v1.16` `vehicle_gps_position` + calibrated `EKF2_GPS_DELAY`), `H2` 5-epoch step/meaconing chi-square monitor (`TTD < 0.3 s`), `H3a` Lagged-Baseline Bias-Compensated 2D Vector MCUSUM (`15 s` ring buffer `L2 = 150`, `5 s` guard band `L1 = 50`, land `< 20%` water `k_d = 0.25, h_t = 2.0` -> `<= 1.4 s` P95 TTD and `<= 0.85 m` P99 slip on `0.5–2.0 m/s` ramps; water `> 35%` adaptive `k_d = 0.35, h_t = 3.4` requiring corroboration), and `H3b` telescoping `1.5 s / 4.0 s` 2D position-increment GLRT. Latch freezes GNSS into EKF2, executes `2.0 s` pose ring-buffer rollback, and seeds `sigma_init = 3.5 m` (spoof) or `2.0 m` (clean jam).
   - **Layer 3 (0.5–1 Hz Visual Geo-Registration + Time-Varying CBVF CTR Inflation):** `XFeat + LighterGlue` (`Apache-2.0`) local ortho matcher (`0.5–1.0 m` MBTiles cache, North-up yaw derotation + radalt scale warp, 3-gate acceptance: `>= 25` RANSAC inliers `< 2.5 px` RMSE, convex hull `> 25%`, 2D Mahalanobis gate, Joseph-form Kalman update) + `MegaLoc` (`MIT`) lost-in-space re-init (`sigma_pos > 50 m`). Propagates `P_pos` via random-walk diffusion `q_rw` + systematic drift `(kappa_drift * d_ol)^2` and injects directional `3 * sqrt(n^T P_pos n) + b_map` (`b_map = 2.0 m`) AND time-varying Lie-derivative **`dh/dt = -r_dot_infl`** into frozen `CbvfSafetyFilter::step_100hz` strictly through public `HazardPoint{kind=3}` (`radius_h_m` + outward expansion `vel_ned = r_dot_infl * er`) against `LTFJ` and confirmed DHMİ AIP `LTBX` (`40°59'26"N, 029°12'58"E`), driving the 3-state Navigation FSM (`Bounded GNSS-Denied Cruise` -> `Degraded Caution` -> `Deterministic Visual-SLZ Abort`).
4. **ROS 2 M3 Nodes, VertiGCS M3 Panel, Wave-2 CMake & CI:** Host-testable C++20 cores with `#ifdef VERTIRTA_WITH_ROS2` wrappers (`vertirta_forward_hazard_node` publishing directly onto existing `/vertirta/wire_capsules` and `/vertirta/intruders`; `vertirta_gnss_denied_node` subscribing to PX4 `v1.16` `/fmu/out/vehicle_gps_position`), collapsible `apps/vertigcs/` M3 telemetry panel + Deck.gl wire-capsule/`3σ` inflated-CTR overlay (`0 B` added to `flight_core` or WASM), Wave-2 root `CMakeLists.txt` integration, and `.github/workflows/m3_perception_voice_forensics.yml`.

**Tech Stack:** C++20 (`gcc-14`/`g++-14`, `-fno-fast-math -ffp-contract=off -fno-associative-math -fno-reciprocal-math -frounding-math`, Catch2 v3.7.1, ASan/UBSan), ONNX Runtime C++20 API (optional compile-time guard `VERTIRTA_HAS_ONNXRUNTIME` with deterministic CPU mock fallback for CI), Gazebo Harmonic (`gz-sensors8`/`gz-sim8` behind `#ifdef VERTIRTA_HAS_GZ_SENSORS`), ROS 2 Jazzy (`rclcpp`, `px4_msgs`, `vertirta_msgs` behind `#ifdef VERTIRTA_WITH_ROS2`), Python 3.11 (`uv run --no-project`, `pytest`, `numpy`, `pyyaml`, `onnx`), TypeScript 5 + Vite 6 + MapLibre GL + Deck.gl + Vitest (`apps/vertigcs/`).

**Canonical Spec & Errata Locks:**
- Master Design Spec: `docs/superpowers/specs/2026-11-22-vertirta-v3-v4-expansion-design.md` (§1.2 Invariants, §2.2 Directory Map, §5 M3 Architecture, §6 License Firewall, §7 Exit Gates)
- Forensic Errata & Locks: `docs/research/01-DR08-13-FORENSIC-ERRATA-AND-LOCKS.md` (Overrides `#1`–`#8`)
- Research Ground Truth: `docs/research/DR-10-forward-radar-vision-gnss-denied-vio.md` (subject to all `01-DR08-13` overrides below)
- Frozen Configs: `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`

---

## 0. Global Constraints, Canonical Overrides & Multi-Agent File-Ownership Split

### 0.1 Mandatory `01-DR08-13` Overrides Superseding `DR-10` Draft Text
Every task in Plan 6A **MUST** enforce these canonical overrides rather than copying stale figures from `DR-10`:

| # | `DR-10` Draft Value (STALE — DO NOT USE) | Canonical Lock (`01-DR08-13` + Frozen Configs) | Enforcement Point in Plan 6A |
|---|---|---|---|
| **O-1** | `4–5 m` rotor span, `25 m/s` cruise (`DR-10` L28, L61) | `R_veh = 2.10 m`, `v_max = 22.0 m/s`, `m = 240 kg`, `a_xy_max = 5.5 m/s^2` (`config/vehicle_x8_240.yaml`) | `WireCapsuleConfig::r_veh_m = 2.10`, `v_max_nominal_mps = 22.0`, `v_max_radar_degraded_mps = 15.0` |
| **O-2** | `LTBX` ARP `40°59'56"N, 029°12'58"E` (`DR-10` L813) | Verified DHMİ AIP Turkey `LTBX` ARP **`40°59'26"N, 029°12'58"E`** (`40.990556° N, 29.216111° E`), Kartal datum `(40.8860° N, 29.1880° E, 12.0 m AMSL, N = +36.7 m)`, `max_lon_near_j2 = 29.1895° E` (`config/istanbul_ltfj_airspace.yaml`) | `CtrInflationConfig` defaults + `test_visual_reloc_and_ctr_inflation.cpp` exact lat/lon lock |
| **O-3** | PX4 `v1.18` `vehicle_gnss` / `SENS_GNSS0_DELAY` (`DR-10` L363, L412) | **PX4 `v1.16`**: `px4_msgs::msg::VehicleGpsPosition` (`/fmu/out/vehicle_gps_position`), `EKF2_GPS_DELAY`, `UXRCE_DDS_SYNCT=1` | `vertirta_gnss_denied_node` subscriber & `GnssSpoofGuard` timestamp delay calibration |
| **O-4** | Assumed `2×` FP16 speedup on TU116 & `1280×720` nadir camera (`DR-10` L159, L215) | Live physical GTX 1650 SUPER proofs (`v2.0.0`, `v3.1.0`, `v3.5.1`) proved **FP16 is `2.5–3.4×` SLOWER than FP32+CUDA Graph on TU116** (`v3.5.1`: `8.89 ms` p50, `11.70 ms` p99, `155 MiB` VRAM). Nadir SLZ camera stays locked at **`512×512` `80°` HFOV (`fx = 305.09 px`)**; forward camera is strictly decoupled `1280×720` `45°` HFOV -> `1280×320 -> 640×640` split-stack. | `ForwardOnnxConfig::enable_fp16 = false` default; empirical FP32-vs-FP16 benchmark gated in Plan 6C (`scripts/bench_gtx1650s_m3_timesliced.py`) |
| **O-5** | Direct edit of CBVF barrier law for `dh/dt` | **`core/rta_companion/` (`CbvfSafetyFilter`) is FROZEN (0 edits).** Inject both directional `3*sqrt(n^T P n) + b_map` into `HazardPoint::radius_h_m` AND time-varying `dh/dt = -r_dot_infl` into `HazardPoint::vel_ned = r_dot_infl * er` (`kind = 3`, `z_eff < -5 m` vertical-wall branch in `cbvf_safety_filter.cpp` L202–235). | `CtrInflationAdapter::build_inflated_hazards()` + exact numerical verification against `CbvfSafetyFilter::step_100hz` |
| **O-6** | `1σ` (`sigma_pos < 25 m / 25–65 m / >= 65 m`, `DR-10` L825) vs `3σ` phrasing ambiguity | Parameterize 3-state FSM explicitly on isotropic `1σ` `sigma_pos_m` (`< 25.0 m` Bounded Cruise -> `3σ < 75.0 m`; `[25.0, 65.0) m` Degraded Caution `12–15 m/s` -> `3σ in [75, 195) m`; `>= 65.0 m` Deterministic Visual-SLZ Abort -> `3σ >= 195.0 m` before `80 m` CBVF infeasibility floor), AND track nominal steady-state `3σ` target (`<= 25.0 m`, achieved `8.8–17.2 m` under `0.5 m` ortho XFeat+LighterGlue) as a distinct KPI flag (`nominal_3sigma_met`). | `GnssNavFsmConfig` with both `sigma_caution_m = 25.0`, `sigma_abort_m = 65.0`, and `nominal_3sigma_target_m = 25.0` |

### 0.2 Hard License & Artifact Firewall (Enforced in Code + CI)
- **Allowed Forward Wire/Intruder Sources:** `TTPLA` (`Apache-2.0` — **strictly isolated to decoupled `45°` forward wire head; STILL BANNED from `512×512` nadir SLZ in `ml/slz_perception/`!**), `Mendeley TEİAŞ` (`CC BY 4.0`), `Purdue UAV` (`BSD-3`), Apache-2.0 Gazebo OGRE2 procedural catenary/crane synthesis.
- **Allowed Forward & Geo-Reloc Models:** `M-LSD-WireSeg` (`Apache-2.0`), `D-FINE-N` (`Apache-2.0`), `Basalt` (`BSD-3-Clause`), `OKVIS2` (`BSD-3-Clause` ONLY with `USE_NN=OFF` and `SuiteSparse/CHOLMOD` stripped), `XFeat + LighterGlue` (`Apache-2.0`), `MegaLoc` (`MIT`).
- **HARD-BANNED Everywhere (CI Fail on Any Include/Symbol/Dataset Reference):** `BlenderProc` (GPL-3.0), `InsPLAD` / `DDOS` / `Drone-vs-Bird` / `UAVid` (NC/DUA/proprietary), `STN PLAD` / `PAMPC` (GPL-3.0), `Ultralytics` (AGPL-3.0), `SegFormer` / `Deep Hough Transform` / `SuperPoint` / `SuperGlue` (NC), `DINOv3` (custom ITAR), `OpenVINS` / `VINS-Fusion` / `HybVIO` / `ORB-SLAM3` (GPL-3.0), `sqrtVINS` (LGPL-3.0), `DINOv2-SALAD` (GPL-3.0), `Sample4Geo` (unlicensed).
- **QUARANTINED (Zero-Shot Eval / Research-Only — HARD-BANNED from ALL Training Manifests):** `UAVScenes` (`AeroScapes`, `ICG_Semantic_Drone`, and `UAVScenes` are strictly forbidden in `train_sources` and allowed ONLY as quarantined zero-shot eval benchmarks per `01-DR08-13` Override `#7` & `Spec §6.1` for Plan 6B Paper 2), `PLDU`/`PLDM`, `SynthWires`, `WireSegHR`, `SPL`, `SOD4SB`, `Seraphim`, `EfficientLoFTR`, `Sentinel-2` `10 m` imagery (gross-blunder `~45 m` CEP50 fallback only — NEVER primary `LTFJ`/`LTBX` corridor localization).

### 0.3 Zero-Overlap Multi-Agent File-Ownership Split (Phase 2 Execution)
Parallel Phase 2 builders **MUST** build inside isolated `/tmp/vertirta_m3_<agent>` directories using standalone dual-mode CMake guards (`if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)`), NEVER touching `build/dev-linux-x64` or root `CMakeLists.txt` until Wave 2 integration!

| Owner Agent | Phase 2 Wave | Strictly Owned Paths (Plan 6A) | Explicitly Forbidden Paths (Zero Overlap) |
|---|---|---|---|
| **`@vision-slz-eng`** | Wave 1 (Tasks 1, 3, 4) | `core/forward_hazard/**`, `core/gnss_denied/**`, `tests/forward_hazard/CMakeLists.txt`, `tests/forward_hazard/test_forward_hazard_fusion.cpp`, `tests/forward_hazard/test_forward_data_license_gate.py`, `tests/gnss_denied/**`, `config/forward_hazard_and_gnss_denied.yaml` | `sim/gazebo_plugins/**`, `ros2_ws/src/**`, `apps/vertigcs/**`, `core/flight_core/**`, `core/rta_companion/**`, `wasm/**`, `core/voice/**`, `core/forensics/**`, `benchmarks/vertibench_v2/**`, `docs/paper_corl/**`, `scripts/colab/**`, root `CMakeLists.txt` (until Wave 2) |
| **`@flight-stack-auditor`** | Wave 1 (Task 2) | `sim/gazebo_plugins/mmwave_77ghz/**`, `tests/forward_hazard/test_mmwave_77ghz_sensor.cpp` (standalone buildable in Wave 1 via `sim/gazebo_plugins/mmwave_77ghz/tests/CMakeLists.txt`, registered in `tests/forward_hazard/CMakeLists.txt` in Wave 2) | All `@vision-slz-eng`, `@px4-sim-eng`, Plan 6B, Plan 6C paths, and frozen cores |
| **`@px4-sim-eng`** | Wave 1 (Tasks 5, 6) | `ros2_ws/src/vertirta_forward_hazard_node/**`, `ros2_ws/src/vertirta_gnss_denied_node/**`, `apps/vertigcs/src/m3/**`, `apps/vertigcs/tests/m3_panel.test.ts`, plus minimal additive hooks in `apps/vertigcs/index.html`, `apps/vertigcs/src/main.ts`, `apps/vertigcs/src/adapters/maplibre_deck_adapter.ts`, `apps/vertigcs/src/styles/vertigcs.css` | All `core/**`, `sim/**`, `wasm/**`, Plan 6B, Plan 6C paths, root `CMakeLists.txt` |
| **`@vision-slz-eng` (Wave-2 Integrator)** | Wave 2 (Task 7 — after Plan 6A & 6B Wave-1 PASS) | Root `CMakeLists.txt` (additive M3 `if(EXISTS ...)` subdirs), `.github/workflows/m3_perception_voice_forensics.yml` (with pinned `libsodium 1.0.22` + `libz3-dev`), `tests/ci/test_m3_workflow_contracts.py`, `scripts/ci/check_zero_gpl_symbols.py` (additive M3 GPL patterns: `openvins`, `vins_fusion`, `orb_slam`, `hybvio`, `sqrtvins`, `blenderproc`, `dinov2_salad`, `ultralytics`, `sample4geo`, `espeak_ng`, `libespeak`, `piper_phonemize`, `piper1_gpl`) | Frozen `core/flight_core/**`, frozen `core/rta_companion/**`, frozen `wasm/**`, Plan 6B/6C implementation files |

### 0.4 Zero-Waste Colab `G4` Lock
- **Phase 1 (Plans) & Phase 2 (Local C++/Python/ROS2/VertiGCS/Dry-Run Code):** **0.00 CU spent — NOBODY runs `colab new`.** Every ONNX adapter (`M-LSD-WireSeg`, `D-FINE-N`, `XFeat+LighterGlue`, `MegaLoc`) ships a deterministic C++20 synthetic/CPU fallback so all unit, ASan/UBSan, and CI tests pass locally before Phase 3 G4 training exports real `.onnx` weights (orchestrated by Plan 6C).

---

## 1. Review Focus (5 High-Risk Failure Modes to Audit Against)

1. **Failure Mode #1 — Stale `DR-10` Constants Leaking Into Code:** Copying `R_veh = 4.5 m`, `v_max = 25 m/s`, `LTBX = 40°59'56"N`, PX4 `v1.18` `vehicle_gnss`/`SENS_GNSS0_DELAY`, or TU116 `enable_fp16 = true` directly from `DR-10` breaks `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`, PX4 `v1.16` uXRCE-DDS, and `GTX 1650 SUPER` latency (`2.5–3.4×` FP16 regression). *Guard:* Every constant is pinned in `config/forward_hazard_and_gnss_denied.yaml` and asserted in Catch2 tests.
2. **Failure Mode #2 — Radar Grazing-Angle Cliff (`-45 dBsm` past `20°` Skew) & Uncapped Clutter Speed:** Smooth crane hoist cables lack ACSR helical Bragg lobes and drop below radar SNR past `20°` skew; flying at `22 m/s` on vision-only cues exceeds stopping distance (`71.2 m` vs `~45 m` safe wire visibility). *Guard:* `ForwardHazardFusion` activates `> 3°` SVD multi-frame parallax fallback + conservative horizon standoff capsule AND forces `v_max_cap_mps <= 15.0 m/s` (`35.8 m` stop) whenever radar drops below quality threshold, while capping outputs strictly at `WireCapsuleTable::kMaxCapsules` (`64`) and `IntruderTrackerKf::kMaxTracks` (`16`).
3. **Failure Mode #3 — GNSS-Coupled VIO Poisoning & Marmara Wave-Glint False-Alarm Flood:** Feeding raw GNSS into VIO contaminates the `15 s` baseline before a slow `0.5–2.0 m/s` ramp is caught; conversely, running land MCUSUM thresholds (`0.25 / 2.0`) over open Marmara Sea waves causes chronic false alarms when KLT features track moving crests. *Guard:* `BasaltVioAdapter` has ZERO GNSS inputs, masks water pixels from `PIDNet-S` segmentation before KLT feature tracking, and `GnssSpoofGuard` maintains an explicit `L2 = 150` (`15 s`) baseline + `L1 = 50` (`5 s`) guard band with adaptive `k_d = 0.25, h_t = 2.0` (land `< 20%` water -> `<= 1.4 s` P95 TTD) vs `k_d = 0.35, h_t = 3.4` (water `> 35%` + corroboration), executing a `2.0 s` ring-buffer rollback upon latch.
4. **Failure Mode #4 — Missing `dh/dt` Time-Varying Lie-Derivative CTR Margin or Editing Frozen `CbvfSafetyFilter`:** Inflating CTR walls only statically (`radius_h_m += 3*sigma`) ignores the rate of covariance growth `d/dt(3*sqrt(n^T P n))`, causing barrier infeasibility during sustained GNSS-denied cruise near the `LTFJ` J2 pinch (`29.19212° E`). *Guard:* `CtrInflationAdapter` injects BOTH `radius_h_m = r_base + 3*sqrt(n^T P n) + 2.0` AND `vel_ned = r_dot_infl * er` into frozen `CbvfSafetyFilter::step_100hz` (`kind = 3`, `z_eff < -5 m` branch) with ZERO edits to `core/rta_companion/`, verified by `0` `LTFJ`/`LTBX` breaches across a 10-minute Monte Carlo drift run.
5. **Failure Mode #5 — GPL/LGPL/NC Contamination, Nadir/Forward Dataset Cross-Leak, or `flight_core`/WASM Size Regression:** Accidentally pulling `OpenVINS`/`ORB-SLAM3`/`SuiteSparse`/`BlenderProc`/`Ultralytics`, letting `TTPLA` leak into `ml/slz_perception/` nadir training, or adding M3 code to `core/flight_core/` or `wasm/` breaks the Apache-2.0 license firewall or the `28,886 B` Cortex-M7 / `38,470 B` WASM locks. *Guard:* `test_forward_data_license_gate.py`, extended `check_zero_gpl_symbols.py`, `check_arch_isolation.py`, and CI binary-size gates enforce hard failure on any violation.

---

## 2. File Structure Overview

```text
config/
  forward_hazard_and_gnss_denied.yaml           # Canonical M3 params (R_veh=2.10, LTBX 40°59'26"N, PX4 v1.16, FP16=false)

core/forward_hazard/
  CMakeLists.txt                                # Dual-mode (standalone Wave 1 + root Wave 2); links vertirta::rta_companion
  include/vertirta/forward_hazard/
    forward_hazard_provider.hpp                 # IForwardHazardProvider port + ForwardSensorBundle + ForwardHazardOutput
    horizon_split_stack.hpp                     # 45° HFOV 1280x720 -> 1280x320 -> 640x640 split-stack (f_net=1545.1 px, 16px overlap)
    onnx_forward_detectors.hpp                  # M-LSD-WireSeg (clDice) + D-FINE-N ORT C++20 adapter (enable_fp16=false TU116 lock)
    se3_pose_ring_buffer.hpp                    # Fixed-capacity 200 Hz (400-sample, 2.0 s) zero-heap SE(3) SLERP pose ring buffer
    wire_span_ekf.hpp                           # 6-state wire-span EKF + chi^2<=9.21 interp-plane gating + >3° SVD parallax fallback
    forward_hazard_fusion.hpp                   # Main 30 Hz fusion engine -> WireCapsuleTable (<=64) + IntruderTrackerKf (<=16) + 15 m/s cap
  src/
    horizon_split_stack.cpp
    onnx_forward_detectors.cpp
    se3_pose_ring_buffer.cpp
    wire_span_ekf.cpp
    forward_hazard_fusion.cpp

sim/gazebo_plugins/mmwave_77ghz/
  CMakeLists.txt                                # Dual-mode; pure C++20 physics lib always builds; gz-sensors8/gz-sim8 behind VERTIRTA_HAS_GZ_SENSORS
  include/vertirta/sim/
    mmwave_77ghz_physics.hpp                    # 77 GHz Fresnel glint, ACSR Bragg lobes, -45 dBsm >20° cliff, Swerling I/III, Doppler separation
    mmwave_77ghz_sensor.hpp                     # #ifdef VERTIRTA_HAS_GZ_SENSORS gz::sensors::Sensor + gz::sim::System wrappers
  src/
    mmwave_77ghz_physics.cpp
    mmwave_77ghz_sensor.cpp
  sdf/
    mmwave_77ghz_sensor_snippet.sdf             # <sensor type="custom" gz:type="mmwave_77ghz"> drop-in SDF snippet
  tests/
    CMakeLists.txt                              # Wave-1 standalone Catch2 test runner

core/gnss_denied/
  CMakeLists.txt                                # Dual-mode; links vertirta::rta_companion (read-only public API)
  include/vertirta/gnss_denied/
    basalt_vio_adapter.hpp                      # Strictly GNSS-decoupled Basalt (BSD-3) VIO adapter + 0.80 m stereo + radalt + water mask
    gnss_spoof_guard.hpp                        # 10 Hz H0..H3 detector (H1 RF/PX4 v1.16, H2 5-ep chi^2, H3a 15s/5s 2D MCUSUM, H3b GLRT, 2s rollback)
    visual_georeloc_adapter.hpp                 # 0.5-1 Hz XFeat+LighterGlue (Apache-2.0) + MegaLoc (MIT) + 3-gate RANSAC/hull/Mahalanobis + Joseph KF
    ctr_inflation_fsm.hpp                       # P_pos propagation, 3*sqrt(n^T P n)+b_map + dh/dt -> frozen CbvfSafetyFilter, 3-state FSM
  src/
    basalt_vio_adapter.cpp
    gnss_spoof_guard.cpp
    visual_georeloc_adapter.cpp
    ctr_inflation_fsm.cpp

ros2_ws/src/vertirta_forward_hazard_node/
  CMakeLists.txt                                # ament_cmake + pure C++20 core + #ifdef VERTIRTA_WITH_ROS2
  package.xml
  config/forward_hazard_node.yaml
  include/vertirta_forward_hazard_node/forward_hazard_core.hpp
  src/forward_hazard_node.cpp
  test/CMakeLists.txt
  test/test_forward_hazard_node.cpp

ros2_ws/src/vertirta_gnss_denied_node/
  CMakeLists.txt                                # ament_cmake + pure C++20 core + #ifdef VERTIRTA_WITH_ROS2 (PX4 v1.16 vehicle_gps_position)
  package.xml
  config/gnss_denied_node.yaml
  include/vertirta_gnss_denied_node/gnss_denied_core.hpp
  src/gnss_denied_node.cpp
  test/CMakeLists.txt
  test/test_gnss_denied_node.cpp

apps/vertigcs/
  src/m3/
    m3_types.ts                                 # WireCapsuleOverlay, IntruderOverlay, GnssDeniedTelemetry, CtrInflationOverlay
    m3_sim_replay.ts                            # Deterministic Kartal-Maltepe M3 demo scenario (wire encounter + slow-ramp spoof + 3σ CTR inflation)
    m3_deck_layer.ts                            # Deck.gl 3D catenary wire capsules + dynamic 3σ inflated LTFJ/LTBX CTR walls
    m3_panel.ts                                 # Collapsible M3 Perception & GNSS-Denied HUD panel (zero flight_core/WASM bloat)
  tests/m3_panel.test.ts                        # Vitest unit tests for M3 types, replay state machine, and Deck.gl layer generation
  index.html                                    # Minimal additive M3 collapsible panel section
  src/main.ts                                   # Minimal additive bindM3Panel(...) wiring
  src/adapters/maplibre_deck_adapter.ts         # Minimal additive updateM3Snapshot(...) overlay slot
  src/styles/vertigcs.css                       # Minimal additive M3 panel styling

tests/forward_hazard/
  CMakeLists.txt
  test_forward_hazard_fusion.cpp                # Catch2: split-stack, SE(3) SLERP, EKF/SVD fallback, <=64/<=16 caps, 15 m/s degraded cap, zero-heap
  test_mmwave_77ghz_sensor.cpp                  # Catch2: Fresnel 1/R^3, Bragg lobes, -45 dBsm >20° cliff, Swerling I/III, Doppler + azimuth leakage
  test_forward_data_license_gate.py             # Pytest: TTPLA forward-only (still banned in nadir SLZ), TEIAS/Purdue allow, GPL/NC/DUA hard-ban

tests/gnss_denied/
  CMakeLists.txt
  test_gnss_spoof_mcusum_vio.cpp                # Catch2: H0..H3 TTD (<=1.4 s P95 on 0.5-2.0 m/s ramps), water-glint FAR, 2.0 s rollback, zero-GNSS VIO
  test_visual_reloc_and_ctr_inflation.cpp       # Catch2: XFeat 3-gate + Joseph KF, MegaLoc re-init, 3σ+dh/dt into frozen CbvfSafetyFilter, 0 LTFJ/LTBX breaches

tests/ci/
  test_m3_workflow_contracts.py                 # Verifies .github/workflows/m3_perception_voice_forensics.yml locks, zero-GPL, 28,886 B flight_core, WASM

.github/workflows/
  m3_perception_voice_forensics.yml             # Full M3 CI: x86_64/arm64 C++20 + ASan/UBSan + Python + Zero-GPL + Arch Isolation + VertiGCS + Size Locks
```

---

## 3. Concrete TDD Implementation Tasks

### Task 1: Forward Wire/Intruder Radar-Vision Fusion (`core/forward_hazard/`) & Forward Dataset License Gate (`@vision-slz-eng` — Wave 1)

**Files:**
- Create: `config/forward_hazard_and_gnss_denied.yaml`
- Create: `core/forward_hazard/CMakeLists.txt`
- Create: `core/forward_hazard/include/vertirta/forward_hazard/{forward_hazard_provider,horizon_split_stack,onnx_forward_detectors,se3_pose_ring_buffer,wire_span_ekf,forward_hazard_fusion}.hpp`
- Create: `core/forward_hazard/src/{horizon_split_stack,onnx_forward_detectors,se3_pose_ring_buffer,wire_span_ekf,forward_hazard_fusion}.cpp`
- Create: `tests/forward_hazard/CMakeLists.txt`
- Create: `tests/forward_hazard/test_forward_hazard_fusion.cpp`
- Create: `tests/forward_hazard/test_forward_data_license_gate.py`

**Exact Engineering Contracts:**
- **`IForwardHazardProvider`** (`forward_hazard_provider.hpp`):
  - Pure C++20 port abstracting forward hazard extraction (`bool step(const ForwardSensorBundle& in, ForwardHazardOutput& out) noexcept`).
  - `ForwardHazardOutput` carries: fixed-size `vertirta::rta::WireCapsuleTable` (`<= 64` capsules, `sizeof(WireCapsule) <= 48 B`, `R_veh = 2.10 m`), `vertirta::rta::IntruderTrackerKf` tracks (`<= 16` tracks), `double recommended_v_max_mps` (`22.0` nominal, clamped to **`<= 15.0 m/s`** when `radar_degraded || vision_only_grazing_fallback`), `bool radar_degraded`, `bool grazing_parallax_active`, `uint32_t active_wire_spans`, `uint32_t active_intruders`.
- **`HorizonSplitStackPreprocessor`** (`horizon_split_stack.hpp/.cpp`):
  - Input: `1280×720` `45°` HFOV narrow-FOV forward RGB image (`f_native = 1545.1 px`), IMU roll/pitch stabilization (`±20°` derotation around optical axis), horizon ROI crop `1280×320` -> stacked vertically into `640×640` tensor (`top = [0..639]` + `bottom = [640..1279]` with `16 px` horizontal overlap `[624..639]` / `[640..655]` and explicit normalized `(u, v)` coordinate map mapping back to `1280×720` native pixels with `< 0.5 px` round-trip error).
  - Explicitly asserts decoupling from the locked `512×512` `80°` nadir SLZ camera (`fx = 305.09 px`).
- **`OnnxForwardDetectors`** (`onnx_forward_detectors.hpp/.cpp`):
  - Hosts `M-LSD-WireSeg` (`clDice`-trained thin-wire line-segment + probability head, `Apache-2.0`) and `D-FINE-N` (small-object bird/drone/pylon detector, `Apache-2.0`).
  - `ForwardOnnxConfig`: `enable_fp16 = false` (mandatory `TU116` default per `01-DR08-13` Override `#6`), `enable_cuda_graph = true`, `vram_budget_mib = 300`, `time_slice_phase` (`CRUISE_FORWARD` vs `DESCENT_NADIR_SLZ` sharing one CUDA context).
  - Behind `#ifdef VERTIRTA_HAS_ONNXRUNTIME`, invokes real `Ort::Session`; without ONNX Runtime (or when `use_deterministic_mock = true`), executes a deterministic zero-heap synthetic projection backend so Catch2 & CI run identically on CPU-only runners.
- **`Se3PoseRingBuffer`** (`se3_pose_ring_buffer.hpp/.cpp`):
  - Fixed `std::array<TimedPoseSe3, 400>` (`200 Hz × 2.0 s`), zero heap allocation, exact quaternion SLERP + linear NED position interpolation for asynchronous 30 Hz camera and 15–20 Hz radar timestamps, rejecting extrapolation `> 25 ms` as stale.
- **`WireSpanEkf` & `ForwardHazardFusion`** (`wire_span_ekf.hpp/.cpp`, `forward_hazard_fusion.hpp/.cpp`):
  - Associates 30 Hz `M-LSD-WireSeg` line segments via 3D camera interpretation planes (`n_cam = (u0 × u1) / ||u0 × u1||`) transformed through `Se3PoseRingBuffer` into local NED, gated by 2-DOF Mahalanobis `chi^2 <= 9.21` (`p = 0.01`).
  - Fuses 15–20 Hz radar glint/Bragg points (range `R`, azimuth `az`, elevation `el`, radial Doppler `v_r`, SNR) into a 6-state span EKF `[p_mid_n, p_mid_e, p_mid_d, dir_az, dir_el, sag_m]`.
  - **Grazing-angle fallback (`-45 dBsm` cliff past `20°` skew):** when radar lacks returns on a persistent vision wire track (`>= 3` consecutive frames), triangulates via multi-frame SVD once camera baseline parallax exceeds **`3.0°`**; before `3.0°` parallax is reached, emits a conservative horizon standoff capsule (`r_standoff = 35.0 m`, `radius_m = 2.10 + 1.50 = 3.60 m`) AND asserts `radar_degraded = true` -> **`recommended_v_max_mps <= 15.0 m/s`** (`35.8 m` stopping distance at `a_xy = 5.5 m/s^2`, `t_react = 0.6 s`).
  - Tessellates each confirmed catenary into `<= 4` piecewise `WireCapsule` segments (`R_veh = 2.10 m` clearance added by `WireCapsuleTable`, capped strictly at `<= 64` capsules prioritized by minimum time-to-collision).
  - Routes Doppler-separated dynamic returns (`|v_res| > gate`) AND static-gate returns confirmed by `D-FINE-N` (hovering drones/birds!) into `IntruderTrackerKf` (`<= 16` tracks).

**TDD Checklist:**
- [ ] **Step 1.1:** Write `config/forward_hazard_and_gnss_denied.yaml` pinning all canonical M3 parameters (`R_veh: 2.10`, `v_max_nominal_mps: 22.0`, `v_max_radar_degraded_mps: 15.0`, `forward_hfov_deg: 45.0`, `forward_f_net_px: 1545.1`, `enable_fp16: false`, `ltbx_arp_lat_deg: 40.990556`, `ltbx_arp_lon_deg: 29.216111`, `px4_gps_topic: /fmu/out/vehicle_gps_position`, `px4_gps_delay_param: EKF2_GPS_DELAY`).
- [ ] **Step 1.2:** Write failing Catch2 tests in `tests/forward_hazard/test_forward_hazard_fusion.cpp` covering:
  - `1280×320 -> 640×640` split-stack coordinate round-trip (`< 0.5 px` across left/right halves and `16 px` overlap seam) + IMU roll derotation.
  - `Se3PoseRingBuffer` 200 Hz SLERP accuracy (`< 1e-4 rad` orientation, `< 1e-4 m` translation) and `> 25 ms` stale rejection.
  - Interpretation-plane `chi^2 <= 9.21` gating + radar-vision 6-state EKF convergence (`< 0.6 m` RMS wire localization at `80 m`).
  - Radar `-45 dBsm` grazing dropout -> conservative horizon standoff capsule before `3°` parallax, SVD triangulation after `> 3°` parallax, AND mandatory `recommended_v_max_mps <= 15.0 m/s`.
  - Hovering drone (`|v_res| < 0.3 m/s` inside static Doppler gate) rescued by `D-FINE-N` cross-check into `IntruderTrackerKf`.
  - Overflow stress test (`120` candidate spans, `32` candidate intruders) verifying strict `<= 64` `WireCapsuleTable` and `<= 16` `IntruderTrackerKf` limits, `sizeof(WireCapsule) <= 48`, and zero dynamic allocations in `step()`.
- [ ] **Step 1.3:** Write failing pytest license firewall test `tests/forward_hazard/test_forward_data_license_gate.py` verifying:
  - Forward-wire manifest allows ONLY `TTPLA` (`Apache-2.0`), `Mendeley TEİAŞ` (`CC BY 4.0`), `Purdue UAV` (`BSD-3`), `Gazebo_Istanbul_Catenary_Synth` (`Apache-2.0`).
  - Hard-rejects `BlenderProc`, `InsPLAD`, `DDOS`, `Drone-vs-Bird`, `UAVid`, `STN PLAD`, `PAMPC`, `Ultralytics`, `SegFormer`, `Deep Hough Transform`, `DINOv3` everywhere, AND rejects `UAVScenes` (alongside `AeroScapes`/`ICG_Semantic_Drone`) from `train_sources` while permitting `UAVScenes` strictly under quarantined zero-shot eval manifests (`01-DR08-13` Override `#7`).
  - Explicitly verifies `TTPLA` remains BANNED inside `ml/slz_perception/data/license_gate.py` (nadir SLZ firewall uncompromised!).
- [ ] **Step 1.4:** Implement `core/forward_hazard/` headers, sources, and dual-mode `core/forward_hazard/CMakeLists.txt` (do NOT create `CMakeLists.txt` until all listed `.cpp` files exist on disk!).
- [ ] **Step 1.5:** Build & verify in isolated `/tmp/vertirta_m3_vision` (Release + ASan/UBSan + pytest):
  - `wsl -d Ubuntu -- bash -lc 'cmake -S /mnt/c/PROJECTS/AEGIS-VTOL/core/forward_hazard -B /tmp/vertirta_m3_vision/fwd_rel -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/vertirta_m3_vision/fwd_rel -j && ctest --test-dir /tmp/vertirta_m3_vision/fwd_rel --output-on-failure'`
  - Repeat with `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`.
  - Run `uv run --no-project --python 3.11 --with pytest --with pyyaml pytest tests/forward_hazard/test_forward_data_license_gate.py -v`.

---

### Task 2: 77 GHz mmWave Radar Physics & Gazebo Harmonic Sensor Plugin (`sim/gazebo_plugins/mmwave_77ghz/`) (`@flight-stack-auditor` — Wave 1)

**Files:**
- Create: `sim/gazebo_plugins/mmwave_77ghz/include/vertirta/sim/mmwave_77ghz_physics.hpp`
- Create: `sim/gazebo_plugins/mmwave_77ghz/src/mmwave_77ghz_physics.cpp`
- Create: `sim/gazebo_plugins/mmwave_77ghz/include/vertirta/sim/mmwave_77ghz_sensor.hpp`
- Create: `sim/gazebo_plugins/mmwave_77ghz/src/mmwave_77ghz_sensor.cpp`
- Create: `sim/gazebo_plugins/mmwave_77ghz/sdf/mmwave_77ghz_sensor_snippet.sdf`
- Create: `sim/gazebo_plugins/mmwave_77ghz/CMakeLists.txt`
- Create: `sim/gazebo_plugins/mmwave_77ghz/tests/CMakeLists.txt`
- Create: `tests/forward_hazard/test_mmwave_77ghz_sensor.cpp`

**Exact Engineering Contracts:**
- **`MmWave77GHzPhysics`** (`mmwave_77ghz_physics.hpp/.cpp`, pure C++20, zero Gazebo dependency so it ALWAYS builds & tests in WSL2 and CI):
  - **Radar Profiles:** `Tier1B_CascadedMimo` (`AWR2243`/`ARS540`: `f_c = 77.0 GHz`, `lambda = 3.893 mm`, `P_tx = 13 dBm`, `G_ant = 16 dBi`, `N_ch = 192` virtual MIMO, `B = 1.0 GHz`, `T_cpi = 20 ms`, `NF = 12 dB`, `az_bw = 1.4°`, `el_bw = 2.0°`, `max_range_m = 150 m`) and `Tier2_ActivePa` (`P_tx = 24 dBm`, `G_ant = 20 dBi`, `max_range_m = 250 m`).
  - **Wire Scattering RCS Model (`compute_wire_rcs_dbsm`):**
    - Normal-incidence Fresnel specular glint (`|theta_skew| <= 2.5°`): `sigma_glint = (pi / lambda) * d_wire * R * L_eff_factor`, reproducing `+1.96 dBsm` (`10 mm` wire) to `+5.94 dBsm` (`25 mm` wire) at `R = 100 m`, scaling with **`1 / R^3`** received power (not `1 / R^4` point target!).
    - Helical stranded ACSR Bragg diffraction lobes (`wire_kind == STRANDED_ACSR`): discrete harmonic peaks at `theta_skew in [7.0°, 14.1°]` (`sigma_bragg in [-14, -6] dBsm`), strictly ceasing for `|theta_skew| > 15.0°`.
    - Smooth crane hoist rope (`wire_kind == SMOOTH_CRANE_ROPE`): **ZERO** Bragg lobes outside normal specular glint.
    - **Grazing/Skew Cliff (`|theta_skew| > 20.0°`):** RCS plunges to **`<= -45.0 dBsm`**, reproducing the physical radar blind spot that triggers Task 1's vision-parallax + `15 m/s` fallback.
  - **Swerling Fluctuation & Detection:** Deterministic seeded PRNG (PCG32/SplitMix64, zero `std::rand`) implementing **Swerling I** (`17.0 dB` required SNR at `P_d = 0.90, P_fa = 1e-6`), **Swerling III** (`13.0 dB`), and **Swerling 0/V** non-fluctuating (`8.5 dB` reference) detection margin checks.
  - **Ego-Motion-Compensated Doppler Separation (`classify_doppler_target`):**
    - Computes `v_res = v_r_meas + dot(v_ego_sensor, u_hat_target)`.
    - Computes off-boresight azimuth leakage variance `sigma_v_leak = |v_ego_xy| * sin(|az_rad|) * sigma_az_rad`, widening the `3σ` static clutter gate smoothly from **`±0.64 m/s`** (boresight `az = 0°`) to **`±1.01 m/s`** at `|v_ego| = 20 m/s, |az| = 30°`.
    - Returns `DYNAMIC_INTRUDER` if `|v_res| > gate_3sigma_mps`, else `STATIC_OR_HOVER_CANDIDATE` (flagged for Task 1's `D-FINE-N` cross-check to catch hovering drones/birds).
- **`MmWave77GHzSensor` & `MmWave77GHzSystem`** (`mmwave_77ghz_sensor.hpp/.cpp`):
  - Guarded cleanly by `#ifdef VERTIRTA_HAS_GZ_SENSORS` (auto-detected via `find_package(gz-sensors8 QUIET)` / `find_package(gz-sim8 QUIET)` in `CMakeLists.txt`).
  - Subclasses `gz::sensors::Sensor` (`Load`, `Update`) and `gz::sim::System` (`Configure`, `PreUpdate`, `PostUpdate`), parsing `<sensor type="custom" gz:type="mmwave_77ghz">` from `mmwave_77ghz_sensor_snippet.sdf` and emitting radar point detections (`range, az, el, v_r, snr_db, rcs_dbsm, doppler_class`).

**TDD Checklist:**
- [ ] **Step 2.1:** Write failing Catch2 tests in `tests/forward_hazard/test_mmwave_77ghz_sensor.cpp` asserting:
  - Normal-incidence `10 mm` (`+1.96 ± 0.4 dBsm`) and `25 mm` (`+5.94 ± 0.4 dBsm`) Fresnel RCS at `100 m` and exact `1/R^3` received-power slope (`~9.0 dB` per factor-of-2 range change vs `12.0 dB` for point targets).
  - Stranded ACSR Bragg peaks present in `7.0°–14.1°`, absent `> 15.0°`; smooth crane rope has zero Bragg peaks.
  - Skew angle `> 20.0°` yields `rcs_dbsm <= -45.0 dBsm` and zero Tier-1B detections at `60 m`.
  - Swerling I (`17.0 dB`), Swerling III (`13.0 dB`), and Swerling 0 (`8.5 dB`) SNR thresholds match analytical tables within `±0.3 dB`.
  - Ego-motion Doppler compensation: static pole at `20 m/s` / `30°` azimuth falls inside widened `±1.01 m/s` gate (`STATIC_OR_HOVER_CANDIDATE`); boresight gate equals `±0.64 ± 0.05 m/s`; moving drone at `v_res = 2.5 m/s` classifies `DYNAMIC_INTRUDER`.
  - SDF snippet `mmwave_77ghz_sensor_snippet.sdf` validates XML structure and declares `<sensor type="custom" gz:type="mmwave_77ghz">` with pure LF line endings.
- [ ] **Step 2.2:** Implement `mmwave_77ghz_physics.{hpp,cpp}`, `mmwave_77ghz_sensor.{hpp,cpp}`, `mmwave_77ghz_sensor_snippet.sdf`, and dual-mode `CMakeLists.txt` + `tests/CMakeLists.txt`.
- [ ] **Step 2.3:** Build & verify in isolated `/tmp/vertirta_m3_radar` (Release + ASan/UBSan):
  - `wsl -d Ubuntu -- bash -lc 'cmake -S /mnt/c/PROJECTS/AEGIS-VTOL/sim/gazebo_plugins/mmwave_77ghz -B /tmp/vertirta_m3_radar/rel -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release -DVERTIRTA_BUILD_TESTS=ON && cmake --build /tmp/vertirta_m3_radar/rel -j && ctest --test-dir /tmp/vertirta_m3_radar/rel --output-on-failure'`

---

### Task 3: GNSS-Denied Layer 1 (`Basalt` BSD-3 Decoupled VIO) & Layer 2 (`H0..H3` Anti-Spoofing/Jamming Guard) (`@vision-slz-eng` — Wave 1)

**Files:**
- Create: `core/gnss_denied/include/vertirta/gnss_denied/basalt_vio_adapter.hpp`
- Create: `core/gnss_denied/src/basalt_vio_adapter.cpp`
- Create: `core/gnss_denied/include/vertirta/gnss_denied/gnss_spoof_guard.hpp`
- Create: `core/gnss_denied/src/gnss_spoof_guard.cpp`
- Create: `tests/gnss_denied/test_gnss_spoof_mcusum_vio.cpp`

**Exact Engineering Contracts:**
- **`BasaltVioAdapter`** (`basalt_vio_adapter.hpp/.cpp`):
  - **Strict Decoupling Contract:** API accepts ONLY belly stereo frame pairs (`baseline_m = 0.80 m` rigid carbon-boom default, `>= 0.50 m` floor), 200 Hz IMU pre-integration packets, nadir radar altimeter (`radalt_agl_m`) + barometric altitude (`baro_alt_m`) metric scale lock, and optional `PIDNet-S` `512×512` water-class bitmask (`water_fraction in [0, 1]`). **NO GNSS input struct or setter exists anywhere in `BasaltVioAdapter`** — compile-time proof against pre-detection spoof poisoning!
  - Purges KLT optical-flow tracks falling on `PIDNet-S` water pixels (`class == WATER`) to stop Marmara Sea wave-glint bias; computes exact `water_fraction` fed to Layer 2.
  - Optional `#ifdef VERTIRTA_HAS_BASALT` links BSD-3 `basalt-headers`/`basalt`; optional `#ifdef VERTIRTA_HAS_OKVIS2_BSD3` enforces `#if defined(OKVIS_USE_NN) || defined(SUITESPARSE_VERSION) #error "GPL/NC forbidden" #endif`. Default host/CI build uses an exact deterministic 200 Hz IMU+stereo+radalt error-state visual-inertial propagator for reproducible Monte Carlo testing.
- **`GnssSpoofGuard`** (`gnss_spoof_guard.hpp/.cpp`, 10 Hz update):
  - Consumes PX4 `v1.16` GNSS measurement struct (`GnssFixV116`: `stamp_ns`, `lat_deg`, `lon_deg`, `alt_hae_m`, `vel_ned`, `cn0_avg_dbhz`, `agc_cnt`, `hdop`, `eph_m`, `epv_m`, `fix_type`, `ekf2_gps_delay_ms`) + strictly decoupled `BasaltVioAdapter` pose/velocity + `water_fraction`.
  - Maintains a fixed-size `15 s` (`L2 = 150` epochs at 10 Hz) circular innovation history buffer and **`5 s` (`L1 = 50` epochs) guard band**, plus a `2.0 s` (`20`-epoch at 10 Hz / `400`-sample at 200 Hz) pre-detection pose rollback snapshot ring.
  - **4-Hypothesis Decision Engine (`GnssHypothesis`):**
    - `H0_NOMINAL`: Clean GNSS + VIO consistency.
    - `H1_RF_JAMMING` (`TTD < 0.2 s`): Triggers on rapid C/N0 drop (`< 24 dB-Hz` or `Delta C/N0 < -10 dB` over `0.2 s`), AGC saturation anomaly, `hdop > 2.5`, or loss of 3D lock -> latches GNSS-denied with **`sigma_init_m = 2.0 m`** (clean jam, minimal pre-detection bias).
    - `H2_STEP_MEACONING` (`TTD < 0.3 s`): 5-epoch windowed Mahalanobis chi-square innovation test (`chi^2_10dof > 23.21`, `p = 0.01`, or instantaneous step jump `> 4.0 m`) -> latches spoof with `2.0 s` rollback and **`sigma_init_m = 3.5 m`**.
    - `H3_SLOW_RAMP_SPOOF` (combined `H3a` + `H3b`):
      - **`H3a` Lagged-Baseline Bias-Compensated 2D Vector MCUSUM:** Estimates VIO drift bias strictly from the lagged window `[k - L2, k - L1)` (`t - 15 s` to `t - 5 s`), subtracts it from the current 2D velocity residual `r_v[k] = v_gnss_xy[k] - v_vio_xy[k] - b_hat_lag`, normalizes by `Sigma_r`, and updates vector MCUSUM `S_k = max(0, ||S_{k-1} + z_k|| - k_d) * unit(S_{k-1} + z_k)`.
      - **Adaptive Sea/Land Thresholds (`DR-10` Table):**
        - Land (`water_fraction < 0.20`): `k_d = 0.25`, `h_t = 2.0` -> **`P95 TTD <= 1.4 s`**, **`P99 undetected slip <= 0.85 m`** on `0.5–2.0 m/s` slow ramps!
        - Transition (`0.20 <= water_fraction <= 0.35`): linear interpolation of `(k_d, h_t)`.
        - Open Water (`water_fraction > 0.35`): `k_d = 0.35`, `h_t = 3.4` AND requires corroboration from either `H3b` or `H1` C/N0-AGC anomaly before latching, preventing Marmara wave-glint false-alarm floods (`FAR < 1e-4 / hr`).
      - **`H3b` Telescoping 2D Position-Increment GLRT:** Evaluates `1.5 s` and `4.0 s` position-increment consistency `||(p_gnss(t) - p_gnss(t-W)) - (p_vio(t) - p_vio(t-W)) - W * b_hat_lag||^2 / sigma_W^2` to catch ultra-slow `0.3–0.5 m/s` drift ramps and corroborate over water.
    - **Latch & Rollback Action:** Freezes GNSS fusion into EKF2 (`gnss_fused_allowed = false`), emits `rollback_pose_ned` from `t_latch - 2.0 s` propagated forward exclusively by decoupled `Basalt` VIO deltas (`2.0 s` rollback strips pre-detection spoof bias), and initializes `sigma_pos_m = 3.5 m` (spoof) or `2.0 m` (clean jam).

**TDD Checklist:**
- [ ] **Step 3.1:** Write failing Catch2 tests in `tests/gnss_denied/test_gnss_spoof_mcusum_vio.cpp` asserting:
  - `H1` RF jamming C/N0 drop & HDOP `> 2.5` triggers in **`<= 0.20 s`** with `sigma_init_m == 2.0`.
  - `H2` `6.0 m` meaconing step triggers in **`<= 0.30 s`** with `2.0 s` rollback active and `sigma_init_m == 3.5`.
  - `H3` 100-seed Monte Carlo across `0.5, 1.0, 1.5, 2.0 m/s` slow-ramp spoofing over land (`water_fraction = 0.05`) achieves **`P95 TTD <= 1.40 s`** and **`P99 post-rollback slip <= 0.85 m`**!
  - Open Marmara Sea (`water_fraction = 0.85`) wave-glint stress (`0.25 m/s` VIO drift jitter for `300 s`) produces **`0` false alarms** while still latching when corroborated by `H3b` or `H1` during real spoofing.
  - Water-pixel masking in `BasaltVioAdapter` rejects `100%` of synthetic wave-glint KLT features on `PIDNet-S` water pixels while retaining land features, and static_asserts zero GNSS member dependency.
- [ ] **Step 3.2:** Implement `basalt_vio_adapter.{hpp,cpp}` and `gnss_spoof_guard.{hpp,cpp}`.
- [ ] **Step 3.3:** Build & verify in isolated `/tmp/vertirta_m3_gnss` (Release + ASan/UBSan).

---

### Task 4: GNSS-Denied Layer 3 (`XFeat+LighterGlue` / `MegaLoc` Visual Geo-Reloc) & Dynamic `3σ + dh/dt` CBVF CTR Inflation + 3-State FSM (`@vision-slz-eng` — Wave 1)

**Files:**
- Create: `core/gnss_denied/include/vertirta/gnss_denied/visual_georeloc_adapter.hpp`
- Create: `core/gnss_denied/src/visual_georeloc_adapter.cpp`
- Create: `core/gnss_denied/include/vertirta/gnss_denied/ctr_inflation_fsm.hpp`
- Create: `core/gnss_denied/src/ctr_inflation_fsm.cpp`
- Create: `core/gnss_denied/CMakeLists.txt`
- Create: `tests/gnss_denied/CMakeLists.txt`
- Create: `tests/gnss_denied/test_visual_reloc_and_ctr_inflation.cpp`

**Exact Engineering Contracts:**
- **`VisualGeoRelocAdapter`** (`visual_georeloc_adapter.hpp/.cpp`, `0.5–1.0 Hz` local reloc):
  - **Local `0.5–1.0 m` Ortho MBTiles Interface:** Fetches local North-up ortho tile patch around current predicted Kartal local-TMERC coordinate computed purely via `vertirta::core::geodesy` (zero `libproj` / `GDAL` dependency, preserving pure C++20 portability alongside `check_arch_isolation.py`; equivalent to `+proj=tmerc +lat_0=40.886 +lon_0=29.188`), applies yaw North-up derotation and radalt GSD scale warp.
  - **Matcher Backends:** `XFeat + LighterGlue` (`Apache-2.0`, local matching) and `MegaLoc` (`MIT`, lost-in-space global retrieval triggered ONLY when `sigma_pos_m > 50.0 m`). Both keep `enable_fp16 = false` TU116 default behind `#ifdef VERTIRTA_HAS_ONNXRUNTIME` with a deterministic synthetic feature-correspondence backend for host/CI testing.
  - **Mandatory 3-Gate Acceptance Filter (`GeoRelocDecision`):**
    1. RANSAC inlier count `>= 25` AND reprojection RMSE `< 2.5 px`.
    2. Spatial distribution: normalized convex-hull area ratio of inliers **`> 0.25`** (rejects narrow road-strip / single-roof false locks).
    3. 2D Mahalanobis innovation gate: `(z - Hx)^T S^{-1} (z - Hx) <= 9.21` (`chi^2_2, 0.99`), bypassed ONLY during validated `MegaLoc` multi-frame lost-in-space re-init (`>= 2` spatially consistent keyframes within `15 m`).
  - **Joseph-Form Covariance Update:** Updates `P_pos` via numerically positive-definite Joseph form `P^+ = (I - KH) P^- (I - KH)^T + K R K^T` (`R` scaled by ortho GSD: `0.5 m` ortho -> `sigma_meas ~ 1.5–2.5 m`; explicitly documents & flags `10 m` Sentinel-2 imagery as `GROSS_BLUNDER_ONLY` with `sigma_meas >= 45.0 m` CEP50 so it can NEVER tighten `LTFJ`/`LTBX` corridor covariance!).
- **`CtrInflationFsm`** (`ctr_inflation_fsm.hpp/.cpp`, 10–100 Hz):
  - **Covariance Growth Model:** Between visual geo-reloc updates, propagates `2×2` horizontal covariance `P_pos` using VIO random-walk diffusion `q_rw * dt` plus systematic path-length drift `(kappa_drift * d_ol)^2` (default `kappa_drift = 0.008` = `0.8%` distance traveled; `q_rw = 0.04 m^2/s`), computing isotropic `sigma_pos_m = sqrt(lambda_max(P_pos))` and directional `sigma_n = sqrt(n^T P_pos n)` along each CTR wall outward normal `n`.
  - **Exact Time-Varying CBVF CTR Inflation Coupling into Frozen `CbvfSafetyFilter::step_100hz` (`0` edits to `core/rta_companion/`!):**
    - Uses canonical `config/istanbul_ltfj_airspace.yaml` geometry: Kartal datum `(40.8860° N, 29.1880° E)`, `LTFJ` ARP `(40.898333° N, 29.309167° E, R = 9260 m)` + NW J1/J2 pinch (`40.912786° N, 29.192122° E`), and **confirmed DHMİ AIP `LTBX` ARP (`40.990556° N, 29.216111° E, R = 9260 m` — `40°59'26"N, 029°12'58"E`, NEVER stale `40°59'56"N`)**.
    - Calls existing `vertirta::rta::AirspaceExclusion::extract_ctr_hazards(...)` to obtain nominal `HazardPoint` (`kind = 3`, `z_eff < -5 m`).
    - For each CTR wall hazard with outward normal `er = (pos_ned - h.pos_ned) / ||pos_ned - h.pos_ned||`:
      - Directional inflation radius: `delta_r_m = 3.0 * sqrt(er^T P_pos er) + b_map_m` (`b_map_m = 2.0 m`).
      - Outward expansion rate (Lie-derivative `dh/dt = -r_dot_infl`): `r_dot_infl_mps = max(0.0, (delta_r_m - prev_delta_r_m) / dt)` (low-pass bounded by analytic upper bound `3.0 * (er^T P_dot er) / (2 * max(sigma_n, 0.5))`).
      - Writes `h.radius_h_m += delta_r_m` AND **`h.vel_ned = {er.x * r_dot_infl_mps, er.y * r_dot_infl_mps, 0.0}`** (per verified `cbvf_safety_filter.cpp` L209–234, this makes `v_r = er · vel_ned - r_dot_infl_mps`, injecting exact `∂h/∂t` directly into the frozen barrier and QP!).
  - **3-State Navigation FSM (`GnssDeniedNavState`):**
    - `BOUNDED_GNSS_DENIED_CRUISE`: `sigma_pos_m < 25.0 m` (`3σ < 75.0 m`; tracks `nominal_3sigma_met = (3 * sigma_pos_m < 25.0)` for `0.5 m` ortho steady-state `8.8–17.2 m` verification). Normal speed cap (`<= 22.0 m/s`, subject to forward-hazard cap).
    - `DEGRADED_CAUTION`: `25.0 m <= sigma_pos_m < 65.0 m` (`3σ in [75.0, 195.0) m`). Enforces **`v_max_cap_mps = 12.0–15.0 m/s`** (default `13.5 m/s`, configurable `[12.0, 15.0]`) and emits westward safety bias away from `LTFJ`/`LTBX` (`max_lon_near_j2 = 29.1895° E`).
    - `DETERMINISTIC_VISUAL_SLZ_ABORT`: `sigma_pos_m >= 65.0 m` (`3σ >= 195.0 m`, latching strictly BEFORE the `80 m` CBVF corridor infeasibility floor) -> commands deterministic descent onto top-ranked `PIDNet-S` SLZ site (`VertiRTA` `SlzBayesMap`). Hysteresis (`-5.0 m`) prevents rapid chattering; `DETERMINISTIC_VISUAL_SLZ_ABORT` is latched until landing or validated recovery.

**TDD Checklist:**
- [ ] **Step 4.1:** Write failing Catch2 tests in `tests/gnss_denied/test_visual_reloc_and_ctr_inflation.cpp` asserting:
  - `VisualGeoRelocAdapter` rejects `< 25` inliers, rejects `>= 2.5 px` RMSE, rejects `< 0.25` convex-hull ratio, rejects `chi^2 > 9.21` outliers, accepts valid `0.5 m` ortho match with SPD Joseph-form `P_pos` reduction (`3σ` steady-state in `[8.8, 17.2] m < 25 m`).
  - `MegaLoc` triggers when `sigma_pos_m > 50.0 m` and recovers lock after `2` consistent keyframes; `Sentinel-2` `10 m` tile is tagged `GROSS_BLUNDER_ONLY` (`sigma_meas >= 45 m`) and NEVER shrinks `P_pos` below `45 m`.
  - `CtrInflationFsm` uses verified DHMİ AIP `LTBX` ARP `40.990556° N, 29.216111° E` (explicitly rejecting stale `40.99889° N`).
  - **Closed-loop Frozen `CbvfSafetyFilter::step_100hz` Proof:** Simulate an aircraft cruising at `18 m/s` parallel to the `LTFJ` NW J2 pinch under sustained GNSS denial (`10 min` drift + `0.5 Hz` reloc dropout) feeding `CtrInflationFsm` inflated hazards directly into **unmodified** `vertirta::rta::CbvfSafetyFilter::step_100hz` -> verify **`0` `LTFJ`/`LTBX` breaches**, verify `r_dot_infl > 0` increases braking margin vs static-only inflation (`dh/dt` test), and verify deterministic transition `BOUNDED_CRUISE -> DEGRADED_CAUTION (v_max <= 15 m/s) -> DETERMINISTIC_VISUAL_SLZ_ABORT (sigma_pos >= 65 m)`.
- [ ] **Step 4.2:** Implement `visual_georeloc_adapter.{hpp,cpp}`, `ctr_inflation_fsm.{hpp,cpp}`, dual-mode `core/gnss_denied/CMakeLists.txt`, and `tests/gnss_denied/CMakeLists.txt`.
- [ ] **Step 4.3:** Build & verify in isolated `/tmp/vertirta_m3_gnss` (Release + ASan/UBSan):
  - `wsl -d Ubuntu -- bash -lc 'cmake -S /mnt/c/PROJECTS/AEGIS-VTOL/core/gnss_denied -B /tmp/vertirta_m3_gnss/rel -G Ninja -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/vertirta_m3_gnss/rel -j && ctest --test-dir /tmp/vertirta_m3_gnss/rel --output-on-failure'`

---

### Task 5: ROS 2 M3 Nodes (`vertirta_forward_hazard_node` & `vertirta_gnss_denied_node`) (`@px4-sim-eng` — Wave 1)

**Files:**
- Create: `ros2_ws/src/vertirta_forward_hazard_node/{CMakeLists.txt,package.xml,config/forward_hazard_node.yaml}`
- Create: `ros2_ws/src/vertirta_forward_hazard_node/{include/vertirta_forward_hazard_node/forward_hazard_core.hpp,src/forward_hazard_node.cpp,test/CMakeLists.txt,test/test_forward_hazard_node.cpp}`
- Create: `ros2_ws/src/vertirta_gnss_denied_node/{CMakeLists.txt,package.xml,config/gnss_denied_node.yaml}`
- Create: `ros2_ws/src/vertirta_gnss_denied_node/{include/vertirta_gnss_denied_node/gnss_denied_core.hpp,src/gnss_denied_node.cpp,test/CMakeLists.txt,test/test_gnss_denied_node.cpp}`

**Exact Engineering Contracts (Matching `vertirta_perception_node` & `vertirta_uspace_node` Pattern):**
- **Host-Testable Pure C++20 Core + `#ifdef VERTIRTA_WITH_ROS2` Wrapper:** Both packages split logic into header/lib cores (`forward_hazard_core.hpp`, `gnss_denied_core.hpp`) tested directly via Catch2 without ROS 2 installed, while `src/*_node.cpp` compiles under `ament_cmake` with `VERTIRTA_WITH_ROS2=1`.
- **`vertirta_forward_hazard_node` ROS 2 Topic Contract (Zero Breaking Changes to `vertirta_msgs`!):**
  - Subscribes: `/vertirta/forward_camera/image_raw` (`sensor_msgs/Image`, `1280×720` `45°` HFOV), `/vertirta/mmwave_77ghz/detections` (`sensor_msgs/PointCloud2`), `/fmu/out/vehicle_odometry` (`px4_msgs/VehicleOdometry`), `/vertirta/imu_200hz` (`sensor_msgs/Imu`).
  - Publishes directly to existing RTA topics: **`/vertirta/wire_capsules`** (`vertirta_msgs/WireCapsuleArray`, `<= 64` spans), **`/vertirta/intruders`** (`vertirta_msgs/IntruderArray`, `<= 16` tracks), plus `/vertirta/forward_hazard_markers` (`visualization_msgs/MarkerArray`), `/vertirta/forward_hazard_status` (`std_msgs/String` JSON), and `/vertirta/forward_speed_cap_mps` (`std_msgs/Float32`).
- **`vertirta_gnss_denied_node` ROS 2 Topic Contract (Strict PX4 `v1.16` Lock!):**
  - Subscribes: **`/fmu/out/vehicle_gps_position`** (`px4_msgs/VehicleGpsPosition` — NEVER `v1.18` `vehicle_gnss`!), `/fmu/out/vehicle_odometry` (`px4_msgs/VehicleOdometry`), `/vertirta/belly_stereo/left/image_raw` & `right/image_raw` (`sensor_msgs/Image`), `/vertirta/radalt/range` (`sensor_msgs/Range`), `/vertirta/slz_water_fraction` (`std_msgs/Float32`).
  - Publishes: `/vertirta/gnss_denied_status` (`std_msgs/String` JSON carrying `H0..H3`, `sigma_pos_m`, `three_sigma_m`, `dh_dt_mps`, `nav_fsm_state`, `v_max_cap_mps`, `ltfj_inflated_margin_m`, `ltbx_inflated_margin_m`), `/vertirta/gnss_denied_odometry` (`nav_msgs/Odometry`), `/vertirta/ctr_inflation_markers` (`visualization_msgs/MarkerArray`), `/vertirta/gnss_speed_cap_mps` (`std_msgs/Float32`).
  - Uses explicit `EKF2_GPS_DELAY` YAML parameter (`gnss_denied_node.yaml`, default `110.0 ms`) and `UXRCE_DDS_SYNCT=1` time-sync assertion.

**TDD Checklist:**
- [ ] **Step 5.1:** Write failing Catch2 tests `test_forward_hazard_node.cpp` and `test_gnss_denied_node.cpp` verifying message serialization, topic name locks (`/vertirta/wire_capsules`, `/vertirta/intruders`, `/fmu/out/vehicle_gps_position`), `<= 64` / `<= 16` caps, radar-degraded `<= 15 m/s` cap propagation, and `H3` spoof -> `DEGRADED_CAUTION` -> `VISUAL_SLZ_ABORT` state broadcast.
- [ ] **Step 5.2:** Implement both ROS 2 packages (`CMakeLists.txt`, `package.xml`, YAML configs, core headers, `#ifdef VERTIRTA_WITH_ROS2` nodes).
- [ ] **Step 5.3:** Build & test standalone host cores in isolated `/tmp/vertirta_m3_ros2` (Release + ASan/UBSan) and verify YAML/XML lint + pure LF line endings.

---

### Task 6: `VertiGCS` M3 Perception & GNSS-Denied Telemetry Panel (`apps/vertigcs/`) (`@px4-sim-eng` — Wave 1)

**Files:**
- Create: `apps/vertigcs/src/m3/m3_types.ts`
- Create: `apps/vertigcs/src/m3/m3_sim_replay.ts`
- Create: `apps/vertigcs/src/m3/m3_deck_layer.ts`
- Create: `apps/vertigcs/src/m3/m3_panel.ts`
- Create: `apps/vertigcs/tests/m3_panel.test.ts`
- Modify (minimal additive hooks ONLY): `apps/vertigcs/index.html`, `apps/vertigcs/src/main.ts`, `apps/vertigcs/src/adapters/maplibre_deck_adapter.ts`, `apps/vertigcs/src/styles/vertigcs.css`

**Exact Engineering Contracts:**
- **ZERO Bloat to `flight_core` or WASM:** `core/flight_core/` (`28,886 B`) and `wasm/` (`38,470 B`) are **NOT TOUCHED**! All M3 GCS visualization lives purely in browser TypeScript (`apps/vertigcs/src/m3/`), mirroring the exact `apps/vertigcs/src/fleet/` M2 architecture.
- **Collapsible M3 Telemetry Panel (`#m3-panel`):**
  - Displays: Forward Hazard status (`M-LSD-WireSeg` + `D-FINE-N` + `77 GHz mmWave` active wire capsules `N/64`, intruders `M/16`, grazing fallback badge, speed cap `22.0 / 15.0 m/s`), GNSS-Denied Guard (`H0..H3` hypothesis badge, `MCUSUM` score & land/water mode, `2.0 s` rollback indicator), Visual Geo-Reloc (`XFeat+LighterGlue` inliers / `MegaLoc` re-init status), and 3-State Navigation FSM (`BOUNDED_CRUISE` / `DEGRADED_CAUTION` / `VISUAL_SLZ_ABORT`, `1σ` & `3σ` uncertainty, `dh/dt` CTR inflation rate, `LTFJ`/`LTBX` inflated clearance margins).
  - Interactive deterministic scenario buttons (matching Sandbox WASM demo ergonomics): `"Simulate Forward Wire + Grazing Cliff (15 m/s)"`, `"Inject 1.0 m/s Slow-Ramp GNSS Spoof (H3 MCUSUM)"`, `"Simulate Marmara Open-Water VIO Drift (3σ CTR Inflation)"`, `"Reset M3 Nominal"`.
- **Deck.gl 3D Overlay (`m3_deck_layer.ts`):**
  - Renders active 3D wire capsules (`PathLayer` / `LineLayer` colored by class: distribution, transmission, crane cable) and dynamic translucent `3σ + b_map` inflated `LTFJ` (with NW J2 pinch) & verified `LTBX` (`40°59'26"N`) CTR exclusion boundaries on top of `MapLibreDeckAdapter`.
- **Pages Compatibility:** Both `npm run build` and `npx vite build --base /VertiRTA/` must pass with zero warnings/errors, preserving relative asset resolution.

**TDD Checklist:**
- [ ] **Step 6.1:** Write failing Vitest tests in `apps/vertigcs/tests/m3_panel.test.ts` testing `M3SimReplay` scenario transitions (`wire_grazing` -> `vMaxCapMps <= 15.0`, `slow_ramp_spoof` -> `H3_SLOW_RAMP_SPOOF` in `<= 1.4 s` with `rollbackActive = true`, `water_drift` -> `DEGRADED_CAUTION` -> `VISUAL_SLZ_ABORT`), `buildM3DeckLayers()` layer generation & `LTBX` `40.990556 N` coordinate lock, and DOM panel render/collapse state.
- [ ] **Step 6.2:** Implement `apps/vertigcs/src/m3/*` and wire additive hooks in `index.html`, `main.ts`, `maplibre_deck_adapter.ts`, and `vertigcs.css`.
- [ ] **Step 6.3:** Run `npm test` (Vitest), `npm run build`, and `npx vite build --base /VertiRTA/` inside `apps/vertigcs/`, plus verify `wasm/` binary unchanged (`38,470 B < 150 KB`).

---

### Task 7: Wave-2 Root `CMakeLists.txt` Wiring, Extended Zero-GPL Gate & M3 CI Workflow (`@vision-slz-eng` / Wave-2 Integrator)

> **STRICT WAVE-2 GATE:** Execute Task 7 ONLY AFTER all Wave-1 builders (`Plan 6A` Tasks 1–6 AND `Plan 6B` Wave-1 C++ modules) pass standalone builds and tests!

**Files:**
- Modify: `CMakeLists.txt` (additive `if(EXISTS ...)` subdirs ONLY)
- Modify: `scripts/ci/check_zero_gpl_symbols.py` (additive M3 GPL/AGPL/LGPL symbol/header patterns)
- Create: `.github/workflows/m3_perception_voice_forensics.yml`
- Create: `tests/ci/test_m3_workflow_contracts.py`

**Exact Engineering Contracts:**
- **Root `CMakeLists.txt` Wave-2 Wiring:**
  - Append conditional `if(EXISTS ...)` subdirectories for Plan 6A (`core/forward_hazard`, `core/gnss_denied`, `sim/gazebo_plugins/mmwave_77ghz`, `tests/forward_hazard`, `tests/gnss_denied`) AND clean Wave-2 hook points for Plan 6B (`core/voice`, `core/forensics`, `tests/voice`, `tests/forensics` — guarded by `if(EXISTS ...)` so builds succeed cleanly regardless of merge order).
  - Add `tests/forward_hazard/test_mmwave_77ghz_sensor.cpp` into `tests/forward_hazard/CMakeLists.txt` linking `vertirta::mmwave_77ghz_physics`.
- **Extended `scripts/ci/check_zero_gpl_symbols.py`:**
  - Add M3 runtime/source bans across both Plan 6A (`core/forward_hazard`, `core/gnss_denied`, `sim/gazebo_plugins/mmwave_77ghz`) and Plan 6B (`core/voice`, `core/forensics`), keeping `schema_version` backward-compatible with M2 tests: `openvins`, `vins_fusion`, `orb_slam`, `hybvio`, `sqrtvins`, `blenderproc`, `dinov2_salad`, `ultralytics`, `sample4geo`, `espeak_ng`, `libespeak`, `piper_phonemize`, `piper1_gpl` (strictly keeping `whisper.cpp` MIT, `sherpa-onnx >= 2.0.0` Apache-2.0, `Basalt` BSD-3, `XFeat` Apache-2.0, `MegaLoc` MIT allowed!).
- **`.github/workflows/m3_perception_voice_forensics.yml`:**
  - Triggers on `push` (`main`, scoped paths for M3 `core/{forward_hazard,gnss_denied,voice,forensics}/**`, `sim/gazebo_plugins/mmwave_77ghz/**`, `ros2_ws/src/vertirta_{forward_hazard,gnss_denied,voice}_node/**`, `apps/vertigcs/**`, `benchmarks/vertibench_v2/**`, `scripts/colab/**`, `tests/{forward_hazard,gnss_denied,voice,forensics,ci}/**`, `CMakeLists.txt`), `pull_request`, and `workflow_dispatch`.
  - **Pinned Immutable `libsodium 1.0.22` + Native `libz3-dev` Setup (Mirroring Proven M2 `m2_uspace_and_physics.yml` CVE Lock):** Both `m3_cpp_matrix` and `m3_sanitizers_gpl_and_size_locks` install `libz3-dev` and build/cache immutable `libsodium-1.0.22.tar.gz` (`SHA-256: adbdd8f16149e81ac6078a03aca6fc03b592b89ef7b5ed83841c086191be3349`) into `~/.local/vertirta-libsodium-1.0.22` (NEVER vulnerable Ubuntu apt `libsodium 1.0.18`!), passing `-DVERTIRTA_REQUIRE_LIBSODIUM=ON -DVERTIRTA_REQUIRE_Z3=ON` so Plan 6B's `core/forensics/` Ed25519 signing and M2's `fleet_z3` never fall back.
  - **Job 1 (`m3_cpp_matrix`):** `ubuntu-24.04` (`x86_64`) + `ubuntu-24.04-arm` (`arm64`) Release C++20 build (`gcc-14`/`g++-14`, `-DVERTIRTA_ENABLE_STRICT_FP=ON -DVERTIRTA_REQUIRE_LIBSODIUM=ON -DVERTIRTA_REQUIRE_Z3=ON`), running all M3 Catch2 tests (`test_forward_hazard_fusion`, `test_mmwave_77ghz_sensor`, `test_gnss_spoof_mcusum_vio`, `test_visual_reloc_and_ctr_inflation`, plus Plan 6B `test_voice_*` / `test_forensics_*` if present) and standalone ROS 2 node host cores.
  - **Job 2 (`m3_sanitizers_gpl_and_size_locks`):**
    - ASan + UBSan on all M3 C++20 test binaries (built with native `libsodium 1.0.22` + `libz3-dev`, `-DVERTIRTA_REQUIRE_LIBSODIUM=ON -DVERTIRTA_REQUIRE_Z3=ON`).
    - `python3 scripts/ci/check_zero_gpl_symbols.py` scanning all built M3 libraries/tests AND `core/forward_hazard/`, `core/gnss_denied/`, `sim/gazebo_plugins/mmwave_77ghz/`, `core/voice/`, `core/forensics/`.
    - `python3 scripts/ci/check_arch_isolation.py` proving `core/flight_core/` and `core/rta_companion/` remain 100% isolated from AI/ROS/ONNX/Python/PROJ.
    - Cortex-M7 `arm-none-eabi-gcc` bare-metal build + `scripts/ci/check_bare_metal_size.py` + `scripts/ci/check_no_heap_symbols.py` proving `flight_core` stays frozen at **`<= 32,768 B` (`28,886 B` exact non-regression)**.
    - `scripts/ci/check_public_hygiene.py` proving zero secrets/BOM/CRLF/personal paths.
  - **Job 3 (`m3_python_and_dryrun_contracts`):** Runs `pytest` on `tests/forward_hazard/test_forward_data_license_gate.py`, `tests/ci/test_m3_workflow_contracts.py`, Plan 6B Python tests (when present), and Plan 6C pre-G4 dry-run manifest/script syntax checks (`python -m py_compile` + `--dry-run`, spending `0 CU`).
  - **Job 4 (`m3_vertigcs_build_and_wasm_lock`):** Node 20 `npm ci && npm test && npm run build && npx vite build --base /VertiRTA/` inside `apps/vertigcs/`, plus WASM binary size lock check (`<= 150 KB`, preserving `38,470 B`).

**TDD Checklist:**
- [ ] **Step 7.1:** Write failing `tests/ci/test_m3_workflow_contracts.py` asserting all 4 CI jobs, `x86_64`+`arm64` matrix, pinned `libsodium-1.0.22` (`SHA-256: adbdd8f16149e81ac6078a03aca6fc03b592b89ef7b5ed83841c086191be3349`) + `-DVERTIRTA_REQUIRE_LIBSODIUM=ON`, `libz3-dev` + `-DVERTIRTA_REQUIRE_Z3=ON`, ASan/UBSan, `check_zero_gpl_symbols.py` (incl. `espeak_ng`, `libespeak`, `piper_phonemize`, `piper1_gpl` + VIO/vision bans), `check_arch_isolation.py`, `check_bare_metal_size.py`, `check_public_hygiene.py`, `VertiGCS` `--base /VertiRTA/`, and `0 CU` dry-run lock exist in `.github/workflows/m3_perception_voice_forensics.yml`.
- [ ] **Step 7.2:** Update root `CMakeLists.txt`, extend `scripts/ci/check_zero_gpl_symbols.py`, and author `.github/workflows/m3_perception_voice_forensics.yml`.
- [ ] **Step 7.3:** Execute full unified WSL2 Release + ASan/UBSan + ARM Cortex-M7 + Python + VertiGCS verification suite below (zero regressions across M0/M1/M2/M3!).

---

## 4. Exact End-to-End Verification Commands

Run every command below before claiming Plan 6A complete (NEVER touch `build/dev-linux-x64` during Wave 1; clean up `/tmp/vertirta_m3_*` after verification):

```powershell
# 1. Unified WSL2 C++20 Release Build & Full CTest (M0+M1+M2+M3, strict FP)
wsl -d Ubuntu -- bash -lc '
  set -euo pipefail
  cmake -S /mnt/c/PROJECTS/AEGIS-VTOL -B /tmp/vertirta_m3_all_rel -G Ninja \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_BUILD_TYPE=Release -DVERTIRTA_BUILD_COMPANION=ON \
    -DVERTIRTA_BUILD_TESTS=ON -DVERTIRTA_ENABLE_STRICT_FP=ON
  cmake --build /tmp/vertirta_m3_all_rel -j
  ctest --test-dir /tmp/vertirta_m3_all_rel --output-on-failure -j4
'

# 2. ASan + UBSan on All Plan 6A C++20 Targets
wsl -d Ubuntu -- bash -lc '
  set -euo pipefail
  cmake -S /mnt/c/PROJECTS/AEGIS-VTOL -B /tmp/vertirta_m3_all_asan -G Ninja \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
    -DVERTIRTA_BUILD_COMPANION=ON -DVERTIRTA_BUILD_TESTS=ON -DVERTIRTA_ENABLE_STRICT_FP=ON
  cmake --build /tmp/vertirta_m3_all_asan --target \
    test_forward_hazard_fusion \
    test_mmwave_77ghz_sensor \
    test_gnss_spoof_mcusum_vio \
    test_visual_reloc_and_ctr_inflation -j
  ctest --test-dir /tmp/vertirta_m3_all_asan -R "(forward_hazard|mmwave_77ghz|gnss_spoof|visual_reloc)" --output-on-failure
'

# 3. Zero-GPL Symbol Gate, Architecture Isolation & Public Hygiene
wsl -d Ubuntu -- bash -lc '
  set -euo pipefail
  python3 /mnt/c/PROJECTS/AEGIS-VTOL/scripts/ci/check_zero_gpl_symbols.py \
    --targets /tmp/vertirta_m3_all_rel/tests/forward_hazard/test_forward_hazard_fusion \
              /tmp/vertirta_m3_all_rel/tests/forward_hazard/test_mmwave_77ghz_sensor \
              /tmp/vertirta_m3_all_rel/tests/gnss_denied/test_gnss_spoof_mcusum_vio \
              /tmp/vertirta_m3_all_rel/tests/gnss_denied/test_visual_reloc_and_ctr_inflation \
    --source-roots /mnt/c/PROJECTS/AEGIS-VTOL/core/forward_hazard \
                   /mnt/c/PROJECTS/AEGIS-VTOL/core/gnss_denied \
                   /mnt/c/PROJECTS/AEGIS-VTOL/sim/gazebo_plugins/mmwave_77ghz
  python3 /mnt/c/PROJECTS/AEGIS-VTOL/scripts/ci/check_arch_isolation.py --repo-root /mnt/c/PROJECTS/AEGIS-VTOL
'
uv run --no-project --python 3.11 --with pytest --with pyyaml pytest tests/forward_hazard/test_forward_data_license_gate.py tests/ci/test_m3_workflow_contracts.py -v
uv run --no-project --python 3.11 --with pyyaml python scripts/ci/check_public_hygiene.py

# 4. VertiGCS Vitest + Production Build + GitHub Pages Base Build (Zero WASM/flight_core Bloat)
Push-Location apps/vertigcs
npm test
npm run build
npx vite build --base /VertiRTA/
Pop-Location

# 5. Cleanup Isolated Build Dirs
wsl -d Ubuntu -- bash -lc 'rm -rf /tmp/vertirta_m3_all_rel /tmp/vertirta_m3_all_asan'
```

---

## 5. Exit Gates (Definition of Done for Plan 6A)

- [ ] **Zero Frozen-Core Drift:** `git diff core/flight_core/ core/rta_companion/ wasm/` is **EMPTY** (`flight_core` remains `28,886 B`, WASM remains `38,470 B`).
- [ ] **Forward Wire/Intruder & 77 GHz Radar:** `1280×320 -> 640×640` split-stack round-trip `< 0.5 px`; `1/R^3` Fresnel glint + ACSR Bragg (`7°–14.1°`) + `-45 dBsm` cliff (`> 20°`) + Doppler azimuth leakage (`±0.64 -> ±1.01 m/s`) verified in Catch2; grazing fallback forces `v_max <= 15.0 m/s`; strict `<= 64` capsules (`R_veh = 2.10 m`) & `<= 16` tracks.
- [ ] **GNSS-Denied VIO, Anti-Spoofing & `3σ + dh/dt` CTR Safety:** Zero GNSS inputs in `BasaltVioAdapter`; `H3` `0.5–2.0 m/s` ramps pass `P95 TTD <= 1.4 s` and `P99 slip <= 0.85 m` over land + `0` Marmara wave-glint false alarms; `XFeat+LighterGlue` 3-gate + Joseph KF keeps nominal `3σ < 25 m`; unmodified `CbvfSafetyFilter::step_100hz` achieves `0` `LTFJ`/`LTBX` (`40°59'26"N`) breaches with `dh/dt` Lie-derivative margin active.
- [ ] **License, Hygiene & Zero-Waste `G4` Lock:** `check_zero_gpl_symbols.py`, `check_arch_isolation.py`, `check_public_hygiene.py`, and `test_forward_data_license_gate.py` (confirming `TTPLA` forward-only & still banned in nadir SLZ) all PASS; `0.00 CU` spent until Phase 3.
