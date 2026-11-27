# VertiRTA `v1.0.0` — Cross-Benchmark Comparative Engineering Audit & Defect Ledger

**Audit Scope:** Read-only principal GNC & systems-engineering audit of `VertiRTA` (`v1.0.0`, initial pre-release audit at `20ff6c6`, with ledger items **B1–B7 resolved prior to `v1.0.0` release**) against six leading open-source UAV/eVTOL and autonomy reference stacks.  
**Assurance Posture:** Open-source research & software-verification baseline — zero airworthiness certification, Type Certificate, or DO-178C/DO-254 compliance claims. All metrics below are read directly from checked-in source files and SHA-256-verified JSON artifacts under `artifacts/v1.0.0/`, `artifacts/v3.1.0/`, `artifacts/v3.5.1/`, and `artifacts/v4.0.0/`.

---

## 1. Executive Verdict

`VertiRTA` sits in an unusual engineering niche: it is **far more rigorous than typical academic UAV repositories** on embedded determinism, formal/SIL verification, control-theoretic fault boundaries, and license hygiene, yet it **honestly lags production autopilots (PX4, ArduPilot, Autoware, industrial eVTOL stacks)** on physical flight-hardware validation, closed-loop Gazebo/SITL endurance, and real-world perception generalization.

| Dimension | Typical Academic UAV Repo | Production Stacks (PX4 / ArduPilot / Autoware) | `VertiRTA v1.0.0` Ground Truth |
| --- | --- | --- | --- |
| **Bare-Metal Determinism** | None (ROS/Python nodes, dynamic `std::vector`) | RTOS-hardened (NuttX/ChibiOS), hundreds of KiB–MiB | **`28,886 B` Cortex-M7 Flash / `0 B` lib RAM / `2,960 B` stack DAG / `0` heap / `0` recursion** |
| **Formal & Protocol SIL** | Unit tests only | Extensive SITL + hardware flight-hour corpus | **`ESBMC v8.7.0` `1,905/1,905` properties + Renode `v1.17.0` dual-FCC CAN-FD `12.50 ms` failover** |
| **Cross-ISA Numerical Parity** | Untested | Hardware-specific tuning | **0-ULP same-ISA & `max_abs_diff = 0.0` `x86_64 <-> arm64` (`p99 = 0.736 us` Cobalt 100)** |
| **Physical Hardware & Flight Logs** | Occasional indoor MoCap | Millions of real flight hours across hundreds of boards | **0 physical STM32H7 HITL runs (`PENDING_PHYSICAL_STM32H7_HITL`); 0 real vehicle flight hours** |
| **Perception Generalization** | Overfit to 1–2 benchmarks | Multi-sensor LiDAR/stereo/radar industrial stacks | **Closed locally: `Depth-Anything-V2-Small` 3D normal + OOD veto (`-25.68%` facade area), real TEIAS WireSeg/D-FINE v1.1 (`240/240 -> 11/240` full / `0/240` crop, `628 -> 1/0` boxes), quarantined `AeroScapes` (`0.2468`) & `UAVScenes` (`0.2259`); `ICG Drone` registration remains open** |
| **License & Supply-Chain Hygiene** | Mixed GPL/NC/SA dependencies | PX4 (BSD-3), ArduPilot (GPL-3), Autoware (Apache-2.0) | **Strict Apache-2.0 runtime + `check_zero_gpl_symbols.py` (`0` GPL across `20` targets) + `libsodium 1.0.22`** |

---

## 2. Six-Stack Comparative Engineering Matrix

### 2.1 PX4 Autopilot (`v1.15` / `v1.16` — BSD-3-Clause)
- **Reference Capabilities:** Production NuttX/Linux flight stack with dynamic control allocation (`control_allocator`), centralized severity-ranked failsafe state machine, `EKF2` multi-instance voting, and `uXRCE-DDS` (`subscription_multi` in `v1.15+`) bridging directly to ROS 2 Jazzy/Humble.
- **Where VertiRTA Exceeds PX4:**
  - **Formally bounded Simplex RTA & 4D HJ-CBVF:** PX4's failsafe state machine is rule-based (RC/GCS/battery/motor saturation triggers) without backward-reachable-set invariance. `VertiRTA` runs a GPU-solved `101x71x101x71` (`102.85M`-node) 4D `HJI-CBVF` + 2D `CEL` `mmap` lookup table (`artifacts/v4.0.0/g4_jax_4d_hires_and_lc_manifest.json`, `0` soundness violations) with `StaticDaqpSolver` (`daqp_auxiliary_norec.c`, `0` recursion).
  - **Same-Arm Coaxial Flat-Spin Survivability:** Stock control allocation capsizes on same-arm coaxial dual-motor loss (`0.00%` survival in `artifacts/v3.1.0/paired_30k_ral_stats.json`). `VertiRTA`'s Option B1b active flat-spin (`|Omega_z| = 2.2 rad/s`, `+10.6 deg` phase lead) achieves `100.00%` (`1,500/1,500`) in the 10k Tier-1 atlas (and exposes real wind/gust cliffs in the new 1,000-flight dispersion campaign; see Section 4).
  - **Bare-Metal Formal & Dual-FCC Proof:** `ESBMC v8.7.0` (`1,905/1,905`) + Renode dual-FCC OpenCyphal CAN-FD (`12.50 ms` failover) has no direct equivalent in upstream PX4 CI.
- **Where VertiRTA Lags PX4:**
  - PX4 runs on dozens of certified/flight-proven Pixhawk FMUv6X/v6C boards with real ESC telemetry, DShot/DroneCAN drivers, and closed-loop Gazebo lockstep CI. `VertiRTA`'s `px4_overlay/` (`12050_gz_vertirta_x8_240`, `mc_indi_wls_control`, 4-part `GZBridge` patch) is checked in and contract-tested, **but full closed-loop PX4 + Gazebo Harmonic SITL remains `PENDING_LOCAL_PX4_GZ_SITL` (`Sn5`/`Sn6`)** with zero recorded closed-loop Gazebo lockstep CI video.

### 2.2 ArduPilot (`Copter` / `QuadPlane` — GPL-3.0)
- **Reference Capabilities:** Battle-tested `EKF3` multi-lane affinity + automatic lane switching (`EK3_IMU_MASK`), onboard Lua scripting (`APM/scripts`, e.g., `ahrs-source-gps-optflow.lua`) for runtime GNSS-denied source transitions, and mature `QuadPlane` VTOL transition logic.
- **Where VertiRTA Exceeds ArduPilot:**
  - **Commercial-Friendly Apache-2.0 Licensing:** ArduPilot is `GPL-3.0`; `VertiRTA` enforces a zero-GPL runtime firewall (`scripts/ci/check_zero_gpl_symbols.py`) and air-gaps offline OpenFOAM (`tools/cfd_offline_gpl/`).
  - **Dedicated Slow-Ramp GNSS Spoof Guard:** Rather than relying on ad-hoc innovation thresholds, `core/gnss_denied/` implements a 10 Hz M-CUSUM spoof guard with rollback and monotonic `3 sigma` CTR inflation, verified zero-shot on real ETH Zurich `EuRoC MAV V1_01_easy` (`29,120` IMU rows, `7/7` gates PASS, `TTD = 0.1 s`, `0` clean false alarms, `0.137 m` rollback slip in `artifacts/v4.0.0/realworld_euroc_mav_gnss_audit.json`).
  - **W3C XSD Regulatory Forensics:** Real `PX4 ULog` (`sample.ulg`, `68.83 s`) -> `.vrta` -> ED-155A hash-chained sidecar -> W3C XSD-valid ECCAIRS2 `E5X` (`artifacts/v4.0.0/realworld_px4_ulog_forensics_proof.json`), absent in ArduPilot/PX4.
- **Where VertiRTA Lags ArduPilot:**
  - Zero physical flight logs, zero hardware-in-the-loop sensor drivers (baro, mag, airspeed, dual-GPS moving-baseline yaw), and no field-proven VTOL stall/wind-shear transition flight tests on a physical `QuadPlane`/Lift+Cruise airframe.

### 2.3 UZH RPG `Agilicious` (Academic NC) & TU Delft INDI-WLS
- **Reference Capabilities:** `Agilicious` (restricted non-commercial academic license) and TU Delft's MAVLab INDI-WLS stacks represent the state of the art in agile quadrotor/tilt-rotor INDI control, motor-dynamics compensation, and high-speed flight.
- **Where VertiRTA Exceeds Them:**
  - **Unencumbered Apache-2.0 + Freestanding MCU Core:** `Agilicious` requires a restricted academic-only license agreement and depends on ROS/Eigen dynamic allocations on companion PCs. `VertiRTA`'s `libvertirta_flight_core` is 100% Apache-2.0, `0`-heap, `-fno-exceptions -fno-rtti`, `28,886 B` Cortex-M7 Flash.
  - **Dual-Airframe Coupled Battery + Coaxial Physics:** Models `112S12P` (`403.2 V`, `21.77 kWh`) 1-RC Thevenin sag, asymmetric BLDC spool/coast inertia (`tau_up = 85 ms`, `tau_coast = 424 ms`), coaxial `eta_coax = 0.85`, and `Lift+Cruise 8+1` quintic Hermite transition (`max_alt_err = 0.1977 m`, `17x12` WLS `p99 = 0.98 us`).
- **Where VertiRTA Lags Them:**
  - Both UZH RPG and TU Delft have validated their INDI-WLS controllers on real flying hardware at high speeds, wind tunnels, and real actuator-failure drop tests. `VertiRTA`'s INDI-WLS is currently verified only in C++20 6-DOF simulation, Renode SIL, and WASM (`VertiGCS`).

### 2.4 Autoware Universe (Apache-2.0)
- **Reference Capabilities:** Production-grade ROS 2 autonomous driving stack featuring explicit ODD monitoring, Minimal Risk Maneuver (`MRM`) state machines, system error monitors, and typed perception-to-planning contracts.
- **Where VertiRTA Exceeds Autoware:**
  - **Hard Real-Time Bare-Metal Safety Core:** Autoware's safety monitor runs in Linux ROS 2 user space; `VertiRTA` isolates its `400 Hz` Simplex arbiter inside a `28.21 KiB` bare-metal Cortex-M7 library with zero dynamic memory, plus a zero-install `< 39 KiB` WASM + WebGL `VertiGCS` console (`https://fahrifurkansagir.github.io/VertiRTA/`).
  - **Neuro-Symbolic Z3 + Conformal Co-Pilot Boundary:** Autoware lacks a formal SMT (`QF_LRA`/`QF_RDL`) + Mondrian conformal gate for human-in-the-loop bilingual contingency commands.
- **Where VertiRTA Lags Autoware:**
  - Autoware runs full multi-sensor 3D point-cloud perception (LiDAR + camera + radar fusion) on real urban roads with live ODD degradation handling. `VertiRTA` now closes the pure-2D oblique-wall gap on real video via an Apache-2.0 `Depth-Anything-V2-Small` monocular 3D surface-normal + OOD fail-closed veto (`artifacts/v4.0.0/realworld_drone_video_temporal_proof.json`, vetoing `-25.68%` vertical-facade frame area to Hazard and cutting centroid jitter `-88.86%`), yet still lacks multi-sensor hardware LiDAR/radar calibration on a physical airframe.

### 2.5 Basalt VIO (BSD-3) / FAST-LIO2 (GPL-2.0) / EGO-Planner (GPL-3.0)
- **Reference Capabilities:** Gold-standard onboard GNSS-denied odometry (`Basalt` visual-inertial, `FAST-LIO2` tightly coupled LiDAR-inertial `ikd-Tree`) and real-time ESDF-free local trajectory replanning (`EGO-Planner`).
- **Where VertiRTA Exceeds Them:**
  - **License-Clean Aviation Integration:** Both `FAST-LIO2` (`GPL-2.0`) and `EGO-Planner` (`GPL-3.0`) are copyleft-incompatible with proprietary/Apache-2.0 avionics. `VertiRTA`'s robocentric `3.094 MiB` FIESTA-style ESDF + 1D `WireCapsules` + `XFeat` visual relocalization (`36/36` RANSAC, `CEP50 = 0.218 m` on Kartal ortho) is 100% Apache-2.0 and integrates directly into the `100 Hz` CBVF-QP safety filter.
- **Where VertiRTA Lags Them:**
  - `VertiRTA` does **not** implement a full tightly-coupled visual-inertial or LiDAR-inertial SLAM frontend (its `EuRoC MAV` audit validates the M-CUSUM spoof guard, rollback, and `3 sigma` CTR inflation against VIO/IMU streams rather than running a full onboard keyframe VIO solver like `Basalt`).
  - Historical `v1.0.0` `M-LSD-WireSeg` + `D-FINE-N` (`800` procedural frames, `0` real powerline pairs; `artifacts/v4.0.0/g4_forward_wire_metrics.json`) suffered `240/240` roofline false positives (`586` spurious boxes). **Resolved locally (`v1.1` ONNX, `artifacts/v4.0.0/local_gtx1650s_forward_wire_v2_metrics.json`):** retraining with real MIT Mendeley TEIAS powerlines + `646/153` permissive urban rooftop hard negatives + static `grid_xy` per-class Sigmoid Focal BCE + NMS reduced zero-shot 240-frame NASA FPs to **`11/240 (4.58%)`** full-frame (`0/240` center-crop) and **`1`** box (`0` center-crop).

### 2.6 NASA / FAA `ASTM F3269-21` Simplex RTA Reference Architectures
- **Reference Capabilities:** Standard-defining Simplex Run-Time Assurance architecture separating untrusted Complex Functions (`CF`) from verified Recovery Control Functions (`RCF`) via deterministic Boundary Monitors (`BM`) and a priority arbiter.
- **Where VertiRTA Stands Out:**
  - One of the most complete open-source `ASTM F3269-21` implementations publicly available: automated AST/symbol isolation gate (`scripts/ci/check_arch_isolation.py`, `37` files, `0` violations), 3-tier arbiter (`CEL > CSFL > CBVF > CF`) with anti-chatter dwell/hysteresis, `SCSC-141C` GSN v3 living safety case (`docs/SAFETY_CASE.md`, `Sn1–Sn18`) with `G5`/`G6` programmatically locked `[UNDEVELOPED]`, and strict `DAL E` read-only isolation of LLM/SLZ advisories.
- **Where VertiRTA Lags NASA/FAA Industrial RTA Programs:**
  - Formal proofs (`ESBMC v8.7.0`) are bounded model checks on 6 harnesses (not full DO-178C/DO-333 deductive proofs or TQL-5 qualified tools); timing measurements are Linux user-space telemetry (not static analytical WCET like aiT/AbsInt on real STM32H753 silicon); and hardware-in-the-loop verification (`G5`) remains unexecuted.

---

## 3. Honest ML, Physics & Silicon Gaps (Updated Post-Local Recovery)

Status of every negative ablation and operational defect measured in `v1.0.0` artifacts and subsequent local recovery runs:

1. **NASA Drone Video Forward-Wire False Positives (`240/240` -> `11/240` full / `0/240` crop) — RESOLVED LOCALLY:** Retraining `M-LSD-WireSeg v1.1` + `D-FINE-N v1.1` on real MIT TEIAS powerlines + `646/153` permissive urban rooftop hard negatives + per-class Sigmoid Focal BCE/NMS (`artifacts/v4.0.0/local_gtx1650s_forward_wire_v2_metrics.json`) cut zero-shot NASA FPs from `240/240` to **`11/240 (4.58%)`** full-frame (`0/240` center-crop) and boxes from `628 / 586` to **`1` / `0`**.
2. **Qwen3-4B v3 Conformal Set Size (`|C| = 7.803 -> 2.502 / 2.058`, `ECE = 0.1744 -> 0.0321`) — RESOLVED LOCALLY:** Exact FP32 layer-streaming GBNF BPE-Trie re-scoring (`artifacts/v4.0.0/local_gtx1650s_qwen3_v3_conformal_raps.json`, `peak_vram = 837.5 MiB`, zero quantization loss, zero test leakage, plus parity-aliasing bugfix in `ml/copilot/calibrate_conformal.py`) reduced Lang-APS ECE to **`0.0321`** (`96.00%` cov, `|C| = 3.486`, `28.5%` singl.), SNR-APS to **`|C| = 2.347`** (`95.08%`, `40.0%` singl.), Lang-LAC to **`|C| = 2.502`** (`96.08%`, `40.0%` singl.), and SNR-LAC to **`|C| = 2.058`** (`95.58%`, `45.5%` singl.; 100-resplit bootstrap `95.00 ± 0.79%`, `|C| = 1.924 ± 0.075`). Bucket-3 `ADVERSARIAL` branch top-1 remains `43.24%` (`100.0%` `NOMINAL` & `AMBIGUOUS_ASR`, `89.67%` `FAULT`), transparently bounding residual multi-set rates.
3. **Whisper BVI-LoRA Negative Ablation — OPEN (Zero-Shot Retained as Default):** Aggressive 3-epoch LoRA (`r=32, alpha=64`) degraded WER vs. Zero-Shot (`13.14% -> 17.97%` `S0_Clean`, `31.74% -> 35.36%` `S3_Garbled`; `g4_whisper_bvi_lora_eval.json`). Local CPU voice ATC proof (`local_voice_atc_live_proof.json`) was upgraded to zero-shot `faster-whisper-large-v3-turbo` INT8, cutting Turkish FLEURS+BVI WER ~2× (`12.58%–27.81%`) with `100%` Z3 SAT (`24/24`).
4. **X8-240 Same-Arm CEL Dispersion Cliff (`29/100`) — FORENSICALLY ROOT-CAUSED (Physical Boundary):** Forensic rigid-body audit (`tests/integration/test_monte_carlo_dispersion.cpp`) proved **zero runner bugs** and derived the exact directional authority envelope: orthogonal actuated axis `cg_v ∈ [-5.0, +5.0] cm` is **`100%` safe**, while unactuated dead-arm axis `cg_u` has physical spin-up impulse boundary **`[-3.50, +1.00] cm`** at `240 kg` (`[-2.25, +0.25] cm` at `264 kg`), compounded by pre-fault descent timing during delayed faults.
5. **3D Depth Surface-Normal Veto (`|n_z| < 0.85`) on Real Video — RESOLVED LOCALLY:** Apache-2.0 `Depth-Anything-V2-Small` 3D normal + OOD fail-closed veto (`artifacts/v4.0.0/realworld_drone_video_temporal_proof.json`) vetoes `-25.68%` sunlit vertical facade area to Hazard, cutting flicker `-53.26%` and SLZ centroid jitter `-88.86%`.
6. **Paper 2 CoRL External Eval-Only Cells & Synthetic Provenance — PARTIALLY RESOLVED:** `UAVScenes` (`300` frames across all `20` scenes, raw pixels deleted) is now evaluated zero-shot (`mIoU = 0.2259`, `FSR = 12.55%`, `realworld_uavscenes_zeroshot_eval.json`); only manual-registration `ICG Drone` (`2` cells) and human-native `VertiBench-v2` recordings remain open.
7. **Local `GTX 1650 SUPER` Hardware Edge Gate — FULLY RESOLVED (`6 / 6 PASS`):** Shared-BFC single-context ORT CUDA allocator + CPU-overlapped 1 Hz `xfeat_512` (`gtx1650s_m3_timesliced_edge_gate.json`) passes **all 6 hardware gates**: **Peak VRAM `286.0 MiB < 300.0 MiB` PASS** (`274.0 MiB` Cruise / `286.0 MiB` Terminal), **Terminal `p99 = 9.07 ms < 12.0 ms PASS`** (`125.65 FPS`), and **Cruise `p99 = 12.30 ms < 25.0 ms PASS`** (`89.84 FPS`).

---

## 4. Concrete Codebase Bug & Quick-Fix Ledger (File + Line Verified)

Every item below was audited at pre-release commit `20ff6c6` and tracked to resolution before finalizing `v1.0.0`:

| ID | Priority / Status | File & Line | Concrete Defect / Gap | Resolution / Recommended Fix |
| --- | --- | --- | --- | --- |
| **B1** | **P0 — RESOLVED (`v1.0.0`)** | `CMakeLists.txt:2` | `project(VertiRTA VERSION 3.0.0 LANGUAGES C CXX)` exported `VertiRTAConfigVersion.cmake` with `VERSION 3.0.0` + `COMPATIBILITY SameMajorVersion`, breaking `find_package(VertiRTA 1.0 REQUIRED)`. | **Resolved:** Aligned root `CMakeLists.txt` to `VERSION 1.0.0` matching `pyproject.toml`, `sdk/python/vertirta/_version.py`, `apps/vertigcs/package.json`, `CITATION.cff`, and `.zenodo.json`. |
| **B2** | **P0 — RESOLVED (`v1.0.0`)** | `CMakeLists.txt:26-46` & `.github/workflows/m3_perception_voice_forensics.yml` | `benchmarks/monte_carlo_dispersion` (`vertirta_mc1000_cli`) was omitted from `_vertirta_opt_dir` and CI path triggers. | **Resolved:** Wired `benchmarks/monte_carlo_dispersion` into root CMake and M3 CI build/smoke verification. |
| **B3** | **P0 — RESOLVED (`v1.0.0`)** | `.github/workflows/product_and_wasm.yml` & `pyproject.toml:81` | Both CI and `cibuildwheel` ran only `test_vertirta_sdk.py`, skipping `test_vertirta_m3_sdk.py`. | **Resolved:** Both CI and `cibuildwheel` now execute both `test_vertirta_sdk.py` and `test_vertirta_m3_sdk.py`. |
| **B4** | **P1 — RESOLVED (`v1.0.0`)** | Subproject `CMakeLists.txt` & `ros2_ws/src/*/package.xml` | Subprojects previously mixed pre-1.0 version numbers (`2.0.0`, `3.0.0`, `3.5.0`, `3.5.1`, `4.0.0`). | **Resolved:** Aligned subproject CMake `project(... VERSION 1.0.0)` and all `ros2_ws/src/*/package.xml` packages to `1.0.0`. |
| **B5** | **P1 — RESOLVED (`v1.0.0`)** | `.github/workflows/m3_perception_voice_forensics.yml` | Job `m3_vertigcs_build_and_wasm_lock` pinned Node `20` while the other workflows pinned Node `22`. | **Resolved:** Aligned to Node `22` and explicit VertiGCS build/test scope. |
| **B6** | **P1 — RESOLVED (`v1.0.0`)** | `README.md` | Documentation drift on workflow badges, Vitest count (`17/17`), `Sn1–Sn19`, and surfacing NASA drone HUD & MC-1000 dispersion. | **Resolved:** Redesigned `README.md` with `7/7` CI green, `17/17` Vitest, `Sn1–Sn19`, moving NASA Goddard HUD preview, and MC-1000 dispersion visual. |
| **B7** | **P1 — RESOLVED (`v1.0.0`)** | `docs/SAFETY_CASE.md` | Supplemental evidence table needed `Sn19` alignment and removal of pre-1.0 tag references. | **Resolved:** Added `Sn19` and aligned `docs/SAFETY_CASE.md` with `v1.0.0`. |
| **B8** | **P1 — AUDITED & REGRESSION-LOCKED (`v1.0.x`)** | `artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json:484-525`, `tests/integration/test_monte_carlo_dispersion.cpp` | `X8-240` `SAME_ARM_DUAL_KILL_CEL` shows `29/100` safe recoveries under unmodeled `±5 cm` lateral CG shift vs `1,500/1,500` nominal-CG Tier-1 10k atlas. | **Forensically verified (0 runner bugs):** Orthogonal `cg_v ∈ [-5.0, +5.0] cm` is `100%` safe; unactuated `cg_u` obeys rigid-body boundary `[-3.50, +1.00] cm` at `240 kg`, locked via Catch2 regression test. |
| **B9** | **P2 — RESOLVED LOCALLY** | `core/slz_bayes/` & `scripts/realworld/run_real_drone_video_temporal_hud.py` | Real drone video pipeline lacked a 3D depth/stereo surface-normal veto (`|n_z| < 0.85`), permitting sunlit vertical facades to trigger `Safe_Flat_Roof`. | **Resolved:** Wired Apache-2.0 `Depth-Anything-V2-Small` 3D surface-normal + OOD fail-closed veto (`-25.68%` facade area vetoed, flicker `-53.26%`, jitter `-88.86%`, `realworld_drone_video_temporal_proof.json`). |
| **B10** | **P2 — PARTIALLY RESOLVED LOCALLY (3/4 CLOSED)** | `docs/research/OPTIONAL_G4_V1_1_BATCH_SPEC.md` | Four ML/data gaps originally slated for cloud GPU: WireSeg/D-FINE roofline FPs (`240/240`), Qwen3 conformal `|C|=7.80`, Whisper BVI-LoRA negative ablation, and Paper-2 `UAVScenes` + `ICG Drone` zero-shot eval. | **Closed locally without cloud GPU:** (1) TEIAS + hardneg WireSeg/D-FINE v1.1 (`240/240 -> 11/240` full / `0/240` crop, `628 -> 1/0` boxes); (2) exact FP32 local Qwen3 GBNF-tree conformal recovery (`ECE 0.1744 -> 0.0321`, `|C| 7.80 -> 2.50/2.06`, `40–45.5%` singletons); (3) `UAVScenes` zero-shot (`0.2259 / 12.55%`, `300f/20` scenes); plus `faster-whisper-large-v3-turbo` local CPU voice proof. **Remaining open:** optional low-LR Whisper BVI-LoRA (`G4-3`) & manual-registration `ICG Drone` (`G4-4` partial). |

---

## 5. Resolution Summary & Remaining Open Roadmap

- **Resolved in `v1.0.0` + Local Hardening:** **B1–B7** (CMake/SDK/Docker/CI/docs parity), **B8** (forensic rigid-body `cg_u` boundary + Catch2 lock), **B9** (`Depth-Anything-V2-Small` 3D normal + OOD veto), and **B10 (3 of 4 ML/data items)**: real TEIAS WireSeg/D-FINE v1.1, exact FP32 local `Qwen3-4B v3` conformal recovery, quarantined `UAVScenes` zero-shot eval, refreshed `TU116` `< 12 ms` Terminal / `< 25 ms` Cruise latency gates, plus P0/P1 C++20/Python/Docker/VertiGCS hardening (`36/36` CTest, `18/18` Vitest, `11/11` native M1–M3 SDK).
- **Remaining Open (`OPTIONAL_G4_V1_1_BATCH_SPEC.md` & Hardware):** Optional low-LR Whisper BVI-LoRA replay (`G4-3`), manual-registration `ICG Drone` zero-shot eval (`G4-4`), physical `STM32H753` HITL (`PENDING_PHYSICAL_STM32H7_HITL`), and local `PX4 v1.16` + Gazebo Harmonic SITL (`PENDING_LOCAL_PX4_GZ_SITL`).
