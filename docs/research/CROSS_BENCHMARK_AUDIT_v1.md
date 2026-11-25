# VertiRTA `v1.0.0` — Cross-Benchmark Comparative Engineering Audit & Defect Ledger

**Audit Scope:** Read-only principal GNC & systems-engineering audit of `VertiRTA` (`v1.0.0`, commit `20ff6c6`) against six leading open-source UAV/eVTOL and autonomy reference stacks.  
**Assurance Posture:** `NONE_RESEARCH_TESTBED_ONLY` — zero certification, Type Certificate, or DO-178C/DO-254 compliance claims. All metrics below are read directly from checked-in source files and SHA-256-verified JSON artifacts under `artifacts/v1.0.0/`, `artifacts/v3.1.0/`, `artifacts/v3.5.1/`, and `artifacts/v4.0.0/`.

---

## 1. Executive Verdict

`VertiRTA` sits in an unusual engineering niche: it is **far more rigorous than typical academic UAV repositories** on embedded determinism, formal/SIL verification, control-theoretic fault boundaries, and license hygiene, yet it **honestly lags production autopilots (PX4, ArduPilot, Autoware, industrial eVTOL stacks)** on physical flight-hardware validation, closed-loop Gazebo/SITL endurance, and real-world perception generalization.

| Dimension | Typical Academic UAV Repo | Production Stacks (PX4 / ArduPilot / Autoware) | `VertiRTA v1.0.0` Ground Truth |
| --- | --- | --- | --- |
| **Bare-Metal Determinism** | None (ROS/Python nodes, dynamic `std::vector`) | RTOS-hardened (NuttX/ChibiOS), hundreds of KiB–MiB | **`28,886 B` Cortex-M7 Flash / `0 B` lib RAM / `2,960 B` stack DAG / `0` heap / `0` recursion** |
| **Formal & Protocol SIL** | Unit tests only | Extensive SITL + hardware flight-hour corpus | **`ESBMC v8.7.0` `1,905/1,905` properties + Renode `v1.17.0` dual-FCC CAN-FD `12.50 ms` failover** |
| **Cross-ISA Numerical Parity** | Untested | Hardware-specific tuning | **0-ULP same-ISA & `max_abs_diff = 0.0` `x86_64 <-> arm64` (`p99 = 0.736 us` Cobalt 100)** |
| **Physical Hardware & Flight Logs** | Occasional indoor MoCap | Millions of real flight hours across hundreds of boards | **0 physical STM32H7 HITL runs (`PENDING_PHYSICAL_STM32H7_HITL`); 0 real vehicle flight hours** |
| **Perception Generalization** | Overfit to 1–2 benchmarks | Multi-sensor LiDAR/stereo/radar industrial stacks | **Honest gaps: NASA roofline WireSeg FP `240/240`, no 3D normal veto on oblique walls yet** |
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
  - Autoware runs full multi-sensor 3D point-cloud perception (LiDAR + camera + radar fusion) on real urban roads with live ODD degradation handling. `VertiRTA`'s `PIDNet-S v2.1` + `M-LSD-WireSeg` + `D-FINE-N` stack currently runs 2D RGB on real video without a live 3D stereo/LiDAR surface-normal veto (`|n_z| < 0.85`), causing bright sunlit vertical walls on the NASA Goddard drone video to partially trigger `Safe_Flat_Roof`/`Safe_Flat_Ground` (`artifacts/v4.0.0/realworld_drone_video_temporal_proof.json:187`).

### 2.5 Basalt VIO (BSD-3) / FAST-LIO2 (GPL-2.0) / EGO-Planner (GPL-3.0)
- **Reference Capabilities:** Gold-standard onboard GNSS-denied odometry (`Basalt` visual-inertial, `FAST-LIO2` tightly coupled LiDAR-inertial `ikd-Tree`) and real-time ESDF-free local trajectory replanning (`EGO-Planner`).
- **Where VertiRTA Exceeds Them:**
  - **License-Clean Aviation Integration:** Both `FAST-LIO2` (`GPL-2.0`) and `EGO-Planner` (`GPL-3.0`) are copyleft-incompatible with proprietary/Apache-2.0 avionics. `VertiRTA`'s robocentric `3.094 MiB` FIESTA-style ESDF + 1D `WireCapsules` + `XFeat` visual relocalization (`36/36` RANSAC, `CEP50 = 0.218 m` on Kartal ortho) is 100% Apache-2.0 and integrates directly into the `100 Hz` CBVF-QP safety filter.
- **Where VertiRTA Lags Them:**
  - `VertiRTA` does **not** implement a full tightly-coupled visual-inertial or LiDAR-inertial SLAM frontend (its `EuRoC MAV` audit validates the M-CUSUM spoof guard, rollback, and `3 sigma` CTR inflation against VIO/IMU streams rather than running a full onboard keyframe VIO solver like `Basalt`).
  - `M-LSD-WireSeg` + `D-FINE-N` were trained on `800` procedural Apache-2.0 frames (`0` real `TTPLA` pairs ingested due to missing local/HF cache during Wave 1; `artifacts/v4.0.0/g4_forward_wire_metrics.json:56`), causing **`240/240` frames (`100%`) roofline false positives** (`mean 5,982 wire_px`, `586` spurious `D-FINE-N` boxes) on the real NASA Goddard downward-oblique drone video (`realworld_drone_video_temporal_proof.json:171-184`), locking the CBVF speed clamp at `15.0 m/s` on every frame.

### 2.6 NASA / FAA `ASTM F3269-21` Simplex RTA Reference Architectures
- **Reference Capabilities:** Standard-defining Simplex Run-Time Assurance architecture separating untrusted Complex Functions (`CF`) from verified Recovery Control Functions (`RCF`) via deterministic Boundary Monitors (`BM`) and a priority arbiter.
- **Where VertiRTA Stands Out:**
  - One of the most complete open-source `ASTM F3269-21` implementations publicly available: automated AST/symbol isolation gate (`scripts/ci/check_arch_isolation.py`, `37` files, `0` violations), 3-tier arbiter (`CEL > CSFL > CBVF > CF`) with anti-chatter dwell/hysteresis, `SCSC-141C` GSN v3 living safety case (`docs/SAFETY_CASE.md`, `Sn1–Sn18`) with `G5`/`G6` programmatically locked `[UNDEVELOPED]`, and strict `DAL E` read-only isolation of LLM/SLZ advisories.
- **Where VertiRTA Lags NASA/FAA Industrial RTA Programs:**
  - Formal proofs (`ESBMC v8.7.0`) are bounded model checks on 6 harnesses (not full DO-178C/DO-333 deductive proofs or TQL-5 qualified tools); timing measurements are Linux user-space telemetry (not static analytical WCET like aiT/AbsInt on real STM32H753 silicon); and hardware-in-the-loop verification (`G5`) remains unexecuted.

---

## 3. Honest ML, Physics & Silicon Gaps (Zero Sugarcoating)

Every negative ablation and operational defect measured in `v1.0.0` artifacts:

1. **NASA Drone Video Forward-Wire False Positives (`240/240` frames):** `M-LSD-WireSeg` + `D-FINE-N` (`0` real `TTPLA` frames trained) fired on building rooflines and facade seams in `240/240` frames (`realworld_drone_video_temporal_proof.json:173-184`), clamping `v_max = 15.0 m/s` continuously (`0/240` frames at `22.0 m/s`).
2. **Qwen3-4B v3 Conformal Set Size Explosion (`|C| = 7.803`, `singleton = 0.0%`):** In `artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json:1120-1126`, although Mondrian coverage hit `96.0%` (Lang) and `95.25%` (SNR), mean prediction set size blew up to **`7.803`** (Lang) / **`7.601`** (SNR) with **`0.0%` singleton execution** (`100%` multi-set clarification!), making the conformal gate operationally unusable without temperature-scaled RAPS (`v3.5.1` previously achieved `|C| = 3.125`, `22.86%` singleton). Furthermore, `ADVERSARIAL` Bucket-3 raw AST accuracy is only `44.93%–46.28%` (`59.8%` gated safe reject).
3. **Whisper BVI-LoRA Negative Ablation:** Aggressive 3-epoch LoRA (`r=32, alpha=64`) **degraded WER across every SNR stratum** vs. Zero-Shot (`whisper-large-v3-turbo`: `13.14% -> 17.97%` on `S0_Clean`, `31.74% -> 35.36%` on `S3_Garbled`; `g4_whisper_bvi_lora_eval.json:133-220`) while doubling inference latency (`14.88 ms -> 28.58 ms`).
4. **X8-240 Same-Arm CEL Dispersion Cliff (`29/100` under `3.5 m/s` gate):** In `artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json:484-525`, while `Lift+Cruise 8+1` achieved **`500/500` (`100%`)** safe recoveries (`CEP95 = 1.97 m`), `X8-240` under `SAME_ARM_DUAL_KILL_CEL` achieved only **`29/100`** safe recoveries (`CEP95 = 216.26 m`, `vz_p50 = 20.28 m/s`, `max_tilt = 179.53 deg`) under the dispersion runner's `3.5 m/s` touchdown / `60 deg` tilt gate — a stark contrast to the Tier-1 10k atlas (`1,500/1,500`) that demands root-cause reconciliation in `v1.0.1` (likely initial-altitude / wind-shear / Option B1b engagement divergence between `mc_1000_runner.cpp` and `vertirta_6dof_batch_cli`).
5. **No 3D Depth/LiDAR Surface-Normal Veto (`|n_z| < 0.85`) on Real Video:** `PIDNet-S v2.1` local recovery achieved `Val mIoU = 0.4669`, `FSR = 2.81%`, `OOD AUROC = 0.7554`, `AeroScapes FSR = 3.43%`, but on oblique real drone RGB without metric depth fusion, sunlit vertical concrete walls still leak into `Safe_Flat_Roof`.
6. **Paper 2 CoRL Open Eval-Only Cells & Synthetic Provenance:** `docs/paper_corl/corl_metrics_manifest.json:179-200` still holds 4 `PENDING_EVAL_ONLY_REAL_IMAGERY` cells (`icg_drone_{miou,fsr}`, `uavscenes_{miou,fsr}`), and `VertiBench-v2` remains `PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2` (`human_native_audit_completed: false`).
7. **Local `GTX 1650 SUPER` Soft-Gate Bounds:** Cruise VRAM delta `366.0 MiB > 300 MiB` and terminal `PIDNet-S v2.1` WDDM `p99 = 13.36 ms > 12.0 ms` (`artifacts/v4.0.0/gtx1650s_m3_timesliced_edge_gate.json`).

---

## 4. Concrete Codebase Bug & Quick-Fix Ledger (File + Line Verified)

Every item below was verified directly in the repository at commit `20ff6c6`:

| ID | Priority | File & Line | Concrete Defect / Gap | Recommended Fix |
| --- | --- | --- | --- | --- |
| **B1** | **P0** | `CMakeLists.txt:2` | `project(VertiRTA VERSION 3.0.0 LANGUAGES C CXX)` exports `VertiRTAConfigVersion.cmake` with `VERSION 3.0.0` + `COMPATIBILITY SameMajorVersion` (`line 154–158`). Any external consumer calling `find_package(VertiRTA 1.0 REQUIRED)` for `v1.0.0` **fails with a major-version mismatch**! | Bump `project(VertiRTA VERSION 1.0.0 LANGUAGES C CXX)` to match `pyproject.toml:7`, `sdk/python/vertirta/_version.py:3`, `apps/vertigcs/package.json:3`, `CITATION.cff:5`, `.zenodo.json:4`. |
| **B2** | **P0** | `CMakeLists.txt:26-46` & `.github/workflows/m3_perception_voice_forensics.yml:6-86` | `benchmarks/monte_carlo_dispersion` (`vertirta_mc1000_cli`) is **omitted from `_vertirta_opt_dir`** in root `CMakeLists.txt` AND **missing from `push.paths`/`pull_request.paths`** in `m3_perception_voice_forensics.yml`. The 1,000-flight Monte Carlo suite is completely unbuilt and untested in CI. | Add `benchmarks/monte_carlo_dispersion` to `CMakeLists.txt:42` and `benchmarks/monte_carlo_dispersion/**` to `m3_perception_voice_forensics.yml` paths + add a smoke build/run step. |
| **B3** | **P0** | `.github/workflows/product_and_wasm.yml:145` & `pyproject.toml:80` | Both CI and `cibuildwheel` run ONLY `pytest tests/sdk/test_vertirta_sdk.py -v`, **completely skipping `tests/sdk/test_vertirta_m3_sdk.py`**! M3 nanobind bindings (`VertiForwardHazard`, `VertiGnssDenied`, `VertiMondrianVoiceGate`, `VertiForensicsEngine`) and M3 CLI commands (`forensics-export`, `gnss-spoof-sim`, `m3-demo`) have zero CI test execution. | Change both to `pytest tests/sdk/test_vertirta_sdk.py tests/sdk/test_vertirta_m3_sdk.py -v` (or `tests/sdk/ -v`). |
| **B4** | **P1** | `sdk/python/CMakeLists.txt:2` (and `core/forward_hazard/CMakeLists.txt:4`, `core/gnss_denied/CMakeLists.txt:4`, `ros2_ws/src/*/package.xml`) | Stale mixed versions across subprojects: `vertirta_python_sdk` is `VERSION 3.0.0`, standalone M3 CMake targets are `VERSION 4.0.0`, and `ros2_ws/src/*/package.xml` mix `1.0.0`, `2.0.0`, `3.5.0`, `4.0.0`. | Align `sdk/python/CMakeLists.txt:2`, standalone CMake targets, and `ros2_ws/src/*/package.xml` to `1.0.0` for clean `v1.0.0` release hygiene. |
| **B5** | **P1** | `.github/workflows/m3_perception_voice_forensics.yml:361-384` | Job `m3_vertigcs_build_and_wasm_lock` advertises `+ WASM Size Lock` in its name (`line 362`), yet **never compiles WASM or checks `.wasm` size**, and pins `node-version: "20"` (`line 371`) while the other 3 workflows (`m2_uspace_and_physics.yml:335`, `product_and_wasm.yml:73`, `deploy_pages.yml:66`) pin `"22"`. | Either add the `emsdk 4.0.4` + `<= 153600 B` check or rename job to `M3 VertiGCS Vitest & Production/Pages Build`; align `node-version: "22"`. |
| **B6** | **P1** | `README.md:6-11, 166, 210, 224, 253, 315, 318` | Multiple documentation drifts: (a) missing CI badge for `m3_perception_voice_forensics.yml` (6 of 7 badged); (b) line 166 still says `6/6 GREEN AT v3.5.0 bf8adb1`; (c) Vitest counts listed as `10/10` (`lines 224, 253`) and `12/12` (`lines 210, 315`), whereas `apps/vertigcs/tests/*.test.ts` now has **17** tests across 5 files; (d) line 318 says `Sn1–Sn14` instead of `Sn1–Sn18`; (e) `monte_carlo_1000_dispersion_report.json` and `docs/assets/vertirta_real_drone_hud_preview.gif` are not yet surfaced in `README.md`. | Add 7th workflow badge, update Scope Matrix to `7/7`, unify Vitest count (`17/17`), update `Sn1–Sn18`, and add the real-drone HUD preview + 1,000-flight dispersion summary (including the honest X8 CEL cliff). |
| **B7** | **P1** | `docs/SAFETY_CASE.md:45` | Mermaid GSN header still stamps `commit=bf8adb1 (v3.5.0)` rather than `v1.0.0`. | Re-run `verification/gsn/build_gsn.py` at the `v1.0.0` release commit (or update header stamp in sync with `Sn15–Sn18`). |
| **B8** | **P1** | `artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json:484-525` | `X8-240` `SAME_ARM_DUAL_KILL_CEL` shows `29/100` safe recoveries (`vz_p50 = 20.28 m/s`, `max_tilt = 179.53 deg`) vs `1,500/1,500` in Tier-1 10k atlas (`README.md:149`). | Audit `benchmarks/monte_carlo_dispersion/src/mc_1000_runner.cpp` vs `vertirta_6dof_batch_cli` to verify whether this is a real high-wind/low-altitude Option B1b physics cliff or a runner initialization/latch bug; document explicitly in `README.md`. |
| **B9** | **P2** | `core/slz_bayes/` & `scripts/realworld/run_real_drone_video_temporal_hud.py` | Real drone video pipeline lacks a 3D depth/stereo surface-normal veto (`|n_z| < 0.85`), permitting sunlit vertical facades to trigger `Safe_Flat_Roof`. | Wire monocular depth (`DAv2-Small`) or stereo normal-map veto (`|n_z| >= 0.85`) before SLZ scoring in `v1.1`. |
| **B10** | **P2** | `docs/research/OPTIONAL_G4_V1_1_BATCH_SPEC.md` (ML gaps) | Four ML/data gaps require GPU compute: WireSeg roofline FPs (`240/240`), Qwen3 Mondrian `|C|=7.80 / singleton=0%`, Whisper BVI-LoRA negative ablation, and 2 Paper-2 `PENDING` eval-only benchmarks (`ICG Drone`, `UAVScenes`). | Execute single consolidated Optional `v1.1` G4 batch per `docs/research/OPTIONAL_G4_V1_1_BATCH_SPEC.md` when G4 capacity opens. |

---

## 5. Summary Recommendation Before Tagging `v1.0.0`

- **Fix immediately (`0.00 CU`, < 30 min local PR):** **B1** (`CMakeLists.txt` `VERSION 1.0.0`), **B2** (wire `benchmarks/monte_carlo_dispersion` into CMake + M3 CI), **B3** (run `test_vertirta_m3_sdk.py` in SDK CI), and **B4–B7** (manifest/CI/README/SAFETY_CASE alignment) plus **B8** root-cause check on the X8 CEL dispersion delta.
- **Defer cleanly to Optional `v1.1`:** All 4 GPU/ML upgrades (**B9–B10**) are isolated in [`OPTIONAL_G4_V1_1_BATCH_SPEC.md`](OPTIONAL_G4_V1_1_BATCH_SPEC.md) and do **not** block an honest, rock-solid `v1.0.0` release.
