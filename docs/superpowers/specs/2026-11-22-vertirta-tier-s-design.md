# VertiRTA — Tier-S Architecture & Engineering Design Specification

- **Status:** **v1.2 FROZEN — All 7 Deep Research Tracks (`DR-1`..`DR-7`) Ingested, Forensically Audited (31 Critical Errata Patched) & Reconciled**
- **Date:** 2026-11-22
- **Canonical Project Name:** `VertiRTA` (*Vertical Run-Time Assurance*; C++ core library `libvertirta`, namespace `vertirta::`, ROS 2 packages `vertirta_*`)
- **License:** Apache 2.0 (strictly zero CC-BY-NC, zero ShareAlike-trained weights, zero defense-excluded licenses)
- **Target Public GitHub Repo:** `VertiRTA` (staged locally in `c:\PROJECTS\AEGIS-VTOL` during Phase 0; published as clean `VertiRTA` repo at Phase 1)
- **Companion Audit Lock:** `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (mandatory errata overrides for all `DR-01..07.md` files)

---

## 1. Intent & Why

### Core Mission
Build an industry- and research-lab-caliber (**Tier-S**), 100% open-source **Run-Time Assurance (RTA)**, **Fault-Tolerant Flight Control (FTC)**, **Formal Verification**, and **Closed-Loop Bilingual Neuro-Symbolic Contingency** architecture for a full-scale **8-rotor coaxial (`X8`) urban eVTOL** operating along the real georeferenced **Istanbul Anatolian Coast corridor (Maltepe – Dragos – Kartal Coast, bounded directly on the east/north by real Sabiha Gökçen `LTFJ` and Samandıra `LTBX` CTR airspace walls)**.

### Strategic & Career Context
- **Author Profile:** New-grad Electrical-Electronics Engineer (Istanbul / Kartal) targeting top R&D engineering roles at **Cezeri Yapay Zeka Robotik Teknolojileri** (#1 target: 8-rotor coaxial eVTOL, triple-redundant flight safety, modern C++20/ROS 2 robotics, Cezeri AI Hub Turkish LLM/CV), **Baykar**, **Teknopark Istanbul defense/UAV** (DASAL, Altınay, STM, HAVELSAN, TUSAŞ), **TÜBİTAK BİLGEM / AI Institute**, **Ford Otosan**, **Siemens Turkey**, **Milvus Robotics**, and international eVTOL/defense-tech startups.
- **Portfolio Complementarity (Zero Overlap):**
  1. *University Graduation Project:* Jetson Orin Nano UGV (YOLOv8 + ROS 2 + LiDAR SLAM + serial motor control) → proves physical embedded hardware & debugging.
  2. *OpenXVS (separate closed-core repo):* Fixed-wing aviation, 5 GPU-trained CV models + Rust `no_std`/Kani + KSFO 3DGS → proves heavy fixed-wing CV & formal methods mindset.
  3. **`VertiRTA` (this 100% public open-source repo):** Closes the #1 missing hiring artifact — **auditable, zero-dependency, production-grade Modern C++20/C++17 + ROS 2 Jazzy + PX4 v1.16 rotary-wing/coaxial eVTOL flight-safety core (`libvertirta`)** — elevated with real Colab G4 96 GB reachability/RLVR/active-vision breakthroughs and public research artifacts (`VertiBench-TR/EN` on Hugging Face).
- **Real-World Engineering Impact:** Existing open-source PX4/ROS 2 projects (`ntnu-arl/composite_cbf`, `gtfactslab`, `cbfkit`, stock PX4 `control_allocator`) do not combine a zero-dependency embedded-ready RTA core (`-fno-exceptions -fno-rtti`, 0-heap proof, bit-exact flight-recorder replay) with **in-loop 400 Hz 3-IMU TMR FDIR**, **same-arm dual-rotor active-spin relaxed-yaw FTC**, **4D fault-degraded Hamilton-Jacobi CBVF safety filtering**, and **CI-linked GSN assurance**. `libvertirta` directly fills this open-source aerospace tooling gap.

---

## 2. Approaches Considered & Selected Architecture

Four execution/ambition approaches were evaluated against real host hardware (**AMD Ryzen 5 5600X 6C/12T, 31.9 GB RAM, GTX 1650 SUPER 4 GB VRAM, WSL2 Ubuntu, no local Jetson Orin Nano**) and cloud compute (**Google Colab G4 `RTX PRO 6000 Blackwell` 96 GB VRAM + free public GitHub Actions `ubuntu-24.04-arm` Arm Neoverse N2**):

1. **Rejected — 8-Pillar Unpruned Draft ("Frankenstein Breadth"):** Stacking PX4 inner loop + offboard ROS 2 INDI + NMPC + CBF-QP + `isaac_ros_nvblox` + monolithic 4-dataset CV (`ICG+UAVid+Mid-Air+TTPLA`) + online 3D/4D `STL-MILP` + `airnoiseUAM` + multi-eVTOL U-Space ATC + conflated `SC-VTOL + SORA 2.5` compliance claims. *Why rejected:* Control authority wars, WSL2/CUDA crash (`nvblox`), NP-hard MILP timeouts, CC-BY-NC/SA license taint, rural-to-urban domain collapse (`Mid-Air`), and fatal regulatory contradictions.
2. **Rejected — Conservative Safe Capstone (`Approach A`):** Stock PX4 `CA_FAILURE_MODE=1` toggle, hand-tuned kinematic HOCBF without reachability proof, generic box world, 200-run table, one-shot SFT JSON parser, and indefinite deferral of learned visual SLZ. *Why rejected:* Safe and buildable, but reads as "Top-1% Student Capstone++" rather than an industry/research-lab flagship.
3. **Rejected — 100% Colab-First Interactive Dev:** Running daily C++20/PX4/Gazebo editing and debugging inside ephemeral Colab containers (Ubuntu 22.04, no Docker, ~6.7 CU/hr burn, idle resets). *Why rejected:* Kills edit-compile-debug velocity and burns GPU quota on trivial builds.
4. **SELECTED — Tier-S `VertiRTA` (3 Deep Pillars + 3-Tier Hybrid Compute + Build-in-Public Release Train):**
   - Keeps every anti-bloat kill dead while pushing **3 tightly coupled aerospace pillars** to ETH RPG / TU Delft MAVLab / Stanford ASL / Joby / Cezeri Principal Engineer depth.
   - Splits compute strictly by physical strengths:
     - **Local WSL2 Ubuntu 24.04 (Ryzen 5600X / 32 GB RAM):** Instant C++/PX4/ROS 2 dev, 10k C++20 + 1k headless PX4/Gazebo physics, native Windows Foxglove (`:8891`) + CesiumJS (`:5180`), and optional full-stack PX4+ROS 2+ORT smoke verification (`<300 MB` VRAM on GTX 1650).
     - **Google Colab G4 (`RTX PRO 6000 Blackwell`, 96 GB VRAM):** Batch JAX 4D HJ-Reachability, `Qwen3-8B/4B` Z3-GRPO RLVR, Apache-2.0 `DINOv2` distillation, sensor-only Gazebo Harmonic EGL 8,000-frame data gen + standalone 6-DOF C++20 Active Vision 3-way closed-loop ablation (zero fragile Ubuntu 22.04 PX4/ROS 2 porting inside Colab!), and headless **Blender 4.2+ Cycles OptiX** 4K `.vrta` real-Istanbul cinema.
     - **Free Public GitHub Actions (`ubuntu-24.04-arm` Arm Neoverse N2 + `ubuntu-24.04` x86_64):** Every-commit 4-tier ARM64 + x86_64 determinism CI, 0-heap proof, static stack bound, bare-metal MCU footprint check, and `.vrta` replay regression.
   - Ships `v1.0.0` in Weeks 1–3 (immediately job-application-ready) and `v2.0.0` in Weeks 4–6 (second viral wave + `VertiBench-TR/EN` + Zenodo DOI / arXiv whitepaper).

---

## 3. Unified Mathematical Chain & 3 Tier-S Pillars

All subsystems belong to **one coherent, whiteboard-derivable closed-loop chain**:

```text
Coaxial X8 (240 kg, T/W = 2.5) B(eta_coax) + 1-RC Thevenin Voltage Sag
  -> 400 Hz PX4-Internal [C++17 libvertirta_flight_core]:
     In-Loop 3-IMU TMR FDIR + Reduced-Attitude (n_B) + INDI Rate + Prioritized WLS-AMS
     (Nominal/1M/2M-Cross: Full 4-DOF CSFL | Same-Arm Dual: Active Diff-Yaw Spin-Stabilized CEL Descent)
  -> Colab G4 Offline JAX 4D Fault-Degraded CBVF V_f(r, v_r, z, v_z) (41x29x41x29 int16, ~2.83 MB/regime)
  -> 100 Hz Zero-Heap [C++20 libvertirta_rta_companion]:
     Supervisory TMR State + mmap 4D CBVF LUT (<2 us) + DAQP HJI-CBVF-QP + .vrta Bit-Exact Logger
  -> 20 Hz CPU ROG-Map ESDF (64x64x32 m @ 0.5 m, ~2 MB) + Analytic WireCapsules + ADS-B KF Intruders
     + Georeferenced Maltepe-Dragos-Kartal Reachable SLZ (bounded East by LTFJ CTR & North by LTBX CTR)
     [v2.0: + DINOv2-Distilled Active Uncertainty-Driven Inspection Orbit via ISlzPriorProvider]
  -> 0.5-1 Hz Async EASA AI L1/2A Contingency Co-Pilot (Strictly Outside ASTM F3269 RTA)
     [v2.0: Qwen3-8B/4B Z3-GRPO RLVR + llama.cpp GBNF + Z3 LTFJ/LTBX UNSAT Repair + C++20 STL rho + KnowNo Conformal]
```

### Pillar 1 — `libvertirta` Two-Target Reusable Library + 400 Hz PX4 Core + JAX 4D CBVF (`v1.0.0`)
- **Clean Two-Target CMake Split (`FetchContent`, Zero ROS/Gazebo/CUDA Deps):**
  1. **`libvertirta_flight_core` (Portable C++17 Subset — Linked Directly into PX4 v1.16 & MCU-Gated):** Contains `common`, `tmr_fdi`, `coax_envelope`, `indi_wls`, `daqp_qp`, and `vrec` (`.vrta` logger). Compiled with `-Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti`, verified in CI with `operator new` **0-byte heap tripwire**, `-fstack-usage` static stack gate, LibFuzzer NaN/Inf/Byzantine fuzzing, and **`arm-none-eabi-g++` bare-metal Cortex-M7/STM32H7 footprint gate (`< 256 KB` Flash / `< 128 KB` RAM)** — zero upstream PX4 C++20 toolchain hacking required!
  2. **`libvertirta_rta_companion` (Full Modern C++20 — Companion ARM64/x86_64 & ROS 2 Jazzy):** Links `libvertirta_flight_core` + `rog_esdf` (~2 MB L3-friendly static ring buffer) + `mmap`-backed 4D JAX CBVF LUTs + `slz_reach` + `stl_monitor` (`v2.0`).
- **400 Hz PX4-Internal Control & In-Loop TMR Ownership:**
  - `libvertirta_flight_core`'s **3-IMU MVS + Chi-Square TMR Voter** runs **in-the-loop at 400 Hz inside PX4** across 3 independent physical Gazebo IMUs (`EKF2_MULTI_IMU=3`, `SYS_FAILURE_EN=1`), yielding 8 ticks inside `< 20 ms` for real gust-immune Chi-Square persistence; voted health (`tmr_state`, `chi2_residual`, `active_imu_mask`) streams at 100 Hz to ROS 2 for supervisory `Fail-Safe Divert` and `.vrta` logging.
  - PX4-internal **`mc_indi_wls_control`** owns **reduced-attitude ($n_B$) + 400 Hz discrete INDI rate + Prioritized WLS Active-Set Allocation** (replacing stock `mc_att_control` + `mc_rate_control` + `control_allocator` in RTA mode), receiving **thrust + desired acceleration/thrust-direction setpoints** from `vertirta_rta_node` via `px4-ros2-interface-lib` (bypassing stock `mc_pos_control` static tilt clamps and full-yaw fighting).
- **Physically Valid Same-Arm Dual-Loss Survival (`T/W = 2.5` + Active Flat Spin `Option B1b` + Voltage-Headroom Winding):**
  - Raising nominal $T/W = 2.5$ ($T_{\max,\text{nom}} = 5{,}886\text{ N}$ split under $\eta_{\text{coax}}=0.85$ as `795.4 N` upper / `676.1 N` lower; upper loss unblanks lower `+17.6%`) with BLDC voltage-headroom winding **$K_{e,\text{bus}} \le 1.36\text{ V}\cdot\text{s/rad}$** (FOC `240 rad/s` cap) and a `320 Wh/kg` (`68 kg`) 22 kWh HV pack guarantees surviving 2-arm $T/W = 1.25$ down to `358.4 V` and $\ge 1.12$ down to `338 V`.
  - Per `DR-1`, non-spinning 2-arm (`Option A`) is rank-2/4 uncontrollable (capsizes) and 3-arm tilted (`Option B2`, 48–70 RPM) fails on `23–32°` lag/rollover. Locked solution: **Option B1b Active Flat Spin (`|\Omega_z| = 2.2 rad/s` = 21 RPM, CCW Arm 0/1, CW Arm 2/3)** with phase lead $\Delta\psi = \operatorname{atan}(\Omega_z \tau_m) = \mp 10.6^\circ$, QR active-set or `double` (`f64`) Cholesky WLS (NEVER `float32` normal equations!), `FD_FAIL_R=0, FD_FAIL_P=0`, and level touchdown (`|v_z| <= 2.5 m/s` >10m, `<= 1.5` <5m, `<= 0.8` contact).
  - **Strict Mode Separation:** `f ∈ {Nominal, 1M, 2M-Cross, Low-Vbat}` execute full 4-DOF hover/diversion **CSFL** guarded by the 100 Hz 4D **JAX HJI-CBVF-QP** (`0` barrier violations inside BRT, with degraded cruise speed cap `v_cap,f` for `2M-Cross` under `1.75 m/s²` gust!). `SAME_ARM_DUAL_LOSS` irreversibly latches **Last-Resort Controlled Emergency Descent (CEL)** bypassing 4-DOF CBVF-QP into inner-loop Option B1b flat-spin descent backed by a `2.32 KiB` sag-indexed 2D vertical braking LUT `V_2D(z, v_z)`.
- **Offline JAX 4D Fault-Degraded CBVF (`StanfordASL/hj_reachability` on Colab G4 96 GB):**
  - Solves exact 4D relative-state HJI-VI $V_f(r, v_r, z, v_z)$ at fine **`81 × 57 × 81 × 57`** (`6.82 GB` FP32 `vmap` across 4 regimes on 96 GB Blackwell) matching `_support_U` & `partial_max_magnitudes` for valid Lax-Friedrichs CFL, then stride-2 decimates & `int16`-quantizes (`q = 0.992 mm`) into **`41 × 29 × 41 × 29` (`~2.83 MB/regime`, `11.31 MB` total across 4 regimes)** via `mmap`+`mlock` in `libvertirta_rta_companion`, storing exact fine-grid margin $\epsilon_{\text{grid}}$ in a 4 KB `.vrta_lut` header (`< 2 µs` lookup, `isfinite` + `regime_idx < 4` guarded).
- **Exteroceptive Sensors (`v1.0.0`) & Dual-Res `3.094 MiB` CPU `ROG-Map`/FIESTA ESDF:**
  - `VertiRTA-X8-240` carries a 16-beam 3D `gpu_lidar` (`60 m` range, `lidar_link` defined; `GZBridge` patched with 3× `PX4Gyroscope/Accelerometer` distinct `DeviceId`s, `sleep_count_limit=5000`, and `_failure_injection_sub` in `GZMixingInterfaceESC`) populating pure-C++20 **Dual-Resolution `3.094 MiB` `ROG-Map`/FIESTA** (`64 × 64 × 32 m`: `0.5 m` Occupancy `1 MiB` + `1.0 m` `131k` 3×18b-DLL FIESTA ESDF `2 MiB` + `96 KiB` queues, `< 2 ms` CPU, fixing upstream `ROG-Map #15`).
  - Analytic `WireCapsules` and ADS-B/Remote-ID KF Dynamic Intruder Ellipsoids (explicitly handling overhead `Δp_z` sign & `smax - ln(2)/β` conservatism!) enter `K <= 16` simultaneous QP hazard rows with `1.5 m` opposing-gradient inflation and staleness margin inflation (`age > 250 ms`).

### Pillar 2 — Authentic Maltepe–Dragos–Kartal (`LTFJ`/`LTBX`-Bounded) Assurance, 2-Tier Verification Pyramid & Photorealistic Earth Replay (`v1.0.0`)
- **Geodetically Consistent Istanbul Airspace (Zero CTR Contradiction!):**
  - Real published **AIP Turkey `AD 2 LTFJ` CTR** (`40°53'54"N, 029°18'33"E` = `40.89833°N, 29.30917°E`, `5 NM = 9.26 km` radius + **NW extension polygon+arc reaching `J2 = 40.91279°N, 29.19212°E` which pulls the boundary 603 m west of `29.200°E`!**, SFC–2,500 ft AMSL = `798.58 m HAE`) and **Samandıra `LTBX` CTR** (SFC–2,000 ft AMSL = `646.35 m HAE`, ARP verified against DHMİ `AD 2.17` PDF before freezing) are enforced as exact **polygon+arc HARD Z3-UNSAT & CBVF NO-ENTRY WALLS** (*simulation reference only — NOT for navigation*).
  - Because `LTFJ`'s 5 NM circle physically engulfs Pendik (`~4.0 NM`) and Kurtköy, our **flyable coastal UAM corridor and all valid Z3-SAT SLZ candidates lie strictly west of the `LTFJ` polygon+arc (`Maltepe Sahil – Dragos – Kartal Coast – Cevizli`, canonical ENU datum at Kartal Coastal Vertiport `40.8860° N, 29.1880° E, 12.0 m AMSL`, ~1.04 km west of `LTFJ` CTR; `ODD.yaml` strictly `< 29.190° E` near `J2`)**. Pendik Marina, Kurtköy, and Sabiha Gökçen sit directly across the eastern CTR wall as real adversarial/prohibited test targets!
- **2-Tier Verification Pyramid (Honest Compute Budget):**
  - *Tier 1 (10,000 runs in `< 90 s` locally/CI):* Deterministic C++20 6-DOF coaxial+Thevenin+Dryden plant running `libvertirta` → dense 4D **Fault Survival Envelope Atlas** + A/B *Hand-Tuned HOCBF vs JAX HJ-CBVF*.
  - *Tier 2 (1,000 runs in ~43–52 min on Ryzen 5600X):* 4 parallel **monotonic hot-resettable** headless PX4 v1.16 + Gazebo Harmonic workers (`gz service` `reset: {model_only: true}` keeping sim-clock monotonic for `UXRCE_DDS_SYNCT=1` & avoiding `px4-ros2-interface-lib` #201/#191!) executing full PX4 EKF2 + `mc_indi_wls_control` + Gazebo dynamics against analytic OSM/DEM/Wire/Intruder hazards, **plus a 50-run Full-Sensor Subset** (2 EGL `--headless-rendering` `gpu_lidar` workers, `10–13 min`) validating end-to-end `ROG-Map` ESDF latency & zero regression.
- **Honest Assurance & Determinism:** `SAFETY_CASE.md` (explicit non-compliance disclaimer; **`SC-VTOL-02` + `MOC-5 Iss. 2 FINAL` (5 Nov 2026)** Cat. Enhanced CSFL + `ASTM F3269-21` 3-Tier Arbiter primary drivers; `SORA 2.5` + **`ED Dec 2026/010/R` (16 Nov 2026)** separated as 25 kg unmanned subscale precursor only; **SHGM `SHT-İHA` (eff. 30 Jul 2026)** M3/İHATTYS real-world hook), MLEAP `ODD.yaml`, `ARP4761A` (ED-135) mini-FHA + corrected FTA cut-sets (2oo3 Triplex = $3\lambda^2 T$, NOT $\lambda^3 T^2$!), **CI-linked SCSC-141C GSN v3 Living Safety Case** (keeping CSFL `n=650` & CEL `n=150` Atlas denominators strictly separate!), and **4-Tier Cloud-ARM64 Determinism Profile** on pinned `ubuntu-24.04-arm` (flagging `steal > 1%` as `NOISY_NEIGHBOR_TAINTED`; **0-ULP SHA-256 bit-identity on Same-ISA** w/ golden hash + **`< 1e-12` tolerance across `x86_64 ↔ ARM64`** with `-fno-fast-math -ffp-contract=off`).
- **3-Layer Photorealistic Earth Visualization (Zero Toy Three.js, Zero UE5 Swamp, 100% ToS-Clean):**
  - *Physics Engine:* Headless Gazebo Harmonic (`gz sim -s -r`, local Kartal topocentric CRS `+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84` — NEVER UTM 35N which causes `115–265 m` drift!).
  - *Live Ops Console (Local Windows 60 FPS, `<1.8 GB` VRAM on GTX 1650 SUPER):* **Foxglove Studio (`ws://localhost:8891`)** + **CesiumJS (`http://localhost:5180`)** with swappable basemap (Esri key/Azure/ion; zero offline tile scraping), Cesium OSM 3D Buildings, `sampleHeightMostDetailed` Gaussian touchdown blend, curvature-compensated `FlatEnuToCesium` (`-(E^2+N^2)/(2 R_gauss)` + `+36.7 m` EGM2008), PBR glTF coaxial X8, translucent `LTFJ`/`LTBX` CTR walls, and live CBVF funnel.
  - *Flagship 4K Photorealistic Cinema (Colab G4 96 GB Blackwell):* **Headless Blender 5.2 LTS Cycles GPU (`OptiX`, driver `>= 575`; fallback 4.5 LTS — Blender 4.2/4.3 lack Blackwell `sm_120`!)** scripted via `bpy` using **100% ToS-clean raw AWS Copernicus `sentinel-cogs` L2A (`35TPF`) + Copernicus GLO-30 (`N40_00_E029_00`, `-dstnodata -9999`) + OSM/Overture footprints (banning `CC BY-NC-SA` EOX/FABDEM!) + procedural PBR shaders (using RGBA socket indices `[6]/[7]/[2]`)** + 3-regime rotor VFX & `hevc_nvenc` HUD — driven **100% deterministically from `.vrta` logs**!

### Pillar 3 — Closed-Loop TR/EN Contingency Co-Pilot (`Z3-GRPO`), `VertiBench-TR/EN` & Active Visual SLZ (`v2.0.0`)
- **Closed-Loop TR/EN Co-Pilot (`EASA AI Level 1/2A`, strictly outside RTA):**
  - Fine-tune **strictly the 2 Apache-2.0 flagship models (`Qwen3-8B` GPU & `Qwen3-4B` local CPU `Q4_K_M` GGUF)** via **SFT → Logic-Verified GRPO (RLVR)** on Colab G4 96 GB (`400 V / 22 kWh` telemetry contract; token-level `dr_grpo` `0.20/0.28` or calibrated GSPO `3e-4/4e-4`; persistent `spawn` Z3 worker pool; exact case-matched `r_z3` codes). Evaluate `Qwen2.5-7B-Instruct` and `Gemma-3-4B-IT` strictly as **zero-shot + GBNF+Z3 baselines**.
  - **Compact `<= 30`-token Symbolic DSL + Eager `libllama` GBNF + Deterministic TR/EN Renderer:** Isolates `Qwen3` `<think>` via empty prefill (`151667,271,151668,271`, `enable_thinking=False`), guarantees `<3 s` Ryzen 5600X CPU latency (avoiding 120-token single-digit BPE truncation!), eliminates numeric hallucination, and enables prefix conformal scoring.
  - **`Z3` C++ API (`QF_LRA`, `<15 ms`, unique `c_dyn_nfz_<idx>` trackers, checking ownship Z + destination pad Z + `cmd_agl` + `floor_amsl_m`)** verifies static straight-line corridor + `LTFJ`/`LTBX` + energy/parameter consistency with UNSAT-core repair; pure-C++20 **Quantitative STL Robustness $\rho(\phi, x, t)$** (`FixedMonotonicWedge` evicting `head_` BEFORE pushing `tail_`) monitors trajectories live; **`KnowNo` Split Conformal (`n_cal=200, α=0.05, q_lvl=0.955`)** drives binary Turkish clarification.
  - **Public Artifact:** Release **`VertiBench-TR/EN`** (1,800 train / 200 human cal / 560 human test, `<0.15` 5-gram Jaccard CI gate, Apache 2.0) on Hugging Face.
- **Active Uncertainty-Driven 3D SLZ Perception (`ISlzPriorProvider`, 100% Unassailable Apache-2.0 Weights):**
  - **Clean 5-Source Zero-ShareAlike Train Set (Verified Primary Licenses):** Drop tainted `RescueNet` (`CC BY-NC-ND`), `LoveDA` (`CC BY-NC-SA`), `FlyAwareV2` (`GPL-3.0`), `SegFormer` (NVIDIA NC), and `OpenEarthMap` default regions (`CC BY-NC-SA`). Train **`PIDNet-S` (`MIT`, ~189 MB FP16 VRAM, static `AvgPool2d` PAPPM)** exclusively on **`FLAIR-1` (`etalab-2.0`) + `OpenEarthMap` programmatic `41/100` permissive whitelist + `FloodNet-Supervised v1.0` (`CDLA-Permissive-1.0`) + `SkyScenes` (`MIT`) + 8K Author Colab-EGL Gazebo-Istanbul**, distilled from frozen Apache-2.0 **`DINOv2-ViT-L/14` + `Depth Anything V2 Small`**. Keep `AeroScapes`, `ICG`, and `UAVScenes` strictly **EVAL-ONLY**.
  - **Fixed Math & Active Orbit:** Single-Pass Free-Energy OOD with explicit OOD variance floor / `Λ_k` reset; cell-local `δp = p_W - c_ij` covariance (zero float32 cancellation); 2-phase `APPROACH_ORBIT` → `ACTIVE_INSPECT_ORBIT` (12 s timer starts only at `z=22 m`, tilted camera avoiding 80° HFOV clipping, complete `step_max_m`/`clearance_r_m` reject bitmask); Mod B monocular veto `>30 m AGL`; Gazebo `rgbd_camera` raw `depth_image` back-projection (`gz-sensors#545`).
  - **Clean Gate 5 Execution Split:** Sensor-only Colab-G4 Gazebo Harmonic EGL for 8k data gen + standalone 6-DOF C++20 3-way Active Vision closed-loop ablation (`N >= 60`), plus full PX4 + ROS 2 + ORT FP16 smoke verification on local WSL2 + GTX 1650 SUPER (`<300 MB` VRAM).
- **Culmination (`v2.0.0`):** Authentic 6–8 page LaTeX Technical Report published with **Zenodo DOI** (and arXiv `cs.RO` submission) backed 100% by frozen real `v1.0.0` + `v2.0.0` metrics.

---

## 4. Locked Cross-Pillar Engineering Contracts (Single Source of Truth)

| Contract | Locked Value / Rule |
|---|---|
| **Vehicle Baseline (`config/vehicle_x8_240.yaml`)** | **`VertiRTA-X8-240`** — Full-scale single-passenger coaxial X8 eVTOL: $m = 240.0\text{ kg}$ ($W = 2{,}354.4\text{ N}$), $J = \mathrm{diag}(85, 95, 140)\text{ kg}\cdot\text{m}^2$, 4 arms at $\pm 45^\circ, \pm 135^\circ$, $L = 1.35\text{ m}$, $R = 0.75\text{ m}$, $\Delta z_{\text{coax}} = 0.35\text{ m}$, $\eta_{\text{coax}} \in [0.82, 0.88]$, $\tau_m = 0.085\text{ s}$, 400 V / 22 kWh HV Li-ion 1-RC Thevenin pack, **nominal $T/W = 2.5$ ($T_{\max,\text{nom}} = 5{,}886\text{ N}$, ~736 N/rotor → 2 surviving orthogonal arms hold $1.25 \times mg$ nominal / $\ge 1.12 \times mg$ under 10% sag!)**, $v_{\max} = 22\text{ m/s}$, SLZ footprint $R_{\text{SLZ}} \ge 5.0\text{ m}$, $\theta \le 8^\circ$, $\sigma_z \le 0.15\text{ m}$. *(25 kg subscale referenced only as a Stage-A SORA 2.5 flight-test note.)* |
| **Coordinate Frames, Airspace & Geoid (`config/istanbul_ltfj_airspace.yaml`)** | PX4 internal = `FRD`/`NED` (Hamilton scalar-first quat); ROS 2 / `libvertirta` / Gazebo / `ROG-Map` = `FLU`/`ENU`; Earth Viz / `Z3` Airspace = `WGS84` ↔ `ECEF`. Compile-time strong types (`Vec3Ned`, `Vec3Enu`, `Vec3Frd`, `Vec3Flu`). Canonical ENU Datum Origin (Kartal Coastal Vertiport): **`40.8860° N, 29.1880° E, 12.0 m AMSL`** (~1.0 km west of `LTFJ` 5 NM CTR edge). Flyable corridor & valid SLZs strictly west of `29.200° E` (**Maltepe – Dragos – Kartal Coast – Cevizli**); `LTFJ` & `LTBX` CTRs = hard Z3-UNSAT / CBVF exclusion walls. Istanbul `EGM2008` geoid undulation **$N_{\text{EGM2008}} = +36.7\text{ m}$** ($h_{\text{WGS84}} = H_{\text{AMSL}} + N$). |
| **Library, LUT & Determinism Contracts** | `libvertirta_flight_core` = portable C++17 (`<256 KB` Flash / `<128 KB` RAM bare-metal `arm-none-eabi` gate, 0 heap, `-fstack-usage`, linked into PX4); `libvertirta_rta_companion` = full C++20 (`mmap` 4D CBVF LUT `41×29×41×29` `int16` = `~2.83 MB/regime`, `~11.3 MB` total for 4 full-DOF regimes, `<2 µs` lookup + `ROG-Map` ~2 MB). `.vrta` replay = **0-ULP bit-identical on Same-ISA** and **`< 1e-12` across `x86_64 ↔ ARM64`** (`-fno-fast-math -ffp-contract=off`). |
| **Loop Frequencies & Isolation** | `400 Hz` PX4 internal (`libvertirta_flight_core`: in-loop 3-IMU TMR + reduced-attitude $n_B$ + INDI + WLS-AMS) → `100 Hz` `libvertirta_rta_companion` (`DAQP` HJI-CBVF-QP + `.vrta` logger, bypassing `mc_pos_control` via thrust/attitude setpoints) → `20 Hz` CPU `ROG-Map` ESDF (`64×64×32 m @ 0.5 m`) & Active SLZ (lock-free SPSC, `age > 250 ms` margin inflation) → `0.5–1 Hz` async TR/EN Co-Pilot (`ModeExecutorBase`, `HOLD_LOITER` fallback). |
| **Port Isolation Lock** | Foxglove Bridge WS = **`8891`** (never `8765`); CesiumJS = **`5180`**; `MicroXRCEAgent` UDP = **`8888`**. Forbidden: `4173, 4174, 8765, 9222, 9223`. |
| **2-Page Cold-Derivation Lock (`docs/MATH_DERIVATIONS.md`)** | 5 whiteboard pillars: (1) Coaxial $B(\eta_{\text{coax}})$, 1-RC Thevenin sag, discrete INDI + WLS + active-spin relaxed-yaw; (2) 3-IMU MVS + Chi-Square TMR; (3) 4D HJI-CBVF VI + `DAQP` linear inequality; (4) Clopper-Pearson 95% CI vs ARP4761A FTA cut-sets; (5) Free-Energy OOD, Bayesian SLZ log-odds, STL $\rho$, Split Conformal $\hat{q}_{1-\alpha}$. |

---

## 5. Non-Goals & Anti-Goals (Permanent Kills)

- **Hard Regulatory Honesty:** **NOT** DO-178C, Part 21, SC-VTOL, or SORA 2.5 certified; NEVER claim $10^{-9}/\text{FH}$ from Monte Carlo; NEVER conflate SORA 2.5 and SC-VTOL on the same manned operation; NEVER claim same-arm dual-loss spin descent satisfies passenger-grade CSFL (explicitly framed as last-resort CEL survivability).
- **Hard Control & Mapping Kills:** NO offboard ROS 2 INDI, NO `NMPC` stacked onto CBF-QP, NO `isaac_ros_nvblox`, NO voxelized 5–20 mm wires, NO monolithic 6D+ dense HJ grids, NO `airnoiseUAM`/fake $1/r^2$ dBA, NO fake urban CFD, NO multi-eVTOL U-Space ATC slot queuing.
- **Hard AI & License Kills:** NO online 3D/4D `STL-MILP` or `Z3` multi-step trajectory unrolling, NO MC-Dropout or Dense EDL, NO online POMDP tree search or end-to-end RL flight control, NO urban crane claims, NO training on non-commercial, ShareAlike, or defense-excluded assets (`DINOv3`, `Depth Anything V2 ViT-B/L`, `Aya-Expanse-8B`, `Mid-Air`, `UAVid` NC, `AeroScapes` training — eval-only only!).
- **Hard Visualization & Infra Kills:** NO toy Three.js boxes, NO Unreal Engine 5 / Isaac Sim swamps, NO Google 3D Tiles assumptions over Istanbul, NO offline Bing/Esri tile scraping in Blender, NO daily PX4/ROS 2 development inside ephemeral Colab containers.

---

## 6. Release Train & Strict Pass/Fail Gates

| Milestone | Window | Unfakeable Pass Criteria |
|---|---|---|
| **Phase 0.5: Deep Research Ingestion** | Pre-Code | Wave-A DR reports (`DR-1..DR-5`) saved under `docs/research/`; `ODD.yaml`, `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`, and `ARCHITECTURE_ICD.md` frozen. |
| **Gate 0: Host & Real Istanbul World** | Days 1–2 | WSL2 Ubuntu 24.04 + PX4 v1.16 + headless Gazebo Harmonic (Copernicus GLO-30 DEM + OSM for Maltepe–Dragos–Kartal + `LTFJ`/`LTBX` CTR walls) runs `>= 2.0× RTF`; Foxglove `:8891` + CesiumJS `:5180` verified with 0 m geoid-mismatch on rooftop touchdown. |
| **Gate 1: `libvertirta` Core + JAX 4D CBVF** | Days 3–7 | Two-target `FetchContent` core green on `ubuntu-24.04` & `ubuntu-24.04-arm` CI: `-Werror`, `clang-tidy`, ASan/UBSan/TSan, LibFuzzer, **0 heap allocs** (`operator new` trap), `-fstack-usage`, `arm-none-eabi` footprint (`<256 KB` Flash / `<128 KB` RAM on `libvertirta_flight_core`), `.vrta` **0-ULP same-ISA & `<1e-12` cross-ISA** replay, Colab G4 4D JAX CBVF LUT (`41×29×41×29` `int16`, `~11.3 MB` total, `<2 µs`) integrated with `DAQP`. |
| **Gate 2: PX4 400 Hz INDI+WLS + TMR Closed Loop** | Days 8–14 | `VertiRTA-X8-240` (`T/W=2.5`) SDF verified first on stock PX4 `CA_FAILURE_MODE=1` (Days 8–9), then upgraded with PX4-internal `mc_indi_wls_control` + in-loop 400 Hz TMR (Days 10–14): live 3-IMU `failure` isolated `<20 ms`; 1-motor & cross-arm 2-motor pass full CSFL with 0 CBVF-QP violations inside BRT; **same-arm dual-loss active-spin relaxed-yaw descent** lands safely (`|v_z| <= 2.5 m/s`) while stock PX4 flips; 1-RC Thevenin sag contracts $T_{\max}$ & SLZ cone. |
| **Gate 3: `v1.0.0` PUBLIC RELEASE & JOB APPS** | Days 15–21 | **Tier-1 10,000 C++20** + **Tier-2 1,000 hot-reset PX4/Gazebo** (analytic OSM/DEM/Wire/Intruder) + **50-run Full-LiDAR/ROG-Map Subset** Stratified Survival Envelope Atlas (Clopper-Pearson 95% CI + honest failure-cliff map + A/B HOCBF vs HJ-CBVF & Stock PX4 vs INDI+WLS); `SAFETY_CASE.md`, ARP4761A FHA/FTA, CI-linked GSN; committed `.mcap`/`.vrta`; **Colab G4 Blender Cycles OptiX 4K ToS-clean `.vrta` flight-test cinema + split-screen 60s demo**; 45-min cold-derivation whiteboard drill passed. **→ LAUNCH APPLICATIONS IMMEDIATELY.** |
| **Gate 4: `v2.0.0-alpha` TR/EN Co-Pilot (`DR-6`)** | Days 22–28 | `Qwen3-8B` (Colab G4) + `Qwen3-4B Q4_K_M` (local CPU AVX2) SFT + **Z3-GRPO (RLVR)** (+ zero-shot `Qwen2.5-7B`/`Gemma-3-4B` eval baselines) + `llama.cpp` GBNF + `Z3` static/corridor `LTFJ`/`LTBX` UNSAT repair + C++20 STL $\rho$ + `KnowNo` Conformal on 600–800 audited `VertiBench-TR/EN` cases: **100% valid DSL**, **100% unsafe rejection**, `>=95%` conformal coverage, `0 µs` jitter on 100 Hz RTA; **`VertiBench-TR/EN` published on Hugging Face (Apache 2.0)**. |
| **Gate 5: `v2.0.0` Active Visual SLZ + Whitepaper (`DR-7`)** | Days 29–35 | Zero-ShareAlike Apache-2.0 `DINOv2` + `Depth Anything V2 Small` distilled `PIDNet-S`/`DDRNet-23-slim` + Free-Energy OOD → 2.5D Bayesian SLZ + ~150-LOC Active Inspection Orbit plugged into `ISlzPriorProvider`; dual Real (`AeroScapes`/`ICG` eval-only) + Colab-G4 3-way standalone 6-DOF + Gazebo EGL closed-loop ablation (`N >= 60`) + local WSL2 PX4+ROS 2+ORT smoke check; **6–8p LaTeX Technical Report published with Zenodo DOI / arXiv**. |

---

## 7. Left to the Implementation Planner (Post-Approval)

Exact CMake target graphs, unit-test fixtures, PX4 module CMakeLists patches, JAX training scripts, Blender `bpy` shader node setups, and ROS 2 launch configs will be detailed in `superpowers:writing-plans` immediately after this Design Document is approved and Wave-A `DR-1..DR-5` outputs are saved to `docs/research/`.
