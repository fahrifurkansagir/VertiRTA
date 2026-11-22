# VertiRTA (Tier-S) — Master Gemini App Deep Research Prompt Pack (`DR-1` .. `DR-7`)

> **Builder-Checked v1.1 (Zero Cross-Track Contradictions)**
>
> **Usage Protocol:** Copy each prompt inside its fenced block verbatim into **Gemini App Deep Research** (one clean Deep Research session per track). Save the resulting markdown reports to:
> - `docs/research/DR-01-coaxial-physics-indi-wls-hw-transfer.md`
> - `docs/research/DR-02-jax-4d-cbvf-libvertirta-daqp-esdf.md`
> - `docs/research/DR-03-easa-scvtol-astm-f3269-gsn-arm64-ci.md`
> - `docs/research/DR-04-real-world-impact-ltfj-incidents-gap.md`
> - `docs/research/DR-05-photorealistic-istanbul-earth-viz.md`
> - `docs/research/DR-06-tr-en-z3-grpo-copilot-vertibench.md`
> - `docs/research/DR-07-active-visual-slz-dinov2-colab-egl.md`
>
> **Execution Order:**
> - **Wave A (Mandatory before `v1.0.0` implementation):** `DR-1`, `DR-2`, `DR-3`, `DR-4`, `DR-5` (all 5 can be queued concurrently in Gemini App).
> - **Wave B (For `v2.0.0` AI & Active Vision — can also be queued early in parallel):** `DR-6`, `DR-7`.
>
> **Global Locked Cross-Track Baseline (100% Synchronized Across ALL 7 Prompts):**
> - **Vehicle:** `VertiRTA-X8-240` — 240 kg full-scale single-passenger coaxial X8 eVTOL ($W = 2{,}354.4\text{ N}$, $J = \mathrm{diag}(85, 95, 140)\text{ kg}\cdot\text{m}^2$, $L = 1.35\text{ m}$, $R = 0.75\text{ m}$, $\Delta z_{\text{coax}} = 0.35\text{ m}$, $\eta_{\text{coax}} \in [0.82, 0.88]$, $\tau_m = 0.085\text{ s}$, 400 V / 22 kWh 1-RC Thevenin HV Li-ion, **nominal $T/W = 2.5$ / $T_{\max,\text{nom}} = 5{,}886\text{ N}$** so 2 surviving orthogonal arms retain $1.25\times mg$ nominal & $\ge 1.12\times mg$ under 10% sag, $v_{\max} = 22\text{ m/s}$, SLZ footprint $R_{\text{SLZ}} \ge 5.0\text{ m}$, $\theta \le 8^\circ$, $\sigma_z \le 0.15\text{ m}$).
> - **Geodetic Datum & Airspace:** Canonical ENU datum at **Kartal Coastal Vertiport (`40.8860° N, 29.1880° E, 12.0 m AMSL`)**, ~1.0 km west of the real 5 NM `LTFJ` CTR boundary (`~29.200° E`). Flyable corridor & valid Z3-SAT SLZs lie **strictly west of `29.200° E` (`Maltepe – Dragos – Kartal Coast – Cevizli`)**. Real published **`AIP Turkey AD 2 LTFJ` CTR** (engulfing Pendik & Kurtköy) + **`LTBX` Samandıra CTR** = **hard Z3-UNSAT / CBVF exclusion walls** (*simulation reference only*). Istanbul `EGM2008` geoid undulation **$N = +36.7\text{ m}$** ($h_{\text{WGS84}} = H_{\text{AMSL}} + N$).
> - **Library & Compute Split:** `libvertirta_flight_core` (portable C++17, MCU `<256 KB` Flash / `<128 KB` RAM, linked into PX4 v1.16 at 400 Hz) + `libvertirta_rta_companion` (full C++20, `mmap` `41×29×41×29` `int16` 4D CBVF LUT `~11.3 MB` total, ROS 2 Jazzy at 100 Hz). Local WSL2 Ubuntu 24.04 (`Ryzen 5 5600X`, `32 GB RAM`) + Google Colab G4 (`RTX PRO 6000 Blackwell`, `96 GB VRAM`) + GitHub Actions (`ubuntu-24.04-arm` Neoverse N2 & `ubuntu-24.04`).
> - **Port Locks:** Foxglove `:8891`, CesiumJS `:5180`, `MicroXRCEAgent` `:8888`.

---

## `DR-1` — Coaxial X8 Flight Physics, 400 Hz PX4 Core (In-Loop 3-IMU TMR + Reduced-Attitude INDI + WLS + Active-Spin Relaxed-Yaw), 1-RC Thevenin Sag & Hardware Transfer

```markdown
# Deep Research Request (DR-1): Coaxial 8-Rotor eVTOL Flight Physics, 400 Hz PX4-Internal Core (In-Loop 3-IMU TMR, Reduced-Attitude INDI+WLS, Active-Spin Same-Arm Dual-Loss), 1-RC Thevenin Battery Sag & Sim-to-Real Hardware Transfer

## Role & Research Standard
Act as a Principal Flight Dynamics, GNC, and Avionics Engineer (TU Delft MAVLab / ETH RPG / Joby / Cezeri caliber). Prioritize peer-reviewed 2020–2026 literature, official PX4 v1.16 (`release/1.16`) & Gazebo Harmonic LTS source code/documentation, and first-principles aerospace derivations. Every equation must use consistent SI units and explicit coordinate frames. Where open-source or hardware limitations exist, state exact file paths, workarounds, and physical assumptions transparently — zero marketing fluff or hand-waving.

## 1. Locked System Context
We are engineering **`VertiRTA`**, a 100% Apache-2.0 open-source Runtime Assurance (RTA) and Fault-Tolerant Flight Control architecture for **`VertiRTA-X8-240`**, a full-scale **240 kg single-passenger 8-rotor coaxial (`X8`) urban eVTOL**:
- $m = 240.0\text{ kg}$ ($W = 2{,}354.4\text{ N}$), $J = \mathrm{diag}(85, 95, 140)\text{ kg}\cdot\text{m}^2$, 4 arms at $\pm 45^\circ, \pm 135^\circ$, arm radius $L = 1.35\text{ m}$, rotor radius $R = 0.75\text{ m}$, coaxial vertical spacing $\Delta z_{\text{coax}} = 0.35\text{ m}$, 400 V / 22 kWh HV Li-ion pack, **nominal $T/W = 2.5$ ($T_{\max,\text{nom}} = 5{,}886\text{ N}$, ~736 N/rotor)** so 2 surviving orthogonal arms retain $1.25 \times mg$ nominal ($\ge 1.12 \times mg$ under 10% voltage sag), $v_{\max} = 22\text{ m/s}$.
- Operating corridor: **Maltepe – Dragos – Kartal Coast**, canonical ENU datum origin at Kartal Coastal Vertiport **`40.8860° N, 29.1880° E, 12.0 m AMSL`** (~1.0 km west of real 5 NM `LTFJ` Sabiha Gökçen CTR), Istanbul `EGM2008` geoid undulation $N = +36.7\text{ m}$.
- Software split: The flight-critical kernel **`libvertirta_flight_core`** is a zero-dependency, **portable C++17** (`-fno-exceptions -fno-rtti`, 0-heap) static library linked **directly into PX4 Autopilot v1.16 (`400 Hz`)** (no upstream PX4 C++20 toolchain hacking!) and testable standalone against a fast 6-DOF plant and bare-metal `arm-none-eabi-g++` (`Pixhawk 6X / FMU-v6X STM32H7`).

## 2. Numbered Research Questions

1. **Coordinate Frames & Complete Physical Parameter Audit (`VertiRTA-X8-240`):**
   - Provide exact transformation matrices, quaternion conventions (Hamilton scalar-first `[w,x,y,z]` in PX4 `px4_msgs` vs scalar-last `[x,y,z,w]` in ROS 2), and angular-rate/torque sign mappings across **`PX4 FRD/NED` ↔ `ROS 2 FLU/ENU` ↔ `WGS84 Geodetic / ECEF`**, including `AMSL` vs `HAE` altitude conversion using Istanbul $N_{\text{EGM2008}} = +36.7\text{ m}$.
   - Audit and complete our 240 kg, $T/W = 2.5$ coaxial X8 SI parameter table (rotor thrust/torque constants $k_T, k_Q$, hover/max RPM, tip Mach number check, disk loading, realistic rotor/BLDC inertia and first-order lag $\tau_m \approx 0.085\text{ s}$, 400 V pack $R_0, R_1, C_1$). Flag any physical inconsistency and provide corrected values if needed.

2. **Coaxial X8 Aerodynamics & Actuator Dynamics:**
   - Derive the exact control effectiveness matrix $B \in \mathbb{R}^{4 \times 8}$ mapping squared rotor speeds $\omega_i^2$ (or thrusts $T_i$) to body moments and collective thrust $[\tau_x, \tau_y, \tau_z, T]^T$ in PX4 FRD frame.
   - Formulate coaxial upper-to-lower rotor aerodynamic interference based on **Leishman / momentum theory** (lower-rotor thrust efficiency $\eta_{\text{coax}}(\omega_{\text{upper}}) \in [0.82, 0.88]$, swirl recovery, hover vs forward-flight wake skew, vortex-ring-state descent bounds), plus a lightweight C++ `gz-sim` Gazebo Harmonic plugin modeling coaxial interference + **first-order BLDC/ESC lag $\tau_m$ + slew-rate saturation** without CFD overhead.

3. **400 Hz PX4-Internal Reduced-Attitude + INDI + Prioritized WLS Control Allocation (Incl. Active-Spin Same-Arm Dual-Loss):**
   - Architect **`mc_indi_wls_control`** inside PX4 v1.16 linking `libvertirta_flight_core` (C++17) so it owns **reduced-attitude ($n_B$) + 400 Hz discrete INDI rate + Prioritized WLS Active-Set Control Allocation**, bypassing stock `mc_pos_control` static tilt clamps and full-yaw quaternion tracking when outer-loop RTA (`DR-2`) commands thrust + acceleration/thrust-direction setpoints.
   - Derive the exact discrete-time **400 Hz INDI** rate law (TU Delft MAVLab Smeur/Sun/de Visser) with **synchronized 2nd-order Butterworth LPF** ($\omega_n, \zeta$) applied identically to measured gyro rates $\Omega$ (for $\dot{\Omega}_f$) and estimated/measured actuator states $\hat{\omega}_f$ (accounting for $\tau_m = 0.085\text{ s}$).
   - Detail the exact **Prioritized WLS Active-Set Control Allocation** (`wls_alloc`, Härkegård/Smeur) with priority weights $W_v = \mathrm{diag}(w_\phi, w_\theta, w_\psi, w_T)$, box constraints, and exact **Attainable Moment Set (AMS)** contraction across 4 fault regimes: (0) Nominal 8-rotor, (1) Single-motor loss, (2) Cross-arm dual-motor loss, and (3) **Same-arm dual coaxial-pair loss** (both rotors on Arm $k$ dead).
   - **CRITICAL COAXIAL SPIN-UP PHYSICS:** On a coaxial X8, losing both rotors on Arm $k$ removes **1 CW + 1 CCW rotor**, leaving balanced CW/CCW pairs on surviving arms (unlike a quadcopter losing 1 rotor!). Prove geometrically why static non-spinning balance forces the diagonally opposite arm to zero thrust (relying on the 2 orthogonal surviving arms carrying $1.25 \times mg$ at $T/W=2.5$), AND analyze whether **active differential CW/CCW yaw-torque spin-up** (Sun et al. / Mueller & D'Andrea Primary-Axis Reduced-Attitude control) vs **non-spinning 2-arm symmetric hover/descent** (with opposite arm zeroed and remaining 4 rotors carrying $1.25 \times mg$) is physically superior for a 240 kg ($J_{zz} = 140\text{ kg}\cdot\text{m}^2$) coaxial X8! Give exact equilibrium equations, required spin rate $\bar{\Omega}_z$ (if spinning) or pitch-authority limits (if zeroing opposite arm), and flare/touchdown sink-rate limits (`|v_z| <= 1.5–2.5 m/s`).
   - Specify an online **Motor FDI residual observer** fusing `esc_status` RPM telemetry and INDI torque residuals isolating 1–2 motor losses in `< 100 ms` without false triggers during `12–15 m/s` gusts.
   - Define the exact **Cross-Layer Capability Envelope Struct** (`fault_bitmask`, `tmr_state`, `T_max`, `a_xy_max`, `tilt_max`, `flight_regime`, `V_term`, `SoC`, `P_hover_deg`) published from PX4 uORB over uXRCE-DDS to ROS 2 (`DR-2`).

4. **Coupled 1-RC Thevenin Battery Voltage-Sag & Fault-Degraded Induced-Power Model:**
   - Derive the discrete-time **1-RC Thevenin ECM** ($V_{\text{term}} = V_{\text{oc}}(\text{SoC}) - I R_0 - V_{RC}$) coupled to propulsion power ($P = P_{\text{induced}} + P_{\text{profile}} + P_{\text{parasitic}} + P_{\text{avionics}}$), quantifying how 1–2 failed rotors raise surviving disk loading ($P_i \propto T^{3/2}/\sqrt{A_{\text{eff}}}$), spike current $I$, cause voltage sag, dynamically contract $T_{\max}(V_{\text{term}})$ (verifying surviving 2-arm $T/W \ge 1.12$ under 10% sag!), and shrink the **3D Energy-Reachable SLZ Cone**.

5. **PX4 v1.16 Multi-IMU SITL (`GZBridge`), `v1.0.0` Exteroceptive Sensor SDF & In-Loop 400 Hz TMR FDIR Voter:**
   - Audit upstream **PX4 v1.16** (`src/modules/simulation/gz_bridge/GZBridge.cpp`, `ROMFS/px4fmu_common/init.d-posix/airframes/`, `Tools/simulation/gz/models/`): verify whether stock `GZBridge.cpp` supports 3 simultaneous Gazebo Harmonic IMU sensors or requires a targeted patch; provide exact SDF for `VertiRTA-X8-240` (8 coaxial motors, 3 physical IMUs, dual baro, GNSS, and lightweight 16-beam 3D LiDAR for `v1.0.0` `ROG-Map`), airframe params (`CA_ROTOR_COUNT=8`, `EKF2_MULTI_IMU=3`, `SENS_IMU_MODE=0`, `SYS_FAILURE_EN=1`), and MAVLink `failure` CLI commands.
   - Formulate the **in-loop 400 Hz C++17 TMR FDIR Voter** (`libvertirta_flight_core`) consuming the 3 raw IMU channels inside PX4 (giving 8 ticks inside `< 20 ms` for gust-immune Chi-Square persistence!) via Mid-Value Select + Chi-Square Innovation / Pairwise Disparity Parity Equations + persistent fault latching (`3-of-3 TMR -> 2-of-2 Duplex -> Fail-Safe Divert`), streaming voted health upward at 100 Hz to ROS 2.

6. **Real-World Sim-to-Real Hardware Transfer Reality:**
   - Critically analyze exact physical & avionics gaps between PX4 SITL/Gazebo Harmonic and real coaxial eVTOL hardware: (a) real coaxial wake/ground-effect/forward-flight aerodynamics vs momentum derating, (b) real 400 V HV BLDC/ESC thermal & current derating + telemetry jitter vs 1st-order lag, (c) real MEMS IMU structural vibration harmonics (blade-pass frequency $N_b \cdot \text{RPM}/60$, dynamic notch filtering, Allan variance bias instability & ARW) vs Gazebo white-noise IMUs, and (d) exact engineering roadmap & bench-test gates to port `libvertirta_flight_core` + `mc_indi_wls_control` onto **Pixhawk 6X (FMU-v6X STM32H7, NuttX RT)** + **ARM64 companion over Ethernet/UART uXRCE-DDS**.

## 3. Required Concrete Outputs
- Complete coordinate-transform + geoid table and audited `VertiRTA-X8-240` ($T/W = 2.5$) SI parameter table.
- Full equations for $B \in \mathbb{R}^{4\times 8}$, $\eta_{\text{coax}}$, discrete INDI + Butterworth LPF, WLS active-set C++17 pseudocode, rigorous **Same-Arm Dual-Loss coaxial trade analysis (active differential-yaw spin vs 2-arm symmetric descent)**, 1-RC Thevenin sag + $T_{\max}(V_{\text{term}})$, and 400 Hz in-loop TMR MVS/Chi-Square voter.
- Exact PX4 v1.16 module/patch blueprint (`mc_indi_wls_control` owning reduced-attitude+rate+allocator, `GZBridge.cpp` multi-IMU audit, `octo_coax` SDF + 16-beam LiDAR, `dds_topics.yaml` capability struct).
- Structured **Sim-to-Real Gap & Hardware Transfer Matrix**.

## 4. Strict Out-of-Scope Boundaries
Do **NOT** cover: Hamilton-Jacobi Reachability / CBVF PDEs, `DAQP`/CBF-QP solvers, `ROG-Map` ESDF, `.vrta` replay, or Monte Carlo harnesses (`DR-2`); EASA SC-VTOL / ARP4761A / GSN certification documents (`DR-3`); UAM industry incident / operational analysis (`DR-4`); photorealistic Cesium/Blender Earth visualization (`DR-5`); Turkish/English LLM, Z3, or STL mission parsing (`DR-6`); or computer-vision SLZ perception training (`DR-7`).
```

---

## `DR-2` — Offline JAX 4D Fault-Degraded HJ-Reachability (CBVF), Two-Target `libvertirta` (`DAQP`), CPU `ROG-Map` ESDF, Bit-Exact `.vrta` Replay & 2-Tier Verification Pyramid

```markdown
# Deep Research Request (DR-2): Offline JAX 4D Fault-Degraded Hamilton-Jacobi Reachability (CBVF), Two-Target Zero-Heap `libvertirta` (`DAQP`), CPU `ROG-Map` ESDF, Bit-Exact `.vrta` Replay & 2-Tier Verification Pyramid

## Role & Research Standard
Act as a Principal Safety-Critical Autonomy, Formal Reachability, and Real-Time C++20/C++17 Robotics Systems Architect (Stanford ASL / UC Berkeley Hybrid Systems Lab / MIT REALM / HKU MaRS Lab / Skydio caliber). Ground every mathematical claim, API signature, and benchmark in verified 2021–2026 peer-reviewed papers and upstream open-source repositories (`StanfordASL/hj_reachability`, `DAQP`, `PIQP`, `OSQP`, `ROG-Map`, `Auterion/px4-ros2-interface-lib`). Provide exact equations, mathematically exact byte budgets, compiler flags, and verifiable C++/JAX engineering blueprints — zero hand-waving.

## 1. Locked System Context
We are architecting **`libvertirta`** (split cleanly into two zero-dependency CMake targets) and its **ROS 2 Jazzy + PX4 v1.16** outer-loop safety executive for **`VertiRTA-X8-240`** (240 kg coaxial X8 eVTOL, $T/W = 2.5$, $v_{\max} = 22\text{ m/s}$, SLZ footprint $R_{\text{SLZ}} \ge 5.0\text{ m}$) flying along the **Maltepe – Dragos – Kartal Coast** corridor bounded by real **`LTFJ` & `LTBX` CTR** exclusion walls.
- **Two-Target Library Split:**
  1. `libvertirta_flight_core` (portable C++17, `-fno-exceptions -fno-rtti`, 0 heap, bare-metal `arm-none-eabi-g++` `< 256 KB` Flash / `< 128 KB` RAM): `common`, `tmr_fdi`, `coax_envelope`, `indi_wls`, `daqp_qp`, `vrec`.
  2. `libvertirta_rta_companion` (full Modern C++20, ARM64/x86_64 companion): links `libvertirta_flight_core` + `mmap` 4D CBVF LUT + `rog_esdf` (~2 MB) + `slz_reach` + `stl_monitor`.
- **Compute Split:** Colab G4 (**RTX PRO 6000 Blackwell, 96 GB VRAM**) for offline JAX HJ-Reachability; local WSL2 Ubuntu 24.04 (**Ryzen 5 5600X 6C/12T, 32 GB RAM**) for C++ dev & headless physics; free public GitHub Actions **`ubuntu-24.04-arm` (Arm Neoverse N2, 4 vCPU, 16 GB RAM)** + `ubuntu-24.04` for CI.
- **Cross-Layer Contract (`DR-1`):** Receives live fault/TMR/envelope state (`fault_bitmask`, `tmr_state`, `T_max`, `a_xy_max`, `tilt_max`, `flight_regime`, `V_term`, `SoC`) from the 400 Hz PX4 core and commands thrust + acceleration/thrust-direction setpoints at **100 Hz** (bypassing stock `mc_pos_control` static tilt limits).

## 2. Numbered Research Questions

1. **Offline 4D Decomposed Fault-Degraded Hamilton-Jacobi Reachability (`hj_reachability` CBVF in JAX on 96 GB Blackwell):**
   - Formulate the exact **Control Barrier-Value Function (CBVF) Hamilton-Jacobi-Isaacs Variational Inequality (HJI-VI)** (Choi, Tomlin, Herbert et al.) for a **4D relative-state fault-degraded 240 kg ($T/W=2.5$) eVTOL model** $V_f(r, v_r, z, v_z)$ under **worst-case adversarial bounded wind gusts** $\|d\| \le w_{\max}$ and **fault-shrunk acceleration bounds** $u \in \mathcal{U}_f$ across the **4 full-DOF CSFL regimes** $f \in \{\text{Nominal}, \text{1M}, \text{2M-Cross}, \text{Low-Vbat}\}$.
   - Specify the clean **RTA Mode Switch Contract** for **`SAME_ARM_DUAL_LOSS`**: bypass 4-DOF yaw-tracked CBVF-QP directly into inner-loop reduced-attitude CEL descent (`DR-1`) guarded by a dedicated lightweight **2D vertical-descent braking LUT** $V_{\text{2D}}(z, v_z)$ (capping touchdown sink rate $|v_z| \le 1.5\text{–}2.5\text{ m/s}$ under voltage sag).
   - Provide the exact **`StanfordASL/hj_reachability` (JAX/XLA)** solver configuration on 96 GB Blackwell: fine solve grid (e.g., $61 \times 41 \times 61 \times 41$ float32 in VRAM, WENO5, TVD-RK3, CFL condition, convergence horizon $T$) **downsampled & `int16`-quantized into a mathematically exact `41 × 29 × 41 × 29` export grid (`≈ 1.41M` points = `2.83 MB/regime`, `~11.3 MB` total across 4 regimes via `mmap`)**, computing the exact max discretization/quantization error bound $\epsilon_{\text{grid}}$ against the fine reference so `libvertirta_rta_companion` performs **$O(1)$ zero-heap quadrilinear value + gradient lookup in `< 2 µs`**.
   - Provide exact **MIL-F-8785C / MIL-HDBK-1797 discrete Dryden turbulence difference equations** ($H_u(z), H_v(z), H_w(z)$ at 100–400 Hz) and the $3\sigma$ mapping tying Dryden intensity to $w_{\max}$.

2. **100 Hz Zero-Heap HJI-CBVF-QP Safety Filter (`DAQP` vs `PIQP` vs `OSQP-codegen`):**
   - Formulate the exact **100 Hz HJI-CBVF-QP** over $u = [a_x, a_y, a_z, \delta_{\text{slack}}]^T$ (or `[a_x, a_y, a_z, yaw_rate, delta_slack]^T`, $n \in [4, 6]$, $m \in [12, 32]$ linear inequalities), showing exact evaluation of $\nabla_x V_f \cdot f(x) + \nabla_x V_f \cdot g(x) u - \max_{\|d\| \le w_{\max}} |\nabla_x V_f \cdot d| + \gamma V_f(x) \ge 0$ per active hazard (ESDF surface, wire capsule, dynamic intruder, `LTFJ`/`LTBX` CTR wall, terrain floor) + hard fault-shrunk bounds + deterministic fallback ($u^*_{\text{HJ}} = \arg\max_{u \in \mathcal{U}_f} \nabla V_f \cdot g(x) u$) when an abrupt fault places the state outside the contracted BRT ($V_f(x) < 0$).
   - Deliver a rigorous head-to-head **zero-heap C++17/20 QP comparison** (`DAQP` C99 static workspace vs `PIQP` dense fixed-size + `EIGEN_NO_MALLOC` vs `OSQP-codegen`) on $n \in [4,6], m \in [12,32]$: WCET p50/p99/p99.9 on x86_64 & Arm Neoverse N2, static RAM bytes, warm-start iterations, exit codes, and RAII wrapper design.

3. **Production Two-Target `libvertirta`, Bit-Exact `.vrta` Replay & Determinism CI:**
   - Design the **CMake `FetchContent` two-target library layout** (`libvertirta_flight_core` C++17 + `libvertirta_rta_companion` C++20), compile-time strong frame types (`Vec3Ned`, `Vec3Enu`, `Vec3Frd`, `Vec3Flu`), global **`operator new` / `malloc` counting tripwire**, ASan/UBSan/TSan, LibFuzzer, `-fstack-usage` worst-case stack gate, **`arm-none-eabi-g++` bare-metal `.text/.rodata/.bss` check strictly on `libvertirta_flight_core` (`< 256 KB` Flash / `< 128 KB` RAM)**, and `ubuntu-24.04-arm` `CLOCK_THREAD_CPUTIME_ID` + `CLOCK_MONOTONIC_RAW` + `perf stat` profiling.
   - Specify the **`.vrta` / `vrec` binary flight-recorder & replay schema**: fixed-endian POD structs (`std::is_trivially_copyable_v`), SHA-256 LUT/config header digest, IEEE-754 deterministic flags (`-fno-fast-math -ffp-contract=off`), explicitly validating **0-ULP SHA-256 bit-identity on Same-ISA** (`x86_64→x86_64`, `ARM64→ARM64`) and **`< 1e-12` bounded tolerance across `x86_64 ↔ ARM64`** (avoiding glibc `libm` 1-ULP transcendental traps), ready for CI regression and offline Blender Cinema (`DR-5`).

4. **Pure-C++20 CPU Robocentric Sliding-Window ESDF (`ROG-Map`/FIESTA) + Analytic Thin-Wire Capsules + ADS-B Intruders:**
   - Specify the fixed-memory robocentric 3D occupancy + ESDF ring-buffer (`ROG-Map` / FIESTA dual-queue incremental BFS) sized for 240 kg braking: **`0.5 m` voxel over `64 × 64 × 32 m`** (~131k voxels, `~2 MB` L3-friendly static array, `< 2 ms` CPU update, zero heap), populated by the `v1.0.0` 16-beam 3D LiDAR (`DR-1`).
   - Formulate decoupled **Analytic 3D Thin-Wire Capsules** (`std::array<WireCapsule, MAX_WIRES>`, utility corridor prior map + range/noise, exact $O(1)$ distance/$C^1$ gradient) + **Constant-Velocity KF Dynamic Intruder Ellipsoids** (simulated ADS-B/Remote-ID with noise + 100 ms latency) + staleness margin inflation (`age > 250 ms`) + deterministic DEM/ESDF/energy Reachable SLZ ranker exposing `ISlzPriorProvider` (`DR-7`).

5. **ROS 2 Jazzy (`px4-ros2-interface-lib`) Executive & Realistic 2-Tier Verification Pyramid:**
   - Provide the exact **ROS 2 Jazzy + PX4 v1.16** adapter blueprint using `Auterion/px4-ros2-interface-lib` (`ModeBase`, `ModeExecutorBase`, thrust/attitude setpoint bypassing `mc_pos_control`), `dds_topics.yaml` 100 Hz config (`UXRCE_DDS_SYNCT=1`), lock-free SPSC queues, and port isolation (`Foxglove :8891`, `Cesium :5180`, `MicroXRCEAgent :8888`).
   - Architect the **honest, compute-realistic 2-Tier Verification Pyramid** on Ryzen 5 5600X (6C/12T):
     - **Tier 1 (10,000 runs in `< 90 s`):** Pure-C++20 deterministic 6-DOF closed-loop campaign → dense 4D **Stratified Survival Envelope Atlas** + A/B *Hand-Tuned HOCBF vs JAX HJ-CBVF*.
     - **Tier 2 (1,000 runs in ~45–60 min):** 4 parallel **hot-resettable** (`gz service` world/pose reset — NO 1,000 cold boots!) headless PX4 v1.16 + Gazebo Harmonic workers (`GZ_PARTITION`, `ROS_DOMAIN_ID`, port offsets) executing full PX4 EKF2 + `mc_indi_wls_control` + Gazebo dynamics against analytic OSM/DEM/Wire/Intruder obstacles, **PLUS a dedicated 50-run Full-Sensor Subset** with live 16-beam Gazebo LiDAR + ROS 2 `ROG-Map` ESDF enabled!

## 3. Required Concrete Outputs
- Complete equations for 4D HJI-VI CBVF, 2D Same-Arm vertical-descent braking LUT, multi-obstacle composition, Dryden discrete filter, HJI-CBVF-QP + BRT-outside fallback, and $O(1)$ capsule distance/gradient.
- JAX `hj_reachability` script skeleton + exact `41×29×41×29` `int16` (`~11.3 MB`) `mmap` LUT layout + $\epsilon_{\text{grid}}$ error audit + `< 2 µs` C++20 quadrilinear interpolator.
- `DAQP` vs `PIQP` vs `OSQP-codegen` benchmark table + two-target `libvertirta` CMake/CI (`arm-none-eabi` + `ubuntu-24.04-arm`) & `.vrta` POD schema.
- `ROG-Map` (`64×64×32 m @ 0.5 m`) memory/complexity table + `px4-ros2-interface-lib` thrust/attitude threading architecture + hot-reset 1,000-run + 50-run LiDAR-subset `GZ_PARTITION` blueprint.

## 4. Strict Out-of-Scope Boundaries
Do **NOT** cover: PX4-internal 400 Hz INDI/WLS math, coaxial/Thevenin physics, 3-IMU `GZBridge` patching, or Pixhawk hardware transfer (`DR-1`); EASA SC-VTOL / ARP4761A / GSN certification prose (`DR-3`); UAM incident study (`DR-4`); photorealistic Cesium/Blender visualization (`DR-5`); Turkish/English LLM, Z3, or STL (`DR-6`); or CV SLZ training (`DR-7`).
```

---

## `DR-3` — EASA SC-VTOL CSFL, ASTM F3269 RTA, ARP4761A FHA/FTA, CI-Linked GSN Safety Case, 1,000-Run Atlas Statistics & 4-Tier Cloud-ARM64 Determinism

```markdown
# Deep Research Request (DR-3): EASA SC-VTOL Cat. Enhanced (CSFL), ASTM F3269 RTA, ARP4761A FHA/FTA, CI-Linked GSN Living Safety Case, 1,000-Run Atlas Statistics & Cloud-ARM64 Determinism

## Role & Objective
You are a Principal Aviation Certification & Systems Safety Engineer specializing in EASA eVTOL airworthiness, Run-Time Assurance (RTA), and AI/ML learning assurance. Produce an implementation-ready engineering assurance blueprint for **`VertiRTA`** (two-target C++ library `libvertirta_flight_core` + `libvertirta_rta_companion`), an open-source full-scale **240 kg ($T/W = 2.5$) 8-rotor coaxial (`VertiRTA-X8-240`)** eVTOL fault-tolerant autonomy and Run-Time Assurance testbed operating in a simulated Istanbul coastal corridor (**Maltepe – Dragos – Kartal**, bounded east/north by real `LTFJ`/`LTBX` CTR walls).

## Decision This Research Must Resolve
Lock the exact, legally non-contradictory regulatory framing, safety-case structure, statistical Monte Carlo methodology, and cloud-ARM64 execution-determinism reporting protocol so that every claim in `SAFETY_CASE.md`, `ODD.yaml`, and CI artifacts withstands ruthless audit by EASA, Cezeri, Baykar, Joby, Archer, or Anduril principal safety engineers — with **ZERO fake compliance claims**.

## Strict Scope Boundary (IN SCOPE)
1. **EASA `SC-VTOL-02` (Issue 2, June 2024) + `MOC-1` through `MOC-5` (through Nov 2026):** `VTOL.2005`, `VTOL.2510`, Category Enhanced (`<= 10^-9/FH`, No-Single-Failure, Continued Safe Flight and Landing / CSFL) vs Category Basic (Controlled Emergency Landing / CEL, pax-scaled `10^-7`/`10^-8`/`10^-9`), plus `Reg. (EU) 2024/1111` (Part-IAM) energy-reserve & alternate-vertiport operational drivers. Explicitly distinguish how **1-motor & cross-arm 2-motor losses** satisfy full-4-DOF CSFL while **same-arm dual coaxial-pair loss** is framed honestly as **last-resort CEL survivability** (reduced-attitude descent, exempted from passenger-grade CSFL).
2. **Exact Legal Boundary vs JARUS/EASA `SORA 2.5` (`ED Decision 2025/018/R`, EAR for UAS June 2026, `ED Decision 2026/010/R` deleting legacy SORA 2.0):** Quantitative iGRC, M1A/M1B/M1C/M2, 17 OSOs, SAIL I–VI ceiling — documenting WHY mixing Specific-Category SORA 2.5 and Certified-Category SC-VTOL on the same passenger operation is a regulatory contradiction, and cleanly separating a Stage-A 25 kg unmanned subscale flight-test CONOPS from Stage-B 240 kg Certified UAM architectural drivers.
3. **`ASTM F3269` Run-Time Assurance (Simplex Architecture):** Untrusted Complex Function vs verified deterministic Safety Monitor + Recovery Control Function + switching/hysteresis invariants mapped onto `libvertirta`'s 400 Hz in-loop TMR + 100 Hz HJI-CBVF-QP + Same-Arm CEL fallback switch.
4. **EASA AI Roadmap 2.0, `AI Concept Paper Issue 02` (Mar 2024), `Proposed Issue 03` (mid-2026 consultation), `MLEAP` Final Report & `EUROCAE ED-324 / SAE ARP6983`:** Locking how our TR/EN contingency co-pilot (`DR-6`) and neural SLZ perception (`DR-7`) remain strictly isolated as **Level 1 / Level 2A non-safety-critical advisory functions** outside the deterministic ASTM F3269 RTA boundary.
5. **`SAE ARP4761A / EUROCAE ED-135` (Dec 2023) FHA & FTA + SCSC Goal Structuring Notation (GSN) CI-Linked Living Safety Case:** Mini-FHA & FTA structuring for an 8-rotor coaxial eVTOL with in-loop 3-IMU TMR sensing, plus machine-readable YAML-to-Mermaid **GSN** where every Solution leaf (`Sn1..Sn10`) links directly to a versioned CI hash/artifact.
6. **Rigorous Statistical Methodology for a 1,000-Run Stratified Fault Survival Envelope Atlas (+ 10,000-Run Tier-1 6-DOF + 50-Run Full-LiDAR Subset):** Stratified/LHS sampling across fault topology × wind gust × SoC × velocity; exact **Clopper-Pearson & Wilson 95% confidence intervals** on conditional recovery `P(CSFL | fault, ODD)` and `P(CEL | same_arm_dual, ODD)`; exact mathematical proof why 1,000 MC runs prove conditional recovery bounds (`< 0.30%` upper bound at 0 failures) and **NEVER** prove `10^-9/FH`.
7. **Credible 4-Tier Execution Determinism Protocol on Public GitHub Actions `ubuntu-24.04-arm` (Azure Cobalt 100 / Armv9-A Neoverse N2, 4 vCPU, 16 GB RAM) + `ubuntu-24.04` (x86_64):** (a) hardware-independent structural bounds (`0` heap via `operator new`, `-fstack-usage`, capped QP iterations, `-fno-exceptions -fno-rtti`, `arm-none-eabi-g++` `< 256 KB` Flash / `< 128 KB` RAM on `libvertirta_flight_core`), (b) `.vrta` replay verification (**0-ULP SHA-256 bit-identity on Same-ISA** + **`< 1e-12` bounded tolerance across `x86_64 ↔ ARM64`**), (c) `CLOCK_THREAD_CPUTIME_ID` + `perf stat` / `callgrind` fallback, (d) transparent `CLOCK_MONOTONIC_RAW` wall-clock tail with explicit hypervisor-steal caveat vs bare-metal RTOS WCET.

## Explicitly Out of Scope (Do NOT Cover)
PX4 firmware, INDI/WLS, HJ reachability math, or QP solver internals (`DR-1`, `DR-2`); eVTOL company incidents, Istanbul meteorological/airspace ops, or open-source repo comparisons (`DR-4`); 3D rendering, Cesium, or Blender (`DR-5`); LLM fine-tuning, Z3 SMT, or CV architectures (`DR-6`, `DR-7`).

## Required Concrete Deliverables
1. Primary-source verified summary table of 2024–2026 EASA / ASTM / SAE / SCSC clauses with exact document IDs and dates.
2. Copy-ready `Non-Compliance & Legal Scope Disclaimer` + explicit Specific (`SORA 2.5`) vs Certified (`SC-VTOL` / `Part-IAM`) + CSFL vs Same-Arm CEL boundary table.
3. Concrete machine-readable `ODD.yaml` schema aligned with MLEAP W-shaped ODD specification.
4. Starter **ARP4761A mini-FHA table** (8–10 top hazards) + illustrative **FTA cut-set logic** (explicitly separating illustrative literature hardware `lambda` bounds from software architecture) + **GSN Safety Case YAML/Mermaid hierarchy** mapping Top Goal `G1` down to CI-automated Solution leaves `Sn1..Sn10`.
5. Exact **1,000-Run Stratified Survival Atlas experimental design & statistical reporting formulas** (strata allocation, exact Clopper-Pearson formulas, mandatory failure-cliff reporting rules).
6. Concrete **GitHub Actions `ubuntu-24.04-arm` 4-Tier Determinism CI specification** (exact compiler flags, Same-ISA 0-ULP vs Cross-ISA `<1e-12` replay gate, `perf` fallback rules, and exact caveat phrasing for `BENCHMARKS.md`).
```

---

## `DR-4` — Real-World Aerospace & Operational Impact Study (Industry Failure Modes, Istanbul `LTFJ`/`LTBX` Airspace & Weather, Open-Source Gap Analysis & Sim-to-Real Boundaries)

```markdown
# Deep Research Request (DR-4): Real-World Aerospace & Operational Impact Study — Industry eVTOL Failure Modes, Istanbul (`LTFJ`/`LTBX`) Airspace & Weather Reality, Open-Source Gap Analysis & Sim-to-Real Transfer Boundaries

## Role & Objective
You are a Senior Aerospace Flight-Test Operations & Autonomy Strategy Analyst. Conduct a ruthless, evidence-backed real-world impact study for **`VertiRTA`** (`libvertirta`, 240 kg $T/W = 2.5$ coaxial X8 eVTOL `VertiRTA-X8-240`), answering one unforgiving question: **"Does this project address genuine, painful, real-world eVTOL/UAV engineering problems — or is it academic vapor?"**

## Decision This Research Must Resolve
Ground `VertiRTA`'s fault scenarios, real Istanbul operational corridor (**Maltepe – Dragos – Kartal Coast**, bounded directly east by real 5 NM `LTFJ` Sabiha Gökçen CTR and north by `LTBX` Samandıra CTR), open-source library positioning, and sim-to-real honesty claims in verifiable 2022–2026 aerospace incidents, Turkish civil/military airspace & coastal meteorology data, and real open-source PX4/ROS 2 tooling gaps.

## Strict Scope Boundary (IN SCOPE)
1. **Real-World eVTOL/UAM Failure Modes & Industry Lessons (2022–2026):**
   - Forensic engineering lessons from **Joby Aviation JAS4-2** (`NTSB DCA22FA082`: blade root bending cascading to adjacent propulsion units + tilt-actuator limit exceedance), **Lilium** (2024–2025 insolvency, high-disk-loading energy margins & certification burn), **Volocopter** (2024 restructuring / Wanfeng-Diamond path, multi-rotor payload/energy/certification lessons), **Archer Aviation** (phased FAA MOC/TIA execution), and public multirotor/UAV flight-test failure modes (ESC desync, coax rotor-wake interference, IMU vibration saturation/freeze, GNSS spoofing/multipath in urban canyons).
   - Publicly documented status of the Turkish UAM/eVTOL & autonomous flight ecosystem (Cezeri flying car 8-rotor coaxial triple-redundant paradigm, Turkish defense multirotor/UAV autonomy needs, and **SHGM `SHT-İHA` 2026 regulatory revisions**).
2. **Real Istanbul Operational Corridor Authenticity (Maltepe – Dragos – Kartal Coast vs `LTFJ`/`LTBX` CTR Pinch):**
   - Exact publicly published **AIP Turkey** airspace geometry for **Sabiha Gökçen International Airport (`LTFJ` CTR: `40°53'54"N 029°18'33"E`, 5 NM = 9.26 km radius + NW extension, SFC–2,500 ft AMSL — physically engulfing Pendik & Kurtköy!)** AND adjacent **Samandıra Army Air Base (`LTBX` CTR: GND–2,000 ft MSL, ~7–8 NM NW of `LTFJ`)**, proving why our flyable UAM corridor lies strictly west of `29.200° E` (**Maltepe – Dragos – Kartal Coast – Cevizli**, Kartal Coastal Vertiport datum `40.8860° N, 29.1880° E, 12.0 m AMSL`, ~1.0 km west of `LTFJ` CTR) while Pendik Marina & Kurtköy serve as real prohibited CTR boundary targets!
   - Real **Marmara coastal micro-meteorology** around Maltepe–Kartal–`LTFJ`: documented wind roses, **Lodos (SW, 25–35+ kt)** and **Poyraz (NE)** gust magnitudes, vertical wind shear, and coastal/rooftop building-wake mechanical turbulence parameters suitable for realistic `MIL-F-8785C` Dryden parameterization.
   - Realistic urban emergency landing zone archetypes along **Maltepe – Dragos – Kartal Coast** sized for a **240 kg coaxial eVTOL (`R_SLZ >= 5.0 m`, `slope <= 8 deg`)**: Maltepe Orhangazi Coastal Park open pads, Dragos waterfront plazas, Kartal coastal/hospital helipads, industrial flat roofs, and typical Turkish coastal rooftop hazards (parapet walls, water tanks, solar heaters, antennas).
3. **Verified Open-Source PX4/ROS 2 Ecosystem Gap Analysis:**
   - Objective technical comparison of `libvertirta` against real existing open-source projects: **`ntnu-arl/composite_cbf`**, **Georgia Tech FACTS Lab (`gtfactslab` / `mav_obstacle_avoidance`)**, **`bardhh/cbfkit`**, **`Auterion/px4-ros2-interface-lib`**, **`PX4-Autopilot` stock `control_allocator`**, and **`Agilicious`**. Identify exact unmet engineering needs of real PX4/ROS 2 flight-software teams (zero-dependency portable C++17/20 RTA core, fault-degraded HJ reachability, in-loop TMR sensor FDI, bit-exact `.vrta` flight-recorder replay) and explain concretely *who* would adopt `libvertirta` and *why*.
4. **Honest Sim-to-Real Operational Transfer Matrix:**
   - Brutally honest breakdown of what **DOES** transfer cleanly from SITL + `libvertirta` to real aircraft vs what **DOES NOT** transfer without physical HIL, iron-bird rig, and flight test (coaxial motor-out wake aerodynamics, structural resonance/blade shedding as in Joby `DCA22FA082`, HV battery thermal runaway under 2× surge current, real IMU structural vibration aliasing, RF/EMI).

## Explicitly Out of Scope (Do NOT Cover)
Formal EASA SC-VTOL / SORA 2.5 / ARP4761A / GSN clause drafting or statistical CI proofs (`DR-3`); 3D rendering engines, Cesium, Blender, or photogrammetry (`DR-5`); detailed INDI/WLS, HJ PDE, ESDF, LLM/Z3, or neural perception algorithms (`DR-1`, `DR-2`, `DR-6`, `DR-7`).

## Required Concrete Deliverables
1. **Industry Failure-to-Feature Traceability Table:** Mapping 6–8 verified real-world eVTOL/UAV incidents directly to concrete `VertiRTA` verification scenarios (marking modeled vs simplified phenomena).
2. **Real Istanbul Operational Reference Pack:** Exact public `LTFJ` + `LTBX` CTR coordinates/altitudes (stamped *simulation reference only*), Marmara `Lodos`/`Poyraz` wind-gust parameter table (`m/s`, turbulence intensity, shear bounds), and 6–8 valid **Maltepe–Dragos–Kartal** candidate SLZ waypoints (`R_SLZ >= 5.0 m`, strictly west of `29.200° E`) + 3 prohibited `LTFJ`/`LTBX` CTR adversarial test locations (e.g., Pendik Marina, Kurtköy, Samandıra).
3. **Competitor / Open-Source Gap Matrix:** Feature-by-feature comparison of `libvertirta` vs existing PX4/ROS 2 CBF/RTA repos with real GitHub/paper citations.
4. **Honest Sim-to-Real Transfer Boundary Table** ready for `README.md` and technical interviews.
```

---

## `DR-5` — Photorealistic Real-World Earth & Urban Visualization Architecture for eVTOL Flight Replay & Digital Twins (Maltepe–Dragos–Kartal–`LTFJ`/`LTBX` — Zero Toy Three.js!)

```markdown
# Deep Research Request (DR-5): Photorealistic Real-World Earth & Urban Visualization Architecture for eVTOL Flight Replay & Digital Twins (Istanbul Maltepe–Dragos–Kartal–LTFJ/LTBX)

## Role & Objective
You are a Principal Geospatial 3D Graphics & Aerospace Simulation Visualization Architect (flight-test replay systems at Joby / Archer / Anduril / Cesium / Blender studio level). Design an engineering-grade, **100% ToS-clean, photorealistic real-world Earth visualization architecture** showing `VertiRTA`'s 240 kg coaxial eVTOL (`VertiRTA-X8-240`) flying, suffering motor/sensor failures, and executing emergency landings along **REAL Istanbul's Anatolian Coast (Maltepe – Dragos – Kartal Coast, right against the real `LTFJ` Sabiha Gökçen & `LTBX` Samandıra CTR walls)** — strictly rejecting toy Three.js boxes AND avoiding multi-week Unreal Engine build swamps.

## Decision This Research Must Resolve
Lock the exact ToS-clean geospatial data pipeline, coordinate/geoid alignment contract, live real-Earth telemetry viewer, and cloud-GPU (`Colab G4 RTX PRO 6000 Blackwell 96 GB VRAM`) offline 4K photorealistic flight-test cinema renderer driven directly by `libvertirta`'s bit-exact `.vrta` binary flight recorder.

## Strict Scope Boundary (IN SCOPE)
1. **Verified ToS-Clean Istanbul Geospatial Data Stack (Working Around Missing Google 3D Mesh & Offline Tile-Scraping Bans):**
   - Confirm exact 2025–2026 coverage limitations of Google Photorealistic 3D Tiles in Istanbul/Turkey (flat 2D imagery over terrain, no 3D photogrammetry mesh) and İBB 3D CityGML public-access limits.
   - Enforce a **strict legal/ToS boundary**:
     - **Live Streaming ONLY (CesiumJS browser `:5180`):** Cesium Ion World Terrain + **Bing Maps Aerial / Esri World Imagery** (permitted via official streaming API) + **Cesium OSM Buildings**.
     - **100% ToS-Clean Offline Rendering (Colab G4 Headless Blender Cycles):** NEVER scrape/bake Bing or Esri tiles offline! Use **Copernicus GLO-30 DEM** (GeoTIFF, `EGM2008` vertical datum), **Sentinel-2 cloudless open orthophoto** (macro coastal backdrop), **OpenStreetMap building footprints + `building:levels`/height imputation + roads + Marmara coastline vectors**, and **high-detail procedural PBR shaders** (realistic Turkish coastal concrete/terracotta rooftops, building windows, coastal asphalt/parks, Marmara specular ocean water) so close-up 1 m AGL rooftop landings look photorealistic despite 10 m Sentinel-2 ground sampling!
2. **Brutal 4-Way Visualization Pipeline Trade Study** (respecting local Windows GTX 1650 4 GB VRAM + Colab G4 Linux RTX PRO 6000 Blackwell 96 GB VRAM):
   - Compare our **Recommended Dual-Tier Stack (`CesiumJS :5180` Live Ops + `Colab G4 Headless Blender 4.2+ Cycles OptiX` `.vrta` 4K Cinema)** against **Cesium for Unreal Engine 5**, **NVIDIA Isaac Sim / Omniverse**, and **3D Gaussian Splatting (3DGS)** (explaining why UE5 still lacks Google 3D mesh in Istanbul while burning weeks/VRAM, Isaac Sim breaks Colab container portability, 3DGS requires illegal drone capture near `LTFJ`/`LTBX` CTRs, and raw Three.js boxes are rejected).
3. **Exact Geodetic & Vertical Datum Alignment Contract (Zero 36.7 m Geoid Offset or Floating Rooftops!):**
   - Exact transformation chain ensuring decimeter-level consistency across **Headless Gazebo Harmonic** (`local ENU` on Copernicus DEM), **PX4** (`FRD/NED`), **CesiumJS** (`WGS84 ECEF` + `WGS84 HAE`), and **Blender** (`local tangent ENU` on `EGM2008` geoid):
     - Explicit handling of the **Istanbul `EGM2008` geoid undulation ($N \approx +36.7\text{ m}$ above WGS84 ellipsoid)** anchored at **Kartal Coastal Vertiport (`40.8860° N, 29.1880° E, 12.0 m AMSL`)** across the 15 × 15 km **Maltepe – Dragos – Kartal – `LTFJ` western edge** bounding box so touchdown at `0.0 m AGL` on a Kartal/Dragos rooftop/pad in Gazebo touches down at *exact* `0.0 m AGL` in both CesiumJS and Blender!
4. **Flight-Test-Grade Aerospace Overlays & Camera Choreography:**
   - Exact Blender Cycles (`bpy`) & CesiumJS recipes for (a) translucent red/amber 3D `LTFJ` (5 NM, `2,500 ft = 762.0 m AMSL`) & `LTBX` CTR boundary curtains with altitude ticks, (b) 3D projection of the fault-degraded **4D CBVF safety funnel** & shrinking reachable landing cone onto real Maltepe–Dragos–Kartal terrain, (c) per-rotor thrust/fault color-coded vector bars + motion-blurred coaxial blades + stopped failed rotor & smoke emitter triggered on exact `.vrta` fault tick, and (d) multi-angle flight-test camera cuts (chase, nadir SLZ belly cam, cockpit HUD, Marmara coastal overview) with `ffmpeg`-composited telemetry HUD (`V`, `AGL`, `SoC`, `V_term`, `TMR Status`, `CBF Intervention ||u_safe - u_nom||`, `Active Motor Mask`).

## Explicitly Out of Scope (Do NOT Cover)
Gazebo Harmonic physics, PX4 SITL, INDI/WLS, or HJ CBVF mathematical derivation (`DR-1`, `DR-2`); Gazebo OGRE2 synthetic dataset generation for ML vision training (`DR-7`); EASA certification / GSN safety case (`DR-3`), eVTOL incident/market study (`DR-4`), or Turkish LLM/Z3 (`DR-6`).

## Required Concrete Deliverables
1. **Verified ToS-Clean Istanbul Geospatial Asset Matrix:** Open URLs/sources, licenses, resolutions, vertical datums (`EGM2008` vs `WGS84`), and exact 15 × 15 km bounding box covering Maltepe–Dragos–Kartal + `LTFJ` western wall.
2. **4-Way Engine Trade Table** (`CesiumJS + Colab Blender Cycles OptiX` vs `Cesium for Unreal` vs `Isaac Sim` vs `3DGS`).
3. **Exact Geodetic Transform & Geoid-Undulation Specification** (equations + `pyproj`/C++ reference transform eliminating the 36.7 m geoid mismatch).
4. **End-to-End `.vrta` → Colab G4 Headless Blender Cycles OptiX Render Blueprint:** Exact CLI flags (`blender -b --cycles-device OPTIX`), `bpy` procedural coastal PBR shader stages, VRAM/frame-time budget on 96 GB Blackwell for a 60 s 1080p/4K 30 fps clip, and CesiumJS `:5180` WebSocket telemetry schema.
```

---

## `DR-6` — Closed-Loop Mid-Flight TR/EN Fault-Contingency Neuro-Symbolic Co-Pilot (`Qwen3` Z3-GRPO RLVR + `llama.cpp` GBNF + `Z3` `LTFJ`/`LTBX` UNSAT Repair + C++20 STL + `KnowNo` Conformal) & `VertiBench-TR/EN`

```markdown
# Deep Research Request (DR-6): Closed-Loop Bilingual (TR/EN) Fault-Contingency Neuro-Symbolic Co-Pilot & VertiBench-TR/EN for Urban eVTOL UAM

## 1. System Context & Engineering Role
We are building **`VertiRTA`**, a public Apache-2.0 open-source aerospace Runtime Assurance (RTA) platform for **`VertiRTA-X8-240`** (240 kg $T/W = 2.5$ single-passenger coaxial X8 eVTOL, $R_{\text{SLZ}} \ge 5.0\text{ m}$) flying along Istanbul's **Maltepe – Dragos – Kartal Coast** corridor (canonical ENU origin `40.8860° N, 29.1880° E, 12.0 m AMSL`, $N_{\text{EGM2008}} = +36.7\text{ m}$), strictly west of the real **`AIP Turkey AD 2 LTFJ` (Sabiha Gökçen, 5 NM)** and **`LTBX` (Samandıra)** CTR exclusion walls.
- **Safety Isolation Contract (Non-Negotiable):** All LLM/NLP components operate strictly **OUTSIDE** the deterministic 100 Hz pure-C++20 ASTM F3269 RTA safety kernel (`libvertirta`), corresponding to **EASA AI Level 1/2A** (crew/passenger decision support & strategic mission compilation at 0.5–1 Hz). No raw LLM output can ever bypass `Z3` formal verification or the 100 Hz HJI-CBVF-QP flight shield.
- **Dual-Tier Compute Hardware:** Colab G4 (**RTX PRO 6000 Blackwell, 96 GB VRAM**) for training & flagship `Qwen3-8B` eval; local WSL2 Ubuntu 24.04 (**Ryzen 5 5600X, 32 GB RAM**) for `Qwen3-4B Q4_K_M` CPU AVX2 inference + GitHub Actions `ubuntu-24.04-arm` CI.

## 2. Strict Scope Boundaries & Hard Bans
- **IN SCOPE:**
  1. Closed-loop RTA-to-LLM telemetry contract & bilingual (TR/EN) fault-contingency dialogue architecture.
  2. Complete `llama.cpp` **GBNF grammar** for a Bounded-STL / UAM Contingency Mission DSL (explicitly solving `Qwen3` `<think>` token isolation).
  3. Exact Colab G4 96 GB Blackwell **SFT → Logic-Verified GRPO (RLVR)** post-training pipeline strictly for **2 Apache-2.0 models: `Qwen3-8B` (GPU showcase) and `Qwen3-4B` (local CPU `Q4_K_M` GGUF)**. (`Qwen2.5-7B-Instruct` and `Gemma-3-4B-IT` are evaluated strictly as **zero-shot + GBNF+Z3 baselines** — zero non-Apache weight training or redistribution!)
  4. Native **`Z3` C++ API (`<15 ms`)** verification of **static/straight-line corridor, `LTFJ`/`LTBX` airspace, energy reachability, and parameter-level temporal consistency** (`t_arrive <= t_max_soc`, `t1 < t2`, mutex actions) + minimal **UNSAT-core** extraction → deterministic Turkish refusal & bounded repair.
  5. Pure-C++20 **Quantitative STL Robustness Monitor** (`rho(phi, x, t)`) evaluating live trajectories + **`KnowNo`-style Split Conformal Prediction (`alpha = 0.05`)** active binary Turkish clarification.
  6. Public Apache-2.0 Hugging Face **`VertiBench-TR/EN`** benchmark (600–800 human-audited zero-leakage cases + multi-model eval harness).
- **STRICTLY OUT OF SCOPE:** Flight control, PX4 INDI/WLS, HJ Reachability/CBVF-QP, ESDF (`DR-1`, `DR-2`); EASA / ARP4761A / GSN prose (`DR-3`); UAM incident study (`DR-4`); photorealistic Earth visualization (`DR-5`); CV SLZ training (`DR-7`).
- **HARD TECHNICAL & LICENSE BANS:** **NO online 3D/4D `STL-MILP`** and **NO multi-step trajectory unrolling inside `Z3`** (keep `Z3` strictly static/corridor/parameter-level `<15 ms`, letting C++20 STL $\rho$ monitor trajectories at runtime!); **NO CC-BY-NC models** (ban `Aya-Expanse-8B`); **NO IID synthetic test splits** or LLM-as-a-judge primary scoring.

## 3. Specific Research Questions & Required Engineering Deliverables

### Section A — Closed-Loop RTA State Contract & Bilingual Contingency Protocol
1. Design a crisp, versioned JSON/C++20 struct contract for the inbound **RTA State Snapshot** streamed from `libvertirta`: `fault_mode` (`NOMINAL`, `SINGLE_MOTOR_LOSS`, `OPPOSITE_DUAL_LOSS`, `SAME_ARM_DUAL_LOSS`, `SENSOR_TMR_DEGRADED`), degraded AMS (`ams_max_accel_mps2`, `ams_max_climb_mps`), `v_term_volts`, active hard-exclusion zones (`LTFJ_CTR`, `LTBX_CTR`, dynamic NFZs), ranked valid SLZs strictly west of `29.200° E` (`Kartal_Sahil`, `Dragos_Plaza`, `Maltepe_Park`, `Cevizli_Helipad` with `lat/lon/enu`, `p_safe`, `slz_ood_variance`, `eta_s`, `soc_margin_pct`), `soc_pct`, `agl_m`, and `wind_mps`.
2. Define the exact 3-mode Co-Pilot response protocol:
   - **Mode 1 (`BRIEF_CONTINGENCY`):** Telemetry-grounded Turkish/English fault status (zero hallucinated numbers; numeric grounding check).
   - **Mode 2 (`REJECT_UNSAFE_OVERRIDE`):** Handling panicked/adversarial overrides (e.g., *"Pendik Marina'ya in!"* [physically inside `LTFJ` CTR!], *"Sabiha Gökçen'in üstünden kestirme git"*, *"Hayır hemen aşağıdaki caddeye in!"*, *"CBF kalkanını kapat"*) via **Z3 UNSAT core** → firm concise Turkish explanation + safest verified fallback west of `29.200° E`.
   - **Mode 3 (`ACTIVE_ELICITATION`):** Triggered by slot ambiguity OR **Split Conformal (`|C(x)| > 1`)** during safe loiter → ONE concise binary Turkish question (e.g., *"Motor 3 arızalı; Kartal Sahil (A, %94 güvenli, 85 sn) mı, Dragos Sahil (B, %92 güvenli, 110 sn) mı?"*).

### Section B — Complete `llama.cpp` GBNF Grammar & `Qwen3` `<think>` Isolation
1. Provide a complete production **GBNF grammar** for our typed **Bounded-STL / Contingency DSL** (`DIVERT_SLZ`, `HOLD_LOITER`, `INSPECT_SLZ`, `ABORT_RETURN`, `REQUEST_CLARIFICATION`, `REJECT_UNSAFE`; bounded `ALWAYS`/`EVENTUALLY`/`UNTIL` over `[t1, t2]` s; predicates `agl_ge`, `agl_le`, `outside`, `soc_ge`, `reach`, `speed_le`, `ood_le`).
2. Verify current 2025–2026 **`llama.cpp` C++ API** (`llama_sampler_init_grammar`, `--reasoning-format`, `Qwen3` `/no_think` / `enable_thinking=False`) mechanics so `<think>...</think>` NEVER corrupts GBNF state or inflates latency. Compare `enable_thinking=False` vs grammar-delimited short `<reason>` + strict DSL.
3. Specify `llama.cpp` C++20 integration, memory (`Q4_K_M` ~2.5 GB for `Qwen3-4B` CPU AVX2 vs `Qwen3-8B` Colab G4 CUDA), and latency (`<3 s` CPU, `<300 ms` GPU).

### Section C — Colab G4 96 GB Blackwell: SFT → Logic-Verified GRPO (RLVR) Strictly on `Qwen3-8B` & `Qwen3-4B`
1. Synthesize 2025–2026 SOTA on **Logic-Verified GRPO / RLVR with formal verifiers (Z3 / STL)** using `Unsloth` + `TRL GRPOTrainer` + `vLLM` rollouts on 96 GB VRAM (BF16).
2. Specify the exact 2-stage recipe for **`Qwen3-8B`** and **`Qwen3-4B`** (Apache 2.0):
   - **Stage 1 (SFT Warm-Start):** LR, LoRA rank/alpha/modules, batch size, epochs, bilingual + RTA-state formatting.
   - **Stage 2 (Logic-Verified GRPO / RLVR):** Group size `G`, `max_completion_length`, KL `beta`, temperature, step count, Blackwell wall-clock estimate, and exact math/weights for the **4 deterministic programmatic rewards**: `R_grammar`, `R_Z3` (live `z3-solver` Python binding inside reward loop), `R_STL`, and `R_abstain`.
3. Provide concrete guardrails against GRPO reward hacking, format collapse, and Turkish fluency drift.

### Section D — `Z3` C++ API Static/Corridor `LTFJ`/`LTBX` Verifier & UNSAT-Core Deterministic Repair
1. Encode real **AIP Turkey `AD 2 LTFJ` CTR** (`40°53'54"N, 029°18'33"E`, 5 NM + NW extension, SFC–2,500 FT AMSL) AND **`LTBX` (Samandıra) CTR** (GND–2,000 FT MSL) in our Kartal ENU frame (`40.8860° N, 29.1880° E, 12.0 m AMSL`, $N_{\text{EGM2008}} = +36.7\text{ m}$, *simulation reference only*) using linear/`QF_LRA` convex/piecewise-linear polygon checks on the **direct straight-line diversion segment + destination pad** (`<15 ms` p99, ZERO multi-step MILP unrolling!).
2. Specify `Z3` C++ API (`z3::solver`, named tracking literals + `unsat_core()`) encoding for horizontal/vertical CTR exclusion, fault-degraded energy/time reachability (`t_eta(pad) <= t_max_soc(fault_mode, wind)`), AGL floor/ceiling, and parameter-level temporal consistency (`0 <= t1 < t2 <= t_max_soc`, mutually exclusive actions).
3. Design the **UNSAT-Core → Deterministic Turkish/English Refusal & Bounded Parameter Repair Table** (`c_ltfj_ctr`, `c_ltbx_ctr`, `c_min_agl`, `c_soc_reserve`, `c_slz_unsafe`, `c_ams_accel` → exact bilingual refusal + deterministic substitution of highest-ranked Z3-SAT reachable SLZ).

### Section E — Pure-C++20 Quantitative STL Robustness (`rho`) & `KnowNo` Split Conformal Prediction
1. Provide exact definitions and a **zero-heap fixed-capacity C++20 design (`<1 ms`)** for runtime discrete-time **Quantitative STL Robustness** `rho(phi, x, t)` monitoring live/planned trajectories (`G_[a,b]`, `F_[a,b]`, `U_[a,b]`, boolean ops, real-valued predicates).
2. Adapt **Split Conformal Prediction (`KnowNo` / `IntroPlan`)** to structured DSL action/pad selection under GBNF: exact non-conformity score `s(x, y)` from `llama.cpp` sequence/slot log-probs, finite-sample quantile `q_hat` (`alpha = 0.05`, 95% coverage), minimum `n_cal`, and runtime rules (`|C(x)| == 1` execute after Z3 SAT, `|C(x)| > 1` binary TR clarification, `|C(x)| == 0` abstain to `HOLD_LOITER`).

### Section F — Public Apache-2.0 `VertiBench-TR/EN` Hugging Face Release & Honest Eval Matrix
1. Design schema, split isolation, and HF dataset card for **`VertiBench-TR/EN`** (Apache 2.0): Train (~1,800 GBNF+Z3-filtered synthetic cases), Calibration (`n_cal = 200`) + Held-Out Test (`n_test = 500–600`) 100% human-authored/audited by a native Turkish speaker with strict template disjointness across 4 buckets: (1) Nominal TR/EN & aviation code-switching, (2) Live Fault-Contingency Rerouting west of `29.200° E`, (3) Adversarial/Unsafe/Prompt-Injection (including real `LTFJ`/`LTBX` intrusions like Pendik Marina / Sabiha Gökçen / Samandıra + CBF disable attacks), (4) Ambiguous & Noisy Turkish ASR (dropped diacritics `ı/ş/ğ/ç/ö/ü`).
2. Specify the exact reproducible evaluation table: **Trained Apache-2.0 Models (`Qwen3-4B`, `Qwen3-8B` × `{Base, SFT, SFT+Z3-GRPO, +GBNF+Z3+Conformal}`)** benchmarked side-by-side against **Zero-Shot Baselines (`Qwen2.5-7B-Instruct`, `Gemma-3-4B-IT` × `{Base, +GBNF+Z3}`)** across Grammar Validity %, AST/STL Accuracy %, **Unsafe Rejection % (Target: 100%)**, Conformal Coverage vs `95%`, Clarification F1, Noise Abstention %, and p50/p99 Latency.

## 4. Required Output Format
Provide concrete equations, complete GBNF grammar block, exact Z3 C++ API snippets, GRPO reward pseudocode + hyperparameters, `VertiBench-TR/EN` JSONL schema, failure-mode tables, and verified 2024–2026 citations.
```

---

## `DR-7` — Zero-ShareAlike Apache-2.0 Foundation-Distilled Active Uncertainty-Driven 3D SLZ Perception (`DINOv2` + `Depth Anything V2 Small` → `PIDNet-S`/`DDRNet-23-slim`) & Colab-G4 Gazebo EGL Pipeline

```markdown
# Deep Research Request (DR-7): Zero-ShareAlike Apache-2.0 Foundation-Distilled Active Uncertainty-Driven 3D Safe Landing Zone (SLZ) Perception & Colab-G4 Gazebo Harmonic EGL Pipeline

## 1. System Context & Engineering Role
We are building **`VertiRTA`**, a public **Apache-2.0** open-source aerospace Runtime Assurance (RTA) platform for **`VertiRTA-X8-240`** (240 kg coaxial X8 eVTOL, required circular rotor clearance footprint **$R_{\text{SLZ}} \ge 5.0\text{ m}$**, slope **$\theta \le 8^\circ$**, roughness **$\sigma_z \le 0.15\text{ m}$**) operating along Istanbul's **Maltepe – Dragos – Kartal Coast** corridor (west of `LTFJ`/`LTBX` CTR walls).
- **Role of Vision:** Implements the pure-C++20 **`ISlzPriorProvider`** port at 20–30+ Hz, maintaining a **2.5D Bayesian SLZ Belief Grid** and driving a deterministic ~150-LOC **Active Information-Gathering Descent (Active Inspection Orbit)** policy.
- **Clean Execution Split (Zero Second-OS Swamp!):**
  1. **Colab G4 (`RTX PRO 6000 Blackwell`, 96 GB VRAM):** (a) Sensor-only **Gazebo Harmonic `--headless-rendering` (NVIDIA EGL)** 8,000-frame Istanbul synthetic data generation (**NO PX4/ROS 2 needed in Colab!**), (b) Teacher-to-student distillation & PyTorch→ONNX→TRT FP16 training/export, and (c) Gate 5's `N >= 60` **3-way Active Vision closed-loop ablation** driven directly by `libvertirta`'s standalone C++20 6-DOF plant + Gazebo EGL camera sensor + C++20 ONNX CUDA/TRT FP16!
  2. **Local WSL2 Ubuntu 24.04 + GTX 1650 (4 GB VRAM):** Full PX4 v1.16 + ROS 2 Jazzy + C++20 ORT FP16 closed-loop smoke verification (`512×512` `PIDNet-S`/`DDRNet-23-slim` uses `< 300 MB` VRAM).

## 2. Strict Scope Boundaries & Unassailable License / Technical Locks
- **ZERO-SHAREALIKE, ZERO-NC, DEFENSE-SAFE LICENSE LOCK (100% Clean Apache-2.0 Published Weights!):**
  - **ALLOWED TEACHERS & TRAINING DATA (Purely Permissive ONLY):**
    - `DINOv2` (`DINOv2-ViT-L/14` / `ViT-B/14`) — **Apache 2.0**.
    - `Depth Anything V2 Small (ViT-S)` — **Apache 2.0**.
    - Author-generated **Colab-EGL Gazebo Harmonic Istanbul Dataset** (8,000 frames) — **100% Author-Owned Apache 2.0**.
    - **Purely Permissive Real & Synthetic Urban Aerial Datasets (`MIT` / `Apache-2.0` / `CC-BY-4.0` / `CDLA-Permissive` ONLY):** Verify exact licenses for **`OpenEarthMap`**, **`RescueNet` (official CDLA-Permissive release vs NC mirror)**, **`DroneDeploy`**, **`SynDrone` / `FlyAwareV2` (verify data license vs Apache-2.0 code license)**, and **`LoveDA`**.
  - **STRICTLY RESTRICTED TO ZERO-SHOT EVAL-ONLY (NEVER Baked Into Published Weights!):**
    - **`AeroScapes`** (`CC BY-SA 4.0` — ShareAlike copyleft risk for commercial/defense weight reuse → **EVAL-ONLY!**).
    - **`ICG Semantic Drone Dataset`** (Non-Commercial → **EVAL-ONLY!**).
    - **`UAVScenes`** (ICCV 2025, `CC BY-NC-SA 4.0` → **EVAL-ONLY** for real LiDAR depth/slope ablation!).
  - **STRICTLY BANNED ENTIRELY:**
    - **`DINOv3`** (Meta custom license with military/defense exclusions).
    - **`Depth Anything V2 ViT-B / ViT-L`** (`CC-BY-NC-4.0`).
    - **`Mid-Air`** (100% rural forest + `CC BY-NC-SA 4.0`) and **`UAVid`** (`CC BY-NC-SA 4.0`).
- **HARD TECHNICAL BANS:** NO MC-Dropout ($O(N)$ latency), NO Dense Evidential Deep Learning (EDL / "Mirage of EDL"), NO monolithic 4-dataset model, NO urban crane claims, NO cramming 1–2 px thin wires (`TTPLA`) into `512×512` nadir SLZ (only define a decoupled `IForwardHazardProvider` V2.1 stub), NO online POMDP tree search or RL flight control.
- **STRICTLY OUT OF SCOPE:** PX4 flight control, INDI/WLS, HJ-CBVF-QP, 3D ESDF (`DR-1`, `DR-2`); EASA / ARP4761A / GSN prose (`DR-3`); UAM incident study (`DR-4`); photorealistic Cesium/Blender cinema (`DR-5`); Turkish LLM/Z3 co-pilot (`DR-6`).

## 3. Specific Research Questions & Required Engineering Deliverables

### Section A — Verified Purely-Permissive Dataset Manifest & Canonical 5-Class SLZ Taxonomy
1. Audit exact 2025–2026 licenses and direct download sources for purely permissive (`MIT`/`Apache-2.0`/`CC-BY-4.0`/`CDLA-Permissive`) real & synthetic urban aerial segmentation/depth datasets (`OpenEarthMap`, `RescueNet`, `DroneDeploy`, `SynDrone`/`FlyAwareV2`, `LoveDA`, `FloodNet`), confirming a **100% zero-ShareAlike, zero-NC training manifest** alongside our 3 zero-shot **EVAL-ONLY** benchmarks (`AeroScapes`, `ICG`, `UAVScenes`).
2. Define the exact **Canonical 5-Class SLZ Semantic Taxonomy** mapping table across every selected train & eval dataset (`Safe_Flat_Ground`, `Safe_Flat_Roof`, `Hazard_Obstacle_Structure`, `Hazard_Dynamic_Human_Vehicle`, `Hazard_Water_Vegetation`, plus `ignore_index = 255`).

### Section B — Colab G4 96 GB Foundation-to-Real-Time Student Distillation & C++20 ONNX/TRT Export
1. Rigorously compare **`PIDNet-S`** vs **`DDRNet-23-slim`** (and `SegFormer-B0` baseline) at **`512×512`** on aerial boundary sharpness, ONNX opset 17+ / TensorRT FP16 graph compatibility, verified VRAM (`< 300 MB` FP16 on GTX 1650 & high FPS on RTX Blackwell / Orin-class reference), and auxiliary flatness/normal head support. **Pick ONE winner.**
2. Design the exact **Colab G4 96 GB Teacher-to-Student Distillation Pipeline**:
   - Frozen **`DINOv2-ViT-L/14`** (Apache 2.0) dense patch embeddings projected into student penultimate features (`L_feat_dino`),
   - Frozen **`Depth Anything V2 Small`** (Apache 2.0) relative depth / surface normals (`L_geom_dav2`) supervising real RGB frames lacking depth GT,
   - Combined loss `L_total = L_ce_lovasz + lambda_bdry * L_bdry + lambda_dino * L_feat_dino + lambda_geom * L_geom_dav2 + lambda_ood * L_energy_margin` and exact Real vs Gazebo-Istanbul batch sampling ratio preventing sim-to-real collapse.
3. Specify exact **PyTorch → ONNX → C++20 ONNX Runtime (`CUDAExecutionProvider` & `TensorrtExecutionProvider` FP16)** export, static `1x3x512x512` pre-allocated buffer pipeline, and TRT engine caching (explicitly tagging hardware for all FPS metrics).

### Section C — Temperature-Calibrated Single-Pass Free-Energy OOD & Dual-Mode 3D Geometric Fusion
1. Specify math and calibration protocol for **Single-Pass Free-Energy OOD** ($E(x) = -T \log \sum_c \exp(f_c(x)/T)$, compared vs `MaxLogit`), including temperature `T`, threshold `tau_ood`, and lightweight synthetic Outlier Exposure (unknown rooftop props/tarps/glare).
2. Design the **Dual-Mode 3D Geometric Flatness & Clearance Pipeline** in pure C++20:
   - **Mode A (Primary Live SITL / RGB-D):** Back-project metric depth + pose → fast deterministic SVD/PCA plane fit per `0.5 m` cell yielding slope $\theta \le 8^\circ$, roughness $\sigma_z \le 0.15\text{ m}$, and circular rotor clearance **$R_{\text{SLZ}} \ge 5.0\text{ m}$** (sized for `VertiRTA-X8-240`).
   - **Mode B (RGB-Only Monocular Fallback / Ablation):** `Depth Anything V2 Small` (Apache 2.0) with **Metric3D-style canonical camera normalization** + **barometer/rangefinder AGL scale anchor** (`scale = z_agl / median_center_depth`) quantifying real monocular slope error vs RGB-D ground truth on `UAVScenes` (eval-only).
3. Derive exact update equations for the **2.5D Bayesian SLZ Belief Grid** (`0.5 m` cell, `80 m × 80 m` local ring buffer): log-odds semantic safety fusion, inverse-variance OOD uncertainty propagation (`sigma^2_cell`), exponential temporal decay for dynamic hazards, and circular `R_SLZ = 5.0 m` pad aggregation.

### Section D — Deterministic ~150-LOC C++20 Active Information-Gathering Descent (Active Inspection Orbit)
1. Formulate a transparent, deterministic C++20 **Active Inspection Orbit State Machine** (~150 LOC, zero POMDP bloat) bridging the 2.5D Bayesian SLZ Grid → `ISlzPriorProvider`:
   - **Transitions:** `CANDIDATE_EVAL` → `{COMMIT_TOUCHDOWN | ACTIVE_INSPECT_ORBIT | REJECT_DIVERT_NEXT}`.
   - **Commit Rule:** `P_safe(pad) >= 0.85` AND `theta_max <= 8 deg` AND `sigma_z <= 0.15 m` AND `sigma^2_pad < tau_commit`.
   - **Active Inspection Trigger:** Candidate passes geometric/semantic screening with high epistemic/OOD variance (`sigma^2_pad >= tau_commit`) **AND** energy safety gate holds: `SoC_current > SoC_reserve + Delta_SoC_orbit(fault_mode, wind)`.
   - **Next-Best-View Orbit Geometry:** Descent altitude (`z_inspect = 20–25 m` AGL), offset radius (`r_inspect = 12–18 m`), `120°–180°` arc keeping sensor centered on TLOF centroid, max inspection budget (`<= 12 s`) to collapse Bayesian variance before final commit/reject.
2. Define the exact C++20 **`ISlzPriorProvider`** header interface consumed by `libvertirta` (`DR-2`) and the Co-Pilot (`DR-6`).

### Section E — Sensor-Only Colab G4 Gazebo Harmonic EGL Pipeline & Dual-Domain Evaluation Protocol
1. Provide a battle-tested, lightweight setup blueprint for running **sensor-only Gazebo Harmonic (`gz sim -s -r --headless-rendering`, OGRE2 + NVIDIA EGL, ZERO PX4/ROS 2 required in Colab!)** inside **Google Colab G4**:
   - Exact env vars (`__NV_PRIME_RENDER_OFFLOAD=1`, `__GLX_VENDOR_LIBRARY_NAME=nvidia`, `__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`, `DISPLAY=""`), `eglinfo` check, and fast `< 2 min` Drive-cached `gz-harmonic` restore.
   - Clean **C++20 `gz-transport` bridge** letting `libvertirta`'s standalone 6-DOF coaxial plant step camera poses inside Gazebo EGL and ingest live RGB-D frames into C++20 ONNX CUDA/TRT FP16 for the `N >= 60` closed-loop ablation without needing Ubuntu 22.04 PX4/ROS 2!
2. Design the **8,000-Frame Automated Maltepe–Dragos–Kartal Domain-Randomized Data Generator** (Copernicus DEM + OSM buildings + coastal rooftops/helipads/cars/pedestrians/clutter, randomized sun angle, haze/fog, altitude 15–120 m, pitch 0°–35°, sensor noise/blur → PNG RGB, float32 metric depth, normals, 5-class masks).
3. Design the **Honest Dual-Domain Benchmark & 3-Way Closed-Loop Ablation Protocol**:
   - **Table 1 (Zero-Shot Real-World Eval):** mIoU, SLZ F1, **False-Safe Rate**, OOD AUROC & FPR@95 on held-out zero-shot **`AeroScapes`** + **`ICG Semantic Drone`** + `Depth Anything V2 Small` baro-anchored slope error on **`UAVScenes`**.
   - **Table 2 (Live Colab-G4 6-DOF + Gazebo EGL 3-Way Closed-Loop Ablation, `N >= 60` randomized rooftop hazard trials + Local WSL2 PX4+ROS 2+ORT Smoke Check):** Compare **(1) Blind DEM-Only SLZ** vs **(2) Passive Single-Pass Vision SLZ** vs **(3) Active Uncertainty-Orbit SLZ** under injected dynamic rooftop pedestrians, OOD debris, and harsh shadow/fog — reporting Safe Touchdown %, False-Commit %, Active Orbit Trigger %, Extra Energy Cost (% SoC), and hardware-tagged C++20 ORT FP16 Latency (Colab RTX Blackwell + Local GTX 1650).

## 4. Required Output Format
Provide exact license audit tables, class-remapping JSON, mathematical loss & Bayesian update equations, C++20 `ISlzPriorProvider` & Active Orbit state machine, verified sensor-only Colab-EGL + `gz-transport` launch commands & SDF sensor blocks, training hyperparameter tables, and verified 2024–2026 references.
```
