# VertiRTA — Technical Interview & Whiteboard Architecture Defense Manual (TR / EN)

> **Audience:** Principal Robotics, Flight Control, Avionics & Autonomy Interviewers (Cezeri, Baykar, TUSAŞ, ASELSAN, HAVELSAN, Skydio, Archer, Joby, Volocopter).
> **Evidence Baseline:** `v1.0.0` (`ef9be75`), `v2.0.0-alpha`, and `v2.0.0` verified on `main`. Every quantitative claim below maps 1:1 to a checked-in artifact or test.

---

## 0. 30-Second Elevator Pitch (TR / EN)

- **EN:** *VertiRTA is an Apache-2.0 Vertical Run-Time Assurance and Fault-Tolerant Flight Control stack for a 240 kg coaxial X8 urban eVTOL operating in the Maltepe–Dragos–Kartal Istanbul coastal corridor bounded by real DHMİ `LTFJ`/`LTBX` CTR walls. It enforces an `ASTM F3269-21` 3-tier Simplex boundary (`CEL > CSFL > CBVF > CF`) coupling a `28,886 B` bare-metal Cortex-M7 C++17 flight core (`400 Hz`, zero heap, zero recursion) to a `100 Hz` C++20 Hamilton-Jacobi CBVF safety filter, an advisory Z3-verified TR/EN conformal co-pilot (`DAL E`), and a zero-ShareAlike 2.5D Bayesian active visual SLZ stack (`1.5625 MiB` grid).*
- **TR:** *VertiRTA, İstanbul Maltepe–Dragos–Kartal kıyı koridorunda (`LTFJ`/`LTBX` CTR duvarlarıyla sınırlı) görev yapan 240 kg eş eksenli X8 eVTOL için geliştirilmiş %100 açık kaynak (`Apache-2.0`) deterministik uçuş kontrol ve Çalışma Zamanı Güvence (RTA) mimarisidir. `28.886 B` Cortex-M7 Flash'a sığan sıfır-heap, sıfır-özyinelemeli `400 Hz` C++17 uçuş çekirdeğini, `100 Hz` C++20 Hamilton-Jacobi CBVF güvenlik filtresi, Z3-doğrulamalı TR/EN yardımcı pilot (`DAL E`, sıfır eyleyici yetkisi) ve sıfır-ShareAlike aktif görsel SLZ algılama katmanıyla deterministik `ASTM F3269-21` yalıtımı altında birleştirir.*

---

## 1. Pillar 1 — Bare-Metal Flight Core (`400 Hz` C++17) & HJI-CBVF RTA (`100 Hz` C++20)

### 1.1 Why `float64` QR/Cholesky WLS Instead of `float32` Normal Equations?
- **Physics & Numerical Trap (`DR-1` #1):** Forming normal equations $H = A^\top W A$ for 8×4 Weighted Least Squares (WLS) control allocation squares the condition number: $\kappa(H_{64}) = 6.385 \times 10^8$. In IEEE-754 `float32`, rounding destroys positive-definiteness ($\lambda_{\min}(H_{32}) = -8.545 \times 10^{-4} < 0$), causing Cholesky factorization to fail on **every single 400 Hz tick**.
- **Defense / Fix:** Exploit STM32H753's hardware double-precision FPU (`FPv5-D16`) for `wls_alloc_8x4_f64` inside `IndiWlsController::step_400hz`, and eliminate mutual recursion in upstream `DAQP v0.10.0` (`auxiliary.c`) via CMake-generated bounded iterative `daqp_auxiliary_norec.c` + non-copyable/non-movable `StaticDaqpSolver`.
- **Verified Bare-Metal Proof (`arm-none-eabi-g++ 14.2.1 -Os`, `artifacts/v1.0.0/benchmarks/tier1_structural.json`):**
  - Flash (`.text + .data`): **`28,886 B` (`28.21 KiB`)** across 11 objects (budget `< 262,144 B` / `256 KiB`).
  - Library Static RAM (`.data + .bss`): **`0 B`** (caller owns static `.bss` arenas; budget `< 131,072 B`).
  - Max Static Stack Frame (`-fstack-usage`): **`1,496 B`** (`<= 4,096 B`, 77 functions, 0 dynamic frames).
  - Max Call-Graph DAG (`-fcallgraph-info=su,da`): **`2,960 B`** (`<= 16,384 B`, 98 nodes, **0 recursion cycles**; worst chain `IndiWlsController::step_400hz -> wls_alloc_8x4_f64`).
  - Zero-Heap Audit (`arm-none-eabi-nm` + `-Wl,--wrap=malloc/calloc/realloc`): **`0` forbidden heap symbols**.
  - `ASTM F3269-21` Isolation (`scripts/ci/check_arch_isolation.py`): **`37` files scanned, `0` violations**.

### 1.2 Coaxial Aerodynamics, Voltage-Headroom Winding & Option B1b Active Flat-Spin (CSFL vs. CEL)
- **Coaxial Physics (`config/vehicle_x8_240.yaml`):** `m = 240.0 kg`, `W = 2,353.6 N`, nominal `T/W = 2.5` (`5,886 N`), 8 three-blade rotors (`R = 0.75 m`, `arm = 1.35 m`, coaxial lower-rotor hover efficiency `eta_coax = 0.85`, swirl recovery `0.96`). Upper-rotor loss unblanks the lower rotor (`+17.6%` thrust recovery); lower-rotor loss does **not** give wake relief to the upper rotor.
- **Battery Mass-Closure & Voltage-Headroom Lock (`112S12P`, `403.2 V`, `21.77 kWh`, `68 kg @ 320 Wh/kg`):** Under 1-RC Thevenin sag (`R_0 = 0.1363 Ohm`, `R_1 = 0.0840 Ohm`, `C_1 = 300 F`), a high-$K_e$ winding collapses thrust quadratically near 360 V. Locking $K_{e,\text{bus}} \le 1.36\text{ V}\cdot\text{s/rad}$ with a software FOC cap (`240 rad/s`) preserves 2-arm $T/W = 1.25$ down to `358.4 V` and $T/W \ge 1.12$ down to `338.0 V`.
- **Same-Arm Dual-Rotor Loss (`2M-SameArm`) — Why Option B1b Beats Option A & Option B2:**
  - *Option A (non-spinning 2-arm hover):* Rank-2/4 roll/pitch uncontrollable -> **unilateral capsize (`0.00%` survival across `1,500` runs)**.
  - *Option B2 (tilted 3-arm spin, 48–70 RPM):* Fails on `23–32°` actuator lag and `15–20°` touchdown rollover.
  - *Option B1b (Active Flat-Spin CEL, irreversible latch `<= 1 tick / 2.5 ms`):* Abandons yaw control to spin flat at $|\Omega_z| = 2.2\text{ rad/s}$ (`21 RPM`), applying actuator-lag phase lead $\Delta\psi = +\arctan(\Omega_z \tau_{\text{up}}) = +10.6^\circ$ (with coast-down lift matching and collective unload during spin-up), landing level at $|v_z| \le 0.8\text{ m/s}$ -> **`100.00%` (`1,500 / 1,500`) CEL survival** (95% Clopper-Pearson upper fail bound `0.001995`).
  - **Honest Regulatory Framing:** Same-arm coaxial dual-loss is **explicitly exempted from passenger-grade CSFL** and framed strictly as last-resort **Controlled Emergency Landing (CEL)** survivability.

### 1.3 4D HJI-CBVF (`VRTALUT1`) vs. Hand-Tuned HOCBF (`+51.97 pp` CSFL Gain)
- **Why Naive HOCBF Fails Under Saturation & Wind:** Analytic High-Order CBFs assume unbounded control authority and ignore actuator limits + wind disturbance $d_{ar}$. When a cross-arm dual motor fails (`2M-Cross`), HOCBF commands infeasible deceleration and breaches obstacles/CTR boundaries (**`44.85%` CSFL** across `n = 6,500`, 95% CP upper fail bound `0.561743`).
- **VertiRTA HJI-CBVF-QP Architecture (`100 Hz`, `libvertirta_rta_companion`):**
  - Offline JAX Hamilton-Jacobi-Isaacs PDE precomputes a 4D Control Barrier Value Function LUT (`VRTALUT1`, pre-faulted `mmap` avoiding page-fault tail spikes) + 2D CEL braking LUT (`VRTACEL2`).
  - Runtime QP (`StaticDaqpSolver`) enforces $\dot{V}(x, u) \ge -\gamma V(x)$ with `u*_HJ` optimal-escape cone fallback when the QP saturates, `+1.5 m` clutter inflation, overhead intruder `s_z` sign fix (so climbing into an overhead intruder never reads as escaping), and an explicit **tall-wall `z_eff < -5.0 m` horizontal radial braking guard** (so `762 m` `LTFJ` / `609.6 m` `LTBX` CTR walls never hallucinate climb-over escapes).
  - **Live 30,000-Simulation A/B Atlas (`artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json`, `seed = 20261122`, `77.32 s`):**
    - `HJ-CBVF + INDI-WLS`: **CSFL `96.82%`** (`6,293 / 6,500`, `k = 207`, CP95 upper `0.035665`), **CEL `100.00%`** (`1,500 / 1,500`, `k = 0`, CP95 upper `0.001995`).
    - `HOCBF + INDI-WLS`: **CSFL `44.85%`** (CP95 upper `0.561743`), CEL `100.00%`.
    - `HJ-CBVF + Stock CA`: CSFL `96.82%`, **CEL `0.00%`** (`0 / 1,500` capsize).
    - **Zero Cherry-Picking:** Every single one of the **`244 / 10,000`** failed runs (`207` in extreme `S3` cross-arm dual + `37` in `S6` out-of-ODD stress) is published with seed, state, invariant, and root cause in `artifacts/v1.0.0/tier1_atlas/failure_cliff_ledger.json`.
  - **Per-Tick Latency (WSL2 x86_64 Release, `100 Hz` HJI-CBVF-QP + `400 Hz` INDI-WLS vs `10,000 us` budget):** Tier 3 `CLOCK_THREAD_CPUTIME_ID` (`100k` ticks) `p50 = 1.2400 us`, `p99 = 1.9000 us`, `p99.9 = 11.5690 us`; Tier 4 `CLOCK_MONOTONIC_RAW` (`50k` ticks) `p50 = 0.5400 us`, `p99 = 0.9500 us`, `p99.9 = 1.5900 us`, `steal = 0.0%` (`CLEAN`).
- **Dual-Res `3.094 MiB` Robocentric ESDF + 1D `WireCapsules`:** `0.5 m` Occupancy (`1 MiB`) + `1.0 m` `16 B` bit-packed FIESTA ESDF (`2 MiB` + `96 KiB` queues). Thin powerlines (`5–20 mm`) are **never** rasterized into coarse voxels (which wastes RAM or bloats corridors); they enter analytically as 1D line-segment capsules (`WireCapsuleTable`, `K <= 16` QP constraints).

---

## 2. Pillar 2 — Verified TR/EN Neuro-Symbolic Contingency Co-Pilot (`v2.0.0-alpha`, `DAL E`)

### 2.1 Zero-Heap Online STL Monitor & `FixedMonotonicWedge<601>` Wrap-Around Fix
- **The Subtle Ring-Buffer Bug (`DR-6` #4):** In a sliding-window monotonic deque of capacity `Cap = 601` tracking a `600`-tick horizon (`span = 600`), pushing a new sample at `tau = 601` *before* evicting the expired `head_` overwrites `head_` in-place and emits **false-safe robustness**.
- **Defense / Fix (`include/vertirta/stl/stl_monitor.hpp`):** Always evict expired `head_` (`tau - front.tau > span`) **before** pushing `tail_`, fail closed on `NaN` (`-inf`), and implement `OnlineBoundedUntil<601>` with instant RoSI violation — verified `0.00` error vs. textbook oracle over 200 traces and `0` heap allocations under `ScopedZeroHeapGuard` + `-Wl,--wrap=malloc`.

### 2.2 Four Critical `QF_LRA` Z3 Verifier Fixes & Deterministic UNSAT Repair (`DR-6` #6)
1. **Unique Dynamic NFZ Trackers:** Label each dynamic NFZ `c_dyn_nfz_0..3` uniquely so 2+ active NFZs never overwrite each other's UNSAT-core assumption literal.
2. **Full 3D Vertical Slab Check:** Check `ownship_u`, `dest_u`, `cmd_agl`, AND `floor_amsl_m` (never 2D-only projection!).
3. **AMS Dynamic Capability Enforcement:** Bind `c_ams_speed`, `c_ams_climb`, and `c_ams_accel` to the live degraded flight envelope (`400 V / 22 kWh`, `v_term_volts` ref `368.5 V` — never hobby `44.2 V`!).
4. **Action-Conditioned Surface Checks:** Skip `c_slz_unsafe` / `c_slz_ood` when action is `HOLD_LOITER` (loitering in air does not touch down).
- **Real Microsoft Z3 Cross-Check:** `tests/copilot/test_z3_smt2_crosscheck.py` parses all 10 emitted `(set-logic QF_LRA)` + `(check-sat-assuming (...))` SMT-LIB2 formulas in real `z3-solver`, matching C++20 `verify_mission_qflra` 100% on SAT/UNSAT and every violated core label. An 8-priority 1-pass repair deterministically repairs `Pendik_Marina` (`LTFJ` CTR violation) -> `Kartal_Sahil` in `< 15 ms`.

### 2.3 `KnowNo` Split Conformal Prediction (`q_lvl = 0.955`) + `< 5.0%` SoC Liveness Override
- **Formula:** Given $n = 200$ calibration nonconformity scores and target miscoverage $\alpha = 0.05$, finite-sample split conformal computes quantile level $q_{\text{lvl}} = \frac{\lceil (n + 1)(1 - \alpha) \rceil}{n} = \frac{191}{200} = 0.955$, guaranteeing $\mathbb{P}(y^* \in \mathcal{C}(x)) \ge 1 - \alpha$ under exchangeability.
- **Decision Regimes:** $|\mathcal{C}| = 1 \to$ `EXECUTE_SINGLE`, $|\mathcal{C}| > 1 \to$ `ACTIVE_ELICITATION` (asks focused TR/EN clarification question), $|\mathcal{C}| = 0 \to$ `ABSTAIN_LOITER`.
- **Safety Liveness Override:** If loiter energy margin above the reserve drops below 5% (`soc_pct - soc_reserve_floor_pct < 5.0%`, i.e., `< 17.0%` total SoC above the `12.0%` reserve floor), indefinite clarification/loiter is fatal -> deterministic `LIVENESS_EMERGENCY_COMMIT` hands immediately to RTA's nearest Z3-verified reachable SLZ.
- **Compact `<= 30`-Token GBNF & Strict Isolation:** `vertirta_mission_node` runs the LLM on an async `std::jthread` behind a lock-free `LockFreeSnapshotBuffer` (TSan verified `0` data races); it has **zero actuation write path** into `flight_core` (`DAL E` advisory only).

---

## 3. Pillar 3 — Active Visual SLZ Perception (`v2.0.0`, Zero-ShareAlike `Apache-2.0`)

### 3.1 Zero-ShareAlike License Firewall & Static-PAPPM `PIDNet-S`
- **Why Competitors' Datasets Are Legally Toxic:** `RescueNet` (`CC BY-NC-ND`), `LoveDA` (`CC BY-NC-SA`/GE-ToS), `FlyAwareV2` (`GPL-3.0`), `DINOv3` (ITAR/military ban), `SegFormer` (NVIDIA NC), and unlisted `OpenEarthMap` regions (`CC BY-NC-SA`) contaminate Apache-2.0/defense weights.
- **VertiRTA's Clean 5-Source Gate (`ml/slz_perception/data/license_gate.py`):** Programmatically locks training to `FLAIR-1` (`etalab-2.0`) + `OpenEarthMap` 41/100 whitelist (`CC BY 4.0` + `DL-DE-BY-2.0` ONLY) + `FloodNet-Supervised v1.0` (`CDLA-Permissive-1.0`) + `SkyScenes` (`MIT`, with depth-normal wall filter $|n_z| < 0.85 \to \text{Obstacle}$) + procedural `Gazebo_Istanbul_8K` (Copernicus GLO-30 blocks, OSM `ODbL` strictly isolated). Eval-only sets (`AeroScapes`, `ICG`, `UAVScenes`) are hard-blocked from `split="train"`.
- **Single-Pass Free-Energy OOD & Static-PAPPM:** Replaces `AdaptiveAvgPool2d` with fixed `AvgPool2d` + global mean so ONNX Opset 17 exports a single static `1x3x512x512` subgraph (verified by `onnx.checker.check_model`, zero `dino_proj` nodes). Bans MC-Dropout ($O(N)$ latency) and Dense EDL (epistemic mirage) in favor of single-pass boundary-compensated Helmholtz Free Energy:
  $$E(u) = -T \log \sum_{c=0}^{4} \exp\!\left(\frac{f_c(u)}{T}\right), \qquad \tilde{E}(u) = E(u) - \kappa_b\,b_{\text{up}}(u) \quad (\kappa_b = 1.5, \; 64\times 64 \xrightarrow{\text{bilinear}} 512\times 512)$$

### 3.2 Cell-Local `1.5625 MiB` 2.5D Bayesian Grid & OOD Variance Floor (`DR-7` #2 & #4)
- **Memory Lock:** `160 × 160` robocentric ring buffer at `0.5 m` resolution (`80 m × 80 m`) with `alignas(64) SlzCell` (`sizeof == 64 B`) = **`1,638,400 B` (`1.5625 MiB`)**, zero heap allocations in hot loop (`ScopedZeroHeapGuard` + `-Wl,--wrap=malloc`). Rotor-clearance disk $R = 5.0\text{ m}$ contains exactly **`317` cells** (`CountCircularDiskCells(5.0f, 0.5f)`).
- **Eliminating `float32` Catastrophic Cancellation:** Global-ENU $\Sigma = \frac{1}{N}M - \bar{p}\bar{p}^\top$ at Maltepe `(-3500 m, +3200 m)` loses 7 digits of mantissa and yields `NaN`/negative eigenvalues. Storing **cell-local shifted moments** $\delta p = p_W - c_{ij}$ (`|dx|, |dy| <= 0.25 m`) and merging `3×3` windows in `double` via the parallel-axis theorem yields `0.00011 m` (`0.11 mm`) zero-noise floor at `(-3500, +3200, 15.0)` and `6.0° ± 0.35°` on noisy sloped roofs. Cells with `< 12` points stay conservatively invalid (`theta > 80°`).
- **Solving OOD Precision Collapse:** Purely additive inverse-variance precision $\Lambda_k$ cannot raise $\sigma^2_{\text{cell}}$ when a new OOD tarp appears on a previously clean roof (40 clean frames) and still collapses `< 0.08` over repeated OOD frames. `SlzBeliefGrid` enforces an explicit **OOD variance floor (`sigma2_floor >= 0.12`) + $\Lambda_k$ cap/reset** whenever $\tilde{E} > \tau_{\text{ood}}$, holding $\sigma^2_{\text{pad}} \ge 0.08$ across all 25+ repeated OOD frames.

### 3.3 2-Phase Active Inspection Orbit (`22 kWh`) & Harmonized 11-Bit `SlzReasonBits`
- **Why 2 Phases (`APPROACH_ORBIT -> ACTIVE_INSPECT_ORBIT`):** Arming the `12.0 s` orbit timer at `80 m` cruise altitude guarantees timeout during the `15–19 s` descent. Phase 1 absorbs descent and arms `orbit_start_ns_` strictly on arrival (`|agl - 22.0| <= 2.5 m`, `|r - 15.0| <= 3.0 m`), uses guarded `now_ns > start_ns` anti-underflow arithmetic, tilts the 2-axis gimbal to $\arctan(22 / 15) = 55.7^\circ$ (preventing `80°` HFOV far-edge pad clipping), and gates energy on the **locked `22 kWh`, `R = 0.75 m` pack (`~40.5 kW`, `kBaseSoc12s = 0.614%` × `1.25` safety = `0.768%`, NEVER `13.2 kWh / 1.36%`)**.
- **Mod B Monocular Veto & `ISlzPriorProvider`:** Affine monocular anchoring uses full roll+pitch slant $z_{\text{AGL}} / (-d_{W,z})$, adds altitude variance floor $0.04 + (z_{\text{AGL}}/60)^2 \cdot 0.25$, and vetoes `COMMIT_TOUCHDOWN` above `30 m` AGL (`REASON_MONO_ALT_VETO`). All 11 `SlzReasonBits` (`bits 0..10`) are bit-exact across `slz_prior_provider.hpp`, `core/slz_bayes/`, and `SlzPrior.msg`, with `250 ms` staleness fail-closed guarding.
- **Local 60-Seed Synthetic 6-DOF Smoke (`active_vision_ablation_runner`):** `active_orbit` `60/60` (`100%` safe, `0%` false commit, max orbit cost `0.5944% <= 1.0%`, `step_p99 = 0.2491 ms`) vs. `passive_vision` (`63.33%` safe, `36.67%` false commit) vs. `blind_dem` (`33.33%` safe, `66.67%` false commit). Real Colab EGL 60-seed & zero-shot accuracy metrics remain honestly marked `PENDING_COLAB_G4`.

---

## 4. Tough Principal-Engineer Questions & Honest Answers

| Tough Question | Crisp, Defensible Answer |
| --- | --- |
| *"Does your 10,000-run Monte Carlo prove EASA `10^-9 / FH`?"* | **No — never.** Proving $\le 10^{-9}/\text{FH}$ at 95% confidence with zero failures requires `2,995,732,273` independent flight hours (`math.log1p(-1e-9)`). Our atlas strictly bounds *conditional* simulated recovery inside `verification/ODD.yaml` and keeps CSFL (`n=6,500`) vs. CEL (`n=1,500`) pools separate. |
| *"Is your `p99 = 1.90 us` Linux timing a DO-178C WCET proof?"* | **No.** Linux user-space timings (including cloud ARM64 Cobalt 100) are CI performance regression telemetry with hypervisor/OS jitter. Our hardware-independent guarantees are Tier 1 (`28,886 B` Flash, `0 B` RAM, `2,960 B` DAG, `0` recursion, `0` heap), Tier 2 (0-ULP `.vrta` replay), and planned `v3.0` real STM32H7 silicon `DWT->CYCCNT` HITL + `ESBMC` BMC. |
| *"Can a `240 kg` passenger eVTOL fly under `SORA 2.5`?"* | **No.** `Reg. (EU) 2019/947` Art. 6 & `ED Dec 2025/018/R` strictly exclude passenger carriage from `SORA 2.5`. We separate **Stage-A** (`25 kg` unmanned subscale `SORA 2.5`, 0 pax) from **Stage-B** (`240 kg` simulated `SC-VTOL-02` / `Part-IAM` + Turkish `SHGM SHT-İHA` `M3` + `İHATTYS` geofencing design alignment). |
| *"Can the LLM Co-Pilot or neural SLZ model crash the aircraft?"* | **Architecturally blocked (`ASTM F3269-21`).** Both run in isolated ROS 2 nodes (`DAL E` advisory snapshots, `0` actuation write path). `libvertirta_flight_core` enforces the hard priority arbiter (`CEL > CSFL > CBVF > CF`), `250 ms` staleness failover, and `check_arch_isolation.py` programmatically bans any AI/ROS/Python dependency inside `core/{flight_core,rta_companion}`. |
| *"Why didn't you test `PIDNet-S` on a real GPU locally?"* | We **did** benchmark the exact exported static-PAPPM ONNX Opset-17 architecture (FP32 & FP16) on physical local **`NVIDIA GTX 1650 SUPER (4 GB, TU116)`** silicon (`artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json`), explicitly tagged `ARCHITECTURE_LATENCY_VRAM_BENCHMARK_ONLY_UNTRAINED_WEIGHTS` so silicon speed/VRAM is never conflated with untrained accuracy. |

---

## 5. Local Physical Silicon Benchmark (`GTX 1650 SUPER`, Architecture-Only Untrained Weights)

Measured live via [`scripts/run_local_gtx1650s_onnx_bench.py`](../scripts/run_local_gtx1650s_onnx_bench.py) on physical **`NVIDIA GeForce GTX 1650 SUPER` (`4096 MiB`, Turing `TU116`, `sm_75`, driver `621.24`, ORT `1.23.2` CUDA 12 / cuDNN 9)** exporting the actual static-PAPPM `PIDNetSlzDeployWrapper` (`8,003,817` deploy params, `1x3x512x512`, Opset 17, `200` warmup + `500` timed iterations). Raw artifact: [`artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json`](../artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json) (`weights_status: ARCHITECTURE_LATENCY_VRAM_BENCHMARK_ONLY_UNTRAINED_WEIGHTS`).

| Execution Mode (`kSameAsRequested` Arena) | ONNX Size | Net `p50` | Net `p99` | Net `max` | Net Mean FPS | E2E Host I/O FPS | `cudaMemGetInfo` VRAM Delta | `< 12 ms` p99 / `< 300 MiB` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`ort_cuda_fp32_cuda_graph` (Recommended TU116)** | `30.505 MB` | **`8.0586 ms`** | **`9.0617 ms`** | **`9.3698 ms`** | **`122.96 FPS`** | **`89.21 FPS`** | **`126.0 MiB`** (`135.0` `nvidia-smi`) | **PASS / PASS** |
| `ort_cuda_fp32` (Eager) | `30.505 MB` | `8.9686 ms` | `15.0456 ms` | `21.5362 ms` | `108.65 FPS` | `93.07 FPS` | `132.0 MiB` (`128.0` `nvidia-smi`) | `FAIL` (WDDM jitter) / `PASS` |
| `ort_cuda_fp16` (Converted FP16) | `15.303 MB` | `28.6717 ms` | `31.1580 ms` | `41.7204 ms` | `34.81 FPS` | `32.87 FPS` | `92.0 MiB` (`80.0` `nvidia-smi`) | `FAIL` (`0.313x` vs FP32) / `PASS` |

> **Key Hardware Takeaway for Interviews:** Because `GTX 1650 SUPER (TU116)` lacks Tensor Cores, converted FP16 runs **`3.2×` slower (`0.313x` speedup)** than FP32 under ORT-CUDA, directly vindicating `DR-7`'s mandatory TU116 FP32-vs-FP16 regression test. Enabling **CUDA Graph** on FP32 cuts WDDM kernel-launch tail jitter by **40%** (`p99: 15.05 ms -> 9.06 ms`, `max: 9.37 ms`) while holding GPU VRAM to **`126.0 MiB`**. TensorRT FP16 remains optimal on Tensor-Core GPUs (`Colab L4/A100`, `Jetson Orin`). Accuracy metrics (`mIoU`, `FSR`, `AUROC`) remain **`PENDING_COLAB_G4`** until trained weights exist.
