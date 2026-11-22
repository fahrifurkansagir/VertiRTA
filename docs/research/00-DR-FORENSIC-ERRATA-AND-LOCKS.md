# VertiRTA — Deep Research Forensic Audit, Errata & Plan-Shaping Locks (`DR-1` .. `DR-7`)

- **Audit Date:** 2026-11-22
- **Status:** **`7 / 7` COMPLETE & FORENSICALLY AUDITED — Ready for `writing-plans` (Phase 1 Implementation Planning)**

---

## 1. File Inventory & Coverage Matrix

| Track | Canonical Repo File (`docs/research/`) | Source File (`Downloads\`) | Size | Status |
|---|---|---|---|---|
| **`DR-1`** | `DR-01-coaxial-physics-indi-wls-hw-transfer.md` | `qweqwe.md` | 95.4 KB | Complete (EN) — **4 Critical / 3 Moderate errata** |
| **`DR-2`** | `DR-02-jax-4d-cbvf-libvertirta-daqp-esdf.md` | `gemini-code-1795353430248.md` | 89.9 KB | Complete (TR) — **5 Critical / 2 Moderate errata** |
| **`DR-3`** | `DR-03-easa-scvtol-astm-f3269-gsn-arm64-ci.md` | `gemini-code-1795353351799.md` | 88.2 KB | Complete (TR) — **5 Critical / 4 Moderate errata** |
| **`DR-4`** | `DR-04-real-world-impact-ltfj-incidents-gap.md` | `gemini3.md` | 71.4 KB | Complete (EN, 0 newlines / flattened export) — **4 Critical errata** |
| **`DR-5`** | `DR-05-photorealistic-istanbul-earth-viz.md` | `gemini-code-1795353611878.md` | 76.1 KB | Complete (TR, clean headings) — **3 Critical errata** *(duplicate `gemini.md` skipped)* |
| **`DR-6`** | `DR-06-tr-en-z3-grpo-copilot-vertibench.md` | `gemini-code-1795352673080.md` | 92.6 KB | Complete (EN+TR) — **6 Critical / 2 Moderate errata** |
| **`DR-7`** | `DR-07-active-visual-slz-dinov2-colab-egl.md` | `gemini-code-1795353420552.md` | 70.4 KB | Complete (TR) — **4 Critical / 4 Moderate errata** |

---

## 2. Mandatory Design-Spec Upgrades Discovered by Forensic Audit

These verified 2025–2026 findings **override** earlier assumptions in `2026-11-22-vertirta-tier-s-design.md` and MUST be folded into `v1.2` before `writing-plans`:

1. **Dual-Res `ROG-Map` + FIESTA Memory Fix (`DR-2`):** `64×64×32 m @ 0.5 m` = `1,048,576` voxels (8× larger than `131k`). Lock **Dual-Res `3.094 MiB`**: `0.5 m` Occupancy (`1 MiB`) + `1.0 m` (`131k`) FIESTA ESDF (`2 MiB` + `96 KiB` queues), patching upstream `ROG-Map #15` uninitialized `safe_margin_i`.
2. **Gazebo Harmonic `gz_bridge` & `px4-ros2-interface-lib` Real Bugs (`DR-2`, `DR-4`):**
   - `VEHICLE_CMD_INJECT_FAILURE` bypasses Gazebo `gz_bridge` → inject IMU/GNSS faults at `gz_bridge`/Gazebo plugin or voter layer!
   - `deferFailsafesSync(true, 60)` does NOT block attitude-limit failsafe → set `FD_FAIL_R=0, FD_FAIL_P=0` for `SAME_ARM_DUAL_LOSS` CEL.
   - `px4-ros2-interface-lib` real upstream bugs **#201** (`waitForFMU` across PX4 restarts), **#167** (active mode reschedule crash), **#191** (multi-vehicle clash) → use `gz service` `reset: {model_only: true}` (keeps sim-clock monotonic for `UXRCE_DDS_SYNCT=1`) and harden worker isolation.
   - Gazebo Harmonic lacks CPU LiDAR → 50-run Full-Sensor subset uses 2 EGL `--headless-rendering` `gpu_lidar` workers (`10–13 min`).
3. **Exact `LTFJ` Polygon+Arc Airspace (`DR-4`, `DR-5`, `DR-6`):** AIP NW join point `J2` (`40.91279°N, 29.19212°E`) sits `5.395 NM` from `LTFJ` ARP and pulls the CTR wall **603 m west of `29.200°E`**. Encode exact **polygon + arc** (`LTFJ` `798.58 m HAE`, `LTBX` `646.35 m HAE` — verify exact `LTBX` ARP `40°59'56"N` vs `40°59'26"N` against DHMİ `AD 2.17` PDF before freezing `config/istanbul_ltfj_airspace.yaml`!).
4. **Blackwell Blender & Commercial Geo-License Lock (`DR-5`):**
   - Blender 4.2/4.3 lack NVIDIA Blackwell (`sm_120`) OptiX → lock **Blender 5.2 LTS** (driver `>= 575`, fallback **4.5 LTS** for `535–574`).
   - **EOX Sentinel-2 Cloudless** & **FABDEM** are `CC BY-NC-SA 4.0` → pull raw open AWS `s3://sentinel-cogs/sentinel-s2-l2a-cogs/` (`35TPF`) + Copernicus GLO-30 (`N40_00_E029_00`, `EGM2008`) with OSM/Overture flattening.
   - Add Earth curvature drop `-(E^2 + N^2)/(2 R_gauss)` (`+1.66 m` Dragos to `+8.82 m` corner) alongside `+36.7 m` EGM2008 geoid undulation!
5. **EASA Nov 2026 & SHGM Jul 2026 Grounding (`DR-3`, `DR-4`):**
   - Verified **`SC-VTOL MOC-5 Iss. 2 FINAL` (5 Nov 2026)**, **`ED Dec 2026/010/R` (16 Nov 2026)** retiring SORA 2.0 on 1 Jan 2027, **`EASA AI CP Proposed Iss. 03` (Jun 2026)**, and **SHGM `SHT-İHA` (eff. 30 Jul 2026)** mapping `VertiRTA-X8-240` to **M3 (>=150 kg)** + **İHATTYS** volumetric geofencing.
6. **Clean Zero-ShareAlike Vision Training Set (`DR-7` — Overrides `design.md`):**
   - Verified `RescueNet` (`CC BY-NC-ND 4.0`), `LoveDA` (`CC BY-NC-SA`), `FlyAwareV2` (`GPL-3.0`), `SegFormer` (NVIDIA NC), `DINOv3` (ITAR/military ban), and `OpenEarthMap` default regions (`CC BY-NC-SA`) are tainted.
   - **Lock clean 5-source train set:** `FLAIR-1` (`etalab-2.0`) + `OpenEarthMap` programmatic `41/100` permissive whitelist + `FloodNet-Supervised v1.0` (`CDLA-Permissive-1.0`) + `SkyScenes` (`MIT`) + Author 8K Gazebo-Istanbul (OSM `ODbL` isolated from training). Winner backbone: **`PIDNet-S` (MIT, ~189 MB FP16 VRAM)** swapping PAPPM `AdaptiveAvgPool2d` → static `AvgPool2d` for single-subgraph TRT FP16.
7. **Coaxial X8 Active Flat Spin (`Option B1b`), Voltage-Headroom Winding & 4-Part PX4 `GZBridge` Lock (`DR-1`):**
   - **Same-Arm Dual-Loss:** Non-spinning 2-arm (`Option A`) is rank-2/4 uncontrollable (capsizes); 3-arm tilted (`Option B2`, 48–70 RPM) fails on `23–32°` lag & `15–20°` rollover. **Lock Option B1b Flat Spin `|Ω_z| = 2.2 rad/s` (21 RPM, CCW Arm 0/1, CW Arm 2/3)** with phase lead `Δψ = atan(Ω_z * τ_m) = ∓10.6°`, level touchdown (`|v_z| <= 2.5` >10m, `<= 1.5` <5m, `<= 0.8` contact).
   - **BLDC Voltage-Headroom Lock (`K_e,bus <= 1.36 V·s/rad`) + `320 Wh/kg` (`68 kg`) 22 kWh Pack:** Eliminates quadratic 360 V sag collapse; holds `T/W = 1.25` to `358.4 V` & `>= 1.12` down to `338 V` with software FOC `240 rad/s` cap. Coaxial split (`η_coax = 0.85`): `795.4 N` upper + `676.1 N` lower (upper failure unblanks lower rotor `+17.6%`).
   - **400 Hz PX4-Internal `mc_indi_wls_control` + 4-Part `GZBridge` Patch:** Single `wq:rate_ctrl` module (`~40 µs`, `<4.8 KB` DTCM): 3-IMU MVS+`χ²_24` (`20 ms`) → `S²` reduced attitude → 8 Hz prewarped Butterworth INDI (`+G2`) → **QR active-set or `double` (`f64`) Cholesky WLS** (NEVER `float32` normal equations!) → dual-channel `<85 ms` Motor FDI. `GZBridge` patch: (a) `3× PX4Gyroscope/Accelerometer` distinct `DeviceId`s (inherits native `failure_injection::process()`), (b) `imu_sensor_0..2` + `air_pressure_sensor_0..1`, (c) `sleep_count_limit 1000→5000` for `gpu_lidar`, (d) `_failure_injection_sub` in `GZMixingInterfaceESC::updateOutputs()`. Exact 72 B `VertirtaCapabilityEnvelope.msg` (adding `float32 chi2_residual` + `uint32 _pad`).

---

## 3. DO-NOT-COPY Errata Register (31 Critical Bugs Caught Across `DR-1`..`DR-7`)

Implementation agents MUST NEVER blindly copy code/numbers from the raw DR files without applying these exact patches:

- **`DR-1` (4 Critical):**
  1. **`float32` WLS Normal-Equations Cholesky Crash:** `cond(H64) = 6.385e8`, `min_eig(H32) = -8.545e-4 < 0` → Cholesky fails on EVERY 400 Hz tick! Fix: **QR active-set on $A \in \mathbb{R}^{12 \times 8}$ (`cond = 2.5e4`) OR `double` (`f64`) Cholesky** (STM32H753 `FPv5-D16` hardware `f64`).
  2. **Broken `||` & `-nostdinc++` Violation in `wls_alloc_8x4`:** Fix split `||` (L350–352) and replace `std::clamp` (`<algorithm>`) with freestanding `vertirta::clamp_f` so bare-metal `-nostdinc++` NuttX build stays pure.
  3. **Airframe `12050_gz_vertirta_x8_240` Startup Contradiction:** Do not source `rc.mc_defaults` without overriding stock `mc_pos/att/rate/allocator` (in RTA mode); explicitly add **`FD_FAIL_R=0, FD_FAIL_P=0`** + `CA_ROTORi_CT`/`CA_FAILURE_MODE=1` (for Gate 2 Day 8–9 stock baseline).
  4. **240 kg MTOW vs 22 kWh Mass + Plugin/Msg Fixes:** Lock high-energy-density `320 Wh/kg` (`68 kg`) pack in `vehicle_x8_240.yaml`; add `float32 chi2_residual` + `uint32 _pad` to `VertirtaCapabilityEnvelope.msg` (true 72 B); fix `VertirtaCoaxialPropulsionSystem` sign (`v_iu + v_air.Z()` on climb), wire `tau_coast_`, apply `0.96*k_Q0_` to lower rotors, define `lidar_link` with `<max>60.0</max>`, use exact `LTFJ` ARP `40.89833°N` + NW `J2`, and verify `gz-sim` `navsat` elevation datum (`AMSL` vs `HAE`) so EKF2 doesn't double-add `+36.7 m`.

- **`DR-2` (5 Critical):**
  1. `_support_U` vs `partial_max_magnitudes` mismatch (`14.06` vs `6.87 m/s²`) breaks Lax-Friedrichs CFL — match both bounds.
  2. `smax_4(0,0)` is `+0.173 m` optimistic (subtract `ln(2)/β`); overhead intruder `|Δp_z|` flips sign without flipping `v_z` (climbing into overhead intruder reads as escaping!).
  3. `2M-Cross` under `d_ar=1.75 m/s²` wind needs `9.95 s / 109.5 m` braking (`> T=6.0 s, r_max=58 m`) — enforce degraded cruise speed cap `v_cap,f` inside BRT or expand grid/horizon.
  4. `MmapCbvf4DLut::eval` lacks `std::isfinite` and `regime_idx < 4` guards (NaN → UB/OOB segfault under LibFuzzer).
  5. Mangled `||` & markdown URLs in CMake/C++; pin exact `DAQP` `v0.10.0` API headers/symbols and remove `DAQP_NO_SOFT_WEIGHTS` clash; align Kartal datum (`40.8860°N`, not `40.8880°N`) and `σ_z <= 0.15 m` (not `0.35 m`).
- **`DR-3` (5 Critical):**
  1. `inline` global `operator new` in header is ill-formed ISO C++17 and poisons companion binaries — move to standalone link-test TU + `nm -u` (checking `_Znwj/_Znaj`/aligned `new`).
  2. Broken `||` newlines in `determinism.yml`; `-Wl,--wrap=malloc` breaks `vertirta_replay_cli` glibc I/O unless isolated; add `-fcallgraph-info` DAG + `-Werror -Wvla` to Cortex-M7 job.
  3. Keep 100 Hz HJI-CBVF-QP + 11.3 MB LUT in `libvertirta_rta_companion` (not `<256 KB` `flight_core`).
  4. Fix `ODD.yaml` `lon_deg` upper bound (`29.2100` is 925 m inside `LTFJ` CTR!) and add `EGM2008 +36.7 m`, `R_SLZ>=5.0 m`, `σ_z<=0.15 m`, `θ<=8°`, `age>250 ms`.
  5. Fix Triplex 2oo3 FTA math (`3*λ²*T = 3e-8/FH`, NOT `λ³*T² = 1e-12/FH` — 30,000× error!) and keep CSFL (`n=650`) vs CEL (`n=150`) Atlas pools strictly separate.
- **`DR-4` (4 Critical):**
  1. Entire `gemini3.md` has 0 newlines (flattened single line) — keep strictly as reference or reformat before quoting.
  2. Coaxial 1-motor OEI math wrongly assumes `2×` thrust (`588.4 N`, `2.43×` power) on faulted arm; WLS offloading cuts peak to `60.0%` (`441.5 N`, `1.84×`) / min-peak `53.3%` (`392.4 N`, `1.54×`), and wake relief only occurs on *upper*-rotor loss.
  3. Urban wind shear `U(z)=W_20*(z/6)^0.30` is physically inverted (produces urban winds 45% faster than open water!) — DO NOT use `U_urban` values in SITL.
  4. `SLZ-07` distance to `LTFJ` ignores NW corner `J2` (`1.635 km`, not `2.34 km`).
- **`DR-5` (3 Critical):**
  1. `gdalwarp` to `EPSG:32635` (UTM 35N) introduces `1.432°` meridian convergence (`115–265 m` horizontal drift!) + clips 360–390 m of BBOX + `-dstnodata 0.0` deletes Marmara Sea! Fix: warp into local Kartal CRS (`+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84`) with `-dstnodata -9999`.
  2. NEVER link `PROJ 9` / `throw` inside `libvertirta_flight_core` (`vertirta_geodesy.hpp`) — keep `constexpr` `<1 KB` bilinear geoid patch in flight core and `pyproj`/PROJ strictly in offline viz tools (asserting `20 < N0 < 50`).
  3. Fix `bpy` `ShaderNodeMix` RGBA sockets (`inputs[6]/[7]`, `outputs[2]`, NOT string `"A"/"B"/"Result"` which bind hidden FLOAT sockets!), fix mangled markdown links/`||`, and rename `[lon, lat, hae]` field cleanly.
- **`DR-6` (6 Critical):**
  1. Fix hobby `12S` (`44.2 V`) → locked `400 V / 22 kWh` HV pack (`~350–380 V`).
  2. Fix GRPO `r_z3` uppercase `"LTFJ_CTR"` vs lowercase `"c_ltfj_ctr"` (100% Bucket-3 zero reward!), fix `ast.target` on `HOLD_LOITER`/`REQUEST_CLARIFICATION`, and replace per-step `fork` pool with persistent `spawn` pool.
  3. Fix GSPO sequence-level clipping (`3e-4 / 4e-4` per `arXiv:2507.18071`, or use token-level `dr_grpo` with `0.20/0.28`).
  4. Fix `FixedMonotonicWedge` STL ring-buffer bug (evict `head_` BEFORE pushing `tail_`, or `Cap=602` — otherwise overwrites `head_` and emits false-safe!).
  5. Compact DSL to `<= 30` tokens under Qwen3 single-digit BPE (or decode `ACTION/TARGET/PARAMS` and synthesize STL deterministically) — otherwise `max_tokens=64` truncates every valid output (`~120 tok`) and blows Ryzen 5600X latency to `~7 s`.
  6. Fix `z3_corridor_verifier.cpp`: unique `"c_dyn_nfz_" + idx` tracker names, check BOTH destination pad Z & `cmd_agl` & `floor_amsl_m`, check `cmd_speed_max`/`ams_max_climb`, skip `p_safe`/`ood_var` on `HOLD_LOITER`, add `VertiLlamaEngine` RAII destructor + `n_batch` chunking.
- **`DR-7` (4 Critical):**
  1. Fix `13.2 kWh` → `22 kWh` and `R=0.80 m` → `R=0.75 m` (`7.069 m²`), and drop duplicate `√2` factor in coaxial induced-power formula (`~40.5 kW / 0.61% SoC` per 12 s).
  2. Fix additive `Λ_k` OOD precision collapse (cannot raise `σ²` when new tarp appears) via explicit OOD variance floor / `Λ_k` cap/reset.
  3. Fix Active Orbit 12 s timer starting before cruise-to-22 m descent (`APPROACH_ORBIT` phase needed), include `step_max_m`/`clearance_r_m` in reject bitmask, guard unsigned timestamp underflow, and add camera tilt/gimbal so `80°` HFOV doesn't clip far edge of 5 m pad during orbit (`atan(20/22)=42.3° > 40°`).
  4. Fix `float32` catastrophic cancellation in global-ENU `Σ = M/N - p̄p̄ᵀ` (accumulate cell-local `δp = p_W - c_ij` or Welford), fix `SlzOrtRunner` `cudaStream_t` leak, and fix Mod B affine shift + Metric3D FOV crop trap.
