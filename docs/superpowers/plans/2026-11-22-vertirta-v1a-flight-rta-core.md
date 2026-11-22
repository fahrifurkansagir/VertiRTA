# VertiRTA v1.0.0 Flight & RTA Core (Plan 1A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify `VertiRTA` v1.0.0's flight-critical core — two-target zero-heap `libvertirta`, offline Colab G4 JAX 4D/2D fault-degraded CBVF LUTs, 400 Hz PX4 v1.16 internal INDI+WLS/TMR control (including same-arm dual-loss active flat spin), Gazebo Harmonic coaxial/Thevenin/Dryden SITL, and the 100 Hz ROS 2 Jazzy `px4-ros2-interface-lib` RTA stack.

**Architecture:** All flight-critical control, 3-IMU TMR voting, coaxial effectiveness, 1-RC Thevenin sag, static `DAQP v0.10.0` solving, and `.vrta` logging live in portable zero-dependency C++17 `libvertirta_flight_core` (linked directly into PX4 v1.16's 400 Hz `mc_indi_wls_control` and bare-metal Cortex-M7 gated). Modern C++20 `libvertirta_rta_companion` links `flight_core` with `mmap` 4D/2D JAX CBVF LUTs, 100 Hz HJI-CBVF-QP + irreversible CEL latch, Dual-Res `3.094 MiB` `ROG-Map`/FIESTA ESDF, analytic `WireCapsules`, ADS-B KF intruders, and `ISlzPriorProvider`, bridged to PX4 via a 4-thread lock-free SPSC ROS 2 Jazzy executive.

**Tech Stack:** C++17 / C++20, CMake 3.22+ (`FetchContent`), `DAQP v0.10.0` (C99 static), Catch2 v3, PX4 Autopilot v1.16 (`release/1.16`), Gazebo Harmonic (`gz-sim8`), ROS 2 Jazzy, `Auterion/px4-ros2-interface-lib` (`release/1.16`), Python 3.12 + JAX / `StanfordASL/hj_reachability` v0.7.0 (Colab G4 Blackwell 96 GB).

**Spec:** `docs/superpowers/specs/2026-11-22-vertirta-tier-s-design.md` (v1.2 FROZEN) and `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (7/7 Forensic Errata Lock).

## Global Constraints

- **Vehicle (`config/vehicle_x8_240.yaml`):** `VertiRTA-X8-240` full-scale coaxial X8: `m=240.0 kg`, `g=9.80665 m/s²` (`W=2353.6 N`), `J=diag(85,95,140) kg·m²`, 4 arms `±45°/±135°`, `L=1.35 m` (`d=0.95459 m`), `R=0.75 m`, `Δz_coax=0.35 m`, `η_coax,0=0.85` (`k_T0=1.3809e-2`, `k_Tl=1.1738e-2`, `k_Q0=1.0488e-3`, `k_Ql=1.0068e-3`), `J_r=0.135 kg·m²`, `τ_up=0.085 s`, `τ_coast=0.424 s`, `ω_max=240.0 rad/s`, `T/W=2.5` (`5,886 N`; `795.4 N` upper / `676.1 N` lower), `K_e,bus<=1.36 V·s/rad` (`R_eff=0.72 Ω`), `320 Wh/kg` (`68 kg`) `400 V / 22 kWh` (`112S12P`, `54.0 Ah`, `R0=136.3 mΩ`, `R1=84.0 mΩ`, `C1=300.0 F`), `v_max=22.0 m/s`, `R_SLZ>=5.0 m`, `θ<=8°`, `σ_z<=0.15 m`.
- **Geodesy & Airspace (`config/istanbul_ltfj_airspace.yaml`):** Kartal ENU datum `40.8860°N, 29.1880°E, 12.0 m AMSL`, `N_EGM2008=+36.7 m` (`h_HAE=48.7 m`). Exact `LTFJ` ARP `40.89833°N, 29.30917°E` (`5 NM=9,260 m` + NW polygon/arc join `J2=40.91279°N, 29.19212°E`, ceiling `798.58 m HAE`) & `LTBX` (`646.35 m HAE`, ARP verified vs DHMİ `AD 2.17`); flyable corridor & SLZs strictly `<29.190°E` near `J2`.
- **Two-Target Library & Determinism:** `vertirta_flight_core` (`ALIAS vertirta::flight_core`) = freestanding C++17 (NO `<algorithm>`, NO `PROJ 9`, NO exceptions/RTTI), `-Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti -fno-threadsafe-statics -fno-fast-math -ffp-contract=off`, `<256 KB` Flash / `<128 KB` RAM bare-metal gate, `0` heap allocs (never put `inline operator new` in headers!). `vertirta_rta_companion` (`ALIAS vertirta::rta_companion`) = C++20 linking `vertirta::flight_core` + `mmap` 4D/2D LUTs + Dual-Res `3.094 MiB` ESDF + `slz_reach` + `slz_prior_provider`. Same-ISA `.vrta` = `0-ULP` SHA-256; cross-ISA `<1e-12`.
- **DR-1/DR-2 Errata Locks:** WLS uses QR active-set on `A∈R^{12×8}` OR `f64` Cholesky (NEVER `f32` normal eqs); `FD_FAIL_R=0, FD_FAIL_P=0`; `VertirtaCapabilityEnvelope.msg` exact `72 B` (w/ `float32 chi2_residual` + `uint32 _pad`); Gazebo plugin fixes `v_iu+v_air.Z()`, `tau_coast_`, `0.96*k_Q0_` lower; JAX matches `_support_U`/`partial_max_magnitudes` CFL, `smax - ln(2)/β`, `2M-Cross` `v_cap,f`, overhead intruder `Δp_z` sign; `MmapCbvf4DLut::eval` guards `std::isfinite` & `regime_idx<4`; `SlidingConfig` zero-init (`ROG-Map #15`); `px4-ros2-interface-lib` handles #201/#167/#191.
- **Ports & Scope Boundary:** Live ports `Foxglove :8891`, `Cesium :5180`, `MicroXRCEAgent :8888`; Tier-2 `:8900–8903`/`:8910–8911`; forbidden `4173,4174,8765,9222,9223`. Plan 1A excludes CI workflows/Assurance/GSN/Atlas/Cesium/Blender (Plan 1B), TR/EN LLM/Z3 (Plan 2), and visual SLZ ML (Plan 3).

## Review Focus

1. **NaN / Inf / OOB `regime_idx` & Corrupt LUTs:** Non-finite inputs or `regime_idx>=4` in `TmrVoter`, `MmapCbvf4DLut::eval`, or `StaticDaqpSolver` must never UB/OOB-read; they deterministically fault-latch or return fallback (`Tasks 3, 5, 7, 8`).
2. **Overhead Intruder Climbing Sign Flip & `r→0` Singularity:** Climbing (`v_z>0`) toward an overhead intruder (`Δp_z<0`, `r<0.5 m`) must produce negative closing velocity (triggers climb braking) and clamp `max(r, 0.5 m)` (`Task 8`).
3. **Irreversible `SAME_ARM_DUAL_LOSS` CEL Latch Under Flicker:** Transient regime flicker back to `NOMINAL` must never unlatch CEL; handover to Option B1b flat spin (`±2.2 rad/s`, `Δψ=∓10.6°`) and 2D vertical braking LUT stays monotonic (`Tasks 4, 8, 12`).
4. **ESDF/Intruder Staleness (`>250 ms`/`>1 s`/`>2 s`) & Clutter "Leaky Corner":** Stale inputs inflate radii by `ΔR_stale`, cap speed `8.0 m/s` (`>1 s`), and divert (`>2 s`); opposing gradients (`e_r,i·e_r,j<-0.5`) add `δ_clutter=1.5 m` (`Tasks 8, 9`).
5. **Double-Applied `+36.7 m` Geoid & PX4↔ROS 2 Quaternion Sign Mismatch:** Cross-frame additions fail at compile time; heading-035° quat round-trip and Gazebo `navsat` vs PX4 AMSL/HAE checks catch `+36.7 m` double-addition (`Tasks 2, 11, 12`).

---

## File Structure

- `CMakeLists.txt`, `CMakePresets.json`, `scripts/setup_wsl2_ubuntu24.sh` — Root CMake (`vertirta_flight_core`/`vertirta::flight_core`, `vertirta_rta_companion`/`vertirta::rta_companion`, Catch2, optional `add_subdirectory` hooks for Plan 2/3 `core/{stl,copilot,slz_bayes}` & `sim/colab_egl`), presets (`dev-linux-x64`, `asan-ubsan`, `tsan`, `arm-cortex-m7`), Ubuntu 24.04 WSL2 bootstrap.
- `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml` — Frozen vehicle SI parameters and Maltepe–Dragos–Kartal / `LTFJ` / `LTBX` geodesy & CTR walls.
- `core/flight_core/` (`CMakeLists.txt`, `include/vertirta/core/{math_utils,frames,geodesy,tmr_voter,coax_envelope,indi_wls,motor_fdi,daqp_qp,vrta_recorder,sim_plant_6dof}.hpp`, `src/*.cpp`) — Freestanding zero-heap C++17 flight kernel.
- `reachability/` (`cbvf_4d_dynamics.py`, `solve_cbvf_luts.py`, `solve_cel_2d_lut.py`, `verify_lut_soundness.py`) — Colab G4 JAX `hj_reachability` 4D CBVF (`11.31 MB`) & 2D CEL (`6.97 KiB`) LUT pipeline.
- `core/rta_companion/` (`CMakeLists.txt`, `include/vertirta/rta/{mmap_cbvf_lut,cel_2d_lut,dryden_wind,cbvf_safety_filter,robocentric_esdf,wire_capsules,intruder_kf,slz_prior_provider,slz_reach}.hpp`, `src/*.cpp`) — C++20 companion RTA kernel (`mmap` LUTs, 100 Hz HJI-CBVF-QP, `3.094 MiB` ESDF, canonical `ISlzPriorProvider`/`SlzPrior`/`SlzProviderSnapshot`).
- `tools/cli/` (`CMakeLists.txt`, `vertirta_replay_cli.cpp`, `vertirta_bench_cli.cpp`, `vertirta_6dof_batch_cli.cpp`) — Standalone glibc-safe CLI runners consumed by Plan 1B Tiers 1–4 CI & Tier-1 10k 6-DOF Survival Atlas.
- `px4_overlay/` (`msg/{VertirtaCapabilityEnvelope,VertirtaRtaSetpoint}.msg`, `modules/mc_indi_wls_control/`, `patches/px4_v116_gz_bridge_multi_imu.patch`, `config/dds_topics_vertirta.yaml`, `airframes/{12050_gz_vertirta_x8_240,rc.vertirta_apps}`) — PX4 v1.16 400 Hz overlay & 4-part `GZBridge` patch.
- `sim/` (`gz_plugins/src/{vertirta_coaxial_propulsion,vertirta_dryden_wind}.cpp`, `models/vertirta_x8_240/model.sdf`, `worlds/kartal_coast.sdf`) — Gazebo Harmonic coaxial X8 model, physics plugins, 400 Hz `kartal_coast` world.
- `ros2_ws/src/` (`vertirta_msgs`, `vertirta_px4_bridge`, `vertirta_rta_node`, `vertirta_map_node`, `vertirta_fault_injector`, `vertirta_bringup`) — ROS 2 Jazzy adapters, 4-thread SPSC 100 Hz RT node, 20 Hz ESDF, Foxglove `:8891`.
- `tests/` (`config/`, `flight_core/`, `reachability/`, `rta_companion/`, `sim/`, `integration/`) — Catch2 unit tests, standalone `zero_heap_guard`, pytest checks, TSan SPSC, Gate-2 closed-loop smoke test.

---

### Task 1: Repository Bootstrap, CMake Presets & Frozen Configs

**Files:** Create `CMakeLists.txt`, `CMakePresets.json`, `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`, `scripts/setup_wsl2_ubuntu24.sh` | Test `tests/config/test_config_contracts.py`
**Interfaces:** Consumes `design.md` §4 & errata locks. Produces `config/*.yaml` and CMake presets `dev-linux-x64`, `asan-ubsan`, `tsan`, `arm-cortex-m7`.

- [ ] **Step 1: Write failing config contract test**
```python
def test_vehicle_and_airspace_contracts():
    v = yaml.safe_load(Path("config/vehicle_x8_240.yaml").read_text())
    a = yaml.safe_load(Path("config/istanbul_ltfj_airspace.yaml").read_text())
    assert v["mass_kg"] == 240.0 and v["tw_nominal"] == 2.5 and v["battery"]["pack_mass_kg"] == 68.0
    assert v["propulsion"]["ke_bus_vs_rad"] <= 1.36 and v["propulsion"]["eta_coax_hover"] == 0.85
    assert (a["datum"]["lat_deg"], a["datum"]["lon_deg"], a["datum"]["egm2008_n_m"]) == (40.8860, 29.1880, 36.7)
    assert a["odd_bounds"]["max_lon_near_j2_deg"] < 29.190 and a["airspace"]["ltfj_ctr"]["nw_j2_lon_deg"] == 29.19212
    assert set(a["ports"].values()).isdisjoint({4173, 4174, 8765, 9222, 9223})
```
- [ ] **Step 2: Run test to verify failure** — `pytest tests/config/test_config_contracts.py -v` → FAIL (`FileNotFoundError`).
- [ ] **Step 3: Create root `CMakeLists.txt`, `CMakePresets.json`, frozen YAMLs, and `scripts/setup_wsl2_ubuntu24.sh`** — Lock `VertiRTA-X8-240` SI params (`320 Wh/kg` 68 kg pack, `112S12P`, `R0=0.1363`, `R1=0.084`, `C1=300.0`, `K_e,bus=1.36`, `d=0.95459`, `R=0.75`, `R_SLZ>=5.0`, `θ<=8°`, `σ_z<=0.15 m`), exact `LTFJ` ARP `40.89833°N, 29.30917°E` + `J2` (`798.58 m HAE`) & `LTBX` (`646.35 m HAE`), ports (`8891,5180,8888`), `VERTIRTA_STRICT_FP_FLAGS`, conditional optional `add_subdirectory` hooks for `core/stl`, `core/copilot`, `core/slz_bayes`, and `sim/colab_egl` (preventing root `CMakeLists.txt` overlap with Plans 2 & 3), and idempotent `--check-only` WSL2 Ubuntu 24.04 bootstrap script.
- [ ] **Step 4: Verify pass** — `pytest tests/config/test_config_contracts.py -v && cmake --preset dev-linux-x64` → PASS.
- [ ] **Step 5: Commit** — `git add CMakeLists.txt CMakePresets.json config/ scripts/setup_wsl2_ubuntu24.sh tests/config/ && git commit -m "chore: bootstrap CMake presets and frozen VertiRTA-X8-240/LTFJ configs"`

---

### Task 2: `libvertirta_flight_core` Strong Frame Types, Quaternions & Constexpr EGM2008 Patch

**Files:** Create `core/flight_core/CMakeLists.txt`, `core/flight_core/include/vertirta/core/{math_utils,frames,geodesy}.hpp`, `core/flight_core/src/frames.cpp` | Test `tests/flight_core/test_frames_geodesy.cpp`
**Interfaces:** Consumes root strict FP flags. Produces `vertirta_flight_core` (`ALIAS vertirta::flight_core`) with freestanding `clamp_f`/`clamp_d` (NO `<algorithm>`), `Vec3{Ned,Enu,Frd,Flu}` in `namespace vertirta::core` (re-exported via `using` in `namespace vertirta`), `QuatPx4FrdNed{w,x,y,z}`, `QuatRosFluEnu{x,y,z,w}`, `{enu_to_ned,ned_to_enu,flu_to_frd,frd_to_flu,px4_to_ros_quat,ros_to_px4_quat}`, `GeodeticWgs84{lat_deg,lon_deg,h_hae_m}`, `egm2008_undulation_m(lat,lon)`, `amsl_to_hae(h_amsl,lat,lon)`, `hae_to_amsl(h_hae,lat,lon)`, `geodetic_to_local_enu(...)`, `local_enu_to_geodetic(...)`.

- [ ] **Step 1: Write failing Catch2 test (pins Review Focus #5)**
```cpp
TEST_CASE("Strong frames block cross-frame add, match heading-035 quat, and keep EGM2008 +36.7 m exact") {
  static_assert(!std::is_invocable_v<decltype([](auto a, auto b){ return a + b; }), Vec3Ned, Vec3Enu>);
  const QuatPx4FrdNed q_px4{0.945380, 0.098027, 0.121511, 0.286149};
  const auto q_ros = px4_to_ros_quat(q_px4);
  REQUIRE_THAT(q_ros.x, WithinAbs(0.155237, 1e-5)); REQUIRE_THAT(q_ros.w, WithinAbs(0.870823, 1e-5));
  const auto rt = ros_to_px4_quat(q_ros);
  REQUIRE(std::abs(rt.w*q_px4.w + rt.x*q_px4.x + rt.y*q_px4.y + rt.z*q_px4.z) > 1.0 - 1e-12);
  const double N = egm2008_undulation_m(40.8860, 29.1880);
  REQUIRE(N > 20.0); REQUIRE(N < 50.0); REQUIRE_THAT(amsl_to_hae(12.0, 40.8860, 29.1880), WithinAbs(48.7, 0.15));
  const auto ltfj = geodetic_to_local_enu({40.898333, 29.309167, 131.7});
  REQUIRE_THAT(std::hypot(ltfj.x, ltfj.y), WithinAbs(10309.0, 40.0));
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_frames_geodesy` → FAIL.
- [ ] **Step 3: Implement freestanding C++17 `vertirta_flight_core` frames & `constexpr` bilinear EGM2008 patch** — Configure STATIC C++17 target (`-fno-exceptions -fno-rtti -fno-threadsafe-statics -ffunction-sections -fdata-sections`, zero `<algorithm>`, zero `PROJ 9`). Implement `<1 KB` `4×4` bilinear EGM2008 patch (`[40.75..41.15°N, 29.00..29.45°E]` anchored to `LTFJ/LTBA/LTFM` `36.58/36.88/37.19 m`, Kartal `+36.7 m`) and exact WGS84↔ECEF↔ENU transforms.
- [ ] **Step 4: Verify pass** — `ctest --preset dev-linux-x64 -R test_frames_geodesy --output-on-failure` → PASS.
- [ ] **Step 5: Commit** — `git add core/flight_core/ tests/flight_core/test_frames_geodesy.cpp && git commit -m "feat(flight_core): add strong frame types, quats, and constexpr EGM2008 patch"`

---

### Task 3: `libvertirta_flight_core` 400 Hz 3-IMU MVS + `χ²_24` TMR FDIR Voter

**Files:** Create `core/flight_core/include/vertirta/core/tmr_voter.hpp`, `core/flight_core/src/tmr_voter.cpp` | Test `tests/flight_core/test_tmr_voter.cpp`
**Interfaces:** Consumes `Vec3Frd`, `clamp_f`. Produces `enum class TmrState : uint8_t { TMR_3OF3=0, DUPLEX_2OF2=1, SIMPLEX_ANALYTICAL=2, FAILSAFE_DIVERT=3 }`, `ImuSampleFrd`, `TmrVoteOutput{gyro_voted, accel_voted, state, health_mask, max_chi2_residual, isolation_latency_ms}`, `TmrVoter::step(const ImuSampleFrd imus[3], const Vec3Frd& model_gyro_pred, bool model_valid) noexcept -> TmrVoteOutput`.

- [ ] **Step 1: Write failing Catch2 tests (pins Review Focus #1 NaN guard)**
```cpp
TEST_CASE("TMR masks tick-0 hardover, isolates 0.12 rad/s IMU0 bias <=20 ms, rejects gusts, and latches NaN") {
  TmrVoter voter;
  // 1) 200 ticks common-mode 1.5 rad/s gust on all 3 IMUs -> TMR_3OF3, max_chi2_residual < 72.23
  // 2) Inject +0.12 rad/s on IMU0 -> tick 0 MVS unaffected (<1e-4); by tick 8 (20 ms) DUPLEX_2OF2, health_mask==0b110
  // 3) Inject NaN on IMU2 -> immediate bit-2 clear; restore finite -> monotonic (never unlatches)
  // 4) Second fault without analytical 3-sigma agreement -> FAILSAFE_DIVERT
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_tmr_voter` → FAIL.
- [ ] **Step 3: Implement `TmrVoter` in `core/flight_core/src/tmr_voter.cpp`** — Coordinate-wise MVS in `TMR_3OF3`, `0.02 Hz` inter-IMU bias tracker frozen above `0.5 * 72.23`, 8-tick (`20 ms`) ring buffer computing `Q_ij ~ χ²_24` (`T_24=72.23`, `σ_g_eff=0.015 rad/s`, `σ_a_eff=0.25 m/s²`), immediate `std::isfinite` NaN/Inf latch, monotonic `TMR→DUPLEX→SIMPLEX/FAILSAFE_DIVERT` with `3σ/6σ` model tie-breaker. Zero heap, zero `<algorithm>`.
- [ ] **Step 4: Verify pass** — `ctest --preset dev-linux-x64 -R test_tmr_voter --output-on-failure` → PASS.
- [ ] **Step 5: Commit** — `git add core/flight_core/ tests/flight_core/test_tmr_voter.cpp && git commit -m "feat(flight_core): add 400 Hz 3-IMU MVS and chi2_24 TMR FDIR voter"`

---

### Task 4: `libvertirta_flight_core` Coaxial `B(η_coax)`, 1-RC Thevenin Sag, INDI + `f64`/QR WLS, Option B1b Spin & Motor FDI

**Files:** Create `core/flight_core/include/vertirta/core/{coax_envelope,indi_wls,motor_fdi}.hpp`, `core/flight_core/src/{coax_envelope,indi_wls,motor_fdi}.cpp` | Test `tests/flight_core/test_indi_wls_thevenin.cpp`
**Interfaces:** Consumes `Vec3Frd`, `Vec3Ned`, `QuatPx4FrdNed`, `clamp_f`. Produces `enum class FlightRegime : uint8_t { NOMINAL=0, SINGLE_LOSS=1, CROSS_DUAL=2, SAME_ARM_ACTIVE_SPIN=3, CRITICAL_DESCENT=4 }`, `CapabilityEnvelope`, `CoaxEffectiveness::{build_B, two_arm_tw_at_vterm}`, `TheveninBattery1RC::{step, omega_lim_from_vterm}`, `IndiWlsController::step_400hz(...)`, `MotorFdiObserver::step_400hz(...)`.

- [ ] **Step 1: Write failing Catch2 tests (pins DR-1 errata #1/#2/#4 & Review Focus #3)**
```cpp
TEST_CASE("Coaxial unblanking, K_e=1.36 sag floor, f64/QR WLS stability, Option B1b spin, and <85 ms FDI") {
  double B[4][8]; float umax[8];
  CoaxEffectiveness::build_B(0x00, 0.f, B, umax, 400.f);
  REQUIRE_THAT(umax[0], WithinAbs(795.4f, 1.5f)); REQUIRE_THAT(umax[4], WithinAbs(676.1f, 1.5f));
  CoaxEffectiveness::build_B(0x01, 0.f, B, umax, 400.f); REQUIRE_THAT(umax[4], WithinAbs(795.4f, 1.5f));
  REQUIRE(CoaxEffectiveness::two_arm_tw_at_vterm(358.4f) >= 1.249f);
  REQUIRE(CoaxEffectiveness::two_arm_tw_at_vterm(338.0f) >= 1.120f);
  // 10k-tick f64/QR WLS across Regimes 0-3 -> 0 Cholesky failures, finite outputs
  // Arm 0/1 dual loss -> spin_rate_ref == -2.2f (CCW), phase lead == +10.6 deg; Arm 2/3 -> +2.2f (CW), -10.6 deg
  // FDI: ESC cut isolated <60 ms; 303.7 N*m blade loss rho>0.88 isolated <85 ms; 45 N*m gust -> 0x00
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_indi_wls_thevenin` → FAIL.
- [ ] **Step 3: Implement `coax_envelope`, `indi_wls`, and `motor_fdi`** — Implement `B_T` (`d=0.95459`, `c_m,u=0.07590`, `c_m,l=0.08572`, upper-loss lower unblanking `+17.6%`, Glauert `f_ov` wake skew, `K_e,bus=1.36`), 1-RC Thevenin ECM (`R0=0.1363, R1=0.084, C1=300.0, Q=54.0 Ah`), `S²` reduced attitude (`k_n=2.6`, `K_Omega=diag(5,5,3)`), synchronized prewarped 8 Hz Butterworth biquads (`b=[0.003622, 0.007243, 0.003622]`, `a=[1, -1.822695, 0.837182]`), motor-lag observer (`τ_up=0.085, d=2`, `G2` inertia term), **Active-Set WLS strictly via `double` (`f64`) Cholesky or QR on `A∈R^{12×8}`** (fixing `DR-1` erratum #1; freestanding `clamp_f`, fixing erratum #2), **Option B1b Active Flat Spin** (`±2.2 rad/s`, rotated `W_v,diag=diag(100,10,4,30)`, `Δψ=atan(Ω_z*τ_m)=∓10.6°`, Arm 1 `M5<=150 N` assist), and dual-channel Motor FDI (`D_w=diag(1/300,1/300,1/25,1/300)`).
- [ ] **Step 4: Verify pass** — `ctest --preset dev-linux-x64 -R test_indi_wls_thevenin --output-on-failure` → PASS.
- [ ] **Step 5: Commit** — `git add core/flight_core/ tests/flight_core/test_indi_wls_thevenin.cpp && git commit -m "feat(flight_core): add coaxial INDI+f64-WLS, Option B1b spin, Thevenin sag, and Motor FDI"`

---

### Task 5: `libvertirta_flight_core` Static `DAQP v0.10.0` Wrapper, `.vrta` 256 B Logger & 6-DOF Coaxial Plant

**Files:** Create `core/flight_core/include/vertirta/core/{daqp_qp,vrta_recorder,sim_plant_6dof}.hpp`, `core/flight_core/src/{daqp_qp,vrta_recorder,sim_plant_6dof}.cpp`, `tests/flight_core/{zero_heap_guard.hpp,zero_heap_guard.cpp,test_daqp_vrec_plant.cpp}` | Modify `core/flight_core/CMakeLists.txt` (`FetchContent` `darnstrom/daqp` `v0.10.0`)
**Interfaces:** Consumes Tasks 2–4. Produces `StaticDaqpSolver<N_MAX=6, M_MAX=32>` (`<=6.2 KiB`), canonical naturally-aligned little-endian POD `VrtaFileHeader` (`128 B`: `char magic[8]="VRTA0001"`, `uint32_t schema_ver=1, reserved0=0`, `double datum_lat_deg, datum_lon_deg, datum_amsl_m, egm2008_n_m`, `uint8_t lut_sha256[32]`, `char git_sha[40]`, `uint8_t reserved1[4]`, `uint32_t header_crc32c`) + `VrtaFrame100Hz` (`256 B`: `uint64_t seq, t_sim_us`, `double pos_enu[3]`, `float vel_enu[3], q_frd_ned[4], omega_frd[3], a_nom_enu[3], a_safe_enu[3], rotor_rpm[8], v_term_v, soc_pct, cbvf_min_v, ltfj_margin_m, ltbx_margin_m, chi2_residual, agl_m`, `uint8_t active_motor_mask, flight_regime, tmr_state, cel_latched`, `uint32_t flags`, `uint8_t pad[80]`, `uint32_t crc32c`), `CoaxialPlant6Dof::step_400hz(...)`, and test-only `ScopedZeroHeapGuard` (standalone TU `operator new` + `-Wl,--wrap=malloc`).

- [ ] **Step 1: Write failing Catch2 tests (pins 0-heap, NaN fallback & Option B1b vs Option A proof)**
```cpp
TEST_CASE("StaticDaqpSolver 0-heap, Vrta POD sizes, and 6-DOF Option B1b survival vs Option A capsize") {
  static_assert(sizeof(StaticDaqpSolver<6, 32>) <= 6400);
  static_assert(sizeof(VrtaFileHeader) == 128 && std::is_trivially_copyable_v<VrtaFileHeader>);
  static_assert(sizeof(VrtaFrame100Hz) == 256 && std::is_trivially_copyable_v<VrtaFrame100Hz>);
  // 1) Under ScopedZeroHeapGuard: feasible n=5,m=28 QP -> OPTIMAL (KKT<1e-9, alloc==0); NaN/infeasible -> used_fallback==true
  // 2) 15 s 6-DOF flight from 35 m AGL w/ M0+M4 cut + 5 m/s gust:
  //    Option A (spin=0) -> roll/pitch > 60 deg (capsize);
  //    Option B1b (spin=-2.2) -> lands |v_z|<=0.8 m/s, tilt<=15 deg, CRC32C valid & 0-ULP SHA-256 identical on rerun!
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_daqp_vrec_plant` → FAIL.
- [ ] **Step 3: Wire `DAQP v0.10.0` C99 core and implement `daqp_qp`, `vrta_recorder`, `sim_plant_6dof`, and `zero_heap_guard`** — Compile only static C99 DAQP `.c` units (no dynamic `daqp_setup`/`calloc`, fix `DR-2` erratum #5 removing `DAQP_NO_SOFT_WEIGHTS` symbol clash and broken `||`). Implement `VrtaFileHeader`/`VrtaFrame100Hz`/CRC32C, 400 Hz RK4 `CoaxialPlant6Dof`, and test-only `zero_heap_guard.cpp` (never an `inline` header override!).
- [ ] **Step 4: Verify pass (incl. ASan/UBSan)** — `ctest --preset dev-linux-x64 --output-on-failure && ctest --preset asan-ubsan --output-on-failure` → PASS.
- [ ] **Step 5: Commit** — `git add core/flight_core/ tests/flight_core/ && git commit -m "feat(flight_core): add DAQP v0.10.0 static wrapper, .vrta 256B logger, and 6-DOF plant"`

---

### Task 6: Offline Colab G4 JAX 4D Fault-Degraded CBVF & 2D CEL Braking LUT Pipeline

**Files:** Create `reachability/{cbvf_4d_dynamics,solve_cbvf_luts,solve_cel_2d_lut,verify_lut_soundness}.py` | Test `tests/reachability/test_cbvf_dynamics_cpu.py`
**Interfaces:** Consumes `config/vehicle_x8_240.yaml`. Produces `.vrta_lut` (`11,313,864 B`: `4096 B` header `VRTALUT1` + `eps_grid[4]` + `v_cap_f[4]` + SHA-256 + `4 × 41×29×41×29` `int16`) and `.vrta_cel2d` (`4096 B` header + `3 × 41×29` `int16` = `6.97 KiB` for `T/W ∈ {1.25, 1.20, 1.15}`).

- [ ] **Step 1: Write failing pytest soundness tests (pins DR-2 critical errata #1/#2/#3)**
```python
def test_cfl_bounds_conservative_smax_2m_vcap_and_cel_2d():
    dyn = VertiRta4DDynamics(a_thrust_max=2.50*9.80665, tilt_max_rad=np.deg2rad(35.0))
    pm = dyn.partial_max_magnitudes(jnp.array([10., 0., 10., 0.]), 0., 0., None)
    assert float(pm[1]) >= dyn.max_achievable_ur() + dyn.d_ar - 1e-4  # Erratum #1: matches 14.06 m/s^2!
    for sr, sz in [(0.0, 0.0), (1.0, 1.0), (-2.0, 0.5)]:
        assert float(conservative_smax_4(sr, sz, beta=4.0)) <= max(sr, sz) + 1e-6  # Erratum #2: -ln(2)/beta
    assert compute_degraded_v_cap("2M-Cross", d_ar=1.75, r_max=58.0, horizon_s=6.0) < 16.5  # Erratum #3
    # Verify 2D CEL LUT (41x29 x 3 sag tiers) matches z_req=(vz^2-4)/(2*a_net) within 0.35 m
```
- [ ] **Step 2: Run test to verify failure** — `pytest tests/reachability/test_cbvf_dynamics_cpu.py -v` → FAIL.
- [ ] **Step 3: Implement JAX `hj_reachability` v0.7.0 4D CBVF & 2D CEL solvers** — Fix `partial_max_magnitudes` horizontal bound to `a_T * sin(tilt_max)` (matched `_support_U` Lax-Friedrichs CFL), subtract `ln(2)/4.0` from `smax_4`, embed `v_cap_f` in header for `2M-Cross`/`Low-Vbat`. `solve_cbvf_luts.py` runs `81×57×81×57` FP32 on Colab G4 Blackwell (`--quick-cpu-fixture` for fast local test LUT) → exact stride-2 `41×29×41×29` `int16` (`q=0.992 mm`) + fine-grid `eps_grid_f` audit. `solve_cel_2d_lut.py` exports `3 × 2.32 KiB` vertical braking LUT (`v_td,max=2.0 m/s`).
- [ ] **Step 4: Verify pass** — `pytest tests/reachability/test_cbvf_dynamics_cpu.py -v && python reachability/verify_lut_soundness.py --quick-fixture` → PASS (`11,313,864 B` & `6,970 B` payload verified).
- [ ] **Step 5: Commit** — `git add reachability/ tests/reachability/ && git commit -m "feat(reachability): add JAX 4D CBVF and 2D CEL LUT pipeline with CFL/smax fixes"`

---

### Task 7: `libvertirta_rta_companion` `MmapCbvf4DLut`, `Cel2DLut` & Discrete MIL-F-8785C Dryden Filter

**Files:** Create `core/rta_companion/CMakeLists.txt`, `core/rta_companion/include/vertirta/rta/{mmap_cbvf_lut,cel_2d_lut,dryden_wind}.hpp`, `core/rta_companion/src/{mmap_cbvf_lut,cel_2d_lut,dryden_wind}.cpp` | Test `tests/rta_companion/test_lut_dryden.cpp`
**Interfaces:** Consumes `vertirta::flight_core`, `.vrta_lut`/`.vrta_cel2d`. Produces `vertirta_rta_companion` (`ALIAS vertirta::rta_companion`), `MmapCbvf4DLut::{load, eval(regime,r,vr,z,vz)->CbvfQueryResult}`, `Cel2DLut::{load, eval(v_term,z_agl,vz_up)->Cel2DQuery}`, `DrydenTurbulenceFilter::step(dt,h_agl,v_air,w20,eta[3])->Vec3Ned`.

- [ ] **Step 1: Write failing Catch2 tests (pins DR-2 erratum #4 & Review Focus #1)**
```cpp
TEST_CASE("MmapCbvf4DLut guards NaN/Inf and regime>=4, verifies SHA-256, and Dryden preserves >=99.5% var") {
  MmapCbvf4DLut lut; REQUIRE(lut.load("tests/fixtures/quick_test.vrta_lut"));
  REQUIRE_FALSE(lut.eval(4, 10., 0., 10., 0.).valid);
  REQUIRE_FALSE(lut.eval(0, std::nan(""), 0., 10., 0.).valid);
  REQUIRE_FALSE(lut.eval(0, 10., std::numeric_limits<double>::infinity(), 10., 0.).valid);
  REQUIRE(lut.eval(0, 15., -5., 20., -1.).valid); // 10k-query p99 < 2.0 us in Release
  // 200k-tick Tustin Dryden @ 100 Hz, h=60 m, W20=15.43 m/s -> empirical std within 3% of Table 4
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_lut_dryden` → FAIL.
- [ ] **Step 3: Implement `vertirta_rta_companion` (C++20) LUT loaders & Tustin Dryden filter** — `mmap`+`mlock`+SHA-256 check; guard `!loaded_ || regime_idx >= 4 || !std::isfinite(...)` before indexing; 16-corner quadrilinear value & analytic gradient subtracting `eps_grid_[regime_idx]`; exact 1st-order `H_u` and 2nd-order `H_v, H_w` Tustin Dryden difference equations scaled by `sqrt(pi/dt)`.
- [ ] **Step 4: Verify pass** — `ctest --preset dev-linux-x64 -R test_lut_dryden --output-on-failure` → PASS.
- [ ] **Step 5: Commit** — `git add core/rta_companion/ tests/rta_companion/test_lut_dryden.cpp && git commit -m "feat(rta_companion): add guarded MmapCbvf4DLut, Cel2DLut, and Tustin Dryden filter"`

---

### Task 8: `libvertirta_rta_companion` 100 Hz HJI-CBVF-QP Safety Filter, Irreversible CEL Latch & Standalone CLI Runners

**Files:** Create `core/rta_companion/include/vertirta/rta/cbvf_safety_filter.hpp`, `core/rta_companion/src/cbvf_safety_filter.cpp`, `tools/cli/{CMakeLists.txt,vertirta_replay_cli.cpp,vertirta_bench_cli.cpp,vertirta_6dof_batch_cli.cpp}` | Test `tests/rta_companion/test_cbvf_safety_filter.cpp`
**Interfaces:** Consumes `StaticDaqpSolver`, `CapabilityEnvelope`, `MmapCbvf4DLut`, `Cel2DLut`, `CoaxialPlant6Dof`, `DrydenTurbulenceFilter`, `ScopedZeroHeapGuard`. Produces `HazardPoint`, `SafetyFilterOutput`, `CbvfSafetyFilter::step_100hz(pos_ned, vel_ned, a_nom_ned, yaw_rate_nom, env, std::span<const HazardPoint>) noexcept -> SafetyFilterOutput`, plus glibc-safe CLI binaries `vertirta_replay_cli` (`--in <.vrta> --out-json <json>`), `vertirta_bench_cli` (`--clock monotonic_raw|thread_cputime --ticks <N>`), and `vertirta_6dof_batch_cli` (`--manifest <json> --cbf-mode hocbf|hj_cbvf --alloc-mode stock_ca|indi_wls`).

- [ ] **Step 1: Write failing Catch2 tests (pins Review Focus #2, #3, #4 & DR-2 erratum #2)**
```cpp
TEST_CASE("Overhead intruder climb brakes, clutter inflates 1.5 m, BRT-outside uses u*_HJ, CEL latch is irreversible") {
  // 1) Ownship z_ned=-20 m climbing (v_D=-5 m/s) under intruder z_ned=-28 m (r=0.1 m):
  //    mapped v_z_rel < 0 (closing), r clamped to 0.5 m, and a_safe.z > 0 (brakes climb)!
  // 2) Two hazards at +E/-E (e_r1.e_r2 < -0.5) -> active margin inflated +1.5 m.
  // 3) min_V < -0.25 m -> used_hj_fallback==true; SAME_ARM_ACTIVE_SPIN -> cel_latched==true, stays true on NOMINAL flicker.
  // 4) 1,000 ticks under ScopedZeroHeapGuard -> alloc_count == 0.
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_cbvf_safety_filter` → FAIL.
- [ ] **Step 3: Implement `CbvfSafetyFilter` and standalone `tools/cli/` binaries** — Pack `K<=16` hazard rows + `12` actuator polytope rows (8-gon horizontal disk + 4 tilt-thrust cone rows) into `StaticDaqpSolver<5, 32>`. Preserve `s_z = (p_up - p_obs_up >= 0) ? +1 : -1` (`v_z_rel = s_z*(v_up - v_obs_up)`, `dV/da_D = -s_z*V_vz`), clamp `max(r, 0.5 m)`, add `δ_clutter=1.5 m` for opposing gradients (`e_r,i·e_r,j < -0.5`), enforce `v_cap_f` damping in `2M-Cross`/`Low-Vbat`, compute closed-form `u*_HJ` cone projection when `min_V < -0.25 m` or non-optimal QP, and irreversibly latch `SAME_ARM_ACTIVE_SPIN` to `Cel2DLut`. Build `vertirta_replay_cli`, `vertirta_bench_cli`, and `vertirta_6dof_batch_cli` in `tools/cli/` linked to `vertirta::rta_companion` (NEVER passing `-Wl,--wrap=malloc` so glibc I/O stays safe for Plan 1B Tiers 2–4 & Tier-1 10k Atlas).
- [ ] **Step 4: Verify pass** — `cmake --build --preset dev-linux-x64 --target test_cbvf_safety_filter vertirta_replay_cli vertirta_bench_cli vertirta_6dof_batch_cli && ctest --preset dev-linux-x64 -R test_cbvf_safety_filter --output-on-failure && ./build/dev-linux-x64/tools/cli/vertirta_bench_cli --clock monotonic_raw --ticks 1000` → PASS.
- [ ] **Step 5: Commit** — `git add core/rta_companion/ tools/cli/ tests/rta_companion/test_cbvf_safety_filter.cpp && git commit -m "feat(rta_companion): add 100 Hz HJI-CBVF-QP filter, CEL latch, and Plan-1B CLI binaries"`

---

### Task 9: `libvertirta_rta_companion` Dual-Res `3.094 MiB` `ROG-Map`/FIESTA ESDF, `WireCapsules`, Intruder KF & Canonical `ISlzPriorProvider`

**Files:** Create `core/rta_companion/include/vertirta/rta/{robocentric_esdf,wire_capsules,intruder_kf,slz_prior_provider,slz_reach}.hpp`, `core/rta_companion/src/{robocentric_esdf,wire_capsules,intruder_kf,slz_reach}.cpp` | Test `tests/rta_companion/test_esdf_hazards_slz.cpp`
**Interfaces:** Consumes `frames`, `geodesy`, `CapabilityEnvelope`, `HazardPoint`. Produces `RobocentricEsdfMap` (`<=3.15 MiB` static), `WireCapsuleTable` (`std::array<WireCapsule, 64>`), `IntruderTrackerKf` (`std::array<IntruderTrack, 16>`), canonical `vertirta::rta::SlzPrior` (`id[32]`, `lat_deg`, `lon_deg`, `elev_amsl_m`, `Vec3Enu center_enu`, `p_safe`, `slz_ood_variance`, `theta_max_deg`, `sigma_z_m`, `step_max_m`, `clearance_r_m`, `h_dyn_max`, `eta_s`, `soc_margin_pct`, `rank`, `reason_mask`), `vertirta::rta::SlzProviderSnapshot` (`std::array<SlzPrior, 8> priors`, `count`, `stamp_ns`, `stale`, `active_decision_state`, `gimbal_pitch_rad`, `gimbal_yaw_rad`), **`class vertirta::rta::ISlzPriorProvider`** (`query_priors(const Vec3Enu&, float soc_pct, uint64_t now_ns) const noexcept -> SlzProviderSnapshot`), and `SlzReachRanker` (v1.0 DEM/ESDF geometric provider).

- [ ] **Step 1: Write failing Catch2 tests (pins `3.094 MiB` budget, `ROG-Map #15`, `LTFJ J2`, & Review Focus #4)**
```cpp
TEST_CASE("Dual-Res ESDF <=3.15 MiB & <2 ms, WireCapsule O(1) grad, staleness >250 ms, and LTFJ J2 SLZ veto") {
  static_assert(sizeof(RobocentricEsdfMap) <= 3300000);
  // 1) Zero-init SlidingConfig, insert 5,760-pt wall at x=12 m, slide 10 m -> query(2,0,0) ~10 m in <2 ms.
  // 2) Wire (10,-5,20)->(10,5,20), R_w=3.5 m -> query(15,0,20) gives s_w=1.5 m, grad=[1,0,0].
  // 3) Intruder/ESDF age 0.75 s -> inflated by DeltaR_stale; age >1.0 s -> v_cap==8.0 m/s; age >2.0 s -> divert flag.
  // 4) SLZ at 40.9128 N, 29.1930 E (inside LTFJ J2 pinch) or sigma_z>0.15 m or slope>8 deg -> REJECTED; Kartal coast pad -> #1.
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_esdf_hazards_slz` → FAIL.
- [ ] **Step 3: Implement Dual-Res `ROG-Map`/FIESTA ESDF, `WireCapsules`, `IntruderTrackerKf`, and `SlzReachRanker`** — `0.5 m` log-odds ring buffer (`128×128×64`, `1.00 MiB`) driving `occ_cnt` (`0..8`) hysteresis into `1.0 m` (`64×64×32=131,072`) `16 B` bit-packed (`3×18b` DLL) FIESTA ESDF (`2.00 MiB`) + `3×8k` queues (`96 KiB`), zero-initializing `SlidingConfig` (`ROG-Map #15`). Implement `WireCapsuleTable`, 6-state CV-KF + exact `LTFJ`/`LTBX` polygon+arc exclusion, staleness inflation (`>250 ms`/`>1 s`/`>2 s`), and `SlzReachRanker` exposing `ISlzPriorProvider` (`σ_z<=0.15 m`, `θ<=8°`, `R>=5.0 m`, `E_req<=0.85 E_rem`).
- [ ] **Step 4: Verify pass** — `ctest --preset dev-linux-x64 -R test_esdf_hazards_slz --output-on-failure` → PASS.
- [ ] **Step 5: Commit** — `git add core/rta_companion/ tests/rta_companion/test_esdf_hazards_slz.cpp && git commit -m "feat(rta_companion): add Dual-Res 3.094 MiB ESDF, WireCapsules, KF intruders, and ISlzPriorProvider"`

---

### Task 10: PX4 v1.16 Internal `mc_indi_wls_control`, 72 B Envelope Msg & 4-Part `GZBridge` Patch

**Files:** Create `px4_overlay/msg/{VertirtaCapabilityEnvelope,VertirtaRtaSetpoint}.msg`, `px4_overlay/modules/mc_indi_wls_control/{CMakeLists.txt,mc_indi_wls_control.hpp,mc_indi_wls_control.cpp}`, `px4_overlay/patches/px4_v116_gz_bridge_multi_imu.patch`, `px4_overlay/config/dds_topics_vertirta.yaml`, `px4_overlay/airframes/{12050_gz_vertirta_x8_240,rc.vertirta_apps}` | Test `tests/flight_core/test_px4_overlay_contracts.py`
**Interfaces:** Consumes `vertirta_flight_core` (Tasks 2–5). Produces `VertirtaCapabilityEnvelope.msg` (exact `72 B` packed w/ `float32 chi2_residual` + `uint32 _pad`) on `/fmu/out/vertirta_capability_envelope` (`100 Hz`), `/fmu/in/vertirta_rta_setpoint`, and airframe `12050_gz_vertirta_x8_240` (`VERTIRTA_USE_INDI=0/1`).

- [ ] **Step 1: Write failing contract test (pins DR-1 errata #3 & #5)**
```python
def test_envelope_msg_is_72_bytes_and_airframe_locks_cel_params():
    msg = Path("px4_overlay/msg/VertirtaCapabilityEnvelope.msg").read_text()
    assert "float32 chi2_residual" in msg and sum_uorb_field_bytes(msg) == 72
    af = Path("px4_overlay/airframes/12050_gz_vertirta_x8_240").read_text()
    for p in ["CA_ROTOR_COUNT 8", "EKF2_MULTI_IMU 3", "SENS_IMU_MODE 0", "SYS_FAILURE_EN 1",
              "FD_FAIL_R 0", "FD_FAIL_P 0", "CA_FAILURE_MODE 1", "VERTIRTA_USE_INDI"]:
        assert p in af
```
- [ ] **Step 2: Run test to verify failure** — `pytest tests/flight_core/test_px4_overlay_contracts.py -v` → FAIL.
- [ ] **Step 3: Author PX4 v1.16 overlay, 4-part `GZBridge` patch, and `mc_indi_wls_control`** — Run `mc_indi_wls_control` at 400 Hz on `wq:rate_ctrl` linking `vertirta_flight_core` (`TmrVoter→IndiWlsController→MotorFdiObserver→TheveninBattery1RC→actuator_motors` + 100 Hz envelope). Patch `GZBridge.{hpp,cpp}` & `GZMixingInterfaceESC.cpp`: (a) `3× PX4Gyroscope/Accelerometer` distinct `DeviceId`s (inherits `failure_injection::process()`), (b) `imu_sensor_0..2` + `air_pressure_sensor_0..1`, (c) `sleep_count_limit 1000→5000`, (d) `_failure_injection_sub` zeroing cut rotors. Separate `VERTIRTA_USE_INDI=0` (stock baseline) vs `1` (`rc.vertirta_apps` excluding stock `mc_pos/att/rate/allocator`) and set `FD_FAIL_R=0, FD_FAIL_P=0`, `SIM_GZ_EC_MAX=240`.
- [ ] **Step 4: Verify pass & WSL2 PX4 build** — `pytest tests/flight_core/test_px4_overlay_contracts.py -v` and `PX4_EXTERNAL_MODULES_LOCATION=... make px4_sitl_default` → PASS.
- [ ] **Step 5: Commit** — `git add px4_overlay/ tests/flight_core/test_px4_overlay_contracts.py && git commit -m "feat(px4): add 400 Hz mc_indi_wls_control, 72B envelope msg, and 4-part GZBridge patch"`

---

### Task 11: Gazebo Harmonic `VertiRTA-X8-240` SDF, Coaxial/Thevenin & Dryden Plugins & Kartal World

**Files:** Create `sim/gz_plugins/CMakeLists.txt`, `sim/gz_plugins/src/{vertirta_coaxial_propulsion,vertirta_dryden_wind}.cpp`, `sim/models/vertirta_x8_240/model.sdf`, `sim/worlds/kartal_coast.sdf` | Test `tests/sim/test_gz_plugins.cpp`
**Interfaces:** Consumes `TheveninBattery1RC`, `CoaxEffectiveness`, `DrydenTurbulenceFilter`. Produces `libvertirta_coaxial_propulsion.so`, `libvertirta_dryden_wind.so`, `vertirta_x8_240/model.sdf`, `kartal_coast.sdf`.

- [ ] **Step 1: Write failing plugin & SDF validation test (pins DR-1 errata #6/#7 & Review Focus #5)**
```cpp
TEST_CASE("Coaxial plugin fixes climb inflow/tau_coast/0.96*k_Q0 and SDF defines lidar_link 60m + 3 IMUs") {
  // 1) Climbing (+5 m/s FLU Z) yields v_iu + v_air.Z() > v_iu (narrower wake skew chi); fault-off uses tau_coast_/drag.
  // 2) Lower rotor hover torque == 0.96*k_Q0_*w^2, thrust == 0.85*k_T0_*w^2.
  // 3) model.sdf has explicit <link name="lidar_link">, <max>60.0</max>, imu_sensor_0..2, air_pressure_sensor_0..1.
  // 4) kartal_coast.sdf: max_step_size=0.0025, datum (40.8860,29.1880), no duplicate world plugins, verified navsat datum.
}
```
- [ ] **Step 2: Run test to verify failure** — `cmake --build --preset dev-linux-x64 --target test_gz_plugins` → FAIL.
- [ ] **Step 3: Implement Gazebo Harmonic plugins, `vertirta_x8_240/model.sdf`, and `kartal_coast.sdf`** — Fix `v_iu + v_air.Z()` on climb, wire `tau_coast_`, apply `0.96*k_Q0_` lower swirl recovery, enforce `K_e,bus=1.36` voltage ceiling, define dedicated `lidar_link` (`<max>60.0</max>`, both `<gz_frame_id>`/`<frame_id>`), 3 heterogeneous 400 Hz IMUs (`ICM-42688-P`, `ICM-45686`, `BMI088`) + 2 baros + navsat + mag, and 400 Hz lockstep Kartal world (without duplicate global plugins loaded by PX4 `server.config`, verifying `navsat` elevation datum so EKF2 does not double-add `+36.7 m`).
- [ ] **Step 4: Verify pass & headless RTF** — `ctest --preset dev-linux-x64 -R test_gz_plugins --output-on-failure` and `HEADLESS=1 gz sim -s -r sim/worlds/kartal_coast.sdf --iterations 2000` → PASS (`>= 2.0× RTF`).
- [ ] **Step 5: Commit** — `git add sim/ tests/sim/test_gz_plugins.cpp && git commit -m "feat(sim): add Gazebo Harmonic coaxial/Thevenin/Dryden plugins, X8 SDF, and Kartal world"`

---

### Task 12: ROS 2 Jazzy `vertirta_*` Adapter Stack, 4-Thread SPSC RTA Node & Gate-2 Smoke Test

**Files:** Create `ros2_ws/src/vertirta_msgs/` (`msg/{SlzPrior,SlzPriorArray,WireCapsuleArray,IntruderArray,RtaStateSnapshot}.msg`, `srv/InjectFault.srv`, `action/ExecuteContingency.action`), `ros2_ws/src/vertirta_px4_bridge/` (`VertiRtaMode`, `VertiRtaExecutor`), `ros2_ws/src/vertirta_rta_node/`, `ros2_ws/src/vertirta_map_node/`, `ros2_ws/src/vertirta_fault_injector/`, `ros2_ws/src/vertirta_bringup/launch/vertirta_sitl.launch.py` | Test `tests/integration/test_spsc_tsan.cpp`, `tests/integration/test_gate2_closed_loop.py`
**Interfaces:** Consumes `vertirta::rta_companion`, `px4_msgs` (`release/1.16` + Task 10 `VertirtaCapabilityEnvelope`), `Auterion/px4-ros2-interface-lib` (`release/1.16`). Produces canonical ROS 2 contracts: Action `/vertirta/execute_contingency` (`vertirta_msgs/action/ExecuteContingency`), Topics `/vertirta/rta_state_snapshot` (`vertirta_msgs/msg/RtaStateSnapshot`, 1 Hz + on-fault), `/vertirta/slz_priors` (`vertirta_msgs/msg/SlzPriorArray`, plus Plan 3 `/vertirta/slz/gimbal_cmd`), `/fmu/out/vertirta_capability_envelope` (`px4_msgs/msg/VertirtaCapabilityEnvelope`, `72 B`), Foxglove `:8891`, `MicroXRCEAgent :8888`, and CRC-valid `.vrta` logs.

- [ ] **Step 1: Write failing TSan 4-thread SPSC test & Gate-2 closed-loop SITL test (pins Review Focus #3 & #5)**
```python
# tests/integration/test_gate2_closed_loop.py (plus tests/integration/test_spsc_tsan.cpp: 4-thread SPSC+seqlock 1M ops, 0 races)
def test_gate2_imu_fdi_csfl_and_same_arm_spin_vs_stock_px4():
    # 1) Stock PX4 (VERTIRTA_USE_INDI=0) w/ M0+M4 cut @ 35 m AGL -> verifies capsize (roll/pitch > 60 deg)
    # 2) VertiRTA (VERTIRTA_USE_INDI=1): (a) IMU0 stuck -> DUPLEX_2OF2 <=20 ms; (b) M0+M5 cross-arm -> CSFL 0 CBVF/CTR breach;
    #    (c) M0+M4 same-arm -> irreversible CEL latch, Option B1b flat spin (~2.2 rad/s), lands |v_z|<=2.5/>10m, <=0.8 contact, valid .vrta!
```
- [ ] **Step 2: Run tests to verify failure** — `ctest --preset tsan -R test_spsc_tsan` and `pytest tests/integration/test_gate2_closed_loop.py -v` → FAIL.
- [ ] **Step 3: Implement ROS 2 Jazzy `vertirta_*` packages & bringup** — Build `vertirta_px4_bridge` on `px4-ros2-interface-lib` (`VertiRtaMode : ModeBase` @ `100 Hz` publishing `AttitudeSetpointType` in CSFL and `VehicleThrustSetpoint`/`VertirtaRtaSetpoint` in CEL; `VertiRtaExecutor : ModeExecutorBase` hosting `/vertirta/execute_contingency`, calling `deferFailsafesSync(true, 60)` on CEL, and applying #201/#167/#191 mitigations: persistent node + monotonic `model_only` reset, no mid-callback reschedule, isolated namespaces). Build `vertirta_rta_node` with 4 threads (Thread 1 DDS `SingleThreadedExecutor` publishing `/vertirta/rta_state_snapshot` ↔ `alignas(64)` lock-free SPSC ↔ Thread 2 `SCHED_FIFO` 100 Hz zero-ROS zero-heap RT loop ↔ Seqlock Thread 3 `vertirta_map_node` 20 Hz ESDF subscribing `/vertirta/slz_priors` ↔ Thread 4 async `.vrta` writer). Configure `vertirta_bringup` (Foxglove `:8891`, `MicroXRCEAgent :8888`).
- [ ] **Step 4: Verify pass** — `cd ros2_ws && colcon build --symlink-install && ./src/px4-ros2-interface-lib/scripts/check-message-compatibility.py`, `ctest --preset tsan -R test_spsc_tsan`, and `pytest tests/integration/test_gate2_closed_loop.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add ros2_ws/ tests/integration/ && git commit -m "feat(ros2): add px4-ros2-interface-lib bridge, 4-thread SPSC RTA node, and Gate-2 test"`

---

## Cross-Plan Interfaces Produced by Plan 1A

- **For Plan 1B (Assurance, CI, 2-Tier Atlas & Earth Viz):** CMake targets `vertirta_flight_core` (`ALIAS vertirta::flight_core`, C++17 MCU-gated) & `vertirta_rta_companion` (`ALIAS vertirta::rta_companion`, C++20); glibc-safe CLI targets `vertirta_replay_cli`, `vertirta_bench_cli`, and `vertirta_6dof_batch_cli` (`tools/cli/`); canonical binary schema `VrtaFileHeader` (`128 B`, `magic="VRTA0001"`) + `VrtaFrame100Hz` (`256 B` + CRC32C); `config/*.yaml`, `12050_gz_vertirta_x8_240` (`VERTIRTA_USE_INDI=0/1`), `sim/models/vertirta_x8_240/model.sdf`, and `sim/worlds/kartal_coast.sdf` (`world_name="kartal_coast"`).
- **For Plan 2 (`v2.0.0-alpha` TR/EN Co-Pilot):** ROS 2 Action `/vertirta/execute_contingency` (`vertirta_msgs/action/ExecuteContingency`, `HOLD_LOITER` fallback, isolated from Thread 2); `/vertirta/rta_state_snapshot` (`vertirta_msgs/msg/RtaStateSnapshot`, 1 Hz, `400 V / 22 kWh`, `chi2_residual`, ranked `SlzPrior`s); `/fmu/out/vertirta_capability_envelope` (`px4_msgs/msg/VertirtaCapabilityEnvelope`, `72 B`); root CMake optional subdir hooks for `core/stl` & `core/copilot`; test-only `ScopedZeroHeapGuard` (`tests/flight_core/zero_heap_guard.hpp`).
- **For Plan 3 (`v2.0.0` Active Visual SLZ):** Canonical C++20 `vertirta::rta::ISlzPriorProvider`, `SlzPrior`, and `SlzProviderSnapshot` (`core/rta_companion/include/vertirta/rta/slz_prior_provider.hpp`), `vertirta::core::FlightRegime`, `vertirta::Vec3Enu`; ROS 2 `/vertirta/slz_priors` (`vertirta_msgs/msg/SlzPriorArray`, plus `/vertirta/slz/gimbal_cmd`) & `/fmu/out/vertirta_capability_envelope` (`px4_msgs/msg/VertirtaCapabilityEnvelope`, `72 B`); standalone `CoaxialPlant6Dof` + `CbvfSafetyFilter` C++20 runtime + root CMake hooks for `core/slz_bayes` & `sim/colab_egl`.
