# VertiRTA Phase 3 Forensic Errata, Cross-DR Conflict Resolution & Hard Locks (`DR-08..DR-13`)

**Status:** FROZEN CANONICAL OVERRIDE (`v3.0.0 -> v4.0.0` Baseline)  
**Companion To:** [00-DR-FORENSIC-ERRATA-AND-LOCKS.md](00-DR-FORENSIC-ERRATA-AND-LOCKS.md) (`DR-01..DR-07`)  
**Audited Inputs:** `DR-08` through `DR-13` (293 KB total, 6/6 reports audited end-to-end against frozen `v2.0.0` truth).

---

## 1. Completeness & Extraction Verdict (Google Docs Export)

All 6 Phase 3 Deep Research reports arrived intact and were extracted verbatim as pure-LF UTF-8 Markdown into `docs/research/`:

| Report | File | Size | Stripped Inline Math/Icons (`[STRIPPED_MATH_OR_ICON]`) | Verdict |
| --- | --- | ---: | ---: | --- |
| `DR-08` | `DR-08-stm32h7-renode-cyphal-esbmc-misra.md` | 51.2 KB | 1 | **NON-BLOCKING** — All hardware errata (`ES0392 Rev 16`), MPU `R0..R7`, `o1heap v3.0.0` `32 KiB`, Cyphal `Subject-ID 7509/1024`, Renode `SetGlobalQuantum "0.000025"`, and ESBMC/linter flags intact. |
| `DR-09` | `DR-09-lift-cruise-openfoam-pod-bvi-acoustics.md` | 49.4 KB | 244 | **NON-BLOCKING** — Standard equations (`IAirframeDynamics`, WLS, quintic Hermite, NASA RP-1218 BPM, Amiet-Lowson, ISO 9613-1/2:2024, POD-RBF) and dBA tables intact. Optional raw `.md` re-export nice-to-have before `M2 (v3.5)` only for author's exact 2025 BPM non-compactness / `Delta L_BVI` / `VrtaWndNode64[0..31]` channel table (otherwise cleanly re-derived against NASA TM-2018-219895). |
| `DR-10` | `DR-10-forward-radar-vision-gnss-denied-vio.md` | 49.3 KB | 63 | **NON-BLOCKING** — All license verdicts, thresholds, bounds, and physics tables intact; closed-form radar/MCUSUM/CBVF equations standard. |
| `DR-11` | `DR-11-multi-evtol-uspace-part-is-cbf-orca.md` | 56.0 KB | 54 | **NON-BLOCKING** — All schemas bind to upstream ASTM `utm.yaml`, `opendroneid-core-c`, RFC 9575, RFC 6479, 1-RC thermal, 2nd-order ECBF (`k1=1.0, k2=1.4`), and RHR curl. |
| `DR-12` | `DR-12-voice-atc-fleet-z3-eccairs2-forensics.md` | 49.8 KB | 29 | **NON-BLOCKING** — Every contract recovered from Section 5 summary table, `Ren et al. CoRL 2023` Mondrian conformal, `QF_RDL`/`QF_LRA`, and `CDR (EU) 2020/2034` ERCS text. |
| `DR-13` | `DR-13-vertigcs-wasm-nanobind-sdk-opencore-papers.md` | 48.1 KB | 5 | **NON-BLOCKING** — All 5 markers were decorative CLI icons; every Emscripten/MapLibre/Deck.gl/nanobind/Beta-Binomial floor is explicit in prose. |

**Open Field Verification (Non-Blocking for `M1 v3.0`):**
- Verify `LTBX` (Samandira) ARP latitude (`40°59'26"N` = `40.990556°N` frozen in `istanbul_ltfj_airspace.yaml` vs `40°59'56"N` = `40.99889°N` in `DR-10/11`, `927 m` delta) against the official DHMI Turkey AIP PDF before `M2 (v3.5)`.

---

## 2. Master Cross-DR Conflict & Canonical Override Table

Whenever `DR-08..DR-13` drift from our frozen `v2.0.0` ground truth (`config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`, `artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json`, `core/flight_core/`), the **Canonical Winner** below strictly overrides the DR text:

| # | Conflict Domain | Drift Found in `DR-08..DR-13` | Canonical Winner & Mandatory Engineering Override |
| --- | --- | --- | --- |
| 1 | **Coaxial X8 Geometry, Inertia, RPM & Spin** | `DR-09/11`: `D=1.65 m (R=0.825 m)`, `L=1.55 m`, `I=[92,92,158]`, `tau=35 ms`, `10-20 deg` tilted spin; `DR-10`: `4-5 m` span, `25 m/s`; `DR-12`: `1,850 RPM` cap, `90 kt`, `4-6 pax`. | **`config/vehicle_x8_240.yaml`:** `R=0.75 m (D=1.50 m)`, `L=1.35 m`, `R_veh=2.10 m`, `z/D=0.233`, `eta_coax=0.85`, `I=[85,95,140]`, `tau_up/down=85/120 ms`, `omega_max=240 rad/s (2,292 RPM)`, `v_max=22.0 m/s (42.8 kt)`, `1` pax, `240 kg MTOW`, Option B1b flat spin `|Omega_z|=2.2 rad/s`. (`90 kt` applies strictly to `DR-09` Lift+Cruise.) |
| 2 | **HV Battery Pack & Thermal Scaling** | `DR-11`: `112S11P` (`98 kg` pack — breaks `240 kg` MTOW) and lower `R0/R1`. | **`112S12P` (`68 kg, 21.77 kWh, 403.2 V nom, R0=0.1363 Ohm, R1=0.0840 Ohm`):** Scale `DR-11`'s pad-side `15 C` active liquid chiller (`260 W/K`, `40/45/50 C` derate) to `68 kg` (`C_th ~= 70.7 kJ/K`). Apply `25% / 20%` as U-Space dispatch policy strictly above our `10% / 12%` airborne reserve floor. |
| 3 | **Istanbul Geodesy, Pads & `LTFJ` CTR** | `DR-09`: ISTMarina (`29.2057 E`) lies **`+1.14 km` INSIDE `LTFJ` CTR**; `DR-11`: `N=37.35 m`, second ENU origin `(40.9080, 29.1470)`, eastern `29.1880 E` wall slicing Kartal; `DR-12`: stale Kartal `40.8880 N`; `DR-13`: Bosphorus CLI demo. | **`config/istanbul_ltfj_airspace.yaml`:** Kartal ENU datum `(40.8860 N, 29.1880 E, 12.0 m AMSL, N=+36.7 m)`, exact `LTFJ` NW `J2` polygon+arc (`max_lon_near_j2=29.1895 E`), `<1 KB` `5'x5'` EGM2008 table anchored to `36.7 m` at Kartal. Keep CFD/fleet/CLI domains strictly west of `29.1895 E`; add Cevizli west of `29.188 E` + **Kartal Dr. Lutfi Kirdar Sehir Hastanesi active helipad** as a Tier-2 hard separation cylinder separate from its soft acoustic penalty. |
| 4 | **400 Hz INDI Placement & PX4 Version** | `DR-09`: 400 Hz INDI-WLS offboard over uXRCE-DDS (`UXRCE_DDS_SYNCT=false`); `DR-10`: PX4 `v1.18` (`vehicle_gnss`, `SENS_GNSS0_DELAY`). | **PX4 `v1.16` internal `wq:rate_ctrl` (`UXRCE_DDS_SYNCT=1`, `vehicle_gps_position`, `EKF2_GPS_DELAY`):** Implement `Lift+Cruise 8+1` inside `core/airframes/lift_cruise/` + PX4 VTOL internal rate overlay; never run 400 Hz inner loop offboard. |
| 5 | **Cortex-M7 `flight_core` & `.vrta` Freeze** | `DR-12`: embed ASR transcripts, Z3 traces, and Ed25519 signatures into 100 Hz `.vrta` frames. | **HARD FREEZE `28,886 B Flash / 0 B RAM` `libvertirta_flight_core.a` & `256 B VRTA0001` binary frames!** Keep `o1heap` `32 KiB` DTCM arena strictly in BSP/Cyphal bus wrapper (`firmware/`). Log Voice/Z3/GSN/chained-SHA256/Ed25519 forensics strictly in companion append-only **`.vrta_events.jsonl`** sidecar linked by `(seq, t_sim_us)`. |
| 6 | **Physical TU116 GPU & Nadir Camera** | `DR-10`: projects `2x` TU116 FP16 speedup, `500-990 MB` ORT-CUDA VRAM, `7.6M` PIDNet-S params, `1280x720` nadir camera. | **Live `artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json`:** Physical GTX 1650 SUPER measured **`126.0 MiB` ORT-CUDA FP32+CUDA Graph (`8.06 ms` p50 / `9.06 ms` p99)** while FP16 regressed **`3.2x` slower (`28.67 ms`)**; deploy params = `8.00M`; locked nadir = `512x512` (`80 deg` HFOV, `fx=305.09 px`). Keep mandatory empirical TU116 FP32-vs-FP16 benchmark gate. |
| 7 | **Conformal Calibration, `UAVid` & Papers** | `v2.0` pooled `n_cal=200` (`q_lvl=0.955`) under-covers Turkish under exact Beta inversion; `DR-12` mistakes `0.955` for `1-alpha` and drops `<5%` SoC override; `DR-13` cites banned `UAVid`. | Lock `alpha=0.05` + `<5.0%` SoC margin `LIVENESS_EMERGENCY_COMMIT`. Freeze `VertiBench-v1`; upgrade **`VertiBench-v2`** to language-group-conditional **`N_cal >= 750-1,000/lang`** (`1,500-2,000` total) + SNR Mondrian (`n_g >= 250/stratum`). Keep **`UAVid` HARD-BANNED** (use `AeroScapes`/`ICG`/`UAVScenes`). Keep `docs/paper/` as arXiv/Zenodo omnibus; split peer-reviewed submissions into **Paper 1 (IEEE RA-L -> IROS 2027/ICRA 2028)** and **Paper 2 (CoRL 2027)**. |
| 8 | **Intruder Slots & `viz/cesium/` Continuity** | `DR-11` requires `19` peers (`R_z=36 m` downwash) vs `IntruderTrackerKf::kMaxTracks=16`; `DR-13` switches primary GCS to MapLibre+Deck.gl. | Keep `CbvfSafetyFilter` (`StaticDaqpSolver<5,32>`) untouched: `core/uspace_fleet/` runs 20-agent tracking and feeds **Top-10 CBF-ORCA + 2 FATO + 2 CTR rows (`<= 14 <= 15` slots)** into RTA. Build unified `apps/vertigcs/` on **MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4**, while retaining `viz/cesium/` behind `IGeoViewportAdapter` for enterprise 3D globe mode. |

---

## 3. Master License, Silicon, Security & Regulatory Firewall (`v2.0 -> v4.0`)

### 3.1 New Hard Bans (Zero Tolerance in Runtime, Weights, or Bundled Datasets)
- **VIO / SLAM:** `OpenVINS` (**GPL-3.0**), `VINS-Fusion` (**GPL-3.0**), `HybVIO` (**GPL-3.0**), `ORB-SLAM3` (**GPL-3.0**), `sqrtVINS` (**LGPL-3.0**).
- **Synthetic Rendering & CFD Air-Gap:** `BlenderProc` (**GPL-3.0** / `bpy` script GPL policy) -> never import in runtime or Apache-2.0 training pipelines; keep existing offline `viz/blender/` and new `tools/cfd_offline_gpl/` (`OpenFOAM v2412/v2606`, GPL-3.0) strictly air-gapped offline generating pure numeric/media assets (`VRTAWND1`), verified by an `ldd`/`nm` **zero-GPL symbol CI gate** on every runtime binary.
- **Voice ASR/TTS & Audio Datasets:** `OHF-Voice/piper1-gpl` (`piper-tts >= 1.7.0`, GPL-3.0), **ALL prebuilt `tr_TR` Piper voices** (`dfki` `CC BY-NC-SA`, `fahrettin`/`fettah` NC `lessac` lineage), `en_US-lessac`/`ryan`/`hfc`/`l2arctic`, `mms-tts-*`, `XTTS-v2`, `OmniVoice`, `Supertonic`, `matcha-icefall` (`#4265`), `UWB-ATCC`, `ATCO2`, `ESC-50`, `DEMAND`, unfiltered `FSD50K`. Quarantine `ATCOSIM` (unverified ISO license) and audit `pocket-tts-turkish` TR-Wiki `CC BY-SA` lineage before any redistribution.
- **Vision / Localization Datasets & Models:** `UAVid` (banned since `DR-7`), `DINOv2-SALAD` (`GPL-3.0`), `Sample4Geo` (unlicensed; taints `CAMP`/`DAC`), `EfficientLoFTR` (custom), `Deep Hough Transform` (`CC BY-NC`), `Ultralytics` (`AGPL-3.0`), `InsPLAD`/`DDOS`/`Drone-vs-Bird` (`NC`), `STN PLAD`/`PAMPC` (`GPL-3.0`). Quarantine `PLDU/PLDM`, `SynthWires`, `WireSegHR`, `SPL`, `SOD4SB`, `Seraphim`.

### 3.2 Mandatory Clean Stack, Version Pins & Hardware/Regulatory Locks
- **Bare-Metal Silicon & Formal (`DR-08`):**
  - STM32H7 Rev V check **`DBGMCU->IDCODE == 0x2003`** (reject Rev Y `0x1003` due to no-workaround FDCAN RAM corruption Erratum `2.24.2`); 480 MHz `VOS0` **`FLASH_ACR.LATENCY = 4, WRHIGHFREQ = 0b10`**; bottom `1 KB` ITCM (`0x00000000-0x000003FF`) MPU `AP_NONE/XN=1` null trap + `VTOR = 0x00000400`; forbid Write-Through MPU (Erratum `2.1.1`); disable FPU lazy stacking (`LSPEN = 0`); unlock `DWT->LAR = 0xC5ACCE55`; poll `FPSCR[4:0]` (Erratum `2.1.6`).
  - OpenCyphal CAN-FD: **MIT** `libcanard v3.2+/v4` + **`o1heap v3.0.0`** (`32 KiB` DTCM in BSP wrapper) + **Apache-2.0** `CMSIS-Core` + **Nunavut C11** (never experimental C++). Renode `v1.17.0` (`SetGlobalQuantum "0.000025"`) = functional SIL only; physical `DWT->CYCCNT` = Empirical HWM only.
  - **`ESBMC v8.7 --32 --floatbv --nan-check --overflow-check`** strictly on **`ubuntu-24.04` x86_64** (ARM64 fails `--32`); framed honestly as **Supplemental Engineering Verification** (zero `DO-330` TQSP claim; Clang vs GCC `14.2.1` mismatch `FM.A-7[9]`).
  - Static Profile: **`clang-tidy 22` + `cppcheck 2.22.0 --platform=unix32 --funsigned-char`** labeled strictly **"MISRA-Aligned High-Integrity C++17 Profile"** + 5-class gap register (never claim OSS MISRA C++:2023 compliance).
- **Forward Perception & GNSS-Denied (`DR-10`):**
  - Primary VIO: **`Basalt` (`BSD-3`)** (or `OKVIS2` `BSD-3` ONLY with `USE_NN=OFF` and GPL `SuiteSparse/CHOLMOD` stripped); Visual Reloc: **`XFeat + LighterGlue` (`Apache-2.0`)** + **`MegaLoc` (`MIT`)**; Forward Wire Dataset: **`TTPLA` (`Apache-2.0`, decoupled `45 deg` forward wire head ONLY)** + **`Mendeley TEIAS` (`CC BY 4.0`)** + **`Purdue UAV` (`BSD-3`)** + Gazebo OGRE2 (`Apache-2.0`).
  - Scale realism: `10 m` Sentinel-2 is gross-blunder only (`~45 m` CEP50); local VPS requires `0.5-1.0 m` ortho (sim MBTiles cache or licensed GCP).
- **U-Space Fleet & Part-IS Cybersecurity (`DR-11`):**
  - **`Part-IS (EU) 2023/203` (`NIS2 != Part-IS`)**: patch OMG DDS Security 1.2 `IV_s` replay gap via **`40 B VertiRtaSecureHeader` + `ReplayWindow128`**, Mode B crypto (AES-256-GCM 20 Hz + Ed25519 1 Hz & P0), **`BEST_EFFORT 3x` UDP burst** for P0, separate SHM MC vs UDP-Crypto profiles.
  - Mandatory CVE pins: **`libsodium >= 1.0.21`** and **`Fast DDS >= 2.14.6 / >= 3.4.1`** (`CVE-2024-28231` / `2025-62799` / `2026-22590`).
- **Voice ATC & Forensics (`DR-12`):**
  - Mandatory pin: **`sherpa-onnx >= 2.0.0`** (PR `#4055` removes GPL-3.0 `espeak-ng`/`piper-phonemize`); ASR = **`whisper.cpp` (`MIT`)** (`small` fast tier + `large-v3-turbo` `audio_ctx=256-384`, `translate=false`, `no_context=true`, `carry_initial_prompt=true`) + `Silero VAD` (`MIT`); clean TTS = `en_US-ljspeech-medium` (PD) + `canberkkkkkk/ema-lightning` (`Apache-2.0`) / `ahakan/okur-tts` (`Apache-2.0 + CC BY 4.0`) / clean-room TR VITS on `sherpa-onnx >= 2.0.0`.
  - Regulatory Forensics (`CIR (EU) 2026/1821`, `SHGM SHY-OLAY`, `ED-155A`, `CDR 2020/2034` ERCS): append-only `.vrta` + `.vrta_events.jsonl` -> **draft-only `Incident`** (`ID 431`, never auto-`Accident`/`Serious Incident`), rotor/control/SW faults mapped to CICTT **`SCF-NP`**, ERCS Barrier `#8 = 0` on high-energy MAC/LOC-I, zero-digit GBNF + `verify_numeric_grounding()`.
- **Productization & Governance (`DR-13`):**
  - **Open-Core Governance:** Permanent **Apache-2.0 + DCO 1.1** (NO CLA) on `flight_core`, `.vrta`, and single-operator `VertiGCS`.
  - **`VertiGCS` Web/WASM:** **MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4 `MapView` (`z >= 8`)** + single-threaded Emscripten `-sENVIRONMENT=worker` flat `extern "C"` `.wasm` (`< 150 KB`; **HARD BAN `-pthread`, `coi-serviceworker`, `embind`**) + self-hosted `.pmtiles` + Copernicus GLO-30 (`DOI 10.5270/ESA-c5d3d65`, EGM2008->HAE).
  - **SDK & Packaging:** `nanobind` + `scikit-build-core` + `cibuildwheel v4` (`cp312-abi3`, skip `cp31?t-*`) + `FILE_SET HEADERS` `VertiRTAConfig.cmake` + read-only non-root `docker compose`.
  - **Standards Registry:** Bump `docs/standards_registry.yaml` to `v2.0.0` adding `Part-IS (EU) 2022/1645` & `2023/203`, `CIR (EU) 2026/1821`, `CDR (EU) 2020/2034`, `EUROCAE ED-155A`, `SHGM SHY-OLAY`, and `SHGM SHT-21 Ek-6`.

---

## 4. Clean 3-Release Expansion Roadmap (`M1 v3.0.0` -> `M2 v3.5.0` -> `M3 v4.0.0`)

| Milestone | Scope & Owned Modules (0 B Added to `28,886 B` `flight_core`) |
| --- | --- |
| **`M1 (v3.0.0)` — Productization + Silicon/Formal Core** | `apps/vertigcs/` (MapLibre v6 + Deck.gl v9.4 + `<150 KB` WASM Worker: GH Pages sandbox + `:5180/:8891` live ops + `IGeoViewportAdapter` keeping `viz/cesium/`), `sdk/python/` (`pip install vertirta` `nanobind` + `VertiVecEnv` Gymnasium + Typer CLI + `VertiRTAConfig.cmake` + non-root `docker compose`), `firmware/stm32h7_hitl/` + `firmware/renode_cyphal/` (`<20 ms` CAN-FD failover CI), `verification/formal/` (`ESBMC v8.7` x86_64 CI + MISRA-aligned profile), `docs/standards_registry.yaml v2.0.0`, DCO 1.1 governance, `docs/paper_ral/` (Paper 1 double-anonymous + McNemar/`rliable`). |
| **`M2 (v3.5.0)` — Multi-eVTOL U-Space + Full-Envelope Physics** | `core/uspace_fleet/` (`1-2` Hero PX4 + `19` SoA C++20 agents `0.37-0.57 ms/tick`, Top-10 3D CBF-ORCA + RHR curl, `<500 ms` P0 preemption, `Part-IS` `40 B` header + `ReplayWindow128` + Mode B crypto + `RTO<=10 s` FSM, 4-stand `15 C` liquid-chilled vertiports + `FATO-EMG`), `core/fleet_z3/` (`QF_RDL` STN `<3 ms` + bounded `|N_local|<=6` `QF_LRA` `<25 ms` p99), `ros2_ws/src/vertirta_uspace_node/` (ASTM F3548-21 / F3411-22a / RFC 9575), `core/airframes/lift_cruise/` (`IAirframeDynamics` + `8+1` hover-OEI `<=79.2% T_max` + PX4 VTOL overlay), `tools/cfd_offline_gpl/` -> `core/wind_acoustics/` (`13.89 MiB VRTAWND1` + `eps_rom_p95` CBVF inflation + `<1 ms` 180-node BVI/BPM/ISO 9613 soft noise cost). |
| **`M3 (v4.0.0)` — GNSS-Denied / Forward Hazards + Voice ATC & Forensics** | `core/forward_hazard/` (`45 deg` IMU-stabilized split-stack `1280x320->640x640` + `M-LSD-WireSeg`/`D-FINE-N` + custom Gazebo `MmWave77GHzSensor` -> `WireCapsuleTable`/`IntruderTrackerKf`), `core/gnss_denied/` (`Basalt` BSD-3 VIO + 4-hypothesis Lagged 2D MCUSUM + `XFeat+LighterGlue` + `3sigma` CBVF `dh/dt` inflation), `core/voice/` (`whisper.cpp` dual-tier + `sherpa-onnx >= 2.0.0` `<250 ms` TTFA + 4-stratum SNR Mondrian KnowNo), `core/forensics/` (`.vrta_events.jsonl` -> deterministic `E5X`/ERCS + zero-digit GBNF), `VertiBench-v2` (`N_cal >= 750-1,000/lang` + SNR Mondrian), `docs/paper_corl/` (Paper 2 double-anonymous, zero `UAVid`). |
