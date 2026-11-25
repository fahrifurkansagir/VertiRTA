# VertiRTA `v1.0.0` — Technical System Card

> **Principal-Engineer Executive Brief · Research & Simulation Testbed Only**  
> **License:** Apache-2.0 (permanent open-core lock) + DCO 1.1 (`Signed-off-by:`, No CLA) · **Zero Airworthiness / Certification Claims**  
> **Live Browser Sandbox:** [`https://fahrifurkansagir.github.io/VertiRTA/`](https://fahrifurkansagir.github.io/VertiRTA/) (`38,470 B` `.wasm` Worker, zero tokens)  
> **Deep Audits:** [`docs/research/CROSS_BENCHMARK_AUDIT_v1.md`](research/CROSS_BENCHMARK_AUDIT_v1.md) · [`docs/research/OPTIONAL_G4_V1_1_BATCH_SPEC.md`](research/OPTIONAL_G4_V1_1_BATCH_SPEC.md) · [`docs/SAFETY_CASE.md`](SAFETY_CASE.md)

---

## 1. System Identity & Legal Scope

`VertiRTA` is an open-source, deterministic flight-control and `ASTM F3269-21` 3-tier Simplex Run-Time Assurance (RTA) testbed engineered for two `240 kg` urban eVTOL reference airframes — coaxial **`VertiRTA-X8-240`** and transition **`VertiRTA-LC-240` (`Lift+Cruise 8+1`)** — operating along the Istanbul Maltepe–Dragos–Kartal coastal corridor bounded by real DHMİ AIP Turkey **`LTFJ`** (Sabiha Gökçen, `< 29.1895° E`) and **`LTBX`** (Samandıra, `40°59'26"N, 029°12'58"E`) controlled airspace walls plus the Kartal Dr. Lütfi Kırdar Hospital Tier-2 helipad keepout cylinder (`160 m`).

- **Zero Certification Claims:** References to EASA `SC-VTOL-02`, `MOC-5`, `Part-IAM (EU) 2024/1111`, `Part-IS (EU) 2023/203`, `SORA 2.5`, `SHGM SHT-İHA`, `DO-178C`, `DO-254`, and `SCSC-141C` are architectural design drivers ONLY. `G5` (physical hardware qualification) and `G6` (certification lifecycle / RTOS WCET) remain permanently `[UNDEVELOPED]` in [`docs/SAFETY_CASE.md`](SAFETY_CASE.md).
- **Strict AI Isolation (`DAL E`):** All neural perception (`PIDNet-S`, `M-LSD-WireSeg`, `D-FINE-N`, `XFeat`) and bilingual LLM/voice modules (`Qwen3-4B`, `faster-whisper`, `sherpa-onnx`) operate outside the deterministic RTA boundary with **zero closed-loop actuation authority**.

---

## 2. Core Deterministic Flight & Assurance Stack

| Subsystem | Verified Metric (`v1.0.0`) | Gate / Contract | Artifact |
| --- | --- | --- | --- |
| **Cortex-M7 Flight Core (`libvertirta_flight_core.a`)** | **`28,886 B`** Flash (`.text+.data`), **`0 B`** static RAM, **`1,496 B`** max frame, **`2,960 B`** static call-graph DAG, **`0`** heap, **`0`** recursion | `< 256 KiB` Flash, `0` heap, `0` recursion (`arm-none-eabi 14.2.1`) | [`bare_metal_size_cortex_m7.json`](../artifacts/v1.0.0/benchmarks/bare_metal_size_cortex_m7.json) |
| **Supplemental BMC (`ESBMC v8.7.0 --32`)** | **`1,905 / 1,905`** properties `VERIFICATION SUCCESSFUL` (`6/6` direct-source harnesses, `0` failed) | `0` failed properties (supplemental, non-`DO-330`) | [`.github/workflows/formal_and_hil.yml`](../.github/workflows/formal_and_hil.yml) |
| **Dual-FCC CAN-FD SIL (`Renode v1.17.0`)** | **`3 / 3`** scenarios PASS, active-to-standby failover **`12.50 ms < 20.0 ms`**, `4` babbling-idiot CRC rejects | Functional SIL (`libcanard` + `o1heap v3.0.0` outside `flight_core`) | [`firmware/renode_cyphal/`](../firmware/renode_cyphal/CMakeLists.txt) |
| **Cross-ISA & Live Browser `.wasm`** | `x86_64 <-> arm64` **`0.0`** max diff (`p99 = 0.736 us` Cobalt 100); **`38,470 B`** single-threaded `.wasm` Worker on GitHub Pages | `< 1e-12` cross-ISA, `< 150 KiB` WASM, `0` SAB/COI | [`tier2_cross_isa.json`](../artifacts/v1.0.0/benchmarks/cloud_ci/tier2_cross_isa.json) |
| **`102.85M`-Node Dual-Airframe 4D JAX PDE** | `101×71×101×71` (`51.42M` nodes × 2 airframes, `435.73 s`), **`0`** violations, `soundness_passed: true` | Monotone `V_nom >= V_oei` (`0.0 m` max viol) | [`g4_jax_4d_hires_and_lc_manifest.json`](../artifacts/v4.0.0/g4_jax_4d_hires_and_lc_manifest.json) |
| **Zero-GPL & Test Suite Hygiene** | **`7 / 7`** GitHub Actions CI workflows, **`36 / 36`** CTest, **`17 / 17`** Vitest (`5` files), **`0`** GPL symbols across runtime targets | Strict Apache-2.0 runtime (`check_zero_gpl_symbols.py`) | [`scripts/ci/check_zero_gpl_symbols.py`](../scripts/ci/check_zero_gpl_symbols.py) |

---

## 3. Dual-Airframe `1,000 + 180` Native C++20 Monte Carlo Dispersion & Failure-Boundary Proof

Executed live via native C++20 (`master_seed = 20261125`, `12` threads, `11.11 s` wall-clock; [`monte_carlo_1000_dispersion_report.json`](../artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json), plot [`vertirta_mc1000_envelope_dispersion.png`](assets/vertirta_mc1000_envelope_dispersion.png)):

- **Zero Airspace Breaches:** **`0 / 1,000`** (`0.00%`, 95% one-sided CP upper bound `0.002991`) across `LTFJ` (min margin `941.0 m`), `LTBX` (`2,240.8 m`), and Kartal Hospital (`2,890.7 m`).
- **`VertiRTA-LC-240` (`8+1` Lift+Cruise):** **`500 / 500` (`100.0%`)** safe recoveries across all 5 operational regimes (`Nominal Turbulent`, `1M CSFL`, `Same-Arm Dual-Kill`, `GNSS Slow-Ramp Spoof`, `Compound Low-SoC + Wind + Fault`) with **`CEP50 = 0.80 m` / `CEP95 = 1.97 m`** (`max_td_err = 3.52 m`, `vz_p99 = 0.81 m/s`).
- **`VertiRTA-X8-240` (`429 / 500`, `85.8%`) — Honest CG-Shift Contrast:** Passes **`400 / 400` (`100.0%`)** across `Nominal Turbulent`, `1M CSFL`, `GNSS Slow-Ramp Spoof` (`200/200` spoof flights detected across both airframes, `mean TTD = 0.178 s`, `max TTD = 0.295 s`), and `Compound Low-SoC + Wind + Fault`. Under **unmodeled `±5 cm` lateral CG shift** combined with **same-arm coaxial dual-motor kill (`CEL`)**, active flat-spin recovery drops to **`29 / 100`** (`CEP50 = 41.08 m`) — explicitly contrasted against **nominal-CG Tier-1 10k `1,500 / 1,500` (`100.0%`)** ([`paired_30k_ral_stats.json`](../artifacts/v3.1.0/paired_30k_ral_stats.json)). This exposes a genuine physical control authority cliff (uncompensated off-axis CG torque during 3-boom flat spin) rather than hiding it.
- **`6×6` Thrust-Saturation Failure-Boundary Sweep (`180` Flights, `X8-240` Same-Arm CEL):** `0%`–`+7%` mass overload achieves `100%` survival across tested gusts (`13` green cells); first breakdown cliff occurs at **`+14%` mass overload @ `10 m/s` gust** (`5` partial transition cells); **`>= +21%` mass overload** hits a hard **`0%` thrust-saturation wall** (`18` red cells).

---

## 4. Zero-GPL Native C++20 MAVLink v2 UDP/Offboard RTA Shield

Clean-room Apache-2.0 C++20 MAVLink v2 wire-protocol shield (`0xFD` framing, `8` `common.xml` `CRC_EXTRA` message IDs, zero GPL/MAVSDK dependencies; [`tools/mavlink_rta_bridge/`](../tools/mavlink_rta_bridge/CMakeLists.txt), [`mavlink_v2_rta_bridge_proof.json`](../artifacts/v4.0.0/mavlink_v2_rta_bridge_proof.json)):

- **Live POSIX UDP Loopback Audit (`2,500` packets, `922.56 ms`):** **`0`** valid packets dropped, **`0`** unsafe packets leaked to FCU, **`0`** corrupted packets leaked.
- **100% Adversarial Interception:** **`1,181 / 1,181`** unsafe offboard setpoints clamped (`566` CBVF, `861` `LTFJ`/`LTBX`/Hospital NFZ, `689` plunge/overspeed), **`40 / 40`** deadman heartbeat-timeout overrides, **`240 / 240`** airborne `DISARM`/`FLIGHT_TERMINATION` commands denied (`COMMAND_ACK` emitted), **`239 / 239`** corrupted-CRC spoofed frames rejected.
- **Wire-to-Wire Latency:** **`p50 = 18.0 us`**, `p95 = 55.5 us`, **`p99 = 124.4 us`** (`mean = 29.7 us`).

---

## 5. Real-World Sensor, Edge Silicon & Forensics Proofs

| Domain | Ground-Truth Measured Proof (`v1.0.0`) | Artifact |
| --- | --- | --- |
| **Moving NASA Goddard Drone Video Temporal SLZ HUD (`GTX 1650 SUPER`)** | `240` live frames (`PD-USGov-NASA` `SVS14443`, `1920×1080`, `0.00 CU`): homography-warped EMA + hysteresis cuts safe/hazard pixel flicker **`2.19% -> 0.96%` (`-56.25%`)**, SLZ centroid jitter **`1.217 -> 0.461 px` (`-62.15%`)**, `PIDNet-S v2.1` **`77.35 FPS`** (`mean 12.93 ms`), `RANSAC 3-gate = 100%` | [`realworld_drone_video_temporal_proof.json`](../artifacts/v4.0.0/realworld_drone_video_temporal_proof.json), [`vertirta_real_drone_hud_preview.gif`](assets/vertirta_real_drone_hud_preview.gif) |
| **`PIDNet-S v2.1` Local `GTX 1650 SUPER` Recovery** | `0.00 CU` FP16 recovery (`pidnet_slz_v21_local_gtx_opset17.onnx`, `32.07 MB`): **`Val mIoU = 0.4669`**, **`FSR = 2.81%`**, **`OOD AUROC = 0.7554`**, zero-shot **`AeroScapes = 0.2468`** (`FSR 3.43%`) | [`local_gtx1650s_pidnet_v21_recovery_metrics.json`](../artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json), [`vertirta_pidnet_v21_slz_grid.jpg`](assets/vertirta_pidnet_v21_slz_grid.jpg) |
| **GNSS-Denied `EuRoC MAV V1_01_easy` & `XFeat`** | Quarantined zero-shot ETH Zurich `EuRoC MAV` (`29,120` IMU rows, `143.56 s`, C++20 `cpp_nanobind`): **`7 / 7` gates PASS** (`0` clean FA, `1.25 m/s` spoof `TTD = 0.1 s`, `0.137 m` rollback, monotonic `3σ` CTR `6.00 -> 8.84 m`, `0` breaches); Kartal `XFeat` **`36/36` RANSAC `CEP50 = 0.218 m`** | [`realworld_euroc_mav_gnss_audit.json`](../artifacts/v4.0.0/realworld_euroc_mav_gnss_audit.json) |
| **Real `PX4 ULog -> W3C XSD E5X` & Live Istanbul Feeds** | Real `PX4 sample.ulg` (`68.83 s`) -> `400` `.vrta` frames + ED-155A hash chain -> **`W3C XSD`-valid `E5X`** (`SCF-NP`, `ERCS I5`, `Barrier-8 = 0`, `DRAFT_INCIDENT_ONLY`); live Istanbul `METAR (LTFJ)` / `Open-Meteo` / `adsb.lol` (`10` aircraft, `0` well-clear violations, `52.77%` SoC) | [`realworld_px4_ulog_forensics_proof.json`](../artifacts/v4.0.0/realworld_px4_ulog_forensics_proof.json), [`realworld_istanbul_live_feed_audit.json`](../artifacts/v4.0.0/realworld_istanbul_live_feed_audit.json) |
| **Offline Zero-GPL Voice ATC & `TU116` Cruise Gate** | `sherpa-onnx v2.0.3` (`espeak-ng` deleted, `0` GPL) TTS **`TTFA p50 = 71.7 ms`** (`< 250 ms`), `faster-whisper` + Mondrian + Z3 `24/24` turns **`100%` Z3 SAT**; Cruise time-sliced **`19.83 ms p99 / 56.25 FPS`** | [`local_voice_atc_live_proof.json`](../artifacts/v4.0.0/local_voice_atc_live_proof.json), [`gtx1650s_m3_timesliced_edge_gate.json`](../artifacts/v4.0.0/gtx1650s_m3_timesliced_edge_gate.json) |

---

## 6. Explicit Honest Limitations & Open `v1.1` Roadmap

Every negative result, unexecuted hardware target, and domain-shift failure mode is disclosed verbatim:

1. **No Physical `STM32H753` Silicon Capture Yet (`PENDING_PHYSICAL_STM32H7_HITL`):** Cross-compiled `STM32H753` Rev V ELF (`0` veneers, `ES0392 Rev 16` locks) and Renode CAN-FD SIL (`12.50 ms`) are verified, but on-silicon `DWT->CYCCNT` empirical high-water-mark timing awaits physical board hardware.
2. **No Local `PX4 v1.16` / `Gazebo Harmonic` Closed-Loop SITL Capture Yet (`PENDING_LOCAL_PX4_GZ_SITL`):** `Sn5` (1,000-run hot-reset SITL) and `Sn6` (50-run EGL LiDAR SITL) remain open; overlays, SDF models, and MAVLink v2 UDP bridge are checked in and unit-tested.
3. **Real-Video Cross-View Domain Shift (`PIDNet-S v2.1` + Procedural `M-LSD-WireSeg` / `D-FINE-N`):** On the oblique NASA Goddard drone video, bright white vertical facades under oblique sunlight partially trigger 2D RGB safe classes (requiring depth/normal fusion before touchdown), and procedural-trained `M-LSD-WireSeg` (`0` real `TTPLA` pairs) + `D-FINE-N` fire roofline-seam false positives (`240/240` frames clamped to `15.0 m/s` caution cap).
4. **`Qwen3-4B v3` Conformal Set Size (`|C| = 7.80`, `0.0%` Singleton Exec) & RAPS Upgrade Ready:** Root-cause audit ([`qwen3_v3_conformal_raps_diagnostic.json`](../artifacts/v4.0.0/qwen3_v3_conformal_raps_diagnostic.json)) traced `|C| = 7.803` (`96.0%` Lang / `95.25%` SNR coverage) to candidate template/EOS mismatch (`RC1`), `gamma = 1.0` boilerplate dilution (`RC2`), naive LAC tail collapse (`RC3`), and shared scalar `tau` (`RC4`). `Mondrian-RAPS` is implemented and unit-tested in `ml/copilot/calibrate_conformal.py` (`0.00 CU`); live re-scoring requires an optional `~0.15 CU` finalize pass (spec in [`OPTIONAL_G4_V1_1_BATCH_SPEC.md`](research/OPTIONAL_G4_V1_1_BATCH_SPEC.md)).
5. **Whisper Zero-Shot Beats Aggressive BVI-LoRA:** BVI-noise LoRA degraded WER on every SNR stratum (`13.14%–31.74%` ZS vs. `17.97%–35.36%` LoRA on `whisper-large-v3-turbo`); Zero-Shot remains the runtime default.
6. **Physical `TU116` (`4 GB`) Soft-Gate Bounds & `X8-240` CG-Shifted CEL:** Cruise VRAM delta (`366.0 MiB > 300 MiB`), terminal WDDM `p99 = 13.36 ms > 12 ms`, and `X8-240` `±5 cm` CG-shifted CEL (`29/100`) are logged as open engineering targets for `v1.1` (see [`CROSS_BENCHMARK_AUDIT_v1.md`](research/CROSS_BENCHMARK_AUDIT_v1.md)).
