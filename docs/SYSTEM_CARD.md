# VertiRTA `v1.0.0` — Technical System Card

> **Principal-Engineer Technical Summary · Apache-2.0 + DCO 1.1 + [`NOTICE`](../NOTICE)**  
> **Author:** Fahri Furkan Sagir (`@fahrifurkansagir`) · **Zero Airworthiness Certification Claims**  
> **Live Browser Console:** [`https://fahrifurkansagir.github.io/VertiRTA/`](https://fahrifurkansagir.github.io/VertiRTA/) (`38,470 B` `.wasm` Worker, zero tokens)  
> **Deep Audits:** [`docs/research/CROSS_BENCHMARK_AUDIT_v1.md`](research/CROSS_BENCHMARK_AUDIT_v1.md) · [`docs/research/OPTIONAL_G4_V1_1_BATCH_SPEC.md`](research/OPTIONAL_G4_V1_1_BATCH_SPEC.md) · [`docs/SAFETY_CASE.md`](SAFETY_CASE.md)

---

## 1. System Identity & Legal Scope

`VertiRTA` is an open-source, deterministic flight-control and `ASTM F3269-21` 3-tier Simplex Run-Time Assurance (RTA) software platform engineered for two `240 kg` urban eVTOL reference airframes — coaxial **`VertiRTA-X8-240`** and transition **`VertiRTA-LC-240` (`Lift+Cruise 8+1`)** — operating along the Istanbul Maltepe–Dragos–Kartal coastal corridor bounded by real DHMİ AIP Turkey **`LTFJ`** (Sabiha Gökçen, `< 29.1895° E`) and **`LTBX`** (Samandıra, `40°59'26"N, 029°12'58"E`) controlled airspace walls plus the Kartal Dr. Lütfi Kırdar Hospital Tier-2 helipad keepout cylinder (`160 m`).

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
| **Zero-GPL & Test Suite Hygiene** | **`7 / 7`** GitHub Actions CI workflows, **`36 / 36`** CTest, **`18 / 18`** Vitest (`5` files), **`11 / 11`** M1–M3 native sdist SDK, **`0`** GPL symbols across runtime targets | Strict Apache-2.0 runtime (`check_zero_gpl_symbols.py`) | [`scripts/ci/check_zero_gpl_symbols.py`](../scripts/ci/check_zero_gpl_symbols.py) |

---

## 3. Dual-Airframe `1,000 + 180` Native C++20 Monte Carlo Dispersion & Failure-Boundary Proof

Executed live via native C++20 (`master_seed = 20261125`, `12` threads, `11.11 s` wall-clock; [`monte_carlo_1000_dispersion_report.json`](../artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json), plot [`vertirta_mc1000_envelope_dispersion.png`](assets/vertirta_mc1000_envelope_dispersion.png)):

- **Zero Airspace Breaches:** **`0 / 1,000`** (`0.00%`, 95% one-sided CP upper bound `0.002991`) across `LTFJ` (min margin `941.0 m`), `LTBX` (`2,240.8 m`), and Kartal Hospital (`2,890.7 m`).
- **`VertiRTA-LC-240` (`8+1` Lift+Cruise):** **`500 / 500` (`100.0%`)** safe recoveries across all 5 operational regimes (`Nominal Turbulent`, `1M CSFL`, `Same-Arm Dual-Kill`, `GNSS Slow-Ramp Spoof`, `Compound Low-SoC + Wind + Fault`) with **`CEP50 = 0.80 m` / `CEP95 = 1.97 m`** (`max_td_err = 3.52 m`, `vz_p99 = 0.81 m/s`).
- **`VertiRTA-X8-240` (`429 / 500`, `85.8%`) — Forensic CG-Shift Physics Boundary:** Passes **`400 / 400` (`100.0%`)** across `Nominal Turbulent`, `1M CSFL`, `GNSS Slow-Ramp Spoof` (`200/200` spoof flights detected across both airframes, `mean TTD = 0.178 s`, `max TTD = 0.295 s`), and `Compound Low-SoC + Wind + Fault`. Under **unmodeled `±5 cm` lateral CG shift** combined with **same-arm coaxial dual-motor kill (`CEL`)**, active flat-spin recovery drops to **`29 / 100`** (`CEP50 = 41.08 m`) vs. nominal-CG Tier-1 10k `1,500 / 1,500` (`100.0%`) ([`paired_30k_ral_stats.json`](../artifacts/v3.1.0/paired_30k_ral_stats.json)). Forensic rigid-body analysis (`tests/integration/test_monte_carlo_dispersion.cpp`) confirms zero runner bugs and pins the exact directional control authority envelope: the orthogonal actuated axis `cg_v ∈ [-5.0, +5.0] cm` is **100% safe** across all 4 arms, whereas the unactuated dead-arm axis obeys a physical spin-up impulse boundary **`cg_u ∈ [-3.50, +1.00] cm`** at `240 kg` (`[-2.25, +0.25] cm` at `264 kg`).
- **`6×6` Thrust-Saturation Failure-Boundary Sweep (`180` Flights, `X8-240` Same-Arm CEL):** `0%`–`+7%` mass overload achieves `100%` survival across tested gusts (`13` green cells); first breakdown cliff occurs at **`+14%` mass overload @ `10 m/s` gust** (`5` partial transition cells); **`>= +21%` mass overload** hits a hard **`0%` thrust-saturation wall** (`18` red cells).

---

## 4. Zero-GPL Native C++20 MAVLink v2 UDP/Offboard RTA Shield

Clean-room Apache-2.0 C++20 MAVLink v2 wire-protocol shield (`0xFD` framing, `8` `common.xml` `CRC_EXTRA` message IDs, zero GPL/MAVSDK dependencies; [`tools/mavlink_rta_bridge/`](../tools/mavlink_rta_bridge/CMakeLists.txt), [`mavlink_v2_rta_bridge_proof.json`](../artifacts/v4.0.0/mavlink_v2_rta_bridge_proof.json)):

- **Live POSIX UDP Loopback Audit (`2,500` packets, `922.56 ms`):** **`0`** valid packets dropped, **`0`** unsafe packets leaked to FCU, **`0`** corrupted packets leaked.
- **100% Adversarial Interception & Protocol Hardening:** **`1,181 / 1,181`** unsafe offboard setpoints clamped (`566` CBVF, `861` `LTFJ`/`LTBX`/Hospital NFZ, `689` plunge/overspeed), **`40 / 40`** deadman heartbeat-timeout overrides, **`240 / 240`** airborne `DISARM`/`FLIGHT_TERMINATION` commands denied (`COMMAND_ACK` emitted), **`239 / 239`** corrupted-CRC spoofed frames rejected; additionally hardened fail-closed against non-`MAV_FRAME_LOCAL_NED`, `FORCE_SET` bit `0x0200`, pre-heartbeat/stale-position/`NaN` disarm, and non-loopback LAN `--fcu-host` binding.
- **Wire-to-Wire Latency:** **`p50 = 18.0 us`**, `p95 = 55.5 us`, **`p99 = 124.4 us`** (`mean = 29.7 us`).

---

## 5. Real-World Sensor, Edge Silicon & Forensics Proofs

| Domain | Ground-Truth Measured Proof (`v1.0.0`) | Artifact |
| --- | --- | --- |
| **4-Scene Real Drone Video 3D-Normal + Motion Veto (`GTX 1650 SUPER`)** | `720` live frames across 4 permissive scenes (`NASA` oblique `50.2°`, `USDA` pure-nadir `82.1°`, `Aruba` coastal surf `50.6°`, `Toronto` forward horizon `18.8°`): 3D surface-normal + Helmholtz OOD + **rigid-surface homography residual motion veto** blocks `25.5%` sunlit urban facades (`-53.8%` flicker), preserves flat nadir ground (`0.17%` veto vs. `100%` legacy bug), cuts crashing ocean-foam false-safe **`44.2% -> 4.8%` (`37.6%` blocked)**, and runs `M-LSD-WireSeg v1.1` + `D-FINE-N v1.1` in-domain on `180/180` horizon frames with `0` FP boxes/alerts (`12/12` gates PASS) | [`realworld_multivideo_stress_proof.json`](../artifacts/v4.0.0/realworld_multivideo_stress_proof.json), [`realworld_drone_video_temporal_proof.json`](../artifacts/v4.0.0/realworld_drone_video_temporal_proof.json), [`vertirta_real_drone_hud_preview.gif`](assets/vertirta_real_drone_hud_preview.gif) |
| **`PIDNet-S v2.1` Local Recovery & Quarantined Real SLZ Transfer** | Local FP16 recovery (`pidnet_slz_v21_local_gtx_opset17.onnx`, `32.07 MB`): **`Val mIoU = 0.4669`**, **`FSR = 2.81%`**, **`OOD AUROC = 0.7554`**; quarantined zero-shot **`AeroScapes = 0.2468`** (`FSR 3.43%`) & **`UAVScenes = 0.2259`** (`FSR 12.55%`, `300` frames / `20` scenes, raw pixels deleted) | [`local_gtx1650s_pidnet_v21_recovery_metrics.json`](../artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json), [`realworld_uavscenes_zeroshot_eval.json`](../artifacts/v4.0.0/realworld_uavscenes_zeroshot_eval.json), [`vertirta_pidnet_v21_slz_grid.jpg`](assets/vertirta_pidnet_v21_slz_grid.jpg) |
| **`Qwen3-4B v3` Exact Local FP32 Conformal Recovery** | Local `GTX 1650 SUPER` exact FP32 layer-streaming GBNF-tree export (`peak_vram = 837.5 MiB`, zero 4-bit quantization loss, zero test leakage, synthetic parity `8.60e-8 < 1e-5`, monolithic BF16 `3/3` top-1): **ECE `0.1744 -> 0.0321`** (Lang-APS `96.00%`, `|C| = 3.486`), **SNR-APS `95.08%`, `|C| = 2.347` (`40.0%` singl.)**, **Lang-LAC `96.08%`, `|C| = 2.502` (`40.0%` singl.)**, **SNR-LAC `95.58%`, `|C| = 2.058` (`45.5%` singl.)**, 100-resplit SNR bootstrap **`95.00 ± 0.79%`, `|C| = 1.924 ± 0.075`** | [`local_gtx1650s_qwen3_v3_conformal_raps.json`](../artifacts/v4.0.0/local_gtx1650s_qwen3_v3_conformal_raps.json) |
| **GNSS-Denied `EuRoC MAV V1_01_easy` & `XFeat`** | Quarantined zero-shot ETH Zurich `EuRoC MAV` (`29,120` IMU rows, `143.56 s`, C++20 `cpp_nanobind`): **`7 / 7` gates PASS** (`0` clean FA, `1.25 m/s` spoof `TTD = 0.1 s`, `0.137 m` rollback, monotonic `3σ` CTR `6.00 -> 8.84 m`, `0` breaches, `NaN`/non-PSD hardened); Kartal `XFeat` **`36/36` RANSAC `CEP50 = 0.218 m`** | [`realworld_euroc_mav_gnss_audit.json`](../artifacts/v4.0.0/realworld_euroc_mav_gnss_audit.json) |
| **Real `PX4 ULog -> W3C XSD E5X` & Live Istanbul Feeds** | Real `PX4 sample.ulg` (`68.83 s`) -> `400` `.vrta` frames + ED-155A hash chain (`\uXXXX` surrogate round-trip hardened) -> **`W3C XSD`-valid `E5X`** (`SCF-NP`, `ERCS I5`, `Barrier-8 = 0`, `DRAFT_INCIDENT_ONLY`); live Istanbul `METAR (LTFJ)` / `Open-Meteo` / `adsb.lol` (`10` aircraft, `0` well-clear violations, `52.77%` SoC) | [`realworld_px4_ulog_forensics_proof.json`](../artifacts/v4.0.0/realworld_px4_ulog_forensics_proof.json), [`realworld_istanbul_live_feed_audit.json`](../artifacts/v4.0.0/realworld_istanbul_live_feed_audit.json) |
| **Offline Zero-GPL Voice ATC & `6/6` `TU116` Edge Gate** | `faster-whisper-large-v3-turbo` INT8 CPU (TR FLEURS+BVI WER cut ~2× to `12.58%–27.81%`, EN `6.45%–26.61%`) + `sherpa-onnx v2.0.3` (`0` GPL, **`TTFA p50 = 67.15 ms`**) + `24/24` **`100%` Z3 SAT**; physical `TU116` shared-BFC time-sliced **`local_gtx_v11`** passes **`6 / 6`** hardware gates: **Peak VRAM `286.0 MiB < 300 MiB`** (`274.0 MiB` Cruise / `286.0 MiB` Terminal), Terminal **`9.07 ms p99 (< 12 ms PASS, 125.65 FPS)`**, Cruise **`12.30 ms p99 (< 25 ms PASS, 89.84 FPS)`** | [`local_voice_atc_live_proof.json`](../artifacts/v4.0.0/local_voice_atc_live_proof.json), [`gtx1650s_m3_timesliced_edge_gate.json`](../artifacts/v4.0.0/gtx1650s_m3_timesliced_edge_gate.json) |

---

## 6. Explicit Honest Limitations & Open Roadmap

Every remaining negative result, unexecuted hardware target, and physical boundary is disclosed verbatim:

1. **No Physical `STM32H753` Silicon Capture Yet (`PENDING_PHYSICAL_STM32H7_HITL`):** Cross-compiled `STM32H753` Rev V ELF (`0` veneers, `ES0392 Rev 16` locks) and Renode CAN-FD SIL (`12.50 ms`) are verified, but on-silicon `DWT->CYCCNT` empirical high-water-mark timing awaits physical board hardware.
2. **No Local `PX4 v1.16` / `Gazebo Harmonic` Closed-Loop SITL Capture Yet (`PENDING_LOCAL_PX4_GZ_SITL`):** `Sn5` (1,000-run hot-reset SITL) and `Sn6` (50-run EGL LiDAR SITL) remain open; overlays, SDF models, and MAVLink v2 UDP bridge are checked in and unit-tested.
3. **Whisper Zero-Shot Beats Aggressive BVI-LoRA:** BVI-noise LoRA degraded WER on every SNR stratum (`13.14%–31.74%` ZS vs. `17.97%–35.36%` LoRA on `whisper-large-v3-turbo`); Zero-Shot remains the runtime default pending optional low-LR clean-replay fine-tuning ([`OPTIONAL_G4_V1_1_BATCH_SPEC.md`](research/OPTIONAL_G4_V1_1_BATCH_SPEC.md)).
4. **Manual-Registration External SLZ Benchmark (`ICG Drone`) & Human-Native `VertiBench-v2`:** `AeroScapes` and `UAVScenes` are evaluated zero-shot (`raw_pixels_deleted: true`), while `ICG Drone` requires manual academic registration (`PENDING_EVAL_ONLY_REAL_IMAGERY`) and `VertiBench-v2` retains its honest `PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2` provenance tag (with Bucket-3 adversarial nuance at `43.24%` branch top-1 driving residual multi-branch sets).
5. **Physical `X8-240` `cg_u` Authority Boundary:** Under collinear same-arm CEL, `X8-240` obeys the verified rigid-body boundary `cg_u ∈ [-3.50, +1.00] cm` at `240 kg` (`29/100` under unmodeled `±5 cm` CG shift; see [`CROSS_BENCHMARK_AUDIT_v1.md`](research/CROSS_BENCHMARK_AUDIT_v1.md)). The previous `TU116` `354 MiB` Cruise VRAM soft-gate is now **CLOSED (`6/6 PASS`, `274 MiB` Cruise / `286 MiB` peak < `300 MiB`)**.
