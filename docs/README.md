# VertiRTA Documentation Guide

Use this index to read **VertiRTA** from top to bottom like an engineering book — or jump directly to the track that matches your discipline.

---

## Read VertiRTA Like a Book (6 Chapters)

| Chapter | Start Here | What You Learn |
| --- | --- | --- |
| **1. System Overview & Metrics** | [`README.md`](../README.md) → [`SYSTEM_CARD.md`](SYSTEM_CARD.md) | System mission, Istanbul `LTFJ`/`LTBX` corridor ODD, live 3D WASM console, and SHA-256-backed `v1.0.0` measurements |
| **2. Core Architecture & Trust Boundaries** | [`core/README.md`](../core/README.md) | 400 Hz freestanding MCU core vs. 100 Hz C++20 RTA companion vs. `DAL E` advisory isolation (`ASTM F3269-21`) |
| **3. Flight Physics & Control Math** | [`MATH_DERIVATIONS.md`](MATH_DERIVATIONS.md) → [`BENCHMARKS.md`](../BENCHMARKS.md) | Coaxial X8-240 & Lift+Cruise 8+1 physics, INDI-WLS, Option B1b flat-spin CEL, 4D HJI-CBVF PDE, and 4-tier determinism |
| **4. Safety Assurance & Formal Proofs** | [`SAFETY_CASE.md`](SAFETY_CASE.md) → [`FHA_FMEA_TRACEABILITY.md`](FHA_FMEA_TRACEABILITY.md) → [`assurance/rtm_v4_traceability.yaml`](assurance/rtm_v4_traceability.yaml) | Living `SCSC-141C` GSN v3 (`Sn1–Sn20`), `FHA-01..10`, `R01..R48` bidirectional traceability, `ESBMC v8.7` & `Renode` SIL |
| **5. Offboard Shield, Perception & Forensics** | [`tools/mavlink_rta_bridge/README.md`](../tools/mavlink_rta_bridge/README.md) → [`research/CROSS_BENCHMARK_AUDIT_v1.md`](research/CROSS_BENCHMARK_AUDIT_v1.md) | Zero-GPL MAVLink v2 UDP shield, 3D-normal-vetoed SLZ, `M-LSD-WireSeg`/`D-FINE-N v1.1`, `Qwen3-4B` conformal RAPS, `E5X` forensics, and 6-stack comparative audit |
| **6. Deep Benchmarks, Papers & Roadmap** | [`MILESTONE_HISTORY.md`](MILESTONE_HISTORY.md) → [`paper_ral/`](paper_ral/) & [`paper_corl/`](paper_corl/) → [`research/OPTIONAL_G4_V1_1_BATCH_SPEC.md`](research/OPTIONAL_G4_V1_1_BATCH_SPEC.md) | Full 30,000-run nominal-CG atlas, reproducible LaTeX manuscripts, and honest remaining `v1.1` / hardware targets |

---

## Role-Based Fast Tracks (10-Minute Onboarding)

### Flight Controls & GNC Engineer
1. [`MATH_DERIVATIONS.md`](MATH_DERIVATIONS.md) — exact coaxial `eta_coax = 0.85`, 112S12P Thevenin sag, Option B1b flat-spin (`2.2 rad/s`, `+10.6°` phase lead).
2. [`core/README.md`](../core/README.md) — `flight_core` 400 Hz inner loop vs. `rta_companion` 100 Hz CBVF-QP filter & `Lift+Cruise 8+1` transition INDI-WLS.
3. [`MILESTONE_HISTORY.md`](MILESTONE_HISTORY.md) — 30,000-run Tier-1 atlas + 1,000+180 Monte Carlo dispersion & directional `cg_u` boundary (`[-3.50, +1.00] cm`).
4. [`tools/mavlink_rta_bridge/README.md`](../tools/mavlink_rta_bridge/README.md) — live offboard setpoint clamping & PX4/FCU UDP wiring.

### Embedded Systems & Firmware Engineer
1. [`BENCHMARKS.md`](../BENCHMARKS.md) — Cortex-M7 Flash (`28,886 B`), static RAM (`0 B`), max frame (`1,496 B`), static stack DAG (`2,960 B`), `0` heap / `0` recursion.
2. [`firmware/stm32h7_hitl/README.md`](../firmware/stm32h7_hitl/README.md) — STM32H753 ITCM/DTCM linker layout (`ES0392 Rev 16`, `0` veneers).
3. [`firmware/renode_cyphal/`](../firmware/renode_cyphal/) — dual-FCC OpenCyphal CAN-FD failover (`12.50 ms < 20.0 ms`, `libcanard` + `o1heap v3.0.0`).
4. [`verification/formal/`](../verification/formal/) — `ESBMC v8.7.0` (`1,905/1,905` properties), `Cppcheck`, and `MISRA_GAP_REGISTER.md`.

### Autonomy, Vision & ML Engineer
1. [`SYSTEM_CARD.md`](SYSTEM_CARD.md) — Sections 5 & 6 (`PIDNet-S v2.1`, Apache-2.0 `Depth-Anything-V2-Small` 3D normal veto, `WireSeg`/`D-FINE-N v1.1`, `Qwen3-4B` FP32 conformal RAPS, `UAVScenes`, `faster-whisper-large-v3-turbo`).
2. [`research/CROSS_BENCHMARK_AUDIT_v1.md`](research/CROSS_BENCHMARK_AUDIT_v1.md) — direct engineering comparison against PX4, ArduPilot, Agilicious, Autoware, Basalt/FAST-LIO2, and NASA RTA.
3. [`paper_corl/README.md`](paper_corl/README.md) — advisory ML & real-imagery transfer reproducibility package.
4. [`research/OPTIONAL_G4_V1_1_BATCH_SPEC.md`](research/OPTIONAL_G4_V1_1_BATCH_SPEC.md) — verified closed-locally items vs. optional GPU/registration backlog.

### Aviation Safety, Assurance & Security Engineer
1. [`SAFETY_CASE.md`](SAFETY_CASE.md) — CI-hashed `SCSC-141C` GSN v3 (`Sn1–Sn20`) with `G5`/`G6` programmatically locked `[UNDEVELOPED]`.
2. [`standards_registry.yaml`](standards_registry.yaml) & [`assurance/rtm_v4_traceability.yaml`](assurance/rtm_v4_traceability.yaml) — machine-auditable regulatory & `R01..R48` requirements mapping.
3. [`FHA_FMEA_TRACEABILITY.md`](FHA_FMEA_TRACEABILITY.md) — hazard analysis (`FHA-01..10`) and dual-FCC Markov reliability derivation.
4. [`SECURITY.md`](../SECURITY.md) — `Part-IS (EU) 2023/203`, `ReplayWindow128`, `libsodium >= 1.0.21`, trust boundaries, and disclosure policy.

---

## Complete Folder Map

| Path | Contents |
| --- | --- |
| [`SYSTEM_CARD.md`](SYSTEM_CARD.md) | Concise technical summary of every verified subsystem and limitation |
| [`SAFETY_CASE.md`](SAFETY_CASE.md) | Living `SCSC-141C` GSN v3 safety case & evidence hashes |
| [`MATH_DERIVATIONS.md`](MATH_DERIVATIONS.md) | Cold-derivation lock for coaxial aerodynamics, battery sag, CEL, and FTA |
| [`FHA_FMEA_TRACEABILITY.md`](FHA_FMEA_TRACEABILITY.md) | Functional Hazard Assessment (`FHA-01..10`) and failure-mode traceability |
| [`MILESTONE_HISTORY.md`](MILESTONE_HISTORY.md) | Tier 1–4 latency tables, 30,000-run survival heatmaps, and pre-1.0 progression |
| [`standards_registry.yaml`](standards_registry.yaml) | Regulatory/standards date and status registry |
| [`assurance/rtm_v4_traceability.yaml`](assurance/rtm_v4_traceability.yaml) | Bidirectional machine-verified Requirements Traceability Matrix (`R01..R48`) |
| [`research/`](research/) | Comparative engineering audit (`CROSS_BENCHMARK_AUDIT_v1.md`), `v1.1` spec, and technical literature notes (`DR-01..DR-13`) |
| [`paper/`](paper/), [`paper_ral/`](paper_ral/), [`paper_corl/`](paper_corl/) | Reproducible LaTeX technical reports and deterministic figure/table verification scripts |
| [`assets/`](assets/) | Verified README visuals (`vertirta_real_drone_hud_preview.gif`, `vertirta_mc1000_envelope_dispersion.png`, `vertirta_vertigcs_live_3d.png`, `vertirta_pidnet_v21_slz_grid.jpg`) |
