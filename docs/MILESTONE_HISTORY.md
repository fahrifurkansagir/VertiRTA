# VertiRTA — Detailed Milestone Verification & Historical Benchmarks

This document preserves the full historical milestone breakdown (`v0.1.0-proto` through `v3.5.1` and consolidated `v1.0.0`) moved out of the front-page [`README.md`](../README.md) to keep the repository landing page concise and scannable. For the 1-page technical summary, see [`docs/SYSTEM_CARD.md`](SYSTEM_CARD.md); for raw `v1.0.0` Tier 1–4 telemetry, see [`BENCHMARKS.md`](../BENCHMARKS.md).

> **Scope Reminder:** Every milestone below is an open-source research, simulation, and software-verification testbed created by **Fahri Furkan Sagir** under **Apache-2.0 + `NOTICE`** — NOT a certified airborne product.

---

## 1. Tier 1–4 Deterministic Execution & Bare-Metal Cortex-M7 Baseline

### Tier 1 — Bare-Metal ARM Cortex-M7 Footprint, Stack/DAG, Zero-Heap & Sanitizers

| Metric | Target / Tool | Measured Value | Budget / Gate | Status | Artifact |
| --- | --- | --- | --- | --- | --- |
| Cortex-M7 Flash (`.text + .data`) | `arm-none-eabi-size` (`-Os`, `11` objs) | `28,886 B` (`28.21 KiB`) | `< 262,144 B` (`256 KiB`) | `PASS` | [`bare_metal_size_cortex_m7.json`](../artifacts/v1.0.0/benchmarks/bare_metal_size_cortex_m7.json) |
| Cortex-M7 Library Static RAM (`.data + .bss`) | `arm-none-eabi-size` (caller owns static `.bss`) | `0 B` | `< 131,072 B` (`128 KiB`) | `PASS` | [`bare_metal_size_cortex_m7.json`](../artifacts/v1.0.0/benchmarks/bare_metal_size_cortex_m7.json) |
| Cortex-M7 Max Static Frame (`-fstack-usage`) | `arm-none-eabi-g++ 14.2.1` (`77` fns, `0` dynamic) | `1,496 B` | `<= 4,096 B` | `PASS` | [`stack_usage_cortex_m7.json`](../artifacts/v1.0.0/benchmarks/stack_usage_cortex_m7.json) |
| Cortex-M7 Max Call-Graph DAG (`-fcallgraph-info`) | `arm-none-eabi-g++ 14.2.1` (`98` nodes, `0` recursion) | `2,960 B` | `<= 16,384 B` | `PASS` | [`stack_usage_cortex_m7.json`](../artifacts/v1.0.0/benchmarks/stack_usage_cortex_m7.json) |
| Host x86_64 Max Frame / Call-Graph DAG | `g++ 15.2.0` (`74` fns, `105` nodes, `0` recursion) | `2,112 B` / `5,280 B` | `<= 4,096 B` / `<= 16,384 B` | `PASS` | [`stack_usage_x86_64.json`](../artifacts/v1.0.0/benchmarks/stack_usage_x86_64.json) |
| Zero-Heap Symbol Audit (x86_64 & Cortex-M7) | `nm` & `arm-none-eabi-nm` + `-Wl,--wrap=malloc` | `0` / `0` forbidden | `0` symbols | `PASS` | [`no_heap_x86_64.json`](../artifacts/v1.0.0/benchmarks/no_heap_x86_64.json), [`no_heap_cortex_m7.json`](../artifacts/v1.0.0/benchmarks/no_heap_cortex_m7.json) |
| `ASTM F3269-21` Core Isolation | `check_arch_isolation.py` (`37` files) | `0` violations | `0` violations | `PASS` | [`arch_isolation_report.json`](../artifacts/v1.0.0/benchmarks/arch_isolation_report.json) |

### Tiers 2, 3 & 4 — Deterministic `.vrta` Replay & Per-Tick Execution Latency

| Tier | Target Runner / Arch | Mode / Scope | Ticks / Frames | `mean` (us) | `p50` (us) | `p99` (us) | `p99.9` (us) | `max` (us) | Steal / Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Tier 2** | `x86_64` & `arm64` | Same-ISA 0-ULP `.vrta` Replay (`corridor_faults.vrta`, `512,128 B`, `1,400` CEL frames) | `2,000` frames (`20.0 s`) | — | — | — | — | — | `PASS` (`run1 == run2` SHA-256) |
| **Tier 2** | `x86_64 <-> arm64` CI | Cross-ISA Parity (`ubuntu-24.04` ↔ Azure Cobalt 100 `ubuntu-24.04-arm`) | `2,000` frames | — | — | — | — | — | **`PASS` (`max_abs_diff = 0.0`, `0` mode div)** |
| **Tier 3** | `x86_64` (Local WSL2) | `CLOCK_THREAD_CPUTIME_ID` | `100,000` ticks | `1.2841` | `1.2400` | `1.9000` | `11.5690` | `55.3490` | `MEASURED_LOCAL` |
| **Tier 4** | `x86_64` (Local WSL2) | `CLOCK_MONOTONIC_RAW` + `/proc/stat` | `50,000` ticks | `0.5780` | `0.5400` | `0.9500` | `1.5900` | `32.0690` | `0.0%` (`CLEAN`) |
| **Tier 3/4** | `x86_64` (`ubuntu-24.04` CI) | Callgrind `723,331 Ir/tick` + `CLOCK_MONOTONIC_RAW` | `20,000` ticks | `0.6570` | `0.6790` | `0.8430` | `4.0220` | `17.8920` | `0.0%` (`CLEAN / PASS`) |
| **Tier 3/4** | `arm64` (`ubuntu-24.04-arm`) | Azure Cobalt 100: Callgrind `604,543 Ir/tick` + `CLOCK_MONOTONIC_RAW` | `20,000` ticks | `0.7212` | `0.7200` | **`0.7360`** | **`0.8640`** | `4.1440` | `0.0%` (`CLEAN / PASS`) |

> **WCET Caveat:** User-space Linux timings are CI regression telemetry ONLY and do NOT constitute DO-178C bare-metal RTOS WCET certification evidence.

### Live Tier-1 10,000-Run 6-DOF Survival Envelope Atlas + A/B Ablations (`30,000` Simulations, Nominal CG)

| Configuration | CSFL Pool (`S2+S3+S5`, `n = 6,500`) | CSFL 95% CP Upper Fail Bound | CEL Pool (`S4` Same-Arm, `n = 1,500`) | CEL 95% CP Upper Fail Bound | Wall-Clock (`10k`) |
| --- | --- | --- | --- | --- | --- |
| **`HJ-CBVF + INDI-WLS` (Primary)** | **`96.82%`** (`6,293 / 6,500`, `k = 207`) | `0.035665` | **`100.00%`** (`1,500 / 1,500`, `k = 0`) | `0.001995` | `33.083 s` |
| `HOCBF + INDI-WLS` (CBVF Ablation) | `44.85%` | `0.561743` | `100.00%` | `0.001995` | `23.111 s` |
| `HJ-CBVF + Stock CA` (Allocation Ablation) | `96.82%` | `0.035665` | `0.00%` (`0 / 1,500` capsize) | `1.000000` | `17.667 s` |

- Exact paired 30k RA-L stats (`v3.1.0` full JAX 4D PDE): **`95.98%` CSFL** (`(3324, 0)`, `p_holm < 10^-16`) / **`100.00%` CEL** (`(1500, 0)`), logged in [`artifacts/v3.1.0/paired_30k_ral_stats.json`](../artifacts/v3.1.0/paired_30k_ral_stats.json).
- Every single failed simulation is preserved in [`artifacts/v1.0.0/tier1_atlas/failure_cliff_ledger.json`](../artifacts/v1.0.0/tier1_atlas/failure_cliff_ledger.json).

![VertiRTA Tier-1 10,000-Run Survival Heatmap](../artifacts/v1.0.0/tier1_atlas/survival_heatmap.png)

![VertiRTA A/B Ablation Survival & Margin CDF Comparison](../artifacts/v1.0.0/tier1_atlas/ab_cdf.png)

---

## 2. Historical Pre-1.0 Research Milestone Progression

- **`v0.1.0-proto` (Archived from early prototype tag, `2026-11-22`):** Deterministic C++17/C++20 `flight_core` (`28,886 B` Flash / `0 B` RAM) + `rta_companion`, 30,000-sim Tier-1 6-DOF atlas, CI-hashed `SCSC-141C` GSN v3 safety case, PX4 v1.16 / ROS 2 Jazzy / Gazebo Harmonic overlays, and dual-layer Istanbul Earth visualization.
- **`v2.0.0-alpha` & `v2.0.0` (Advisory Only, `2026-11-22`):** TR/EN neuro-symbolic Z3-verified contingency co-pilot (`core/stl/`, `core/copilot/`, Apache-2.0 `VertiBench-TR/EN` `< 0.15` Jaccard ARM64 CI) + Active Visual SLZ (`core/slz_bayes/`, cell-local `1.5625 MiB` grid, zero-ShareAlike 5-source `PIDNet-S`, Colab EGL rig & 60-seed ablation).
- **`v3.0.0` (`M1`, `dc28e37`):** Bare-metal `STM32H753` ITCM/DTCM HITL firmware (`ES0392 Rev 16` locks), dual-FCC Renode `v1.17.0` OpenCyphal CAN-FD SIL (`3/3`, `12.50 ms` failover), `ESBMC v8.7.0 --32` (`1,905 / 1,905` properties), `< 150 KiB` single-threaded Emscripten `.wasm` (`38,470 B`) + MapLibre/Deck.gl `VertiGCS`, relocatable `VertiRTAConfig.cmake` + `nanobind` `cp312-abi3` SDK, non-root Docker, `DCO 1.1`, and double-anonymous IEEE RA-L Paper 1 (`docs/paper_ral/`).
- **`v3.1.0` (`6869842`, Pre-release):** Live 1-click GitHub Pages `VertiGCS` WASM sandbox + real Colab `T4`/`L4` GPU campaigns (`19.35 CU`): full `81×57×81×57` JAX 4D `HJI-CBVF` PDE + paired 30k McNemar/`rliable` IQM, `Qwen3-4B` SFT→Z3-GRPO, zero-ShareAlike `PIDNet-S` distillation verified on physical `GTX 1650 SUPER`.
- **`v3.5.0` (`M2`, `bf8adb1`):** Zero-heap `Lift+Cruise 8+1` `IAirframeDynamics` & 400 Hz transition INDI-WLS (`0.198 m` alt err, `0.98 us` WLS p99), air-gapped offline `GPL-3.0` OpenFOAM POD-RBF → Apache-2.0 `mmap` `VRTAWND1` (`p99 = 0.272 us`, `0` GPL across runtime targets) + coaxial BVI/BPM `dBA` + Kartal Hospital Tier-2 helipad cylinder, 20-eVTOL SoA `FleetSimEngine` (`0.151 ms/tick`) + `Part-IS (EU) 2023/203` (`40 B` header, `ReplayWindow128`, `libsodium-1.0.22`, `RTO <= 10 s`) + separate-context Fleet Z3.
- **`v3.5.1` (Phase B, Pre-release):** Offline bundled MapLibre worker, `PIDNet-S v2` (`val_mIoU = 0.4848`, `val_FSR = 4.03%`, physical Win10 `GTX 1650 SUPER` `p99 = 11.695 ms < 12 ms`, `155 MiB` VRAM), `Qwen3-4B v2` O(1) GBNF DFA + Z3 + NLL scaling (`AST/STL 91.96%`, `ECE 0.1501`, `|C| = 3.125`), STM32H753 `PLL1Q = 80 MHz` FDCAN1, and official DHMİ AIP `LTBX` (`40°59'26"N, 029°12'58"E`).
- **`v1.0.0` (`Latest` Stable SemVer, `2026-11-25`):** Consolidates all milestones + `102.85M`-node dual-airframe JAX PDE, `PIDNet-S v2.1` local GTX recovery (`0.4669` mIoU / `2.81%` FSR / `0.7554` OOD AUROC), `EuRoC MAV 7/7`, `PX4 ULog → W3C XSD E5X`, live Istanbul feeds, Zero-GPL `sherpa-onnx` TTS (`71.7 ms`), real moving NASA Goddard drone temporal SLZ HUD (`-56.25%` flicker / `-62.15%` jitter), `1,000+180` C++20 Monte Carlo dispersion & failure boundary, Zero-GPL MAVLink v2 UDP RTA shield (`p50 = 18.0 us`, `0` leaks), RAPS conformal upgrade + Qwen3 diagnostic, `docs/SYSTEM_CARD.md`, and `7/7` live CI GREEN.
