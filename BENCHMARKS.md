# VertiRTA — 4-Tier Execution Determinism & Bare-Metal Footprint Benchmarks (`v1.0.0`)

> **Evidence Commit:** `ef9be75` (`main`) | **Artifact Directory:** [`artifacts/v1.0.0/`](artifacts/v1.0.0/)
> **Local Measurement Host:** WSL2 Ubuntu (x86_64, `g++ 15.2.0 -O2` strict IEEE-754 + `arm-none-eabi-g++ 14.2.1 -Os` Cortex-M7).
> **Cloud Runner Status:** `ubuntu-24.04-arm` (Azure Cobalt 100) rows are explicitly marked **`PENDING_GITHUB_ACTIONS_ARM64`** until executed on GitHub Actions — zero cloud numbers are fabricated or extrapolated.

## Execution Environment & WCET Scope Caveat

> All CI execution benchmarks in this repository are collected on public GitHub-hosted virtualized runners:
> - **ARM64:** `ubuntu-24.04-arm` (Microsoft Azure **Cobalt 100** / Armv9.0-A **Neoverse N2**, 4 vCPU, 16 GB RAM, Ubuntu 24.04 LTS, Azure Linux kernel).
> - **x86_64:** `ubuntu-24.04` (4 vCPU, 16 GB RAM, Ubuntu 24.04 LTS, Azure Linux kernel).
>
> **Strict Interpretation Rules (NO Bare-Metal WCET Claims):**
> 1. **Hypervisor Steal & Multi-Tenant Jitter:** `CLOCK_MONOTONIC_RAW` wall-clock tail latencies (`p99`, `p99.9`, `max`) measured inside multi-tenant cloud VMs include host hypervisor scheduling preemption (reported alongside `/proc/stat` CPU `steal` % and involuntary context switches). Runs exceeding `1.0%` hypervisor steal are explicitly flagged **`NOISY_NEIGHBOR_TAINTED`**.
> 2. **NOT Certifiable RTOS WCET Evidence:** Linux user-space wall-clock timings on virtualized Cobalt 100 / x86_64 runners are indicative CI performance regression metrics ONLY. They do **NOT** constitute Worst-Case Execution Time (WCET) certification evidence under DO-178C or EASA `SC-VTOL-02` / `MOC VTOL.2510`.
> 3. **Hardware-Independent Structural Guarantees:** Auditable determinism claims rest on **Tier 1** (link-time 0-heap trap + `nm -u` ban, `-fstack-usage` + `-fcallgraph-info=su,da` DAG, Cortex-M7 `< 256 KB` Flash / `< 128 KB` RAM), **Tier 2** (0-ULP SHA-256 Same-ISA & `< 1e-12` / 0 mode divergence Cross-ISA), and **Tier 3** (`CLOCK_THREAD_CPUTIME_ID` + `perf stat` / `valgrind --tool=callgrind` deterministic instruction count).

---

## Tier 1 — Hardware-Independent Structural & Bare-Metal Cortex-M7 Footprint (`libvertirta_flight_core`)

| Metric | Target / Tool | Measured Value | Budget / Gate | Status | Artifact |
| --- | --- | --- | --- | --- | --- |
| Cortex-M7 Flash (`.text + .data`) | `arm-none-eabi-size` (`-Os`, `11` objs) | `28886 B` (`28.21 KiB`) | `< 262144 B` (`256 KiB`) | `PASS` | [`bare_metal_size_cortex_m7.json`](artifacts/v1.0.0/benchmarks/bare_metal_size_cortex_m7.json) |
| Cortex-M7 Library Static RAM (`.data + .bss`) | `arm-none-eabi-size` (caller owns static `.bss` arenas) | `0 B` | `< 131072 B` (`128 KiB`) | `PASS` | [`bare_metal_size_cortex_m7.json`](artifacts/v1.0.0/benchmarks/bare_metal_size_cortex_m7.json) |
| Cortex-M7 Max Static Frame (`-fstack-usage`) | `arm-none-eabi-g++` (`77` fns, `0` dynamic) | `1496 B` | `<= 4096 B` | `PASS` | [`stack_usage_cortex_m7.json`](artifacts/v1.0.0/benchmarks/stack_usage_cortex_m7.json) |
| Cortex-M7 Max Call-Graph DAG (`-fcallgraph-info`) | `arm-none-eabi-g++` (`98` nodes, `0` recursion) | `2960 B` | `<= 16384 B` | `PASS` | [`stack_usage_cortex_m7.json`](artifacts/v1.0.0/benchmarks/stack_usage_cortex_m7.json) |
| Host x86_64 Max Static Frame (`flight_core` + noheap test) | `g++ 15.2.0` (`74` fns, `3` bounded realigned) | `2112 B` | `<= 4096 B` | `PASS` | [`stack_usage_x86_64.json`](artifacts/v1.0.0/benchmarks/stack_usage_x86_64.json) |
| Host x86_64 Max Call-Graph DAG (`flight_core` + noheap test) | `g++ 15.2.0` (`105` nodes, `0` recursion) | `5280 B` | `<= 16384 B` | `PASS` | [`stack_usage_x86_64.json`](artifacts/v1.0.0/benchmarks/stack_usage_x86_64.json) |
| Zero-Heap Symbol Audit (x86_64 & Cortex-M7) | `nm` & `arm-none-eabi-nm` + `-Wl,--wrap=malloc` | `0` / `0` forbidden | `0` symbols | `PASS` | [`no_heap_x86_64.json`](artifacts/v1.0.0/benchmarks/no_heap_x86_64.json), [`no_heap_cortex_m7.json`](artifacts/v1.0.0/benchmarks/no_heap_cortex_m7.json) |
| `ASTM F3269-21` Core Isolation (`core/{flight_core,rta_companion}`) | `check_arch_isolation.py` (`37` files) | `0` violations | `0` violations | `PASS` | [`arch_isolation_report.json`](artifacts/v1.0.0/benchmarks/arch_isolation_report.json) |
| Release x86_64 Unit & No-Heap Test Suite | `ctest --preset release-linux-x64` | `8 / 8` passed (`1.00 s`) | `100%` pass | `PASS` | [`ctest_release_x86_64.xml`](artifacts/v1.0.0/benchmarks/ctest_release_x86_64.xml) |

> **Worst-Case Cortex-M7 Stack Call Chain (`2,960 B`):** `IndiWlsController::step_400hz` -> `wls_alloc_8x4_f64`.

---

## Tier 2 — `.vrta` Deterministic Replay (`Same-ISA` & `x86_64 <-> ARM64`)

- **Golden Fixture:** [`tests/fixtures/corridor_faults.vrta`](tests/fixtures/corridor_faults.vrta) (`2,000` frames @ `100 Hz` = `20.0 s`, `512128 B`, `1,400` CEL-latched frames, Castagnoli CRC-32C verified by `viz.cesium.vrta_reader`, SHA-256 `2f331efa044ad2fd7e702156649577df101fc05dc8bf1023241a2dda28b662fe`).
- **Same-ISA x86_64 0-ULP SHA-256 Match (`run1 == run2`):** `True` (`c1a217314681936dc987dab329911257b8d339ce09501e1fa694bfe66c72374e`, artifact [`tier2_same_isa_x86_64.json`](artifacts/v1.0.0/tier2_same_isa_x86_64.json)).
- **Cross-ISA (`x86_64 <-> ubuntu-24.04-arm` Cobalt 100):** `PENDING_GITHUB_ACTIONS_ARM64` (CI enforces `< 1e-12` continuous max-abs tolerance and `0` discrete mode divergences via `.github/workflows/determinism.yml`).

---

## Tier 3 & Tier 4 — Per-Tick CPU-Time Latency (`100k` Ticks) & Transparent Wall-Clock Tail (`50k` Ticks)

Measured on Release `vertirta_bench_cli` (`lut_loaded: true`: full `100 Hz` HJI-CBVF-QP safety filter + `400 Hz` INDI-WLS control allocator step per tick; `100 Hz` outer frame budget = `10,000 us`):

| Target Runner / Arch | Clock / Work-Unit Mode | Ticks | `mean` (us) | `p50` (us) | `p99` (us) | `p99.9` (us) | `max` (us) | Steal % | Taint / Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `x86_64` (Local WSL2 Ubuntu) | Tier 3 `CLOCK_THREAD_CPUTIME_ID` | `100000` | `1.2841` | `1.2400` | `1.9000` | `11.5690` | `55.3490` | `N/A` | `MEASURED_LOCAL` |
| `x86_64` (Local WSL2 Ubuntu) | Tier 4 `CLOCK_MONOTONIC_RAW` + `/proc/stat` | `50000` | `0.5780` | `0.5400` | `0.9500` | `1.5900` | `32.0690` | `0.0%` | `CLEAN` |
| `x86_64` (`ubuntu-24.04` CI) | Tier 3 `perf stat` / `valgrind --tool=callgrind` `Ir/tick` | `1000` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `N/A` | `PENDING_GITHUB_ACTIONS_X86_PMU` |
| `arm64` (`ubuntu-24.04-arm` Azure Cobalt 100) | Tier 3 `Ir/tick` + Tier 4 `CLOCK_MONOTONIC_RAW` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING` | `PENDING_GITHUB_ACTIONS_ARM64` |

Raw benchmark artifacts: [`tier3_thread_cputime_x86_64.json`](artifacts/v1.0.0/benchmarks/tier3_thread_cputime_x86_64.json), [`tier4_wallclock_x86_64.json`](artifacts/v1.0.0/benchmarks/tier4_wallclock_x86_64.json).

---

## Tier-1 6-DOF Survival Envelope Atlas (`10,000` Stratified LHS Runs x `3` A/B Configs = `30,000` Simulations)

Executed live via Release `vertirta_6dof_batch_cli` (`seed = 20261122`, total wall-clock `77.32 s` across `30,000` simulations). Full Clopper-Pearson/Wilson statistics: [`tier1_10k_clopper_pearson.json`](artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json); zero-cherry-picking ledger (`244` failures): [`failure_cliff_ledger.json`](artifacts/v1.0.0/tier1_atlas/failure_cliff_ledger.json).

| Configuration | CSFL Pool (`S2+S3+S5`, `n=6,500`) | CSFL 95% CP Upper Fail Bound | CEL Pool (`S4` Same-Arm, `n=1,500`) | CEL 95% CP Upper Fail Bound | Wall-Clock (`10k`) |
| --- | --- | --- | --- | --- | --- |
| **`HJ-CBVF + INDI-WLS` (Primary)** | **`96.82%`** (`6293/6500`, `k=207`) | `0.035665` | **`100.00%`** (`1500/1500`, `k=0`) | `0.001995` | `33.083 s` |
| `HOCBF + INDI-WLS` (CBVF Ablation) | `44.85%` | `0.561743` | `100.00%` | `0.001995` | `23.111 s` |
| `HJ-CBVF + Stock CA` (Allocation Ablation) | `96.82%` | `0.035665` | `0.00%` | `1.000000` | `17.667 s` |

### Per-Stratum Breakdown (`HJ-CBVF + INDI-WLS`, `n = 10,000`)

| Stratum | Description | `n` | Failures (`k`) | Empirical Pass Rate | 95% One-Sided CP Upper Fail Bound |
| --- | --- | --- | --- | --- | --- |
| `S1` | Nominal Corridor Operations | `1000` | `0` | `100.00%` | `0.002991` |
| `S2` | Single-Motor Loss (`1M`, CSFL Pool) | `2500` | `0` | `100.00%` | `0.001198` |
| `S3` | Cross-Arm Dual-Motor Loss (`2M-Cross`, CSFL Pool) | `2500` | `207` | `91.72%` | `0.092433` |
| `S4` | Same-Arm Coaxial Dual Loss (`2M-SameArm`, CEL Pool) | `1500` | `0` | `100.00%` | `0.001995` |
| `S5` | TMR Sensor Fault + Low-SoC Reserve (CSFL Pool) | `1500` | `0` | `100.00%` | `0.001995` |
| `S6` | Out-of-ODD Boundary Stress | `1000` | `37` | `96.30%` | `0.048390` |

> **Statistical Scope Reminder:** Monte Carlo bounds conditional recovery inside `verification/ODD.yaml` ONLY and **never proves `<= 10^-9 / FH`** (exact zero-failure 95% confidence horizon = `2,995,732,273` independent FH via `math.log1p(-1e-9)`). Tier-2 PX4/Gazebo 1k (`Sn5`) and Tier-3 EGL LiDAR 50 (`Sn6`) remain **`PENDING_LOCAL_PX4_GZ_SITL`** on hosts without ROS 2 Jazzy + Gazebo Harmonic + PX4 SITL installed.

---

## Living Safety Case (`SCSC-141C` GSN v3) Evidence SHA-256 Registry

Stamped into [`docs/SAFETY_CASE.md`](docs/SAFETY_CASE.md) at commit `ef9be75` via `verification/gsn/build_gsn.py` ([`gsn_evidence_hashes.json`](artifacts/v1.0.0/gsn_evidence_hashes.json)):

| GSN Solution | Evidence Status / SHA-256 | Source Artifact |
| --- | --- | --- |
| `Sn1` | `81b5dd7af0d3ae182ab7c465482672107e988b4f54c801c93ba93379dcfde244` | [`arch_isolation_report.json`](artifacts/v1.0.0/benchmarks/arch_isolation_report.json) |
| `Sn2` | `f32fd1a2e1d0f831e3bfea3aa0d4efa7af7c2e29898c0b23c49098fe03b34a5d` | [`ctest_release_x86_64.xml`](artifacts/v1.0.0/benchmarks/ctest_release_x86_64.xml) |
| `Sn3` | `ee90ea4f7917351f6f8ed05e1fbdb69ed72eb65414d987ed48de4834ce508d1f` | [`odd_part_iam_validation.json`](artifacts/v1.0.0/benchmarks/odd_part_iam_validation.json) |
| `Sn4` | `5ed4bba011c951a02f4ec225902b08f497b7fa8c61d566987df8b0aebf86e230` | [`tier1_10k_clopper_pearson.json`](artifacts/v1.0.0/tier1_atlas/tier1_10k_clopper_pearson.json) |
| `Sn5` | `PENDING_LOCAL_PX4_GZ_SITL` | Tier-2 1k PX4/Gazebo SITL (deferred to SITL host) |
| `Sn6` | `PENDING_LOCAL_PX4_GZ_SITL` | Tier-3 50 EGL LiDAR SITL (deferred to SITL host) |
| `Sn7` | `950f9821abbc7e4c5bfe13116d11a55c9fb9eebce9f87118f3c2805676a6bd5d` | [`tier1_structural.json`](artifacts/v1.0.0/benchmarks/tier1_structural.json) |
| `Sn8` | `f7dd4e2bdaf3ffb48fd5c525c479d620e3ca6683819c1bf8a3391b6763b6fe37` | [`tier2_same_isa_x86_64.json`](artifacts/v1.0.0/tier2_same_isa_x86_64.json) (`x86_64 <-> ARM64` `PENDING_GITHUB_ACTIONS_ARM64`) |
| `Sn9` | `c08da5de22cf7347d012b233a083194f863d8111b0729ab26892e623218b4c0b` | [`tier3_thread_cputime_x86_64.json`](artifacts/v1.0.0/benchmarks/tier3_thread_cputime_x86_64.json) (`ARM64`/`PMU` `PENDING_GITHUB_ACTIONS_ARM64`) |
| `Sn10` | `4e21d42b8bbea9c49a215e26f69ead62d9fbc42dac605c23cb655566f842bac8` | [`tier4_wallclock_x86_64.json`](artifacts/v1.0.0/benchmarks/tier4_wallclock_x86_64.json) (`ARM64` `PENDING_GITHUB_ACTIONS_ARM64`) |
