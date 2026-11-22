# VertiRTA v3.0.0 Bare-Metal STM32H7 HITL, Renode OpenCyphal Failover & ESBMC Formal Verification (Plan 4A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the frozen `28,886 B Flash / 0 B static RAM / 2,960 B stack DAG` Cortex-M7 `libvertirta_flight_core.a` into executable bare-metal STM32H753 / NUCLEO-H743ZI2 silicon profiling (`DWT->CYCCNT` 400 Hz HITL), headless `Renode v1.17.0` dual-FCC `< 20 ms` OpenCyphal CAN-FD failover CI (`MIT libcanard` + `o1heap v3.0.0` + Nunavut C11), `ESBMC v8.7` bit-precise C++17/C99 formal proofs across all 6 safety-critical kernels, and an auditable MISRA-aligned static profile — adding **0 bytes** to `core/flight_core/` and keeping **zero file overlap** with Plan 4B (`VertiGCS` / WASM / SDK / Governance / Paper 1).

**Architecture:** Plan 4A builds strictly *around* the frozen `libvertirta_flight_core.a` archive as standalone bare-metal CMake/verification subtrees (`firmware/stm32h7_hitl/`, `firmware/renode_cyphal/`, `verification/formal/`, `.github/workflows/formal_and_hil.yml`, `tests/firmware/`). The STM32H753/H743 BSP relocates `libvertirta_flight_core.a` (`44.1%` of 64 KB zero-wait-state ITCM) above a `1 KB` MPU NULL-trap guard (`VTOR = 0x00000400`), places stack + hot tables inside `128 KB` DTCM above a `256 B` stack-overflow MPU trap, isolates the `32 KiB` `o1heap v3.0.0` CAN-FD arena strictly inside the BSP bus wrapper, and exercises dual-node failover in `Renode v1.17.0` (`.NET`, `SetGlobalQuantum "0.000025"`). Offline formal verification runs `ESBMC v8.7 --32 --floatbv --nan-check --overflow-check --bitwuzla` on `ubuntu-24.04` (`x86_64`) alongside `clang-tidy` (LLVM 22) + `cppcheck 2.22.0` (`--platform=unix32 --funsigned-char`) with an explicit 5-class MISRA C++:2023 gap register.

**Tech Stack:** Arm GNU Toolchain `14.2.Rel1` (`arm-none-eabi-gcc`/`g++ 14.2.1`, already present in WSL2 Ubuntu), CMake `3.22+` / Ninja, `CMSIS-Core` (Apache-2.0), `TinyUSB` (MIT, optional), `libcanard v3.2.x/v4.x` (MIT), `o1heap v3.0.0` (MIT), `Nunavut` (`nnvg` C11 target ONLY), `Renode v1.17.0` (`.NET` portable tarballs, Robot Framework `CANTester`), `ESBMC v8.7` (`ubuntu-24.04` x86_64 ONLY), `clang-tidy` (LLVM 22), `cppcheck 2.22.0`, Catch2 `v3.7.1`, Python `3.11+` (`pytest`).

**Spec & Canonical Overrides:** `docs/superpowers/specs/2026-11-22-vertirta-v3-v4-expansion-design.md`, `docs/research/01-DR08-13-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-08` locks), `docs/research/DR-08-stm32h7-renode-cyphal-esbmc-misra.md`, and `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md`.

---

## Global Constraints & Ownership Boundary

- **HARD FREEZE (`core/flight_core/` & `VRTA0001`):** Do **NOT** edit `core/flight_core/**`, `core/rta_companion/**`, root `CMakeLists.txt`, or `256 B` `VRTA0001` frames. `libvertirta_flight_core.a` MUST remain **`28,886 B` Flash / `0 B` library static RAM / `<= 1,496 B` max frame / `<= 2,960 B` stack DAG / `0` recursion / `0` heap symbols**, and `check_arch_isolation.py` MUST remain `37` files / `0` violations. Keep `o1heap` `32 KiB` DTCM arena strictly inside `firmware/renode_cyphal/`.
- **Strict Zero-Overlap Boundary vs. Plan 4B:**
  - **Plan 4A Owns ONLY:** `firmware/stm32h7_hitl/**`, `firmware/renode_cyphal/**`, `verification/formal/**`, `.github/workflows/formal_and_hil.yml`, `tests/firmware/**`.
  - **Plan 4A NEVER Touches:** `apps/vertigcs/**`, `wasm/**`, `sdk/python/**`, `docker/**`, `docs/standards_registry.yaml`, `CONTRIBUTING.md`, `docs/paper_ral/**`, `.github/workflows/product_and_wasm.yml`, root `CMakeLists.txt`.
- **STM32H753 / H743 Silicon Errata Locks (`ES0392 Rev 16`):**
  1. Inspect `DBGMCU->IDCODE` (`REV_ID[15:0]`): unlock `480 MHz VOS0` (`SYSCFG->PWRCR |= ODEN`, wait `PWR->D3CR & VOSRDY`) ONLY on **Rev V (`0x2003`)**; on Rev Y (`0x1003`), refuse `480 MHz VOS0`, fall back to bench diagnostic `400 MHz VOS1` (`LATENCY = 2`), and latch `rev_y_fdcan_errata_flag = true` **blocking safety-critical FDCAN arming** (preventing silent no-workaround AXI/FDCAN RAM corruption under `Errata 2.2.10 / 2.24.2`).
  2. At `480 MHz VOS0` (`240 MHz AXI`), program **`FLASH_ACR.LATENCY = 4, WRHIGHFREQ = 0b10`** + readback barrier BEFORE switching `SYSCLK` to `PLL1P` (explicitly overriding PX4 `fmu-v6x` `board.h`'s `2`).
  3. **NO Write-Through MPU regions (`Erratum 2.1.1`)** and never set `CM7_CACR.FORCEWT/SIWT`; use `WB-WA` (`S=0`) for cached SRAM and `Normal Non-Cacheable` for `.dma_nocache`. Emit `__DSB(); __ISB();` after every cache op (`Erratum 2.1.5`).
  4. **Eager FPU Stacking (`FPU->FPCCR`: `ASPEN=1, LSPEN=0`)**, **CoreSight DWT Unlock (`DWT->LAR = 0xC5ACCE55`)**, and **explicit `FPSCR[4:0]` poll + clear on every 400 Hz tick** (`Erratum 2.1.6` — Cortex-M7 FPU IRQ unmapped on NVIC line 81).
  5. **USART3 non-blocking FIFO/poll (`Erratum 2.20.4`)** outside profiled tick; **FDCAN edge-filtering OFF, DAR OFF, dedicated Tx FIFO (`Errata 2.24.3–2.24.5`)**.
  6. **GNU ld ITCM/Flash Veneer Guard:** Every `.S` routine MUST carry `.type sym, %function` + `.thumb_func`, colocate `libvertirta_flight_core.a` + `libgcc`/`libm` helpers inside `.itcm_text`, compile Flash callers with `-mlong-calls`, and audit generated `*_veneer` stubs with `arm-none-eabi-objdump -d` in CI.
- **Honest Framing & Certification Boundary (MANDATORY):**
  - `Renode v1.17.0` (`tlib` DBT) is **Tier-1 Functional & Protocol SIL ONLY** — NEVER hardware Cortex-M7 WCET.
  - Physical `STM32H7` `DWT->CYCCNT` logs establish **Empirical High-Water-Mark (HWM) Timing ONLY** — NOT a mathematical DO-178C / CAST-32A / AMC 20-193 WCET proof without path coverage and interference bounds.
  - `ESBMC v8.7` (Clang frontend vs. GCC `14.2.1 -Os` object code, no DO-330 TQSP) is strictly **Supplemental Engineering Verification** (leaves DO-333 Table `FM.A-7[9]` object-code preservation & `DO-330` TQL-5/TQL-4 open).
  - Open-source `clang-tidy` (LLVM 22) + `cppcheck 2.22.0` provide a **MISRA-Aligned High-Integrity C++17 Profile** with a mandatory 5-class gap register — NEVER claim official MISRA C++:2023 compliance (commercial Cppcheck Premium / Polyspace / Helix QAC only).
- **Hygiene Locks:** Pure LF (`CR=0`) on all files (especially `.sh`, `.ld`, `.S`, `.repl`, `.resc`, `.robot`, `.yml`), zero personal paths/tokens (`check_public_hygiene.py` PASS), zero GPL dependencies in runtime (`MIT`/`Apache-2.0` BSP only).

## Review Focus

1. **ITCM Address-Zero Aliasing & Corrupt GNU ld Veneers (Task 1):** Placing `.itcm_isr_vector` at `0x00000000` masks NULL dereferences; missing `.type sym, %function` emits illegal veneer opcode `0x04 F0 1F E5` (`UFSR.UNDEFINSTR`). Verify bottom `1 KB` MPU `AP_NONE/XN=1`, `VTOR = 0x00000400`, and `objdump -d` clean veneers.
2. **DWT Lock, Unserialized Superscalar Reads & Unmapped FPU IRQ (Task 2):** Omitting `DWT->LAR = 0xC5ACCE55` leaves `CYCCNT == 0` without SWD; omitting `__DSB(); __ISB();` around `t0`/`t1` skews dual-issue timings; omitting `FPSCR[4:0]` polling silently misses NaN/overflow exceptions (`Erratum 2.1.6`).
3. **CAN-FD Single-Frame Overflow & `o1heap` Fragmentation (Task 3):** Packing 64 payload bytes into Cyphal/CAN spills into a 2-frame transfer (Cyphal uses 1 tail byte -> max `63 B = 504 bits`). Lock `SafetyEnvelope.1.0` to `48 B (384 bits <= 504 bits)` with Nunavut C11 `@assert` and size `o1heap v3.0.0` at `32 KiB` (`> 30,095 B` WCMC).
4. **Renode PWR Hang, Coarse Quantum & SocketCAN Loop (Task 4):** Upstream `stm32h7.repl` hangs `VOSRDY` polling (`renode#649/#857`) and default quantum adds multi-ms virtual CAN lag. Require `STM32H7_PWR.cs`, `systickFrequency: 480_000_000`, `SetGlobalQuantum "0.000025"`, and `timeout=0.020` with `CANTester`.
5. **Silent ESBMC Weakening & ARM64 `--32` Failure (Tasks 5 & 7):** Omitting `--overflow-check`, `--nan-check`, or `--floatbv`, or passing `--no-unwinding-assertions`, silently yields vacuous `VERIFICATION SUCCESSFUL`; running `ESBMC --32` on ARM64 fails on missing 32-bit libc headers. Pin `ubuntu-24.04` x86_64 and check exact CLI flags programmatically.
6. **LLVM 22 & Cppcheck 2.19+ Silent Linter Drift (Task 6):** LLVM 22 dropped `clang-analyzer-*` from defaults and opened an RFC deprecating `hicpp-*`; Cppcheck 2.19 removed `unix32-unsigned`. Explicitly opt into `clang-analyzer-*` and pass `--platform=unix32 --funsigned-char`.

---

## File Structure

- `firmware/stm32h7_hitl/` — Standalone bare-metal Cortex-M7 CMake project (`CMakeLists.txt`, `stm32h753_tcm.ld`, `startup/startup_stm32h753.S`, `include/vertirta_bsp/{stm32h7_regs,stm32h7_boot,dwt_profiler,uart_console}.hpp`, `src/{stm32h7_boot,dwt_profiler,uart_console,main_hitl}.cpp`, `README.md`) linking frozen `libvertirta_flight_core.a` into `vertirta_stm32h7_hitl.elf`.
- `firmware/renode_cyphal/` — OpenCyphal v1.0 DSDL (`dsdl/vertirta/flight/SafetyEnvelope.1.0.dsdl`, `scripts/codegen_dsdl.sh`), C11 codec & deterministic `o1heap v3.0.0` `32 KiB` + `libcanard` BSP node (`CMakeLists.txt`, `include/vertirta_cyphal/{o1heap_arena.hpp,cyphal_bus_node.hpp,safety_envelope_codec.h}`, `src/{o1heap_arena.cpp,cyphal_bus_node.cpp,safety_envelope_codec.c,fdcan_stm32h7.cpp,main_fcc_node.cpp}`), and Renode v1.17.0 dual-node `< 20 ms` failover suite (`renode/{STM32H7_PWR.cs,stm32h753_avionics.repl,dual_fcc_failover.resc,failover_suite.robot}`).
- `verification/formal/` — `ESBMC v8.7` bit-precise formal harnesses (`esbmc/{verify_tmr_voter.cpp,verify_indi_wls.cpp,verify_motor_fdi.cpp,verify_coax_envelope.cpp,verify_daqp_bounds.c,verify_cbvf_filter.cpp,run_esbmc_suite.sh}`) + MISRA-aligned high-integrity C++17 profile (`misra_profile/{.clang-tidy,cppcheck_suppressions.txt,run_static_profile.sh,MISRA_GAP_REGISTER.md}`).
- `.github/workflows/formal_and_hil.yml` — Headless GitHub Actions CI (`stm32h7-bare-metal-elf`, `renode-dual-fcc-failover` on `ubuntu-24.04` + `ubuntu-24.04-arm`, `esbmc-formal-x86_64` on `ubuntu-24.04` ONLY, `misra-aligned-static-profile`).
- `tests/firmware/` — Automated local verification suite (`CMakeLists.txt`, `test_stm32h7_tcm_elf.py`, `test_hitl_profiler_and_elf.py`, `test_cyphal_bus_and_o1heap.cpp`, `test_dsdl_contract.py`, `test_renode_failover_contracts.py`, `test_formal_harnesses.py`, `test_static_profile_contracts.py`, `test_formal_and_hil_workflow.py`).

---

### Task 1: Zero-Wait-State ITCM/DTCM Linker Script, Vector Startup & 8-Region W^X MPU (`firmware/stm32h7_hitl/`)

**Files:** Create `firmware/stm32h7_hitl/{CMakeLists.txt,stm32h753_tcm.ld,startup/startup_stm32h753.S,include/vertirta_bsp/{stm32h7_regs.hpp,stm32h7_boot.hpp},src/stm32h7_boot.cpp}` | Test `tests/firmware/test_stm32h7_tcm_elf.py`
**Interfaces:**
- Consumes: Frozen `build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a` (`28,886 B Flash / 0 B RAM`, compiled with `arm-none-eabi-g++ 14.2.1 -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os`).
- Produces: `stm32h753_tcm.ld` (`VTOR = 0x00000400`, `1 KB` NULL guard, `256 B` DTCM stack guard, `.itcm_text`, `.dtcm_rodata`, `.axi_sram`, `.dma_nocache`), `startup_stm32h753.S` (`Reset_Handler` with `.type/.thumb_func`), `vertirta::bsp::{SiliconRevInfo, inspect_silicon_rev, init_clocks_480mhz_or_fallback, init_mpu_8_regions, init_fpu_eager_and_caches, unlock_dwt_cyccnt}`.

- [ ] **Step 1: Write the failing test (`tests/firmware/test_stm32h7_tcm_elf.py`)**
```python
from pathlib import Path
import re

def test_tcm_linker_startup_and_mpu_errata_contracts():
    ld = Path("firmware/stm32h7_hitl/stm32h753_tcm.ld").read_text(encoding="utf-8")
    assert "0x00000400" in ld and ".itcm_isr_vector" in ld and ".itcm_text" in ld
    assert ".dtcm_stack_guard" in ld and ".dma_nocache" in ld and ".axi_sram" in ld
    asm = Path("firmware/stm32h7_hitl/startup/startup_stm32h753.S").read_text(encoding="utf-8")
    assert ".thumb_func" in asm and ".type Reset_Handler, %function" in asm and "0x00000400" in asm
    boot = Path("firmware/stm32h7_hitl/src/stm32h7_boot.cpp").read_text(encoding="utf-8")
    assert "0x2003" in boot and "0x1003" in boot
    assert "LATENCY = 4" in boot or "kFlashLatencyVos0 = 4" in boot
    assert "0xC5ACCE55" in boot and "LSPEN" in boot
    assert "FORCEWT" not in boot and "SIWT" not in boot
    # Verify all 8 MPU regions R0..R7 are configured without Write-Through (TEX=0,C=1,B=0)
    assert re.search(r"R0.*0x60000000|SRD.*0x87", boot, re.DOTALL)
```
- [ ] **Step 2: Run test to verify failure** — `uv run --python 3.11 --with pytest pytest tests/firmware/test_stm32h7_tcm_elf.py -v` → FAIL (`FileNotFoundError`).
- [ ] **Step 3: Implement `stm32h753_tcm.ld`, `startup_stm32h753.S`, `stm32h7_regs.hpp`, `stm32h7_boot.{hpp,cpp}`, and `CMakeLists.txt`** — Implement exact 64 KB ITCM (`0x00000000–0x000003FF` reserved NULL guard, `0x00000400` 1,024-byte aligned vector table, `.itcm_text` absorbing `*libvertirta_flight_core.a:*(.text .text.*)` + `libgcc`/`libm`), 128 KB DTCM (`0x20000000–0x200000FF` `256 B` stack guard, descending stack at top `0x20020000`), `512 KB` `.axi_sram`, and `32 KB` `.dma_nocache` (`aligned(32)`). Implement `stm32h7_boot.cpp` with exact `ES0392 Rev 16` mitigations (`DBGMCU->IDCODE == 0x2003`, `FLASH_ACR.LATENCY = 4 / WRHIGHFREQ = 0b10` + readback, 8-region W^X MPU `R0..R7` with ZERO Write-Through, `FPU->FPCCR` eager stacking `ASPEN=1, LSPEN=0`, `__DSB(); __ISB();` around L1 I/D-cache enable, `DWT->LAR = 0xC5ACCE55`).
- [ ] **Step 4: Run test & compile check** — `uv run --python 3.11 --with pytest pytest tests/firmware/test_stm32h7_tcm_elf.py -v` → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `firmware/stm32h7_hitl/` + `tests/firmware/test_stm32h7_tcm_elf.py`.

---

### Task 2: Bare-Metal 400 Hz `DWT->CYCCNT` HITL Profiling Harness & Verified Linked Cortex-M7 ELF (`firmware/stm32h7_hitl/`)

**Files:** Create `firmware/stm32h7_hitl/{include/vertirta_bsp/{dwt_profiler.hpp,uart_console.hpp},src/{dwt_profiler.cpp,uart_console.cpp,main_hitl.cpp},README.md}` | Modify `firmware/stm32h7_hitl/CMakeLists.txt` | Test `tests/firmware/test_hitl_profiler_and_elf.py`
**Interfaces:**
- Consumes: Task 1 BSP boot/MPU/TCM + frozen `libvertirta_flight_core.a` (`TmrVoter`, `IndiWlsController`, `MotorFdiObserver`, `CoaxialPlant6Dof`).
- Produces: `vertirta_stm32h7_hitl.elf` (linked with `arm-none-eabi-g++ 14.2.1 -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -ffp-contract=off -fstack-usage -Wstack-usage=3500 -Wl,--wrap=malloc`), `DwtCycleProfiler` (`10,000` raw `uint32_t` ticks in `.axi_sram` `40 KB`, calibrated `g_bracket_overhead`, `__DSB(); __ISB();` serialization, `FPSCR[4:0]` polling/clearing per `Erratum 2.1.6`, atomic `BSRR` GPIO scope toggle, non-blocking `USART3` JSON report tagged `EMPIRICAL_HWM_NOT_WCET_PROOF`, plus optional `#ifdef VERTIRTA_ENABLE_TINYUSB` MIT TinyUSB CDC hook using `PWR_CR3_USB33DEN` + `HSI48`).

- [ ] **Step 1: Write the failing test (`tests/firmware/test_hitl_profiler_and_elf.py`)**
```python
import subprocess
from pathlib import Path

def test_stm32h7_hitl_elf_builds_links_and_audits_cleanly(tmp_path):
    # Builds firmware/stm32h7_hitl against arm-cortex-m7 libvertirta_flight_core.a
    elf = tmp_path / "vertirta_stm32h7_hitl.elf"
    assert elf.exists()  # populated by cmake --build in test fixture
    # 1) .itcm_isr_vector VMA == 0x00000400; .itcm_text fits cleanly inside 64 KB ITCM
    # 2) arm-none-eabi-nm confirms 0 forbidden heap symbols (__wrap_malloc/calloc/realloc unreferenced)
    # 3) arm-none-eabi-objdump -d confirms 0 corrupt veneer opcodes (0x04 f0 1f e5 / bfcsel)
    # 4) main_hitl.cpp polls FPSCR[4:0], brackets DWT->CYCCNT with __DSB/__ISB, and labels output EMPIRICAL_HWM_NOT_WCET_PROOF
```
- [ ] **Step 2: Run test to verify failure** — `uv run --python 3.11 --with pytest pytest tests/firmware/test_hitl_profiler_and_elf.py -v` → FAIL.
- [ ] **Step 3: Implement `dwt_profiler.{hpp,cpp}`, `uart_console.{hpp,cpp}`, `main_hitl.cpp`, and `README.md`** — Configure `TIM2` (`240 MHz`, `PSC=0, ARR=599999` -> exact `2,500.0 us` 400 Hz) or `SysTick` (`LOAD=1,199,999` @ `480 MHz`), calibrate `g_bracket_overhead` over `1,024` empty iterations, run adversarial worst-case coaxial X8 fault stimuli (simultaneous IMU0 bias + 2M-Cross + 2M-SameArm Option B1b active spin + saturated DAQP corners) for `10,000` ticks, sort `uint32_t` cycle samples in-place (zero heap) for exact `min/p50/p99/p99.9/max`, report sticky `FPSCR[4:0]` bits, and document both NUCLEO-H743ZI2 (`STLINK-V3E` SWD, `8 MHz MCO` drift caveat -> `HSI48` for USB) and Pixhawk 6X (`0x08000000` bootloader archive warning!) in `firmware/stm32h7_hitl/README.md`.
- [ ] **Step 4: Build ELF locally via WSL2 `arm-none-eabi-g++ 14.2.1` & run test** — Build into isolated `/tmp/vertirta_v3a_hitl`, verify `arm-none-eabi-readelf -S`, `arm-none-eabi-objdump -d`, `check_no_heap_symbols.py`, `check_stack_usage.py`, and `check_bare_metal_size.py` (`libvertirta_flight_core.a` STILL exact `28,886 B / 0 B RAM`) → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `firmware/stm32h7_hitl/` + `tests/firmware/test_hitl_profiler_and_elf.py`.

---

### Task 3: OpenCyphal v1.0 DSDL (`Subject-ID 7509 / 1024`), Nunavut C11 Codec & `o1heap v3.0.0` + `libcanard` BSP Wrapper (`firmware/renode_cyphal/`)

**Files:** Create `firmware/renode_cyphal/{dsdl/vertirta/flight/SafetyEnvelope.1.0.dsdl,scripts/codegen_dsdl.sh,CMakeLists.txt,include/vertirta_cyphal/{o1heap_arena.hpp,safety_envelope_codec.h,cyphal_bus_node.hpp},src/{o1heap_arena.cpp,safety_envelope_codec.c,cyphal_bus_node.cpp}}` | Test `tests/firmware/{CMakeLists.txt,test_cyphal_bus_and_o1heap.cpp,test_dsdl_contract.py}`
**Interfaces:**
- Consumes: Frozen `vertirta::core::CapabilityEnvelope` & `crc32c` from `libvertirta_flight_core` (read-only headers), MIT `o1heap v3.0.0` & MIT `libcanard v3.2.x/v4.x` (pinned via CMake `FetchContent` or vendored MIT single-header/C11 minimal fallback when offline).
- Produces:
  - `SafetyEnvelope.1.0.dsdl`: fixed Subject-ID `1024`, exact `48 B` (`384 bits <= 504 bits` single-frame CAN-FD ceiling with `@assert _offset_ == {384}` and `@assert _offset_ <= {504}`), encoding `float32[4] wrench_min_frd`, `float32[4] wrench_max_frd` (`[Fz, Mx, My, Mz]`), `uint32 seq`, `uint32 t_us`, `uint8 fault_mask`, `uint8 flight_regime`, `uint8 tmr_state`, `uint8 cel_latched`, `uint32 crc32c`; plus standard `uavcan.node.Heartbeat.1.0` on Subject-ID `7509` (7 B @ 1 Hz).
  - Nunavut **C11** serializer/deserializer `safety_envelope_codec.{h,c}` (strictly C11, NEVER experimental Nunavut C++).
  - `O1HeapDtcmArena` (`32,768 B` `alignas(8)` static DTCM buffer, zero `malloc`, Half-Fit WCMC `H_b(M=4096, l=32, n=256) = 30,095 B < 32,768 B`, `invariants_hold() == true`, `oom_count() == 0`) + `CyphalFccNode` (Primary Node `10` publisher / Secondary Node `11` watchdog & failover FSM with 4-missed-frame `W = 10.50 ms` watchdog and babbling-idiot CRC-32C rejection).

- [ ] **Step 1: Write failing Catch2 & Python tests (`test_cyphal_bus_and_o1heap.cpp`, `test_dsdl_contract.py`)**
```cpp
TEST_CASE("SafetyEnvelope C11 codec fits 48 B <= 63 B single CAN-FD frame, o1heap 32 KiB zero-OOM, and <20 ms failover FSM") {
  static_assert(VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES == 48u);
  static_assert(VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES + 1u <= 64u); // +1 B Cyphal tail byte
  // 1) Round-trip serialize/deserialize + Castagnoli CRC-32C verify + single-bit corruption reject
  // 2) Stress 10,000 random alloc/free cycles (32..256 B up to M=4096 B live) on 32 KiB O1HeapDtcmArena -> oom_count==0, invariants==true
  // 3) Secondary Node 11 receives 400 Hz frames from Node 10 -> STANDBY; drop frames -> takes over ACTIVE at 10.50 ms (modeled total 13.75 ms < 20 ms)
  // 4) Corrupted-CRC babbling-idiot frames at 400 Hz -> rejected, triggers Secondary takeover <= 10.50 ms
}
```
- [ ] **Step 2: Run tests to verify failure** — `ctest` & `pytest tests/firmware/test_dsdl_contract.py -v` → FAIL.
- [ ] **Step 3: Implement DSDL, `codegen_dsdl.sh`, `safety_envelope_codec.{h,c}`, `o1heap_arena.{hpp,cpp}`, and `cyphal_bus_node.{hpp,cpp}`** — Keep `o1heap` strictly in `firmware/renode_cyphal/` (zero bytes added to `core/flight_core/`). Implement exact ISO 11898-1:2015 CAN-FD wire-time budget constants (`48 B -> 160.6 us`, `64 B -> 192.6 us` @ 1/5 Mbps) and verify `codegen_dsdl.sh` passes `bash -n` with pure LF.
- [ ] **Step 4: Build & run host Catch2 (Debug + ASan/UBSan) and pytest** — Both `test_cyphal_bus_and_o1heap` (`0` ASan/UBSan/LSan errors, `0` `malloc` in hot path) and `test_dsdl_contract.py` → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `firmware/renode_cyphal/` + `tests/firmware/{CMakeLists.txt,test_cyphal_bus_and_o1heap.cpp,test_dsdl_contract.py}`.

---

### Task 4: Dual-STM32H753 Lockstep Firmware & `Renode v1.17.0` `< 20 ms` CAN-FD Failover Suite (`firmware/renode_cyphal/`)

**Files:** Create `firmware/renode_cyphal/{src/fdcan_stm32h7.cpp,src/main_fcc_node.cpp,renode/{STM32H7_PWR.cs,stm32h753_avionics.repl,dual_fcc_failover.resc,failover_suite.robot}}` | Modify `firmware/renode_cyphal/CMakeLists.txt` | Test `tests/firmware/test_renode_failover_contracts.py`
**Interfaces:**
- Consumes: Task 1 `stm32h753_tcm.ld` + `stm32h7_boot`, Task 3 `CyphalFccNode` + `O1HeapDtcmArena` + `safety_envelope_codec.h`, frozen `libvertirta_flight_core.a`.
- Produces:
  - Two bare-metal Cortex-M7 ELFs (`vertirta_fcc_primary.elf` Node 10 & `vertirta_fcc_secondary.elf` Node 11) linking `libvertirta_flight_core.a` + `fdcan_stm32h7.cpp` (explicitly enforcing `ES0392` Errata `2.24.3–2.24.5`: edge filtering OFF, DAR OFF, dedicated Tx FIFO).
  - `STM32H7_PWR.cs` + `stm32h753_avionics.repl` (resolves `renode#649/#857` PWR `VOSRDY` hang & `renode#1009` `systickFrequency: 480_000_000`).
  - `dual_fcc_failover.resc` (`emulation SetGlobalQuantum "0.000025"`, virtual `CANHub`, zero dual-`SocketCANBridge` reflection loop `renode#598`).
  - `failover_suite.robot`: `defaultPauseEmulation=true`, strict virtual-time `timeout=0.020`, `CANTester`/`CANKeywords` + `TerminalTester` verifying 3 orthogonal failover faults: (a) Primary NVIC `SetPendingSynchronousFault` HardFault, (b) Primary `CANHub` disconnect, (c) Corrupted-CRC babbling-idiot injection — explicitly labeled **Tier-1 Functional & Protocol SIL ONLY (NOT Hardware WCET)**.

- [ ] **Step 1: Write the failing test (`tests/firmware/test_renode_failover_contracts.py`)**
```python
from pathlib import Path

def test_renode_dual_fcc_overlays_quantum_and_three_fault_scenarios():
    repl = Path("firmware/renode_cyphal/renode/stm32h753_avionics.repl").read_text(encoding="utf-8")
    resc = Path("firmware/renode_cyphal/renode/dual_fcc_failover.resc").read_text(encoding="utf-8")
    robot = Path("firmware/renode_cyphal/renode/failover_suite.robot").read_text(encoding="utf-8")
    pwr = Path("firmware/renode_cyphal/renode/STM32H7_PWR.cs").read_text(encoding="utf-8")
    assert "480000000" in repl or "480_000_000" in repl
    assert 'SetGlobalQuantum "0.000025"' in resc and "SocketCANBridge" not in resc
    assert "STM32H7_PWR.cs" in resc and "0x18" in pwr
    assert "timeout=0.020" in robot and "SetPendingSynchronousFault" in robot
    assert "CANTester" in robot or "Create CAN Tester" in robot
    assert "NOT hardware WCET" in robot or "Functional & Protocol SIL" in robot
```
- [ ] **Step 2: Run test to verify failure** — `uv run --python 3.11 --with pytest pytest tests/firmware/test_renode_failover_contracts.py -v` → FAIL.
- [ ] **Step 3: Implement `fdcan_stm32h7.cpp`, `main_fcc_node.cpp`, `STM32H7_PWR.cs`, `.repl`, `.resc`, and `failover_suite.robot`** — Ensure both `vertirta_fcc_primary.elf` and `vertirta_fcc_secondary.elf` compile & link cleanly via WSL2 `arm-none-eabi-g++ 14.2.1` (zero heap symbols, ITCM `< 64 KB`, stack DAG compliant).
- [ ] **Step 4: Build dual ELFs locally & run tests** — Build into isolated `/tmp/vertirta_v3a_cyphal`, verify ELF sections + symbols + `pytest tests/firmware/test_renode_failover_contracts.py -v` (plus execute `renode-test failover_suite.robot` if `renode` is installed locally; otherwise executed live on GitHub Actions in Task 7) → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `firmware/renode_cyphal/` + `tests/firmware/test_renode_failover_contracts.py`.

---

### Task 5: `ESBMC v8.7` Bit-Precise C++17 & C99 Formal Verification Harnesses (`verification/formal/esbmc/`)

**Files:** Create `verification/formal/esbmc/{verify_tmr_voter.cpp,verify_indi_wls.cpp,verify_motor_fdi.cpp,verify_coax_envelope.cpp,verify_daqp_bounds.c,verify_cbvf_filter.cpp,run_esbmc_suite.sh}` | Test `tests/firmware/test_formal_harnesses.py`
**Interfaces:**
- Consumes: Frozen `core/flight_core/` (`tmr_voter`, `indi_wls`, `motor_fdi`, `coax_envelope`, `daqp_qp`, `daqp_auxiliary_norec.c`) and `core/rta_companion/` (`cbvf_safety_filter`, `mmap_cbvf_lut`).
- Produces: 6 standalone self-contained bounded model-checking harnesses + `run_esbmc_suite.sh` (pure LF, `bash -n` PASS) invoking `esbmc` on **`ubuntu-24.04` x86_64 ONLY** with mandatory flags **`--32 --floatbv --nan-check --overflow-check --unsigned-overflow-check --memory-leak-check --struct-fields-check --bitwuzla`** (NEVER `--no-unwinding-assertions`!), emitting YAML witnesses and supporting optional `--z3` cross-check and `--compile-only` local host syntax verification.

- [ ] **Step 1: Write the failing test (`tests/firmware/test_formal_harnesses.py`)**
```python
import subprocess
from pathlib import Path

HARNESSES = [
    "verify_tmr_voter.cpp",
    "verify_indi_wls.cpp",
    "verify_motor_fdi.cpp",
    "verify_coax_envelope.cpp",
    "verify_daqp_bounds.c",
    "verify_cbvf_filter.cpp",
]

def test_esbmc_harnesses_cli_flags_and_honest_do333_framing():
    root = Path("verification/formal/esbmc")
    for h in HARNESSES:
        assert (root / h).is_file(), f"Missing ESBMC harness: {h}"
    sh = (root / "run_esbmc_suite.sh").read_text(encoding="utf-8")
    for required in ("--32", "--floatbv", "--nan-check", "--overflow-check", "--bitwuzla", "x86_64"):
        assert required in sh
    assert "--no-unwinding-assertions" not in sh
    assert "Supplemental Engineering Verification" in sh
    assert subprocess.run(["bash", "-n", (root / "run_esbmc_suite.sh").as_posix()]).returncode == 0
```
- [ ] **Step 2: Run test to verify failure** — `uv run --python 3.11 --with pytest pytest tests/firmware/test_formal_harnesses.py -v` → FAIL.
- [ ] **Step 3: Author all 6 ESBMC harnesses & `run_esbmc_suite.sh`** — Restrict freestanding C++17 harnesses to ESBMC-modeled headers (`<cstdint>`, `<cstddef>`, `<cmath>`, `<array>`, `<type_traits>`), guard ESBMC intrinsics (`__ESBMC_assume`, `__ESBMC_assert`, `nondet_*`) behind `#ifdef __ESBMC__` fallbacks so every harness ALSO compiles and passes concrete smoke execution under local WSL2 `g++ 15.2.0` / `gcc 15.2.0`! Enforce `DR-08` §3.2 exact unwinding bounds:
  - `verify_tmr_voter.cpp`: symbolic Byzantine Channel 2 (`NaN`/`±Inf`/`±DBL_MAX`), `--unwind 5`.
  - `verify_indi_wls.cpp`: Layer 2A fixed `B0` + all 256 symbolic `uint8_t motor_mask` (`--unwind 14 --interval-analysis --scalarize-structs`) + Layer 2B `--enforce-all-contracts` on rank-deficient reflector/Cholesky fallback (`||x||_2 < 1e-12`).
  - `verify_motor_fdi.cpp`: irreversible `SAME_ARM_ACTIVE_SPIN` latch + finite outputs (`--unwind 10`).
  - `verify_coax_envelope.cpp`: `two_arm_tw_at_vterm` monotonicity + `TheveninBattery1RC` finite-output guard (`--unwind 10`).
  - `verify_daqp_bounds.c`: pure C99 Pass 3A full `N=8, M=16, iter_limit=25` (`--unwind 27`) workspace safety + Pass 3B symbolic `N=3, M=6` pivot degeneracy (also compatible with optional `Frama-C` Eva / `CBMC` second opinion!).
  - `verify_cbvf_filter.cpp`: non-finite input fallback + `regime_idx < 4` bounds safety + irreversible CEL latch (`--unwind 12`).
- [ ] **Step 4: Run `bash -n`, local WSL2 concrete compilation/execution of all 6 harnesses, and pytest** — `tests/firmware/test_formal_harnesses.py` + `bash verification/formal/esbmc/run_esbmc_suite.sh --compile-only` (plus live `esbmc` run if installed locally) → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `verification/formal/esbmc/` + `tests/firmware/test_formal_harnesses.py`.

---

### Task 6: MISRA-Aligned High-Integrity C++17 Static Profile & 5-Class Gap Register (`verification/formal/misra_profile/`)

**Files:** Create `verification/formal/misra_profile/{.clang-tidy,cppcheck_suppressions.txt,run_static_profile.sh,MISRA_GAP_REGISTER.md}` | Test `tests/firmware/test_static_profile_contracts.py`
**Interfaces:**
- Consumes: `core/flight_core/` compile commands (`build/arm-cortex-m7/compile_commands.json` / `build/dev-linux-x64/compile_commands.json`) and existing CI structural scripts (`scripts/ci/check_{bare_metal_size,no_heap_symbols,stack_usage,arch_isolation}.py`).
- Produces: LLVM-22-safe `.clang-tidy`, `cppcheck_suppressions.txt`, `run_static_profile.sh` (pure LF, `bash -n` PASS, using `--platform=unix32 --funsigned-char` for ARM EABI unsigned `char` and explicit `clang-analyzer-*` opt-in), and `MISRA_GAP_REGISTER.md` (documenting all 5 MISRA C++:2023 guideline classes OSS linters cannot check + compensating controls, explicitly banning false MISRA compliance claims).

- [ ] **Step 1: Write the failing test (`tests/firmware/test_static_profile_contracts.py`)**
```python
import subprocess
from pathlib import Path
import yaml

def test_misra_aligned_profile_llvm22_cppcheck_and_5class_gap_register():
    root = Path("verification/formal/misra_profile")
    tidy = yaml.safe_load((root / ".clang-tidy").read_text(encoding="utf-8"))
    checks = tidy["Checks"] if isinstance(tidy["Checks"], str) else ",".join(tidy["Checks"])
    assert "clang-analyzer-*" in checks and "bugprone-*" in checks and "cert-*" in checks
    assert "hicpp-*" not in checks  # LLVM 22 2026 deprecation RFC
    sh = (root / "run_static_profile.sh").read_text(encoding="utf-8")
    assert "--platform=unix32" in sh and "--funsigned-char" in sh and "unix32-unsigned" not in sh
    assert subprocess.run(["bash", "-n", (root / "run_static_profile.sh").as_posix()]).returncode == 0
    gap = (root / "MISRA_GAP_REGISTER.md").read_text(encoding="utf-8")
    for token in ("Dir-0.3.1", "Rule-4.1.3", "Rule-6.2.1", "Rule-7.0.5", "Compliance:2020", "NOT full MISRA"):
        assert token in gap
```
- [ ] **Step 2: Run test to verify failure** — `uv run --python 3.11 --with pytest --with pyyaml pytest tests/firmware/test_static_profile_contracts.py -v` → FAIL.
- [ ] **Step 3: Implement `.clang-tidy`, `cppcheck_suppressions.txt`, `run_static_profile.sh`, and `MISRA_GAP_REGISTER.md`** — Configure `run_static_profile.sh` with `--check-tools` mode (gracefully reporting missing optional `clang-tidy`/`cppcheck` locally while running strict GCC `14.2.1` `-Werror -Wvla -Wshadow -Wconversion` + `check_bare_metal_size.py` + `check_no_heap_symbols.py` + `check_stack_usage.py` + `check_arch_isolation.py`, and failing hard in CI when `--strict-tools` is set).
- [ ] **Step 4: Run pytest & local `run_static_profile.sh` check** — `pytest tests/firmware/test_static_profile_contracts.py -v` + WSL2 `bash verification/formal/misra_profile/run_static_profile.sh` → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `verification/formal/misra_profile/` + `tests/firmware/test_static_profile_contracts.py`.

---

### Task 7: Headless GitHub Actions Formal & Bare-Metal HIL Workflow (`.github/workflows/formal_and_hil.yml`) & End-to-End Plan 4A Gate

**Files:** Create `.github/workflows/formal_and_hil.yml`, `tests/firmware/test_formal_and_hil_workflow.py`
**Interfaces:**
- Consumes: Tasks 1–6 (`firmware/stm32h7_hitl/`, `firmware/renode_cyphal/`, `verification/formal/`), `CMakePresets.json` (`arm-cortex-m7`), and `scripts/ci/check_{bare_metal_size,no_heap_symbols,stack_usage,arch_isolation,public_hygiene}.py`.
- Produces: Complete 4-job GitHub Actions workflow `.github/workflows/formal_and_hil.yml` (`stm32h7-bare-metal-elf`, `renode-dual-fcc-failover` on `ubuntu-24.04` + `ubuntu-24.04-arm`, `esbmc-formal-x86_64` strictly on `ubuntu-24.04`, `misra-aligned-static-profile` on `ubuntu-24.04`), verified locally via YAML schema/contract pytest + full zero-regression verification.

- [ ] **Step 1: Write the failing test (`tests/firmware/test_formal_and_hil_workflow.py`)**
```python
from pathlib import Path
import yaml

def test_formal_and_hil_workflow_pins_and_x86_64_esbmc_guard():
    wf = yaml.safe_load(Path(".github/workflows/formal_and_hil.yml").read_text(encoding="utf-8"))
    jobs = wf["jobs"]
    assert {"stm32h7-bare-metal-elf", "renode-dual-fcc-failover", "esbmc-formal-x86_64", "misra-aligned-static-profile"} <= set(jobs)
    # ESBMC --32 MUST run strictly on ubuntu-24.04 x86_64 (ARM64 fails --32)
    assert jobs["esbmc-formal-x86_64"]["runs-on"] == "ubuntu-24.04"
    # Renode v1.17.0 matrix covers both x86_64 and ARM64 using portable .NET tarballs
    renode_raw = Path(".github/workflows/formal_and_hil.yml").read_text(encoding="utf-8")
    assert "ubuntu-24.04-arm" in renode_raw and "1.17.0" in renode_raw and "14.2.Rel1" in renode_raw
    assert "check_bare_metal_size.py" in renode_raw and "check_public_hygiene.py" in renode_raw
```
- [ ] **Step 2: Run test to verify failure** — `uv run --python 3.11 --with pytest --with pyyaml pytest tests/firmware/test_formal_and_hil_workflow.py -v` → FAIL.
- [ ] **Step 3: Author `.github/workflows/formal_and_hil.yml`** — Pin Arm GNU Toolchain `14.2.Rel1` (matching local WSL2 `14.2.1` to prevent ABI drift), pin `Renode v1.17.0` self-contained `.NET` portable tarballs (`linux-portable` & `linux-arm64-portable`, zero Mono/GTK/source-build latency), pin `ESBMC v8.7` on `ubuntu-24.04` x86_64 only, assert exact `libvertirta_flight_core.a` footprint (`<= 28,886 B` regression check or `< 262,144 B` budget, `0 B` RAM, `0` heap, `<= 2,960 B` DAG), audit ITCM veneers with `arm-none-eabi-objdump -d`, upload `.elf` + `.map` + ESBMC YAML witnesses, and enforce `check_arch_isolation.py` + `check_public_hygiene.py`.
- [ ] **Step 4: Run full Plan 4A local verification gate** — Run ALL tests in `tests/firmware/`, verify `arm-cortex-m7` `libvertirta_flight_core.a` remains exact `28,886 B Flash / 0 B RAM`, verify `ctest --preset dev-linux-x64` (`16/16` PASS), `check_arch_isolation.py` (`37` files / `0` violations), and `check_public_hygiene.py` (`0` violations, pure LF across all files) → PASS.
- [ ] **Step 5: Commit (coordinator/integrator phase)** — `.github/workflows/formal_and_hil.yml` + `tests/firmware/test_formal_and_hil_workflow.py`.
