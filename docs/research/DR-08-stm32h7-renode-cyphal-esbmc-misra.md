# Avionics Execution, Dual-Node CAN-FD Failover, and Formal Verification Blueprint for libvertirta_flight_core.a

Deploying libvertirta_flight_core.a (28,886 B Flash, 0 B static RAM, 2,960 B verified stack DAG, 0 recursion, compiled with arm-none-eabi-g++ 14.2.1 -mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -Os) into a flight-grade STM32H753 / NUCLEO-H743ZI2 architecture requires strictly isolating three verification domains: silicon-level cycle profiling on physical Cortex-M7 hardware, deterministic multi-node bus failover in headless CI, and formal absence-of-runtime-error proofs coupled with static coding-standard gates. Placing the entire 28,886 B flight-control archive inside the STM32H7's 64 KB zero-wait-state ITCM (44.1% utilization) and its descending stack plus hot read-only tables inside the 128 KB DTCM eliminates L1 cache-miss and Flash wait-state jitter across the 400 Hz (2,500 µs; 1,200,000 cycles at 480 MHz) control frame. Paired with a dual-STM32H753 Renode v1.17.0 (September 2026) OpenCyphal/CAN-FD testbench running MIT-licensed libcanard and o1heap v3.0.0, this architecture yields a modeled worst-case bus failover latency of 13.75 ms (6.25 ms margin below 20 ms) under 400 Hz single-frame safety supervision. [1][2][3]

Three hard certification boundaries govern how evidence from this 2024–2026 open-source toolchain must be framed to airworthiness authorities and technical auditors. First, dynamic binary translators (Renode and QEMU) cannot produce certifiable Cortex-M7 worst-case execution time (WCET) evidence because they omit superscalar dual-issue pairing, L1 cache associativity, branch-target caches, Flash wait states, and AXI bus contention; Renode evidence is strictly functional and protocol-level, whereas physical DWT->CYCCNT profiling establishes empirical high-water-mark (HWM) timing that must still be paired with path/interference analysis under DO-178C and CAST-32A / AMC 20-193. Second, among the three open-source formal verifiers evaluated (ESBMC, CBMC, and Frama-C), ESBMC v8.7 (November 2026) provides the most mature Clang-based C++17 frontend with native IEEE-754 Bitwuzla/Z3 SMT theories, yet it ships without a DO-330 Tool Qualification Support Pack (TQSP) and parses source through Clang rather than the flight compiler (GCC 14.2.1), leaving DO-333 Table FM.A-7 Objective 9 (source-to-object property preservation) and TQL-5/TQL-4 qualification open. Third, neither upstream clang-tidy (LLVM 22) nor open-source cppcheck (v2.22.0) provides official MISRA C++:2023 rule tables—their combined profile acts as a CI defect filter that cannot verify MISRA's 4 directives, undecidable lifetime/aliasing rules, cross-translation-unit ODR rules, or exact essential-type promotion semantics. [1][2][3]

Verification Pillar

2026 SOTA Tool / Target

Primary Evidence Output

Hard Certification / Technical Boundary

On-Target Cycle Profiling

Physical NUCLEO-H743ZI2 (MB1364) / Pixhawk 6X (FMUv6X, STM32H753 Rev V) + DWT->CYCCNT

Genuine cycle-accurate min/p99/max execution cycles under worst-case 400 Hz HITL stimuli

Empirical HWM only; not a mathematical WCET proof without path coverage and interference bounds (CAST-32A / AMC 20-193)

Multi-Node Bus Failover CI

Renode v1.17.0 (.NET, Sep 2026) + libcanard + o1heap v3.0.0 + Robot Framework

Deterministic <20 ms virtual-time failover (13.75 ms modeled budget) under NVIC fault injection on ubuntu-24.04 and ubuntu-24.04-arm

Functional/protocol SIL only; physical transceiver arbitration, bit-timing drift, and MCU latency require two-board HIL validation

Formal Methods (C++17 / C99)

ESBMC v8.7 (Nov 2026, Clang 18–22, Bitwuzla/Z3) + CBMC 6.x / Frama-C 33 (C99/C11)

Bit-precise IEEE-754 --floatbv, --nan-check, --overflow-check, and array/pointer safety proofs

Unqualified under DO-330 (no TQL-5/TQL-4 kit); GCC 14.2.1 -Os object-code preservation (FM.A-7[9]) unproven by source BMC

Static Coding Profile

clang-tidy (LLVM 22) + cppcheck (2.22.0 OSS) + arm-none-eabi-g++ 14.2.1 warnings

Automated CI gate catching UB, narrowing, casts, uninitialized members, and dead code

Zero official MISRA C++:2023 rule tables in OSS tools; Cppcheck MISRA C++:2023 requires commercial Premium

1. Bare-Metal STM32H753 / NUCLEO-H743ZI2 Execution & Hardware DWT->CYCCNT Profiling

1.1 Silicon Revision Hazards, Clock Tree, and Target Hardware Deltas

Bringing up bare-metal flight firmware across a ~$25 NUCLEO-H743ZI2 (MB1364) evaluation board and a Pixhawk 6X (FMUv6X, STM32H753; note that some FMUv6X variants such as ARKV6X populate STM32H743IIK6) requires explicit handling of silicon revision differences and errata documented in ES0392 Rev 16 (October 2026). Both STM32H743 and STM32H753 share the identical Arm Cortex-M7 core with double-precision FPv5-DP-D16 FPU, 16 KB 2-way I-Cache, 16 KB 4-way D-Cache (32-byte cache lines), 2 MB dual-bank Flash, 64 KB ITCM, 128 KB DTCM, 512 KB D1 AXI SRAM, 288 KB D2 SRAM (SRAM1/2/3), 64 KB D3 SRAM4, and 4 KB backup SRAM; the STM32H753 adds hardware cryptographic acceleration absent on the H743. [1][2][3][4][5][6][7][8][9][10][11][12][13]

Startup code must inspect DBGMCU->IDCODE (REV_ID[15:0]) before configuring PLL1. Revision Y/W (0x1003) is capped at 400 MHz (VOS1) and exhibits severe system errata—including AXI SRAM data read corruption (Erratum 2.2.10) and dual-FDCAN message RAM read corruption (Erratum 2.24.2, marked N with no workaround)—whereas Revision V (0x2003) unlocks 480 MHz (VOS0) via SYSCFG->PWRCR |= SYSCFG_PWRCR_ODEN (polling PWR->D3CR & PWR_D3CR_VOSRDY) and eliminates both memory corruptions. Note that ES0392 Rev 16 still lists Erratum 2.2.21 ("480 MHz maximum CPU frequency not available") with partial-workaround status (P) across revision columns, so production boards must be verified against the exact ST package/temperature/VOS0 LDO derating rules (1.26–1.40 V core supply). [1][2][3][4][5][6][7][8][9][10][11][12][13]

Flash wait states represent a major bare-metal bringup trap. At fCPU = 480 MHz (VOS0), the AXI/HCLK domain runs at 240 MHz (fCPU / 2, APB1–4 at 120 MHz). Per RM0433 Table 17, driving the Flash interface at 225–240 MHz under VOS0 requires FLASH_ACR.LATENCY = 4 and FLASH_ACR.WRHIGHFREQ = 0b10 (under VOS1 at 200 MHz AXI for 400 MHz Rev Y, LATENCY = 2 and WRHIGHFREQ = 0b10). Although PX4's boards/px4/fmu-v6x/nuttx-config/include/board.h defines #define BOARD_FLASH_WAITSTATES 2 alongside its 480 MHz PLL1 setup, standalone bare-metal firmware clocking AXI at 240 MHz must program LATENCY = 4, WRHIGHFREQ = 0b10 before switching SYSCLK to PLL1P and read back FLASH->ACR to confirm synchronization. [1][2][3][4][5][6][7][8][9][10][11][12][13]

Hardware / Silicon Parameter

NUCLEO-H743ZI2 (MB1364 / UM2407)

Pixhawk 6X (FMUv6X / DS-012)

Bare-Metal Failure Mode & Mitigation

MCU & Silicon Check

STM32H743ZIT6 (LQFP144); Rev V (0x2003) on MB1364 boards

STM32H753 (FMUv6X standard; Rev V 0x2003)

Read DBGMCU_IDC; fall back to 400 MHz VOS1 if 0x1003 (Rev Y)

HSE Clock Source

8 MHz MCO from STLINK-V3E on PF0/PH0 (SB45 ON, requires HSEBYP=1)

16 MHz crystal (HSEBYP=0), M1=1, N1=60, P1=2 -> 480 MHz

Community measurements show STLINK-V3E MCO derives from internal HSI (~8.08 MHz, ~1% drift); use HSI48 for USB or fit 25 MHz crystal X3 (SB3/4 ON, SB45 OFF)

Debug / Telemetry UART

USART3 VCP on PD8 (TX, SB19 ON) / PD9 (RX, SB12 ON)

USART3 (FMU Debug JST SM10B), UART7 (TELEM1), UART5 (TELEM2)

Erratum 2.20.4 (Rev V): DMA stream can lock on USART transfers; use TX FIFO + IRQ or non-blocking poll outside the profiled tick

USB-CDC Interface

USB2_OTG_FS on CN13 (PA11 DM, PA12 DP, PA9 VBUS, PA10 ID; SB76 ON)

USB2_OTG_FS Type-C (PA11 DM, PA12 DP); PLL3Q = 48 MHz

Enable USB 3.3 V supply detector (PWR_CR3_USB33DEN) before touching DWC2 registers

FDCAN1 / FDCAN2 Pins

Routed to ST morpho/Zio headers (external 3.3 V CAN-FD transceiver required)

FDCAN1: PD0 (RX) / PD1 (TX); FDCAN2: PB12 (RX), TX pin verified against full board.h / carrier DS-010/DS-012 schematic

Errata 2.24.3–2.24.5 (all revs): Edge-filtering desync, Tx FIFO inversion under mixed buffer/priority use, DAR arbitration failure; disable edge filtering and DAR, use dedicated Tx FIFO, and verify exact ES0392 Sec. 2.24 workarounds

Flash Boot Offset

Direct boot at 0x08000000 via STLINK-V3E SWD

PX4 bootloader occupies base of 0x08000000

SWD flashing at 0x08000000 overwrites the PX4 bootloader; archive original bootloader or offset .flash_isr_vector

Three Cortex-M7 core errata from ES0392 Rev 16 directly shape low-level BSP design across all STM32H743/H753 silicon revisions. First, Erratum 2.1.1 (Arm 1259864, Category A) causes silent data corruption when a specific sequence of stores, linefills or cache maintenance without DSB, and loads executes against Write-Through D-Cache memory; firmware must therefore never configure any MPU region as Write-Through and never set CM7_CACR.FORCEWT or SIWT, using strictly Write-Back (WB-WA) for cached SRAM and Normal Non-Cacheable for DMA buffers. Second, Erratum 2.1.5 creates an inconsistent memory view if a store follows a cache invalidate without an intervening barrier, while Erratum 2.1.4 risks D-Cache corruption if both cache error bank registers (DEBR0, DEBR1) are locked; enforce __DSB(); __ISB(); after every cache maintenance operation and leave error bank registers unlocked. Third, Erratum 2.1.6 (no workaround, N) leaves the Cortex-M7 FPU interrupt unmapped on NVIC line 81; because hardware FPU exceptions (IOC, DZC, OFC, UFC) never raise an interrupt, the 400 Hz tick epilogue must explicitly read and clear FPSCR[4:0] on every iteration. [1][2][3][4][5][6]

1.2 Zero-Wait-State ITCM/DTCM Linker Architecture and GNU ld Veneer Safety

Because libvertirta_flight_core.a consumes 28,886 B of code and 0 B of static RAM with a 2,960 B static stack DAG, placing the entire control archive and hot interrupt handlers inside ITCM (0x00000000, 64 KB) and all stack frames plus read-only control tables inside DTCM (0x20000000, 128 KB) turns the STM32H7 into a zero-wait-state Harvard execution engine. On Cortex-M7, TCM regions inherently behave as non-cacheable, non-shared Normal memory and bypass L1 I-Cache and D-Cache completely, eliminating cache-conflict jitter during QR-WLS factorizations and DAQP active-set iterations. [1][2][3][4][5][6]

Executing Cortex-M7 firmware concurrently from ITCM (0x00000000) and Flash (0x08000000) exposes two structural hazards that must be neutralized directly in the linker script and assembly startup. First, because ITCM maps directly to address zero, a null-pointer dereference silently reads or executes live ITCM contents rather than trapping. At the same time, the Cortex-M7 NVIC supports up to 240 external interrupts plus 16 system exceptions (256 × 4 B = 1,024 B), mandating 1,024-byte alignment for SCB->VTOR (0xE000ED08). Reserving 0x00000000–0x000003FF (1,024 B) as an MPU AP_NONE / XN=1 guard band traps null accesses as immediate MemManage faults while landing the relocated .itcm_isr_vector table cleanly on the 1,024-byte boundary 0x00000400. [1][2][3][4][5]

Second, the 128 MB distance between ITCM and Flash exceeds the direct Thumb-2 near-branch immediate range, compelling GNU ld to synthesize jump veneers (__*_veneer). Documented STM32H7 bringup failures under -O3 show that any assembly routine missing .type <sym>, %function causes GNU ld to emit corrupt veneer bytes (disassembling as illegal opcode 0x04 F0 1F E5 / invalid ARMv8-M bfcsel), raising UFSR.UNDEFINSTR (CFSR = 0x00010000) promoted to HardFault (HFSR = 0x40000000). Mitigate this structurally by annotating every .S routine with .type sym, %function and .thumb_func, co-locating libvertirta_flight_core.a alongside its libgcc/libm helpers and linker veneers inside .itcm_text so the 400 Hz control path performs zero cross-region jumps, compiling Flash-resident harness callers with -mlong-calls, and auditing all generated *_veneer stubs via arm-none-eabi-objdump -d in CI. [1][2][3][4][5]

The linker script stm32h753_tcm.ld below implements this zero-wait-state layout for both NUCLEO-H743ZI2 and Pixhawk 6X STM32H753 targets: [1][2][3][4][5]

In Reset_Handler—prior to __libc_init_array, prior to enabling L1 caches, and while ITCM remains writable—startup code copies .itcm_isr_vector, .itcm_text, .dtcm_rodata, and .data from their Flash LMAs into ITCM/DTCM VMAs, zeroes .bss, executes __DSB(); __ISB();, and writes SCB->VTOR = 0x00000400UL;. Because D2 peripheral DMA masters and the USB OTG internal DMA engine cannot directly access DTCM, every USB endpoint buffer and DMA descriptor must carry __attribute__((section(".dma_nocache"), aligned(32))). [1][2][3][4][5][6]

1.3 Cortex-M7 MPU, Eager FPU Stacking, and L1 Cache Initialization

Hardware bringup on Cortex-M7 must strictly obey a five-stage boot sequence so speculative linefills never fault unpopulated address space (Erratum 2.1.2 / ERR011573) and stale cache tags never corrupt SRAM: [1][2][3][4][5][6]

MPU Region

Base Address

Size (ARM_MPU_REGION_SIZE_*)

Attributes (TEX, S, C, B)

AP / XN / SRD

Architectural Purpose

R0

0x00000000

4 GB (0x1F)

TEX=0, S=0, C=0, B=0 (Strongly-Ordered)

AP=0 (None), XN=1, SRD=0x87

AN4838 background speculative-access firewall blocking unpopulated external memory 0x60000000–0xDFFFFFFF

R1

0x08000000

2 MB (0x14)

TEX=1, S=0, C=1, B=1 (Normal, WB-WA)

AP=6 (RO), XN=0, SRD=0x00

Internal Flash; RO protects code, WB avoids Erratum 2.1.1 WT corruption

R2

0x00000000

64 KB (0x0F)

TEX=1, S=0, C=0, B=0 (Normal, NC)

AP=6 (RO), XN=0, SRD=0x00

ITCM locked W^X read-only after boot copy; TCM inherently bypasses L1 cache

R3

0x00000000

1 KB (0x09)

TEX=0, S=0, C=0, B=0 (Strongly-Ordered)

AP=0 (None), XN=1, SRD=0x00

Overrides R2 bottom 1 KB (R3 > R2); hardware NULL-pointer MemManage trap

R4

0x20000000

128 KB (0x10)

TEX=1, S=0, C=0, B=0 (Normal, NC)

AP=3 (Full), XN=1, SRD=0x00

DTCM stack and hot data; XN=1 enforces strict W^X

R5

0x20000[span_92](start_span)[span_92](end_span)000

256 B (0x07)

TEX=0, S=0, C=0, B=0 (Strongly-Ordered)

AP=0 (None), XN=1, SRD=0x00

Overrides R4 bottom 256 B (R5 > R4); deterministic stack-overflow MemManage trap

R6

0x24000000

512 KB (0x12)

TEX=1, S=0, C=1, B=1 (Normal, WB-WA)

AP=3 (Full), XN=1, SRD=0x00

D1 AXI SRAM; S=0 mandatory because S=1 disables Cortex-M7 D-Cache

R7

0x30000000

32 KB (0x0E)

TEX=1, S=0, C=0, B=0 (Normal, NC)

AP=3 (Full), XN=1, SRD=0x00

D2 SRAM1 .dma_nocache for TinyUSB DWC2 and DMA buffers (eliminates manual cache maintenance)

Cortex-M7 defaults FPU->FPCCR to lazy stacking (ASPEN=1, LSPEN=1), which keeps non-FPU ISR entry at 12 cycles instead of 29 cycles but creates a non-preemptible deferred-save stall of up to 17 cycles whenever a nested or preempting handler executes its first floating-point instruction. Because libvertirta_flight_core.a heavily exercises FPv5-DP-D16 double-precision registers on every 400 Hz frame, disabling lazy stacking (ASPEN=1, LSPEN=0) trades a constant 17 cycles at exception entry for zero nested deferral jitter. Likewise, unlike Cortex-M4, Cortex-M7 resets with the CoreSight DWT software lock active: standalone firmware running without an attached SWD debugger must write 0xC5ACCE55 to DWT->LAR (0xE0001FB0) or DWT->CYCCNT remains stuck at zero. [1][2][3][4]

1.4 Minimal Apache-2.0/MIT Bare-Metal UART + USB-CDC 400 Hz HITL Profiling Harness

To capture authentic DWT->CYCCNT min/p99/max execution cycles without RTOS context-switching jitter or vendor HAL bloat, construct the HITL harness strictly from CMSIS-Core (Apache-2.0), TinyUSB (MIT, OPT_MCU_STM32H7, RHPort 0 mapped to USB2_OTG_FS), and direct register drivers for USART3 (PD8/PD9) and TIM2. Configure TinyUSB with CFG_TUSB_MCU=OPT_MCU_STM32H7, BOARD_TUD_RHPORT=0, CFG_TUSB_OS=OPT_OS_NONE, and CFG_TUD_CDC=1. Enable the dedicated USB 3.3 V power supply detector via PWR->CR3 |= PWR_CR3_USB33DEN and wait for PWR_CR3_USB33RDY (matching HAL_PWREx_EnableUSBVoltageDetector()), route internal HSI48 (48 MHz—avoiding the ~8.08 MHz STLINK-V3E MCO drift reported on NUCLEO-H743ZI2) or PLL3Q (48 MHz on Pixhawk 6X) to RCC->D2CCIP2R.USBSEL, configure PA11/PA12 to AF10_OTG2_HS, place TinyUSB buffers in .dma_nocache, and forward OTG_FS_IRQHandler() to tusb_int_handler(0, true). Drive the 400 Hz tick via 32-bit TIM2 (2 × PCLK1 = 240 MHz kernel clock when HCLK = 240 MHz and APB1 = /2) with PSC = 0 and ARR = 599999 (exact 2,500.0 µs period), or via SysTick (LOAD = 1,199,999 at 480 MHz) for minimal peripheral dependency. [1][2][3][4][5][6][7]

Because the 6-stage superscalar Cortex-M7 reorders and dual-issues instructions across unserialized reads, every measurement must bracket the flight-core call with __DSB(); __ISB(); before reading both t0 and t1, subtract an empirically calibrated empty-bracket baseline g_bracket_overhead (averaged over 1,024 iterations at startup), toggle a spare board GPIO via atomic BSRR for external oscilloscope cross-verification, and store raw 32-bit cycle deltas in a static .axi_sram array (e.g., 10,000 samples = 40 KB in 512 KB AXI SRAM for exact offline or on-target p50/p99/p99.9/max sorting with zero heap and zero quantization loss). Note that DWT->CYCCNT wraps every 8.95 s at 480 MHz (10.74 s at 400 MHz), so unsigned subtraction t1 - t0 remains exact for any control tick under 8.9 s. [1][2][3][4][5][6][7]

Build and link the harness with exact ABI parity against libvertirta_flight_core.a using arm-none-eabi-g++ 14.2.1 (14.2.Rel1), enforcing compile-time stack budgets (-fstack-usage -Wstack-usage=3500), zero-heap link assertions (-Wl,--wrap=malloc), and disabled FMA contraction (-ffp-contract=off to preserve IEEE-754 rounding parity with ESBMC --floatbv): [1][2][3][4][5]

1.5 Why Renode and QEMU Cannot Produce Certifiable Cortex-M7 WCET

Neither Renode (tlib dynamic binary translation) nor QEMU (TCG dynamic binary translation) models Cortex-M7 microarchitectural timing. Both engines translate guest basic blocks into host machine code and advance virtual time using a fixed scalar instruction rate (Renode's integer cpu PerformanceInMips, default 100 MIPS; QEMU's -icount fixed quantum per instruction). ArduPilot's official Renode documentation explicitly cautions that Renode is a functional developer tool whose peripheral models are not physically or timing accurate. [1][2][3][4]

Microarchitectural Mechanism

Physical STM32H753 Cortex-M7 Silicon

Renode / QEMU DBT Behavior

Why Emulated WCET Is Invalid

Superscalar Dual-Issue Pipeline

6-stage pipeline pairing up to 2 instructions/cycle; independent STM32H730 microbenchmarks show pairing order alone shifts throughput by ≥2×

Fixed scalar CPI derived from constant MIPS or -icount

Cannot predict dual-issue stalls, FPU pipeline hazards, or register scoreboarding

L1 I/D-Cache & Bus Contention

16 KB 2-way I-Cache + 16 KB 4-way D-Cache (32 B lines, pseudo-random eviction); AXI misses and DMA bursts incur large multi-cycle stalls

Flat sysbus memory arrays with uniform single-step cost

Completely hides cache-miss tails, eviction anomalies, and DMA/AXI interference

Branch Predictor & BTAC

Always-enabled branch predictor + single-cycle BTAC with multi-cycle misprediction flushes

No branch-predictor or BTAC state modeled in translated blocks

Flattens data-dependent branching variance inside QR-WLS and DAQP active-set loops

Flash Wait States & MMIO Bridges

4 wait states at 240 MHz AXI (VOS0) plus 480 MHz → 120 MHz APB synchronization latency

Synchronous C/C# MMIO callbacks execute in zero virtual time

Masks APB write-buffer stalls and stale peripheral status reads in tight polling loops

Interrupt Ingestion & FPU Stacking

12-cycle core stacking, tail-chaining, multi-cycle LDM interruption, and 17-cycle FPU lazy-stacking deferral

Pending IRQs deferred until translated block exits

Introduces artificial TB-boundary jitter while missing real mid-sequence interrupt preemption

Host Memory Ordering

Armv7-M weakly ordered memory model requiring DMB/DSB/ISB

x86_64 host enforces hardware Total Store Order (TSO)

Silently masks missing barrier bugs (E[span_163](start_span)[span_163](end_span)rratum 2.1.5) that fail on physical silicon

Framing Hardware vs. Emulated Evidence Honestly (DO-178C / CAST-32A / AMC 20-193): Emulated Renode runs belong strictly in Tier-1 Functional & Protocol SIL—supporting deterministic state-machine regression, OpenCyphal serialization/deserialization, and multi-node fault-handling logic, never execution-time budgets. Physical DWT->CYCCNT logs collected on NUCLEO-H743ZI2 and Pixhawk 6X establish Empirical High-Water-Mark (HWM) timing on real silicon under adversarial stimuli (simultaneous IMU dropout, motor FDI fault, and saturated coaxial X8 DAQP active-set corners). Because empirical testing alone cannot guarantee that the worst-case input path and worst-case hardware microstate coincided—and because analytical WCET is insoluble in the general case as noted in CAST-32A / AMC 20-193 literature citing Wilhelm et al.—full DO-178C Table A-6 / AMC 20-193 (MCP_Resource_Usage_4, MCP_Software_1) timing compliance combines deterministic TCM isolation (eliminating cache/Flash variance), parametric active-set iteration certification (e.g., via DAQP's companion ASCertain.jl), and either static WCET analysis or qualified hybrid measurement suites (such as Rapita RapiTime or LDRA TBrun) with explicit interference margins. [1][2][3][4][5]

2. Renode Dual-STM32H753 Lockstep & OpenCyphal CAN-FD Failover in GitHub Actions CI

2.1 Dual-Node Virtual Platform Architecture and STM32H7 Peripheral Overlays

Renode v1.17.0 (released September 2026) establishes a clean, Mono-free foundation for STM32H7 avionics CI: it completes the migration to native .NET (dotnet), distributes standalone portable tarballs for both x86_64 (linux-portable) and aarch64 (linux-arm64-portable), introduces CANTester and CANKeywords in the Robot Framework suite, upgrades STM32 FDCAN / MCAN peripheral support, and ships a Nucleo H753ZI platform alongside Antmicro's open-hardware STM32H753 Renode Reference Platform. [1][2][3][4][5]

Two documented Renode STM32H7 peripheral limitations must be addressed via a platform overlay (or an ad-hoc C# peripheral model) so unmodified bare-metal ELF images boot identically in Renode and on silicon. First, upstream stm32h7.repl lacks full emulation of the PWR controller (renode#649, renode#857), notably leaving PWR_D3CR (offset 0x18, VOSRDY) and supply-readiness bits unmodeled or returning zero—hanging bare-metal VOS0/LDO polling loops. Second, RCC kernel clock updates do not dynamically propagate to timer models (renode#1009), so deterministic 400 Hz scheduling in Renode should rely on SysTick (systickFrequency: 480_000_000) or explicitly configured timer frequencies. For simple readiness polling, sysbus Tag entries unblock startup; if firmware performs read-modify-write loops across multiple PWR registers (PWR_CR1, PWR_CR3, PWR_CSR1, PWR_D3CR), load a small custom C# STM32H7_PWR.cs via include $ORIGIN/STM32H7_PWR.cs before loading the .repl, which Renode compiles transparently on startup. [1][2][3][4]

Platform Overlay (renode/stm32h753_avionics.repl):

Dual-Node Emulation Script (renode/dual_fcc_failover.resc):

Tightening emulation SetGlobalQuantum "0.000025" (25 µs) is mandatory for sub-20 ms testing: Renode processes all inter-node CANHub frame deliveries strictly at synchronization phases between virtual-time quanta, so leaving a coarse default quantum injects multi-millisecond virtual synchronization lag that obscures failover behavior. When bridging to Linux vcan0 locally, never attach SocketCANBridge to both machines simultaneously, as doing so creates an infinite packet reflection loop (renode#598). [1][2][3]

2.2 OpenCyphal v1.0 DSDL, libcanard, and o1heap v3.0.0 Deterministic Sizing

OpenCyphal v1.0 mandates uavcan.node.Heartbeat.1.0 on fixed 13-bit Subject-ID 7509 (0x1D55, 7-byte serialized payload: uint32 uptime, Health.1.0 health, Mode.1.0 mode, uint8 vendor_specific_status_code, @extent 12 * 8) with MAX_PUBLICATION_PERIOD = 1 s and OFFLINE_TIMEOUT = 3 s. Because a 3-second timeout is 150× slower than the 20 ms failover limit of an unstable coaxial X8 multirotor, nodes must publish standard Heartbeat.1.0 at 1 Hz on Subject-ID 7509 for network-wide Cyphal compliance while running a dedicated 400 Hz (2.5 ms) single-frame CAN-FD supervision subject vertirta.flight.SafetyEnvelope.1.0 on Subject-ID 1024. [1][2][3]

Sizing the custom DSDL schema requires respecting Cyphal/CAN's 1-byte tail byte (SOT/EOT/Toggle/Transfer-ID): a single-frame CAN-FD transfer (CANARD_MTU_CAN_FD = 64 bytes) holds at most 63 payload bytes (504 bits). Packing a 6-axis float32[6] min + max capability wrench (48 B) alongside sequence, timestamp, health masks, and CRC-32C (16 B) yields 64 payload bytes—silently spilling into a 2-frame transfer with a 2-byte transfer CRC and doubled bus latency. Scoping the coaxial X8 attainable wrench to its 4 controlled axes ([Fz, Mx, My, Mz] as float32[4] min + float32[4] max = 32 B, or float16[6] min + max = 24 B if full 6-DOF is required) keeps the serialized payload at exactly 48 bytes (384 bits), leaving 15 bytes of headroom inside a single 64-byte CAN-FD frame (49 B on-wire including the Cyphal tail byte, padded to DLC 15 / 64 B, or exact 48 B DLC 14 if timestamp_us is stored as a 16-bit tick delta):

Generate header-only C11 serializers using Nunavut (nnvg); avoid Nunavut's C++ target, which upstream documentation explicitly marks as experimental and subject to redesign or removal: [1][2][3]

Choosing Between libcanard v3.2.x/v4.x and v5.0 (2026 Ecosystem): Standalone bare-metal Cyphal v1.0 flight controllers should pin MIT-licensed libcanard v3.2.x (or v4.x), which implements the ~1,000-LOC purely reactive C transport with O(log n) Cavl AVL trees, CANARD_MTU_CAN_FD = 64, flex-array removal for MISRA C alignment, and simple canardInit / canardTxInit / canardTxPush / canardRxSubscribe / canardRxAccept calls. The newest v5.0 rework targets OpenCyphal's Cy stack, adding Cyphal v1.1 16-bit subject-IDs, priority-preemptive multi-frame RX reassemblers, automatic hardware acceptance-filter callbacks, and split memory resource arenas (canard_mem_set_t: tx_transfer, tx_frame, rx_session, rx_payload, rx_filters) via canard_new / canard_subscribe_13b / canard_publish_13b / canard_poll; adopt v5.0 only if Cyphal v1.1 interoperability or concurrent multi-priority multi-frame RX preemption is required. [1][2][3][4][5][6][7]

o1heap v3.0.0 Deterministic Arena Sizing: MIT-licensed o1heap v3.0.0 (2026) halves per-fragment metadata overhead from 4× to 2× pointer width, reducing O1HEAP_ALIGNMENT = a on 32-bit Cortex-M7 from 16 bytes (v2.x) to 8 bytes, while adding o1heapReallocate and maintaining strict O(1) constant-time execution (the o1heap v3.0.0 README benchmark on an RP2350 Cortex-M33 reports ≤125 cycles for allocation and ≤115 cycles for deallocation, independent of heap fill or fragmentation; Cortex-M7 figures must be measured on-target via DWT->CYCCNT). Because o1heap is not thread-safe, serialize all libcanard + o1heap accesses inside one context or guard them with BASEPRI. [1][2][3][4][5][6][7]

Using the refined Half-Fit worst-case memory consumption (WCMC) formula from the o1heap v3.0.0 specification:

where a = 8 B, n_f = ceil(n / l), M_f = ceil(M / l), and k = M_f - n_f + 1, sizing for peak live libcanard memory M = 4,096 B (32 queued 64-byte CAN-FD frames plus active RX sessions), minimum allocation l = 32 B, and maximum single-frame allocation n = 128 B yields H_b = 20,661 B (20.18 KiB). Even allowing 256-byte multi-frame diagnostic allocations (n = 256 B, M = 4,096 B) bounds worst-case fragmented consumption at H_b = 30,095 B (29.39 KiB). Allocating a static 32,768 B (32 KiB) alignas(O1HEAP_ALIGNMENT) arena in DTCM mathematically prevents OOM under arbitrary fragmentation, verified continuously via o1heapDoInvariantsHold(heap) and o1heapGetDiagnostics(heap).oom_count == 0. [1][2][3][4][5][6]

2.3 Deterministic <20 ms Failover Budget, NVIC Fault Injection, and Robot Framework Suite

Meeting a hard < 20 ms failover ceiling begins with ISO 11898-1:2015 worst-case bit-stuffed wire time for 29-bit extended-ID CAN-FD frames (required by Cyphal/CAN). At 1 Mbps nominal arbitration (57 conservative nominal bits including SOF, 29-bit ID, control bits, 8 worst-case dynamic stuff bits, BRS, CRC delimiter, ACK, EOF, and IFS) and 5 Mbps data phase (ESI, DLC, dynamically stuffed payload, 4-bit Gray stuff count, CRC-17/21, and fixed stuff bits), a 48-byte payload takes 160.6 µs and a full 64-byte payload takes 192.6 µs. [1][2][3][4][5]

When the Primary FCC (Node 10) emits SafetyEnvelope at 400 Hz (2.5 ms period), the worst-case crash timing occurs immediately after Primary completes transmitting frame k. Configuring the Secondary Safety Monitor (Node 11) with a 4-missed-frame watchdog (W = 10.50 ms, including 0.50 ms jitter allowance), reserving 0.40 ms for worst-case bus blocking plus the Exceptional-priority handover frame (two 64-byte frames at 1/5 Mbps), an analytical 0.35 ms software allowance for FDCAN ISR + libcanard + o1heap + failover FSM (to be empirically bounded on physical STM32H753 silicon via DWT->CYCCNT), and 2.50 ms for the first active 400 Hz actuator frame yields a modeled worst-case handover of 13.75 ms (6.25 ms margin below 20 ms)—or 11.25 ms with a 3-missed-frame watchdog (W = 8.00 ms).

Fault-Injection Mechanics in Renode (NVIC.cs): Renode's Cortex-M NVIC implementation (NVIC.cs) models architectural synchronous fault escalation to HardFault and Lockup (SetPendingSynchronousFault), asynchronous system exception pending (SetPendingIRQ), and external IRQ line assertion (OnGPIO, adding the 16-exception vector offset internally), alongside direct register writes (sysbus.nvic WriteDoubleWord, exercised in Renode's own tests/unit-tests/mpu.robot). To test three orthogonal failure modes headlessly without relying on host wall-clock timing, the Robot Framework suite below uses defaultPauseEmulation=true and enforces a strict virtual-time timeout (timeout=0.020) on Wait For Line On Uart—meaning the test fails deterministically if more than 20.0 ms of simulated virtual time elapses between fault injection and Secondary takeover: [1][2][3][4]

Alongside TerminalTester assertions, Renode v1.17.0 ships native CANTester and CANKeywords in the Robot Framework suite so CI tests can assert raw 29-bit Cyphal extended CAN-FD frames directly on canHub—independently confirming Subject-ID 7509 and 1024 priority arbitration without trusting UART console prints alone. For offline post-mortem debugging of failed runs, enable emulation LogCANTraffic locally to inspect the internal CANHub stream in Wireshark. [1][2][3][4]

2.4 Headless GitHub Actions Matrix on ubuntu-24.04 and ubuntu-24.04-arm

Although Antmicro publishes antmicro/renode-test-action@v5, its action.yml clones and builds Renode from source on cache miss (installing dotnet-sdk-10.0, libgtk-3-dev, automake, gcc, etc.). For fast, reproducible headless CI across both ubuntu-24.04 (x86_64) and ubuntu-24.04-arm (aarch64 GitHub-hosted Arm64 runners; note ubuntu-latest transitions to Ubuntu 26.04 in November 2026), fetching the pinned self-contained Renode v1.17.0 .NET portable tarball eliminates Mono, GTK/X11, and source-build latency while matching the exact SHA-256 digests published in the v1.17.0 GitHub release. Because libvertirta_flight_core.a was compiled with arm-none-eabi-g++ 14.2.1 (Arm GNU Toolchain 14.2.Rel1), install the matching Arm 14.2.Rel1 toolchain tarball for each host architecture rather than Ubuntu 24.04's older apt package to prevent libgcc/libstdc++ ABI drift: [1][2][3][4]

3. Formal Verification with ESBMC (C++17 Clang Frontend) & Open-Source MISRA C++:2023 Static Profile

3.1 ESBMC v8.7 Verification Strategy, CLI Configurations, and Operational Traps

Between 2023 and November 2026, ESBMC matured from its initial Clang C++ frontend (v7.3, TACAS 2023) through integrated interval abstract interpretation (v7.4, TACAS 2024), modern C++ move/exception lowering (v7.6), incremental concurrency (v7.7, 2025), and in-memory VFS / LLVM 18–22 frontend upgrades (v8.5–v8.6, August–October 2026) to ESBMC v8.7 (November 20, 2026). Three v8.6–v8.7 capabilities directly fit libvertirta_flight_core.a: native C++ API Bitwuzla as the default SMT solver with native IEEE-754 Float32/Float64 theory and overhauled IR-IEEE NaN-preservation/rounding semantics; C++17 freestanding AST and operational-model coverage including structured bindings, if/switch init-statements, CTAD, std::span, std::clamp, std::optional, std::variant, and complete <type_traits>; and single-invocation modular contracts (__ESBMC_requires, __ESBMC_ensures, __ESBMC_assigns, --enforce-all-contracts), k-induction (--k-induction), interval pruning (--interval-analysis), IC3/PDR (--ts-pdr), struct scalarization (--scalarize-structs), and YAML correctness/violation witnesses. [1][2][3][4][5][6]

Four subtle CLI traps can silently weaken or break ESBMC verification runs and must be controlled explicitly in CI. First, ESBMC enables array bounds, pointer safety, division-by-zero, pointer alignment, and UB-shift checks by default (no --bounds-check flag is required; only --no-bounds-check disables it), whereas signed integer overflow (--overflow-check), unsigned overflow (--unsigned-overflow-check), floating-point NaN (--nan-check), memory leaks (--memory-leak-check), and over-sized struct field reads (--struct-fields-check) are opt-in and OFF by default. Omitting --overflow-check emits zero signed-overflow verification conditions regardless of unwind depth, and omitting --floatbv reverts to fixed-point encoding that misses IEEE-754 NaN comparisons and rounding bugs. Second, never pass --no-unwinding-assertions on proof runs: unwinding assertions must remain active so an insufficient --unwind K fails loudly instead of returning a false VERIFICATION SUCCESSFUL. Third, to match Cortex-M7 ILP32 widths (sizeof(int)=4, sizeof(long)=4, sizeof(void*)=4), pass --32—and run all ESBMC CI jobs on ubuntu-24.04 (x86_64), because the Linux ARM64 ESBMC build (esbmc-linux-armv8.zip) explicitly fails on --32 due to missing 32-bit libc headers. Fourth, ESBMC invokes Clang with -nostdinc++ against its bundled operational models (src/cpp/library/), so freestanding harnesses should restrict standard includes to modeled headers (<cstdint>, <cstddef>, <cmath>, <array>, <type_traits>, <algorithm>). [1][2][3][4][5][6]

3.2 Concrete Proof Harnesses and Unwinding Bounds for libvertirta_flight_core.a

Harness 1: 400 Hz 3-IMU TMR Voter (proofs/verify_tmr_voter.cpp)

This harness verifies both nominal precision and Byzantine single-IMU fault isolation across all three axes when Channel 2 carries an unconstrained symbolic IEEE-754 double (including NaN, +Inf, -Inf, subnormals, and ±DBL_MAX). Because the voter iterates over exactly 3 IMUs × 3 axes, --unwind 5 strictly exceeds the loop bound (3 + 1) with unwinding assertions enabled.

Harness 2: Double-Precision QR-WLS INDI Coaxial X8 Control Allocation (proofs/verify_qrwls_indi.cpp)

Feeding 96 unconstrained symbolic IEEE-754 double entries (12 × 8 weighted control effectiveness matrix) directly into a full Householder QR factorization (~600+ double-precision multiplications plus 8 square roots) causes solver timeout in bit-precise SMT engines. Make the proof tractable without sacrificing rigor via a two-layer decomposition: in Layer 2A, fix the nominal coaxial X8 rotor effectiveness matrix B0, symbolize all 256 possible 8-motor FDI failure bitmasks (uint8_t motor_mask, including single-, dual-, and total-rotor failure) alongside bounded symbolic 4-axis wrench demands nu, and unwind all matrix loops (--unwind 14 > 12 + 1) with --interval-analysis and --scalarize-structs. In Layer 2B, apply --enforce-all-contracts on the scalar/vector Householder reflector and triangular back-substitution kernels with fully symbolic double inputs to verify that rank-deficient column norms (||x||_2 < 1e-12) cleanly take the regularized fallback path without division by zero or NaN propagation. [1][2][3]

Harness 3: Static C99 DAQP Active-Set QP Buffer Safety (proofs/verify_daqp_bounds.c)

DAQP solves the strictly convex quadratic program:

using recursive rank-one [STRIPPED_MATH_OR_ICON] updates on a fixed static C99 workspace (packed lower-triangular L of length n(n+1)/2, diagonal D[n], working set WS[n], dual vector lam[n], constraint status sense[m]). The two primary safety hazards are out-of-bounds array indexing during active-set add/drop swaps (n_active > n or negative index) and near-zero pivot division when linearly dependent constraints enter the working set. Because active-set iteration count depends on runtime data, cap iter_limit = 25 in flight firmware (certifying true worst-case iteration counts parametrically offline with ASCertain.jl) and execute two ESBMC passes: Pass 3A proves workspace index, bounds, pointer, and overflow safety across the full N=8, M=16, iter_limit=25 problem (--unwind 27 > max(25, 16, 8) + 1), while Pass 3B exercises fully symbolic H and A matrices on a reduced N=3, M=6 instance to stress pivot degeneracy and NaN immunity. [1][2][3][4][5][6][7][8][9][10]

3.3 Comparative Analysis: ESBMC vs CBMC vs Frama-C for Freestanding C++17

No single open-source verifier covers a mixed C++17 / C99 / C11 avionics stack equally well. As of late 2026, pairing ESBMC v8.7 on the C++17 control library with Frama-C 33 (or CBMC 6.x) on the pure C99/C11 dependencies closes each engine's blind spots. [1][2][3]

Technical Criterion

ESBMC v8.7 (Nov 2026)

CBMC 6.x (2024–2026)

Frama-C 33.0 / 34.0-beta + Frama-Clang

C++17 Frontend Maturity

Strongest OSS C++17 path: Clang/LLVM 18–22 AST handles templates, structured bindings, if-init, CTAD, <type_traits>, std::span

Limited for modern C++: In-tree custom C/C++ parsers lag Clang on C++17/20 templates and structured bindings

Unsuitable for C++17 core: Core CIL AST is C-only; Frama-Clang v0.0.20 / v0.0.21~beta (Nov 2026) remains explicitly experimental with incomplete C++17 support

Solver / Proof Engine

Multi-solver SMT (Bitwuzla, Z3 4.13+, CVC5, MathSAT, Boolector, Yices) + interval domain + single-call BMC / k-induction / --ts-pdr

SAT-first bit-blasting (optional SMT export); bounded only—no built-in k-induction in a single engine

Sound abstract interpretation (Eva) + deductive weakest-precondition (WP via ACSL + external provers); naturally handles unbounded C loops

Modular Contracts

Single-invocation __ESBMC_requires / __ESBMC_ensures / __ESBMC_assigns

Mature Dynamic Frame Condition Checking (DFCC), requires 3-step goto-cc -> goto-instrument -> cbmc

Mature ACSL contracts; direct Airbus A380 DO-333 heritage via Caveat predecessor

Recommended Role in This Stack

Primary verifier for C++17 libvertirta_flight_core.a (TMR voter, motor FDI, QR-WLS INDI)

Independent second-opinion BMC on pure C99 DAQP and C11 libcanard / o1heap

Sound unbounded RTE verification (Eva + WP) strictly on pure C99 DAQP and C11 o1heap / libcanard

3.4 Honest Mapping to DO-333 / ED-216 Credits and DO-330 TQL-5 Qualification Gaps

DO-333 / ED-216 defines a formal method as a formal model paired with a formal analysis—spanning abstract interpretation, model checking, and theorem proving—and mandates evidence of soundness showing that the analysis never provides unwarranted confidence. Under DO-330 / ED-215, tool qualification is required whenever a tool eliminates, reduces, or automates a DO-178C/DO-333 process without independent verification of its output. Verification tools unable to inject errors into executable object code (EOC) qualify at TQL-5, except when formal verification replaces or reduces another verification activity (such as substituting proofs for low-level robustness testing), which typically elevates qualification to TQL-4. [1][2]

DO-333 / ED-216 Annex A Objective

Technical Support From ESBMC v8.7

Honest Certification Gap / Blocker

Table FM.A-5 [FM1, FM3, FM6] Source complies with LLR; verifiable; accurate & consistent (no RTEs)

Verifies absence of bounds violations, null dereferences, signed overflow, div-by-zero, and NaN across symbolic inputs within checked loop bounds

Requires DO-330 TQL-5 (automating source review) or TQL-4 (reducing testing); every __ESBMC_assume must undergo independent review (FM.A-5[FM12]) to prevent vacuous proofs

Table FM.A-5 [FM10, FM11, FM13] Formal cases/results correct; soundness evidence

Reproducible CI harnesses, pinned solver builds, YAML witnesses, dual-solver cross-checks (--bitwuzla + --z3)

Soundness gap: SMT solvers exhibit occasional FP/theory bugs (e.g., ESBMC #7898); NASA CR-2017-219371 documents solver-specific behavioral variance and proof-rule trust gaps requiring per-solver qualification

Table FM.A-6 [3, 4] EOC complies with & is robust with LLR

Formal RTE proofs cover extreme numerical corners (NaN, ±Inf, saturation) beyond hand-written robustness vectors

Cannot replace target HIL integration testing: hardware interrupts, MPU/cache faults, CAN-FD arbitration, and DWT->CYCCNT timing require STM32H753 silicon

Table FM.A-7 [5–8] Structural coverage & dead-code verification

ESBMC --branch-coverage, --assertion-coverage, and automated test generation assist coverage discovery

Does not satisfy DAL A MC/DC object-code traceability on its own; requires qualified coverage tooling or formal requirements-completeness proof per DO-333

Table FM.A-7 [9] Property preservation between source code and EOC

None directly — ESBMC parses C++17 via Clang/LLVM 18–22, whereas target silicon executes GCC 14.2.1 -Os object code

Compiler-mismatch gap: GCC -Os FMA contraction, UB optimizations, or ABI lowering can invalidate source proofs unless constrained (-ffp-contract=off -fno-strict-aliasing) and backed by compiler/object-code verification

Defensible PSAC Framing: Do not claim DO-333 test-elimination credit for an unqualified ESBMC setup. No off-the-shelf DO-330 TQSP (Tool Qualification Plan, Tool Operational Requirements, RTVM, TVPR, TQAS) exists for ESBMC, and the certification applicant of record bears full DO-330 accountability for writing and executing operational verification cases tied to the exact OS, ESBMC commit, CLI flags, and SMT solver configuration. Declare ESBMC in the Plan for Software Aspects of Certification (PSAC) as Supplemental Engineering Verification (incurring zero DO-330 qualification cost while all mandatory DO-178C Table A-6/A-7 target tests and structural coverage remain fully executed), retaining the harnesses and YAML witnesses as a high-leverage asset should a DAL A/B program later fund either applicant-owned TQL-5 qualification or migration of safety kernels to qualified commercial provers (such as Polyspace Code Prover or SPARK/GNATprove). [1]

3.5 Open-Source MISRA C++:2023 Static Profile and Unverifiable Rules

MISRA C++:2023 (published October 2023 for ISO/IEC 14882:2017 C++17, superseding MISRA C++:2008 and AUTOSAR C++14) defines 179 guidelines: 4 Directives and 175 Rules, categorized as Mandatory (no deviations permitted), Required (formal deviation required), or Advisory, and classified as Decidable or Undecidable. [1]

Neither upstream clang-tidy (LLVM 22) nor open-source cppcheck (v2.22.0, September 2026) provides official MISRA C++:2023 compliance. LLVM Discourse threads through 2025–2026 confirm that upstream clang-tidy lacks a MISRA module due to MISRA copyright/licensing constraints incompatible with LLVM's permissive license. Note two major LLVM 22.1.0 breaking changes that directly affect CI configs: clang-analyzer-* checks were removed from default checks (requiring explicit opt-in), non-system header diagnostics are now enabled by default, and LLVM maintainers opened a 2026 RFC to deprecate/remove the 5-check hicpp-* module after Perforce placed HIC++ behind a registration wall. Similarly, open-source Cppcheck provides zero MISRA C++:2023 checks (offering only partial MISRA C:2012 via misra.py plus its explicitly unsound flow-sensitive bug engine); full MISRA C++:2023 support exists strictly in commercial Cppcheck Premium (since v24.5.0, current v26.9.0, with pricing quoted via sales and practitioner reports citing roughly €400/seat/year). [1]

Production .clang-tidy Profile (LLVM 22) Approximating MISRA C++:2023 for Freestanding C++17:

Mandatory Cross-Target CI Invocation (GCC 14.2.1 + clang-tidy 22 + Cppcheck 2.22.0): Note that since Cppcheck 2.19 removed unix32-unsigned and defaults unix32 to signed char, ARM EABI builds (where char is unsigned by default) must explicitly pass --platform=unix32 --funsigned-char: [1][2][3]

For the pure C99/C11 components (daqp, libcanard, o1heap), optionally add cppcheck --addon=misra against open-source Cppcheck's partial MISRA C:2012 checker with a user-supplied rule-text file. [1][2]

Explicit Gap Analysis — MISRA C++:2023 Rules Open-Source Linters CANNOT Verify: Claiming full MISRA C++:2023 compliance from clang-tidy + cppcheck OSS alone fails any safety audit. Five guideline classes lie structurally beyond open-source static linters and require explicit compensating controls: [1][2]

Unverifiable MISRA C++:2023 Class

Exact Guideline IDs & Requirements

Why clang-tidy + Cppcheck OSS Fail

Required Compensating Control

All 4 Directives

Dir-0.3.1 (appropriate FP arithmetic), Dir-0.3.2 (preconditions respected), Dir-5.7.2 (no commented-out code), Dir-15.8.1 (self-assignment handling)

Depend on external design specs, numerical conditioning, or intent not checkable from syntax alone

ESBMC --floatbv / --nan-check + numerical conditioning analysis + peer review checklist

Undecidable UB, Lifetime, Aliasing & Uninitialized Reads

Rule-4.1.3 (no UB), Rule-4.6.1 (evaluation sequencing), Rule-6.8.1 (lifetime), Rule-8.7.1/8.7.2/8.9.1 (pointer arithmetic/relational ops within same array), Rule-8.18.1 (Mandatory: no overlapping copy), Rule-11.6.2 (Mandatory: no uninitialized read), Rule-8.2.10 (no direct/indirect recursion), Rule-15.1.1 (dynamic type in ctor/dtor), Rule-28.3.1 (side-effect-free predicates)

Fundamentally undecidable statically; AST matchers and Cppcheck's unsound flow engine yield both false negatives and false positives across pointers and branches

ESBMC v8.7 closes 4.1.3, 6.8.1, 8.7.1, 8.7.2, 11.6.2 on verified kernels; GCC -fcallgraph-info=su,da + DAG cycle script closes 8.2.10 (0 recursion)

System-Wide Cross-TU & ODR Rules

Rule-6.2.1 (One-Definition Rule), Rule-6.2.2 (identical cross-TU declaration types), Rule-6.2.3 (single entity implementation), Rule-6.5.1/6.5.2 (linkage scope across TUs)

clang-tidy inspects one translation unit at a time; Cppcheck CTU depth caps at 10 with incomplete C++ template/linkage resolution

Single-header declaration discipline, -flto -Wodr link-time ODR diagnostics, and ELF symbol audit scripts

Exact MISRA Type-Category & Promotion Rules

Rule-7.0.5 (integral promotion/usual arithmetic conversions must not change signedness or type category, e.g. uint8_t == uint16_t promoting to int), Rule-7.0.6 (appropriate numeric assignment excluding compound assignments), Rule-6.9.2 (explicit-width integer types), Rule-8.0.1 (MISRA precedence parentheses), Rule-5.13.4 (unsigned literal U suffixes)

bugprone-narrowing-conversions and GCC -Wconversion do not implement MISRA's 4-category model (Character/Integer/Floating/Other) or exact rule exceptions

MISRA Compliance:2020 Governance Artifacts

Guidelines Recategorization Plan (GRP), Guidelines Compliance Summary (GCS), traceable deviation records, and verbatim copyrighted headline mappings

Open-source linters cannot legally bundle copyrighted MISRA headline text or emit formal compliance matrices

Maintain manual deviation register now; adopt a qualified commercial tool prior to formal certification freeze

Operational Readiness Sequence

Execute bringup and CI integration in three gated phases to guarantee zero silent assumptions:

1. Physical Silicon Bringup (NUCLEO-H743ZI2 -> Pixhawk 6X): Link libvertirta_flight_core.a with stm32h753_tcm.ld, verify DBGMCU_IDC == 0x2003 (Rev V) and FLASH_ACR.LATENCY == 4 at 480 MHz VOS0, activate the 8-region W^X MPU + eager FPU (FPCCR.LSPEN=0) + L1 caches + DWT->LAR = 0xC5ACCE55, inspect objdump -d -j .itcm_text for clean veneers, and log 10,000+ worst-case 400 Hz HITL ticks (cross-checked on an oscilloscope via atomic GPIO toggle) to record genuine hardware min/p99/max execution cycles.

2. Headless Dual-Node CI Gate (ubuntu-24.04 + ubuntu-24.04-arm): Pin Renode v1.17.0 portable .NET tarballs, Arm GNU Toolchain 14.2.Rel1, Nunavut C11 DSDL codegen (enforcing @assert _offset_ <= {504} for single-frame CAN-FD), libcanard + 32 KiB o1heap v3.0.0 arena, and require green Robot Framework < 20 ms virtual-time failover runs under NVIC fault, bus-disconnect, and corrupted-CRC babbling-idiot injection on every pull request.

3. Formal & Static CI Gate (ubuntu-24.04 x86_64): Run clang-tidy 22 (--target=arm-none-eabi), cppcheck 2.22.0 (--platform=unix32 --funsigned-char), strict GCC 14.2.1 -Werror + call-graph DAG checks, ESBMC v8.7 (--32 --floatbv --nan-check --overflow-check --memory-leak-check --struct-fields-check --bitwuzla with unwinding assertions enabled) across the three flight-core proof harnesses, and optionally Frama-C 33 Eva on static C99 DAQP and C11 o1heap/libcanard—documenting all evidence honestly inside DO-178C / DO-333 / DO-330 boundaries.

1. https://www.st.com/resource/en/datasheet/stm32h753vi.pdf (Datasheet - STM32H753xI - STMicroelectronics)

2. https://community.st.com/stm32-mcus-products-25/what-is-the-speed-of-sram-in-stm32h7-150940 (What is the speed of SRAM in STM32H7? - ST Community)

3. https://embeddedinterview.com/articles/interrupt-latency/ (Interrupt Latency on Cortex-M, Explained | EmbeddedInterview)

4. https://renode.readthedocs.io/en/latest/advanced/time_framework.html (Time framework - Renode - documentation - Read the Docs)

5. https://www.tasking.com/files/content/technical-papers/Following-the-recommendations-of-CAST-32A-AMC-20-193-white-paper-v2.1.pdf (Following the recommendations of CAST-32A & A(M)C 20-193)

6. https://ldra.com/wp-content/uploads/2023/07/LDRA-SYSGO-Leads-gen-paper_v4_20221028-1.pdf (A Practical Approach to CAST-32A & AMC 20-193 Compliance - LDRA)

7. https://scitools.com/do-330-qualification-kits (DO-330 Tool Qualification Kits - SciTools)

8. https://github.com/esbmc/esbmc/releases (Releases · esbmc/esbmc - GitHub)

9. https://learn.adacore.com/booklets/adacore-technologies-for-airborne-software/chapters/analysis.html (4. Compliance with DO-178C/ED-12C Guidance: Analysis)

10. https://discourse.llvm.org/t/will-clang-frontend-plan-accept-misra-check-tools/84754 (Will clang frontend plan/accept misra check tools?)

11. https://codeant.ai/blogs/misra-c-plus-plus-2023 (MISRA C++ 2023 Guidelines, Rules, PDF Access, and Tools (2026))

12. https://help.perforce.com/qac/enforcement/doc/MISRA_M2CPP.html (MISRA C++:2023)

13. https://cppcheck.sourceforge.io/ (Cppcheck - A tool for static C/C++ code analysis)
