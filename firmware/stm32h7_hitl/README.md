# VertiRTA STM32H753 / NUCLEO-H743ZI2 Bare-Metal 400 Hz `DWT->CYCCNT` HITL Harness

Standalone Apache-2.0 bare-metal Cortex-M7 profiling firmware that relocates the frozen **`28,886 B Flash / 0 B static RAM`** `libvertirta_flight_core.a` archive into zero-wait-state 64 KB ITCM (`44.1%` utilization) above a `1 KB` MPU NULL-pointer guard (`SCB->VTOR = 0x00000400`) and executes `10,000` worst-case 400 Hz coaxial X8 fault ticks (`TmrVoter` + `IndiWlsController` + `MotorFdiObserver` + `StaticDaqpSolver` + `CoaxialPlant6Dof`) with serialized `DWT->CYCCNT` timing and oscilloscope GPIO toggling (`PB0`).

> **Timing Honesty Lock (`DO-178C` / `CAST-32A` / `AMC 20-193`):**
> On-target `DWT->CYCCNT` logs (`EMPIRICAL_HWM_NOT_WCET_PROOF`) capture genuine **Empirical High-Water-Mark (HWM)** silicon cycle timings across adversarial stimuli. They are **NOT** a standalone mathematical WCET proof without path coverage and interference bounds.

## Hardware Bringup & Silicon Errata Notes (`ES0392 Rev 16`)

| Target Board | Clock / Debug Interface | Critical Safety Lock |
| --- | --- | --- |
| **NUCLEO-H743ZI2** (`MB1364`, ~$25) | `USART3` VCP (`PD8`/`PD9`, `115200 8N1`), `STLINK-V3E` SWD at `0x08000000`, scope trigger `PB0` | `STLINK-V3E` 8 MHz MCO exhibits ~1% HSI drift; USB uses internal `HSI48` (`PWR_CR3_USB33DEN`). |
| **Pixhawk 6X** (`FMUv6X`, `STM32H753` Rev V) | `USART3` FMU Debug (`JST SM10B`), `SWD` | **Archive the original PX4 bootloader at `0x08000000` before flashing via SWD**, or offset `.flash_isr_vector` above the bootloader. |

- **Rev V Gate (`DBGMCU->IDCODE == 0x2003`):** `480 MHz VOS0` (`240 MHz AXI`, `FLASH_ACR.LATENCY = 4`, `WRHIGHFREQ = 0b10`) activates strictly on Rev V. On Rev Y (`0x1003`), firmware drops to bench-diagnostic `400 MHz VOS1` (`LATENCY = 2`) and latches `rev_y_fdcan_errata_flag = true` to **block safety-critical FDCAN arming** (`Erratum 2.24.2`).
- **Core Errata Neutralized:** Zero Write-Through MPU regions (`Erratum 2.1.1`), `__DSB(); __ISB();` barriers after cache maintenance (`Erratum 2.1.5`), eager FPU stacking (`FPU->FPCCR.LSPEN = 0`), `DWT->LAR = 0xC5ACCE55`, per-tick `FPSCR[4:0]` polling/clearing (`Erratum 2.1.6`), and non-DMA `USART3` polling (`Erratum 2.20.4`).

## Cross-Compiling Locally (WSL2 / Linux `arm-none-eabi-g++ 14.2.1`)

```bash
cmake --preset arm-cortex-m7 && cmake --build --preset arm-cortex-m7
cmake -S firmware/stm32h7_hitl -B /tmp/vertirta_v3a_hitl -G Ninja \
  -DVERTIRTA_FLIGHT_CORE_LIB=$PWD/build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a
cmake --build /tmp/vertirta_v3a_hitl
arm-none-eabi-readelf -S /tmp/vertirta_v3a_hitl/vertirta_stm32h7_hitl.elf
arm-none-eabi-objdump -d -j .itcm_text /tmp/vertirta_v3a_hitl/vertirta_stm32h7_hitl.elf | grep veneer || true
```
