// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vertirta::bsp {

struct SiliconRevInfo {
  uint16_t dev_id{0u};
  uint16_t rev_id{0u};
  uint32_t sysclk_hz{480'000'000u};
  uint32_t axi_hz{240'000'000u};
  uint32_t flash_latency{4u};
  bool is_rev_v{false};
  // ES0392 Rev 16 Erratum 2.24.2 (Rev Y 0x1003 no-workaround FDCAN message RAM corruption):
  // when true, firmware refuses 480 MHz VOS0 AND blocks safety-critical FDCAN arming.
  bool rev_y_fdcan_errata_flag{false};
};

// Reads DBGMCU_IDC (0x5C001000) to classify STM32H753 / H743 silicon revision.
[[nodiscard]] SiliconRevInfo inspect_silicon_rev() noexcept;

// Configures PLL1/VOS0 (480 MHz, FLASH_ACR.LATENCY=4, WRHIGHFREQ=0b10) on Rev V (0x2003),
// or bench-diagnostic 400 MHz VOS1 fallback + rev_y_fdcan_errata_flag latch on Rev Y (0x1003).
void init_clocks_480mhz_or_fallback(SiliconRevInfo& rev) noexcept;

// Configures exact 8-region W^X MPU (R0..R7) with bottom 1 KB ITCM NULL trap (R3),
// bottom 256 B DTCM stack trap (R5), D2 SRAM1 .dma_nocache (R7), and ZERO Write-Through
// regions (ES0392 Erratum 2.1.1).
void init_mpu_8_regions() noexcept;

// Enables CP10/CP11, disables FPU lazy stacking (FPCCR: ASPEN=1, LSPEN=0), and enables
// L1 I/D caches with full DSB/ISB barriers (ES0392 Erratum 2.1.5).
void init_fpu_eager_and_caches() noexcept;

// Unlocks Cortex-M7 CoreSight DWT software lock (DWT->LAR = 0xC5ACCE55) and enables CYCCNT.
void unlock_dwt_cyccnt() noexcept;

// Optional USB 3.3 V supply detector + HSI48 clock routing for MIT TinyUSB CDC (DR-08 Sec 1.4).
void init_usb_otg_fs_clock_hsi48() noexcept;

// Configures 400 Hz periodic tick via TIM2 (240 MHz, PSC=0, ARR=599999) or SysTick.
void init_400hz_systick(uint32_t sysclk_hz) noexcept;
void init_400hz_tim2() noexcept;

}  // namespace vertirta::bsp
