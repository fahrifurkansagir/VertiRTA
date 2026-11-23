// SPDX-License-Identifier: Apache-2.0
#include "vertirta_bsp/stm32h7_boot.hpp"

#include "vertirta_bsp/stm32h7_regs.hpp"

namespace vertirta::bsp {

namespace {

constexpr uint32_t make_mpu_rasr(
    uint32_t xn,
    uint32_t ap,
    uint32_t tex,
    uint32_t s,
    uint32_t c,
    uint32_t b,
    uint32_t srd,
    uint32_t size_enc) noexcept {
  return ((xn & 1u) << 28u) |
         ((ap & 7u) << 24u) |
         ((tex & 7u) << 19u) |
         ((s & 1u) << 18u) |
         ((c & 1u) << 17u) |
         ((b & 1u) << 16u) |
         ((srd & 0xFFu) << 8u) |
         ((size_enc & 0x1Fu) << 1u) |
         1u;  // ENABLE
}

void program_mpu_region(uint32_t rnr, uint32_t base_addr, uint32_t rasr) noexcept {
  regs::reg32(regs::kMpuRnr) = rnr;
  regs::reg32(regs::kMpuRbar) = base_addr;
  regs::reg32(regs::kMpuRasr) = rasr;
}

void invalidate_icache() noexcept {
  regs::dsb();
  regs::isb();
  regs::reg32(regs::kScbIciallu) = 0u;
  regs::dsb();
  regs::isb();
}

void invalidate_dcache() noexcept {
  regs::reg32(regs::kScbCsselr) = 0u;  // Select Level 1 Data Cache
  regs::dsb();
  const uint32_t ccsidr = regs::reg32(regs::kScbCcsidr);
  const uint32_t sets = ((ccsidr >> 13u) & 0x7FFFu);
  const uint32_t ways = ((ccsidr >> 3u) & 0x3FFu);
  for (uint32_t set = 0u; set <= sets; ++set) {
    for (uint32_t way = 0u; way <= ways; ++way) {
      const uint32_t dcisw = ((way & 0x3u) << 30u) | ((set & 0x1FFu) << 5u);
      regs::reg32(regs::kScbDcsw) = dcisw;
    }
  }
  regs::dsb();
  regs::isb();
}

}  // namespace

SiliconRevInfo inspect_silicon_rev() noexcept {
  const uint32_t idc = regs::reg32(regs::kDbgMcuIdcAddr);
  SiliconRevInfo info{};
  info.dev_id = static_cast<uint16_t>(idc & 0x0FFFu);
  info.rev_id = static_cast<uint16_t>((idc >> 16u) & 0xFFFFu);
#ifdef VERTIRTA_RENODE_SIL
  // Virtual Renode SIL target may leave DBGMCU_IDCODE uninitialized (0x0000); allow Rev V mode ONLY in SIL builds.
  const bool rev_v_ok = (info.rev_id == regs::kStm32h7RevV || info.rev_id == 0x0000u);
#else
  // Physical STM32H753 silicon lock: strictly require verified Rev V (0x2003); block 0x0000/0x1003 (Erratum 2.24.2).
  const bool rev_v_ok = (info.rev_id == regs::kStm32h7RevV);
#endif
  if (rev_v_ok) {
    info.is_rev_v = true;
    info.rev_y_fdcan_errata_flag = false;
    info.sysclk_hz = 480'000'000u;
    info.axi_hz = 240'000'000u;
    info.flash_latency = regs::kFlashLatencyVos0;
  } else {
    // Rev Y (0x1003) or unknown: refuse 480 MHz VOS0 & block safety-critical FDCAN arming.
    info.is_rev_v = false;
    info.rev_y_fdcan_errata_flag = true;
    info.sysclk_hz = 400'000'000u;
    info.axi_hz = 200'000'000u;
    info.flash_latency = regs::kFlashLatencyVos1;
  }
  return info;
}

void init_clocks_480mhz_or_fallback(SiliconRevInfo& rev) noexcept {
  // 1. Enable SYSCFG clock (APB4 bit 1) and configure LDO supply.
  regs::reg32(regs::kRccApb4enr) |= (1u << 1u);
  regs::dsb();
  regs::reg32(regs::kPwrCr3) = (regs::reg32(regs::kPwrCr3) & ~(1u << 2u)) | (1u << 1u);  // LDOEN

  // Select VOS1 first (D3CR.VOS = 0b11 << 14) and wait for VOSRDY (bit 13).
  regs::reg32(regs::kPwrD3cr) = (regs::reg32(regs::kPwrD3cr) & ~(3u << 14u)) | (3u << 14u);
  for (uint32_t spin = 0u; spin < 100000u; ++spin) {
    if ((regs::reg32(regs::kPwrD3cr) & (1u << 13u)) != 0u) {
      break;
    }
  }

  if (rev.is_rev_v && rev.rev_id != regs::kStm32h7RevY) {
    // Rev V (0x2003): unlock VOS0 overdrive (SYSCFG_PWRCR.ODEN = bit 0) -> 480 MHz CPU / 240 MHz AXI.
    regs::reg32(regs::kSyscfgPwrcr) |= 1u;
    for (uint32_t spin = 0u; spin < 100000u; ++spin) {
      if ((regs::reg32(regs::kPwrD3cr) & (1u << 13u)) != 0u) {
        break;
      }
    }
    // RM0433 Table 17: 225-240 MHz AXI under VOS0 mandates LATENCY = 4, WRHIGHFREQ = 0b10.
    constexpr uint32_t kFlashAcrVos0 =
        (regs::kFlashLatencyVos0 & 0xFu) | ((regs::kFlashWrHighFreq10 & 0x3u) << 4u);
    regs::reg32(regs::kFlashAcr) = kFlashAcrVos0;
    const uint32_t acr_rb = regs::reg32(regs::kFlashAcr);  // Readback barrier before SYSCLK switch
    (void)acr_rb;
    regs::dsb();
    regs::isb();
    rev.sysclk_hz = 480'000'000u;
    rev.axi_hz = 240'000'000u;
    rev.flash_latency = regs::kFlashLatencyVos0;
  } else {
    // Rev Y (0x1003): bench diagnostic 400 MHz VOS1 fallback; latch rev_y_fdcan_errata_flag.
    rev.rev_y_fdcan_errata_flag = true;
    constexpr uint32_t kFlashAcrVos1 =
        (regs::kFlashLatencyVos1 & 0xFu) | ((regs::kFlashWrHighFreq10 & 0x3u) << 4u);
    regs::reg32(regs::kFlashAcr) = kFlashAcrVos1;
    const uint32_t acr_rb = regs::reg32(regs::kFlashAcr);
    (void)acr_rb;
    regs::dsb();
    regs::isb();
    rev.sysclk_hz = 400'000'000u;
    rev.axi_hz = 200'000'000u;
    rev.flash_latency = regs::kFlashLatencyVos1;
  }

  // Configure domain dividers: D1CPRE=/1 (480 MHz), HPRE=/2 (240 MHz AXI), APB1-4=/2 (120 MHz).
  regs::reg32(regs::kRccD1cfgr) = (0u << 8u) | (4u << 4u) | (8u << 0u);
  regs::reg32(regs::kRccD2cfgr) = (4u << 8u) | (4u << 4u);
  regs::reg32(regs::kRccD3cfgr) = (4u << 4u);

  // Configure PLL1 from HSI (64 MHz / M=4 = 16 MHz):
  //   Rev V: N=60 -> 960 MHz VCO -> DIVP1=/2 (480 MHz SYSCLK), DIVQ1=/12 (80 MHz FDCAN1 kernel clock)
  //   Rev Y: N=50 -> 800 MHz VCO -> DIVP1=/2 (400 MHz SYSCLK), DIVQ1=/10 (80 MHz FDCAN1 kernel clock)
  const uint32_t pll_n = rev.is_rev_v ? 59u : 49u;  // (N - 1)
  const uint32_t pll_q = rev.is_rev_v ? 11u : 9u;   // (Q - 1): /12 (Rev V) or /10 (Rev Y) -> 80 MHz PLL1Q
  regs::reg32(regs::kRccPllckselr) = (0u << 0u) | (4u << 4u);  // PLLSRC=HSI, DIVM1=4
  regs::reg32(regs::kRccPllcfgr) =
      (1u << 17u) | (1u << 16u) | (3u << 2u);  // DIVQ1EN | DIVP1EN | PLL1RGE=8-16 MHz
  regs::reg32(regs::kRccPll1divr) =
      (pll_q << 16u) | (1u << 9u) | (pll_n << 0u);  // DIVQ1=/12(RevV) /10(RevY), DIVP1=/2
  regs::reg32(regs::kRccCr) |= (1u << 24u);  // PLL1ON
  for (uint32_t spin = 0u; spin < 100000u; ++spin) {
    if ((regs::reg32(regs::kRccCr) & (1u << 25u)) != 0u) {
      break;
    }
  }
  regs::reg32(regs::kRccCfgr) = (regs::reg32(regs::kRccCfgr) & ~7u) | 3u;  // SW = PLL1P
  regs::dsb();
  regs::isb();
}

void init_mpu_8_regions() noexcept {
  // Disable MPU prior to reprogramming (DR-08 Sec 1.3; ZERO Write-Through regions per Erratum 2.1.1).
  regs::dsb();
  regs::reg32(regs::kMpuCtrl) = 0u;

  // R0: 0x00000000, 4 GB (0x1F), Strongly-Ordered, AP=0, XN=1, SRD=0x87
  //     AN4838 speculative-access firewall blocking unpopulated 0x60000000-0xDFFFFFFF.
  program_mpu_region(0u, 0x00000000UL, make_mpu_rasr(1u, 0u, 0u, 0u, 0u, 0u, 0x87u, 0x1Fu));

  // R1: 0x08000000, 2 MB (0x14), Normal WB-WA (TEX=1,S=0,C=1,B=1), AP=6 (RO), XN=0
  program_mpu_region(1u, 0x08000000UL, make_mpu_rasr(0u, 6u, 1u, 0u, 1u, 1u, 0x00u, 0x14u));

  // R2: 0x00000000, 64 KB (0x0F), Normal NC (TEX=1,S=0,C=0,B=0), AP=6 (RO W^X), XN=0
  program_mpu_region(2u, 0x00000000UL, make_mpu_rasr(0u, 6u, 1u, 0u, 0u, 0u, 0x00u, 0x0Fu));

  // R3: 0x00000000, 1 KB (0x09), Strongly-Ordered, AP=0 (None), XN=1 -> hardware NULL trap (R3 > R2).
  program_mpu_region(3u, 0x00000000UL, make_mpu_rasr(1u, 0u, 0u, 0u, 0u, 0u, 0x00u, 0x09u));

  // R4: 0x20000000, 128 KB (0x10), Normal NC (DTCM), AP=3 (Full), XN=1 (strict W^X).
  program_mpu_region(4u, 0x20000000UL, make_mpu_rasr(1u, 3u, 1u, 0u, 0u, 0u, 0x00u, 0x10u));

  // R5: 0x20000000, 256 B (0x07), Strongly-Ordered, AP=0 (None), XN=1 -> stack-overflow trap (R5 > R4).
  program_mpu_region(5u, 0x20000000UL, make_mpu_rasr(1u, 0u, 0u, 0u, 0u, 0u, 0x00u, 0x07u));

  // R6: 0x24000000, 512 KB (0x12), Normal WB-WA (S=0 mandatory for D-Cache), AP=3, XN=1 (D1 AXI SRAM).
  program_mpu_region(6u, 0x24000000UL, make_mpu_rasr(1u, 3u, 1u, 0u, 1u, 1u, 0x00u, 0x12u));

  // R7: 0x30000000, 32 KB (0x0E), Normal NC (TEX=1,S=0,C=0,B=0), AP=3, XN=1 (D2 SRAM1 .dma_nocache).
  program_mpu_region(7u, 0x30000000UL, make_mpu_rasr(1u, 3u, 1u, 0u, 0u, 0u, 0x00u, 0x0Eu));

  // Enable MemManage/BusFault/UsageFault handlers (SHCSR bits 16..18) and MPU (PRIVDEFENA | ENABLE).
  regs::reg32(regs::kScbShcsr) |= (7u << 16u);
  regs::reg32(regs::kMpuCtrl) = (1u << 2u) | 1u;
  regs::dsb();
  regs::isb();
}

void init_fpu_eager_and_caches() noexcept {
  // 1. Grant full CP10/CP11 access (bits 20..23).
  regs::reg32(regs::kScbCpacr) |= (0xFu << 20u);
  regs::dsb();
  regs::isb();

  // 2. Disable FPU lazy stacking (ASPEN=1, LSPEN=0) -> constant deterministic exception entry.
  uint32_t fpccr = regs::reg32(regs::kFpuFpccr);
  fpccr |= regs::kFpuFpccrAspen;
  fpccr &= ~regs::kFpuFpccrLspen;
  regs::reg32(regs::kFpuFpccr) = fpccr;
  regs::dsb();
  regs::isb();

  // 3. Invalidate and enable L1 I-Cache (CCR bit 17) & D-Cache (CCR bit 16) with barriers (Erratum 2.1.5).
  invalidate_icache();
  invalidate_dcache();
  regs::reg32(regs::kScbCcr) |= (1u << 17u) | (1u << 16u);
  regs::dsb();
  regs::isb();
}

void unlock_dwt_cyccnt() noexcept {
  // Enable CoreDebug TRCENA (bit 24), unlock Cortex-M7 DWT->LAR (0xC5ACCE55), reset & enable CYCCNT.
  regs::reg32(regs::kCoreDebugDemcr) |= (1u << 24u);
  regs::dsb();
  regs::reg32(regs::kDwtLar) = regs::kDwtLarUnlockKey;
  regs::dsb();
  regs::reg32(regs::kDwtCyccnt) = 0u;
  regs::reg32(regs::kDwtCtrl) |= 1u;  // CYCCNTENA
  regs::dsb();
  regs::isb();
}

void init_usb_otg_fs_clock_hsi48() noexcept {
  // Enable USB 3.3 V supply detector (PWR_CR3_USB33DEN bit 24) and route HSI48 (RCC_D2CCIP2R.USBSEL=0b11).
  regs::reg32(regs::kPwrCr3) |= (1u << 24u);
  for (uint32_t spin = 0u; spin < 100000u; ++spin) {
    if ((regs::reg32(regs::kPwrCr3) & (1u << 26u)) != 0u) {  // USB33RDY
      break;
    }
  }
  regs::reg32(regs::kRccCr) |= (1u << 12u);  // HSI48ON
  for (uint32_t spin = 0u; spin < 100000u; ++spin) {
    if ((regs::reg32(regs::kRccCr) & (1u << 13u)) != 0u) {  // HSI48RDY
      break;
    }
  }
  regs::reg32(regs::kRccD2ccip2r) = (regs::reg32(regs::kRccD2ccip2r) & ~(3u << 20u)) | (3u << 20u);
}

void init_400hz_systick(uint32_t sysclk_hz) noexcept {
  const uint32_t reload = (sysclk_hz / 400u) - 1u;  // 1,199,999 @ 480 MHz -> exact 2,500.0 us
  regs::reg32(regs::kSysTickCtrl) = 0u;
  regs::reg32(regs::kSysTickLoad) = reload;
  regs::reg32(regs::kSysTickVal) = 0u;
  regs::reg32(regs::kSysTickCtrl) = 0x7u;  // CLKSOURCE=CPU | TICKINT | ENABLE
}

void init_400hz_tim2() noexcept {
  // 240 MHz TIM2 kernel clock -> PSC=0, ARR=599999 -> exact 2,500.0 us (400 Hz).
  regs::reg32(regs::kRccApb1lenr) |= 1u;  // TIM2EN
  regs::dsb();
  regs::reg32(regs::kTim2Cr1) = 0u;
  regs::reg32(regs::kTim2Psc) = 0u;
  regs::reg32(regs::kTim2Arr) = 599999u;
  regs::reg32(regs::kTim2Egr) = 1u;
  regs::reg32(regs::kTim2Sr) = 0u;
  regs::reg32(regs::kTim2Cr1) = 1u;  // CEN
}

}  // namespace vertirta::bsp
