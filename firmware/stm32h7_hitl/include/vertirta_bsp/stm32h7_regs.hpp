// SPDX-License-Identifier: Apache-2.0
// Minimal Apache-2.0 register map for STM32H753 / STM32H743 Cortex-M7 bare-metal HITL BSP.
// Zero vendor HAL dependency; addresses verified against ST RM0433 & ARMv7-M Architecture Reference Manual.
#pragma once

#include <cstdint>

namespace vertirta::bsp::regs {

inline constexpr uintptr_t kScbBase       = 0xE000ED00UL;
inline constexpr uintptr_t kMpuBase       = 0xE000ED90UL;
inline constexpr uintptr_t kFpuBase       = 0xE000EF34UL;
inline constexpr uintptr_t kCoreDebugBase = 0xE000EDF0UL;
inline constexpr uintptr_t kDwtBase       = 0xE0001000UL;
inline constexpr uintptr_t kSysTickBase   = 0xE000E010UL;

inline constexpr uintptr_t kDbgMcuIdcAddr = 0x5C001000UL;
inline constexpr uintptr_t kFlashBase     = 0x52002000UL;
inline constexpr uintptr_t kRccBase       = 0x58024400UL;
inline constexpr uintptr_t kPwrBase       = 0x58024800UL;
inline constexpr uintptr_t kSyscfgBase    = 0x58000400UL;
inline constexpr uintptr_t kGpioBBase     = 0x58020400UL;
inline constexpr uintptr_t kGpioDBase     = 0x58020C00UL;
inline constexpr uintptr_t kTim2Base      = 0x40000000UL;
inline constexpr uintptr_t kUsart3Base    = 0x40004800UL;
inline constexpr uintptr_t kFdcan1Base    = 0x4000A000UL;
inline constexpr uintptr_t kFdcanMsgRam   = 0x4000AC00UL;

// CoreSight / Cortex-M7 SCB offsets
inline constexpr uintptr_t kScbVtor     = kScbBase + 0x08UL;
inline constexpr uintptr_t kScbCcr      = kScbBase + 0x14UL;
inline constexpr uintptr_t kScbShcsr    = kScbBase + 0x24UL;
inline constexpr uintptr_t kScbCpacr    = kScbBase + 0x88UL;
inline constexpr uintptr_t kScbIciallu  = 0xE000EF50UL;
inline constexpr uintptr_t kScbDcsw     = 0xE000EF60UL;
inline constexpr uintptr_t kScbCsselr   = kScbBase + 0x84UL;
inline constexpr uintptr_t kScbCcsidr   = kScbBase + 0x80UL;

// Cortex-M7 MPU offsets
inline constexpr uintptr_t kMpuCtrl = kMpuBase + 0x04UL;
inline constexpr uintptr_t kMpuRnr  = kMpuBase + 0x08UL;
inline constexpr uintptr_t kMpuRbar = kMpuBase + 0x0CUL;
inline constexpr uintptr_t kMpuRasr = kMpuBase + 0x10UL;

// Cortex-M7 FPU / DWT / CoreDebug offsets
inline constexpr uintptr_t kFpuFpccr     = kFpuBase;  // 0xE000EF34
inline constexpr uintptr_t kCoreDebugDemcr = kCoreDebugBase + 0x0CUL;  // 0xE000EDFC
inline constexpr uintptr_t kDwtCtrl      = kDwtBase + 0x00UL;
inline constexpr uintptr_t kDwtCyccnt    = kDwtBase + 0x04UL;
inline constexpr uintptr_t kDwtLar       = kDwtBase + 0xFB0UL;  // 0xE0001FB0

// STM32H7 PWR / SYSCFG / FLASH / RCC offsets (RM0433)
inline constexpr uintptr_t kFlashAcr      = kFlashBase + 0x00UL;
inline constexpr uintptr_t kPwrCr3        = kPwrBase + 0x0CUL;
inline constexpr uintptr_t kPwrD3cr       = kPwrBase + 0x18UL;
inline constexpr uintptr_t kSyscfgPwrcr   = kSyscfgBase + 0x2CUL;
inline constexpr uintptr_t kRccCr         = kRccBase + 0x00UL;
inline constexpr uintptr_t kRccCfgr       = kRccBase + 0x10UL;
inline constexpr uintptr_t kRccD1cfgr     = kRccBase + 0x18UL;
inline constexpr uintptr_t kRccD2cfgr     = kRccBase + 0x1CUL;
inline constexpr uintptr_t kRccD3cfgr     = kRccBase + 0x20UL;
inline constexpr uintptr_t kRccPllckselr  = kRccBase + 0x28UL;
inline constexpr uintptr_t kRccPllcfgr    = kRccBase + 0x2CUL;
inline constexpr uintptr_t kRccPll1divr   = kRccBase + 0x30UL;
inline constexpr uintptr_t kRccD2ccip2r   = kRccBase + 0x54UL;
inline constexpr uintptr_t kRccAhb4enr    = kRccBase + 0xE0UL;
inline constexpr uintptr_t kRccApb1lenr   = kRccBase + 0xE8UL;
inline constexpr uintptr_t kRccApb1henr   = kRccBase + 0xECUL;
inline constexpr uintptr_t kRccApb4enr    = kRccBase + 0xF4UL;

// SysTick offsets
inline constexpr uintptr_t kSysTickCtrl = kSysTickBase + 0x00UL;
inline constexpr uintptr_t kSysTickLoad = kSysTickBase + 0x04UL;
inline constexpr uintptr_t kSysTickVal  = kSysTickBase + 0x08UL;

// TIM2 offsets
inline constexpr uintptr_t kTim2Cr1 = kTim2Base + 0x00UL;
inline constexpr uintptr_t kTim2Psc = kTim2Base + 0x28UL;
inline constexpr uintptr_t kTim2Arr = kTim2Base + 0x2CUL;
inline constexpr uintptr_t kTim2Egr = kTim2Base + 0x14UL;
inline constexpr uintptr_t kTim2Sr  = kTim2Base + 0x10UL;

// GPIO offsets
inline constexpr uintptr_t kGpioModerOff = 0x00UL;
inline constexpr uintptr_t kGpioBsrrOff  = 0x18UL;
inline constexpr uintptr_t kGpioAfrhOff  = 0x24UL;

// USART3 offsets (RM0433)
inline constexpr uintptr_t kUsartCr1 = kUsart3Base + 0x00UL;
inline constexpr uintptr_t kUsartBrr = kUsart3Base + 0x0CUL;
inline constexpr uintptr_t kUsartIsr = kUsart3Base + 0x1CUL;
inline constexpr uintptr_t kUsartTdr = kUsart3Base + 0x28UL;

// Key Silicon & Architectural Constants (ES0392 Rev 16)
inline constexpr uint16_t kStm32h7RevV          = 0x2003u;
inline constexpr uint16_t kStm32h7RevY          = 0x1003u;
inline constexpr uint32_t kDwtLarUnlockKey      = 0xC5ACCE55UL;
inline constexpr uint32_t kFlashLatencyVos0     = 4u;    // 225-240 MHz AXI @ VOS0 (RM0433 Table 17)
inline constexpr uint32_t kFlashLatencyVos1     = 2u;    // 200 MHz AXI @ VOS1 (Rev Y bench fallback)
inline constexpr uint32_t kFlashWrHighFreq10    = 2u;    // 0b10
inline constexpr uint32_t kItcmVtorAddr         = 0x00000400UL;
inline constexpr uint32_t kFpuFpccrAspen        = 1u << 31u;
inline constexpr uint32_t kFpuFpccrLspen        = 1u << 30u;
inline constexpr uint32_t kFpscrExceptionMask   = 0x1Fu;  // IOC | DZC | OFC | UFC | IXC [4:0]

inline volatile uint32_t& reg32(uintptr_t addr) noexcept {
  return *reinterpret_cast<volatile uint32_t*>(addr);
}

inline void dsb() noexcept {
#if defined(__arm__) || defined(__thumb__)
  __asm__ volatile("dsb sy" ::: "memory");
#endif
}

inline void isb() noexcept {
#if defined(__arm__) || defined(__thumb__)
  __asm__ volatile("isb sy" ::: "memory");
#endif
}

}  // namespace vertirta::bsp::regs
