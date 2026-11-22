// SPDX-License-Identifier: Apache-2.0
#include "vertirta_bsp/uart_console.hpp"

#include "vertirta_bsp/stm32h7_regs.hpp"

namespace vertirta::bsp {

void usart3_init(uint32_t apb1_hz, uint32_t baud) noexcept {
  // Enable GPIOB (scope pin PB0), GPIOD (PD8 TX / PD9 RX), and USART3 (APB1L bit 18).
  regs::reg32(regs::kRccAhb4enr) |= (1u << 1u) | (1u << 3u);
  regs::reg32(regs::kRccApb1lenr) |= (1u << 18u);
  regs::dsb();

  // PB0 as push-pull output for oscilloscope BSRR trigger.
  uint32_t gpiob_moder = regs::reg32(regs::kGpioBBase + regs::kGpioModerOff);
  gpiob_moder = (gpiob_moder & ~3u) | 1u;
  regs::reg32(regs::kGpioBBase + regs::kGpioModerOff) = gpiob_moder;

  // PD8 (TX) & PD9 (RX) as AF7 (USART3).
  uint32_t gpiod_moder = regs::reg32(regs::kGpioDBase + regs::kGpioModerOff);
  gpiod_moder = (gpiod_moder & ~(0xFu << 16u)) | (0xAu << 16u);
  regs::reg32(regs::kGpioDBase + regs::kGpioModerOff) = gpiod_moder;

  uint32_t gpiod_afrh = regs::reg32(regs::kGpioDBase + regs::kGpioAfrhOff);
  gpiod_afrh = (gpiod_afrh & ~0xFFu) | 0x77u;  // AF7 on pins 8 and 9
  regs::reg32(regs::kGpioDBase + regs::kGpioAfrhOff) = gpiod_afrh;

  // Configure USART3 non-DMA TX/RX (FIFO mode optional, avoids Erratum 2.20.4 DMA stream lock).
  regs::reg32(regs::kUsartCr1) = 0u;
  regs::reg32(regs::kUsartBrr) = (baud > 0u) ? (apb1_hz / baud) : 1041u;
  regs::reg32(regs::kUsartCr1) = (1u << 3u) | (1u << 2u) | (1u << 0u);  // TE | RE | UE
}

void usart3_write_char(char c) noexcept {
  for (uint32_t spin = 0u; spin < 100000u; ++spin) {
    if ((regs::reg32(regs::kUsartIsr) & (1u << 7u)) != 0u) {  // TXE / TXFNF
      break;
    }
  }
  regs::reg32(regs::kUsartTdr) = static_cast<uint32_t>(static_cast<uint8_t>(c));
}

void usart3_write_str(const char* s) noexcept {
  if (s == nullptr) {
    return;
  }
  while (*s != '\0') {
    usart3_write_char(*s++);
  }
}

void usart3_write_u32(uint32_t v) noexcept {
  char buf[12]{};
  int pos = 0;
  if (v == 0u) {
    usart3_write_char('0');
    return;
  }
  while (v > 0u && pos < 11) {
    buf[pos++] = static_cast<char>('0' + (v % 10u));
    v /= 10u;
  }
  while (pos > 0) {
    usart3_write_char(buf[--pos]);
  }
}

void usart3_write_hex32(uint32_t v) noexcept {
  static constexpr char kHex[] = "0123456789ABCDEF";
  usart3_write_str("0x");
  for (int shift = 28; shift >= 0; shift -= 4) {
    usart3_write_char(kHex[(v >> static_cast<unsigned>(shift)) & 0xFu]);
  }
}

}  // namespace vertirta::bsp
