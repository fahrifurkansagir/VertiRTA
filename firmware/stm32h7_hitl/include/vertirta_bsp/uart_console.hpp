// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vertirta::bsp {

// Non-blocking / polling USART3 console (PD8 TX / PD9 RX on NUCLEO-H743ZI2 & Pixhawk 6X FMU Debug).
// Intentionally avoids USART DMA per ES0392 Rev 16 Erratum 2.20.4 and runs strictly outside the
// DWT->CYCCNT profiled 400 Hz control tick.
void usart3_init(uint32_t apb1_hz = 120'000'000u, uint32_t baud = 115'200u) noexcept;
void usart3_write_char(char c) noexcept;
void usart3_write_str(const char* s) noexcept;
void usart3_write_u32(uint32_t v) noexcept;
void usart3_write_hex32(uint32_t v) noexcept;

}  // namespace vertirta::bsp
