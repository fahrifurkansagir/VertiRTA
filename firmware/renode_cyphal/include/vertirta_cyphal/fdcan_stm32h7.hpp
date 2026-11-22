// SPDX-License-Identifier: Apache-2.0
// Bare-Metal STM32H753 FDCAN1 (Bosch M_CAN) Register Driver enforcing ES0392 Errata 2.24.2-2.24.5.
#pragma once

#include <cstdint>

#include "vertirta_bsp/stm32h7_boot.hpp"
#include "vertirta_cyphal/cyphal_bus_node.hpp"

namespace vertirta::cyphal {

// Initializes STM32H753 FDCAN1 in 1 Mbps nominal / 5 Mbps CAN-FD data mode with:
//   - Erratum 2.24.2 guard: REFUSES arming if rev.rev_y_fdcan_errata_flag == true (Rev Y)
//   - Erratum 2.24.3: Edge filtering explicitly OFF (CCCR.EFBI = 0)
//   - Erratum 2.24.4: DAR mode explicitly OFF (CCCR.DAR = 0; DAR=1 triggers wrong arbitration/control fields)
//   - Erratum 2.24.5: Dedicated Tx FIFO mode (TXBC.TFQM = 0), 64-byte CAN-FD elements
[[nodiscard]] bool fdcan1_init(const vertirta::bsp::SiliconRevInfo& rev) noexcept;

// Non-blocking CAN-FD TX enqueue into dedicated Tx FIFO (returns false if full or uninitialized).
[[nodiscard]] bool fdcan1_transmit(const CanFdFrame& frame) noexcept;

// Non-blocking CAN-FD RX pop from Rx FIFO0 (returns true if a 29-bit extended frame was read).
[[nodiscard]] bool fdcan1_receive(CanFdFrame& out_frame) noexcept;

// Maps byte count (0..64) <-> ISO 11898-1:2015 CAN-FD DLC code (0..15).
[[nodiscard]] std::uint8_t canfd_len_to_dlc(std::uint8_t len) noexcept;
[[nodiscard]] std::uint8_t canfd_dlc_to_len(std::uint8_t dlc) noexcept;

}  // namespace vertirta::cyphal
