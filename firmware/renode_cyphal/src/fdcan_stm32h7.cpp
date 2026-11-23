// SPDX-License-Identifier: Apache-2.0
#include "vertirta_cyphal/fdcan_stm32h7.hpp"

#include "vertirta_bsp/stm32h7_regs.hpp"

namespace vertirta::cyphal {

namespace {

constexpr std::uint32_t kFdcan1Base = 0x4000A000u;
constexpr std::uint32_t kFdcanMsgRamBase = 0x4000AC00u;

constexpr std::uint32_t kFdcanDbtp = kFdcan1Base + 0x00Cu;
constexpr std::uint32_t kFdcanCccr = kFdcan1Base + 0x018u;
constexpr std::uint32_t kFdcanNbtp = kFdcan1Base + 0x01Cu;
constexpr std::uint32_t kFdcanGfc = kFdcan1Base + 0x080u;
constexpr std::uint32_t kFdcanSidfc = kFdcan1Base + 0x084u;
constexpr std::uint32_t kFdcanXidfc = kFdcan1Base + 0x088u;
constexpr std::uint32_t kFdcanRxf0c = kFdcan1Base + 0x0A0u;
constexpr std::uint32_t kFdcanRxf0s = kFdcan1Base + 0x0A4u;
constexpr std::uint32_t kFdcanRxf0a = kFdcan1Base + 0x0A8u;
constexpr std::uint32_t kFdcanRxesc = kFdcan1Base + 0x0BCu;
constexpr std::uint32_t kFdcanTxbc = kFdcan1Base + 0x0C0u;
constexpr std::uint32_t kFdcanTxfqs = kFdcan1Base + 0x0C4u;
constexpr std::uint32_t kFdcanTxesc = kFdcan1Base + 0x0C8u;
constexpr std::uint32_t kFdcanTxbar = kFdcan1Base + 0x0D0u;

constexpr std::uint32_t kRccD2ccip1r = vertirta::bsp::regs::kRccBase + 0x050u;
constexpr std::uint32_t kRccApb1henr = vertirta::bsp::regs::kRccBase + 0x0E4u;

// CCCR bit masks (RM0433 + ES0392 Errata 2.24.3 / 2.24.4).
constexpr std::uint32_t kCccrInit = (1u << 0u);
constexpr std::uint32_t kCccrCce = (1u << 1u);
constexpr std::uint32_t kCccrDar = (1u << 6u);   // MUST stay 0 (DAR mode OFF; ES0392 Erratum 2.24.4)
constexpr std::uint32_t kCccrFdoe = (1u << 8u);  // 1 = CAN-FD operation enabled
constexpr std::uint32_t kCccrBrse = (1u << 9u);  // 1 = Bit-Rate Switch enabled
constexpr std::uint32_t kCccrEfbi = (1u << 13u); // MUST stay 0 (Edge Filtering OFF; ES0392 Erratum 2.24.3)

// Message RAM partition (word offsets from kFdcanMsgRamBase = 0x4000AC00):
//   Rx FIFO0: 16 elements x 18 words (72 B: 2 header + 16 data) = 288 words [0x000..0x480)
//   Tx FIFO :  8 elements x 18 words (72 B: 2 header + 16 data) = 144 words [0x480..0x6C0)
constexpr std::uint32_t kRxFifo0Count = 16u;
constexpr std::uint32_t kTxFifoCount = 8u;
constexpr std::uint32_t kElementWords = 18u;
constexpr std::uint32_t kRxFifo0WordOffset = 0u;
constexpr std::uint32_t kTxFifoWordOffset = kRxFifo0Count * kElementWords;

constexpr std::uint32_t kPollLimit = 200000u;

bool g_fdcan1_ready = false;

bool wait_init_bit(bool expect_set) noexcept {
  for (std::uint32_t i = 0u; i < kPollLimit; ++i) {
    const bool is_set = (vertirta::bsp::regs::reg32(kFdcanCccr) & kCccrInit) != 0u;
    if (is_set == expect_set) {
      return true;
    }
  }
  return false;
}

}  // namespace

std::uint8_t canfd_len_to_dlc(std::uint8_t len) noexcept {
  if (len <= 8u) {
    return len;
  }
  if (len <= 12u) {
    return 9u;
  }
  if (len <= 16u) {
    return 10u;
  }
  if (len <= 20u) {
    return 11u;
  }
  if (len <= 24u) {
    return 12u;
  }
  if (len <= 32u) {
    return 13u;
  }
  if (len <= 48u) {
    return 14u;
  }
  return 15u;
}

std::uint8_t canfd_dlc_to_len(std::uint8_t dlc) noexcept {
  static constexpr std::uint8_t kTable[16] = {
      0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 12u, 16u, 20u, 24u, 32u, 48u, 64u};
  return kTable[dlc & 0x0Fu];
}

bool fdcan1_init(const vertirta::bsp::SiliconRevInfo& rev) noexcept {
  g_fdcan1_ready = false;

  // ES0392 Erratum 2.24.2 / 2.2.10 hard safety interlock: NEVER arm FDCAN on Rev Y silicon.
  if (rev.rev_y_fdcan_errata_flag) {
    vertirta::bsp::regs::reg32(kFdcanCccr) |= kCccrInit;
    return false;
  }

  // Enable FDCAN1 peripheral clock on APB1H (bit 8) and select PLL1Q kernel clock (0b01).
  vertirta::bsp::regs::reg32(kRccD2ccip1r) =
      (vertirta::bsp::regs::reg32(kRccD2ccip1r) & ~(3u << 28u)) | (1u << 28u);
  vertirta::bsp::regs::reg32(kRccApb1henr) |= (1u << 8u);
  const std::uint32_t clk_rb = vertirta::bsp::regs::reg32(kRccApb1henr);
  (void)clk_rb;

  // Enter initialization mode (INIT=1) then enable configuration change (CCE=1); fail-closed on timeout.
  vertirta::bsp::regs::reg32(kFdcanCccr) |= kCccrInit;
  if (!wait_init_bit(true)) {
    g_fdcan1_ready = false;
    return false;
  }
  vertirta::bsp::regs::reg32(kFdcanCccr) |= kCccrCce;

  // Configure CAN-FD + BRS with BOTH EFBI=0 (Erratum 2.24.3) and DAR=0 (Erratum 2.24.4) cleared.
  std::uint32_t cccr = vertirta::bsp::regs::reg32(kFdcanCccr);
  cccr |= (kCccrFdoe | kCccrBrse);
  cccr &= ~(kCccrEfbi | kCccrDar);
  vertirta::bsp::regs::reg32(kFdcanCccr) = cccr;

  // Nominal 1 Mbps & Data 5 Mbps bit timing (assuming 80 MHz FDCAN kernel clock):
  //   Nominal: NBRP=0 (/1), NTSEG1=62, NTSEG2=15, NSJW=15 -> 80 / (1+63+16) = 1.0 Mbps (80% SP)
  //   Data   : DBRP=0 (/1), DTSEG1=11, DTSEG2=2,  DSJW=2  -> 80 / (1+12+3)  = 5.0 Mbps (81.25% SP)
  vertirta::bsp::regs::reg32(kFdcanNbtp) =
      (15u << 25u) | (0u << 16u) | (62u << 8u) | 15u;
  vertirta::bsp::regs::reg32(kFdcanDbtp) =
      (0u << 16u) | (11u << 8u) | (2u << 4u) | 2u;

  // Accept all 29-bit extended frames into Rx FIFO0; reject non-matching standard 11-bit frames.
  vertirta::bsp::regs::reg32(kFdcanGfc) = (2u << 4u) | (0u << 2u);
  vertirta::bsp::regs::reg32(kFdcanSidfc) = 0u;
  vertirta::bsp::regs::reg32(kFdcanXidfc) = 0u;

  // Zero-initialize Message RAM partition.
  for (std::uint32_t w = 0u; w < ((kRxFifo0Count + kTxFifoCount) * kElementWords); ++w) {
    vertirta::bsp::regs::reg32(kFdcanMsgRamBase + (w * 4u)) = 0u;
  }

  // Configure 64-byte CAN-FD element sizes for Rx FIFO0 (F0DS=0b111) and Tx FIFO (TBDS=0b111).
  vertirta::bsp::regs::reg32(kFdcanRxesc) = 0x7u;
  vertirta::bsp::regs::reg32(kFdcanTxesc) = 0x7u;

  // Configure Rx FIFO0: blocking mode, 16 elements at word offset 0.
  vertirta::bsp::regs::reg32(kFdcanRxf0c) =
      (kRxFifo0Count << 16u) | (kRxFifo0WordOffset << 2u);

  // Configure Tx FIFO (Erratum 2.24.5: TFQM=0 dedicated Tx FIFO mode, 8 elements).
  vertirta::bsp::regs::reg32(kFdcanTxbc) =
      (0u << 30u) | (kTxFifoCount << 24u) | (kTxFifoWordOffset << 2u);

  // Leave initialization mode -> normal CAN-FD bus operation; fail-closed if kernel clock stalls INIT clear.
  vertirta::bsp::regs::reg32(kFdcanCccr) &= ~(kCccrInit | kCccrCce);
  if (!wait_init_bit(false)) {
    vertirta::bsp::regs::reg32(kFdcanCccr) |= kCccrInit;
    g_fdcan1_ready = false;
    return false;
  }
  vertirta::bsp::regs::dsb();
  vertirta::bsp::regs::isb();

  g_fdcan1_ready = true;
  return true;
}

bool fdcan1_transmit(const CanFdFrame& frame) noexcept {
  if (!g_fdcan1_ready) {
    return false;
  }
  const std::uint32_t txfqs = vertirta::bsp::regs::reg32(kFdcanTxfqs);
  if ((txfqs & (1u << 21u)) != 0u) {
    return false;  // Tx FIFO full (TFQF == 1)
  }
  const std::uint32_t put_idx = (txfqs >> 16u) & 0x1Fu;
  if (put_idx >= kTxFifoCount) {
    return false;
  }

  const std::uint32_t elem_addr =
      kFdcanMsgRamBase + ((kTxFifoWordOffset + (put_idx * kElementWords)) * 4u);

  const std::uint8_t dlc = canfd_len_to_dlc(frame.len);
  const std::uint32_t t0 = (1u << 30u) | (frame.ext_id & 0x1FFFFFFFu);  // XTD = 1
  const std::uint32_t t1 = (1u << 21u) | (1u << 20u) | (static_cast<std::uint32_t>(dlc) << 16u);

  vertirta::bsp::regs::reg32(elem_addr + 0u) = t0;
  vertirta::bsp::regs::reg32(elem_addr + 4u) = t1;

  for (std::uint32_t w = 0u; w < 16u; ++w) {
    const std::uint32_t b = w * 4u;
    const std::uint32_t word =
        static_cast<std::uint32_t>(frame.data[b + 0u]) |
        (static_cast<std::uint32_t>(frame.data[b + 1u]) << 8u) |
        (static_cast<std::uint32_t>(frame.data[b + 2u]) << 16u) |
        (static_cast<std::uint32_t>(frame.data[b + 3u]) << 24u);
    vertirta::bsp::regs::reg32(elem_addr + 8u + (w * 4u)) = word;
  }

  vertirta::bsp::regs::dsb();
  vertirta::bsp::regs::reg32(kFdcanTxbar) = (1u << put_idx);
  return true;
}

bool fdcan1_receive(CanFdFrame& out_frame) noexcept {
  if (!g_fdcan1_ready) {
    return false;
  }
  const std::uint32_t rxf0s = vertirta::bsp::regs::reg32(kFdcanRxf0s);
  const std::uint32_t fill_level = rxf0s & 0x7Fu;
  if (fill_level == 0u) {
    return false;
  }

  const std::uint32_t get_idx = (rxf0s >> 8u) & 0x3Fu;
  if (get_idx >= kRxFifo0Count) {
    return false;
  }

  const std::uint32_t elem_addr =
      kFdcanMsgRamBase + ((kRxFifo0WordOffset + (get_idx * kElementWords)) * 4u);

  const std::uint32_t r0 = vertirta::bsp::regs::reg32(elem_addr + 0u);
  const std::uint32_t r1 = vertirta::bsp::regs::reg32(elem_addr + 4u);
  const bool is_ext = (r0 & (1u << 30u)) != 0u;
  const std::uint8_t dlc = static_cast<std::uint8_t>((r1 >> 16u) & 0x0Fu);
  const std::uint8_t len = canfd_dlc_to_len(dlc);

  out_frame.ext_id = r0 & 0x1FFFFFFFu;
  out_frame.len = len;
  for (std::uint32_t w = 0u; w < 16u; ++w) {
    const std::uint32_t word = vertirta::bsp::regs::reg32(elem_addr + 8u + (w * 4u));
    const std::uint32_t b = w * 4u;
    out_frame.data[b + 0u] = static_cast<std::uint8_t>(word & 0xFFu);
    out_frame.data[b + 1u] = static_cast<std::uint8_t>((word >> 8u) & 0xFFu);
    out_frame.data[b + 2u] = static_cast<std::uint8_t>((word >> 16u) & 0xFFu);
    out_frame.data[b + 3u] = static_cast<std::uint8_t>((word >> 24u) & 0xFFu);
  }

  // Acknowledge RX FIFO0 element read.
  vertirta::bsp::regs::reg32(kFdcanRxf0a) = get_idx;
  return is_ext;
}

}  // namespace vertirta::cyphal
