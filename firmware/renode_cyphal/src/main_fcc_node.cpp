// SPDX-License-Identifier: Apache-2.0
// Bare-Metal STM32H753 Dual-FCC OpenCyphal CAN-FD Node Firmware (Node 10 Primary / Node 11 Secondary)
// Links frozen libvertirta_flight_core.a inside 64 KB zero-wait-state ITCM and isolates the
// 32 KiB o1heap v3.0.0 CAN-FD arena strictly inside DTCM.

#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta_bsp/dwt_profiler.hpp"
#include "vertirta_bsp/stm32h7_boot.hpp"
#include "vertirta_bsp/uart_console.hpp"
#include "vertirta_cyphal/cyphal_bus_node.hpp"
#include "vertirta_cyphal/fdcan_stm32h7.hpp"
#include "vertirta_cyphal/o1heap_arena.hpp"
#include "vertirta_cyphal/safety_envelope_codec.h"

#ifndef VERTIRTA_FCC_NODE_ID
#define VERTIRTA_FCC_NODE_ID 10u
#endif

namespace {

constexpr std::uint8_t kNodeId = static_cast<std::uint8_t>(VERTIRTA_FCC_NODE_ID);
constexpr vertirta::cyphal::FccRole kNodeRole =
    (kNodeId == vertirta::cyphal::CyphalFccNode::kPrimaryNodeId)
        ? vertirta::cyphal::FccRole::PRIMARY
        : vertirta::cyphal::FccRole::SECONDARY;

volatile std::uint32_t g_systick_400hz_count = 0u;

// Static DTCM allocations: 32 KiB o1heap arena + reactive Cyphal node + frozen flight_core kernels.
vertirta::cyphal::O1HeapDtcmArena g_o1heap_arena;
vertirta::cyphal::CyphalFccNode g_cyphal_node(kNodeRole, kNodeId, g_o1heap_arena);

vertirta::core::TmrVoter g_tmr_voter;
vertirta::core::IndiWlsController g_indi_wls;
vertirta::core::MotorFdiObserver g_fdi_obs;

vertirta_flight_SafetyEnvelope_1_0 compute_local_envelope(std::uint32_t tick_idx,
                                                          std::uint32_t now_us) noexcept {
  // Run frozen libvertirta_flight_core.a 400 Hz supervision step inside ITCM.
  vertirta::core::ImuSampleFrd imus[3]{};
  for (int i = 0; i < 3; ++i) {
    imus[i].valid = true;
    imus[i].accel_m_s2 = {0.0, 0.0, -vertirta::core::params::kGravity};
    imus[i].gyro_rad_s = {0.0, 0.0, 0.0};
    imus[i].t_us = static_cast<std::uint64_t>(now_us);
  }
  const auto tmr_out = g_tmr_voter.step(imus, {0.0, 0.0, 0.0}, false);

  float omega_esc[8]{};
  bool esc_valid[8]{};
  for (int i = 0; i < 8; ++i) {
    omega_esc[i] = static_cast<float>(vertirta::core::params::kOmegaHover);
    esc_valid[i] = true;
  }

  float omega_hat[8]{};
  g_indi_wls.get_omega_hat(omega_hat);
  const std::uint8_t fdi_mask = g_fdi_obs.step_400hz(
      g_indi_wls.filtered_gyro_frd(),
      g_indi_wls.filtered_dgyro_frd(),
      static_cast<float>(tmr_out.accel_voted.z),
      omega_hat,
      omega_esc,
      esc_valid,
      0.0f,
      398.0f,
      0x00u);

  vertirta::core::IndiInput in{};
  in.q_frd_ned = {1.0, 0.0, 0.0, 0.0};
  in.gyro_frd = tmr_out.gyro_voted;
  in.accel_des_ned = {0.0, 0.0, -vertirta::core::params::kGravity};
  in.yaw_rate_des = 0.0;
  in.v_horiz_mps = 0.0f;
  in.v_term_v = 398.0f;
  in.fault_mask = fdi_mask;
  in.esc_omega_rad_s = omega_esc;
  in.esc_valid = esc_valid;

  float omega_cmd[8]{};
  (void)g_indi_wls.step_400hz(in, omega_cmd);

  const auto env = vertirta::core::CoaxEffectiveness::compute_envelope(
      fdi_mask,
      static_cast<std::uint8_t>(tmr_out.state),
      tmr_out.health_mask,
      398.0f,
      92.0f,
      0.80f,
      0.0f,
      0.0f,
      0.0f,
      0u);

  vertirta_flight_SafetyEnvelope_1_0 msg{};
  msg.wrench_min_frd[0] = -env.t_max_n;
  msg.wrench_min_frd[1] = -250.0f;
  msg.wrench_min_frd[2] = -250.0f;
  msg.wrench_min_frd[3] = -120.0f;
  msg.wrench_max_frd[0] = 0.0f;
  msg.wrench_max_frd[1] = 250.0f;
  msg.wrench_max_frd[2] = 250.0f;
  msg.wrench_max_frd[3] = 120.0f;
  msg.seq = tick_idx;
  msg.t_us = now_us;
  msg.fault_mask = env.fault_mask;
  msg.flight_regime = static_cast<std::uint8_t>(env.regime);
  msg.tmr_state = env.tmr_state;
  msg.cel_latched = 0u;
  msg.crc32c = 0u;
  return msg;
}

}  // namespace

extern "C" void SysTick_Handler(void) {
  g_systick_400hz_count = g_systick_400hz_count + 1u;
}

extern "C" void* __wrap_malloc(std::size_t) {
  for (;;) {
    __asm volatile("bkpt #1");
  }
}

extern "C" void* __wrap_calloc(std::size_t, std::size_t) {
  for (;;) {
    __asm volatile("bkpt #2");
  }
}

extern "C" void* __wrap_realloc(void*, std::size_t) {
  for (;;) {
    __asm volatile("bkpt #3");
  }
}

int main() {
  auto rev = vertirta::bsp::inspect_silicon_rev();
  vertirta::bsp::init_clocks_480mhz_or_fallback(rev);
  vertirta::bsp::init_mpu_8_regions();
  vertirta::bsp::init_fpu_eager_and_caches();
  vertirta::bsp::unlock_dwt_cyccnt();
  vertirta::bsp::usart3_init(rev.axi_hz / 2u, 115200u);

  const bool fdcan_ok = vertirta::cyphal::fdcan1_init(rev);
  if (!fdcan_ok) {
    vertirta::bsp::usart3_write_str("[FCC] FDCAN_ARM_BLOCKED REV_Y_ERRATUM_2_24_2\n");
    for (;;) {
      __asm volatile("wfi");
    }
  }

  g_o1heap_arena.reset();
  g_cyphal_node.reset(0u);
  vertirta::bsp::init_400hz_systick(rev.sysclk_hz);

  vertirta::bsp::usart3_write_str(
      (kNodeRole == vertirta::cyphal::FccRole::PRIMARY)
          ? "[FCC10] BOOT_OK ROLE=PRIMARY NODE_ID=10 SUBJECT=1024\n"
          : "[FCC11] BOOT_OK ROLE=SECONDARY NODE_ID=11 STANDBY SUBJECT=1024\n");

  std::uint32_t last_tick = 0u;
  for (;;) {
    // Poll incoming CAN-FD frames continuously between 400 Hz SysTick ticks.
    vertirta::cyphal::CanFdFrame rx{};
    const std::uint32_t now_us = g_systick_400hz_count * VERTIRTA_FAILOVER_PERIOD_US;
    while (vertirta::cyphal::fdcan1_receive(rx)) {
      (void)g_cyphal_node.on_rx_frame(rx, now_us);
    }

    const std::uint32_t cur_tick = g_systick_400hz_count;
    if (cur_tick != last_tick) {
      last_tick = cur_tick;
      const std::uint32_t tick_us = cur_tick * VERTIRTA_FAILOVER_PERIOD_US;
      const auto env = compute_local_envelope(cur_tick, tick_us);
      const bool failover_triggered = g_cyphal_node.tick(tick_us, env);

      // Drain priority-ordered TX queue out to STM32H7 FDCAN1.
      vertirta::cyphal::CanFdFrame tx{};
      while (g_cyphal_node.pop_tx_frame(tx)) {
        (void)vertirta::cyphal::fdcan1_transmit(tx);
      }

      if (failover_triggered) {
        vertirta::bsp::usart3_write_str("[FCC11] FAILOVER_ACTIVE TAKEOVER_US=");
        vertirta::bsp::usart3_write_u32(g_cyphal_node.takeover_latency_us());
        vertirta::bsp::usart3_write_str(" CRC_REJECTS=");
        vertirta::bsp::usart3_write_u32(g_cyphal_node.rx_crc_reject_count());
        vertirta::bsp::usart3_write_str(" AUTHORITY=NODE11\n");
      }

      (void)vertirta::bsp::poll_and_clear_fpscr_flags();
    }
  }
}
