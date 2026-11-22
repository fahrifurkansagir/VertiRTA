// SPDX-License-Identifier: Apache-2.0
// Bare-Metal STM32H753 / NUCLEO-H743ZI2 400 Hz DWT->CYCCNT HITL Profiling Harness (DR-08 Sec 1.4).
// Measures Empirical High-Water-Mark (HWM) silicon cycle counts for libvertirta_flight_core.a under
// adversarial worst-case coaxial X8 fault stimuli (NOT a standalone DO-178C WCET proof).

#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/daqp_qp.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta_bsp/dwt_profiler.hpp"
#include "vertirta_bsp/stm32h7_boot.hpp"
#include "vertirta_bsp/uart_console.hpp"

#ifdef VERTIRTA_ENABLE_TINYUSB
// Optional MIT TinyUSB CDC hook (CFG_TUSB_MCU=OPT_MCU_STM32H7, BOARD_TUD_RHPORT=0, OPT_OS_NONE).
extern "C" void tud_int_handler(uint8_t rhport);
extern "C" void OTG_FS_IRQHandler() { tud_int_handler(0u); }
#endif

namespace {

using vertirta::bsp::CycleSummary;
using vertirta::bsp::DwtCycleProfiler;
using vertirta::bsp::SiliconRevInfo;
using vertirta::core::CoaxialPlant6Dof;
using vertirta::core::ImuSampleFrd;
using vertirta::core::IndiInput;
using vertirta::core::IndiWlsController;
using vertirta::core::MotorFdiObserver;
using vertirta::core::PlantState;
using vertirta::core::StaticDaqpSolver;
using vertirta::core::TmrVoter;
using vertirta::core::Vec3Frd;
using vertirta::core::Vec3Ned;

// Static DTCM-resident flight-control arenas (0 B heap; flight_core library itself stays 0 B RAM).
TmrVoter g_voter{};
IndiWlsController g_ctrl{true};
MotorFdiObserver g_fdi{};
CoaxialPlant6Dof g_plant{};
StaticDaqpSolver<5, 32> g_qp{};
DwtCycleProfiler g_profiler{};

#if defined(__arm__) || defined(__thumb__)
__attribute__((section(".dma_nocache"), aligned(32)))
#endif
uint8_t g_dma_staging_buf[512]{};

// Placed in ITCM alongside libvertirta_flight_core.a so the 400 Hz step executes zero-wait-state.
#if defined(__arm__) || defined(__thumb__)
__attribute__((section(".itcm_text.hitl_step"), noinline))
#endif
uint8_t execute_adversarial_400hz_tick(uint32_t tick) noexcept {
  const PlantState st = g_plant.state();

  // Inject adversarial fault schedule across 10,000 ticks (25 s @ 400 Hz):
  //   0..2499   : Nominal 8-rotor hover + coastal gust
  //   2500..4999: IMU0 +0.15 rad/s bias + Single-Motor loss (M0)
  //   5000..7499: Cross-Arm Dual-Motor loss (M0 + M1)
  //   7500..9999: Same-Arm Coaxial Dual-Motor loss (M0 + M4 -> Option B1b active flat spin)
  uint8_t fault_mask = 0x00u;
  if (tick >= 7500u) {
    fault_mask = static_cast<uint8_t>((1u << 0u) | (1u << 4u));
  } else if (tick >= 5000u) {
    fault_mask = static_cast<uint8_t>((1u << 0u) | (1u << 1u));
  } else if (tick >= 2500u) {
    fault_mask = static_cast<uint8_t>(1u << 0u);
  }

  ImuSampleFrd imus[3]{};
  for (int i = 0; i < 3; ++i) {
    imus[i].gyro_rad_s = st.omega_frd;
    imus[i].accel_m_s2 = {0.0, 0.0, -9.80665};
    imus[i].t_us = static_cast<uint64_t>(tick) * 2500ULL;
    imus[i].valid = true;
  }
  if (tick >= 2500u && tick < 5000u) {
    imus[0].gyro_rad_s.x += 0.15;
  }

  const auto vote = g_voter.step(imus, st.omega_frd, true);

  float omega_esc[8]{};
  bool esc_valid[8]{};
  for (int i = 0; i < 8; ++i) {
    const bool failed = ((fault_mask >> i) & 1u) != 0u;
    omega_esc[i] = failed ? 0.0f : static_cast<float>(st.rotor_omega[i]);
    esc_valid[i] = true;
  }

  float omega_hat[8]{};
  g_ctrl.get_omega_hat(omega_hat);
  const uint8_t fdi_mask = g_fdi.step_400hz(
      g_ctrl.filtered_gyro_frd(),
      g_ctrl.filtered_dgyro_frd(),
      static_cast<float>(vote.accel_voted.z),
      omega_hat,
      omega_esc,
      esc_valid,
      4.0f,
      st.battery.v_term_v,
      fault_mask);

  IndiInput in{};
  in.q_frd_ned = st.q_frd_ned;
  in.gyro_frd = vote.gyro_voted;
  in.accel_des_ned = {0.5, -0.3, -9.80665};
  in.yaw_rate_des = 0.0;
  in.v_horiz_mps = 4.0f;
  in.v_term_v = st.battery.v_term_v;
  in.fault_mask = fdi_mask;
  in.esc_omega_rad_s = omega_esc;
  in.esc_valid = esc_valid;

  float omega_cmd[8]{};
  const auto wls_res = g_ctrl.step_400hz(in, omega_cmd);

  // Exercise static DAQP corner every 4th tick (100 Hz outer-loop cadence) inside bounded budget.
  if ((tick & 3u) == 0u) {
    const double H_diag[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    const double f[5] = {-0.5, 0.3, 0.0, 0.0, 0.0};
    const double A[1 * 5] = {1.0, 0.0, 0.0, 0.0, 0.0};
    const double x_lo[5] = {-6.0, -6.0, -4.0, 0.0, 0.0};
    const double x_hi[5] = {6.0, 6.0, 4.0, 100.0, 100.0};
    const double b_lo[1] = {-100.0};
    const double b_hi[1] = {5.0};
    const auto qp_res = g_qp.solve(5, 1, H_diag, f, A, x_lo, x_hi, b_lo, b_hi, true);
    g_dma_staging_buf[0] = static_cast<uint8_t>(qp_res.exit_flag & 0xFF);
  }

  const Vec3Ned gust{3.5, -2.0, 0.5};
  (void)g_plant.step_400hz(omega_cmd, fault_mask, gust);
  return static_cast<uint8_t>(wls_res.iters ^ fdi_mask);
}

void emit_json_report(const SiliconRevInfo& rev, const CycleSummary& s, uint8_t checksum) noexcept {
  using namespace vertirta::bsp;
  usart3_write_str("{\"schema\":\"vertirta.stm32h7_hitl/v1.0.0\",");
  usart3_write_str("\"timing_evidence_class\":\"EMPIRICAL_HWM_NOT_WCET_PROOF\",");
  usart3_write_str("\"rev_id\":\"");
  usart3_write_hex32(rev.rev_id);
  usart3_write_str("\",\"sysclk_hz\":");
  usart3_write_u32(rev.sysclk_hz);
  usart3_write_str(",\"flash_latency\":");
  usart3_write_u32(rev.flash_latency);
  usart3_write_str(",\"rev_y_fdcan_errata_flag\":");
  usart3_write_str(rev.rev_y_fdcan_errata_flag ? "true" : "false");
  usart3_write_str(",\"ticks\":");
  usart3_write_u32(s.count);
  usart3_write_str(",\"bracket_overhead_cycles\":");
  usart3_write_u32(s.bracket_overhead_cycles);
  usart3_write_str(",\"min_cycles\":");
  usart3_write_u32(s.min_cycles);
  usart3_write_str(",\"p50_cycles\":");
  usart3_write_u32(s.p50_cycles);
  usart3_write_str(",\"p99_cycles\":");
  usart3_write_u32(s.p99_cycles);
  usart3_write_str(",\"p999_cycles\":");
  usart3_write_u32(s.p999_cycles);
  usart3_write_str(",\"max_cycles\":");
  usart3_write_u32(s.max_cycles);
  usart3_write_str(",\"fpscr_sticky_or\":\"");
  usart3_write_hex32(s.fpscr_sticky_or);
  usart3_write_str("\",\"fpscr_nonzero_ticks\":");
  usart3_write_u32(s.fpscr_nonzero_ticks);
  usart3_write_str(",\"checksum\":");
  usart3_write_u32(checksum);
  usart3_write_str("}\r\n");
}

}  // namespace

int main() {
  using namespace vertirta::bsp;

  // Five-stage Cortex-M7 bare-metal boot sequence (ES0392 Rev 16 compliant).
  SiliconRevInfo rev = inspect_silicon_rev();
  init_clocks_480mhz_or_fallback(rev);
  init_mpu_8_regions();
  init_fpu_eager_and_caches();
  unlock_dwt_cyccnt();
#ifdef VERTIRTA_ENABLE_TINYUSB
  init_usb_otg_fs_clock_hsi48();
#endif
  usart3_init(rev.axi_hz / 2u, 115'200u);
  init_400hz_tim2();

  g_voter.reset();
  g_ctrl.reset();
  g_fdi.reset();
  g_plant.reset({0.0, 0.0, -35.0}, {4.0, 0.0, 0.0}, 0.65f);
  g_profiler.reset();
  g_profiler.calibrate_overhead(1024u);

  uint8_t checksum = 0u;
  for (uint32_t tick = 0u; tick < DwtCycleProfiler::kMaxSamples; ++tick) {
    scope_gpio_set(true);
    const uint32_t t0 = read_dwt_cyccnt_serialized();
    checksum ^= execute_adversarial_400hz_tick(tick);
    const uint32_t t1 = read_dwt_cyccnt_serialized();
    scope_gpio_set(false);
    // ES0392 Erratum 2.1.6: poll and clear unmapped Cortex-M7 FPU sticky flags every 400 Hz tick.
    const uint32_t fpscr_flags = poll_and_clear_fpscr_flags();
    g_profiler.record_sample(t0, t1, fpscr_flags);
  }

  const CycleSummary summary = g_profiler.finalize_summary();
  emit_json_report(rev, summary, checksum);
  return 0;
}
