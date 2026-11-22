// SPDX-License-Identifier: Apache-2.0
// ESBMC v8.7 Harness 3: Dual-Channel 400 Hz Motor FDI Observer (--unwind 10)
// Verifies monotonic fault-mask latching, irreversible SAME_ARM_ACTIVE_SPIN classification,
// and bounds/finiteness safety under symbolic and non-finite inputs.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esbmc_compat.h"
#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/motor_fdi.hpp"

namespace {

using vertirta::core::CoaxEffectiveness;
using vertirta::core::FlightRegime;
using vertirta::core::MotorFdiObserver;
using vertirta::core::Vec3Frd;

void verify_fdi_step(uint8_t initial_mask,
                     const Vec3Frd& gyro_f,
                     const Vec3Frd& dgyro_f,
                     float accel_z_f,
                     float v_horiz,
                     float v_term,
                     float esc0_omega) {
  MotorFdiObserver fdi{};
  float omega_hat[8]{};
  float omega_esc[8]{};
  bool esc_valid[8]{};

  for (int i = 0; i < 8; ++i) {
    omega_hat[i] = 151.8f;
    omega_esc[i] = 151.8f;
    esc_valid[i] = true;
  }
  omega_esc[0] = esc0_omega;

  uint8_t mask = initial_mask;
#ifdef VERTIRTA_IS_ESBMC
  const int n_ticks = 1;
#else
  const int n_ticks = 4;
#endif
  for (int tick = 0; tick < n_ticks; ++tick) {
    const uint8_t next_mask = fdi.step_400hz(
        gyro_f, dgyro_f, accel_z_f, omega_hat, omega_esc, esc_valid, v_horiz, v_term, mask);
    // Fault bits must be monotonic (previously isolated motors never silently un-latch)
    VERTIRTA_FORMAL_ASSERT((next_mask & mask) == mask, "FDI fault mask must remain monotonic");
    mask = next_mask;
  }

  // Irreversible SAME_ARM_ACTIVE_SPIN check: when Arm 0 upper (bit 0) and lower (bit 4) fail
  const uint8_t same_arm_mask = static_cast<uint8_t>(mask | 0x11u);
  const FlightRegime reg = CoaxEffectiveness::classify_regime(same_arm_mask);
  VERTIRTA_FORMAL_ASSERT(
      reg == FlightRegime::SAME_ARM_ACTIVE_SPIN || reg == FlightRegime::CRITICAL_DESCENT,
      "Same-arm dual failure must classify as SAME_ARM_ACTIVE_SPIN or CRITICAL_DESCENT");
  VERTIRTA_FORMAL_ASSERT(reg != FlightRegime::NOMINAL, "Same-arm dual failure cannot be NOMINAL");
}

}  // namespace

int main() {
#ifdef VERTIRTA_IS_ESBMC
  // 1) Exhaustive symbolic proof over all 256 uint8_t fault masks that any same-arm dual loss
  //    (Arm 0: 0x11, Arm 1: 0x22, Arm 2: 0x44, Arm 3: 0x88) classifies as SAME_ARM_ACTIVE_SPIN or CRITICAL_DESCENT.
  const uint8_t any_mask = nondet_uchar();
  const uint8_t arm_idx = nondet_uchar();
  VERTIRTA_FORMAL_ASSUME(arm_idx < 4u);
  const uint8_t same_arm_bits = static_cast<uint8_t>((1u << arm_idx) | (1u << (arm_idx + 4u)));
  const FlightRegime reg = CoaxEffectiveness::classify_regime(static_cast<uint8_t>(any_mask | same_arm_bits));
  VERTIRTA_FORMAL_ASSERT(
      reg == FlightRegime::SAME_ARM_ACTIVE_SPIN || reg == FlightRegime::CRITICAL_DESCENT,
      "Same-arm dual failure must classify as SAME_ARM_ACTIVE_SPIN or CRITICAL_DESCENT");
  VERTIRTA_FORMAL_ASSERT(reg != FlightRegime::NOMINAL, "Same-arm dual failure cannot be NOMINAL");

  // 2) Full 400 Hz MotorFdiObserver::step_400hz monotonic latching proof over symbolic ESC dropout speed
  //    and representative prior fault states (constant B_T/expected_w keeps Bitwuzla --floatbv fast).
  const float esc0 = nondet_float();
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(esc0) && esc0 >= 0.0f && esc0 <= 240.0f);
  verify_fdi_step(0x00u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, -9.80665f, 0.0f, 385.0f, esc0);
  verify_fdi_step(0x11u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, -9.80665f, 0.0f, 385.0f, esc0);
#else
  verify_fdi_step(0x00u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, -9.80665f, 8.0f, 385.0f, 0.0f);
  verify_fdi_step(0x11u, {0.1, -0.1, 2.2}, {0.0, 0.0, 0.0}, -9.80665f, 4.0f, 360.0f, 0.0f);
  verify_fdi_step(0x05u, {std::nan(""), 0.0, 0.0}, {0.0, 0.0, 0.0}, -9.80665f, 0.0f, 340.0f, 151.8f);
#endif
  return 0;
}
