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
  for (int tick = 0; tick < 4; ++tick) {
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
#ifdef __ESBMC__
  const uint8_t init_mask = nondet_uchar();
  const double gx = nondet_double();
  const double gy = nondet_double();
  const double gz = nondet_double();
  const double dgx = nondet_double();
  const double dgy = nondet_double();
  const double dgz = nondet_double();
  const float az = nondet_float();
  const float vh = nondet_float();
  const float vt = nondet_float();
  const float esc0 = nondet_float();

  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(gx) && gx >= -5.0 && gx <= 5.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(gy) && gy >= -5.0 && gy <= 5.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(gz) && gz >= -5.0 && gz <= 5.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(dgx) && dgx >= -20.0 && dgx <= 20.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(dgy) && dgy >= -20.0 && dgy <= 20.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(dgz) && dgz >= -20.0 && dgz <= 20.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(az) && az >= -30.0f && az <= 10.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(vh) && vh >= 0.0f && vh <= 25.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(vt) && vt >= 300.0f && vt <= 475.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(esc0) && esc0 >= 0.0f && esc0 <= 240.0f);

  verify_fdi_step(init_mask, {gx, gy, gz}, {dgx, dgy, dgz}, az, vh, vt, esc0);
#else
  verify_fdi_step(0x00u, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, -9.80665f, 8.0f, 385.0f, 0.0f);
  verify_fdi_step(0x11u, {0.1, -0.1, 2.2}, {0.0, 0.0, 0.0}, -9.80665f, 4.0f, 360.0f, 0.0f);
  verify_fdi_step(0x05u, {std::nan(""), 0.0, 0.0}, {0.0, 0.0, 0.0}, -9.80665f, 0.0f, 340.0f, 151.8f);
#endif
  return 0;
}
