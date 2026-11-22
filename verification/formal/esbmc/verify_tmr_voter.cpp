// SPDX-License-Identifier: Apache-2.0
// ESBMC v8.7 Harness 1: 400 Hz 3-IMU TMR FDIR Voter (DR-08 Section 3.2, --unwind 5)
// Verifies nominal precision and Byzantine Channel-2 isolation (NaN / +-Inf / +-DBL_MAX / subnormal)
// across all 3 axes with strictly finite voted gyro & accel outputs.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esbmc_compat.h"
#include "vertirta/core/tmr_voter.hpp"

namespace {

using vertirta::core::ImuSampleFrd;
using vertirta::core::TmrState;
using vertirta::core::TmrVoteOutput;
using vertirta::core::TmrVoter;
using vertirta::core::Vec3Frd;

void check_byzantine_channel2(double gx,
                              double gy,
                              double gz,
                              double ax,
                              double ay,
                              double az,
                              double byz_gx,
                              double byz_gy,
                              double byz_gz,
                              double byz_ax,
                              double byz_ay,
                              double byz_az) {
  TmrVoter voter{};
  ImuSampleFrd imus[3]{};

  // Healthy IMUs 0 and 1 agree within normal vibration floor
  imus[0].gyro_rad_s = {gx, gy, gz};
  imus[0].accel_m_s2 = {ax, ay, az};
  imus[0].valid = true;

  imus[1].gyro_rad_s = {gx + 0.004, gy - 0.004, gz + 0.002};
  imus[1].accel_m_s2 = {ax + 0.05, ay - 0.05, az + 0.05};
  imus[1].valid = true;

  // Channel 2 carries an unconstrained Byzantine double (NaN, +-Inf, +-DBL_MAX, step fault)
  imus[2].gyro_rad_s = {byz_gx, byz_gy, byz_gz};
  imus[2].accel_m_s2 = {byz_ax, byz_ay, byz_az};
  imus[2].valid = true;

  const Vec3Frd model_pred{gx, gy, gz};

  for (int tick = 0; tick < 3; ++tick) {
    const uint64_t t_us = static_cast<uint64_t>(tick + 1) * 2500ULL;
    imus[0].t_us = t_us;
    imus[1].t_us = t_us;
    imus[2].t_us = t_us;

    const TmrVoteOutput out = voter.step(imus, model_pred, true);

    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out.gyro_voted.x), "voted gyro.x must be finite");
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out.gyro_voted.y), "voted gyro.y must be finite");
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out.gyro_voted.z), "voted gyro.z must be finite");
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out.accel_voted.x), "voted accel.x must be finite");
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out.accel_voted.y), "voted accel.y must be finite");
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out.accel_voted.z), "voted accel.z must be finite");
    VERTIRTA_FORMAL_ASSERT(!std::isnan(out.max_chi2_residual) && out.max_chi2_residual >= 0.0f,
                           "chi2 residual must be non-NaN and non-negative");
    if (!vertirta_is_finite_f(out.max_chi2_residual)) {
      // A finite Byzantine spike exceeding sqrt(FLT_MAX * 2 * sigma^2) saturates float max_chi2_residual
      // to +Inf on tick 0 only, and MUST trigger immediate tick-0 isolation of Channel 2 into DUPLEX_2OF2.
      VERTIRTA_FORMAL_ASSERT(tick == 0, "chi2 +Inf saturation may only occur on tick 0 before isolation");
      VERTIRTA_FORMAL_ASSERT((out.health_mask & 0x04u) == 0u, "huge finite Byzantine IMU 2 must be isolated");
      VERTIRTA_FORMAL_ASSERT(out.state == TmrState::DUPLEX_2OF2, "must transition to DUPLEX_2OF2 on tick 0");
    }
    VERTIRTA_FORMAL_ASSERT((out.health_mask & ~0x07u) == 0u, "health_mask limited to 3 IMUs");
    VERTIRTA_FORMAL_ASSERT(
        static_cast<uint8_t>(out.state) <= static_cast<uint8_t>(TmrState::FAILSAFE_DIVERT),
        "TMR state within enum bounds");

    // Healthy pair consensus must prevent Byzantine IMU 2 from pulling voted gyro off-envelope
    const double err_x = std::fabs(out.gyro_voted.x - gx);
    const double err_y = std::fabs(out.gyro_voted.y - gy);
    const double err_z = std::fabs(out.gyro_voted.z - gz);
    VERTIRTA_FORMAL_ASSERT(err_x <= 0.25, "Byzantine IMU 2 must not corrupt voted gyro.x");
    VERTIRTA_FORMAL_ASSERT(err_y <= 0.25, "Byzantine IMU 2 must not corrupt voted gyro.y");
    VERTIRTA_FORMAL_ASSERT(err_z <= 0.25, "Byzantine IMU 2 must not corrupt voted gyro.z");
  }
}

}  // namespace

int main() {
#ifdef __ESBMC__
  const double gx = nondet_double();
  const double gy = nondet_double();
  const double gz = nondet_double();
  const double ax = nondet_double();
  const double ay = nondet_double();
  const double az = nondet_double();

  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(gx) && gx >= -4.0 && gx <= 4.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(gy) && gy >= -4.0 && gy <= 4.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(gz) && gz >= -4.0 && gz <= 4.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(ax) && ax >= -30.0 && ax <= 30.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(ay) && ay >= -30.0 && ay <= 30.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(az) && az >= -30.0 && az <= 30.0);

  // Completely unconstrained symbolic IEEE-754 double on Byzantine Channel 2
  const double byz_gx = nondet_double();
  const double byz_gy = nondet_double();
  const double byz_gz = nondet_double();
  const double byz_ax = nondet_double();
  const double byz_ay = nondet_double();
  const double byz_az = nondet_double();

  check_byzantine_channel2(gx, gy, gz, ax, ay, az, byz_gx, byz_gy, byz_gz, byz_ax, byz_ay, byz_az);
#else
  // Concrete smoke sweep over adversarial IEEE-754 Byzantine channel values
  const double nan_v = std::nan("");
  const double pos_inf = HUGE_VAL;
  const double neg_inf = -HUGE_VAL;
  const double dbl_max = 1.7976931348623157e+308;
  const double cases[11] = {
      nan_v, pos_inf, neg_inf, dbl_max, -dbl_max, 1.0e300, -1.0e300, 1.0e-310, 0.0, 5.5, -12.0};
  for (std::size_t i = 0; i < 11u; ++i) {
    const double b = cases[i];
    check_byzantine_channel2(0.12, -0.08, 0.25, 0.3, -0.2, -9.80665, b, -b, b, b, -b, b);
  }
#endif
  return 0;
}
