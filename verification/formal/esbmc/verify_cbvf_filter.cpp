// SPDX-License-Identifier: Apache-2.0
// ESBMC v8.7 Harness 6: 100 Hz HJI-CBVF Safety Filter & 4D LUT Guard (--unwind 12)
// Verifies DR-2 Erratum #4 (regime_idx < 4 & non-finite LUT guard), non-finite filter fallback,
// and irreversible SAME_ARM_ACTIVE_SPIN CEL latch persistence.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "esbmc_compat.h"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace {

using vertirta::core::CapabilityEnvelope;
using vertirta::core::FlightRegime;
using vertirta::core::Vec3Ned;
using vertirta::rta::CbvfQueryResult;
using vertirta::rta::CbvfSafetyFilter;
using vertirta::rta::HazardPoint;
using vertirta::rta::MmapCbvf4DLut;
using vertirta::rta::SafetyFilterOutput;

void verify_lut_guards(uint8_t bad_regime, double r, double vr, double z, double vz) {
  MmapCbvf4DLut lut{};
  const CbvfQueryResult q_bad = lut.eval(bad_regime, r, vr, z, vz);
  VERTIRTA_FORMAL_ASSERT(!q_bad.valid, "unloaded / OOB regime_idx >= 4 LUT eval must return invalid");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(q_bad.value_m), "LUT fallback value must be finite");
}

void verify_filter_fallback_and_cel_latch(double ax_nom, double ay_nom, double az_nom) {
  CbvfSafetyFilter filter{nullptr, nullptr};
  CapabilityEnvelope env{};
  env.regime = FlightRegime::NOMINAL;
  env.v_term_v = 390.0f;
  env.soc = 0.70f;
  env.a_xy_max = 6.87f;
  env.tilt_max_rad = 0.45f;
  env.t_max_n = 5886.0f;

  HazardPoint haz[2]{};
  haz[0].pos_ned = {20.0, 0.0, -35.0};
  haz[0].radius_h_m = 4.0;
  haz[0].radius_z_m = 3.0;
  haz[0].kind = 0u;

  haz[1].pos_ned = {0.0, 25.0, -35.0};
  haz[1].radius_h_m = 4.0;
  haz[1].radius_z_m = 3.0;
  haz[1].kind = 1u;

#ifndef VERTIRTA_IS_ESBMC
  const SafetyFilterOutput out1 = filter.step_100hz(
      Vec3Ned{0.0, 0.0, -35.0},
      Vec3Ned{10.0, 0.0, 0.0},
      Vec3Ned{ax_nom, ay_nom, az_nom},
      0.0,
      env,
      std::span<const HazardPoint>{haz, 2u});

  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out1.a_safe_ned.x), "a_safe_ned.x must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out1.a_safe_ned.y), "a_safe_ned.y must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out1.a_safe_ned.z), "a_safe_ned.z must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out1.yaw_rate_safe), "yaw_rate_safe must be finite");
  VERTIRTA_FORMAL_ASSERT(!out1.cel_latched, "nominal tick must not latch CEL");
#endif

  // Trigger SAME_ARM_ACTIVE_SPIN -> must latch CEL irreversibly with finite closed-form descent
  env.regime = FlightRegime::SAME_ARM_ACTIVE_SPIN;
  const SafetyFilterOutput out2 = filter.step_100hz(
      Vec3Ned{0.0, 0.0, -35.0},
      Vec3Ned{2.0, 0.0, 1.5},
      Vec3Ned{ax_nom, ay_nom, az_nom},
      0.0,
      env,
      std::span<const HazardPoint>{haz, 2u});
  VERTIRTA_FORMAL_ASSERT(out2.cel_latched && filter.is_cel_latched(),
                         "SAME_ARM_ACTIVE_SPIN must latch CEL");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out2.a_safe_ned.x), "CEL a_safe_ned.x must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out2.a_safe_ned.y), "CEL a_safe_ned.y must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out2.a_safe_ned.z), "CEL a_safe_ned.z must be finite");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out2.yaw_rate_safe), "CEL yaw_rate_safe must be finite");

  // Even if caller reverts regime to NOMINAL, CEL latch must remain irreversibly armed until reset()
  env.regime = FlightRegime::NOMINAL;
  const SafetyFilterOutput out3 = filter.step_100hz(
      Vec3Ned{0.0, 0.0, -33.0},
      Vec3Ned{1.0, 0.0, 1.2},
      Vec3Ned{ax_nom, ay_nom, az_nom},
      0.0,
      env,
      std::span<const HazardPoint>{haz, 2u});
  VERTIRTA_FORMAL_ASSERT(out3.cel_latched && filter.is_cel_latched(),
                         "CEL latch must remain irreversible until reset()");
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(out3.a_safe_ned.z), "CEL a_safe_ned.z must be finite");

  filter.reset();
  VERTIRTA_FORMAL_ASSERT(!filter.is_cel_latched(), "reset() must clear CEL latch");
}

}  // namespace

int main() {
#ifdef VERTIRTA_IS_ESBMC
  const uint8_t reg = nondet_uchar();
  VERTIRTA_FORMAL_ASSUME(reg >= 4u);
  const float r_f = nondet_float();
  verify_lut_guards(reg, static_cast<double>(r_f), 0.0, 10.0, 0.0);

  const float ax_f = nondet_float();
  const float ay_f = nondet_float();
  const float az_f = nondet_float();
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(ax_f) && ax_f >= -10.0f && ax_f <= 10.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(ay_f) && ay_f >= -10.0f && ay_f <= 10.0f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(az_f) && az_f >= -10.0f && az_f <= 10.0f);
  verify_filter_fallback_and_cel_latch(
      static_cast<double>(ax_f), static_cast<double>(ay_f), static_cast<double>(az_f));
#else
  const double nan_v = std::nan("");
  verify_lut_guards(4u, 10.0, 0.0, 10.0, 0.0);
  verify_lut_guards(255u, nan_v, HUGE_VAL, -HUGE_VAL, nan_v);
  verify_filter_fallback_and_cel_latch(2.0, -1.5, 0.5);
  verify_filter_fallback_and_cel_latch(nan_v, 0.0, 0.0);
#endif
  return 0;
}
