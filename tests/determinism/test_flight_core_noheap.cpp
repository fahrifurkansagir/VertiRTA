// Freestanding zero-heap runtime test exercising libvertirta_flight_core
// (frames, geodesy, and 400 Hz 3-IMU TMR FDIR voter) with ZERO heap allocations
// and ZERO Catch2/iostream/STL container dependencies.

#include <cmath>
#include <cstdint>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/core/tmr_voter.hpp"

namespace {

using vertirta::core::GeodeticWgs84;
using vertirta::core::ImuSampleFrd;
using vertirta::core::QuatPx4FrdNed;
using vertirta::core::TmrConfig;
using vertirta::core::TmrState;
using vertirta::core::TmrVoteOutput;
using vertirta::core::TmrVoter;
using vertirta::core::Vec3Enu;
using vertirta::core::Vec3Frd;

bool near(double a, double b, double tol) noexcept {
  return std::fabs(a - b) <= tol;
}

int test_frames_and_geodesy() noexcept {
  const Vec3Enu enu{120.5, -340.25, 45.0};
  const Vec3Enu round_enu = vertirta::core::ned_to_enu(vertirta::core::enu_to_ned(enu));
  if (!near(enu.x, round_enu.x, 1e-12) || !near(enu.y, round_enu.y, 1e-12) ||
      !near(enu.z, round_enu.z, 1e-12)) {
    return 1;
  }

  const QuatPx4FrdNed q_in{0.9238795325112867, 0.1, -0.2, 0.3249196962329063};
  const QuatPx4FrdNed q_out = vertirta::core::ros_to_px4_quat(vertirta::core::px4_to_ros_quat(q_in));
  if (!near(q_in.w, q_out.w, 1e-12) || !near(q_in.x, q_out.x, 1e-12) ||
      !near(q_in.y, q_out.y, 1e-12) || !near(q_in.z, q_out.z, 1e-12)) {
    return 2;
  }

  const double n_kartal = vertirta::core::egm2008_undulation_m(40.8860, 29.1880);
  if (!(n_kartal > 36.4 && n_kartal < 37.0)) {
    return 3;
  }

  const GeodeticWgs84 g0{40.9125, 29.1460, 120.0};
  const Vec3Enu local = vertirta::core::geodetic_to_local_enu(g0);
  const GeodeticWgs84 g1 = vertirta::core::local_enu_to_geodetic(local);
  if (!near(g0.lat_deg, g1.lat_deg, 1e-7) || !near(g0.lon_deg, g1.lon_deg, 1e-7) ||
      !near(g0.h_hae_m, g1.h_hae_m, 1e-3)) {
    return 4;
  }
  return 0;
}

int test_tmr_400hz_isolation_and_nan_guard() noexcept {
  TmrVoter voter{TmrConfig{}};
  const Vec3Frd nominal_gyro{0.01, -0.02, 0.005};
  const Vec3Frd nominal_accel{0.0, 0.0, -9.80665};

  TmrVoteOutput out{};
  for (uint64_t tick = 0; tick < 20; ++tick) {
    ImuSampleFrd imus[3]{
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
    };
    out = voter.step(imus, nominal_gyro, true);
  }
  if (out.state != TmrState::TMR_3OF3 || out.health_mask != 0x07U) {
    return 5;
  }

  // Inject persistent step fault on IMU 2; verify isolation within <= 20 ms (8 ticks)
  for (uint64_t tick = 20; tick < 36; ++tick) {
    ImuSampleFrd imus[3]{
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {{1.25, -0.80, 0.50}, nominal_accel, tick * 2500ULL, true},
    };
    out = voter.step(imus, nominal_gyro, true);
  }
  if (out.state != TmrState::DUPLEX_2OF2 || out.health_mask != 0x03U || out.isolation_latency_ms > 20U) {
    return 6;
  }

  // NaN guard: channel 1 goes NaN -> voted output must stay strictly finite
  ImuSampleFrd nan_imus[3]{
      {nominal_gyro, nominal_accel, 90000ULL, true},
      {{NAN, 0.0, 0.0}, nominal_accel, 90000ULL, true},
      {{1.25, -0.80, 0.50}, nominal_accel, 90000ULL, true},
  };
  out = voter.step(nan_imus, nominal_gyro, true);
  if (!out.gyro_voted.is_finite() || !out.accel_voted.is_finite()) {
    return 7;
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = test_frames_and_geodesy(); rc != 0) {
    return rc;
  }
  if (const int rc = test_tmr_400hz_isolation_and_nan_guard(); rc != 0) {
    return rc;
  }
  return 0;
}
