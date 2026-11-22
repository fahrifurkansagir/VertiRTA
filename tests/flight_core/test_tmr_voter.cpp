#include <cmath>
#include <cstdint>
#include <limits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/core/tmr_voter.hpp"

using Catch::Matchers::WithinAbs;
using namespace vertirta::core;

namespace {

ImuSampleFrd make_imu(uint64_t tick, double gx, double gy, double gz,
                      double ax = 0.0, double ay = 0.0, double az = -9.80665) {
  return ImuSampleFrd{{gx, gy, gz}, {ax, ay, az}, tick * 2500ULL, true};
}

}  // namespace

TEST_CASE("TMR MVS masks tick-0 hardover and isolates IMU0 bias <= 20 ms via chi2_24 > 72.23") {
  TmrVoter voter;
  bool isolated = false;
  uint64_t isolated_tick = 999;
  float chi2_at_isolation = 0.0f;

  for (uint64_t k = 0; k < 20; ++k) {
    ImuSampleFrd imus[3] = {
        make_imu(k, 0.02, -0.01, 0.005),
        make_imu(k, 0.02, -0.01, 0.005),
        make_imu(k, 0.02, -0.01, 0.005),
    };
    imus[0].gyro_rad_s.x += 0.075;  // Slightly above 0.0637 rad/s MDB -> isolates inside 8 ticks (20 ms)
    const TmrVoteOutput out = voter.step(imus, {0.02, -0.01, 0.005}, true);
    if (k == 0) {
      // MVS passively masks single-channel hardover with zero tick-0 error
      REQUIRE_THAT(out.gyro_voted.x, WithinAbs(0.02, 1e-6));
      REQUIRE(out.state == TmrState::TMR_3OF3);
    }
    if (!isolated && out.state == TmrState::DUPLEX_2OF2) {
      isolated = true;
      isolated_tick = k;
      chi2_at_isolation = out.max_chi2_residual;
    }
    if (k == 7) {
      REQUIRE(out.state == TmrState::DUPLEX_2OF2);
      REQUIRE(out.health_mask == 0x06);  // Bit 0 cleared, IMUs 1 & 2 healthy
      REQUIRE(out.isolation_latency_ms <= 20);
      REQUIRE(out.max_chi2_residual < 1.0f);  // Active surviving pair (1,2) is clean post-isolation
    }
  }
  REQUIRE(isolated);
  REQUIRE(isolated_tick <= 7);
  REQUIRE(chi2_at_isolation > 72.23f);
}

TEST_CASE("TMR rejects common-mode 15 m/s gust maneuvers with zero false alarms") {
  TmrVoter voter;
  for (uint64_t k = 0; k < 200; ++k) {
    const double t = static_cast<double>(k) * 0.0025;
    const double wx = 1.5 * std::sin(12.0 * t);
    const double wy = -1.2 * std::cos(9.0 * t);
    const double wz = 0.6 * std::sin(5.0 * t);
    const double az = -9.80665 - 3.5 * std::sin(8.0 * t);
    ImuSampleFrd imus[3] = {
        make_imu(k, wx + 0.001, wy - 0.001, wz, 0.1, -0.1, az),
        make_imu(k, wx - 0.001, wy + 0.001, wz, 0.1, -0.1, az),
        make_imu(k, wx, wy, wz + 0.001, 0.1, -0.1, az),
    };
    const TmrVoteOutput out = voter.step(imus, {wx, wy, wz}, true);
    REQUIRE(out.state == TmrState::TMR_3OF3);
    REQUIRE(out.health_mask == 0x07);
    REQUIRE(out.max_chi2_residual < 72.23f);
  }
}

TEST_CASE("TMR latches NaN/Inf immediately, never unlatches, and degrades to SIMPLEX or FAILSAFE_DIVERT") {
  // Part A: NaN on IMU2 -> immediate bit-2 clear, monotonic (stays DUPLEX after finite restored)
  {
    TmrVoter voter;
    ImuSampleFrd imus[3] = {
        make_imu(0, 0.0, 0.0, 0.0),
        make_imu(0, 0.0, 0.0, 0.0),
        make_imu(0, std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0),
    };
    const TmrVoteOutput out0 = voter.step(imus, {0.0, 0.0, 0.0}, true);
    REQUIRE(out0.state == TmrState::DUPLEX_2OF2);
    REQUIRE(out0.health_mask == 0x03);
    REQUIRE(out0.gyro_voted.is_finite());

    // Restore finite IMU2 -> must NEVER auto-unlatch
    imus[2] = make_imu(1, 0.0, 0.0, 0.0);
    const TmrVoteOutput out1 = voter.step(imus, {0.0, 0.0, 0.0}, true);
    REQUIRE(out1.state == TmrState::DUPLEX_2OF2);
    REQUIRE(out1.health_mask == 0x03);

    // Inject second fault on IMU1 WITHOUT analytical model -> FAILSAFE_DIVERT
    TmrVoteOutput last{};
    for (uint64_t k = 2; k < 15; ++k) {
      ImuSampleFrd s[3] = {
          make_imu(k, 0.0, 0.0, 0.0),
          make_imu(k, 0.25, 0.0, 0.0),  // Large bias on IMU1
          make_imu(k, 0.0, 0.0, 0.0),
      };
      last = voter.step(s, {0.0, 0.0, 0.0}, false);
    }
    REQUIRE(last.state == TmrState::FAILSAFE_DIVERT);
  }

  // Part B: Second fault WITH clear analytical 3-sigma vs 6-sigma tie-breaker -> SIMPLEX_ANALYTICAL
  {
    TmrVoter voter;
    ImuSampleFrd imus[3] = {
        make_imu(0, 0.0, 0.0, 0.0),
        make_imu(0, 0.0, 0.0, 0.0),
        make_imu(0, 0.0, 0.0, std::numeric_limits<double>::infinity()),
    };
    (void)voter.step(imus, {0.0, 0.0, 0.0}, true);  // Drops IMU2 -> DUPLEX (IMU0 & IMU1)

    TmrVoteOutput last{};
    for (uint64_t k = 1; k < 15; ++k) {
      ImuSampleFrd s[3] = {
          make_imu(k, 0.01, 0.0, 0.0),   // Within 3*sigma (0.045 rad/s) of model (0.0)
          make_imu(k, 0.25, 0.0, 0.0),   // > 6*sigma (0.090 rad/s) away from model
          make_imu(k, 0.0, 0.0, 0.0),
      };
      last = voter.step(s, {0.0, 0.0, 0.0}, true);
    }
    REQUIRE(last.state == TmrState::SIMPLEX_ANALYTICAL);
    REQUIRE(last.health_mask == 0x01);  // Only IMU0 remains healthy
    REQUIRE_THAT(last.gyro_voted.x, WithinAbs(0.01, 1e-6));
  }
}
