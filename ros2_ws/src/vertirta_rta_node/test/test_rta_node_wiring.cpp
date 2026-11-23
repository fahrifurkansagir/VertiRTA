// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta_px4_bridge/deadman_guard.hpp"
#include "vertirta_rta_node/lockfree_transport.hpp"
#include "vertirta_rta_node/rta_safety_core.hpp"

using namespace vertirta::ros;
using namespace vertirta::px4_bridge;

TEST_CASE("RtaSafetyCore runs real CbvfSafetyFilter on Seqlock intruder/wire hazards and subtracts DEM terrain AGL") {
  // 1) DEM terrain AGL subtraction (vs flat-ground -pos_ned[2] bug)
  const float ridge_elev = estimate_kartal_corridor_terrain_rel_m(/*east_m=*/1200.0, /*north_m=*/1500.0);
  REQUIRE(ridge_elev > 15.0f);
  REQUIRE_THAT(compute_terrain_agl_m(-50.0, ridge_elev), Catch::Matchers::WithinAbs(50.0f - ridge_elev, 1e-4));
  REQUIRE(compute_terrain_agl_m(-10.0, 25.0f) == 0.0f);

  // 2) Real CbvfSafetyFilter steering away from oncoming intruder in Seqlock snapshot
  FourThreadRtaTransport transport{};
  PerceptionHazardSnapshot haz{};
  haz.version = 1ULL;
  haz.stamp_ns = 1'000'000'000ULL;
  haz.terrain_elev_m = 20.0f;
  haz.hazard_count = 2U;
  // Intruder 10 m directly ahead (+North), approaching at -8 m/s
  haz.hazards[0].pos_ned[0] = 10.0;
  haz.hazards[0].pos_ned[1] = 0.0;
  haz.hazards[0].pos_ned[2] = -40.0;
  haz.hazards[0].vel_ned[0] = -8.0;
  haz.hazards[0].radius_h_m = 5.0f;
  haz.hazards[0].radius_z_m = 4.0f;
  haz.hazards[0].gamma = 1.4f;
  haz.hazards[0].kind = 2U;  // INTRUDER
  // Thin wire capsule 15 m east
  haz.hazards[1].pos_ned[0] = 0.0;
  haz.hazards[1].pos_ned[1] = 15.0;
  haz.hazards[1].pos_ned[2] = -40.0;
  haz.hazards[1].radius_h_m = 2.5f;
  haz.hazards[1].radius_z_m = 2.5f;
  haz.hazards[1].gamma = 1.4f;
  haz.hazards[1].kind = 1U;  // WIRE
  transport.map_to_rt_seqlock.publish(haz);

  TelemetryIngressFrame tel{};
  tel.stamp_ns = 1'020'000'000ULL;  // 20 ms fresh
  tel.pos_ned[0] = 0.0;
  tel.pos_ned[1] = 0.0;
  tel.pos_ned[2] = -40.0;
  tel.vel_ned[0] = 6.0f;  // Flying straight toward intruder!
  tel.flight_regime = 0U;

  RtaSafetyCore core{};
  // Nominal command asks to accelerate +3 m/s^2 directly into the intruder
  const auto cmd = core.step_100hz(tel, transport.map_to_rt_seqlock.read(), {3.0, 0.0, 0.0}, 0.0);
  REQUIRE(cmd.valid);
  REQUIRE_FALSE(cmd.perception_stale);
  // Real CbvfSafetyFilter must brake / deflect away from the +North intruder (< +3.0 m/s^2!)
  REQUIRE(cmd.accel_ned[0] < 1.5f);

  // 3) CEL latch on SAME_ARM_ACTIVE_SPIN uses terrain-subtracted AGL (40 m - 36 m ridge = 4 m AGL -> 0.8 m/s)
  PerceptionHazardSnapshot low_haz = haz;
  low_haz.terrain_elev_m = 36.0f;  // AGL = 40 - 36 = 4.0 m (< 5 m touchdown band!)
  tel.flight_regime = 3U;          // SAME_ARM_ACTIVE_SPIN
  tel.fault_bitmask = 0x11U;
  const auto cel_cmd = core.step_100hz(tel, low_haz, {0.0, 0.0, 0.0}, 0.0);
  REQUIRE(cel_cmd.cel_latched);
  REQUIRE(cel_cmd.control_mode == RtaSafetyCore::kModeCelControlledLanding);
  REQUIRE_THAT(cel_cmd.vz_limit_mps, Catch::Matchers::WithinAbs(0.8f, 1e-4));
  REQUIRE_THAT(cel_cmd.yawspeed_sp, Catch::Matchers::WithinAbs(-2.2f, 1e-4));
}

TEST_CASE("VertiRtaMode 100 ms rta_cmd staleness deadman guard engages fail-closed hold") {
  const std::array<float, 3> stale_accel = {4.5f, -3.2f, 1.1f};

  // Fresh command (50 ms old <= 100 ms): passes through untouched
  const auto fresh = apply_rta_deadman_fallback(
      /*now_ns=*/1'050'000'000ULL, /*last_rx_ns=*/1'000'000'000ULL, /*cel=*/false, stale_accel, 0.4f, 2.5f);
  REQUIRE_FALSE(fresh.deadman_triggered);
  REQUIRE_THAT(fresh.accel_ned[0], Catch::Matchers::WithinAbs(4.5f, 1e-5));

  // Stale command (150 ms old > 100 ms) in nominal flight: zero-accel level hold + vz capped <= 1.5
  const auto stale_nom = apply_rta_deadman_fallback(
      /*now_ns=*/1'150'000'000ULL, /*last_rx_ns=*/1'000'000'000ULL, /*cel=*/false, stale_accel, 0.4f, 2.5f);
  REQUIRE(stale_nom.deadman_triggered);
  REQUIRE(stale_nom.accel_ned[0] == 0.0f);
  REQUIRE(stale_nom.accel_ned[1] == 0.0f);
  REQUIRE(stale_nom.accel_ned[2] == 0.0f);
  REQUIRE(stale_nom.yawspeed_sp == 0.0f);
  REQUIRE(stale_nom.vz_limit_mps <= 1.5f);

  // Stale command during CEL latch: preserves -2.2 rad/s flat-spin descent safely
  const auto stale_cel = apply_rta_deadman_fallback(
      /*now_ns=*/1'150'000'000ULL, /*last_rx_ns=*/1'000'000'000ULL, /*cel=*/true, stale_accel, 0.0f, 2.5f);
  REQUIRE(stale_cel.deadman_triggered);
  REQUIRE_THAT(stale_cel.yawspeed_sp, Catch::Matchers::WithinAbs(-2.2f, 1e-5));
  REQUIRE(stale_cel.vz_limit_mps <= 1.5f);

  // Never received any rta_cmd (last_rx_ns == 0): fail-closed immediately
  REQUIRE(is_rta_cmd_stale(500'000'000ULL, 0ULL));
}
