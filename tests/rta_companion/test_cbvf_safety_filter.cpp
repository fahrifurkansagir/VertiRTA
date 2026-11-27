#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>

#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "zero_heap_guard.hpp"

using namespace vertirta::core;
using namespace vertirta::rta;
using vertirta::test::ScopedZeroHeapGuard;

namespace {

std::string fixture_path(const char* rel) {
  return std::string(VERTIRTA_SOURCE_DIR) + "/" + rel;
}

}  // namespace

TEST_CASE("Overhead intruder climb brakes, clutter inflates 1.5 m, BRT-outside uses u*_HJ, CEL latch is irreversible") {
  static_assert(!std::is_copy_constructible_v<CbvfSafetyFilter>);
  static_assert(!std::is_move_constructible_v<CbvfSafetyFilter>);

  MmapCbvf4DLut lut;
  Cel2DLut cel;
  REQUIRE(lut.load(fixture_path("tests/fixtures/quick_test.vrta_lut")));
  REQUIRE(cel.load(fixture_path("tests/fixtures/quick_test.vrta_cel2d")));

  CbvfSafetyFilter filter(&lut, &cel);
  const CapabilityEnvelope env_nom =
      CoaxEffectiveness::compute_envelope(0x00, 0, 0x07, 400.0f, 92.0f, 0.80f, 0.0f, 0.0f, 0.0f, 0);

  // 1) Overhead intruder sign fix (DR-2 Erratum #2 & Review Focus #2):
  // Ownship z_ned=-20 m climbing (v_D=-5 m/s -> v_up=+5 m/s) under overhead intruder at z_ned=-28 m (r=0.1 m).
  // Must map vz_rel < 0 (closing), clamp r >= 0.5 m, and command a_safe_ned.z > 0 (brake climb!).
  {
    const std::array<HazardPoint, 1> hazards = {{
        HazardPoint{.pos_ned = {0.1, 0.0, -28.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 2},
    }};
    const SafetyFilterOutput out = filter.step_100hz(
        {0.0, 0.0, -20.0},
        {0.0, 0.0, -5.0},
        {0.0, 0.0, -1.5},  // Nominal requests continuing climb!
        0.0,
        env_nom,
        hazards);
    REQUIRE(out.active_hazards == 1);
    REQUIRE(std::isfinite(out.a_safe_ned.z));
    REQUIRE(out.a_safe_ned.z > 0.0);  // Positive a_D brakes upward climb!
    REQUIRE(out.last_mapped_r_min >= 0.5);
    REQUIRE(out.last_mapped_vz_min < 0.0);  // Closing onto overhead intruder!
  }

  // 2) Opposing-gradient clutter inflation (+1.5 m when e_r1 . e_r2 < -0.5)
  {
    const std::array<HazardPoint, 1> single_haz = {{
        HazardPoint{.pos_ned = {0.0, 10.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
    }};
    const std::array<HazardPoint, 2> opp_haz = {{
        HazardPoint{.pos_ned = {0.0, +10.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
        HazardPoint{.pos_ned = {0.0, -10.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
    }};
    const auto out_single = filter.step_100hz(
        {0.0, 0.0, -20.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0, env_nom, single_haz);
    const auto out_opp = filter.step_100hz(
        {0.0, 0.0, -20.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.0, env_nom, opp_haz);
    REQUIRE_FALSE(out_single.clutter_inflated);
    REQUIRE(out_opp.clutter_inflated);
    REQUIRE(out_opp.min_cbvf_v <= out_single.min_cbvf_v - 1.49);
  }

  // 3) Degraded 2M-Cross v_cap_f speed-cap damping when v_h > v_cap_f(2)
  {
    const CapabilityEnvelope env_cross =
        CoaxEffectiveness::compute_envelope(0x21, 0, 0x07, 385.0f, 105.0f, 0.60f, 0.0f, 0.0f, 0.0f, 40);
    const std::array<HazardPoint, 1> far_haz = {{
        HazardPoint{.pos_ned = {100.0, 0.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
    }};
    const auto out_fast = filter.step_100hz(
        {0.0, 0.0, -30.0}, {19.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, 0.0, env_cross, far_haz);
    REQUIRE(out_fast.a_safe_ned.x < 0.0);  // Brakes down toward degraded v_cap_f!
  }

  // 4) Deep BRT violation (min_V < -0.25 m) -> deterministic closed-form u*_HJ fallback
  //    a) Finite-height obstacle at z_rel=0 (14 m/s closing at r=3 m): HJI escapes vertically (a_D < -5 m/s^2)
  //    b) Tall wall/CTR pillar (radius_z=40 m blocking vertical escape): HJI brakes horizontally (a_N < -5 m/s^2)
  {
    const std::array<HazardPoint, 1> deep_haz = {{
        HazardPoint{.pos_ned = {3.0, 0.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
    }};
    const auto out_brt_z = filter.step_100hz(
        {0.0, 0.0, -20.0}, {14.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, 0.0, env_nom, deep_haz);
    REQUIRE(out_brt_z.min_cbvf_v < -0.25);
    REQUIRE(out_brt_z.used_hj_fallback);
    REQUIRE(out_brt_z.a_safe_ned.z < -5.0);  // Vertical climb escape dominates over impossible 3m stop!

    const std::array<HazardPoint, 1> tall_wall = {{
        HazardPoint{.pos_ned = {3.0, 0.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .radius_z_m = 40.0, .kind = 3},
    }};
    const auto out_brt_r = filter.step_100hz(
        {0.0, 0.0, -20.0}, {14.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, 0.0, env_nom, tall_wall);
    REQUIRE(out_brt_r.min_cbvf_v < -0.25);
    REQUIRE(out_brt_r.used_hj_fallback);
    REQUIRE(out_brt_r.a_safe_ned.x < -5.0);  // Full horizontal braking away from tall +N wall!
  }

  // 5) 1,000 ticks under ScopedZeroHeapGuard -> strictly 0 heap allocations!
  {
    const std::array<HazardPoint, 4> four_haz = {{
        HazardPoint{.pos_ned = {18.0, 2.0, -25.0}, .vel_ned = {-1.0, 0.0, 0.0}, .kind = 0},
        HazardPoint{.pos_ned = {12.0, 14.0, -22.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 1},
        HazardPoint{.pos_ned = {-25.0, 10.0, -30.0}, .vel_ned = {2.0, -1.0, 0.0}, .kind = 2},
        HazardPoint{.pos_ned = {0.0, 40.0, -25.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 3},
    }};
    ScopedZeroHeapGuard guard;
    for (int k = 0; k < 1000; ++k) {
      const auto out = filter.step_100hz(
          {0.0, 0.0, -25.0},
          {8.0, 1.0, -0.5},
          {1.0, 0.2, 0.0},
          0.1,
          env_nom,
          four_haz);
      REQUIRE(std::isfinite(out.a_safe_ned.x));
      REQUIRE(std::isfinite(out.a_safe_ned.y));
      REQUIRE(std::isfinite(out.a_safe_ned.z));
    }
    REQUIRE(guard.alloc_count() == 0);
  }

  // 6) Irreversible SAME_ARM_ACTIVE_SPIN CEL latch + Cel2DLut braking
  {
    const CapabilityEnvelope env_cel =
        CoaxEffectiveness::compute_envelope(0x11, 0, 0x07, 350.0f, 140.0f, 0.30f, -2.2f, -2.1f, 0.0f, 45);
    const auto out_cel = filter.step_100hz(
        {0.0, 0.0, -4.0},
        {0.0, 0.0, +4.5},  // Excessive 4.5 m/s sink at 4 m AGL!
        {0.0, 0.0, +1.0},
        0.0,
        env_cel,
        {});
    REQUIRE(out_cel.cel_latched);
    REQUIRE(out_cel.a_safe_ned.z < -0.5);  // Strong upward flare braking!

    // Flicker back to NOMINAL envelope -> CEL latch MUST remain irreversible!
    const auto out_flicker = filter.step_100hz(
        {0.0, 0.0, -15.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, 0.0, env_nom, {});
    REQUIRE(out_flicker.cel_latched);
    REQUIRE(filter.is_cel_latched());
  }
}

TEST_CASE("CbvfSafetyFilter fails closed on NaN/Inf in ego state, nominal cmd, envelope, hazards, and CEL",
          "[cbvf][nan_hardening]") {
  MmapCbvf4DLut lut;
  Cel2DLut cel;
  REQUIRE(lut.load(fixture_path("tests/fixtures/quick_test.vrta_lut")));
  REQUIRE(cel.load(fixture_path("tests/fixtures/quick_test.vrta_cel2d")));

  CbvfSafetyFilter filter(&lut, &cel);
  const CapabilityEnvelope env_nom =
      CoaxEffectiveness::compute_envelope(0x00, 0, 0x07, 400.0f, 92.0f, 0.80f, 0.0f, 0.0f, 0.0f, 0);

  constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
  constexpr double kInf = std::numeric_limits<double>::infinity();
  constexpr float kNaNf = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInff = std::numeric_limits<float>::infinity();

  auto require_all_finite = [](const SafetyFilterOutput& o) {
    REQUIRE(std::isfinite(o.a_safe_ned.x));
    REQUIRE(std::isfinite(o.a_safe_ned.y));
    REQUIRE(std::isfinite(o.a_safe_ned.z));
    REQUIRE(std::isfinite(o.yaw_rate_safe));
    REQUIRE(std::isfinite(o.slack));
    REQUIRE(std::isfinite(o.min_cbvf_v));
    REQUIRE(std::isfinite(o.cel_v_2d));
  };

  // 1) Non-finite ego pos_ned or vel_ned -> immediate fail-closed safe hover fallback.
  {
    const auto o_pos_nan = filter.step_100hz(
        {kNaN, 0.0, -20.0}, {5.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, 0.2, env_nom, {});
    require_all_finite(o_pos_nan);
    REQUIRE(o_pos_nan.used_hj_fallback);
    REQUIRE(o_pos_nan.a_safe_ned.x == 0.0);
    REQUIRE(o_pos_nan.a_safe_ned.y == 0.0);
    REQUIRE(o_pos_nan.a_safe_ned.z == 0.0);
    REQUIRE(o_pos_nan.yaw_rate_safe == 0.0);

    const auto o_vel_inf = filter.step_100hz(
        {0.0, 0.0, -20.0}, {0.0, -kInf, 0.0}, {1.0, 0.0, 0.0}, 0.2, env_nom, {});
    require_all_finite(o_vel_inf);
    REQUIRE(o_vel_inf.used_hj_fallback);
    REQUIRE(o_vel_inf.a_safe_ned.x == 0.0);
    REQUIRE(o_vel_inf.a_safe_ned.y == 0.0);
    REQUIRE(o_vel_inf.a_safe_ned.z == 0.0);
    REQUIRE(o_vel_inf.yaw_rate_safe == 0.0);
  }

  // 2) Non-finite a_nom_ned & yaw_rate_nom sanitized to 0 before QP -> 100% finite outputs.
  {
    const std::array<HazardPoint, 1> haz = {{
        HazardPoint{.pos_ned = {15.0, 0.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
    }};
    const auto o_nom = filter.step_100hz(
        {0.0, 0.0, -20.0}, {4.0, 0.0, 0.0}, {kNaN, kInf, -kInf}, kNaN, env_nom, haz);
    require_all_finite(o_nom);
    REQUIRE(std::abs(o_nom.yaw_rate_safe) <= 1.5);
  }

  // 3) Corrupted CapabilityEnvelope fields (std::clamp(NaN, lo, hi) trap!) -> conservative finite bounds.
  {
    CapabilityEnvelope env_bad = env_nom;
    env_bad.a_xy_max = kNaNf;
    env_bad.tilt_max_rad = kNaNf;
    env_bad.t_max_n = kInff;
    env_bad.v_term_v = kNaNf;
    env_bad.soc = kNaNf;
    env_bad.spin_rate_meas = kNaNf;
    env_bad.spin_rate_ref = kNaNf;

    const std::array<HazardPoint, 1> haz = {{
        HazardPoint{.pos_ned = {12.0, 0.0, -20.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
    }};
    const auto o_env = filter.step_100hz(
        {0.0, 0.0, -20.0}, {3.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, 0.1, env_bad, haz);
    require_all_finite(o_env);
    REQUIRE(std::hypot(o_env.a_safe_ned.x, o_env.a_safe_ned.y) <= 0.55);
  }

  // 4) Non-finite fields inside HazardPoint (pos_ned NaN skipped; vel/accel/radius/gamma NaN sanitized).
  {
    const std::array<HazardPoint, 2> bad_haz = {{
        HazardPoint{
            .pos_ned = {kNaN, 0.0, -20.0},
            .vel_ned = {0.0, 0.0, 0.0},
            .accel_ned = {0.0, 0.0, 0.0},
            .radius_h_m = 0.0,
            .radius_z_m = 0.0,
            .gamma = 1.2,
            .kind = 0,
        },
        HazardPoint{
            .pos_ned = {6.0, 0.0, -20.0},
            .vel_ned = {kNaN, kInf, 0.0},
            .accel_ned = {kNaN, 0.0, -kInf},
            .radius_h_m = kNaN,
            .radius_z_m = kInf,
            .gamma = kNaN,
            .kind = 2,
        },
    }};
    const auto o_haz = filter.step_100hz(
        {0.0, 0.0, -20.0}, {6.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, 0.0, env_nom, bad_haz);
    require_all_finite(o_haz);
    REQUIRE(o_haz.active_hazards == 1u);
    REQUIRE(o_haz.a_safe_ned.x < 1.0);  // Valid-position obstacle still actively avoided!
  }

  // 5) Latched CEL mode with NaN a_nom_ned & NaN env fields (and NaN pos_ned), verified zero-heap!
  {
    CapabilityEnvelope env_cel =
        CoaxEffectiveness::compute_envelope(0x11, 0, 0x07, 350.0f, 140.0f, 0.30f, -2.2f, -2.1f, 0.0f, 45);
    env_cel.v_term_v = kNaNf;
    env_cel.t_max_n = kNaNf;
    env_cel.spin_rate_meas = kNaNf;
    env_cel.spin_rate_ref = kNaNf;

    ScopedZeroHeapGuard guard;
    const auto o_cel = filter.step_100hz(
        {0.0, 0.0, -4.0}, {1.0, -1.0, +3.5}, {kNaN, kNaN, kNaN}, kNaN, env_cel, {});
    require_all_finite(o_cel);
    REQUIRE(o_cel.cel_latched);
    REQUIRE(o_cel.a_safe_ned.x == 0.0);
    REQUIRE(o_cel.a_safe_ned.y == 0.0);
    REQUIRE(o_cel.yaw_rate_safe == 0.0);

    const auto o_cel_nan_pos = filter.step_100hz(
        {0.0, 0.0, kNaN}, {0.0, 0.0, 2.0}, {0.0, 0.0, 0.0}, 0.0, env_cel, {});
    require_all_finite(o_cel_nan_pos);
    REQUIRE(o_cel_nan_pos.cel_latched);
    REQUIRE(o_cel_nan_pos.used_hj_fallback);
    REQUIRE(o_cel_nan_pos.a_safe_ned.z == 0.0);
    REQUIRE(guard.alloc_count() == 0u);
  }
}
