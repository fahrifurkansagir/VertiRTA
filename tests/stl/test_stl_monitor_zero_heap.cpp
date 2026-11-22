#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "vertirta/stl/stl_monitor.hpp"
#include "zero_heap_guard.hpp"

namespace {

using vertirta::copilot::BoundedStlFormula;
using vertirta::copilot::EnuPoint;
using vertirta::copilot::StlAstNode;
using vertirta::copilot::StlNodeKind;
using vertirta::copilot::ZoneId;
using vertirta::stl::FixedMonotonicWedge;
using vertirta::stl::HalfSpace2D;
using vertirta::stl::OnlineBoundedUntil;
using vertirta::stl::RosiBound;
using vertirta::stl::StlMonitorEngine;
using vertirta::stl::StlSignalSample;

[[nodiscard]] float textbook_until_at(const std::vector<float>& r1,
                                      const std::vector<float>& r2,
                                      uint32_t t,
                                      uint32_t a,
                                      uint32_t b) {
  float best = -std::numeric_limits<float>::infinity();
  for (uint32_t k = t + a; k <= t + b; ++k) {
    float min_r1 = +std::numeric_limits<float>::infinity();
    for (uint32_t j = t; j <= k; ++j) {
      min_r1 = std::min(min_r1, r1[j]);
    }
    best = std::max(best, std::min(r2[k], min_r1));
  }
  return best;
}

[[nodiscard]] BoundedStlFormula build_kartal_diversion_formula() noexcept {
  BoundedStlFormula f{};
  f.target_enu = EnuPoint{0.0, 0.0, 0.0};
  f.includes_ltfj_exclusion = true;
  f.includes_ltbx_exclusion = true;
  f.until_target_radius_m = 15.0f;
  f.horizon_s = 100U;

  auto add = [&](StlAstNode n) -> uint8_t {
    const uint8_t idx = f.node_count++;
    f.nodes[idx] = n;
    return idx;
  };

  // Clause 1: ALWAYS[0, 77](agl_ge(35) & outside(LTFJ) & outside(LTBX) & speed_le(22))
  const uint8_t p_agl = add(StlAstNode{StlNodeKind::PRED_AGL_GE, 0xFFU, 0xFFU, 0, 0, 35.0f, ZoneId::LTFJ_CTR});
  const uint8_t p_ltfj = add(StlAstNode{StlNodeKind::PRED_OUTSIDE_ZONE, 0xFFU, 0xFFU, 0, 0, 0.0f, ZoneId::LTFJ_CTR});
  const uint8_t p_ltbx = add(StlAstNode{StlNodeKind::PRED_OUTSIDE_ZONE, 0xFFU, 0xFFU, 0, 0, 0.0f, ZoneId::LTBX_CTR});
  const uint8_t p_spd = add(StlAstNode{StlNodeKind::PRED_SPEED_LE, 0xFFU, 0xFFU, 0, 0, 22.0f, ZoneId::LTFJ_CTR});
  const uint8_t a1 = add(StlAstNode{StlNodeKind::OP_AND, p_agl, p_ltfj, 0, 0, 0.0f, ZoneId::LTFJ_CTR});
  const uint8_t a2 = add(StlAstNode{StlNodeKind::OP_AND, p_ltbx, p_spd, 0, 0, 0.0f, ZoneId::LTFJ_CTR});
  const uint8_t body1 = add(StlAstNode{StlNodeKind::OP_AND, a1, a2, 0, 0, 0.0f, ZoneId::LTFJ_CTR});
  const uint8_t c1 = add(StlAstNode{StlNodeKind::OP_ALWAYS, body1, 0xFFU, 0U, 77U, 0.0f, ZoneId::LTFJ_CTR});

  // Clause 2: ALWAYS[0, 100](soc_ge(12))
  const uint8_t p_soc = add(StlAstNode{StlNodeKind::PRED_SOC_GE, 0xFFU, 0xFFU, 0, 0, 12.0f, ZoneId::LTFJ_CTR});
  const uint8_t c2 = add(StlAstNode{StlNodeKind::OP_ALWAYS, p_soc, 0xFFU, 0U, 100U, 0.0f, ZoneId::LTFJ_CTR});

  // Clause 3: UNTIL[0, 100](agl_ge(35), reach(Kartal_Sahil, 15))
  const uint8_t p_agl2 = add(StlAstNode{StlNodeKind::PRED_AGL_GE, 0xFFU, 0xFFU, 0, 0, 35.0f, ZoneId::LTFJ_CTR});
  const uint8_t p_reach = add(StlAstNode{StlNodeKind::PRED_REACH_TARGET, 0xFFU, 0xFFU, 0, 0, 15.0f, ZoneId::LTFJ_CTR});
  const uint8_t c3 = add(StlAstNode{StlNodeKind::OP_UNTIL, p_agl2, p_reach, 0U, 100U, 0.0f, ZoneId::LTFJ_CTR});

  const uint8_t top12 = add(StlAstNode{StlNodeKind::OP_AND, c1, c2, 0, 0, 0.0f, ZoneId::LTFJ_CTR});
  f.root_index = add(StlAstNode{StlNodeKind::OP_AND, top12, c3, 0, 0, 0.0f, ZoneId::LTFJ_CTR});
  return f;
}

}  // namespace

TEST_CASE("FixedMonotonicWedge evicts expired head BEFORE tail push at Cap=601, span=600, tau=601") {
  FixedMonotonicWedge<601U, true> min_w;
  for (uint32_t tau = 0U; tau <= 600U; ++tau) {
    REQUIRE(min_w.push_and_query(tau, static_cast<float>(tau) - 5.0f, 600U) ==
            Catch::Approx(-5.0f));
  }
  REQUIRE(min_w.size() == 601U);

  // Critical wraparound tick tau=601: index 0 (-5.0) must expire BEFORE tail pushes 601.
  REQUIRE(min_w.push_and_query(601U, 596.0f, 600U) == Catch::Approx(-4.0f));
  REQUIRE(min_w.size() == 601U);

  // Negative safety violation at tau=602 must immediately dominate and survive.
  REQUIRE(min_w.push_and_query(602U, -9.5f, 600U) == Catch::Approx(-9.5f));

  // NaN input must fail closed to -inf.
  const float nan_out =
      min_w.push_and_query(603U, std::numeric_limits<float>::quiet_NaN(), 600U);
  REQUIRE(std::isinf(nan_out));
  REQUIRE(nan_out < 0.0f);

  // Verify max wedge wraparound symmetrically on strictly decreasing inputs.
  FixedMonotonicWedge<601U, false> max_w;
  for (uint32_t tau = 0U; tau <= 600U; ++tau) {
    REQUIRE(max_w.push_and_query(tau, 1000.0f - static_cast<float>(tau), 600U) ==
            Catch::Approx(1000.0f));
  }
  REQUIRE(max_w.push_and_query(601U, 399.0f, 600U) == Catch::Approx(999.0f));
}

TEST_CASE("OnlineBoundedUntil matches textbook discrete STL over 200 seeded traces and triggers instant RoSI violation") {
  std::mt19937 rng(42U);
  std::uniform_real_distribution<float> dist(-20.0f, 20.0f);

  float max_abs_err = 0.0f;
  for (int trial = 0; trial < 200; ++trial) {
    const uint32_t b = static_cast<uint32_t>(5 + (trial % 45));
    const uint32_t a = static_cast<uint32_t>(trial % (b + 1U));
    const uint32_t trace_len = b + 25U;

    std::vector<float> r1(trace_len);
    std::vector<float> r2(trace_len);
    for (uint32_t t = 0U; t < trace_len; ++t) {
      r1[t] = dist(rng);
      r2[t] = dist(rng);
    }

    OnlineBoundedUntil<601U> op;
    op.configure(a, b);
    for (uint32_t tau = 0U; tau < trace_len; ++tau) {
      float exact_rho = 0.0f;
      RosiBound rosi{};
      const bool ready = op.step(tau, r1[tau], r2[tau], exact_rho, rosi);
      if (tau < b) {
        REQUIRE_FALSE(ready);
      } else {
        REQUIRE(ready);
        const float expected = textbook_until_at(r1, r2, tau - b, a, b);
        max_abs_err = std::max(max_abs_err, std::abs(exact_rho - expected));
      }
    }
  }
  REQUIRE(max_abs_err == Catch::Approx(0.0f).margin(1e-6f));

  // Instant RoSI violation detection without waiting b=100 ticks:
  OnlineBoundedUntil<601U> live_u;
  live_u.configure(0U, 100U);
  float rho = 0.0f;
  RosiBound rosi{};
  for (uint32_t t = 0U; t <= 3U; ++t) {
    const float r1_v = (t == 3U) ? -2.0f : 5.0f;
    const bool ready = live_u.step(t, r1_v, -10.0f, rho, rosi);
    REQUIRE_FALSE(ready);
  }
  REQUIRE(rosi.definitively_violated());
  REQUIRE(rosi.sup == Catch::Approx(-2.0f));
}

TEST_CASE("StlMonitorEngine runs with 0 heap allocations and <1 ms WCET on calibrated Kartal_Sahil diversion") {
  StlMonitorEngine engine;
  const std::array<HalfSpace2D, 3> ltfj_west = {{
      {-0.985235, -0.171209, -703.66},
      {-0.923880, -0.382683, -549.88},
      {-0.877927, +0.478795, +1251.06},
  }};
  const std::array<HalfSpace2D, 3> ltbx_south = {{
      {-0.707107, -0.707107, -1127.69},
      {-0.382683, -0.923880, -3078.10},
      {0.0, -1.0, -3127.31},
  }};
  engine.set_zone_halfspaces(ZoneId::LTFJ_CTR, ltfj_west);
  engine.set_zone_halfspaces(ZoneId::LTBX_CTR, ltbx_south);

  const BoundedStlFormula formula = build_kartal_diversion_formula();

  double max_step_us = 0.0;
  vertirta::stl::StlTickOutput final_out{};

  {
    vertirta::test::ScopedZeroHeapGuard heap_guard;
    engine.configure(formula);

    // 101-tick calibrated SINGLE_MOTOR_LOSS diversion from (-1150, +950) to Kartal_Sahil (0, 0).
    // Cruise finishes at t=78 s (speed 19.13 m/s -> margin +2.87 m/s, AGL >= 45.75 m),
    // followed by terminal vertical descent flare t=79..100 s over Kartal_Sahil (SoC 34% -> 20%).
    for (uint32_t tau = 0U; tau <= 100U; ++tau) {
      StlSignalSample s{};
      if (tau <= 78U) {
        const double frac = static_cast<double>(tau) / 78.0;
        s.east_m = -1150.0 * (1.0 - frac);
        s.north_m = 950.0 * (1.0 - frac);
        s.agl_m = 65.0f - static_cast<float>(frac) * 19.25f;  // 65.0 -> 45.75 m
        s.groundspeed_mps = 19.13f;                           // 22.0 - 19.13 = +2.87 m/s
      } else {
        const double flare = static_cast<double>(tau - 78U) / 22.0;
        s.east_m = 0.0;
        s.north_m = 0.0;
        s.agl_m = 45.75f * static_cast<float>(1.0 - flare);  // Flare down to 0 m touchdown
        s.groundspeed_mps = 1.5f;
      }
      s.soc_pct = 34.0f - 0.14f * static_cast<float>(tau);  // 34.0% -> 20.0% (+8.0% above 12%)
      s.ood_var = 0.03f;

      const auto t0 = std::chrono::steady_clock::now();
      final_out = engine.step(tau, s);
      const auto t1 = std::chrono::steady_clock::now();
      const double step_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      max_step_us = std::max(max_step_us, step_us);
      REQUIRE_FALSE(final_out.violated);
    }

    REQUIRE(heap_guard.alloc_count() == 0U);
    REQUIRE(heap_guard.bytes_allocated() == 0U);
  }

  REQUIRE(final_out.mission_complete);
  REQUIRE(final_out.mission_rho == Catch::Approx(2.87f).margin(0.05f));
  REQUIRE(max_step_us < 1000.0);
}
