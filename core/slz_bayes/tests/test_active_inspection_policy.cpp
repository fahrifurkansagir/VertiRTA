// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vertirta/perception/active_inspection_policy.hpp"
#include "vertirta/perception/slz_belief_grid.hpp"
#include "vertirta/perception/visual_slz_prior_provider.hpp"
#include "zero_heap_guard.hpp"

using Catch::Approx;
using vertirta::Vec3Enu;
using vertirta::core::FlightRegime;
using vertirta::perception::ActiveInspectionPolicy;
using vertirta::perception::DepthMode;
using vertirta::perception::EnergyGateInput;
using vertirta::perception::IForwardHazardProvider;
using vertirta::perception::NullForwardHazardProvider;
using vertirta::perception::REASON_CLEARANCE_INSUFF;
using vertirta::perception::REASON_CTR_VIOLATION;
using vertirta::perception::REASON_DYNAMIC_HAZARD;
using vertirta::perception::REASON_ENERGY_UNREACHABLE;
using vertirta::perception::REASON_MONO_ALT_VETO;
using vertirta::perception::REASON_NONE;
using vertirta::perception::REASON_OBSTACLE_CLEARANCE;
using vertirta::perception::REASON_OOD_UNCERTAIN;
using vertirta::perception::REASON_ORBIT_TIMEOUT;
using vertirta::perception::REASON_ROUGHNESS_EXCEEDED;
using vertirta::perception::REASON_SLOPE_EXCEEDED;
using vertirta::perception::REASON_STEP_EXCEEDED;
using vertirta::perception::SlzBeliefGrid;
using vertirta::perception::SlzDecisionState;
using vertirta::perception::SlzPointObservation;
using vertirta::perception::VisualSlzPriorProvider;
using vertirta::rta::ISlzPriorProvider;
using vertirta::rta::SlzPrior;

namespace {

SlzPrior MakeCleanRoofPrior(const char* id = "Dragos_Roof_01") noexcept {
  SlzPrior p{};
  std::strncpy(p.id, id, sizeof(p.id) - 1u);
  p.center_enu = {0.0, 0.0, 12.0};
  p.p_safe = 0.94f;
  p.slz_ood_variance = 0.03f;
  p.theta_max_deg = 2.5f;
  p.sigma_z_m = 0.04f;
  p.step_max_m = 0.06f;
  p.clearance_r_m = 6.5f;
  p.h_dyn_max = 0.02f;
  p.reason_mask = REASON_NONE;
  return p;
}

SlzPrior MakeUncertainRoofPrior(float slz_ood_variance, const char* id = "Dragos_Roof_01") noexcept {
  SlzPrior p = MakeCleanRoofPrior(id);
  p.p_safe = 0.78f;
  p.slz_ood_variance = slz_ood_variance;
  return p;
}

}  // namespace

TEST_CASE("Harmonized SlzReasonBits match SlzPrior.msg and slz_prior_provider.hpp exactly",
          "[slz_policy][bits]") {
  REQUIRE(REASON_NONE == 0u);
  REQUIRE(REASON_CTR_VIOLATION == 1u);
  REQUIRE(REASON_ROUGHNESS_EXCEEDED == 2u);
  REQUIRE(REASON_SLOPE_EXCEEDED == 4u);
  REQUIRE(REASON_CLEARANCE_INSUFF == 8u);
  REQUIRE(REASON_ENERGY_UNREACHABLE == 16u);
  REQUIRE(REASON_OBSTACLE_CLEARANCE == 32u);
  REQUIRE(REASON_STEP_EXCEEDED == 64u);
  REQUIRE(REASON_OOD_UNCERTAIN == 128u);
  REQUIRE(REASON_MONO_ALT_VETO == 256u);
  REQUIRE(REASON_ORBIT_TIMEOUT == 512u);
  REQUIRE(REASON_DYNAMIC_HAZARD == 1024u);
}

TEST_CASE("ActiveInspectionPolicy 22kWh energy gate, 2-phase orbit, reasons, Mod B veto & underflow",
          "[slz_policy]") {
  ActiveInspectionPolicy pol;

  // 1) Locked 22 kWh, R = 0.75 m => ~40.5 kW => 0.614% * 1.25 = 0.7675% (NEVER 13.2 kWh / 1.36%).
  const float nom = pol.EstimateOrbitDeltaSocPct({50.0f, 20.0f, 0.0f, FlightRegime::NOMINAL});
  REQUIRE(nom == Approx(0.614f * 1.25f).margin(0.03f));
  const float cross_wind = pol.EstimateOrbitDeltaSocPct({50.0f, 20.0f, 10.0f, FlightRegime::CROSS_DUAL});
  REQUIRE(cross_wind < 1.35f);
  REQUIRE(cross_wind > nom);

  // 2) APPROACH_ORBIT absorbs 16 s descent from 80 m -> 24 m AGL BEFORE arming 12 s orbit timer.
  const auto unc = MakeUncertainRoofPrior(0.30f);
  const auto d0 = pol.Step(
      unc, {40.0, 0.0, 92.0}, 80.0f, DepthMode::RGBD_PRIMARY,
      {50.0f, 20.0f, 5.0f, FlightRegime::NOMINAL}, 1'000'000'000ULL);
  REQUIRE(d0.state == SlzDecisionState::APPROACH_ORBIT);
  REQUIRE(d0.inspect_setpoint.has_value());

  // 16 s later on the 15 m / 22 m ring: enters ACTIVE_INSPECT_ORBIT with elapsed_s == 0.0.
  const auto s1 = pol.Step(
      unc, {16.0, 0.0, 36.0}, 24.0f, DepthMode::RGBD_PRIMARY,
      {49.0f, 20.0f, 5.0f, FlightRegime::NOMINAL}, 17'000'000'000ULL);
  REQUIRE(s1.state == SlzDecisionState::ACTIVE_INSPECT_ORBIT);
  REQUIRE(s1.inspect_setpoint.has_value());
  REQUIRE(s1.inspect_setpoint->elapsed_s == Approx(0.0f).margin(1e-3f));
  REQUIRE(s1.inspect_setpoint->gimbal_pitch_rad == Approx(std::atan2(22.0f, 15.0f)).margin(0.05f));
  // Gimbal yaw at +X looking back toward TLOF (0,0) is +-pi.
  REQUIRE(std::fabs(s1.inspect_setpoint->gimbal_yaw_rad) == Approx(3.14159265f).margin(0.05f));

  // Backwards timestamp inside ACTIVE_INSPECT_ORBIT must NOT underflow into timeout.
  const auto back = pol.Step(
      unc, {15.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {49.0f, 20.0f, 5.0f, FlightRegime::NOMINAL}, 16'000'000'000ULL);
  REQUIRE(back.state == SlzDecisionState::ACTIVE_INSPECT_ORBIT);
  REQUIRE(back.inspect_setpoint.has_value());
  REQUIRE(back.inspect_setpoint->elapsed_s == Approx(0.0f).margin(1e-5f));

  // 11.5 s into orbit: still active.
  const auto mid = pol.Step(
      unc, {0.0, 15.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {48.0f, 20.0f, 5.0f, FlightRegime::NOMINAL}, 28'500'000'000ULL);
  REQUIRE(mid.state == SlzDecisionState::ACTIVE_INSPECT_ORBIT);

  // 12.1 s into orbit: cleanly times out with REASON_ORBIT_TIMEOUT.
  const auto tout = pol.Step(
      unc, {-15.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {48.0f, 20.0f, 5.0f, FlightRegime::NOMINAL}, 29'100'000'000ULL);
  REQUIRE(tout.state == SlzDecisionState::REJECT_DIVERT_NEXT);
  REQUIRE((tout.reason_mask & REASON_ORBIT_TIMEOUT) != 0u);

  // 3) Complete 6-criterion reject bits never return REASON_NONE; Mod B > 30 m vetoes commit.
  auto parapet = MakeCleanRoofPrior("Parapet_Roof");
  parapet.step_max_m = 0.32f;
  const auto r_step = pol.Step(
      parapet, {0.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {50.0f, 20.0f, 0.0f, FlightRegime::NOMINAL}, 30'000'000'000ULL);
  REQUIRE(r_step.state == SlzDecisionState::REJECT_DIVERT_NEXT);
  REQUIRE((r_step.reason_mask & REASON_STEP_EXCEEDED) != 0u);

  auto tight = MakeCleanRoofPrior("Tight_Roof");
  tight.clearance_r_m = 4.2f;
  const auto r_clr = pol.Step(
      tight, {0.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {50.0f, 20.0f, 0.0f, FlightRegime::NOMINAL}, 31'000'000'000ULL);
  REQUIRE(r_clr.state == SlzDecisionState::REJECT_DIVERT_NEXT);
  REQUIRE((r_clr.reason_mask & REASON_CLEARANCE_INSUFF) != 0u);

  const auto mono_hi = pol.Step(
      MakeCleanRoofPrior("Mono_Roof"), {0.0, 0.0, 50.0}, 38.0f, DepthMode::MONO_FALLBACK,
      {50.0f, 20.0f, 0.0f, FlightRegime::NOMINAL}, 32'000'000'000ULL);
  REQUIRE(mono_hi.state != SlzDecisionState::COMMIT_TOUCHDOWN);
  REQUIRE((mono_hi.reason_mask & REASON_MONO_ALT_VETO) != 0u);

  // Same Mod B candidate commits cleanly once descended <= 30 m AGL.
  const auto mono_lo = pol.Step(
      MakeCleanRoofPrior("Mono_Roof"), {0.0, 0.0, 34.0}, 22.0f, DepthMode::MONO_FALLBACK,
      {50.0f, 20.0f, 0.0f, FlightRegime::NOMINAL}, 33'000'000'000ULL);
  REQUIRE(mono_lo.state == SlzDecisionState::COMMIT_TOUCHDOWN);
  REQUIRE(mono_lo.reason_mask == REASON_NONE);

  // 4) Low SoC reserve blocks orbit with REASON_ENERGY_UNREACHABLE; CEL refuses orbit.
  pol.Reset();
  const auto low_soc = pol.Step(
      unc, {15.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {20.4f, 20.0f, 0.0f, FlightRegime::NOMINAL}, 40'000'000'000ULL);
  REQUIRE(low_soc.state == SlzDecisionState::REJECT_DIVERT_NEXT);
  REQUIRE((low_soc.reason_mask & REASON_ENERGY_UNREACHABLE) != 0u);

  const auto cel_no_orbit = pol.Step(
      unc, {15.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
      {80.0f, 20.0f, 0.0f, FlightRegime::SAME_ARM_ACTIVE_SPIN}, 41'000'000'000ULL);
  REQUIRE(cel_no_orbit.state == SlzDecisionState::REJECT_DIVERT_NEXT);
}

TEST_CASE("VisualSlzPriorProvider implements ISlzPriorProvider with 250 ms staleness & zero heap",
          "[slz_policy][provider][zero_heap]") {
  SlzBeliefGrid grid;
  const Vec3Enu pad{0.0, 0.0, 12.0};
  std::array<SlzPointObservation, 169> obs{};
  std::size_t idx = 0u;
  for (int iy = -6; iy <= 6; ++iy) {
    for (int ix = -6; ix <= 6; ++ix) {
      obs[idx++] = {
          {static_cast<double>(ix) * 0.85, static_cast<double>(iy) * 0.85, 12.0},
          0.96f,
          0.0f,
          -12.0f,
          1.0f,
      };
    }
  }
  grid.SetOrigin(pad);
  grid.UpdateFrame(obs, {0.0, 0.0, 34.0}, 1.0, 0.5f);

  VisualSlzPriorProvider provider;
  ISlzPriorProvider& rta_port = provider;

  std::array<SlzPrior, 2> catalog{};
  catalog[0] = MakeUncertainRoofPrior(0.25f, "Kartal_Vertiport");
  catalog[1] = MakeCleanRoofPrior("LTFJ_Blocked");
  catalog[1].reason_mask = REASON_CTR_VIOLATION;

  // Decoupled V2.1 IForwardHazardProvider stub returns empty span cleanly.
  NullForwardHazardProvider wire_stub;
  const IForwardHazardProvider& fwd_port = wire_stub;
  REQUIRE(fwd_port.QueryForwardHazards(1'000'000'000ULL).empty());

  {
    vertirta::test::ScopedZeroHeapGuard guard;
    const auto dec = provider.UpdateFromGrid(
        grid, catalog, {15.0, 0.0, 34.0}, 22.0f, DepthMode::RGBD_PRIMARY,
        {55.0f, 20.0f, 4.0f, FlightRegime::NOMINAL}, 1'000'000'000ULL);
    REQUIRE(dec.timestamp_ns == 1'000'000'000ULL);

    // Fresh within 250 ms contract.
    const auto fresh = rta_port.query_priors({15.0, 0.0, 34.0}, 55.0f, 1'100'000'000ULL);
    REQUIRE_FALSE(fresh.stale);
    REQUIRE(fresh.count == 2u);
    // CTR-clean candidate ranked first.
    REQUIRE(std::strncmp(fresh.priors[0].id, "Kartal_Vertiport", 16) == 0);

    // Stale at 300 ms > 250 ms contract.
    const auto stale = rta_port.query_priors({15.0, 0.0, 34.0}, 55.0f, 1'300'000'000ULL);
    REQUIRE(stale.stale);

    // Backwards timestamp treated as stale fail-closed.
    const auto backwards = rta_port.query_priors({15.0, 0.0, 34.0}, 55.0f, 900'000'000ULL);
    REQUIRE(backwards.stale);

    REQUIRE(guard.alloc_count() == 0u);
    REQUIRE(guard.bytes_allocated() == 0u);
  }
}
