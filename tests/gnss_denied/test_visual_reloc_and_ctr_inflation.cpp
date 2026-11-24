#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>

#include "vertirta/core/geodesy.hpp"
#include "vertirta/gnss_denied/ctr_inflation_fsm.hpp"
#include "vertirta/gnss_denied/visual_georeloc_adapter.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/intruder_kf.hpp"

using Catch::Approx;
using namespace vertirta::gnss_denied;

TEST_CASE("VisualGeoRelocAdapter 3-gate acceptance, Joseph-form SPD update, MegaLoc re-lock, and Sentinel-2 gross-blunder guard",
          "[gnss_denied][georeloc]") {
  VisualGeoRelocAdapter reloc{};
  REQUIRE_FALSE(reloc.config().enable_fp16);  // TU116 Override #6

  const vertirta::core::Vec3Ned prior_pos{100.0, -50.0, -40.0};
  const vertirta::core::GeodeticWgs84 exact_geo =
      vertirta::core::local_enu_to_geodetic(vertirta::core::ned_to_enu(prior_pos),
                                            vertirta::core::kKartalDatum);
  const Cov2D prior_cov{25.0, 3.0, 36.0};  // sigma_max ~ 6.1 m

  VisualMatchObservation obs{};
  obs.matched_geo = exact_geo;
  obs.ransac_inliers = 48u;
  obs.reproj_rmse_px = 1.4;
  obs.convex_hull_ratio = 0.42;
  obs.map_tier = OrthoMapTier::kLocalHighResOrtho05m;
  obs.valid = true;

  // Gate 1a: reject < 25 inliers
  {
    auto bad = obs;
    bad.ransac_inliers = 24u;
    const auto d = reloc.fuse_observation(prior_pos, prior_cov, bad);
    REQUIRE_FALSE(d.accepted);
    REQUIRE(d.reason == GeoRelocRejectReason::kTooFewInliers);
  }
  // Gate 1b: reject >= 2.5 px RMSE
  {
    auto bad = obs;
    bad.reproj_rmse_px = 2.6;
    const auto d = reloc.fuse_observation(prior_pos, prior_cov, bad);
    REQUIRE_FALSE(d.accepted);
    REQUIRE(d.reason == GeoRelocRejectReason::kHighReprojRmse);
  }
  // Gate 2: reject <= 0.25 convex hull ratio
  {
    auto bad = obs;
    bad.convex_hull_ratio = 0.22;
    const auto d = reloc.fuse_observation(prior_pos, prior_cov, bad);
    REQUIRE_FALSE(d.accepted);
    REQUIRE(d.reason == GeoRelocRejectReason::kLowConvexHullCoverage);
  }
  // Gate 3: reject Mahalanobis outlier (chi^2 > 9.21)
  {
    auto outlier = obs;
    outlier.matched_geo = vertirta::core::local_enu_to_geodetic(
        vertirta::core::ned_to_enu({prior_pos.x + 80.0, prior_pos.y + 80.0, prior_pos.z}),
        vertirta::core::kKartalDatum);
    const auto d = reloc.fuse_observation(prior_pos, prior_cov, outlier);
    REQUIRE_FALSE(d.accepted);
    REQUIRE(d.reason == GeoRelocRejectReason::kMahalanobisGateFailed);
    REQUIRE(d.mahalanobis_chi2 > 9.21);
  }

  // Sentinel-2 10 m (~45 m CEP50) MUST be rejected as gross-blunder only when prior_sigma <= 45 m!
  {
    auto s2 = obs;
    s2.map_tier = OrthoMapTier::kSentinel2GrossBlunder10m;
    const auto d = reloc.fuse_observation(prior_pos, prior_cov, s2);
    REQUIRE_FALSE(d.accepted);
    REQUIRE(d.reason == GeoRelocRejectReason::kSentinel2GrossBlunderOnly);
  }

  // Valid 0.5 m ortho XFeat+LighterGlue match -> SPD Joseph-form update keeps 3*sigma < 25 m!
  {
    const auto d = reloc.fuse_observation(prior_pos, prior_cov, obs);
    REQUIRE(d.accepted);
    REQUIRE(d.updated_cov.min_eigenvalue() > 0.0);
    REQUIRE(d.updated_cov.sigma_max() < prior_cov.sigma_max());
    REQUIRE(3.0 * d.updated_cov.sigma_max() < 25.0);
  }

  // MegaLoc lost-in-space re-init when sigma_pos > 50 m requires 2 consistent keyframes within 15 m!
  {
    const Cov2D lost_cov{3600.0, 0.0, 3600.0};  // sigma = 60 m > 50 m
    auto ml = obs;
    ml.from_megaloc = true;
    ml.matched_geo = vertirta::core::local_enu_to_geodetic(
        vertirta::core::ned_to_enu({prior_pos.x + 120.0, prior_pos.y - 90.0, prior_pos.z}),
        vertirta::core::kKartalDatum);

    const auto first = reloc.fuse_observation(prior_pos, lost_cov, ml);
    REQUIRE_FALSE(first.accepted);
    REQUIRE(first.megaloc_requested);
    REQUIRE(first.reason == GeoRelocRejectReason::kMegaLocAwaitingCorroboration);

    // Second spatially consistent MegaLoc keyframe within 5 m triggers re-lock!
    auto ml2 = ml;
    ml2.matched_geo = vertirta::core::local_enu_to_geodetic(
        vertirta::core::ned_to_enu({prior_pos.x + 123.0, prior_pos.y - 88.0, prior_pos.z}),
        vertirta::core::kKartalDatum);
    const auto second = reloc.fuse_observation(prior_pos, lost_cov, ml2);
    REQUIRE(second.accepted);
    REQUIRE(second.megaloc_relocked);
    REQUIRE(second.updated_cov.sigma_max() < 10.0);
  }
}

TEST_CASE("CtrInflationFsm enforces verified DHMI AIP LTBX 40d59m26sN, exact dh/dt CBVF margin, 0 CTR breaches, and 3-state FSM",
          "[gnss_denied][ctr_inflation][cbvf]") {
  CtrInflationFsm fsm{};

  // Verify DHMI AIP LTBX ARP lock (40.990556 N, 29.216111 E) — NOT stale DR-10 40.99889 N!
  REQUIRE(fsm.config().ltbx_arp_lat_deg == Approx(40.990556).margin(1e-6));
  REQUIRE(fsm.config().ltbx_arp_lon_deg == Approx(29.216111).margin(1e-6));
  REQUIRE(std::abs(fsm.config().ltbx_arp_lat_deg - 40.99889) > 0.005);

  // 1. Verify dh/dt = -r_dot_infl increases braking conservatism inside UNMODIFIED CbvfSafetyFilter::step_100hz!
  // Place aircraft cleanly outside BOTH LTFJ (~325 m west of NW edge) and verified LTBX (40d59m26sN, ~460 m south),
  // cruising east at 14 m/s toward LTFJ CTR so BOTH CTR walls are actively evaluated.
  const vertirta::core::GeodeticWgs84 near_j2{40.9050, 29.1920, vertirta::core::amsl_to_hae(52.0, 40.9050, 29.1920)};
  REQUIRE(vertirta::rta::AirspaceExclusion::signed_margin_ltfj_m(near_j2.lat_deg, near_j2.lon_deg, 52.0) > 100.0);
  REQUIRE(vertirta::rta::AirspaceExclusion::signed_margin_ltfj_m(near_j2.lat_deg, near_j2.lon_deg, 52.0) < 480.0);
  REQUIRE(vertirta::rta::AirspaceExclusion::signed_margin_ltbx_m(near_j2.lat_deg, near_j2.lon_deg, 52.0) > 200.0);
  const vertirta::core::Vec3Ned pos_near_j2 =
      vertirta::core::enu_to_ned(vertirta::core::geodetic_to_local_enu(near_j2, vertirta::core::kKartalDatum));
  const vertirta::core::Vec3Ned vel_east{0.0, 14.0, 0.0};

  fsm.reset(12.0);
  fsm.propagate_open_loop(1.0, 18.0);  // Non-zero cov_rate_ => r_dot_infl > 0!
  std::array<vertirta::rta::HazardPoint, 2> hazards_with_dhdt{};
  const auto st_dhdt = fsm.evaluate_and_build_hazards(0.1, pos_near_j2, 52.0, hazards_with_dhdt);
  REQUIRE(st_dhdt.emitted_ctr_hazards >= 1u);
  REQUIRE(st_dhdt.ltfj_r_dot_infl_mps > 0.0);
  REQUIRE(st_dhdt.ltfj_delta_r_m > 3.0 * 12.0);

  // Build identical static-only hazard (h.vel_ned = 0) and prove dh/dt reduces min_cbvf_v strictly!
  auto hazards_static = hazards_with_dhdt;
  for (std::size_t i = 0; i < st_dhdt.emitted_ctr_hazards; ++i) {
    hazards_static[i].vel_ned = {0.0, 0.0, 0.0};
  }

  vertirta::core::CapabilityEnvelope env{};
  env.regime = vertirta::core::FlightRegime::NOMINAL;
  env.a_xy_max = 5.5f;
  env.tilt_max_rad = 0.45f;
  env.t_max_n = 3600.0f;
  env.v_term_v = 400.0f;
  env.soc = 0.80f;

  vertirta::rta::CbvfSafetyFilter cbvf_dynamic(nullptr, nullptr);
  vertirta::rta::CbvfSafetyFilter cbvf_static(nullptr, nullptr);
  const auto out_dyn = cbvf_dynamic.step_100hz(
      pos_near_j2, vel_east, {0.0, 2.0, 0.0}, 0.0, env,
      std::span<const vertirta::rta::HazardPoint>(hazards_with_dhdt.data(), st_dhdt.emitted_ctr_hazards));
  const auto out_stat = cbvf_static.step_100hz(
      pos_near_j2, vel_east, {0.0, 2.0, 0.0}, 0.0, env,
      std::span<const vertirta::rta::HazardPoint>(hazards_static.data(), st_dhdt.emitted_ctr_hazards));

  REQUIRE(out_dyn.min_cbvf_v < out_stat.min_cbvf_v);

  // 2. Closed-loop sustained GNSS-denied cruise parallel to LTFJ western edge into UNMODIFIED CbvfSafetyFilter:
  //    Aircraft cruises south-southeast (pushing east into LTFJ wall while traveling south along the coast at ~14 m/s):
  //    Verify 0 LTFJ & 0 LTBX breaches AND deterministic 3-state FSM progression:
  //      BOUNDED_CRUISE (<25 m) -> DEGRADED_CAUTION (25-65 m, 12-15 m/s) -> DETERMINISTIC_VISUAL_SLZ_ABORT (>=65 m)!
  fsm.reset(3.5);
  vertirta::rta::CbvfSafetyFilter cbvf_loop(nullptr, nullptr);
  vertirta::core::Vec3Ned pos = pos_near_j2;
  vertirta::core::Vec3Ned vel{-14.0, 8.0, 0.0};

  bool saw_bounded = false;
  bool saw_caution = false;
  bool saw_abort = false;
  const double dt = 0.1;  // 10 Hz loop up to 750 s sustained open-loop GNSS denial

  for (int k = 0; k < 7500; ++k) {
    const double speed_xy = std::hypot(vel.x, vel.y);
    fsm.propagate_open_loop(dt, speed_xy);

    std::array<vertirta::rta::HazardPoint, 2> ctr_haz{};
    const auto st = fsm.evaluate_and_build_hazards(dt, pos, 52.0, ctr_haz);

    if (st.nav_state == GnssDeniedNavState::kBoundedGnssDeniedCruise) {
      saw_bounded = true;
      REQUIRE(st.v_max_cap_mps == Approx(22.0));
    } else if (st.nav_state == GnssDeniedNavState::kDegradedCaution) {
      saw_caution = true;
      REQUIRE(st.v_max_cap_mps >= 12.0);
      REQUIRE(st.v_max_cap_mps <= 15.0);
      REQUIRE(st.westward_bias_active);
    } else if (st.nav_state == GnssDeniedNavState::kDeterministicVisualSlzAbort) {
      saw_abort = true;
      REQUIRE(st.visual_slz_abort_latched);
      REQUIRE(st.sigma_pos_m >= 65.0);
    }

    // Nominal command cruises south along the coast while pushing east toward LTFJ;
    // unmodified CbvfSafetyFilter + dynamic 3sigma+dh/dt CTR wall must repel cleanly!
    const vertirta::core::Vec3Ned a_nom{
        (-14.0 - vel.x) * 0.8,
        st.westward_bias_active ? (-3.0 - vel.y) * 0.8 : (8.0 - vel.y) * 0.8,
        0.0,
    };
    const auto safe = cbvf_loop.step_100hz(
        pos, vel, a_nom, 0.0, env,
        std::span<const vertirta::rta::HazardPoint>(ctr_haz.data(), st.emitted_ctr_hazards));

    vel.x += safe.a_safe_ned.x * dt;
    vel.y += safe.a_safe_ned.y * dt;
    const double vxy = std::hypot(vel.x, vel.y);
    if (vxy > st.v_max_cap_mps && vxy > 1e-6) {
      vel.x *= st.v_max_cap_mps / vxy;
      vel.y *= st.v_max_cap_mps / vxy;
    }
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;

    // Assert ZERO physical LTFJ or LTBX CTR breaches throughout the entire 10-minute GNSS-denied flight!
    const auto geo_now =
        vertirta::core::local_enu_to_geodetic(vertirta::core::ned_to_enu(pos),
                                              vertirta::core::kKartalDatum);
    REQUIRE(vertirta::rta::AirspaceExclusion::signed_margin_ltfj_m(geo_now.lat_deg, geo_now.lon_deg, 52.0) > 0.0);
    REQUIRE(vertirta::rta::AirspaceExclusion::signed_margin_ltbx_m(geo_now.lat_deg, geo_now.lon_deg, 52.0) > 0.0);

    if (saw_abort) {
      break;  // Aircraft initiates deterministic Visual-SLZ descent before 80 m CBVF infeasibility floor!
    }
  }

  REQUIRE(saw_bounded);
  REQUIRE(saw_caution);
  REQUIRE(saw_abort);
}
