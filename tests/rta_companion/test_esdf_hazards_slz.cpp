#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/rta/intruder_kf.hpp"
#include "vertirta/rta/robocentric_esdf.hpp"
#include "vertirta/rta/slz_prior_provider.hpp"
#include "vertirta/rta/slz_reach.hpp"
#include "vertirta/rta/wire_capsules.hpp"

using Catch::Matchers::WithinAbs;
using namespace vertirta::core;
using namespace vertirta::rta;

TEST_CASE("Dual-Res RobocentricEsdfMap <=3.15 MiB, zero-init SlidingConfig, wall distance & <2 ms slide/update") {
  static_assert(sizeof(EsdfCell) == 16, "Bit-packed FIESTA EsdfCell must be exact 16 B");
  static_assert(sizeof(RobocentricEsdfMap) <= 3300000, "RobocentricEsdfMap exceeds 3.15 MiB static budget");

  // ROG-Map #15 guard: SlidingConfig default-initializes every field cleanly to 0
  const SlidingConfig cfg{};
  REQUIRE(cfg.safe_margin_i == 0);
  REQUIRE(cfg.origin_ix == 0);
  REQUIRE(cfg.origin_iy == 0);
  REQUIRE(cfg.origin_iz == 0);

  auto map = std::make_unique<RobocentricEsdfMap>();
  map->reset({0.0, 0.0, 0.0});

  // Build a 5,760-pt 16-beam LiDAR wall at x = 12.0 m (y in [-10, 10], z in [-6, 6])
  std::vector<Vec3Enu> pts;
  pts.reserve(5760);
  for (int i = 0; i < 360; ++i) {
    const double y = -10.0 + 20.0 * static_cast<double>(i) / 359.0;
    for (int j = 0; j < 16; ++j) {
      const double z = -6.0 + 12.0 * static_cast<double>(j) / 15.0;
      pts.push_back({12.0, y, z});
    }
  }
  REQUIRE(pts.size() == 5760u);

  // Slide 10 m and insert wall, timing the update + slide
  const auto t0 = std::chrono::steady_clock::now();
  map->slide_to({10.0, 0.0, 0.0});
  map->insert_point_cloud({2.0, 0.0, 0.0}, pts, 1000000000ULL);
  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  double dist = 0.0;
  Vec3Enu grad{};
  REQUIRE(map->query_dist_grad({2.0, 0.0, 0.0}, dist, grad));
  REQUIRE_THAT(dist, WithinAbs(10.0, 1.25));  // 1.0 m ESDF grid resolution
  REQUIRE(grad.x < -0.75);                    // Gradient points away from x=12 wall (-X)

#if !defined(__SANITIZE_ADDRESS__)
  REQUIRE(elapsed_ms < 20.0);  // First-fill full-wave BFS; steady-state < 2 ms below
#else
  REQUIRE(elapsed_ms < 250.0);
#endif

  // Steady-state incremental update with identical wall -> < 2.0 ms!
  const auto t2 = std::chrono::steady_clock::now();
  map->insert_point_cloud({2.0, 0.0, 0.0}, pts, 1100000000ULL);
  const auto t3 = std::chrono::steady_clock::now();
  const double steady_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
#if !defined(__SANITIZE_ADDRESS__)
  REQUIRE(steady_ms < 2.0);
#else
  REQUIRE(steady_ms < 35.0);
#endif
}

TEST_CASE("WireCapsuleTable O(1) distance & C1 gradient and IntruderTrackerKf + staleness + LTFJ/LTBX CTR") {
  WireCapsuleTable wires;
  REQUIRE(wires.add_capsule({10.0, -5.0, 20.0}, {10.0, 5.0, 20.0}, 3.5, 1u));
  REQUIRE(wires.size() == 1u);

  WireQueryResult wres{};
  REQUIRE(wires.query_nearest_single({15.0, 0.0, 20.0}, wres));
  REQUIRE_THAT(wres.signed_dist_m, WithinAbs(1.5, 1e-3));
  REQUIRE_THAT(wres.grad_unit.x, WithinAbs(1.0, 1e-3));
  REQUIRE_THAT(wres.grad_unit.y, WithinAbs(0.0, 1e-3));
  REQUIRE_THAT(wres.grad_unit.z, WithinAbs(0.0, 1e-3));

  // Intruder CV-KF tracking + 100 ms forward prediction
  IntruderTrackerKf kf;
  kf.update_measurement(42u, {100.0, 50.0, 30.0}, {-10.0, 0.0, 0.0}, 1000000000ULL);
  std::array<HazardPoint, 4> haz{};
  const std::size_t n_h = kf.extract_hazards(1100000000ULL, haz);  // +100 ms later
  REQUIRE(n_h == 1u);
  // 100 ms at vx_enu=-10 m/s -> E moves 100 -> 99 m (NED y = 99 m)
  REQUIRE_THAT(haz[0].pos_ned.y, WithinAbs(99.0, 0.25));

  // Exact staleness escalation policy (>250 ms inflate, >1.0 s v_cap=8.0, >2.0 s divert)
  const StalenessPolicy p_fresh = compute_staleness_policy(0.10);
  const StalenessPolicy p_mid = compute_staleness_policy(0.75);
  const StalenessPolicy p_slow = compute_staleness_policy(1.25);
  const StalenessPolicy p_divert = compute_staleness_policy(2.50);
  REQUIRE(p_fresh.delta_r_stale_m == 0.0);
  REQUIRE(p_mid.delta_r_stale_m > 1.0);
  REQUIRE_THAT(p_slow.v_cap_mps, WithinAbs(8.0, 1e-6));
  REQUIRE_FALSE(p_slow.divert_to_slz);
  REQUIRE(p_divert.divert_to_slz);
  REQUIRE_THAT(p_divert.v_cap_mps, WithinAbs(8.0, 1e-6));

  // DHMI AIP LTFJ polygon+arc (incl. NW J2 pinch!) & LTBX CTR exclusion
  // Kartal Coastal Vertiport (40.8860 N, 29.1880 E): outside both LTFJ and LTBX
  REQUIRE(AirspaceExclusion::signed_margin_ltfj_m(40.8860, 29.1880, 50.0) > 500.0);
  REQUIRE(AirspaceExclusion::signed_margin_ltbx_m(40.8860, 29.1880, 50.0) > 1000.0);
  // LTFJ NW J2 pinch (40.9128 N, 29.1930 E): INSIDE LTFJ CTR -> negative margin!
  REQUIRE(AirspaceExclusion::signed_margin_ltfj_m(40.9128, 29.1930, 50.0) < 0.0);
  // Kartal Merkez (40.90639 N, 29.21111 E): INSIDE LTFJ 5 NM arc -> negative margin!
  REQUIRE(AirspaceExclusion::signed_margin_ltfj_m(40.90639, 29.21111, 50.0) < 0.0);
}

TEST_CASE("SlzReachRanker implements ISlzPriorProvider, vetoes LTFJ J2 / roughness / slope, and ranks Kartal #1") {
  SlzReachRanker ranker;
  ranker.load_default_istanbul_priors();

  // Add 4 invalid candidates that MUST be rejected by hard gates:
  // 1) Inside LTFJ J2 pinch (40.9128 N, 29.1930 E)
  SlzPrior bad_j2{};
  std::strncpy(bad_j2.id, "BAD_LTFJ_J2", sizeof(bad_j2.id) - 1);
  bad_j2.lat_deg = 40.9128;
  bad_j2.lon_deg = 29.1930;
  bad_j2.elev_amsl_m = 20.0;
  bad_j2.clearance_r_m = 15.0f;
  bad_j2.theta_max_deg = 1.0f;
  bad_j2.sigma_z_m = 0.05f;
  bad_j2.p_safe = 0.99f;
  REQUIRE(ranker.add_candidate(bad_j2));

  // 2) Roughness sigma_z = 0.25 m > 0.15 m
  SlzPrior bad_rough = bad_j2;
  std::strncpy(bad_rough.id, "BAD_ROUGH", sizeof(bad_rough.id) - 1);
  bad_rough.lat_deg = 40.8870;
  bad_rough.lon_deg = 29.1850;
  bad_rough.sigma_z_m = 0.25f;
  REQUIRE(ranker.add_candidate(bad_rough));

  // 3) Slope theta = 11 deg > 8 deg
  SlzPrior bad_slope = bad_rough;
  std::strncpy(bad_slope.id, "BAD_SLOPE", sizeof(bad_slope.id) - 1);
  bad_slope.sigma_z_m = 0.05f;
  bad_slope.theta_max_deg = 11.0f;
  REQUIRE(ranker.add_candidate(bad_slope));

  // 4) Radius 3.5 m < 5.0 m
  SlzPrior bad_rad = bad_slope;
  std::strncpy(bad_rad.id, "BAD_RADIUS", sizeof(bad_rad.id) - 1);
  bad_rad.theta_max_deg = 2.0f;
  bad_rad.clearance_r_m = 3.5f;
  REQUIRE(ranker.add_candidate(bad_rad));

  // Query via polymorphic ISlzPriorProvider interface near Kartal Coastal Vertiport
  const ISlzPriorProvider& provider = ranker;
  const SlzProviderSnapshot snap = provider.query_priors({0.0, 0.0, 40.0}, 50.0f, 5000000000ULL);
  REQUIRE_FALSE(snap.stale);
  REQUIRE(snap.count == 3u);  // Only the 3 valid Istanbul coastal pads survive!
  REQUIRE(std::string_view(snap.priors[0].id) == "Kartal_Coastal_Vertiport");
  REQUIRE(snap.priors[0].rank == 1u);
  REQUIRE(snap.priors[0].reason_mask == 0u);
  for (uint8_t i = 0; i < snap.count; ++i) {
    REQUIRE(std::string_view(snap.priors[i].id).substr(0, 4) != "BAD_");
    REQUIRE(snap.priors[i].sigma_z_m <= 0.15f);
    REQUIRE(snap.priors[i].theta_max_deg <= 8.0f);
    REQUIRE(snap.priors[i].clearance_r_m >= 5.0f);
  }
}
