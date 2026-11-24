#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "vertirta/gnss_denied/basalt_vio_adapter.hpp"
#include "vertirta/gnss_denied/gnss_spoof_guard.hpp"

using Catch::Approx;
using namespace vertirta::gnss_denied;

TEST_CASE("BasaltVioAdapter enforces strict zero-GNSS decoupling and purges 100% of PIDNet-S water features",
          "[gnss_denied][vio]") {
  STATIC_REQUIRE_FALSE(BasaltVioAdapter::kAllowsGnssFeedback);

  BasaltVioAdapter vio{};
  REQUIRE(vio.config().stereo_baseline_m == Approx(0.80));
  REQUIRE(vio.config().stereo_baseline_m >= 0.50);
  vio.reset({0.0, 0.0, -40.0}, {12.0, 0.0, 0.0});

  VioSensorFrame f{};
  f.stamp_ns = 1'000'000'000ULL;
  f.dt_s = 0.1;
  f.imu_accel_ned = {0.0, 0.0, 0.0};
  f.has_stereo_frame = true;
  f.stereo_delta_vel_ned = {12.0, 0.0, 0.0};
  f.radalt_agl_m = 40.0;
  f.radalt_valid = true;

  // Populate 60 features: 45 on Marmara Sea water pixels + 15 on shoreline land pixels.
  f.feature_count = 60u;
  for (std::size_t i = 0; i < 60u; ++i) {
    f.features[i] = StereoKltFeature{
        .u_px = 100.0 + static_cast<double>(i) * 5.0,
        .v_px = 240.0,
        .disparity_px = 8.0,
        .flow_dx_px = 2.0,
        .flow_dy_px = 0.0,
        .is_water_pixel = (i < 45u),
        .valid = true,
    };
  }

  VioStateEstimate est{};
  REQUIRE(vio.step(f, est));
  REQUIRE(est.valid);
  REQUIRE(est.metric_scale_locked);
  REQUIRE(est.purged_water_features == 45u);
  REQUIRE(est.kept_land_features == 15u);
  REQUIRE(est.water_fraction == Approx(0.75));
}

TEST_CASE("GnssSpoofGuard H1 RF jamming <= 0.2 s (sigma_init=2.0) and H2 step meaconing <= 0.3 s (2.0 s rollback, sigma_init=3.5)",
          "[gnss_denied][spoof_guard]") {
  // --- Test H1 RF Jamming ---
  GnssSpoofGuard guard{};
  VioStateEstimate vio{};
  vio.valid = true;
  vio.water_fraction = 0.05;

  GnssSpoofGuardStatus st{};
  for (int k = 0; k < 40; ++k) {
    const double t = 0.1 * static_cast<double>(k);
    vio.stamp_ns = static_cast<uint64_t>(k) * 100'000'000ULL;
    vio.pos_ned = {15.0 * t, 0.0, -40.0};
    vio.vel_ned = {15.0, 0.0, 0.0};

    GnssFixV116 g{};
    g.stamp_ns = vio.stamp_ns;
    g.pos_ned = vio.pos_ned;
    g.vel_ned = vio.vel_ned;
    g.cn0_avg_dbhz = (k >= 35) ? 20.0 : 42.0;  // Sudden C/N0 drop at k=35
    g.hdop = (k >= 35) ? 3.2 : 0.9;
    st = guard.step_10hz(g, vio);
    if (st.latched) {
      REQUIRE((k - 35) * 0.1 <= 0.20 + 1e-9);
      break;
    }
  }
  REQUIRE(st.latched);
  REQUIRE(st.hypothesis == GnssHypothesis::kH1RfJamming);
  REQUIRE_FALSE(st.gnss_fused_allowed);
  REQUIRE(st.sigma_init_m == Approx(2.0));

  // --- Test H2 Step Meaconing (6.0 m jump -> TTD <= 0.3 s, 2.0 s rollback, sigma_init=3.5 m) ---
  guard.reset();
  for (int k = 0; k < 50; ++k) {
    const double t = 0.1 * static_cast<double>(k);
    vio.stamp_ns = static_cast<uint64_t>(k) * 100'000'000ULL;
    vio.pos_ned = {15.0 * t, 0.0, -40.0};
    vio.vel_ned = {15.0, 0.0, 0.0};

    GnssFixV116 g{};
    g.stamp_ns = vio.stamp_ns;
    g.pos_ned = vio.pos_ned;
    if (k >= 30) {
      g.pos_ned.y += 6.0;  // 6.0 m instantaneous meaconing step east toward LTFJ
    }
    g.vel_ned = vio.vel_ned;
    st = guard.step_10hz(g, vio);
    if (st.latched) {
      REQUIRE((k - 30) * 0.1 <= 0.30 + 1e-9);
      // Rollback must strip the 6.0 m east spoof bias cleanly!
      REQUIRE(std::abs(st.rollback_pos_ned.y - 0.0) < 0.05);
      break;
    }
  }
  REQUIRE(st.latched);
  REQUIRE(st.hypothesis == GnssHypothesis::kH2StepMeaconing);
  REQUIRE(st.rollback_executed);
  REQUIRE(st.sigma_init_m == Approx(3.5));
}

TEST_CASE("GnssSpoofGuard H3 100-seed Monte Carlo slow ramps (0.5-2.0 m/s): P95 TTD <= 1.4 s, P99 post-rollback slip <= 0.85 m, 0 water-glint FAR",
          "[gnss_denied][mcusum]") {
  // 1. Land 100-seed Monte Carlo across 0.5, 1.0, 1.5, 2.0 m/s ramps (DR-10 Table 2.3: sigma_w=0.14, sigma_b=0.10, tau=60 s)
  const double ramp_rates[] = {0.5, 1.0, 1.5, 2.0};
  for (double ramp_mps : ramp_rates) {
    std::vector<double> ttds;
    std::vector<double> slips;
    ttds.reserve(100);
    slips.reserve(100);

    for (uint32_t seed = 0; seed < 100u; ++seed) {
      std::mt19937_64 rng(10000ULL * static_cast<uint64_t>(ramp_mps * 10.0) + seed + 1ULL);
      std::normal_distribution<double> w_vel(0.0, 0.14);
      std::normal_distribution<double> w_pos(0.0, 0.15);
      std::normal_distribution<double> ou_noise(0.0, 1.0);

      GnssSpoofGuard guard{};
      double bias_n = 0.08 * ou_noise(rng);
      double bias_e = 0.08 * ou_noise(rng);
      const double phi = std::exp(-0.1 / 60.0);
      const double ou_step = 0.10 * std::sqrt(1.0 - phi * phi);

      // Warm up 18.0 s (180 epochs) so 15 s lagged baseline + 5 s guard band are fully populated.
      const int k_onset = 180;
      double true_n = 0.0;
      double true_e = 0.0;
      double vio_n = 0.0;
      double vio_e = 0.0;
      double spoof_e = 0.0;

      bool caught = false;
      for (int k = 0; k < k_onset + 80; ++k) {
        bias_n = phi * bias_n + ou_step * ou_noise(rng);
        bias_e = phi * bias_e + ou_step * ou_noise(rng);

        true_n += 15.0 * 0.1;
        vio_n += (15.0 - bias_n) * 0.1;
        vio_e += (0.0 - bias_e) * 0.1;

        double spoof_v_e = 0.0;
        if (k >= k_onset) {
          spoof_v_e = ramp_mps;
          spoof_e += spoof_v_e * 0.1;
        }

        VioStateEstimate vio{};
        vio.stamp_ns = static_cast<uint64_t>(k) * 100'000'000ULL;
        vio.pos_ned = {vio_n, vio_e, -40.0};
        vio.vel_ned = {15.0 - bias_n, -bias_e, 0.0};
        vio.water_fraction = 0.05;
        vio.valid = true;

        GnssFixV116 g{};
        g.stamp_ns = vio.stamp_ns;
        g.pos_ned = {true_n + w_pos(rng), true_e + spoof_e + w_pos(rng), -40.0};
        g.vel_ned = {15.0 + w_vel(rng), spoof_v_e + w_vel(rng), 0.0};
        g.valid = true;

        const auto st = guard.step_10hz(g, vio);
        if (k >= k_onset && st.latched) {
          const double ttd = static_cast<double>(k - k_onset + 1) * 0.1;
          const double post_rollback_err_e = std::abs(st.rollback_pos_ned.y - true_e);
          ttds.push_back(ttd);
          slips.push_back(post_rollback_err_e);
          caught = true;
          break;
        }
      }
      REQUIRE(caught);
    }

    std::sort(ttds.begin(), ttds.end());
    std::sort(slips.begin(), slips.end());
    const double p95_ttd = ttds[94];
    const double p99_slip = slips[98];
    REQUIRE(p95_ttd <= 1.40 + 1e-9);
    REQUIRE(p99_slip <= 0.85 + 1e-9);
  }

  // 2. Open Marmara Sea (water_fraction=0.85) wave-glint stress: 300 s (3000 epochs) with 0.25 m/s VIO drift jitter -> 0 false alarms!
  GnssSpoofGuard sea_guard{};
  std::mt19937_64 sea_rng(424242ULL);
  std::normal_distribution<double> sea_w(0.0, 0.14);
  double sea_true_n = 0.0;
  double sea_vio_n = 0.0;
  double sea_vio_e = 0.0;
  for (int k = 0; k < 3000; ++k) {
    const double t = 0.1 * static_cast<double>(k);
    // Slowly varying 0.25 m/s Marmara wave-glint bias over 300 s
    const double wave_bias_e = 0.25 * std::sin(2.0 * 3.14159265358979323846 * t / 90.0);
    sea_true_n += 14.0 * 0.1;
    sea_vio_n += 14.0 * 0.1;
    sea_vio_e += (-wave_bias_e) * 0.1;

    VioStateEstimate vio{};
    vio.stamp_ns = static_cast<uint64_t>(k) * 100'000'000ULL;
    vio.pos_ned = {sea_vio_n, sea_vio_e, -40.0};
    vio.vel_ned = {14.0, -wave_bias_e, 0.0};
    vio.water_fraction = 0.85;
    vio.valid = true;

    GnssFixV116 g{};
    g.stamp_ns = vio.stamp_ns;
    g.pos_ned = {sea_true_n, 0.0, -40.0};
    g.vel_ned = {14.0 + sea_w(sea_rng), sea_w(sea_rng), 0.0};
    g.valid = true;

    const auto st = sea_guard.step_10hz(g, vio);
    REQUIRE(st.active_k_d_mps == Approx(0.35));
    REQUIRE(st.active_h_t_mps == Approx(3.4));
    REQUIRE_FALSE(st.latched);
  }

  // Now inject a real 1.5 m/s spoof over open water -> H3a + H3b corroboration MUST latch cleanly!
  bool sea_caught = false;
  double sea_spoof_e = 0.0;
  for (int k = 3000; k < 3060; ++k) {
    sea_true_n += 14.0 * 0.1;
    sea_vio_n += 14.0 * 0.1;
    sea_spoof_e += 1.5 * 0.1;

    VioStateEstimate vio{};
    vio.stamp_ns = static_cast<uint64_t>(k) * 100'000'000ULL;
    vio.pos_ned = {sea_vio_n, sea_vio_e, -40.0};
    vio.vel_ned = {14.0, 0.0, 0.0};
    vio.water_fraction = 0.85;
    vio.valid = true;

    GnssFixV116 g{};
    g.stamp_ns = vio.stamp_ns;
    g.pos_ned = {sea_true_n, sea_spoof_e, -40.0};
    g.vel_ned = {14.0, 1.5, 0.0};
    g.valid = true;

    const auto st = sea_guard.step_10hz(g, vio);
    if (st.latched) {
      REQUIRE(st.hypothesis == GnssHypothesis::kH3SlowRampSpoof);
      REQUIRE(st.rollback_executed);
      sea_caught = true;
      break;
    }
  }
  REQUIRE(sea_caught);
}
