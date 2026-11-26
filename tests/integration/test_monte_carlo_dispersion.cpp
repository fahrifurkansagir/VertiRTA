// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "vertirta/benchmarks/mc_1000_runner.hpp"

using Catch::Approx;
using namespace vertirta::benchmarks;

TEST_CASE("Exact 95% Wilson and Clopper-Pearson binomial confidence intervals",
          "[integration][monte_carlo][stats]") {
  // Zero breaches in 1,000 flights -> exact 1-sided 95% upper bound = 1 - 0.05^(1/1000) ~ 0.0029912
  // Two-sided 95% upper bound = 1 - 0.025^(1/1000) ~ 0.0036821
  const auto ci0 = compute_binomial_ci95(0U, 1000U);
  REQUIRE(ci0.rate == Approx(0.0).margin(1e-12));
  REQUIRE(ci0.clopper_pearson_lo == Approx(0.0).margin(1e-12));
  REQUIRE(ci0.clopper_pearson_hi == Approx(0.00368209).margin(1e-5));
  REQUIRE(ci0.clopper_pearson_one_sided_hi == Approx(0.00299124).margin(1e-5));
  REQUIRE(ci0.wilson_lo == Approx(0.0).margin(1e-12));
  REQUIRE(ci0.wilson_hi > 0.0035);
  REQUIRE(ci0.wilson_hi < 0.0040);

  // 1000/1000 safe -> exact 2-sided 95% lower bound = 0.025^(1/1000) ~ 0.9963179
  const auto ci1 = compute_binomial_ci95(1000U, 1000U);
  REQUIRE(ci1.rate == Approx(1.0));
  REQUIRE(ci1.clopper_pearson_lo == Approx(0.99631791).margin(1e-5));
  REQUIRE(ci1.clopper_pearson_hi == Approx(1.0));

  // Interior case: 98/100 -> Clopper-Pearson 95% CI ~ [0.9296, 0.9976]
  const auto ci98 = compute_binomial_ci95(98U, 100U);
  REQUIRE(ci98.clopper_pearson_lo == Approx(0.9296).margin(5e-4));
  REQUIRE(ci98.clopper_pearson_hi == Approx(0.9976).margin(5e-4));
  REQUIRE(ci98.wilson_lo < ci98.rate);
  REQUIRE(ci98.wilson_hi > ci98.rate);
}

TEST_CASE("Deterministic 50-flight Monte Carlo dispersion subset executes cleanly with zero airspace breaches",
          "[integration][monte_carlo][subset50]") {
  MonteCarloCampaignConfig cfg{};
  cfg.master_seed = 20261125ULL;
  cfg.flights_per_regime_per_airframe = 5U;  // 5 x 5 regimes x 2 airframes = 50 flights
  cfg.stress_seeds_per_cell = 1U;            // 6 x 6 x 1 = 36 extreme boundary flights
  cfg.num_threads = 4U;

  const auto rep = run_monte_carlo_campaign(cfg);
  REQUIRE(rep.operational_flights_total == 50U);
  REQUIRE(rep.stress_flights_total == 36U);
  REQUIRE(rep.overall.flights == 50U);
  REQUIRE(rep.overall.airspace_breaches == 0U);
  // Honest physics: LC 8+1 achieves 100% (25/25), X8 non-CEL achieves 100% (20/20),
  // while X8 Same-Arm CEL under random +/-5 cm CG offset exposes genuine unactuated-axis sensitivity.
  REQUIRE(rep.overall.safe_recoveries >= 45U);
  REQUIRE(rep.by_airframe[0].flights == 25U);
  REQUIRE(rep.by_airframe[1].flights == 25U);
  REQUIRE(rep.by_airframe[1].safe_recoveries == 25U);
  REQUIRE(rep.gnss_spoof_summary.spoof_flights == 10U);
  REQUIRE(rep.gnss_spoof_summary.detected_count == 10U);
  REQUIRE(rep.gnss_spoof_summary.rollback_ok_count == 10U);
  REQUIRE(rep.gnss_spoof_summary.max_ttd_s <= 2.2);
  REQUIRE(rep.report_sha256.size() == 64U);

  // Honest failure boundary check: both green safe region AND red physical thrust-saturation wall exist!
  REQUIRE(rep.failure_boundary.green_cells_100pct >= 8U);
  REQUIRE(rep.failure_boundary.red_cells_0pct >= 12U);
  REQUIRE(rep.failure_boundary.failed_stress_flights >= 12U);
}

TEST_CASE("Extreme +35% mass overload under same-arm dual-kill genuinely triggers thrust-saturation breakdown",
          "[integration][monte_carlo][failure_boundary]") {
  FlightSampleConfig extreme{};
  extreme.flight_id = 999999U;
  extreme.seed = 2026112599ULL;
  extreme.airframe = AirframeKind::kCoaxialX8_240;
  extreme.regime = FaultRegimeKind::kSameArmDualKillCel;
  extreme.mass_kg = 324.0;  // +35% overload > 2-arm physical thrust ceiling (~2,943 N = 300 kg)
  extreme.cg_x_m = 0.03;
  extreme.cg_y_m = -0.03;
  extreme.wind_speed_mps = 24.0;
  extreme.wind_dir_rad = 3.14159265;
  extreme.gust_sigma_scale = 1.25;
  extreme.init_agl_m = 45.0;
  extreme.init_east_m = -220.0;
  extreme.init_soc = 0.42;
  extreme.fault_time_s = 1.5;
  extreme.fault_index = 0U;

  const auto out = run_single_flight(extreme);
  REQUIRE(out.cel_latched);
  REQUIRE_FALSE(out.safe_recovery);
  REQUIRE(out.failure_mode != "NONE");
  REQUIRE((out.touchdown_vz_mps > 3.5 || out.max_tilt_deg > 60.0 || out.battery_collapse));
}

TEST_CASE("Sealed 1,000-flight campaign artifact passes SHA-256 and pure UTF-8 LF structural integrity",
          "[integration][monte_carlo][artifact]") {
#ifdef VERTIRTA_REPO_ROOT
  const std::filesystem::path artifact_path =
      std::filesystem::path(VERTIRTA_REPO_ROOT) /
      "artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json";
#else
  const std::filesystem::path artifact_path =
      "artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json";
#endif

  REQUIRE(std::filesystem::exists(artifact_path));
  REQUIRE(std::filesystem::file_size(artifact_path) < 500U * 1024U);

  std::string err;
  INFO("SHA-256 verify error: " << err);
  REQUIRE(verify_mc_report_json_sha256(artifact_path, &err));

  std::ifstream ifs(artifact_path, std::ios::binary);
  REQUIRE(ifs.is_open());
  const std::string raw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  REQUIRE(raw.find('\r') == std::string::npos);
  REQUIRE(raw.find(": nan") == std::string::npos);
  REQUIRE(raw.find(",nan") == std::string::npos);
  REQUIRE(raw.find("[nan") == std::string::npos);
  REQUIRE(raw.find("NaN") == std::string::npos);
  REQUIRE(raw.find(": inf") == std::string::npos);
  REQUIRE(raw.find(",inf") == std::string::npos);
  REQUIRE(raw.find("Infinity") == std::string::npos);
  REQUIRE(raw.find("\"operational_flights_total\": 1000") != std::string::npos);
  REQUIRE(raw.find("\"stress_flights_total\": 180") != std::string::npos);
  REQUIRE(raw.find("vertirta.monte_carlo_1000_dispersion_report/v4.0.0") != std::string::npos);
}

TEST_CASE("X8 same-arm CEL exhibits exact directional CG-axis sensitivity (cg_v safe vs cg_u cliff)",
          "[integration][monte_carlo][cg_physics]") {
  // Arm 0 lies along e_u = (+1, +1)/sqrt(2); orthogonal actuated axis is e_v = (-1, +1)/sqrt(2).
  // Bidirectional surviving-arm torque tau_u rejects +/-5 cm cg_v, whereas unactuated-axis torque
  // tau_v = -L*T_opp - cg_u*T_tot triggers genuine rigid-body tipping beyond cg_u >= +1.25 cm.
  constexpr double kInvSqrt2 = 0.70710678118654752440;
  auto make_hover_cel = [](AirframeKind af, double cgx, double cgy) {
    FlightSampleConfig sc{};
    sc.flight_id = 777000U;
    sc.seed = 2026112577ULL;
    sc.airframe = af;
    sc.regime = FaultRegimeKind::kSameArmDualKillCel;
    sc.mass_kg = 240.0;
    sc.cg_x_m = cgx;
    sc.cg_y_m = cgy;
    sc.wind_speed_mps = 0.0;
    sc.gust_sigma_scale = 0.0;
    sc.init_agl_m = 45.0;
    sc.init_east_m = -220.0;
    sc.init_north_m = 0.0;
    sc.init_vx_mps = 0.0;
    sc.slz_offset_n_m = 0.0;
    sc.init_soc = 0.70;
    sc.fault_time_s = 0.20;
    sc.fault_index = 0U;
    sc.imu_gyro_sigma_rads = 0.0;
    sc.alt_noise_sigma_m = 0.0;
    return sc;
  };

  // 1. Nominal CG (0 cm) hover CEL survives cleanly
  const auto o_nom = run_single_flight(make_hover_cel(AirframeKind::kCoaxialX8_240, 0.0, 0.0));
  REQUIRE(o_nom.cel_latched);
  REQUIRE(o_nom.safe_recovery);
  REQUIRE(o_nom.max_tilt_deg < 45.0);
  REQUIRE(o_nom.touchdown_vz_mps < 1.0);

  // 2. +/-5 cm CG shift purely along actuated orthogonal axis e_v survives cleanly
  const auto o_cgv_pos = run_single_flight(
      make_hover_cel(AirframeKind::kCoaxialX8_240, -0.05 * kInvSqrt2, +0.05 * kInvSqrt2));
  const auto o_cgv_neg = run_single_flight(
      make_hover_cel(AirframeKind::kCoaxialX8_240, +0.05 * kInvSqrt2, -0.05 * kInvSqrt2));
  REQUIRE(o_cgv_pos.safe_recovery);
  REQUIRE(o_cgv_pos.max_tilt_deg < 45.0);
  REQUIRE(o_cgv_neg.safe_recovery);
  REQUIRE(o_cgv_neg.max_tilt_deg < 45.0);

  // 3. +2.5 cm CG shift toward dead Arm 0 (cg_u > +1.25 cm cliff) triggers genuine attitude upset
  const auto o_cgu_cliff = run_single_flight(
      make_hover_cel(AirframeKind::kCoaxialX8_240, +0.025 * kInvSqrt2, +0.025 * kInvSqrt2));
  REQUIRE(o_cgu_cliff.cel_latched);
  REQUIRE_FALSE(o_cgu_cliff.safe_recovery);
  REQUIRE(o_cgu_cliff.failure_mode == "ATTITUDE_UPSET");
  REQUIRE(o_cgu_cliff.max_tilt_deg > 60.0);

  // 4. Lift+Cruise 8+1 retains full rank-4 bidirectional control under corner (+5 cm, +5 cm) CG shift
  const auto o_lc = run_single_flight(make_hover_cel(AirframeKind::kLiftCruise_240_8p1, 0.05, 0.05));
  REQUIRE(o_lc.safe_recovery);
  REQUIRE(o_lc.max_tilt_deg < 35.0);
  REQUIRE(o_lc.touchdown_vz_mps < 1.5);
}
