// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "vertirta/sim/mmwave_77ghz_physics.hpp"
#include "vertirta/sim/mmwave_77ghz_sensor.hpp"

using namespace vertirta::sim;

namespace {

constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

}  // namespace

TEST_CASE("77 GHz normal-incidence Fresnel wire glint matches +1.96..+5.94 dBsm at 100 m and 1/R^3 slope") {
  MmWave77GHzPhysics radar(MmWaveRadarConfig::make_tier1b_cascaded());
  REQUIRE(radar.wavelength_m() == Catch::Approx(0.0038934).margin(1e-5));

  const double rcs_10 = radar.compute_wire_rcs_dbsm(0.010, 100.0, 0.0, WireKind::kStrandedAcsr);
  const double rcs_15 = radar.compute_wire_rcs_dbsm(0.015, 100.0, 0.0, WireKind::kStrandedAcsr);
  const double rcs_20 = radar.compute_wire_rcs_dbsm(0.020, 100.0, 0.0, WireKind::kStrandedAcsr);
  const double rcs_25 = radar.compute_wire_rcs_dbsm(0.025, 100.0, 0.0, WireKind::kStrandedAcsr);

  REQUIRE(rcs_10 == Catch::Approx(1.96).margin(0.4));
  REQUIRE(rcs_15 == Catch::Approx(3.72).margin(0.4));
  REQUIRE(rcs_20 == Catch::Approx(4.97).margin(0.4));
  REQUIRE(rcs_25 == Catch::Approx(5.94).margin(0.4));

  // Verify 1/R^3 received-power scaling (~9.03 dB per 2x range change) vs 1/R^4 point target (~12.04 dB).
  const double pr_wire_50 = radar.compute_received_power_dbm(
      radar.compute_wire_rcs_dbsm(0.015, 50.0, 0.0, WireKind::kStrandedAcsr), 50.0);
  const double pr_wire_100 = radar.compute_received_power_dbm(
      radar.compute_wire_rcs_dbsm(0.015, 100.0, 0.0, WireKind::kStrandedAcsr), 100.0);
  REQUIRE((pr_wire_50 - pr_wire_100) == Catch::Approx(9.03).margin(0.15));

  const double pr_pt_50 = radar.compute_received_power_dbm(-18.0, 50.0);
  const double pr_pt_100 = radar.compute_received_power_dbm(-18.0, 100.0);
  REQUIRE((pr_pt_50 - pr_pt_100) == Catch::Approx(12.04).margin(0.05));
}

TEST_CASE("ACSR stranded Bragg lobes appear in 7..14.1 deg, vanish >15 deg, smooth crane rope lacks Bragg, and >20 deg plunges <=-45 dBsm") {
  MmWave77GHzPhysics radar(MmWaveRadarConfig::make_tier1b_cascaded());

  // 1st Bragg order (~7.0 deg) and 2nd Bragg order (~14.1 deg) for p=16 mm ACSR
  const double rcs_acsr_7 = radar.compute_wire_rcs_dbsm(0.015, 60.0, 7.0, WireKind::kStrandedAcsr, 0.016);
  const double rcs_acsr_14 = radar.compute_wire_rcs_dbsm(0.015, 60.0, 14.1, WireKind::kStrandedAcsr, 0.016);
  const double rcs_acsr_16 = radar.compute_wire_rcs_dbsm(0.015, 60.0, 16.0, WireKind::kStrandedAcsr, 0.016);
  const double rcs_smooth_7 = radar.compute_wire_rcs_dbsm(0.015, 60.0, 7.0, WireKind::kSmoothCraneRope, 0.016);
  const double rcs_smooth_14 = radar.compute_wire_rcs_dbsm(0.015, 60.0, 14.1, WireKind::kSmoothCraneRope, 0.016);

  REQUIRE(rcs_acsr_7 >= -14.0);
  REQUIRE(rcs_acsr_7 <= -6.0);
  REQUIRE(rcs_acsr_14 >= -15.0);
  REQUIRE(rcs_acsr_14 <= -6.0);
  REQUIRE(rcs_acsr_7 > rcs_smooth_7 + 12.0);
  REQUIRE(rcs_acsr_14 > rcs_smooth_14 + 12.0);
  REQUIRE(rcs_acsr_16 < -30.0);  // Bragg ceased above 15.0 deg

  // Grazing cliff past 20.0 deg skew -> <= -45 dBsm and zero Tier-1B detections at 60 m
  const double rcs_skew_21 = radar.compute_wire_rcs_dbsm(0.015, 60.0, 21.0, WireKind::kStrandedAcsr, 0.016);
  REQUIRE(rcs_skew_21 <= -45.0);

  const std::array<WireTargetDesc, 2> wires{{
      WireTargetDesc{101u, Vec3f64{60.0, 0.0, 0.0}, 0.015, 0.0, 0.016, 0.0, WireKind::kStrandedAcsr},
      WireTargetDesc{102u, Vec3f64{60.0, 2.0, 0.0}, 0.015, 22.0, 0.016, 0.0, WireKind::kStrandedAcsr},
  }};
  std::array<RadarDetection, 16> dets{};
  const std::size_t n = radar.simulate_scan(Vec3f64{18.0, 0.0, 0.0}, wires, {}, dets, true);
  REQUIRE(n == 1u);
  REQUIRE(dets[0].target_id == 101u);  // Normal glint detected; >20 deg skewed wire dropped
  REQUIRE(dets[0].is_wire_return);
}

TEST_CASE("Swerling 0/III/I required SNR thresholds and ego-motion Doppler azimuth-leakage gate match DR-10") {
  REQUIRE(MmWave77GHzPhysics::required_snr_db(SwerlingModel::kSwerling0) == Catch::Approx(8.5).margin(0.3));
  REQUIRE(MmWave77GHzPhysics::required_snr_db(SwerlingModel::kSwerlingIII) == Catch::Approx(13.0).margin(0.3));
  REQUIRE(MmWave77GHzPhysics::required_snr_db(SwerlingModel::kSwerlingI) == Catch::Approx(17.0).margin(0.3));

  MmWave77GHzPhysics radar(MmWaveRadarConfig::make_tier1b_cascaded());
  const Vec3f64 v_ego{20.0, 0.0, 0.0};

  // Boresight static pole: v_r_meas = -20.0 -> v_res = 0.0, gate = +-0.64 m/s
  const auto boresight = radar.classify_doppler_target(-20.0, v_ego, 0.0, 0.0);
  REQUIRE(boresight.doppler_residual_mps == Catch::Approx(0.0).margin(1e-6));
  REQUIRE(boresight.static_gate_3sigma_mps == Catch::Approx(0.64).margin(0.05));
  REQUIRE(boresight.target_class == DopplerTargetClass::kStaticOrHoverCandidate);

  // Off-boresight 30 deg static pole at 20 m/s cruise: widened gate +-1.01 m/s
  const double az30 = 30.0 * kDegToRad;
  const double vr_static_30 = -20.0 * std::cos(az30);
  const auto off30 = radar.classify_doppler_target(vr_static_30 + 0.80, v_ego, az30, 0.0);
  REQUIRE(off30.static_gate_3sigma_mps == Catch::Approx(1.01).margin(0.05));
  REQUIRE(off30.target_class == DopplerTargetClass::kStaticOrHoverCandidate);

  // Moving rogue drone (v_res = 2.5 m/s) classifies cleanly as DYNAMIC_INTRUDER
  const auto drone = radar.classify_doppler_target(vr_static_30 + 2.5, v_ego, az30, 0.0);
  REQUIRE(drone.doppler_residual_mps == Catch::Approx(2.5).margin(1e-6));
  REQUIRE(drone.target_class == DopplerTargetClass::kDynamicIntruder);

  // Sensor bridge scan + deterministic repeatability check
  MmWave77GHzSensorBridge bridge(MmWaveRadarConfig::make_tier2_active_pa());
  const std::array<PointTargetDesc, 1> pts{{
      PointTargetDesc{201u, Vec3f64{80.0, 10.0, 0.0}, Vec3f64{-5.0, 0.0, 0.0}, -18.0, SwerlingModel::kSwerlingI},
  }};
  const std::size_t k = bridge.step_scan(1000000ULL, v_ego, {}, pts, true);
  REQUIRE(k == 1u);
  REQUIRE(bridge.last_detections()[0].target_id == 201u);
  REQUIRE(bridge.last_detections()[0].doppler_class == DopplerTargetClass::kDynamicIntruder);
}

TEST_CASE("Gazebo Harmonic mmWave 77 GHz SDF snippet has pure LF and declares custom gz:type=mmwave_77ghz") {
  const std::string sdf_path =
      std::string(VERTIRTA_REPO_ROOT) + "/sim/gazebo_plugins/mmwave_77ghz/sdf/mmwave_77ghz_sensor_snippet.sdf";
  std::ifstream in(sdf_path, std::ios::binary);
  REQUIRE(in.is_open());
  std::ostringstream ss;
  ss << in.rdbuf();
  const std::string text = ss.str();

  REQUIRE(text.find('\r') == std::string::npos);
  REQUIRE(text.find("<?xml version=\"1.0\"") != std::string::npos);
  REQUIRE(text.find("<sensor name=\"vertirta_mmwave_77ghz\" type=\"custom\" gz:type=\"mmwave_77ghz\">") != std::string::npos);
  REQUIRE(text.find("</sdf>") != std::string::npos);
}
