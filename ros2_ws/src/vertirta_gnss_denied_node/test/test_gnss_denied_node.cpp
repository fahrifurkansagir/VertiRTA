// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "vertirta_gnss_denied_node/gnss_denied_core.hpp"

using namespace vertirta::gnss_denied_node;

TEST_CASE("GnssDeniedCore locks PX4 v1.16 vehicle_gps_position, verified DHMI LTBX ARP, and UXRCE_DDS_SYNCT") {
  REQUIRE(std::string(kTopicVehicleGpsPosition) == "/fmu/out/vehicle_gps_position");
  REQUIRE(std::string(kTopicVehicleGpsPosition).find("vehicle_gnss") == std::string::npos);
  REQUIRE(std::string(kTopicGnssDeniedStatus) == "/vertirta/gnss_denied_status");
  REQUIRE(std::string(kTopicGnssDeniedOdometry) == "/vertirta/gnss_denied_odometry");
  REQUIRE(std::string(kTopicCtrInflationMarkers) == "/vertirta/ctr_inflation_markers");
  REQUIRE(std::string(kTopicGnssSpeedCap) == "/vertirta/gnss_speed_cap_mps");

  // Confirmed DHMI AIP LTBX ARP: 40 deg 59 min 26 sec N, 029 deg 12 min 58 sec E (NEVER stale 40.99889 N!)
  REQUIRE(kVerifiedLtbxArpLatDeg == Catch::Approx(40.990556).margin(1e-6));
  REQUIRE(kVerifiedLtbxArpLonDeg == Catch::Approx(29.216111).margin(1e-6));

  GnssDeniedConfig cfg{};
  REQUIRE(cfg.ekf2_gps_delay_ms == Catch::Approx(110.0));
  REQUIRE(cfg.uxrce_dds_synct_enabled);
}

TEST_CASE("GnssDeniedCore detects H3 slow-ramp spoof <= 1.4 s, rolls back 2.0 s, and escalates FSM to VISUAL_SLZ_ABORT") {
  GnssDeniedConfig cfg{};
  GnssDeniedNodeCore core{cfg};

  const double dt_s = 0.1;  // 10 Hz GNSS/VIO guard update
  double t_s = 0.0;
  double true_e = -500.0;
  double true_n = 200.0;
  double spoof_pos_bias_e = 0.0;

  // 16 s clean nominal warmup over land (water_fraction = 0.05) to fill 15 s lagged baseline.
  for (int i = 0; i < 160; ++i) {
    t_s += dt_s;
    true_e += 12.0 * dt_s;
    GnssDeniedStepInput in{};
    in.t_s = t_s;
    in.dt_s = dt_s;
    in.gps_valid = true;
    in.timesync_ok = true;
    in.cn0_dbhz = 42.0;
    in.hdop = 0.9;
    in.water_fraction = 0.05;
    in.pos_gnss_enu = {true_e, true_n, 45.0};
    in.vel_gnss_enu = {12.0, 0.0, 0.0};
    in.pos_vio_enu = {true_e, true_n, 45.0};
    in.vel_vio_enu = {12.0, 0.0, 0.0};
    const auto st = core.step(in);
    REQUIRE(st.hypothesis == SpoofHypothesis::H0_NOMINAL);
    REQUIRE(st.nav_state == NavFsmState::BOUNDED_GNSS_DENIED_CRUISE);
  }

  // Inject 1.5 m/s slow-ramp spoof over land -> MUST latch H3 in <= 1.4 s with 2.0 s rollback active.
  double ttd_s = 99.0;
  const double spoof_start_s = t_s;
  GnssDeniedOutput latched_out{};
  for (int i = 0; i < 30; ++i) {
    t_s += dt_s;
    true_e += 12.0 * dt_s;
    spoof_pos_bias_e += 1.5 * dt_s;
    GnssDeniedStepInput in{};
    in.t_s = t_s;
    in.dt_s = dt_s;
    in.gps_valid = true;
    in.timesync_ok = true;
    in.cn0_dbhz = 42.0;
    in.hdop = 0.9;
    in.water_fraction = 0.05;
    in.pos_gnss_enu = {true_e + spoof_pos_bias_e, true_n, 45.0};
    in.vel_gnss_enu = {13.5, 0.0, 0.0};
    in.pos_vio_enu = {true_e, true_n, 45.0};
    in.vel_vio_enu = {12.0, 0.0, 0.0};
    latched_out = core.step(in);
    if (latched_out.spoof_latched) {
      ttd_s = t_s - spoof_start_s;
      break;
    }
  }
  REQUIRE(latched_out.spoof_latched);
  REQUIRE(latched_out.hypothesis == SpoofHypothesis::H3_SLOW_RAMP_SPOOF);
  REQUIRE(ttd_s <= 1.401);
  REQUIRE(latched_out.rollback_applied);
  REQUIRE_FALSE(latched_out.gnss_fused_allowed);
  REQUIRE(latched_out.sigma_pos_m >= 3.5);

  // Sustained GNSS denial + reloc dropout over open water -> DEGRADED_CAUTION (<= 15 m/s) -> VISUAL_SLZ_ABORT (sigma >= 65 m).
  bool saw_degraded = false;
  bool saw_abort = false;
  bool saw_positive_dh_dt = false;
  GnssDeniedOutput last{};
  for (int i = 0; i < 6500; ++i) {
    t_s += dt_s;
    true_n += 12.0 * dt_s;
    GnssDeniedStepInput in{};
    in.t_s = t_s;
    in.dt_s = dt_s;
    in.gps_valid = false;
    in.timesync_ok = true;
    in.water_fraction = 0.85;
    in.pos_vio_enu = {true_e, true_n, 45.0};
    in.vel_vio_enu = {0.0, 12.0, 0.0};
    last = core.step(in);
    if (last.dh_dt_mps > 0.0) {
      saw_positive_dh_dt = true;
    }
    if (last.nav_state == NavFsmState::DEGRADED_CAUTION) {
      saw_degraded = true;
      REQUIRE(last.v_max_cap_mps <= 15.0);
    }
    if (last.nav_state == NavFsmState::DETERMINISTIC_VISUAL_SLZ_ABORT) {
      saw_abort = true;
      REQUIRE(last.sigma_pos_m >= 65.0);
      break;
    }
  }
  REQUIRE(saw_degraded);
  REQUIRE(saw_abort);
  REQUIRE(saw_positive_dh_dt);

  const std::string json = core.to_status_json(last);
  REQUIRE(json.find("\"hypothesis\":\"H3_SLOW_RAMP_SPOOF\"") != std::string::npos);
  REQUIRE(json.find("\"nav_fsm_state\":\"DETERMINISTIC_VISUAL_SLZ_ABORT\"") != std::string::npos);
  REQUIRE(json.find("\"dh_dt_mps\":") != std::string::npos);
  REQUIRE(json.find("\"ltfj_inflated_margin_m\":") != std::string::npos);
  REQUIRE(json.find("\"ltbx_inflated_margin_m\":") != std::string::npos);
}

TEST_CASE("GnssDeniedCore suppresses Marmara wave-glint false alarms over open water without corroboration") {
  GnssDeniedConfig cfg{};
  GnssDeniedNodeCore core{cfg};
  const double dt_s = 0.1;
  double t_s = 0.0;
  double e = -600.0;

  for (int i = 0; i < 600; ++i) {
    t_s += dt_s;
    e += 10.0 * dt_s;
    const double wave_jitter = 0.22 * std::sin(t_s * 1.7);
    GnssDeniedStepInput in{};
    in.t_s = t_s;
    in.dt_s = dt_s;
    in.gps_valid = true;
    in.timesync_ok = true;
    in.cn0_dbhz = 41.0;
    in.hdop = 1.0;
    in.water_fraction = 0.85;
    in.pos_gnss_enu = {e, 0.0, 45.0};
    in.vel_gnss_enu = {10.0, 0.0, 0.0};
    in.pos_vio_enu = {e, 0.0, 45.0};
    in.vel_vio_enu = {10.0 + wave_jitter, 0.0, 0.0};
    const auto out = core.step(in);
    REQUIRE_FALSE(out.spoof_latched);
  }
}
