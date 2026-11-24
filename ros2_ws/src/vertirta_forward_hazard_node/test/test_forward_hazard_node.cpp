// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vertirta_forward_hazard_node/forward_hazard_core.hpp"

using namespace vertirta::forward_hazard_node;

TEST_CASE("ForwardHazardCore locks ROS 2 topics, split-stack geometry, and R_veh = 2.10 m") {
  REQUIRE(std::string(kTopicForwardCamera) == "/vertirta/forward_camera/image_raw");
  REQUIRE(std::string(kTopicMmWaveDetections) == "/vertirta/mmwave_77ghz/detections");
  REQUIRE(std::string(kTopicVehicleOdometry) == "/fmu/out/vehicle_odometry");
  REQUIRE(std::string(kTopicImu200Hz) == "/vertirta/imu_200hz");
  REQUIRE(std::string(kTopicWireCapsules) == "/vertirta/wire_capsules");
  REQUIRE(std::string(kTopicIntruders) == "/vertirta/intruders");
  REQUIRE(std::string(kTopicForwardHazardMarkers) == "/vertirta/forward_hazard_markers");
  REQUIRE(std::string(kTopicForwardHazardStatus) == "/vertirta/forward_hazard_status");
  REQUIRE(std::string(kTopicForwardSpeedCap) == "/vertirta/forward_speed_cap_mps");
  REQUIRE(kCanonicalVehicleRadiusM == Catch::Approx(2.10));

  // Verify 1280x320 -> 640x640 split-stack coordinate round-trip < 0.5 px across both halves.
  for (double u : {12.5, 319.25, 639.5, 640.5, 980.75, 1278.5}) {
    for (double v : {0.5, 160.0, 319.5}) {
      const auto stacked = pack_split_stack_1280x320_to_640x640(u, v);
      REQUIRE(stacked.u >= 0.0);
      REQUIRE(stacked.u <= 640.0);
      REQUIRE(stacked.v >= 0.0);
      REQUIRE(stacked.v <= 640.0);
      const auto recovered = unpack_split_stack_640x640_to_1280x320(stacked.u, stacked.v);
      REQUIRE(std::abs(recovered.u - u) < 0.5);
      REQUIRE(std::abs(recovered.v - v) < 0.5);
    }
  }
}

TEST_CASE("ForwardHazardCore enforces <= 64 wire capsules, <= 16 intruders, and <= 15 m/s grazing cap") {
  ForwardHazardConfig cfg{};
  REQUIRE_FALSE(cfg.enable_fp16);  // TU116 safe FP32 default
  ForwardHazardNodeCore core{cfg};

  const uint64_t t0_ns = 10'000'000'000ULL;
  core.push_odometry(OdometrySample{t0_ns, {0.0, 0.0, 45.0}, {12.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
  core.push_imu(ImuSample{t0_ns, 0.0, 0.0, 0.0});

  // Submit 72 candidate wire segments and 22 dynamic intruders -> must strictly cap at 64 and 16.
  ForwardSensorBatch batch{};
  batch.stamp_ns = t0_ns + 50'000'000ULL;
  batch.radar_healthy = true;
  for (uint32_t i = 0; i < 72U; ++i) {
    WireObservation obs{};
    obs.id = 100U + i;
    obs.p0_enu = {40.0 + static_cast<double>(i) * 1.5, -15.0, 45.0};
    obs.p1_enu = {40.0 + static_cast<double>(i) * 1.5, 15.0, 45.0};
    obs.wire_radius_m = 0.02;
    obs.range_sigma_m = 0.25;
    obs.prior_sigma_m = 0.30;
    obs.sag_sigma_m = 0.20;
    obs.voltage_class = (i % 3U == 0U) ? WireVoltageClass::TRANSMISSION : WireVoltageClass::DISTRIBUTION;
    obs.radar_confirmed = true;
    obs.skew_deg = 8.0;
    batch.wires.push_back(obs);
  }
  for (uint32_t j = 0; j < 22U; ++j) {
    IntruderObservation intr{};
    intr.track_id = 500U + j;
    intr.pos_enu = {80.0 + static_cast<double>(j) * 4.0, 10.0, 48.0};
    intr.vel_enu = {-5.0, 1.0, 0.0};
    intr.doppler_mps = -5.0;
    intr.source_type = IntruderSourceType::TRACKER_FUSED;
    batch.intruders.push_back(intr);
  }

  const auto out = core.step(batch);
  REQUIRE(out.wire_msg.ids.size() == 64U);
  REQUIRE(out.intruder_msg.track_ids.size() == 16U);
  REQUIRE(core.wire_table().size() == 64U);
  REQUIRE(core.intruder_tracker().active_count() == 16U);
  REQUIRE_FALSE(out.wire_msg.stale);
  REQUIRE_FALSE(out.radar_grazing_fallback);
  REQUIRE(out.v_max_cap_mps == Catch::Approx(22.0));
  for (float r : out.wire_msg.radius_m) {
    REQUIRE(static_cast<double>(r) > kCanonicalVehicleRadiusM);
  }

  // Grazing skew > 20 deg without radar confirmation MUST engage <= 15.0 m/s CBVF speed cap.
  ForwardSensorBatch grazing_batch{};
  grazing_batch.stamp_ns = t0_ns + 150'000'000ULL;
  grazing_batch.radar_healthy = true;
  WireObservation grazing_wire{};
  grazing_wire.id = 900U;
  grazing_wire.p0_enu = {55.0, -25.0, 45.0};
  grazing_wire.p1_enu = {75.0, 25.0, 45.0};
  grazing_wire.radar_confirmed = false;
  grazing_wire.skew_deg = 28.0;
  grazing_wire.voltage_class = WireVoltageClass::CRANE_CABLE;
  grazing_batch.wires.push_back(grazing_wire);

  const auto out_grazing = core.step(grazing_batch);
  REQUIRE(out_grazing.radar_grazing_fallback);
  REQUIRE(out_grazing.v_max_cap_mps <= 15.0);
  const std::string status_json = core.to_status_json(out_grazing);
  REQUIRE(status_json.find("\"radar_grazing_fallback\":true") != std::string::npos);
  REQUIRE(status_json.find("\"v_max_cap_mps\":15.00") != std::string::npos);
  REQUIRE(status_json.find("\"wire_count\":1") != std::string::npos);
}
