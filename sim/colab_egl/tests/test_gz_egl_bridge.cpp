// SPDX-License-Identifier: Apache-2.0
#include "vertirta/sim/gz_egl_bridge.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("Gimbal quaternion is normalized and aims optical axis at TLOF", "[sim][colab_egl]") {
  // Active inspection orbit geometry: z_inspect = 22 m AGL, r_inspect = 15 m
  // => pitch_down = atan2(22, 15) ~= 0.9721 rad (~55.71 deg below horizon).
  const double pitch_down = 0.9721;
  const auto q = vertirta::sim::ComputeGimbalQuaternionEnu(pitch_down, /*yaw_rad=*/0.0);
  const double norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
  REQUIRE(norm_sq == Catch::Approx(1.0).margin(1e-9));

  const auto opt_axis = vertirta::sim::RotateVectorByQuat(q, 1.0, 0.0, 0.0);
  CHECK(opt_axis[0] == Catch::Approx(std::cos(pitch_down)).margin(1e-9));
  CHECK(opt_axis[1] == Catch::Approx(0.0).margin(1e-9));
  CHECK(opt_axis[2] == Catch::Approx(-std::sin(pitch_down)).margin(1e-9));

  // Pure nadir (pitch_down = pi/2) points straight down (0, 0, -1).
  const auto q_nadir = vertirta::sim::ComputeGimbalQuaternionEnu(0.5 * kPi, 0.75);
  const auto nadir_axis = vertirta::sim::RotateVectorByQuat(q_nadir, 1.0, 0.0, 0.0);
  CHECK(nadir_axis[0] == Catch::Approx(0.0).margin(1e-9));
  CHECK(nadir_axis[1] == Catch::Approx(0.0).margin(1e-9));
  CHECK(nadir_axis[2] == Catch::Approx(-1.0).margin(1e-9));

  // Yaw +90 deg CCW from East rotates horizontal component onto +Y (North).
  const auto q_north = vertirta::sim::ComputeGimbalQuaternionEnu(pitch_down, 0.5 * kPi);
  const auto north_axis = vertirta::sim::RotateVectorByQuat(q_north, 1.0, 0.0, 0.0);
  CHECK(north_axis[0] == Catch::Approx(0.0).margin(1e-9));
  CHECK(north_axis[1] == Catch::Approx(std::cos(pitch_down)).margin(1e-9));
  CHECK(north_axis[2] == Catch::Approx(-std::sin(pitch_down)).margin(1e-9));

  // Non-finite inputs fall back deterministically to unit identity quaternion.
  const auto q_nan =
      vertirta::sim::ComputeGimbalQuaternionEnu(std::numeric_limits<double>::quiet_NaN(), 0.0);
  CHECK(q_nan.w == 1.0);
  CHECK(q_nan.x == 0.0);
  CHECK(q_nan.y == 0.0);
  CHECK(q_nan.z == 0.0);
}

TEST_CASE("GzEglSensorBridge rejects unpaired, malformed, and >2ms skewed RGB-D frames",
          "[sim][colab_egl]") {
  vertirta::sim::GzEglSensorBridge bridge;
  std::vector<uint8_t> rgb;
  std::vector<float> depth;
  uint64_t ts = 0;

  // Uninitialized / empty bridge rejects copy
  REQUIRE_FALSE(bridge.CopyLatestRgbd(rgb, depth, ts));
  REQUIRE_FALSE(bridge.StepSensorRigPose(0.0, 0.0, 22.0, 1.0, 0.0, 0.0, 0.0));

  REQUIRE(bridge.Init("istanbul_coast", "vertirta_sensor_rig"));
  CHECK(bridge.IsInitialized());
  CHECK(bridge.SetPoseServiceName() == "/world/istanbul_coast/set_pose/blocking");

  // Non-finite or zero-norm quaternion pose rejected
  CHECK_FALSE(bridge.StepSensorRigPose(std::numeric_limits<double>::quiet_NaN(), 0.0, 22.0, 1.0,
                                       0.0, 0.0, 0.0));
  CHECK_FALSE(bridge.StepSensorRigPose(0.0, 0.0, 22.0, 0.0, 0.0, 0.0, 0.0));

  // Malformed buffer sizes rejected
  std::vector<uint8_t> bad_rgb(16u, 7u);
  std::vector<float> bad_depth(16u, 22.0f);
  CHECK_FALSE(bridge.InjectTestRgbFrame(bad_rgb, 1'000'000'000ULL));
  CHECK_FALSE(bridge.InjectTestDepthFrame(bad_depth, 1'000'000'000ULL));

  // RGB-only (unpaired) rejected
  std::vector<uint8_t> valid_rgb(vertirta::sim::GzEglSensorBridge::kRgbBytes, 42u);
  std::vector<float> valid_depth(vertirta::sim::GzEglSensorBridge::kDepthPixels, 22.5f);
  REQUIRE(bridge.InjectTestRgbFrame(valid_rgb, 1'000'000'000ULL));
  REQUIRE_FALSE(bridge.CopyLatestRgbd(rgb, depth, ts));

  // Depth with 2.000001 ms skew (> 2 ms lock) rejected
  REQUIRE(bridge.InjectTestDepthFrame(valid_depth, 1'002'000'001ULL));
  REQUIRE_FALSE(bridge.CopyLatestRgbd(rgb, depth, ts));

  // Depth with exact 2.0 ms boundary (<= 2 ms lock) accepted
  REQUIRE(bridge.InjectTestDepthFrame(valid_depth, 1'002'000'000ULL));
  REQUIRE(bridge.CopyLatestRgbd(rgb, depth, ts));
  CHECK(rgb.size() == vertirta::sim::GzEglSensorBridge::kRgbBytes);
  CHECK(depth.size() == vertirta::sim::GzEglSensorBridge::kDepthPixels);
  CHECK(rgb[0] == 42u);
  CHECK(depth[0] == Catch::Approx(22.5f));
  CHECK(ts == 1'002'000'000ULL);
}
