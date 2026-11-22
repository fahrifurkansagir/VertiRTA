// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#ifdef VERTIRTA_HAS_GZ_TRANSPORT
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/image.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/transport/Node.hh>
#endif

namespace vertirta::sim {

/// Hamilton scalar-first unit quaternion (w, x, y, z) in ENU world frame.
struct QuatWxyz {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

/// Compute unit ENU quaternion for a 2-axis gimbal pointing camera optical +X axis
/// at azimuth `yaw_rad` (CCW from East) and depression angle `pitch_down_rad` below horizon
/// (`pitch_down_rad = pi/2` is straight-down nadir; `atan2(22, 15) ~= 0.9721 rad` aims at
/// TLOF center from a 22 m AGL / 15 m radius inspection orbit).
[[nodiscard]] QuatWxyz ComputeGimbalQuaternionEnu(double pitch_down_rad,
                                                  double yaw_rad) noexcept;

/// Rotate a 3D vector by unit quaternion `q` (pure helper for geometry/FOV checks).
[[nodiscard]] std::array<double, 3> RotateVectorByQuat(const QuatWxyz& q,
                                                       double vx,
                                                       double vy,
                                                       double vz) noexcept;

/// Sensor-only Gazebo Harmonic EGL transport bridge for kinematic `vertirta_sensor_rig`.
/// Subscribes to raw `/slz_rgbd/image` + `/slz_rgbd/depth_image` (bypassing gz-sensors#545
/// PointCloud2 axis bug) and enforces strict RGB-D timestamp pairing `|t_rgb - t_depth| <= 2 ms`.
class GzEglSensorBridge {
 public:
  static constexpr uint32_t kWidth = 512u;
  static constexpr uint32_t kHeight = 512u;
  static constexpr size_t kRgbBytes =
      static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight) * 3u;
  static constexpr size_t kDepthPixels =
      static_cast<size_t>(kWidth) * static_cast<size_t>(kHeight);
  static constexpr uint64_t kMaxPairSkewNs = 2'000'000ULL;  // 2 ms pairing contract

  GzEglSensorBridge() = default;
  ~GzEglSensorBridge() = default;

  GzEglSensorBridge(const GzEglSensorBridge&) = delete;
  GzEglSensorBridge& operator=(const GzEglSensorBridge&) = delete;
  GzEglSensorBridge(GzEglSensorBridge&&) = delete;
  GzEglSensorBridge& operator=(GzEglSensorBridge&&) = delete;

  /// Initialize transport subscriptions and `/world/<world>/set_pose/blocking` service path.
  bool Init(const std::string& world = "istanbul_coast",
            const std::string& rig_model_name = "vertirta_sensor_rig");

  /// Synchronously teleport kinematic `vertirta_sensor_rig` via Gazebo `set_pose/blocking`.
  bool StepSensorRigPose(double x,
                         double y,
                         double z,
                         double qw,
                         double qx,
                         double qy,
                         double qz);

  /// Copy latest RGB + depth pair ONLY when both frames are present and
  /// `|t_rgb - t_depth| <= kMaxPairSkewNs` (2 ms).
  [[nodiscard]] bool CopyLatestRgbd(std::vector<uint8_t>& rgb_out,
                                    std::vector<float>& depth_out,
                                    uint64_t& stamp_ns_out) const;

  /// Deterministic frame injection hooks for unit tests & offline replay verification.
  bool InjectTestRgbFrame(std::span<const uint8_t> rgb, uint64_t stamp_ns);
  bool InjectTestDepthFrame(std::span<const float> depth, uint64_t stamp_ns);

  [[nodiscard]] static constexpr bool HasGzTransport() noexcept {
#ifdef VERTIRTA_HAS_GZ_TRANSPORT
    return true;
#else
    return false;
#endif
  }

  [[nodiscard]] bool IsInitialized() const noexcept { return initialized_.load(); }
  [[nodiscard]] const std::string& SetPoseServiceName() const noexcept { return set_pose_srv_; }
  [[nodiscard]] static constexpr uint64_t MaxPairSkewNs() noexcept { return kMaxPairSkewNs; }

 private:
#ifdef VERTIRTA_HAS_GZ_TRANSPORT
  void OnRgb(const gz::msgs::Image& msg);
  void OnDepth(const gz::msgs::Image& msg);
  gz::transport::Node node_;
#endif

  std::string set_pose_srv_{"/world/istanbul_coast/set_pose/blocking"};
  std::string rig_model_name_{"vertirta_sensor_rig"};
  mutable std::mutex mtx_;
  std::vector<uint8_t> rgb_buf_;
  std::vector<float> depth_buf_;
  uint64_t rgb_stamp_ns_{0};
  uint64_t depth_stamp_ns_{0};
  bool has_rgb_{false};
  bool has_depth_{false};
  std::array<double, 7> last_pose_enu_{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  std::atomic<bool> initialized_{false};
};

}  // namespace vertirta::sim
