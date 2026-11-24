// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/intruder_kf.hpp"
#include "vertirta/rta/wire_capsules.hpp"

namespace vertirta::forward_hazard_node {

// Canonical M3 ROS 2 topic contract (Plan 6A Task 5)
inline constexpr const char* kTopicForwardCamera = "/vertirta/forward_camera/image_raw";
inline constexpr const char* kTopicMmWaveDetections = "/vertirta/mmwave_77ghz/detections";
inline constexpr const char* kTopicVehicleOdometry = "/fmu/out/vehicle_odometry";
inline constexpr const char* kTopicImu200Hz = "/vertirta/imu_200hz";
inline constexpr const char* kTopicWireCapsules = "/vertirta/wire_capsules";
inline constexpr const char* kTopicIntruders = "/vertirta/intruders";
inline constexpr const char* kTopicForwardHazardMarkers = "/vertirta/forward_hazard_markers";
inline constexpr const char* kTopicForwardHazardStatus = "/vertirta/forward_hazard_status";
inline constexpr const char* kTopicForwardSpeedCap = "/vertirta/forward_speed_cap_mps";

inline constexpr double kCanonicalVehicleRadiusM = 2.10;
inline constexpr std::size_t kMaxWireCapsules = vertirta::rta::WireCapsuleTable::kMaxCapsules;
inline constexpr std::size_t kMaxIntruders = vertirta::rta::IntruderTrackerKf::kMaxTracks;
inline constexpr std::size_t kPoseRingCapacity = 64U;

enum class WireVoltageClass : uint8_t {
  DISTRIBUTION = 0U,
  TRANSMISSION = 1U,
  CRANE_CABLE = 2U,
};

enum class IntruderSourceType : uint8_t {
  ADSB = 0U,
  REMOTE_ID = 1U,
  TRACKER_FUSED = 2U,
};

struct Pixel2D {
  double u{0.0};
  double v{0.0};
};

// Exact lossless 1280x320 horizon crop -> 640x640 split-stack packing:
//   Left half  u in [0, 640),   v in [0, 320) -> top    (u,         v)
//   Right half u in [640, 1280], v in [0, 320) -> bottom (u - 640.0, v + 320.0)
[[nodiscard]] constexpr Pixel2D pack_split_stack_1280x320_to_640x640(double u, double v) noexcept {
  const double uc = std::clamp(u, 0.0, 1280.0);
  const double vc = std::clamp(v, 0.0, 320.0);
  if (uc < 640.0) {
    return Pixel2D{uc, vc};
  }
  return Pixel2D{uc - 640.0, vc + 320.0};
}

[[nodiscard]] constexpr Pixel2D unpack_split_stack_640x640_to_1280x320(double su, double sv) noexcept {
  const double suc = std::clamp(su, 0.0, 640.0);
  const double svc = std::clamp(sv, 0.0, 640.0);
  if (svc < 320.0) {
    return Pixel2D{suc, svc};
  }
  return Pixel2D{suc + 640.0, svc - 320.0};
}

struct ForwardHazardConfig {
  double r_veh_m{kCanonicalVehicleRadiusM};
  double v_max_nominal_mps{22.0};
  double v_max_radar_degraded_mps{15.0};
  double radar_grazing_skew_deg{20.0};
  double stale_timeout_s{0.25};
  bool enable_fp16{false};  // TU116 safe FP32 default
};

struct OdometrySample {
  uint64_t stamp_ns{0U};
  core::Vec3Enu pos_enu{};
  core::Vec3Enu vel_enu{};
  double roll_rad{0.0};
  double pitch_rad{0.0};
  double yaw_rad{0.0};
};

struct ImuSample {
  uint64_t stamp_ns{0U};
  double roll_rate_rps{0.0};
  double pitch_rate_rps{0.0};
  double yaw_rate_rps{0.0};
};

struct WireObservation {
  uint32_t id{0U};
  core::Vec3Enu p0_enu{};
  core::Vec3Enu p1_enu{};
  double wire_radius_m{0.02};
  double prior_sigma_m{0.35};
  double range_sigma_m{0.25};
  double sag_sigma_m{0.20};
  double skew_deg{0.0};
  bool radar_confirmed{true};
  WireVoltageClass voltage_class{WireVoltageClass::DISTRIBUTION};
};

struct IntruderObservation {
  uint32_t track_id{0U};
  core::Vec3Enu pos_enu{};
  core::Vec3Enu vel_enu{};
  double radius_horiz_m{12.0};
  double half_height_m{6.0};
  double doppler_mps{0.0};
  IntruderSourceType source_type{IntruderSourceType::TRACKER_FUSED};
};

struct ForwardSensorBatch {
  uint64_t stamp_ns{0U};
  bool radar_healthy{true};
  std::vector<WireObservation> wires{};
  std::vector<IntruderObservation> intruders{};
};

// Wire-compatible POD mirrors of ros2_ws/src/vertirta_msgs/msg/{WireCapsuleArray,IntruderArray}.msg
struct WireCapsuleArrayOut {
  uint64_t stamp_ns{0U};
  bool stale{false};
  std::vector<uint32_t> ids{};
  std::vector<core::Vec3Enu> p0_enu{};
  std::vector<core::Vec3Enu> p1_enu{};
  std::vector<float> radius_m{};
  std::vector<uint8_t> voltage_class{};
};

struct IntruderArrayOut {
  uint64_t stamp_ns{0U};
  bool stale{false};
  std::vector<uint32_t> track_ids{};
  std::vector<core::Vec3Enu> pos_enu{};
  std::vector<core::Vec3Enu> vel_enu{};
  std::vector<float> radius_horiz_m{};
  std::vector<float> half_height_m{};
  std::vector<float> age_s{};
  std::vector<uint8_t> source_type{};
};

struct ForwardHazardStepOutput {
  WireCapsuleArrayOut wire_msg{};
  IntruderArrayOut intruder_msg{};
  double v_max_cap_mps{22.0};
  bool radar_grazing_fallback{false};
  bool radar_degraded{false};
  double sensor_age_ms{0.0};
};

class ForwardHazardNodeCore {
 public:
  explicit ForwardHazardNodeCore(ForwardHazardConfig cfg = {}) noexcept : cfg_(cfg) {}

  void push_odometry(const OdometrySample& sample) noexcept {
    odom_ring_[odom_head_] = sample;
    odom_head_ = (odom_head_ + 1U) % kPoseRingCapacity;
    if (odom_count_ < kPoseRingCapacity) {
      ++odom_count_;
    }
    last_odom_ = sample;
    has_odom_ = true;
  }

  void push_imu(const ImuSample& sample) noexcept {
    imu_ring_[imu_head_] = sample;
    imu_head_ = (imu_head_ + 1U) % kPoseRingCapacity;
    if (imu_count_ < kPoseRingCapacity) {
      ++imu_count_;
    }
    last_imu_ = sample;
  }

  [[nodiscard]] const vertirta::rta::WireCapsuleTable& wire_table() const noexcept {
    return wire_table_;
  }

  [[nodiscard]] const vertirta::rta::IntruderTrackerKf& intruder_tracker() const noexcept {
    return intruder_kf_;
  }

  [[nodiscard]] const ForwardHazardConfig& config() const noexcept { return cfg_; }

  [[nodiscard]] ForwardHazardStepOutput step(const ForwardSensorBatch& batch) {
    ForwardHazardStepOutput out{};
    out.wire_msg.stamp_ns = batch.stamp_ns;
    out.intruder_msg.stamp_ns = batch.stamp_ns;

    double age_s = 0.0;
    if (has_odom_ && batch.stamp_ns >= last_odom_.stamp_ns) {
      age_s = static_cast<double>(batch.stamp_ns - last_odom_.stamp_ns) * 1e-9;
    }
    out.sensor_age_ms = age_s * 1000.0;
    const bool stale = (age_s > cfg_.stale_timeout_s);
    out.wire_msg.stale = stale;
    out.intruder_msg.stale = stale;

    wire_table_.clear();
    intruder_kf_.clear();

    bool grazing_fallback = false;
    const std::size_t wire_n = std::min(batch.wires.size(), kMaxWireCapsules);
    out.wire_msg.ids.reserve(wire_n);
    out.wire_msg.p0_enu.reserve(wire_n);
    out.wire_msg.p1_enu.reserve(wire_n);
    out.wire_msg.radius_m.reserve(wire_n);
    out.wire_msg.voltage_class.reserve(wire_n);

    for (std::size_t i = 0U; i < wire_n; ++i) {
      const auto& w = batch.wires[i];
      const bool unconfirmed_grazing =
          (!w.radar_confirmed) && (std::abs(w.skew_deg) > cfg_.radar_grazing_skew_deg);
      if (unconfirmed_grazing || !batch.radar_healthy) {
        grazing_fallback = true;
      }

      const double sigma_range = w.radar_confirmed ? w.range_sigma_m : std::max(w.range_sigma_m, 1.25);
      const double r_cap = std::max(
          cfg_.r_veh_m + 0.25,
          w.wire_radius_m + cfg_.r_veh_m +
              3.0 * std::hypot(w.prior_sigma_m, sigma_range) + w.sag_sigma_m);

      if (wire_table_.add_capsule(w.p0_enu, w.p1_enu, r_cap, w.id)) {
        out.wire_msg.ids.push_back(w.id);
        out.wire_msg.p0_enu.push_back(w.p0_enu);
        out.wire_msg.p1_enu.push_back(w.p1_enu);
        out.wire_msg.radius_m.push_back(static_cast<float>(r_cap));
        out.wire_msg.voltage_class.push_back(static_cast<uint8_t>(w.voltage_class));
      }
    }

    const std::size_t intr_n = std::min(batch.intruders.size(), kMaxIntruders);
    out.intruder_msg.track_ids.reserve(intr_n);
    out.intruder_msg.pos_enu.reserve(intr_n);
    out.intruder_msg.vel_enu.reserve(intr_n);
    out.intruder_msg.radius_horiz_m.reserve(intr_n);
    out.intruder_msg.half_height_m.reserve(intr_n);
    out.intruder_msg.age_s.reserve(intr_n);
    out.intruder_msg.source_type.reserve(intr_n);

    for (std::size_t i = 0U; i < intr_n; ++i) {
      const auto& intr = batch.intruders[i];
      intruder_kf_.update_measurement(intr.track_id, intr.pos_enu, intr.vel_enu, batch.stamp_ns);
      out.intruder_msg.track_ids.push_back(intr.track_id);
      out.intruder_msg.pos_enu.push_back(intr.pos_enu);
      out.intruder_msg.vel_enu.push_back(intr.vel_enu);
      out.intruder_msg.radius_horiz_m.push_back(static_cast<float>(intr.radius_horiz_m));
      out.intruder_msg.half_height_m.push_back(static_cast<float>(intr.half_height_m));
      out.intruder_msg.age_s.push_back(static_cast<float>(age_s));
      out.intruder_msg.source_type.push_back(static_cast<uint8_t>(intr.source_type));
    }

    out.radar_degraded = !batch.radar_healthy;
    out.radar_grazing_fallback = grazing_fallback;
    out.v_max_cap_mps =
        (grazing_fallback || out.radar_degraded || stale)
            ? std::min(cfg_.v_max_nominal_mps, cfg_.v_max_radar_degraded_mps)
            : cfg_.v_max_nominal_mps;

    return out;
  }

  [[nodiscard]] std::string to_status_json(const ForwardHazardStepOutput& out) const {
    char buf[384];
    std::snprintf(
        buf,
        sizeof(buf),
        "{\"wire_count\":%zu,\"wire_cap\":%zu,\"intruder_count\":%zu,\"intruder_cap\":%zu,"
        "\"radar_grazing_fallback\":%s,\"radar_degraded\":%s,\"v_max_cap_mps\":%.2f,"
        "\"r_veh_m\":%.2f,\"enable_fp16\":%s,\"sensor_age_ms\":%.1f}",
        out.wire_msg.ids.size(),
        kMaxWireCapsules,
        out.intruder_msg.track_ids.size(),
        kMaxIntruders,
        out.radar_grazing_fallback ? "true" : "false",
        out.radar_degraded ? "true" : "false",
        out.v_max_cap_mps,
        cfg_.r_veh_m,
        cfg_.enable_fp16 ? "true" : "false",
        out.sensor_age_ms);
    return std::string(buf);
  }

 private:
  ForwardHazardConfig cfg_{};
  vertirta::rta::WireCapsuleTable wire_table_{};
  vertirta::rta::IntruderTrackerKf intruder_kf_{};
  std::array<OdometrySample, kPoseRingCapacity> odom_ring_{};
  std::size_t odom_head_{0U};
  std::size_t odom_count_{0U};
  OdometrySample last_odom_{};
  bool has_odom_{false};
  std::array<ImuSample, kPoseRingCapacity> imu_ring_{};
  std::size_t imu_head_{0U};
  std::size_t imu_count_{0U};
  ImuSample last_imu_{};
};

}  // namespace vertirta::forward_hazard_node
