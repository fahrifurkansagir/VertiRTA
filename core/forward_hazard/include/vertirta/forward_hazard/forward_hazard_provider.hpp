#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/intruder_kf.hpp"
#include "vertirta/rta/wire_capsules.hpp"

namespace vertirta::forward_hazard {

// Canonical airframe & forward-hazard locks (01-DR08-13 Override #1 + config/vehicle_x8_240.yaml).
inline constexpr double kCanonicalVehicleRadiusM = 2.10;
inline constexpr double kCanonicalNominalVMaxMps = 22.0;
inline constexpr double kCanonicalRadarDegradedVMaxMps = 15.0;
inline constexpr double kCanonicalAccelXyMaxMps2 = 5.5;
inline constexpr std::size_t kMaxForwardLineDetections = 64;
inline constexpr std::size_t kMaxForwardBoxDetections = 32;
inline constexpr std::size_t kMaxRadarDetections = 128;

// 2D thin-wire line segment detected in native 1280x720 forward camera coordinates.
struct WireLineDetection2D {
  double u0_px{0.0};
  double v0_px{0.0};
  double u1_px{0.0};
  double v1_px{0.0};
  double score{0.0};
  uint8_t voltage_class{0};  // 0=distribution, 1=transmission, 2=crane_cable
};

// 2D small-object intruder bounding box from D-FINE-N (Apache-2.0).
struct IntruderBoxDetection2D {
  double cx_px{0.0};
  double cy_px{0.0};
  double width_px{0.0};
  double height_px{0.0};
  double score{0.0};
  uint8_t class_id{0};  // 0=drone, 1=bird, 2=aircraft, 3=pylon
};

enum class RadarDopplerClass : uint8_t {
  kStaticOrHoverCandidate = 0,
  kDynamicIntruder = 1,
};

// Single 77 GHz 4D mmWave radar detection in forward sensor FRD frame.
struct RadarPointDetection {
  double range_m{0.0};
  double az_rad{0.0};   // +right from boresight (+Y_FRD)
  double el_rad{0.0};   // +down from boresight (+Z_FRD)
  double radial_vel_mps{0.0};
  double residual_doppler_mps{0.0};
  double snr_db{0.0};
  double rcs_dbsm{-50.0};
  RadarDopplerClass doppler_class{RadarDopplerClass::kStaticOrHoverCandidate};
  bool valid{false};
};

// 200 Hz aircraft pose sample in local Kartal NED with Hamilton R_{FRD -> NED}.
struct TimedPoseSe3 {
  uint64_t stamp_ns{0};
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  core::QuatPx4FrdNed q_frd_to_ned{};
  bool valid{false};
};

// Fixed-capacity, zero-heap sensor bundle passed into IForwardHazardProvider::step().
struct ForwardSensorBundle {
  uint64_t frame_stamp_ns{0};
  uint64_t radar_stamp_ns{0};
  double imu_roll_rad{0.0};
  double imu_pitch_rad{0.0};

  // Optional 1280x720 synthetic/pre-extracted cues or detector detections.
  std::array<WireLineDetection2D, kMaxForwardLineDetections> wire_lines{};
  std::size_t wire_line_count{0};

  std::array<IntruderBoxDetection2D, kMaxForwardBoxDetections> intruder_boxes{};
  std::size_t intruder_box_count{0};

  std::array<RadarPointDetection, kMaxRadarDetections> radar_points{};
  std::size_t radar_point_count{0};

  // Explicitly signals radar front-end health (false when jammed, degraded, or >20 deg grazing dropout).
  bool radar_healthy{true};
  bool camera_healthy{true};
};

// Fixed-capacity output feeding frozen WireCapsuleTable (<=64) and IntruderTrackerKf (<=16).
struct ForwardHazardOutput {
  uint64_t stamp_ns{0};
  rta::WireCapsuleTable wire_capsules{};
  rta::IntruderTrackerKf intruder_tracker{};

  double recommended_v_max_mps{kCanonicalNominalVMaxMps};
  bool radar_degraded{false};
  bool grazing_parallax_active{false};
  bool standoff_capsule_emitted{false};
  uint32_t active_wire_spans{0};
  uint32_t active_wire_capsules{0};
  uint32_t active_intruders{0};
  uint32_t hover_intruders_rescued{0};
};

// Clean port interface for M3 forward wire/intruder hazard providers (zero heap, noexcept).
class IForwardHazardProvider {
 public:
  virtual ~IForwardHazardProvider() noexcept = default;

  virtual void reset() noexcept = 0;

  virtual void push_pose_200hz(const TimedPoseSe3& pose) noexcept = 0;

  [[nodiscard]] virtual bool step(
      const ForwardSensorBundle& bundle,
      ForwardHazardOutput& out) noexcept = 0;
};

}  // namespace vertirta::forward_hazard
