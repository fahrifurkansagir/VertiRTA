// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "vertirta/sim/mmwave_77ghz_physics.hpp"

#ifdef VERTIRTA_HAS_GZ_SENSORS
#include <gz/sensors/Sensor.hh>
#include <gz/sim/System.hh>
#endif

namespace vertirta::sim {

// Host-testable sensor adapter wrapping MmWave77GHzPhysics.
// When Gazebo Harmonic (gz-sensors8 / gz-sim8) is installed, also exposes
// MmWave77GHzSensor (gz::sensors::Sensor) and MmWave77GHzSystem (gz::sim::System)
// registered for <sensor type="custom" gz:type="mmwave_77ghz">.
class MmWave77GHzSensorBridge {
 public:
  static constexpr std::size_t kMaxDetectionsPerScan = 128;

  explicit MmWave77GHzSensorBridge(
      const MmWaveRadarConfig& cfg = MmWaveRadarConfig::make_tier1b_cascaded(),
      std::string topic = "/vertirta/mmwave_77ghz/detections") noexcept;

  [[nodiscard]] const MmWave77GHzPhysics& physics() const noexcept { return physics_; }
  [[nodiscard]] std::string_view topic() const noexcept { return topic_; }
  [[nodiscard]] uint64_t scan_count() const noexcept { return scan_count_; }

  [[nodiscard]] std::size_t step_scan(
      uint64_t stamp_ns,
      const Vec3f64& ego_vel_sensor_mps,
      std::span<const WireTargetDesc> wires,
      std::span<const PointTargetDesc> points,
      bool deterministic_mean_rcs = true) noexcept;

  [[nodiscard]] std::span<const RadarDetection> last_detections() const noexcept {
    return std::span<const RadarDetection>(buffer_.data(), last_count_);
  }
  [[nodiscard]] uint64_t last_stamp_ns() const noexcept { return last_stamp_ns_; }

 private:
  MmWave77GHzPhysics physics_{};
  std::string topic_;
  std::array<RadarDetection, kMaxDetectionsPerScan> buffer_{};
  std::size_t last_count_{0};
  uint64_t last_stamp_ns_{0};
  uint64_t scan_count_{0};
};

#ifdef VERTIRTA_HAS_GZ_SENSORS
class MmWave77GHzSensor final : public gz::sensors::Sensor {
 public:
  MmWave77GHzSensor() = default;
  ~MmWave77GHzSensor() override = default;

  bool Load(const sdf::Sensor& sdf) override;
  bool Update(const std::chrono::steady_clock::duration& now) override;

 private:
  MmWave77GHzSensorBridge bridge_{};
};

class MmWave77GHzSystem final : public gz::sim::System,
                                public gz::sim::ISystemConfigure,
                                public gz::sim::ISystemPreUpdate,
                                public gz::sim::ISystemPostUpdate {
 public:
  MmWave77GHzSystem() = default;
  ~MmWave77GHzSystem() override = default;

  void Configure(
      const gz::sim::Entity& entity,
      const std::shared_ptr<const sdf::Element>& sdf,
      gz::sim::EntityComponentManager& ecm,
      gz::sim::EventManager& event_mgr) override;
  void PreUpdate(
      const gz::sim::UpdateInfo& info,
      gz::sim::EntityComponentManager& ecm) override;
  void PostUpdate(
      const gz::sim::UpdateInfo& info,
      const gz::sim::EntityComponentManager& ecm) override;

 private:
  MmWave77GHzSensorBridge bridge_{};
};
#endif  // VERTIRTA_HAS_GZ_SENSORS

}  // namespace vertirta::sim
