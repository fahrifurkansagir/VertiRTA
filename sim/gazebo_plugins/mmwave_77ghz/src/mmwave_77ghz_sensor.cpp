// SPDX-License-Identifier: Apache-2.0
#include "vertirta/sim/mmwave_77ghz_sensor.hpp"

#include <utility>

namespace vertirta::sim {

MmWave77GHzSensorBridge::MmWave77GHzSensorBridge(
    const MmWaveRadarConfig& cfg,
    std::string topic) noexcept
    : physics_(cfg), topic_(std::move(topic)) {}

std::size_t MmWave77GHzSensorBridge::step_scan(
    uint64_t stamp_ns,
    const Vec3f64& ego_vel_sensor_mps,
    std::span<const WireTargetDesc> wires,
    std::span<const PointTargetDesc> points,
    bool deterministic_mean_rcs) noexcept {
  last_count_ = physics_.simulate_scan(
      ego_vel_sensor_mps,
      wires,
      points,
      std::span<RadarDetection>(buffer_.data(), buffer_.size()),
      deterministic_mean_rcs);
  last_stamp_ns_ = stamp_ns;
  ++scan_count_;
  return last_count_;
}

#ifdef VERTIRTA_HAS_GZ_SENSORS
bool MmWave77GHzSensor::Load(const sdf::Sensor& sdf) {
  if (!gz::sensors::Sensor::Load(sdf)) {
    return false;
  }
  MmWaveRadarConfig cfg = MmWaveRadarConfig::make_tier1b_cascaded();
  std::string topic = sdf.Topic().empty() ? "/vertirta/mmwave_77ghz/detections" : sdf.Topic();
  bridge_ = MmWave77GHzSensorBridge(cfg, std::move(topic));
  return true;
}

bool MmWave77GHzSensor::Update(const std::chrono::steady_clock::duration& now) {
  const auto ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  (void)bridge_.step_scan(ns, Vec3f64{}, {}, {}, true);
  return true;
}

void MmWave77GHzSystem::Configure(
    const gz::sim::Entity& /*entity*/,
    const std::shared_ptr<const sdf::Element>& /*sdf*/,
    gz::sim::EntityComponentManager& /*ecm*/,
    gz::sim::EventManager& /*event_mgr*/) {
  bridge_ = MmWave77GHzSensorBridge(MmWaveRadarConfig::make_tier1b_cascaded());
}

void MmWave77GHzSystem::PreUpdate(
    const gz::sim::UpdateInfo& /*info*/,
    gz::sim::EntityComponentManager& /*ecm*/) {}

void MmWave77GHzSystem::PostUpdate(
    const gz::sim::UpdateInfo& info,
    const gz::sim::EntityComponentManager& /*ecm*/) {
  if (info.paused) {
    return;
  }
  const auto ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count());
  (void)bridge_.step_scan(ns, Vec3f64{}, {}, {}, true);
}
#endif  // VERTIRTA_HAS_GZ_SENSORS

}  // namespace vertirta::sim
