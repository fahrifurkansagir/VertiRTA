#pragma once

#include <cstdint>

#include "vertirta/forward_hazard/forward_hazard_provider.hpp"
#include "vertirta/forward_hazard/horizon_split_stack.hpp"
#include "vertirta/forward_hazard/onnx_forward_detectors.hpp"
#include "vertirta/forward_hazard/se3_pose_ring_buffer.hpp"
#include "vertirta/forward_hazard/wire_span_ekf.hpp"

namespace vertirta::forward_hazard {

struct ForwardFusionConfig {
  SplitStackConfig split_stack{};
  ForwardOnnxConfig onnx{};
  WireSpanEkfConfig wire_ekf{};
  double vehicle_radius_m{kCanonicalVehicleRadiusM};
  double v_max_nominal_mps{kCanonicalNominalVMaxMps};
  double v_max_radar_degraded_mps{kCanonicalRadarDegradedVMaxMps};
  double dfine_hover_assoc_px{45.0};
  double min_radar_snr_db{10.0};
  double radar_assoc_gate_m{15.0};
};

// Concrete 30 Hz forward wire + intruder radar-vision fusion implementation of IForwardHazardProvider.
class ForwardHazardFusion final : public IForwardHazardProvider {
 public:
  explicit ForwardHazardFusion(const ForwardFusionConfig& cfg = ForwardFusionConfig{}) noexcept;

  void reset() noexcept override;

  void push_pose_200hz(const TimedPoseSe3& pose) noexcept override;

  [[nodiscard]] bool step(
      const ForwardSensorBundle& bundle,
      ForwardHazardOutput& out) noexcept override;

  void set_flight_phase(InferenceFlightPhase phase) noexcept { detectors_.set_flight_phase(phase); }

  [[nodiscard]] const ForwardFusionConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const Se3PoseRingBuffer& pose_buffer() const noexcept { return pose_ring_; }

 private:
  ForwardFusionConfig cfg_{};
  HorizonSplitStackPreprocessor split_stack_{};
  OnnxForwardDetectors detectors_{};
  Se3PoseRingBuffer pose_ring_{};
  WireSpanEkfBank wire_ekf_{};
  rta::IntruderTrackerKf intruder_kf_{};
  uint32_t next_radar_track_id_{1000};
};

}  // namespace vertirta::forward_hazard
