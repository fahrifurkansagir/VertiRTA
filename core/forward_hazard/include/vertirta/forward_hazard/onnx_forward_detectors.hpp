#pragma once

#include <cstddef>
#include <cstdint>

#include "vertirta/forward_hazard/forward_hazard_provider.hpp"
#include "vertirta/forward_hazard/horizon_split_stack.hpp"

namespace vertirta::forward_hazard {

enum class InferenceFlightPhase : uint8_t {
  kCruiseForward = 0,     // Runs M-LSD-WireSeg + D-FINE-N in shared CUDA context
  kDescentNadirSlz = 1,   // Yields GPU context to locked 512x512 PIDNet-S nadir SLZ
};

struct ForwardOnnxConfig {
  // 01-DR08-13 Override #6: FP16 is 2.5-3.4x SLOWER on TU116 GTX 1650 SUPER.
  // Must stay false by default until empirical profiling proves otherwise.
  bool enable_fp16{false};
  bool enable_cuda_graph{true};
  bool use_deterministic_mock{true};
  uint32_t vram_budget_mib{300};
  double wire_score_thresh{0.35};
  double intruder_score_thresh{0.30};
  double seam_merge_dist_px{12.0};
  InferenceFlightPhase phase{InferenceFlightPhase::kCruiseForward};
};

struct ForwardDetectorRawOutput {
  std::array<WireLineDetection2D, kMaxForwardLineDetections> wire_lines{};
  std::size_t wire_line_count{0};
  std::array<IntruderBoxDetection2D, kMaxForwardBoxDetections> intruder_boxes{};
  std::size_t intruder_box_count{0};
  bool used_onnx_runtime{false};
  bool phase_yielded_to_slz{false};
  uint32_t estimated_vram_mib{0};
};

// Permissive Apache-2.0 M-LSD-WireSeg (clDice) + D-FINE-N ONNX Runtime adapter.
// Behind #ifdef VERTIRTA_HAS_ONNXRUNTIME, binds Ort::Session; otherwise runs
// deterministic zero-heap pass-through / seam-dedup mock for host & CI tests.
class OnnxForwardDetectors {
 public:
  explicit OnnxForwardDetectors(
      const ForwardOnnxConfig& cfg = ForwardOnnxConfig{},
      const SplitStackConfig& split_cfg = SplitStackConfig{}) noexcept;

  [[nodiscard]] const ForwardOnnxConfig& config() const noexcept { return cfg_; }
  void set_flight_phase(InferenceFlightPhase phase) noexcept { cfg_.phase = phase; }

  // Filters, seam-deduplicates across the 16 px split-stack overlap, and validates detections.
  [[nodiscard]] bool infer_from_bundle(
      const ForwardSensorBundle& bundle,
      ForwardDetectorRawOutput& out) const noexcept;

  // Compile-time check exposing whether real ONNX Runtime C++ headers were linked.
  [[nodiscard]] static constexpr bool compiled_with_onnxruntime() noexcept {
#ifdef VERTIRTA_HAS_ONNXRUNTIME
    return true;
#else
    return false;
#endif
  }

 private:
  ForwardOnnxConfig cfg_{};
  HorizonSplitStackPreprocessor split_stack_{};
};

}  // namespace vertirta::forward_hazard
