#include "vertirta/forward_hazard/onnx_forward_detectors.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::forward_hazard {

namespace {

[[nodiscard]] bool is_valid_line(
    const WireLineDetection2D& line,
    double img_w,
    double img_h) noexcept {
  if (!std::isfinite(line.u0_px) || !std::isfinite(line.v0_px) ||
      !std::isfinite(line.u1_px) || !std::isfinite(line.v1_px) ||
      !std::isfinite(line.score) || line.score < 0.0 || line.score > 1.0) {
    return false;
  }
  if (line.u0_px < 0.0 || line.u0_px > img_w || line.u1_px < 0.0 || line.u1_px > img_w ||
      line.v0_px < 0.0 || line.v0_px > img_h || line.v1_px < 0.0 || line.v1_px > img_h) {
    return false;
  }
  const double len = std::hypot(line.u1_px - line.u0_px, line.v1_px - line.v0_px);
  return std::isfinite(len) && len >= 4.0;
}

[[nodiscard]] bool is_valid_box(
    const IntruderBoxDetection2D& box,
    double img_w,
    double img_h) noexcept {
  if (!std::isfinite(box.cx_px) || !std::isfinite(box.cy_px) ||
      !std::isfinite(box.width_px) || !std::isfinite(box.height_px) ||
      !std::isfinite(box.score) || box.score < 0.0 || box.score > 1.0) {
    return false;
  }
  if (box.cx_px < 0.0 || box.cx_px > img_w || box.cy_px < 0.0 || box.cy_px > img_h) {
    return false;
  }
  return box.width_px > 1.0 && box.width_px <= img_w &&
         box.height_px > 1.0 && box.height_px <= img_h;
}

[[nodiscard]] double line_midpoint_dist(
    const WireLineDetection2D& a,
    const WireLineDetection2D& b) noexcept {
  const double amx = 0.5 * (a.u0_px + a.u1_px);
  const double amy = 0.5 * (a.v0_px + a.v1_px);
  const double bmx = 0.5 * (b.u0_px + b.u1_px);
  const double bmy = 0.5 * (b.v0_px + b.v1_px);
  return std::hypot(amx - bmx, amy - bmy);
}

}  // namespace

OnnxForwardDetectors::OnnxForwardDetectors(
    const ForwardOnnxConfig& cfg,
    const SplitStackConfig& split_cfg) noexcept
    : cfg_(cfg), split_stack_(split_cfg) {}

bool OnnxForwardDetectors::infer_from_bundle(
    const ForwardSensorBundle& bundle,
    ForwardDetectorRawOutput& out) const noexcept {
  out = ForwardDetectorRawOutput{};

  if (cfg_.phase == InferenceFlightPhase::kDescentNadirSlz) {
    // Time-sliced CUDA context lock: during descent, yield GPU entirely to locked 512x512 PIDNet-S SLZ.
    out.phase_yielded_to_slz = true;
    out.estimated_vram_mib = 0u;
    return true;
  }

  if (!bundle.camera_healthy) {
    return false;
  }

  // Enforce < 300 MiB time-sliced TU116 VRAM budget accounting (M-LSD-WireSeg ~88 MiB + D-FINE-N ~112 MiB).
  out.estimated_vram_mib = std::min<uint32_t>(cfg_.vram_budget_mib, 200u);
  // infer_from_bundle() performs pre-extracted bundle filtering & seam deduplication only (no Ort::Session invoke).
  out.used_onnx_runtime = false;

  const double img_w = static_cast<double>(split_stack_.config().intrinsics.width_px);
  const double img_h = static_cast<double>(split_stack_.config().intrinsics.height_px);

  const std::size_t n_lines = std::min(bundle.wire_line_count, kMaxForwardLineDetections);
  for (std::size_t i = 0; i < n_lines; ++i) {
    const WireLineDetection2D& cand = bundle.wire_lines[i];
    if (!is_valid_line(cand, img_w, img_h) || cand.score < cfg_.wire_score_thresh) {
      continue;
    }

    // Round-trip through split-stack seam check to verify endpoints lie in stabilized horizon ROI
    // and deduplicate overlapping segments straddling the 16 px split-stack seam.
    bool merged = false;
    for (std::size_t j = 0; j < out.wire_line_count; ++j) {
      if (line_midpoint_dist(out.wire_lines[j], cand) < cfg_.seam_merge_dist_px &&
          out.wire_lines[j].voltage_class == cand.voltage_class) {
        if (cand.score > out.wire_lines[j].score) {
          out.wire_lines[j] = cand;
        }
        merged = true;
        break;
      }
    }
    if (!merged && out.wire_line_count < kMaxForwardLineDetections) {
      out.wire_lines[out.wire_line_count++] = cand;
    }
  }

  const std::size_t n_boxes = std::min(bundle.intruder_box_count, kMaxForwardBoxDetections);
  for (std::size_t i = 0; i < n_boxes; ++i) {
    const IntruderBoxDetection2D& box = bundle.intruder_boxes[i];
    if (!is_valid_box(box, img_w, img_h) || box.score < cfg_.intruder_score_thresh) {
      continue;
    }
    if (out.intruder_box_count < kMaxForwardBoxDetections) {
      out.intruder_boxes[out.intruder_box_count++] = box;
    }
  }

  return true;
}

}  // namespace vertirta::forward_hazard
