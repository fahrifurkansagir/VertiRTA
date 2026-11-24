#include "vertirta/forward_hazard/horizon_split_stack.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::forward_hazard {

namespace {

constexpr double kEps = 1e-9;

[[nodiscard]] double clamp_derotate(double angle_rad, double max_abs_rad) noexcept {
  if (!std::isfinite(angle_rad)) {
    return 0.0;
  }
  return std::clamp(angle_rad, -max_abs_rad, max_abs_rad);
}

}  // namespace

HorizonSplitStackPreprocessor::HorizonSplitStackPreprocessor(const SplitStackConfig& cfg) noexcept
    : cfg_(cfg) {}

StackedPixelCoord HorizonSplitStackPreprocessor::native_to_stacked(
    double u_px,
    double v_px,
    double imu_roll_rad,
    double imu_pitch_rad,
    bool prefer_right_half) const noexcept {
  StackedPixelCoord out{};
  if (!std::isfinite(u_px) || !std::isfinite(v_px)) {
    return out;
  }

  const double roll = clamp_derotate(imu_roll_rad, cfg_.max_derotate_rad);
  const double pitch = clamp_derotate(imu_pitch_rad, cfg_.max_derotate_rad);

  // IMU horizon stabilization: pitch shifts horizon vertically by fy * tan(pitch);
  // roll counter-rotates pixel around principal point (cx, cy).
  const double dx = u_px - cfg_.intrinsics.cx_px;
  const double dy = v_px - (cfg_.intrinsics.cy_px + cfg_.intrinsics.fy_px * std::tan(pitch));
  const double c = std::cos(-roll);
  const double s = std::sin(-roll);
  const double u_stab = cfg_.intrinsics.cx_px + c * dx - s * dy;
  const double v_stab = cfg_.intrinsics.cy_px + s * dx + c * dy;

  const double roi_top = static_cast<double>(cfg_.roi_top_y_px);
  const double roi_h = static_cast<double>(cfg_.roi_height_px);
  const double roi_w = static_cast<double>(cfg_.roi_width_px);
  if (u_stab < 0.0 || u_stab >= roi_w || v_stab < roi_top || v_stab >= (roi_top + roi_h)) {
    return out;
  }

  // Split 1280x320 horizon into left [0..639+ov/2] and right [640-ov/2..1279] mapped onto 640x320 halves.
  const double half_w = static_cast<double>(cfg_.stacked_width_px);   // 640
  const double half_h = static_cast<double>(cfg_.stacked_height_px) * 0.5;  // 320
  const double ov_half = 0.5 * static_cast<double>(cfg_.seam_overlap_px);   // 8 px each side -> 16 px seam

  const double left_end = half_w + ov_half;    // 648
  const double right_start = half_w - ov_half; // 632
  const bool in_seam = (u_stab >= right_start && u_stab < left_end);

  const bool use_right = in_seam ? prefer_right_half : (u_stab >= half_w);
  const double y_rel = (v_stab - roi_top) * (half_h / roi_h);

  if (!use_right) {
    // Top half covers native u_stab in [0, half_w + ov_half) scaled linearly into [0, half_w)
    const double scale_x = half_w / left_end;
    out.x_stack = u_stab * scale_x;
    out.y_stack = y_rel;
    out.half_id = 0u;
  } else {
    // Bottom half covers native u_stab in [half_w - ov_half, roi_w) scaled linearly into [0, half_w)
    const double span_right = roi_w - right_start;
    const double scale_x = half_w / span_right;
    out.x_stack = (u_stab - right_start) * scale_x;
    out.y_stack = half_h + y_rel;
    out.half_id = 1u;
  }

  out.in_overlap_seam = in_seam;
  out.valid = (out.x_stack >= 0.0 && out.x_stack < half_w &&
               out.y_stack >= 0.0 && out.y_stack < static_cast<double>(cfg_.stacked_height_px));
  return out;
}

NativePixelCoord HorizonSplitStackPreprocessor::stacked_to_native(
    double x_stack,
    double y_stack,
    double imu_roll_rad,
    double imu_pitch_rad) const noexcept {
  NativePixelCoord out{};
  if (!std::isfinite(x_stack) || !std::isfinite(y_stack)) {
    return out;
  }
  const double stack_w = static_cast<double>(cfg_.stacked_width_px);
  const double stack_h = static_cast<double>(cfg_.stacked_height_px);
  if (x_stack < 0.0 || x_stack >= stack_w || y_stack < 0.0 || y_stack >= stack_h) {
    return out;
  }

  const double half_w = stack_w;
  const double half_h = stack_h * 0.5;
  const double roi_top = static_cast<double>(cfg_.roi_top_y_px);
  const double roi_h = static_cast<double>(cfg_.roi_height_px);
  const double roi_w = static_cast<double>(cfg_.roi_width_px);
  const double ov_half = 0.5 * static_cast<double>(cfg_.seam_overlap_px);

  double u_stab = 0.0;
  double v_stab = 0.0;
  if (y_stack < half_h) {
    const double left_end = half_w + ov_half;
    u_stab = x_stack * (left_end / half_w);
    v_stab = roi_top + y_stack * (roi_h / half_h);
  } else {
    const double right_start = half_w - ov_half;
    const double span_right = roi_w - right_start;
    u_stab = right_start + x_stack * (span_right / half_w);
    v_stab = roi_top + (y_stack - half_h) * (roi_h / half_h);
  }

  // Exact inverse of IMU roll/pitch stabilization.
  const double roll = clamp_derotate(imu_roll_rad, cfg_.max_derotate_rad);
  const double pitch = clamp_derotate(imu_pitch_rad, cfg_.max_derotate_rad);
  const double dx_s = u_stab - cfg_.intrinsics.cx_px;
  const double dy_s = v_stab - cfg_.intrinsics.cy_px;
  const double c = std::cos(roll);
  const double s = std::sin(roll);
  out.u_px = cfg_.intrinsics.cx_px + c * dx_s - s * dy_s;
  out.v_px = cfg_.intrinsics.cy_px + cfg_.intrinsics.fy_px * std::tan(pitch) + s * dx_s + c * dy_s;
  out.valid = std::isfinite(out.u_px) && std::isfinite(out.v_px);
  return out;
}

core::Vec3Frd HorizonSplitStackPreprocessor::pixel_to_unit_ray_frd(
    double u_px,
    double v_px) const noexcept {
  // Forward camera FRD: +X optical forward, +Y image right, +Z image down.
  const double x = 1.0;
  const double y = (u_px - cfg_.intrinsics.cx_px) / cfg_.intrinsics.fx_px;
  const double z = (v_px - cfg_.intrinsics.cy_px) / cfg_.intrinsics.fy_px;
  const double inv_n = 1.0 / std::sqrt(x * x + y * y + z * z);
  return {x * inv_n, y * inv_n, z * inv_n};
}

bool HorizonSplitStackPreprocessor::segment_to_interp_plane_frd(
    double u0_px,
    double v0_px,
    double u1_px,
    double v1_px,
    core::Vec3Frd& normal_unit_frd) const noexcept {
  if (!std::isfinite(u0_px) || !std::isfinite(v0_px) ||
      !std::isfinite(u1_px) || !std::isfinite(v1_px)) {
    return false;
  }
  const core::Vec3Frd r0 = pixel_to_unit_ray_frd(u0_px, v0_px);
  const core::Vec3Frd r1 = pixel_to_unit_ray_frd(u1_px, v1_px);
  const double nx = r0.y * r1.z - r0.z * r1.y;
  const double ny = r0.z * r1.x - r0.x * r1.z;
  const double nz = r0.x * r1.y - r0.y * r1.x;
  const double n = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (n < 1e-6) {
    return false;
  }
  normal_unit_frd = {nx / n, ny / n, nz / n};
  return true;
}

bool HorizonSplitStackPreprocessor::project_frd_to_pixel(
    const core::Vec3Frd& pt_frd,
    double& u_px,
    double& v_px) const noexcept {
  if (!pt_frd.is_finite() || pt_frd.x <= kEps) {
    return false;
  }
  u_px = cfg_.intrinsics.cx_px + cfg_.intrinsics.fx_px * (pt_frd.y / pt_frd.x);
  v_px = cfg_.intrinsics.cy_px + cfg_.intrinsics.fy_px * (pt_frd.z / pt_frd.x);
  return (u_px >= 0.0 && u_px < static_cast<double>(cfg_.intrinsics.width_px) &&
          v_px >= 0.0 && v_px < static_cast<double>(cfg_.intrinsics.height_px));
}

}  // namespace vertirta::forward_hazard
