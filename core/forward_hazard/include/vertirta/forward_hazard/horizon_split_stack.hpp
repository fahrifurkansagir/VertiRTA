#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/frames.hpp"

namespace vertirta::forward_hazard {

// Decoupled 45 deg HFOV narrow-FOV forward camera geometry (strictly separate from 512x512 80 deg nadir SLZ).
struct ForwardCameraIntrinsics {
  uint32_t width_px{1280};
  uint32_t height_px{720};
  double hfov_deg{45.0};
  double fx_px{1545.1};
  double fy_px{1545.1};
  double cx_px{640.0};
  double cy_px{360.0};

  // Locked nadir SLZ reference constants asserted unchanged (01-DR08-13 Override #6).
  static constexpr uint32_t kLockedNadirSlzWidthPx = 512;
  static constexpr uint32_t kLockedNadirSlzHeightPx = 512;
  static constexpr double kLockedNadirSlzHfovDeg = 80.0;
  static constexpr double kLockedNadirSlzFxPx = 305.09;
};

struct SplitStackConfig {
  ForwardCameraIntrinsics intrinsics{};
  uint32_t roi_width_px{1280};
  uint32_t roi_height_px{320};
  uint32_t roi_top_y_px{200};  // Centered around horizon cy=360: [200, 520)
  uint32_t stacked_width_px{640};
  uint32_t stacked_height_px{640};
  uint32_t seam_overlap_px{16};
  double max_derotate_rad{0.3490658504};  // 20 deg
};

struct StackedPixelCoord {
  double x_stack{0.0};
  double y_stack{0.0};
  uint8_t half_id{0};  // 0 = top (left horizon half), 1 = bottom (right horizon half)
  bool in_overlap_seam{false};
  bool valid{false};
};

struct NativePixelCoord {
  double u_px{0.0};
  double v_px{0.0};
  bool valid{false};
};

// IMU-stabilized 1280x720 -> 1280x320 horizon ROI -> 640x640 vertical split-stack preprocessor.
class HorizonSplitStackPreprocessor {
 public:
  explicit HorizonSplitStackPreprocessor(const SplitStackConfig& cfg = SplitStackConfig{}) noexcept;

  [[nodiscard]] const SplitStackConfig& config() const noexcept { return cfg_; }

  // Project a native 1280x720 pixel (after IMU roll/pitch stabilization) into 640x640 stacked coordinates.
  // If prefer_right_half is true inside the 16 px overlap seam, maps into the bottom half (right horizon).
  [[nodiscard]] StackedPixelCoord native_to_stacked(
      double u_px,
      double v_px,
      double imu_roll_rad,
      double imu_pitch_rad,
      bool prefer_right_half = false) const noexcept;

  // Exact inverse mapping from 640x640 detector output back into native 1280x720 camera pixels.
  [[nodiscard]] NativePixelCoord stacked_to_native(
      double x_stack,
      double y_stack,
      double imu_roll_rad,
      double imu_pitch_rad) const noexcept;

  // Convert native 1280x720 camera pixel to unit ray in camera FRD (+X forward, +Y right, +Z down).
  [[nodiscard]] core::Vec3Frd pixel_to_unit_ray_frd(double u_px, double v_px) const noexcept;

  // Compute unit camera-frame interpretation-plane normal for a 2D line segment (u0,v0)->(u1,v1).
  // Returns false if endpoints are nearly collinear (< 1e-6 rad separation).
  [[nodiscard]] bool segment_to_interp_plane_frd(
      double u0_px,
      double v0_px,
      double u1_px,
      double v1_px,
      core::Vec3Frd& normal_unit_frd) const noexcept;

  // Project a 3D point in forward camera FRD (+X > 0) back into native 1280x720 pixels.
  [[nodiscard]] bool project_frd_to_pixel(
      const core::Vec3Frd& pt_frd,
      double& u_px,
      double& v_px) const noexcept;

 private:
  SplitStackConfig cfg_{};
};

}  // namespace vertirta::forward_hazard
