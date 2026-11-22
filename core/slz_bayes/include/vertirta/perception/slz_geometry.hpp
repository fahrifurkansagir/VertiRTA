// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "vertirta/core/frames.hpp"

namespace vertirta::perception {

struct CameraIntrinsics {
  float fx{305.09f};
  float fy{305.09f};
  float cx{255.5f};
  float cy{255.5f};
  uint32_t width{512u};
  uint32_t height{512u};
  float z_min_m{0.8f};
  float z_max_m{150.0f};
};

// Row-major 3x3 rotation matrix.
struct Mat3d {
  double m[9]{
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
  };

  [[nodiscard]] static constexpr Mat3d Identity() noexcept { return {}; }

  [[nodiscard]] constexpr core::Vec3Enu mul_vec(double x, double y, double z) const noexcept {
    return {
        m[0] * x + m[1] * y + m[2] * z,
        m[3] * x + m[4] * y + m[5] * z,
        m[6] * x + m[7] * y + m[8] * z,
    };
  }
};

// Camera pose in local ENU world frame.
// r_wc rotates from Gazebo/ROS camera body frame (X forward, Y left, Z up) to ENU world.
// BackprojectDepthRgbd applies fixed R_{opt->cam} before r_wc, bypassing gz-sensors#545.
struct CameraPoseEnu {
  Mat3d r_wc{Mat3d::Identity()};
  core::Vec3Enu t_wc{};
};

// Fixed optical (x-right, y-down, z-forward) -> camera body (x-forward, y-left, z-up) matrix.
inline constexpr Mat3d kRotOptToCam{{
    0.0,  0.0, 1.0,
   -1.0,  0.0, 0.0,
    0.0, -1.0, 0.0,
}};

// Builds a camera body-to-ENU pose looking along yaw_rad (CCW from East) pitched downward
// by pitch_down_rad (0 = horizontal, +pi/2 = straight nadir down).
[[nodiscard]] CameraPoseEnu MakeGimbalCameraPoseEnu(
    const core::Vec3Enu& position_enu,
    double yaw_rad,
    double pitch_down_rad) noexcept;

// Extracts unit optical axis (+Z_opt) expressed in ENU world coordinates.
[[nodiscard]] core::Vec3Enu OpticalAxisEnu(const CameraPoseEnu& pose) noexcept;

// Mod A (Primary RGB-D): backprojects raw depth_image [H*W] to ENU world points using
// deterministic R_wc * R_{opt->cam} (bypassing gz-sensors#545 PointCloud2 axis bug).
// Zero heap allocations; returns number of valid points written to points_out.
[[nodiscard]] std::size_t BackprojectDepthRgbd(
    std::span<const float> depth_image,
    const CameraIntrinsics& intr,
    const CameraPoseEnu& pose_enu,
    std::span<core::Vec3Enu> points_out,
    std::span<uint32_t> pixel_indices_out = {},
    uint32_t stride = 1u) noexcept;

// Mod B (Monocular Fallback): affine / scale anchoring of relative inverse-depth disparity
// using full roll+pitch optical slant z_AGL / (-d_Wz) (or optional laser rangefinder rng_m,
// and optional far-field depth anchor far_m). Zero heap allocations.
// Returns altitude-dependent monocular geometric variance floor sigma^2_{geo,mono}(agl_m).
[[nodiscard]] float AnchorMonocularDepthAffine(
    std::span<const float> disp,
    const core::Vec3Enu& opt_axis_enu,
    float agl_m,
    std::optional<float> rng_m,
    std::optional<float> far_m,
    std::span<float> depth_out) noexcept;

}  // namespace vertirta::perception
