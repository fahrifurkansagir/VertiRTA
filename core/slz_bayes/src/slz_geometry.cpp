// SPDX-License-Identifier: Apache-2.0
#include "vertirta/perception/slz_geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace vertirta::perception {

namespace {

[[nodiscard]] float SelectQuantileStack(
    std::array<float, 256>& buf,
    std::size_t count,
    float q) noexcept {
  if (count == 0u) {
    return 0.0f;
  }
  const std::size_t idx = static_cast<std::size_t>(
      std::clamp(q, 0.0f, 1.0f) * static_cast<float>(count - 1u));
  auto nth = buf.begin() + static_cast<std::ptrdiff_t>(idx);
  std::nth_element(buf.begin(), nth, buf.begin() + static_cast<std::ptrdiff_t>(count));
  return *nth;
}

}  // namespace

CameraPoseEnu MakeGimbalCameraPoseEnu(
    const core::Vec3Enu& position_enu,
    double yaw_rad,
    double pitch_down_rad) noexcept {
  const double cy = std::cos(yaw_rad);
  const double sy = std::sin(yaw_rad);
  const double cp = std::cos(pitch_down_rad);
  const double sp = std::sin(pitch_down_rad);

  // Camera body axes expressed in ENU:
  // X_cam (forward/optical +Z) = (cp*cy, cp*sy, -sp)
  // Y_cam (left)               = (-sy, cy, 0)
  // Z_cam (up = X_cam x Y_cam) = (sp*cy, sp*sy, cp)
  CameraPoseEnu pose{};
  pose.r_wc.m[0] = cp * cy;
  pose.r_wc.m[1] = -sy;
  pose.r_wc.m[2] = sp * cy;

  pose.r_wc.m[3] = cp * sy;
  pose.r_wc.m[4] = cy;
  pose.r_wc.m[5] = sp * sy;

  pose.r_wc.m[6] = -sp;
  pose.r_wc.m[7] = 0.0;
  pose.r_wc.m[8] = cp;

  pose.t_wc = position_enu;
  return pose;
}

core::Vec3Enu OpticalAxisEnu(const CameraPoseEnu& pose) noexcept {
  const core::Vec3Enu opt_in_cam = kRotOptToCam.mul_vec(0.0, 0.0, 1.0);
  return pose.r_wc.mul_vec(opt_in_cam.x, opt_in_cam.y, opt_in_cam.z);
}

std::size_t BackprojectDepthRgbd(
    std::span<const float> depth_image,
    const CameraIntrinsics& intr,
    const CameraPoseEnu& pose_enu,
    std::span<core::Vec3Enu> points_out,
    std::span<uint32_t> pixel_indices_out,
    uint32_t stride) noexcept {
  const std::size_t expected =
      static_cast<std::size_t>(intr.width) * static_cast<std::size_t>(intr.height);
  if (expected == 0u || depth_image.size() != expected ||
      intr.fx <= 1e-6f || intr.fy <= 1e-6f || points_out.empty()) {
    return 0u;
  }

  const uint32_t step = (stride == 0u) ? 1u : stride;
  const double inv_fx = 1.0 / static_cast<double>(intr.fx);
  const double inv_fy = 1.0 / static_cast<double>(intr.fy);
  const double cx = static_cast<double>(intr.cx);
  const double cy = static_cast<double>(intr.cy);
  const bool write_idx = (pixel_indices_out.size() >= points_out.size());

  std::size_t written = 0u;
  for (uint32_t v = 0; v < intr.height && written < points_out.size(); v += step) {
    const std::size_t row_off = static_cast<std::size_t>(v) * intr.width;
    const double y_norm = (static_cast<double>(v) - cy) * inv_fy;

    for (uint32_t u = 0; u < intr.width && written < points_out.size(); u += step) {
      const std::size_t pix = row_off + u;
      const float z = depth_image[pix];
      if (!std::isfinite(z) || z < intr.z_min_m || z > intr.z_max_m) {
        continue;
      }

      const double zd = static_cast<double>(z);
      const double x_opt = (static_cast<double>(u) - cx) * inv_fx * zd;
      const double y_opt = y_norm * zd;
      const double z_opt = zd;

      // Deterministic optical -> camera body -> ENU transform (bypasses gz-sensors#545).
      const core::Vec3Enu p_cam = kRotOptToCam.mul_vec(x_opt, y_opt, z_opt);
      const core::Vec3Enu p_rot = pose_enu.r_wc.mul_vec(p_cam.x, p_cam.y, p_cam.z);
      points_out[written] = p_rot + pose_enu.t_wc;
      if (write_idx) {
        pixel_indices_out[written] = static_cast<uint32_t>(pix);
      }
      ++written;
    }
  }
  return written;
}

float AnchorMonocularDepthAffine(
    std::span<const float> disp,
    const core::Vec3Enu& opt_axis_enu,
    float agl_m,
    std::optional<float> rng_m,
    std::optional<float> far_m,
    std::span<float> depth_out) noexcept {
  constexpr float kEps = 1e-4f;
  constexpr float kZMin = 0.5f;
  constexpr float kZMax = 150.0f;

  if (disp.empty() || disp.size() != depth_out.size() ||
      !std::isfinite(agl_m) || agl_m <= 0.0f || !opt_axis_enu.is_finite()) {
    std::fill(depth_out.begin(), depth_out.end(), 0.0f);
    return 1.0f;
  }

  const double opt_norm = opt_axis_enu.norm();
  if (opt_norm < 1e-9) {
    std::fill(depth_out.begin(), depth_out.end(), 0.0f);
    return 1.0f;
  }
  const double neg_d_wz = -opt_axis_enu.z / opt_norm;
  if (neg_d_wz < 0.05) {
    // Optical axis aimed near/above horizon: downward slant anchor invalid.
    std::fill(depth_out.begin(), depth_out.end(), 0.0f);
    return 1.0f;
  }

  // Altitude-dependent monocular geometric variance floor (DR-7 C.2):
  // sigma^2_{geo,mono}(z_AGL) = 0.04 + (z_AGL / 60.0)^2 * 0.25
  const float ratio = agl_m / 60.0f;
  const float sigma2_geo_mono = 0.04f + ratio * ratio * 0.25f;

  // Target center slant distance: rangefinder along optical axis if available, else z_AGL / (-d_Wz).
  float z_anchor = static_cast<float>(static_cast<double>(agl_m) / neg_d_wz);
  if (rng_m.has_value() && std::isfinite(*rng_m) && *rng_m >= kZMin && *rng_m <= kZMax) {
    z_anchor = *rng_m;
  }
  z_anchor = std::clamp(z_anchor, kZMin, kZMax);

  // Sample central 20% core region (zero heap, fixed stack buffer).
  std::array<float, 256> core_zrel{};
  std::array<float, 256> core_disp{};
  std::size_t core_n = 0u;

  const std::size_t total = disp.size();
  const std::size_t side = static_cast<std::size_t>(std::sqrt(static_cast<double>(total)) + 0.5);
  if (side * side == total && side >= 8u) {
    const std::size_t margin = (side * 4u) / 10u;  // central 20%: [0.4*S, 0.6*S)
    const std::size_t box_w = std::max<std::size_t>(1u, side - 2u * margin);
    const std::size_t box_pixels = box_w * box_w;
    const std::size_t stride = std::max<std::size_t>(1u, box_pixels / core_zrel.size());
    std::size_t k = 0u;
    for (std::size_t y = margin; y < side - margin && core_n < core_zrel.size(); ++y) {
      for (std::size_t x = margin; x < side - margin && core_n < core_zrel.size(); ++x) {
        if ((k++ % stride) != 0u) {
          continue;
        }
        const float d = disp[y * side + x];
        if (!std::isfinite(d)) {
          continue;
        }
        const float d_pos = std::max(d, 0.0f);
        core_disp[core_n] = d_pos;
        core_zrel[core_n] = 1.0f / (d_pos + kEps);
        ++core_n;
      }
    }
  } else {
    const std::size_t stride = std::max<std::size_t>(1u, total / core_zrel.size());
    for (std::size_t i = 0; i < total && core_n < core_zrel.size(); i += stride) {
      const float d = disp[i];
      if (!std::isfinite(d)) {
        continue;
      }
      const float d_pos = std::max(d, 0.0f);
      core_disp[core_n] = d_pos;
      core_zrel[core_n] = 1.0f / (d_pos + kEps);
      ++core_n;
    }
  }

  if (core_n == 0u) {
    std::fill(depth_out.begin(), depth_out.end(), 0.0f);
    return 1.0f;
  }

  const float med_zrel = SelectQuantileStack(core_zrel, core_n, 0.5f);
  const float med_disp = SelectQuantileStack(core_disp, core_n, 0.5f);

  // Optional 2-point affine inverse-depth anchoring when far-field anchor is supplied.
  bool use_affine = false;
  float aff_a = 1.0f;
  float aff_b = 0.0f;
  if (far_m.has_value() && std::isfinite(*far_m) && *far_m > z_anchor + 0.5f) {
    std::array<float, 256> full_disp{};
    std::size_t full_n = 0u;
    const std::size_t stride = std::max<std::size_t>(1u, total / full_disp.size());
    for (std::size_t i = 0; i < total && full_n < full_disp.size(); i += stride) {
      if (std::isfinite(disp[i])) {
        full_disp[full_n++] = std::max(disp[i], 0.0f);
      }
    }
    if (full_n >= 4u) {
      const float d_far = SelectQuantileStack(full_disp, full_n, 0.10f);
      const float denom = med_disp - d_far;
      if (denom > 1e-4f) {
        const float inv_near = 1.0f / z_anchor;
        const float inv_far = 1.0f / std::clamp(*far_m, z_anchor + 0.5f, kZMax);
        aff_a = (inv_near - inv_far) / denom;
        aff_b = inv_near - aff_a * med_disp;
        use_affine = std::isfinite(aff_a) && std::isfinite(aff_b) && (aff_a > 0.0f);
      }
    }
  }

  const float scale = z_anchor / std::max(med_zrel, 1e-6f);
  for (std::size_t i = 0; i < total; ++i) {
    const float d = disp[i];
    if (!std::isfinite(d)) {
      depth_out[i] = 0.0f;
      continue;
    }
    const float d_pos = std::max(d, 0.0f);
    float z_m = 0.0f;
    if (use_affine) {
      const float inv_z = std::max(aff_a * d_pos + aff_b, 1.0f / kZMax);
      z_m = 1.0f / inv_z;
    } else {
      z_m = scale / (d_pos + kEps);
    }
    depth_out[i] = std::clamp(z_m, kZMin, kZMax);
  }

  return sigma2_geo_mono;
}

}  // namespace vertirta::perception
