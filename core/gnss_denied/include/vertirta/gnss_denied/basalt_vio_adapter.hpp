#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/frames.hpp"

// Compile-time license firewall for optional OKVIS2 BSD-3 backend (01-DR08-13 & Plan 6A Section 0.2).
#if defined(VERTIRTA_HAS_OKVIS2_BSD3)
#if defined(OKVIS_USE_NN) || defined(SUITESPARSE_VERSION) || defined(CHOLMOD_VERSION)
#error "OKVIS2 must compile with USE_NN=OFF and zero GPL SuiteSparse/CHOLMOD symbols."
#endif
#endif

namespace vertirta::gnss_denied {

inline constexpr std::size_t kMaxVioFeaturesPerFrame = 128;

// Single tracked optical-flow feature from belly stereo camera pair.
// Tagged with PIDNet-S nadir water-pixel mask so Marmara Sea wave-glint tracks are purged.
struct StereoKltFeature {
  double u_px{0.0};
  double v_px{0.0};
  double disparity_px{0.0};
  double flow_dx_px{0.0};
  double flow_dy_px{0.0};
  bool is_water_pixel{false};
  bool valid{false};
};

// Strictly GNSS-free sensor packet feeding Basalt (BSD-3-Clause) decoupled VIO.
// SECURITY BOUNDARY: NO GNSS position, velocity, or covariance field exists anywhere in this struct.
struct VioSensorFrame {
  uint64_t stamp_ns{0};
  double dt_s{0.005};  // 200 Hz IMU nominal step (or 10 Hz summary step)
  core::Vec3Ned imu_accel_ned{};  // Gravity-compensated specific force in local NED
  core::Vec3Ned stereo_delta_vel_ned{};  // Stereo visual velocity observation (optional)
  bool has_stereo_frame{false};

  double radalt_agl_m{0.0};
  bool radalt_valid{false};
  double baro_alt_amsl_m{0.0};
  bool baro_valid{false};

  // Optional PIDNet-S nadir water-fraction override [0, 1] when full feature array is not populated.
  double pidnet_water_fraction{-1.0};

  std::array<StereoKltFeature, kMaxVioFeaturesPerFrame> features{};
  std::size_t feature_count{0};
};

struct BasaltVioConfig {
  double stereo_baseline_m{0.80};      // Rigid carbon-boom belly stereo baseline (min 0.50 m)
  double stereo_baseline_min_m{0.50};
  bool mask_pidnet_water_pixels{true};
  double radalt_scale_blend{0.35};
  double baro_alt_blend{0.20};
  double stereo_vel_blend{0.40};
};

struct VioStateEstimate {
  uint64_t stamp_ns{0};
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  double water_fraction{0.0};
  uint32_t kept_land_features{0};
  uint32_t purged_water_features{0};
  bool metric_scale_locked{false};
  bool valid{false};
};

// Strictly GNSS-decoupled Basalt (BSD-3-Clause) VIO adapter (zero heap, noexcept).
// Compile-time & architectural invariant: accepts ZERO GNSS feedback so pre-detection
// spoofing can never poison the VIO reference velocity used by GnssSpoofGuard.
class BasaltVioAdapter {
 public:
  static constexpr bool kAllowsGnssFeedback = false;

  explicit BasaltVioAdapter(const BasaltVioConfig& cfg = BasaltVioConfig{}) noexcept;

  void reset(const core::Vec3Ned& init_pos_ned = {}, const core::Vec3Ned& init_vel_ned = {}) noexcept;

  [[nodiscard]] bool step(const VioSensorFrame& frame, VioStateEstimate& out) noexcept;

  [[nodiscard]] const BasaltVioConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] const VioStateEstimate& state() const noexcept { return state_; }

  [[nodiscard]] static constexpr bool compiled_with_basalt_native() noexcept {
#ifdef VERTIRTA_HAS_BASALT
    return true;
#else
    return false;
#endif
  }

 private:
  BasaltVioConfig cfg_{};
  VioStateEstimate state_{};
};

}  // namespace vertirta::gnss_denied
