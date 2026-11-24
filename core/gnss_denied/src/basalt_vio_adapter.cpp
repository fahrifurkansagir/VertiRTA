#include "vertirta/gnss_denied/basalt_vio_adapter.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::gnss_denied {

BasaltVioAdapter::BasaltVioAdapter(const BasaltVioConfig& cfg) noexcept : cfg_(cfg) {
  if (cfg_.stereo_baseline_m < cfg_.stereo_baseline_min_m) {
    cfg_.stereo_baseline_m = cfg_.stereo_baseline_min_m;
  }
}

void BasaltVioAdapter::reset(
    const core::Vec3Ned& init_pos_ned,
    const core::Vec3Ned& init_vel_ned) noexcept {
  state_ = VioStateEstimate{};
  state_.pos_ned = init_pos_ned;
  state_.vel_ned = init_vel_ned;
  state_.valid = true;
}

bool BasaltVioAdapter::step(const VioSensorFrame& frame, VioStateEstimate& out) noexcept {
  if (!std::isfinite(frame.dt_s) || frame.dt_s <= 0.0 || frame.dt_s > 1.0 ||
      !frame.imu_accel_ned.is_finite()) {
    out = state_;
    return false;
  }

  // 1. Purge Marmara Sea wave-glint KLT features using PIDNet-S water-pixel mask.
  uint32_t kept_land = 0u;
  uint32_t purged_water = 0u;
  const std::size_t n_feat = std::min(frame.feature_count, kMaxVioFeaturesPerFrame);
  for (std::size_t i = 0; i < n_feat; ++i) {
    const StereoKltFeature& f = frame.features[i];
    if (!f.valid) {
      continue;
    }
    if (cfg_.mask_pidnet_water_pixels && f.is_water_pixel) {
      ++purged_water;
    } else {
      ++kept_land;
    }
  }

  double water_frac = state_.water_fraction;
  if (frame.pidnet_water_fraction >= 0.0 && frame.pidnet_water_fraction <= 1.0 &&
      std::isfinite(frame.pidnet_water_fraction)) {
    water_frac = frame.pidnet_water_fraction;
  } else if ((kept_land + purged_water) > 0u) {
    water_frac = static_cast<double>(purged_water) /
                 static_cast<double>(kept_land + purged_water);
  }

  // 2. Pure IMU + belly stereo (0.80 m rigid carbon-boom baseline) propagation — ZERO GNSS coupling.
  const double dt = frame.dt_s;
  state_.vel_ned = state_.vel_ned + frame.imu_accel_ned * dt;

  // Apply stereo visual velocity update only when sufficient non-water land features remain.
  const bool stereo_usable = frame.has_stereo_frame && frame.stereo_delta_vel_ned.is_finite() &&
                             (kept_land > 0u || water_frac < 0.85);
  if (stereo_usable) {
    const double alpha = std::clamp(cfg_.stereo_vel_blend * (1.0 - 0.6 * water_frac), 0.05, 0.80);
    state_.vel_ned = state_.vel_ned * (1.0 - alpha) + frame.stereo_delta_vel_ned * alpha;
  }

  state_.pos_ned = state_.pos_ned + state_.vel_ned * dt;

  // 3. Nadir radar altimeter & barometer vertical metric scale lock.
  bool scale_locked = false;
  if (frame.radalt_valid && std::isfinite(frame.radalt_agl_m) && frame.radalt_agl_m >= 0.0) {
    // In local Kartal NED, D = -alt_above_datum (approximate ground elevation near datum = 0).
    const double target_d = -frame.radalt_agl_m;
    state_.pos_ned.z += cfg_.radalt_scale_blend * (target_d - state_.pos_ned.z);
    scale_locked = true;
  } else if (frame.baro_valid && std::isfinite(frame.baro_alt_amsl_m)) {
    const double target_d = -(frame.baro_alt_amsl_m - 12.0);
    state_.pos_ned.z += cfg_.baro_alt_blend * (target_d - state_.pos_ned.z);
    scale_locked = true;
  }

  state_.stamp_ns = frame.stamp_ns;
  state_.water_fraction = water_frac;
  state_.kept_land_features = kept_land;
  state_.purged_water_features = purged_water;
  state_.metric_scale_locked = scale_locked;
  state_.valid = state_.pos_ned.is_finite() && state_.vel_ned.is_finite();

  out = state_;
  return state_.valid;
}

}  // namespace vertirta::gnss_denied
