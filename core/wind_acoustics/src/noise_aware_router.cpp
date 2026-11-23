// SPDX-License-Identifier: Apache-2.0
#include "vertirta/wind_acoustics/noise_aware_router.hpp"

#include <algorithm>
#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::wind_acoustics {

namespace {

[[nodiscard]] inline double smooth_softplus(double x, double beta) noexcept {
  const double b = std::max(beta, 0.25);
  const double z = x / b;
  if (z > 20.0) {
    return x;
  }
  if (z < -20.0) {
    return 0.0;
  }
  return b * std::log1p(std::exp(z));
}

}  // namespace

NoiseAwareRouter::NoiseAwareRouter(
    rta::CbvfSafetyFilter* safety_filter,
    const MmapVrtaWindLut* wind_lut,
    NoiseRouterConfig cfg) noexcept
    : safety_filter_(safety_filter),
      wind_lut_(wind_lut),
      cfg_(cfg),
      receptors_(build_istanbul_64_receptors()),
      ground_receptors_(extract_ground_receptors(receptors_)) {}

double NoiseAwareRouter::eval_softplus_noise_cost(
    const AcousticHemisphere180& hemi,
    const core::Vec3Enu& pos_enu,
    const core::Vec3Enu& vel_enu,
    double* out_max_dba,
    double* out_hospital_dba) const noexcept {
  std::array<double, kNumIstanbulReceptors> dba{};
  acoustics_.propagate_to_receptors(hemi, pos_enu, vel_enu, ground_receptors_, dba);

  double cost = 0.0;
  double max_val = 0.0;
  for (std::size_t k = 0; k < kNumIstanbulReceptors; ++k) {
    const double lk = dba[k];
    if (lk > max_val) {
      max_val = lk;
    }
    const double excess = lk - receptors_[k].threshold_dba(cfg_.period);
    const double sp = smooth_softplus(excess, cfg_.softplus_beta_dba);
    cost += receptors_[k].soft_weight * sp;
  }
  if (out_max_dba != nullptr) {
    *out_max_dba = max_val;
  }
  if (out_hospital_dba != nullptr) {
    *out_hospital_dba = dba[0];  // Receptor 0 is Kartal Dr. Lutfi Kirdar Hospital
  }
  return cost;
}

NoiseRouterOutput NoiseAwareRouter::step_100hz(
    const core::Vec3Ned& pos_ned,
    const core::Vec3Ned& vel_ned,
    const core::Vec3Ned& a_track_ned,
    double yaw_rate_nom,
    const core::CapabilityEnvelope& env,
    const AcousticFlightState& acoustic_state,
    std::span<const rta::HazardPoint> corridor_hazards) noexcept {
  NoiseRouterOutput out{};
  const core::Vec3Enu pos_enu = core::ned_to_enu(pos_ned);
  const core::Vec3Enu vel_enu = core::ned_to_enu(vel_ned);

  // 1) Query VRTAWND1 wind ROM (or fail-closed conservative fallback when unloaded)
  if (wind_lut_ != nullptr && wind_lut_->is_loaded()) {
    (void)wind_lut_->query_enu(pos_enu, out.wind_sample);
  } else {
    out.wind_sample.valid = false;
    out.wind_sample.eps_rom_p95_mps = MmapVrtaWindLut::kConservativeFallbackEpsMps;
    out.wind_sample.sigma_rms_mps = MmapVrtaWindLut::kConservativeFallbackSigmaMps;
    out.wind_sample.sigma_enu = {MmapVrtaWindLut::kConservativeFallbackSigmaMps,
                                 MmapVrtaWindLut::kConservativeFallbackSigmaMps,
                                 MmapVrtaWindLut::kConservativeFallbackSigmaMps};
  }

  // 2) Event-driven 180-node acoustic hemisphere cache (DR-09 Sec 3.2):
  //    Refresh ONLY when |dRPM|/RPM > 2%, |dV_air| > 2 m/s, or airframe/heading/FPA shifts materially.
  auto rel_diff_gt_2pct = [](double a, double b) noexcept {
    const double denom = std::max(std::abs(b), 1.0);
    return (std::abs(a - b) / denom) > 0.02;
  };
  const bool need_hemi_refresh =
      !hemi_cache_valid_ ||
      (acoustic_state.airframe != cached_acoustic_state_.airframe) ||
      (acoustic_state.lift_rotors_locked != cached_acoustic_state_.lift_rotors_locked) ||
      rel_diff_gt_2pct(acoustic_state.lift_rpm, cached_acoustic_state_.lift_rpm) ||
      rel_diff_gt_2pct(acoustic_state.pusher_rpm, cached_acoustic_state_.pusher_rpm) ||
      (std::abs(acoustic_state.airspeed_mps - cached_acoustic_state_.airspeed_mps) > 2.0) ||
      (std::abs(acoustic_state.flight_path_angle_deg - cached_acoustic_state_.flight_path_angle_deg) > 1.0) ||
      (std::abs(acoustic_state.coax_z_over_d - cached_acoustic_state_.coax_z_over_d) > 1e-3) ||
      (std::abs(core::wrap_pi_d(acoustic_state.heading_enu_rad - cached_acoustic_state_.heading_enu_rad)) > 0.05);

  if (need_hemi_refresh) {
    acoustics_.synthesize_hemisphere(acoustic_state, cached_hemi_);
    cached_acoustic_state_ = acoustic_state;
    hemi_cache_valid_ = true;
    ++hemisphere_recompute_count_;
    out.hemisphere_recomputed = true;
  }
  const AcousticHemisphere180& hemi = cached_hemi_;
  out.tier1_noise_cost = eval_softplus_noise_cost(
      hemi, pos_enu, vel_enu, &out.max_receptor_dba, &out.kartal_hospital_dba);

  // 3) Tier-1 Soft Outer Planner: immediately shed 100% of soft noise preference within 1 tick
  //    (10 ms) whenever degraded/OEI regime or severe coastal downdraft is present.
  const bool degraded_or_downdraft =
      (env.regime != core::FlightRegime::NOMINAL) ||
      (out.wind_sample.valid && out.wind_sample.total_wind_enu.z < cfg_.downdraft_shed_mps);

  core::Vec3Ned a_ref_soft = a_track_ned;
  if (cfg_.quiet_mode_enabled && !degraded_or_downdraft) {
    const double dt = std::clamp(cfg_.lookahead_s, 1.0, 10.0);
    const double step_xy = std::min(cfg_.max_soft_accel_delta_mps2, 0.45 * static_cast<double>(env.a_xy_max));
    const double step_z = std::min(1.2, 0.35 * step_xy);

    // 9 deterministic cross-track (lateral E/W) + climb (-D) perturbations preserving forward flow
    const std::array<core::Vec3Ned, 9> deltas{{
        {0.0, 0.0, 0.0},
        {0.0, -step_xy, 0.0},                // Full seaward (west) lateral offset
        {0.0, -0.5 * step_xy, 0.0},          // Gentle seaward lateral offset
        {0.0, +0.5 * step_xy, 0.0},          // Gentle inland lateral offset
        {0.0, +step_xy, 0.0},                // Full inland lateral offset
        {0.0, -step_xy, -step_z},            // Seaward + climb (raises slant range & moves offshore)
        {0.0, -0.5 * step_xy, -step_z},      // Gentle seaward + climb
        {0.0, +0.5 * step_xy, -step_z},      // Gentle inland + climb
        {0.0, 0.0, -step_z},                 // Pure climb
    }};

    double best_obj = 1e30;
    for (const auto& d_a : deltas) {
      const core::Vec3Ned a_cand = a_track_ned + d_a;
      const core::Vec3Ned p_pred_ned =
          pos_ned + vel_ned * dt + a_cand * (0.5 * dt * dt);
      const core::Vec3Ned v_pred_ned = vel_ned + a_cand * dt;
      const core::Vec3Enu p_pred_enu = core::ned_to_enu(p_pred_ned);
      const core::Vec3Enu v_pred_enu = core::ned_to_enu(v_pred_ned);

      // Soft penalty on drifting below 35 m AGL or above 145 m AGL ceiling
      double alt_pen = 0.0;
      if (p_pred_enu.z < 35.0) {
        alt_pen = 50.0 * (35.0 - p_pred_enu.z);
      } else if (p_pred_enu.z > 145.0) {
        alt_pen = 50.0 * (p_pred_enu.z - 145.0);
      }

      const double track_pen = 0.5 * d_a.norm_sq();
      const double noise_pen =
          cfg_.lambda_noise * eval_softplus_noise_cost(hemi, p_pred_enu, v_pred_enu);
      const double obj = track_pen + noise_pen + alt_pen;
      if (obj < best_obj) {
        best_obj = obj;
        a_ref_soft = a_cand;
      }
    }
  } else if (cfg_.quiet_mode_enabled && degraded_or_downdraft) {
    out.noise_preference_shed = true;
  }
  out.a_ref_soft_ned = a_ref_soft;

  // 4) Tier-2 Hard RTA Safety Filter:
  //    Always prepend Kartal Hospital active helipad (kind=3) + LTFJ CTR western wall (kind=3),
  //    append up to 12 corridor hazards (total <= 14 <= 15 RTA slots), inflate monotonically by
  //    VRTAWND1 eps_rom_p95, and project through frozen CbvfSafetyFilter::step_100hz.
  std::array<rta::HazardPoint, kMaxRouterActiveHazards> raw_hazards{};
  std::size_t n_haz = 0u;
  raw_hazards[n_haz++] = build_kartal_hospital_helipad_hazard();
  raw_hazards[n_haz++] = build_ltfj_ctr_wall_hazard(pos_ned);

  const std::size_t rem_slots = kMaxRouterActiveHazards - n_haz;
  const std::size_t n_copy = std::min(corridor_hazards.size(), rem_slots);
  for (std::size_t i = 0; i < n_copy; ++i) {
    raw_hazards[n_haz++] = corridor_hazards[i];
  }

  std::array<rta::HazardPoint, kMaxRouterActiveHazards> inflated_hazards{};
  const double v_horiz = std::hypot(vel_ned.x, vel_ned.y);
  const std::size_t n_inflated = inflate_hazards_with_wind_rom(
      std::span<const rta::HazardPoint>(raw_hazards.data(), n_haz),
      out.wind_sample,
      v_horiz,
      static_cast<double>(env.a_xy_max),
      std::span<rta::HazardPoint>(inflated_hazards.data(), n_haz));
  out.total_hazards_fed_to_rta = static_cast<uint16_t>(n_inflated);

  if (safety_filter_ != nullptr) {
    out.rta_out = safety_filter_->step_100hz(
        pos_ned,
        vel_ned,
        a_ref_soft,
        yaw_rate_nom,
        env,
        std::span<const rta::HazardPoint>(inflated_hazards.data(), n_inflated));

    const double rta_override_norm = (out.rta_out.a_safe_ned - a_ref_soft).norm();
    if (rta_override_norm > 0.15 || out.rta_out.used_hj_fallback || out.rta_out.cel_latched) {
      out.noise_preference_shed = true;
    }
  } else {
    out.rta_out.a_safe_ned = a_ref_soft;
    out.rta_out.yaw_rate_safe = yaw_rate_nom;
  }

  return out;
}

}  // namespace vertirta::wind_acoustics
