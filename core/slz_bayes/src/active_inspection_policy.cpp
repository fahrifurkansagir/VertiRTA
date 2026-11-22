// SPDX-License-Identifier: Apache-2.0
#include "vertirta/perception/active_inspection_policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vertirta::perception {

namespace {

[[nodiscard]] float GuardedElapsedSeconds(uint64_t now_ns, uint64_t start_ns) noexcept {
  if (now_ns <= start_ns) {
    return 0.0f;
  }
  return static_cast<float>(now_ns - start_ns) * 1e-9f;
}

}  // namespace

ActiveInspectionPolicy::ActiveInspectionPolicy(Config cfg) noexcept : cfg_(cfg) {}

void ActiveInspectionPolicy::Reset() noexcept {
  state_ = SlzDecisionState::CANDIDATE_EVAL;
  std::memset(active_id_, 0, sizeof(active_id_));
  active_center_ = {};
  has_active_ = false;
  approach_start_ns_ = 0u;
  orbit_start_ns_ = 0u;
  start_azimuth_rad_ = 0.0f;
}

float ActiveInspectionPolicy::EstimateOrbitDeltaSocPct(const EnergyGateInput& e) const noexcept {
  float fault_mult = 1.00f;
  switch (e.regime) {
    case core::FlightRegime::NOMINAL:
      fault_mult = 1.00f;
      break;
    case core::FlightRegime::SINGLE_LOSS:
      fault_mult = 1.18f;
      break;
    case core::FlightRegime::CROSS_DUAL:
      fault_mult = 1.36f;
      break;
    case core::FlightRegime::SAME_ARM_ACTIVE_SPIN:
    case core::FlightRegime::CRITICAL_DESCENT:
      fault_mult = 1.60f;
      break;
  }
  const float wind_clamped = std::clamp(std::isfinite(e.wind_mps) ? e.wind_mps : 0.0f, 0.0f, 18.0f);
  const float wind_mult = 1.0f + 0.012f * wind_clamped;
  return kBaseSoc12s * fault_mult * wind_mult * cfg_.soc_safety_mult;
}

bool ActiveInspectionPolicy::SameCandidate(const rta::SlzPrior& c) const noexcept {
  if (!has_active_) {
    return false;
  }
  const bool active_has_id = (active_id_[0] != '\0');
  const bool cand_has_id = (c.id[0] != '\0');
  if (active_has_id || cand_has_id) {
    return std::strncmp(active_id_, c.id, sizeof(active_id_)) == 0;
  }
  const double dx = c.center_enu.x - active_center_.x;
  const double dy = c.center_enu.y - active_center_.y;
  return (dx * dx + dy * dy) <= 1.0;
}

void ActiveInspectionPolicy::LatchCandidateIdentity(const rta::SlzPrior& c) noexcept {
  std::memcpy(active_id_, c.id, sizeof(active_id_));
  active_center_ = c.center_enu;
  has_active_ = true;
}

bool ActiveInspectionPolicy::IsAtOrbitRing(
    const core::Vec3Enu& vehicle_enu,
    const core::Vec3Enu& tlof_enu,
    float agl_m) const noexcept {
  const float r_horiz = static_cast<float>(
      std::hypot(vehicle_enu.x - tlof_enu.x, vehicle_enu.y - tlof_enu.y));
  return (std::fabs(agl_m - cfg_.z_inspect_agl_m) <= cfg_.arrival_agl_tol_m) &&
         (std::fabs(r_horiz - cfg_.r_inspect_m) <= cfg_.arrival_r_tol_m);
}

uint32_t ActiveInspectionPolicy::CollectRejectReasonMask(
    const rta::SlzPrior& c,
    bool mono_hi,
    bool var_ok) const noexcept {
  uint32_t mask = REASON_NONE;
  if ((c.reason_mask & REASON_CTR_VIOLATION) != 0u) mask |= REASON_CTR_VIOLATION;
  if (c.sigma_z_m > cfg_.sigma_z_commit_m) mask |= REASON_ROUGHNESS_EXCEEDED;
  if (c.theta_max_deg > cfg_.theta_commit_deg) mask |= REASON_SLOPE_EXCEEDED;
  if (c.clearance_r_m < cfg_.r_slz_min_m) mask |= REASON_CLEARANCE_INSUFF;
  if (c.p_safe < cfg_.p_safe_commit) mask |= REASON_OBSTACLE_CLEARANCE;
  if (c.step_max_m > cfg_.step_max_m) mask |= REASON_STEP_EXCEEDED;
  if (!var_ok) mask |= REASON_OOD_UNCERTAIN;
  if (mono_hi) mask |= REASON_MONO_ALT_VETO;
  if (c.h_dyn_max >= cfg_.h_dyn_commit_max) mask |= REASON_DYNAMIC_HAZARD;
  return mask;
}

ActiveInspectionDecision ActiveInspectionPolicy::Reject(
    ActiveInspectionDecision& out,
    uint32_t mask) noexcept {
  state_ = SlzDecisionState::REJECT_DIVERT_NEXT;
  has_active_ = false;
  out.state = state_;
  out.reason_mask = (mask != REASON_NONE) ? mask : REASON_OBSTACLE_CLEARANCE;
  out.inspect_setpoint.reset();
  return out;
}

ActiveInspectSetpoint ActiveInspectionPolicy::ComputeApproachSetpoint(
    const core::Vec3Enu& vehicle_enu,
    const core::Vec3Enu& tlof_enu,
    float agl_m,
    float approach_elapsed_s) const noexcept {
  const float az = static_cast<float>(
      std::atan2(vehicle_enu.y - tlof_enu.y, vehicle_enu.x - tlof_enu.x));
  const core::Vec3Enu target{
      tlof_enu.x + static_cast<double>(cfg_.r_inspect_m) * std::cos(static_cast<double>(az)),
      tlof_enu.y + static_cast<double>(cfg_.r_inspect_m) * std::sin(static_cast<double>(az)),
      tlof_enu.z + static_cast<double>(cfg_.z_inspect_agl_m),
  };
  const float r_now = std::max(
      0.5f, static_cast<float>(std::hypot(vehicle_enu.x - tlof_enu.x, vehicle_enu.y - tlof_enu.y)));
  const float z_now = std::max(1.0f, std::isfinite(agl_m) ? agl_m : cfg_.z_inspect_agl_m);
  const float gimbal_pitch = std::atan2(z_now, r_now);
  const float gimbal_yaw = static_cast<float>(
      std::atan2(tlof_enu.y - vehicle_enu.y, tlof_enu.x - vehicle_enu.x));
  return {
      target,
      tlof_enu,
      cfg_.approach_speed_mps,
      0.0f,
      approach_elapsed_s,
      gimbal_pitch,
      gimbal_yaw,
  };
}

ActiveInspectSetpoint ActiveInspectionPolicy::ComputeOrbitSetpoint(
    const core::Vec3Enu& tlof_enu,
    float elapsed_s) const noexcept {
  const float frac = std::clamp(elapsed_s / std::max(cfg_.max_budget_s, 1e-3f), 0.0f, 1.0f);
  const float az = start_azimuth_rad_ + frac * cfg_.arc_total_rad;
  const core::Vec3Enu target{
      tlof_enu.x + static_cast<double>(cfg_.r_inspect_m) * std::cos(static_cast<double>(az)),
      tlof_enu.y + static_cast<double>(cfg_.r_inspect_m) * std::sin(static_cast<double>(az)),
      tlof_enu.z + static_cast<double>(cfg_.z_inspect_agl_m),
  };
  const float gimbal_pitch = std::atan2(cfg_.z_inspect_agl_m, cfg_.r_inspect_m);
  const float gimbal_yaw = static_cast<float>(
      std::atan2(tlof_enu.y - target.y, tlof_enu.x - target.x));
  return {
      target,
      tlof_enu,
      (cfg_.r_inspect_m * cfg_.arc_total_rad) / std::max(cfg_.max_budget_s, 1e-3f),
      frac * cfg_.arc_total_rad,
      elapsed_s,
      gimbal_pitch,
      gimbal_yaw,
  };
}

ActiveInspectionDecision ActiveInspectionPolicy::Step(
    const rta::SlzPrior& candidate,
    const core::Vec3Enu& vehicle_enu,
    float agl_m,
    DepthMode mode,
    const EnergyGateInput& energy,
    uint64_t now_ns) noexcept {
  ActiveInspectionDecision out{};
  out.timestamp_ns = now_ns;
  out.depth_mode = mode;
  out.candidate = candidate;

  const bool ctr_ok = (candidate.reason_mask & REASON_CTR_VIOLATION) == 0u;
  const bool geom_ok = (candidate.theta_max_deg <= cfg_.theta_commit_deg) &&
                       (candidate.sigma_z_m <= cfg_.sigma_z_commit_m) &&
                       (candidate.step_max_m <= cfg_.step_max_m) &&
                       (candidate.clearance_r_m >= cfg_.r_slz_min_m);
  const bool sem_ok = (candidate.p_safe >= cfg_.p_safe_commit) &&
                      (candidate.h_dyn_max < cfg_.h_dyn_commit_max);
  const bool var_ok = (candidate.slz_ood_variance < cfg_.tau_commit);
  const bool mono_hi = (mode == DepthMode::MONO_FALLBACK) && (agl_m > cfg_.mono_alt_veto_m);
  const bool orbit_allowed_regime =
      (energy.regime != core::FlightRegime::SAME_ARM_ACTIVE_SPIN) &&
      (energy.regime != core::FlightRegime::CRITICAL_DESCENT);

  // 1) Immediate commit if all 6 criteria, OOD threshold, CTR clearance, and Mod B alt gate pass.
  if (ctr_ok && geom_ok && sem_ok && var_ok && !mono_hi) {
    state_ = SlzDecisionState::COMMIT_TOUCHDOWN;
    LatchCandidateIdentity(candidate);
    out.state = state_;
    out.reason_mask = REASON_NONE;
    return out;
  }

  const uint32_t base_reason = CollectRejectReasonMask(candidate, mono_hi, var_ok);
  const uint32_t orbit_reason = (mono_hi ? REASON_MONO_ALT_VETO : REASON_NONE) |
                                (!var_ok ? REASON_OOD_UNCERTAIN : REASON_NONE);

  // 2) Active Phase 1: APPROACH_ORBIT (absorbs cruise-to-22 m descent before arming 12 s timer).
  if (state_ == SlzDecisionState::APPROACH_ORBIT && SameCandidate(candidate)) {
    const float approach_elapsed_s = GuardedElapsedSeconds(now_ns, approach_start_ns_);
    if (!ctr_ok || !orbit_allowed_regime ||
        candidate.p_safe < cfg_.p_safe_abort ||
        candidate.h_dyn_max >= cfg_.h_dyn_abort_max ||
        candidate.clearance_r_m < cfg_.r_slz_min_m ||
        candidate.step_max_m > cfg_.step_max_m) {
      return Reject(out, base_reason);
    }
    if (approach_elapsed_s >= cfg_.approach_timeout_s) {
      return Reject(out, base_reason | REASON_ORBIT_TIMEOUT);
    }
    if (energy.soc_current_pct <= (energy.soc_reserve_pct + EstimateOrbitDeltaSocPct(energy))) {
      return Reject(out, base_reason | REASON_ENERGY_UNREACHABLE);
    }

    if (IsAtOrbitRing(vehicle_enu, candidate.center_enu, agl_m)) {
      // Arm 12 s inspection budget STRICTLY upon arrival at 22 m AGL / 15 m radius.
      state_ = SlzDecisionState::ACTIVE_INSPECT_ORBIT;
      orbit_start_ns_ = now_ns;
      start_azimuth_rad_ = static_cast<float>(std::atan2(
          vehicle_enu.y - candidate.center_enu.y,
          vehicle_enu.x - candidate.center_enu.x));
      out.state = state_;
      out.reason_mask = (orbit_reason != REASON_NONE) ? orbit_reason : base_reason;
      out.inspect_setpoint = ComputeOrbitSetpoint(candidate.center_enu, 0.0f);
      return out;
    }

    out.state = SlzDecisionState::APPROACH_ORBIT;
    out.reason_mask = (orbit_reason != REASON_NONE) ? orbit_reason : base_reason;
    out.inspect_setpoint = ComputeApproachSetpoint(
        vehicle_enu, candidate.center_enu, agl_m, approach_elapsed_s);
    return out;
  }

  // 3) Active Phase 2: ACTIVE_INSPECT_ORBIT (12 s budgeted 150 deg arc around TLOF).
  if (state_ == SlzDecisionState::ACTIVE_INSPECT_ORBIT && SameCandidate(candidate)) {
    const float elapsed_s = GuardedElapsedSeconds(now_ns, orbit_start_ns_);
    if (!ctr_ok || !orbit_allowed_regime ||
        candidate.p_safe < cfg_.p_safe_abort ||
        candidate.h_dyn_max >= cfg_.h_dyn_abort_max ||
        candidate.clearance_r_m < cfg_.r_slz_min_m ||
        (candidate.step_max_m > cfg_.step_max_m && agl_m <= 28.0f)) {
      return Reject(out, base_reason);
    }
    if (candidate.theta_max_deg > cfg_.theta_screen_deg && agl_m <= 28.0f) {
      return Reject(out, base_reason | REASON_SLOPE_EXCEEDED);
    }
    if (elapsed_s >= cfg_.max_budget_s) {
      return Reject(out, base_reason | REASON_ORBIT_TIMEOUT);
    }
    const float rem_frac = std::clamp(1.0f - elapsed_s / std::max(cfg_.max_budget_s, 1e-3f), 0.0f, 1.0f);
    const float rem_soc = EstimateOrbitDeltaSocPct(energy) * rem_frac;
    if (energy.soc_current_pct <= (energy.soc_reserve_pct + rem_soc)) {
      return Reject(out, base_reason | REASON_ENERGY_UNREACHABLE);
    }

    out.state = SlzDecisionState::ACTIVE_INSPECT_ORBIT;
    out.reason_mask = (orbit_reason != REASON_NONE) ? orbit_reason : base_reason;
    out.inspect_setpoint = ComputeOrbitSetpoint(candidate.center_enu, elapsed_s);
    return out;
  }

  // 4) Initial screening for active inspection eligibility.
  const bool screen_ok = ctr_ok && orbit_allowed_regime &&
                         (candidate.p_safe >= cfg_.p_safe_screen) &&
                         (candidate.h_dyn_max < cfg_.h_dyn_abort_max) &&
                         (candidate.theta_max_deg <= cfg_.theta_screen_deg) &&
                         (candidate.sigma_z_m <= cfg_.sigma_z_screen_m) &&
                         (candidate.step_max_m <= cfg_.step_max_m) &&
                         (candidate.clearance_r_m >= cfg_.r_slz_min_m);
  const bool high_var_trigger = (!var_ok) || mono_hi;

  if (!screen_ok || !high_var_trigger) {
    return Reject(out, base_reason);
  }

  if (energy.soc_current_pct <= (energy.soc_reserve_pct + EstimateOrbitDeltaSocPct(energy))) {
    return Reject(out, base_reason | REASON_ENERGY_UNREACHABLE);
  }

  LatchCandidateIdentity(candidate);
  if (IsAtOrbitRing(vehicle_enu, candidate.center_enu, agl_m)) {
    state_ = SlzDecisionState::ACTIVE_INSPECT_ORBIT;
    orbit_start_ns_ = now_ns;
    start_azimuth_rad_ = static_cast<float>(std::atan2(
        vehicle_enu.y - candidate.center_enu.y,
        vehicle_enu.x - candidate.center_enu.x));
    out.state = state_;
    out.reason_mask = orbit_reason;
    out.inspect_setpoint = ComputeOrbitSetpoint(candidate.center_enu, 0.0f);
    return out;
  }

  state_ = SlzDecisionState::APPROACH_ORBIT;
  approach_start_ns_ = now_ns;
  out.state = state_;
  out.reason_mask = orbit_reason;
  out.inspect_setpoint = ComputeApproachSetpoint(vehicle_enu, candidate.center_enu, agl_m, 0.0f);
  return out;
}

}  // namespace vertirta::perception
