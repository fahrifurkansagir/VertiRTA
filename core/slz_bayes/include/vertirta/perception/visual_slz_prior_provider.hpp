// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>

#include "vertirta/core/frames.hpp"
#include "vertirta/perception/active_inspection_policy.hpp"
#include "vertirta/perception/slz_belief_grid.hpp"
#include "vertirta/rta/slz_prior_provider.hpp"

namespace vertirta::perception {

// Implements Plan 1A's canonical zero-heap vertirta::rta::ISlzPriorProvider port
// using the 2.5D Bayesian SLZ belief grid and 2-phase ActiveInspectionPolicy.
class VisualSlzPriorProvider final : public rta::ISlzPriorProvider {
 public:
  static constexpr uint64_t kDefaultStalenessWarnNs = 250'000'000ULL;  // 250 ms contract

  explicit VisualSlzPriorProvider(
      ActiveInspectionPolicy::Config policy_cfg = {},
      uint64_t staleness_warn_ns = kDefaultStalenessWarnNs) noexcept
      : policy_(policy_cfg), staleness_warn_ns_(staleness_warn_ns) {}

  // Evaluates up to 8 candidate priors against the live belief grid, steps the active
  // inspection state machine on the top-ranked candidate, and caches a zero-heap snapshot.
  ActiveInspectionDecision UpdateFromGrid(
      const SlzBeliefGrid& grid,
      std::span<const rta::SlzPrior> catalog_candidates,
      const core::Vec3Enu& vehicle_enu,
      float agl_m,
      DepthMode mode,
      const EnergyGateInput& energy,
      uint64_t now_ns) noexcept {
    rta::SlzProviderSnapshot snap{};
    snap.stamp_ns = now_ns;
    snap.stale = false;

    const std::size_t n = std::min<std::size_t>(catalog_candidates.size(), snap.priors.size());
    snap.count = static_cast<uint8_t>(n);

    for (std::size_t i = 0; i < n; ++i) {
      rta::SlzPrior eval = grid.EvaluateDisk(catalog_candidates[i].center_enu,
                                             policy_.config().r_slz_min_m);
      std::memcpy(eval.id, catalog_candidates[i].id, sizeof(eval.id));
      eval.lat_deg = catalog_candidates[i].lat_deg;
      eval.lon_deg = catalog_candidates[i].lon_deg;
      eval.elev_amsl_m = catalog_candidates[i].elev_amsl_m;
      eval.center_enu = catalog_candidates[i].center_enu;
      eval.eta_s = catalog_candidates[i].eta_s;
      eval.soc_margin_pct = energy.soc_current_pct - energy.soc_reserve_pct;
      eval.reason_mask |= (catalog_candidates[i].reason_mask & REASON_CTR_VIOLATION);
      snap.priors[i] = eval;
    }

    // Deterministic ranking: CTR-clean first, then higher p_safe, then lower OOD variance.
    std::sort(snap.priors.begin(), snap.priors.begin() + static_cast<std::ptrdiff_t>(n),
              [](const rta::SlzPrior& a, const rta::SlzPrior& b) noexcept {
                const bool a_ctr = (a.reason_mask & REASON_CTR_VIOLATION) != 0u;
                const bool b_ctr = (b.reason_mask & REASON_CTR_VIOLATION) != 0u;
                if (a_ctr != b_ctr) return !a_ctr;
                if (a.p_safe != b.p_safe) return a.p_safe > b.p_safe;
                return a.slz_ood_variance < b.slz_ood_variance;
              });

    for (std::size_t i = 0; i < n; ++i) {
      snap.priors[i].rank = static_cast<uint8_t>(i);
    }

    ActiveInspectionDecision decision{};
    decision.timestamp_ns = now_ns;
    decision.depth_mode = mode;

    if (n > 0u) {
      decision = policy_.Step(snap.priors[0], vehicle_enu, agl_m, mode, energy, now_ns);
      snap.priors[0].reason_mask = decision.reason_mask;
      snap.active_decision_state = static_cast<uint8_t>(decision.state);
      if (decision.inspect_setpoint.has_value()) {
        snap.gimbal_pitch_rad = decision.inspect_setpoint->gimbal_pitch_rad;
        snap.gimbal_yaw_rad = decision.inspect_setpoint->gimbal_yaw_rad;
      }
    } else {
      snap.active_decision_state = static_cast<uint8_t>(SlzDecisionState::REJECT_DIVERT_NEXT);
      decision.state = SlzDecisionState::REJECT_DIVERT_NEXT;
      decision.reason_mask = REASON_CLEARANCE_INSUFF;
    }

    last_soc_reserve_pct_ = energy.soc_reserve_pct;
    latest_snapshot_ = snap;
    last_decision_ = decision;
    return decision;
  }

  // Directly updates cached snapshot from an already-evaluated decision (e.g. ROS 2 node).
  void CommitDecisionSnapshot(
      const ActiveInspectionDecision& decision,
      std::span<const rta::SlzPrior> ranked_priors = {}) noexcept {
    rta::SlzProviderSnapshot snap{};
    snap.stamp_ns = decision.timestamp_ns;
    snap.stale = false;
    snap.active_decision_state = static_cast<uint8_t>(decision.state);
    if (decision.inspect_setpoint.has_value()) {
      snap.gimbal_pitch_rad = decision.inspect_setpoint->gimbal_pitch_rad;
      snap.gimbal_yaw_rad = decision.inspect_setpoint->gimbal_yaw_rad;
    }
    if (!ranked_priors.empty()) {
      const std::size_t n = std::min<std::size_t>(ranked_priors.size(), snap.priors.size());
      snap.count = static_cast<uint8_t>(n);
      for (std::size_t i = 0; i < n; ++i) {
        snap.priors[i] = ranked_priors[i];
        snap.priors[i].rank = static_cast<uint8_t>(i);
      }
    } else {
      snap.count = 1u;
      snap.priors[0] = decision.candidate;
      snap.priors[0].reason_mask = decision.reason_mask;
      snap.priors[0].rank = 0u;
    }
    latest_snapshot_ = snap;
    last_decision_ = decision;
  }

  [[nodiscard]] rta::SlzProviderSnapshot query_priors(
      const core::Vec3Enu& /*vehicle_enu*/,
      float soc_pct,
      uint64_t now_ns) const noexcept override {
    rta::SlzProviderSnapshot out = latest_snapshot_;
    const bool is_stale = (out.stamp_ns == 0u) || (now_ns < out.stamp_ns) ||
                          ((now_ns - out.stamp_ns) > staleness_warn_ns_);
    out.stale = is_stale;
    for (uint8_t i = 0; i < out.count && i < out.priors.size(); ++i) {
      out.priors[i].soc_margin_pct = soc_pct - last_soc_reserve_pct_;
    }
    return out;
  }

  void Reset() noexcept {
    policy_.Reset();
    latest_snapshot_ = {};
    last_decision_ = {};
    last_soc_reserve_pct_ = 12.0f;
  }

  [[nodiscard]] const ActiveInspectionPolicy& policy() const noexcept { return policy_; }
  [[nodiscard]] const ActiveInspectionDecision& last_decision() const noexcept {
    return last_decision_;
  }

 private:
  ActiveInspectionPolicy policy_{};
  uint64_t staleness_warn_ns_{kDefaultStalenessWarnNs};
  float last_soc_reserve_pct_{12.0f};
  rta::SlzProviderSnapshot latest_snapshot_{};
  ActiveInspectionDecision last_decision_{};
};

}  // namespace vertirta::perception
