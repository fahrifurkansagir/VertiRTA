#include "vertirta/copilot/conformal_gate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vertirta::copilot {

std::vector<float> normalized_branch_probs(std::span<const float> sum_logprobs,
                                           std::span<const std::size_t> token_counts,
                                           float gamma,
                                           float tau) {
  const std::size_t n = std::min(sum_logprobs.size(), token_counts.size());
  std::vector<float> probs(n, 0.0f);
  if (n == 0U) {
    return probs;
  }
  const double inv_tau = 1.0 / std::max(1e-6, static_cast<double>(tau));
  std::vector<double> scaled(n, 0.0);
  double max_s = -1e300;
  for (std::size_t i = 0U; i < n; ++i) {
    const double len = static_cast<double>(std::max<std::size_t>(1U, token_counts[i]));
    const double norm_lp =
        static_cast<double>(sum_logprobs[i]) / std::pow(len, static_cast<double>(gamma));
    scaled[i] = norm_lp * inv_tau;
    if (scaled[i] > max_s) {
      max_s = scaled[i];
    }
  }
  double sum_exp = 0.0;
  for (std::size_t i = 0U; i < n; ++i) {
    scaled[i] = std::exp(scaled[i] - max_s);
    sum_exp += scaled[i];
  }
  if (sum_exp <= 0.0) {
    const float uniform = 1.0f / static_cast<float>(n);
    std::fill(probs.begin(), probs.end(), uniform);
    return probs;
  }
  for (std::size_t i = 0U; i < n; ++i) {
    probs[i] = static_cast<float>(scaled[i] / sum_exp);
  }
  return probs;
}

float compute_split_conformal_qhat(std::span<const float> nonconformity_scores,
                                   float alpha) noexcept {
  if (nonconformity_scores.empty()) {
    return 1.0f;
  }
  const std::size_t n = nonconformity_scores.size();
  std::vector<float> sorted(nonconformity_scores.begin(), nonconformity_scores.end());
  std::sort(sorted.begin(), sorted.end());

  const double clamped_alpha = std::clamp(static_cast<double>(alpha), 0.0, 1.0);
  const double raw_k = std::ceil(static_cast<double>(n + 1U) * (1.0 - clamped_alpha));
  const std::size_t k = static_cast<std::size_t>(
      std::clamp(raw_k, 1.0, static_cast<double>(n)));
  return sorted[k - 1U];
}

ConformalDecision evaluate_conformal_gate(std::span<const CandidateScore> candidates,
                                          float q_hat,
                                          const RtaStateSnapshotV1& st) noexcept {
  ConformalDecision out{};
  const float clamped_q = std::clamp(q_hat, 0.0f, 1.0f);
  out.p_min_threshold = 1.0f - clamped_q;

  std::vector<const CandidateScore*> safe_set;
  std::vector<const CandidateScore*> all_sat;
  safe_set.reserve(candidates.size());
  all_sat.reserve(candidates.size());

  for (const auto& c : candidates) {
    if (!c.z3_sat) {
      continue;
    }
    all_sat.push_back(&c);
    if (c.prob >= out.p_min_threshold) {
      safe_set.push_back(&c);
    }
  }

  auto desc_by_prob = [](const CandidateScore* a, const CandidateScore* b) {
    return a->prob > b->prob;
  };
  std::sort(safe_set.begin(), safe_set.end(), desc_by_prob);
  std::sort(all_sat.begin(), all_sat.end(), desc_by_prob);
  out.safe_set_size = safe_set.size();

  if (safe_set.size() == 1U) {
    out.outcome = ConformalOutcome::EXECUTE_SINGLE;
    out.opt_a = safe_set[0]->id;
    return out;
  }

  // Deterministic safety liveness override: never loiter/clarify when loiter margin < 5.0% SoC.
  const float loiter_soc_margin = st.soc_pct - st.soc_reserve_floor_pct;
  if (loiter_soc_margin < kLoiterLivenessMinSocMarginPct) {
    out.outcome = ConformalOutcome::LIVENESS_EMERGENCY_COMMIT;
    // Prefer highest-ranked Z3-SAT SLZ from snapshot; fall back to highest-prob Z3-SAT candidate.
    const std::size_t n_slz =
        (st.num_slz <= kMaxSlzCandidates) ? st.num_slz : kMaxSlzCandidates;
    for (std::size_t i = 0U; i < n_slz; ++i) {
      const auto slz_id = st.ranked_slzs[i].id_view();
      for (const CandidateScore* c : all_sat) {
        if (c->id == slz_id) {
          out.opt_a = c->id;
          return out;
        }
      }
    }
    if (!all_sat.empty()) {
      out.opt_a = all_sat[0]->id;
    } else if (n_slz > 0U) {
      out.opt_a = std::string(st.ranked_slzs[0].id_view());
    } else {
      out.opt_a = "HOLD_LOITER";
    }
    return out;
  }

  if (safe_set.size() > 1U) {
    out.outcome = ConformalOutcome::ACTIVE_ELICITATION;
    out.opt_a = safe_set[0]->id;
    out.opt_b = safe_set[1]->id;
    return out;
  }

  out.outcome = ConformalOutcome::ABSTAIN_LOITER;
  out.opt_a = "HOLD_LOITER";
  return out;
}

}  // namespace vertirta::copilot
