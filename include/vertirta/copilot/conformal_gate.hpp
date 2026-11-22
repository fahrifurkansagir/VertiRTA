#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vertirta/copilot/rta_state_contract.hpp"

namespace vertirta::copilot {

inline constexpr float kDefaultConformalAlpha = 0.05f;
inline constexpr float kLoiterLivenessMinSocMarginPct = 5.0f;

struct CandidateScore {
  std::string id;
  float prob{0.0f};
  bool z3_sat{false};
};

enum class ConformalOutcome : uint8_t {
  EXECUTE_SINGLE = 0,
  ACTIVE_ELICITATION = 1,
  ABSTAIN_LOITER = 2,
  LIVENESS_EMERGENCY_COMMIT = 3,
};

struct ConformalDecision {
  ConformalOutcome outcome{ConformalOutcome::ABSTAIN_LOITER};
  std::string opt_a;
  std::string opt_b;
  std::size_t safe_set_size{0U};
  float p_min_threshold{0.0f};
};

// Length-normalized softmax probabilities over discrete GBNF action/target branches (gamma=0.5, tau=1.0).
[[nodiscard]] std::vector<float> normalized_branch_probs(
    std::span<const float> sum_logprobs,
    std::span<const std::size_t> token_counts,
    float gamma = 0.5f,
    float tau = 1.0f);

// Finite-sample Split Conformal quantile: ceil((n_cal + 1) * (1 - alpha)) / n_cal order statistic.
[[nodiscard]] float compute_split_conformal_qhat(std::span<const float> nonconformity_scores,
                                                 float alpha = kDefaultConformalAlpha) noexcept;

// Evaluate KnowNo conformal prediction set intersected with Z3-SAT candidates,
// enforcing the deterministic <5.0% SoC loiter liveness override.
[[nodiscard]] ConformalDecision evaluate_conformal_gate(
    std::span<const CandidateScore> candidates,
    float q_hat,
    const RtaStateSnapshotV1& st) noexcept;

}  // namespace vertirta::copilot
