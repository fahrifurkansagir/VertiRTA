// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vertirta/voice/voice_types.hpp"

namespace vertirta::voice {

struct VoiceCandidateScore {
  std::string id{};
  float prob{0.0f};
  bool z3_sat{false};
};

// Lightweight view mirroring vertirta::copilot::RtaStateSnapshotV1 safety-liveness fields
// so core/voice builds 100% standalone while remaining zero-copy compatible with core/copilot.
struct VoiceRtaLivenessState {
  float soc_pct{50.0f};
  float soc_reserve_floor_pct{12.0f};
  std::uint64_t seq{0U};
  std::uint64_t t_sim_us{0U};

  template <typename SnapshotT>
  [[nodiscard]] static VoiceRtaLivenessState from_snapshot(const SnapshotT& st) noexcept {
    VoiceRtaLivenessState out{};
    out.soc_pct = st.soc_pct;
    out.soc_reserve_floor_pct = st.soc_reserve_floor_pct;
    return out;
  }
};

struct VoiceTurnSignals {
  float snr_db{12.0f};
  VoiceLang lang{VoiceLang::TR};
  float nbest_shannon_entropy_norm{0.0f};  // [0, 1]
  float min_safety_slot_prob{1.0f};        // [0, 1]
  float nospeech_prob{0.0f};               // [0, 1]
  float mayday_acoustic_weight{0.0f};      // [0, 1]
};

struct MondrianGateConfig {
  float alpha{kDefaultConformalAlpha};                                   // Locked 0.05
  float loiter_liveness_min_soc_margin_pct{kLoiterLivenessMinSocMarginPct};  // Locked 5.0%
  float mayday_asymmetric_weight_floor{kDefaultMaydayAsymmetricWeightFloor}; // >= 0.15
  float nospeech_abstain_threshold{0.70f};
  float lambda_entropy{0.12f};
  float lambda_slot_uncertainty{0.18f};
  float lambda_nospeech{0.25f};
  // Calibrated finite-sample quantiles q_hat per SnrStratum [S0, S1, S2, S3].
  std::array<float, 4U> snr_q_hat{0.35f, 0.45f, 0.60f, 0.78f};
  // Calibrated finite-sample quantiles q_hat per VoiceLang [TR, EN, MIX].
  std::array<float, 3U> lang_q_hat{0.42f, 0.42f, 0.50f};
};

struct MondrianGateDecision {
  VoiceGateAction decision{VoiceGateAction::ABSTAIN_HOLD_SAFE};
  SnrStratum stratum{SnrStratum::S3_GARBLED};
  VoiceLang lang{VoiceLang::TR};
  float effective_q_hat{1.0f};
  float acoustic_penalty{0.0f};
  float p_min_threshold{0.0f};
  std::size_t conformal_set_size{0U};
  std::string selected_action_id{};
  std::string clarify_option_a{};
  std::string clarify_option_b{};
  std::string clarify_reason{};  // "LOW_ASR" in S2/S3, "AMBIG" in S0/S1
};

[[nodiscard]] MondrianGateConfig make_default_mondrian_config() noexcept;

// Compute exact finite-sample split-conformal quantile ceil((n+1)*(1-alpha))/n.
[[nodiscard]] float compute_stratum_q_hat(
    std::span<const float> nonconformity_scores,
    float alpha = kDefaultConformalAlpha) noexcept;

// 4-Stratum SNR + Language-Group Mondrian Conformal KnowNo Gate with MAYDAY priority
// and deterministic < 5.0% SoC margin LIVENESS_EMERGENCY_COMMIT override.
class MondrianVoiceConformalGate {
 public:
  explicit MondrianVoiceConformalGate(MondrianGateConfig cfg = make_default_mondrian_config()) noexcept;

  [[nodiscard]] const MondrianGateConfig& config() const noexcept { return cfg_; }

  [[nodiscard]] MondrianGateDecision evaluate(
      const VoiceTurnSignals& sig,
      std::span<const VoiceCandidateScore> candidates,
      const VoiceRtaLivenessState& rta_state) const;

 private:
  MondrianGateConfig cfg_{};
};

}  // namespace vertirta::voice
