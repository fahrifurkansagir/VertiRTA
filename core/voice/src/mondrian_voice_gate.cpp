// SPDX-License-Identifier: Apache-2.0
#include "vertirta/voice/mondrian_voice_gate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vertirta::voice {

MondrianGateConfig make_default_mondrian_config() noexcept {
  return MondrianGateConfig{};
}

float compute_stratum_q_hat(
    std::span<const float> nonconformity_scores,
    float alpha) noexcept {
  if (nonconformity_scores.empty() || !std::isfinite(alpha)) {
    return 1.0f;
  }
  std::vector<float> sorted;
  sorted.reserve(nonconformity_scores.size());
  for (const float s : nonconformity_scores) {
    if (!std::isfinite(s)) {
      return 1.0f;
    }
    sorted.push_back(s);
  }
  std::sort(sorted.begin(), sorted.end());

  const std::size_t n = sorted.size();
  const double clamped_alpha = std::clamp(static_cast<double>(alpha), 1e-6, 1.0 - 1e-6);
  const double raw_k = std::ceil(static_cast<double>(n + 1U) * (1.0 - clamped_alpha));
  const std::size_t k = static_cast<std::size_t>(std::clamp(raw_k, 1.0, static_cast<double>(n)));
  return std::clamp(sorted[k - 1U], 0.0f, 1.0f);
}

MondrianVoiceConformalGate::MondrianVoiceConformalGate(MondrianGateConfig cfg) noexcept
    : cfg_(cfg) {
  cfg_.alpha = std::isfinite(cfg_.alpha)
                   ? std::clamp(cfg_.alpha, 1e-4f, 0.50f)
                   : kDefaultConformalAlpha;
  cfg_.loiter_liveness_min_soc_margin_pct =
      std::isfinite(cfg_.loiter_liveness_min_soc_margin_pct)
          ? std::max(kLoiterLivenessMinSocMarginPct, cfg_.loiter_liveness_min_soc_margin_pct)
          : kLoiterLivenessMinSocMarginPct;
  cfg_.mayday_asymmetric_weight_floor =
      std::isfinite(cfg_.mayday_asymmetric_weight_floor)
          ? std::max(kDefaultMaydayAsymmetricWeightFloor, cfg_.mayday_asymmetric_weight_floor)
          : kDefaultMaydayAsymmetricWeightFloor;
  cfg_.nospeech_abstain_threshold = std::isfinite(cfg_.nospeech_abstain_threshold)
                                        ? std::clamp(cfg_.nospeech_abstain_threshold, 0.0f, 1.0f)
                                        : 0.70f;
  cfg_.lambda_entropy = std::isfinite(cfg_.lambda_entropy) ? std::max(0.0f, cfg_.lambda_entropy) : 0.12f;
  cfg_.lambda_slot_uncertainty =
      std::isfinite(cfg_.lambda_slot_uncertainty) ? std::max(0.0f, cfg_.lambda_slot_uncertainty) : 0.18f;
  cfg_.lambda_nospeech = std::isfinite(cfg_.lambda_nospeech) ? std::max(0.0f, cfg_.lambda_nospeech) : 0.25f;
  for (float& q : cfg_.snr_q_hat) {
    q = std::isfinite(q) ? std::clamp(q, 0.0f, 1.0f) : 1.0f;
  }
  for (float& q : cfg_.lang_q_hat) {
    q = std::isfinite(q) ? std::clamp(q, 0.0f, 1.0f) : 1.0f;
  }
}

MondrianGateDecision MondrianVoiceConformalGate::evaluate(
    const VoiceTurnSignals& sig,
    std::span<const VoiceCandidateScore> candidates,
    const VoiceRtaLivenessState& rta_state) const {
  MondrianGateDecision out{};
  out.lang = sig.lang;

  // Fail-closed finite guard on all acoustic/probability inputs BEFORE classify_snr_stratum,
  // std::clamp, MAYDAY check, or abstain check. Never allow NaN/Inf to propagate through std::clamp
  // or bypass abstain/MAYDAY logic.
  const bool sig_finite =
      std::isfinite(sig.snr_db) &&
      std::isfinite(sig.nbest_shannon_entropy_norm) &&
      std::isfinite(sig.min_safety_slot_prob) &&
      std::isfinite(sig.nospeech_prob) &&
      std::isfinite(sig.mayday_acoustic_weight);
  if (!sig_finite) {
    out.stratum = SnrStratum::S3_GARBLED;
    out.effective_q_hat = 1.0f;
    out.acoustic_penalty = 0.95f;
    out.p_min_threshold = 1.0f;
    out.conformal_set_size = 0U;
    out.decision = VoiceGateAction::ABSTAIN_HOLD_SAFE;
    out.selected_action_id = "HOLD_LOITER";
    return out;
  }

  out.stratum = classify_snr_stratum(sig.snr_db);

  const auto snr_idx = static_cast<std::size_t>(out.stratum);
  const auto lang_idx = static_cast<std::size_t>(out.lang);
  // Conservative hierarchical Mondrian quantile: never under-cover either SNR stratum or language group.
  const float q_snr = cfg_.snr_q_hat[std::min<std::size_t>(snr_idx, 3U)];
  const float q_lang = cfg_.lang_q_hat[std::min<std::size_t>(lang_idx, 2U)];
  out.effective_q_hat = std::clamp(std::max(q_snr, q_lang), 0.0f, 1.0f);

  const float h_norm = std::clamp(sig.nbest_shannon_entropy_norm, 0.0f, 1.0f);
  const float slot_unc = 1.0f - std::clamp(sig.min_safety_slot_prob, 0.0f, 1.0f);
  const float ns = std::clamp(sig.nospeech_prob, 0.0f, 1.0f);
  out.acoustic_penalty = std::clamp(
      cfg_.lambda_entropy * h_norm + cfg_.lambda_slot_uncertainty * slot_unc + cfg_.lambda_nospeech * ns,
      0.0f, 0.95f);

  // Branch 1: MAYDAY / PAN-PAN asymmetric emergency priority (w_m >= 0.15) ALWAYS preempts silence/clarify.
  if (sig.mayday_acoustic_weight >= cfg_.mayday_asymmetric_weight_floor) {
    out.decision = VoiceGateAction::EMERGENCY_PREEMPT;
    out.p_min_threshold = 1.0f - out.effective_q_hat;
    // Prefer highest-probability finite Z3-SAT candidate if supplied, else deterministic safe abort.
    const VoiceCandidateScore* best_sat = nullptr;
    for (const auto& c : candidates) {
      if (c.z3_sat && std::isfinite(c.prob) && (best_sat == nullptr || c.prob > best_sat->prob)) {
        best_sat = &c;
      }
    }
    out.selected_action_id = (best_sat != nullptr) ? best_sat->id : std::string("ABORT_RETURN:Kartal_Sahil");
    out.conformal_set_size = (best_sat != nullptr) ? 1U : 0U;
    return out;
  }

  // Build Z3-SAT conformal set C_g with noise-widened threshold:
  // Higher acoustic uncertainty (H_nbest, 1 - p_min_slot, p_nospeech) widens the conformal
  // prediction set C_g (lowers p_min_threshold), preventing noisy turns from falsely collapsing
  // into overconfident singletons and forcing clarification (or liveness commit when <5% SoC).
  out.p_min_threshold = std::clamp(1.0f - out.effective_q_hat - out.acoustic_penalty, 0.0f, 1.0f);

  std::vector<const VoiceCandidateScore*> safe_set;
  std::vector<const VoiceCandidateScore*> all_sat;
  safe_set.reserve(candidates.size());
  all_sat.reserve(candidates.size());
  for (const auto& c : candidates) {
    if (!c.z3_sat || !std::isfinite(c.prob)) {
      continue;
    }
    all_sat.push_back(&c);
    if (c.prob >= out.p_min_threshold) {
      safe_set.push_back(&c);
    }
  }
  auto desc_prob = [](const VoiceCandidateScore* a, const VoiceCandidateScore* b) {
    return a->prob > b->prob;
  };
  std::sort(safe_set.begin(), safe_set.end(), desc_prob);
  std::sort(all_sat.begin(), all_sat.end(), desc_prob);
  out.conformal_set_size = safe_set.size();

  // Fail closed to abstain if battery reserve telemetry is non-finite.
  if (!std::isfinite(rta_state.soc_pct) || !std::isfinite(rta_state.soc_reserve_floor_pct)) {
    out.decision = VoiceGateAction::ABSTAIN_HOLD_SAFE;
    out.selected_action_id = "HOLD_LOITER";
    return out;
  }

  // Branch 2: Deterministic < 5.0% SoC margin LIVENESS_EMERGENCY_COMMIT override (Erratum #7).
  // Never loiter or request clarification when battery reserve margin is below 5.0%.
  const float loiter_soc_margin = rta_state.soc_pct - rta_state.soc_reserve_floor_pct;
  if (loiter_soc_margin < cfg_.loiter_liveness_min_soc_margin_pct && safe_set.size() != 1U) {
    out.decision = VoiceGateAction::LIVENESS_EMERGENCY_COMMIT;
    if (!safe_set.empty()) {
      out.selected_action_id = safe_set.front()->id;
    } else if (!all_sat.empty()) {
      out.selected_action_id = all_sat.front()->id;
    } else {
      out.selected_action_id = "ABORT_RETURN:Kartal_Sahil";
    }
    return out;
  }

  // Branch 3: High nospeech probability or empty conformal set -> fail-closed safe hold/abstain.
  if (ns > cfg_.nospeech_abstain_threshold || safe_set.empty()) {
    out.decision = VoiceGateAction::ABSTAIN_HOLD_SAFE;
    out.selected_action_id = "HOLD_LOITER";
    return out;
  }

  // Branch 4: Exact singleton Z3-SAT conformal set -> execute immediately.
  if (safe_set.size() == 1U) {
    out.decision = VoiceGateAction::EXECUTE_SINGLE;
    out.selected_action_id = safe_set.front()->id;
    return out;
  }

  // Branch 5: Multi-element conformal set -> targeted bilingual clarification.
  out.decision = VoiceGateAction::REQUEST_CLARIFICATION;
  out.clarify_option_a = safe_set[0]->id;
  out.clarify_option_b = safe_set[1]->id;
  out.clarify_reason =
      (out.stratum == SnrStratum::S2_HOVER || out.stratum == SnrStratum::S3_GARBLED)
          ? "LOW_ASR"
          : "AMBIG";
  return out;
}

}  // namespace vertirta::voice
