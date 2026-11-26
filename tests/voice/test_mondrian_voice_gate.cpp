// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "vertirta/voice/mondrian_voice_gate.hpp"

using namespace vertirta::voice;

TEST_CASE("MondrianVoiceConformalGate enforces canonical S0..S3 strata, alpha=0.05, MAYDAY w_m>=0.15, and <5% SoC liveness") {
  const MondrianGateConfig cfg = make_default_mondrian_config();
  REQUIRE_THAT(cfg.alpha, Catch::Matchers::WithinAbs(0.05f, 1e-6f));
  REQUIRE(cfg.mayday_asymmetric_weight_floor >= 0.15f);
  REQUIRE_THAT(cfg.loiter_liveness_min_soc_margin_pct, Catch::Matchers::WithinAbs(5.0f, 1e-6f));

  REQUIRE(classify_snr_stratum(18.0f) == SnrStratum::S0_CLEAN);
  REQUIRE(snr_stratum_name(SnrStratum::S0_CLEAN) == "S0_Clean");
  REQUIRE(classify_snr_stratum(11.0f) == SnrStratum::S1_CRUISE);
  REQUIRE(snr_stratum_name(SnrStratum::S1_CRUISE) == "S1_Cruise");
  REQUIRE(classify_snr_stratum(4.5f) == SnrStratum::S2_HOVER);
  REQUIRE(snr_stratum_name(SnrStratum::S2_HOVER) == "S2_Hover");
  REQUIRE(classify_snr_stratum(0.5f) == SnrStratum::S3_GARBLED);
  REQUIRE(snr_stratum_name(SnrStratum::S3_GARBLED) == "S3_Garbled");

  MondrianVoiceConformalGate gate(cfg);

  // 1) MAYDAY in S3_GARBLED with w_m = 0.22 (>= 0.15) triggers immediate EMERGENCY_PREEMPT
  VoiceTurnSignals mayday_sig{};
  mayday_sig.snr_db = 0.2f;
  mayday_sig.lang = VoiceLang::TR;
  mayday_sig.nospeech_prob = 0.20f;
  mayday_sig.mayday_acoustic_weight = 0.22f;
  mayday_sig.min_safety_slot_prob = 0.41f;
  mayday_sig.nbest_shannon_entropy_norm = 0.62f;

  VoiceRtaLivenessState nominal_rta{.soc_pct = 45.0f, .soc_reserve_floor_pct = 12.0f};
  const auto emg = gate.evaluate(mayday_sig, {}, nominal_rta);
  REQUIRE(emg.decision == VoiceGateAction::EMERGENCY_PREEMPT);
  REQUIRE(emg.stratum == SnrStratum::S3_GARBLED);
  REQUIRE(emg.selected_action_id == "ABORT_RETURN:Kartal_Sahil");

  // 2) <5.0% SoC loiter margin forces LIVENESS_EMERGENCY_COMMIT even when conformal set has 2 candidates
  VoiceRtaLivenessState low_soc{.soc_pct = 15.8f, .soc_reserve_floor_pct = 12.0f};  // 3.8% < 5.0%
  std::vector<VoiceCandidateScore> ambig_cands{
      {.id = "DIVERT_SLZ:Kartal_Sahil", .prob = 0.56f, .z3_sat = true},
      {.id = "DIVERT_SLZ:Dragos_Plaza", .prob = 0.44f, .z3_sat = true},
  };
  VoiceTurnSignals hover_sig{};
  hover_sig.snr_db = 5.0f;
  hover_sig.lang = VoiceLang::EN;
  hover_sig.nospeech_prob = 0.05f;
  hover_sig.min_safety_slot_prob = 0.85f;
  hover_sig.nbest_shannon_entropy_norm = 0.20f;

  const auto liveness = gate.evaluate(hover_sig, ambig_cands, low_soc);
  REQUIRE(liveness.decision == VoiceGateAction::LIVENESS_EMERGENCY_COMMIT);
  REQUIRE(liveness.selected_action_id == "DIVERT_SLZ:Kartal_Sahil");

  // 3) Same ambiguous hover turn with ample SoC triggers REQUEST_CLARIFICATION with RSN=LOW_ASR
  const auto clarify = gate.evaluate(hover_sig, ambig_cands, nominal_rta);
  REQUIRE(clarify.decision == VoiceGateAction::REQUEST_CLARIFICATION);
  REQUIRE(clarify.clarify_reason == "LOW_ASR");
  REQUIRE(clarify.clarify_option_a == "DIVERT_SLZ:Kartal_Sahil");
  REQUIRE(clarify.clarify_option_b == "DIVERT_SLZ:Dragos_Plaza");

  // 4) Z3-UNSAT candidate is ALWAYS excluded even at prob = 0.99
  std::vector<VoiceCandidateScore> unsafe_cands{
      {.id = "DIVERT_SLZ:LTFJ_CTR", .prob = 0.99f, .z3_sat = false},
  };
  const auto abstain = gate.evaluate(hover_sig, unsafe_cands, nominal_rta);
  REQUIRE(abstain.decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  REQUIRE(abstain.conformal_set_size == 0U);
}

TEST_CASE("compute_stratum_q_hat computes exact finite-sample order statistic ceil((n+1)*(1-alpha))/n") {
  std::vector<float> scores;
  scores.reserve(200U);
  for (std::size_t i = 1U; i <= 200U; ++i) {
    scores.push_back(static_cast<float>(i) / 200.0f);
  }
  // ceil(201 * 0.95) = ceil(190.95) = 191 -> scores[190] = 191/200 = 0.955
  const float q = compute_stratum_q_hat(scores, 0.05f);
  REQUIRE_THAT(q, Catch::Matchers::WithinAbs(0.955f, 1e-6f));
}

TEST_CASE("MondrianVoiceConformalGate fails closed to ABSTAIN_HOLD_SAFE on NaN/Inf SNR, probabilities, candidates, and SoC") {
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float pinf = std::numeric_limits<float>::infinity();
  const float ninf = -std::numeric_limits<float>::infinity();

  MondrianVoiceConformalGate gate(make_default_mondrian_config());
  const VoiceRtaLivenessState nominal_rta{.soc_pct = 55.0f, .soc_reserve_floor_pct = 12.0f};
  const std::vector<VoiceCandidateScore> strong_singleton{
      {.id = "DIVERT_SLZ:Kartal_Sahil", .prob = 0.96f, .z3_sat = true},
  };

  auto make_clean_sig = []() {
    VoiceTurnSignals s{};
    s.snr_db = 18.0f;
    s.lang = VoiceLang::EN;
    s.nbest_shannon_entropy_norm = 0.05f;
    s.min_safety_slot_prob = 0.98f;
    s.nospeech_prob = 0.02f;
    s.mayday_acoustic_weight = 0.0f;
    return s;
  };

  // Sanity: clean turn executes singleton.
  REQUIRE(gate.evaluate(make_clean_sig(), strong_singleton, nominal_rta).decision ==
          VoiceGateAction::EXECUTE_SINGLE);

  // Non-finite snr_db (NaN, +Inf, -Inf) MUST fail closed to ABSTAIN_HOLD_SAFE (never S0_CLEAN / EXECUTE_SINGLE).
  for (const float bad_snr : {qnan, pinf, ninf}) {
    auto s = make_clean_sig();
    s.snr_db = bad_snr;
    const auto d = gate.evaluate(s, strong_singleton, nominal_rta);
    REQUIRE(d.decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
    REQUIRE(d.selected_action_id == "HOLD_LOITER");
    REQUIRE(d.stratum == SnrStratum::S3_GARBLED);
    REQUIRE(std::isfinite(d.acoustic_penalty));
    REQUIRE(std::isfinite(d.p_min_threshold));
  }

  // Non-finite probability inputs MUST fail closed to ABSTAIN_HOLD_SAFE (never bypassing abstain or faking MAYDAY).
  {
    auto s = make_clean_sig();
    s.nospeech_prob = qnan;
    REQUIRE(gate.evaluate(s, strong_singleton, nominal_rta).decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  }
  {
    auto s = make_clean_sig();
    s.nbest_shannon_entropy_norm = qnan;
    REQUIRE(gate.evaluate(s, strong_singleton, nominal_rta).decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  }
  {
    auto s = make_clean_sig();
    s.min_safety_slot_prob = qnan;
    REQUIRE(gate.evaluate(s, strong_singleton, nominal_rta).decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  }
  {
    auto s = make_clean_sig();
    s.mayday_acoustic_weight = qnan;
    REQUIRE(gate.evaluate(s, strong_singleton, nominal_rta).decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  }
  {
    auto s = make_clean_sig();
    s.mayday_acoustic_weight = pinf;
    // +Inf corrupt MAYDAY weight must NOT falsely trigger EMERGENCY_PREEMPT; fails closed to ABSTAIN!
    REQUIRE(gate.evaluate(s, strong_singleton, nominal_rta).decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  }

  // Non-finite candidate probability MUST be excluded and fail closed to ABSTAIN_HOLD_SAFE.
  {
    const std::vector<VoiceCandidateScore> nan_cands{
        {.id = "DIVERT_SLZ:Kartal_Sahil", .prob = qnan, .z3_sat = true},
        {.id = "DIVERT_SLZ:Dragos_Plaza", .prob = pinf, .z3_sat = true},
    };
    const auto d = gate.evaluate(make_clean_sig(), nan_cands, nominal_rta);
    REQUIRE(d.decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
    REQUIRE(d.conformal_set_size == 0U);
  }

  // Non-finite SoC reserve state fails closed to ABSTAIN_HOLD_SAFE.
  {
    const VoiceRtaLivenessState nan_rta{.soc_pct = qnan, .soc_reserve_floor_pct = 12.0f};
    const auto d = gate.evaluate(make_clean_sig(), strong_singleton, nan_rta);
    REQUIRE(d.decision == VoiceGateAction::ABSTAIN_HOLD_SAFE);
  }

  // compute_stratum_q_hat fails closed to 1.0f on NaN alpha or NaN/Inf nonconformity scores.
  const std::vector<float> good_scores{0.1f, 0.2f, 0.3f};
  REQUIRE_THAT(compute_stratum_q_hat(good_scores, qnan), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
  const std::vector<float> bad_scores{0.1f, qnan, 0.3f};
  REQUIRE_THAT(compute_stratum_q_hat(bad_scores, 0.05f), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
}
