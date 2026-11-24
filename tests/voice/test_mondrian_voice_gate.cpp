// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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
