#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <numeric>
#include <vector>

#include "vertirta/copilot/conformal_gate.hpp"

using vertirta::copilot::CandidateScore;
using vertirta::copilot::ConformalOutcome;

TEST_CASE("Split Conformal quantile matches ceil((200+1)*0.95)/200 = 0.955 for n_cal=200, alpha=0.05") {
  std::vector<float> scores(200U);
  for (std::size_t i = 0U; i < 200U; ++i) {
    scores[i] = static_cast<float>(i + 1U) / 200.0f;
  }
  REQUIRE(vertirta::copilot::compute_split_conformal_qhat(scores, 0.05f) ==
          Catch::Approx(191.0f / 200.0f));
}

TEST_CASE("normalized_branch_probs length-normalizes logprobs and sums to 1.0") {
  const std::array<float, 3> lps = {{-1.0f, -2.0f, -4.0f}};
  const std::array<std::size_t, 3> lens = {{4U, 4U, 4U}};
  const auto probs = vertirta::copilot::normalized_branch_probs(lps, lens, 0.5f, 1.0f);
  REQUIRE(probs.size() == 3U);
  const float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
  REQUIRE(sum == Catch::Approx(1.0f).margin(1e-5f));
  REQUIRE(probs[0] > probs[1]);
  REQUIRE(probs[1] > probs[2]);
}

TEST_CASE("Conformal gate routes |C_safe|==1 -> EXECUTE, >1 -> CLARIFY, ==0 -> ABSTAIN, and <5% SoC -> LIVENESS") {
  auto st = vertirta::copilot::make_reference_400v_sml_snapshot();  // soc=34%, reserve=12% -> margin 22% > 5%

  // q_hat = 0.68 -> p_min = 0.32
  const std::array<CandidateScore, 2> c_single = {{
      {"Kartal_Sahil", 0.89f, true},
      {"Dragos_Plaza", 0.07f, false},
  }};
  const auto d1 = vertirta::copilot::evaluate_conformal_gate(c_single, 0.68f, st);
  REQUIRE(d1.outcome == ConformalOutcome::EXECUTE_SINGLE);
  REQUIRE(d1.opt_a == "Kartal_Sahil");
  REQUIRE(d1.safe_set_size == 1U);

  const std::array<CandidateScore, 3> c_ambig = {{
      {"Kartal_Sahil", 0.44f, true},
      {"Cevizli_Helipad", 0.37f, true},
      {"Dragos_Plaza", 0.14f, false},
  }};
  const auto d2 = vertirta::copilot::evaluate_conformal_gate(c_ambig, 0.68f, st);
  REQUIRE(d2.outcome == ConformalOutcome::ACTIVE_ELICITATION);
  REQUIRE(d2.opt_a == "Kartal_Sahil");
  REQUIRE(d2.opt_b == "Cevizli_Helipad");
  REQUIRE(d2.safe_set_size == 2U);

  const std::array<CandidateScore, 2> c_noise = {{
      {"Kartal_Sahil", 0.24f, true},
      {"Cevizli_Helipad", 0.20f, true},
  }};
  const auto d0 = vertirta::copilot::evaluate_conformal_gate(c_noise, 0.68f, st);
  REQUIRE(d0.outcome == ConformalOutcome::ABSTAIN_LOITER);
  REQUIRE(d0.opt_a == "HOLD_LOITER");
  REQUIRE(d0.safe_set_size == 0U);

  // Deterministic <5.0% SoC loiter liveness override bypasses clarification & abstention!
  st.soc_pct = 16.5f;  // Margin = 16.5 - 12.0 = 4.5% < 5.0% floor
  const auto d_live_ambig = vertirta::copilot::evaluate_conformal_gate(c_ambig, 0.68f, st);
  REQUIRE(d_live_ambig.outcome == ConformalOutcome::LIVENESS_EMERGENCY_COMMIT);
  REQUIRE(d_live_ambig.opt_a == "Kartal_Sahil");

  const auto d_live_noise = vertirta::copilot::evaluate_conformal_gate(c_noise, 0.68f, st);
  REQUIRE(d_live_noise.outcome == ConformalOutcome::LIVENESS_EMERGENCY_COMMIT);
  REQUIRE(d_live_noise.opt_a == "Kartal_Sahil");
}
