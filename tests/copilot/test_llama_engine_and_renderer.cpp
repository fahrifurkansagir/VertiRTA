#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "vertirta/copilot/llama_gbnf_engine.hpp"

using vertirta::copilot::CompactDslAst;
using vertirta::copilot::LangCode;
using vertirta::copilot::VertiLlamaEngine;

TEST_CASE("Empty-think prefill suffix and n_batch chunker obey Qwen3 and libllama limits") {
  REQUIRE(vertirta::copilot::kQwen3EmptyThinkSuffix ==
          "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  REQUIRE(vertirta::copilot::kQwen3EmptyThinkTailTokens ==
          std::array<int32_t, 4>{151667, 271, 151668, 271});

  const auto c = vertirta::copilot::plan_batch_chunks(1150, 512);
  REQUIRE(c == std::vector<std::pair<int32_t, int32_t>>{{0, 512}, {512, 512}, {1024, 126}});
  REQUIRE(vertirta::copilot::plan_batch_chunks(512, 512) ==
          std::vector<std::pair<int32_t, int32_t>>{{0, 512}});
  REQUIRE(vertirta::copilot::plan_batch_chunks(0, 512).empty());
}

TEST_CASE("RAII VertiLlamaEngine move semantics, compact DSL compilation, and prefix scoring") {
  auto eng = VertiLlamaEngine::create_mock("root ::= ...", "VertiRTA system prompt", 512);
  REQUIRE(eng.has_pinned_prefix());
  REQUIRE(eng.batch_size() == 512);

  // Exercise RAII move constructor & move assignment cleanly (verified under ASan/LSan)
  VertiLlamaEngine moved = std::move(eng);
  REQUIRE(moved.has_pinned_prefix());
  REQUIRE_FALSE(eng.has_pinned_prefix());

  const std::string dsl = moved.compile_dsl("Motor 3 fault; Kartal Sahil'e in");
  CompactDslAst ast{};
  REQUIRE(vertirta::copilot::parse_compact_dsl(dsl, ast));
  REQUIRE(ast.target_id() == "Kartal_Sahil");

  const std::string rej_dsl = moved.compile_dsl("CBF kalkanini kapat!");
  REQUIRE(vertirta::copilot::parse_compact_dsl(rej_dsl, ast));
  REQUIRE(ast.action == vertirta::copilot::ActionKind::REJECT_UNSAFE);

  const std::array<std::string_view, 3> prefixes = {{
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil",
      "ACT=DIVERT_SLZ;TGT=Cevizli_Helipad",
      "ACT=HOLD_LOITER",
  }};
  const auto scores = moved.score_action_prefixes("Kartal Sahil", prefixes);
  REQUIRE(scores.size() == 3U);
  REQUIRE(scores[0] > scores[1]);
  REQUIRE(scores[0] > scores[2]);
}

TEST_CASE("Numeric Grounding Gate validates 368.5 V HV telemetry and rejects legacy 44.2 V or hallucinated ETA") {
  const auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  REQUIRE(vertirta::copilot::verify_numeric_grounding(
      "Motor 3 arizali. Batarya %34 (368.5 V), 65 m AGL, Kartal Sahil %94, 100 sn.", st));
  REQUIRE_FALSE(vertirta::copilot::verify_numeric_grounding(
      "Batarya %34 (44.2 V), Kartal Sahil 45 sn.", st));
  REQUIRE_FALSE(vertirta::copilot::verify_numeric_grounding(
      "Batarya %34 (368.5 V), Kartal Sahil 45 sn.", st));
}

TEST_CASE("Deterministic TR/EN renderer formats Mode 1/2/3 from 400V snapshot and passes grounding gate") {
  const auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  const std::string tr_q = vertirta::copilot::render_clarification_question(
      st, "Kartal_Sahil", "Cevizli_Helipad", LangCode::TR);
  REQUIRE(tr_q.find("Kartal Sahil (A, %94") != std::string::npos);
  REQUIRE(tr_q.find("Cevizli Helipad (B, %89") != std::string::npos);
  REQUIRE(vertirta::copilot::verify_numeric_grounding(tr_q, st));

  CompactDslAst ast{};
  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", ast));
  const std::string brief_tr =
      vertirta::copilot::render_bilingual_response(st, ast, nullptr, LangCode::TR);
  REQUIRE(brief_tr.find("368.5 V") != std::string::npos);
  REQUIRE(brief_tr.find("44.2") == std::string::npos);
  REQUIRE(vertirta::copilot::verify_numeric_grounding(brief_tr, st));

  const std::string brief_en =
      vertirta::copilot::render_bilingual_response(st, ast, nullptr, LangCode::EN);
  REQUIRE(brief_en.find("368.5 V") != std::string::npos);
  REQUIRE(vertirta::copilot::verify_numeric_grounding(brief_en, st));
}
