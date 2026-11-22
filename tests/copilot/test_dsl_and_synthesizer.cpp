#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "vertirta/copilot/rta_state_contract.hpp"

using vertirta::copilot::ActionKind;
using vertirta::copilot::ClarifyReason;
using vertirta::copilot::CompactDslAst;
using vertirta::copilot::LangCode;
using vertirta::copilot::ModeCode;
using vertirta::copilot::RtaStateSnapshotV1;
using vertirta::copilot::ViolationCode;

TEST_CASE("RtaStateSnapshotV1 locks 400V/22kWh HV pack and kMaxDynamicNfz=4") {
  const RtaStateSnapshotV1 st = vertirta::copilot::make_reference_400v_sml_snapshot();
  REQUIRE(st.pack_nominal_volts == Catch::Approx(400.0f));
  REQUIRE(st.pack_energy_kwh == Catch::Approx(22.0f));
  REQUIRE(st.v_term_volts == Catch::Approx(368.5f));
  REQUIRE(st.v_term_volts > 320.0f);
  REQUIRE(st.soc_reserve_floor_pct == Catch::Approx(12.0f));
  REQUIRE(st.dyn_nfzs.size() == 4U);
  REQUIRE(st.shield_active);
  REQUIRE(st.num_slz == 4U);
  REQUIRE(st.find_slz("Kartal_Sahil") != nullptr);
  REQUIRE(st.find_slz("Cevizli_Helipad") != nullptr);
}

TEST_CASE("populate_slzs_from_provider maps ISlzPriorProvider snapshot cleanly without heap") {
  vertirta::rta::SlzProviderSnapshot prov{};
  prov.count = 2U;
  std::strncpy(prov.priors[0].id, "Kartal_Sahil", sizeof(prov.priors[0].id) - 1U);
  prov.priors[0].lat_deg = 40.8860;
  prov.priors[0].lon_deg = 29.1880;
  prov.priors[0].p_safe = 0.94f;
  prov.priors[0].eta_s = 100.0f;
  prov.priors[0].rank = 1U;

  std::strncpy(prov.priors[1].id, "Pendik_Trap", sizeof(prov.priors[1].id) - 1U);
  prov.priors[1].lon_deg = 29.2397;
  prov.priors[1].rank = 2U;

  RtaStateSnapshotV1 st{};
  vertirta::copilot::populate_slzs_from_provider(prov, st);
  REQUIRE(st.num_slz == 2U);
  REQUIRE(st.ranked_slzs[0].id_view() == "Kartal_Sahil");
  REQUIRE(st.ranked_slzs[0].west_of_29200e);
  REQUIRE_FALSE(st.ranked_slzs[1].west_of_29200e);
}

TEST_CASE("Compact DSL parses all 6 actions and rejects <think>, CRLF, trailing garbage, and invalid SLZs") {
  CompactDslAst ast{};

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", ast));
  REQUIRE(ast.action == ActionKind::DIVERT_SLZ);
  REQUIRE(ast.target_id() == "Kartal_Sahil");
  REQUIRE(ast.eta_lim_s == 100);
  REQUIRE(ast.agl_min_m == 35);
  REQUIRE(ast.agl_max_m == 120);
  REQUIRE(ast.speed_max_mps == 18);
  REQUIRE(ast.lang == LangCode::TR);
  REQUIRE(ast.mode == ModeCode::BRIEF);

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=INSPECT_SLZ;TGT=Cevizli_Helipad;ETA=80;AGL=30,90;SPD=15\nLNG=MIX;MOD=BRIEF", ast));
  REQUIRE(ast.action == ActionKind::INSPECT_SLZ);
  REQUIRE(ast.target_id() == "Cevizli_Helipad");

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=ABORT_RETURN;TGT=Dragos_Plaza;ETA=210;AGL=40,120;SPD=16\nLNG=EN;MOD=BRIEF", ast));
  REQUIRE(ast.action == ActionKind::ABORT_RETURN);

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF", ast));
  REQUIRE(ast.action == ActionKind::HOLD_LOITER);
  REQUIRE(ast.duration_s == 30);
  REQUIRE(ast.agl_min_m == 60);

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=REQUEST_CLARIFICATION;A=Kartal_Sahil;B=Cevizli_Helipad;RSN=AMBIG\nLNG=TR;MOD=CLARIFY",
      ast));
  REQUIRE(ast.action == ActionKind::REQUEST_CLARIFICATION);
  REQUIRE(ast.opt_a() == "Kartal_Sahil");
  REQUIRE(ast.opt_b() == "Cevizli_Helipad");
  REQUIRE(ast.clarify_reason == ClarifyReason::AMBIG);

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=REJECT_UNSAFE;VIO=LTFJ_CTR;FB=Kartal_Sahil\nLNG=TR;MOD=REJECT", ast));
  REQUIRE(ast.action == ActionKind::REJECT_UNSAFE);
  REQUIRE(ast.violation_code == ViolationCode::LTFJ_CTR);
  REQUIRE(ast.fallback_target() == "Kartal_Sahil");

  // Hard rejections
  REQUIRE_FALSE(vertirta::copilot::parse_compact_dsl(
      "<think>x</think>\nACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF", ast));
  REQUIRE_FALSE(vertirta::copilot::parse_compact_dsl(
      "ACT=DIVERT_SLZ;TGT=Pendik_Marina;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", ast));
  REQUIRE_FALSE(vertirta::copilot::parse_compact_dsl(
      "ACT=HOLD_LOITER;DUR=30;AGL=60\r\nLNG=TR;MOD=BRIEF", ast));
  REQUIRE_FALSE(vertirta::copilot::parse_compact_dsl(
      "ACT=HOLD_LOITER;DUR=30;AGL=60;EXTRA=1\nLNG=TR;MOD=BRIEF", ast));
}

TEST_CASE("synthesize_canonical_stl deterministically emits CTR exclusions and <=32 AST nodes") {
  const auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  CompactDslAst ast{};
  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=100;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", ast));

  const auto stl = vertirta::copilot::synthesize_canonical_stl(ast, st);
  REQUIRE(stl.node_count > 0U);
  REQUIRE(stl.node_count <= 32U);
  REQUIRE(stl.root_index < stl.node_count);
  REQUIRE(stl.includes_ltfj_exclusion);
  REQUIRE(stl.includes_ltbx_exclusion);
  REQUIRE(stl.until_target_radius_m == Catch::Approx(15.0f));
  REQUIRE(stl.horizon_s == 100U);

  REQUIRE(vertirta::copilot::parse_compact_dsl(
      "ACT=HOLD_LOITER;DUR=45;AGL=60\nLNG=TR;MOD=BRIEF", ast));
  const auto loiter_stl = vertirta::copilot::synthesize_canonical_stl(ast, st);
  REQUIRE(loiter_stl.node_count > 0U);
  REQUIRE(loiter_stl.node_count <= 32U);
  REQUIRE(loiter_stl.includes_ltfj_exclusion);
  REQUIRE(loiter_stl.includes_ltbx_exclusion);
  REQUIRE(loiter_stl.horizon_s == 45U);
}

TEST_CASE("GBNF grammar file exists on disk and enforces compact ACT= syntax without verbose STL=") {
  const std::string path = std::string(VERTIRTA_SOURCE_DIR) + "/config/vertirta_dsl_v1.gbnf";
  std::ifstream ifs(path);
  REQUIRE(ifs.is_open());
  std::ostringstream oss;
  oss << ifs.rdbuf();
  const std::string gbnf = oss.str();
  REQUIRE(gbnf.find("\"ACT=\"") != std::string::npos);
  REQUIRE(gbnf.find("STL=") == std::string::npos);
  REQUIRE(gbnf.find('\r') == std::string::npos);
}
