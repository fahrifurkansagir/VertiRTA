#include "vertirta/copilot/rta_state_contract.hpp"

#include <algorithm>

namespace vertirta::copilot {

namespace {

[[nodiscard]] uint8_t push_node(BoundedStlFormula& f, const StlAstNode& node) noexcept {
  if (f.node_count >= kMaxStlNodes) {
    return 0xFFU;
  }
  const uint8_t idx = f.node_count++;
  f.nodes[idx] = node;
  return idx;
}

[[nodiscard]] uint8_t make_and(BoundedStlFormula& f, uint8_t a, uint8_t b) noexcept {
  StlAstNode n{};
  n.kind = StlNodeKind::OP_AND;
  n.left = a;
  n.right = b;
  return push_node(f, n);
}

[[nodiscard]] uint8_t make_always(BoundedStlFormula& f, uint32_t ta, uint32_t tb,
                                  uint8_t child) noexcept {
  StlAstNode n{};
  n.kind = StlNodeKind::OP_ALWAYS;
  n.t_a = ta;
  n.t_b = tb;
  n.left = child;
  return push_node(f, n);
}

[[nodiscard]] uint8_t make_until(BoundedStlFormula& f, uint32_t ta, uint32_t tb,
                                 uint8_t left_child, uint8_t right_child) noexcept {
  StlAstNode n{};
  n.kind = StlNodeKind::OP_UNTIL;
  n.t_a = ta;
  n.t_b = tb;
  n.left = left_child;
  n.right = right_child;
  return push_node(f, n);
}

[[nodiscard]] uint8_t make_pred(BoundedStlFormula& f, StlNodeKind k, float param,
                                ZoneId z = ZoneId::LTFJ_CTR) noexcept {
  StlAstNode n{};
  n.kind = k;
  n.param = param;
  n.zone = z;
  return push_node(f, n);
}

}  // namespace

BoundedStlFormula synthesize_canonical_stl(const CompactDslAst& ast,
                                           const RtaStateSnapshotV1& st) noexcept {
  BoundedStlFormula f{};
  if (!ast.valid) {
    return f;
  }

  const float agl_min = static_cast<float>(std::clamp(ast.agl_min_m, 30, 150));
  const float spd_max = static_cast<float>(std::clamp(ast.speed_max_mps, 5, 22));
  const float soc_floor = st.soc_reserve_floor_pct;

  // Resolve target pad coordinates if present in snapshot
  std::string_view tgt_name = ast.target_id();
  if (ast.action == ActionKind::REJECT_UNSAFE && ast.fallback_target() != "HOLD_LOITER") {
    tgt_name = ast.fallback_target();
  }
  const SlzCandidate* slz = tgt_name.empty() ? nullptr : st.find_slz(tgt_name);
  if (slz != nullptr) {
    f.target_enu = slz->enu;
  } else {
    f.target_enu = st.pos_enu;
  }

  const bool is_diversion = (ast.action == ActionKind::DIVERT_SLZ ||
                             ast.action == ActionKind::INSPECT_SLZ ||
                             ast.action == ActionKind::ABORT_RETURN);

  if (is_diversion) {
    const uint32_t eta = static_cast<uint32_t>(std::clamp(ast.eta_lim_s, 10, 600));
    const uint32_t t_cruise = (eta > 22U) ? (eta - 22U) : ((eta * 7U) / 10U);
    f.horizon_s = eta;
    f.until_target_radius_m = kDefaultReachRadiusM;

    // Clause 1: ALWAYS[0, t_cruise](agl_ge & outside(LTFJ) & outside(LTBX) & speed_le)
    const uint8_t p_agl = make_pred(f, StlNodeKind::PRED_AGL_GE, agl_min);
    const uint8_t p_ltfj = make_pred(f, StlNodeKind::PRED_OUTSIDE_ZONE, 0.0f, ZoneId::LTFJ_CTR);
    const uint8_t p_ltbx = make_pred(f, StlNodeKind::PRED_OUTSIDE_ZONE, 0.0f, ZoneId::LTBX_CTR);
    const uint8_t p_spd = make_pred(f, StlNodeKind::PRED_SPEED_LE, spd_max);
    f.includes_ltfj_exclusion = true;
    f.includes_ltbx_exclusion = true;

    const uint8_t c1_a = make_and(f, p_agl, p_ltfj);
    const uint8_t c1_b = make_and(f, p_ltbx, p_spd);
    const uint8_t c1_body = make_and(f, c1_a, c1_b);
    const uint8_t clause1 = make_always(f, 0U, t_cruise, c1_body);

    // Clause 2: ALWAYS[0, eta](soc_ge(reserve))
    const uint8_t p_soc = make_pred(f, StlNodeKind::PRED_SOC_GE, soc_floor);
    const uint8_t clause2 = make_always(f, 0U, eta, p_soc);

    // Clause 3: UNTIL[0, eta](agl_ge, reach(target, 15))
    const uint8_t p_agl_u = make_pred(f, StlNodeKind::PRED_AGL_GE, agl_min);
    const uint8_t p_reach = make_pred(f, StlNodeKind::PRED_REACH_TARGET, kDefaultReachRadiusM);
    const uint8_t clause3 = make_until(f, 0U, eta, p_agl_u, p_reach);

    const uint8_t top12 = make_and(f, clause1, clause2);
    f.root_index = make_and(f, top12, clause3);
    // Total nodes = 15 (strictly <= 32)
    return f;
  }

  // HOLD_LOITER, REQUEST_CLARIFICATION, or REJECT_UNSAFE safe-stay invariant
  const uint32_t dur =
      static_cast<uint32_t>(std::clamp((ast.duration_s > 0) ? ast.duration_s : ast.eta_lim_s, 10, 300));
  f.horizon_s = dur;
  f.until_target_radius_m = 0.0f;

  const uint8_t p_agl = make_pred(f, StlNodeKind::PRED_AGL_GE, agl_min);
  const uint8_t p_ltfj = make_pred(f, StlNodeKind::PRED_OUTSIDE_ZONE, 0.0f, ZoneId::LTFJ_CTR);
  const uint8_t p_ltbx = make_pred(f, StlNodeKind::PRED_OUTSIDE_ZONE, 0.0f, ZoneId::LTBX_CTR);
  const uint8_t p_soc = make_pred(f, StlNodeKind::PRED_SOC_GE, soc_floor);
  f.includes_ltfj_exclusion = true;
  f.includes_ltbx_exclusion = true;

  const uint8_t a1 = make_and(f, p_agl, p_ltfj);
  const uint8_t a2 = make_and(f, p_ltbx, p_soc);
  const uint8_t body = make_and(f, a1, a2);
  f.root_index = make_always(f, 0U, dur, body);
  // Total nodes = 8 (strictly <= 32)
  return f;
}

}  // namespace vertirta::copilot
