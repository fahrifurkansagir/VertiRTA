#include "vertirta/stl/stl_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vertirta::stl {

void StlMonitorEngine::set_zone_halfspaces(copilot::ZoneId zone,
                                           std::span<const HalfSpace2D> facets) noexcept {
  const std::size_t z_idx = static_cast<std::size_t>(zone);
  if (z_idx >= kMaxZones) {
    return;
  }
  auto& z = zones_[z_idx];
  const std::size_t n = (facets.size() <= kMaxPolyFacets) ? facets.size() : kMaxPolyFacets;
  z.count = static_cast<uint8_t>(n);
  z.active = (n > 0U);
  for (std::size_t i = 0U; i < n; ++i) {
    z.facets[i] = facets[i];
  }
}

void StlMonitorEngine::configure(const copilot::BoundedStlFormula& formula) noexcept {
  formula_ = formula;
  configured_ = (formula_.node_count > 0U && formula_.root_index < formula_.node_count &&
                 formula_.node_count <= copilot::kMaxStlNodes);

  for (auto& w : always_wedges_) {
    w.reset();
  }
  for (auto& w : eventually_wedges_) {
    w.reset();
  }
  tracks_.fill(TemporalTrack{});

  uint8_t next_always = 0U;
  uint8_t next_eventually = 0U;
  uint8_t next_until = 0U;

  for (uint8_t i = 0U; i < formula_.node_count; ++i) {
    const auto& node = formula_.nodes[i];
    if (node.kind == copilot::StlNodeKind::OP_ALWAYS) {
      tracks_[i].mission_acc = +std::numeric_limits<float>::infinity();
      if (next_always < kMaxAlwaysSlots) {
        tracks_[i].slot = next_always++;
      }
    } else if (node.kind == copilot::StlNodeKind::OP_EVENTUALLY) {
      tracks_[i].mission_acc = -std::numeric_limits<float>::infinity();
      if (next_eventually < kMaxEventuallySlots) {
        tracks_[i].slot = next_eventually++;
      }
    } else if (node.kind == copilot::StlNodeKind::OP_UNTIL) {
      if (next_until < kMaxUntilSlots) {
        tracks_[i].slot = next_until;
        until_ops_[next_until].configure(node.t_a, node.t_b);
        ++next_until;
      }
    }
  }
}

float StlMonitorEngine::eval_outside_zone(copilot::ZoneId zone, double east_m,
                                          double north_m) const noexcept {
  const std::size_t z_idx = static_cast<std::size_t>(zone);
  if (z_idx >= kMaxZones || !zones_[z_idx].active) {
    return +10000.0f;  // Inactive zone imposes no barrier
  }
  if (!std::isfinite(east_m) || !std::isfinite(north_m)) {
    return -std::numeric_limits<float>::infinity();
  }
  const auto& z = zones_[z_idx];
  double max_margin = -std::numeric_limits<double>::infinity();
  for (uint8_t i = 0U; i < z.count; ++i) {
    const double m = z.facets[i].ax * east_m + z.facets[i].ay * north_m - z.facets[i].b;
    if (m > max_margin) {
      max_margin = m;
    }
  }
  return static_cast<float>(max_margin);
}

StlTickOutput StlMonitorEngine::step(uint32_t tau, const StlSignalSample& s) noexcept {
  StlTickOutput out{};
  if (!configured_) {
    out.violated = true;
    return out;
  }

  std::array<float, copilot::kMaxStlNodes> cur_val{};
  std::array<float, copilot::kMaxStlNodes> mission_val{};
  std::array<bool, copilot::kMaxStlNodes> node_locked{};
  std::array<RosiBound, copilot::kMaxStlNodes> rosi_arr{};

  const float neg_inf = -std::numeric_limits<float>::infinity();
  const float pos_inf = +std::numeric_limits<float>::infinity();

  for (uint8_t i = 0U; i < formula_.node_count; ++i) {
    const auto& n = formula_.nodes[i];
    switch (n.kind) {
      case copilot::StlNodeKind::PRED_AGL_GE: {
        const float v = std::isfinite(s.agl_m) ? (s.agl_m - n.param) : neg_inf;
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::PRED_AGL_LE: {
        const float v = std::isfinite(s.agl_m) ? (n.param - s.agl_m) : neg_inf;
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::PRED_OUTSIDE_ZONE: {
        const float v = eval_outside_zone(n.zone, s.east_m, s.north_m);
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::PRED_SOC_GE: {
        const float v = std::isfinite(s.soc_pct) ? (s.soc_pct - n.param) : neg_inf;
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::PRED_REACH_TARGET: {
        float v = neg_inf;
        if (std::isfinite(s.east_m) && std::isfinite(s.north_m)) {
          const double de = s.east_m - formula_.target_enu.east_m;
          const double dn = s.north_m - formula_.target_enu.north_m;
          v = n.param - static_cast<float>(std::hypot(de, dn));
        }
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::PRED_SPEED_LE: {
        const float v = std::isfinite(s.groundspeed_mps) ? (n.param - s.groundspeed_mps) : neg_inf;
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::PRED_OOD_LE: {
        const float v = std::isfinite(s.ood_var) ? (n.param - s.ood_var) : neg_inf;
        cur_val[i] = v;
        mission_val[i] = v;
        node_locked[i] = true;
        rosi_arr[i] = RosiBound{v, v};
        break;
      }
      case copilot::StlNodeKind::OP_NOT: {
        const uint8_t c = n.left;
        cur_val[i] = -cur_val[c];
        mission_val[i] = -mission_val[c];
        node_locked[i] = node_locked[c];
        rosi_arr[i] = RosiBound{-rosi_arr[c].sup, -rosi_arr[c].inf};
        break;
      }
      case copilot::StlNodeKind::OP_AND: {
        const uint8_t l = n.left;
        const uint8_t r = n.right;
        cur_val[i] = std::min(cur_val[l], cur_val[r]);
        mission_val[i] = std::min(mission_val[l], mission_val[r]);
        node_locked[i] = node_locked[l] && node_locked[r];
        rosi_arr[i] = RosiBound{std::min(rosi_arr[l].inf, rosi_arr[r].inf),
                                std::min(rosi_arr[l].sup, rosi_arr[r].sup)};
        break;
      }
      case copilot::StlNodeKind::OP_OR: {
        const uint8_t l = n.left;
        const uint8_t r = n.right;
        cur_val[i] = std::max(cur_val[l], cur_val[r]);
        mission_val[i] = std::max(mission_val[l], mission_val[r]);
        node_locked[i] = node_locked[l] && node_locked[r];
        rosi_arr[i] = RosiBound{std::max(rosi_arr[l].inf, rosi_arr[r].inf),
                                std::max(rosi_arr[l].sup, rosi_arr[r].sup)};
        break;
      }
      case copilot::StlNodeKind::OP_ALWAYS: {
        const float cv = cur_val[n.left];
        auto& tr = tracks_[i];
        const uint32_t span_ba = (n.t_b >= n.t_a) ? (n.t_b - n.t_a) : 0U;
        if (tr.slot < kMaxAlwaysSlots) {
          cur_val[i] = always_wedges_[tr.slot].push_and_query(tau, cv, span_ba);
        } else {
          cur_val[i] = cv;
        }
        if (tau >= n.t_a && tau <= n.t_b) {
          tr.mission_acc = std::min(tr.mission_acc, cv);
          tr.started = true;
        }
        if (tau >= n.t_b) {
          tr.locked = true;
        }
        mission_val[i] = tr.started ? tr.mission_acc : neg_inf;
        node_locked[i] = tr.locked;
        rosi_arr[i].inf = tr.locked ? tr.mission_acc : neg_inf;
        rosi_arr[i].sup = tr.started ? tr.mission_acc : pos_inf;
        break;
      }
      case copilot::StlNodeKind::OP_EVENTUALLY: {
        const float cv = cur_val[n.left];
        auto& tr = tracks_[i];
        const uint32_t span_ba = (n.t_b >= n.t_a) ? (n.t_b - n.t_a) : 0U;
        if (tr.slot < kMaxEventuallySlots) {
          cur_val[i] = eventually_wedges_[tr.slot].push_and_query(tau, cv, span_ba);
        } else {
          cur_val[i] = cv;
        }
        if (tau >= n.t_a && tau <= n.t_b) {
          tr.mission_acc = std::max(tr.mission_acc, cv);
          tr.started = true;
        }
        if (tau >= n.t_b) {
          tr.locked = true;
        }
        mission_val[i] = tr.started ? tr.mission_acc : neg_inf;
        node_locked[i] = tr.locked;
        rosi_arr[i].inf = tr.started ? tr.mission_acc : neg_inf;
        rosi_arr[i].sup = tr.locked ? tr.mission_acc : pos_inf;
        break;
      }
      case copilot::StlNodeKind::OP_UNTIL: {
        const float r1 = cur_val[n.left];
        const float r2 = cur_val[n.right];
        auto& tr = tracks_[i];
        float exact_u = neg_inf;
        RosiBound u_rosi{};
        if (tr.slot < kMaxUntilSlots) {
          (void)until_ops_[tr.slot].step(tau, r1, r2, exact_u, u_rosi);
          tr.locked = until_ops_[tr.slot].mission_locked();
          tr.mission_acc = until_ops_[tr.slot].mission_rho();
        }
        cur_val[i] = exact_u;
        mission_val[i] = tr.locked ? tr.mission_acc : u_rosi.inf;
        node_locked[i] = tr.locked;
        rosi_arr[i] = u_rosi;
        break;
      }
      case copilot::StlNodeKind::EMPTY:
      default:
        cur_val[i] = neg_inf;
        mission_val[i] = neg_inf;
        node_locked[i] = false;
        rosi_arr[i] = RosiBound{neg_inf, neg_inf};
        break;
    }
  }

  const uint8_t root = formula_.root_index;
  out.rosi = rosi_arr[root];
  out.streaming_rho = cur_val[root];
  out.mission_complete = node_locked[root];
  out.mission_rho = node_locked[root] ? mission_val[root] : out.rosi.inf;
  out.violated = out.rosi.definitively_violated();
  return out;
}

}  // namespace vertirta::stl
