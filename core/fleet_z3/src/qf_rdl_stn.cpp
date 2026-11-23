// SPDX-License-Identifier: Apache-2.0
#include "vertirta/fleet_z3/fleet_z3_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef VERTIRTA_HAS_Z3
#include <z3++.h>
#endif

namespace vertirta::fleet_z3 {

namespace {

constexpr std::int64_t kScale = 10000LL;
constexpr std::int64_t kInf = std::numeric_limits<std::int64_t>::max() / 4LL;
// Deterministic rlimit (250000) bounds solver work; 100 ms wall timeout absorbs
// arm64/sanitizer cloud VM steal (mirrors core/copilot/z3_corridor_verifier.cpp).
constexpr std::uint32_t kTier1Z3TimeoutMs = 100U;

[[nodiscard]] std::int64_t scale_1e4(double v) noexcept {
  if (!std::isfinite(v)) {
    return 0LL;
  }
  return static_cast<std::int64_t>(std::llround(v * static_cast<double>(kScale)));
}

[[nodiscard]] double unscale_1e4(std::int64_t q) noexcept {
  return static_cast<double>(q) / static_cast<double>(kScale);
}

[[nodiscard]] std::string smt_real_q(std::int64_t num) {
  if (num == 0LL) {
    return "0.0";
  }
  if (num < 0LL) {
    return "(- (/ " + std::to_string(-num) + " 10000))";
  }
  return "(/ " + std::to_string(num) + " 10000)";
}

[[nodiscard]] std::string sanitize_ident(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 4U);
  for (char c : raw) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  if (out.empty() || (out[0] >= '0' && out[0] <= '9')) {
    out.insert(out.begin(), 'v');
  }
  return out;
}

// Directed difference constraint: x[v] - x[u] <= w_q / 10000.
struct DiffEdge {
  std::size_t u{0U};
  std::size_t v{0U};
  std::int64_t w_q{0LL};
  std::string label{};
};

}  // namespace

struct QfRdlStnSolver::Impl {
#ifdef VERTIRTA_HAS_Z3
  struct RdlConfigHolder {
    z3::config cfg;
    RdlConfigHolder() { cfg.set("auto_config", false); }
  };

  // Strictly isolated Tier-1 context (never shared with Tier-2 QfLraLocalSolver).
  // Uses exact 1e-4 s integer-tick QF_IDL isomorphism (totally unimodular difference matrix:
  // x_v - x_u <= w_q/10000 over R iff X_v - X_u <= w_q over Z), achieving ~1 ms on Z3 4.8.12.
  RdlConfigHolder cfg_holder;
  z3::context ctx_rdl;
  z3::solver solver_rdl;
  std::vector<z3::expr> cached_nodes;

  Impl() : cfg_holder(), ctx_rdl(cfg_holder.cfg), solver_rdl(ctx_rdl, "QF_IDL") {
    constexpr std::size_t kMaxNodes = 1U + 2U * kMaxFleetAircraft;
    cached_nodes.reserve(kMaxNodes);
    for (std::size_t k = 0U; k < kMaxNodes; ++k) {
      const std::string nm = "n" + std::to_string(k);
      cached_nodes.push_back(ctx_rdl.int_const(nm.c_str()));
    }
    z3::params fast_params(ctx_rdl);
    fast_params.set("unsat_core", false);
    fast_params.set("rlimit", 250000U);
    fast_params.set("timeout", static_cast<unsigned>(kTier1Z3TimeoutMs));
    solver_rdl.set(fast_params);

    // Pre-warm Z3 allocator & QF_IDL difference-logic graph engine with a realistic
    // multi-edge difference chain so tick #1 has zero cold initialization spike.
    solver_rdl.push();
    solver_rdl.add((cached_nodes[0U] - cached_nodes[1U]) <=
                   ctx_rdl.int_val(static_cast<std::int64_t>(0)));
    solver_rdl.add((cached_nodes[1U] - cached_nodes[2U]) <=
                   ctx_rdl.int_val(static_cast<std::int64_t>(-1800000)));
    solver_rdl.add((cached_nodes[2U] - cached_nodes[0U]) <=
                   ctx_rdl.int_val(static_cast<std::int64_t>(9000000)));
    (void)solver_rdl.check();
    solver_rdl.pop();
  }
#else
  std::uint64_t fallback_tag{0x52444C5F53544E31ULL};  // "RDL_STN1"
#endif
};

QfRdlStnSolver::QfRdlStnSolver() : impl_(std::make_unique<Impl>()) {}
QfRdlStnSolver::~QfRdlStnSolver() = default;
QfRdlStnSolver::QfRdlStnSolver(QfRdlStnSolver&&) noexcept = default;
QfRdlStnSolver& QfRdlStnSolver::operator=(QfRdlStnSolver&&) noexcept = default;

std::uintptr_t QfRdlStnSolver::context_tag() const noexcept {
  if (!impl_) {
    return 0U;
  }
#ifdef VERTIRTA_HAS_Z3
  return reinterpret_cast<std::uintptr_t>(&impl_->ctx_rdl);
#else
  return reinterpret_cast<std::uintptr_t>(&impl_->fallback_tag);
#endif
}

SolverBackend QfRdlStnSolver::backend() const noexcept {
#ifdef VERTIRTA_HAS_Z3
  return SolverBackend::NATIVE_Z3_DUAL_CONTEXT;
#else
  return SolverBackend::DETERMINISTIC_EXACT_FALLBACK;
#endif
}

FleetScheduleResult QfRdlStnSolver::solve(const FleetScheduleRequest& req) const {
  const auto t0 = std::chrono::steady_clock::now();
  FleetScheduleResult res{};
  res.tier_used = SolverTier::TIER1_QF_RDL_STN;
  res.backend_used = backend();
  res.smt2_logic = "QF_RDL";

  const std::size_t n = std::min(req.aircraft.size(), kMaxFleetAircraft);
  const std::size_t num_nodes = 1U + 2U * n;  // Node 0 = t_zero; 2*i+1 = dep_i; 2*i+2 = arr_i

  std::vector<std::string> var_names(num_nodes);
  var_names[0U] = "t_zero";
  std::vector<std::string> safe_ids(n);
  for (std::size_t i = 0U; i < n; ++i) {
    safe_ids[i] = sanitize_ident(req.aircraft[i].aircraft_id.empty()
                                     ? ("UA_" + std::to_string(i))
                                     : req.aircraft[i].aircraft_id);
    var_names[1U + 2U * i] = "dep_" + safe_ids[i];
    var_names[2U + 2U * i] = "arr_" + safe_ids[i];
  }

  std::vector<DiffEdge> edges;
  edges.reserve(6U * n + 16U);

  double max_wake_enforced = 0.0;
  bool any_dep_fato_mutex = false;

  for (std::size_t i = 0U; i < n; ++i) {
    const auto& a = req.aircraft[i];
    const std::size_t dep_node = 1U + 2U * i;
    const std::size_t arr_node = 2U + 2U * i;

    // dep_i >= earliest_dep_s  <=>  t_zero - dep_i <= -earliest_dep_s
    edges.push_back(DiffEdge{dep_node, 0U, -scale_1e4(a.earliest_dep_s),
                             "c_dep_lb_" + safe_ids[i]});
    // arr_i - dep_i >= min_transit_s  <=>  dep_i - arr_i <= -min_transit_s
    edges.push_back(DiffEdge{arr_node, dep_node, -scale_1e4(a.min_transit_s),
                             "c_transit_min_" + safe_ids[i]});
    // arr_i - dep_i <= max_transit_s
    edges.push_back(DiffEdge{dep_node, arr_node, scale_1e4(a.max_transit_s),
                             "c_transit_max_" + safe_ids[i]});
    // arr_i <= battery_max_arr_s  <=>  arr_i - t_zero <= battery_max_arr_s
    edges.push_back(DiffEdge{0U, arr_node, scale_1e4(a.battery_max_arr_s),
                             "c_battery_" + safe_ids[i]});

    if (a.crosses_ltfj_wake_zone) {
      const double wake_s = effective_ltfj_wake_s(a, req.ltfj_heavy_wake_s);
      max_wake_enforced = std::max(max_wake_enforced, wake_s);
      const double wake_clear_s = a.ltfj_heavy_arrival_s + wake_s;
      // arr_i >= wake_clear_s  <=>  t_zero - arr_i <= -wake_clear_s
      edges.push_back(DiffEdge{arr_node, 0U, -scale_1e4(wake_clear_s),
                               "c_ltfj_wake_" + safe_ids[i]});
    }
  }

  // Fixed-order minimal precedence edges for aircraft sharing a FATO or < 60 m dependent group.
  // Connecting each aircraft to its immediate next mutex successor is exact by transitivity
  // (every sep_s >= 0) and eliminates O(N^2) redundant edges -> 0 Booleans in pure QF_RDL STN.
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
    const double la = req.aircraft[lhs].nominal_dep_s + req.aircraft[lhs].min_transit_s;
    const double ra = req.aircraft[rhs].nominal_dep_s + req.aircraft[rhs].min_transit_s;
    if (la != ra) {
      return la < ra;
    }
    return req.aircraft[lhs].aircraft_id < req.aircraft[rhs].aircraft_id;
  });

  for (std::size_t oi = 0U; oi < n; ++oi) {
    const std::size_t i = order[oi];
    const auto& ai = req.aircraft[i];
    for (std::size_t oj = oi + 1U; oj < n; ++oj) {
      const std::size_t j = order[oj];
      const auto& aj = req.aircraft[j];
      if (!shares_fato_mutex(ai, aj, req.dependent_fato_spacing_m)) {
        continue;
      }
      const bool dep_group = (ai.fato_id != aj.fato_id);
      if (dep_group) {
        any_dep_fato_mutex = true;
      }
      const double sep_s = std::max(ai.fato_dwell_s, req.siso_wake_spacing_s);
      const std::size_t arr_i = 2U + 2U * i;
      const std::size_t arr_j = 2U + 2U * j;
      // arr_j - arr_i >= sep_s  <=>  arr_i - arr_j <= -sep_s
      const std::string prefix = dep_group ? "c_dependent_fato_sep_" : "c_fato_sep_";
      edges.push_back(DiffEdge{arr_j, arr_i, -scale_1e4(sep_s),
                               prefix + safe_ids[i] + "_" + safe_ids[j]});
      break;  // Immediate mutex successor enforces full transitive chain for this FATO/group
    }
  }

  // Exact integer Bellman-Ford on reversed graph (y[k] = -x[k]) computing earliest feasible times
  // and irreducible negative-cycle UNSAT core in O(V * E).
  std::vector<std::int64_t> y_dist(num_nodes, kInf);
  std::vector<std::size_t> parent_node(num_nodes, num_nodes);
  std::vector<std::size_t> parent_edge(num_nodes, edges.size());
  y_dist[0U] = 0LL;

  std::size_t cycle_seed = num_nodes;
  for (std::size_t iter = 0U; iter < num_nodes; ++iter) {
    bool changed = false;
    cycle_seed = num_nodes;
    for (std::size_t ei = 0U; ei < edges.size(); ++ei) {
      const auto& e = edges[ei];
      // Constraint x[v] - x[u] <= w  <=>  y[u] <= y[v] + w  (reversed edge v -> u).
      if (y_dist[e.v] < kInf && y_dist[e.v] + e.w_q < y_dist[e.u]) {
        y_dist[e.u] = y_dist[e.v] + e.w_q;
        parent_node[e.u] = e.v;
        parent_edge[e.u] = ei;
        changed = true;
        cycle_seed = e.u;
      }
    }
    if (!changed) {
      break;
    }
  }

  bool bf_sat = (cycle_seed == num_nodes);
  std::vector<std::string> bf_core;
  if (!bf_sat) {
    std::size_t cur = cycle_seed;
    for (std::size_t step = 0U; step < num_nodes; ++step) {
      if (parent_node[cur] < num_nodes) {
        cur = parent_node[cur];
      }
    }
    const std::size_t start = cur;
    for (std::size_t guard = 0U; guard <= num_nodes; ++guard) {
      const std::size_t pe = parent_edge[cur];
      if (pe < edges.size()) {
        bf_core.push_back(edges[pe].label);
      }
      cur = parent_node[cur];
      if (cur == start || cur >= num_nodes) {
        break;
      }
    }
    std::sort(bf_core.begin(), bf_core.end());
    bf_core.erase(std::unique(bf_core.begin(), bf_core.end()), bf_core.end());
  }

#ifdef VERTIRTA_HAS_Z3
  {
    z3::context& ctx = impl_->ctx_rdl;
    z3::solver& solver = impl_->solver_rdl;
    const auto& znodes = impl_->cached_nodes;

    // Fast path: 0-Boolean unit integer-difference inequalities with incremental push()/pop()
    // (~1 ms on Ubuntu 24.04 libz3 4.8.12; exact 1e-4 s isomorphism to QF_RDL).
    solver.push();
    for (const auto& e : edges) {
      solver.add((znodes[e.v] - znodes[e.u]) <= ctx.int_val(static_cast<std::int64_t>(e.w_q)));
    }
    const z3::check_result zres = solver.check();
    solver.pop();

    if (zres == z3::sat) {
      res.sat = true;
    } else if (zres == z3::unsat) {
      // UNSAT slow path: extract tracked minimal unsat_core only when conflict occurs.
      res.sat = false;
      z3::solver core_solver(ctx, "QF_IDL");
      z3::params core_params(ctx);
      core_params.set("unsat_core", true);
      core_params.set("rlimit", 250000U);
      core_params.set("timeout", static_cast<unsigned>(kTier1Z3TimeoutMs));
      core_solver.set(core_params);

      z3::expr_vector assumptions(ctx);
      for (const auto& e : edges) {
        z3::expr b = ctx.bool_const(e.label.c_str());
        core_solver.add(z3::implies(
            b, (znodes[e.v] - znodes[e.u]) <= ctx.int_val(static_cast<std::int64_t>(e.w_q))));
        assumptions.push_back(b);
      }
      if (core_solver.check(assumptions) == z3::unsat) {
        z3::expr_vector core = core_solver.unsat_core();
        for (unsigned ci = 0U; ci < core.size(); ++ci) {
          res.unsat_core_labels.push_back(core[ci].decl().name().str());
        }
      }
      if (res.unsat_core_labels.empty()) {
        res.unsat_core_labels = bf_core;
      }
    } else {
      res.timed_out = true;
      res.sat = bf_sat;
      res.unsat_core_labels = bf_sat ? std::vector<std::string>{} : bf_core;
    }
  }
#else
  res.sat = bf_sat;
  res.unsat_core_labels = bf_core;
#endif

  res.dependent_fato_mutex_enforced = any_dep_fato_mutex;
  res.ltfj_wake_buffer_s = max_wake_enforced;

  if (res.sat) {
    res.schedule.reserve(n);
    for (std::size_t i = 0U; i < n; ++i) {
      const double dep_s = unscale_1e4(-y_dist[1U + 2U * i]);
      const double arr_s = unscale_1e4(-y_dist[2U + 2U * i]);
      const double nom_arr = req.aircraft[i].nominal_dep_s + req.aircraft[i].min_transit_s;
      const double delay_s = std::max(0.0, arr_s - nom_arr);
      res.schedule.push_back(ScheduledAircraftSlot{
          req.aircraft[i].aircraft_id,
          req.aircraft[i].fato_id,
          dep_s,
          arr_s,
          delay_s,
          delay_s > 1e-3,
      });
    }
  }

  if (req.emit_smt2) {
    std::ostringstream smt;
    smt << "(set-logic QF_RDL)\n";
    for (const auto& vn : var_names) {
      smt << "(declare-const " << vn << " Real)\n";
    }
    smt << "(assert (= t_zero 0.0))\n";
    if (res.sat) {
      // Pure 0-Boolean QF_RDL STN fast path
      for (const auto& e : edges) {
        smt << "(assert (<= (- " << var_names[e.v] << ' ' << var_names[e.u] << ") "
            << smt_real_q(e.w_q) << "))\n";
      }
      smt << "(check-sat)\n";
    } else {
      // Tracked QF_RDL assertions for exact UNSAT core cross-verification
      for (const auto& e : edges) {
        smt << "(declare-const " << e.label << " Bool)\n";
        smt << "(assert (=> " << e.label << " (<= (- " << var_names[e.v] << ' '
            << var_names[e.u] << ") " << smt_real_q(e.w_q) << ")))\n";
      }
      smt << "(check-sat-assuming (";
      for (std::size_t ei = 0U; ei < edges.size(); ++ei) {
        if (ei > 0U) {
          smt << ' ';
        }
        smt << edges[ei].label;
      }
      smt << "))\n(get-unsat-core)\n";
    }
    res.smt2_audit = smt.str();
  }

  const auto t1 = std::chrono::steady_clock::now();
  res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return res;
}

}  // namespace vertirta::fleet_z3
