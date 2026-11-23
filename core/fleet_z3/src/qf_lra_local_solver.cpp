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

struct LraEdge {
  std::size_t u{0U};
  std::size_t v{0U};
  std::int64_t w_q{0LL};
  std::string label{};
};

#ifdef VERTIRTA_HAS_Z3
[[nodiscard]] z3::expr z3_qreal(z3::context& c, std::int64_t q) {
  const std::int64_t abs_q = (q < 0LL) ? -q : q;
  const std::int64_t g = std::gcd(abs_q, kScale);
  const std::int64_t num = q / g;
  const std::int64_t den = kScale / g;
  const std::string rat =
      (den == 1LL) ? std::to_string(num) : (std::to_string(num) + "/" + std::to_string(den));
  return c.real_val(rat.c_str());
}
#endif

}  // namespace

struct QfLraLocalSolver::Impl {
#ifdef VERTIRTA_HAS_Z3
  z3::context ctx_lra;
  // Persistent incremental solver on strictly separate Tier-2 context.
  z3::solver solver_lra;

  Impl() : ctx_lra(), solver_lra(ctx_lra) {}
#else
  std::uint64_t fallback_tag{0x4C52415F4C4F4332ULL};  // "LRA_LOC2"
#endif
};

QfLraLocalSolver::QfLraLocalSolver() : impl_(std::make_unique<Impl>()) {}
QfLraLocalSolver::~QfLraLocalSolver() = default;
QfLraLocalSolver::QfLraLocalSolver(QfLraLocalSolver&&) noexcept = default;
QfLraLocalSolver& QfLraLocalSolver::operator=(QfLraLocalSolver&&) noexcept = default;

std::uintptr_t QfLraLocalSolver::context_tag() const noexcept {
  if (!impl_) {
    return 0U;
  }
#ifdef VERTIRTA_HAS_Z3
  return reinterpret_cast<std::uintptr_t>(&impl_->ctx_lra);
#else
  return reinterpret_cast<std::uintptr_t>(&impl_->fallback_tag);
#endif
}

SolverBackend QfLraLocalSolver::backend() const noexcept {
#ifdef VERTIRTA_HAS_Z3
  return SolverBackend::NATIVE_Z3_DUAL_CONTEXT;
#else
  return SolverBackend::DETERMINISTIC_EXACT_FALLBACK;
#endif
}

FleetScheduleResult QfLraLocalSolver::check_unrepaired(
    const FleetScheduleRequest& req,
    const std::vector<std::size_t>& local_indices) const {
  const auto t0 = std::chrono::steady_clock::now();
  FleetScheduleResult res{};
  res.tier_used = SolverTier::TIER2_QF_LRA_LOCAL;
  res.backend_used = backend();
  res.smt2_logic = "QF_LRA";

  std::vector<std::size_t> subset = local_indices;
  if (subset.size() > kMaxLocalNeighborhood) {
    subset.resize(kMaxLocalNeighborhood);
  }
  const std::size_t m = subset.size();
  res.local_neighborhood_size = m;

  std::vector<std::string> safe_ids(m);
  std::vector<std::string> dep_vars(m);
  std::vector<std::string> arr_vars(m);
  std::vector<std::int64_t> nom_arr_q(m);
  std::vector<std::string> pin_labels(m);
  for (std::size_t k = 0U; k < m; ++k) {
    const auto& a = req.aircraft[subset[k]];
    safe_ids[k] = sanitize_ident(a.aircraft_id.empty() ? ("UA_" + std::to_string(subset[k]))
                                                       : a.aircraft_id);
    dep_vars[k] = "dep_" + safe_ids[k];
    arr_vars[k] = "arr_" + safe_ids[k];
    nom_arr_q[k] = scale_1e4(a.nominal_dep_s + a.min_transit_s);
    const bool approved_nom =
        a.is_existing_approved && (a.priority == PriorityClass::APPROVED_NOMINAL);
    pin_labels[k] = approved_nom ? ("c_art10_9_preserve_" + safe_ids[k])
                                 : ("c_req_slot_" + safe_ids[k]);
  }

  std::vector<std::string> trackers;
  std::ostringstream smt;
  smt << "(set-logic QF_LRA)\n";
  for (std::size_t k = 0U; k < m; ++k) {
    smt << "(declare-const " << dep_vars[k] << " Real)\n";
    smt << "(declare-const " << arr_vars[k] << " Real)\n";
  }

  auto emit_tracked = [&](const std::string& label, const std::string& expr) {
    trackers.push_back(label);
    smt << "(declare-const " << label << " Bool)\n";
    smt << "(assert (=> " << label << ' ' << expr << "))\n";
  };

#ifdef VERTIRTA_HAS_Z3
  z3::context& ctx = impl_->ctx_lra;
  z3::solver& solver = impl_->solver_lra;
  solver.reset();
  z3::params zparams(ctx);
  zparams.set("unsat_core", true);
  zparams.set("rlimit", 250000U);
  zparams.set("timeout", static_cast<unsigned>(kTier2TimeoutMs));
  solver.set(zparams);

  std::vector<z3::expr> zdep;
  std::vector<z3::expr> zarr;
  zdep.reserve(m);
  zarr.reserve(m);
  for (std::size_t k = 0U; k < m; ++k) {
    zdep.push_back(ctx.real_const(dep_vars[k].c_str()));
    zarr.push_back(ctx.real_const(arr_vars[k].c_str()));
  }
  z3::expr_vector assumptions(ctx);
  auto add_z3_tracked = [&](const std::string& label, const z3::expr& expr) {
    z3::expr b = ctx.bool_const(label.c_str());
    solver.add(z3::implies(b, expr));
    assumptions.push_back(b);
  };
#endif

  std::vector<std::string> violated_core;
  auto mark_core = [&](const std::string& lbl) { violated_core.push_back(lbl); };

  double max_wake_enforced = 0.0;
  bool any_dep_mutex = false;

  for (std::size_t k = 0U; k < m; ++k) {
    const auto& a = req.aircraft[subset[k]];
    const std::int64_t early_q = scale_1e4(a.earliest_dep_s);
    const std::int64_t min_t_q = scale_1e4(a.min_transit_s);
    const std::int64_t max_t_q = scale_1e4(a.max_transit_s);
    const std::int64_t batt_q = scale_1e4(a.battery_max_arr_s);

    const std::string l_dep = "c_dep_window_" + safe_ids[k];
    const std::string l_tr = "c_transit_" + safe_ids[k];
    const std::string l_bat = "c_battery_" + safe_ids[k];

    emit_tracked(l_dep, "(>= " + dep_vars[k] + " " + smt_real_q(early_q) + ")");
    emit_tracked(l_tr, "(and (>= (- " + arr_vars[k] + " " + dep_vars[k] + ") " +
                           smt_real_q(min_t_q) + ") (<= (- " + arr_vars[k] + " " + dep_vars[k] +
                           ") " + smt_real_q(max_t_q) + "))");
    emit_tracked(l_bat, "(<= " + arr_vars[k] + " " + smt_real_q(batt_q) + ")");
    emit_tracked(pin_labels[k], "(= " + arr_vars[k] + " " + smt_real_q(nom_arr_q[k]) + ")");

#ifdef VERTIRTA_HAS_Z3
    add_z3_tracked(l_dep, zdep[k] >= z3_qreal(ctx, early_q));
    add_z3_tracked(l_tr, ((zarr[k] - zdep[k]) >= z3_qreal(ctx, min_t_q)) &&
                             ((zarr[k] - zdep[k]) <= z3_qreal(ctx, max_t_q)));
    add_z3_tracked(l_bat, zarr[k] <= z3_qreal(ctx, batt_q));
    add_z3_tracked(pin_labels[k], zarr[k] == z3_qreal(ctx, nom_arr_q[k]));
#endif

    // Check pinned slot feasibility
    if (nom_arr_q[k] - min_t_q < early_q || min_t_q > max_t_q) {
      mark_core(l_dep);
      mark_core(l_tr);
      mark_core(pin_labels[k]);
    }
    if (nom_arr_q[k] > batt_q) {
      mark_core(l_bat);
      mark_core(pin_labels[k]);
    }

    if (a.crosses_ltfj_wake_zone) {
      const std::string l_ats = "c_ats_clearance_" + safe_ids[k];
      emit_tracked(l_ats, a.ats_clearance_granted ? "true" : "false");
#ifdef VERTIRTA_HAS_Z3
      add_z3_tracked(l_ats, ctx.bool_val(a.ats_clearance_granted));
#endif
      if (!a.ats_clearance_granted) {
        mark_core(l_ats);
      }

      const double wake_s = effective_ltfj_wake_s(a, req.ltfj_heavy_wake_s);
      max_wake_enforced = std::max(max_wake_enforced, wake_s);
      const std::int64_t clear_q = scale_1e4(a.ltfj_heavy_arrival_s + wake_s);
      const std::string l_wake = "c_ltfj_wake_" + safe_ids[k];
      emit_tracked(l_wake, "(>= " + arr_vars[k] + " " + smt_real_q(clear_q) + ")");
#ifdef VERTIRTA_HAS_Z3
      add_z3_tracked(l_wake, zarr[k] >= z3_qreal(ctx, clear_q));
#endif
      if (nom_arr_q[k] < clear_q) {
        mark_core(l_wake);
        mark_core(pin_labels[k]);
      }
    }
  }

  // Pairwise FATO & < 60 m dependent FATO mutexes with Art. 10(8)/(9)/(10) directional priority.
  for (std::size_t i = 0U; i < m; ++i) {
    for (std::size_t j = i + 1U; j < m; ++j) {
      const auto& ai = req.aircraft[subset[i]];
      const auto& aj = req.aircraft[subset[j]];
      if (!shares_fato_mutex(ai, aj, req.dependent_fato_spacing_m)) {
        continue;
      }
      const bool dep_group = (ai.fato_id != aj.fato_id);
      if (dep_group) {
        any_dep_mutex = true;
      }
      const std::int64_t sep_ij_q = scale_1e4(std::max(ai.fato_dwell_s, req.siso_wake_spacing_s));
      const std::int64_t sep_ji_q = scale_1e4(std::max(aj.fato_dwell_s, req.siso_wake_spacing_s));
      const std::string l_mtx =
          (dep_group ? "c_dependent_fato_mutex_" : "c_fato_mutex_") + safe_ids[i] + "_" +
          safe_ids[j];

      bool pair_ok = true;
      if (ai.priority < aj.priority) {
        emit_tracked(l_mtx, "(>= (- " + arr_vars[j] + " " + arr_vars[i] + ") " +
                                smt_real_q(sep_ij_q) + ")");
#ifdef VERTIRTA_HAS_Z3
        add_z3_tracked(l_mtx, (zarr[j] - zarr[i]) >= z3_qreal(ctx, sep_ij_q));
#endif
        pair_ok = (nom_arr_q[j] - nom_arr_q[i]) >= sep_ij_q;
      } else if (aj.priority < ai.priority) {
        emit_tracked(l_mtx, "(>= (- " + arr_vars[i] + " " + arr_vars[j] + ") " +
                                smt_real_q(sep_ji_q) + ")");
#ifdef VERTIRTA_HAS_Z3
        add_z3_tracked(l_mtx, (zarr[i] - zarr[j]) >= z3_qreal(ctx, sep_ji_q));
#endif
        pair_ok = (nom_arr_q[i] - nom_arr_q[j]) >= sep_ji_q;
      } else {
        emit_tracked(l_mtx, "(or (>= (- " + arr_vars[j] + " " + arr_vars[i] + ") " +
                                smt_real_q(sep_ij_q) + ") (>= (- " + arr_vars[i] + " " +
                                arr_vars[j] + ") " + smt_real_q(sep_ji_q) + "))");
#ifdef VERTIRTA_HAS_Z3
        add_z3_tracked(l_mtx, ((zarr[j] - zarr[i]) >= z3_qreal(ctx, sep_ij_q)) ||
                                  ((zarr[i] - zarr[j]) >= z3_qreal(ctx, sep_ji_q)));
#endif
        pair_ok = ((nom_arr_q[j] - nom_arr_q[i]) >= sep_ij_q) ||
                  ((nom_arr_q[i] - nom_arr_q[j]) >= sep_ji_q);
      }

      if (!pair_ok) {
        mark_core(l_mtx);
        mark_core(pin_labels[i]);
        mark_core(pin_labels[j]);
      }
    }
  }

  smt << "(check-sat-assuming (";
  for (std::size_t ti = 0U; ti < trackers.size(); ++ti) {
    if (ti > 0U) {
      smt << ' ';
    }
    smt << trackers[ti];
  }
  smt << "))\n(get-unsat-core)\n";

  std::sort(violated_core.begin(), violated_core.end());
  violated_core.erase(std::unique(violated_core.begin(), violated_core.end()), violated_core.end());

#ifdef VERTIRTA_HAS_Z3
  const z3::check_result zres = solver.check(assumptions);
  if (zres == z3::sat) {
    res.sat = true;
  } else if (zres == z3::unsat) {
    res.sat = false;
    z3::expr_vector core = solver.unsat_core();
    for (unsigned ci = 0U; ci < core.size(); ++ci) {
      res.unsat_core_labels.push_back(core[ci].decl().name().str());
    }
    if (res.unsat_core_labels.empty()) {
      res.unsat_core_labels = violated_core;
    }
  } else {
    res.timed_out = true;
    res.sat = violated_core.empty();
    res.unsat_core_labels = violated_core;
  }
#else
  res.sat = violated_core.empty();
  res.unsat_core_labels = violated_core;
#endif

  res.dependent_fato_mutex_enforced = any_dep_mutex;
  res.ltfj_wake_buffer_s = max_wake_enforced;
  if (req.emit_smt2) {
    res.smt2_audit = smt.str();
  }
  const auto t1 = std::chrono::steady_clock::now();
  res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return res;
}

FleetScheduleResult QfLraLocalSolver::solve_repaired(
    const FleetScheduleRequest& req,
    const std::vector<std::size_t>& local_indices) const {
  const auto t0 = std::chrono::steady_clock::now();
  FleetScheduleResult res{};
  res.tier_used = SolverTier::TIER2_QF_LRA_LOCAL;
  res.backend_used = backend();
  res.smt2_logic = "QF_LRA";

  std::vector<std::size_t> ordered = local_indices;
  if (ordered.size() > kMaxLocalNeighborhood) {
    ordered.resize(kMaxLocalNeighborhood);
  }
  const std::size_t m = ordered.size();
  res.local_neighborhood_size = m;

  // Priority-first ordering strictly enforcing Regulation (EU) 2021/664 Art. 10(8)/(9)/(10).
  std::sort(ordered.begin(), ordered.end(), [&](std::size_t lhs, std::size_t rhs) {
    const auto& a = req.aircraft[lhs];
    const auto& b = req.aircraft[rhs];
    if (a.priority != b.priority) {
      return static_cast<std::uint8_t>(a.priority) < static_cast<std::uint8_t>(b.priority);
    }
    if (a.is_existing_approved != b.is_existing_approved) {
      return a.is_existing_approved && !b.is_existing_approved;
    }
    const double la = a.nominal_dep_s + a.min_transit_s;
    const double ra = b.nominal_dep_s + b.min_transit_s;
    if (la != ra) {
      return la < ra;
    }
    return a.aircraft_id < b.aircraft_id;
  });

  bool has_preemption = false;
  bool has_mayday = false;
  for (std::size_t idx : ordered) {
    const auto prio = req.aircraft[idx].priority;
    if (prio == PriorityClass::EMERGENCY_MAYDAY) {
      has_mayday = true;
      has_preemption = true;
    } else if (prio == PriorityClass::SPECIAL_OPS_EMS_SAR) {
      has_preemption = true;
    }
  }

  const std::size_t num_nodes = 1U + 2U * m;  // 0 = t_zero; 2*k+1 = dep_k; 2*k+2 = arr_k
  std::vector<std::string> safe_ids(m);
  std::vector<std::string> dep_vars(m);
  std::vector<std::string> arr_vars(m);
  for (std::size_t k = 0U; k < m; ++k) {
    const auto& a = req.aircraft[ordered[k]];
    safe_ids[k] = sanitize_ident(a.aircraft_id.empty() ? ("UA_" + std::to_string(ordered[k]))
                                                       : a.aircraft_id);
    dep_vars[k] = "dep_" + safe_ids[k];
    arr_vars[k] = "arr_" + safe_ids[k];
  }

  std::vector<LraEdge> edges;
  edges.reserve(8U * m + 20U);
  std::vector<std::string> bool_failures;
  std::vector<std::string> trackers;

  std::ostringstream smt;
  smt << "(set-logic QF_LRA)\n";
  smt << "(declare-const t_zero Real)\n";
  for (std::size_t k = 0U; k < m; ++k) {
    smt << "(declare-const " << dep_vars[k] << " Real)\n";
    smt << "(declare-const " << arr_vars[k] << " Real)\n";
  }
  smt << "(assert (= t_zero 0.0))\n";

  auto emit_tracked = [&](const std::string& label, const std::string& expr) {
    trackers.push_back(label);
    smt << "(declare-const " << label << " Bool)\n";
    smt << "(assert (=> " << label << ' ' << expr << "))\n";
  };

#ifdef VERTIRTA_HAS_Z3
  z3::context& ctx = impl_->ctx_lra;
  z3::solver& solver = impl_->solver_lra;
  solver.reset();
  z3::params zparams(ctx);
  zparams.set("unsat_core", true);
  zparams.set("rlimit", 250000U);
  zparams.set("timeout", static_cast<unsigned>(kTier2TimeoutMs));
  solver.set(zparams);

  z3::expr zzero = ctx.real_const("t_zero");
  solver.add(zzero == z3_qreal(ctx, 0LL));
  std::vector<z3::expr> zdep;
  std::vector<z3::expr> zarr;
  zdep.reserve(m);
  zarr.reserve(m);
  for (std::size_t k = 0U; k < m; ++k) {
    zdep.push_back(ctx.real_const(dep_vars[k].c_str()));
    zarr.push_back(ctx.real_const(arr_vars[k].c_str()));
  }
  z3::expr_vector assumptions(ctx);
  auto add_z3_tracked = [&](const std::string& label, const z3::expr& expr) {
    z3::expr b = ctx.bool_const(label.c_str());
    solver.add(z3::implies(b, expr));
    assumptions.push_back(b);
  };
#endif

  double max_wake_enforced = 0.0;
  bool any_dep_mutex = false;

  for (std::size_t k = 0U; k < m; ++k) {
    const auto& a = req.aircraft[ordered[k]];
    const std::size_t dep_node = 1U + 2U * k;
    const std::size_t arr_node = 2U + 2U * k;

    const std::int64_t early_q = scale_1e4(a.earliest_dep_s);
    const std::int64_t min_t_q = scale_1e4(a.min_transit_s);
    const std::int64_t max_t_q = scale_1e4(a.max_transit_s);
    const std::int64_t batt_q = scale_1e4(a.battery_max_arr_s);

    const std::string l_dep = "c_dep_window_" + safe_ids[k];
    const std::string l_tr = "c_transit_" + safe_ids[k];
    const std::string l_bat = "c_battery_" + safe_ids[k];

    edges.push_back(LraEdge{dep_node, 0U, -early_q, l_dep});
    edges.push_back(LraEdge{arr_node, dep_node, -min_t_q, l_tr});
    edges.push_back(LraEdge{dep_node, arr_node, max_t_q, l_tr});
    edges.push_back(LraEdge{0U, arr_node, batt_q, l_bat});

    emit_tracked(l_dep, "(>= (- " + dep_vars[k] + " t_zero) " + smt_real_q(early_q) + ")");
    emit_tracked(l_tr, "(and (>= (- " + arr_vars[k] + " " + dep_vars[k] + ") " +
                           smt_real_q(min_t_q) + ") (<= (- " + arr_vars[k] + " " + dep_vars[k] +
                           ") " + smt_real_q(max_t_q) + "))");
    emit_tracked(l_bat, "(<= (- " + arr_vars[k] + " t_zero) " + smt_real_q(batt_q) + ")");

#ifdef VERTIRTA_HAS_Z3
    add_z3_tracked(l_dep, (zdep[k] - zzero) >= z3_qreal(ctx, early_q));
    add_z3_tracked(l_tr, ((zarr[k] - zdep[k]) >= z3_qreal(ctx, min_t_q)) &&
                             ((zarr[k] - zdep[k]) <= z3_qreal(ctx, max_t_q)));
    add_z3_tracked(l_bat, (zarr[k] - zzero) <= z3_qreal(ctx, batt_q));
#endif

    // Art. 10(9): Without MAYDAY/EMS preemption, approved nominal U-plans remain immutable.
    if (!has_preemption && a.is_existing_approved &&
        a.priority == PriorityClass::APPROVED_NOMINAL) {
      const std::int64_t nom_q = scale_1e4(a.nominal_dep_s + a.min_transit_s);
      const std::string l_pres = "c_art10_9_preserve_" + safe_ids[k];
      edges.push_back(LraEdge{arr_node, 0U, -nom_q, l_pres});
      edges.push_back(LraEdge{0U, arr_node, nom_q, l_pres});
      emit_tracked(l_pres, "(= (- " + arr_vars[k] + " t_zero) " + smt_real_q(nom_q) + ")");
#ifdef VERTIRTA_HAS_Z3
      add_z3_tracked(l_pres, (zarr[k] - zzero) == z3_qreal(ctx, nom_q));
#endif
    }

    if (a.crosses_ltfj_wake_zone) {
      const std::string l_ats = "c_ats_clearance_" + safe_ids[k];
      emit_tracked(l_ats, a.ats_clearance_granted ? "true" : "false");
#ifdef VERTIRTA_HAS_Z3
      add_z3_tracked(l_ats, ctx.bool_val(a.ats_clearance_granted));
#endif
      if (!a.ats_clearance_granted) {
        bool_failures.push_back(l_ats);
      }

      const double wake_s = effective_ltfj_wake_s(a, req.ltfj_heavy_wake_s);
      max_wake_enforced = std::max(max_wake_enforced, wake_s);
      const std::int64_t clear_q = scale_1e4(a.ltfj_heavy_arrival_s + wake_s);
      const std::string l_wake = "c_ltfj_wake_" + safe_ids[k];
      edges.push_back(LraEdge{arr_node, 0U, -clear_q, l_wake});
      emit_tracked(l_wake, "(>= (- " + arr_vars[k] + " t_zero) " + smt_real_q(clear_q) + ")");
#ifdef VERTIRTA_HAS_Z3
      add_z3_tracked(l_wake, (zarr[k] - zzero) >= z3_qreal(ctx, clear_q));
#endif
    }
  }

  // Priority-ordered FATO & < 60 m dependent-FATO downwash mutex separation.
  for (std::size_t i = 0U; i < m; ++i) {
    for (std::size_t j = i + 1U; j < m; ++j) {
      const auto& ai = req.aircraft[ordered[i]];
      const auto& aj = req.aircraft[ordered[j]];
      if (!shares_fato_mutex(ai, aj, req.dependent_fato_spacing_m)) {
        continue;
      }
      const bool dep_group = (ai.fato_id != aj.fato_id);
      if (dep_group) {
        any_dep_mutex = true;
      }
      const std::int64_t sep_q = scale_1e4(std::max(ai.fato_dwell_s, req.siso_wake_spacing_s));
      const std::size_t arr_i = 2U + 2U * i;
      const std::size_t arr_j = 2U + 2U * j;
      const std::string l_mtx =
          (dep_group ? "c_dependent_fato_mutex_" : "c_fato_mutex_") + safe_ids[i] + "_" +
          safe_ids[j];
      // arr_j - arr_i >= sep_q  <=>  arr_i - arr_j <= -sep_q
      edges.push_back(LraEdge{arr_j, arr_i, -sep_q, l_mtx});
      emit_tracked(l_mtx, "(>= (- " + arr_vars[j] + " " + arr_vars[i] + ") " + smt_real_q(sep_q) +
                              ")");
#ifdef VERTIRTA_HAS_Z3
      add_z3_tracked(l_mtx, (zarr[j] - zarr[i]) >= z3_qreal(ctx, sep_q));
#endif
    }
  }

  smt << "(check-sat-assuming (";
  for (std::size_t ti = 0U; ti < trackers.size(); ++ti) {
    if (ti > 0U) {
      smt << ' ';
    }
    smt << trackers[ti];
  }
  smt << "))\n(get-unsat-core)\n";

  // Exact integer Bellman-Ford on reversed graph (y[k] = -x[k]) solving the exact LRA system.
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

  std::vector<std::string> fb_core = bool_failures;
  if (cycle_seed != num_nodes) {
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
        fb_core.push_back(edges[pe].label);
      }
      cur = parent_node[cur];
      if (cur == start || cur >= num_nodes) {
        break;
      }
    }
  }
  std::sort(fb_core.begin(), fb_core.end());
  fb_core.erase(std::unique(fb_core.begin(), fb_core.end()), fb_core.end());

#ifdef VERTIRTA_HAS_Z3
  const z3::check_result zres = solver.check(assumptions);
  if (zres == z3::sat) {
    res.sat = true;
  } else if (zres == z3::unsat) {
    res.sat = false;
    z3::expr_vector core = solver.unsat_core();
    for (unsigned ci = 0U; ci < core.size(); ++ci) {
      res.unsat_core_labels.push_back(core[ci].decl().name().str());
    }
    if (res.unsat_core_labels.empty()) {
      res.unsat_core_labels = fb_core;
    }
  } else {
    res.timed_out = true;
    res.sat = false;
    res.unsat_core_labels = {"c_tier2_qflra_timeout"};
  }
#else
  res.sat = fb_core.empty();
  res.unsat_core_labels = fb_core;
#endif

  res.dependent_fato_mutex_enforced = any_dep_mutex;
  res.ltfj_wake_buffer_s = max_wake_enforced;

  if (res.sat) {
    bool all_existing_preserved = true;
    double max_new_nom_delay = 0.0;
    bool any_shifted = false;

    res.schedule.reserve(m);
    for (std::size_t k = 0U; k < m; ++k) {
      const auto& a = req.aircraft[ordered[k]];
      const double dep_s = unscale_1e4(-y_dist[1U + 2U * k]);
      const double arr_s = unscale_1e4(-y_dist[2U + 2U * k]);
      const double nom_arr = a.nominal_dep_s + a.min_transit_s;
      const double delay_s = std::max(0.0, arr_s - nom_arr);
      const bool shifted = std::abs(arr_s - nom_arr) > 1e-3;
      if (shifted) {
        any_shifted = true;
      }
      if (a.is_existing_approved && a.priority == PriorityClass::APPROVED_NOMINAL && shifted) {
        all_existing_preserved = false;
      }
      if (a.priority == PriorityClass::NEW_NOMINAL) {
        max_new_nom_delay = std::max(max_new_nom_delay, delay_s);
      }
      res.schedule.push_back(ScheduledAircraftSlot{
          a.aircraft_id,
          a.fato_id,
          dep_s,
          arr_s,
          delay_s,
          shifted,
      });
    }

    res.repaired = any_shifted;
    res.existing_plan_preserved = all_existing_preserved;
    res.new_nominal_delayed_s = max_new_nom_delay;
    res.mayday_granted_priority = has_mayday;
  } else {
    res.tier_used = SolverTier::FALLBACK_SAFE_HOLD;
  }

  if (req.emit_smt2) {
    res.smt2_audit = smt.str();
  }
  const auto t1 = std::chrono::steady_clock::now();
  res.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return res;
}

}  // namespace vertirta::fleet_z3
