#include "vertirta/copilot/z3_corridor_verifier.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef VERTIRTA_HAS_Z3
#include <z3++.h>
#endif

namespace vertirta::copilot {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSatEps = 1e-3;
constexpr double kBufferM = 150.0;
constexpr double kCtrRadiusM = 9260.0;

// Round to 1e-4 rational grid so C++ QF_LRA evaluation and SMT-LIB2 (/ num 10000) match bit-for-bit.
[[nodiscard]] int64_t scale_1e4(double v) noexcept {
  if (!std::isfinite(v)) {
    return 0;
  }
  return static_cast<int64_t>(std::llround(v * 10000.0));
}

[[nodiscard]] double quant_1e4(double v) noexcept {
  return static_cast<double>(scale_1e4(v)) / 10000.0;
}

[[nodiscard]] std::string smt_real(double v) {
  const int64_t num = scale_1e4(v);
  if (num == 0) {
    return "0.0";
  }
  if (num < 0) {
    return "(- (/ " + std::to_string(-num) + " 10000))";
  }
  return "(/ " + std::to_string(num) + " 10000)";
}

[[nodiscard]] double cross_2d(const std::array<double, 2>& o,
                              const std::array<double, 2>& a,
                              const std::array<double, 2>& b) noexcept {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

[[nodiscard]] ConvexPoly2D build_convex_hull_poly(std::vector<std::array<double, 2>> pts,
                                                  double floor_enu_u,
                                                  double ceil_enu_u,
                                                  bool has_elevated_floor) {
  std::sort(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
    return (a[0] < b[0]) || (a[0] == b[0] && a[1] < b[1]);
  });
  pts.erase(std::unique(pts.begin(), pts.end(), [](const auto& a, const auto& b) {
              return std::abs(a[0] - b[0]) < 1e-6 && std::abs(a[1] - b[1]) < 1e-6;
            }),
            pts.end());

  std::vector<std::array<double, 2>> hull;
  hull.reserve(pts.size() * 2U);
  for (const auto& p : pts) {
    while (hull.size() >= 2U && cross_2d(hull[hull.size() - 2U], hull.back(), p) <= 0.0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
  const std::size_t lower_size = hull.size();
  for (std::size_t idx = pts.size(); idx-- > 0U;) {
    const auto& p = pts[idx];
    while (hull.size() > lower_size && cross_2d(hull[hull.size() - 2U], hull.back(), p) <= 0.0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }
  if (!hull.empty()) {
    hull.pop_back();
  }

  ConvexPoly2D poly;
  poly.floor_enu_u_m = floor_enu_u;
  poly.ceil_enu_u_m = ceil_enu_u;
  poly.has_elevated_floor = has_elevated_floor;
  poly.vertices = hull;
  poly.halfspaces.reserve(hull.size());

  const std::size_t m = hull.size();
  for (std::size_t i = 0U; i < m; ++i) {
    const auto& v0 = hull[i];
    const auto& v1 = hull[(i + 1U) % m];
    const double dx = v1[0] - v0[0];
    const double dy = v1[1] - v0[1];
    const double len = std::hypot(dx, dy);
    if (len < 1e-9) {
      continue;
    }
    // Outward unit normal for CCW polygon edge v0 -> v1 is (+dy/len, -dx/len).
    const double ax = dy / len;
    const double ay = -dx / len;
    const double b = ax * v0[0] + ay * v0[1];
    poly.halfspaces.push_back({ax, ay, b});
  }
  return poly;
}

void append_circum_ngon_vertices(double cx, double cy, double r, int n_sides,
                                 std::vector<std::array<double, 2>>& out) {
  const double r_v = r / std::cos(kPi / static_cast<double>(n_sides));
  for (int k = 0; k < n_sides; ++k) {
    const double tv = (2.0 * kPi * static_cast<double>(k) + kPi) / static_cast<double>(n_sides);
    out.push_back({cx + r_v * std::cos(tv), cy + r_v * std::sin(tv)});
  }
}

#ifndef VERTIRTA_HAS_Z3
[[nodiscard]] bool eval_segment_disjoint_2d_exact(double e0, double n0, double e1, double n1,
                                                  const ConvexPoly2D& poly) noexcept {
  const double qe0 = quant_1e4(e0);
  const double qn0 = quant_1e4(n0);
  const double qe1 = quant_1e4(e1);
  const double qn1 = quant_1e4(n1);

  for (const auto& hs : poly.halfspaces) {
    const double ax = quant_1e4(hs[0]);
    const double ay = quant_1e4(hs[1]);
    const double b = quant_1e4(hs[2]);
    if (ax * qe0 + ay * qn0 > b + kSatEps && ax * qe1 + ay * qn1 > b + kSatEps) {
      return true;
    }
  }

  const double de = qe1 - qe0;
  const double dn = qn1 - qn0;
  bool all_pos = !poly.vertices.empty();
  bool all_neg = !poly.vertices.empty();
  for (const auto& v : poly.vertices) {
    const double cross = de * quant_1e4(v[1] - qn0) - dn * quant_1e4(v[0] - qe0);
    if (!(cross > kSatEps)) {
      all_pos = false;
    }
    if (!(cross < -kSatEps)) {
      all_neg = false;
    }
  }
  return all_pos || all_neg;
}
#endif

[[nodiscard]] std::string smt_segment_disjoint(double e0, double n0, const ConvexPoly2D& poly) {
  const double qe0 = quant_1e4(e0);
  const double qn0 = quant_1e4(n0);
  std::ostringstream oss;
  oss << "(or";
  for (const auto& hs : poly.halfspaces) {
    const double ax = quant_1e4(hs[0]);
    const double ay = quant_1e4(hs[1]);
    const double b = quant_1e4(hs[2]);
    if (ax * qe0 + ay * qn0 > b + kSatEps) {
      oss << " (> (+ (* " << smt_real(ax) << " e1) (* " << smt_real(ay) << " n1)) (+ "
          << smt_real(b) << " " << smt_real(kSatEps) << "))";
    }
  }
  if (!poly.vertices.empty()) {
    oss << " (and";
    for (const auto& v : poly.vertices) {
      oss << " (> (- (* (- e1 " << smt_real(qe0) << ") " << smt_real(v[1] - qn0)
          << ") (* (- n1 " << smt_real(qn0) << ") " << smt_real(v[0] - qe0) << ")) "
          << smt_real(kSatEps) << ")";
    }
    oss << ") (and";
    for (const auto& v : poly.vertices) {
      oss << " (< (- (* (- e1 " << smt_real(qe0) << ") " << smt_real(v[1] - qn0)
          << ") (* (- n1 " << smt_real(qn0) << ") " << smt_real(v[0] - qe0) << ")) (- "
          << smt_real(kSatEps) << "))";
    }
    oss << ")";
  }
  oss << ")";
  return oss.str();
}

[[nodiscard]] std::string smt_zone_clearance(double e0, double n0, const ConvexPoly2D& poly) {
  std::ostringstream oss;
  const std::string ceil_plus = "(+ " + smt_real(poly.ceil_enu_u_m) + " " + smt_real(kSatEps) + ")";
  const std::string above_expr =
      "(and (> z0 " + ceil_plus + ") (> z1 " + ceil_plus + ") (> z_cmd_min " + ceil_plus + "))";
  if (poly.has_elevated_floor) {
    const std::string floor_minus =
        "(- " + smt_real(poly.floor_enu_u_m) + " " + smt_real(kSatEps) + ")";
    const std::string below_expr = "(and (< z0 " + floor_minus + ") (< z1 " + floor_minus +
                                   ") (< z_cmd_max " + floor_minus + "))";
    oss << "(or " << smt_segment_disjoint(e0, n0, poly) << " " << above_expr << " " << below_expr
        << ")";
  } else {
    oss << "(or " << smt_segment_disjoint(e0, n0, poly) << " " << above_expr << ")";
  }
  return oss.str();
}

#ifdef VERTIRTA_HAS_Z3
[[nodiscard]] z3::expr z3_qreal(z3::context& c, double val) {
  return c.real_val(scale_1e4(val), 10000);
}

[[nodiscard]] z3::expr z3_segment_disjoint(z3::context& c, double e0, double n0,
                                           const z3::expr& e1, const z3::expr& n1,
                                           const ConvexPoly2D& poly) {
  const double qe0 = quant_1e4(e0);
  const double qn0 = quant_1e4(n0);
  z3::expr_vector disj(c);
  z3::expr eps = z3_qreal(c, kSatEps);
  for (const auto& hs : poly.halfspaces) {
    const double ax = quant_1e4(hs[0]);
    const double ay = quant_1e4(hs[1]);
    const double b = quant_1e4(hs[2]);
    if (ax * qe0 + ay * qn0 > b + kSatEps) {
      disj.push_back(z3_qreal(c, ax) * e1 + z3_qreal(c, ay) * n1 > z3_qreal(c, b) + eps);
    }
  }
  if (!poly.vertices.empty()) {
    z3::expr_vector all_pos(c);
    z3::expr_vector all_neg(c);
    z3::expr de = e1 - z3_qreal(c, qe0);
    z3::expr dn = n1 - z3_qreal(c, qn0);
    for (const auto& v : poly.vertices) {
      z3::expr cr = de * z3_qreal(c, v[1] - qn0) - dn * z3_qreal(c, v[0] - qe0);
      all_pos.push_back(cr > eps);
      all_neg.push_back(cr < -eps);
    }
    disj.push_back(z3::mk_and(all_pos));
    disj.push_back(z3::mk_and(all_neg));
  }
  if (disj.empty()) {
    return c.bool_val(false);
  }
  return z3::mk_or(disj);
}
#endif

}  // namespace

ConvexPoly2D make_circum_8gon(double cx, double cy, double r, double floor_enu_u,
                              double ceil_enu_u, bool has_elevated_floor) {
  std::vector<std::array<double, 2>> pts;
  pts.reserve(8U);
  append_circum_ngon_vertices(cx, cy, r, 8, pts);
  return build_convex_hull_poly(std::move(pts), floor_enu_u, ceil_enu_u, has_elevated_floor);
}

AirspacePolys load_istanbul_airspace_polys() {
  AirspacePolys polys;

  // LTFJ CTR (Sabiha Gokcen): Center ENU (+10210.23, +1376.73), R_buf = 9410.0 m
  // plus NW J1 (+11528.77, +11280.95) and J2 (+347.28, +2974.68) with 150 m buffer.
  // Ceiling = 762.0 m AMSL = 750.0 m ENU (relative to 12.0 m Kartal datum).
  std::vector<std::array<double, 2>> ltfj_pts;
  ltfj_pts.reserve(32U);
  append_circum_ngon_vertices(10210.23, 1376.73, kCtrRadiusM + kBufferM, 16, ltfj_pts);
  append_circum_ngon_vertices(11528.77, 11280.95, kBufferM, 8, ltfj_pts);
  append_circum_ngon_vertices(347.28, 2974.68, kBufferM, 8, ltfj_pts);
  polys.ltfj = build_convex_hull_poly(std::move(ltfj_pts), -12.0, 750.0, false);

  // LTBX CTR (Samandira): Center ENU (+2365.23, +12537.31), R_buf = 9410.0 m.
  // Ceiling = 609.6 m AMSL = 597.6 m ENU.
  std::vector<std::array<double, 2>> ltbx_pts;
  ltbx_pts.reserve(16U);
  append_circum_ngon_vertices(2365.23, 12537.31, kCtrRadiusM + kBufferM, 16, ltbx_pts);
  polys.ltbx = build_convex_hull_poly(std::move(ltbx_pts), -12.0, 597.6, false);

  return polys;
}

std::vector<stl::HalfSpace2D> to_stl_halfspaces(const ConvexPoly2D& poly) {
  std::vector<stl::HalfSpace2D> out;
  out.reserve(poly.halfspaces.size());
  for (const auto& hs : poly.halfspaces) {
    out.push_back(stl::HalfSpace2D{hs[0], hs[1], hs[2]});
  }
  return out;
}

CandidateMission build_candidate_mission(const CompactDslAst& ast,
                                         const RtaStateSnapshotV1& st) noexcept {
  CandidateMission m{};
  m.action = ast.action;
  m.cmd_agl_min_m = static_cast<double>(ast.agl_min_m);
  m.cmd_agl_max_m = static_cast<double>(ast.agl_max_m);
  m.cmd_speed_max_mps = static_cast<double>(ast.speed_max_mps);
  m.cmd_climb_req_mps = std::min(1.5, static_cast<double>(st.ams.ams_max_climb_mps));
  m.cmd_accel_req_mps2 = std::min(1.5, static_cast<double>(st.ams.ams_max_accel_mps2));
  m.t1_s = 0.0;
  m.active_action_count = 1;
  m.requests_shield_disable = (ast.violation_code == ViolationCode::SHIELD_BYPASS);

  if (ast.action == ActionKind::HOLD_LOITER || ast.action == ActionKind::REQUEST_CLARIFICATION) {
    m.action = ActionKind::HOLD_LOITER;
    m.target_id = "CURRENT_POS";
    m.target_e_m = st.pos_enu.east_m;
    m.target_n_m = st.pos_enu.north_m;
    m.target_u_m = st.pos_enu.up_m;
    m.target_p_safe = 0.0;
    m.target_ood_var = 1.0;
    m.target_eta_s = static_cast<double>((ast.duration_s > 0) ? ast.duration_s : 30);
    m.t2_s = m.target_eta_s;
    return m;
  }

  const std::string_view tgt =
      (ast.action == ActionKind::REJECT_UNSAFE) ? ast.fallback_target() : ast.target_id();
  m.target_id = std::string(tgt);
  if (const SlzCandidate* slz = st.find_slz(tgt); slz != nullptr) {
    m.target_e_m = slz->enu.east_m;
    m.target_n_m = slz->enu.north_m;
    m.target_u_m = slz->enu.up_m;
    m.target_p_safe = static_cast<double>(slz->p_safe);
    m.target_ood_var = static_cast<double>(slz->slz_ood_variance);
    m.target_eta_s = static_cast<double>(slz->eta_s);
    m.t2_s = static_cast<double>((ast.eta_lim_s > 0) ? ast.eta_lim_s
                                                     : static_cast<int32_t>(std::ceil(slz->eta_s)));
  } else {
    m.target_p_safe = 0.0;
    m.target_ood_var = 1.0;
    m.target_eta_s = static_cast<double>((ast.eta_lim_s > 0) ? ast.eta_lim_s : 100);
    m.t2_s = m.target_eta_s;
  }
  return m;
}

Z3VerifyResult verify_mission_qflra(const RtaStateSnapshotV1& st,
                                    const CandidateMission& m,
                                    const AirspacePolys& polys,
                                    bool emit_smt2) {
  const auto t_start = std::chrono::steady_clock::now();
  Z3VerifyResult res{};

  const bool is_loiter = (m.action == ActionKind::HOLD_LOITER);
  const double e0 = st.pos_enu.east_m;
  const double n0 = st.pos_enu.north_m;
  const double e1 = is_loiter ? e0 : m.target_e_m;
  const double n1 = is_loiter ? n0 : m.target_n_m;

  // DR-6 Fix #2: Full 3D vertical slab bounds (ownship Z, destination pad Z, cmd AGL, terrain)
  const double z0 = st.pos_enu.up_m;
  const double z1 = is_loiter ? z0 : m.target_u_m;
  const double terrain_0 = st.pos_enu.up_m - static_cast<double>(st.agl_m);
  const double terrain_1 = is_loiter ? terrain_0 : m.target_u_m;
  const double z_cmd_min = m.cmd_agl_min_m + std::min(terrain_0, terrain_1);
  const double z_cmd_max = m.cmd_agl_max_m + std::max(terrain_0, terrain_1);

  const double t_max_soc =
      std::max(0.0, (static_cast<double>(st.soc_pct) - static_cast<double>(st.soc_reserve_floor_pct)) /
                        std::max(1e-3, static_cast<double>(st.ams.burn_rate_soc_pct_per_s)));

  // Build active dynamic NFZ polygons with unique tracker names c_dyn_nfz_0..3 (DR-6 Fix #1)
  struct TrackedDynNfz {
    std::string label;
    ConvexPoly2D poly;
  };
  std::vector<TrackedDynNfz> dyn_polys;
  const uint8_t n_nfz =
      (st.num_dyn_nfz <= kMaxDynamicNfz) ? st.num_dyn_nfz : static_cast<uint8_t>(kMaxDynamicNfz);
  dyn_polys.reserve(n_nfz);
  for (uint8_t k = 0U; k < n_nfz; ++k) {
    if (!st.dyn_nfzs[k].active) {
      continue;
    }
    const auto& nfz = st.dyn_nfzs[k];
    const double floor_enu = static_cast<double>(nfz.floor_amsl_m) - kKartalOriginAmslM;
    const double ceil_enu = static_cast<double>(nfz.ceil_amsl_m) - kKartalOriginAmslM;
    const bool has_elev_floor = (nfz.floor_amsl_m > 0.0f);
    dyn_polys.push_back(TrackedDynNfz{
        "c_dyn_nfz_" + std::to_string(k),
        make_circum_8gon(nfz.center_enu.east_m, nfz.center_enu.north_m,
                         static_cast<double>(nfz.radius_m) + 50.0, floor_enu, ceil_enu,
                         has_elev_floor),
    });
  }

  // Emit complete auditable SMT-LIB2 (Z3 Issue #2589: explicit check-sat-assuming + get-unsat-core)
  if (emit_smt2) {
    std::vector<std::string> trackers;
    std::ostringstream smt;
    smt << "(set-logic QF_LRA)\n"
        << "(set-option :produce-unsat-cores true)\n"
        << "(set-option :smt.core.minimize true)\n"
        << "(set-option :rlimit 250000)\n"
        << "(set-option :timeout 12)\n"
        << "(declare-const e1 Real)\n(declare-const n1 Real)\n"
        << "(declare-const z0 Real)\n(declare-const z1 Real)\n"
        << "(declare-const z_cmd_min Real)\n(declare-const z_cmd_max Real)\n"
        << "(declare-const agl_min Real)\n(declare-const agl_max Real)\n"
        << "(declare-const t_eta Real)\n(declare-const t1 Real)\n(declare-const t2 Real)\n"
        << "(declare-const p_safe Real)\n(declare-const ood_var Real)\n"
        << "(declare-const accel Real)\n(declare-const speed Real)\n(declare-const climb Real)\n"
        << "(declare-const n_act Real)\n";

    smt << "(assert (= e1 " << smt_real(e1) << "))\n"
        << "(assert (= n1 " << smt_real(n1) << "))\n"
        << "(assert (= z0 " << smt_real(z0) << "))\n"
        << "(assert (= z1 " << smt_real(z1) << "))\n"
        << "(assert (= z_cmd_min " << smt_real(z_cmd_min) << "))\n"
        << "(assert (= z_cmd_max " << smt_real(z_cmd_max) << "))\n"
        << "(assert (= agl_min " << smt_real(m.cmd_agl_min_m) << "))\n"
        << "(assert (= agl_max " << smt_real(m.cmd_agl_max_m) << "))\n"
        << "(assert (= t_eta " << smt_real(m.target_eta_s) << "))\n"
        << "(assert (= t1 " << smt_real(m.t1_s) << "))\n"
        << "(assert (= t2 " << smt_real(m.t2_s) << "))\n"
        << "(assert (= p_safe " << smt_real(m.target_p_safe) << "))\n"
        << "(assert (= ood_var " << smt_real(m.target_ood_var) << "))\n"
        << "(assert (= accel " << smt_real(m.cmd_accel_req_mps2) << "))\n"
        << "(assert (= speed " << smt_real(m.cmd_speed_max_mps) << "))\n"
        << "(assert (= climb " << smt_real(m.cmd_climb_req_mps) << "))\n"
        << "(assert (= n_act " << smt_real(static_cast<double>(m.active_action_count)) << "))\n";

    auto add_tracked = [&](const std::string& label, const std::string& expr) {
      trackers.push_back(label);
      smt << "(declare-const " << label << " Bool)\n"
          << "(assert (=> " << label << " " << expr << "))\n";
    };

    add_tracked("c_shield_bypass", (!m.requests_shield_disable && st.shield_active) ? "true" : "false");
    add_tracked("c_action_mutex", "(= n_act 1.0)");
    if (st.ltfj_ctr_active) {
      add_tracked("c_ltfj_ctr", smt_zone_clearance(e0, n0, polys.ltfj));
    }
    if (st.ltbx_ctr_active) {
      add_tracked("c_ltbx_ctr", smt_zone_clearance(e0, n0, polys.ltbx));
    }
    for (const auto& dn : dyn_polys) {
      add_tracked(dn.label, smt_zone_clearance(e0, n0, dn.poly));
    }
    add_tracked("c_soc_reserve", "(<= t_eta " + smt_real(t_max_soc) + ")");
    add_tracked("c_min_agl", "(>= agl_min " + smt_real(kDefaultMinAglM) + ")");
    add_tracked("c_max_agl", "(<= agl_max " + smt_real(kDefaultMaxAglM) + ")");
    if (!is_loiter) {
      add_tracked("c_slz_unsafe", "(>= p_safe " + smt_real(kDefaultMinPsafe) + ")");
      add_tracked("c_slz_ood", "(<= ood_var " + smt_real(kDefaultMaxOodVar) + ")");
    }
    add_tracked("c_ams_accel", "(<= accel " + smt_real(st.ams.ams_max_accel_mps2) + ")");
    add_tracked("c_ams_speed", "(<= speed " + smt_real(st.ams.max_cruise_airspeed_mps) + ")");
    add_tracked("c_ams_climb", "(<= climb " + smt_real(st.ams.ams_max_climb_mps) + ")");
    add_tracked("c_temporal_order", "(and (>= t1 0.0) (< t1 t2) (<= t2 " + smt_real(t_max_soc) + "))");

    smt << "(check-sat-assuming (";
    for (std::size_t i = 0U; i < trackers.size(); ++i) {
      if (i > 0U) {
        smt << ' ';
      }
      smt << trackers[i];
    }
    smt << "))\n(get-unsat-core)\n";
    res.auditable_smt2 = smt.str();
  }

#ifdef VERTIRTA_HAS_Z3
  z3::context ctx;
  z3::solver s(ctx, "QF_LRA");
  z3::params p(ctx);
  p.set("unsat_core", true);
  p.set("core.minimize", true);
  p.set("rlimit", 250000U);
  p.set("timeout", 12U);
  s.set(p);

  z3::expr ze1 = ctx.real_const("e1");
  z3::expr zn1 = ctx.real_const("n1");
  z3::expr zz0 = ctx.real_const("z0");
  z3::expr zz1 = ctx.real_const("z1");
  z3::expr zzmin = ctx.real_const("z_cmd_min");
  z3::expr zzmax = ctx.real_const("z_cmd_max");
  z3::expr zagl_min = ctx.real_const("agl_min");
  z3::expr zagl_max = ctx.real_const("agl_max");
  z3::expr zt_eta = ctx.real_const("t_eta");
  z3::expr zt1 = ctx.real_const("t1");
  z3::expr zt2 = ctx.real_const("t2");
  z3::expr zp_safe = ctx.real_const("p_safe");
  z3::expr zood = ctx.real_const("ood_var");
  z3::expr zaccel = ctx.real_const("accel");
  z3::expr zspeed = ctx.real_const("speed");
  z3::expr zclimb = ctx.real_const("climb");
  z3::expr zn_act = ctx.real_const("n_act");

  s.add(ze1 == z3_qreal(ctx, e1));
  s.add(zn1 == z3_qreal(ctx, n1));
  s.add(zz0 == z3_qreal(ctx, z0));
  s.add(zz1 == z3_qreal(ctx, z1));
  s.add(zzmin == z3_qreal(ctx, z_cmd_min));
  s.add(zzmax == z3_qreal(ctx, z_cmd_max));
  s.add(zagl_min == z3_qreal(ctx, m.cmd_agl_min_m));
  s.add(zagl_max == z3_qreal(ctx, m.cmd_agl_max_m));
  s.add(zt_eta == z3_qreal(ctx, m.target_eta_s));
  s.add(zt1 == z3_qreal(ctx, m.t1_s));
  s.add(zt2 == z3_qreal(ctx, m.t2_s));
  s.add(zp_safe == z3_qreal(ctx, m.target_p_safe));
  s.add(zood == z3_qreal(ctx, m.target_ood_var));
  s.add(zaccel == z3_qreal(ctx, m.cmd_accel_req_mps2));
  s.add(zspeed == z3_qreal(ctx, m.cmd_speed_max_mps));
  s.add(zclimb == z3_qreal(ctx, m.cmd_climb_req_mps));
  s.add(zn_act == z3_qreal(ctx, static_cast<double>(m.active_action_count)));

  auto z3_zone_expr = [&](const ConvexPoly2D& poly) {
    z3::expr eps = z3_qreal(ctx, kSatEps);
    z3::expr ceil_e = z3_qreal(ctx, poly.ceil_enu_u_m) + eps;
    z3::expr above = (zz0 > ceil_e) && (zz1 > ceil_e) && (zzmin > ceil_e);
    z3::expr horiz = z3_segment_disjoint(ctx, e0, n0, ze1, zn1, poly);
    if (poly.has_elevated_floor) {
      z3::expr floor_e = z3_qreal(ctx, poly.floor_enu_u_m) - eps;
      z3::expr below = (zz0 < floor_e) && (zz1 < floor_e) && (zzmax < floor_e);
      return horiz || above || below;
    }
    return horiz || above;
  };

  s.add(ctx.bool_val(!m.requests_shield_disable && st.shield_active), "c_shield_bypass");
  s.add(zn_act == z3_qreal(ctx, 1.0), "c_action_mutex");
  if (st.ltfj_ctr_active) {
    s.add(z3_zone_expr(polys.ltfj), "c_ltfj_ctr");
  }
  if (st.ltbx_ctr_active) {
    s.add(z3_zone_expr(polys.ltbx), "c_ltbx_ctr");
  }
  for (const auto& dn : dyn_polys) {
    s.add(z3_zone_expr(dn.poly), dn.label.c_str());
  }
  s.add(zt_eta <= z3_qreal(ctx, t_max_soc), "c_soc_reserve");
  s.add(zagl_min >= z3_qreal(ctx, kDefaultMinAglM), "c_min_agl");
  s.add(zagl_max <= z3_qreal(ctx, kDefaultMaxAglM), "c_max_agl");
  if (!is_loiter) {
    s.add(zp_safe >= z3_qreal(ctx, kDefaultMinPsafe), "c_slz_unsafe");
    s.add(zood <= z3_qreal(ctx, kDefaultMaxOodVar), "c_slz_ood");
  }
  s.add(zaccel <= z3_qreal(ctx, st.ams.ams_max_accel_mps2), "c_ams_accel");
  s.add(zspeed <= z3_qreal(ctx, st.ams.max_cruise_airspeed_mps), "c_ams_speed");
  s.add(zclimb <= z3_qreal(ctx, st.ams.ams_max_climb_mps), "c_ams_climb");
  s.add((zt1 >= z3_qreal(ctx, 0.0)) && (zt1 < zt2) && (zt2 <= z3_qreal(ctx, t_max_soc)),
        "c_temporal_order");

  const z3::check_result status = s.check();
  if (status == z3::sat) {
    res.is_sat = true;
  } else if (status == z3::unsat) {
    z3::expr_vector core = s.unsat_core();
    for (unsigned i = 0U; i < core.size(); ++i) {
      res.unsat_core_labels.push_back(core[i].decl().name().str());
    }
  } else {
    res.timed_out = true;
    res.unsat_core_labels.push_back("c_solver_timeout");
  }
#else
  auto check_zone_clear = [&](const ConvexPoly2D& poly) noexcept -> bool {
    if (eval_segment_disjoint_2d_exact(e0, n0, e1, n1, poly)) {
      return true;
    }
    const double qz0 = quant_1e4(z0);
    const double qz1 = quant_1e4(z1);
    const double qzmin = quant_1e4(z_cmd_min);
    const double qzmax = quant_1e4(z_cmd_max);
    const double qceil = quant_1e4(poly.ceil_enu_u_m);
    const double qfloor = quant_1e4(poly.floor_enu_u_m);
    const bool above = (qz0 > qceil + kSatEps) && (qz1 > qceil + kSatEps) && (qzmin > qceil + kSatEps);
    const bool below = poly.has_elevated_floor && (qz0 < qfloor - kSatEps) &&
                       (qz1 < qfloor - kSatEps) && (qzmax < qfloor - kSatEps);
    return above || below;
  };

  // Mathematically exact QF_LRA evaluator matching the emitted SMT-LIB2 rational assertions 1:1.
  if (m.requests_shield_disable || !st.shield_active) {
    res.unsat_core_labels.push_back("c_shield_bypass");
  }
  if (quant_1e4(static_cast<double>(m.active_action_count)) != 1.0) {
    res.unsat_core_labels.push_back("c_action_mutex");
  }
  if (st.ltfj_ctr_active && !check_zone_clear(polys.ltfj)) {
    res.unsat_core_labels.push_back("c_ltfj_ctr");
  }
  if (st.ltbx_ctr_active && !check_zone_clear(polys.ltbx)) {
    res.unsat_core_labels.push_back("c_ltbx_ctr");
  }
  for (const auto& dn : dyn_polys) {
    if (!check_zone_clear(dn.poly)) {
      res.unsat_core_labels.push_back(dn.label);
    }
  }
  if (quant_1e4(m.target_eta_s) > quant_1e4(t_max_soc)) {
    res.unsat_core_labels.push_back("c_soc_reserve");
  }
  if (quant_1e4(m.cmd_agl_min_m) < quant_1e4(kDefaultMinAglM)) {
    res.unsat_core_labels.push_back("c_min_agl");
  }
  if (quant_1e4(m.cmd_agl_max_m) > quant_1e4(kDefaultMaxAglM)) {
    res.unsat_core_labels.push_back("c_max_agl");
  }
  if (!is_loiter) {
    if (quant_1e4(m.target_p_safe) < quant_1e4(kDefaultMinPsafe)) {
      res.unsat_core_labels.push_back("c_slz_unsafe");
    }
    if (quant_1e4(m.target_ood_var) > quant_1e4(kDefaultMaxOodVar)) {
      res.unsat_core_labels.push_back("c_slz_ood");
    }
  }
  if (quant_1e4(m.cmd_accel_req_mps2) > quant_1e4(st.ams.ams_max_accel_mps2)) {
    res.unsat_core_labels.push_back("c_ams_accel");
  }
  if (quant_1e4(m.cmd_speed_max_mps) > quant_1e4(st.ams.max_cruise_airspeed_mps)) {
    res.unsat_core_labels.push_back("c_ams_speed");
  }
  if (quant_1e4(m.cmd_climb_req_mps) > quant_1e4(st.ams.ams_max_climb_mps)) {
    res.unsat_core_labels.push_back("c_ams_climb");
  }
  if (!(quant_1e4(m.t1_s) >= 0.0 && quant_1e4(m.t1_s) < quant_1e4(m.t2_s) &&
        quant_1e4(m.t2_s) <= quant_1e4(t_max_soc))) {
    res.unsat_core_labels.push_back("c_temporal_order");
  }
  res.is_sat = res.unsat_core_labels.empty();
#endif

  const auto t_end = std::chrono::steady_clock::now();
  res.wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return res;
}

}  // namespace vertirta::copilot
