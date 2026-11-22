#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "vertirta/copilot/z3_corridor_verifier.hpp"

namespace {

using vertirta::copilot::ActionKind;
using vertirta::copilot::AirspacePolys;
using vertirta::copilot::CandidateMission;
using vertirta::copilot::ConvexPoly2D;
using vertirta::copilot::DynamicNfzCircle;
using vertirta::copilot::EnuPoint;
using vertirta::copilot::RtaStateSnapshotV1;

[[nodiscard]] bool contains_label(const std::vector<std::string>& labels,
                                  const std::string& target) {
  return std::find(labels.begin(), labels.end(), target) != labels.end();
}

[[nodiscard]] DynamicNfzCircle make_nfz(const char* name,
                                        double e,
                                        double n,
                                        float r,
                                        float floor_amsl,
                                        float ceil_amsl) {
  DynamicNfzCircle c{};
  c.set_id(name);
  c.center_enu = EnuPoint{e, n, 0.0};
  c.radius_m = r;
  c.floor_amsl_m = floor_amsl;
  c.ceil_amsl_m = ceil_amsl;
  c.active = true;
  return c;
}

[[nodiscard]] CandidateMission make_divert_to_slz(const RtaStateSnapshotV1& st,
                                                  const std::string& id) {
  const auto* slz = st.find_slz(id);
  REQUIRE(slz != nullptr);
  CandidateMission m{};
  m.action = ActionKind::DIVERT_SLZ;
  m.target_id = id;
  m.target_e_m = slz->enu.east_m;
  m.target_n_m = slz->enu.north_m;
  m.target_u_m = slz->enu.up_m;
  m.target_p_safe = static_cast<double>(slz->p_safe);
  m.target_ood_var = static_cast<double>(slz->slz_ood_variance);
  m.target_eta_s = static_cast<double>(slz->eta_s);
  m.cmd_agl_min_m = 35.0;
  m.cmd_agl_max_m = 120.0;
  m.cmd_speed_max_mps = 18.0;
  m.cmd_climb_req_mps = 1.5;
  m.cmd_accel_req_mps2 = 1.5;
  m.t1_s = 0.0;
  m.t2_s = m.target_eta_s;
  return m;
}

[[nodiscard]] CandidateMission make_divert_to_coord(double e,
                                                    double n,
                                                    double u,
                                                    double p_safe,
                                                    double eta_s = 100.0) {
  CandidateMission m{};
  m.action = ActionKind::DIVERT_SLZ;
  m.target_id = "CUSTOM_TARGET";
  m.target_e_m = e;
  m.target_n_m = n;
  m.target_u_m = u;
  m.target_p_safe = p_safe;
  m.target_ood_var = 0.03;
  m.target_eta_s = eta_s;
  m.cmd_agl_min_m = 35.0;
  m.cmd_agl_max_m = 120.0;
  m.cmd_speed_max_mps = 18.0;
  m.cmd_climb_req_mps = 1.5;
  m.cmd_accel_req_mps2 = 1.5;
  m.t1_s = 0.0;
  m.t2_s = eta_s;
  return m;
}

[[nodiscard]] CandidateMission make_hold_loiter(const RtaStateSnapshotV1& st, double dur_s) {
  CandidateMission m{};
  m.action = ActionKind::HOLD_LOITER;
  m.target_id = "CURRENT_POS";
  m.target_e_m = st.pos_enu.east_m;
  m.target_n_m = st.pos_enu.north_m;
  m.target_u_m = st.pos_enu.up_m;
  m.target_p_safe = 0.0;
  m.target_ood_var = 1.0;
  m.target_eta_s = dur_s;
  m.cmd_agl_min_m = 35.0;
  m.cmd_agl_max_m = 120.0;
  m.cmd_speed_max_mps = 5.0;
  m.cmd_climb_req_mps = 1.0;
  m.cmd_accel_req_mps2 = 1.0;
  m.t1_s = 0.0;
  m.t2_s = dur_s;
  return m;
}

// Cyrus-Beck exact parametric line-segment vs convex polygon intersection oracle.
[[nodiscard]] bool segment_intersects_convex_poly_oracle(double e0,
                                                         double n0,
                                                         double e1,
                                                         double n1,
                                                         const ConvexPoly2D& poly) noexcept {
  double t_min = 0.0;
  double t_max = 1.0;
  const double de = e1 - e0;
  const double dn = n1 - n0;
  for (const auto& hs : poly.halfspaces) {
    const double num = hs[2] - (hs[0] * e0 + hs[1] * n0);
    const double den = hs[0] * de + hs[1] * dn;
    if (std::abs(den) < 1e-12) {
      if (num < 0.0) {
        return false;
      }
    } else if (den > 0.0) {
      t_max = std::min(t_max, num / den);
    } else {
      t_min = std::max(t_min, num / den);
    }
    if (t_min > t_max) {
      return false;
    }
  }
  return t_min <= t_max;
}

}  // namespace

TEST_CASE("DR-6 Fix #1: Z3 assigns unique c_dyn_nfz_0..3 trackers across 2+ dynamic NFZs") {
  auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  st.num_dyn_nfz = 2U;
  st.dyn_nfzs[0] = make_nfz("HELI_0", -575.0, 475.0, 200.0f, 0.0f, 200.0f);
  st.dyn_nfzs[1] = make_nfz("CRANE_1", -150.0, 120.0, 150.0f, 0.0f, 200.0f);

  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();
  const auto res = vertirta::copilot::verify_mission_qflra(st, make_divert_to_slz(st, "Kartal_Sahil"), polys);
  REQUIRE_FALSE(res.is_sat);
  REQUIRE(contains_label(res.unsat_core_labels, "c_dyn_nfz_0"));
  REQUIRE(contains_label(res.unsat_core_labels, "c_dyn_nfz_1"));
  REQUIRE(res.auditable_smt2.find("(declare-const c_dyn_nfz_0 Bool)") != std::string::npos);
  REQUIRE(res.auditable_smt2.find("(declare-const c_dyn_nfz_1 Bool)") != std::string::npos);
}

TEST_CASE("DR-6 Fix #2: Z3 checks ownship Z, destination Z, cmd_agl, and dynamic NFZ floor_amsl_m") {
  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();

  // Case A: Ownship at 800 m ENU (> LTFJ 750 m ceil) descending into Pendik_Marina (-8.8 m ENU) -> MUST BE UNSAT!
  auto st_hi = vertirta::copilot::make_reference_400v_sml_snapshot();
  st_hi.pos_enu.up_m = 800.0;
  st_hi.agl_m = 800.0f;
  const auto m_pendik = make_divert_to_coord(4360.4, -2022.3, -8.8, 0.95, 100.0);
  const auto r_desc = vertirta::copilot::verify_mission_qflra(st_hi, m_pendik, polys);
  REQUIRE_FALSE(r_desc.is_sat);
  REQUIRE(contains_label(r_desc.unsat_core_labels, "c_ltfj_ctr"));

  // Case B: Elevated dynamic NFZ (floor 180 m AMSL = 168 m ENU, ceil 350 m AMSL = 338 m ENU) across path.
  auto st_nfz = vertirta::copilot::make_reference_400v_sml_snapshot();
  st_nfz.num_dyn_nfz = 1U;
  st_nfz.dyn_nfzs[0] = make_nfz("ELEV_NFZ", -575.0, 475.0, 200.0f, 180.0f, 350.0f);

  auto m_under = make_divert_to_slz(st_nfz, "Kartal_Sahil");
  m_under.cmd_agl_max_m = 120.0;  // z0=65, z1=0, z_cmd_max=120 < 168 m floor -> clears cleanly underneath!
  REQUIRE(vertirta::copilot::verify_mission_qflra(st_nfz, m_under, polys).is_sat);

  auto m_hit_floor = m_under;
  st_nfz.pos_enu.up_m = 175.0;  // Ownship penetrates elevated NFZ floor!
  const auto r_floor = vertirta::copilot::verify_mission_qflra(st_nfz, m_hit_floor, polys);
  REQUIRE_FALSE(r_floor.is_sat);
  REQUIRE(contains_label(r_floor.unsat_core_labels, "c_dyn_nfz_0"));
}

TEST_CASE("DR-6 Fix #3 & #4: Z3 checks cmd_speed_max/ams_max_climb and skips p_safe/ood_var on HOLD_LOITER") {
  const auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();

  auto fast = make_divert_to_slz(st, "Kartal_Sahil");
  fast.cmd_speed_max_mps = 25.0;
  REQUIRE(contains_label(vertirta::copilot::verify_mission_qflra(st, fast, polys).unsat_core_labels,
                         "c_ams_speed"));

  auto steep = make_divert_to_slz(st, "Kartal_Sahil");
  steep.cmd_climb_req_mps = 4.0;
  REQUIRE(contains_label(vertirta::copilot::verify_mission_qflra(st, steep, polys).unsat_core_labels,
                         "c_ams_climb"));

  auto hard_acc = make_divert_to_slz(st, "Kartal_Sahil");
  hard_acc.cmd_accel_req_mps2 = 5.0;
  REQUIRE(contains_label(vertirta::copilot::verify_mission_qflra(st, hard_acc, polys).unsat_core_labels,
                         "c_ams_accel"));

  // HOLD_LOITER with p_safe=0.0, ood_var=1.0 must return SAT (mid-air loiter has no ground pad)
  const auto loiter = make_hold_loiter(st, 30.0);
  REQUIRE(vertirta::copilot::verify_mission_qflra(st, loiter, polys).is_sat);
}

TEST_CASE("AIP LTFJ NW J2 & LTBX geometry separates all 4 coastal SLZs from Pendik/Yakacik/J2 traps") {
  const auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();

  REQUIRE(vertirta::copilot::verify_mission_qflra(st, make_divert_to_slz(st, "Kartal_Sahil"), polys).is_sat);
  REQUIRE(vertirta::copilot::verify_mission_qflra(st, make_divert_to_slz(st, "Cevizli_Helipad"), polys).is_sat);

  // Dragos_Plaza is clear of LTFJ/LTBX but violates c_soc_reserve at 34% SoC
  const auto r_dragos = vertirta::copilot::verify_mission_qflra(st, make_divert_to_slz(st, "Dragos_Plaza"), polys);
  REQUIRE_FALSE(r_dragos.is_sat);
  REQUIRE(contains_label(r_dragos.unsat_core_labels, "c_soc_reserve"));
  REQUIRE_FALSE(contains_label(r_dragos.unsat_core_labels, "c_ltfj_ctr"));
  REQUIRE_FALSE(contains_label(r_dragos.unsat_core_labels, "c_ltbx_ctr"));

  // NW J2 pinch point (+347.28, +2974.68) must be strictly inside LTFJ convex hull
  const auto r_j2 = vertirta::copilot::verify_mission_qflra(
      st, make_divert_to_coord(347.28, 2974.68, 0.0, 0.95, 100.0), polys);
  REQUIRE_FALSE(r_j2.is_sat);
  REQUIRE(contains_label(r_j2.unsat_core_labels, "c_ltfj_ctr"));

  // Yakacik_Intrude (+1432.0, +4331.3) violates BOTH LTFJ and LTBX
  const auto r_yakacik = vertirta::copilot::verify_mission_qflra(
      st, make_divert_to_coord(1432.0, 4331.3, 96.4, 0.95, 100.0), polys);
  REQUIRE_FALSE(r_yakacik.is_sat);
  REQUIRE(contains_label(r_yakacik.unsat_core_labels, "c_ltfj_ctr"));
  REQUIRE(contains_label(r_yakacik.unsat_core_labels, "c_ltbx_ctr"));
}

TEST_CASE("UNSAT-core repair resolves Pendik_Marina -> Kartal_Sahil in 1 pass (<15 ms) and appends check-sat-assuming") {
  const auto st = vertirta::copilot::make_reference_400v_sml_snapshot();
  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();
  const auto m_pendik = make_divert_to_coord(4360.4, -2022.3, -8.8, 0.95, 333.3);
  const auto r = vertirta::copilot::verify_mission_qflra(st, m_pendik, polys, /*emit_smt2=*/true);
  REQUIRE_FALSE(r.is_sat);
  REQUIRE(contains_label(r.unsat_core_labels, "c_ltfj_ctr"));
  REQUIRE(contains_label(r.unsat_core_labels, "c_soc_reserve"));
  REQUIRE(r.auditable_smt2.find("(check-sat-assuming (") != std::string::npos);

  const auto rep = vertirta::copilot::resolve_unsat_and_repair(st, m_pendik, r, polys);
  REQUIRE(rep.verified_sat);
  REQUIRE(rep.repaired);
  REQUIRE(rep.primary_unsat_label == "c_ltfj_ctr");
  REQUIRE(rep.fallback_target_id == "Kartal_Sahil");
  REQUIRE(rep.wall_ms < 15.0);
  REQUIRE(rep.explanation_tr.find("LTFJ") != std::string::npos);

  // Priority 1 (c_shield_bypass) beats c_ltfj_ctr
  auto m_bypass = m_pendik;
  m_bypass.requests_shield_disable = true;
  const auto r_byp = vertirta::copilot::verify_mission_qflra(st, m_bypass, polys, false);
  const auto rep_byp = vertirta::copilot::resolve_unsat_and_repair(st, m_bypass, r_byp, polys);
  REQUIRE(rep_byp.primary_unsat_label == "c_shield_bypass");
  REQUIRE(rep_byp.verified_sat);

  // Priority 6 bounded AGL clamp keeps Kartal_Sahil in 1 pass
  auto m_low_agl = make_divert_to_slz(st, "Kartal_Sahil");
  m_low_agl.cmd_agl_min_m = 15.0;
  const auto r_agl = vertirta::copilot::verify_mission_qflra(st, m_low_agl, polys, false);
  const auto rep_agl = vertirta::copilot::resolve_unsat_and_repair(st, m_low_agl, r_agl, polys);
  REQUIRE(rep_agl.primary_unsat_label == "c_min_agl");
  REQUIRE(rep_agl.verified_sat);
  REQUIRE(rep_agl.repaired_mission.cmd_agl_min_m == Catch::Approx(35.0));
}

TEST_CASE("1000-segment randomized sweep confirms 0 SAT/oracle mismatches and p99 <15 ms") {
  const auto base_st = vertirta::copilot::make_reference_400v_sml_snapshot();
  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();
  std::mt19937 rng(2026U);
  std::uniform_real_distribution<double> dist_e(-6000.0, 12000.0);
  std::uniform_real_distribution<double> dist_n(-4000.0, 12000.0);

  std::vector<double> wall_ms;
  wall_ms.reserve(1000U);
  int mismatches = 0;

  for (int i = 0; i < 1000; ++i) {
    auto st = base_st;
    st.pos_enu.east_m = dist_e(rng);
    st.pos_enu.north_m = dist_n(rng);
    st.ltbx_ctr_active = false;  // Isolate LTFJ 2D geometry comparison
    const double e1 = dist_e(rng);
    const double n1 = dist_n(rng);

    const auto m = make_divert_to_coord(e1, n1, 0.0, 0.95, 50.0);
    const auto r = vertirta::copilot::verify_mission_qflra(st, m, polys, /*emit_smt2=*/false);
    wall_ms.push_back(r.wall_ms);

    const bool oracle_hits =
        segment_intersects_convex_poly_oracle(st.pos_enu.east_m, st.pos_enu.north_m, e1, n1, polys.ltfj);
    const bool z3_hits = contains_label(r.unsat_core_labels, "c_ltfj_ctr");
    if (oracle_hits != z3_hits) {
      ++mismatches;
    }
  }
  REQUIRE(mismatches == 0);
  std::sort(wall_ms.begin(), wall_ms.end());
  const double p99 = wall_ms[989U];
  REQUIRE(p99 < 15.0);
}
