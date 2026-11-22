#include <iostream>
#include <string>
#include <vector>

#include "vertirta/copilot/z3_corridor_verifier.hpp"

namespace {

using vertirta::copilot::ActionKind;
using vertirta::copilot::CandidateMission;
using vertirta::copilot::DynamicNfzCircle;
using vertirta::copilot::EnuPoint;
using vertirta::copilot::RtaStateSnapshotV1;

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 32U);
  for (char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else {
      out += c;
    }
  }
  return out;
}

void emit_case(const std::string& name,
               const RtaStateSnapshotV1& st,
               const CandidateMission& m,
               const vertirta::copilot::AirspacePolys& polys) {
  const auto res = vertirta::copilot::verify_mission_qflra(st, m, polys, /*emit_smt2=*/true);
  std::cout << "{\"name\":\"" << json_escape(name) << "\",\"cpp_is_sat\":"
            << (res.is_sat ? "true" : "false") << ",\"cpp_unsat_core\":[";
  for (std::size_t i = 0U; i < res.unsat_core_labels.size(); ++i) {
    if (i > 0U) {
      std::cout << ',';
    }
    std::cout << '"' << json_escape(res.unsat_core_labels[i]) << '"';
  }
  std::cout << "],\"smt2\":\"" << json_escape(res.auditable_smt2) << "\"}\n";
}

CandidateMission make_slz_mission(const RtaStateSnapshotV1& st, const std::string& id) {
  const auto* slz = st.find_slz(id);
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

}  // namespace

int main() {
  const auto base_st = vertirta::copilot::make_reference_400v_sml_snapshot();
  const auto polys = vertirta::copilot::load_istanbul_airspace_polys();

  // 1. Nominal Kartal_Sahil -> SAT
  emit_case("kartal_sahil_sat", base_st, make_slz_mission(base_st, "Kartal_Sahil"), polys);

  // 2. Pendik_Marina -> UNSAT (c_ltfj_ctr, c_soc_reserve)
  {
    CandidateMission m = make_slz_mission(base_st, "Kartal_Sahil");
    m.target_id = "Pendik_Marina";
    m.target_e_m = 4360.4;
    m.target_n_m = -2022.3;
    m.target_u_m = -8.8;
    m.target_eta_s = 333.3;
    m.t2_s = 333.3;
    emit_case("pendik_marina_unsat", base_st, m, polys);
  }

  // 3. Ownship 800m descending into Pendik_Marina -> UNSAT (c_ltfj_ctr)
  {
    auto st = base_st;
    st.pos_enu.up_m = 800.0;
    st.agl_m = 800.0f;
    CandidateMission m = make_slz_mission(base_st, "Kartal_Sahil");
    m.target_e_m = 4360.4;
    m.target_n_m = -2022.3;
    m.target_u_m = -8.8;
    emit_case("vertical_800m_to_ltfj_unsat", st, m, polys);
  }

  // 4. 2x dynamic NFZs -> UNSAT (c_dyn_nfz_0, c_dyn_nfz_1)
  {
    auto st = base_st;
    st.num_dyn_nfz = 2U;
    st.dyn_nfzs[0].set_id("HELI_0");
    st.dyn_nfzs[0].center_enu = EnuPoint{-575.0, 475.0, 0.0};
    st.dyn_nfzs[0].radius_m = 200.0f;
    st.dyn_nfzs[0].floor_amsl_m = 0.0f;
    st.dyn_nfzs[0].ceil_amsl_m = 200.0f;
    st.dyn_nfzs[0].active = true;
    st.dyn_nfzs[1].set_id("CRANE_1");
    st.dyn_nfzs[1].center_enu = EnuPoint{-150.0, 120.0, 0.0};
    st.dyn_nfzs[1].radius_m = 150.0f;
    st.dyn_nfzs[1].floor_amsl_m = 0.0f;
    st.dyn_nfzs[1].ceil_amsl_m = 200.0f;
    st.dyn_nfzs[1].active = true;
    emit_case("dual_dyn_nfz_unsat", st, make_slz_mission(st, "Kartal_Sahil"), polys);
  }

  // 5. Elevated dynamic NFZ cleared below floor -> SAT
  {
    auto st = base_st;
    st.num_dyn_nfz = 1U;
    st.dyn_nfzs[0].set_id("ELEV_0");
    st.dyn_nfzs[0].center_enu = EnuPoint{-575.0, 475.0, 0.0};
    st.dyn_nfzs[0].radius_m = 200.0f;
    st.dyn_nfzs[0].floor_amsl_m = 180.0f;
    st.dyn_nfzs[0].ceil_amsl_m = 350.0f;
    st.dyn_nfzs[0].active = true;
    emit_case("elevated_nfz_under_floor_sat", st, make_slz_mission(st, "Kartal_Sahil"), polys);
  }

  // 6. AMS speed violation -> UNSAT (c_ams_speed)
  {
    auto m = make_slz_mission(base_st, "Kartal_Sahil");
    m.cmd_speed_max_mps = 25.0;
    emit_case("ams_speed_unsat", base_st, m, polys);
  }

  // 7. AMS climb violation -> UNSAT (c_ams_climb)
  {
    auto m = make_slz_mission(base_st, "Kartal_Sahil");
    m.cmd_climb_req_mps = 4.0;
    emit_case("ams_climb_unsat", base_st, m, polys);
  }

  // 8. HOLD_LOITER skipping p_safe/ood_var -> SAT
  {
    CandidateMission m{};
    m.action = ActionKind::HOLD_LOITER;
    m.target_id = "CURRENT_POS";
    m.target_e_m = base_st.pos_enu.east_m;
    m.target_n_m = base_st.pos_enu.north_m;
    m.target_u_m = base_st.pos_enu.up_m;
    m.target_p_safe = 0.0;
    m.target_ood_var = 1.0;
    m.target_eta_s = 30.0;
    m.cmd_agl_min_m = 35.0;
    m.cmd_agl_max_m = 120.0;
    m.cmd_speed_max_mps = 5.0;
    m.cmd_climb_req_mps = 1.0;
    m.cmd_accel_req_mps2 = 1.0;
    m.t1_s = 0.0;
    m.t2_s = 30.0;
    emit_case("hold_loiter_surface_skip_sat", base_st, m, polys);
  }

  // 9. Shield bypass attempt -> UNSAT (c_shield_bypass)
  {
    auto m = make_slz_mission(base_st, "Kartal_Sahil");
    m.requests_shield_disable = true;
    emit_case("shield_bypass_unsat", base_st, m, polys);
  }

  // 10. Yakacik_Intrude -> UNSAT (c_ltfj_ctr, c_ltbx_ctr)
  {
    auto m = make_slz_mission(base_st, "Kartal_Sahil");
    m.target_e_m = 1432.0;
    m.target_n_m = 4331.3;
    m.target_u_m = 96.4;
    emit_case("yakacik_dual_ctr_unsat", base_st, m, polys);
  }

  return 0;
}
