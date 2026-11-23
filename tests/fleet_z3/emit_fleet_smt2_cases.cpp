// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <string>
#include <vector>

#include "vertirta/fleet_z3/fleet_z3_scheduler.hpp"

namespace {

using namespace vertirta::fleet_z3;

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

void emit_row(const std::string& name, const FleetScheduleResult& res) {
  std::cout << "{\"name\":\"" << json_escape(name) << "\",\"logic\":\""
            << json_escape(res.smt2_logic) << "\",\"cpp_is_sat\":"
            << (res.sat ? "true" : "false") << ",\"cpp_unsat_core\":[";
  for (std::size_t i = 0U; i < res.unsat_core_labels.size(); ++i) {
    if (i > 0U) {
      std::cout << ',';
    }
    std::cout << '"' << json_escape(res.unsat_core_labels[i]) << '"';
  }
  std::cout << "],\"smt2\":\"" << json_escape(res.smt2_audit) << "\"}\n";
}

}  // namespace

int main() {
  FleetZ3Scheduler sched{};

  // 1. Tier-1 QF_RDL STN SAT: 20-aircraft nominal Anatolian corridor timetable
  emit_row("tier1_qf_rdl_20_aircraft_sat",
           sched.check_tier1_stn(make_nominal_20_aircraft_request()));

  // 2. Tier-1 QF_RDL STN UNSAT: battery reserve negative cycle
  emit_row("tier1_qf_rdl_battery_cycle_unsat",
           sched.check_tier1_stn(make_tier1_unsat_battery_cycle_request()));

  // 3. Tier-2 QF_LRA UNSAT (pre-repair): Art. 10(9) new nominal clashing with approved U-plan
  emit_row("tier2_qf_lra_art10_9_clash_unsat",
           sched.check_tier2_local_unrepaired(make_nominal_slot_clash_request()));

  // 4. Tier-2 QF_LRA SAT (repaired): Art. 10(9) approved plan preserved, new nominal shifted >= 60 s
  emit_row("tier2_qf_lra_art10_9_repaired_sat",
           sched.schedule(make_nominal_slot_clash_request()));

  // 5. Tier-2 QF_LRA UNSAT (pre-repair): < 60 m dependent FATO downwash mutex + LTFJ 150 s wake conflict
  emit_row("tier2_qf_lra_mayday_dependent_fato_clash_unsat",
           sched.check_tier2_local_unrepaired(make_mayday_dependent_fato_request(45.0, 150.0)));

  // 6. Tier-2 QF_LRA SAT (repaired): Art. 10(10) MAYDAY preempts nominal + < 60 m FATO mutex + 150 s LTFJ wake
  emit_row("tier2_qf_lra_mayday_dependent_fato_repaired_sat",
           sched.schedule(make_mayday_dependent_fato_request(45.0, 150.0)));

  // 7. Tier-2 QF_LRA UNSAT: DLR EUREKA 2026 missing ATS clearance inside LTFJ controlled corridor
  emit_row("tier2_qf_lra_missing_ats_clearance_unsat",
           sched.schedule(make_missing_ats_clearance_request()));

  // 8. Tier-2 QF_LRA SAT: Art. 10(8) EMS/SAR special operations preemption on shared FATO
  emit_row("tier2_qf_lra_art10_8_ems_sar_repaired_sat",
           sched.schedule(make_ems_sar_preemption_request()));

  return 0;
}
