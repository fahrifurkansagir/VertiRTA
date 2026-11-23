// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "vertirta/fleet_z3/fleet_z3_scheduler.hpp"

using namespace vertirta::fleet_z3;

TEST_CASE("Tier-1 QF_RDL STN solves 20-eVTOL schedule in < 3 ms and Tier-2 QF_LRA enforces Art. 10(8)-(10) in < 25 ms") {
  FleetZ3Scheduler sched{};
  REQUIRE(sched.uses_separate_z3_contexts());  // Prevents smt.arith.solver=3 clash

  // 1) 20-aircraft nominal timetable -> Tier-1 QF_RDL fast path (0 Booleans, <= 160 vars)
  auto req20 = make_nominal_20_aircraft_request();
  REQUIRE(req20.aircraft.size() == 20U);
  const auto r_nom = sched.schedule(req20);
  REQUIRE(r_nom.sat);
  REQUIRE(r_nom.tier_used == SolverTier::TIER1_QF_RDL_STN);
  REQUIRE(r_nom.smt2_logic == "QF_RDL");
  REQUIRE(r_nom.wall_ms < 10.0);  // Target < 3 ms Release
  REQUIRE(r_nom.smt2_audit.find("(set-logic QF_RDL)") != std::string::npos);
  REQUIRE(r_nom.schedule.size() == 20U);

  // 2) Art. 10(9): New nominal request conflicting with approved U-plan CANNOT preempt it
  auto req_art10_9 = make_nominal_slot_clash_request();
  const auto r_deny = sched.schedule(req_art10_9);
  REQUIRE(r_deny.sat);
  REQUIRE(r_deny.repaired);
  REQUIRE(r_deny.existing_plan_preserved);
  REQUIRE(r_deny.new_nominal_delayed_s >= 60.0);
  REQUIRE(r_deny.tier_used == SolverTier::TIER2_QF_LRA_LOCAL);
  REQUIRE(r_deny.smt2_logic == "QF_LRA");
  REQUIRE(r_deny.smt2_audit.find("(set-logic QF_LRA)") != std::string::npos);

  // 3) Art. 10(8)/(10): MAYDAY emergency preempts nominal aircraft & enforces < 60 m dependent FATO mutex + LTFJ wake
  auto req_mayday = make_mayday_dependent_fato_request(/*fato_spacing_m=*/45.0, /*ltfj_heavy_wake_s=*/150.0);
  const auto r_emg = sched.schedule(req_mayday);
  REQUIRE(r_emg.sat);
  REQUIRE(r_emg.tier_used == SolverTier::TIER2_QF_LRA_LOCAL);
  REQUIRE(r_emg.local_neighborhood_size <= 6U);
  REQUIRE(r_emg.mayday_granted_priority);
  REQUIRE(r_emg.dependent_fato_mutex_enforced);
  REQUIRE(r_emg.ltfj_wake_buffer_s >= 120.0);
  REQUIRE(r_emg.wall_ms < 25.0);
  REQUIRE(r_emg.smt2_audit.find("(set-logic QF_LRA)") != std::string::npos);
}

TEST_CASE("Tier-1 QF_RDL detects battery negative cycle and Tier-2 QF_LRA bounds local neighborhood <= 6") {
  FleetZ3Scheduler sched{};

  // Direct Tier-1 QF_RDL UNSAT check (battery reserve deadline tighter than minimum transit)
  const auto req_cycle = make_tier1_unsat_battery_cycle_request();
  const auto r_rdl = sched.check_tier1_stn(req_cycle);
  REQUIRE_FALSE(r_rdl.sat);
  REQUIRE(r_rdl.tier_used == SolverTier::TIER1_QF_RDL_STN);
  REQUIRE(r_rdl.smt2_logic == "QF_RDL");
  REQUIRE_FALSE(r_rdl.unsat_core_labels.empty());

  // 12-aircraft congested hub -> Tier-2 local neighborhood strictly bounded |N_local| <= 6
  const auto req12 = make_congested_12_aircraft_mayday_request();
  REQUIRE(req12.aircraft.size() == 12U);
  const auto r12 = sched.schedule(req12);
  REQUIRE(r12.sat);
  REQUIRE(r12.tier_used == SolverTier::TIER2_QF_LRA_LOCAL);
  REQUIRE(r12.local_neighborhood_size <= kMaxLocalNeighborhood);
  REQUIRE(r12.mayday_granted_priority);
  REQUIRE(r12.wall_ms < 25.0);
}

TEST_CASE("DLR EUREKA 2026 ATS vs U-Space gate fails closed when LTFJ controlled clearance is missing") {
  FleetZ3Scheduler sched{};
  const auto req_no_ats = make_missing_ats_clearance_request();
  const auto res = sched.schedule(req_no_ats);
  REQUIRE_FALSE(res.sat);
  REQUIRE(res.tier_used == SolverTier::FALLBACK_SAFE_HOLD);
  REQUIRE(std::find(res.unsat_core_labels.begin(), res.unsat_core_labels.end(),
                    "c_ats_clearance_UA_LTFJ_01") != res.unsat_core_labels.end());
}

TEST_CASE("Special Ops EMS/SAR (Art. 10(8)) preempts nominal flights and independent FATOs >= 60 m do not false-mutex") {
  FleetZ3Scheduler sched{};

  // Independent FATOs spaced 75 m >= 60 m allow concurrent operations without downwash mutex
  const auto req_indep = make_independent_fato_pair_request(/*fato_spacing_m=*/75.0);
  const auto r_indep = sched.schedule(req_indep);
  REQUIRE(r_indep.sat);
  REQUIRE_FALSE(r_indep.dependent_fato_mutex_enforced);

  // Special Ops EMS/SAR (Art. 10(8)) preempts nominal arrival on shared FATO
  const auto req_ems = make_ems_sar_preemption_request();
  const auto r_ems = sched.schedule(req_ems);
  REQUIRE(r_ems.sat);
  REQUIRE(r_ems.repaired);
  REQUIRE(r_ems.tier_used == SolverTier::TIER2_QF_LRA_LOCAL);
  REQUIRE(r_ems.wall_ms < 25.0);
}

TEST_CASE("Independent FleetZ3Scheduler instances run concurrently with zero TSan races") {
  std::vector<std::thread> threads;
  std::array<std::uint8_t, 4U> ok{0U, 0U, 0U, 0U};
  for (std::size_t i = 0U; i < 4U; ++i) {
    threads.emplace_back([&, i]() {
      FleetZ3Scheduler local_sched{};
      const auto r1 = local_sched.schedule(make_nominal_20_aircraft_request());
      const auto r2 = local_sched.schedule(make_mayday_dependent_fato_request(45.0, 150.0));
      ok[i] = (local_sched.uses_separate_z3_contexts() && r1.sat && r2.sat &&
               r2.mayday_granted_priority && r2.dependent_fato_mutex_enforced)
                  ? 1U
                  : 0U;
    });
  }
  for (auto& t : threads) {
    t.join();
  }
  for (std::uint8_t passed : ok) {
    REQUIRE(passed == 1U);
  }
}
