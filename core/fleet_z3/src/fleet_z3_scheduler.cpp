// SPDX-License-Identifier: Apache-2.0
#include "vertirta/fleet_z3/fleet_z3_scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

namespace vertirta::fleet_z3 {

namespace {

[[nodiscard]] bool needs_tier2_local_solver(const FleetScheduleRequest& req) noexcept {
  const std::size_t n = std::min(req.aircraft.size(), kMaxFleetAircraft);
  for (std::size_t i = 0U; i < n; ++i) {
    const auto& a = req.aircraft[i];
    if (a.priority == PriorityClass::EMERGENCY_MAYDAY ||
        a.priority == PriorityClass::SPECIAL_OPS_EMS_SAR) {
      return true;
    }
    if (a.crosses_ltfj_wake_zone && !a.ats_clearance_granted) {
      return true;
    }
  }
  // Check for nominal slot clashes (Art. 10(9)) or < 60 m dependent FATO overlaps.
  for (std::size_t i = 0U; i < n; ++i) {
    for (std::size_t j = i + 1U; j < n; ++j) {
      const auto& ai = req.aircraft[i];
      const auto& aj = req.aircraft[j];
      if (!shares_fato_mutex(ai, aj, req.dependent_fato_spacing_m)) {
        continue;
      }
      const double ai_arr = ai.nominal_dep_s + ai.min_transit_s;
      const double aj_arr = aj.nominal_dep_s + aj.min_transit_s;
      const double sep_req =
          std::max({ai.fato_dwell_s, aj.fato_dwell_s, req.siso_wake_spacing_s});
      if (std::abs(ai_arr - aj_arr) < sep_req - 1e-6) {
        return true;
      }
      if (ai.priority != aj.priority) {
        return true;
      }
    }
  }
  return false;
}

[[nodiscard]] std::vector<std::size_t> select_bounded_local_neighborhood(
    const FleetScheduleRequest& req) {
  const std::size_t n = std::min(req.aircraft.size(), kMaxFleetAircraft);
  std::vector<std::size_t> selected;
  selected.reserve(kMaxLocalNeighborhood);

  auto add_unique = [&](std::size_t idx) {
    if (selected.size() >= kMaxLocalNeighborhood) {
      return;
    }
    if (std::find(selected.begin(), selected.end(), idx) == selected.end()) {
      selected.push_back(idx);
    }
  };

  // 1. Always pin EMERGENCY_MAYDAY, SPECIAL_OPS_EMS_SAR, and missing-ATS aircraft first.
  for (std::size_t i = 0U; i < n; ++i) {
    const auto& a = req.aircraft[i];
    if (a.priority == PriorityClass::EMERGENCY_MAYDAY ||
        a.priority == PriorityClass::SPECIAL_OPS_EMS_SAR ||
        (a.crosses_ltfj_wake_zone && !a.ats_clearance_granted)) {
      add_unique(i);
    }
  }

  // 2. Add conflicting FATO / < 60 m dependent-FATO peers interacting with pinned aircraft or each other.
  std::vector<std::size_t> conflict_peers;
  for (std::size_t i = 0U; i < n; ++i) {
    for (std::size_t j = i + 1U; j < n; ++j) {
      const auto& ai = req.aircraft[i];
      const auto& aj = req.aircraft[j];
      if (!shares_fato_mutex(ai, aj, req.dependent_fato_spacing_m)) {
        continue;
      }
      const double ai_arr = ai.nominal_dep_s + ai.min_transit_s;
      const double aj_arr = aj.nominal_dep_s + aj.min_transit_s;
      const double sep_req =
          std::max({ai.fato_dwell_s, aj.fato_dwell_s, req.siso_wake_spacing_s});
      if (std::abs(ai_arr - aj_arr) < sep_req * 2.0) {
        conflict_peers.push_back(i);
        conflict_peers.push_back(j);
      }
    }
  }

  // Sort candidate peers by priority then proximity to pinned emergency arrival time.
  double anchor_arr = 0.0;
  if (!selected.empty()) {
    const auto& anchor = req.aircraft[selected.front()];
    anchor_arr = anchor.nominal_dep_s + anchor.min_transit_s;
  }
  std::sort(conflict_peers.begin(), conflict_peers.end(), [&](std::size_t l, std::size_t r) {
    const auto& al = req.aircraft[l];
    const auto& ar = req.aircraft[r];
    if (al.priority != ar.priority) {
      return static_cast<std::uint8_t>(al.priority) < static_cast<std::uint8_t>(ar.priority);
    }
    const double dl = std::abs((al.nominal_dep_s + al.min_transit_s) - anchor_arr);
    const double dr = std::abs((ar.nominal_dep_s + ar.min_transit_s) - anchor_arr);
    if (dl != dr) {
      return dl < dr;
    }
    return al.aircraft_id < ar.aircraft_id;
  });

  for (std::size_t idx : conflict_peers) {
    add_unique(idx);
  }

  // 3. If still empty (e.g. single-aircraft wake/battery conflict), take up to kMaxLocalNeighborhood.
  for (std::size_t i = 0U; i < n && selected.size() < kMaxLocalNeighborhood; ++i) {
    if (selected.empty() || req.aircraft[i].crosses_ltfj_wake_zone) {
      add_unique(i);
    }
  }
  if (selected.empty()) {
    for (std::size_t i = 0U; i < std::min(n, kMaxLocalNeighborhood); ++i) {
      add_unique(i);
    }
  }

  return selected;
}

}  // namespace

bool shares_fato_mutex(const AircraftSlotRequest& a,
                       const AircraftSlotRequest& b,
                       double default_spacing_m) noexcept {
  if (a.fato_id == b.fato_id) {
    return true;
  }
  if (!a.dependent_group_id.empty() && a.dependent_group_id == b.dependent_group_id) {
    const double eff_spacing =
        std::min({a.fato_spacing_m, b.fato_spacing_m, default_spacing_m});
    return eff_spacing < kDependentFatoSpacingThresholdM;
  }
  return false;
}

double effective_ltfj_wake_s(const AircraftSlotRequest& a,
                             double request_default_wake_s) noexcept {
  if (!a.crosses_ltfj_wake_zone) {
    return 0.0;
  }
  const double raw = std::max(a.ltfj_wake_buffer_s, request_default_wake_s);
  return std::clamp(raw, kMinLtfjHeavyWakeBufferS, kMaxLtfjHeavyWakeBufferS);
}

FleetZ3Scheduler::FleetZ3Scheduler() : rdl_solver_(), lra_solver_() {}

bool FleetZ3Scheduler::uses_separate_z3_contexts() const noexcept {
  const std::uintptr_t t1 = rdl_solver_.context_tag();
  const std::uintptr_t t2 = lra_solver_.context_tag();
  return (t1 != 0U) && (t2 != 0U) && (t1 != t2);
}

SolverBackend FleetZ3Scheduler::backend() const noexcept { return rdl_solver_.backend(); }

FleetScheduleResult FleetZ3Scheduler::check_tier1_stn(const FleetScheduleRequest& req) const {
  return rdl_solver_.solve(req);
}

FleetScheduleResult FleetZ3Scheduler::check_tier2_local_unrepaired(
    const FleetScheduleRequest& req) const {
  const auto local_idx = select_bounded_local_neighborhood(req);
  return lra_solver_.check_unrepaired(req, local_idx);
}

FleetScheduleResult FleetZ3Scheduler::schedule(const FleetScheduleRequest& req) const {
  const auto t_start = std::chrono::steady_clock::now();

  // Fast Path: Tier-1 QF_RDL STN (0 Booleans, <= 160 vars, < 3 ms) for nominal timetables.
  if (!needs_tier2_local_solver(req)) {
    auto r_stn = rdl_solver_.solve(req);
    if (r_stn.sat) {
      return r_stn;
    }
  }

  // Conflict / Emergency / Preemption Path: Tier-2 bounded local QF_LRA (|N_local| <= 6, try-for 18 ms).
  const auto local_idx = select_bounded_local_neighborhood(req);
  auto r_lra = lra_solver_.solve_repaired(req, local_idx);
  const auto t_end = std::chrono::steady_clock::now();
  r_lra.wall_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
  return r_lra;
}

// --- Canonical Istanbul Anatolian Coastal Corridor Scenarios (Frozen < 29.1895 E) ---

FleetScheduleRequest make_nominal_20_aircraft_request() {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.dependent_fato_spacing_m = 80.0;
  req.emit_smt2 = true;

  static constexpr const char* kFatos[4] = {
      "VPT-MLT-F1", "VPT-DRG-F1", "VPT-CVZ-F1", "VPT-KRT-F1"};

  req.aircraft.reserve(20U);
  for (std::size_t i = 0U; i < 20U; ++i) {
    const std::size_t fato_idx = i % 4U;
    const std::size_t wave = i / 4U;  // 5 staggered waves per vertiport, 75 s apart >= 60 s SISO
    AircraftSlotRequest a{};
    a.aircraft_id = "UA_" + std::string(i < 9U ? "0" : "") + std::to_string(i + 1U);
    a.fato_id = kFatos[fato_idx];
    a.fato_spacing_m = 80.0;
    a.priority = PriorityClass::APPROVED_NOMINAL;
    a.earliest_dep_s = static_cast<double>(wave) * 75.0;
    a.nominal_dep_s = a.earliest_dep_s;
    a.min_transit_s = 180.0;
    a.max_transit_s = 260.0;
    a.fato_dwell_s = 60.0;
    a.battery_max_arr_s = 900.0;
    a.is_existing_approved = true;
    req.aircraft.push_back(std::move(a));
  }
  return req;
}

FleetScheduleRequest make_nominal_slot_clash_request() {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.dependent_fato_spacing_m = 80.0;
  req.emit_smt2 = true;

  // Existing approved U-plan at Kartal FATO-1 arriving t = 240 s
  AircraftSlotRequest approved{};
  approved.aircraft_id = "UA_APP_01";
  approved.fato_id = "VPT-KRT-F1";
  approved.priority = PriorityClass::APPROVED_NOMINAL;
  approved.earliest_dep_s = 60.0;
  approved.nominal_dep_s = 60.0;
  approved.min_transit_s = 180.0;
  approved.max_transit_s = 300.0;
  approved.fato_dwell_s = 60.0;
  approved.battery_max_arr_s = 900.0;
  approved.is_existing_approved = true;

  // New nominal entrant requesting overlapping arrival at t = 240 s (Art. 10(9) must yield >= 60 s)
  AircraftSlotRequest entrant{};
  entrant.aircraft_id = "UA_NEW_02";
  entrant.fato_id = "VPT-KRT-F1";
  entrant.priority = PriorityClass::NEW_NOMINAL;
  entrant.earliest_dep_s = 60.0;
  entrant.nominal_dep_s = 60.0;
  entrant.min_transit_s = 180.0;
  entrant.max_transit_s = 360.0;
  entrant.fato_dwell_s = 60.0;
  entrant.battery_max_arr_s = 900.0;
  entrant.is_existing_approved = false;

  req.aircraft.push_back(std::move(approved));
  req.aircraft.push_back(std::move(entrant));
  return req;
}

FleetScheduleRequest make_mayday_dependent_fato_request(double fato_spacing_m,
                                                        double ltfj_heavy_wake_s) {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.dependent_fato_spacing_m = fato_spacing_m;
  req.ltfj_heavy_wake_s = ltfj_heavy_wake_s;
  req.emit_smt2 = true;

  // Approved nominal arriving on Kartal FATO-1 (adjacent 45 m < 60 m dependent downwash group)
  AircraftSlotRequest nom{};
  nom.aircraft_id = "UA_NOM_01";
  nom.fato_id = "VPT-KRT-F1";
  nom.dependent_group_id = "VPT-KRT-DEP";
  nom.fato_spacing_m = fato_spacing_m;
  nom.priority = PriorityClass::APPROVED_NOMINAL;
  nom.earliest_dep_s = 60.0;
  nom.nominal_dep_s = 60.0;
  nom.min_transit_s = 180.0;
  nom.max_transit_s = 360.0;
  nom.fato_dwell_s = 60.0;
  nom.battery_max_arr_s = 900.0;
  nom.is_existing_approved = true;

  // MAYDAY emergency (Art. 10(10)) diverting to Kartal FATO-EMG (< 60 m from FATO-1) crossing LTFJ wake
  AircraftSlotRequest mayday{};
  mayday.aircraft_id = "UA_MAYDAY_01";
  mayday.fato_id = "VPT-KRT-EMG";
  mayday.dependent_group_id = "VPT-KRT-DEP";
  mayday.fato_spacing_m = fato_spacing_m;
  mayday.priority = PriorityClass::EMERGENCY_MAYDAY;
  mayday.earliest_dep_s = 40.0;
  mayday.nominal_dep_s = 40.0;  // Unrepaired arrival 220 s clashes with LTFJ wake (250 s) & nom (240 s)
  mayday.min_transit_s = 180.0;
  mayday.max_transit_s = 300.0;
  mayday.fato_dwell_s = 60.0;
  mayday.battery_max_arr_s = 600.0;
  mayday.is_existing_approved = false;
  mayday.crosses_ltfj_wake_zone = true;
  mayday.ltfj_heavy_arrival_s = 100.0;
  mayday.ltfj_wake_buffer_s = ltfj_heavy_wake_s;
  mayday.ats_clearance_granted = true;

  req.aircraft.push_back(std::move(nom));
  req.aircraft.push_back(std::move(mayday));
  return req;
}

FleetScheduleRequest make_tier1_unsat_battery_cycle_request() {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.emit_smt2 = true;

  AircraftSlotRequest a{};
  a.aircraft_id = "UA_LOWBATT_01";
  a.fato_id = "VPT-CVZ-F1";
  a.priority = PriorityClass::APPROVED_NOMINAL;
  a.earliest_dep_s = 100.0;
  a.nominal_dep_s = 100.0;
  a.min_transit_s = 200.0;
  a.max_transit_s = 300.0;
  a.battery_max_arr_s = 250.0;  // Impossible: earliest arrival = 300 s > 250 s battery reserve
  a.is_existing_approved = true;
  req.aircraft.push_back(std::move(a));
  return req;
}

FleetScheduleRequest make_congested_12_aircraft_mayday_request() {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.dependent_fato_spacing_m = 45.0;
  req.ltfj_heavy_wake_s = 150.0;
  req.emit_smt2 = true;

  AircraftSlotRequest mayday{};
  mayday.aircraft_id = "UA_MAYDAY_99";
  mayday.fato_id = "VPT-KRT-EMG";
  mayday.dependent_group_id = "VPT-KRT-DEP";
  mayday.fato_spacing_m = 45.0;
  mayday.priority = PriorityClass::EMERGENCY_MAYDAY;
  mayday.earliest_dep_s = 40.0;
  mayday.nominal_dep_s = 40.0;
  mayday.min_transit_s = 180.0;
  mayday.max_transit_s = 300.0;
  mayday.fato_dwell_s = 60.0;
  mayday.battery_max_arr_s = 700.0;
  mayday.is_existing_approved = false;
  mayday.crosses_ltfj_wake_zone = true;
  mayday.ltfj_heavy_arrival_s = 90.0;
  mayday.ltfj_wake_buffer_s = 150.0;
  mayday.ats_clearance_granted = true;
  req.aircraft.push_back(std::move(mayday));

  for (std::size_t i = 0U; i < 11U; ++i) {
    AircraftSlotRequest a{};
    a.aircraft_id = "UA_CONG_" + std::string(i < 9U ? "0" : "") + std::to_string(i + 1U);
    a.fato_id = (i < 6U) ? "VPT-KRT-F1" : "VPT-DRG-F1";
    a.dependent_group_id = (i < 6U) ? "VPT-KRT-DEP" : "";
    a.fato_spacing_m = (i < 6U) ? 45.0 : 90.0;
    a.priority = PriorityClass::APPROVED_NOMINAL;
    a.earliest_dep_s = 45.0 + static_cast<double>(i) * 15.0;
    a.nominal_dep_s = a.earliest_dep_s;
    a.min_transit_s = 180.0;
    a.max_transit_s = 600.0;
    a.fato_dwell_s = 60.0;
    a.battery_max_arr_s = 1200.0;
    a.is_existing_approved = true;
    req.aircraft.push_back(std::move(a));
  }
  return req;
}

FleetScheduleRequest make_missing_ats_clearance_request() {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.ltfj_heavy_wake_s = 150.0;
  req.emit_smt2 = true;

  AircraftSlotRequest a{};
  a.aircraft_id = "UA_LTFJ_01";
  a.fato_id = "VPT-KRT-F1";
  a.priority = PriorityClass::APPROVED_NOMINAL;
  a.earliest_dep_s = 100.0;
  a.nominal_dep_s = 100.0;
  a.min_transit_s = 180.0;
  a.max_transit_s = 300.0;
  a.battery_max_arr_s = 900.0;
  a.crosses_ltfj_wake_zone = true;
  a.ltfj_heavy_arrival_s = 100.0;
  a.ltfj_wake_buffer_s = 150.0;
  a.ats_clearance_granted = false;  // DLR EUREKA 2026: USSP authorization cannot bypass ATS
  req.aircraft.push_back(std::move(a));
  return req;
}

FleetScheduleRequest make_independent_fato_pair_request(double fato_spacing_m) {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.dependent_fato_spacing_m = fato_spacing_m;
  req.emit_smt2 = true;

  for (std::size_t i = 0U; i < 2U; ++i) {
    AircraftSlotRequest a{};
    a.aircraft_id = "UA_IND_0" + std::to_string(i + 1U);
    a.fato_id = (i == 0U) ? "VPT-MLT-F1" : "VPT-MLT-F2";
    a.dependent_group_id = "VPT-MLT-PAIR";
    a.fato_spacing_m = fato_spacing_m;  // >= 60 m => independent concurrent FATO operations allowed
    a.priority = PriorityClass::APPROVED_NOMINAL;
    a.earliest_dep_s = 60.0;
    a.nominal_dep_s = 60.0;
    a.min_transit_s = 180.0;
    a.max_transit_s = 300.0;
    a.fato_dwell_s = 60.0;
    a.battery_max_arr_s = 900.0;
    a.is_existing_approved = true;
    req.aircraft.push_back(std::move(a));
  }
  return req;
}

FleetScheduleRequest make_ems_sar_preemption_request() {
  FleetScheduleRequest req{};
  req.siso_wake_spacing_s = 60.0;
  req.emit_smt2 = true;

  AircraftSlotRequest nom{};
  nom.aircraft_id = "UA_NOM_10";
  nom.fato_id = "VPT-DRG-F1";
  nom.priority = PriorityClass::APPROVED_NOMINAL;
  nom.earliest_dep_s = 60.0;
  nom.nominal_dep_s = 60.0;
  nom.min_transit_s = 180.0;
  nom.max_transit_s = 360.0;
  nom.fato_dwell_s = 60.0;
  nom.battery_max_arr_s = 900.0;
  nom.is_existing_approved = true;

  // Regulation (EU) 2021/664 Art. 10(8) Special Ops EMS/SAR preempting nominal flight
  AircraftSlotRequest ems{};
  ems.aircraft_id = "UA_EMS_01";
  ems.fato_id = "VPT-DRG-F1";
  ems.priority = PriorityClass::SPECIAL_OPS_EMS_SAR;
  ems.earliest_dep_s = 60.0;
  ems.nominal_dep_s = 60.0;
  ems.min_transit_s = 180.0;
  ems.max_transit_s = 300.0;
  ems.fato_dwell_s = 60.0;
  ems.battery_max_arr_s = 700.0;
  ems.is_existing_approved = false;

  req.aircraft.push_back(std::move(nom));
  req.aircraft.push_back(std::move(ems));
  return req;
}

}  // namespace vertirta::fleet_z3
