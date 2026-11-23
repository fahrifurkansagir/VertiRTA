// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vertirta::fleet_z3 {

inline constexpr std::size_t kMaxFleetAircraft = 20U;
inline constexpr std::size_t kMaxLocalNeighborhood = 6U;
inline constexpr double kDependentFatoSpacingThresholdM = 60.0;  // EASA/UK CAA PTS-VPT-DSN
inline constexpr double kMinLtfjHeavyWakeBufferS = 120.0;        // Heavy fixed-wing wake floor
inline constexpr double kMaxLtfjHeavyWakeBufferS = 180.0;        // Heavy fixed-wing wake ceiling
inline constexpr double kDefaultSisoWakeSpacingS = 60.0;         // Calm/nominal SISO wake spacing
inline constexpr std::uint32_t kTier2TimeoutMs = 18U;            // try-for 18 ms (< 25 ms p99)

enum class SolverTier : std::uint8_t {
  TIER1_QF_RDL_STN = 1U,
  TIER2_QF_LRA_LOCAL = 2U,
  FALLBACK_SAFE_HOLD = 3U,
};

// Regulation (EU) 2021/664 Art. 10(8) / 10(9) / 10(10) strategic priority ordering.
enum class PriorityClass : std::uint8_t {
  EMERGENCY_MAYDAY = 0U,    // Art. 10(10): absolute preemption
  SPECIAL_OPS_EMS_SAR = 1U, // Art. 10(8): EMS/SAR special operations preemption
  APPROVED_NOMINAL = 2U,    // Art. 10(9): existing approved U-plan (cannot be preempted by new nominal)
  NEW_NOMINAL = 3U,         // Art. 10(9): new nominal entrant (must yield to existing approved plans)
};

enum class SolverBackend : std::uint8_t {
  NATIVE_Z3_DUAL_CONTEXT = 1U,
  DETERMINISTIC_EXACT_FALLBACK = 2U,
};

struct AircraftSlotRequest {
  std::string aircraft_id{};
  std::string fato_id{"VPT-KRT-F1"};
  std::string dependent_group_id{};  // Non-empty + fato_spacing_m < 60 m => shared downwash mutex
  double fato_spacing_m{80.0};
  PriorityClass priority{PriorityClass::APPROVED_NOMINAL};
  double earliest_dep_s{0.0};
  double nominal_dep_s{0.0};
  double min_transit_s{180.0};
  double max_transit_s{300.0};
  double fato_dwell_s{60.0};         // EASA PTS-VPT-DSN.C.200 FATO exclusivity + wake dwell
  double battery_max_arr_s{900.0};   // Latest arrival respecting 20%/12% reserve floor
  bool is_existing_approved{true};
  bool crosses_ltfj_wake_zone{false};
  double ltfj_heavy_arrival_s{-1000.0};
  double ltfj_wake_buffer_s{0.0};    // Enforced >= 120 s when crosses_ltfj_wake_zone is true
  bool ats_clearance_granted{true};  // DLR EUREKA 2026: USSP authorization != ATS clearance
};

struct FleetScheduleRequest {
  std::vector<AircraftSlotRequest> aircraft{};
  double siso_wake_spacing_s{kDefaultSisoWakeSpacingS};
  double dependent_fato_spacing_m{80.0};
  double ltfj_heavy_wake_s{0.0};
  bool emit_smt2{true};
};

struct ScheduledAircraftSlot {
  std::string aircraft_id{};
  std::string fato_id{};
  double assigned_dep_s{0.0};
  double assigned_arr_s{0.0};
  double delay_s{0.0};
  bool preempted_or_shifted{false};
};

struct FleetScheduleResult {
  bool sat{false};
  bool repaired{false};
  bool timed_out{false};
  SolverTier tier_used{SolverTier::TIER1_QF_RDL_STN};
  SolverBackend backend_used{SolverBackend::DETERMINISTIC_EXACT_FALLBACK};
  std::size_t local_neighborhood_size{0U};
  bool existing_plan_preserved{true};
  double new_nominal_delayed_s{0.0};
  bool mayday_granted_priority{false};
  bool dependent_fato_mutex_enforced{false};
  double ltfj_wake_buffer_s{0.0};
  double wall_ms{0.0};
  std::vector<ScheduledAircraftSlot> schedule{};
  std::vector<std::string> unsat_core_labels{};
  std::string smt2_audit{};
  std::string smt2_logic{"QF_RDL"};
};

// Tier-1 dedicated QF_RDL Simple Temporal Network solver owning its own isolated Z3 context.
class QfRdlStnSolver {
 public:
  QfRdlStnSolver();
  ~QfRdlStnSolver();
  QfRdlStnSolver(const QfRdlStnSolver&) = delete;
  QfRdlStnSolver& operator=(const QfRdlStnSolver&) = delete;
  QfRdlStnSolver(QfRdlStnSolver&&) noexcept;
  QfRdlStnSolver& operator=(QfRdlStnSolver&&) noexcept;

  [[nodiscard]] std::uintptr_t context_tag() const noexcept;
  [[nodiscard]] SolverBackend backend() const noexcept;
  [[nodiscard]] FleetScheduleResult solve(const FleetScheduleRequest& req) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Tier-2 dedicated bounded local QF_LRA solver (|N_local| <= 6) owning a strictly separate Z3 context.
class QfLraLocalSolver {
 public:
  QfLraLocalSolver();
  ~QfLraLocalSolver();
  QfLraLocalSolver(const QfLraLocalSolver&) = delete;
  QfLraLocalSolver& operator=(const QfLraLocalSolver&) = delete;
  QfLraLocalSolver(QfLraLocalSolver&&) noexcept;
  QfLraLocalSolver& operator=(QfLraLocalSolver&&) noexcept;

  [[nodiscard]] std::uintptr_t context_tag() const noexcept;
  [[nodiscard]] SolverBackend backend() const noexcept;

  // Evaluates local neighborhood with nominal slots frozen (detects conflicts & extracts UNSAT core).
  [[nodiscard]] FleetScheduleResult check_unrepaired(
      const FleetScheduleRequest& req,
      const std::vector<std::size_t>& local_indices) const;

  // Solves local neighborhood with 1-pass Art. 10(8)/(9)/(10) repair & verifies SAT/UNSAT in QF_LRA.
  [[nodiscard]] FleetScheduleResult solve_repaired(
      const FleetScheduleRequest& req,
      const std::vector<std::size_t>& local_indices) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Orchestrates Tier-1 QF_RDL STN fast path (< 3 ms) -> Tier-2 bounded local QF_LRA (< 25 ms p99).
// Permanently bans global fleet STL-MILP; maintains two strictly separate Z3 contexts.
class FleetZ3Scheduler {
 public:
  FleetZ3Scheduler();
  ~FleetZ3Scheduler() = default;
  FleetZ3Scheduler(const FleetZ3Scheduler&) = delete;
  FleetZ3Scheduler& operator=(const FleetZ3Scheduler&) = delete;
  FleetZ3Scheduler(FleetZ3Scheduler&&) noexcept = default;
  FleetZ3Scheduler& operator=(FleetZ3Scheduler&&) noexcept = default;

  [[nodiscard]] bool uses_separate_z3_contexts() const noexcept;
  [[nodiscard]] SolverBackend backend() const noexcept;

  [[nodiscard]] FleetScheduleResult schedule(const FleetScheduleRequest& req) const;
  [[nodiscard]] FleetScheduleResult check_tier1_stn(const FleetScheduleRequest& req) const;
  [[nodiscard]] FleetScheduleResult check_tier2_local_unrepaired(
      const FleetScheduleRequest& req) const;

 private:
  QfRdlStnSolver rdl_solver_;
  QfLraLocalSolver lra_solver_;
};

// Shared deterministic helper checks (FATO conflict & rational quantization).
[[nodiscard]] bool shares_fato_mutex(const AircraftSlotRequest& a,
                                     const AircraftSlotRequest& b,
                                     double default_spacing_m) noexcept;
[[nodiscard]] double effective_ltfj_wake_s(const AircraftSlotRequest& a,
                                           double request_default_wake_s) noexcept;

// Canonical Istanbul Anatolian coastal corridor test & benchmark scenario builders.
[[nodiscard]] FleetScheduleRequest make_nominal_20_aircraft_request();
[[nodiscard]] FleetScheduleRequest make_nominal_slot_clash_request();
[[nodiscard]] FleetScheduleRequest make_mayday_dependent_fato_request(
    double fato_spacing_m = 45.0,
    double ltfj_heavy_wake_s = 150.0);
[[nodiscard]] FleetScheduleRequest make_tier1_unsat_battery_cycle_request();
[[nodiscard]] FleetScheduleRequest make_congested_12_aircraft_mayday_request();
[[nodiscard]] FleetScheduleRequest make_missing_ats_clearance_request();
[[nodiscard]] FleetScheduleRequest make_independent_fato_pair_request(double fato_spacing_m = 75.0);
[[nodiscard]] FleetScheduleRequest make_ems_sar_preemption_request();

}  // namespace vertirta::fleet_z3
