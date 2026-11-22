#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vertirta/copilot/rta_state_contract.hpp"
#include "vertirta/stl/stl_monitor.hpp"

namespace vertirta::copilot {

struct ConvexPoly2D {
  std::vector<std::array<double, 3>> halfspaces;  // {ax, ay, b} for ax*E + ay*N <= b
  std::vector<std::array<double, 2>> vertices;    // CCW convex hull vertices {vE, vN}
  double floor_enu_u_m{-12.0};
  double ceil_enu_u_m{750.0};
  bool has_elevated_floor{false};
};

struct AirspacePolys {
  ConvexPoly2D ltfj;
  ConvexPoly2D ltbx;
};

struct CandidateMission {
  ActionKind action{ActionKind::DIVERT_SLZ};
  std::string target_id;
  double target_e_m{0.0};
  double target_n_m{0.0};
  double target_u_m{0.0};
  double target_p_safe{0.0};
  double target_ood_var{1.0};
  double target_eta_s{0.0};
  double cmd_agl_min_m{35.0};
  double cmd_agl_max_m{120.0};
  double cmd_speed_max_mps{18.0};
  double cmd_climb_req_mps{1.5};
  double cmd_accel_req_mps2{1.5};
  double t1_s{0.0};
  double t2_s{100.0};
  bool requests_shield_disable{false};
  int active_action_count{1};
};

struct Z3VerifyResult {
  bool is_sat{false};
  bool timed_out{false};
  double wall_ms{0.0};
  std::vector<std::string> unsat_core_labels;
  std::string auditable_smt2;
};

struct RepairDecision {
  bool verified_sat{false};
  bool repaired{false};
  std::string primary_unsat_label;
  std::string fallback_target_id;
  CandidateMission repaired_mission{};
  double wall_ms{0.0};
  std::string explanation_tr;
  std::string explanation_en;
};

// Build exact conservative 150 m-buffered convex hulls for AIP Turkey AD 2 LTFJ (incl. NW J1/J2)
// and AD 2 LTBX in canonical Kartal ENU (Simulation reference only — NOT for navigation).
[[nodiscard]] AirspacePolys load_istanbul_airspace_polys();

[[nodiscard]] ConvexPoly2D make_circum_8gon(double cx, double cy, double r, double floor_enu_u,
                                            double ceil_enu_u, bool has_elevated_floor);

[[nodiscard]] std::vector<stl::HalfSpace2D> to_stl_halfspaces(const ConvexPoly2D& poly);

[[nodiscard]] CandidateMission build_candidate_mission(const CompactDslAst& ast,
                                                       const RtaStateSnapshotV1& st) noexcept;

// Pure QF_LRA static corridor & parameter verifier (<15 ms budget, rlimit=250000, timeout=12 ms).
// Uses native Z3 C++ API when VERTIRTA_HAS_Z3 is defined, and mathematically exact QF_LRA SAT
// evaluation otherwise; always emits auditable SMT-LIB2 ending with (check-sat-assuming (...)).
[[nodiscard]] Z3VerifyResult verify_mission_qflra(const RtaStateSnapshotV1& st,
                                                  const CandidateMission& m,
                                                  const AirspacePolys& polys,
                                                  bool emit_smt2 = true);

// 8-priority deterministic UNSAT-core refusal & 1-pass bounded parameter repair.
[[nodiscard]] RepairDecision resolve_unsat_and_repair(const RtaStateSnapshotV1& st,
                                                      const CandidateMission& m,
                                                      const Z3VerifyResult& res,
                                                      const AirspacePolys& polys);

}  // namespace vertirta::copilot
