// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"

namespace vertirta::uspace {

// Strict separation ellipsoid radii (DR-11 + Plan 5B Lock)
inline constexpr double kNominalSepRxyM = 12.0;
inline constexpr double kNominalSepRzM = 36.0;
inline constexpr double kCelSepRxyM = 25.0;
inline constexpr double kCelSepRzM = 75.0;
inline constexpr std::size_t kMaxCbfOrcaPeers = 10U;
inline constexpr std::size_t kMaxRtaHazardSlots = 14U;  // <= 15 slots in frozen StaticDaqpSolver<5,32>

struct PeerKinematics {
  uint32_t agent_id{0U};
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  bool is_priority0_cel{false};
  bool is_suspect_non_coop{false};
};

struct CbfOrcaCommand {
  core::Vec3Ned accel_cmd_ned{};
  std::array<rta::HazardPoint, kMaxRtaHazardSlots> rta_hazards{};
  uint16_t rta_hazard_count{0U};
  double min_ellipsoidal_ratio{1e6};
  bool curl_active{false};
};

struct AntipodalBenchmarkResult {
  uint32_t deadlock_count{0U};
  double min_ellipsoidal_sep_ratio{1e6};
  double mean_final_goal_err_m{0.0};
  uint16_t max_hazard_rows_fed_to_rta{0U};
};

struct CelPreemptionEncounterResult {
  double escape_onset_ms{0.0};
  double v2v_preemption_latency_ms{0.0};
  double ussp_fato_vacate_latency_ms{0.0};
  double min_separation_m{0.0};
};

// Computes 3D ellipsoidal separation ratio sqrt((dx^2+dy^2)/R_xy^2 + dz^2/R_z^2).
[[nodiscard]] double ellipsoidal_separation_ratio(
    const core::Vec3Ned& p_i,
    const core::Vec3Ned& p_j,
    double r_xy_m,
    double r_z_m) noexcept;

// Computes decentralized 3D ellipsoidal CBF-ORCA acceleration command with:
//   - Anisotropic scaling S = diag(1, 1, R_xy / R_z)
//   - 2nd-order ECBF (k1 = 1.0, k2 = 1.4) + ORCA reciprocity (beta_ij = 0.5 vs 1.0 for P0/suspect)
//   - Risk-gated orthogonal Right-Hand-Rule curl (e_r x z_hat) breaking antipodal deadlocks
//   - std::nth_element Top-10 pruning pinning Priority-0 peers unconditionally (<= 14 <= 15 RTA slots)
[[nodiscard]] CbfOrcaCommand compute_cbf_orca_3d(
    const PeerKinematics& ego,
    const core::Vec3Ned& a_nom_ned,
    std::span<const PeerKinematics> peers,
    std::span<const rta::HazardPoint> static_hazards) noexcept;

// Runs deterministic N-agent symmetric antipodal circle merge verifying zero deadlock and <= 14 RTA slots.
[[nodiscard]] AntipodalBenchmarkResult run_antipodal_merge_benchmark(
    uint32_t n_agents,
    double radius_m,
    double duration_s) noexcept;

// Runs head-on Priority-0 CEL encounter with BEST_EFFORT 3x UDP burst (5 ms spacing) & USSP FATO vacate.
[[nodiscard]] CelPreemptionEncounterResult run_headon_cel_preemption_test(
    double closing_speed_mps,
    double comm_delay_ms) noexcept;

}  // namespace vertirta::uspace
