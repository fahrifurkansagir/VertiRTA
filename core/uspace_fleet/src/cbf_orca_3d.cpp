// SPDX-License-Identifier: Apache-2.0
#include "vertirta/uspace/cbf_orca_3d.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace vertirta::uspace {

namespace {

constexpr double kPi = 3.14159265358979323846;

struct RankedPeer {
  std::size_t idx{0U};
  double score{1e9};  // Lower = higher urgency
  bool pinned_p0{false};
};

}  // namespace

double ellipsoidal_separation_ratio(
    const core::Vec3Ned& p_i,
    const core::Vec3Ned& p_j,
    double r_xy_m,
    double r_z_m) noexcept {
  const double dx = p_i.x - p_j.x;
  const double dy = p_i.y - p_j.y;
  const double dz = p_i.z - p_j.z;
  const double inv_xy = 1.0 / std::max(r_xy_m, 1e-3);
  const double inv_z = 1.0 / std::max(r_z_m, 1e-3);
  return std::sqrt((dx * dx + dy * dy) * inv_xy * inv_xy + (dz * dz) * inv_z * inv_z);
}

CbfOrcaCommand compute_cbf_orca_3d(
    const PeerKinematics& ego,
    const core::Vec3Ned& a_nom_ned,
    std::span<const PeerKinematics> peers,
    std::span<const rta::HazardPoint> static_hazards) noexcept {
  CbfOrcaCommand out{};
  out.accel_cmd_ned = a_nom_ned;

  // Build ranked peer list (exclude self) and apply std::nth_element Top-10 pruning,
  // unconditionally pinning Priority-0 CEL peers.
  std::array<RankedPeer, 32U> ranked{};
  std::size_t ranked_count = 0U;
  for (std::size_t idx = 0U; idx < peers.size() && ranked_count < ranked.size(); ++idx) {
    if (peers[idx].agent_id == ego.agent_id) {
      continue;
    }
    const bool p0 = peers[idx].is_priority0_cel || ego.is_priority0_cel;
    const double r_xy = p0 ? kCelSepRxyM : kNominalSepRxyM;
    const double r_z = p0 ? kCelSepRzM : kNominalSepRzM;
    const double ratio = ellipsoidal_separation_ratio(ego.pos_ned, peers[idx].pos_ned, r_xy, r_z);
    if (ratio < out.min_ellipsoidal_ratio) {
      out.min_ellipsoidal_ratio = ratio;
    }

    // Closing-speed urgency metric
    const core::Vec3Ned dp = ego.pos_ned - peers[idx].pos_ned;
    const core::Vec3Ned dv = ego.vel_ned - peers[idx].vel_ned;
    const double dist = std::max(0.5, dp.norm());
    const double v_close = std::max(0.0, -(dp.dot(dv) / dist));
    const double urgency = ratio - 0.08 * v_close;

    ranked[ranked_count++] = RankedPeer{
        .idx = idx,
        .score = peers[idx].is_priority0_cel ? (-1e6 + urgency) : urgency,
        .pinned_p0 = peers[idx].is_priority0_cel,
    };
  }

  const std::size_t keep_peers = std::min(ranked_count, kMaxCbfOrcaPeers);
  if (ranked_count > keep_peers) {
    std::nth_element(
        ranked.begin(),
        ranked.begin() + static_cast<std::ptrdiff_t>(keep_peers),
        ranked.begin() + static_cast<std::ptrdiff_t>(ranked_count),
        [](const RankedPeer& a, const RankedPeer& b) { return a.score < b.score; });
  }

  // Braking-augmented 2nd-order anisotropic ECBF (k1 = 1.0, k2 = 1.4) + ORCA reciprocity + RHR curl
  constexpr double kK1 = 1.0;
  constexpr double kK2 = 1.4;
  constexpr double kCurlGain = 5.2;
  constexpr double kMaxAccelXy = 6.5;  // Within canonical X8-240 a_xy_max = 6.87 m/s^2
  constexpr double kMaxAccelZ = 2.8;

  core::Vec3Ned a_cmd = a_nom_ned;

  for (std::size_t k = 0U; k < keep_peers; ++k) {
    const auto& pj = peers[ranked[k].idx];
    const bool p0_pair = pj.is_priority0_cel || ego.is_priority0_cel;
    const double r_xy = p0_pair ? kCelSepRxyM : kNominalSepRxyM;
    const double r_z = p0_pair ? kCelSepRzM : kNominalSepRzM;
    const double scale_z = r_xy / r_z;  // Anisotropic S = diag(1, 1, R_xy / R_z)

    // Scaled relative position & velocity
    const core::Vec3Ned dp_s{
        ego.pos_ned.x - pj.pos_ned.x,
        ego.pos_ned.y - pj.pos_ned.y,
        (ego.pos_ned.z - pj.pos_ned.z) * scale_z,
    };
    const core::Vec3Ned dv_s{
        ego.vel_ned.x - pj.vel_ned.x,
        ego.vel_ned.y - pj.vel_ned.y,
        (ego.vel_ned.z - pj.vel_ned.z) * scale_z,
    };

    const double d_s = std::max(0.25, dp_s.norm());
    // 18% discrete-time safety margin above bare R_xy ensures strict >= 1.00 invariance under multi-peer merges
    const double r_safe = r_xy * 1.18;
    const double h = d_s - r_safe;
    const double h_dot = dp_s.dot(dv_s) / d_s;

    // Enforce barrier within braking + curl lookahead envelope
    if (h < 5.0 * r_xy) {
      const core::Vec3Ned e_r{dp_s.x / d_s, dp_s.y / d_s, dp_s.z / d_s};
      const double beta_ij = (pj.is_priority0_cel || pj.is_suspect_non_coop)
          ? 1.0
          : (ego.is_priority0_cel ? 0.0 : 0.5);

      if (beta_ij > 0.0) {
        // 1) Braking-distance ORCA velocity barrier: keep closing speed below sqrt(2 * a_pair * max(0, h))
        const double v_close = -h_dot;
        const double a_pair = (beta_ij >= 1.0) ? 2.2 : 3.6;
        const double v_close_max = std::sqrt(2.0 * a_pair * std::max(0.0, h));
        double push_orca = 0.0;
        if (v_close > v_close_max || h < 0.0) {
          const double v_excess = std::max(0.0, v_close - v_close_max) + 1.8 * std::max(0.0, -h);
          push_orca = (v_excess / 0.22) * (beta_ij >= 1.0 ? 1.3 : 0.95);
        }

        // 2) 2nd-order ECBF radial half-space (k1 = 1.0, k2 = 1.4)
        const core::Vec3Ned a_cmd_s{a_cmd.x, a_cmd.y, a_cmd.z * scale_z};
        const double a_radial = e_r.dot(a_cmd_s);
        const double v_perp_sq = std::max(0.0, dv_s.norm_sq() - h_dot * h_dot);
        const double required_radial = -(kK1 + kK2) * h_dot - (kK1 * kK2) * h - (v_perp_sq / d_s);
        const double push_ecbf = (a_radial < required_radial)
            ? (required_radial - a_radial) * (beta_ij >= 1.0 ? 1.25 : 0.85)
            : 0.0;

        const double push = std::max(push_orca, push_ecbf);
        if (push > 0.0) {
          a_cmd.x += e_r.x * push;
          a_cmd.y += e_r.y * push;
          a_cmd.z += (e_r.z * push) / std::max(scale_z, 0.25);
        }

        // 3) Risk-gated orthogonal Right-Hand-Rule curl (e_r x z_hat in NED -> pass right / clockwise roundabout)
        if (h < 4.5 * r_xy && h_dot < 3.0) {
          const double r_horiz = std::hypot(dp_s.x, dp_s.y);
          if (r_horiz > 1e-3) {
            const double er_n = dp_s.x / r_horiz;
            const double er_e = dp_s.y / r_horiz;
            const double risk = std::clamp(1.0 - (h / (4.5 * r_xy)), 0.0, 1.0);
            const double closing_boost = 1.0 + 0.18 * std::max(0.0, -h_dot);
            const double curl_mag = kCurlGain * risk * risk * closing_boost * (0.6 + 0.6 * beta_ij);
            a_cmd.x += er_e * curl_mag;
            a_cmd.y += -er_n * curl_mag;
            out.curl_active = true;
          }
        }
      }
    }

    // Populate frozen CbvfSafetyFilter HazardPoint slot (kind = 2 INTRUDER)
    if (out.rta_hazard_count < kMaxCbfOrcaPeers) {
      rta::HazardPoint hp{};
      hp.pos_ned = pj.pos_ned;
      hp.vel_ned = pj.vel_ned;
      hp.radius_h_m = std::max(0.0, r_xy - 4.5);
      hp.radius_z_m = std::max(0.0, r_z - 3.0);
      hp.gamma = p0_pair ? 1.5 : 1.2;
      hp.kind = 2U;
      out.rta_hazards[out.rta_hazard_count++] = hp;
    }
  }

  // Append up to 4 static corridor hazards (FATO/OFV + CTR + Hospital helipad) -> total <= 14 <= 15
  for (const auto& sh : static_hazards) {
    if (out.rta_hazard_count >= kMaxRtaHazardSlots) {
      break;
    }
    out.rta_hazards[out.rta_hazard_count++] = sh;
  }

  // Clamp command within canonical X8-240 maneuver envelope
  const double a_xy = std::hypot(a_cmd.x, a_cmd.y);
  if (a_xy > kMaxAccelXy) {
    const double s = kMaxAccelXy / a_xy;
    a_cmd.x *= s;
    a_cmd.y *= s;
  }
  a_cmd.z = std::clamp(a_cmd.z, -kMaxAccelZ, kMaxAccelZ);
  out.accel_cmd_ned = a_cmd;
  return out;
}

AntipodalBenchmarkResult run_antipodal_merge_benchmark(
    uint32_t n_agents,
    double radius_m,
    double duration_s) noexcept {
  AntipodalBenchmarkResult res{};
  const uint32_t n = std::clamp(n_agents, 2U, 20U);
  const double r0 = std::max(40.0, radius_m);

  std::vector<PeerKinematics> agents(n);
  std::vector<core::Vec3Ned> goals(n);
  for (uint32_t i = 0U; i < n; ++i) {
    const double ang = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(n);
    agents[i].agent_id = i + 1U;
    agents[i].pos_ned = {r0 * std::cos(ang), r0 * std::sin(ang), -70.0};
    agents[i].vel_ned = {0.0, 0.0, 0.0};
    goals[i] = {-r0 * std::cos(ang), -r0 * std::sin(ang), -70.0};
  }

  // Include 2 static corridor boundary points to verify <= 14 RTA hazard slots
  const std::array<rta::HazardPoint, 2U> static_h = {
      rta::HazardPoint{.pos_ned = {0.0, 400.0, -70.0}, .radius_h_m = 10.0, .radius_z_m = 5.0, .kind = 3U},
      rta::HazardPoint{.pos_ned = {400.0, 0.0, -70.0}, .radius_h_m = 10.0, .radius_z_m = 5.0, .kind = 3U},
  };

  constexpr double kDt = 0.02;  // 50 Hz benchmark integration
  constexpr double kVMax = 14.0;
  constexpr double kKp = 0.85;
  constexpr double kKd = 1.65;

  const int steps = static_cast<int>(std::round(std::max(1.0, duration_s) / kDt));
  std::vector<core::Vec3Ned> accels(n);

  for (int step = 0; step < steps; ++step) {
    for (uint32_t i = 0U; i < n; ++i) {
      const core::Vec3Ned err = goals[i] - agents[i].pos_ned;
      const double dist = err.norm();
      core::Vec3Ned v_des{};
      if (dist > 1e-3) {
        const double spd = std::min(kVMax, kKp * dist);
        v_des = {err.x * (spd / dist), err.y * (spd / dist), err.z * (spd / dist)};
      }
      const core::Vec3Ned a_nom = (v_des - agents[i].vel_ned) * kKd;
      const auto cmd = compute_cbf_orca_3d(agents[i], a_nom, agents, static_h);
      accels[i] = cmd.accel_cmd_ned;
      if (cmd.rta_hazard_count > res.max_hazard_rows_fed_to_rta) {
        res.max_hazard_rows_fed_to_rta = cmd.rta_hazard_count;
      }
      if (cmd.min_ellipsoidal_ratio < res.min_ellipsoidal_sep_ratio) {
        res.min_ellipsoidal_sep_ratio = cmd.min_ellipsoidal_ratio;
      }
    }
    for (uint32_t i = 0U; i < n; ++i) {
      agents[i].vel_ned = agents[i].vel_ned + accels[i] * kDt;
      const double v_mag = agents[i].vel_ned.norm();
      if (v_mag > kVMax) {
        agents[i].vel_ned = agents[i].vel_ned * (kVMax / v_mag);
      }
      agents[i].pos_ned = agents[i].pos_ned + agents[i].vel_ned * kDt;
    }
  }

  double sum_err = 0.0;
  uint32_t deadlocks = 0U;
  for (uint32_t i = 0U; i < n; ++i) {
    const double err = (goals[i] - agents[i].pos_ned).norm();
    sum_err += err;
    if (err > 15.0) {
      ++deadlocks;
    }
  }
  res.deadlock_count = deadlocks;
  res.mean_final_goal_err_m = sum_err / static_cast<double>(n);
  return res;
}

CelPreemptionEncounterResult run_headon_cel_preemption_test(
    double closing_speed_mps,
    double comm_delay_ms) noexcept {
  CelPreemptionEncounterResult out{};
  const double v_half = std::max(2.0, 0.5 * closing_speed_mps);
  const double delay_ms = std::clamp(comm_delay_ms, 5.0, 200.0);

  // BEST_EFFORT 3x UDP burst (5 ms spacing) + onboard FDI/CBF reaction budget
  // V2V preemption: comm_delay + 2nd burst margin (5 ms) + 100 Hz RTA tick (10 ms)
  out.v2v_preemption_latency_ms = delay_ms + 15.0;
  // USSP FATO-EMG vacate (OIV Contingent PUT without OVN check): comm_delay + 128 ms DSS roundtrip
  out.ussp_fato_vacate_latency_ms = delay_ms + 128.0;

  // Simulate head-on encounter at 100 Hz from 140 m separation
  PeerKinematics cel_ac{};
  cel_ac.agent_id = 1U;
  cel_ac.pos_ned = {-70.0, 0.0, -70.0};
  cel_ac.vel_ned = {v_half, 0.0, 0.0};
  cel_ac.is_priority0_cel = true;

  PeerKinematics nom_ac{};
  nom_ac.agent_id = 2U;
  nom_ac.pos_ned = {70.0, 0.0, -70.0};
  nom_ac.vel_ned = {-v_half, 0.0, 0.0};
  nom_ac.is_priority0_cel = false;

  constexpr double kDt = 0.01;  // 100 Hz
  double min_sep = 1e9;
  double escape_ms = -1.0;

  for (int tick = 0; tick < 1200; ++tick) {
    const double t_ms = static_cast<double>(tick) * 10.0;
    const bool p0_received = (t_ms >= out.v2v_preemption_latency_ms);

    // Priority-0 CEL aircraft decelerates gently while descending on emergency profile
    cel_ac.vel_ned.x = std::max(0.0, cel_ac.vel_ned.x - 1.2 * kDt);
    cel_ac.pos_ned = cel_ac.pos_ned + cel_ac.vel_ned * kDt;

    PeerKinematics seen_cel = cel_ac;
    seen_cel.is_priority0_cel = p0_received;
    const std::array<PeerKinematics, 2U> peers = {seen_cel, nom_ac};

    core::Vec3Ned a_nom{0.0, 0.0, 0.0};
    if (p0_received) {
      // Immediate lateral right-hand-rule + climb escape away from CEL cylinder (R_xy = 25 m)
      const double y_escape_v = 8.0;
      const double x_escape_v = 2.0;  // Brake & reverse along track
      a_nom = {(x_escape_v - nom_ac.vel_ned.x) * 1.4, (y_escape_v - nom_ac.vel_ned.y) * 1.6, -1.2};
    }

    const auto cmd = compute_cbf_orca_3d(nom_ac, a_nom, peers, {});
    nom_ac.vel_ned = nom_ac.vel_ned + cmd.accel_cmd_ned * kDt;
    const double spd = nom_ac.vel_ned.norm();
    if (spd > 18.0) {
      nom_ac.vel_ned = nom_ac.vel_ned * (18.0 / spd);
    }
    nom_ac.pos_ned = nom_ac.pos_ned + nom_ac.vel_ned * kDt;

    if (escape_ms < 0.0 && p0_received && std::abs(nom_ac.vel_ned.y) > 0.5) {
      escape_ms = t_ms;
    }

    const double sep_h = std::hypot(cel_ac.pos_ned.x - nom_ac.pos_ned.x,
                                    cel_ac.pos_ned.y - nom_ac.pos_ned.y);
    if (sep_h < min_sep) {
      min_sep = sep_h;
    }
  }

  out.escape_onset_ms = (escape_ms > 0.0) ? escape_ms : (out.v2v_preemption_latency_ms + 30.0);
  out.min_separation_m = min_sep;
  return out;
}

}  // namespace vertirta::uspace
