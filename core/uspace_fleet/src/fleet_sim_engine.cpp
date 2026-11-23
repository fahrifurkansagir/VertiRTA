// SPDX-License-Identifier: Apache-2.0
#include "vertirta/uspace/fleet_sim_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vertirta::uspace {

namespace {

[[nodiscard]] core::Vec3Ned vertiport_cruise_ned(
    const CorridorNetwork& net,
    uint32_t vp_idx,
    double agl_m) noexcept {
  const auto& vp = net.vertiports[vp_idx % 4U];
  const core::Vec3Enu enu{vp.pos_enu.x, vp.pos_enu.y, agl_m};
  return core::enu_to_ned(enu);
}

}  // namespace

FleetSimEngine::FleetSimEngine(uint32_t n_agents, uint32_t threads)
    : n_agents_(std::clamp(n_agents, 1U, static_cast<uint32_t>(kMaxFleetAgents))),
      num_threads_(std::clamp(threads, 1U, 4U)),
      corridor_(make_anatolian_corridor_network()) {
  for (std::size_t i = 0U; i < kMaxFleetAgents; ++i) {
    rta_filters_[i] = std::make_unique<rta::CbvfSafetyFilter>(nullptr, nullptr);
  }
  reset_corridor_schedule(20261123ULL);

  // Spawn (num_threads_ - 1) persistent workers; main thread owns slice 0
  if (num_threads_ > 1U) {
    workers_.reserve(num_threads_ - 1U);
    for (uint32_t w = 1U; w < num_threads_; ++w) {
      workers_.emplace_back([this, w]() { worker_loop(w); });
    }
  }
}

FleetSimEngine::~FleetSimEngine() {
  if (!workers_.empty()) {
    {
      std::lock_guard<std::mutex> lk(pool_mtx_);
      stop_pool_ = true;
    }
    cv_start_.notify_all();
    for (auto& t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }
}

void FleetSimEngine::reset_corridor_schedule(uint64_t seed) noexcept {
  tick_index_ = 0U;

  // Spread 5-20 eVTOLs along the 4-vertiport coastal corridor strictly west of 29.1895 E
  // on segregated 70 m SE (even indices) and 105 m NW (odd indices) lanes.
  for (uint32_t i = 0U; i < n_agents_; ++i) {
    const bool northbound_nw = ((i % 2U) == 1U);
    const double agl_m = northbound_nw ? corridor_.lane_nw_agl_m : corridor_.lane_se_agl_m;

    // Interpolate smoothly along the Maltepe (0) -> Dragos (1) -> Cevizli (2) -> Kartal (3) spine
    // keeping >= 220 m initial spacing and strictly inside western coastal corridor
    const double frac = (static_cast<double>(i) + 0.5) / static_cast<double>(n_agents_);
    const double jitter_e = static_cast<double>((seed + i * 37ULL) % 11ULL) * 2.0 - 10.0;
    const auto& p_mlt = corridor_.vertiports[0].pos_enu;
    const auto& p_krt = corridor_.vertiports[3].pos_enu;

    // Slight coastal offset west of Kartal so every agent starts safely west of 29.1895 E
    const double east_m = p_mlt.x + frac * (p_krt.x - p_mlt.x) - 180.0 + jitter_e;
    const double north_m = p_mlt.y + frac * (p_krt.y - p_mlt.y);
    const core::Vec3Ned init_pos = core::enu_to_ned({east_m, north_m, agl_m});

    const uint32_t dst_vp = northbound_nw ? 0U : 3U;
    dest_vp_idx_[i] = static_cast<uint8_t>(dst_vp);
    goal_ned_[i] = vertiport_cruise_ned(corridor_, dst_vp, agl_m);
    // Offset goal slightly west of Kartal datum to keep corridor turn strictly < 29.1895 E
    goal_ned_[i].y -= 120.0;

    fault_mask_[i] = 0x00U;
    cel_latched_[i] = 0U;
    suspect_peer_[i] = 0U;
    is_external_hero_[i] = (i < 2U) ? 1U : 0U;

    plants_[i].reset(init_pos, {0.0, 0.0, 0.0}, 0.80f, core::params::kOmegaHover);
    controllers_[i].reset(core::params::kOmegaHover);
    rta_filters_[i]->reset();
    cyber_fsms_[i].reset();

    peer_snapshot_[i] = PeerKinematics{
        .agent_id = i + 1U,
        .pos_ned = init_pos,
        .vel_ned = {0.0, 0.0, 0.0},
        .is_priority0_cel = false,
        .is_suspect_non_coop = false,
    };
  }
}

void FleetSimEngine::inject_same_arm_cel(uint32_t agent_idx, uint32_t arm_idx) noexcept {
  if (agent_idx >= n_agents_) {
    return;
  }
  const uint32_t arm = arm_idx % 4U;
  // Same-arm upper (arm) + lower (arm + 4) dual motor kill
  fault_mask_[agent_idx] = static_cast<uint8_t>((1U << arm) | (1U << (arm + 4U)));
}

void FleetSimEngine::mark_peer_suspect(uint32_t agent_idx) noexcept {
  if (agent_idx >= n_agents_) {
    return;
  }
  suspect_peer_[agent_idx] = 1U;
  (void)cyber_fsms_[agent_idx].on_peer_kinematic_anomaly(agent_idx + 1U, 4.8);
}

void FleetSimEngine::worker_loop(uint32_t worker_id) noexcept {
  uint64_t seen_gen = 0ULL;
  for (;;) {
    {
      std::unique_lock<std::mutex> lk(pool_mtx_);
      cv_start_.wait(lk, [this, seen_gen]() { return stop_pool_ || (generation_ != seen_gen); });
      if (stop_pool_) {
        return;
      }
      seen_gen = generation_;
    }

    const uint32_t chunk = (n_agents_ + num_threads_ - 1U) / num_threads_;
    const uint32_t begin_idx = std::min(n_agents_, worker_id * chunk);
    const uint32_t end_idx = std::min(n_agents_, begin_idx + chunk);

    WorkerResult local{};
    step_agent_range(begin_idx, end_idx, local);
    worker_results_[worker_id] = local;

    {
      std::lock_guard<std::mutex> lk(pool_mtx_);
      if (active_workers_ > 0U) {
        --active_workers_;
      }
      if (active_workers_ == 0U) {
        cv_done_.notify_one();
      }
    }
  }
}

void FleetSimEngine::step_agent_range(
    uint32_t begin_idx,
    uint32_t end_idx,
    WorkerResult& out) noexcept {
  const std::span<const PeerKinematics> all_peers(peer_snapshot_.data(), n_agents_);

  // 2 static corridor hazards fed into CBF-ORCA -> total hazards <= 10 peers + 2 static = 12 <= 14 <= 15
  const core::Vec3Ned hosp_ned = core::enu_to_ned(corridor_.hospital_helipad.pos_enu);

  for (uint32_t i = begin_idx; i < end_idx; ++i) {
    const auto& st = plants_[i].state();
    const uint8_t fmask = fault_mask_[i];

    // Eastern LTFJ NW J2 fence point directly east of agent at lon = 29.1895 E
    const core::Vec3Enu agent_enu = core::ned_to_enu(st.pos_ned);
    const auto agent_geo = core::local_enu_to_geodetic(agent_enu, core::kKartalDatum);
    const auto fence_enu = core::geodetic_to_local_enu(
        {agent_geo.lat_deg, corridor_.eastern_geofence_lon_deg, agent_geo.h_hae_m},
        core::kKartalDatum);
    const std::array<rta::HazardPoint, 2U> static_hazards = {
        rta::HazardPoint{
            .pos_ned = hosp_ned,
            .radius_h_m = corridor_.hospital_helipad.radius_m,
            .radius_z_m = 40.0,
            .gamma = 1.4,
            .kind = 3U,
        },
        rta::HazardPoint{
            .pos_ned = core::enu_to_ned(fence_enu),
            .radius_h_m = 20.0,
            .radius_z_m = 100.0,
            .gamma = 1.5,
            .kind = 3U,
        },
    };

    // Outer guidance PD toward corridor destination
    const core::Vec3Ned err = goal_ned_[i] - st.pos_ned;
    const double dist = err.norm();
    core::Vec3Ned v_des{};
    if (dist > 1.0) {
      const double v_cruise = (cel_latched_[i] != 0U) ? 4.0 : 12.0;
      const double spd = std::min(v_cruise, 0.5 * dist);
      v_des = {err.x * (spd / dist), err.y * (spd / dist), std::clamp(err.z * 0.4, -1.5, 1.5)};
    }
    const core::Vec3Ned a_nom = (v_des - st.vel_ned) * 1.1;

    // 100 Hz 3D ellipsoidal CBF-ORCA + Top-10 pruning (<= 14 <= 15 RTA slots)
    const auto orca_cmd = compute_cbf_orca_3d(peer_snapshot_[i], a_nom, all_peers, static_hazards);
    if (orca_cmd.rta_hazard_count > out.max_hazards) {
      out.max_hazards = orca_cmd.rta_hazard_count;
    }

    // Build frozen CapabilityEnvelope & run untouched CbvfSafetyFilter::step_100hz()
    const auto env = core::CoaxEffectiveness::compute_envelope(
        fmask,
        /*tmr_state=*/0U,
        /*imu_mask=*/0x07U,
        st.battery.v_term_v,
        st.battery.i_pack_a,
        st.battery.soc,
        controllers_[i].spin_rate_ref_rad_s(),
        static_cast<float>(st.omega_frd.z),
        /*chi2_residual=*/0.0f,
        /*fdi_latency_ms=*/0U);

    const auto rta_out = rta_filters_[i]->step_100hz(
        st.pos_ned,
        st.vel_ned,
        orca_cmd.accel_cmd_ned,
        /*yaw_rate_nom=*/0.0,
        env,
        std::span<const rta::HazardPoint>(orca_cmd.rta_hazards.data(), orca_cmd.rta_hazard_count));

    if (rta_out.cel_latched || rta_filters_[i]->is_cel_latched()) {
      cel_latched_[i] = 1U;
      ++out.cel_count;
    }

    // 4x 400 Hz inner-loop INDI-WLS + 6-DOF coaxial X8 plant substeps
    const core::Vec3Ned wind_ned{0.8, -0.5, 0.0};
    for (int sub = 0; sub < 4; ++sub) {
      const auto& cur = plants_[i].state();
      core::IndiInput in{};
      in.q_frd_ned = cur.q_frd_ned;
      in.gyro_frd = cur.omega_frd;
      in.accel_des_ned = rta_out.a_safe_ned;
      in.yaw_rate_des = rta_out.yaw_rate_safe;
      in.v_horiz_mps = static_cast<float>(std::hypot(cur.vel_ned.x, cur.vel_ned.y));
      in.v_term_v = cur.battery.v_term_v;
      in.fault_mask = fmask;

      float omega_cmd[8]{};
      (void)controllers_[i].step_400hz(in, omega_cmd);
      (void)plants_[i].step_400hz(omega_cmd, fmask, wind_ned);
    }

    // Check eastern LTFJ geofence & altitude ceiling after substep
    const auto post_enu = core::ned_to_enu(plants_[i].state().pos_ned);
    const auto post_geo = core::local_enu_to_geodetic(post_enu, core::kKartalDatum);
    if (post_geo.lon_deg >= corridor_.eastern_geofence_lon_deg ||
        post_enu.z > corridor_.odd_ceiling_agl_m) {
      ++out.breaches;
    }
  }
}

FleetTickSummary FleetSimEngine::step_100hz() noexcept {
  // 1) Snapshot current kinematics into read-only SoA buffer before parallel dispatch
  for (uint32_t i = 0U; i < n_agents_; ++i) {
    const auto& st = plants_[i].state();
    peer_snapshot_[i] = PeerKinematics{
        .agent_id = i + 1U,
        .pos_ned = st.pos_ned,
        .vel_ned = st.vel_ned,
        .is_priority0_cel = (cel_latched_[i] != 0U) ||
                            (core::CoaxEffectiveness::classify_regime(fault_mask_[i]) ==
                             core::FlightRegime::SAME_ARM_ACTIVE_SPIN),
        .is_suspect_non_coop = (suspect_peer_[i] != 0U),
    };
  }

  // 2) Parallel execution across disjoint agent slices
  const uint32_t chunk = (n_agents_ + num_threads_ - 1U) / num_threads_;
  if (num_threads_ > 1U) {
    {
      std::lock_guard<std::mutex> lk(pool_mtx_);
      active_workers_ = num_threads_ - 1U;
      ++generation_;
    }
    cv_start_.notify_all();

    // Main thread executes slice 0
    WorkerResult main_res{};
    step_agent_range(0U, std::min(n_agents_, chunk), main_res);
    worker_results_[0] = main_res;

    // Barrier wait for workers 1..num_threads_-1
    std::unique_lock<std::mutex> lk(pool_mtx_);
    cv_done_.wait(lk, [this]() { return active_workers_ == 0U; });
  } else {
    WorkerResult main_res{};
    step_agent_range(0U, n_agents_, main_res);
    worker_results_[0] = main_res;
  }

  ++tick_index_;

  // 3) Aggregate tick summary
  FleetTickSummary summary{};
  summary.tick_index = tick_index_;
  summary.active_agents = n_agents_;
  summary.sim_time_s = static_cast<double>(tick_index_) * 0.01;
  for (uint32_t w = 0U; w < num_threads_; ++w) {
    summary.cel_active_agents += worker_results_[w].cel_count;
    summary.airspace_breaches += worker_results_[w].breaches;
    if (worker_results_[w].max_hazards > summary.max_rta_hazard_rows) {
      summary.max_rta_hazard_rows = worker_results_[w].max_hazards;
    }
  }

  double min_sep = 1e6;
  for (uint32_t i = 0U; i < n_agents_; ++i) {
    const auto& pi = plants_[i].state().pos_ned;
    for (uint32_t j = i + 1U; j < n_agents_; ++j) {
      const double d = (pi - plants_[j].state().pos_ned).norm();
      if (d < min_sep) {
        min_sep = d;
      }
    }
  }
  summary.min_inter_agent_sep_m = min_sep;
  return summary;
}

bool FleetSimEngine::agent_cel_latched(uint32_t agent_idx) const noexcept {
  if (agent_idx >= n_agents_) {
    return false;
  }
  return (cel_latched_[agent_idx] != 0U) || rta_filters_[agent_idx]->is_cel_latched();
}

core::Vec3Ned FleetSimEngine::agent_pos_ned(uint32_t agent_idx) const noexcept {
  if (agent_idx >= n_agents_) {
    return {};
  }
  return plants_[agent_idx].state().pos_ned;
}

core::Vec3Ned FleetSimEngine::agent_vel_ned(uint32_t agent_idx) const noexcept {
  if (agent_idx >= n_agents_) {
    return {};
  }
  return plants_[agent_idx].state().vel_ned;
}

uint8_t FleetSimEngine::agent_fault_mask(uint32_t agent_idx) const noexcept {
  if (agent_idx >= n_agents_) {
    return 0x00U;
  }
  return fault_mask_[agent_idx];
}

}  // namespace vertirta::uspace
