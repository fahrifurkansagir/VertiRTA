#include "vertirta/gnss_denied/ctr_inflation_fsm.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::gnss_denied {

namespace {

[[nodiscard]] core::Vec3Ned arp_local_ned(double lat_deg, double lon_deg) noexcept {
  const core::GeodeticWgs84 g{lat_deg, lon_deg, core::amsl_to_hae(12.0, lat_deg, lon_deg)};
  return core::enu_to_ned(core::geodetic_to_local_enu(g, core::kKartalDatum));
}

}  // namespace

CtrInflationFsm::CtrInflationFsm(const CtrInflationConfig& cfg) noexcept : cfg_(cfg) {
  reset(2.0);
}

void CtrInflationFsm::reset(double sigma_init_m) noexcept {
  const double s = std::max(0.5, sigma_init_m);
  cov_ = {s * s, 0.0, s * s};
  cov_rate_ = {0.0, 0.0, 0.0};
  open_loop_dist_m_ = 0.0;
  prev_ltfj_delta_r_m_ = 3.0 * s + cfg_.b_map_m;
  prev_ltbx_delta_r_m_ = 3.0 * s + cfg_.b_map_m;
  has_prev_delta_ = false;
  nav_state_ = GnssDeniedNavState::kBoundedGnssDeniedCruise;
  abort_latched_ = false;
}

void CtrInflationFsm::propagate_open_loop(double dt_s, double horiz_speed_mps) noexcept {
  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    return;
  }
  const double v = std::max(0.0, std::isfinite(horiz_speed_mps) ? horiz_speed_mps : 0.0);
  const double ds = v * dt_s;

  // Covariance growth: diffusion q_rw * dt + incremental systematic path-length drift d((kappa * d_ol)^2).
  const double k2 = cfg_.kappa_drift_fraction * cfg_.kappa_drift_fraction;
  const double d_sys = k2 * (2.0 * open_loop_dist_m_ * ds + ds * ds);
  const double d_rw = cfg_.q_rw_m2_per_s * dt_s;
  const double dp = d_rw + d_sys;

  open_loop_dist_m_ += ds;
  cov_.p_nn += dp;
  cov_.p_ee += dp;

  const double dp_dt = dp / dt_s;
  cov_rate_ = {dp_dt, 0.0, dp_dt};
}

void CtrInflationFsm::apply_reloc_update(const Cov2D& updated_cov) noexcept {
  cov_ = updated_cov;
  cov_rate_ = {0.0, 0.0, 0.0};
  open_loop_dist_m_ = 0.0;
}

CtrInflationStatus CtrInflationFsm::evaluate_and_build_hazards(
    double dt_s,
    const core::Vec3Ned& own_pos_ned,
    double alt_amsl_m,
    std::span<rta::HazardPoint> out_hazards) noexcept {
  CtrInflationStatus st{};

  const double sigma_pos = cov_.sigma_max();
  st.sigma_pos_m = sigma_pos;
  st.three_sigma_pos_m = 3.0 * sigma_pos;
  st.nominal_3sigma_met = (st.three_sigma_pos_m < cfg_.nominal_3sigma_target_m);

  // --- 3-State GNSS-Denied Navigation FSM (with hysteresis & latched deterministic Visual-SLZ abort) ---
  if (abort_latched_ || sigma_pos >= cfg_.sigma_abort_m) {
    abort_latched_ = true;
    nav_state_ = GnssDeniedNavState::kDeterministicVisualSlzAbort;
  } else if (nav_state_ == GnssDeniedNavState::kDegradedCaution) {
    if (sigma_pos < (cfg_.sigma_caution_m - cfg_.fsm_hysteresis_m)) {
      nav_state_ = GnssDeniedNavState::kBoundedGnssDeniedCruise;
    }
  } else if (sigma_pos >= cfg_.sigma_caution_m) {
    nav_state_ = GnssDeniedNavState::kDegradedCaution;
  }

  st.nav_state = nav_state_;
  st.visual_slz_abort_latched = abort_latched_;
  st.westward_bias_active = (nav_state_ != GnssDeniedNavState::kBoundedGnssDeniedCruise);
  if (nav_state_ == GnssDeniedNavState::kBoundedGnssDeniedCruise) {
    st.v_max_cap_mps = cfg_.v_max_nominal_mps;
  } else {
    st.v_max_cap_mps = std::clamp(cfg_.v_max_caution_mps, 12.0, 15.0);
  }

  // Convert own local Kartal NED to WGS84 strictly via pure vertirta::core::geodesy (zero libproj).
  const core::GeodeticWgs84 own_geo =
      core::local_enu_to_geodetic(core::ned_to_enu(own_pos_ned), core::kKartalDatum);

  st.ltfj_nominal_margin_m =
      rta::AirspaceExclusion::signed_margin_ltfj_m(own_geo.lat_deg, own_geo.lon_deg, alt_amsl_m);
  st.ltbx_nominal_margin_m =
      rta::AirspaceExclusion::signed_margin_ltbx_m(own_geo.lat_deg, own_geo.lon_deg, alt_amsl_m);

  // Build directional 3*sqrt(er^T P er) + b_map AND exact dh/dt = -r_dot_infl CTR HazardPoints (kind=3)
  // directly compatible with unmodified frozen CbvfSafetyFilter::step_100hz (cbvf_safety_filter.cpp L202-234).
  std::size_t n_emit = 0u;
  auto emit_ctr_wall = [&](double nominal_margin_m,
                           double arp_lat_deg,
                           double arp_lon_deg,
                           double radius_z_m,
                           double& prev_delta_r,
                           double& out_delta_r,
                           double& out_r_dot,
                           double& out_inflated_margin) noexcept {
    const core::Vec3Ned arp_ned = arp_local_ned(arp_lat_deg, arp_lon_deg);
    const double to_arp_n = arp_ned.x - own_pos_ned.x;
    const double to_arp_e = arp_ned.y - own_pos_ned.y;
    const double d_arp = std::max(1.0, std::hypot(to_arp_n, to_arp_e));

    // Safe outward unit normal er points FROM the CTR wall TOWARD the aircraft (= opposite to_arp).
    const double er_n = -to_arp_n / d_arp;
    const double er_e = -to_arp_e / d_arp;

    const double var_n = cov_.directional_variance(er_n, er_e);
    const double sigma_n = std::sqrt(std::max(0.04, var_n));
    const double delta_r = 3.0 * sigma_n + cfg_.b_map_m;

    // Analytic outward wall-expansion rate: d/dt [3*sqrt(er^T P er)] = 3*(er^T P_dot er)/(2*sigma_n)
    const double var_dot = std::max(0.0, cov_rate_.directional_variance(er_n, er_e));
    const double r_dot_analytic = 3.0 * var_dot / (2.0 * std::max(sigma_n, 0.25));
    double r_dot_fd = 0.0;
    if (has_prev_delta_ && std::isfinite(dt_s) && dt_s > 1e-4) {
      r_dot_fd = std::max(0.0, (delta_r - prev_delta_r) / dt_s);
    }
    const double r_dot = std::clamp(std::max(r_dot_analytic, r_dot_fd), 0.0, cfg_.max_r_dot_infl_mps);

    prev_delta_r = delta_r;
    out_delta_r = delta_r;
    out_r_dot = r_dot;
    out_inflated_margin = nominal_margin_m - delta_r;

    // Activate hazard whenever inflated wall lies within 500 m of aircraft (handles large 3sigma safely!).
    if (n_emit < out_hazards.size() && out_inflated_margin < cfg_.ctr_base_activation_m) {
      const double wall_dist = std::max(nominal_margin_m, 0.5);
      // Place virtual wall along direction toward ARP at nominal wall distance, then inflate by delta_r
      // and give outward expansion velocity +r_dot * er -> CbvfSafetyFilter computes:
      //   r_eff = wall_dist - delta_r
      //   v_r   = er . (vel_ned - (+r_dot * er)) = er . vel_ned - r_dot  (EXACT dh/dt = -r_dot!)
      out_hazards[n_emit++] = rta::HazardPoint{
          .pos_ned = {own_pos_ned.x - er_n * wall_dist, own_pos_ned.y - er_e * wall_dist, own_pos_ned.z},
          .vel_ned = {er_n * r_dot, er_e * r_dot, 0.0},
          .accel_ned = {0.0, 0.0, 0.0},
          .radius_h_m = delta_r,
          .radius_z_m = radius_z_m,
          .gamma = 1.5,
          .kind = 3u,
      };
    }
  };

  emit_ctr_wall(st.ltfj_nominal_margin_m, cfg_.ltfj_arp_lat_deg, cfg_.ltfj_arp_lon_deg, 700.0,
                prev_ltfj_delta_r_m_, st.ltfj_delta_r_m, st.ltfj_r_dot_infl_mps,
                st.ltfj_inflated_margin_m);
  emit_ctr_wall(st.ltbx_nominal_margin_m, cfg_.ltbx_arp_lat_deg, cfg_.ltbx_arp_lon_deg, 600.0,
                prev_ltbx_delta_r_m_, st.ltbx_delta_r_m, st.ltbx_r_dot_infl_mps,
                st.ltbx_inflated_margin_m);

  has_prev_delta_ = true;
  st.emitted_ctr_hazards = n_emit;
  return st;
}

}  // namespace vertirta::gnss_denied
