// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/intruder_kf.hpp"

namespace vertirta::gnss_denied_node {

// Strict PX4 v1.16 ROS 2 Topic Contract (Plan 6A Task 5)
inline constexpr const char* kTopicVehicleGpsPosition = "/fmu/out/vehicle_gps_position";
inline constexpr const char* kTopicVehicleOdometry = "/fmu/out/vehicle_odometry";
inline constexpr const char* kTopicBellyStereoLeft = "/vertirta/belly_stereo/left/image_raw";
inline constexpr const char* kTopicBellyStereoRight = "/vertirta/belly_stereo/right/image_raw";
inline constexpr const char* kTopicRadaltRange = "/vertirta/radalt/range";
inline constexpr const char* kTopicSlzWaterFraction = "/vertirta/slz_water_fraction";
inline constexpr const char* kTopicGnssDeniedStatus = "/vertirta/gnss_denied_status";
inline constexpr const char* kTopicGnssDeniedOdometry = "/vertirta/gnss_denied_odometry";
inline constexpr const char* kTopicCtrInflationMarkers = "/vertirta/ctr_inflation_markers";
inline constexpr const char* kTopicGnssSpeedCap = "/vertirta/gnss_speed_cap_mps";

// Confirmed DHMI AIP Turkey coordinates (Kartal datum, LTFJ, LTBX 40 deg 59 min 26 sec N, 029 deg 12 min 58 sec E)
inline constexpr double kKartalDatumLatDeg = 40.8860;
inline constexpr double kKartalDatumLonDeg = 29.1880;
inline constexpr double kLtfjArpLatDeg = 40.898333;
inline constexpr double kLtfjArpLonDeg = 29.309167;
inline constexpr double kVerifiedLtbxArpLatDeg = 40.990556;
inline constexpr double kVerifiedLtbxArpLonDeg = 29.216111;
inline constexpr double kCtrRadiusM = 9260.0;
inline constexpr double kMapBiasM = 2.0;

enum class SpoofHypothesis : uint8_t {
  H0_NOMINAL = 0U,
  H1_RF_JAMMING = 1U,
  H2_STEP_MEACONING = 2U,
  H3_SLOW_RAMP_SPOOF = 3U,
};

enum class NavFsmState : uint8_t {
  BOUNDED_GNSS_DENIED_CRUISE = 0U,
  DEGRADED_CAUTION = 1U,
  DETERMINISTIC_VISUAL_SLZ_ABORT = 2U,
};

[[nodiscard]] constexpr const char* spoof_hypothesis_str(SpoofHypothesis h) noexcept {
  switch (h) {
    case SpoofHypothesis::H0_NOMINAL:
      return "H0_NOMINAL";
    case SpoofHypothesis::H1_RF_JAMMING:
      return "H1_RF_JAMMING";
    case SpoofHypothesis::H2_STEP_MEACONING:
      return "H2_STEP_MEACONING";
    case SpoofHypothesis::H3_SLOW_RAMP_SPOOF:
      return "H3_SLOW_RAMP_SPOOF";
  }
  return "H0_NOMINAL";
}

[[nodiscard]] constexpr const char* nav_fsm_state_str(NavFsmState s) noexcept {
  switch (s) {
    case NavFsmState::BOUNDED_GNSS_DENIED_CRUISE:
      return "BOUNDED_GNSS_DENIED_CRUISE";
    case NavFsmState::DEGRADED_CAUTION:
      return "DEGRADED_CAUTION";
    case NavFsmState::DETERMINISTIC_VISUAL_SLZ_ABORT:
      return "DETERMINISTIC_VISUAL_SLZ_ABORT";
  }
  return "BOUNDED_GNSS_DENIED_CRUISE";
}

struct GnssDeniedConfig {
  double ekf2_gps_delay_ms{110.0};
  bool uxrce_dds_synct_enabled{true};
  double land_mcusum_kd{0.25};
  double land_mcusum_ht{2.0};
  double water_mcusum_kd{0.35};
  double water_mcusum_ht{3.4};
  double vel_residual_sigma_mps{0.25};
  double step_meaconing_thresh_m{5.0};
  double rollback_window_s{2.0};
  double q_rw_m2_per_s{0.04};
  double kappa_drift_land{0.008};
  double kappa_drift_water{0.012};
  double fsm_caution_sigma_m{25.0};
  double fsm_abort_sigma_m{65.0};
  double v_max_nominal_mps{22.0};
  double v_max_caution_mps{13.5};
};

struct VisualRelocMeasurement {
  bool valid{false};
  int inlier_count{0};
  double reproj_rmse_px{9.9};
  double hull_area_ratio{0.0};
  double meas_sigma_m{2.0};
  core::Vec3Enu pos_enu{};
};

struct GnssDeniedStepInput {
  double t_s{0.0};
  double dt_s{0.1};
  bool gps_valid{true};
  bool timesync_ok{true};
  double cn0_dbhz{42.0};
  double hdop{0.9};
  double water_fraction{0.05};
  core::Vec3Enu pos_gnss_enu{};
  core::Vec3Enu vel_gnss_enu{};
  core::Vec3Enu pos_vio_enu{};
  core::Vec3Enu vel_vio_enu{};
  VisualRelocMeasurement visual_reloc{};
};

struct GnssDeniedOutput {
  SpoofHypothesis hypothesis{SpoofHypothesis::H0_NOMINAL};
  NavFsmState nav_state{NavFsmState::BOUNDED_GNSS_DENIED_CRUISE};
  bool spoof_latched{false};
  bool gnss_fused_allowed{true};
  bool rollback_applied{false};
  double mcusum_norm{0.0};
  double sigma_pos_m{1.5};
  double three_sigma_m{4.5};
  double ctr_inflation_m{6.5};
  double dh_dt_mps{0.0};
  double v_max_cap_mps{22.0};
  double ltfj_inflated_margin_m{0.0};
  double ltbx_inflated_margin_m{0.0};
  core::Vec3Enu est_pos_enu{};
  core::Vec3Enu est_vel_enu{};
};

class GnssDeniedNodeCore {
 public:
  explicit GnssDeniedNodeCore(GnssDeniedConfig cfg = {}) noexcept : cfg_(cfg) {}

  [[nodiscard]] const GnssDeniedConfig& config() const noexcept { return cfg_; }

  [[nodiscard]] GnssDeniedOutput step(const GnssDeniedStepInput& in) {
    const double dt = std::clamp(in.dt_s, 1e-3, 1.0);
    const double wf = std::clamp(in.water_fraction, 0.0, 1.0);

    HistorySample hist{};
    hist.t_s = in.t_s;
    hist.pos_gnss = in.pos_gnss_enu;
    hist.vel_gnss = in.vel_gnss_enu;
    hist.pos_vio = in.pos_vio_enu;
    hist.vel_vio = in.vel_vio_enu;
    history_.push_back(hist);
    while (history_.size() > 220U) {
      history_.pop_front();
    }

    // 1. Lagged baseline bias estimation strictly from [t - 15 s, t - 5 s)
    double b_lag_e = 0.0;
    double b_lag_n = 0.0;
    std::size_t lag_cnt = 0U;
    for (const auto& s : history_) {
      const double age = in.t_s - s.t_s;
      if (age >= 5.0 && age <= 15.0) {
        b_lag_e += (s.vel_gnss.x - s.vel_vio.x);
        b_lag_n += (s.vel_gnss.y - s.vel_vio.y);
        ++lag_cnt;
      }
    }
    if (lag_cnt > 0U) {
      b_lag_e /= static_cast<double>(lag_cnt);
      b_lag_n /= static_cast<double>(lag_cnt);
    }

    // 2. Adaptive land / transition / open-water MCUSUM thresholds
    double k_d = cfg_.land_mcusum_kd;
    double h_t = cfg_.land_mcusum_ht;
    if (wf > 0.35) {
      k_d = cfg_.water_mcusum_kd;
      h_t = cfg_.water_mcusum_ht;
    } else if (wf >= 0.20) {
      const double alpha = (wf - 0.20) / 0.15;
      k_d = cfg_.land_mcusum_kd + alpha * (cfg_.water_mcusum_kd - cfg_.land_mcusum_kd);
      h_t = cfg_.land_mcusum_ht + alpha * (cfg_.water_mcusum_ht - cfg_.land_mcusum_ht);
    }

    // 3. Evaluate 4-hypothesis GNSS spoof/jam detector if not yet latched
    if (!spoof_latched_) {
      const bool rf_jam = (!in.gps_valid) || (in.cn0_dbhz < 28.0) || (in.hdop > 2.5) ||
                          (cfg_.uxrce_dds_synct_enabled && !in.timesync_ok);
      const double step_err = std::hypot(
          in.pos_gnss_enu.x - in.pos_vio_enu.x,
          in.pos_gnss_enu.y - in.pos_vio_enu.y);
      const bool step_meacon = in.gps_valid && (step_err > cfg_.step_meaconing_thresh_m);

      // H3a: 2D vector MCUSUM on bias-compensated velocity residual
      const double rv_e = (in.vel_gnss_enu.x - in.vel_vio_enu.x - b_lag_e) / cfg_.vel_residual_sigma_mps;
      const double rv_n = (in.vel_gnss_enu.y - in.vel_vio_enu.y - b_lag_n) / cfg_.vel_residual_sigma_mps;
      const double sx = mcusum_e_ + rv_e;
      const double sy = mcusum_n_ + rv_n;
      const double s_mag = std::hypot(sx, sy);
      if (s_mag > k_d && s_mag > 1e-9) {
        const double scale = (s_mag - k_d) / s_mag;
        mcusum_e_ = sx * scale;
        mcusum_n_ = sy * scale;
      } else {
        mcusum_e_ = 0.0;
        mcusum_n_ = 0.0;
      }
      const double mcusum_norm = std::hypot(mcusum_e_, mcusum_n_);

      // H3b: 1.5 s / 4.0 s telescoping position-increment GLRT (water corroboration)
      bool h3b_corroborated = false;
      for (const double win_s : {1.5, 4.0}) {
        const HistorySample* past = find_sample_at_age(in.t_s, win_s);
        if (past != nullptr) {
          const double d_gnss_e = in.pos_gnss_enu.x - past->pos_gnss.x;
          const double d_gnss_n = in.pos_gnss_enu.y - past->pos_gnss.y;
          const double d_vio_e = in.pos_vio_enu.x - past->pos_vio.x;
          const double d_vio_n = in.pos_vio_enu.y - past->pos_vio.y;
          const double inc_err = std::hypot(
              (d_gnss_e - d_vio_e) - win_s * b_lag_e,
              (d_gnss_n - d_vio_n) - win_s * b_lag_n);
          const double sigma_w = 0.45 * std::sqrt(win_s);
          const double glrt = (inc_err * inc_err) / (sigma_w * sigma_w);
          if (glrt > 9.21) {
            h3b_corroborated = true;
          }
        }
      }

      if (rf_jam) {
        spoof_latched_ = true;
        hypothesis_ = SpoofHypothesis::H1_RF_JAMMING;
        rollback_applied_ = false;
        p_var_m2_ = std::max(p_var_m2_, 2.0 * 2.0);
        open_loop_dist_m_ = 0.0;
      } else if (step_meacon) {
        spoof_latched_ = true;
        hypothesis_ = SpoofHypothesis::H2_STEP_MEACONING;
        apply_vio_rollback(in);
        p_var_m2_ = std::max(p_var_m2_, 3.5 * 3.5);
        open_loop_dist_m_ = 0.0;
      } else if (mcusum_norm > h_t) {
        const bool water_ok = (wf <= 0.35) || h3b_corroborated || (in.cn0_dbhz < 32.0);
        if (water_ok) {
          spoof_latched_ = true;
          hypothesis_ = SpoofHypothesis::H3_SLOW_RAMP_SPOOF;
          apply_vio_rollback(in);
          p_var_m2_ = std::max(p_var_m2_, 3.5 * 3.5);
          open_loop_dist_m_ = 0.0;
        }
      }
    }

    // 4. Pose & horizontal covariance propagation + Joseph visual geo-reloc update
    core::Vec3Enu est_pos = in.pos_gnss_enu;
    if (spoof_latched_ || !in.gps_valid) {
      est_pos = {in.pos_vio_enu.x + vio_offset_e_, in.pos_vio_enu.y + vio_offset_n_, in.pos_vio_enu.z};
      const double speed = std::hypot(in.vel_vio_enu.x, in.vel_vio_enu.y);
      open_loop_dist_m_ += speed * dt;
      const double kappa = (wf > 0.35) ? cfg_.kappa_drift_water : cfg_.kappa_drift_land;
      const double d_step = speed * dt;
      p_var_m2_ += cfg_.q_rw_m2_per_s * dt + (2.0 * kappa * kappa * open_loop_dist_m_ * d_step);

      // 3-Gate XFeat+LighterGlue visual reloc acceptance + Joseph update
      if (in.visual_reloc.valid &&
          in.visual_reloc.inlier_count >= 25 &&
          in.visual_reloc.reproj_rmse_px < 2.5 &&
          in.visual_reloc.hull_area_ratio > 0.25 &&
          in.visual_reloc.meas_sigma_m < 40.0) {
        const double r_var = in.visual_reloc.meas_sigma_m * in.visual_reloc.meas_sigma_m;
        const double innov_e = in.visual_reloc.pos_enu.x - est_pos.x;
        const double innov_n = in.visual_reloc.pos_enu.y - est_pos.y;
        const double s_var = p_var_m2_ + r_var;
        const double maha2 = (innov_e * innov_e + innov_n * innov_n) / std::max(s_var, 1e-6);
        const bool megaloc_reinit = (std::sqrt(p_var_m2_) > 50.0);
        if (maha2 <= 9.21 || megaloc_reinit) {
          const double k_gain = p_var_m2_ / s_var;
          vio_offset_e_ += k_gain * innov_e;
          vio_offset_n_ += k_gain * innov_n;
          est_pos.x += k_gain * innov_e;
          est_pos.y += k_gain * innov_n;
          const double one_minus_k = 1.0 - k_gain;
          p_var_m2_ = (one_minus_k * one_minus_k) * p_var_m2_ + (k_gain * k_gain) * r_var;
          open_loop_dist_m_ = 0.0;
        }
      }
    } else {
      p_var_m2_ = 1.5 * 1.5;
      open_loop_dist_m_ = 0.0;
    }

    const double sigma_pos_m = std::sqrt(std::max(p_var_m2_, 0.25));
    const double three_sigma_m = 3.0 * sigma_pos_m;
    const double ctr_inflation_m = three_sigma_m + kMapBiasM;
    const double dh_dt_mps =
        has_prev_infl_ ? std::max(0.0, (ctr_inflation_m - prev_ctr_inflation_m_) / dt) : 0.0;
    prev_ctr_inflation_m_ = ctr_inflation_m;
    has_prev_infl_ = true;

    // 5. 3-State Navigation FSM with hysteresis & latched SLZ abort
    if (nav_state_ != NavFsmState::DETERMINISTIC_VISUAL_SLZ_ABORT) {
      if (sigma_pos_m >= cfg_.fsm_abort_sigma_m) {
        nav_state_ = NavFsmState::DETERMINISTIC_VISUAL_SLZ_ABORT;
      } else if (sigma_pos_m >= cfg_.fsm_caution_sigma_m) {
        nav_state_ = NavFsmState::DEGRADED_CAUTION;
      } else if (nav_state_ == NavFsmState::DEGRADED_CAUTION &&
                 sigma_pos_m < (cfg_.fsm_caution_sigma_m - 5.0)) {
        nav_state_ = NavFsmState::BOUNDED_GNSS_DENIED_CRUISE;
      }
    }

    // 6. Compute inflated LTFJ & verified DHMI AIP LTBX margins
    const double lat_deg = kKartalDatumLatDeg + (est_pos.y / 111052.0);
    const double lon_deg = kKartalDatumLonDeg + (est_pos.x / 84255.0);
    const double alt_amsl_m = 12.0 + est_pos.z;
    const double raw_ltfj_m =
        vertirta::rta::AirspaceExclusion::signed_margin_ltfj_m(lat_deg, lon_deg, alt_amsl_m);
    const double ltbx_de = (lon_deg - kVerifiedLtbxArpLonDeg) * 84255.0;
    const double ltbx_dn = (lat_deg - kVerifiedLtbxArpLatDeg) * 111052.0;
    const double raw_ltbx_m = std::hypot(ltbx_de, ltbx_dn) - kCtrRadiusM;

    GnssDeniedOutput out{};
    out.hypothesis = hypothesis_;
    out.nav_state = nav_state_;
    out.spoof_latched = spoof_latched_;
    out.gnss_fused_allowed = !spoof_latched_;
    out.rollback_applied = rollback_applied_;
    out.mcusum_norm = std::hypot(mcusum_e_, mcusum_n_);
    out.sigma_pos_m = sigma_pos_m;
    out.three_sigma_m = three_sigma_m;
    out.ctr_inflation_m = ctr_inflation_m;
    out.dh_dt_mps = dh_dt_mps;
    out.v_max_cap_mps =
        (nav_state_ == NavFsmState::BOUNDED_GNSS_DENIED_CRUISE)
            ? cfg_.v_max_nominal_mps
            : std::min(cfg_.v_max_caution_mps, 15.0);
    out.ltfj_inflated_margin_m = raw_ltfj_m - ctr_inflation_m;
    out.ltbx_inflated_margin_m = raw_ltbx_m - ctr_inflation_m;
    out.est_pos_enu = est_pos;
    out.est_vel_enu = spoof_latched_ ? in.vel_vio_enu : in.vel_gnss_enu;
    return out;
  }

  [[nodiscard]] std::string to_status_json(const GnssDeniedOutput& out) const {
    char buf[512];
    std::snprintf(
        buf,
        sizeof(buf),
        "{\"hypothesis\":\"%s\",\"nav_fsm_state\":\"%s\",\"spoof_latched\":%s,"
        "\"gnss_fused_allowed\":%s,\"rollback_applied\":%s,\"mcusum_norm\":%.3f,"
        "\"sigma_pos_m\":%.2f,\"three_sigma_m\":%.2f,\"ctr_inflation_m\":%.2f,"
        "\"dh_dt_mps\":%.4f,\"v_max_cap_mps\":%.2f,"
        "\"ltfj_inflated_margin_m\":%.2f,\"ltbx_inflated_margin_m\":%.2f,"
        "\"ekf2_gps_delay_ms\":%.1f,\"uxrce_dds_synct\":%d}",
        spoof_hypothesis_str(out.hypothesis),
        nav_fsm_state_str(out.nav_state),
        out.spoof_latched ? "true" : "false",
        out.gnss_fused_allowed ? "true" : "false",
        out.rollback_applied ? "true" : "false",
        out.mcusum_norm,
        out.sigma_pos_m,
        out.three_sigma_m,
        out.ctr_inflation_m,
        out.dh_dt_mps,
        out.v_max_cap_mps,
        out.ltfj_inflated_margin_m,
        out.ltbx_inflated_margin_m,
        cfg_.ekf2_gps_delay_ms,
        cfg_.uxrce_dds_synct_enabled ? 1 : 0);
    return std::string(buf);
  }

 private:
  struct HistorySample {
    double t_s{0.0};
    core::Vec3Enu pos_gnss{};
    core::Vec3Enu vel_gnss{};
    core::Vec3Enu pos_vio{};
    core::Vec3Enu vel_vio{};
  };

  [[nodiscard]] const HistorySample* find_sample_at_age(double now_s, double target_age_s) const noexcept {
    const HistorySample* best = nullptr;
    double best_err = 1e9;
    for (const auto& s : history_) {
      const double age = now_s - s.t_s;
      const double err = std::abs(age - target_age_s);
      if (err < best_err) {
        best_err = err;
        best = &s;
      }
    }
    return (best_err <= 0.35) ? best : nullptr;
  }

  void apply_vio_rollback(const GnssDeniedStepInput& in) noexcept {
    const HistorySample* anchor = find_sample_at_age(in.t_s, cfg_.rollback_window_s);
    if (anchor != nullptr) {
      vio_offset_e_ = anchor->pos_gnss.x - anchor->pos_vio.x;
      vio_offset_n_ = anchor->pos_gnss.y - anchor->pos_vio.y;
      rollback_applied_ = true;
    } else {
      vio_offset_e_ = 0.0;
      vio_offset_n_ = 0.0;
      rollback_applied_ = true;
    }
  }

  GnssDeniedConfig cfg_{};
  std::deque<HistorySample> history_{};
  SpoofHypothesis hypothesis_{SpoofHypothesis::H0_NOMINAL};
  NavFsmState nav_state_{NavFsmState::BOUNDED_GNSS_DENIED_CRUISE};
  bool spoof_latched_{false};
  bool rollback_applied_{false};
  double mcusum_e_{0.0};
  double mcusum_n_{0.0};
  double vio_offset_e_{0.0};
  double vio_offset_n_{0.0};
  double p_var_m2_{2.25};
  double open_loop_dist_m_{0.0};
  double prev_ctr_inflation_m_{0.0};
  bool has_prev_infl_{false};
};

}  // namespace vertirta::gnss_denied_node
