#include "vertirta/gnss_denied/gnss_spoof_guard.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::gnss_denied {

GnssSpoofGuard::GnssSpoofGuard(const GnssSpoofGuardConfig& cfg) noexcept : cfg_(cfg) {}

void GnssSpoofGuard::reset() noexcept {
  for (auto& r : ring_) {
    r = EpochRecord{};
  }
  write_idx_ = 0u;
  count_ = 0u;
  mcusum_n_ = 0.0;
  mcusum_e_ = 0.0;
  h2_sq_ring_.fill(0.0);
  h2_count_ = 0u;
  has_prev_pos_ = false;
  prev_gnss_pos_ = {};
  prev_vio_pos_ = {};
  latched_ = false;
  latched_hyp_ = GnssHypothesis::kH0Nominal;
  latched_sigma_init_m_ = 0.0;
  latched_rollback_anchor_ned_ = {};
  latched_vio_anchor_ned_ = {};
}

const GnssSpoofGuard::EpochRecord& GnssSpoofGuard::history_back(std::size_t steps_ago) const noexcept {
  // steps_ago == 0 returns the most recently written record at (write_idx_ - 1).
  const std::size_t offset = (steps_ago + 1u) % kMaxHistory;
  const std::size_t idx = (write_idx_ + kMaxHistory - offset) % kMaxHistory;
  return ring_[idx];
}

GnssSpoofGuardStatus GnssSpoofGuard::step_10hz(
    const GnssFixV116& gnss,
    const VioStateEstimate& vio) noexcept {
  GnssSpoofGuardStatus st{};
  st.stamp_ns = gnss.stamp_ns > 0ULL ? gnss.stamp_ns : vio.stamp_ns;
  st.water_fraction = std::clamp(vio.water_fraction, 0.0, 1.0);

  // If already latched, stay strictly GNSS-denied and propagate rollback pose purely via VIO deltas.
  if (latched_) {
    st.hypothesis = latched_hyp_;
    st.gnss_fused_allowed = false;
    st.latched = true;
    st.rollback_executed = (latched_hyp_ != GnssHypothesis::kH1RfJamming);
    st.sigma_init_m = latched_sigma_init_m_;
    st.rollback_pos_ned = latched_rollback_anchor_ned_ + (vio.pos_ned - latched_vio_anchor_ned_);
    st.mcusum_norm_mps = std::hypot(mcusum_n_, mcusum_e_);
    return st;
  }

  // Adaptive sea/land MCUSUM parameters (DR-10 Table 2.3):
  //   Land (<20% water):  k_d = 0.25 m/s, h_t = 2.0 m/s
  //   Water (>35% water): k_d = 0.35 m/s, h_t = 3.4 m/s (+ corroboration required)
  double water_blend = 0.0;
  if (st.water_fraction > cfg_.land_water_frac_max) {
    const double span = std::max(1e-3, cfg_.open_water_frac_min - cfg_.land_water_frac_max);
    water_blend = std::clamp((st.water_fraction - cfg_.land_water_frac_max) / span, 0.0, 1.0);
  }
  const double k_d = cfg_.land_k_d_mps + water_blend * (cfg_.water_k_d_mps - cfg_.land_k_d_mps);
  const double h_t = cfg_.land_h_t_mps + water_blend * (cfg_.water_h_t_mps - cfg_.land_h_t_mps);
  const bool over_open_water = (st.water_fraction > cfg_.open_water_frac_min);
  st.active_k_d_mps = k_d;
  st.active_h_t_mps = h_t;

  // --- Monitor A (H1: RF Jamming / Interference, TTD <= 0.2 s) ---
  bool cn0_rapid_drop = false;
  if (count_ >= 2u) {
    const double prev_cn0 = history_back(1u).cn0_dbhz;
    cn0_rapid_drop = (prev_cn0 - gnss.cn0_avg_dbhz) >= cfg_.h1_cn0_drop_db;
  }
  const bool agc_anomaly = (gnss.agc_ratio < 0.15 || gnss.agc_ratio > 0.95);
  const bool rf_anomaly = (!gnss.valid) || (gnss.fix_type < 3u) || gnss.rf_jam_flag ||
                          gnss.rf_spoof_flag || (gnss.cn0_avg_dbhz < cfg_.h1_cn0_min_dbhz) ||
                          cn0_rapid_drop || agc_anomaly || (gnss.hdop > cfg_.h1_hdop_max);

  // Record current epoch in 15 s circular buffer.
  EpochRecord rec{};
  rec.stamp_ns = st.stamp_ns;
  rec.gnss_pos_ned = gnss.pos_ned;
  rec.vio_pos_ned = vio.pos_ned;
  rec.v_res_n = gnss.vel_ned.x - vio.vel_ned.x;
  rec.v_res_e = gnss.vel_ned.y - vio.vel_ned.y;
  rec.cn0_dbhz = gnss.cn0_avg_dbhz;
  rec.valid = gnss.valid && vio.valid;

  ring_[write_idx_] = rec;
  write_idx_ = (write_idx_ + 1u) % kMaxHistory;
  if (count_ < kMaxHistory) {
    ++count_;
  }

  // --- Monitor B (H2: Step / Meaconing 5-epoch chi^2 & single-epoch jump, TTD <= 0.3 s) ---
  bool h2_triggered = false;
  if (has_prev_pos_ && gnss.valid && vio.valid) {
    const double dgn_n = gnss.pos_ned.x - prev_gnss_pos_.x;
    const double dgn_e = gnss.pos_ned.y - prev_gnss_pos_.y;
    const double dvio_n = vio.pos_ned.x - prev_vio_pos_.x;
    const double dvio_e = vio.pos_ned.y - prev_vio_pos_.y;
    const double jump_n = dgn_n - dvio_n;
    const double jump_e = dgn_e - dvio_e;
    const double jump_m = std::hypot(jump_n, jump_e);

    const double inv_var = 1.0 / std::max(0.05, cfg_.h2_pos_sigma_m * cfg_.h2_pos_sigma_m);
    const double ep_chi2 = (jump_n * jump_n + jump_e * jump_e) * inv_var;
    h2_sq_ring_[h2_count_ % 5u] = ep_chi2;
    ++h2_count_;

    double sum_chi2 = 0.0;
    const std::size_t n_win = std::min<std::size_t>(h2_count_, 5u);
    for (std::size_t i = 0; i < n_win; ++i) {
      sum_chi2 += h2_sq_ring_[i];
    }
    st.h2_chi2_10dof = sum_chi2;

    if (jump_m > cfg_.h2_step_jump_m || (n_win >= 5u && sum_chi2 > cfg_.h2_chi2_10dof_thresh)) {
      h2_triggered = true;
    }
  }
  if (gnss.valid && vio.valid) {
    prev_gnss_pos_ = gnss.pos_ned;
    prev_vio_pos_ = vio.pos_ned;
    has_prev_pos_ = true;
  }

  // --- Monitor C1 (H3a: Lagged-Baseline [t-15s, t-5s] Bias-Compensated 2D Vector MCUSUM) ---
  double b_lag_n = 0.0;
  double b_lag_e = 0.0;
  if (count_ > cfg_.guard_band_epochs_l1 + 1u) {
    const std::size_t lag_start = cfg_.guard_band_epochs_l1;
    const std::size_t lag_end = std::min(count_, cfg_.lag_baseline_epochs_l2);
    double sum_n = 0.0;
    double sum_e = 0.0;
    std::size_t n_lag = 0u;
    for (std::size_t ago = lag_start; ago < lag_end; ++ago) {
      const EpochRecord& old_r = history_back(ago);
      if (old_r.valid) {
        sum_n += old_r.v_res_n;
        sum_e += old_r.v_res_e;
        ++n_lag;
      }
    }
    if (n_lag > 0u) {
      b_lag_n = sum_n / static_cast<double>(n_lag);
      b_lag_e = sum_e / static_cast<double>(n_lag);
    }
  }

  const double r_n = rec.v_res_n - b_lag_n;
  const double r_e = rec.v_res_e - b_lag_e;
  const double un = mcusum_n_ + r_n;
  const double ue = mcusum_e_ + r_e;
  const double norm_u = std::hypot(un, ue);
  if (norm_u > k_d) {
    const double scale = (norm_u - k_d) / norm_u;
    mcusum_n_ = un * scale;
    mcusum_e_ = ue * scale;
  } else {
    mcusum_n_ = 0.0;
    mcusum_e_ = 0.0;
  }
  st.mcusum_norm_mps = std::hypot(mcusum_n_, mcusum_e_);

  // --- Monitor C2 (H3b: Telescoping 1.5 s / 4.0 s Unbiased 2D Position-Increment GLRT) ---
  auto compute_glrt_m = [&](std::size_t win_epochs) noexcept -> double {
    if (count_ <= win_epochs) {
      return 0.0;
    }
    const EpochRecord& prev_w = history_back(win_epochs);
    if (!prev_w.valid || !rec.valid) {
      return 0.0;
    }
    const double dt_w = static_cast<double>(win_epochs) / std::max(cfg_.rate_hz, 1.0);
    const double dn = (rec.gnss_pos_ned.x - prev_w.gnss_pos_ned.x) -
                      (rec.vio_pos_ned.x - prev_w.vio_pos_ned.x) - dt_w * b_lag_n;
    const double de = (rec.gnss_pos_ned.y - prev_w.gnss_pos_ned.y) -
                      (rec.vio_pos_ned.y - prev_w.vio_pos_ned.y) - dt_w * b_lag_e;
    return std::hypot(dn, de);
  };

  st.h3b_glrt_short_m = compute_glrt_m(cfg_.glrt_short_epochs);
  st.h3b_glrt_long_m = compute_glrt_m(cfg_.glrt_long_epochs);

  const bool h3a_hit = (st.mcusum_norm_mps > h_t);
  const bool h3b_short_hit = (st.h3b_glrt_short_m > cfg_.glrt_short_thresh_m);
  const bool h3b_long_hit = (st.h3b_glrt_long_m > cfg_.glrt_long_thresh_m);
  const bool h3b_hit = h3b_short_hit || h3b_long_hit;

  // Over open Marmara Sea water (>35%), require corroboration so wave-glint drift never causes FAR floods.
  const bool h3_triggered = over_open_water
                                ? ((h3a_hit && (h3b_hit || rf_anomaly)) ||
                                   (h3b_short_hit && h3b_long_hit))
                                : (h3a_hit || h3b_hit);

  // Priority arbitration: H1 RF jamming -> H2 Step/Meaconing -> H3 Slow-Ramp Spoof.
  if (rf_anomaly || h2_triggered || h3_triggered) {
    latched_ = true;
    if (rf_anomaly && !gnss.rf_spoof_flag && !h2_triggered && !h3_triggered) {
      latched_hyp_ = GnssHypothesis::kH1RfJamming;
      latched_sigma_init_m_ = cfg_.sigma_init_jam_m;
      // Clean RF jam: anchor at last clean GNSS pose (1 epoch back if available), no 2.0 s spoof rewind needed.
      const EpochRecord& clean_ref = (count_ >= 2u) ? history_back(1u) : rec;
      latched_rollback_anchor_ned_ = clean_ref.gnss_pos_ned;
      latched_vio_anchor_ned_ = clean_ref.vio_pos_ned;
      st.rollback_executed = false;
    } else {
      latched_hyp_ = h2_triggered ? GnssHypothesis::kH2StepMeaconing : GnssHypothesis::kH3SlowRampSpoof;
      latched_sigma_init_m_ = cfg_.sigma_init_spoof_m;
      // Execute 2.0 s pose ring-buffer rollback to strip pre-detection spoofing bias!
      const std::size_t rb_steps = std::min(cfg_.rollback_epochs, count_ - 1u);
      const EpochRecord& rb_ref = history_back(rb_steps);
      latched_rollback_anchor_ned_ = rb_ref.gnss_pos_ned;
      latched_vio_anchor_ned_ = rb_ref.vio_pos_ned;
      st.rollback_executed = true;
    }

    st.hypothesis = latched_hyp_;
    st.gnss_fused_allowed = false;
    st.latched = true;
    st.sigma_init_m = latched_sigma_init_m_;
    st.rollback_pos_ned = latched_rollback_anchor_ned_ + (vio.pos_ned - latched_vio_anchor_ned_);
    return st;
  }

  st.hypothesis = GnssHypothesis::kH0Nominal;
  st.gnss_fused_allowed = true;
  st.latched = false;
  st.rollback_pos_ned = gnss.pos_ned;
  return st;
}

}  // namespace vertirta::gnss_denied
