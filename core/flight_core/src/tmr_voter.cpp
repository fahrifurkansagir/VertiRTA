#include "vertirta/core/tmr_voter.hpp"

#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

namespace {

constexpr int kPairA[3] = {0, 0, 1};
constexpr int kPairB[3] = {1, 2, 2};

// For sensor i, pairs (kSensorPairs[i][0], kSensorPairs[i][1]) include i; kOppPair[i] excludes i.
constexpr int kSensorPairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
constexpr int kOppPair[3] = {2, 1, 0};

[[nodiscard]] inline bool is_healthy(uint8_t mask, int idx) noexcept {
  return (mask & static_cast<uint8_t>(1u << idx)) != 0u;
}

[[nodiscard]] inline int count_healthy(uint8_t mask) noexcept {
  int c = 0;
  for (int i = 0; i < 3; ++i) {
    if (is_healthy(mask, i)) {
      ++c;
    }
  }
  return c;
}

[[nodiscard]] inline bool sample_is_clean(const ImuSampleFrd& s) noexcept {
  return s.valid && s.gyro_rad_s.is_finite() && s.accel_m_s2.is_finite();
}

[[nodiscard]] inline double sum_ring(const double buf[TmrVoter::kWindowTicks]) noexcept {
  double s = 0.0;
  for (int k = 0; k < TmrVoter::kWindowTicks; ++k) {
    s += buf[k];
  }
  return s;
}

}  // namespace

TmrVoter::TmrVoter(const TmrConfig& cfg) noexcept : cfg_(cfg) {
  reset();
}

void TmrVoter::reset() noexcept {
  state_ = TmrState::TMR_3OF3;
  health_mask_ = 0x07;
  tick_count_ = 0;
  ring_idx_ = 0;
  simplex_drift_ticks_ = 0;
  last_isolation_latency_ms_ = 0;
  last_valid_gyro_ = {};
  last_valid_accel_ = {0.0, 0.0, -9.80665};
  for (int p = 0; p < 3; ++p) {
    bias_gyro_pair_[p] = {};
    bias_accel_pair_[p] = {};
    pair_onset_tick_[p] = 0;
    pair_onset_active_[p] = false;
    for (int k = 0; k < kWindowTicks; ++k) {
      q_gyro_ring_[p][k] = 0.0;
      q_accel_ring_[p][k] = 0.0;
    }
  }
}

TmrVoteOutput TmrVoter::step(
    const ImuSampleFrd imus[3],
    const Vec3Frd& model_gyro_pred,
    bool model_valid) noexcept {
  const uint64_t now_tick = tick_count_++;
  const double agree_lim = cfg_.analytical_agree_sigma * cfg_.sigma_gyro_eff;
  const double reject_lim = cfg_.analytical_reject_sigma * cfg_.sigma_gyro_eff;
  const bool model_ok = model_valid && model_gyro_pred.is_finite();

  // 1. Immediate monotonic latch on invalid or NaN/Inf IMU channels (Review Focus #1).
  bool hard_latched_this_tick = false;
  for (int i = 0; i < 3; ++i) {
    if (is_healthy(health_mask_, i) && !sample_is_clean(imus[i])) {
      health_mask_ = static_cast<uint8_t>(health_mask_ & ~(1u << i));
      hard_latched_this_tick = true;
    }
  }

  if (hard_latched_this_tick) {
    const int n_ok = count_healthy(health_mask_);
    if (n_ok == 2 && state_ == TmrState::TMR_3OF3) {
      state_ = TmrState::DUPLEX_2OF2;
      last_isolation_latency_ms_ = 0;
    } else if (n_ok == 1 && state_ < TmrState::SIMPLEX_ANALYTICAL) {
      int survivor = -1;
      for (int i = 0; i < 3; ++i) {
        if (is_healthy(health_mask_, i)) {
          survivor = i;
        }
      }
      const double err = (survivor >= 0 && model_ok)
                             ? (imus[survivor].gyro_rad_s - model_gyro_pred).norm()
                             : 1e9;
      state_ = (err <= agree_lim) ? TmrState::SIMPLEX_ANALYTICAL : TmrState::FAILSAFE_DIVERT;
      last_isolation_latency_ms_ = 0;
    } else if (n_ok == 0) {
      state_ = TmrState::FAILSAFE_DIVERT;
      last_isolation_latency_ms_ = 0;
    }
  }

  // 2. Pairwise parity residuals & 8-tick (20 ms) chi^2_24 sliding window.
  const double inv_var_g = 1.0 / (2.0 * cfg_.sigma_gyro_eff * cfg_.sigma_gyro_eff);
  const double inv_var_a = 1.0 / (2.0 * cfg_.sigma_accel_eff * cfg_.sigma_accel_eff);
  const double inst_half_thresh = 0.5 * (cfg_.chi2_24_threshold / static_cast<double>(kWindowTicks));

  double q_pair_win[3] = {0.0, 0.0, 0.0};
  double max_inst_q = 0.0;
  double max_win_q = 0.0;
  Vec3Frd dg_pair[3]{};
  Vec3Frd da_pair[3]{};

  for (int p = 0; p < 3; ++p) {
    const int a = kPairA[p];
    const int b = kPairB[p];
    if (is_healthy(health_mask_, a) && is_healthy(health_mask_, b)) {
      dg_pair[p] = (imus[a].gyro_rad_s - imus[b].gyro_rad_s) - bias_gyro_pair_[p];
      da_pair[p] = (imus[a].accel_m_s2 - imus[b].accel_m_s2) - bias_accel_pair_[p];
      const double qg = dg_pair[p].norm_sq() * inv_var_g;
      const double qa = da_pair[p].norm_sq() * inv_var_a;
      q_gyro_ring_[p][ring_idx_] = qg;
      q_accel_ring_[p][ring_idx_] = qa;

      const double q_inst = (qg > qa) ? qg : qa;
      if (q_inst > max_inst_q) {
        max_inst_q = q_inst;
      }
      if (q_inst > inst_half_thresh) {
        if (!pair_onset_active_[p]) {
          pair_onset_active_[p] = true;
          pair_onset_tick_[p] = now_tick;
        }
      } else {
        pair_onset_active_[p] = false;
      }

      const double qg_sum = sum_ring(q_gyro_ring_[p]);
      const double qa_sum = sum_ring(q_accel_ring_[p]);
      q_pair_win[p] = (qg_sum > qa_sum) ? qg_sum : qa_sum;
      if (q_pair_win[p] > max_win_q) {
        max_win_q = q_pair_win[p];
      }
    } else {
      q_gyro_ring_[p][ring_idx_] = 0.0;
      q_accel_ring_[p][ring_idx_] = 0.0;
      pair_onset_active_[p] = false;
    }
  }
  ring_idx_ = (ring_idx_ + 1) % kWindowTicks;

  // Freeze slow 0.02 Hz inter-IMU bias tracker whenever parity exceeds 50% threshold.
  if (max_win_q < 0.5 * cfg_.chi2_24_threshold && max_inst_q < inst_half_thresh) {
    const double alpha = 1.0 - std::exp(-kTwoPi * cfg_.bias_tracker_hz * cfg_.dt_s);
    for (int p = 0; p < 3; ++p) {
      if (is_healthy(health_mask_, kPairA[p]) && is_healthy(health_mask_, kPairB[p])) {
        bias_gyro_pair_[p] = bias_gyro_pair_[p] + dg_pair[p] * alpha;
        bias_accel_pair_[p] = bias_accel_pair_[p] + da_pair[p] * alpha;
      }
    }
  }

  // 3. Monotonic FDIR state transitions.
  if (state_ == TmrState::TMR_3OF3) {
    int isolated = -1;
    for (int i = 0; i < 3; ++i) {
      const int p1 = kSensorPairs[i][0];
      const int p2 = kSensorPairs[i][1];
      const int po = kOppPair[i];
      if (q_pair_win[p1] > cfg_.chi2_24_threshold &&
          q_pair_win[p2] > cfg_.chi2_24_threshold &&
          q_pair_win[po] <= cfg_.chi2_24_threshold) {
        isolated = i;
        const uint64_t onset = (pair_onset_active_[p1] && pair_onset_active_[p2])
                                   ? ((pair_onset_tick_[p1] < pair_onset_tick_[p2])
                                          ? pair_onset_tick_[p1]
                                          : pair_onset_tick_[p2])
                                   : now_tick;
        const double lat_ms = static_cast<double>(now_tick - onset + 1ULL) * cfg_.dt_s * 1000.0;
        last_isolation_latency_ms_ = static_cast<uint16_t>(clamp_d(lat_ms + 0.5, 0.0, 65535.0));
        break;
      }
    }
    if (isolated >= 0) {
      health_mask_ = static_cast<uint8_t>(health_mask_ & ~(1u << isolated));
      state_ = TmrState::DUPLEX_2OF2;
    } else if (q_pair_win[0] > cfg_.chi2_24_threshold &&
               q_pair_win[1] > cfg_.chi2_24_threshold &&
               q_pair_win[2] > cfg_.chi2_24_threshold) {
      state_ = TmrState::FAILSAFE_DIVERT;
      last_isolation_latency_ms_ = static_cast<uint16_t>(kWindowTicks * cfg_.dt_s * 1000.0 + 0.5);
    }
  } else if (state_ == TmrState::DUPLEX_2OF2) {
    for (int p = 0; p < 3; ++p) {
      const int a = kPairA[p];
      const int b = kPairB[p];
      if (is_healthy(health_mask_, a) && is_healthy(health_mask_, b)) {
        if (q_pair_win[p] > cfg_.chi2_24_threshold) {
          const uint64_t onset = pair_onset_active_[p] ? pair_onset_tick_[p] : now_tick;
          const double lat_ms = static_cast<double>(now_tick - onset + 1ULL) * cfg_.dt_s * 1000.0;
          last_isolation_latency_ms_ = static_cast<uint16_t>(clamp_d(lat_ms + 0.5, 0.0, 65535.0));

          if (model_ok) {
            const double err_a = (imus[a].gyro_rad_s - model_gyro_pred).norm();
            const double err_b = (imus[b].gyro_rad_s - model_gyro_pred).norm();
            if (err_a <= agree_lim && err_b > reject_lim) {
              health_mask_ = static_cast<uint8_t>(health_mask_ & ~(1u << b));
              state_ = TmrState::SIMPLEX_ANALYTICAL;
            } else if (err_b <= agree_lim && err_a > reject_lim) {
              health_mask_ = static_cast<uint8_t>(health_mask_ & ~(1u << a));
              state_ = TmrState::SIMPLEX_ANALYTICAL;
            } else {
              state_ = TmrState::FAILSAFE_DIVERT;
            }
          } else {
            state_ = TmrState::FAILSAFE_DIVERT;
          }
        }
        break;
      }
    }
  } else if (state_ == TmrState::SIMPLEX_ANALYTICAL) {
    int survivor = -1;
    for (int i = 0; i < 3; ++i) {
      if (is_healthy(health_mask_, i) && sample_is_clean(imus[i])) {
        survivor = i;
        break;
      }
    }
    if (survivor < 0) {
      state_ = TmrState::FAILSAFE_DIVERT;
    } else if (model_ok) {
      const double err = (imus[survivor].gyro_rad_s - model_gyro_pred).norm();
      if (err > reject_lim) {
        if (++simplex_drift_ticks_ >= kWindowTicks) {
          health_mask_ = static_cast<uint8_t>(health_mask_ & ~(1u << survivor));
          state_ = TmrState::FAILSAFE_DIVERT;
          last_isolation_latency_ms_ = static_cast<uint16_t>(kWindowTicks * cfg_.dt_s * 1000.0 + 0.5);
        }
      } else {
        simplex_drift_ticks_ = 0;
      }
    }
  }

  // 4. Compute voted IMU output according to active redundancy tier.
  TmrVoteOutput out{};
  out.state = state_;
  out.health_mask = health_mask_;
  out.max_chi2_residual = static_cast<float>(max_win_q);
  out.isolation_latency_ms = last_isolation_latency_ms_;

  if (state_ == TmrState::TMR_3OF3) {
    out.gyro_voted = {
        median3_d(imus[0].gyro_rad_s.x, imus[1].gyro_rad_s.x, imus[2].gyro_rad_s.x),
        median3_d(imus[0].gyro_rad_s.y, imus[1].gyro_rad_s.y, imus[2].gyro_rad_s.y),
        median3_d(imus[0].gyro_rad_s.z, imus[1].gyro_rad_s.z, imus[2].gyro_rad_s.z),
    };
    out.accel_voted = {
        median3_d(imus[0].accel_m_s2.x, imus[1].accel_m_s2.x, imus[2].accel_m_s2.x),
        median3_d(imus[0].accel_m_s2.y, imus[1].accel_m_s2.y, imus[2].accel_m_s2.y),
        median3_d(imus[0].accel_m_s2.z, imus[1].accel_m_s2.z, imus[2].accel_m_s2.z),
    };
    last_valid_gyro_ = out.gyro_voted;
    last_valid_accel_ = out.accel_voted;
  } else if (state_ == TmrState::DUPLEX_2OF2) {
    Vec3Frd g_sum{};
    Vec3Frd a_sum{};
    int cnt = 0;
    for (int i = 0; i < 3; ++i) {
      if (is_healthy(health_mask_, i) && sample_is_clean(imus[i])) {
        g_sum = g_sum + imus[i].gyro_rad_s;
        a_sum = a_sum + imus[i].accel_m_s2;
        ++cnt;
      }
    }
    if (cnt > 0) {
      const double inv_cnt = 1.0 / static_cast<double>(cnt);
      out.gyro_voted = g_sum * inv_cnt;
      out.accel_voted = a_sum * inv_cnt;
      last_valid_gyro_ = out.gyro_voted;
      last_valid_accel_ = out.accel_voted;
    } else {
      out.gyro_voted = last_valid_gyro_;
      out.accel_voted = last_valid_accel_;
    }
  } else if (state_ == TmrState::SIMPLEX_ANALYTICAL) {
    bool found = false;
    for (int i = 0; i < 3; ++i) {
      if (is_healthy(health_mask_, i) && sample_is_clean(imus[i])) {
        out.gyro_voted = imus[i].gyro_rad_s;
        out.accel_voted = imus[i].accel_m_s2;
        last_valid_gyro_ = out.gyro_voted;
        last_valid_accel_ = out.accel_voted;
        found = true;
        break;
      }
    }
    if (!found) {
      out.gyro_voted = model_ok ? model_gyro_pred : last_valid_gyro_;
      out.accel_voted = last_valid_accel_;
    }
  } else {
    // FAILSAFE_DIVERT: guarantee finite output using analytical prediction or last valid sample.
    out.gyro_voted = model_ok ? model_gyro_pred : last_valid_gyro_;
    out.accel_voted = last_valid_accel_;
  }

  return out;
}

}  // namespace vertirta::core
