#pragma once

#include <cstdint>

#include "vertirta/core/frames.hpp"

namespace vertirta::core {

enum class TmrState : uint8_t {
  TMR_3OF3 = 0,
  DUPLEX_2OF2 = 1,
  SIMPLEX_ANALYTICAL = 2,
  FAILSAFE_DIVERT = 3,
};

struct ImuSampleFrd {
  Vec3Frd gyro_rad_s{};
  Vec3Frd accel_m_s2{};
  uint64_t t_us{0};
  bool valid{true};
};

struct TmrVoteOutput {
  Vec3Frd gyro_voted{};
  Vec3Frd accel_voted{};
  TmrState state{TmrState::TMR_3OF3};
  uint8_t health_mask{0x07};
  float max_chi2_residual{0.0f};
  uint16_t isolation_latency_ms{0};
};

struct TmrConfig {
  double dt_s{0.0025};                 // 400 Hz tick period
  double sigma_gyro_eff{0.015};        // [rad/s] post-notch vibration floor
  double sigma_accel_eff{0.25};        // [m/s^2] post-notch vibration floor
  double chi2_24_threshold{72.23};     // P_FA = 1e-6 for 24-DOF (8 ticks x 3 axes)
  double bias_tracker_hz{0.02};        // Slow inter-IMU bias tracker cutoff
  double analytical_agree_sigma{3.0};  // 3-sigma agreement for SIMPLEX_ANALYTICAL
  double analytical_reject_sigma{6.0}; // 6-sigma outlier rejection for SIMPLEX_ANALYTICAL
};

// Zero-heap, freestanding C++17 400 Hz 3-IMU MVS + chi^2_24 TMR FDIR Voter (DR-1 Section 5.3).
class TmrVoter {
 public:
  static constexpr int kWindowTicks = 8;  // 8 x 2.5 ms = 20.0 ms isolation window

  explicit TmrVoter(const TmrConfig& cfg = TmrConfig{}) noexcept;

  void reset() noexcept;

  [[nodiscard]] TmrVoteOutput step(
      const ImuSampleFrd imus[3],
      const Vec3Frd& model_gyro_pred,
      bool model_valid) noexcept;

  [[nodiscard]] TmrState state() const noexcept { return state_; }
  [[nodiscard]] uint8_t health_mask() const noexcept { return health_mask_; }

 private:
  TmrConfig cfg_{};
  TmrState state_{TmrState::TMR_3OF3};
  uint8_t health_mask_{0x07};
  uint64_t tick_count_{0};
  int ring_idx_{0};

  double q_gyro_ring_[3][kWindowTicks]{};
  double q_accel_ring_[3][kWindowTicks]{};
  Vec3Frd bias_gyro_pair_[3]{};
  Vec3Frd bias_accel_pair_[3]{};

  uint64_t pair_onset_tick_[3]{};
  bool pair_onset_active_[3]{false, false, false};
  int simplex_drift_ticks_{0};

  uint16_t last_isolation_latency_ms_{0};
  Vec3Frd last_valid_gyro_{};
  Vec3Frd last_valid_accel_{0.0, 0.0, -9.80665};
};

}  // namespace vertirta::core
