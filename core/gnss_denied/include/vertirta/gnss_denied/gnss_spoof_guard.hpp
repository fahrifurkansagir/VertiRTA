#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/frames.hpp"
#include "vertirta/gnss_denied/basalt_vio_adapter.hpp"

namespace vertirta::gnss_denied {

// Canonical PX4 v1.16 GNSS fix contract (01-DR08-13 Override #4: /fmu/out/vehicle_gps_position + EKF2_GPS_DELAY).
struct GnssFixV116 {
  uint64_t stamp_ns{0};
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  double cn0_avg_dbhz{40.0};
  double agc_ratio{0.50};     // Nominal ~0.4-0.7; <0.15 or >0.95 indicates RF jamming/saturation
  double hdop{0.9};
  double eph_m{1.2};
  double epv_m{1.8};
  double ekf2_gps_delay_ms{110.0};
  uint8_t fix_type{3};        // >= 3 for valid 3D fix
  bool rf_jam_flag{false};
  bool rf_spoof_flag{false};
  bool valid{true};
};

enum class GnssHypothesis : uint8_t {
  kH0Nominal = 0,
  kH1RfJamming = 1,
  kH2StepMeaconing = 2,
  kH3SlowRampSpoof = 3,
};

struct GnssSpoofGuardConfig {
  double rate_hz{10.0};
  std::size_t lag_baseline_epochs_l2{150};  // 15.0 s at 10 Hz
  std::size_t guard_band_epochs_l1{50};     // 5.0 s guard band at 10 Hz
  std::size_t rollback_epochs{20};          // 2.0 s pose rollback at 10 Hz

  // Monitor A (H1 RF Jamming, TTD < 0.2 s)
  double h1_cn0_min_dbhz{24.0};
  double h1_cn0_drop_db{10.0};
  double h1_hdop_max{2.5};

  // Monitor B (H2 Step/Meaconing, TTD < 0.3 s)
  std::size_t h2_window_epochs{5};
  double h2_chi2_10dof_thresh{23.21};
  double h2_step_jump_m{4.0};
  double h2_pos_sigma_m{0.85};

  // Monitor C1 (H3a Lagged-Baseline Bias-Compensated 2D Vector MCUSUM, DR-10 Table 2.3)
  double land_water_frac_max{0.20};
  double open_water_frac_min{0.35};
  double land_k_d_mps{0.25};
  double land_h_t_mps{2.0};
  double water_k_d_mps{0.35};
  double water_h_t_mps{3.4};

  // Monitor C2 (H3b Telescoping 2D Position-Increment GLRT, 1.5 s & 4.0 s windows)
  std::size_t glrt_short_epochs{15};  // 1.5 s
  std::size_t glrt_long_epochs{40};   // 4.0 s
  double glrt_short_thresh_m{1.70};
  double glrt_long_thresh_m{2.56};

  // Post-latch uncertainty initialization
  double sigma_init_spoof_m{3.5};
  double sigma_init_jam_m{2.0};
};

struct GnssSpoofGuardStatus {
  uint64_t stamp_ns{0};
  GnssHypothesis hypothesis{GnssHypothesis::kH0Nominal};
  bool gnss_fused_allowed{true};
  bool latched{false};
  bool rollback_executed{false};

  double mcusum_norm_mps{0.0};
  double active_k_d_mps{0.25};
  double active_h_t_mps{2.0};
  double h2_chi2_10dof{0.0};
  double h3b_glrt_short_m{0.0};
  double h3b_glrt_long_m{0.0};
  double water_fraction{0.0};

  // On latch: pre-detection pose from (t_latch - 2.0 s) propagated forward exclusively by decoupled VIO deltas.
  core::Vec3Ned rollback_pos_ned{};
  double sigma_init_m{0.0};
};

// 10 Hz 4-Hypothesis (H0..H3) Anti-Spoofing & RF Jamming Guard (zero heap, noexcept).
class GnssSpoofGuard {
 public:
  static constexpr std::size_t kMaxHistory = 200;  // >= L2 (150) + safety margin

  explicit GnssSpoofGuard(const GnssSpoofGuardConfig& cfg = GnssSpoofGuardConfig{}) noexcept;

  void reset() noexcept;

  [[nodiscard]] GnssSpoofGuardStatus step_10hz(
      const GnssFixV116& gnss,
      const VioStateEstimate& vio) noexcept;

  [[nodiscard]] bool is_latched() const noexcept { return latched_; }
  [[nodiscard]] const GnssSpoofGuardConfig& config() const noexcept { return cfg_; }

 private:
  struct EpochRecord {
    uint64_t stamp_ns{0};
    core::Vec3Ned gnss_pos_ned{};
    core::Vec3Ned vio_pos_ned{};
    double v_res_n{0.0};  // v_gnss_n - v_vio_n
    double v_res_e{0.0};  // v_gnss_e - v_vio_e
    double cn0_dbhz{40.0};
    bool valid{false};
  };

  [[nodiscard]] const EpochRecord& history_back(std::size_t steps_ago) const noexcept;

  GnssSpoofGuardConfig cfg_{};
  std::array<EpochRecord, kMaxHistory> ring_{};
  std::size_t write_idx_{0};
  std::size_t count_{0};

  // Crosier 2D vector MCUSUM state (m/s units per DR-10 Table 2.3)
  double mcusum_n_{0.0};
  double mcusum_e_{0.0};

  // Last 5-epoch step-innovation window for H2 chi^2
  std::array<double, 5> h2_sq_ring_{};
  std::size_t h2_count_{0};
  bool has_prev_pos_{false};
  core::Vec3Ned prev_gnss_pos_{};
  core::Vec3Ned prev_vio_pos_{};

  bool latched_{false};
  GnssHypothesis latched_hyp_{GnssHypothesis::kH0Nominal};
  double latched_sigma_init_m_{0.0};
  core::Vec3Ned latched_rollback_anchor_ned_{};
  core::Vec3Ned latched_vio_anchor_ned_{};
};

}  // namespace vertirta::gnss_denied
