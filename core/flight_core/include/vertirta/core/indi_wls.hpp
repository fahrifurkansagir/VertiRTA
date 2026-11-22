#pragma once

#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"

namespace vertirta::core {

struct WlsResult {
  double x[8]{};         // Allocated delta_u [N]
  int8_t W[8]{};         // Working set: -1 lower, +1 upper, 0 free
  uint8_t iters{0};
  uint8_t sat_mask{0};
  bool cholesky_failed{false};
};

// Zero-heap freestanding f64 Active-Set WLS Allocator (DR-1 Erratum #1: f64 Cholesky + QR fallback).
[[nodiscard]] WlsResult wls_alloc_8x4_f64(
    const double G[4][8],
    const double v[4],
    const double xmin[8],
    const double xmax[8],
    const double Wv[4],
    const double Wu[8],
    const double ud_minus_u0[8],
    double gamma,
    int8_t W_warm[8],
    uint8_t max_iter = 12) noexcept;

struct IndiInput {
  QuatPx4FrdNed q_frd_ned{};
  Vec3Frd gyro_frd{};
  Vec3Ned accel_des_ned{};
  double yaw_rate_des{0.0};
  float v_horiz_mps{0.0f};
  float v_term_v{403.2f};
  uint8_t fault_mask{0x00};
  const float* esc_omega_rad_s{nullptr};  // Optional 8-element ESC RPM feedback
  const bool* esc_valid{nullptr};         // Optional 8-element validity flags
};

// 400 Hz Reduced-Attitude (S^2) + Synchronized 8 Hz Butterworth INDI + f64 WLS + Option B1b Spin.
class IndiWlsController {
 public:
  explicit IndiWlsController(bool allow_active_spin = true) noexcept;

  void reset(double initial_omega_rad_s = params::kOmegaHover) noexcept;
  void set_allow_active_spin(bool allow) noexcept { allow_active_spin_ = allow; }

  [[nodiscard]] WlsResult step_400hz(const IndiInput& in, float omega_cmd_out[8]) noexcept;

  [[nodiscard]] float spin_rate_ref_rad_s() const noexcept { return spin_rate_ref_; }
  [[nodiscard]] float phase_lead_deg() const noexcept { return phase_lead_deg_; }
  [[nodiscard]] Vec3Frd filtered_gyro_frd() const noexcept { return gyro_f_; }
  [[nodiscard]] Vec3Frd filtered_dgyro_frd() const noexcept { return dgyro_f_; }
  void get_omega_hat(float out[8]) const noexcept;

 private:
  struct BiquadState {
    double x1{0.0}, x2{0.0}, y1{0.0}, y2{0.0};
    void reset(double val) noexcept { x1 = x2 = y1 = y2 = val; }
    [[nodiscard]] double step(double x) noexcept;
  };

  bool allow_active_spin_{true};
  bool initialized_{false};
  float spin_rate_ref_{0.0f};
  float phase_lead_deg_{0.0f};

  BiquadState gyro_lpf_[3]{};
  BiquadState thrust_lpf_[8]{};
  Vec3Frd gyro_f_{};
  Vec3Frd gyro_f_prev_{};
  Vec3Frd dgyro_f_{};

  double omega_hat_[8]{};
  double omega_cmd_delay_[2][8]{};
  int delay_idx_{0};
  int8_t wls_warm_[8]{};
};

}  // namespace vertirta::core
