// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

#include "vertirta/airframes/lift_cruise_dynamics.hpp"

namespace vertirta::airframes {

enum class TransitionFaultMode : uint8_t {
  kNominal = 0,
  kLiftRotorOei = 1,
  kPusherLossGlide = 2,
  kPusherLossHoverAbort = 3,
  kStallRecovery = 4,
};

struct Wls12Result {
  std::array<double, kMaxActuators> x{};
  std::array<int8_t, kMaxActuators> W{};
  uint8_t iters{0u};
  uint8_t sat_mask_lo{0u};
  bool cholesky_fallback{false};
};

// C^2-continuous quintic Hermite hover->cruise blend schedule across [v_blend_lo, v_blend_hi].
// Returns 0 for V_a <= 14 m/s, 0.5 at 21 m/s, 1.0 for V_a >= 28 m/s, with zero 1st & 2nd derivatives at endpoints.
[[nodiscard]] double quintic_hermite_blend(
    double v_airspeed_mps,
    double v_blend_lo = LiftCruiseDynamics::kBlendStartMps,
    double v_blend_hi = LiftCruiseDynamics::kBlendEndMps) noexcept;

// Zero-heap IEEE-754 f64 active-set WLS allocator solving up to 17x12 stacked Harkegard objective:
//   min_{du_min <= du <= du_max} gamma * ||W_v (B * du - v_req)||_2^2 + ||W_u (du - ud_minus_u0)||_2^2
[[nodiscard]] Wls12Result wls_alloc_12x5_f64(
    const std::array<std::array<double, kMaxActuators>, kMaxControls>& B,
    const std::array<double, kMaxControls>& v_req,
    const std::array<double, kMaxActuators>& du_min,
    const std::array<double, kMaxActuators>& du_max,
    const std::array<double, kMaxControls>& Wv,
    const std::array<double, kMaxActuators>& Wu,
    const std::array<double, kMaxActuators>& ud_minus_u0,
    double gamma,
    const std::array<int8_t, kMaxActuators>& W_warm,
    uint8_t n_actuators,
    uint8_t max_iter = 12u) noexcept;

struct TransitionCommand {
  double target_airspeed_mps{0.0};
  double target_alt_m{100.0};
  double target_heading_rad{0.0};
  uint16_t fault_mask{0u};
  // Optional PX4 v1.16 bidirectional DShot ESC_STATUS feedback [N] (nullptr uses internal lag model)
  const double* esc_thrust_n{nullptr};
  const bool* esc_valid{nullptr};
};

struct TransitionOutput {
  std::array<double, kMaxActuators> u_cmd{};
  std::array<double, kMaxControls> nu_hedge{};  // Pseudo-Control Hedging residual
  double eta_blend{0.0};
  double airspeed_mps{0.0};
  double alpha_rad{0.0};
  double glide_target_airspeed_mps{0.0};
  double estimated_ld_ratio{0.0};
  TransitionFaultMode fault_mode{TransitionFaultMode::kNominal};
  uint8_t wls_iters{0u};
  bool rotors_locked_cruise{false};
  bool aoa_barrier_active{false};
  bool valid{false};
};

// Zero-heap 400 Hz actuator-synchronized INDI + scheduled f64-WLS controller for VertiRTA-LC-240.
// Keeps physical lift-rotor columns of B(V_a) non-zero prior to rotor-lock engagement, enforces
// AoA / V_stall(n_z) protection with PCH anti-windup, and implements OEI & pusher-loss policies.
class TransitionIndiWlsController {
 public:
  explicit TransitionIndiWlsController(const LiftCruiseDynamics& dynamics) noexcept;

  void reset(const AirframeState6Dof& initial_state) noexcept;

  [[nodiscard]] TransitionOutput step_400hz(
      const AirframeState6Dof& state,
      const core::Vec3Ned& wind_ned,
      const TransitionCommand& cmd) noexcept;

 private:
  struct Biquad15Hz {
    double x1{0.0}, x2{0.0}, y1{0.0}, y2{0.0};
    void reset(double v) noexcept { x1 = x2 = y1 = y2 = v; }
    [[nodiscard]] double step(double x) noexcept;
  };

  const LiftCruiseDynamics& dynamics_;
  bool initialized_{false};
  bool rotors_locked_{false};
  TransitionFaultMode latched_pusher_mode_{TransitionFaultMode::kNominal};
  double eta_state_{0.0};
  double alt_int_{0.0};
  double speed_int_{0.0};

  std::array<Biquad15Hz, 3> gyro_lpf_{};
  std::array<Biquad15Hz, 3> accel_lpf_{};
  std::array<Biquad15Hz, kMaxActuators> act_lpf_{};
  core::Vec3Frd gyro_f_{};
  core::Vec3Frd gyro_f_prev_{};
  core::Vec3Frd accel_f_{};
  std::array<double, kMaxActuators> u_hat_{};
  std::array<double, kMaxActuators> u_f_{};
  std::array<int8_t, kMaxActuators> wls_warm_{};
};

}  // namespace vertirta::airframes
