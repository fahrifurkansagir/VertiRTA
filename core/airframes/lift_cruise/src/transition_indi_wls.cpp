// SPDX-License-Identifier: Apache-2.0
#include "vertirta/airframes/transition_indi_wls.hpp"

#include <algorithm>
#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::airframes {

namespace {

constexpr double kDt400 = 0.0025;  // 400 Hz inner rate loop
// Exact 400 Hz Tustin prewarped 15.0 Hz 2nd-order Butterworth (zeta = 0.70710678)
constexpr double kB0_15 = 0.011906332234389828;
constexpr double kB1_15 = 0.023812664468779656;
constexpr double kB2_15 = 0.011906332234389828;
constexpr double kA1_15 = -1.6692031429311931;
constexpr double kA2_15 = 0.7168284718687524;

void quat_to_euler_zyx(const core::QuatPx4FrdNed& q, double& roll, double& pitch, double& yaw) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  const double inv = (n > 1e-12) ? (1.0 / n) : 1.0;
  const double w = q.w * inv, x = q.x * inv, y = q.y * inv, z = q.z * inv;

  const double sinr_cosp = 2.0 * (w * x + y * z);
  const double cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
  roll = std::atan2(sinr_cosp, cosr_cosp);

  const double sinp = core::clamp_d(2.0 * (w * y - z * x), -1.0, 1.0);
  pitch = std::asin(sinp);

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  yaw = std::atan2(siny_cosp, cosy_cosp);
}

[[nodiscard]] bool solve_cholesky_f64(
    const double H[kMaxActuators][kMaxActuators],
    const double rhs[kMaxActuators],
    const uint8_t free_idx[kMaxActuators],
    uint8_t n_free,
    double p_out[kMaxActuators]) noexcept {
  double L[kMaxActuators][kMaxActuators]{};
  for (uint8_t i = 0; i < n_free; ++i) {
    const uint8_t ri = free_idx[i];
    for (uint8_t j = 0; j <= i; ++j) {
      const uint8_t rj = free_idx[j];
      double sum = H[ri][rj];
      for (uint8_t k = 0; k < j; ++k) {
        sum -= L[i][k] * L[j][k];
      }
      if (i == j) {
        if (sum <= 1e-14 || !core::is_finite_d(sum)) {
          return false;
        }
        L[i][i] = std::sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }

  double y[kMaxActuators]{};
  for (uint8_t i = 0; i < n_free; ++i) {
    double sum = rhs[free_idx[i]];
    for (uint8_t k = 0; k < i; ++k) {
      sum -= L[i][k] * y[k];
    }
    y[i] = sum / L[i][i];
  }

  double x[kMaxActuators]{};
  for (int i = static_cast<int>(n_free) - 1; i >= 0; --i) {
    double sum = y[i];
    for (int k = i + 1; k < static_cast<int>(n_free); ++k) {
      sum -= L[k][i] * x[k];
    }
    x[i] = sum / L[i][i];
  }

  for (uint8_t i = 0; i < kMaxActuators; ++i) {
    p_out[i] = 0.0;
  }
  for (uint8_t i = 0; i < n_free; ++i) {
    p_out[free_idx[i]] = x[i];
  }
  return true;
}

}  // namespace

double quintic_hermite_blend(double v_airspeed_mps, double v_blend_lo, double v_blend_hi) noexcept {
  if (!core::is_finite_d(v_airspeed_mps) || v_blend_hi <= v_blend_lo + 1e-9) {
    return 0.0;
  }
  const double s = core::clamp_d((v_airspeed_mps - v_blend_lo) / (v_blend_hi - v_blend_lo), 0.0, 1.0);
  return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s);
}

Wls12Result wls_alloc_12x5_f64(
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
    uint8_t max_iter) noexcept {
  const uint8_t m = std::min<uint8_t>(n_actuators, kMaxActuators);
  const double g_scale = core::clamp_d(gamma, 1.0, 1e9);

  // Assemble Harkegard normal equations strictly in IEEE-754 double precision:
  //   H = gamma * B^T Wv^2 B + Wu^2,  g0 = gamma * B^T Wv^2 v_req + Wu^2 * (u_d - u_0)
  double H[kMaxActuators][kMaxActuators]{};
  double g0[kMaxActuators]{};
  for (uint8_t i = 0; i < m; ++i) {
    const double wu2 = Wu[i] * Wu[i] + 1e-9;
    H[i][i] = wu2;
    g0[i] = wu2 * ud_minus_u0[i];
    for (uint8_t j = i; j < m; ++j) {
      double sum = 0.0;
      for (uint8_t r = 0; r < kMaxControls; ++r) {
        const double wv2 = g_scale * Wv[r] * Wv[r];
        sum += wv2 * B[r][i] * B[r][j];
      }
      H[i][j] += sum;
      H[j][i] = H[i][j];
    }
    for (uint8_t r = 0; r < kMaxControls; ++r) {
      const double wv2 = g_scale * Wv[r] * Wv[r];
      g0[i] += wv2 * B[r][i] * v_req[r];
    }
  }

  Wls12Result res{};
  for (uint8_t i = 0; i < m; ++i) {
    res.W[i] = W_warm[i];
    if (du_max[i] <= du_min[i] + 1e-12) {
      res.x[i] = du_min[i];
      res.W[i] = -1;
    } else if (res.W[i] < 0) {
      res.x[i] = du_min[i];
    } else if (res.W[i] > 0) {
      res.x[i] = du_max[i];
    } else {
      res.x[i] = core::clamp_d(0.0, du_min[i], du_max[i]);
    }
  }

  for (uint8_t iter = 0; iter < max_iter; ++iter) {
    res.iters = static_cast<uint8_t>(iter + 1u);
    uint8_t free_idx[kMaxActuators]{};
    uint8_t n_free = 0u;
    double d[kMaxActuators]{};
    for (uint8_t i = 0; i < m; ++i) {
      double hx = 0.0;
      for (uint8_t j = 0; j < m; ++j) {
        hx += H[i][j] * res.x[j];
      }
      d[i] = g0[i] - hx;
      if (res.W[i] == 0) {
        free_idx[n_free++] = i;
      }
    }

    double p[kMaxActuators]{};
    if (n_free > 0u) {
      bool ok = solve_cholesky_f64(H, d, free_idx, n_free, p);
      if (!ok) {
        res.cholesky_fallback = true;
        double H_reg[kMaxActuators][kMaxActuators]{};
        for (uint8_t i = 0; i < m; ++i) {
          for (uint8_t j = 0; j < m; ++j) {
            H_reg[i][j] = H[i][j];
          }
          H_reg[i][i] += 1e-4 * (1.0 + std::abs(H[i][i]));
        }
        (void)solve_cholesky_f64(H_reg, d, free_idx, n_free, p);
      }
    }

    // Check feasibility of step x + p along free coordinates
    double alpha = 1.0;
    int blocking = -1;
    int8_t block_sign = 0;
    for (uint8_t k = 0; k < n_free; ++k) {
      const uint8_t i = free_idx[k];
      const double x_new = res.x[i] + p[i];
      if (x_new < du_min[i] - 1e-12 && p[i] < -1e-15) {
        const double a = (du_min[i] - res.x[i]) / p[i];
        if (a < alpha) {
          alpha = std::max(0.0, a);
          blocking = static_cast<int>(i);
          block_sign = -1;
        }
      } else if (x_new > du_max[i] + 1e-12 && p[i] > 1e-15) {
        const double a = (du_max[i] - res.x[i]) / p[i];
        if (a < alpha) {
          alpha = std::max(0.0, a);
          blocking = static_cast<int>(i);
          block_sign = +1;
        }
      }
    }

    for (uint8_t k = 0; k < n_free; ++k) {
      const uint8_t i = free_idx[k];
      res.x[i] = core::clamp_d(res.x[i] + alpha * p[i], du_min[i], du_max[i]);
    }

    if (blocking >= 0) {
      res.W[static_cast<uint8_t>(blocking)] = block_sign;
      continue;
    }

    // No blocking constraint: check KKT multipliers on active set
    double worst_lambda = -1e-9;
    int release_idx = -1;
    for (uint8_t i = 0; i < m; ++i) {
      if (res.W[i] == 0 || du_max[i] <= du_min[i] + 1e-12) {
        continue;
      }
      double hx = 0.0;
      for (uint8_t j = 0; j < m; ++j) {
        hx += H[i][j] * res.x[j];
      }
      const double grad_i = hx - g0[i];  // Positive gradient wants to decrease x
      // Active lower bound (W=-1) optimal if grad_i >= 0; active upper bound (W=+1) optimal if grad_i <= 0
      const double lambda_i = -static_cast<double>(res.W[i]) * grad_i;
      if (lambda_i < worst_lambda) {
        worst_lambda = lambda_i;
        release_idx = static_cast<int>(i);
      }
    }

    if (release_idx < 0) {
      break;  // KKT optimal!
    }
    res.W[static_cast<uint8_t>(release_idx)] = 0;
  }

  for (uint8_t i = 0; i < m; ++i) {
    res.x[i] = core::clamp_d(res.x[i], du_min[i], du_max[i]);
    if (i < 8u && res.W[i] != 0) {
      res.sat_mask_lo |= static_cast<uint8_t>(1u << i);
    }
  }
  return res;
}

double TransitionIndiWlsController::Biquad15Hz::step(double x) noexcept {
  const double y = kB0_15 * x + kB1_15 * x1 + kB2_15 * x2 - kA1_15 * y1 - kA2_15 * y2;
  x2 = x1;
  x1 = x;
  y2 = y1;
  y1 = y;
  return y;
}

TransitionIndiWlsController::TransitionIndiWlsController(const LiftCruiseDynamics& dynamics) noexcept
    : dynamics_(dynamics) {}

void TransitionIndiWlsController::reset(const AirframeState6Dof& initial_state) noexcept {
  initialized_ = true;
  rotors_locked_ = false;
  latched_pusher_mode_ = TransitionFaultMode::kNominal;
  alt_int_ = 0.0;
  speed_int_ = 0.0;
  wls_warm_.fill(0);

  const auto d0 = dynamics_.eval_dynamics(initial_state, {0.0, 0.0, 0.0}, 0u);
  eta_state_ = quintic_hermite_blend(d0.airspeed_mps);

  gyro_lpf_[0].reset(initial_state.omega_frd.x);
  gyro_lpf_[1].reset(initial_state.omega_frd.y);
  gyro_lpf_[2].reset(initial_state.omega_frd.z);
  gyro_f_ = initial_state.omega_frd;
  gyro_f_prev_ = initial_state.omega_frd;

  accel_lpf_[0].reset(d0.accel_body_frd.x);
  accel_lpf_[1].reset(d0.accel_body_frd.y);
  accel_lpf_[2].reset(d0.accel_body_frd.z);
  accel_f_ = d0.accel_body_frd;

  for (uint8_t i = 0; i < kMaxActuators; ++i) {
    u_hat_[i] = initial_state.u_act[i];
    u_f_[i] = initial_state.u_act[i];
    act_lpf_[i].reset(initial_state.u_act[i]);
  }
}

TransitionOutput TransitionIndiWlsController::step_400hz(
    const AirframeState6Dof& state,
    const core::Vec3Ned& wind_ned,
    const TransitionCommand& cmd) noexcept {
  if (!initialized_) {
    reset(state);
  }

  TransitionOutput out{};
  const uint8_t n_act = dynamics_.actuator_count();
  const uint8_t n_lift = dynamics_.lift_rotor_count();
  const uint8_t p_idx = dynamics_.pusher_index();
  const uint8_t a_idx = dynamics_.aileron_index();
  const uint8_t e_idx = dynamics_.elevator_index();
  const uint8_t r_idx = dynamics_.rudder_index();

  // 1. Synchronized 15 Hz Butterworth filtering of IMU rates, specific force, and actuators (Smeur sync)
  const auto dyn_now = dynamics_.eval_dynamics(state, wind_ned, cmd.fault_mask);
  const double v_air = dyn_now.airspeed_mps;
  const double alpha = dyn_now.alpha_rad;
  out.airspeed_mps = v_air;
  out.alpha_rad = alpha;

  gyro_f_prev_ = gyro_f_;
  gyro_f_ = {
      gyro_lpf_[0].step(state.omega_frd.x),
      gyro_lpf_[1].step(state.omega_frd.y),
      gyro_lpf_[2].step(state.omega_frd.z),
  };
  const core::Vec3Frd dgyro_f = (gyro_f_ - gyro_f_prev_) * (1.0 / kDt400);
  accel_f_ = {
      accel_lpf_[0].step(dyn_now.accel_body_frd.x),
      accel_lpf_[1].step(dyn_now.accel_body_frd.y),
      accel_lpf_[2].step(dyn_now.accel_body_frd.z),
  };

  EffectivenessMatrix eff{};
  dynamics_.compute_effectiveness(state, wind_ned, cmd.fault_mask, kDt400, eff);

  for (uint8_t i = 0; i < n_act; ++i) {
    if (cmd.esc_thrust_n != nullptr && cmd.esc_valid != nullptr && i <= p_idx && cmd.esc_valid[i]) {
      u_hat_[i] = cmd.esc_thrust_n[i];
    } else {
      u_hat_[i] = state.u_act[i];
    }
    u_f_[i] = act_lpf_[i].step(u_hat_[i]);
  }

  // 2. Envelope Protection & Fault Policy State Machine (V_stall, OEI, Pusher Failure)
  const uint16_t lift_mask = static_cast<uint16_t>(cmd.fault_mask & ((1u << n_lift) - 1u));
  const bool pusher_failed = (cmd.fault_mask & static_cast<uint16_t>(1u << p_idx)) != 0u;
  const double v_stall_nz = LiftCruiseDynamics::kStallSpeed1gMps * std::sqrt(std::max(dyn_now.load_factor_nz, 0.5));
  double target_v = core::clamp_d(cmd.target_airspeed_mps, 0.0, LiftCruiseDynamics::kVneAirspeedMps);

  double eta_target = quintic_hermite_blend(v_air);
  out.fault_mode = TransitionFaultMode::kNominal;

  if (pusher_failed) {
    // Hysteresis band 24.0..25.1 m/s (1.10..1.15 V_stall) per DR-09 Table 1.5
    if (v_air >= 25.1) {
      latched_pusher_mode_ = TransitionFaultMode::kPusherLossGlide;
    } else if (v_air < 24.0) {
      latched_pusher_mode_ = TransitionFaultMode::kPusherLossHoverAbort;
    } else if (latched_pusher_mode_ == TransitionFaultMode::kNominal) {
      latched_pusher_mode_ = TransitionFaultMode::kPusherLossHoverAbort;
    }
    out.fault_mode = latched_pusher_mode_;
    if (latched_pusher_mode_ == TransitionFaultMode::kPusherLossGlide) {
      out.glide_target_airspeed_mps = LiftCruiseDynamics::kBestGlideAirspeedMps;
      out.estimated_ld_ratio = LiftCruiseDynamics::kBestGlideLd;
      target_v = LiftCruiseDynamics::kBestGlideAirspeedMps;
      eta_target = 1.0;
    } else {
      // Immediate hover abort: eta -> 0 in 0.4 s
      eta_target = 0.0;
      target_v = 0.0;
      rotors_locked_ = false;
      eta_state_ = std::max(0.0, eta_state_ - kDt400 / 0.4);
    }
  } else {
    latched_pusher_mode_ = TransitionFaultMode::kNominal;
    if (lift_mask != 0u) {
      out.fault_mode = TransitionFaultMode::kLiftRotorOei;
    }
    // Back-transition rotor wake-up: if deceleration is commanded below blend ceiling,
    // allow eta_target to track min(v_air, target_v) once v_air drops to <= 29.5 m/s so lift rotors
    // take vertical load smoothly and prevent an endless best-glide lock above 26 m/s.
    if (target_v < LiftCruiseDynamics::kBlendEndMps - 0.5 && v_air <= LiftCruiseDynamics::kBlendEndMps + 1.5) {
      eta_target = std::min(eta_target, quintic_hermite_blend(target_v));
    }
    // Stall floor guard in winged cruise: V_a < 1.08 * V_stall(n_z) releases lock & pulls eta down
    if (rotors_locked_ && v_air < 1.08 * v_stall_nz) {
      rotors_locked_ = false;
      out.fault_mode = TransitionFaultMode::kStallRecovery;
      eta_target = std::min(eta_target, 0.35);
    }
    // Smooth rate-limited eta tracking (full sweep in ~1.0 s)
    const double max_d_eta = 1.0 * kDt400;
    eta_state_ = core::clamp_d(eta_state_ + core::clamp_d(eta_target - eta_state_, -max_d_eta, max_d_eta), 0.0, 1.0);
  }

  // Engage streamwise rotor lock ONLY once V_a >= 28.5 m/s, eta >= 0.999, cruising,
  // AND lift rotors have naturally unloaded (< 35 N max) so lock-out causes zero altitude drop.
  double max_lift_uf = 0.0;
  for (uint8_t i = 0; i < n_lift; ++i) {
    max_lift_uf = std::max(max_lift_uf, u_f_[i]);
  }
  if (v_air >= LiftCruiseDynamics::kBlendEndMps + 0.5 && eta_state_ >= 0.999 &&
      target_v >= LiftCruiseDynamics::kBlendEndMps && !pusher_failed && max_lift_uf < 35.0) {
    rotors_locked_ = true;
  } else if (v_air < LiftCruiseDynamics::kBlendEndMps || target_v < LiftCruiseDynamics::kBlendEndMps - 0.5) {
    rotors_locked_ = false;
  }

  out.eta_blend = eta_state_;
  out.rotors_locked_cruise = rotors_locked_;

  // 3. Outer-Loop Altitude, Airspeed & Attitude Reference Model
  double roll = 0.0, pitch = 0.0, yaw = 0.0;
  quat_to_euler_zyx(state.q_frd_ned, roll, pitch, yaw);

  const double alt_m = -state.pos_ned.z;
  const double alt_err = cmd.target_alt_m - alt_m;  // Positive when below target
  const double vz_ned = state.vel_ned.z;            // Positive downward

  // Desired NED vertical acceleration (negative = upward); respect OEI 20.8% dynamic thrust margin
  const double az_lim_up = (lift_mask != 0u) ? -1.6 : -4.5;
  const double az_des_ned = core::clamp_d(-5.5 * alt_err - 4.8 * vz_ned - 1.5 * alt_int_, az_lim_up, 3.5);

  // Desired longitudinal acceleration in body x (zero integrator immediately on deceleration)
  const double v_err = target_v - v_air;
  if (v_err < -0.3 && speed_int_ > 0.0) {
    speed_int_ = 0.0;
  }
  const double ax_des_body = core::clamp_d(1.8 * v_err + 0.25 * speed_int_, -4.5, 4.0);

  // Wing trim pitch schedule blended smoothly by eta_state_:
  // In cruise (eta=1), wing lift carries 100% of weight + altitude correction via pitch angle.
  const double q_dyn = 0.5 * LiftCruiseDynamics::kRhoAir * v_air * v_air;
  const double gamma_fp = (v_air > 4.0) ? std::atan2(-vz_ned, std::max(v_air, 4.0)) : 0.0;
  double theta_wing = 0.0;
  if (q_dyn * LiftCruiseDynamics::kWingAreaM2 > 200.0) {
    const double lift_req = LiftCruiseDynamics::kWeightN - LiftCruiseDynamics::kMassKg * az_des_ned;
    const double cl_req = core::clamp_d(lift_req / (q_dyn * LiftCruiseDynamics::kWingAreaM2), -0.1, 1.15);
    const double alpha_trim = core::clamp_d(
        (cl_req - LiftCruiseDynamics::kCl0) / LiftCruiseDynamics::kClAlpha, -0.06, 0.175);
    theta_wing = core::clamp_d(gamma_fp + alpha_trim, -0.10, 0.19);
  }
  // Pre-pitch wing slightly ahead of rotor unload (12..25 m/s) during forward transition so wing
  // already carries full 1g weight before streamwise rotor lock engages at 28.5 m/s.
  const double wing_pitch_blend =
      (v_err >= -0.15) ? std::max(eta_state_, quintic_hermite_blend(v_air, 12.0, 25.0)) : eta_state_;
  // During deceleration below 28 m/s with lift rotors active, pitch nose-up to tilt rotor lift vector backward
  const double brake_pitch =
      (!rotors_locked_ && v_err < -0.15) ? core::clamp_d(-0.08 * v_err, 0.0, 0.17) : 0.0;
  const double theta_des = core::clamp_d(
      wing_pitch_blend * theta_wing + (1.0 - wing_pitch_blend) * brake_pitch, -0.10, 0.19);

  // Lateral/heading stabilization (wings level + heading hold)
  const double yaw_err = core::wrap_pi_d(cmd.target_heading_rad - yaw);
  const double phi_des = 0.0;

  // PD rotational virtual acceleration demands [p_dot, q_dot, r_dot]
  const double p_dot_des = core::clamp_d(14.0 * (phi_des - roll) - 6.5 * state.omega_frd.x, -6.0, 6.0);
  double q_dot_des = core::clamp_d(16.0 * (theta_des - pitch) - 7.5 * state.omega_frd.y, -6.0, 6.0);
  const double r_dot_des = core::clamp_d(6.0 * yaw_err - 4.0 * state.omega_frd.z, -4.0, 4.0);

  // AoA ceiling barrier (alpha_max = 12.0 deg = 0.20944 rad, DR-09 Sec 1.5)
  constexpr double kAlphaMaxRad = 12.0 * core::kDegToRad;
  if (v_air > 8.0 && alpha > 0.75 * kAlphaMaxRad) {
    const double q_dot_cap = 12.0 * (kAlphaMaxRad - alpha) - 4.5 * state.omega_frd.y;
    if (q_dot_des > q_dot_cap) {
      q_dot_des = q_dot_cap;
      out.aoa_barrier_active = true;
    }
  }

  // Transform desired specific force into body FRD (g_frd = [-g*sin(theta), 0, +g*cos(theta)]):
  const double fz_body_des = (az_des_ned - LiftCruiseDynamics::kGravity) * std::cos(pitch);
  const double fx_body_des = ax_des_body + LiftCruiseDynamics::kGravity * std::sin(pitch);

  // 4. INDI Incremental Virtual Demands: v_req = nu_des - nu_measured_filtered
  // Note: because state.u_act tracks real physical actuators while u_f_ is LPF-synchronized,
  // B * du produces exact incremental control without blinding rotor columns.
  std::array<double, kMaxControls> v_req{
      p_dot_des - dgyro_f.x,
      q_dot_des - dgyro_f.y,
      r_dot_des - dgyro_f.z,
      fz_body_des - accel_f_.z,
      fx_body_des - accel_f_.x,
  };

  // 5. Airspeed-Scheduled Smeur Priority Weights & Actuator Effort / Preference (DR-09 Sec 1.4)
  // W_v(eta) = diag(10.0, 10.0, 1.5, 5.0 * (1 - 0.98 * eta), 3.0)
  std::array<double, kMaxControls> Wv{
      10.0,
      10.0,
      1.5,
      5.0 * (1.0 - 0.98 * eta_state_),
      3.0,
  };
  // For 4+1 mid-transition OEI below 12 m/s, drop yaw priority to protect roll/pitch
  if (dynamics_.layout() == LiftCruiseLayout::kFourPlusOne && lift_mask != 0u && v_air < 12.0) {
    Wv[2] = 0.05;
  }

  std::array<double, kMaxActuators> Wu{};
  std::array<double, kMaxActuators> ud_minus_u0{};
  const double t_hover_each = LiftCruiseDynamics::kWeightN / static_cast<double>(n_lift);

  // Lift rotors: effort cost ramps 1.0 -> 20.0; preferred thrust decays (1 - eta) * t_hover
  // Only zero lift-rotor bounds after streamwise rotor lock engages at V_a >= 28.5 m/s & eta >= 0.999.
  // Use full physical position bounds [0 - u_f, T_max - u_f] (physical 85 ms motor lag governs u_act slew).
  for (uint8_t i = 0; i < n_lift; ++i) {
    const bool failed = (cmd.fault_mask & static_cast<uint16_t>(1u << i)) != 0u;
    Wu[i] = 1.0 + 19.0 * eta_state_;
    const double u_pref = (failed || rotors_locked_) ? 0.0 : (1.0 - eta_state_) * t_hover_each;
    ud_minus_u0[i] = u_pref - u_f_[i];
    if (rotors_locked_ || failed) {
      eff.du_min[i] = -u_f_[i];
      eff.du_max[i] = -u_f_[i];
    } else {
      eff.du_min[i] = 0.0 - u_f_[i];
      eff.du_max[i] = dynamics_.max_lift_rotor_thrust_n() - u_f_[i];
    }
  }

  // Pusher effort & full position bounds
  Wu[p_idx] = 1.0;
  ud_minus_u0[p_idx] = 0.0;
  if (pusher_failed) {
    eff.du_min[p_idx] = -u_f_[p_idx];
    eff.du_max[p_idx] = -u_f_[p_idx];
  } else {
    eff.du_min[p_idx] = 0.0 - u_f_[p_idx];
    eff.du_max[p_idx] = eff.u_max[p_idx] - u_f_[p_idx];
  }

  // Aero surfaces: effort cost drops 20.0 -> 1.0 as dynamic pressure builds
  const double wu_aero = 20.0 - 19.0 * eta_state_;
  for (uint8_t idx : {a_idx, e_idx, r_idx}) {
    Wu[idx] = wu_aero;
    ud_minus_u0[idx] = 0.0 - u_f_[idx];
    eff.du_min[idx] = eff.u_min[idx] - u_f_[idx];
    eff.du_max[idx] = eff.u_max[idx] - u_f_[idx];
  }

  // 6. Solve warm-started 17x12 f64 Active-Set WLS
  const auto wls = wls_alloc_12x5_f64(
      eff.B, v_req, eff.du_min, eff.du_max, Wv, Wu, ud_minus_u0, 1e6, wls_warm_, n_act, 12u);
  wls_warm_ = wls.W;
  out.wls_iters = wls.iters;

  // 7. Compute actuator commands & Pseudo-Control Hedging (PCH) residual nu_h = v_req - B * du*
  bool all_finite = true;
  for (uint8_t i = 0; i < n_act; ++i) {
    double cmd_i = u_f_[i] + wls.x[i];
    if ((i < n_lift && (rotors_locked_ || ((cmd.fault_mask & static_cast<uint16_t>(1u << i)) != 0u))) ||
        (i == p_idx && pusher_failed)) {
      cmd_i = 0.0;
    } else {
      cmd_i = core::clamp_d(cmd_i, eff.u_min[i], eff.u_max[i]);
    }
    out.u_cmd[i] = cmd_i;
    if (!core::is_finite_d(cmd_i)) {
      all_finite = false;
    }
  }

  for (uint8_t r = 0; r < kMaxControls; ++r) {
    double b_du = 0.0;
    for (uint8_t i = 0; i < n_act; ++i) {
      b_du += eff.B[r][i] * wls.x[i];
    }
    out.nu_hedge[r] = v_req[r] - b_du;
  }

  // Anti-windup integrator updates hedged by PCH residual
  const double hedge_z = core::clamp_d(out.nu_hedge[3], -2.0, 2.0);
  if (std::abs(hedge_z) < 0.8) {
    alt_int_ = core::clamp_d(alt_int_ + alt_err * kDt400, -1.5, 1.5);
  }
  const double hedge_x = core::clamp_d(out.nu_hedge[4], -2.0, 2.0);
  if (std::abs(hedge_x) < 0.8 && !pusher_failed) {
    speed_int_ = core::clamp_d(speed_int_ + v_err * kDt400, -3.0, 3.0);
  }

  out.valid = all_finite;
  return out;
}

}  // namespace vertirta::airframes
