#include "vertirta/core/indi_wls.hpp"

#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

namespace {

// Exact 400 Hz Tustin prewarped 8.0 Hz Butterworth (zeta = 0.7071) biquad (DR-1 Table 3.2)
constexpr double kB0 = 0.003621681514928682;
constexpr double kB1 = 0.007243363029857364;
constexpr double kB2 = 0.003621681514928682;
constexpr double kA1 = -1.8226949251963083;
constexpr double kA2 = 0.8371816512560227;
constexpr double kDt400 = 0.0025;

// Solve symmetric positive definite submatrix H[free][free] * p[free] = d[free] in f64 Cholesky.
[[nodiscard]] bool solve_cholesky_submatrix_f64(
    const double H[8][8],
    const double d[8],
    const int free_idx[8],
    int n_free,
    double p_out[8]) noexcept {
  double L[8][8]{};
  for (int i = 0; i < n_free; ++i) {
    const int ri = free_idx[i];
    for (int j = 0; j <= i; ++j) {
      const int rj = free_idx[j];
      double sum = H[ri][rj];
      for (int k = 0; k < j; ++k) {
        sum -= L[i][k] * L[j][k];
      }
      if (i == j) {
        if (sum <= 1e-14 || !is_finite_d(sum)) {
          return false;
        }
        L[i][i] = std::sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }

  double y[8]{};
  for (int i = 0; i < n_free; ++i) {
    double sum = d[free_idx[i]];
    for (int k = 0; k < i; ++k) {
      sum -= L[i][k] * y[k];
    }
    y[i] = sum / L[i][i];
  }

  double x[8]{};
  for (int i = n_free - 1; i >= 0; --i) {
    double sum = y[i];
    for (int k = i + 1; k < n_free; ++k) {
      sum -= L[k][i] * x[k];
    }
    x[i] = sum / L[i][i];
  }

  for (int i = 0; i < 8; ++i) {
    p_out[i] = 0.0;
  }
  for (int i = 0; i < n_free; ++i) {
    p_out[free_idx[i]] = x[i];
  }
  return true;
}

// Rotate NED vector v into body FRD using R_{FRD->NED}^T from Hamilton scalar-first q_frd_ned.
[[nodiscard]] Vec3Frd ned_to_frd_by_quat(const QuatPx4FrdNed& q, const Vec3Ned& v) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  const double inv = (n > 1e-9) ? (1.0 / n) : 1.0;
  const double w = q.w * inv, x = q.x * inv, y = q.y * inv, z = q.z * inv;

  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - w * z);
  const double r02 = 2.0 * (x * z + w * y);
  const double r10 = 2.0 * (x * y + w * z);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - w * x);
  const double r20 = 2.0 * (x * z - w * y);
  const double r21 = 2.0 * (y * z + w * x);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);

  // R^T * v
  return {
      r00 * v.x + r10 * v.y + r20 * v.z,
      r01 * v.x + r11 * v.y + r21 * v.z,
      r02 * v.x + r12 * v.y + r22 * v.z,
  };
}

}  // namespace

WlsResult wls_alloc_8x4_f64(
    const double G[4][8],
    const double v[4],
    const double xmin[8],
    const double xmax[8],
    const double Wv[4],
    const double Wu[8],
    const double ud_minus_u0[8],
    double gamma,
    int8_t W_warm[8],
    uint8_t max_iter) noexcept {
  // 1. Assemble weighted normal equations strictly in f64 (fixes DR-1 critical erratum #1).
  double H[8][8]{};
  double g0[8]{};
  for (int i = 0; i < 8; ++i) {
    const double reg = gamma * Wu[i] * Wu[i];
    H[i][i] = reg;
    g0[i] = reg * ud_minus_u0[i];
    for (int j = i; j < 8; ++j) {
      double sum = 0.0;
      for (int r = 0; r < 4; ++r) {
        const double w2 = Wv[r] * Wv[r];
        sum += w2 * G[r][i] * G[r][j];
      }
      H[i][j] += sum;
      H[j][i] = H[i][j];
    }
    for (int r = 0; r < 4; ++r) {
      g0[i] += (Wv[r] * Wv[r]) * G[r][i] * v[r];
    }
  }

  // 2. Feasible initialization from warm working set (freestanding clamp_d, no <algorithm>).
  WlsResult res{};
  for (int i = 0; i < 8; ++i) {
    res.W[i] = W_warm[i];
    if (xmax[i] <= xmin[i] + 1e-9) {
      res.x[i] = xmin[i];
      res.W[i] = -1;
    } else if (res.W[i] < 0) {
      res.x[i] = xmin[i];
    } else if (res.W[i] > 0) {
      res.x[i] = xmax[i];
    } else {
      res.x[i] = clamp_d(0.0, xmin[i], xmax[i]);
    }
  }

  // 3. Deterministic active-set loop (<= max_iter).
  for (uint8_t iter = 0; iter < max_iter; ++iter) {
    res.iters = static_cast<uint8_t>(iter + 1u);
    int free_idx[8]{};
    int n_free = 0;
    double d[8]{};
    for (int i = 0; i < 8; ++i) {
      double hx = 0.0;
      for (int j = 0; j < 8; ++j) {
        hx += H[i][j] * res.x[j];
      }
      d[i] = g0[i] - hx;
      if (res.W[i] == 0) {
        free_idx[n_free++] = i;
      }
    }

    double p[8]{};
    if (n_free > 0) {
      if (!solve_cholesky_submatrix_f64(H, d, free_idx, n_free, p)) {
        res.cholesky_failed = true;
        break;
      }
    }

    double alpha = 1.0;
    int block_idx = -1;
    int8_t block_sign = 0;
    for (int k = 0; k < n_free; ++k) {
      const int i = free_idx[k];
      if (p[i] > 1e-10 && res.x[i] + p[i] > xmax[i]) {
        const double a = (xmax[i] - res.x[i]) / p[i];
        if (a < alpha) {
          alpha = a;
          block_idx = i;
          block_sign = +1;
        }
      } else if (p[i] < -1e-10 && res.x[i] + p[i] < xmin[i]) {
        const double a = (xmin[i] - res.x[i]) / p[i];
        if (a < alpha) {
          alpha = a;
          block_idx = i;
          block_sign = -1;
        }
      }
    }

    for (int k = 0; k < n_free; ++k) {
      const int i = free_idx[k];
      res.x[i] = clamp_d(res.x[i] + alpha * p[i], xmin[i], xmax[i]);
    }

    if (block_idx >= 0) {
      res.x[block_idx] = (block_sign > 0) ? xmax[block_idx] : xmin[block_idx];
      res.W[block_idx] = block_sign;
      continue;
    }

    // Check KKT multipliers on active set.
    double min_lambda = -1e-7;
    int drop_idx = -1;
    for (int i = 0; i < 8; ++i) {
      if (res.W[i] == 0 || xmax[i] <= xmin[i] + 1e-9) {
        continue;
      }
      double hx = 0.0;
      for (int j = 0; j < 8; ++j) {
        hx += H[i][j] * res.x[j];
      }
      const double lam = static_cast<double>(res.W[i]) * (g0[i] - hx);
      if (lam < min_lambda) {
        min_lambda = lam;
        drop_idx = i;
      }
    }
    if (drop_idx < 0) {
      break;
    }
    res.W[drop_idx] = 0;
  }

  for (int i = 0; i < 8; ++i) {
    W_warm[i] = res.W[i];
    if (res.W[i] != 0) {
      res.sat_mask = static_cast<uint8_t>(res.sat_mask | (1u << i));
    }
  }
  return res;
}

double IndiWlsController::BiquadState::step(double x) noexcept {
  const double y = kB0 * x + kB1 * x1 + kB2 * x2 - kA1 * y1 - kA2 * y2;
  x2 = x1;
  x1 = x;
  y2 = y1;
  y1 = y;
  return y;
}

IndiWlsController::IndiWlsController(bool allow_active_spin) noexcept
    : allow_active_spin_(allow_active_spin) {
  reset();
}

void IndiWlsController::reset(double initial_omega_rad_s) noexcept {
  initialized_ = false;
  spin_rate_ref_ = 0.0f;
  phase_lead_deg_ = 0.0f;
  gyro_f_ = {};
  gyro_f_prev_ = {};
  dgyro_f_ = {};
  delay_idx_ = 0;

  double k_t[8]{};
  double k_q[8]{};
  CoaxEffectiveness::effective_rotor_constants(0x00, 0.0f, k_t, k_q);
  for (int i = 0; i < 3; ++i) {
    gyro_lpf_[i].reset(0.0);
  }
  for (int i = 0; i < 8; ++i) {
    omega_hat_[i] = initial_omega_rad_s;
    omega_cmd_delay_[0][i] = initial_omega_rad_s;
    omega_cmd_delay_[1][i] = initial_omega_rad_s;
    thrust_lpf_[i].reset(k_t[i] * initial_omega_rad_s * initial_omega_rad_s);
    wls_warm_[i] = 0;
  }
}

void IndiWlsController::get_omega_hat(float out[8]) const noexcept {
  for (int i = 0; i < 8; ++i) {
    out[i] = static_cast<float>(omega_hat_[i]);
  }
}

WlsResult IndiWlsController::step_400hz(const IndiInput& in, float omega_cmd_out[8]) noexcept {
  // 1. Update actuator effectiveness and 1st-order motor-lag observer.
  double B_T[4][8]{};
  float u_max_f[8]{};
  double k_t[8]{};
  double k_q[8]{};
  CoaxEffectiveness::build_B(in.fault_mask, in.v_horiz_mps, B_T, u_max_f, in.v_term_v);
  CoaxEffectiveness::effective_rotor_constants(in.fault_mask, in.v_horiz_mps, k_t, k_q);

  const double w_cap = static_cast<double>(CoaxEffectiveness::omega_lim_from_vterm(in.v_term_v));
  double u_hat_f[8]{};
  double T_hat_f = 0.0;

  if (!initialized_) {
    gyro_lpf_[0].reset(in.gyro_frd.x);
    gyro_lpf_[1].reset(in.gyro_frd.y);
    gyro_lpf_[2].reset(in.gyro_frd.z);
    gyro_f_ = in.gyro_frd;
    gyro_f_prev_ = in.gyro_frd;
    initialized_ = true;
  }

  for (int i = 0; i < 8; ++i) {
    const bool dead = (in.fault_mask & static_cast<uint8_t>(1u << i)) != 0u;
    const double w_delayed = dead ? 0.0 : omega_cmd_delay_[delay_idx_][i];
    const double tau = dead ? params::kTauCoast
                            : ((w_delayed >= omega_hat_[i]) ? params::kTauUp : params::kTauDown);
    const double dw = dead ? (-k_q[i] * omega_hat_[i] * omega_hat_[i] / params::kJr)
                           : clamp_d((w_delayed - omega_hat_[i]) / tau, -params::kAlphaMax, params::kAlphaMax);
    omega_hat_[i] = clamp_d(omega_hat_[i] + dw * kDt400, 0.0, params::kOmegaMax);

    // Optional complementary blend with live ESC RPM telemetry (20 rad/s)
    if (in.esc_omega_rad_s != nullptr && in.esc_valid != nullptr && in.esc_valid[i] &&
        is_finite_f(in.esc_omega_rad_s[i])) {
      omega_hat_[i] += 20.0 * kDt400 * (static_cast<double>(in.esc_omega_rad_s[i]) - omega_hat_[i]);
      omega_hat_[i] = clamp_d(omega_hat_[i], 0.0, params::kOmegaMax);
    }

    const double thrust_inst = k_t[i] * omega_hat_[i] * omega_hat_[i];
    u_hat_f[i] = thrust_lpf_[i].step(thrust_inst);
    T_hat_f += u_hat_f[i];
  }

  // 2. Synchronized 8 Hz Butterworth filtering of voted gyro -> dOmega_f.
  gyro_f_prev_ = gyro_f_;
  gyro_f_ = {
      gyro_lpf_[0].step(in.gyro_frd.x),
      gyro_lpf_[1].step(in.gyro_frd.y),
      gyro_lpf_[2].step(in.gyro_frd.z),
  };
  dgyro_f_ = (gyro_f_ - gyro_f_prev_) * (1.0 / kDt400);

  // 3. Desired inertial force & reduced-attitude S^2 target n_{B,des}.
  const Vec3Ned f_ned{
      params::kMassKg * in.accel_des_ned.x,
      params::kMassKg * in.accel_des_ned.y,
      params::kMassKg * (in.accel_des_ned.z - params::kGravity),
  };
  double T_des = f_ned.norm();
  if (T_des < 50.0) {
    T_des = 50.0;
  }
  const Vec3Ned n_ned_des = f_ned * (1.0 / T_des);  // Points along thrust vector (-Z_FRD when level)
  Vec3Frd n_b_des = ned_to_frd_by_quat(in.q_frd_ned, n_ned_des);

  const FlightRegime regime = CoaxEffectiveness::classify_regime(in.fault_mask);
  const int dead_arm = CoaxEffectiveness::failed_same_arm_index(in.fault_mask);

  // 4. Construct virtual control demand v_req and WLS weights by regime.
  double G_eff[4][8]{};
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 8; ++i) {
      G_eff[r][i] = B_T[r][i];
    }
  }
  // In Regimes 0-2, include mild G2 rotor-inertia yaw augmentation; in Regime 3 use exact steady-state
  // B_T[2] so differential yaw sustains full -55.3/+55.3 N*m spin-up torque without under-allocation.
  if (regime != FlightRegime::SAME_ARM_ACTIVE_SPIN) {
    for (int i = 0; i < 8; ++i) {
      if ((in.fault_mask & static_cast<uint8_t>(1u << i)) == 0u) {
        const double w_lin = (omega_hat_[i] > 50.0) ? omega_hat_[i] : 50.0;
        G_eff[2][i] += 0.25 * static_cast<double>(params::kRotorSpinSign[i]) * params::kJr /
                       (2.0 * k_t[i] * w_lin * params::kTauUp);
      }
    }
  }

  double v_req[4]{};
  double Wv[4] = {100.0, 100.0, 1.0, 25.0};
  double Wu[8] = {1.0, 1.0, 1.0, 1.0, 1.15, 1.15, 1.15, 1.15};
  double ud_minus_u0[8]{};

  // Preferred equal load-sharing baseline
  int n_alive = 0;
  for (int i = 0; i < 8; ++i) {
    if ((in.fault_mask & static_cast<uint8_t>(1u << i)) == 0u) {
      ++n_alive;
    }
  }
  const double t_share = (n_alive > 0) ? (T_des / static_cast<double>(n_alive)) : 0.0;
  for (int i = 0; i < 8; ++i) {
    const bool dead = (in.fault_mask & static_cast<uint8_t>(1u << i)) != 0u;
    ud_minus_u0[i] = dead ? 0.0 : (t_share - u_hat_f[i]);
  }

  bool allow_opp_spin_pulse = false;
  double t_dead_arm = 0.0;

  if (regime == FlightRegime::SAME_ARM_ACTIVE_SPIN && dead_arm >= 0) {
    const int opp_arm = dead_arm ^ 1;  // Arm 0<->1 (FR<->RL), Arm 2<->3 (FL<->RR)
    t_dead_arm = u_hat_f[dead_arm] + u_hat_f[dead_arm + 4];
    // Exact FRD azimuth of dead arm: 0=+45, 1=-135, 2=-45, 3=+135 deg
    static constexpr double kArmCos[4] = {+kInvSqrt2, -kInvSqrt2, +kInvSqrt2, -kInvSqrt2};
    static constexpr double kArmSin[4] = {+kInvSqrt2, -kInvSqrt2, -kInvSqrt2, +kInvSqrt2};
    const double ca = kArmCos[dead_arm];
    const double sa = kArmSin[dead_arm];

    // Option B1b active flat spin vs Option A (allow_active_spin_ == false)
    spin_rate_ref_ = allow_active_spin_ ? ((dead_arm == 0 || dead_arm == 1) ? -2.2f : +2.2f) : 0.0f;
    const double lead_ref_rad = std::atan(static_cast<double>(spin_rate_ref_) * params::kTauUp);
    phase_lead_deg_ = static_cast<float>(lead_ref_rad * kRadToDeg);

    // Apply live actuator-lag phase lead R_z(+atan(r * tau_up)) around +Z_FRD to n_{B,des} (DR-1 Sec 3.4)
    const double r_lead = allow_active_spin_ ? clamp_d(gyro_f_.z, -2.5, 2.5) : 0.0;
    const double lead_rad = std::atan(r_lead * params::kTauUp);
    const double cl = std::cos(lead_rad);
    const double sl = std::sin(lead_rad);
    const Vec3Frd n_lead{
        cl * n_b_des.x - sl * n_b_des.y,
        sl * n_b_des.x + cl * n_b_des.y,
        n_b_des.z,
    };

    // Rotate gyro, dgyro, and thrust-vector error into dead-arm diagonal frame (e_u, e_v, e_z).
    const double p_u = ca * gyro_f_.x + sa * gyro_f_.y;
    const double q_v = -sa * gyro_f_.x + ca * gyro_f_.y;
    const double dp_u = ca * dgyro_f_.x + sa * dgyro_f_.y;
    const double dq_v = -sa * dgyro_f_.x + ca * dgyro_f_.y;
    const double n_u = ca * n_lead.x + sa * n_lead.y;
    const double n_v = -sa * n_lead.x + ca * n_lead.y;

    // Exact 4-state INDI pole-placed reduced-attitude feedback on bidirectional tau_u (DR-1 Sec 3.4)
    double alpha_u_des = 0.0;
    double alpha_v_des = 0.0;
    if (allow_active_spin_) {
      const double sgn_r = (spin_rate_ref_ < 0.0f) ? -1.0 : 1.0;
      const double spin_blend = clamp_d(std::abs(gyro_f_.z) / 2.2, 0.0, 1.0);
      // Exact Ackermann INDI gains at r = +/-2.2 rad/s (poles -0.85+/-1.1j, -1.4+/-0.8j):
      // K_indi = [4.5, -/+2.6426, +/-12.2206, -7.6824]
      const double k_pu = 4.5;
      const double k_qv = -sgn_r * 2.6426 * spin_blend;
      const double k_nu = +sgn_r * 12.2206 * spin_blend;
      const double k_nv = -7.6824;
      alpha_u_des = -(k_pu * p_u + k_qv * q_v + k_nu * n_u + k_nv * n_v);
      // Unilateral stabilization on tau_v (opp_arm fires only when negative torque is helpful)
      alpha_v_des = -3.0 * q_v - 4.0 * n_u;
      allow_opp_spin_pulse =
          (std::abs(gyro_f_.z) < 1.8) && (alpha_v_des < -0.25) && (q_v > -0.10) && (n_u < 0.05);
    } else {
      // Option A non-spinning 2-arm baseline (demonstrates unilateral rank-2/4 instability)
      alpha_u_des = 5.0 * (2.6 * n_v - p_u);
      alpha_v_des = 5.0 * (-2.6 * n_u - q_v);
    }

    const double alpha_z_des = 3.5 * (static_cast<double>(spin_rate_ref_) - gyro_f_.z);

    // Transform G_eff roll/pitch rows into diagonal (u, v) frame so W_v prioritizes bidirectional tau_u!
    double G_diag[4][8]{};
    for (int i = 0; i < 8; ++i) {
      G_diag[0][i] = ca * G_eff[0][i] + sa * G_eff[1][i];
      G_diag[1][i] = -sa * G_eff[0][i] + ca * G_eff[1][i];
      G_diag[2][i] = G_eff[2][i];
      G_diag[3][i] = G_eff[3][i];
    }
    for (int r = 0; r < 4; ++r) {
      for (int i = 0; i < 8; ++i) {
        G_eff[r][i] = G_diag[r][i];
      }
    }

    v_req[0] = params::kJPerp * (alpha_u_des - dp_u);
    v_req[1] = params::kJPerp * (alpha_v_des - dq_v);
    v_req[2] = params::kJzz * (alpha_z_des - dgyro_f_.z);
    v_req[3] = T_des - T_hat_f;

    Wv[0] = 100.0;  // Bidirectional tau_u priority 1
    Wv[1] = 10.0;   // Unilateral tau_v opportunistic
    Wv[2] = allow_active_spin_ ? 4.0 : 1.0;
    Wv[3] = 30.0;   // Collective lift

    // Drive opposite arm preferred thrust to 0 (unload Arm opp_arm so T_2=T_3=W/2)
    for (int i = 0; i < 8; ++i) {
      if (i == opp_arm || i == opp_arm + 4) {
        ud_minus_u0[i] = 0.0 - u_hat_f[i];
      } else if ((in.fault_mask & static_cast<uint8_t>(1u << i)) == 0u) {
        ud_minus_u0[i] = (0.25 * T_des) - u_hat_f[i];
      }
    }
  } else {
    spin_rate_ref_ = 0.0f;
    phase_lead_deg_ = 0.0f;
    // Standard S^2 reduced-attitude error: e_n = n_B x n_{B,des} = [+n_b_des.y, -n_b_des.x, 0]
    constexpr double kKn = 2.6;
    const double wx_ref = kKn * n_b_des.y;
    const double wy_ref = -kKn * n_b_des.x;
    const double wz_ref = in.yaw_rate_des;

    const double dwx_des = 5.0 * (wx_ref - gyro_f_.x);
    const double dwy_des = 5.0 * (wy_ref - gyro_f_.y);
    const double dwz_des = 3.0 * (wz_ref - gyro_f_.z);

    v_req[0] = params::kJxx * (dwx_des - dgyro_f_.x);
    v_req[1] = params::kJyy * (dwy_des - dgyro_f_.y);
    v_req[2] = params::kJzz * (dwz_des - dgyro_f_.z);
    v_req[3] = T_des - T_hat_f;
  }

  // 5. Per-tick box bounds (dead motors locked at 0, idle floor, voltage ceiling, 25 ms slew).
  double xmin[8]{};
  double xmax[8]{};
  const int opp_arm_cap = (regime == FlightRegime::SAME_ARM_ACTIVE_SPIN && dead_arm >= 0) ? (dead_arm ^ 1) : -1;
  for (int i = 0; i < 8; ++i) {
    const bool dead = (in.fault_mask & static_cast<uint8_t>(1u << i)) != 0u;
    if (dead) {
      xmin[i] = 0.0;
      xmax[i] = 0.0;
    } else {
      const double u_idle = k_t[i] * params::kOmegaIdle * params::kOmegaIdle;
      double u_cap = static_cast<double>(u_max_f[i]);
      // In Regime 3, match coast-down thrust on dead arm, then unload opposite arm to idle
      // (with bounded <=150 N lower-rotor spin-up pulse ONLY when attitude/rate safely allow).
      if (opp_arm_cap >= 0 && (i == opp_arm_cap || i == opp_arm_cap + 4)) {
        const double coast_match = u_idle + 0.5 * t_dead_arm;
        if (allow_active_spin_ && allow_opp_spin_pulse && i == opp_arm_cap + 4) {
          u_cap = (coast_match > 150.0) ? coast_match : 150.0;
        } else {
          u_cap = coast_match;
        }
        if (u_cap > static_cast<double>(u_max_f[i])) {
          u_cap = static_cast<double>(u_max_f[i]);
        }
      }
      const double w_lin = (omega_hat_[i] > 50.0) ? omega_hat_[i] : 50.0;
      const double du_slew = 2.0 * k_t[i] * w_lin * params::kAlphaMax * 0.025;
      const double lo = u_idle - u_hat_f[i];
      const double hi = u_cap - u_hat_f[i];
      xmin[i] = (lo > -du_slew) ? lo : -du_slew;
      xmax[i] = (hi < +du_slew) ? hi : +du_slew;
      if (xmax[i] < xmin[i]) {
        xmin[i] = hi;
        xmax[i] = hi;
      }
    }
  }

  const WlsResult res =
      wls_alloc_8x4_f64(G_eff, v_req, xmin, xmax, Wv, Wu, ud_minus_u0, 1e-4, wls_warm_, 12);

  // 6. Convert allocated thrusts -> motor speed commands [rad/s] and push into 2-tick transport FIFO.
  for (int i = 0; i < 8; ++i) {
    const bool dead = (in.fault_mask & static_cast<uint8_t>(1u << i)) != 0u;
    if (dead) {
      omega_cmd_out[i] = 0.0f;
      omega_cmd_delay_[delay_idx_][i] = 0.0;
    } else {
      const double u_cmd = clamp_d(u_hat_f[i] + res.x[i], 0.0, static_cast<double>(u_max_f[i]));
      const double w_cmd = clamp_d(std::sqrt(u_cmd / k_t[i]), params::kOmegaIdle, w_cap);
      omega_cmd_out[i] = static_cast<float>(w_cmd);
      omega_cmd_delay_[delay_idx_][i] = w_cmd;
    }
  }
  delay_idx_ = (delay_idx_ + 1) % 2;
  return res;
}

}  // namespace vertirta::core
