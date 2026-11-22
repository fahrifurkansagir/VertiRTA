#include "vertirta/core/motor_fdi.hpp"

#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

namespace {
constexpr uint16_t kEscPersistTicks = 10;     // 25 ms @ 400 Hz
constexpr uint16_t kWrenchPersistTicks = 20;  // 50 ms @ 400 Hz
constexpr double kWrenchNormThresh = 0.55;
constexpr double kCosineSimThresh = 0.88;
}  // namespace

MotorFdiObserver::MotorFdiObserver() noexcept {
  reset();
}

void MotorFdiObserver::reset() noexcept {
  for (int i = 0; i < 8; ++i) {
    esc_fault_ticks_[i] = 0;
    wrench_fault_ticks_[i] = 0;
  }
  last_isolation_latency_ms_ = 0;
}

uint8_t MotorFdiObserver::step_400hz(
    const Vec3Frd& gyro_f,
    const Vec3Frd& dgyro_f,
    float accel_z_f_frd,
    const float omega_hat[8],
    const float omega_esc[8],
    const bool esc_valid[8],
    float v_horiz_mps,
    float v_term_v,
    uint8_t current_mask) noexcept {
  uint8_t mask = current_mask;

  // Channel 1: ESC RPM innovation (r_omega > 12 rad/s & > 12% of omega_hat for 10 ticks = 25 ms)
  if (omega_esc != nullptr && esc_valid != nullptr) {
    for (int i = 0; i < 8; ++i) {
      if ((mask & static_cast<uint8_t>(1u << i)) != 0u) {
        continue;
      }
      if (esc_valid[i] && is_finite_f(omega_esc[i]) && is_finite_f(omega_hat[i])) {
        const float r_w = omega_hat[i] - omega_esc[i];
        const float min_ref = (omega_hat[i] > 40.0f) ? omega_hat[i] : 40.0f;
        if (r_w > 12.0f && r_w > 0.12f * min_ref) {
          if (++esc_fault_ticks_[i] >= kEscPersistTicks) {
            mask = static_cast<uint8_t>(mask | (1u << i));
            last_isolation_latency_ms_ = static_cast<uint16_t>(esc_fault_ticks_[i] * 2.5 + 0.5);
          }
        } else {
          esc_fault_ticks_[i] = 0;
        }
      }
    }
  }

  // Channel 2: Sensorless 4-DOF INDI wrench residual + weighted cosine similarity
  double B_T[4][8]{};
  float u_max_unused[8]{};
  double k_t[8]{};
  double k_q[8]{};
  CoaxEffectiveness::build_B(mask, v_horiz_mps, B_T, u_max_unused, v_term_v);
  CoaxEffectiveness::effective_rotor_constants(mask, v_horiz_mps, k_t, k_q);

  double T_hat[8]{};
  for (int i = 0; i < 8; ++i) {
    const bool dead = (mask & static_cast<uint8_t>(1u << i)) != 0u;
    const double w = dead ? 0.0 : clamp_d(static_cast<double>(omega_hat[i]), 0.0, params::kOmegaMax);
    T_hat[i] = k_t[i] * w * w;
  }

  double expected_w[4]{};
  for (int r = 0; r < 4; ++r) {
    for (int i = 0; i < 8; ++i) {
      expected_w[r] += B_T[r][i] * T_hat[i];
    }
  }

  // Measured rigid-body wrench: J*dOmega + Omega x (J*Omega) and F_z = -m * a_z_f
  const double jx = params::kJxx * gyro_f.x;
  const double jy = params::kJyy * gyro_f.y;
  const double jz = params::kJzz * gyro_f.z;
  const double meas_w[4] = {
      params::kJxx * dgyro_f.x + (gyro_f.y * jz - gyro_f.z * jy),
      params::kJyy * dgyro_f.y + (gyro_f.z * jx - gyro_f.x * jz),
      params::kJzz * dgyro_f.z + (gyro_f.x * jy - gyro_f.y * jx),
      -params::kMassKg * static_cast<double>(accel_z_f_frd),
  };

  constexpr double kDw[4] = {1.0 / 300.0, 1.0 / 300.0, 1.0 / 25.0, 1.0 / 300.0};
  double rw_s[4]{};
  double rw_norm2 = 0.0;
  for (int r = 0; r < 4; ++r) {
    rw_s[r] = kDw[r] * (meas_w[r] - expected_w[r]);
    rw_norm2 += rw_s[r] * rw_s[r];
  }
  const double rw_norm = std::sqrt(rw_norm2);

  int best_i = -1;
  double best_rho = -1.0;
  if (rw_norm > kWrenchNormThresh) {
    for (int i = 0; i < 8; ++i) {
      if ((mask & static_cast<uint8_t>(1u << i)) != 0u) {
        continue;
      }
      double dot = 0.0;
      double bi_norm2 = 0.0;
      for (int r = 0; r < 4; ++r) {
        const double neg_b = -kDw[r] * B_T[r][i];
        dot += rw_s[r] * neg_b;
        bi_norm2 += neg_b * neg_b;
      }
      if (bi_norm2 > 1e-12) {
        const double rho = dot / (rw_norm * std::sqrt(bi_norm2));
        if (rho > best_rho) {
          best_rho = rho;
          best_i = i;
        }
      }
    }
  }

  for (int i = 0; i < 8; ++i) {
    if (i == best_i && best_rho > kCosineSimThresh) {
      if (++wrench_fault_ticks_[i] >= kWrenchPersistTicks) {
        mask = static_cast<uint8_t>(mask | (1u << i));
        last_isolation_latency_ms_ = static_cast<uint16_t>(wrench_fault_ticks_[i] * 2.5 + 0.5);
      }
    } else {
      wrench_fault_ticks_[i] = 0;
    }
  }

  return mask;
}

}  // namespace vertirta::core
