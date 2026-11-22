#include "vertirta/core/sim_plant_6dof.hpp"

#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

namespace {

constexpr double kDt400 = 0.0025;

struct RigidBodyState {
  Vec3Ned pos;
  Vec3Ned vel;
  QuatPx4FrdNed q;
  Vec3Frd omega;
};

struct RigidBodyDeriv {
  Vec3Ned dpos;
  Vec3Ned dvel;
  QuatPx4FrdNed dq;
  Vec3Frd domega;
};

[[nodiscard]] QuatPx4FrdNed normalize_quat(const QuatPx4FrdNed& q) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-12 || !is_finite_d(n)) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  const double inv = 1.0 / n;
  return {q.w * inv, q.x * inv, q.y * inv, q.z * inv};
}

[[nodiscard]] Vec3Ned frd_to_ned_by_quat(const QuatPx4FrdNed& q, const Vec3Frd& v) noexcept {
  const double w = q.w, x = q.x, y = q.y, z = q.z;
  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - w * z);
  const double r02 = 2.0 * (x * z + w * y);
  const double r10 = 2.0 * (x * y + w * z);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - w * x);
  const double r20 = 2.0 * (x * z - w * y);
  const double r21 = 2.0 * (y * z + w * x);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);
  return {
      r00 * v.x + r01 * v.y + r02 * v.z,
      r10 * v.x + r11 * v.y + r12 * v.z,
      r20 * v.x + r21 * v.y + r22 * v.z,
  };
}

[[nodiscard]] Vec3Frd ned_to_frd_by_quat_local(const QuatPx4FrdNed& q, const Vec3Ned& v) noexcept {
  const double w = q.w, x = q.x, y = q.y, z = q.z;
  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - w * z);
  const double r02 = 2.0 * (x * z + w * y);
  const double r10 = 2.0 * (x * y + w * z);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - w * x);
  const double r20 = 2.0 * (x * z - w * y);
  const double r21 = 2.0 * (y * z + w * x);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);
  return {
      r00 * v.x + r10 * v.y + r20 * v.z,
      r01 * v.x + r11 * v.y + r21 * v.z,
      r02 * v.x + r12 * v.y + r22 * v.z,
  };
}

[[nodiscard]] RigidBodyDeriv eval_deriv(
    const RigidBodyState& s,
    double total_thrust_n,
    const Vec3Frd& rotor_torque_frd,
    const Vec3Ned& wind_ned) noexcept {
  const QuatPx4FrdNed q = normalize_quat(s.q);
  const Vec3Ned v_air_ned = s.vel - wind_ned;
  const double v_air_mag = v_air_ned.norm();

  // Parasitic drag in NED
  const double drag_coeff = 0.5 * params::kRhoAir * params::kFlatPlateAreaM2 * v_air_mag;
  const Vec3Ned f_drag_ned = v_air_ned * (-drag_coeff);

  // Rotor-disk flapping / fuselage gust tipping moment in FRD (<=45 N*m @ 15 m/s gust, DR-1 Sec 3.5)
  const Vec3Frd v_air_frd = ned_to_frd_by_quat_local(q, v_air_ned);
  constexpr double kGustMomentGain = 3.0;  // [N*m / (m/s)] -> 45 N*m @ 15 m/s
  const Vec3Frd tau_gust_frd{
      -kGustMomentGain * v_air_frd.y,
      +kGustMomentGain * v_air_frd.x,
      -2.0 * s.omega.z,  // Mild aerodynamic yaw damping
  };

  const Vec3Ned f_thrust_ned = frd_to_ned_by_quat(q, {0.0, 0.0, -total_thrust_n});
  const Vec3Ned acc_ned{
      (f_thrust_ned.x + f_drag_ned.x) / params::kMassKg,
      (f_thrust_ned.y + f_drag_ned.y) / params::kMassKg,
      (f_thrust_ned.z + f_drag_ned.z) / params::kMassKg + params::kGravity,
  };

  const Vec3Frd tau_tot = rotor_torque_frd + tau_gust_frd;
  const double jx = params::kJxx * s.omega.x;
  const double jy = params::kJyy * s.omega.y;
  const double jz = params::kJzz * s.omega.z;
  const Vec3Frd domega{
      (tau_tot.x - (s.omega.y * jz - s.omega.z * jy)) / params::kJxx,
      (tau_tot.y - (s.omega.z * jx - s.omega.x * jz)) / params::kJyy,
      (tau_tot.z - (s.omega.x * jy - s.omega.y * jx)) / params::kJzz,
  };

  // q_dot = 0.5 * q (x) [0, wx, wy, wz]
  const double wx = s.omega.x, wy = s.omega.y, wz = s.omega.z;
  const QuatPx4FrdNed dq{
      0.5 * (-q.x * wx - q.y * wy - q.z * wz),
      0.5 * (+q.w * wx + q.y * wz - q.z * wy),
      0.5 * (+q.w * wy - q.x * wz + q.z * wx),
      0.5 * (+q.w * wz + q.x * wy - q.y * wx),
  };

  return {s.vel, acc_ned, dq, domega};
}

[[nodiscard]] RigidBodyState add_scaled(const RigidBodyState& s, const RigidBodyDeriv& k, double dt) noexcept {
  return {
      s.pos + k.dpos * dt,
      s.vel + k.dvel * dt,
      {s.q.w + k.dq.w * dt, s.q.x + k.dq.x * dt, s.q.y + k.dq.y * dt, s.q.z + k.dq.z * dt},
      s.omega + k.domega * dt,
  };
}

}  // namespace

CoaxialPlant6Dof::CoaxialPlant6Dof() noexcept {
  reset();
}

void CoaxialPlant6Dof::reset(
    const Vec3Ned& pos_ned,
    const Vec3Ned& vel_ned,
    float initial_soc,
    double initial_rotor_omega) noexcept {
  battery_.reset(initial_soc);
  state_ = PlantState{};
  state_.pos_ned = pos_ned;
  state_.vel_ned = vel_ned;
  state_.q_frd_ned = {1.0, 0.0, 0.0, 0.0};
  state_.omega_frd = {0.0, 0.0, 0.0};
  for (int i = 0; i < 8; ++i) {
    state_.rotor_omega[i] = initial_rotor_omega;
  }
  state_.battery = battery_.state();
  state_.on_ground = (pos_ned.z >= 0.0);
  state_.touchdown_vz_mps = 0.0;
}

PlantState CoaxialPlant6Dof::step_400hz(
    const float omega_cmd_rad_s[8],
    uint8_t fault_mask,
    const Vec3Ned& wind_ned) noexcept {
  const float v_horiz = static_cast<float>(std::hypot(
      state_.vel_ned.x - wind_ned.x, state_.vel_ned.y - wind_ned.y));
  double k_t[8]{};
  double k_q[8]{};
  CoaxEffectiveness::effective_rotor_constants(fault_mask, v_horiz, k_t, k_q);

  const double w_cap = static_cast<double>(
      CoaxEffectiveness::omega_lim_from_vterm(state_.battery.v_term_v));

  // 1. Advance 8 rotor speeds with asymmetric BLDC spin-up/down & unpowered aerodynamic coast-down.
  double total_thrust = 0.0;
  Vec3Frd rotor_torque_frd{};
  float thrust_n[8]{};
  for (int i = 0; i < 8; ++i) {
    const bool dead = (fault_mask & static_cast<uint8_t>(1u << i)) != 0u;
    const double w_target = dead ? 0.0 : clamp_d(static_cast<double>(omega_cmd_rad_s[i]), 0.0, w_cap);
    const double tau = dead ? params::kTauCoast
                            : ((w_target >= state_.rotor_omega[i]) ? params::kTauUp : params::kTauDown);
    const double dw = dead ? (-k_q[i] * state_.rotor_omega[i] * state_.rotor_omega[i] / params::kJr)
                           : clamp_d((w_target - state_.rotor_omega[i]) / tau,
                                     -params::kAlphaMax, params::kAlphaMax);
    state_.rotor_omega[i] = clamp_d(state_.rotor_omega[i] + dw * kDt400, 0.0, params::kOmegaMax);

    // Unpowered rotor coasts down aerodynamically (producing decaying lift, zero stator EM yaw torque).
    const double ti = k_t[i] * state_.rotor_omega[i] * state_.rotor_omega[i];
    const double qi = dead ? 0.0 : (k_q[i] * state_.rotor_omega[i] * state_.rotor_omega[i] + params::kJr * dw);
    thrust_n[i] = static_cast<float>(ti);
    total_thrust += ti;
    rotor_torque_frd.x += -params::kRotorPosY[i] * ti;
    rotor_torque_frd.y += +params::kRotorPosX[i] * ti;
    rotor_torque_frd.z += static_cast<double>(params::kRotorSpinSign[i]) * qi;
  }

  // 2. Step 1-RC Thevenin battery from live propulsion electrical power.
  const float p_elec = CoaxEffectiveness::compute_electrical_power_w(thrust_n, fault_mask, v_horiz);
  state_.battery = battery_.step(static_cast<float>(kDt400), p_elec);

  // 3. RK4 rigid-body integration.
  const RigidBodyState s0{state_.pos_ned, state_.vel_ned, state_.q_frd_ned, state_.omega_frd};
  const RigidBodyDeriv k1 = eval_deriv(s0, total_thrust, rotor_torque_frd, wind_ned);
  const RigidBodyDeriv k2 = eval_deriv(add_scaled(s0, k1, 0.5 * kDt400), total_thrust, rotor_torque_frd, wind_ned);
  const RigidBodyDeriv k3 = eval_deriv(add_scaled(s0, k2, 0.5 * kDt400), total_thrust, rotor_torque_frd, wind_ned);
  const RigidBodyDeriv k4 = eval_deriv(add_scaled(s0, k3, kDt400), total_thrust, rotor_torque_frd, wind_ned);

  const double c6 = kDt400 / 6.0;
  state_.pos_ned = s0.pos + (k1.dpos + k2.dpos * 2.0 + k3.dpos * 2.0 + k4.dpos) * c6;
  state_.vel_ned = s0.vel + (k1.dvel + k2.dvel * 2.0 + k3.dvel * 2.0 + k4.dvel) * c6;
  state_.omega_frd = s0.omega + (k1.domega + k2.domega * 2.0 + k3.domega * 2.0 + k4.domega) * c6;
  state_.q_frd_ned = normalize_quat({
      s0.q.w + (k1.dq.w + 2.0 * k2.dq.w + 2.0 * k3.dq.w + k4.dq.w) * c6,
      s0.q.x + (k1.dq.x + 2.0 * k2.dq.x + 2.0 * k3.dq.x + k4.dq.x) * c6,
      s0.q.y + (k1.dq.y + 2.0 * k2.dq.y + 2.0 * k3.dq.y + k4.dq.y) * c6,
      s0.q.z + (k1.dq.z + 2.0 * k2.dq.z + 2.0 * k3.dq.z + k4.dq.z) * c6,
  });

  // 4. Ground plane contact at z_ned = 0.0 (AGL = 0 m).
  if (state_.pos_ned.z >= 0.0) {
    if (!state_.on_ground) {
      state_.touchdown_vz_mps = state_.vel_ned.z;
    }
    state_.on_ground = true;
    state_.pos_ned.z = 0.0;
    if (state_.vel_ned.z > 0.0) {
      state_.vel_ned.z = 0.0;
    }
  } else {
    state_.on_ground = false;
  }

  return state_;
}

}  // namespace vertirta::core
