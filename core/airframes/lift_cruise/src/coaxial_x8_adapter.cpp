// SPDX-License-Identifier: Apache-2.0
#include "vertirta/airframes/coaxial_x8_adapter.hpp"

#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::airframes {

namespace {

[[nodiscard]] core::Vec3Frd rotate_ned_to_frd(
    const core::QuatPx4FrdNed& q,
    const core::Vec3Ned& v) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  const double inv = (n > 1e-12) ? (1.0 / n) : 1.0;
  const double w = q.w * inv;
  const double x = q.x * inv;
  const double y = q.y * inv;
  const double z = q.z * inv;

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

[[nodiscard]] core::Vec3Ned rotate_frd_to_ned(
    const core::QuatPx4FrdNed& q,
    const core::Vec3Frd& v) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  const double inv = (n > 1e-12) ? (1.0 / n) : 1.0;
  const double w = q.w * inv;
  const double x = q.x * inv;
  const double y = q.y * inv;
  const double z = q.z * inv;

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

}  // namespace

AirframeDerivatives CoaxialX8AirframeAdapter::eval_dynamics(
    const AirframeState6Dof& state,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask) const noexcept {
  using namespace core::params;
  AirframeDerivatives d{};
  d.pos_dot_ned = state.vel_ned;

  const core::Vec3Ned v_rel_ned = state.vel_ned - wind_ned;
  const core::Vec3Frd v_rel_frd = rotate_ned_to_frd(state.q_frd_ned, v_rel_ned);
  const double v_air = v_rel_frd.norm();
  d.airspeed_mps = v_air;
  d.alpha_rad = std::atan2(v_rel_frd.z, std::max(std::abs(v_rel_frd.x), 0.25));
  d.beta_rad = std::asin(core::clamp_d(v_rel_frd.y / std::max(v_air, 0.25), -1.0, 1.0));

  const uint8_t fm8 = static_cast<uint8_t>(fault_mask & 0xFFu);
  const float v_horiz = static_cast<float>(std::hypot(v_rel_ned.x, v_rel_ned.y));
  double B_raw[4][8]{};
  float u_max_raw[8]{};
  core::CoaxEffectiveness::build_B(fm8, v_horiz, B_raw, u_max_raw, bus_voltage_v_);

  double tau_x = 0.0;
  double tau_y = 0.0;
  double tau_z = 0.0;
  double thrust_total = 0.0;
  for (int i = 0; i < 8; ++i) {
    const double u_i = ((fm8 & static_cast<uint8_t>(1u << i)) != 0u)
                           ? 0.0
                           : core::clamp_d(state.u_act[i], 0.0, static_cast<double>(u_max_raw[i]));
    tau_x += B_raw[0][i] * u_i;
    tau_y += B_raw[1][i] * u_i;
    tau_z += B_raw[2][i] * u_i;
    thrust_total += B_raw[3][i] * u_i;
  }

  // Parasitic flat-plate fuselage drag opposing relative airspeed
  const double q_dyn_drag = 0.5 * kRhoAir * kFlatPlateAreaM2 * v_air;
  const core::Vec3Frd f_body{
      -q_dyn_drag * v_rel_frd.x,
      -q_dyn_drag * v_rel_frd.y,
      -thrust_total - q_dyn_drag * v_rel_frd.z,
  };
  d.accel_body_frd = f_body * (1.0 / kMassKg);
  d.load_factor_nz = -d.accel_body_frd.z / kGravity;

  const core::Vec3Ned a_ned = rotate_frd_to_ned(state.q_frd_ned, d.accel_body_frd);
  d.vel_dot_ned = {a_ned.x, a_ned.y, a_ned.z + kGravity};

  // Euler rigid-body rotational equations with diagonal J = diag(85, 95, 140)
  const double p = state.omega_frd.x;
  const double q = state.omega_frd.y;
  const double r = state.omega_frd.z;
  d.omega_dot_frd = {
      (tau_x - (kJzz - kJyy) * q * r) / kJxx,
      (tau_y - (kJxx - kJzz) * r * p) / kJyy,
      (tau_z - (kJyy - kJxx) * p * q) / kJzz,
  };
  return d;
}

void CoaxialX8AirframeAdapter::compute_effectiveness(
    const AirframeState6Dof& state,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask,
    double dt_s,
    EffectivenessMatrix& out) const noexcept {
  using namespace core::params;
  out = EffectivenessMatrix{};
  out.n_actuators = 8u;
  out.n_controls = 4u;

  const uint8_t fm8 = static_cast<uint8_t>(fault_mask & 0xFFu);
  const core::Vec3Ned v_rel = state.vel_ned - wind_ned;
  const float v_horiz = static_cast<float>(std::hypot(v_rel.x, v_rel.y));
  double B_raw[4][8]{};
  float u_max_raw[8]{};
  core::CoaxEffectiveness::build_B(fm8, v_horiz, B_raw, u_max_raw, bus_voltage_v_);

  const double dt = core::clamp_d(dt_s, 1e-4, 0.05);
  for (uint8_t i = 0; i < 8u; ++i) {
    out.B[0][i] = B_raw[0][i] / kJxx;
    out.B[1][i] = B_raw[1][i] / kJyy;
    out.B[2][i] = B_raw[2][i] / kJzz;
    out.B[3][i] = -B_raw[3][i] / kMassKg;  // Positive thrust accelerates along -z (upward in FRD)
    out.B[4][i] = 0.0;

    out.u_min[i] = 0.0;
    out.u_max[i] = static_cast<double>(u_max_raw[i]);
    out.tau_act_s[i] = kTauUp;

    const double max_rate_up = out.u_max[i] / kTauUp;
    const double max_rate_dn = out.u_max[i] / kTauDown;
    out.du_min[i] = std::max(out.u_min[i] - state.u_act[i], -max_rate_dn * dt);
    out.du_max[i] = std::min(out.u_max[i] - state.u_act[i], max_rate_up * dt);
  }
}

}  // namespace vertirta::airframes
