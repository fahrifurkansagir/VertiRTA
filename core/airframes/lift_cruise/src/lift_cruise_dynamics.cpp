// SPDX-License-Identifier: Apache-2.0
#include "vertirta/airframes/lift_cruise_dynamics.hpp"

#include <algorithm>
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

// Smooth sigmoid blend between attached wing theory and flat-plate post-stall aerodynamics
// (consistent with gz::sim::systems::AdvancedLiftDrag).
void compute_cl_cd(double alpha_rad, double& cl_out, double& cd_out) noexcept {
  constexpr double kAlphaStall = 0.262;  // ~15.0 deg
  constexpr double kBlendM = 18.0;
  const double a_abs = std::abs(alpha_rad);
  const double e_pos = std::exp(-kBlendM * (alpha_rad - kAlphaStall));
  const double e_neg = std::exp(kBlendM * (alpha_rad + kAlphaStall));
  const double sigma = (1.0 + e_pos + e_neg) / ((1.0 + e_pos) * (1.0 + e_neg));
  // sigma -> 0 inside attached linear regime (|alpha| < kAlphaStall), sigma -> 1 in deep post-stall.
  const double sigma_stall = core::clamp_d(sigma, 0.0, 1.0);

  const double cl_lin = LiftCruiseDynamics::kCl0 + LiftCruiseDynamics::kClAlpha * alpha_rad;
  const double cl_flat = 2.0 * std::sin(alpha_rad) * std::cos(alpha_rad);
  cl_out = (1.0 - sigma_stall) * cl_lin + sigma_stall * cl_flat;

  constexpr double kInd = 1.0 / (core::kPi * LiftCruiseDynamics::kOswaldE * LiftCruiseDynamics::kAspectRatio);
  const double cd_attached = LiftCruiseDynamics::kCd0 + kInd * cl_lin * cl_lin;
  const double cd_flat = LiftCruiseDynamics::kCd0 + 1.85 * std::sin(a_abs) * std::sin(a_abs);
  cd_out = (1.0 - sigma_stall) * cd_attached + sigma_stall * cd_flat;
}

// Exact inverse of coupled roll-yaw inertia tensor with I_xz != 0.
void apply_inv_inertia(double mx, double my, double mz, core::Vec3Frd& omega_dot) noexcept {
  constexpr double kDet = LiftCruiseDynamics::kIxx * LiftCruiseDynamics::kIzz -
                          LiftCruiseDynamics::kIxz * LiftCruiseDynamics::kIxz;
  omega_dot.x = (LiftCruiseDynamics::kIzz * mx + LiftCruiseDynamics::kIxz * mz) / kDet;
  omega_dot.y = my / LiftCruiseDynamics::kIyy;
  omega_dot.z = (LiftCruiseDynamics::kIxz * mx + LiftCruiseDynamics::kIxx * mz) / kDet;
}

}  // namespace

LiftCruiseDynamics::LiftCruiseDynamics(LiftCruiseLayout layout) noexcept : layout_(layout) {
  if (layout_ == LiftCruiseLayout::kEightPlusOne) {
    // 8 isolated lift rotors on 4 booms (x = +-1.45 m, y_in = 1.20 m, y_out = 2.56 m).
    // Closed-form minimax hover OEI peak: W / (5 + y_in/y_out) = 2353.596 / 5.46875 = 430.37 N = 79.07% T_max.
    t_max_lift_n_ = 544.3;  // 1.85x nominal 294.2 N hover per rotor
    constexpr double kXf = 1.45;
    constexpr double kYin = 1.20;
    constexpr double kYout = 2.56;
    constexpr double kZ = -0.22;  // Slightly above CG in FRD
    rotor_x_m_ = {+kXf, +kXf, +kXf, +kXf, -kXf, -kXf, -kXf, -kXf};
    rotor_y_m_ = {-kYout, -kYin, +kYin, +kYout, -kYout, -kYin, +kYin, +kYout};
    rotor_z_m_ = {kZ, kZ, kZ, kZ, kZ, kZ, kZ, kZ};
    rotor_spin_ = {+1, -1, +1, -1, -1, +1, -1, +1};
  } else {
    // 4+1 PX4 gz_standard_vtol compat layout (4 isolated rotors, T_max = 1059.1 N = 1.80x nominal 588.4 N).
    t_max_lift_n_ = 1059.1;
    constexpr double kXf = 1.45;
    constexpr double kY = 1.65;
    constexpr double kZ = -0.22;
    rotor_x_m_ = {+kXf, +kXf, -kXf, -kXf, 0.0, 0.0, 0.0, 0.0};
    rotor_y_m_ = {-kY, +kY, -kY, +kY, 0.0, 0.0, 0.0, 0.0};
    rotor_z_m_ = {kZ, kZ, kZ, kZ, 0.0, 0.0, 0.0, 0.0};
    rotor_spin_ = {+1, -1, -1, +1, 0, 0, 0, 0};
  }
}

AirframeDerivatives LiftCruiseDynamics::eval_dynamics(
    const AirframeState6Dof& state,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask) const noexcept {
  AirframeDerivatives d{};
  d.pos_dot_ned = state.vel_ned;

  const core::Vec3Ned v_rel_ned = state.vel_ned - wind_ned;
  const core::Vec3Frd v_a_frd = rotate_ned_to_frd(state.q_frd_ned, v_rel_ned);
  const double v_air = v_a_frd.norm();
  d.airspeed_mps = v_air;

  const double u_a = v_a_frd.x;
  const double v_a = v_a_frd.y;
  const double w_a = v_a_frd.z;
  const double v_safe = std::max(v_air, 0.25);
  const double alpha = (v_air > 1.0) ? std::atan2(w_a, std::max(u_a, 0.25)) : 0.0;
  const double beta = (v_air > 1.0) ? std::asin(core::clamp_d(v_a / v_safe, -1.0, 1.0)) : 0.0;
  d.alpha_rad = alpha;
  d.beta_rad = beta;

  const double q_dyn = 0.5 * kRhoAir * v_air * v_air;
  double cl = 0.0;
  double cd = 0.0;
  compute_cl_cd(alpha, cl, cd);

  const double p = state.omega_frd.x;
  const double q = state.omega_frd.y;
  const double r = state.omega_frd.z;
  const double p_hat = (v_air > 2.0) ? (p * kWingSpanM / (2.0 * v_air)) : 0.0;
  const double q_hat = (v_air > 2.0) ? (q * kMeanChordM / (2.0 * v_air)) : 0.0;
  const double r_hat = (v_air > 2.0) ? (r * kWingSpanM / (2.0 * v_air)) : 0.0;

  const double da = core::clamp_d(state.u_act[aileron_index()], -max_surface_rad_, max_surface_rad_);
  const double de = core::clamp_d(state.u_act[elevator_index()], -max_surface_rad_, max_surface_rad_);
  const double dr = core::clamp_d(state.u_act[rudder_index()], -max_surface_rad_, max_surface_rad_);

  // Wing + empennage aerodynamic forces in wind axes -> transformed to body FRD
  const double lift_n = q_dyn * kWingAreaM2 * (cl + 0.35 * de);
  const double drag_n = q_dyn * kWingAreaM2 * (cd + 0.08 * de * de + 0.04 * da * da);
  const double side_n = q_dyn * kWingAreaM2 * (-0.35 * beta + 0.14 * dr);

  const double ca = std::cos(alpha);
  const double sa = std::sin(alpha);
  core::Vec3Frd f_body{
      -drag_n * ca + lift_n * sa,
      side_n,
      -drag_n * sa - lift_n * ca,
  };

  // Pusher propulsion (aligned along +x body FRD; slipstream augments empennage q_emp)
  const uint8_t p_idx = pusher_index();
  const bool pusher_failed = (fault_mask & static_cast<uint16_t>(1u << p_idx)) != 0u;
  const double t_pusher = pusher_failed ? 0.0 : core::clamp_d(state.u_act[p_idx], 0.0, t_max_pusher_n_);
  f_body.x += t_pusher;

  constexpr double kPusherDiskArea = core::kPi * 0.675 * 0.675;  // D_p = 1.35 m
  const double q_emp = q_dyn + 0.35 * (t_pusher / kPusherDiskArea);

  // Post-stall surface authority attenuation
  const double stall_atten = core::clamp_d(1.0 - 0.55 * std::max(0.0, std::abs(alpha) - 0.21) / 0.15, 0.25, 1.0);

  // Aerodynamic moments in FRD
  double mx = q_dyn * kWingAreaM2 * kWingSpanM *
              (-0.08 * beta - 0.52 * p_hat + 0.12 * r_hat + stall_atten * (0.20 * da + 0.015 * dr));
  double my = q_dyn * kWingAreaM2 * kMeanChordM * (0.015 - 0.65 * alpha - 12.0 * q_hat) +
              q_emp * kWingAreaM2 * kMeanChordM * stall_atten * (-1.20 * de);
  double mz = q_dyn * kWingAreaM2 * kWingSpanM * (0.10 * beta - 0.04 * p_hat - 0.18 * r_hat - 0.01 * da) +
              q_emp * kWingAreaM2 * kWingSpanM * stall_atten * (-0.14 * dr);

  // Distributed lift rotors: vertical thrust (-z FRD), edgewise H-force drag, reaction torque
  const uint8_t n_lift = lift_rotor_count();
  for (uint8_t i = 0; i < n_lift; ++i) {
    const bool failed = (fault_mask & static_cast<uint16_t>(1u << i)) != 0u;
    const double ti = failed ? 0.0 : core::clamp_d(state.u_act[i], 0.0, t_max_lift_n_);
    if (ti <= 1e-6) {
      continue;
    }
    // Edgewise hub drag (H-force) scales gently with advance ratio mu = u_a / v_tip
    const double h_force = 0.014 * ti * core::clamp_d(u_a / 122.8, -0.35, 0.35);
    f_body.x -= h_force;
    f_body.z -= ti;

    mx += -rotor_y_m_[i] * ti;  // +y rotor pulling -z creates negative roll (-Mx)
    my += +rotor_x_m_[i] * ti - rotor_z_m_[i] * h_force;  // +x rotor pulling -z pitches nose up (+My)
    mz += static_cast<double>(rotor_spin_[i]) * k_torque_per_thrust_ * ti;
  }

  // Hover-regime low-speed aerodynamic rotational damping (prevents drift in zero airspeed)
  mx -= 2.5 * p;
  my -= 3.0 * q;
  mz -= 2.0 * r;

  d.accel_body_frd = f_body * (1.0 / kMassKg);
  d.load_factor_nz = -d.accel_body_frd.z / kGravity;

  const core::Vec3Ned a_ned = rotate_frd_to_ned(state.q_frd_ned, d.accel_body_frd);
  d.vel_dot_ned = {a_ned.x, a_ned.y, a_ned.z + kGravity};

  // Rigid-body rotational dynamics with coupled I_xz term: J * omega_dot = M - omega x (J * omega)
  const double jw_x = kIxx * p - kIxz * r;
  const double jw_y = kIyy * q;
  const double jw_z = -kIxz * p + kIzz * r;
  const double rhs_x = mx - (q * jw_z - r * jw_y);
  const double rhs_y = my - (r * jw_x - p * jw_z);
  const double rhs_z = mz - (p * jw_y - q * jw_x);
  apply_inv_inertia(rhs_x, rhs_y, rhs_z, d.omega_dot_frd);

  return d;
}

void LiftCruiseDynamics::compute_effectiveness(
    const AirframeState6Dof& state,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask,
    double dt_s,
    EffectivenessMatrix& out) const noexcept {
  out = EffectivenessMatrix{};
  const uint8_t n_lift = lift_rotor_count();
  const uint8_t p_idx = pusher_index();
  const uint8_t a_idx = aileron_index();
  const uint8_t e_idx = elevator_index();
  const uint8_t r_idx = rudder_index();
  out.n_actuators = actuator_count();
  out.n_controls = 5u;

  const core::Vec3Ned v_rel_ned = state.vel_ned - wind_ned;
  const core::Vec3Frd v_a_frd = rotate_ned_to_frd(state.q_frd_ned, v_rel_ned);
  const double v_air = v_a_frd.norm();
  const double alpha = (v_air > 1.0) ? std::atan2(v_a_frd.z, std::max(v_a_frd.x, 0.25)) : 0.0;
  const double q_dyn = 0.5 * kRhoAir * v_air * v_air;

  const bool pusher_failed = (fault_mask & static_cast<uint16_t>(1u << p_idx)) != 0u;
  const double t_pusher = pusher_failed ? 0.0 : core::clamp_d(state.u_act[p_idx], 0.0, t_max_pusher_n_);
  constexpr double kPusherDiskArea = core::kPi * 0.675 * 0.675;
  const double q_emp = q_dyn + 0.35 * (t_pusher / kPusherDiskArea);
  const double stall_atten = core::clamp_d(1.0 - 0.55 * std::max(0.0, std::abs(alpha) - 0.21) / 0.15, 0.25, 1.0);

  const double dt = core::clamp_d(dt_s, 1e-4, 0.05);
  constexpr double kTauLift = 0.085;
  constexpr double kTauPusher = 0.095;
  constexpr double kTauServo = 0.020;  // 50 rad/s aero control surface servo

  // 1. Lift rotor columns (ALWAYS kept physically non-zero unless failed; transition blending happens in WLS weights!)
  for (uint8_t i = 0; i < n_lift; ++i) {
    const bool failed = (fault_mask & static_cast<uint16_t>(1u << i)) != 0u;
    out.tau_act_s[i] = kTauLift;
    out.u_min[i] = 0.0;
    out.u_max[i] = failed ? 0.0 : t_max_lift_n_;
    if (failed) {
      out.du_min[i] = -state.u_act[i];
      out.du_max[i] = -state.u_act[i];
      continue;
    }
    const double dh_dt = 0.014 * core::clamp_d(v_a_frd.x / 122.8, -0.35, 0.35);
    const double dmx = -rotor_y_m_[i];
    const double dmy = +rotor_x_m_[i] - rotor_z_m_[i] * dh_dt;
    const double dmz = static_cast<double>(rotor_spin_[i]) * k_torque_per_thrust_;
    core::Vec3Frd dw{};
    apply_inv_inertia(dmx, dmy, dmz, dw);

    out.B[0][i] = dw.x;
    out.B[1][i] = dw.y;
    out.B[2][i] = dw.z;
    out.B[3][i] = -1.0 / kMassKg;
    // Parasitic H-force is treated as an unmodeled INDI perturbation, never as an active airbrake channel.
    out.B[4][i] = 0.0;

    const double rate_lim = (t_max_lift_n_ / kTauLift) * dt;
    out.du_min[i] = std::max(out.u_min[i] - state.u_act[i], -rate_lim);
    out.du_max[i] = std::min(out.u_max[i] - state.u_act[i], rate_lim);
  }

  // 2. Pusher column
  out.tau_act_s[p_idx] = kTauPusher;
  out.u_min[p_idx] = 0.0;
  out.u_max[p_idx] = pusher_failed ? 0.0 : t_max_pusher_n_;
  if (pusher_failed) {
    out.du_min[p_idx] = -state.u_act[p_idx];
    out.du_max[p_idx] = -state.u_act[p_idx];
  } else {
    out.B[4][p_idx] = 1.0 / kMassKg;
    const double rate_lim = (t_max_pusher_n_ / kTauPusher) * dt;
    out.du_min[p_idx] = std::max(out.u_min[p_idx] - state.u_act[p_idx], -rate_lim);
    out.du_max[p_idx] = std::min(out.u_max[p_idx] - state.u_act[p_idx], rate_lim);
  }

  // 3. Aero surfaces: aileron, elevator, rudder
  const double dmx_da = q_dyn * kWingAreaM2 * kWingSpanM * stall_atten * 0.20;
  const double dmz_da = q_dyn * kWingAreaM2 * kWingSpanM * (-0.01);
  core::Vec3Frd dw_da{};
  apply_inv_inertia(dmx_da, 0.0, dmz_da, dw_da);
  out.B[0][a_idx] = dw_da.x;
  out.B[1][a_idx] = dw_da.y;
  out.B[2][a_idx] = dw_da.z;

  const double dmy_de = q_emp * kWingAreaM2 * kMeanChordM * stall_atten * (-1.20);
  core::Vec3Frd dw_de{};
  apply_inv_inertia(0.0, dmy_de, 0.0, dw_de);
  out.B[0][e_idx] = dw_de.x;
  out.B[1][e_idx] = dw_de.y;
  out.B[2][e_idx] = dw_de.z;
  out.B[3][e_idx] = -(q_dyn * kWingAreaM2 * 0.35) / kMassKg;

  const double dmx_dr = q_dyn * kWingAreaM2 * kWingSpanM * stall_atten * 0.015;
  const double dmz_dr = q_emp * kWingAreaM2 * kWingSpanM * stall_atten * (-0.14);
  core::Vec3Frd dw_dr{};
  apply_inv_inertia(dmx_dr, 0.0, dmz_dr, dw_dr);
  out.B[0][r_idx] = dw_dr.x;
  out.B[1][r_idx] = dw_dr.y;
  out.B[2][r_idx] = dw_dr.z;

  constexpr double kServoRateRadS = 4.0;
  for (uint8_t idx : {a_idx, e_idx, r_idx}) {
    out.tau_act_s[idx] = kTauServo;
    out.u_min[idx] = -max_surface_rad_;
    out.u_max[idx] = +max_surface_rad_;
    const double step_lim = kServoRateRadS * dt;
    out.du_min[idx] = std::max(out.u_min[idx] - state.u_act[idx], -step_lim);
    out.du_max[idx] = std::min(out.u_max[idx] - state.u_act[idx], step_lim);
  }
}

OeiTrimAuditResult LiftCruiseDynamics::audit_oei_trim_feasibility(
    double v_airspeed_mps,
    uint8_t failed_lift_rotor) const noexcept {
  OeiTrimAuditResult res{};
  const uint8_t n_lift = lift_rotor_count();
  if (failed_lift_rotor >= n_lift) {
    return res;
  }

  // Wing lift relief at forward airspeed (trim alpha ~ 8 deg = 0.140 rad during mid-transition OEI)
  const double v_air = std::max(0.0, v_airspeed_mps);
  const double q_dyn = 0.5 * kRhoAir * v_air * v_air;
  const double cl_oei = (v_air > 1.0) ? (kCl0 + kClAlpha * 0.140) : 0.0;  // ~0.943
  const double wing_lift_n = std::min(kWeightN, q_dyn * kWingAreaM2 * cl_oei);
  const double w_rem = std::max(0.0, kWeightN - wing_lift_n);

  if (layout_ == LiftCruiseLayout::kEightPlusOne) {
    // 8+1 retains full 4-DOF controllability across the entire speed envelope (0..38 m/s).
    res.effective_rank = 4u;
    const bool is_outboard = (failed_lift_rotor == 0u || failed_lift_rotor == 3u ||
                              failed_lift_rotor == 4u || failed_lift_rotor == 7u);
    const double y_in = 1.20;
    const double y_out = 2.56;
    const double ratio_io = y_in / y_out;

    if (is_outboard) {
      // Exact closed-form minimax solution: zero diagonally coupled inboard partner,
      // equalize the 4 loaded rotors at T_peak = W_rem / (5 + y_in / y_out),
      // and balance the opposite outboard pair at W_rem/2 - 2*T_peak.
      const double t_peak = w_rem / (5.0 + ratio_io);
      const double t_opp = std::max(0.0, 0.5 * w_rem - 2.0 * t_peak);

      // Map indices symmetrically relative to which outboard rotor failed:
      // Rotor pairs: Front = {0,1,2,3}, Rear = {4,5,6,7}
      const bool front = (failed_lift_rotor < 4u);
      const bool left = (failed_lift_rotor == 0u || failed_lift_rotor == 4u);
      const uint8_t f_out_same = front ? (left ? 0u : 3u) : (left ? 4u : 7u);
      const uint8_t f_in_same = front ? (left ? 1u : 2u) : (left ? 5u : 6u);
      const uint8_t f_in_opp = front ? (left ? 2u : 1u) : (left ? 6u : 5u);
      const uint8_t f_out_opp = front ? (left ? 3u : 0u) : (left ? 7u : 4u);
      const uint8_t r_out_same = front ? (left ? 4u : 7u) : (left ? 0u : 3u);
      const uint8_t r_in_same = front ? (left ? 5u : 6u) : (left ? 1u : 2u);
      const uint8_t r_in_opp = front ? (left ? 6u : 5u) : (left ? 2u : 1u);
      const uint8_t r_out_opp = front ? (left ? 7u : 4u) : (left ? 3u : 0u);

      res.u_trim[f_out_same] = 0.0;
      res.u_trim[f_in_same] = t_peak;
      res.u_trim[f_in_opp] = t_peak;
      res.u_trim[f_out_opp] = t_opp;
      res.u_trim[r_out_same] = t_peak;
      res.u_trim[r_in_same] = t_peak;
      res.u_trim[r_in_opp] = 0.0;
      res.u_trim[r_out_opp] = t_opp;
      res.peak_thrust_n = t_peak;
    } else {
      // Inboard rotor failure is strictly easier than outboard: zero diagonally opposite outboard
      // or redistribute across 7 rotors -> peak <= W_rem / 6.0 (<= 392.3 N = 72.1% T_max).
      const double t_peak = w_rem / 6.0;
      const double t_low = std::max(0.0, 0.5 * w_rem - 2.0 * t_peak);
      for (uint8_t i = 0; i < 8u; ++i) {
        res.u_trim[i] = (i == failed_lift_rotor) ? 0.0 : t_peak;
      }
      // Trim opposing same-row partner slightly lower to maintain exact zero roll/pitch
      const uint8_t opp_in = static_cast<uint8_t>(failed_lift_rotor ^ 3u);
      res.u_trim[opp_in] = t_low;
      res.peak_thrust_n = t_peak;
    }

    res.peak_thrust_ratio = res.peak_thrust_n / t_max_lift_n_;
    res.dynamic_margin_ratio = std::max(0.0, 1.0 - res.peak_thrust_ratio);
    res.trim_feasible_4dof = (res.peak_thrust_n <= t_max_lift_n_);
    return res;
  }

  // 4+1 Quadplane OEI analysis:
  // In hover (V_a < 8 m/s), losing 1 of 4 rotors leaves only 3 lift rotors -> rank 3 < 4 (no 4-DOF trim),
  // and even dropping yaw needs W/2 = 1176.8 N > T_max = 1059.1 N!
  // Above 8 m/s, aileron/elevator/rudder + wing lift restore full 4-DOF rank; aileron roll moment
  // relieves the remaining lift rotors cleanly for V_a >= 12-14 m/s.
  const double aileron_roll_nm = q_dyn * kWingAreaM2 * kWingSpanM * 0.20 * max_surface_rad_;
  const double rudder_yaw_nm = q_dyn * kWingAreaM2 * kWingSpanM * 0.14 * max_surface_rad_;
  const bool aero_restores_4dof = (v_air >= 8.0) && (aileron_roll_nm > 150.0) && (rudder_yaw_nm > 100.0);

  if (!aero_restores_4dof) {
    res.effective_rank = 3u;
    res.trim_feasible_4dof = false;
    res.peak_thrust_n = 0.5 * w_rem;
    res.peak_thrust_ratio = res.peak_thrust_n / t_max_lift_n_;
    res.dynamic_margin_ratio = std::max(0.0, 1.0 - res.peak_thrust_ratio);
    return res;
  }

  res.effective_rank = 4u;
  // With aileron counter-rolling up to aileron_roll_nm over lever y = 1.65 m,
  // the opposing same-row rotor can stay loaded up to delta_T = aileron_roll_nm / (2 * 1.65).
  const double y_arm = 1.65;
  const double roll_relief_n = std::min(0.25 * w_rem, aileron_roll_nm / (2.0 * y_arm));
  const double t_peak = std::max(w_rem / 3.0, 0.5 * w_rem - roll_relief_n);
  for (uint8_t i = 0; i < 4u; ++i) {
    res.u_trim[i] = (i == failed_lift_rotor) ? 0.0 : t_peak;
  }
  res.peak_thrust_n = t_peak;
  res.peak_thrust_ratio = t_peak / t_max_lift_n_;
  res.dynamic_margin_ratio = std::max(0.0, 1.0 - res.peak_thrust_ratio);
  res.trim_feasible_4dof = (t_peak <= t_max_lift_n_);
  return res;
}

}  // namespace vertirta::airframes
