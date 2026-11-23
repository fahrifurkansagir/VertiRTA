// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

#include "vertirta/core/frames.hpp"

namespace vertirta::airframes {

inline constexpr uint8_t kMaxActuators = 12u;
inline constexpr uint8_t kMaxControls = 5u;  // [p_dot, q_dot, r_dot, a_z_body, a_x_body]

struct AirframeState6Dof {
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  core::QuatPx4FrdNed q_frd_ned{};
  core::Vec3Frd omega_frd{};
  // Actual physical actuator states:
  //   X8-240:            [0..7] rotor thrusts [N]
  //   Lift+Cruise (8+1): [0..7] lift thrusts [N], [8] pusher thrust [N], [9..11] aileron/elevator/rudder [rad]
  //   Lift+Cruise (4+1): [0..3] lift thrusts [N], [4] pusher thrust [N], [5..7] aileron/elevator/rudder [rad]
  std::array<double, kMaxActuators> u_act{};
};

struct AirframeDerivatives {
  core::Vec3Ned pos_dot_ned{};
  core::Vec3Ned vel_dot_ned{};
  core::Vec3Frd omega_dot_frd{};
  // Specific force in body FRD (non-gravitational acceleration F_aero_prop / m)
  core::Vec3Frd accel_body_frd{};
  double airspeed_mps{0.0};
  double alpha_rad{0.0};
  double beta_rad{0.0};
  double load_factor_nz{1.0};
};

struct EffectivenessMatrix {
  std::array<std::array<double, kMaxActuators>, kMaxControls> B{};
  std::array<double, kMaxActuators> u_min{};
  std::array<double, kMaxActuators> u_max{};
  std::array<double, kMaxActuators> du_min{};
  std::array<double, kMaxActuators> du_max{};
  std::array<double, kMaxActuators> tau_act_s{};
  uint8_t n_actuators{0u};
  uint8_t n_controls{0u};
};

struct OeiTrimAuditResult {
  bool trim_feasible_4dof{false};
  uint8_t effective_rank{0u};
  double peak_thrust_n{0.0};
  double peak_thrust_ratio{0.0};
  double dynamic_margin_ratio{0.0};
  std::array<double, kMaxActuators> u_trim{};
};

// Zero-allocation stack-only polymorphic airframe dynamics interface (400 Hz safe).
class IAirframeDynamics {
 public:
  virtual ~IAirframeDynamics() = default;

  [[nodiscard]] virtual uint8_t actuator_count() const noexcept = 0;
  [[nodiscard]] virtual uint8_t control_count() const noexcept = 0;
  [[nodiscard]] virtual double mass_kg() const noexcept = 0;
  [[nodiscard]] virtual double max_airspeed_mps() const noexcept = 0;

  [[nodiscard]] virtual AirframeDerivatives eval_dynamics(
      const AirframeState6Dof& state,
      const core::Vec3Ned& wind_ned,
      uint16_t fault_mask) const noexcept = 0;

  virtual void compute_effectiveness(
      const AirframeState6Dof& state,
      const core::Vec3Ned& wind_ned,
      uint16_t fault_mask,
      double dt_s,
      EffectivenessMatrix& out) const noexcept = 0;
};

}  // namespace vertirta::airframes
