// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "vertirta/airframes/airframe_dynamics.hpp"
#include "vertirta/core/coax_envelope.hpp"

namespace vertirta::airframes {

// Canonical adapter wrapping frozen VertiRTA-X8-240 parameters (config/vehicle_x8_240.yaml).
// Explicitly enforces 01-DR08-13 Override #1: D=1.50 m (R=0.75 m), z/D=0.233, I=[85,95,140],
// tau_up/down=85/120 ms, omega_max=240 rad/s, v_max=22.0 m/s (zero DR-09 D=1.65 m drift).
class CoaxialX8AirframeAdapter final : public IAirframeDynamics {
 public:
  explicit CoaxialX8AirframeAdapter(float bus_voltage_v = 403.2f) noexcept
      : bus_voltage_v_(bus_voltage_v) {}

  [[nodiscard]] uint8_t actuator_count() const noexcept override { return 8u; }
  [[nodiscard]] uint8_t control_count() const noexcept override { return 4u; }
  [[nodiscard]] double mass_kg() const noexcept override { return core::params::kMassKg; }
  [[nodiscard]] double max_airspeed_mps() const noexcept override { return 22.0; }

  [[nodiscard]] double rotor_diameter_m() const noexcept {
    return 2.0 * core::params::kRotorRadiusM;  // Exact 1.50 m
  }
  [[nodiscard]] double coax_z_over_d() const noexcept {
    return core::params::kCoaxSpacingM / rotor_diameter_m();  // Exact 0.35 / 1.50 = 0.2333...
  }

  [[nodiscard]] AirframeDerivatives eval_dynamics(
      const AirframeState6Dof& state,
      const core::Vec3Ned& wind_ned,
      uint16_t fault_mask) const noexcept override;

  void compute_effectiveness(
      const AirframeState6Dof& state,
      const core::Vec3Ned& wind_ned,
      uint16_t fault_mask,
      double dt_s,
      EffectivenessMatrix& out) const noexcept override;

 private:
  float bus_voltage_v_{403.2f};
};

}  // namespace vertirta::airframes
