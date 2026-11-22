#pragma once

#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"

namespace vertirta::core {

struct PlantState {
  Vec3Ned pos_ned{};
  Vec3Ned vel_ned{};
  QuatPx4FrdNed q_frd_ned{};
  Vec3Frd omega_frd{};
  double rotor_omega[8]{};
  BatteryState battery{};
  bool on_ground{false};
  double touchdown_vz_mps{0.0};
};

// Deterministic 400 Hz RK4 6-DOF coaxial X8 rigid-body plant with asymmetric motor lag,
// coaxial downwash/unblanking, rotor inertia reaction torque, 1-RC Thevenin battery sag,
// and wind drag/flapping moments. Shared by Plan 1A tests, Plan 1B Tier-1 10k Atlas, and Plan 3.
class CoaxialPlant6Dof {
 public:
  CoaxialPlant6Dof() noexcept;

  void reset(
      const Vec3Ned& pos_ned = {0.0, 0.0, -35.0},
      const Vec3Ned& vel_ned = {0.0, 0.0, 0.0},
      float initial_soc = 0.80f,
      double initial_rotor_omega = params::kOmegaHover) noexcept;

  [[nodiscard]] PlantState step_400hz(
      const float omega_cmd_rad_s[8],
      uint8_t fault_mask,
      const Vec3Ned& wind_ned) noexcept;

  [[nodiscard]] const PlantState& state() const noexcept { return state_; }

 private:
  PlantState state_{};
  TheveninBattery1RC battery_{0.80f};
};

}  // namespace vertirta::core
