// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

#include "vertirta/airframes/airframe_dynamics.hpp"

namespace vertirta::airframes {

enum class LiftCruiseLayout : uint8_t {
  kEightPlusOne = 0,  // Recommended primary: 8 isolated lift rotors + 1 pusher + 3 aero surfaces (12 actuators)
  kFourPlusOne = 1,   // PX4 gz_standard_vtol compat: 4 lift rotors + 1 pusher + 3 aero surfaces (8 actuators)
};

// 240 kg Lift+Cruise transition eVTOL dynamics (VertiRTA-LC-240, Plan 5A Task 1).
// Supports both 8+1 (hover OEI <= 79.2% T_max) and 4+1 (PX4 standard_vtol compat) with full
// 6-DOF FRD sigmoid post-stall wing aerodynamics, I_xz inertia coupling, edgewise rotor drag,
// and relative-airspeed coastal wake coupling. 90 kt (46.3 m/s V_NE) applies strictly here.
class LiftCruiseDynamics final : public IAirframeDynamics {
 public:
  explicit LiftCruiseDynamics(LiftCruiseLayout layout = LiftCruiseLayout::kEightPlusOne) noexcept;

  [[nodiscard]] LiftCruiseLayout layout() const noexcept { return layout_; }
  [[nodiscard]] uint8_t lift_rotor_count() const noexcept {
    return (layout_ == LiftCruiseLayout::kEightPlusOne) ? 8u : 4u;
  }
  [[nodiscard]] uint8_t pusher_index() const noexcept { return lift_rotor_count(); }
  [[nodiscard]] uint8_t aileron_index() const noexcept { return static_cast<uint8_t>(lift_rotor_count() + 1u); }
  [[nodiscard]] uint8_t elevator_index() const noexcept { return static_cast<uint8_t>(lift_rotor_count() + 2u); }
  [[nodiscard]] uint8_t rudder_index() const noexcept { return static_cast<uint8_t>(lift_rotor_count() + 3u); }

  [[nodiscard]] uint8_t actuator_count() const noexcept override {
    return static_cast<uint8_t>(lift_rotor_count() + 4u);  // lift + pusher + 3 surfaces
  }
  [[nodiscard]] uint8_t control_count() const noexcept override { return 5u; }
  [[nodiscard]] double mass_kg() const noexcept override { return kMassKg; }
  [[nodiscard]] double cruise_airspeed_mps() const noexcept { return kCruiseAirspeedMps; }
  [[nodiscard]] double max_airspeed_mps() const noexcept override { return kVneAirspeedMps; }
  [[nodiscard]] double stall_speed_1g_mps() const noexcept { return kStallSpeed1gMps; }
  [[nodiscard]] double best_glide_airspeed_mps() const noexcept { return kBestGlideAirspeedMps; }
  [[nodiscard]] double best_glide_ld_ratio() const noexcept { return kBestGlideLd; }
  [[nodiscard]] double max_lift_rotor_thrust_n() const noexcept { return t_max_lift_n_; }

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

  // Exact analytical / minimax hover & mid-transition OEI controllability audit.
  [[nodiscard]] OeiTrimAuditResult audit_oei_trim_feasibility(
      double v_airspeed_mps,
      uint8_t failed_lift_rotor) const noexcept;

  // Aero sizing constants (exact closure matching DR-09 Table 1.1 & Sec 1.5)
  static constexpr double kMassKg = 240.0;
  static constexpr double kGravity = 9.80665;
  static constexpr double kWeightN = kMassKg * kGravity;  // 2353.596 N
  static constexpr double kRhoAir = 1.225;
  static constexpr double kWingAreaM2 = 5.20;
  static constexpr double kWingSpanM = 6.84;
  static constexpr double kMeanChordM = 0.760;
  static constexpr double kAspectRatio = (kWingSpanM * kWingSpanM) / kWingAreaM2;  // 8.9972
  static constexpr double kOswaldE = 0.82;
  static constexpr double kCd0 = 0.03198;
  static constexpr double kCl0 = 0.25;
  static constexpr double kClAlpha = 4.95;
  static constexpr double kClMax = 1.5507;
  static constexpr double kStallSpeed1gMps = 21.83;
  static constexpr double kBlendStartMps = 14.0;
  static constexpr double kBlendEndMps = 28.0;
  static constexpr double kCruiseAirspeedMps = 38.0;  // 73.9 kt nominal corridor cruise
  static constexpr double kVneAirspeedMps = 46.3;     // 90.0 kt Lift+Cruise envelope ceiling
  static constexpr double kBestGlideAirspeedMps = 29.30;
  static constexpr double kBestGlideLd = 13.46;

  static constexpr double kIxx = 68.0;
  static constexpr double kIyy = 82.0;
  static constexpr double kIzz = 128.0;
  static constexpr double kIxz = 4.2;  // Twin-boom / high-tail cross product of inertia

 private:
  LiftCruiseLayout layout_{LiftCruiseLayout::kEightPlusOne};
  double t_max_lift_n_{544.3};
  double t_max_pusher_n_{950.0};
  double max_surface_rad_{0.4363323129985824};  // 25 deg
  double k_torque_per_thrust_{0.0759};
  std::array<double, 8> rotor_x_m_{};
  std::array<double, 8> rotor_y_m_{};
  std::array<double, 8> rotor_z_m_{};
  std::array<int, 8> rotor_spin_{};
};

}  // namespace vertirta::airframes
