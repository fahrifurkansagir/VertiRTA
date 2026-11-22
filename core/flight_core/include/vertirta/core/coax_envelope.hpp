#pragma once

#include <cstdint>

#include "vertirta/core/frames.hpp"

namespace vertirta::core {

// Frozen VertiRTA-X8-240 SI physical constants (Spec v1.2 + DR-1 Audit Lock)
namespace params {
inline constexpr double kMassKg = 240.0;
inline constexpr double kGravity = 9.80665;
inline constexpr double kWeightN = kMassKg * kGravity;  // 2353.596 N
inline constexpr double kJxx = 85.0;
inline constexpr double kJyy = 95.0;
inline constexpr double kJzz = 140.0;
inline constexpr double kJPerp = 0.5 * (kJxx + kJyy);   // 90.0 kg*m^2
inline constexpr double kArmRadiusM = 1.35;
inline constexpr double kArmXyM = 0.9545941546018393;   // L / sqrt(2)
inline constexpr double kRotorRadiusM = 0.75;
inline constexpr double kDiskAreaM2 = kPi * kRotorRadiusM * kRotorRadiusM;  // 1.76714587 m^2
inline constexpr double kCoaxSpacingM = 0.35;
inline constexpr double kRhoAir = 1.225;

inline constexpr double kKT0 = 0.0138091;     // Upper thrust constant [N/(rad/s)^2]
inline constexpr double kKQ0 = 0.00104876;    // Upper torque constant [N*m/(rad/s)^2]
inline constexpr double kEtaCoax0 = 0.85;     // Lower-rotor hover thrust efficiency
inline constexpr double kSwirlRec0 = 0.96;    // Lower-rotor hover torque swirl recovery
inline constexpr double kKappaIso = 1.15;
inline constexpr double kKappaIntHover = 1.22;
inline constexpr double kSolidity = 0.1210;
inline constexpr double kCd0 = 0.011;
inline constexpr double kFlatPlateAreaM2 = 0.48;
inline constexpr double kDriveEff = 0.88;
inline constexpr double kAvionicsW = 450.0;

inline constexpr double kJr = 0.135;          // Rotor + BLDC bell inertia [kg*m^2]
inline constexpr double kTauUp = 0.085;       // Powered spin-up time constant [s]
inline constexpr double kTauDown = 0.120;     // Powered spin-down time constant [s]
inline constexpr double kTauCoast = 0.424;    // Unpowered aerodynamic coast-down [s]
inline constexpr double kAlphaMax = 487.7;    // Max rotor angular accel [rad/s^2]
inline constexpr double kOmegaIdle = 20.0;    // [rad/s]
inline constexpr double kOmegaHover = 151.8;  // [rad/s]
inline constexpr double kOmegaMax = 240.0;    // [rad/s] FOC software speed cap

// Voltage-headroom winding lock (DR-1): holds 2-arm T/W=1.25 to 358.4 V & >=1.12 to 338 V
inline constexpr double kKeBus = 1.36;        // [V*s/rad]
inline constexpr double kReffHot = 0.72;      // [Ohm]

// 112S12P 400 V / 22 kWh 1-RC Thevenin pack (320 Wh/kg, 68 kg mass-closure lock)
inline constexpr double kBattR0 = 0.1363;     // [Ohm]
inline constexpr double kBattR1 = 0.0840;     // [Ohm]
inline constexpr double kBattC1 = 300.0;      // [F]
inline constexpr double kBattTau1 = kBattR1 * kBattC1;  // 25.2 s
inline constexpr double kBattCapAh = 54.0;    // [Ah]

// PX4 Coaxial X8 rotor layout (FRD): 0..3 upper, 4..7 lower directly beneath
// Spin sign s_i: +1 = CCW (+tau_z FRD), -1 = CW (-tau_z FRD)
inline constexpr double kRotorPosX[8] = {
    +kArmXyM, -kArmXyM, +kArmXyM, -kArmXyM,
    +kArmXyM, -kArmXyM, +kArmXyM, -kArmXyM,
};
inline constexpr double kRotorPosY[8] = {
    +kArmXyM, -kArmXyM, -kArmXyM, +kArmXyM,
    +kArmXyM, -kArmXyM, -kArmXyM, +kArmXyM,
};
inline constexpr int kRotorSpinSign[8] = {+1, +1, -1, -1, -1, -1, +1, +1};
}  // namespace params

enum class FlightRegime : uint8_t {
  NOMINAL = 0,
  SINGLE_LOSS = 1,
  CROSS_DUAL = 2,
  SAME_ARM_ACTIVE_SPIN = 3,
  CRITICAL_DESCENT = 4,
};

struct CapabilityEnvelope {
  uint8_t fault_mask{0x00};
  uint8_t tmr_state{0};
  uint8_t imu_mask{0x07};
  FlightRegime regime{FlightRegime::NOMINAL};
  float t_max_n{5886.0f};
  float a_xy_max{6.87f};
  float tilt_max_rad{0.45f};
  float vz_sink_max{3.0f};
  float v_term_v{403.2f};
  float i_pack_a{92.0f};
  float soc{0.80f};
  float p_hover_deg_w{36230.0f};
  float spin_rate_ref{0.0f};
  float spin_rate_meas{0.0f};
  float chi2_residual{0.0f};
  uint16_t fdi_latency_ms{0};
};

struct BatteryState {
  float v_oc_v{403.2f};
  float v_rc_v{0.0f};
  float v_term_v{403.2f};
  float i_pack_a{0.0f};
  float soc{0.80f};
  bool voltage_collapse{false};
};

class CoaxEffectiveness {
 public:
  [[nodiscard]] static FlightRegime classify_regime(uint8_t fault_mask) noexcept;
  [[nodiscard]] static int failed_same_arm_index(uint8_t fault_mask) noexcept;
  [[nodiscard]] static float omega_lim_from_vterm(float v_term_v) noexcept;
  [[nodiscard]] static float two_arm_tw_at_vterm(float v_term_v) noexcept;
  [[nodiscard]] static double coax_overlap_factor(float v_horiz_mps, double omega_upper) noexcept;

  // Populates B_out (4x8 mapping rotor thrusts -> [tau_x, tau_y, tau_z, T] in FRD)
  // and per-rotor thrust ceiling u_max_n[8], applying upper-loss lower-rotor unblanking (+17.6%).
  static void build_B(
      uint8_t fault_mask,
      float v_horiz_mps,
      double B_out[4][8],
      float u_max_n[8],
      float v_term_v) noexcept;

  // Populates effective k_T_i and k_Q_i for each rotor accounting for coaxial interference & unblanking.
  static void effective_rotor_constants(
      uint8_t fault_mask,
      float v_horiz_mps,
      double k_t_out[8],
      double k_q_out[8]) noexcept;

  [[nodiscard]] static float compute_electrical_power_w(
      const float thrust_n[8],
      uint8_t fault_mask,
      float v_horiz_mps) noexcept;

  [[nodiscard]] static CapabilityEnvelope compute_envelope(
      uint8_t fault_mask,
      uint8_t tmr_state,
      uint8_t imu_mask,
      float v_term_v,
      float i_pack_a,
      float soc,
      float spin_rate_ref,
      float spin_rate_meas,
      float chi2_residual,
      uint16_t fdi_latency_ms) noexcept;
};

class TheveninBattery1RC {
 public:
  explicit TheveninBattery1RC(float initial_soc = 0.80f) noexcept;

  void reset(float initial_soc = 0.80f) noexcept;
  [[nodiscard]] static float voc_from_soc(float soc) noexcept;
  [[nodiscard]] BatteryState step(float dt_s, float p_elec_w) noexcept;
  [[nodiscard]] BatteryState state() const noexcept { return state_; }

 private:
  BatteryState state_{};
};

}  // namespace vertirta::core
