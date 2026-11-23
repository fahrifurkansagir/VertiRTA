// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/frames.hpp"

namespace vertirta::wind_acoustics {

inline constexpr std::size_t kAcousticNumBands = 27u;      // 1/3-octave 25 Hz .. 10 kHz
inline constexpr std::size_t kHemisphereAzNodes = 18u;     // 20 deg azimuth steps [0..340 deg]
inline constexpr std::size_t kHemisphereElNodes = 10u;     // 0 deg (horizon) .. 90 deg (nadir)
inline constexpr std::size_t kHemisphereTotalNodes = kHemisphereAzNodes * kHemisphereElNodes;  // 180
inline constexpr double kHemisphereRefRadiusM = 15.0;      // >= 10 * D_rotor far-field reference
inline constexpr double kPrefPaSq = 4.0e-10;               // (20 uPa)^2

// Canonical X8-240 geometry & propulsion constants locked to config/vehicle_x8_240.yaml (Override #1).
inline constexpr double kX8RotorDiameterM = 1.50;
inline constexpr double kX8RotorRadiusM = 0.75;
inline constexpr double kX8CoaxSpacingM = 0.35;
inline constexpr double kX8CoaxZOverD = kX8CoaxSpacingM / kX8RotorDiameterM;  // 0.233333...
inline constexpr int kX8BladesPerRotor = 3;
inline constexpr double kX8MeanChordM = 0.095;
inline constexpr double kX8HoverRpm = 1449.58;   // 151.8 rad/s * 60 / (2*pi)
inline constexpr double kX8MaxRpm = 2291.83;     // 240.0 rad/s * 60 / (2*pi)

// Canonical Lift+Cruise 8+1 rotor/pusher geometry constants.
inline constexpr double kLcLiftRotorDiameterM = 1.30;
inline constexpr double kLcPusherDiameterM = 1.35;

enum class AcousticAirframeKind : uint8_t {
  kCoaxialX8_240 = 0,
  kLiftCruise_8p1 = 1,
};

struct AcousticFlightState {
  AcousticAirframeKind airframe{AcousticAirframeKind::kCoaxialX8_240};
  double total_mass_kg{240.0};
  double airspeed_mps{0.0};
  double flight_path_angle_deg{0.0};   // Negative = descent; shallow BVI in [-12, -2] deg
  double heading_enu_rad{0.0};         // 0 = East, pi/2 = North
  double lift_rpm{kX8HoverRpm};
  double lift_thrust_total_n{2353.596};
  double pusher_rpm{0.0};
  double pusher_thrust_n{0.0};
  double coax_z_over_d{kX8CoaxZOverD};  // Overrideable for parametric coaxial z/D audits
  double inflow_turbulence_rms_mps{0.8};
  bool lift_rotors_locked{false};
};

struct AcousticHemisphere180 {
  // Stored as acoustic pressure-squared normalized by p_ref^2 (10^(SPL/10)) at r_ref = 15 m
  // for zero-pow/zero-exp fast bilinear propagation across 64+ ground receptors.
  std::array<std::array<float, kAcousticNumBands>, kHemisphereTotalNodes> linear_p2{};
  double heading_enu_rad{0.0};
  double airspeed_mps{0.0};
  double speed_of_sound_mps{343.2};
  bool valid{false};
};

struct GroundReceptor {
  core::Vec3Enu pos_enu{};
  double ground_factor_g{0.2};  // ISO 9613-2:2024 coastal concrete/water G = 0.2
};

// Exact ISO 9613-1:1993 pure-tone atmospheric attenuation coefficient [dB/km].
[[nodiscard]] double compute_iso9613_1_alpha_db_per_km(
    double freq_hz,
    double temp_c = 20.0,
    double rh_pct = 70.0,
    double pressure_kpa = 101.325) noexcept;

// Exact IEC 61672-1 A-weighting gain [dB] at frequency `freq_hz`.
[[nodiscard]] double compute_iec61672_a_weight_db(double freq_hz) noexcept;

// Zero-heap < 1 ms 180-node coaxial BVI + NASA RP-1218 BPM (2025 non-compactness) +
// Amiet-Lowson + ISO 9613-1/2:2024 elevated-source C_geo aeroacoustic engine.
class AeroacousticEngine {
 public:
  explicit AeroacousticEngine(
      double temp_c = 20.0,
      double rh_pct = 70.0,
      double pressure_kpa = 101.325) noexcept;

  static const std::array<double, kAcousticNumBands>& band_center_freqs_hz() noexcept;

  [[nodiscard]] double alpha_atm_db_per_km(std::size_t band_idx) const noexcept {
    return (band_idx < kAcousticNumBands) ? alpha_db_per_km_[band_idx] : 0.0;
  }
  [[nodiscard]] double a_weight_db(std::size_t band_idx) const noexcept {
    return (band_idx < kAcousticNumBands) ? a_weight_db_[band_idx] : 0.0;
  }

  // Synthesize 18x10 = 180-node x 27-band hemisphere at r_ref = 15 m (zero heap).
  void synthesize_hemisphere(
      const AcousticFlightState& state,
      AcousticHemisphere180& out) const noexcept;

  // Propagate 180-node hemisphere to N ground receptors with Doppler, spherical spreading,
  // ISO 9613-1:1993 atmospheric absorption, ISO 9613-2:2024 C_geo ground factor, and A-weighting.
  void propagate_to_receptors(
      const AcousticHemisphere180& hemisphere,
      const core::Vec3Enu& aircraft_enu,
      const core::Vec3Enu& vel_enu,
      std::span<const GroundReceptor> receptors,
      std::span<double> out_dba) const noexcept;

  // Convenience single-receptor dBA evaluation.
  [[nodiscard]] double propagate_single_dba(
      const AcousticHemisphere180& hemisphere,
      const core::Vec3Enu& aircraft_enu,
      const core::Vec3Enu& vel_enu,
      const GroundReceptor& receptor) const noexcept;

 private:
  double temp_c_{20.0};
  double rh_pct_{70.0};
  double pressure_kpa_{101.325};
  double speed_of_sound_mps_{343.2};
  std::array<double, kAcousticNumBands> alpha_db_per_km_{};
  std::array<double, kAcousticNumBands> alpha_neper_per_m_{};
  std::array<double, kAcousticNumBands> a_weight_db_{};
  std::array<double, kAcousticNumBands> a_weight_linear_{};
};

}  // namespace vertirta::wind_acoustics
