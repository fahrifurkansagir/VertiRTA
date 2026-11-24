// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace vertirta::sim {

enum class RadarProfileKind : uint8_t {
  kTier1BCascadedMimo = 0,  // AWR2243 / ARS540 class (13 dBm, 16 dBi, 192 virtual ch, 150 m)
  kTier2ActivePa = 1,       // Active PA/LNA steered front-end (24 dBm, 20 dBi, 250 m)
};

enum class WireKind : uint8_t {
  kStrandedAcsr = 0,     // Helical stranded ACSR conductor (Bragg lobes at 7.0..14.1 deg)
  kSmoothCraneRope = 1,  // Smooth unstranded hoist cable (zero Bragg lobes)
};

enum class SwerlingModel : uint8_t {
  kSwerling0 = 0,    // Non-fluctuating reference (8.5 dB required SNR)
  kSwerlingI = 1,    // Slow scan-to-scan Rayleigh / exponential RCS (17.0 dB required SNR)
  kSwerlingIII = 2,  // Dominant scatterer + clutter chi-sq 4-DOF RCS (13.0 dB required SNR)
};

enum class DopplerTargetClass : uint8_t {
  kStaticOrHoverCandidate = 0,  // Inside 3-sigma static clutter gate (requires D-FINE-N cross-check)
  kDynamicIntruder = 1,         // Outside 3-sigma ego-compensated gate -> IntruderTrackerKf
};

struct Vec3f64 {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct MmWaveRadarConfig {
  RadarProfileKind profile{RadarProfileKind::kTier1BCascadedMimo};
  double carrier_hz{77.0e9};
  double tx_power_dbm{13.0};
  double antenna_gain_dbi{16.0};
  double virtual_channels{192.0};
  double bandwidth_hz{1.0e9};
  double cpi_duration_s{0.020};
  double noise_figure_db{12.0};
  double system_loss_db{9.1};
  double az_beamwidth_deg{1.4};
  double el_beamwidth_deg{2.0};
  double fov_az_half_deg{45.0};
  double fov_el_half_deg{20.0};
  double max_range_m{150.0};
  double sigma_v_radar_mps{0.15};
  double sigma_v_vio_mps{0.1517};
  double sigma_az_rad{0.026045};  // Calibrated: 3-sigma static gate 0.64 m/s boresight -> 1.01 m/s @ 20 m/s, 30 deg
  SwerlingModel swerling{SwerlingModel::kSwerlingI};
  uint64_t rng_seed{20261124ULL};

  [[nodiscard]] static MmWaveRadarConfig make_tier1b_cascaded() noexcept;
  [[nodiscard]] static MmWaveRadarConfig make_tier2_active_pa() noexcept;
};

struct WireTargetDesc {
  uint32_t target_id{0};
  Vec3f64 pos_sensor_m{};      // Closest illuminated point in sensor frame (+X fwd, +Y right, +Z down)
  double diameter_m{0.015};    // 10..25 mm conductor diameter
  double skew_angle_deg{0.0};  // Incidence skew away from wire normal [deg]
  double strand_pitch_m{0.016};
  double rain_rate_mm_hr{0.0};
  WireKind kind{WireKind::kStrandedAcsr};
};

struct PointTargetDesc {
  uint32_t target_id{0};
  Vec3f64 pos_sensor_m{};
  Vec3f64 vel_world_sensor_frame_mps{};  // Target world velocity expressed in sensor axes
  double mean_rcs_dbsm{-18.0};           // e.g. -18 dBsm quadcopter, -25 dBsm bird
  SwerlingModel swerling{SwerlingModel::kSwerlingI};
};

struct RadarDetection {
  uint32_t target_id{0};
  double range_m{0.0};
  double az_rad{0.0};
  double el_rad{0.0};
  double radial_vel_mps{0.0};
  double doppler_residual_mps{0.0};
  double static_gate_3sigma_mps{0.64};
  double rcs_dbsm{-50.0};
  double snr_db{-100.0};
  DopplerTargetClass doppler_class{DopplerTargetClass::kStaticOrHoverCandidate};
  bool is_wire_return{false};
};

struct DopplerClassificationResult {
  double doppler_residual_mps{0.0};
  double static_gate_3sigma_mps{0.64};
  DopplerTargetClass target_class{DopplerTargetClass::kStaticOrHoverCandidate};
};

// Deterministic PCG/SplitMix64 PRNG (zero std::rand, zero heap).
class DeterministicRng {
 public:
  explicit DeterministicRng(uint64_t seed = 20261124ULL) noexcept : state_(seed) {}
  [[nodiscard]] uint64_t next_u64() noexcept;
  [[nodiscard]] double uniform_open01() noexcept;  // (0, 1)
  [[nodiscard]] double exponential_unit() noexcept;
  [[nodiscard]] double gamma_k2_unit_mean() noexcept;

 private:
  uint64_t state_{20261124ULL};
};

// Pure C++20 zero-heap 77 GHz 4D mmWave radar physics engine (DR-10 §1.4).
class MmWave77GHzPhysics {
 public:
  static constexpr double kSpeedOfLightMps = 299792458.0;
  static constexpr double kBoltzmannThermalDensityWHz = 4.0039e-21;  // k_B * 290 K
  static constexpr double kGrazingCliffDeg = 20.0;
  static constexpr double kGrazingFloorDbsm = -48.0;

  explicit MmWave77GHzPhysics(const MmWaveRadarConfig& cfg = MmWaveRadarConfig::make_tier1b_cascaded()) noexcept;

  [[nodiscard]] const MmWaveRadarConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] double wavelength_m() const noexcept { return kSpeedOfLightMps / cfg_.carrier_hz; }

  // Analytical wire scattering RCS [dBsm]:
  //   - Normal Fresnel glint (|skew| <= 2.5 deg): sigma = (pi/2)*d*R -> +1.96..+5.94 dBsm @ 100 m
  //   - Helical ACSR Bragg diffraction lobes in [7.0, 14.1 deg], ceasing > 15.0 deg
  //   - Smooth crane hoist rope: zero Bragg lobes
  //   - Grazing skew cliff (> 20.0 deg): plunges <= -45.0 dBsm
  [[nodiscard]] double compute_wire_rcs_dbsm(
      double diameter_m,
      double range_m,
      double skew_angle_deg,
      WireKind kind,
      double strand_pitch_m = 0.016,
      double rain_rate_mm_hr = 0.0) const noexcept;

  // Monostatic received power [dBm] and post-CPI SNR [dB] for arbitrary RCS [dBsm] and range [m].
  [[nodiscard]] double compute_received_power_dbm(double rcs_dbsm, double range_m) const noexcept;
  [[nodiscard]] double compute_snr_db(double rcs_dbsm, double range_m) const noexcept;

  // Analytical required SNR [dB] for P_d = 0.90, P_fa = 1e-6 (DR-10 §1.4 table).
  [[nodiscard]] static double required_snr_db(SwerlingModel model) noexcept;

  // Stochastic Swerling RCS draw [dBsm] around mean_rcs_dbsm (deterministic via rng).
  [[nodiscard]] double sample_swerling_rcs_dbsm(
      double mean_rcs_dbsm,
      SwerlingModel model,
      DeterministicRng& rng) const noexcept;

  // Ego-motion-compensated Doppler residual and off-boresight azimuth leakage 3-sigma gate.
  [[nodiscard]] DopplerClassificationResult classify_doppler_target(
      double radial_vel_meas_mps,
      const Vec3f64& ego_vel_sensor_mps,
      double az_rad,
      double el_rad) const noexcept;

  // Simulate one scan over wire + point targets into fixed caller buffer (zero heap).
  [[nodiscard]] std::size_t simulate_scan(
      const Vec3f64& ego_vel_sensor_mps,
      std::span<const WireTargetDesc> wires,
      std::span<const PointTargetDesc> points,
      std::span<RadarDetection> out_detections,
      bool deterministic_mean_rcs = true) noexcept;

 private:
  MmWaveRadarConfig cfg_{};
  DeterministicRng rng_{};
};

}  // namespace vertirta::sim
