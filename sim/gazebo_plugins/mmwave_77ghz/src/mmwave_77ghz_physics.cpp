// SPDX-License-Identifier: Apache-2.0
#include "vertirta/sim/mmwave_77ghz_physics.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::sim {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

[[nodiscard]] inline double safe_log10(double v) noexcept {
  return std::log10(std::max(v, 1e-30));
}

[[nodiscard]] inline double db_to_lin(double db) noexcept {
  return std::pow(10.0, db * 0.1);
}

[[nodiscard]] inline double lin_to_db(double lin) noexcept {
  return 10.0 * safe_log10(lin);
}

}  // namespace

MmWaveRadarConfig MmWaveRadarConfig::make_tier1b_cascaded() noexcept {
  MmWaveRadarConfig cfg{};
  cfg.profile = RadarProfileKind::kTier1BCascadedMimo;
  cfg.carrier_hz = 77.0e9;
  cfg.tx_power_dbm = 13.0;
  cfg.antenna_gain_dbi = 16.0;
  cfg.virtual_channels = 192.0;
  cfg.bandwidth_hz = 1.0e9;
  cfg.cpi_duration_s = 0.020;
  cfg.noise_figure_db = 12.0;
  cfg.system_loss_db = 9.1;
  cfg.az_beamwidth_deg = 1.4;
  cfg.el_beamwidth_deg = 2.0;
  cfg.fov_az_half_deg = 45.0;
  cfg.fov_el_half_deg = 20.0;
  cfg.max_range_m = 150.0;
  cfg.sigma_v_radar_mps = 0.15;
  cfg.sigma_v_vio_mps = 0.1517;
  cfg.sigma_az_rad = 0.026045;
  cfg.swerling = SwerlingModel::kSwerlingI;
  return cfg;
}

MmWaveRadarConfig MmWaveRadarConfig::make_tier2_active_pa() noexcept {
  MmWaveRadarConfig cfg = make_tier1b_cascaded();
  cfg.profile = RadarProfileKind::kTier2ActivePa;
  cfg.tx_power_dbm = 24.0;
  cfg.antenna_gain_dbi = 20.0;
  cfg.noise_figure_db = 6.0;
  cfg.max_range_m = 250.0;
  return cfg;
}

uint64_t DeterministicRng::next_u64() noexcept {
  uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27u)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31u);
}

double DeterministicRng::uniform_open01() noexcept {
  // 53-bit double in (0, 1) strictly avoiding 0.0 and 1.0 for log().
  const uint64_t bits = next_u64() >> 11u;
  return (static_cast<double>(bits) + 0.5) * (1.0 / 9007199254740992.0);
}

double DeterministicRng::exponential_unit() noexcept {
  return -std::log(uniform_open01());
}

double DeterministicRng::gamma_k2_unit_mean() noexcept {
  // Sum of two independent Exp(1) has mean 2; divide by 2 -> unit mean chi-sq 4-DOF.
  return 0.5 * (exponential_unit() + exponential_unit());
}

MmWave77GHzPhysics::MmWave77GHzPhysics(const MmWaveRadarConfig& cfg) noexcept
    : cfg_(cfg), rng_(cfg.rng_seed) {}

double MmWave77GHzPhysics::compute_wire_rcs_dbsm(
    double diameter_m,
    double range_m,
    double skew_angle_deg,
    WireKind kind,
    double strand_pitch_m,
    double rain_rate_mm_hr) const noexcept {
  if (!std::isfinite(diameter_m) || !std::isfinite(range_m) || !std::isfinite(skew_angle_deg) ||
      diameter_m <= 0.0 || range_m <= 0.5) {
    return kGrazingFloorDbsm;
  }

  const double abs_skew = std::fabs(skew_angle_deg);
  if (abs_skew > kGrazingCliffDeg) {
    // Physical grazing/skew cliff past 20 deg: RCS plunges beneath urban clutter (<= -45 dBsm).
    const double extra_drop = std::min(6.0, 1.5 * (abs_skew - kGrazingCliffDeg));
    return std::min(-45.5 - extra_drop, -45.5);
  }

  // Stationary-phase Fresnel normal-incidence glint: sigma_glint = (pi / 2) * d * R.
  // At R=100 m: 10mm -> +1.96 dBsm, 15mm -> +3.72 dBsm, 20mm -> +4.97 dBsm, 25mm -> +5.94 dBsm.
  const double sigma_glint_lin = 0.5 * kPi * diameter_m * range_m;
  const double glint_rolloff = std::exp(-0.5 * (abs_skew / 1.15) * (abs_skew / 1.15));
  double sigma_lin = sigma_glint_lin * glint_rolloff;

  // Diffuse cylinder floor between 2.5 deg and 20 deg (~ -28 dBsm decaying smoothly toward -45 dBsm).
  const double t_grazing = std::clamp((abs_skew - 2.5) / (kGrazingCliffDeg - 2.5), 0.0, 1.0);
  const double diffuse_dbsm = -28.0 - 17.0 * t_grazing * t_grazing;
  sigma_lin += db_to_lin(diffuse_dbsm);

  // Helical stranded ACSR Bragg diffraction lobes (2 * p * sin(theta_m) = m * lambda),
  // strictly active in [7.0, 14.1 deg] and ceasing above 15.0 deg.
  if (kind == WireKind::kStrandedAcsr && abs_skew <= 15.0 && strand_pitch_m > 1e-4) {
    const double lambda = wavelength_m();
    // Rain fills inter-strand grooves, attenuating Bragg lobes up to 8 dB in heavy rain.
    const double rain_atten_db = std::clamp(0.16 * std::max(0.0, rain_rate_mm_hr), 0.0, 8.0);
    const double rain_factor = db_to_lin(-rain_atten_db);

    for (int m = 1; m <= 2; ++m) {
      const double sin_tm = (static_cast<double>(m) * lambda) / (2.0 * strand_pitch_m);
      if (sin_tm <= 0.0 || sin_tm >= 0.99) {
        continue;
      }
      const double theta_m_deg = std::asin(sin_tm) * kRadToDeg;
      if (theta_m_deg < 6.0 || theta_m_deg > 15.0) {
        continue;
      }
      const double delta = abs_skew - theta_m_deg;
      const double peak_dbsm = (m == 1) ? -9.5 : -12.5;
      const double lobe_lin = db_to_lin(peak_dbsm) * std::exp(-0.5 * (delta / 0.85) * (delta / 0.85));
      sigma_lin += lobe_lin * rain_factor;
    }
  }

  return std::max(kGrazingFloorDbsm, lin_to_db(sigma_lin));
}

double MmWave77GHzPhysics::compute_received_power_dbm(double rcs_dbsm, double range_m) const noexcept {
  if (!std::isfinite(rcs_dbsm) || !std::isfinite(range_m) || range_m <= 0.25) {
    return -200.0;
  }
  const double p_tx_w = db_to_lin(cfg_.tx_power_dbm - 30.0);
  const double g_lin = db_to_lin(cfg_.antenna_gain_dbi);
  const double lambda = wavelength_m();
  const double sigma_lin = db_to_lin(rcs_dbsm);
  const double loss_lin = db_to_lin(cfg_.system_loss_db);
  const double four_pi_cubed = std::pow(4.0 * kPi, 3.0);
  const double r4 = range_m * range_m * range_m * range_m;
  // Monostatic received power at antenna output including virtual MIMO array gain.
  const double p_rx_w =
      (p_tx_w * g_lin * g_lin * lambda * lambda * sigma_lin * cfg_.virtual_channels) /
      (four_pi_cubed * r4 * loss_lin);
  return lin_to_db(p_rx_w) + 30.0;
}

double MmWave77GHzPhysics::compute_snr_db(double rcs_dbsm, double range_m) const noexcept {
  const double p_rx_dbm = compute_received_power_dbm(rcs_dbsm, range_m);
  const double b_eff_hz = 1.0 / std::max(cfg_.cpi_duration_s, 1e-6);
  const double noise_w = kBoltzmannThermalDensityWHz * b_eff_hz * db_to_lin(cfg_.noise_figure_db);
  const double noise_dbm = lin_to_db(noise_w) + 30.0;
  return p_rx_dbm - noise_dbm;
}

double MmWave77GHzPhysics::required_snr_db(SwerlingModel model) noexcept {
  switch (model) {
    case SwerlingModel::kSwerling0:
      return 8.5;
    case SwerlingModel::kSwerlingIII:
      return 13.0;
    case SwerlingModel::kSwerlingI:
    default:
      return 17.0;
  }
}

double MmWave77GHzPhysics::sample_swerling_rcs_dbsm(
    double mean_rcs_dbsm,
    SwerlingModel model,
    DeterministicRng& rng) const noexcept {
  const double mean_lin = db_to_lin(mean_rcs_dbsm);
  double mult = 1.0;
  switch (model) {
    case SwerlingModel::kSwerling0:
      mult = 1.0;
      break;
    case SwerlingModel::kSwerlingI:
      mult = rng.exponential_unit();
      break;
    case SwerlingModel::kSwerlingIII:
      mult = rng.gamma_k2_unit_mean();
      break;
  }
  return std::max(-80.0, lin_to_db(mean_lin * mult));
}

DopplerClassificationResult MmWave77GHzPhysics::classify_doppler_target(
    double radial_vel_meas_mps,
    const Vec3f64& ego_vel_sensor_mps,
    double az_rad,
    double el_rad) const noexcept {
  const double ce = std::cos(el_rad);
  const Vec3f64 u_hat{ce * std::cos(az_rad), ce * std::sin(az_rad), -std::sin(el_rad)};
  const double ego_radial =
      ego_vel_sensor_mps.x * u_hat.x + ego_vel_sensor_mps.y * u_hat.y + ego_vel_sensor_mps.z * u_hat.z;
  const double v_res = radial_vel_meas_mps + ego_radial;

  const double v_ego_xy = std::hypot(ego_vel_sensor_mps.x, ego_vel_sensor_mps.y);
  const double sigma_leak = v_ego_xy * std::fabs(std::sin(az_rad)) * cfg_.sigma_az_rad;
  const double sigma_tot = std::sqrt(
      cfg_.sigma_v_radar_mps * cfg_.sigma_v_radar_mps +
      cfg_.sigma_v_vio_mps * cfg_.sigma_v_vio_mps +
      sigma_leak * sigma_leak);
  const double gate_3sigma = 3.0 * sigma_tot;

  DopplerClassificationResult out{};
  out.doppler_residual_mps = v_res;
  out.static_gate_3sigma_mps = gate_3sigma;
  out.target_class = (std::fabs(v_res) > gate_3sigma)
                         ? DopplerTargetClass::kDynamicIntruder
                         : DopplerTargetClass::kStaticOrHoverCandidate;
  return out;
}

std::size_t MmWave77GHzPhysics::simulate_scan(
    const Vec3f64& ego_vel_sensor_mps,
    std::span<const WireTargetDesc> wires,
    std::span<const PointTargetDesc> points,
    std::span<RadarDetection> out_detections,
    bool deterministic_mean_rcs) noexcept {
  std::size_t count = 0;
  const double fov_az_rad = cfg_.fov_az_half_deg * kDegToRad;
  const double fov_el_rad = cfg_.fov_el_half_deg * kDegToRad;

  for (const auto& w : wires) {
    if (count >= out_detections.size()) {
      break;
    }
    const double r = std::sqrt(
        w.pos_sensor_m.x * w.pos_sensor_m.x +
        w.pos_sensor_m.y * w.pos_sensor_m.y +
        w.pos_sensor_m.z * w.pos_sensor_m.z);
    if (r <= 0.5 || r > cfg_.max_range_m || w.pos_sensor_m.x <= 0.0) {
      continue;
    }
    const double az = std::atan2(w.pos_sensor_m.y, w.pos_sensor_m.x);
    const double el = std::atan2(-w.pos_sensor_m.z, std::hypot(w.pos_sensor_m.x, w.pos_sensor_m.y));
    if (std::fabs(az) > fov_az_rad || std::fabs(el) > fov_el_rad) {
      continue;
    }

    const double mean_rcs = compute_wire_rcs_dbsm(
        w.diameter_m, r, w.skew_angle_deg, w.kind, w.strand_pitch_m, w.rain_rate_mm_hr);
    const double rcs = deterministic_mean_rcs
                           ? mean_rcs
                           : sample_swerling_rcs_dbsm(mean_rcs, cfg_.swerling, rng_);
    const double snr = compute_snr_db(rcs, r);
    const double req_snr = required_snr_db(deterministic_mean_rcs ? cfg_.swerling : SwerlingModel::kSwerling0);
    if (snr < req_snr) {
      continue;
    }

    const double ce = std::cos(el);
    const Vec3f64 u_hat{ce * std::cos(az), ce * std::sin(az), -std::sin(el)};
    const double v_r_meas = -(ego_vel_sensor_mps.x * u_hat.x +
                              ego_vel_sensor_mps.y * u_hat.y +
                              ego_vel_sensor_mps.z * u_hat.z);
    const auto dcls = classify_doppler_target(v_r_meas, ego_vel_sensor_mps, az, el);

    out_detections[count++] = RadarDetection{
        w.target_id,
        r,
        az,
        el,
        v_r_meas,
        dcls.doppler_residual_mps,
        dcls.static_gate_3sigma_mps,
        rcs,
        snr,
        dcls.target_class,
        true,
    };
  }

  for (const auto& p : points) {
    if (count >= out_detections.size()) {
      break;
    }
    const double r = std::sqrt(
        p.pos_sensor_m.x * p.pos_sensor_m.x +
        p.pos_sensor_m.y * p.pos_sensor_m.y +
        p.pos_sensor_m.z * p.pos_sensor_m.z);
    if (r <= 0.5 || r > cfg_.max_range_m || p.pos_sensor_m.x <= 0.0) {
      continue;
    }
    const double az = std::atan2(p.pos_sensor_m.y, p.pos_sensor_m.x);
    const double el = std::atan2(-p.pos_sensor_m.z, std::hypot(p.pos_sensor_m.x, p.pos_sensor_m.y));
    if (std::fabs(az) > fov_az_rad || std::fabs(el) > fov_el_rad) {
      continue;
    }

    const double rcs = deterministic_mean_rcs
                           ? p.mean_rcs_dbsm
                           : sample_swerling_rcs_dbsm(p.mean_rcs_dbsm, p.swerling, rng_);
    const double snr = compute_snr_db(rcs, r);
    const double req_snr = required_snr_db(deterministic_mean_rcs ? p.swerling : SwerlingModel::kSwerling0);
    if (snr < req_snr) {
      continue;
    }

    const double ce = std::cos(el);
    const Vec3f64 u_hat{ce * std::cos(az), ce * std::sin(az), -std::sin(el)};
    const Vec3f64 v_rel{
        p.vel_world_sensor_frame_mps.x - ego_vel_sensor_mps.x,
        p.vel_world_sensor_frame_mps.y - ego_vel_sensor_mps.y,
        p.vel_world_sensor_frame_mps.z - ego_vel_sensor_mps.z,
    };
    const double v_r_meas = v_rel.x * u_hat.x + v_rel.y * u_hat.y + v_rel.z * u_hat.z;
    const auto dcls = classify_doppler_target(v_r_meas, ego_vel_sensor_mps, az, el);

    out_detections[count++] = RadarDetection{
        p.target_id,
        r,
        az,
        el,
        v_r_meas,
        dcls.doppler_residual_mps,
        dcls.static_gate_3sigma_mps,
        rcs,
        snr,
        dcls.target_class,
        false,
    };
  }

  return count;
}

}  // namespace vertirta::sim
