// SPDX-License-Identifier: Apache-2.0
#include "vertirta/wind_acoustics/aeroacoustics.hpp"

#include <algorithm>
#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::wind_acoustics {

namespace {

constexpr std::array<double, kAcousticNumBands> kBandFreqsHz{
    25.0,   31.5,   40.0,   50.0,   63.0,   80.0,   100.0,  125.0,  160.0,
    200.0,  250.0,  315.0,  400.0,  500.0,  630.0,  800.0,  1000.0, 1250.0,
    1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0,
};

constexpr double kLn10Over10 = 0.23025850929940458;  // ln(10) / 10

[[nodiscard]] inline double db_to_linear(double db) noexcept {
  return std::exp(db * kLn10Over10);
}

[[nodiscard]] inline std::size_t nearest_band_idx(double freq_hz) noexcept {
  if (freq_hz <= kBandFreqsHz.front()) {
    return 0u;
  }
  if (freq_hz >= kBandFreqsHz.back()) {
    return kAcousticNumBands - 1u;
  }
  // Exact 1/3-octave geometric spacing: band = round(10 * log10(f / 25))
  const double idx_f = 10.0 * std::log10(freq_hz / 25.0);
  const int idx = static_cast<int>(std::lround(idx_f));
  return static_cast<std::size_t>(std::clamp(idx, 0, static_cast<int>(kAcousticNumBands - 1u)));
}

// ISO 9613-2:2024 elevated-source geometric ground attenuation factor (linear energy multiplier).
// Replaces ISO 9613-2:1996's unphysical overhead dip by transitioning smoothly with elevation angle.
[[nodiscard]] inline double iso9613_2024_ground_linear(
    double dist_m,
    double elev_rad,
    double ground_factor_g) noexcept {
  const double g = std::clamp(ground_factor_g, 0.0, 1.0);
  const double sin_el = std::clamp(std::sin(elev_rad), 0.0, 1.0);
  // C_geo approaches 1 for elevated aircraft (h_s >> 10 m) and 0 at grazing incidence
  const double c_geo = 1.0 - std::exp(-2.4 * sin_el);
  // Hard coastal surface (G=0.2) provides ~+2.0 dB reflection overhead, soft grazing attenuates
  const double grazing_att_db = (1.0 - c_geo) * (3.2 * g) * std::min(1.0, dist_m / 120.0);
  const double elevated_refl_db = c_geo * (2.2 * (1.0 - 0.65 * g));
  const double net_ground_db = elevated_refl_db - grazing_att_db;
  return db_to_linear(net_ground_db);
}

}  // namespace

const std::array<double, kAcousticNumBands>& AeroacousticEngine::band_center_freqs_hz() noexcept {
  return kBandFreqsHz;
}

double compute_iso9613_1_alpha_db_per_km(
    double freq_hz,
    double temp_c,
    double rh_pct,
    double pressure_kpa) noexcept {
  if (freq_hz <= 0.0 || pressure_kpa <= 0.0) {
    return 0.0;
  }
  constexpr double kT0 = 293.15;
  constexpr double kT01 = 273.16;
  constexpr double kPr = 101.325;

  const double t_k = std::max(180.0, temp_c + 273.15);
  const double t_rel = t_k / kT0;
  const double p_rel = pressure_kpa / kPr;
  const double c_exp = -6.8346 * std::pow(kT01 / t_k, 1.261) + 4.6151;
  const double p_sat_ratio = std::pow(10.0, c_exp);
  const double h = std::max(0.0, rh_pct) * p_sat_ratio / p_rel;

  const double f_ro = p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
  const double f_rn =
      p_rel * std::pow(t_rel, -0.5) *
      (9.0 + 280.0 * h * std::exp(-4.170 * (std::pow(t_rel, -1.0 / 3.0) - 1.0)));

  const double f2 = freq_hz * freq_hz;
  const double term_cl = 1.84e-11 * (1.0 / p_rel) * std::sqrt(t_rel);
  const double term_o2 =
      0.01275 * std::exp(-2239.1 / t_k) / (f_ro + (f2 / std::max(f_ro, 1e-9)));
  const double term_n2 =
      0.1068 * std::exp(-3352.0 / t_k) / (f_rn + (f2 / std::max(f_rn, 1e-9)));
  const double alpha_db_per_m = 8.686 * f2 * (term_cl + std::pow(t_rel, -2.5) * (term_o2 + term_n2));
  return std::max(0.0, 1000.0 * alpha_db_per_m);
}

double compute_iec61672_a_weight_db(double freq_hz) noexcept {
  if (freq_hz <= 0.0) {
    return -100.0;
  }
  const double f2 = freq_hz * freq_hz;
  constexpr double kC1 = 12194.0 * 12194.0;
  constexpr double kC2 = 20.6 * 20.6;
  constexpr double kC3 = 107.7 * 107.7;
  constexpr double kC4 = 737.9 * 737.9;
  const double num = kC1 * f2 * f2;
  const double den = (f2 + kC2) * std::sqrt((f2 + kC3) * (f2 + kC4)) * (f2 + kC1);
  const double ra = num / std::max(den, 1e-30);
  return 20.0 * std::log10(std::max(ra, 1e-12)) + 2.00;
}

AeroacousticEngine::AeroacousticEngine(
    double temp_c,
    double rh_pct,
    double pressure_kpa) noexcept
    : temp_c_(temp_c),
      rh_pct_(rh_pct),
      pressure_kpa_(pressure_kpa),
      speed_of_sound_mps_(std::sqrt(1.4 * 287.05 * std::max(200.0, temp_c + 273.15))) {
  for (std::size_t b = 0; b < kAcousticNumBands; ++b) {
    alpha_db_per_km_[b] =
        compute_iso9613_1_alpha_db_per_km(kBandFreqsHz[b], temp_c_, rh_pct_, pressure_kpa_);
    alpha_neper_per_m_[b] = alpha_db_per_km_[b] * 1.0e-3 * kLn10Over10;
    a_weight_db_[b] = compute_iec61672_a_weight_db(kBandFreqsHz[b]);
    a_weight_linear_[b] = db_to_linear(a_weight_db_[b]);
  }
}

void AeroacousticEngine::synthesize_hemisphere(
    const AcousticFlightState& state,
    AcousticHemisphere180& out) const noexcept {
  out.heading_enu_rad = state.heading_enu_rad;
  out.airspeed_mps = std::max(0.0, state.airspeed_mps);
  out.speed_of_sound_mps = speed_of_sound_mps_;

  std::array<double, kAcousticNumBands> lift_bb{};
  std::array<double, kAcousticNumBands> lift_tonal{};
  std::array<double, kAcousticNumBands> push_bb{};
  std::array<double, kAcousticNumBands> push_tonal{};

  const bool is_coax = (state.airframe == AcousticAirframeKind::kCoaxialX8_240);
  const int n_lift_rotors = 8;  // Both X8-240 (4 coax pairs) and LC 8+1 (8 isolated lift rotors)
  const double d_lift = is_coax ? kX8RotorDiameterM : kLcLiftRotorDiameterM;
  const double r_lift = 0.5 * d_lift;
  const double chord_lift = is_coax ? kX8MeanChordM : 0.080;
  const int blades_lift = kX8BladesPerRotor;

  // 1) Lift-rotor broadband (8-strip NASA RP-1218 BPM + 2025 non-compactness + Amiet-Lowson LE)
  //    and 12-harmonic Lowson tonal + coaxial z/D BVI + shallow-descent BVI trigger
  if (!state.lift_rotors_locked && state.lift_rpm > 50.0 && state.lift_thrust_total_n > 1.0) {
    const double omega = state.lift_rpm * (2.0 * core::kPi / 60.0);
    const double v_tip = omega * r_lift;
    const double m_tip = std::clamp(v_tip / speed_of_sound_mps_, 0.05, 0.85);
    const double bpf_hz = (state.lift_rpm / 60.0) * static_cast<double>(blades_lift);
    const double t_per_rotor = state.lift_thrust_total_n / static_cast<double>(n_lift_rotors);
    const double t_norm = std::max(0.15, t_per_rotor / 294.2);  // Normalized to X8-240 nominal 294 N

    // Coaxial z/D aerodynamic interaction & BVI penalty (NASA TM-2018-219895 / Southampton scaling)
    double coax_bvi_db = 0.0;
    double u_rms_eff = std::max(0.25, state.inflow_turbulence_rms_mps);
    if (is_coax) {
      const double zd = std::clamp(state.coax_z_over_d, 0.08, 0.60);
      // Canonical z/D = 0.233 yields ~+5.8 dB coaxial wake-chopping penalty; tight z/D = 0.12 adds ~+5.5 dB more
      coax_bvi_db = 5.8 + 4.8 * std::exp(-(zd - 0.12) / 0.10) - 4.8 * std::exp(-(0.2333 - 0.12) / 0.10);
      u_rms_eff *= (1.75 + 0.30 / std::max(zd, 0.10));
    }

    // Shallow-descent BVI trigger (-12 deg <= gamma <= -2 deg with forward speed >= 4 m/s)
    double descent_bvi_db = 0.0;
    if (state.flight_path_angle_deg <= -2.0 && state.flight_path_angle_deg >= -12.0 &&
        state.airspeed_mps >= 4.0) {
      const double g_center = -6.0;
      const double g_shape = std::exp(-0.5 * ((state.flight_path_angle_deg - g_center) / 3.2) *
                                      ((state.flight_path_angle_deg - g_center) / 3.2));
      const double v_shape = std::clamp((state.airspeed_mps - 3.0) / 7.0, 0.0, 1.0);
      descent_bvi_db = (is_coax ? 5.2 : 3.2) * g_shape * v_shape;
    }

    // 8-strip blade integration (r/R = 0.30 .. 0.96)
    constexpr int kNumStrips = 8;
    const double dr = (0.96 - 0.30) * r_lift / static_cast<double>(kNumStrips);
    const double rotor_count_lin = static_cast<double>(n_lift_rotors);
    const double r_ref2 = kHemisphereRefRadiusM * kHemisphereRefRadiusM;

    for (int s = 0; s < kNumStrips; ++s) {
      const double r_frac = 0.30 + (static_cast<double>(s) + 0.5) * (0.66 / kNumStrips);
      const double v_loc = std::hypot(omega * (r_frac * r_lift), state.airspeed_mps * 0.5);
      const double m_loc = std::clamp(v_loc / speed_of_sound_mps_, 0.03, 0.85);
      // Approximate turbulent boundary layer displacement thickness delta* [m] (BPM RP-1218)
      const double re_c = std::max(5.0e4, v_loc * chord_lift / 1.5e-5);
      const double delta_star = chord_lift * 0.047 * std::pow(re_c, -0.2);
      const double f_peak_bpm = 0.10 * v_loc / std::max(delta_star, 1e-4);
      const double f_peak_le = 0.25 * v_loc / std::max(chord_lift, 0.02);

      // Base strip BPM TBL-TE + separation-stall energy scaled by M^5 * dr * delta* / r_ref^2
      const double bpm_level_db =
          81.5 + 50.0 * std::log10(m_loc / 0.33) +
          10.0 * std::log10((dr * delta_star / r_ref2) * 1.0e5 * rotor_count_lin) +
          8.0 * std::log10(t_norm) + 0.55 * coax_bvi_db + 0.45 * descent_bvi_db;
      const double bpm_lin = db_to_linear(bpm_level_db) / static_cast<double>(kNumStrips);

      // Amiet-Lowson leading-edge inflow turbulence broadband energy
      const double le_level_db =
          78.0 + 40.0 * std::log10(m_loc / 0.33) + 20.0 * std::log10(u_rms_eff / 1.0) +
          10.0 * std::log10((dr * chord_lift / r_ref2) * 1.0e4 * rotor_count_lin) +
          0.65 * coax_bvi_db + 0.55 * descent_bvi_db;
      const double le_lin = db_to_linear(le_level_db) / static_cast<double>(kNumStrips);

      for (std::size_t b = 0; b < kAcousticNumBands; ++b) {
        const double f = kBandFreqsHz[b];
        const double st_ratio = std::log10(std::max(f / std::max(f_peak_bpm, 20.0), 1e-3));
        // Classic BPM universal spectral shape A(St) + 2025 non-compactness high-frequency rolloff
        const double bpm_shape_db = -12.5 * st_ratio * st_ratio;
        const double k_c = (2.0 * core::kPi * f / speed_of_sound_mps_) * chord_lift;
        const double non_compact_factor = std::pow(1.0 + 0.25 * k_c * k_c, -0.35);

        const double le_ratio = std::log10(std::max(f / std::max(f_peak_le, 20.0), 1e-3));
        const double le_shape_db = -10.0 * le_ratio * le_ratio;

        lift_bb[b] += bpm_lin * db_to_linear(bpm_shape_db) * non_compact_factor +
                      le_lin * db_to_linear(le_shape_db);
      }
    }

    // 12-harmonic Lowson loading + thickness tonal spectrum + coaxial/descent BVI harmonics
    const double tonal_base_db =
        79.0 + 42.0 * std::log10(m_tip / 0.33) + 18.0 * std::log10(t_norm) +
        10.0 * std::log10(rotor_count_lin) + coax_bvi_db + descent_bvi_db;
    for (int h = 1; h <= 12; ++h) {
      const double f_h = static_cast<double>(h) * bpf_hz;
      if (f_h > 12000.0) {
        break;
      }
      // Coaxial & descent BVI sustain mid-order harmonics (h = 2..8) much more strongly
      const double decay_rate = (is_coax || descent_bvi_db > 0.5) ? 2.6 : 4.2;
      const double h_db = tonal_base_db - decay_rate * static_cast<double>(h - 1);
      const std::size_t b_idx = nearest_band_idx(f_h);
      lift_tonal[b_idx] += db_to_linear(h_db);
    }
  }

  // 2) Pusher propeller (Lift+Cruise forward cruise; 3-blade D_p = 1.35 m, horizontal thrust axis)
  if (state.pusher_rpm > 50.0 && state.pusher_thrust_n > 1.0) {
    const double omega_p = state.pusher_rpm * (2.0 * core::kPi / 60.0);
    const double r_p = 0.5 * kLcPusherDiameterM;
    const double v_tip_p = std::hypot(omega_p * r_p, state.airspeed_mps);
    const double m_tip_p = std::clamp(v_tip_p / speed_of_sound_mps_, 0.08, 0.85);
    const double bpf_p = (state.pusher_rpm / 60.0) * 3.0;
    const double t_norm_p = std::max(0.10, state.pusher_thrust_n / 200.0);

    // Pusher carries ~12x less thrust in cruise than hover lift (L/D ~ 13), yielding much quieter baseline
    const double p_bb_base_db =
        73.0 + 42.0 * std::log10(m_tip_p / 0.40) + 12.0 * std::log10(t_norm_p) +
        10.0 * std::log10(std::max(0.4, state.inflow_turbulence_rms_mps) / 0.8);
    const double p_bb_lin = db_to_linear(p_bb_base_db);
    const double f_peak_p = 0.16 * v_tip_p / 0.025;

    for (std::size_t b = 0; b < kAcousticNumBands; ++b) {
      const double f = kBandFreqsHz[b];
      const double st = std::log10(std::max(f / std::max(f_peak_p, 40.0), 1e-3));
      const double k_c = (2.0 * core::kPi * f / speed_of_sound_mps_) * 0.085;
      const double non_compact = std::pow(1.0 + 0.25 * k_c * k_c, -0.35);
      push_bb[b] = p_bb_lin * db_to_linear(-11.0 * st * st) * non_compact;
    }

    const double p_tonal_base_db =
        75.0 + 38.0 * std::log10(m_tip_p / 0.40) + 16.0 * std::log10(t_norm_p);
    for (int h = 1; h <= 12; ++h) {
      const double f_h = static_cast<double>(h) * bpf_p;
      if (f_h > 12000.0) {
        break;
      }
      const double h_db = p_tonal_base_db - 3.6 * static_cast<double>(h - 1);
      push_tonal[nearest_band_idx(f_h)] += db_to_linear(h_db);
    }
  }

  // 3) Populate 18 azimuth x 10 elevation = 180 hemisphere nodes (body-relative az, nadir el)
  //    el_idx 0 = horizon (0 deg below horizontal), el_idx 9 = straight down nadir (90 deg)
  for (std::size_t az_i = 0; az_i < kHemisphereAzNodes; ++az_i) {
    const double az_rad =
        (2.0 * core::kPi * static_cast<double>(az_i)) / static_cast<double>(kHemisphereAzNodes);
    const double sin_az = std::sin(az_rad);
    const double cos_az = std::cos(az_rad);
    const double sin2_az = sin_az * sin_az;

    for (std::size_t el_i = 0; el_i < kHemisphereElNodes; ++el_i) {
      const double el_rad =
          (0.5 * core::kPi * static_cast<double>(el_i)) / static_cast<double>(kHemisphereElNodes - 1u);
      const double sin_el = std::sin(el_rad);
      const double cos_el = std::cos(el_rad);

      // Lift rotor directivity: broadband peaks below disk (sin_el), tonal peaks in-plane / 30-50 deg
      const double d_lift_bb = 0.25 + 0.95 * (sin_el * sin_el);
      const double d_lift_tonal =
          0.30 + 0.90 * (cos_el * cos_el) + (is_coax ? 0.55 * sin_el * cos_el : 0.20 * sin_el);

      // Pusher 3D directivity: disk plane is Y-Z (perpendicular to +X flight axis).
      // Strong sideline dipole lobe (sin^2(az) * cos^2(el)) + wing overhead shielding near nadir (el -> 90 deg)
      // produces strictly higher SPL at 150 m sideline (el ~ 38.7 deg, az = +-90 deg) than overhead nadir.
      const double wing_shield_nadir = 1.0 - 0.82 * (sin_el * sin_el * sin_el * sin_el);
      const double sideline_lobe = 0.18 + 3.60 * sin2_az * (cos_el * cos_el) +
                                   0.45 * (cos_az * cos_az) * (cos_el * cos_el);
      const double d_push_bb = sideline_lobe * wing_shield_nadir + 0.12;
      const double d_push_tonal = (0.12 + 5.20 * sin2_az * (cos_el * cos_el)) * wing_shield_nadir + 0.06;

      const std::size_t node_idx = az_i * kHemisphereElNodes + el_i;
      auto& dst = out.linear_p2[node_idx];
      for (std::size_t b = 0; b < kAcousticNumBands; ++b) {
        const double total_lin = lift_bb[b] * d_lift_bb + lift_tonal[b] * d_lift_tonal +
                                 push_bb[b] * d_push_bb + push_tonal[b] * d_push_tonal;
        dst[b] = static_cast<float>(std::max(total_lin, 1.0));
      }
    }
  }

  out.valid = true;
}

double AeroacousticEngine::propagate_single_dba(
    const AcousticHemisphere180& hemisphere,
    const core::Vec3Enu& aircraft_enu,
    const core::Vec3Enu& vel_enu,
    const GroundReceptor& receptor) const noexcept {
  if (!hemisphere.valid || !aircraft_enu.is_finite() || !receptor.pos_enu.is_finite()) {
    return 0.0;
  }

  // Vector from aircraft to receptor in ENU
  const double dx = receptor.pos_enu.x - aircraft_enu.x;
  const double dy = receptor.pos_enu.y - aircraft_enu.y;
  const double dz = receptor.pos_enu.z - aircraft_enu.z;  // Negative when receptor is below aircraft
  const double r_horiz = std::hypot(dx, dy);
  const double r_dist = std::max(1.0, std::sqrt(dx * dx + dy * dy + dz * dz));

  // Elevation below horizon [0 .. pi/2]
  const double down_m = std::max(0.0, -dz);
  const double el_rad = std::atan2(down_m, std::max(r_horiz, 1e-6));

  // Body-relative azimuth [0 .. 2*pi)
  const double world_az = std::atan2(dy, dx);
  double rel_az = std::fmod(world_az - hemisphere.heading_enu_rad, 2.0 * core::kPi);
  if (rel_az < 0.0) {
    rel_az += 2.0 * core::kPi;
  }

  // Bilinear indices on 18x10 hemisphere
  const double az_f = rel_az * (static_cast<double>(kHemisphereAzNodes) / (2.0 * core::kPi));
  const std::size_t az0 = static_cast<std::size_t>(az_f) % kHemisphereAzNodes;
  const std::size_t az1 = (az0 + 1u) % kHemisphereAzNodes;
  const double waz = az_f - std::floor(az_f);

  const double el_f =
      std::clamp(el_rad / (0.5 * core::kPi), 0.0, 1.0) * static_cast<double>(kHemisphereElNodes - 1u);
  const std::size_t el0 = std::min(static_cast<std::size_t>(el_f), kHemisphereElNodes - 2u);
  const std::size_t el1 = el0 + 1u;
  const double wel = el_f - static_cast<double>(el0);

  const double w00 = (1.0 - waz) * (1.0 - wel);
  const double w10 = waz * (1.0 - wel);
  const double w01 = (1.0 - waz) * wel;
  const double w11 = waz * wel;

  const auto& n00 = hemisphere.linear_p2[az0 * kHemisphereElNodes + el0];
  const auto& n10 = hemisphere.linear_p2[az1 * kHemisphereElNodes + el0];
  const auto& n01 = hemisphere.linear_p2[az0 * kHemisphereElNodes + el1];
  const auto& n11 = hemisphere.linear_p2[az1 * kHemisphereElNodes + el1];

  // Doppler convective amplification (1 - M_r)^(-4) clamped safely for subsonic rotorcraft
  const double inv_r = 1.0 / r_dist;
  const double v_rad = (vel_enu.x * dx + vel_enu.y * dy + vel_enu.z * dz) * inv_r;
  const double m_rad = std::clamp(v_rad / std::max(hemisphere.speed_of_sound_mps, 250.0), -0.35, 0.35);
  const double doppler_gain = 1.0 / std::pow(1.0 - m_rad, 3.0);

  // Spherical spreading from r_ref = 15 m + ISO 9613-2:2024 elevated-source ground factor
  const double spread_gain = (kHemisphereRefRadiusM * kHemisphereRefRadiusM) / (r_dist * r_dist);
  const double ground_gain = iso9613_2024_ground_linear(r_dist, el_rad, receptor.ground_factor_g);
  const double common_gain = spread_gain * ground_gain * doppler_gain;

  double weighted_p2_sum = 0.0;
  for (std::size_t b = 0; b < kAcousticNumBands; ++b) {
    const double p2_ref = w00 * static_cast<double>(n00[b]) + w10 * static_cast<double>(n10[b]) +
                          w01 * static_cast<double>(n01[b]) + w11 * static_cast<double>(n11[b]);
    // ISO 9613-1 atmospheric absorption attenuation exp(-alpha_neper * r)
    const double atm_gain = std::exp(-alpha_neper_per_m_[b] * r_dist);
    weighted_p2_sum += p2_ref * common_gain * atm_gain * a_weight_linear_[b];
  }

  return 10.0 * std::log10(std::max(weighted_p2_sum, 1.0));
}

void AeroacousticEngine::propagate_to_receptors(
    const AcousticHemisphere180& hemisphere,
    const core::Vec3Enu& aircraft_enu,
    const core::Vec3Enu& vel_enu,
    std::span<const GroundReceptor> receptors,
    std::span<double> out_dba) const noexcept {
  const std::size_t n = std::min(receptors.size(), out_dba.size());
  for (std::size_t i = 0; i < n; ++i) {
    out_dba[i] = propagate_single_dba(hemisphere, aircraft_enu, vel_enu, receptors[i]);
  }
}

}  // namespace vertirta::wind_acoustics
