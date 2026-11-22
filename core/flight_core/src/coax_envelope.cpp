#include "vertirta/core/coax_envelope.hpp"

#include <cmath>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

namespace {

[[nodiscard]] inline int popcount8(uint8_t m) noexcept {
  int c = 0;
  for (int i = 0; i < 8; ++i) {
    if ((m & static_cast<uint8_t>(1u << i)) != 0u) {
      ++c;
    }
  }
  return c;
}

[[nodiscard]] inline bool motor_dead(uint8_t mask, int i) noexcept {
  return (mask & static_cast<uint8_t>(1u << i)) != 0u;
}

// Glauert hover-to-forward induced velocity v_{i,u} (4 Newton iterations from hover baseline)
[[nodiscard]] double solve_glauert_vi(double thrust_u, double v_horiz) noexcept {
  const double t_pos = (thrust_u > 1.0) ? thrust_u : 1.0;
  const double vh2 = t_pos / (2.0 * params::kRhoAir * params::kDiskAreaM2);
  double vi = std::sqrt(vh2);
  const double v2 = v_horiz * v_horiz;
  for (int iter = 0; iter < 5; ++iter) {
    const double denom = std::sqrt(v2 + vi * vi);
    const double f = vi * denom - vh2;
    const double df = denom + (vi * vi) / denom;
    vi -= f / df;
    if (vi < 0.5) {
      vi = 0.5;
    }
  }
  return vi;
}

}  // namespace

int CoaxEffectiveness::failed_same_arm_index(uint8_t fault_mask) noexcept {
  for (int arm = 0; arm < 4; ++arm) {
    if (motor_dead(fault_mask, arm) && motor_dead(fault_mask, arm + 4)) {
      return arm;
    }
  }
  return -1;
}

FlightRegime CoaxEffectiveness::classify_regime(uint8_t fault_mask) noexcept {
  const int n_dead = popcount8(fault_mask);
  if (n_dead == 0) {
    return FlightRegime::NOMINAL;
  }
  if (n_dead == 1) {
    return FlightRegime::SINGLE_LOSS;
  }
  int dead_arms = 0;
  for (int arm = 0; arm < 4; ++arm) {
    if (motor_dead(fault_mask, arm) && motor_dead(fault_mask, arm + 4)) {
      ++dead_arms;
    }
  }
  if (dead_arms == 1 && n_dead == 2) {
    return FlightRegime::SAME_ARM_ACTIVE_SPIN;
  }
  if (dead_arms == 0 && n_dead == 2) {
    return FlightRegime::CROSS_DUAL;
  }
  return FlightRegime::CRITICAL_DESCENT;
}

float CoaxEffectiveness::omega_lim_from_vterm(float v_term_v) noexcept {
  const double vt = clamp_d(static_cast<double>(v_term_v), 0.0, 600.0);
  const double a = (params::kReffHot * params::kKQ0) / params::kKeBus;
  const double b = params::kKeBus;
  const double disc = b * b + 4.0 * a * vt;
  const double w_lim = (-b + std::sqrt(disc)) / (2.0 * a);
  return static_cast<float>(clamp_d(w_lim, 0.0, params::kOmegaMax));
}

float CoaxEffectiveness::two_arm_tw_at_vterm(float v_term_v) noexcept {
  const double w = static_cast<double>(omega_lim_from_vterm(v_term_v));
  // 2 surviving orthogonal coaxial arms: 2 * (1.0 + eta_coax) * k_T0 * w^2
  const double t_two_arm = 2.0 * (1.0 + params::kEtaCoax0) * params::kKT0 * w * w;
  return static_cast<float>(t_two_arm / params::kWeightN);
}

double CoaxEffectiveness::coax_overlap_factor(float v_horiz_mps, double omega_upper) noexcept {
  const double vh = clamp_d(static_cast<double>(v_horiz_mps), 0.0, 40.0);
  if (vh < 0.1) {
    return 1.0;
  }
  const double w_u = clamp_d(omega_upper, 40.0, params::kOmegaMax);
  const double vi = solve_glauert_vi(params::kKT0 * w_u * w_u, vh);
  const double chi = std::atan2(vh, (vi > 0.5) ? vi : 0.5);
  const double dx = params::kCoaxSpacingM * std::tan(chi);
  const double xi = clamp_d(dx / (2.0 * params::kRotorRadiusM), 0.0, 1.0);
  return (2.0 / kPi) * (std::acos(xi) - xi * std::sqrt(clamp_d(1.0 - xi * xi, 0.0, 1.0)));
}

void CoaxEffectiveness::effective_rotor_constants(
    uint8_t fault_mask,
    float v_horiz_mps,
    double k_t_out[8],
    double k_q_out[8]) noexcept {
  const double f_ov = coax_overlap_factor(v_horiz_mps, params::kOmegaHover);
  for (int i = 0; i < 4; ++i) {
    k_t_out[i] = params::kKT0;
    k_q_out[i] = params::kKQ0;
    const bool upper_dead = motor_dead(fault_mask, i);
    const double eta_t = upper_dead ? 1.0 : (1.0 - (1.0 - params::kEtaCoax0) * f_ov);
    const double eta_q = upper_dead ? 1.0 : (1.0 - (1.0 - params::kSwirlRec0) * f_ov);
    k_t_out[i + 4] = eta_t * params::kKT0;
    k_q_out[i + 4] = eta_q * params::kKQ0;
  }
}

void CoaxEffectiveness::build_B(
    uint8_t fault_mask,
    float v_horiz_mps,
    double B_out[4][8],
    float u_max_n[8],
    float v_term_v) noexcept {
  double k_t[8]{};
  double k_q[8]{};
  effective_rotor_constants(fault_mask, v_horiz_mps, k_t, k_q);
  const double w_cap = static_cast<double>(omega_lim_from_vterm(v_term_v));
  const double w_cap2 = w_cap * w_cap;

  for (int i = 0; i < 8; ++i) {
    if (motor_dead(fault_mask, i)) {
      B_out[0][i] = 0.0;
      B_out[1][i] = 0.0;
      B_out[2][i] = 0.0;
      B_out[3][i] = 0.0;
      u_max_n[i] = 0.0f;
    } else {
      const double c_m = k_q[i] / k_t[i];
      B_out[0][i] = -params::kRotorPosY[i];
      B_out[1][i] = +params::kRotorPosX[i];
      B_out[2][i] = static_cast<double>(params::kRotorSpinSign[i]) * c_m;
      B_out[3][i] = 1.0;
      u_max_n[i] = static_cast<float>(k_t[i] * w_cap2);
    }
  }
}

float CoaxEffectiveness::compute_electrical_power_w(
    const float thrust_n[8],
    uint8_t fault_mask,
    float v_horiz_mps) noexcept {
  const double vh = clamp_d(static_cast<double>(v_horiz_mps), 0.0, 40.0);
  double k_t[8]{};
  double k_q[8]{};
  effective_rotor_constants(fault_mask, v_horiz_mps, k_t, k_q);
  const double f_ov = coax_overlap_factor(v_horiz_mps, params::kOmegaHover);
  const double kappa_int = 1.0 + (params::kKappaIntHover - 1.0) * f_ov;

  double p_mech = 0.0;
  for (int arm = 0; arm < 4; ++arm) {
    const bool u_alive = !motor_dead(fault_mask, arm);
    const bool l_alive = !motor_dead(fault_mask, arm + 4);
    const int n_arm = (u_alive ? 1 : 0) + (l_alive ? 1 : 0);
    if (n_arm == 0) {
      continue;
    }
    const double tu = u_alive ? clamp_d(static_cast<double>(thrust_n[arm]), 0.0, 1200.0) : 0.0;
    const double tl = l_alive ? clamp_d(static_cast<double>(thrust_n[arm + 4]), 0.0, 1200.0) : 0.0;
    const double t_arm = tu + tl;
    const double a_eff = (n_arm == 2)
                             ? (2.0 / (kappa_int * kappa_int)) * params::kDiskAreaM2
                             : params::kDiskAreaM2;
    const double vh2_arm = t_arm / (2.0 * params::kRhoAir * a_eff);
    double vi = std::sqrt((vh2_arm > 0.01) ? vh2_arm : 0.01);
    for (int it = 0; it < 4; ++it) {
      const double den = std::sqrt(vh * vh + vi * vi);
      vi -= (vi * den - vh2_arm) / (den + (vi * vi) / den);
      if (vi < 0.1) {
        vi = 0.1;
      }
    }
    const double p_ind = params::kKappaIso * t_arm * vi;

    double p_prof = 0.0;
    if (u_alive) {
      const double wu = std::sqrt(tu / k_t[arm]);
      const double v_tip = wu * params::kRotorRadiusM;
      const double mu = (v_tip > 1.0) ? (vh / v_tip) : 0.0;
      p_prof += 0.125 * params::kRhoAir * params::kDiskAreaM2 * params::kSolidity * params::kCd0 *
                v_tip * v_tip * v_tip * (1.0 + 4.65 * mu * mu);
    }
    if (l_alive) {
      const double wl = std::sqrt(tl / k_t[arm + 4]);
      const double v_tip = wl * params::kRotorRadiusM;
      const double mu = (v_tip > 1.0) ? (vh / v_tip) : 0.0;
      p_prof += 0.125 * params::kRhoAir * params::kDiskAreaM2 * params::kSolidity * params::kCd0 *
                v_tip * v_tip * v_tip * (1.0 + 4.65 * mu * mu);
    }
    p_mech += p_ind + p_prof;
  }

  const double p_para = 0.5 * params::kRhoAir * params::kFlatPlateAreaM2 * vh * vh * vh;
  return static_cast<float>((p_mech + p_para) / params::kDriveEff + params::kAvionicsW);
}

CapabilityEnvelope CoaxEffectiveness::compute_envelope(
    uint8_t fault_mask,
    uint8_t tmr_state,
    uint8_t imu_mask,
    float v_term_v,
    float i_pack_a,
    float soc,
    float spin_rate_ref,
    float spin_rate_meas,
    float chi2_residual,
    uint16_t fdi_latency_ms) noexcept {
  CapabilityEnvelope env{};
  env.fault_mask = fault_mask;
  env.tmr_state = tmr_state;
  env.imu_mask = imu_mask;
  env.regime = classify_regime(fault_mask);
  env.v_term_v = v_term_v;
  env.i_pack_a = i_pack_a;
  env.soc = soc;
  env.spin_rate_ref = spin_rate_ref;
  env.spin_rate_meas = spin_rate_meas;
  env.chi2_residual = chi2_residual;
  env.fdi_latency_ms = fdi_latency_ms;

  const double w_cap = static_cast<double>(omega_lim_from_vterm(v_term_v));
  const double v_scale = (w_cap * w_cap) / (params::kOmegaMax * params::kOmegaMax);
  const bool low_vbat = (v_term_v < 365.0f) || (soc < 0.15f);

  switch (env.regime) {
    case FlightRegime::NOMINAL:
      env.t_max_n = static_cast<float>(5886.0 * v_scale);
      env.tilt_max_rad = low_vbat ? 0.384f : 0.6109f;  // 22 deg low-Vbat, 35 deg nom
      env.a_xy_max = low_vbat ? 3.96f : 6.87f;
      env.vz_sink_max = 3.0f;
      env.p_hover_deg_w = 36230.0f;
      break;
    case FlightRegime::SINGLE_LOSS:
      env.t_max_n = static_cast<float>(4470.2 * v_scale);
      env.tilt_max_rad = 0.4887f;  // 28 deg
      env.a_xy_max = 5.21f;
      env.vz_sink_max = 2.8f;
      env.p_hover_deg_w = 37770.0f;
      break;
    case FlightRegime::CROSS_DUAL:
      env.t_max_n = static_cast<float>(3181.6 * v_scale);
      env.tilt_max_rad = 0.3840f;  // 22 deg
      env.a_xy_max = 3.96f;
      env.vz_sink_max = 2.5f;
      env.p_hover_deg_w = 39320.0f;
      break;
    case FlightRegime::SAME_ARM_ACTIVE_SPIN:
    case FlightRegime::CRITICAL_DESCENT:
      env.t_max_n = static_cast<float>(2943.0 * v_scale);
      env.tilt_max_rad = 0.18f;
      env.a_xy_max = 1.50f;
      env.vz_sink_max = 2.5f;
      env.p_hover_deg_w = 51050.0f;
      break;
  }
  return env;
}

TheveninBattery1RC::TheveninBattery1RC(float initial_soc) noexcept {
  reset(initial_soc);
}

void TheveninBattery1RC::reset(float initial_soc) noexcept {
  const float s = clamp_f(initial_soc, 0.0f, 1.0f);
  const float voc = voc_from_soc(s);
  state_ = BatteryState{voc, 0.0f, voc, 0.0f, s, false};
}

float TheveninBattery1RC::voc_from_soc(float soc) noexcept {
  // 112S NMC811 (INR21700-P45B class) piecewise OCV matching DR-1 Table 4.2:
  // SoC 0.00 -> 336.0 V, 0.10 -> 369.6 V, 0.20 -> 384.2 V, 0.50 -> 409.9 V, 1.00 -> 470.4 V
  static constexpr float kSocPts[5] = {0.00f, 0.10f, 0.20f, 0.50f, 1.00f};
  static constexpr float kVocPts[5] = {336.0f, 369.6f, 384.2f, 409.9f, 470.4f};
  const float s = clamp_f(soc, 0.0f, 1.0f);
  for (int i = 0; i < 4; ++i) {
    if (s <= kSocPts[i + 1]) {
      const float t = (s - kSocPts[i]) / (kSocPts[i + 1] - kSocPts[i]);
      return kVocPts[i] + t * (kVocPts[i + 1] - kVocPts[i]);
    }
  }
  return kVocPts[4];
}

BatteryState TheveninBattery1RC::step(float dt_s, float p_elec_w) noexcept {
  const double dt = clamp_d(static_cast<double>(dt_s), 0.0, 1.0);
  const double p = clamp_d(static_cast<double>(p_elec_w), 0.0, 300000.0);
  const double voc = static_cast<double>(voc_from_soc(state_.soc));
  const double v_eff = voc - static_cast<double>(state_.v_rc_v);
  const double disc = v_eff * v_eff - 4.0 * params::kBattR0 * p;

  double i_pack = 0.0;
  bool collapse = false;
  if (disc <= 0.0) {
    collapse = true;
    i_pack = v_eff / (2.0 * params::kBattR0);
  } else {
    i_pack = (v_eff - std::sqrt(disc)) / (2.0 * params::kBattR0);
  }

  const double v_term = v_eff - i_pack * params::kBattR0;
  const double exp_k = std::exp(-dt / params::kBattTau1);
  const double v_rc_next =
      exp_k * static_cast<double>(state_.v_rc_v) + params::kBattR1 * (1.0 - exp_k) * i_pack;
  const double soc_next = clamp_d(
      static_cast<double>(state_.soc) - (dt / (3600.0 * params::kBattCapAh)) * i_pack,
      0.0,
      1.0);

  state_.v_oc_v = static_cast<float>(voc);
  state_.v_term_v = static_cast<float>(v_term);
  state_.i_pack_a = static_cast<float>(i_pack);
  state_.v_rc_v = static_cast<float>(v_rc_next);
  state_.soc = static_cast<float>(soc_next);
  state_.voltage_collapse = collapse;
  return state_;
}

}  // namespace vertirta::core
