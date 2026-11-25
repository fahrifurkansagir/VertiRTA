// SPDX-License-Identifier: Apache-2.0
#include "vertirta/benchmarks/mc_1000_runner.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vertirta/airframes/lift_cruise_dynamics.hpp"
#include "vertirta/airframes/transition_indi_wls.hpp"
#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/math_utils.hpp"
#include "vertirta/forensics/vrta_events_sidecar.hpp"
#include "vertirta/gnss_denied/basalt_vio_adapter.hpp"
#include "vertirta/gnss_denied/ctr_inflation_fsm.hpp"
#include "vertirta/gnss_denied/gnss_spoof_guard.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/dryden_wind.hpp"
#include "vertirta/wind_acoustics/istanbul_receptors.hpp"

namespace vertirta::benchmarks {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kDt400Hz = 0.0025;
constexpr double kDt100Hz = 0.01;

// Safe recovery engineering thresholds (EASA SC-VTOL / CS-27 emergency touchdown & envelope limits)
constexpr double kMaxSafeTouchdownVzMps = 3.5;
constexpr double kMaxSafeTiltDeg = 60.0;
constexpr double kMinSafePackVoltageV = 300.0;

// Cevizli-Dragos Coastal Medical Helipad keepout cylinder outside LTBX 9,260 m CTR
constexpr double kCoastalHospitalLatDeg = 40.90800;
constexpr double kCoastalHospitalLonDeg = 29.15800;
constexpr double kCoastalHospitalAltAmslM = 32.0;

[[nodiscard]] double safe_finite(double v, double fallback = 0.0) noexcept {
  return std::isfinite(v) ? v : fallback;
}

[[nodiscard]] std::string utc_now_iso8601() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
#if defined(_WIN32)
  gmtime_s(&utc_tm, &tt);
#else
  gmtime_r(&tt, &utc_tm);
#endif
  char buf[32]{};
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
  return std::string(buf);
}

[[nodiscard]] std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8U);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out.push_back(c);
        break;
    }
  }
  return out;
}

// ----------------------------------------------------------------------------
// Exact Regularized Incomplete Beta & Binomial Confidence Intervals (95%)
// ----------------------------------------------------------------------------
[[nodiscard]] double beta_continued_fraction(double a, double b, double x) noexcept {
  constexpr int kMaxIter = 200;
  constexpr double kEps = 3.0e-13;
  constexpr double kFpMin = 1.0e-300;

  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < kFpMin) {
    d = kFpMin;
  }
  d = 1.0 / d;
  double h = d;

  for (int m = 1; m <= kMaxIter; ++m) {
    const double m_d = static_cast<double>(m);
    const double m2 = 2.0 * m_d;
    double aa = m_d * (b - m_d) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kFpMin) {
      d = kFpMin;
    }
    c = 1.0 + aa / c;
    if (std::abs(c) < kFpMin) {
      c = kFpMin;
    }
    d = 1.0 / d;
    h *= d * c;

    aa = -(a + m_d) * (qab + m_d) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kFpMin) {
      d = kFpMin;
    }
    c = 1.0 + aa / c;
    if (std::abs(c) < kFpMin) {
      c = kFpMin;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < kEps) {
      break;
    }
  }
  return h;
}

[[nodiscard]] double reg_incomplete_beta(double a, double b, double x) noexcept {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x >= 1.0) {
    return 1.0;
  }
  const double log_bt =
      std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log(1.0 - x);
  const double bt = std::exp(log_bt);
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return bt * beta_continued_fraction(a, b, x) / a;
  }
  return 1.0 - bt * beta_continued_fraction(b, a, 1.0 - x) / b;
}

[[nodiscard]] double inv_reg_beta(double a, double b, double target) noexcept {
  double lo = 0.0;
  double hi = 1.0;
  for (int it = 0; it < 90; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (reg_incomplete_beta(a, b, mid) < target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// ----------------------------------------------------------------------------
// Quaternion & Geometry Helpers
// ----------------------------------------------------------------------------
[[nodiscard]] core::QuatPx4FrdNed normalize_quat(const core::QuatPx4FrdNed& q) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-12 || !std::isfinite(n)) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  const double inv = 1.0 / n;
  return {q.w * inv, q.x * inv, q.y * inv, q.z * inv};
}

[[nodiscard]] core::Vec3Ned frd_to_ned(const core::QuatPx4FrdNed& q, const core::Vec3Frd& v) noexcept {
  const double w = q.w, x = q.x, y = q.y, z = q.z;
  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - w * z);
  const double r02 = 2.0 * (x * z + w * y);
  const double r10 = 2.0 * (x * y + w * z);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - w * x);
  const double r20 = 2.0 * (x * z - w * y);
  const double r21 = 2.0 * (y * z + w * x);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);
  return {
      r00 * v.x + r01 * v.y + r02 * v.z,
      r10 * v.x + r11 * v.y + r12 * v.z,
      r20 * v.x + r21 * v.y + r22 * v.z,
  };
}

[[nodiscard]] core::Vec3Frd ned_to_frd(const core::QuatPx4FrdNed& q, const core::Vec3Ned& v) noexcept {
  const double w = q.w, x = q.x, y = q.y, z = q.z;
  const double r00 = 1.0 - 2.0 * (y * y + z * z);
  const double r01 = 2.0 * (x * y - w * z);
  const double r02 = 2.0 * (x * z + w * y);
  const double r10 = 2.0 * (x * y + w * z);
  const double r11 = 1.0 - 2.0 * (x * x + z * z);
  const double r12 = 2.0 * (y * z - w * x);
  const double r20 = 2.0 * (x * z - w * y);
  const double r21 = 2.0 * (y * z + w * x);
  const double r22 = 1.0 - 2.0 * (x * x + y * y);
  return {
      r00 * v.x + r10 * v.y + r20 * v.z,
      r01 * v.x + r11 * v.y + r21 * v.z,
      r02 * v.x + r12 * v.y + r22 * v.z,
  };
}

void quat_to_euler_zyx(const core::QuatPx4FrdNed& q_in, double& roll, double& pitch, double& yaw) noexcept {
  const auto q = normalize_quat(q_in);
  roll = std::atan2(2.0 * (q.w * q.x + q.y * q.z), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));
  pitch = std::asin(std::clamp(2.0 * (q.w * q.y - q.z * q.x), -1.0, 1.0));
  yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

[[nodiscard]] double compute_tilt_deg(const core::QuatPx4FrdNed& q_in) noexcept {
  const auto q = normalize_quat(q_in);
  const double cos_tilt = std::clamp(1.0 - 2.0 * (q.x * q.x + q.y * q.y), -1.0, 1.0);
  return std::acos(cos_tilt) * kRadToDeg;
}

[[nodiscard]] rta::HazardPoint build_coastal_hospital_hazard() noexcept {
  const double h_hae = core::amsl_to_hae(
      kCoastalHospitalAltAmslM, kCoastalHospitalLatDeg, kCoastalHospitalLonDeg);
  const core::Vec3Enu enu = core::geodetic_to_local_enu(
      {kCoastalHospitalLatDeg, kCoastalHospitalLonDeg, h_hae}, core::kKartalDatum);
  rta::HazardPoint hz{};
  hz.pos_ned = core::enu_to_ned(enu);
  hz.vel_ned = {0.0, 0.0, 0.0};
  hz.accel_ned = {0.0, 0.0, 0.0};
  hz.radius_h_m = wind_acoustics::kKartalHospitalHelipadRadiusHM;  // 160.0 m
  hz.radius_z_m = wind_acoustics::kKartalHospitalHelipadRadiusZM;  // 125.0 m
  hz.gamma = 1.4;
  hz.kind = 3U;
  return hz;
}

// Fast deterministic Box-Muller Gaussian from mt19937_64
class FastGaussian {
 public:
  explicit FastGaussian(uint64_t seed) noexcept : rng_(seed) {}

  [[nodiscard]] double next_normal() noexcept {
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }
    const double u1 = std::max(1e-12, static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0));
    const double u2 = static_cast<double>(rng_() >> 11) * (1.0 / 9007199254740992.0);
    const double mag = std::sqrt(-2.0 * std::log(u1));
    spare_ = mag * std::sin(kTwoPi * u2);
    has_spare_ = true;
    return mag * std::cos(kTwoPi * u2);
  }

 private:
  std::mt19937_64 rng_;
  bool has_spare_{false};
  double spare_{0.0};
};

// ----------------------------------------------------------------------------
// Perturbed 6-DOF Coaxial X8-240 Plant (Exact frozen core physics + mass/CG dispersion)
// ----------------------------------------------------------------------------
struct PerturbedX8Plant {
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  core::QuatPx4FrdNed q_frd_ned{1.0, 0.0, 0.0, 0.0};
  core::Vec3Frd omega_frd{};
  double rotor_omega[8]{};
  core::TheveninBattery1RC battery{0.80f};
  core::BatteryState batt_state{};
  bool on_ground{false};
  double touchdown_vz_mps{0.0};

  double mass_kg{core::params::kMassKg};
  double jxx{core::params::kJxx};
  double jyy{core::params::kJyy};
  double jzz{core::params::kJzz};
  double cg_x_m{0.0};
  double cg_y_m{0.0};

  void reset(
      const core::Vec3Ned& pos0,
      const core::Vec3Ned& vel0,
      float init_soc,
      double mass,
      double cgx,
      double cgy) noexcept {
    mass_kg = mass;
    const double scale = mass / core::params::kMassKg;
    jxx = core::params::kJxx * scale;
    jyy = core::params::kJyy * scale;
    jzz = core::params::kJzz * scale;
    cg_x_m = cgx;
    cg_y_m = cgy;

    battery.reset(init_soc);
    batt_state = battery.state();
    pos_ned = pos0;
    vel_ned = vel0;
    q_frd_ned = {1.0, 0.0, 0.0, 0.0};
    omega_frd = {0.0, 0.0, 0.0};
    const double w0 = core::params::kOmegaHover * std::sqrt(std::max(0.5, scale));
    for (int i = 0; i < 8; ++i) {
      rotor_omega[i] = std::min(w0, core::params::kOmegaMax);
    }
    on_ground = (pos_ned.z >= 0.0);
    touchdown_vz_mps = 0.0;
  }

  struct RbState {
    core::Vec3Ned pos;
    core::Vec3Ned vel;
    core::QuatPx4FrdNed q;
    core::Vec3Frd omega;
  };
  struct RbDeriv {
    core::Vec3Ned dpos;
    core::Vec3Ned dvel;
    core::QuatPx4FrdNed dq;
    core::Vec3Frd domega;
  };

  [[nodiscard]] RbDeriv eval_deriv(
      const RbState& s,
      double total_thrust_n,
      const core::Vec3Frd& rotor_torque_frd,
      const core::Vec3Ned& wind_ned) const noexcept {
    const auto q = normalize_quat(s.q);
    const core::Vec3Ned v_air_ned = s.vel - wind_ned;
    const double v_air_mag = v_air_ned.norm();

    const double drag_coeff = 0.5 * core::params::kRhoAir * core::params::kFlatPlateAreaM2 * v_air_mag;
    const core::Vec3Ned f_drag_ned = v_air_ned * (-drag_coeff);

    const core::Vec3Frd v_air_frd = ned_to_frd(q, v_air_ned);
    constexpr double kGustMomentGain = 3.0;  // 45 N*m @ 15 m/s gust (DR-1 Sec 3.5)
    const core::Vec3Frd tau_gust_frd{
        -kGustMomentGain * v_air_frd.y,
        +kGustMomentGain * v_air_frd.x,
        -2.0 * s.omega.z,
    };

    const core::Vec3Ned f_thrust_ned = frd_to_ned(q, {0.0, 0.0, -total_thrust_n});
    const core::Vec3Ned acc_ned{
        (f_thrust_ned.x + f_drag_ned.x) / mass_kg,
        (f_thrust_ned.y + f_drag_ned.y) / mass_kg,
        (f_thrust_ned.z + f_drag_ned.z) / mass_kg + core::params::kGravity,
    };

    const core::Vec3Frd tau_tot = rotor_torque_frd + tau_gust_frd;
    const double jx = jxx * s.omega.x;
    const double jy = jyy * s.omega.y;
    const double jz = jzz * s.omega.z;
    const core::Vec3Frd domega{
        (tau_tot.x - (s.omega.y * jz - s.omega.z * jy)) / jxx,
        (tau_tot.y - (s.omega.z * jx - s.omega.x * jz)) / jyy,
        (tau_tot.z - (s.omega.x * jy - s.omega.y * jx)) / jzz,
    };

    const double wx = s.omega.x, wy = s.omega.y, wz = s.omega.z;
    const core::QuatPx4FrdNed dq{
        0.5 * (-q.x * wx - q.y * wy - q.z * wz),
        0.5 * (+q.w * wx + q.y * wz - q.z * wy),
        0.5 * (+q.w * wy - q.x * wz + q.z * wx),
        0.5 * (+q.w * wz + q.x * wy - q.y * wx),
    };
    return {s.vel, acc_ned, dq, domega};
  }

  static RbState add_scaled(const RbState& s, const RbDeriv& k, double dt) noexcept {
    return {
        s.pos + k.dpos * dt,
        s.vel + k.dvel * dt,
        {s.q.w + k.dq.w * dt, s.q.x + k.dq.x * dt, s.q.y + k.dq.y * dt, s.q.z + k.dq.z * dt},
        s.omega + k.domega * dt,
    };
  }

  void step_400hz(
      const float omega_cmd_rad_s[8],
      uint8_t fault_mask,
      const core::Vec3Ned& wind_ned) noexcept {
    const float v_horiz = static_cast<float>(
        std::hypot(vel_ned.x - wind_ned.x, vel_ned.y - wind_ned.y));
    double k_t[8]{};
    double k_q[8]{};
    core::CoaxEffectiveness::effective_rotor_constants(fault_mask, v_horiz, k_t, k_q);
    const double w_cap = static_cast<double>(
        core::CoaxEffectiveness::omega_lim_from_vterm(batt_state.v_term_v));

    double total_thrust = 0.0;
    core::Vec3Frd rotor_torque_frd{};
    float thrust_n[8]{};
    for (int i = 0; i < 8; ++i) {
      const bool dead = (fault_mask & static_cast<uint8_t>(1u << i)) != 0u;
      const double w_target = dead ? 0.0 : core::clamp_d(static_cast<double>(omega_cmd_rad_s[i]), 0.0, w_cap);
      const double tau = dead ? core::params::kTauCoast
                              : ((w_target >= rotor_omega[i]) ? core::params::kTauUp : core::params::kTauDown);
      const double dw = dead ? (-k_q[i] * rotor_omega[i] * rotor_omega[i] / core::params::kJr)
                             : core::clamp_d((w_target - rotor_omega[i]) / tau,
                                             -core::params::kAlphaMax, core::params::kAlphaMax);
      rotor_omega[i] = core::clamp_d(rotor_omega[i] + dw * kDt400Hz, 0.0, core::params::kOmegaMax);

      const double ti = k_t[i] * rotor_omega[i] * rotor_omega[i];
      const double qi = dead ? 0.0 : (k_q[i] * rotor_omega[i] * rotor_omega[i] + core::params::kJr * dw);
      thrust_n[i] = static_cast<float>(ti);
      total_thrust += ti;

      // Exact perturbed moment arm relative to shifted CG (cg_x_m, cg_y_m)
      const double rx = core::params::kRotorPosX[i] - cg_x_m;
      const double ry = core::params::kRotorPosY[i] - cg_y_m;
      rotor_torque_frd.x += -ry * ti;
      rotor_torque_frd.y += +rx * ti;
      rotor_torque_frd.z += static_cast<double>(core::params::kRotorSpinSign[i]) * qi;
    }

    const float p_elec = core::CoaxEffectiveness::compute_electrical_power_w(thrust_n, fault_mask, v_horiz);
    batt_state = battery.step(static_cast<float>(kDt400Hz), p_elec);

    const RbState s0{pos_ned, vel_ned, q_frd_ned, omega_frd};
    const RbDeriv k1 = eval_deriv(s0, total_thrust, rotor_torque_frd, wind_ned);
    const RbDeriv k2 = eval_deriv(add_scaled(s0, k1, 0.5 * kDt400Hz), total_thrust, rotor_torque_frd, wind_ned);
    const RbDeriv k3 = eval_deriv(add_scaled(s0, k2, 0.5 * kDt400Hz), total_thrust, rotor_torque_frd, wind_ned);
    const RbDeriv k4 = eval_deriv(add_scaled(s0, k3, kDt400Hz), total_thrust, rotor_torque_frd, wind_ned);

    const double c6 = kDt400Hz / 6.0;
    pos_ned = s0.pos + (k1.dpos + k2.dpos * 2.0 + k3.dpos * 2.0 + k4.dpos) * c6;
    vel_ned = s0.vel + (k1.dvel + k2.dvel * 2.0 + k3.dvel * 2.0 + k4.dvel) * c6;
    omega_frd = s0.omega + (k1.domega + k2.domega * 2.0 + k3.domega * 2.0 + k4.domega) * c6;
    q_frd_ned = normalize_quat({
        s0.q.w + (k1.dq.w + 2.0 * k2.dq.w + 2.0 * k3.dq.w + k4.dq.w) * c6,
        s0.q.x + (k1.dq.x + 2.0 * k2.dq.x + 2.0 * k3.dq.x + k4.dq.x) * c6,
        s0.q.y + (k1.dq.y + 2.0 * k2.dq.y + 2.0 * k3.dq.y + k4.dq.y) * c6,
        s0.q.z + (k1.dq.z + 2.0 * k2.dq.z + 2.0 * k3.dq.z + k4.dq.z) * c6,
    });

    if (pos_ned.z >= 0.0) {
      if (!on_ground) {
        touchdown_vz_mps = vel_ned.z;
      }
      on_ground = true;
      pos_ned.z = 0.0;
      if (vel_ned.z > 0.0) {
        vel_ned.z = 0.0;
      }
    } else {
      on_ground = false;
    }
  }
};

// ----------------------------------------------------------------------------
// Perturbed 6-DOF Lift+Cruise (8+1) Helper (Exact LiftCruiseDynamics + mass/CG/sag)
// ----------------------------------------------------------------------------
[[nodiscard]] double compute_lc_elec_power_w(
    const airframes::LiftCruiseDynamics& plant,
    const airframes::AirframeState6Dof& st,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask) noexcept {
  constexpr double kRho = airframes::LiftCruiseDynamics::kRhoAir;
  constexpr double kDriveEff = 0.88;
  constexpr double kAvionicsW = 450.0;
  constexpr double kLiftDiskArea = kPi * 0.55 * 0.55;
  constexpr double kPusherDiskArea = kPi * 0.675 * 0.675;
  constexpr double kKappaHover = 1.15;

  double p_shaft_w = 0.0;
  const uint8_t n_lift = plant.lift_rotor_count();
  for (uint8_t i = 0U; i < n_lift; ++i) {
    if ((fault_mask & static_cast<uint16_t>(1U << i)) != 0U) {
      continue;
    }
    const double t_n = std::max(0.0, st.u_act[i]);
    if (t_n > 1e-3) {
      const double p_ind = kKappaHover * t_n * std::sqrt(t_n / (2.0 * kRho * kLiftDiskArea));
      p_shaft_w += 1.12 * p_ind;
    }
  }
  const uint8_t p_idx = plant.pusher_index();
  if ((fault_mask & static_cast<uint16_t>(1U << p_idx)) == 0U) {
    const double t_p = std::max(0.0, st.u_act[p_idx]);
    if (t_p > 1e-3) {
      const double v_air = std::max(1.0, (st.vel_ned - wind_ned).norm());
      const double v_i = 0.5 * (-v_air + std::sqrt(v_air * v_air + 2.0 * t_p / (kRho * kPusherDiskArea)));
      p_shaft_w += t_p * (v_air + 1.15 * v_i);
    }
  }
  return p_shaft_w / kDriveEff + kAvionicsW;
}

[[nodiscard]] airframes::AirframeDerivatives eval_lc_perturbed(
    const airframes::LiftCruiseDynamics& plant,
    const airframes::AirframeState6Dof& st,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask,
    double mass_kg,
    double cg_x_m,
    double cg_y_m) noexcept {
  auto d = plant.eval_dynamics(st, wind_ned, fault_mask);
  const double inv_scale = airframes::LiftCruiseDynamics::kMassKg / std::max(50.0, mass_kg);

  // Exact mass perturbation on non-gravitational specific force: F_body / m_actual
  d.accel_body_frd = d.accel_body_frd * inv_scale;
  const core::Vec3Ned a_ned = frd_to_ned(normalize_quat(st.q_frd_ned), d.accel_body_frd);
  d.vel_dot_ned = {a_ned.x, a_ned.y, a_ned.z + airframes::LiftCruiseDynamics::kGravity};

  // Exact CG offset moment from lift & vertical aerodynamic force: -r_cg x F_body
  const double fz_body = d.accel_body_frd.z * mass_kg;  // Negative when thrusting upward
  const double dmx_cg = -cg_y_m * fz_body;              // +cg_y with -Fz gives +Mx
  const double dmy_cg = +cg_x_m * fz_body;              // +cg_x with -Fz gives -My

  constexpr double kDet = airframes::LiftCruiseDynamics::kIxx * airframes::LiftCruiseDynamics::kIzz -
                          airframes::LiftCruiseDynamics::kIxz * airframes::LiftCruiseDynamics::kIxz;
  const double dw_cg_x = (airframes::LiftCruiseDynamics::kIzz * dmx_cg) / kDet;
  const double dw_cg_y = dmy_cg / airframes::LiftCruiseDynamics::kIyy;
  const double dw_cg_z = (airframes::LiftCruiseDynamics::kIxz * dmx_cg) / kDet;

  d.omega_dot_frd.x = (d.omega_dot_frd.x + dw_cg_x) * inv_scale;
  d.omega_dot_frd.y = (d.omega_dot_frd.y + dw_cg_y) * inv_scale;
  d.omega_dot_frd.z = (d.omega_dot_frd.z + dw_cg_z) * inv_scale;
  return d;
}

void step_lc_plant_400hz(
    const airframes::LiftCruiseDynamics& plant,
    airframes::AirframeState6Dof& st,
    const std::array<double, airframes::kMaxActuators>& u_cmd,
    const core::Vec3Ned& wind_ned,
    uint16_t fault_mask,
    double v_term_v,
    double mass_kg,
    double cg_x_m,
    double cg_y_m) noexcept {
  airframes::EffectivenessMatrix eff{};
  plant.compute_effectiveness(st, wind_ned, fault_mask, kDt400Hz, eff);

  // Bus-voltage motor speed/thrust ceiling coupling (same physical winding headroom as X8)
  const double w_cap = static_cast<double>(core::CoaxEffectiveness::omega_lim_from_vterm(static_cast<float>(v_term_v)));
  const double v_scale = std::clamp((w_cap * w_cap) / (core::params::kOmegaMax * core::params::kOmegaMax), 0.35, 1.0);

  const uint8_t n_lift = plant.lift_rotor_count();
  for (uint8_t i = 0U; i < plant.actuator_count(); ++i) {
    const double u_hi = (i < n_lift) ? (eff.u_max[i] * v_scale) : eff.u_max[i];
    const double target = core::clamp_d(u_cmd[i], eff.u_min[i], u_hi);
    if (i < n_lift && ((fault_mask & static_cast<uint16_t>(1U << i)) != 0U)) {
      st.u_act[i] *= std::exp(-kDt400Hz / 0.424);
    } else if (i == plant.pusher_index() && ((fault_mask & static_cast<uint16_t>(1U << i)) != 0U)) {
      st.u_act[i] *= std::exp(-kDt400Hz / 0.35);
    } else {
      const double tau = std::max(0.005, eff.tau_act_s[i]);
      const double alpha_f = 1.0 - std::exp(-kDt400Hz / tau);
      st.u_act[i] += alpha_f * (target - st.u_act[i]);
    }
  }

  auto advance = [](const airframes::AirframeState6Dof& s,
                    const airframes::AirframeDerivatives& d,
                    double h) {
    airframes::AirframeState6Dof out = s;
    out.pos_ned = s.pos_ned + d.pos_dot_ned * h;
    out.vel_ned = s.vel_ned + d.vel_dot_ned * h;
    out.omega_frd = s.omega_frd + d.omega_dot_frd * h;
    const auto& q = s.q_frd_ned;
    const auto& w = s.omega_frd;
    const double qw = q.w + 0.5 * h * (-q.x * w.x - q.y * w.y - q.z * w.z);
    const double qx = q.x + 0.5 * h * (q.w * w.x + q.y * w.z - q.z * w.y);
    const double qy = q.y + 0.5 * h * (q.w * w.y - q.x * w.z + q.z * w.x);
    const double qz = q.z + 0.5 * h * (q.w * w.z + q.x * w.y - q.y * w.x);
    out.q_frd_ned = normalize_quat({qw, qx, qy, qz});
    return out;
  };

  const auto k1 = eval_lc_perturbed(plant, st, wind_ned, fault_mask, mass_kg, cg_x_m, cg_y_m);
  const auto mid = advance(st, k1, 0.5 * kDt400Hz);
  const auto k2 = eval_lc_perturbed(plant, mid, wind_ned, fault_mask, mass_kg, cg_x_m, cg_y_m);
  st = advance(st, k2, kDt400Hz);
}

[[nodiscard]] core::CapabilityEnvelope build_lc_envelope(
    const airframes::LiftCruiseDynamics& plant,
    const core::BatteryState& batt,
    uint16_t fault_mask,
    double p_elec_w) noexcept {
  core::CapabilityEnvelope env{};
  env.fault_mask = static_cast<uint8_t>(fault_mask & 0xFFU);
  env.tmr_state = 0U;
  env.imu_mask = 0x07U;

  uint8_t dead_lift = 0U;
  for (uint8_t i = 0U; i < plant.lift_rotor_count(); ++i) {
    if ((fault_mask & static_cast<uint16_t>(1U << i)) != 0U) {
      ++dead_lift;
    }
  }
  const double w_cap = static_cast<double>(core::CoaxEffectiveness::omega_lim_from_vterm(batt.v_term_v));
  const double v_scale = std::clamp((w_cap * w_cap) / (core::params::kOmegaMax * core::params::kOmegaMax), 0.35, 1.0);

  if (dead_lift == 0U) {
    env.regime = core::FlightRegime::NOMINAL;
    env.t_max_n = static_cast<float>(8.0 * plant.max_lift_rotor_thrust_n() * v_scale);
    env.a_xy_max = 4.2f;
    env.tilt_max_rad = 0.35f;
  } else if (dead_lift == 1U) {
    env.regime = core::FlightRegime::SINGLE_LOSS;
    env.t_max_n = static_cast<float>(5.46875 * plant.max_lift_rotor_thrust_n() * v_scale);
    env.a_xy_max = 3.2f;
    env.tilt_max_rad = 0.28f;
  } else {
    // Inner-boom dual lift-motor loss: balanced 6-rotor thrust ceiling ~ 2755.5 N * v_scale
    // Use CROSS_DUAL (not SAME_ARM_ACTIVE_SPIN) because 8+1 LC stays non-spinning hover-divert!
    env.regime = core::FlightRegime::CROSS_DUAL;
    env.t_max_n = static_cast<float>(2755.5 * v_scale);
    env.a_xy_max = 2.2f;
    env.tilt_max_rad = 0.22f;
  }
  env.vz_sink_max = 3.0f;
  env.v_term_v = batt.v_term_v;
  env.i_pack_a = batt.i_pack_a;
  env.soc = batt.soc;
  env.p_hover_deg_w = static_cast<float>(std::max(0.0, p_elec_w));
  return env;
}

// ----------------------------------------------------------------------------
// X8-240 Single-Flight Closed-Loop Runner
// ----------------------------------------------------------------------------
[[nodiscard]] FlightOutcome run_x8_flight(const FlightSampleConfig& cfg) noexcept {
  const auto t_wall0 = std::chrono::steady_clock::now();
  FlightOutcome out{};
  out.flight_id = cfg.flight_id;
  out.airframe = cfg.airframe;
  out.regime = cfg.regime;
  out.mass_kg = cfg.mass_kg;
  out.cg_offset_m = std::hypot(cfg.cg_x_m, cfg.cg_y_m);
  out.wind_speed_mps = cfg.wind_speed_mps;
  out.init_agl_m = cfg.init_agl_m;
  out.init_soc = cfg.init_soc;

  PerturbedX8Plant plant{};
  const core::Vec3Ned init_pos{cfg.init_north_m, cfg.init_east_m, -cfg.init_agl_m};
  const core::Vec3Ned init_vel{cfg.init_vx_mps, 0.0, 0.0};
  plant.reset(init_pos, init_vel, static_cast<float>(cfg.init_soc), cfg.mass_kg, cfg.cg_x_m, cfg.cg_y_m);

  core::IndiWlsController ctrl(/*allow_active_spin=*/true);
  ctrl.reset(core::params::kOmegaHover * std::sqrt(std::max(0.5, cfg.mass_kg / core::params::kMassKg)));

  rta::CbvfSafetyFilter cbvf(nullptr, nullptr);
  rta::DrydenTurbulenceFilter dryden(cfg.seed ^ 0xA5A55A5A12345678ULL);
  FastGaussian sensor_rng(cfg.seed ^ 0x517CC1B727220A95ULL);

  // Realistic GNSS/INS horizontal position bias (OU process, sigma ~ 0.65 m, tau = 12 s)
  double nav_bias_n = sensor_rng.next_normal() * 0.65;
  double nav_bias_e = sensor_rng.next_normal() * 0.65;
  double vio_drift_n = 0.0;
  double vio_drift_e = 0.0;
  const double vio_drift_rate_n = sensor_rng.next_normal() * 0.008;
  const double vio_drift_rate_e = sensor_rng.next_normal() * 0.008;

  gnss_denied::BasaltVioAdapter vio{};
  vio.reset(plant.pos_ned, plant.vel_ned);
  gnss_denied::GnssSpoofGuard spoof_guard{};
  gnss_denied::CtrInflationFsm ctr_fsm{};
  ctr_fsm.reset(/*sigma_init_m=*/2.0);
  const auto hospital_haz = build_coastal_hospital_hazard();

  // Commanded SLZ aimpoint (west of LTFJ CTR and clear of Coastal Hospital)
  double slz_n = cfg.init_north_m + cfg.slz_offset_n_m;
  double slz_e = (cfg.slz_offset_n_m > 1.0) ? std::min(cfg.init_east_m - 15.0, -140.0) : cfg.init_east_m;
  bool emergency_slz_latched = false;

  const core::Vec3Ned mean_wind_ned{
      cfg.wind_speed_mps * std::cos(cfg.wind_dir_rad),
      cfg.wind_speed_mps * std::sin(cfg.wind_dir_rad),
      0.0,
  };

  uint8_t active_fault_mask = 0x00U;
  bool spoof_detected = false;
  double spoof_ttd_s = -1.0;
  bool spoof_rollback_ok = false;
  core::Vec3Ned nav_pos_offset{0.0, 0.0, 0.0};
  core::Vec3Ned prev_vel_ned = plant.vel_ned;
  gnss_denied::VioStateEstimate vio_est = vio.state();

  double int_ax = 0.0;
  double int_ay = 0.0;
  double int_az = 0.0;
  bool descending = false;
  const uint32_t max_ticks = static_cast<uint32_t>(std::ceil(cfg.max_flight_s / kDt100Hz));

  for (uint32_t tick = 0U; tick < max_ticks; ++tick) {
    const double t_s = static_cast<double>(tick) * kDt100Hz;
    const uint64_t t_ns = static_cast<uint64_t>(tick) * 10'000'000ULL;

    // Inject regime-specific faults
    if (cfg.regime == FaultRegimeKind::kSingleRotorLossCsfl ||
        cfg.regime == FaultRegimeKind::kCompoundLowSocWindFault) {
      if (t_s >= cfg.fault_time_s) {
        active_fault_mask = static_cast<uint8_t>(1U << (cfg.fault_index % 8U));
      }
    } else if (cfg.regime == FaultRegimeKind::kSameArmDualKillCel) {
      if (t_s >= cfg.fault_time_s) {
        const uint8_t arm = static_cast<uint8_t>(cfg.fault_index % 4U);
        active_fault_mask = static_cast<uint8_t>((1U << arm) | (1U << (arm + 4U)));
        if (!emergency_slz_latched) {
          emergency_slz_latched = true;
          // Immediate CEL emergency SLZ directly along fault-capture ground projection
          slz_n = plant.pos_ned.x + plant.vel_ned.x * 1.5;
          slz_e = plant.pos_ned.y + plant.vel_ned.y * 1.5;
          descending = true;
        }
      }
    }

    // Real MIL-F-8785C Dryden turbulence + steady wind
    const double agl_true = std::max(1.0, -plant.pos_ned.z);
    const double v_air_est = std::max(2.0, (plant.vel_ned - mean_wind_ned).norm());
    const double w20_eff = std::max(1.5, cfg.wind_speed_mps * cfg.gust_sigma_scale);
    const core::Vec3Ned gust = dryden.step_seeded(kDt100Hz, agl_true, v_air_est, w20_eff);
    const double gust_gain = 0.40 * cfg.gust_sigma_scale;
    const core::Vec3Ned wind_ned{
        mean_wind_ned.x + std::clamp(gust.x * gust_gain, -6.0, 6.0),
        mean_wind_ned.y + std::clamp(gust.y * gust_gain, -6.0, 6.0),
        std::clamp(gust.z * gust_gain, -2.0, 2.0),
    };
    out.max_wind_mps = std::max(out.max_wind_mps, wind_ned.norm());

    // Evolve realistic correlated GNSS/INS nav bias (or post-rollback VIO drift)
    constexpr double kOuPhi = 0.999167;  // exp(-0.01 / 12.0)
    constexpr double kOuSig = 0.0265;    // 0.65 * sqrt(1 - phi^2)
    nav_bias_n = kOuPhi * nav_bias_n + kOuSig * sensor_rng.next_normal();
    nav_bias_e = kOuPhi * nav_bias_e + kOuSig * sensor_rng.next_normal();
    if (spoof_detected) {
      const double ds = std::hypot(plant.vel_ned.x, plant.vel_ned.y) * kDt100Hz;
      vio_drift_n += vio_drift_rate_n * ds + 0.012 * sensor_rng.next_normal();
      vio_drift_e += vio_drift_rate_e * ds + 0.012 * sensor_rng.next_normal();
    }

    // Honest 100 Hz VIO + 10 Hz GNSS Spoof Guard for GNSS_SLOW_RAMP_SPOOF
    if (cfg.regime == FaultRegimeKind::kGnssSlowRampSpoof) {
      const core::Vec3Ned imu_accel_ned =
          (tick > 0U) ? ((plant.vel_ned - prev_vel_ned) * (1.0 / kDt100Hz)) : core::Vec3Ned{0.0, 0.0, 0.0};
      prev_vel_ned = plant.vel_ned;

      gnss_denied::VioSensorFrame vf{};
      vf.stamp_ns = t_ns;
      vf.dt_s = kDt100Hz;
      vf.imu_accel_ned = imu_accel_ned;
      vf.has_stereo_frame = (tick % 10U == 0U);
      vf.stereo_delta_vel_ned = plant.vel_ned;
      vf.radalt_agl_m = agl_true;
      vf.radalt_valid = true;
      vf.pidnet_water_fraction = 0.08;
      (void)vio.step(vf, vio_est);

      if (t_s >= cfg.fault_time_s && !spoof_detected) {
        // Pre-detection GNSS spoof fools nav estimate westward so aircraft drifts east toward LTFJ!
        const double dt_sp = t_s - cfg.fault_time_s;
        nav_pos_offset = {0.0, -cfg.spoof_rate_mps * dt_sp, 0.0};
      }

      if (tick % 10U == 0U) {
        gnss_denied::GnssFixV116 gfix{};
        gfix.stamp_ns = t_ns;
        gfix.pos_ned = plant.pos_ned;
        gfix.vel_ned = plant.vel_ned;
        if (t_s >= cfg.fault_time_s) {
          const double dt_sp = t_s - cfg.fault_time_s;
          gfix.pos_ned.y += cfg.spoof_rate_mps * dt_sp;
          gfix.vel_ned.y += cfg.spoof_rate_mps;
        }
        const auto gstatus = spoof_guard.step_10hz(gfix, vio_est);
        if (gstatus.latched && !spoof_detected) {
          spoof_detected = true;
          spoof_ttd_s = t_s - cfg.fault_time_s;
          spoof_rollback_ok = gstatus.rollback_executed;
          nav_pos_offset = {0.0, 0.0, 0.0};  // Rolled back to clean VIO anchor
          ctr_fsm.reset(gstatus.sigma_init_m);
          descending = true;
        }
      }
      if (spoof_detected) {
        ctr_fsm.propagate_open_loop(kDt100Hz, std::hypot(plant.vel_ned.x, plant.vel_ned.y));
      }
    }

    // Airspace & Coastal Hospital evaluation
    std::array<rta::HazardPoint, 4> hazards{};
    std::size_t n_haz = 0U;
    std::array<rta::HazardPoint, 2> ctr_haz{};
    const auto ctr_st = ctr_fsm.evaluate_and_build_hazards(
        kDt100Hz, plant.pos_ned, 12.0 - plant.pos_ned.z, ctr_haz);
    out.min_ltfj_margin_m = std::min(out.min_ltfj_margin_m, ctr_st.ltfj_nominal_margin_m);
    out.min_ltbx_margin_m = std::min(out.min_ltbx_margin_m, ctr_st.ltbx_nominal_margin_m);
    out.max_3sigma_m = std::max(out.max_3sigma_m, ctr_st.three_sigma_pos_m);
    if (ctr_st.ltfj_nominal_margin_m <= 0.0) {
      out.ltfj_breach = true;
    }
    if (ctr_st.ltbx_nominal_margin_m <= 0.0) {
      out.ltbx_breach = true;
    }
    for (std::size_t i = 0U;
         i < ctr_st.emitted_ctr_hazards && i < ctr_haz.size() && n_haz < hazards.size();
         ++i) {
      hazards[n_haz++] = ctr_haz[i];
    }

    const double hosp_dist = std::hypot(
        plant.pos_ned.x - hospital_haz.pos_ned.x,
        plant.pos_ned.y - hospital_haz.pos_ned.y);
    out.min_hospital_dist_m = std::min(out.min_hospital_dist_m, hosp_dist);
    if (hosp_dist < hospital_haz.radius_h_m) {
      out.hospital_breach = true;
    }
    if (hosp_dist < 600.0 && n_haz < hazards.size()) {
      hazards[n_haz++] = hospital_haz;
    }

    // Sensor-noisy nav state for outer guidance (GNSS/INS bias + jitter, or VIO drift post-spoof)
    const double alt_meas = std::max(0.0, -plant.pos_ned.z + sensor_rng.next_normal() * cfg.alt_noise_sigma_m);
    const double horiz_err_n = (spoof_detected ? (nav_bias_n * 1.4 + vio_drift_n) : nav_bias_n) +
                               sensor_rng.next_normal() * 0.14;
    const double horiz_err_e = (spoof_detected ? (nav_bias_e * 1.4 + vio_drift_e) : nav_bias_e) +
                               sensor_rng.next_normal() * 0.14;
    const core::Vec3Ned pos_meas{
        plant.pos_ned.x + nav_pos_offset.x + horiz_err_n,
        plant.pos_ned.y + nav_pos_offset.y + horiz_err_e,
        -alt_meas,
    };

    const double err_n = slz_n - pos_meas.x;
    const double err_e = slz_e - pos_meas.y;
    const double horiz_dist = std::hypot(err_n, err_e);

    if (!descending) {
      if (horiz_dist < 22.0 || t_s >= 10.0 || active_fault_mask != 0U) {
        descending = true;
      }
    }

    // Outer 100 Hz guidance -> nominal NED acceleration
    const double v_hor_cap = (active_fault_mask != 0U) ? 6.0 : 8.5;
    const double vn_des = std::clamp(0.45 * err_n, -v_hor_cap, v_hor_cap);
    const double ve_des = std::clamp(0.45 * err_e, -v_hor_cap, v_hor_cap);
    const double evx = vn_des - plant.vel_ned.x;
    const double evy = ve_des - plant.vel_ned.y;

    const bool in_cel_regime =
        cbvf.is_cel_latched() ||
        (core::CoaxEffectiveness::classify_regime(active_fault_mask) ==
         core::FlightRegime::SAME_ARM_ACTIVE_SPIN);

    double a_n_nom = 0.0;
    double a_e_nom = 0.0;
    double a_d_nom = 0.0;

    if (in_cel_regime) {
      // Canonical Regime-3 CEL vertical braking profile (no integrator windup starving spin-up yaw!)
      const double vz_des =
          (alt_meas > 10.0) ? 2.2 : ((alt_meas > 5.0) ? 1.5 : ((alt_meas > 1.0) ? 0.9 : 0.5));
      a_d_nom = std::clamp(2.5 * (vz_des - plant.vel_ned.z), -2.2, 3.0);
    } else {
      int_ax = std::clamp(int_ax + 0.30 * evx * kDt100Hz, -2.0, 2.0);
      int_ay = std::clamp(int_ay + 0.30 * evy * kDt100Hz, -2.0, 2.0);
      a_n_nom = 1.15 * evx + int_ax;
      a_e_nom = 1.15 * evy + int_ay;
      const double a_xy_norm = std::hypot(a_n_nom, a_e_nom);
      const double a_xy_lim = (active_fault_mask != 0U) ? 2.6 : 3.2;  // <= 0.33 g comfort/envelope cap
      if (a_xy_norm > a_xy_lim) {
        const double sc = a_xy_lim / a_xy_norm;
        a_n_nom *= sc;
        a_e_nom *= sc;
      }

      if (!descending) {
        const double alt_err = (-cfg.init_agl_m) - pos_meas.z;
        const double vz_err = (0.55 * alt_err) - plant.vel_ned.z;
        int_az = std::clamp(int_az + 0.45 * vz_err * kDt100Hz, -2.8, 2.5);
        a_d_nom = std::clamp(1.6 * vz_err + int_az, -3.2, 2.4);
      } else {
        double vz_des = 0.0;
        if (horiz_dist > 12.0) {
          vz_des = (alt_meas > 18.0) ? 1.8 : 0.4;
        } else {
          vz_des = (alt_meas > 18.0) ? 2.4 : ((alt_meas > 6.0) ? 1.35 : ((alt_meas > 2.0) ? 0.80 : 0.55));
        }
        const double vz_err = vz_des - plant.vel_ned.z;
        int_az = std::clamp(int_az + 0.55 * vz_err * kDt100Hz, -3.2, 2.2);
        a_d_nom = std::clamp(2.1 * vz_err + int_az, -3.8, 2.8);
      }
    }

    const core::Vec3Ned a_nom{a_n_nom, a_e_nom, a_d_nom};
    const auto env = core::CoaxEffectiveness::compute_envelope(
        active_fault_mask, 0U, 0x07U,
        plant.batt_state.v_term_v, plant.batt_state.i_pack_a, plant.batt_state.soc,
        ctrl.spin_rate_ref_rad_s(), static_cast<float>(plant.omega_frd.z),
        (active_fault_mask != 0U) ? 18.0f : 0.5f,
        (active_fault_mask != 0U) ? 15U : 0U);

    const auto rta_out = cbvf.step_100hz(
        plant.pos_ned, plant.vel_ned, a_nom, 0.0, env,
        std::span<const rta::HazardPoint>(hazards.data(), n_haz));

    const double da_rta = (rta_out.a_safe_ned - a_nom).norm();
    if (da_rta > 0.15 || rta_out.cel_latched || rta_out.used_hj_fallback) {
      out.rta_intervened = true;
    }
    if (rta_out.cel_latched) {
      out.cel_latched = true;
    }

    // 4x 400 Hz inner INDI-WLS + perturbed 6-DOF RK4 substeps
    for (int sub = 0; sub < 4; ++sub) {
      float esc_omega[8]{};
      bool esc_valid[8]{};
      for (int i = 0; i < 8; ++i) {
        esc_omega[i] = static_cast<float>(plant.rotor_omega[i]);
        esc_valid[i] = true;
      }
      core::IndiInput in{};
      in.q_frd_ned = plant.q_frd_ned;
      in.gyro_frd = {
          plant.omega_frd.x + sensor_rng.next_normal() * cfg.imu_gyro_sigma_rads,
          plant.omega_frd.y + sensor_rng.next_normal() * cfg.imu_gyro_sigma_rads,
          plant.omega_frd.z + sensor_rng.next_normal() * cfg.imu_gyro_sigma_rads,
      };
      in.accel_des_ned = rta_out.a_safe_ned;
      in.yaw_rate_des = rta_out.yaw_rate_safe;
      in.v_horiz_mps = static_cast<float>(std::hypot(plant.vel_ned.x, plant.vel_ned.y));
      in.v_term_v = plant.batt_state.v_term_v;
      in.fault_mask = active_fault_mask;
      in.esc_omega_rad_s = esc_omega;
      in.esc_valid = esc_valid;

      float omega_cmd[8]{};
      (void)ctrl.step_400hz(in, omega_cmd);
      plant.step_400hz(omega_cmd, active_fault_mask, wind_ned);
      if (plant.on_ground) {
        break;
      }
    }

    if (!plant.pos_ned.is_finite() || !plant.vel_ned.is_finite() || !plant.omega_frd.is_finite()) {
      out.all_finite = false;
      break;
    }

    out.max_tilt_deg = std::max(out.max_tilt_deg, compute_tilt_deg(plant.q_frd_ned));
    out.min_pack_voltage_v = std::min(out.min_pack_voltage_v, static_cast<double>(plant.batt_state.v_term_v));
    if (plant.batt_state.voltage_collapse) {
      out.battery_collapse = true;
    }

    if (plant.on_ground) {
      out.reached_ground = true;
      out.touchdown_vz_mps = std::abs(plant.touchdown_vz_mps);
      out.touchdown_dx_m = plant.pos_ned.x - slz_n;
      out.touchdown_dy_m = plant.pos_ned.y - slz_e;
      out.touchdown_error_m = std::hypot(out.touchdown_dx_m, out.touchdown_dy_m);
      out.flight_duration_s = t_s + kDt100Hz;
      break;
    }
    out.flight_duration_s = t_s + kDt100Hz;
  }

  out.final_soc = static_cast<double>(plant.batt_state.soc);
  out.airspace_breach = out.ltfj_breach || out.ltbx_breach || out.hospital_breach;
  out.spoof_detected = spoof_detected;
  out.spoof_ttd_s = spoof_ttd_s;
  out.spoof_rollback_ok = spoof_rollback_ok;

  if (!out.reached_ground) {
    out.touchdown_dx_m = plant.pos_ned.x - slz_n;
    out.touchdown_dy_m = plant.pos_ned.y - slz_e;
    out.touchdown_error_m = std::hypot(out.touchdown_dx_m, out.touchdown_dy_m);
    out.touchdown_vz_mps = std::max(0.0, plant.vel_ned.z);
  }

  // Classify outcome honestly with exact physical failure mode
  if (!out.all_finite) {
    out.safe_recovery = false;
    out.failure_mode = "NUMERICAL_DIVERGENCE";
  } else if (out.airspace_breach) {
    out.safe_recovery = false;
    out.failure_mode = "AIRSPACE_BREACH";
  } else if (out.battery_collapse || out.min_pack_voltage_v < kMinSafePackVoltageV) {
    out.safe_recovery = false;
    out.failure_mode = "VOLTAGE_COLLAPSE";
  } else if (out.max_tilt_deg > kMaxSafeTiltDeg) {
    out.safe_recovery = false;
    out.failure_mode = "ATTITUDE_UPSET";
  } else if (!out.reached_ground) {
    out.safe_recovery = false;
    out.failure_mode = "INCOMPLETE_DESCENT";
  } else if (out.touchdown_vz_mps > kMaxSafeTouchdownVzMps) {
    out.safe_recovery = false;
    out.failure_mode = "THRUST_SAT_HARD_IMPACT";
  } else {
    out.safe_recovery = true;
    out.failure_mode = "NONE";
  }

  out.wall_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_wall0).count();
  return out;
}

// ----------------------------------------------------------------------------
// Lift+Cruise 8+1 (VertiRTA-LC-240) Single-Flight Closed-Loop Runner
// ----------------------------------------------------------------------------
[[nodiscard]] FlightOutcome run_lc_flight(const FlightSampleConfig& cfg) noexcept {
  const auto t_wall0 = std::chrono::steady_clock::now();
  FlightOutcome out{};
  out.flight_id = cfg.flight_id;
  out.airframe = cfg.airframe;
  out.regime = cfg.regime;
  out.mass_kg = cfg.mass_kg;
  out.cg_offset_m = std::hypot(cfg.cg_x_m, cfg.cg_y_m);
  out.wind_speed_mps = cfg.wind_speed_mps;
  out.init_agl_m = cfg.init_agl_m;
  out.init_soc = cfg.init_soc;

  airframes::LiftCruiseDynamics lc_plant(airframes::LiftCruiseLayout::kEightPlusOne);
  airframes::AirframeState6Dof lc_st{};
  lc_st.pos_ned = {cfg.init_north_m, cfg.init_east_m, -cfg.init_agl_m};
  lc_st.vel_ned = {4.0, 0.0, 0.0};
  const double t_hover_each = cfg.mass_kg * airframes::LiftCruiseDynamics::kGravity / 8.0;
  for (uint8_t i = 0U; i < 8U; ++i) {
    lc_st.u_act[i] = std::min(t_hover_each, lc_plant.max_lift_rotor_thrust_n());
  }

  // Synchronized 15 Hz 2nd-order Butterworth filters for true sensor-driven INDI (matching TransitionIndiWlsController)
  struct Biquad15 {
    double x1{0.0}, x2{0.0}, y1{0.0}, y2{0.0};
    void reset(double v) noexcept { x1 = x2 = y1 = y2 = v; }
    [[nodiscard]] double step(double x) noexcept {
      constexpr double b0 = 0.011906332234389828;
      constexpr double b1 = 0.023812664468779656;
      constexpr double b2 = 0.011906332234389828;
      constexpr double a1 = -1.6692031429311931;
      constexpr double a2 = 0.7168284718687524;
      const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
      return y;
    }
  };
  std::array<Biquad15, 3> gyro_lpf{};
  std::array<Biquad15, 3> accel_lpf{};
  std::array<Biquad15, airframes::kMaxActuators> act_lpf{};
  for (int i = 0; i < 3; ++i) {
    gyro_lpf[i].reset(0.0);
  }
  accel_lpf[0].reset(0.0);
  accel_lpf[1].reset(0.0);
  accel_lpf[2].reset(-airframes::LiftCruiseDynamics::kGravity);
  for (uint8_t i = 0U; i < airframes::kMaxActuators; ++i) {
    act_lpf[i].reset(lc_st.u_act[i]);
  }
  core::Vec3Frd gyro_f{};
  core::Vec3Frd gyro_f_prev{};
  core::Vec3Frd accel_f{0.0, 0.0, -airframes::LiftCruiseDynamics::kGravity};
  std::array<double, airframes::kMaxActuators> u_f = lc_st.u_act;

  core::TheveninBattery1RC lc_battery{static_cast<float>(cfg.init_soc)};
  core::BatteryState batt_state = lc_battery.state();
  double last_p_elec_w = 35000.0;

  rta::CbvfSafetyFilter cbvf(nullptr, nullptr);
  rta::DrydenTurbulenceFilter dryden(cfg.seed ^ 0xCAFEBABE87654321ULL);
  FastGaussian sensor_rng(cfg.seed ^ 0x9E3779B97F4A7C15ULL);

  // Realistic GNSS/INS horizontal position bias (OU process, sigma ~ 0.65 m, tau = 12 s)
  double nav_bias_n = sensor_rng.next_normal() * 0.65;
  double nav_bias_e = sensor_rng.next_normal() * 0.65;
  double vio_drift_n = 0.0;
  double vio_drift_e = 0.0;
  const double vio_drift_rate_n = sensor_rng.next_normal() * 0.008;
  const double vio_drift_rate_e = sensor_rng.next_normal() * 0.008;

  gnss_denied::BasaltVioAdapter vio{};
  vio.reset(lc_st.pos_ned, lc_st.vel_ned);
  gnss_denied::GnssSpoofGuard spoof_guard{};
  gnss_denied::CtrInflationFsm ctr_fsm{};
  ctr_fsm.reset(/*sigma_init_m=*/2.0);
  const auto hospital_haz = build_coastal_hospital_hazard();

  double slz_n = cfg.init_north_m + 85.0;
  double slz_e = std::min(cfg.init_east_m - 15.0, -140.0);
  bool emergency_slz_latched = false;

  const core::Vec3Ned mean_wind_ned{
      cfg.wind_speed_mps * std::cos(cfg.wind_dir_rad),
      cfg.wind_speed_mps * std::sin(cfg.wind_dir_rad),
      0.0,
  };

  uint16_t active_fault_mask = 0U;
  bool spoof_detected = false;
  double spoof_ttd_s = -1.0;
  bool spoof_rollback_ok = false;
  core::Vec3Ned nav_pos_offset{0.0, 0.0, 0.0};
  core::Vec3Ned prev_vel_ned = lc_st.vel_ned;
  gnss_denied::VioStateEstimate vio_est = vio.state();

  double int_ax = 0.0;
  double int_ay = 0.0;
  double int_az = 0.0;
  bool descending = false;
  std::array<int8_t, airframes::kMaxActuators> wls_warm{};
  const uint32_t max_ticks = static_cast<uint32_t>(std::ceil(cfg.max_flight_s / kDt100Hz));

  const uint8_t n_act = lc_plant.actuator_count();
  const uint8_t n_lift = lc_plant.lift_rotor_count();
  const uint8_t p_idx = lc_plant.pusher_index();
  const uint8_t a_idx = lc_plant.aileron_index();
  const uint8_t e_idx = lc_plant.elevator_index();
  const uint8_t r_idx = lc_plant.rudder_index();

  for (uint32_t tick = 0U; tick < max_ticks; ++tick) {
    const double t_s = static_cast<double>(tick) * kDt100Hz;
    const uint64_t t_ns = static_cast<uint64_t>(tick) * 10'000'000ULL;

    if (cfg.regime == FaultRegimeKind::kSingleRotorLossCsfl ||
        cfg.regime == FaultRegimeKind::kCompoundLowSocWindFault) {
      if (t_s >= cfg.fault_time_s) {
        active_fault_mask = static_cast<uint16_t>(1U << (cfg.fault_index % 8U));
      }
    } else if (cfg.regime == FaultRegimeKind::kSameArmDualKillCel) {
      if (t_s >= cfg.fault_time_s) {
        // Inner-boom dual lift-motor power-bus loss: {1, 5} (left inner) or {2, 6} (right inner)
        const bool right_boom = (cfg.fault_index % 2U) != 0U;
        active_fault_mask = right_boom ? static_cast<uint16_t>((1U << 2U) | (1U << 6U))
                                       : static_cast<uint16_t>((1U << 1U) | (1U << 5U));
        if (!emergency_slz_latched) {
          emergency_slz_latched = true;
          slz_n = lc_st.pos_ned.x + lc_st.vel_ned.x * 2.0;
          slz_e = lc_st.pos_ned.y + lc_st.vel_ned.y * 2.0;
          descending = true;
        }
      }
    }

    const double agl_true = std::max(1.0, -lc_st.pos_ned.z);
    const double v_air_est = std::max(2.0, (lc_st.vel_ned - mean_wind_ned).norm());
    const double w20_eff = std::max(1.5, cfg.wind_speed_mps * cfg.gust_sigma_scale);
    const core::Vec3Ned gust = dryden.step_seeded(kDt100Hz, agl_true, v_air_est, w20_eff);
    const double gust_gain = 0.40 * cfg.gust_sigma_scale;
    const core::Vec3Ned wind_ned{
        mean_wind_ned.x + std::clamp(gust.x * gust_gain, -6.0, 6.0),
        mean_wind_ned.y + std::clamp(gust.y * gust_gain, -6.0, 6.0),
        std::clamp(gust.z * gust_gain, -2.0, 2.0),
    };
    out.max_wind_mps = std::max(out.max_wind_mps, wind_ned.norm());

    constexpr double kOuPhi = 0.999167;
    constexpr double kOuSig = 0.0265;
    nav_bias_n = kOuPhi * nav_bias_n + kOuSig * sensor_rng.next_normal();
    nav_bias_e = kOuPhi * nav_bias_e + kOuSig * sensor_rng.next_normal();
    if (spoof_detected) {
      const double ds = std::hypot(lc_st.vel_ned.x, lc_st.vel_ned.y) * kDt100Hz;
      vio_drift_n += vio_drift_rate_n * ds + 0.012 * sensor_rng.next_normal();
      vio_drift_e += vio_drift_rate_e * ds + 0.012 * sensor_rng.next_normal();
    }

    if (cfg.regime == FaultRegimeKind::kGnssSlowRampSpoof) {
      const core::Vec3Ned imu_accel_ned =
          (tick > 0U) ? ((lc_st.vel_ned - prev_vel_ned) * (1.0 / kDt100Hz)) : core::Vec3Ned{0.0, 0.0, 0.0};
      prev_vel_ned = lc_st.vel_ned;

      gnss_denied::VioSensorFrame vf{};
      vf.stamp_ns = t_ns;
      vf.dt_s = kDt100Hz;
      vf.imu_accel_ned = imu_accel_ned;
      vf.has_stereo_frame = (tick % 10U == 0U);
      vf.stereo_delta_vel_ned = lc_st.vel_ned;
      vf.radalt_agl_m = agl_true;
      vf.radalt_valid = true;
      vf.pidnet_water_fraction = 0.08;
      (void)vio.step(vf, vio_est);

      if (t_s >= cfg.fault_time_s && !spoof_detected) {
        const double dt_sp = t_s - cfg.fault_time_s;
        nav_pos_offset = {0.0, -cfg.spoof_rate_mps * dt_sp, 0.0};
      }

      if (tick % 10U == 0U) {
        gnss_denied::GnssFixV116 gfix{};
        gfix.stamp_ns = t_ns;
        gfix.pos_ned = lc_st.pos_ned;
        gfix.vel_ned = lc_st.vel_ned;
        if (t_s >= cfg.fault_time_s) {
          const double dt_sp = t_s - cfg.fault_time_s;
          gfix.pos_ned.y += cfg.spoof_rate_mps * dt_sp;
          gfix.vel_ned.y += cfg.spoof_rate_mps;
        }
        const auto gstatus = spoof_guard.step_10hz(gfix, vio_est);
        if (gstatus.latched && !spoof_detected) {
          spoof_detected = true;
          spoof_ttd_s = t_s - cfg.fault_time_s;
          spoof_rollback_ok = gstatus.rollback_executed;
          nav_pos_offset = {0.0, 0.0, 0.0};
          ctr_fsm.reset(gstatus.sigma_init_m);
          descending = true;
        }
      }
      if (spoof_detected) {
        ctr_fsm.propagate_open_loop(kDt100Hz, std::hypot(lc_st.vel_ned.x, lc_st.vel_ned.y));
      }
    }

    std::array<rta::HazardPoint, 4> hazards{};
    std::size_t n_haz = 0U;
    std::array<rta::HazardPoint, 2> ctr_haz{};
    const auto ctr_st = ctr_fsm.evaluate_and_build_hazards(
        kDt100Hz, lc_st.pos_ned, 12.0 - lc_st.pos_ned.z, ctr_haz);
    out.min_ltfj_margin_m = std::min(out.min_ltfj_margin_m, ctr_st.ltfj_nominal_margin_m);
    out.min_ltbx_margin_m = std::min(out.min_ltbx_margin_m, ctr_st.ltbx_nominal_margin_m);
    out.max_3sigma_m = std::max(out.max_3sigma_m, ctr_st.three_sigma_pos_m);
    if (ctr_st.ltfj_nominal_margin_m <= 0.0) {
      out.ltfj_breach = true;
    }
    if (ctr_st.ltbx_nominal_margin_m <= 0.0) {
      out.ltbx_breach = true;
    }
    for (std::size_t i = 0U;
         i < ctr_st.emitted_ctr_hazards && i < ctr_haz.size() && n_haz < hazards.size();
         ++i) {
      hazards[n_haz++] = ctr_haz[i];
    }

    const double hosp_dist = std::hypot(
        lc_st.pos_ned.x - hospital_haz.pos_ned.x,
        lc_st.pos_ned.y - hospital_haz.pos_ned.y);
    out.min_hospital_dist_m = std::min(out.min_hospital_dist_m, hosp_dist);
    if (hosp_dist < hospital_haz.radius_h_m) {
      out.hospital_breach = true;
    }
    if (hosp_dist < 600.0 && n_haz < hazards.size()) {
      hazards[n_haz++] = hospital_haz;
    }

    const double alt_meas = std::max(0.0, -lc_st.pos_ned.z + sensor_rng.next_normal() * cfg.alt_noise_sigma_m);
    const double horiz_err_n = (spoof_detected ? (nav_bias_n * 1.4 + vio_drift_n) : nav_bias_n) +
                               sensor_rng.next_normal() * 0.14;
    const double horiz_err_e = (spoof_detected ? (nav_bias_e * 1.4 + vio_drift_e) : nav_bias_e) +
                               sensor_rng.next_normal() * 0.14;
    const core::Vec3Ned pos_meas{
        lc_st.pos_ned.x + nav_pos_offset.x + horiz_err_n,
        lc_st.pos_ned.y + nav_pos_offset.y + horiz_err_e,
        -alt_meas,
    };

    const double err_n = slz_n - pos_meas.x;
    const double err_e = slz_e - pos_meas.y;
    const double horiz_dist = std::hypot(err_n, err_e);

    if (!descending && (horiz_dist < 22.0 || t_s >= 10.0 || active_fault_mask != 0U)) {
      descending = true;
    }

    const bool dual_fault = (cfg.regime == FaultRegimeKind::kSameArmDualKillCel && active_fault_mask != 0U);
    const double v_hor_cap = dual_fault ? 3.5 : ((active_fault_mask != 0U) ? 5.0 : 6.0);
    const double vn_des = std::clamp(0.40 * err_n, -v_hor_cap, v_hor_cap);
    const double ve_des = std::clamp(0.40 * err_e, -v_hor_cap, v_hor_cap);
    const double evx = vn_des - lc_st.vel_ned.x;
    const double evy = ve_des - lc_st.vel_ned.y;
    int_ax = std::clamp(int_ax + 0.35 * evx * kDt100Hz, -2.2, 2.2);
    int_ay = std::clamp(int_ay + 0.35 * evy * kDt100Hz, -2.2, 2.2);

    const double a_xy_lim = dual_fault ? 1.6 : 2.5;
    double a_n_nom = 1.15 * evx + int_ax;
    double a_e_nom = 1.15 * evy + int_ay;
    const double a_xy_mag = std::hypot(a_n_nom, a_e_nom);
    if (a_xy_mag > a_xy_lim) {
      const double sc = a_xy_lim / a_xy_mag;
      a_n_nom *= sc;
      a_e_nom *= sc;
    }
    double a_d_nom = 0.0;
    if (!descending) {
      const double alt_err = (-cfg.init_agl_m) - pos_meas.z;
      const double vz_err = (0.55 * alt_err) - lc_st.vel_ned.z;
      int_az = std::clamp(int_az + 0.45 * vz_err * kDt100Hz, -3.0, 2.5);
      a_d_nom = std::clamp(1.6 * vz_err + int_az, -3.0, 2.2);
    } else {
      double vz_des = 0.0;
      if (horiz_dist > 14.0 && !dual_fault) {
        vz_des = (alt_meas > 18.0) ? 1.7 : 0.4;
      } else {
        vz_des = (alt_meas > 18.0) ? 2.3 : ((alt_meas > 6.0) ? 1.25 : ((alt_meas > 2.0) ? 0.75 : 0.50));
      }
      const double vz_err = vz_des - lc_st.vel_ned.z;
      int_az = std::clamp(int_az + 0.55 * vz_err * kDt100Hz, -3.2, 2.0);
      a_d_nom = std::clamp(2.0 * vz_err + int_az, -3.2, 2.4);
    }

    const core::Vec3Ned a_nom{a_n_nom, a_e_nom, a_d_nom};
    const auto lc_env = build_lc_envelope(lc_plant, batt_state, active_fault_mask, last_p_elec_w);
    const auto rta_out = cbvf.step_100hz(
        lc_st.pos_ned, lc_st.vel_ned, a_nom, 0.0, lc_env,
        std::span<const rta::HazardPoint>(hazards.data(), n_haz));
    if ((rta_out.a_safe_ned - a_nom).norm() > 0.15 || rta_out.used_hj_fallback || dual_fault) {
      out.rta_intervened = true;
    }

    const core::Vec3Ned a_safe = rta_out.a_safe_ned;
    const double tilt_cap = static_cast<double>(lc_env.tilt_max_rad);

    // 4x 400 Hz Lift+Cruise synchronized 15 Hz INDI + WLS 12x5 hover-approach tracking (pusher idle)
    bool hit_ground = false;
    for (int sub = 0; sub < 4; ++sub) {
      double roll = 0.0, pitch = 0.0, yaw = 0.0;
      quat_to_euler_zyx(lc_st.q_frd_ned, roll, pitch, yaw);

      // True IMU measurement (actual perturbed 6-DOF specific force + noisy gyro) -> synchronized 15 Hz LPF
      const auto dyn_imu = eval_lc_perturbed(
          lc_plant, lc_st, wind_ned, active_fault_mask, cfg.mass_kg, cfg.cg_x_m, cfg.cg_y_m);
      const core::Vec3Frd gyro_meas{
          lc_st.omega_frd.x + sensor_rng.next_normal() * cfg.imu_gyro_sigma_rads,
          lc_st.omega_frd.y + sensor_rng.next_normal() * cfg.imu_gyro_sigma_rads,
          lc_st.omega_frd.z + sensor_rng.next_normal() * cfg.imu_gyro_sigma_rads,
      };
      const core::Vec3Frd accel_meas{
          dyn_imu.accel_body_frd.x + sensor_rng.next_normal() * 0.06,
          dyn_imu.accel_body_frd.y + sensor_rng.next_normal() * 0.06,
          dyn_imu.accel_body_frd.z + sensor_rng.next_normal() * 0.06,
      };
      gyro_f_prev = gyro_f;
      gyro_f = {gyro_lpf[0].step(gyro_meas.x), gyro_lpf[1].step(gyro_meas.y), gyro_lpf[2].step(gyro_meas.z)};
      const core::Vec3Frd dgyro_f = (gyro_f - gyro_f_prev) * (1.0 / kDt400Hz);
      accel_f = {accel_lpf[0].step(accel_meas.x), accel_lpf[1].step(accel_meas.y), accel_lpf[2].step(accel_meas.z)};
      for (uint8_t i = 0U; i < n_act; ++i) {
        u_f[i] = act_lpf[i].step(lc_st.u_act[i]);
      }

      const double cy = std::cos(yaw);
      const double sy = std::sin(yaw);
      const double a_fwd = cy * a_safe.x + sy * a_safe.y;
      const double a_right = -sy * a_safe.x + cy * a_safe.y;
      const double az_up = std::max(3.0, airframes::LiftCruiseDynamics::kGravity - a_safe.z);

      const double phi_des = std::clamp(std::atan2(a_right, az_up), -tilt_cap, tilt_cap);
      const double theta_des = std::clamp(std::atan2(-a_fwd, az_up), -tilt_cap, tilt_cap);
      const double yaw_err = core::wrap_pi_d(0.0 - yaw);

      const double p_dot_des = std::clamp(16.0 * (phi_des - roll) - 7.0 * gyro_f.x, -6.0, 6.0);
      const double q_dot_des = std::clamp(16.0 * (theta_des - pitch) - 7.0 * gyro_f.y, -6.0, 6.0);
      const double r_dot_des = std::clamp(6.0 * yaw_err - 4.0 * gyro_f.z, -4.0, 4.0);

      const double cos_tilt = std::max(0.78, std::cos(roll) * std::cos(pitch));
      const double fz_body_des = -(az_up / cos_tilt);

      airframes::EffectivenessMatrix eff{};
      lc_plant.compute_effectiveness(lc_st, {0.0, 0.0, 0.0}, active_fault_mask, kDt400Hz, eff);

      std::array<double, airframes::kMaxActuators> Wu{};
      std::array<double, airframes::kMaxActuators> ud_minus_u0{};
      const double t_pref = airframes::LiftCruiseDynamics::kWeightN / 8.0;
      for (uint8_t i = 0U; i < n_lift; ++i) {
        const bool dead = (active_fault_mask & static_cast<uint16_t>(1U << i)) != 0U;
        Wu[i] = 1.0;
        ud_minus_u0[i] = (dead ? 0.0 : t_pref) - u_f[i];
        if (dead) {
          eff.du_min[i] = -u_f[i];
          eff.du_max[i] = -u_f[i];
        } else {
          eff.du_min[i] = 0.0 - u_f[i];
          eff.du_max[i] = lc_plant.max_lift_rotor_thrust_n() - u_f[i];
        }
      }
      // Pusher unloaded during vertical hover-approach/touchdown
      Wu[p_idx] = 20.0;
      ud_minus_u0[p_idx] = -u_f[p_idx];
      eff.du_min[p_idx] = -u_f[p_idx];
      eff.du_max[p_idx] = -u_f[p_idx];
      for (uint8_t idx : {a_idx, e_idx, r_idx}) {
        Wu[idx] = 20.0;
        ud_minus_u0[idx] = -u_f[idx];
        eff.du_min[idx] = eff.u_min[idx] - u_f[idx];
        eff.du_max[idx] = eff.u_max[idx] - u_f[idx];
      }

      const std::array<double, airframes::kMaxControls> v_req{
          p_dot_des - dgyro_f.x,
          q_dot_des - dgyro_f.y,
          r_dot_des - dgyro_f.z,
          fz_body_des - accel_f.z,
          0.0,
      };
      const std::array<double, airframes::kMaxControls> Wv{10.0, 10.0, 1.5, 6.0, 0.0};
      const auto wls = airframes::wls_alloc_12x5_f64(
          eff.B, v_req, eff.du_min, eff.du_max, Wv, Wu, ud_minus_u0, 1e6, wls_warm, n_act, 12U);
      wls_warm = wls.W;

      std::array<double, airframes::kMaxActuators> u_cmd{};
      for (uint8_t i = 0U; i < n_act; ++i) {
        const bool dead = (i < n_lift) && ((active_fault_mask & static_cast<uint16_t>(1U << i)) != 0U);
        u_cmd[i] = (dead || i == p_idx)
                       ? 0.0
                       : core::clamp_d(u_f[i] + wls.x[i], eff.u_min[i], eff.u_max[i]);
      }

      step_lc_plant_400hz(
          lc_plant, lc_st, u_cmd, wind_ned, active_fault_mask,
          static_cast<double>(batt_state.v_term_v), cfg.mass_kg, cfg.cg_x_m, cfg.cg_y_m);

      if (lc_st.pos_ned.z >= 0.0) {
        out.reached_ground = true;
        out.touchdown_vz_mps = std::abs(lc_st.vel_ned.z);
        lc_st.pos_ned.z = 0.0;
        hit_ground = true;
        break;
      }
    }

    if (!lc_st.pos_ned.is_finite() || !lc_st.vel_ned.is_finite() || !lc_st.omega_frd.is_finite()) {
      out.all_finite = false;
      break;
    }

    last_p_elec_w = compute_lc_elec_power_w(lc_plant, lc_st, wind_ned, active_fault_mask);
    batt_state = lc_battery.step(static_cast<float>(kDt100Hz), static_cast<float>(last_p_elec_w));
    out.min_pack_voltage_v = std::min(out.min_pack_voltage_v, static_cast<double>(batt_state.v_term_v));
    if (batt_state.voltage_collapse) {
      out.battery_collapse = true;
    }
    out.max_tilt_deg = std::max(out.max_tilt_deg, compute_tilt_deg(lc_st.q_frd_ned));
    out.flight_duration_s = t_s + kDt100Hz;

    if (hit_ground) {
      out.touchdown_dx_m = lc_st.pos_ned.x - slz_n;
      out.touchdown_dy_m = lc_st.pos_ned.y - slz_e;
      out.touchdown_error_m = std::hypot(out.touchdown_dx_m, out.touchdown_dy_m);
      break;
    }
  }

  out.final_soc = static_cast<double>(batt_state.soc);
  out.airspace_breach = out.ltfj_breach || out.ltbx_breach || out.hospital_breach;
  out.spoof_detected = spoof_detected;
  out.spoof_ttd_s = spoof_ttd_s;
  out.spoof_rollback_ok = spoof_rollback_ok;

  if (!out.reached_ground) {
    out.touchdown_dx_m = lc_st.pos_ned.x - slz_n;
    out.touchdown_dy_m = lc_st.pos_ned.y - slz_e;
    out.touchdown_error_m = std::hypot(out.touchdown_dx_m, out.touchdown_dy_m);
    out.touchdown_vz_mps = std::max(0.0, lc_st.vel_ned.z);
  }

  if (!out.all_finite) {
    out.safe_recovery = false;
    out.failure_mode = "NUMERICAL_DIVERGENCE";
  } else if (out.airspace_breach) {
    out.safe_recovery = false;
    out.failure_mode = "AIRSPACE_BREACH";
  } else if (out.battery_collapse || out.min_pack_voltage_v < kMinSafePackVoltageV) {
    out.safe_recovery = false;
    out.failure_mode = "VOLTAGE_COLLAPSE";
  } else if (out.max_tilt_deg > kMaxSafeTiltDeg) {
    out.safe_recovery = false;
    out.failure_mode = "ATTITUDE_UPSET";
  } else if (!out.reached_ground) {
    out.safe_recovery = false;
    out.failure_mode = "INCOMPLETE_DESCENT";
  } else if (out.touchdown_vz_mps > kMaxSafeTouchdownVzMps) {
    out.safe_recovery = false;
    out.failure_mode = "THRUST_SAT_HARD_IMPACT";
  } else {
    out.safe_recovery = true;
    out.failure_mode = "NONE";
  }

  out.wall_us =
      std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t_wall0).count();
  return out;
}

// ----------------------------------------------------------------------------
// Quantile & Aggregate Statistics Helpers
// ----------------------------------------------------------------------------
[[nodiscard]] double sorted_quantile(const std::vector<double>& sorted, double q) noexcept {
  if (sorted.empty()) {
    return 0.0;
  }
  const double pos = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1U);
  const std::size_t idx = static_cast<std::size_t>(pos);
  const double frac = pos - static_cast<double>(idx);
  if (idx + 1U >= sorted.size()) {
    return sorted.back();
  }
  return sorted[idx] * (1.0 - frac) + sorted[idx + 1U] * frac;
}

[[nodiscard]] AggregateStats compute_aggregate_stats(const std::vector<const FlightOutcome*>& items) {
  AggregateStats st{};
  st.flights = static_cast<uint32_t>(items.size());
  if (items.empty()) {
    return st;
  }

  std::vector<double> td_errs;
  std::vector<double> vz_vals;
  td_errs.reserve(items.size());
  vz_vals.reserve(items.size());

  double sum_td = 0.0;
  double sum_dx = 0.0;
  double sum_dy = 0.0;
  double sum_wall = 0.0;
  double min_v = 1e9;
  double max_tilt = 0.0;
  double min_ltfj = 1e9;
  double min_ltbx = 1e9;
  double min_hosp = 1e9;

  for (const auto* f : items) {
    if (f->safe_recovery) {
      ++st.safe_recoveries;
    }
    if (f->airspace_breach) {
      ++st.airspace_breaches;
    }
    if (f->rta_intervened) {
      ++st.rta_interventions;
    }
    if (f->cel_latched) {
      ++st.cel_latches;
    }
    td_errs.push_back(f->touchdown_error_m);
    vz_vals.push_back(f->touchdown_vz_mps);
    sum_td += f->touchdown_error_m;
    sum_dx += f->touchdown_dx_m;
    sum_dy += f->touchdown_dy_m;
    sum_wall += f->wall_us;
    min_v = std::min(min_v, f->min_pack_voltage_v);
    max_tilt = std::max(max_tilt, f->max_tilt_deg);
    min_ltfj = std::min(min_ltfj, f->min_ltfj_margin_m);
    min_ltbx = std::min(min_ltbx, f->min_ltbx_margin_m);
    min_hosp = std::min(min_hosp, f->min_hospital_dist_m);
  }

  const double n_d = static_cast<double>(items.size());
  st.recovery_ci95 = compute_binomial_ci95(st.safe_recoveries, st.flights);
  st.breach_ci95 = compute_binomial_ci95(st.airspace_breaches, st.flights);

  std::sort(td_errs.begin(), td_errs.end());
  std::sort(vz_vals.begin(), vz_vals.end());

  st.cep50_m = sorted_quantile(td_errs, 0.50);
  st.cep95_m = sorted_quantile(td_errs, 0.95);
  st.mean_td_error_m = sum_td / n_d;
  st.max_td_error_m = td_errs.back();

  st.mean_dx_m = sum_dx / n_d;
  st.mean_dy_m = sum_dy / n_d;
  double sxx = 0.0, sxy = 0.0, syy = 0.0;
  for (const auto* f : items) {
    const double dx = f->touchdown_dx_m - st.mean_dx_m;
    const double dy = f->touchdown_dy_m - st.mean_dy_m;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }
  const double denom = (items.size() > 1U) ? static_cast<double>(items.size() - 1U) : 1.0;
  st.cov_xx_m2 = sxx / denom;
  st.cov_xy_m2 = sxy / denom;
  st.cov_yy_m2 = syy / denom;

  st.vz_p50_mps = sorted_quantile(vz_vals, 0.50);
  st.vz_p95_mps = sorted_quantile(vz_vals, 0.95);
  st.vz_p99_mps = sorted_quantile(vz_vals, 0.99);
  st.vz_max_mps = vz_vals.back();

  st.min_pack_voltage_v = min_v;
  st.max_tilt_deg = max_tilt;
  st.min_ltfj_margin_m = min_ltfj;
  st.min_ltbx_margin_m = min_ltbx;
  st.min_hospital_dist_m = min_hosp;
  st.mean_wall_us = sum_wall / n_d;
  return st;
}

// Generate Latin-Hypercube stratified vector of n samples in [lo, hi]
[[nodiscard]] std::vector<double> lhs_draw(
    std::mt19937_64& rng,
    uint32_t n,
    double lo,
    double hi) {
  std::vector<double> vals(n, lo);
  if (n == 0U) {
    return vals;
  }
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const double inv_n = 1.0 / static_cast<double>(n);
  for (uint32_t i = 0U; i < n; ++i) {
    const double frac = (static_cast<double>(i) + u01(rng)) * inv_n;
    vals[i] = lo + frac * (hi - lo);
  }
  std::shuffle(vals.begin(), vals.end(), rng);
  return vals;
}

void write_ci95_json(std::ostringstream& os, const ConfidenceInterval95& ci, const std::string& ind) {
  os << "{\n";
  os << ind << "  \"count\": " << ci.count << ",\n";
  os << ind << "  \"total\": " << ci.total << ",\n";
  os << ind << "  \"rate\": " << std::setprecision(6) << safe_finite(ci.rate) << ",\n";
  os << ind << "  \"wilson_95_lo\": " << safe_finite(ci.wilson_lo) << ",\n";
  os << ind << "  \"wilson_95_hi\": " << safe_finite(ci.wilson_hi) << ",\n";
  os << ind << "  \"clopper_pearson_95_lo\": " << safe_finite(ci.clopper_pearson_lo) << ",\n";
  os << ind << "  \"clopper_pearson_95_hi\": " << safe_finite(ci.clopper_pearson_hi) << ",\n";
  os << ind << "  \"clopper_pearson_95_one_sided_hi\": " << safe_finite(ci.clopper_pearson_one_sided_hi) << "\n";
  os << ind << "}" << std::setprecision(4);
}

void write_agg_json(std::ostringstream& os, const AggregateStats& st, const std::string& ind) {
  os << "{\n";
  os << ind << "  \"flights\": " << st.flights << ",\n";
  os << ind << "  \"safe_recoveries\": " << st.safe_recoveries << ",\n";
  os << ind << "  \"airspace_breaches\": " << st.airspace_breaches << ",\n";
  os << ind << "  \"rta_interventions\": " << st.rta_interventions << ",\n";
  os << ind << "  \"cel_latches\": " << st.cel_latches << ",\n";
  os << ind << "  \"recovery_ci95\": ";
  write_ci95_json(os, st.recovery_ci95, ind + "  ");
  os << ",\n";
  os << ind << "  \"airspace_breach_ci95\": ";
  write_ci95_json(os, st.breach_ci95, ind + "  ");
  os << ",\n";
  os << ind << "  \"cep50_m\": " << safe_finite(st.cep50_m) << ",\n";
  os << ind << "  \"cep95_m\": " << safe_finite(st.cep95_m) << ",\n";
  os << ind << "  \"mean_td_error_m\": " << safe_finite(st.mean_td_error_m) << ",\n";
  os << ind << "  \"max_td_error_m\": " << safe_finite(st.max_td_error_m) << ",\n";
  os << ind << "  \"mean_dx_m\": " << safe_finite(st.mean_dx_m) << ",\n";
  os << ind << "  \"mean_dy_m\": " << safe_finite(st.mean_dy_m) << ",\n";
  os << ind << "  \"cov_xx_m2\": " << safe_finite(st.cov_xx_m2) << ",\n";
  os << ind << "  \"cov_xy_m2\": " << safe_finite(st.cov_xy_m2) << ",\n";
  os << ind << "  \"cov_yy_m2\": " << safe_finite(st.cov_yy_m2) << ",\n";
  os << ind << "  \"vz_p50_mps\": " << safe_finite(st.vz_p50_mps) << ",\n";
  os << ind << "  \"vz_p95_mps\": " << safe_finite(st.vz_p95_mps) << ",\n";
  os << ind << "  \"vz_p99_mps\": " << safe_finite(st.vz_p99_mps) << ",\n";
  os << ind << "  \"vz_max_mps\": " << safe_finite(st.vz_max_mps) << ",\n";
  os << ind << "  \"min_pack_voltage_v\": " << safe_finite(st.min_pack_voltage_v) << ",\n";
  os << ind << "  \"max_tilt_deg\": " << safe_finite(st.max_tilt_deg) << ",\n";
  os << ind << "  \"min_ltfj_margin_m\": " << safe_finite(st.min_ltfj_margin_m) << ",\n";
  os << ind << "  \"min_ltbx_margin_m\": " << safe_finite(st.min_ltbx_margin_m) << ",\n";
  os << ind << "  \"min_hospital_dist_m\": " << safe_finite(st.min_hospital_dist_m) << ",\n";
  os << ind << "  \"mean_wall_us\": " << safe_finite(st.mean_wall_us) << "\n";
  os << ind << "}";
}

}  // namespace

const char* airframe_name(AirframeKind kind) noexcept {
  switch (kind) {
    case AirframeKind::kCoaxialX8_240:
      return "VertiRTA-X8-240";
    case AirframeKind::kLiftCruise_240_8p1:
      return "VertiRTA-LC-240-8p1";
  }
  return "UNKNOWN";
}

const char* regime_name(FaultRegimeKind kind) noexcept {
  switch (kind) {
    case FaultRegimeKind::kNominalTurbulent:
      return "NOMINAL_TURBULENT";
    case FaultRegimeKind::kSingleRotorLossCsfl:
      return "SINGLE_ROTOR_LOSS_CSFL";
    case FaultRegimeKind::kSameArmDualKillCel:
      return "SAME_ARM_DUAL_KILL_CEL";
    case FaultRegimeKind::kGnssSlowRampSpoof:
      return "GNSS_SLOW_RAMP_SPOOF";
    case FaultRegimeKind::kCompoundLowSocWindFault:
      return "COMPOUND_LOW_SOC_WIND_FAULT";
  }
  return "UNKNOWN";
}

ConfidenceInterval95 compute_binomial_ci95(uint32_t k, uint32_t n) noexcept {
  ConfidenceInterval95 ci{};
  ci.count = k;
  ci.total = n;
  if (n == 0U) {
    return ci;
  }
  const double kd = static_cast<double>(k);
  const double nd = static_cast<double>(n);
  const double p = kd / nd;
  ci.rate = p;

  // 95% Wilson score interval
  constexpr double kZ = 1.959963984540054;
  constexpr double kZ2 = kZ * kZ;
  const double denom = 1.0 + kZ2 / nd;
  const double center = (p + kZ2 / (2.0 * nd)) / denom;
  const double half = (kZ / denom) * std::sqrt((p * (1.0 - p)) / nd + kZ2 / (4.0 * nd * nd));
  ci.wilson_lo = std::clamp(center - half, 0.0, 1.0);
  ci.wilson_hi = std::clamp(center + half, 0.0, 1.0);

  // Exact 95% two-sided Clopper-Pearson interval (alpha = 0.05) + one-sided 95% upper bound
  if (k == 0U) {
    ci.clopper_pearson_lo = 0.0;
    ci.clopper_pearson_hi = 1.0 - std::pow(0.025, 1.0 / nd);
    ci.clopper_pearson_one_sided_hi = 1.0 - std::pow(0.05, 1.0 / nd);
  } else if (k == n) {
    ci.clopper_pearson_lo = std::pow(0.025, 1.0 / nd);
    ci.clopper_pearson_hi = 1.0;
    ci.clopper_pearson_one_sided_hi = 1.0;
  } else {
    ci.clopper_pearson_lo = inv_reg_beta(kd, nd - kd + 1.0, 0.025);
    ci.clopper_pearson_hi = inv_reg_beta(kd + 1.0, nd - kd, 0.975);
    ci.clopper_pearson_one_sided_hi = inv_reg_beta(kd + 1.0, nd - kd, 0.95);
  }
  return ci;
}

FlightOutcome run_single_flight(const FlightSampleConfig& cfg) noexcept {
  if (cfg.airframe == AirframeKind::kCoaxialX8_240) {
    return run_x8_flight(cfg);
  }
  return run_lc_flight(cfg);
}

MonteCarloCampaignReport run_monte_carlo_campaign(const MonteCarloCampaignConfig& cfg) {
  const auto t_wall0 = std::chrono::steady_clock::now();
  MonteCarloCampaignReport rep{};
  rep.timestamp_utc = utc_now_iso8601();
  rep.master_seed = cfg.master_seed;

  std::mt19937_64 master_rng(cfg.master_seed);

  // --------------------------------------------------------------------------
  // Campaign 1: Build Stratified Latin-Hypercube 1,000-Flight Sample Table
  // --------------------------------------------------------------------------
  std::vector<FlightSampleConfig> op_samples;
  const uint32_t n_stratum = cfg.flights_per_regime_per_airframe;
  op_samples.reserve(static_cast<std::size_t>(n_stratum) * kNumAirframes * kNumRegimes);

  uint32_t next_id = 0U;
  for (std::size_t af_i = 0U; af_i < kNumAirframes; ++af_i) {
    const auto af = static_cast<AirframeKind>(af_i);
    for (std::size_t rg_i = 0U; rg_i < kNumRegimes; ++rg_i) {
      const auto rg = static_cast<FaultRegimeKind>(rg_i);

      const auto mass_v = lhs_draw(master_rng, n_stratum, 216.0, 264.0);
      const auto cgx_v = lhs_draw(master_rng, n_stratum, -0.05, +0.05);
      const auto cgy_v = lhs_draw(master_rng, n_stratum, -0.05, +0.05);
      const double w_lo = (rg == FaultRegimeKind::kCompoundLowSocWindFault)
                              ? 8.0
                              : ((rg == FaultRegimeKind::kSameArmDualKillCel) ? 0.0 : 2.0);
      const double w_hi = (rg == FaultRegimeKind::kSameArmDualKillCel) ? 15.0 : 14.0;
      const auto wind_v = lhs_draw(master_rng, n_stratum, w_lo, w_hi);
      const auto wdir_v = lhs_draw(master_rng, n_stratum, 0.0, kTwoPi);
      const auto agl_v = lhs_draw(master_rng, n_stratum, 35.0, 90.0);
      const double soc_lo = (rg == FaultRegimeKind::kCompoundLowSocWindFault) ? 0.24 : 0.55;
      const double soc_hi = (rg == FaultRegimeKind::kCompoundLowSocWindFault) ? 0.32 : 0.85;
      const auto soc_v = lhs_draw(master_rng, n_stratum, soc_lo, soc_hi);
      const double tf_lo = (rg == FaultRegimeKind::kGnssSlowRampSpoof) ? 2.5 : 2.0;
      const double tf_hi = (rg == FaultRegimeKind::kGnssSlowRampSpoof)
                               ? 6.0
                               : ((rg == FaultRegimeKind::kCompoundLowSocWindFault) ? 10.0 : 12.0);
      const auto tf_v = lhs_draw(master_rng, n_stratum, tf_lo, tf_hi);
      const auto spoof_v = lhs_draw(master_rng, n_stratum, 1.0, 1.8);
      const double east_lo = (rg == FaultRegimeKind::kGnssSlowRampSpoof) ? -190.0 : -300.0;
      const double east_hi = (rg == FaultRegimeKind::kGnssSlowRampSpoof) ? -110.0 : -140.0;
      const auto east_v = lhs_draw(master_rng, n_stratum, east_lo, east_hi);
      const auto north_v = lhs_draw(master_rng, n_stratum, -60.0, +60.0);

      for (uint32_t k = 0U; k < n_stratum; ++k) {
        FlightSampleConfig sc{};
        sc.flight_id = next_id++;
        sc.seed = master_rng();
        sc.airframe = af;
        sc.regime = rg;
        sc.mass_kg = mass_v[k];
        sc.cg_x_m = cgx_v[k];
        sc.cg_y_m = cgy_v[k];
        sc.wind_speed_mps = wind_v[k];
        sc.wind_dir_rad = wdir_v[k];
        sc.gust_sigma_scale = 1.0;
        sc.init_agl_m = agl_v[k];
        sc.init_east_m = east_v[k];
        sc.init_north_m = north_v[k];
        sc.init_vx_mps = (rg == FaultRegimeKind::kSameArmDualKillCel) ? 3.0 : 5.0;
        sc.slz_offset_n_m = (rg == FaultRegimeKind::kSameArmDualKillCel) ? 45.0 : 85.0;
        sc.init_soc = soc_v[k];
        sc.fault_time_s = tf_v[k];
        sc.fault_index = static_cast<uint8_t>(master_rng() & 0x07ULL);
        sc.spoof_rate_mps = (rg == FaultRegimeKind::kGnssSlowRampSpoof) ? spoof_v[k] : 0.0;
        sc.imu_gyro_sigma_rads = 0.012;
        sc.alt_noise_sigma_m = 0.22;
        sc.max_flight_s = 75.0;
        op_samples.push_back(sc);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Campaign 2: Build 6 x 6 x S Extreme Failure-Boundary Sweep Sample Table
  // --------------------------------------------------------------------------
  std::vector<FlightSampleConfig> stress_samples;
  stress_samples.reserve(kStressGridMassLevels * kStressGridWindLevels * cfg.stress_seeds_per_cell);
  uint32_t stress_id = 100000U;
  for (std::size_t mi = 0U; mi < kStressGridMassLevels; ++mi) {
    const double mass_kg = core::params::kMassKg * (1.0 + rep.stress_mass_overload_pcts[mi] / 100.0);
    for (std::size_t wi = 0U; wi < kStressGridWindLevels; ++wi) {
      const double wind_mps = rep.stress_wind_gust_mps[wi];
      for (uint32_t s = 0U; s < cfg.stress_seeds_per_cell; ++s) {
        FlightSampleConfig sc{};
        sc.flight_id = stress_id++;
        sc.seed = master_rng();
        sc.airframe = AirframeKind::kCoaxialX8_240;
        sc.regime = FaultRegimeKind::kSameArmDualKillCel;
        sc.mass_kg = mass_kg;
        // Nominal trim CG + hover-terminal entry so the 6x6 sweep cleanly maps Mass Overload x Wind Gust
        sc.cg_x_m = 0.0;
        sc.cg_y_m = 0.0;
        sc.wind_speed_mps = wind_mps;
        // True headwind into front-right Arm-0 (-135 deg wind velocity vector => +45 deg relative airspeed)
        sc.wind_dir_rad = -2.356194 + (static_cast<double>(s) - 2.0) * 0.25;
        sc.gust_sigma_scale = 0.65;
        sc.init_agl_m = 45.0;
        sc.init_east_m = -220.0;
        sc.init_north_m = 0.0;
        sc.init_vx_mps = 0.0;
        sc.slz_offset_n_m = 0.0;
        sc.init_soc = 0.42;  // Mid-mission SoC so 1-RC battery sag bites honestly under overload
        sc.fault_time_s = 0.20;
        sc.fault_index = 0U;  // Canonical Arm-0 same-arm coaxial dual-rotor kill (0x11)
        sc.imu_gyro_sigma_rads = 0.015;
        sc.alt_noise_sigma_m = 0.25;
        sc.max_flight_s = 55.0;
        stress_samples.push_back(sc);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Deterministic Multi-Threaded Execution (Bit-Identical Regardless of Threads)
  // --------------------------------------------------------------------------
  uint32_t hw = std::thread::hardware_concurrency();
  if (hw == 0U) {
    hw = 4U;
  }
  const uint32_t n_threads = (cfg.num_threads == 0U) ? std::min(hw, 12U) : std::max(1U, cfg.num_threads);
  rep.threads_used = n_threads;

  auto run_batch = [&](const std::vector<FlightSampleConfig>& inputs, std::vector<FlightOutcome>& outputs) {
    outputs.resize(inputs.size());
    if (n_threads <= 1U || inputs.size() <= 4U) {
      for (std::size_t i = 0U; i < inputs.size(); ++i) {
        outputs[i] = run_single_flight(inputs[i]);
      }
      return;
    }
    std::atomic<std::size_t> cursor{0U};
    std::vector<std::thread> workers;
    workers.reserve(n_threads);
    for (uint32_t t = 0U; t < n_threads; ++t) {
      workers.emplace_back([&]() {
        while (true) {
          const std::size_t idx = cursor.fetch_add(1U, std::memory_order_relaxed);
          if (idx >= inputs.size()) {
            break;
          }
          outputs[idx] = run_single_flight(inputs[idx]);
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  };

  run_batch(op_samples, rep.operational_outcomes);
  std::vector<FlightOutcome> stress_outcomes;
  run_batch(stress_samples, stress_outcomes);

  // --------------------------------------------------------------------------
  // Aggregate Campaign 1 Statistics
  // --------------------------------------------------------------------------
  rep.operational_flights_total = static_cast<uint32_t>(rep.operational_outcomes.size());
  std::vector<const FlightOutcome*> all_ptrs;
  all_ptrs.reserve(rep.operational_outcomes.size());
  std::array<std::vector<const FlightOutcome*>, kNumAirframes> af_ptrs{};
  std::array<std::vector<const FlightOutcome*>, kNumRegimes> rg_ptrs{};
  std::array<std::array<std::vector<const FlightOutcome*>, kNumRegimes>, kNumAirframes> strata_ptrs{};

  std::vector<double> spoof_ttds;
  for (const auto& f : rep.operational_outcomes) {
    const auto af_i = static_cast<std::size_t>(f.airframe);
    const auto rg_i = static_cast<std::size_t>(f.regime);
    all_ptrs.push_back(&f);
    af_ptrs[af_i].push_back(&f);
    rg_ptrs[rg_i].push_back(&f);
    strata_ptrs[af_i][rg_i].push_back(&f);
    if (f.regime == FaultRegimeKind::kGnssSlowRampSpoof) {
      ++rep.gnss_spoof_summary.spoof_flights;
      if (f.spoof_detected) {
        ++rep.gnss_spoof_summary.detected_count;
        spoof_ttds.push_back(f.spoof_ttd_s);
      }
      if (f.spoof_rollback_ok) {
        ++rep.gnss_spoof_summary.rollback_ok_count;
      }
      rep.gnss_spoof_summary.max_3sigma_m = std::max(rep.gnss_spoof_summary.max_3sigma_m, f.max_3sigma_m);
    }
  }

  rep.overall = compute_aggregate_stats(all_ptrs);
  for (std::size_t a = 0U; a < kNumAirframes; ++a) {
    rep.by_airframe[a] = compute_aggregate_stats(af_ptrs[a]);
  }
  for (std::size_t r = 0U; r < kNumRegimes; ++r) {
    rep.by_regime[r] = compute_aggregate_stats(rg_ptrs[r]);
  }
  for (std::size_t a = 0U; a < kNumAirframes; ++a) {
    for (std::size_t r = 0U; r < kNumRegimes; ++r) {
      rep.by_airframe_regime[a][r] = compute_aggregate_stats(strata_ptrs[a][r]);
    }
  }

  if (!spoof_ttds.empty()) {
    std::sort(spoof_ttds.begin(), spoof_ttds.end());
    double sum_ttd = 0.0;
    for (double v : spoof_ttds) {
      sum_ttd += v;
    }
    rep.gnss_spoof_summary.mean_ttd_s = sum_ttd / static_cast<double>(spoof_ttds.size());
    rep.gnss_spoof_summary.p95_ttd_s = sorted_quantile(spoof_ttds, 0.95);
    rep.gnss_spoof_summary.max_ttd_s = spoof_ttds.back();
  }

  // --------------------------------------------------------------------------
  // Aggregate Campaign 2 (6 x 6 Honest Failure-Boundary Matrix)
  // --------------------------------------------------------------------------
  rep.stress_flights_total = static_cast<uint32_t>(stress_outcomes.size());
  rep.failure_boundary.total_stress_flights = rep.stress_flights_total;
  rep.stress_matrix.reserve(kStressGridMassLevels * kStressGridWindLevels);

  bool found_first_breakdown = false;
  bool found_zero_row = false;
  double max_full_safe_mass = -1.0;

  std::size_t cursor = 0U;
  for (std::size_t mi = 0U; mi < kStressGridMassLevels; ++mi) {
    const double ov_pct = rep.stress_mass_overload_pcts[mi];
    const double mass_kg = core::params::kMassKg * (1.0 + ov_pct / 100.0);
    bool entire_row_safe = true;
    bool entire_row_zero = true;

    for (std::size_t wi = 0U; wi < kStressGridWindLevels; ++wi) {
      StressGridCell cell{};
      cell.mass_idx = static_cast<uint32_t>(mi);
      cell.wind_idx = static_cast<uint32_t>(wi);
      cell.mass_overload_pct = ov_pct;
      cell.mass_kg = mass_kg;
      cell.wind_gust_mps = rep.stress_wind_gust_mps[wi];
      cell.flights = cfg.stress_seeds_per_cell;
      cell.min_voltage_v = 1e9;

      double sum_vz = 0.0;
      double sum_tilt = 0.0;
      std::map<std::string, uint32_t> fail_counts;

      for (uint32_t s = 0U; s < cfg.stress_seeds_per_cell; ++s) {
        const auto& fo = stress_outcomes[cursor++];
        if (fo.safe_recovery) {
          ++cell.safe_recoveries;
          ++rep.failure_boundary.safe_stress_flights;
        } else {
          ++rep.failure_boundary.failed_stress_flights;
          ++fail_counts[fo.failure_mode];
        }
        sum_vz += fo.touchdown_vz_mps;
        cell.max_td_vz_mps = std::max(cell.max_td_vz_mps, fo.touchdown_vz_mps);
        cell.min_voltage_v = std::min(cell.min_voltage_v, fo.min_pack_voltage_v);
        sum_tilt += fo.max_tilt_deg;
      }

      if (cell.flights > 0U) {
        cell.survival_rate = static_cast<double>(cell.safe_recoveries) / static_cast<double>(cell.flights);
        cell.mean_td_vz_mps = sum_vz / static_cast<double>(cell.flights);
        cell.mean_max_tilt_deg = sum_tilt / static_cast<double>(cell.flights);
      }

      if (!fail_counts.empty()) {
        uint32_t best_c = 0U;
        for (const auto& kv : fail_counts) {
          if (kv.second > best_c) {
            best_c = kv.second;
            cell.dominant_failure_mode = kv.first;
          }
        }
      }

      if (cell.safe_recoveries == cell.flights && cell.flights > 0U) {
        ++rep.failure_boundary.green_cells_100pct;
        entire_row_zero = false;
      } else if (cell.safe_recoveries == 0U) {
        ++rep.failure_boundary.red_cells_0pct;
        entire_row_safe = false;
        if (!found_first_breakdown) {
          found_first_breakdown = true;
          rep.failure_boundary.first_breakdown_mass_overload_pct = ov_pct;
          rep.failure_boundary.first_breakdown_wind_mps = cell.wind_gust_mps;
        }
      } else {
        ++rep.failure_boundary.transition_cells_partial;
        entire_row_safe = false;
        entire_row_zero = false;
        if (!found_first_breakdown) {
          found_first_breakdown = true;
          rep.failure_boundary.first_breakdown_mass_overload_pct = ov_pct;
          rep.failure_boundary.first_breakdown_wind_mps = cell.wind_gust_mps;
        }
      }

      rep.stress_matrix.push_back(cell);
    }

    if (entire_row_safe) {
      max_full_safe_mass = ov_pct;
    }
    if (entire_row_zero && !found_zero_row) {
      found_zero_row = true;
      rep.failure_boundary.zero_survival_mass_overload_pct = ov_pct;
    }
  }
  rep.failure_boundary.max_100pct_safe_mass_overload_pct = std::max(0.0, max_full_safe_mass);

  rep.total_wall_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_wall0).count();

  // Seal canonical JSON with FIPS 180-4 SHA-256 (computed with empty report_sha256 field)
  rep.report_sha256.clear();
  const std::string body = serialize_mc_report_json(rep);
  rep.report_sha256 = forensics::sha256_hex_string(body);
  return rep;
}

std::string serialize_mc_report_json(const MonteCarloCampaignReport& rep) {
  std::ostringstream os;
  os << std::fixed << std::setprecision(4);
  os << "{\n";
  os << "  \"schema_version\": \"" << json_escape(rep.schema_version) << "\",\n";
  os << "  \"timestamp_utc\": \"" << json_escape(rep.timestamp_utc) << "\",\n";
  os << "  \"master_seed\": " << rep.master_seed << ",\n";
  os << "  \"operational_flights_total\": " << rep.operational_flights_total << ",\n";
  os << "  \"stress_flights_total\": " << rep.stress_flights_total << ",\n";
  os << "  \"threads_used\": " << rep.threads_used << ",\n";
  os << "  \"total_wall_ms\": " << safe_finite(rep.total_wall_ms) << ",\n";
  os << "  \"safety_thresholds\": {\n";
  os << "    \"max_safe_touchdown_vz_mps\": " << kMaxSafeTouchdownVzMps << ",\n";
  os << "    \"max_safe_tilt_deg\": " << kMaxSafeTiltDeg << ",\n";
  os << "    \"min_safe_pack_voltage_v\": " << kMinSafePackVoltageV << ",\n";
  os << "    \"hospital_keepout_radius_m\": " << wind_acoustics::kKartalHospitalHelipadRadiusHM << ",\n";
  os << "    \"ltfj_western_wall_lon_deg\": 29.1895,\n";
  os << "    \"ltbx_arp_lat_deg\": 40.990556,\n";
  os << "    \"ltbx_arp_lon_deg\": 29.216111\n";
  os << "  },\n";

  os << "  \"overall\": ";
  write_agg_json(os, rep.overall, "  ");
  os << ",\n";

  os << "  \"by_airframe\": {\n";
  for (std::size_t a = 0U; a < kNumAirframes; ++a) {
    os << "    \"" << airframe_name(static_cast<AirframeKind>(a)) << "\": ";
    write_agg_json(os, rep.by_airframe[a], "    ");
    os << (a + 1U < kNumAirframes ? "," : "") << "\n";
  }
  os << "  },\n";

  os << "  \"by_regime\": {\n";
  for (std::size_t r = 0U; r < kNumRegimes; ++r) {
    os << "    \"" << regime_name(static_cast<FaultRegimeKind>(r)) << "\": ";
    write_agg_json(os, rep.by_regime[r], "    ");
    os << (r + 1U < kNumRegimes ? "," : "") << "\n";
  }
  os << "  },\n";

  os << "  \"by_airframe_regime\": {\n";
  for (std::size_t a = 0U; a < kNumAirframes; ++a) {
    os << "    \"" << airframe_name(static_cast<AirframeKind>(a)) << "\": {\n";
    for (std::size_t r = 0U; r < kNumRegimes; ++r) {
      os << "      \"" << regime_name(static_cast<FaultRegimeKind>(r)) << "\": ";
      write_agg_json(os, rep.by_airframe_regime[a][r], "      ");
      os << (r + 1U < kNumRegimes ? "," : "") << "\n";
    }
    os << "    }" << (a + 1U < kNumAirframes ? "," : "") << "\n";
  }
  os << "  },\n";

  os << "  \"gnss_spoof_summary\": {\n";
  os << "    \"spoof_flights\": " << rep.gnss_spoof_summary.spoof_flights << ",\n";
  os << "    \"detected_count\": " << rep.gnss_spoof_summary.detected_count << ",\n";
  os << "    \"rollback_ok_count\": " << rep.gnss_spoof_summary.rollback_ok_count << ",\n";
  os << "    \"mean_ttd_s\": " << safe_finite(rep.gnss_spoof_summary.mean_ttd_s) << ",\n";
  os << "    \"p95_ttd_s\": " << safe_finite(rep.gnss_spoof_summary.p95_ttd_s) << ",\n";
  os << "    \"max_ttd_s\": " << safe_finite(rep.gnss_spoof_summary.max_ttd_s) << ",\n";
  os << "    \"max_3sigma_m\": " << safe_finite(rep.gnss_spoof_summary.max_3sigma_m) << "\n";
  os << "  },\n";

  os << "  \"failure_boundary_sweep\": {\n";
  os << "    \"stress_airframe\": \"" << json_escape(rep.failure_boundary.stress_airframe) << "\",\n";
  os << "    \"stress_fault_mode\": \"" << json_escape(rep.failure_boundary.stress_fault_mode) << "\",\n";
  os << "    \"mass_overload_pcts\": [";
  for (std::size_t i = 0U; i < kStressGridMassLevels; ++i) {
    os << safe_finite(rep.stress_mass_overload_pcts[i]) << (i + 1U < kStressGridMassLevels ? ", " : "");
  }
  os << "],\n";
  os << "    \"wind_gust_mps\": [";
  for (std::size_t i = 0U; i < kStressGridWindLevels; ++i) {
    os << safe_finite(rep.stress_wind_gust_mps[i]) << (i + 1U < kStressGridWindLevels ? ", " : "");
  }
  os << "],\n";
  os << "    \"summary\": {\n";
  os << "      \"total_stress_flights\": " << rep.failure_boundary.total_stress_flights << ",\n";
  os << "      \"safe_stress_flights\": " << rep.failure_boundary.safe_stress_flights << ",\n";
  os << "      \"failed_stress_flights\": " << rep.failure_boundary.failed_stress_flights << ",\n";
  os << "      \"green_cells_100pct\": " << rep.failure_boundary.green_cells_100pct << ",\n";
  os << "      \"transition_cells_partial\": " << rep.failure_boundary.transition_cells_partial << ",\n";
  os << "      \"red_cells_0pct\": " << rep.failure_boundary.red_cells_0pct << ",\n";
  os << "      \"max_100pct_safe_mass_overload_pct\": " << safe_finite(rep.failure_boundary.max_100pct_safe_mass_overload_pct) << ",\n";
  os << "      \"first_breakdown_mass_overload_pct\": " << safe_finite(rep.failure_boundary.first_breakdown_mass_overload_pct) << ",\n";
  os << "      \"first_breakdown_wind_mps\": " << safe_finite(rep.failure_boundary.first_breakdown_wind_mps) << ",\n";
  os << "      \"zero_survival_mass_overload_pct\": " << safe_finite(rep.failure_boundary.zero_survival_mass_overload_pct) << "\n";
  os << "    },\n";
  os << "    \"grid_cells\": [\n";
  for (std::size_t i = 0U; i < rep.stress_matrix.size(); ++i) {
    const auto& c = rep.stress_matrix[i];
    os << "      {\"mass_idx\": " << c.mass_idx
       << ", \"wind_idx\": " << c.wind_idx
       << ", \"mass_overload_pct\": " << safe_finite(c.mass_overload_pct)
       << ", \"mass_kg\": " << safe_finite(c.mass_kg)
       << ", \"wind_gust_mps\": " << safe_finite(c.wind_gust_mps)
       << ", \"flights\": " << c.flights
       << ", \"safe_recoveries\": " << c.safe_recoveries
       << ", \"survival_rate\": " << safe_finite(c.survival_rate)
       << ", \"mean_td_vz_mps\": " << safe_finite(c.mean_td_vz_mps)
       << ", \"max_td_vz_mps\": " << safe_finite(c.max_td_vz_mps)
       << ", \"min_voltage_v\": " << safe_finite(c.min_voltage_v)
       << ", \"mean_max_tilt_deg\": " << safe_finite(c.mean_max_tilt_deg)
       << ", \"dominant_failure_mode\": \"" << json_escape(c.dominant_failure_mode) << "\"}"
       << (i + 1U < rep.stress_matrix.size() ? "," : "") << "\n";
  }
  os << "    ]\n";
  os << "  },\n";

  // Compact per-flight touchdown table for 100% reproducible dispersion plotting & verification
  // Tuple schema: [airframe_id, regime_id, dx_m, dy_m, td_vz_mps, min_v_term_v, max_tilt_deg, safe_int, breach_int]
  os << "  \"touchdown_samples_schema\": [\"airframe_id\", \"regime_id\", \"dx_m\", \"dy_m\", \"td_vz_mps\", \"min_v_term_v\", \"max_tilt_deg\", \"safe\", \"breach\"],\n";
  os << "  \"touchdown_samples\": [\n";
  os << std::setprecision(2);
  for (std::size_t i = 0U; i < rep.operational_outcomes.size(); ++i) {
    const auto& f = rep.operational_outcomes[i];
    os << "    [" << static_cast<unsigned>(f.airframe)
       << "," << static_cast<unsigned>(f.regime)
       << "," << safe_finite(f.touchdown_dx_m)
       << "," << safe_finite(f.touchdown_dy_m)
       << "," << safe_finite(f.touchdown_vz_mps)
       << "," << safe_finite(f.min_pack_voltage_v)
       << "," << safe_finite(f.max_tilt_deg)
       << "," << (f.safe_recovery ? 1 : 0)
       << "," << (f.airspace_breach ? 1 : 0) << "]"
       << (i + 1U < rep.operational_outcomes.size() ? "," : "") << "\n";
  }
  os << "  ],\n";
  os << "  \"report_sha256\": \"" << json_escape(rep.report_sha256) << "\"\n";
  os << "}\n";
  return os.str();
}

bool write_mc_report_json(
    const MonteCarloCampaignReport& rep,
    const std::filesystem::path& out_path) {
  std::error_code ec;
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path(), ec);
  }
  std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    return false;
  }
  const std::string txt = serialize_mc_report_json(rep);
  ofs.write(txt.data(), static_cast<std::streamsize>(txt.size()));
  return ofs.good();
}

bool verify_mc_report_json_sha256(
    const std::filesystem::path& json_path,
    std::string* err_out) {
  std::ifstream ifs(json_path, std::ios::binary);
  if (!ifs.is_open()) {
    if (err_out) {
      *err_out = "Unable to open report JSON";
    }
    return false;
  }
  const std::string raw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  if (raw.find('\r') != std::string::npos) {
    if (err_out) {
      *err_out = "CR byte detected (must be pure LF)";
    }
    return false;
  }
  const std::string key = "\"report_sha256\": \"";
  const auto pos = raw.rfind(key);
  if (pos == std::string::npos) {
    if (err_out) {
      *err_out = "Missing report_sha256 field";
    }
    return false;
  }
  const auto val_start = pos + key.size();
  const auto val_end = raw.find('"', val_start);
  if (val_end == std::string::npos || (val_end - val_start) != 64U) {
    if (err_out) {
      *err_out = "Malformed report_sha256 hex digest";
    }
    return false;
  }
  const std::string recorded = raw.substr(val_start, 64U);
  std::string body = raw;
  body.erase(val_start, 64U);
  const std::string computed = forensics::sha256_hex_string(body);
  if (computed != recorded) {
    if (err_out) {
      *err_out = "SHA-256 mismatch (recorded=" + recorded + ", computed=" + computed + ")";
    }
    return false;
  }
  return true;
}

}  // namespace vertirta::benchmarks
