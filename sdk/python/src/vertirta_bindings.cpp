// SPDX-License-Identifier: Apache-2.0
// VertiRTA v3.0.0 — nanobind C++20 Python SDK bindings wrapping frozen vertirta::flight_core.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/math_utils.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta/core/vrta_recorder.hpp"

namespace nb = nanobind;

namespace {

using vertirta::core::CoaxEffectiveness;
using vertirta::core::CoaxialPlant6Dof;
using vertirta::core::FlightRegime;
using vertirta::core::ImuSampleFrd;
using vertirta::core::IndiInput;
using vertirta::core::IndiWlsController;
using vertirta::core::MotorFdiObserver;
using vertirta::core::PlantState;
using vertirta::core::TmrVoter;
using vertirta::core::Vec3Enu;
using vertirta::core::Vec3Ned;
using vertirta::core::VrtaFileHeader;
using vertirta::core::VrtaFrame100Hz;
using vertirta::core::clamp_d;
using vertirta::core::crc32c;
using vertirta::core::enu_to_ned;
using vertirta::core::finalize_vrta_frame;
using vertirta::core::finalize_vrta_header;
using vertirta::core::make_default_vrta_header;
using vertirta::core::ned_to_enu;
using vertirta::core::verify_vrta_frame;
namespace params = vertirta::core::params;

constexpr std::size_t kObsDim = 18U;
constexpr std::size_t kActDim = 4U;

constexpr double kKartalLatDeg = 40.8860;
constexpr double kKartalLonDeg = 29.1880;
constexpr double kKartalAmslM = 12.0;
constexpr double kKartalEgm2008NM = 36.7;
constexpr double kMetersPerDegLat = 111052.0;
constexpr double kMetersPerDegLon = 84255.0;
constexpr double kLtfjArpLatDeg = 40.898333;
constexpr double kLtfjArpLonDeg = 29.309167;
constexpr double kLtfjRadiusM = 9260.0;
constexpr double kLtfjJ2MaxLonDeg = 29.1895;
constexpr double kLtbxArpLatDeg = 40.990556;
constexpr double kLtbxArpLonDeg = 29.216111;
constexpr double kLtbxRadiusM = 9260.0;

constexpr uint32_t kFaultNone = 0U;
constexpr uint32_t kFaultSingleMotor = 1U;
constexpr uint32_t kFaultSameArmDual = 2U;
constexpr uint32_t kFaultImuBias = 3U;
constexpr uint32_t kFaultWindGust = 4U;

[[nodiscard]] double signed_margin_ltfj_m(double lat_deg, double lon_deg) noexcept {
  const double d_e = (lon_deg - kLtfjArpLonDeg) * kMetersPerDegLon;
  const double d_n = (lat_deg - kLtfjArpLatDeg) * kMetersPerDegLat;
  const double arc_margin = std::sqrt(d_e * d_e + d_n * d_n) - kLtfjRadiusM;
  const double j2_wall_margin = (kLtfjJ2MaxLonDeg - lon_deg) * kMetersPerDegLon;
  return (arc_margin < j2_wall_margin) ? arc_margin : j2_wall_margin;
}

[[nodiscard]] double signed_margin_ltbx_m(double lat_deg, double lon_deg) noexcept {
  const double d_e = (lon_deg - kLtbxArpLonDeg) * kMetersPerDegLon;
  const double d_n = (lat_deg - kLtbxArpLatDeg) * kMetersPerDegLat;
  return std::sqrt(d_e * d_e + d_n * d_n) - kLtbxRadiusM;
}

struct StepOutcome {
  std::array<float, kObsDim> obs{};
  float reward{0.0f};
  bool terminated{false};
  bool cel_latched{false};
  uint8_t active_motor_mask{0xFFU};
};

class NativeCoaxialSim {
 public:
  explicit NativeCoaxialSim(bool allow_active_spin = true) noexcept
      : indi_(allow_active_spin) {
    reset({0.0, 0.0, 35.0}, {0.0, 0.0, 0.0}, 0.80f, 0U, {0.0, 0.0, 0.0});
  }

  std::array<float, kObsDim> reset(
      const std::array<double, 3>& pos_enu,
      const std::array<double, 3>& vel_enu,
      float initial_soc,
      uint32_t fault_mask,
      const std::array<double, 3>& wind_enu) noexcept {
    const Vec3Enu p_enu{pos_enu[0], pos_enu[1], pos_enu[2]};
    const Vec3Enu v_enu{vel_enu[0], vel_enu[1], vel_enu[2]};
    plant_.reset(enu_to_ned(p_enu), enu_to_ned(v_enu), initial_soc, params::kOmegaHover);
    indi_.reset(params::kOmegaHover);
    tmr_.reset();
    fdi_.reset();
    tick_400hz_ = 0ULL;
    seq_100hz_ = 0ULL;
    injected_fault_mask_ = static_cast<uint8_t>(fault_mask & 0xFFU);
    detected_fault_mask_ = 0x00U;
    imu_fault_active_ = false;
    imu_fault_idx_ = 0U;
    cel_latched_ = false;
    wind_ned_ = enu_to_ned(Vec3Enu{wind_enu[0], wind_enu[1], wind_enu[2]});
    for (float& cmd : omega_cmd_) {
      cmd = static_cast<float>(params::kOmegaHover);
    }
    write_frame({0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0U, 0.0f);
    return build_obs();
  }

  void inject_fault(uint32_t fault_kind, uint32_t target_idx) noexcept {
    switch (fault_kind) {
      case kFaultNone:
        imu_fault_active_ = false;
        wind_ned_ = {0.0, 0.0, 0.0};
        break;
      case kFaultSingleMotor: {
        const uint32_t rotor = target_idx & 7U;
        injected_fault_mask_ =
            static_cast<uint8_t>(injected_fault_mask_ | static_cast<uint8_t>(1U << rotor));
        break;
      }
      case kFaultSameArmDual: {
        const uint32_t arm = target_idx & 3U;
        const uint8_t pair = static_cast<uint8_t>((1U << arm) | (1U << (arm + 4U)));
        injected_fault_mask_ = static_cast<uint8_t>(injected_fault_mask_ | pair);
        break;
      }
      case kFaultImuBias:
        imu_fault_active_ = true;
        imu_fault_idx_ = static_cast<uint8_t>(target_idx % 3U);
        break;
      case kFaultWindGust:
        wind_ned_ = {8.5, -8.5, 0.0};
        break;
      default:
        break;
    }
  }

  StepOutcome step_400hz(
      uint32_t n_ticks,
      double a_nom_e,
      double a_nom_n,
      double a_nom_u,
      double yaw_rate_des) noexcept {
    const uint32_t ticks = std::clamp(n_ticks, 1U, 4000U);
    const Vec3Enu a_nom_enu{
        clamp_d(a_nom_e, -4.0, 4.0),
        clamp_d(a_nom_n, -4.0, 4.0),
        clamp_d(a_nom_u, -3.0, 3.0),
    };
    const double yaw_cmd = clamp_d(yaw_rate_des, -1.0, 1.0);

    uint8_t last_tmr_state = 0U;
    float last_chi2 = 0.0f;
    Vec3Enu last_a_safe_enu = a_nom_enu;

    for (uint32_t s = 0U; s < ticks; ++s) {
      const PlantState ps = plant_.state();
      const uint64_t t_us = tick_400hz_ * 2500ULL;

      ImuSampleFrd imus[3]{};
      for (int k = 0; k < 3; ++k) {
        imus[k].gyro_rad_s = ps.omega_frd;
        imus[k].accel_m_s2 = {0.0, 0.0, -params::kGravity};
        imus[k].t_us = t_us;
        imus[k].valid = true;
      }
      if (imu_fault_active_) {
        imus[imu_fault_idx_].gyro_rad_s.x += 1.25;
        imus[imu_fault_idx_].gyro_rad_s.y -= 0.95;
      }
      const auto vote = tmr_.step(imus, indi_.filtered_gyro_frd(), tick_400hz_ > 4U);
      last_tmr_state = static_cast<uint8_t>(vote.state);
      last_chi2 = vote.max_chi2_residual;

      float omega_hat[8]{};
      float omega_esc[8]{};
      bool esc_valid[8]{};
      indi_.get_omega_hat(omega_hat);
      for (int i = 0; i < 8; ++i) {
        omega_esc[i] = static_cast<float>(ps.rotor_omega[i]);
        esc_valid[i] = true;
      }
      const float v_horiz = static_cast<float>(
          std::sqrt(ps.vel_ned.x * ps.vel_ned.x + ps.vel_ned.y * ps.vel_ned.y));
      detected_fault_mask_ = fdi_.step_400hz(
          indi_.filtered_gyro_frd(),
          indi_.filtered_dgyro_frd(),
          static_cast<float>(-params::kGravity),
          omega_hat,
          omega_esc,
          esc_valid,
          v_horiz,
          ps.battery.v_term_v,
          detected_fault_mask_);

      if (CoaxEffectiveness::classify_regime(detected_fault_mask_) ==
          FlightRegime::SAME_ARM_ACTIVE_SPIN) {
        cel_latched_ = true;
      }

      const Vec3Enu pos_enu = ned_to_enu(ps.pos_ned);
      const Vec3Enu vel_enu = ned_to_enu(ps.vel_ned);
      Vec3Enu a_safe_enu = a_nom_enu;
      if (cel_latched_) {
        const double target_vz = (pos_enu.z > 10.0) ? -2.0 : ((pos_enu.z > 5.0) ? -1.2 : -0.6);
        a_safe_enu.x = clamp_d(-0.6 * vel_enu.x, -1.5, 1.5);
        a_safe_enu.y = clamp_d(-0.6 * vel_enu.y, -1.5, 1.5);
        a_safe_enu.z = clamp_d(1.8 * (target_vz - vel_enu.z), -2.0, 2.0);
      } else {
        const double z_err = 35.0 - pos_enu.z;
        a_safe_enu.z = clamp_d(a_nom_enu.z + 0.8 * z_err - 1.4 * vel_enu.z, -2.5, 2.5);
        if (pos_enu.x > 80.0) {
          a_safe_enu.x = clamp_d(-1.5 - 0.5 * vel_enu.x, -4.0, 0.0);
        }
      }
      last_a_safe_enu = a_safe_enu;

      IndiInput in{};
      in.q_frd_ned = ps.q_frd_ned;
      in.gyro_frd = vote.gyro_voted;
      in.accel_des_ned = enu_to_ned(a_safe_enu);
      in.yaw_rate_des = yaw_cmd;
      in.v_horiz_mps = v_horiz;
      in.v_term_v = ps.battery.v_term_v;
      in.fault_mask = detected_fault_mask_;
      in.esc_omega_rad_s = nullptr;
      in.esc_valid = nullptr;

      (void)indi_.step_400hz(in, omega_cmd_);
      (void)plant_.step_400hz(omega_cmd_, injected_fault_mask_, wind_ned_);

      ++tick_400hz_;
      if ((tick_400hz_ % 4ULL) == 0ULL) {
        ++seq_100hz_;
      }
    }

    write_frame(a_nom_enu, last_a_safe_enu, last_tmr_state, last_chi2);

    StepOutcome out{};
    out.obs = build_obs();
    const PlantState& ps = plant_.state();
    const Vec3Enu pos_enu = ned_to_enu(ps.pos_ned);
    const Vec3Enu vel_enu = ned_to_enu(ps.vel_ned);
    const double agl_m = std::max(0.0, pos_enu.z);
    const double speed = std::sqrt(vel_enu.x * vel_enu.x + vel_enu.y * vel_enu.y + vel_enu.z * vel_enu.z);
    const bool hard_impact = ps.on_ground && (std::abs(ps.touchdown_vz_mps) > 2.5);
    const bool out_of_bounds =
        (std::abs(pos_enu.x) > 500.0) || (std::abs(pos_enu.y) > 500.0) || (agl_m > 155.0);
    const bool battery_empty = (ps.battery.soc <= 0.05f);
    out.terminated = hard_impact || out_of_bounds || battery_empty || (ps.on_ground && cel_latched_);
    out.cel_latched = cel_latched_;
    out.active_motor_mask = latest_frame_.active_motor_mask;

    double reward = 1.0 - 0.01 * std::abs(agl_m - 35.0) - 0.02 * speed;
    if (hard_impact || out_of_bounds || battery_empty) {
      reward -= 25.0;
    } else if (ps.on_ground && cel_latched_) {
      reward += 10.0;
    }
    out.reward = static_cast<float>(reward);
    return out;
  }

  [[nodiscard]] std::array<float, kObsDim> build_obs() const noexcept {
    const PlantState& ps = plant_.state();
    const Vec3Enu pos_enu = ned_to_enu(ps.pos_ned);
    const Vec3Enu vel_enu = ned_to_enu(ps.vel_ned);
    int healthy = 0;
    for (int i = 0; i < 8; ++i) {
      if ((latest_frame_.active_motor_mask & static_cast<uint8_t>(1U << i)) != 0U) {
        ++healthy;
      }
    }
    return {
        static_cast<float>(pos_enu.x),
        static_cast<float>(pos_enu.y),
        static_cast<float>(pos_enu.z),
        static_cast<float>(vel_enu.x),
        static_cast<float>(vel_enu.y),
        static_cast<float>(vel_enu.z),
        static_cast<float>(ps.q_frd_ned.w),
        static_cast<float>(ps.q_frd_ned.x),
        static_cast<float>(ps.q_frd_ned.y),
        static_cast<float>(ps.q_frd_ned.z),
        static_cast<float>(ps.omega_frd.x),
        static_cast<float>(ps.omega_frd.y),
        static_cast<float>(ps.omega_frd.z),
        ps.battery.v_term_v,
        ps.battery.soc * 100.0f,
        static_cast<float>(std::max(0.0, pos_enu.z)),
        static_cast<float>(healthy) / 8.0f,
        cel_latched_ ? 1.0f : 0.0f,
    };
  }

  [[nodiscard]] const VrtaFrame100Hz& latest_frame() const noexcept { return latest_frame_; }

 private:
  void write_frame(
      const Vec3Enu& a_nom_enu,
      const Vec3Enu& a_safe_enu,
      uint8_t tmr_state_byte,
      float chi2_res) noexcept {
    const PlantState& ps = plant_.state();
    const Vec3Enu pos_enu = ned_to_enu(ps.pos_ned);
    const Vec3Enu vel_enu = ned_to_enu(ps.vel_ned);
    const double lat_deg = kKartalLatDeg + pos_enu.y / kMetersPerDegLat;
    const double lon_deg = kKartalLonDeg + pos_enu.x / kMetersPerDegLon;
    const double agl_m = std::max(0.0, pos_enu.z);
    const double ltfj_m = signed_margin_ltfj_m(lat_deg, lon_deg);
    const double ltbx_m = signed_margin_ltbx_m(lat_deg, lon_deg);
    const uint8_t eff_mask = static_cast<uint8_t>(detected_fault_mask_ | injected_fault_mask_);
    const FlightRegime regime = cel_latched_
                                    ? FlightRegime::SAME_ARM_ACTIVE_SPIN
                                    : CoaxEffectiveness::classify_regime(eff_mask);

    VrtaFrame100Hz frame{};
    frame.seq = seq_100hz_;
    frame.t_sim_us = tick_400hz_ * 2500ULL;
    frame.pos_enu[0] = pos_enu.x;
    frame.pos_enu[1] = pos_enu.y;
    frame.pos_enu[2] = pos_enu.z;
    frame.vel_enu[0] = static_cast<float>(vel_enu.x);
    frame.vel_enu[1] = static_cast<float>(vel_enu.y);
    frame.vel_enu[2] = static_cast<float>(vel_enu.z);
    frame.q_frd_ned[0] = static_cast<float>(ps.q_frd_ned.w);
    frame.q_frd_ned[1] = static_cast<float>(ps.q_frd_ned.x);
    frame.q_frd_ned[2] = static_cast<float>(ps.q_frd_ned.y);
    frame.q_frd_ned[3] = static_cast<float>(ps.q_frd_ned.z);
    frame.omega_frd[0] = static_cast<float>(ps.omega_frd.x);
    frame.omega_frd[1] = static_cast<float>(ps.omega_frd.y);
    frame.omega_frd[2] = static_cast<float>(ps.omega_frd.z);
    frame.a_nom_enu[0] = static_cast<float>(a_nom_enu.x);
    frame.a_nom_enu[1] = static_cast<float>(a_nom_enu.y);
    frame.a_nom_enu[2] = static_cast<float>(a_nom_enu.z);
    frame.a_safe_enu[0] = static_cast<float>(a_safe_enu.x);
    frame.a_safe_enu[1] = static_cast<float>(a_safe_enu.y);
    frame.a_safe_enu[2] = static_cast<float>(a_safe_enu.z);
    for (int i = 0; i < 8; ++i) {
      frame.rotor_rpm[i] = static_cast<float>(ps.rotor_omega[i] * (60.0 / (2.0 * vertirta::core::kPi)));
    }
    frame.v_term_v = ps.battery.v_term_v;
    frame.soc_pct = ps.battery.soc * 100.0f;
    frame.cbvf_min_v = static_cast<float>(std::min({ltfj_m, ltbx_m, 120.0 - agl_m, agl_m}));
    frame.ltfj_margin_m = static_cast<float>(ltfj_m);
    frame.ltbx_margin_m = static_cast<float>(ltbx_m);
    frame.chi2_residual = chi2_res;
    frame.agl_m = static_cast<float>(agl_m);
    frame.active_motor_mask = static_cast<uint8_t>((~eff_mask) & 0xFFU);
    frame.flight_regime = static_cast<uint8_t>(regime);
    frame.tmr_state = tmr_state_byte;
    frame.cel_latched = cel_latched_ ? 1U : 0U;
    frame.flags = imu_fault_active_ ? 0x01U : 0x00U;
    finalize_vrta_frame(frame);
    latest_frame_ = frame;
  }

  CoaxialPlant6Dof plant_{};
  IndiWlsController indi_{true};
  TmrVoter tmr_{};
  MotorFdiObserver fdi_{};
  uint64_t tick_400hz_{0ULL};
  uint64_t seq_100hz_{0ULL};
  uint8_t injected_fault_mask_{0x00U};
  uint8_t detected_fault_mask_{0x00U};
  bool imu_fault_active_{false};
  uint8_t imu_fault_idx_{0U};
  bool cel_latched_{false};
  Vec3Ned wind_ned_{};
  float omega_cmd_[8]{};
  VrtaFrame100Hz latest_frame_{};
};

class NativeVecSim {
 public:
  explicit NativeVecSim(std::size_t num_envs, std::size_t num_threads = 0U)
      : envs_(num_envs) {
    if (num_envs == 0U) {
      throw std::invalid_argument("num_envs must be >= 1");
    }
    const std::size_t hw = std::max<std::size_t>(1U, std::thread::hardware_concurrency());
    num_threads_ = (num_threads == 0U) ? std::min(num_envs, hw) : std::min(num_envs, num_threads);
  }

  [[nodiscard]] std::size_t num_envs() const noexcept { return envs_.size(); }
  [[nodiscard]] std::size_t num_threads() const noexcept { return num_threads_; }

  std::array<float, kObsDim> reset_one(
      std::size_t idx,
      const std::array<double, 3>& pos_enu,
      const std::array<double, 3>& vel_enu,
      float initial_soc,
      uint32_t fault_mask,
      const std::array<double, 3>& wind_enu) {
    if (idx >= envs_.size()) {
      throw std::out_of_range("env index out of bounds");
    }
    return envs_[idx].reset(pos_enu, vel_enu, initial_soc, fault_mask, wind_enu);
  }

  void inject_fault_one(std::size_t idx, uint32_t fault_kind, uint32_t target_idx) {
    if (idx >= envs_.size()) {
      throw std::out_of_range("env index out of bounds");
    }
    envs_[idx].inject_fault(fault_kind, target_idx);
  }

  void step_batch(
      nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> actions,
      uint32_t n_ticks,
      nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> obs_out,
      nb::ndarray<float, nb::ndim<1>, nb::c_contig, nb::device::cpu> rew_out,
      nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> term_out,
      nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> cel_out) {
    const std::size_t n = envs_.size();
    if (actions.shape(0) != n || actions.shape(1) != kActDim ||
        obs_out.shape(0) != n || obs_out.shape(1) != kObsDim ||
        rew_out.shape(0) != n || term_out.shape(0) != n || cel_out.shape(0) != n) {
      throw std::invalid_argument("step_batch array shape mismatch");
    }

    const float* act_ptr = actions.data();
    float* obs_ptr = obs_out.data();
    float* rew_ptr = rew_out.data();
    uint8_t* term_ptr = term_out.data();
    uint8_t* cel_ptr = cel_out.data();

    auto run_range = [&](std::size_t lo, std::size_t hi) noexcept {
      for (std::size_t i = lo; i < hi; ++i) {
        const float* a = act_ptr + i * kActDim;
        const StepOutcome res = envs_[i].step_400hz(
            n_ticks,
            static_cast<double>(a[0]) * 4.0,
            static_cast<double>(a[1]) * 4.0,
            static_cast<double>(a[2]) * 3.0,
            static_cast<double>(a[3]) * 1.0);
        std::memcpy(obs_ptr + i * kObsDim, res.obs.data(), kObsDim * sizeof(float));
        rew_ptr[i] = res.reward;
        term_ptr[i] = res.terminated ? 1U : 0U;
        cel_ptr[i] = res.cel_latched ? 1U : 0U;
      }
    };

    // Release Python GIL for parallel zero-copy C++20 stepping across envs
    nb::gil_scoped_release release;
    if (num_threads_ <= 1U || n < 2U) {
      run_range(0U, n);
    } else {
      std::vector<std::thread> workers;
      workers.reserve(num_threads_);
      const std::size_t chunk = (n + num_threads_ - 1U) / num_threads_;
      for (std::size_t t = 0U; t < num_threads_; ++t) {
        const std::size_t lo = t * chunk;
        if (lo >= n) {
          break;
        }
        const std::size_t hi = std::min(n, lo + chunk);
        workers.emplace_back(run_range, lo, hi);
      }
      for (std::thread& w : workers) {
        w.join();
      }
    }
  }

 private:
  std::vector<NativeCoaxialSim> envs_;
  std::size_t num_threads_{1U};
};

nb::bytes py_make_vrta_header_bytes(
    double datum_lat_deg,
    double datum_lon_deg,
    double datum_amsl_m,
    double egm2008_n_m) {
  VrtaFileHeader hdr = make_default_vrta_header();
  hdr.datum_lat_deg = datum_lat_deg;
  hdr.datum_lon_deg = datum_lon_deg;
  hdr.datum_amsl_m = datum_amsl_m;
  hdr.egm2008_n_m = egm2008_n_m;
  finalize_vrta_header(hdr);
  return nb::bytes(reinterpret_cast<const char*>(&hdr), sizeof(VrtaFileHeader));
}

bool py_verify_vrta_frame_bytes(const nb::bytes& data) {
  if (data.size() != sizeof(VrtaFrame100Hz)) {
    return false;
  }
  VrtaFrame100Hz frame{};
  std::memcpy(&frame, data.c_str(), sizeof(VrtaFrame100Hz));
  return verify_vrta_frame(frame);
}

uint32_t py_crc32c_bytes(const nb::bytes& data) {
  return crc32c(data.c_str(), data.size());
}

}  // namespace

NB_MODULE(_vertirta_native, m) {
  m.doc() = "VertiRTA C++17/20 nanobind native flight_core bindings (zero-copy GIL-free batch sim)";

  m.attr("OBS_DIM") = kObsDim;
  m.attr("ACT_DIM") = kActDim;
  m.attr("VRTA_HEADER_SIZE") = sizeof(VrtaFileHeader);
  m.attr("VRTA_FRAME_SIZE") = sizeof(VrtaFrame100Hz);
  m.attr("FAULT_NONE") = kFaultNone;
  m.attr("FAULT_SINGLE_MOTOR") = kFaultSingleMotor;
  m.attr("FAULT_SAME_ARM_DUAL") = kFaultSameArmDual;
  m.attr("FAULT_IMU_BIAS") = kFaultImuBias;
  m.attr("FAULT_WIND_GUST") = kFaultWindGust;
  m.attr("KARTAL_DATUM") = std::make_tuple(kKartalLatDeg, kKartalLonDeg, kKartalAmslM, kKartalEgm2008NM);

  m.def("make_vrta_header_bytes", &py_make_vrta_header_bytes,
        nb::arg("datum_lat_deg") = kKartalLatDeg,
        nb::arg("datum_lon_deg") = kKartalLonDeg,
        nb::arg("datum_amsl_m") = kKartalAmslM,
        nb::arg("egm2008_n_m") = kKartalEgm2008NM);
  m.def("verify_vrta_frame_bytes", &py_verify_vrta_frame_bytes, nb::arg("frame_bytes"));
  m.def("crc32c_bytes", &py_crc32c_bytes, nb::arg("data"));

  nb::class_<NativeCoaxialSim>(m, "NativeCoaxialSim")
      .def(nb::init<bool>(), nb::arg("allow_active_spin") = true)
      .def("reset", &NativeCoaxialSim::reset,
           nb::arg("pos_enu") = std::array<double, 3>{0.0, 0.0, 35.0},
           nb::arg("vel_enu") = std::array<double, 3>{0.0, 0.0, 0.0},
           nb::arg("initial_soc") = 0.80f,
           nb::arg("fault_mask") = 0U,
           nb::arg("wind_enu") = std::array<double, 3>{0.0, 0.0, 0.0})
      .def("inject_fault", &NativeCoaxialSim::inject_fault,
           nb::arg("fault_kind"), nb::arg("target_idx") = 0U)
      .def("step_400hz",
           [](NativeCoaxialSim& self, uint32_t n_ticks, double ae, double an, double au, double yaw) {
             const StepOutcome res = self.step_400hz(n_ticks, ae, an, au, yaw);
             return std::make_tuple(res.obs, res.reward, res.terminated, res.cel_latched, res.active_motor_mask);
           },
           nb::arg("n_ticks") = 4U,
           nb::arg("a_nom_e") = 0.0,
           nb::arg("a_nom_n") = 0.0,
           nb::arg("a_nom_u") = 0.0,
           nb::arg("yaw_rate_des") = 0.0)
      .def("get_obs", &NativeCoaxialSim::build_obs)
      .def("latest_vrta_bytes", [](const NativeCoaxialSim& self) {
        const VrtaFrame100Hz& f = self.latest_frame();
        return nb::bytes(reinterpret_cast<const char*>(&f), sizeof(VrtaFrame100Hz));
      });

  nb::class_<NativeVecSim>(m, "NativeVecSim")
      .def(nb::init<std::size_t, std::size_t>(),
           nb::arg("num_envs"), nb::arg("num_threads") = 0U)
      .def_prop_ro("num_envs", &NativeVecSim::num_envs)
      .def_prop_ro("num_threads", &NativeVecSim::num_threads)
      .def("reset_one", &NativeVecSim::reset_one,
           nb::arg("idx"),
           nb::arg("pos_enu") = std::array<double, 3>{0.0, 0.0, 35.0},
           nb::arg("vel_enu") = std::array<double, 3>{0.0, 0.0, 0.0},
           nb::arg("initial_soc") = 0.80f,
           nb::arg("fault_mask") = 0U,
           nb::arg("wind_enu") = std::array<double, 3>{0.0, 0.0, 0.0})
      .def("inject_fault_one", &NativeVecSim::inject_fault_one,
           nb::arg("idx"), nb::arg("fault_kind"), nb::arg("target_idx") = 0U)
      .def("step_batch", &NativeVecSim::step_batch,
           nb::arg("actions"),
           nb::arg("n_ticks"),
           nb::arg("obs_out"),
           nb::arg("rew_out"),
           nb::arg("term_out"),
           nb::arg("cel_out"));
}
