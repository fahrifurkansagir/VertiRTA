// SPDX-License-Identifier: Apache-2.0
// VertiRTA v3.5.1 — nanobind C++20 Python SDK bindings wrapping vertirta::flight_core & M2 engines.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/math_utils.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta/core/vrta_recorder.hpp"

#if defined(VERTIRTA_SDK_HAS_M2) && VERTIRTA_SDK_HAS_M2
#include "vertirta/airframes/lift_cruise_dynamics.hpp"
#include "vertirta/uspace/fleet_sim_engine.hpp"
#include "vertirta/uspace/part_is_security.hpp"
#endif

#if defined(VERTIRTA_SDK_HAS_M3) && VERTIRTA_SDK_HAS_M3
#include <filesystem>
#include <fstream>
#include "vertirta/forensics/ercs_scorer.hpp"
#include "vertirta/forensics/forensics_pipeline.hpp"
#include "vertirta/forensics/vrta_events_sidecar.hpp"
#include "vertirta/forward_hazard/forward_hazard_fusion.hpp"
#include "vertirta/gnss_denied/ctr_inflation_fsm.hpp"
#include "vertirta/gnss_denied/gnss_spoof_guard.hpp"
#include "vertirta/voice/mondrian_voice_gate.hpp"
#endif

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
      nb::ndarray<uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> cel_out,
      std::optional<nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>>
          step_mask = std::nullopt) {
    const std::size_t n = envs_.size();
    if (actions.shape(0) != n || actions.shape(1) != kActDim ||
        obs_out.shape(0) != n || obs_out.shape(1) != kObsDim ||
        rew_out.shape(0) != n || term_out.shape(0) != n || cel_out.shape(0) != n ||
        (step_mask.has_value() && step_mask->shape(0) != n)) {
      throw std::invalid_argument("step_batch array shape mismatch");
    }

    const float* act_ptr = actions.data();
    float* obs_ptr = obs_out.data();
    float* rew_ptr = rew_out.data();
    uint8_t* term_ptr = term_out.data();
    uint8_t* cel_ptr = cel_out.data();
    const uint8_t* mask_ptr = step_mask.has_value() ? step_mask->data() : nullptr;

    auto run_range = [&](std::size_t lo, std::size_t hi) noexcept {
      for (std::size_t i = lo; i < hi; ++i) {
        if (mask_ptr != nullptr && mask_ptr[i] == 0U) {
          rew_ptr[i] = 0.0f;
          term_ptr[i] = 0U;
          cel_ptr[i] = 0U;
          continue;
        }
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

#if defined(VERTIRTA_SDK_HAS_M2) && VERTIRTA_SDK_HAS_M2
const char* cyber_state_name(vertirta::uspace::CyberState st) noexcept {
  switch (st) {
    case vertirta::uspace::CyberState::SEC_NOMINAL:
      return "SEC_NOMINAL";
    case vertirta::uspace::CyberState::SEC_SUSPECT_PEER:
      return "SEC_SUSPECT_PEER";
    case vertirta::uspace::CyberState::SEC_USSP_QUARANTINE:
      return "SEC_USSP_QUARANTINE";
    case vertirta::uspace::CyberState::SEC_AUTONOMOUS_RTB:
      return "SEC_AUTONOMOUS_RTB";
    default:
      return "SEC_UNKNOWN";
  }
}

class NativeFleetSim {
 public:
  explicit NativeFleetSim(uint32_t n_agents = 20U, uint32_t threads = 1U)
      : engine_(n_agents, threads) {}

  void reset(uint64_t seed = 20261123ULL) {
    engine_.reset_corridor_schedule(seed);
    fsm_.reset();
  }

  void inject_same_arm_cel(uint32_t agent_idx, uint32_t arm_idx = 0U) {
    engine_.inject_same_arm_cel(agent_idx, arm_idx);
  }

  nb::dict step_100hz() {
    vertirta::uspace::FleetTickSummary s{};
    {
      nb::gil_scoped_release release;
      s = engine_.step_100hz();
    }
    nb::dict out;
    out["tick_index"] = s.tick_index;
    out["active_agents"] = s.active_agents;
    out["cel_active_agents"] = s.cel_active_agents;
    out["airspace_breaches"] = s.airspace_breaches;
    out["max_rta_hazard_rows"] = s.max_rta_hazard_rows;
    out["min_inter_agent_sep_m"] = s.min_inter_agent_sep_m;
    out["sim_time_s"] = s.sim_time_s;
    return out;
  }

  [[nodiscard]] uint32_t agent_count() const noexcept { return engine_.agent_count(); }
  [[nodiscard]] bool agent_cel_latched(uint32_t agent_idx) const noexcept {
    return engine_.agent_cel_latched(agent_idx);
  }

  nb::dict check_ussp_geofence(bool sig_valid, double cmd_lon_deg, double cmd_agl_m, double t_s) {
    const auto act = fsm_.on_ussp_command_check(sig_valid, cmd_lon_deg, cmd_agl_m, t_s);
    nb::dict out;
    out["cyber_state"] = std::string(cyber_state_name(fsm_.state()));
    out["drop_ussp_directives"] = act.drop_ussp_directives;
    out["emit_is_or_230_draft"] = act.emit_is_or_230_draft;
    return out;
  }

  nb::dict step_cyber_time(double t_s, bool ussp_auth_restored = false) {
    const auto act = fsm_.step_time(t_s, ussp_auth_restored);
    nb::dict out;
    out["cyber_state"] = std::string(cyber_state_name(fsm_.state()));
    out["divert_to_nearest_fato_emg"] = act.divert_to_nearest_fato_emg;
    out["elapsed_quarantine_s"] = act.elapsed_quarantine_s;
    return out;
  }

 private:
  vertirta::uspace::FleetSimEngine engine_;
  vertirta::uspace::CyberResilienceFsm fsm_{};
};

class NativeLiftCruiseSim {
 public:
  explicit NativeLiftCruiseSim(const std::string& layout = "8+1")
      : dyn_(layout == "4+1" ? vertirta::airframes::LiftCruiseLayout::kFourPlusOne
                             : vertirta::airframes::LiftCruiseLayout::kEightPlusOne) {}

  [[nodiscard]] uint8_t actuator_count() const noexcept { return dyn_.actuator_count(); }
  [[nodiscard]] uint8_t lift_rotor_count() const noexcept { return dyn_.lift_rotor_count(); }
  [[nodiscard]] double cruise_airspeed_mps() const noexcept { return dyn_.cruise_airspeed_mps(); }
  [[nodiscard]] double max_airspeed_mps() const noexcept { return dyn_.max_airspeed_mps(); }
  [[nodiscard]] double best_glide_ld_ratio() const noexcept { return dyn_.best_glide_ld_ratio(); }

  nb::dict audit_oei_trim(double v_airspeed_mps, uint8_t failed_lift_rotor = 0U) const {
    const auto res = dyn_.audit_oei_trim_feasibility(v_airspeed_mps, failed_lift_rotor);
    nb::dict out;
    out["trim_feasible_4dof"] = res.trim_feasible_4dof;
    out["peak_thrust_ratio"] = res.peak_thrust_ratio;
    out["effective_rank"] = res.effective_rank;
    out["v_airspeed_mps"] = v_airspeed_mps;
    return out;
  }

 private:
  vertirta::airframes::LiftCruiseDynamics dyn_;
};
#endif

#if defined(VERTIRTA_SDK_HAS_M3) && VERTIRTA_SDK_HAS_M3
class NativeForwardHazardFusion {
 public:
  NativeForwardHazardFusion() : fusion_(vertirta::forward_hazard::ForwardFusionConfig{}) {}

  void reset() noexcept { fusion_.reset(); }

  void push_pose_200hz(
      uint64_t stamp_ns,
      std::array<double, 3> pos_ned,
      std::array<double, 3> vel_ned) noexcept {
    vertirta::forward_hazard::TimedPoseSe3 pose{};
    pose.stamp_ns = stamp_ns;
    pose.pos_ned = {pos_ned[0], pos_ned[1], pos_ned[2]};
    pose.vel_ned = {vel_ned[0], vel_ned[1], vel_ned[2]};
    pose.valid = true;
    fusion_.push_pose_200hz(pose);
  }

  nb::dict step(
      uint64_t frame_stamp_ns,
      bool radar_healthy,
      const std::vector<std::array<double, 5>>& wire_lines,
      const std::vector<std::array<double, 5>>& radar_points,
      const std::vector<std::array<double, 5>>& intruder_boxes) {
    vertirta::forward_hazard::ForwardSensorBundle bundle{};
    bundle.frame_stamp_ns = frame_stamp_ns;
    bundle.radar_stamp_ns = frame_stamp_ns;
    bundle.radar_healthy = radar_healthy;
    bundle.camera_healthy = true;

    const std::size_t n_wires =
        std::min(wire_lines.size(), vertirta::forward_hazard::kMaxForwardLineDetections);
    bundle.wire_line_count = n_wires;
    for (std::size_t i = 0U; i < n_wires; ++i) {
      auto& w = bundle.wire_lines[i];
      w.u0_px = wire_lines[i][0];
      w.v0_px = wire_lines[i][1];
      w.u1_px = wire_lines[i][2];
      w.v1_px = wire_lines[i][3];
      w.score = wire_lines[i][4];
    }

    const std::size_t n_radar =
        std::min(radar_points.size(), vertirta::forward_hazard::kMaxRadarDetections);
    bundle.radar_point_count = n_radar;
    for (std::size_t i = 0U; i < n_radar; ++i) {
      auto& r = bundle.radar_points[i];
      r.range_m = radar_points[i][0];
      r.az_rad = radar_points[i][1];
      r.el_rad = radar_points[i][2];
      r.radial_vel_mps = radar_points[i][3];
      r.residual_doppler_mps = radar_points[i][3];
      r.snr_db = radar_points[i][4];
      r.rcs_dbsm = -20.0;
      r.doppler_class = (std::abs(radar_points[i][3]) > 1.2)
                            ? vertirta::forward_hazard::RadarDopplerClass::kDynamicIntruder
                            : vertirta::forward_hazard::RadarDopplerClass::kStaticOrHoverCandidate;
      r.valid = true;
    }

    const std::size_t n_boxes =
        std::min(intruder_boxes.size(), vertirta::forward_hazard::kMaxForwardBoxDetections);
    bundle.intruder_box_count = n_boxes;
    for (std::size_t i = 0U; i < n_boxes; ++i) {
      auto& b = bundle.intruder_boxes[i];
      b.cx_px = intruder_boxes[i][0];
      b.cy_px = intruder_boxes[i][1];
      b.width_px = intruder_boxes[i][2];
      b.height_px = intruder_boxes[i][3];
      b.score = intruder_boxes[i][4];
    }

    vertirta::forward_hazard::ForwardHazardOutput out{};
    bool ok = false;
    {
      nb::gil_scoped_release release;
      ok = fusion_.step(bundle, out);
    }

    nb::dict res;
    res["ok"] = ok;
    res["recommended_v_max_mps"] = out.recommended_v_max_mps;
    res["radar_degraded"] = out.radar_degraded;
    res["grazing_parallax_active"] = out.grazing_parallax_active;
    res["standoff_capsule_emitted"] = out.standoff_capsule_emitted;
    res["active_wire_spans"] = out.active_wire_spans;
    res["active_wire_capsules"] = out.active_wire_capsules;
    res["active_intruders"] = out.active_intruders;
    res["hover_intruders_rescued"] = out.hover_intruders_rescued;
    return res;
  }

 private:
  vertirta::forward_hazard::ForwardHazardFusion fusion_;
};

class NativeGnssDeniedSuite {
 public:
  NativeGnssDeniedSuite() : guard_{}, ctr_{} { ctr_.reset(2.0); }

  void reset(double sigma_init_m = 2.0) noexcept {
    guard_.reset();
    ctr_.reset(sigma_init_m);
  }

  nb::dict step_spoof_guard_10hz(
      uint64_t stamp_ns,
      std::array<double, 3> gnss_pos_ned,
      std::array<double, 3> gnss_vel_ned,
      std::array<double, 3> vio_pos_ned,
      std::array<double, 3> vio_vel_ned,
      double water_fraction = 0.05,
      double cn0_dbhz = 40.0,
      double hdop = 0.9,
      bool valid = true) {
    vertirta::gnss_denied::GnssFixV116 fix{};
    fix.stamp_ns = stamp_ns;
    fix.pos_ned = {gnss_pos_ned[0], gnss_pos_ned[1], gnss_pos_ned[2]};
    fix.vel_ned = {gnss_vel_ned[0], gnss_vel_ned[1], gnss_vel_ned[2]};
    fix.cn0_avg_dbhz = cn0_dbhz;
    fix.hdop = hdop;
    fix.valid = valid;
    fix.fix_type = valid ? 3U : 0U;

    vertirta::gnss_denied::VioStateEstimate vio{};
    vio.stamp_ns = stamp_ns;
    vio.pos_ned = {vio_pos_ned[0], vio_pos_ned[1], vio_pos_ned[2]};
    vio.vel_ned = {vio_vel_ned[0], vio_vel_ned[1], vio_vel_ned[2]};
    vio.water_fraction = water_fraction;
    vio.metric_scale_locked = true;
    vio.valid = true;

    vertirta::gnss_denied::GnssSpoofGuardStatus st{};
    {
      nb::gil_scoped_release release;
      st = guard_.step_10hz(fix, vio);
    }

    const char* hyp = "H0_NOMINAL";
    if (st.hypothesis == vertirta::gnss_denied::GnssHypothesis::kH1RfJamming) {
      hyp = "H1_RF_JAMMING";
    } else if (st.hypothesis == vertirta::gnss_denied::GnssHypothesis::kH2StepMeaconing) {
      hyp = "H2_STEP_MEACONING";
    } else if (st.hypothesis == vertirta::gnss_denied::GnssHypothesis::kH3SlowRampSpoof) {
      hyp = "H3_SLOW_RAMP_SPOOF";
    }

    nb::dict out;
    out["hypothesis"] = std::string(hyp);
    out["gnss_fused_allowed"] = st.gnss_fused_allowed;
    out["latched"] = st.latched;
    out["rollback_executed"] = st.rollback_executed;
    out["mcusum_norm_mps"] = st.mcusum_norm_mps;
    out["active_k_d_mps"] = st.active_k_d_mps;
    out["active_h_t_mps"] = st.active_h_t_mps;
    out["sigma_init_m"] = st.sigma_init_m;
    out["rollback_pos_ned"] = std::array<double, 3>{
        st.rollback_pos_ned.x, st.rollback_pos_ned.y, st.rollback_pos_ned.z};
    return out;
  }

  nb::dict propagate_and_evaluate_ctr(
      double dt_s,
      double horiz_speed_mps,
      std::array<double, 3> own_pos_ned,
      double alt_amsl_m = 57.0) {
    std::array<vertirta::rta::HazardPoint, 2> hz{};
    vertirta::gnss_denied::CtrInflationStatus st{};
    {
      nb::gil_scoped_release release;
      ctr_.propagate_open_loop(dt_s, horiz_speed_mps);
      st = ctr_.evaluate_and_build_hazards(
          dt_s, {own_pos_ned[0], own_pos_ned[1], own_pos_ned[2]}, alt_amsl_m, hz);
    }

    const char* nav = "BOUNDED_GNSS_DENIED_CRUISE";
    if (st.nav_state == vertirta::gnss_denied::GnssDeniedNavState::kDegradedCaution) {
      nav = "DEGRADED_CAUTION";
    } else if (st.nav_state ==
               vertirta::gnss_denied::GnssDeniedNavState::kDeterministicVisualSlzAbort) {
      nav = "DETERMINISTIC_VISUAL_SLZ_ABORT";
    }

    nb::dict out;
    out["nav_state"] = std::string(nav);
    out["sigma_pos_m"] = st.sigma_pos_m;
    out["three_sigma_pos_m"] = st.three_sigma_pos_m;
    out["nominal_3sigma_met"] = st.nominal_3sigma_met;
    out["visual_slz_abort_latched"] = st.visual_slz_abort_latched;
    out["westward_bias_active"] = st.westward_bias_active;
    out["v_max_cap_mps"] = st.v_max_cap_mps;
    out["ltfj_inflated_margin_m"] = st.ltfj_inflated_margin_m;
    out["ltbx_inflated_margin_m"] = st.ltbx_inflated_margin_m;
    out["ltfj_delta_r_m"] = st.ltfj_delta_r_m;
    out["ltfj_r_dot_infl_mps"] = st.ltfj_r_dot_infl_mps;
    out["emitted_ctr_hazards"] = st.emitted_ctr_hazards;
    return out;
  }

 private:
  vertirta::gnss_denied::GnssSpoofGuard guard_;
  vertirta::gnss_denied::CtrInflationFsm ctr_;
};

class NativeMondrianVoiceGate {
 public:
  NativeMondrianVoiceGate() : gate_(vertirta::voice::make_default_mondrian_config()) {}

  nb::dict evaluate(
      float snr_db,
      const std::string& lang,
      const std::vector<std::tuple<std::string, float, bool>>& candidates,
      float soc_pct = 50.0f,
      float soc_reserve_floor_pct = 12.0f,
      float mayday_acoustic_weight = 0.0f,
      float nospeech_prob = 0.0f) const {
    vertirta::voice::VoiceTurnSignals sig{};
    sig.snr_db = snr_db;
    sig.lang = (lang == "EN") ? vertirta::voice::VoiceLang::EN
                              : (lang == "MIX" ? vertirta::voice::VoiceLang::MIX
                                               : vertirta::voice::VoiceLang::TR);
    sig.mayday_acoustic_weight = mayday_acoustic_weight;
    sig.nospeech_prob = nospeech_prob;

    std::vector<vertirta::voice::VoiceCandidateScore> cands{};
    cands.reserve(candidates.size());
    for (const auto& tup : candidates) {
      cands.push_back(vertirta::voice::VoiceCandidateScore{
          std::get<0>(tup), std::get<1>(tup), std::get<2>(tup)});
    }

    vertirta::voice::VoiceRtaLivenessState rta{};
    rta.soc_pct = soc_pct;
    rta.soc_reserve_floor_pct = soc_reserve_floor_pct;

    const auto dec = gate_.evaluate(sig, cands, rta);
    const char* act = "ABSTAIN_HOLD_SAFE";
    switch (dec.decision) {
      case vertirta::voice::VoiceGateAction::EXECUTE_SINGLE:
        act = "EXECUTE_SINGLE";
        break;
      case vertirta::voice::VoiceGateAction::REQUEST_CLARIFICATION:
        act = "REQUEST_CLARIFICATION";
        break;
      case vertirta::voice::VoiceGateAction::ABSTAIN_HOLD_SAFE:
        act = "ABSTAIN_HOLD_SAFE";
        break;
      case vertirta::voice::VoiceGateAction::LIVENESS_EMERGENCY_COMMIT:
        act = "LIVENESS_EMERGENCY_COMMIT";
        break;
      case vertirta::voice::VoiceGateAction::EMERGENCY_PREEMPT:
        act = "EMERGENCY_PREEMPT";
        break;
    }

    nb::dict out;
    out["decision"] = std::string(act);
    out["stratum"] = std::string(vertirta::voice::snr_stratum_name(dec.stratum));
    out["lang"] = std::string(vertirta::voice::voice_lang_name(dec.lang));
    out["effective_q_hat"] = dec.effective_q_hat;
    out["p_min_threshold"] = dec.p_min_threshold;
    out["conformal_set_size"] = dec.conformal_set_size;
    out["selected_action_id"] = dec.selected_action_id;
    out["clarify_reason"] = dec.clarify_reason;
    return out;
  }

 private:
  vertirta::voice::MondrianVoiceConformalGate gate_;
};

class NativeForensicsEngine {
 public:
  nb::dict score_ercs(
      const std::string& trigger,
      const std::string& fault_domain,
      const std::string& key_risk_area,
      uint32_t occupant_capacity = 1U,
      std::array<uint8_t, 8> barriers = {1U, 2U, 2U, 1U, 2U, 1U, 1U, 3U},
      uint8_t stopping_barrier_idx = 0U) const {
    vertirta::forensics::ErcsInput in{};
    in.occupant_capacity = occupant_capacity;
    in.barriers = barriers;
    in.stopping_barrier_idx = stopping_barrier_idx;

    if (trigger == "FATO_INCURSION") {
      in.trigger = vertirta::forensics::Cir2026Trigger::FATO_INCURSION;
    } else if (trigger == "USPACE_AIRBORNE_ENCOUNTER") {
      in.trigger = vertirta::forensics::Cir2026Trigger::USPACE_AIRBORNE_ENCOUNTER;
    } else if (trigger == "PART_IS_INFOSEC_ANOMALY") {
      in.trigger = vertirta::forensics::Cir2026Trigger::PART_IS_INFOSEC_ANOMALY;
    } else if (trigger == "UAS_CMU_TECHNICAL_FAULT") {
      in.trigger = vertirta::forensics::Cir2026Trigger::UAS_CMU_TECHNICAL_FAULT;
    } else {
      in.trigger = vertirta::forensics::Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT;
    }

    if (fault_domain == "FLIGHT_CONTROL_OR_SOFTWARE") {
      in.fault_domain = vertirta::forensics::FaultDomain::FLIGHT_CONTROL_OR_SOFTWARE;
    } else if (fault_domain == "DATALINK_OR_USPACE_CMU") {
      in.fault_domain = vertirta::forensics::FaultDomain::DATALINK_OR_USPACE_CMU;
    } else if (fault_domain == "NAVIGATION_GNSS_VIO") {
      in.fault_domain = vertirta::forensics::FaultDomain::NAVIGATION_GNSS_VIO;
    } else {
      in.fault_domain = vertirta::forensics::FaultDomain::ROTOR_ACTUATOR_OR_INVERTER;
    }

    if (key_risk_area == "MAC_HIGH_ENERGY") {
      in.key_risk_area = vertirta::forensics::KeyRiskArea::MAC_HIGH_ENERGY;
    } else if (key_risk_area == "CFIT_HIGH_ENERGY") {
      in.key_risk_area = vertirta::forensics::KeyRiskArea::CFIT_HIGH_ENERGY;
    } else if (key_risk_area == "GROUND_DAMAGE_LOW_ENERGY") {
      in.key_risk_area = vertirta::forensics::KeyRiskArea::GROUND_DAMAGE_LOW_ENERGY;
    } else {
      in.key_risk_area = vertirta::forensics::KeyRiskArea::LOC_I_HIGH_ENERGY;
    }

    const auto r = vertirta::forensics::score_ercs_cdr_2020_2034(in);
    nb::dict out;
    out["cictt_category"] = r.cictt_category;
    out["severity_letter"] = std::string(1U, r.severity_letter);
    out["probability_index"] = r.probability_index;
    out["ercs_matrix_code"] = r.ercs_matrix_code;
    out["effective_barrier_sum"] = r.effective_barrier_sum;
    out["effective_barrier8_score"] = r.effective_barrier8_score;
    out["occurrence_class_id_431"] =
        (r.occurrence_class_id_431 == vertirta::forensics::OccurrenceClassDraft::DRAFT_INCIDENT_ONLY)
            ? std::string("DRAFT_INCIDENT_ONLY")
            : std::string("DRAFT_NOT_DETERMINED");
    out["auto_classified_accident_or_serious_incident"] =
        r.auto_classified_accident_or_serious_incident;
    out["reporting_status"] = r.reporting_status;
    out["ata_jasc_chapter"] = r.ata_jasc_chapter;
    return out;
  }

  nb::dict export_e5x_from_vrta(
      const std::string& vrta_path,
      const std::string& out_e5x_xml_path,
      const std::string& out_audit_json_path,
      const std::string& sidecar_path = "",
      const std::string& sidecar_public_key_hex = "",
      const std::string& file_number = "VRT-2026-0001",
      const std::string& raw_headline = "Envelope protection event along Istanbul coastal corridor",
      uint32_t occupant_capacity = 1U) const {
    vertirta::forensics::ForensicsPipelineRequest req{};
    req.vrta_path = std::filesystem::path(vrta_path);
    req.out_e5x_xml_path = std::filesystem::path(out_e5x_xml_path);
    req.out_audit_json_path = std::filesystem::path(out_audit_json_path);
    if (!sidecar_path.empty()) {
      req.sidecar_path = std::filesystem::path(sidecar_path);
      req.sidecar_public_key_hex = sidecar_public_key_hex;
    } else {
      // Auto-emit a cryptographically chained ED-155A sidecar anchored to the first VRTA0001 frame.
      uint64_t first_seq = 0ULL;
      uint64_t first_t_sim_us = 0ULL;
      std::ifstream vf(req.vrta_path, std::ios::binary);
      if (vf.is_open()) {
        vf.seekg(static_cast<std::streamoff>(sizeof(VrtaFileHeader)), std::ios::beg);
        VrtaFrame100Hz first_frame{};
        if (vf.read(reinterpret_cast<char*>(&first_frame), sizeof(first_frame))) {
          first_seq = first_frame.seq;
          first_t_sim_us = first_frame.t_sim_us;
        }
      }
      const std::filesystem::path auto_sidecar =
          req.out_e5x_xml_path.parent_path() /
          (req.out_e5x_xml_path.stem().string() + ".vrta_events.jsonl");
      std::error_code ec;
      std::filesystem::remove(auto_sidecar, ec);
      vertirta::forensics::VrtaEventsSidecarWriter writer{};
      vertirta::forensics::SidecarEvent ev{};
      ev.seq = first_seq;
      ev.t_sim_us = first_t_sim_us;
      ev.kind = vertirta::forensics::SidecarEventKind::RTA_ENVELOPE_PROTECTION;
      ev.snr_stratum = "S1_Cruise";
      ev.conformal_set_size = 1U;
      ev.asr_transcript_utf8 = "Kartal coastal corridor RTA envelope protection event";
      ev.audio_sha256_hex = vertirta::forensics::sha256_hex_string(ev.asr_transcript_utf8);
      ev.z3_smt2_sha256_hex = vertirta::forensics::sha256_hex_string("(check-sat)");
      ev.gsn_node_id = "G1.1-INDI-WLS";
      (void)writer.append_to_file(auto_sidecar, ev, false);
      req.sidecar_path = auto_sidecar;
      req.sidecar_public_key_hex = writer.public_key_hex();
    }
    req.file_number = file_number;
    req.raw_headline = raw_headline;
    req.ercs_input.occupant_capacity = occupant_capacity;
    req.ercs_input.barriers = {1U, 2U, 2U, 1U, 2U, 1U, 1U, 3U};

    vertirta::forensics::ForensicsPipelineResult res{};
    {
      nb::gil_scoped_release release;
      res = vertirta::forensics::run_forensics_pipeline(req);
    }

    nb::dict out;
    out["ok"] = res.ok;
    out["frames_verified"] = res.frames_verified;
    out["sidecar_events_joined"] = res.sidecar_events_joined;
    out["vrta_sha256_hex"] = res.vrta_sha256_hex;
    out["cictt_category"] = res.ercs.cictt_category;
    out["ercs_matrix_code"] = res.ercs.ercs_matrix_code;
    out["effective_barrier8_score"] = res.ercs.effective_barrier8_score;
    out["occurrence_class_id_431"] =
        (res.ercs.occurrence_class_id_431 ==
         vertirta::forensics::OccurrenceClassDraft::DRAFT_INCIDENT_ONLY)
            ? std::string("DRAFT_INCIDENT_ONLY")
            : std::string("DRAFT_NOT_DETERMINED");
    out["auto_classified_accident_or_serious_incident"] =
        res.ercs.auto_classified_accident_or_serious_incident;
    out["error_message"] = res.error_message;
    return out;
  }
};
#endif

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
           nb::arg("cel_out"),
           nb::arg("step_mask") = nb::none());

#if defined(VERTIRTA_SDK_HAS_M2) && VERTIRTA_SDK_HAS_M2
  m.attr("HAS_M2_BINDINGS") = true;

  nb::class_<NativeFleetSim>(m, "NativeFleetSim")
      .def(nb::init<uint32_t, uint32_t>(),
           nb::arg("n_agents") = 20U, nb::arg("threads") = 1U)
      .def_prop_ro("agent_count", &NativeFleetSim::agent_count)
      .def("reset", &NativeFleetSim::reset, nb::arg("seed") = 20261123ULL)
      .def("inject_same_arm_cel", &NativeFleetSim::inject_same_arm_cel,
           nb::arg("agent_idx"), nb::arg("arm_idx") = 0U)
      .def("step_100hz", &NativeFleetSim::step_100hz)
      .def("agent_cel_latched", &NativeFleetSim::agent_cel_latched, nb::arg("agent_idx"))
      .def("check_ussp_geofence", &NativeFleetSim::check_ussp_geofence,
           nb::arg("sig_valid"), nb::arg("cmd_lon_deg"), nb::arg("cmd_agl_m"), nb::arg("t_s"))
      .def("step_cyber_time", &NativeFleetSim::step_cyber_time,
           nb::arg("t_s"), nb::arg("ussp_auth_restored") = false);

  nb::class_<NativeLiftCruiseSim>(m, "NativeLiftCruiseSim")
      .def(nb::init<const std::string&>(), nb::arg("layout") = "8+1")
      .def_prop_ro("actuator_count", &NativeLiftCruiseSim::actuator_count)
      .def_prop_ro("lift_rotor_count", &NativeLiftCruiseSim::lift_rotor_count)
      .def_prop_ro("cruise_airspeed_mps", &NativeLiftCruiseSim::cruise_airspeed_mps)
      .def_prop_ro("max_airspeed_mps", &NativeLiftCruiseSim::max_airspeed_mps)
      .def_prop_ro("best_glide_ld_ratio", &NativeLiftCruiseSim::best_glide_ld_ratio)
      .def("audit_oei_trim", &NativeLiftCruiseSim::audit_oei_trim,
           nb::arg("v_airspeed_mps"), nb::arg("failed_lift_rotor") = 0U);
#else
  m.attr("HAS_M2_BINDINGS") = false;
#endif

#if defined(VERTIRTA_SDK_HAS_M3) && VERTIRTA_SDK_HAS_M3
  m.attr("HAS_M3_BINDINGS") = true;

  nb::class_<NativeForwardHazardFusion>(m, "NativeForwardHazardFusion")
      .def(nb::init<>())
      .def("reset", &NativeForwardHazardFusion::reset)
      .def("push_pose_200hz", &NativeForwardHazardFusion::push_pose_200hz,
           nb::arg("stamp_ns"), nb::arg("pos_ned"), nb::arg("vel_ned"))
      .def("step", &NativeForwardHazardFusion::step,
           nb::arg("frame_stamp_ns"),
           nb::arg("radar_healthy") = true,
           nb::arg("wire_lines") = std::vector<std::array<double, 5>>{},
           nb::arg("radar_points") = std::vector<std::array<double, 5>>{},
           nb::arg("intruder_boxes") = std::vector<std::array<double, 5>>{});

  nb::class_<NativeGnssDeniedSuite>(m, "NativeGnssDeniedSuite")
      .def(nb::init<>())
      .def("reset", &NativeGnssDeniedSuite::reset, nb::arg("sigma_init_m") = 2.0)
      .def("step_spoof_guard_10hz", &NativeGnssDeniedSuite::step_spoof_guard_10hz,
           nb::arg("stamp_ns"),
           nb::arg("gnss_pos_ned"),
           nb::arg("gnss_vel_ned"),
           nb::arg("vio_pos_ned"),
           nb::arg("vio_vel_ned"),
           nb::arg("water_fraction") = 0.05,
           nb::arg("cn0_dbhz") = 40.0,
           nb::arg("hdop") = 0.9,
           nb::arg("valid") = true)
      .def("propagate_and_evaluate_ctr", &NativeGnssDeniedSuite::propagate_and_evaluate_ctr,
           nb::arg("dt_s"),
           nb::arg("horiz_speed_mps"),
           nb::arg("own_pos_ned"),
           nb::arg("alt_amsl_m") = 57.0);

  nb::class_<NativeMondrianVoiceGate>(m, "NativeMondrianVoiceGate")
      .def(nb::init<>())
      .def("evaluate", &NativeMondrianVoiceGate::evaluate,
           nb::arg("snr_db"),
           nb::arg("lang"),
           nb::arg("candidates"),
           nb::arg("soc_pct") = 50.0f,
           nb::arg("soc_reserve_floor_pct") = 12.0f,
           nb::arg("mayday_acoustic_weight") = 0.0f,
           nb::arg("nospeech_prob") = 0.0f);

  nb::class_<NativeForensicsEngine>(m, "NativeForensicsEngine")
      .def(nb::init<>())
      .def("score_ercs", &NativeForensicsEngine::score_ercs,
           nb::arg("trigger") = "CRITICAL_ENERGY_OR_ENVELOPE_PROT",
           nb::arg("fault_domain") = "ROTOR_ACTUATOR_OR_INVERTER",
           nb::arg("key_risk_area") = "LOC_I_HIGH_ENERGY",
           nb::arg("occupant_capacity") = 1U,
           nb::arg("barriers") = std::array<uint8_t, 8>{1U, 2U, 2U, 1U, 2U, 1U, 1U, 3U},
           nb::arg("stopping_barrier_idx") = 0U)
      .def("export_e5x_from_vrta", &NativeForensicsEngine::export_e5x_from_vrta,
           nb::arg("vrta_path"),
           nb::arg("out_e5x_xml_path"),
           nb::arg("out_audit_json_path"),
           nb::arg("sidecar_path") = "",
           nb::arg("sidecar_public_key_hex") = "",
           nb::arg("file_number") = "VRT-2026-0001",
           nb::arg("raw_headline") = "Envelope protection event along Istanbul coastal corridor",
           nb::arg("occupant_capacity") = 1U);
#else
  m.attr("HAS_M3_BINDINGS") = false;
#endif
}
