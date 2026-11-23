// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta_rta_node/lockfree_transport.hpp"

namespace vertirta::ros {

// Pure C++20 zero-heap Thread-2 RT safety kernel linking real vertirta::rta::CbvfSafetyFilter,
// DEM terrain-subtracted AGL computation, real intruder/wire hazard ingestion, and staleness caps.
// Callable identically from VertiRtaNode (ROS 2) and standalone Catch2 unit tests.
class RtaSafetyCore {
 public:
  static constexpr uint64_t kStaleWarnNs = 250'000'000ULL;
  static constexpr uint64_t kStaleCapNs = 1'000'000'000ULL;
  static constexpr uint64_t kStaleDivertNs = 2'000'000'000ULL;
  static constexpr float kStaleSpeedCapMps = 8.0f;
  static constexpr uint8_t kModeAccelNed = 0U;
  static constexpr uint8_t kModeCelControlledLanding = 2U;

  explicit RtaSafetyCore(
      const rta::MmapCbvf4DLut* cbvf_lut = nullptr,
      const rta::Cel2DLut* cel_lut = nullptr) noexcept
      : filter_(cbvf_lut, cel_lut) {}

  void reset() noexcept {
    filter_.reset();
    cel_latched_ = false;
    rt_seq_ = 0ULL;
  }

  [[nodiscard]] bool is_cel_latched() const noexcept {
    return cel_latched_ || filter_.is_cel_latched();
  }

  [[nodiscard]] RtaCommandEgressFrame step_100hz(
      const TelemetryIngressFrame& tel,
      const PerceptionHazardSnapshot& haz,
      const core::Vec3Ned& a_nom_ned,
      double yaw_rate_nom = 0.0) noexcept {
    if (tel.flight_regime == static_cast<uint8_t>(core::FlightRegime::SAME_ARM_ACTIVE_SPIN) ||
        tel.flight_regime == static_cast<uint8_t>(core::FlightRegime::CRITICAL_DESCENT)) {
      cel_latched_ = true;
    }

    const uint64_t age_ns =
        (tel.stamp_ns >= haz.stamp_ns && haz.stamp_ns > 0ULL) ? (tel.stamp_ns - haz.stamp_ns) : 0ULL;
    const bool stale_warn = haz.stale || (age_ns > kStaleWarnNs);
    const bool stale_cap = (age_ns > kStaleCapNs);
    const bool stale_divert = (age_ns > kStaleDivertNs);

    // DEM-referenced AGL (subtracts terrain elevation relative to Kartal datum)
    const float agl_m = compute_terrain_agl_m(tel.pos_ned[2], haz.terrain_elev_m);

    // Build nominal acceleration with staleness 8.0 m/s speed-cap damping
    core::Vec3Ned a_des = a_nom_ned;
    if (stale_cap) {
      const double v_xy = std::hypot(static_cast<double>(tel.vel_ned[0]), static_cast<double>(tel.vel_ned[1]));
      if (v_xy > static_cast<double>(kStaleSpeedCapMps)) {
        const double scale = -1.5 * (v_xy - static_cast<double>(kStaleSpeedCapMps)) / v_xy;
        a_des.x = static_cast<double>(tel.vel_ned[0]) * scale;
        a_des.y = static_cast<double>(tel.vel_ned[1]) * scale;
      }
    }

    // Convert lock-free Seqlock HazardPods (intruders + wires + ESDF/CTR) into CbvfSafetyFilter span
    std::array<rta::HazardPoint, PerceptionHazardSnapshot::kMaxHazards> hp_buf{};
    const std::size_t n_haz = std::min<std::size_t>(
        haz.hazard_count, PerceptionHazardSnapshot::kMaxHazards);
    for (std::size_t i = 0U; i < n_haz; ++i) {
      const auto& src = haz.hazards[i];
      rta::HazardPoint hp{};
      hp.pos_ned = {src.pos_ned[0], src.pos_ned[1], src.pos_ned[2]};
      hp.vel_ned = {src.vel_ned[0], src.vel_ned[1], src.vel_ned[2]};
      hp.radius_h_m = static_cast<double>(src.radius_h_m) + (stale_warn ? 2.0 : 0.0);
      hp.radius_z_m = static_cast<double>(src.radius_z_m) + (stale_warn ? 1.0 : 0.0);
      hp.gamma = static_cast<double>(src.gamma);
      hp.kind = src.kind;
      hp_buf[i] = hp;
    }

    core::CapabilityEnvelope env{};
    env.fault_mask = tel.fault_bitmask;
    env.tmr_state = tel.tmr_state;
    env.imu_mask = tel.imu_health_mask;
    env.regime = static_cast<core::FlightRegime>(std::min<uint8_t>(tel.flight_regime, 4U));
    env.t_max_n = (tel.t_max_n > 100.0f) ? tel.t_max_n : 5886.0f;
    env.a_xy_max = (tel.a_xy_max > 0.5f) ? tel.a_xy_max : 6.87f;
    env.tilt_max_rad = (tel.tilt_max_rad > 0.05f) ? tel.tilt_max_rad : 0.45f;
    env.vz_sink_max = (tel.vz_sink_max > 0.2f) ? tel.vz_sink_max : 3.0f;
    env.v_term_v = tel.v_term_v;
    env.i_pack_a = tel.i_pack_a;
    env.soc = tel.soc;
    env.chi2_residual = tel.chi2_residual;

    const core::Vec3Ned pos_ned{tel.pos_ned[0], tel.pos_ned[1], tel.pos_ned[2]};
    const core::Vec3Ned vel_ned{
        static_cast<double>(tel.vel_ned[0]),
        static_cast<double>(tel.vel_ned[1]),
        static_cast<double>(tel.vel_ned[2]),
    };

    const auto rta_out = filter_.step_100hz(
        pos_ned,
        vel_ned,
        a_des,
        yaw_rate_nom,
        env,
        std::span<const rta::HazardPoint>(hp_buf.data(), n_haz));

    if (rta_out.cel_latched) {
      cel_latched_ = true;
    }

    RtaCommandEgressFrame cmd{};
    cmd.seq = ++rt_seq_;
    cmd.stamp_ns = tel.stamp_ns;
    cmd.cel_latched = cel_latched_;
    cmd.perception_stale = stale_warn;
    cmd.divert_requested = stale_divert;
    cmd.cbvf_min_margin_m = static_cast<float>(rta_out.min_cbvf_v);
    cmd.valid = true;

    if (cel_latched_) {
      cmd.control_mode = kModeCelControlledLanding;
      cmd.vz_limit_mps = (agl_m > 10.0f) ? 2.5f : ((agl_m > 5.0f) ? 1.5f : 0.8f);
      const float az_safe = (tel.vel_ned[2] > cmd.vz_limit_mps)
          ? std::min(static_cast<float>(rta_out.a_safe_ned.z), -1.5f)
          : static_cast<float>(rta_out.a_safe_ned.z);
      cmd.accel_ned = {
          static_cast<float>(rta_out.a_safe_ned.x),
          static_cast<float>(rta_out.a_safe_ned.y),
          az_safe,
      };
      cmd.yawspeed_sp = -2.2f;
    } else {
      cmd.control_mode = kModeAccelNed;
      cmd.accel_ned = {
          static_cast<float>(rta_out.a_safe_ned.x),
          static_cast<float>(rta_out.a_safe_ned.y),
          static_cast<float>(rta_out.a_safe_ned.z),
      };
      cmd.yawspeed_sp = static_cast<float>(rta_out.yaw_rate_safe);
      cmd.vz_limit_mps = env.vz_sink_max;
    }

    return cmd;
  }

 private:
  rta::CbvfSafetyFilter filter_;
  bool cel_latched_{false};
  uint64_t rt_seq_{0ULL};
};

}  // namespace vertirta::ros
