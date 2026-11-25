// SPDX-License-Identifier: Apache-2.0
#include "vertirta/mavlink/offboard_rta_shield.hpp"

#include <algorithm>
#include <cmath>

#include "vertirta/core/geodesy.hpp"
#include "vertirta/gnss_denied/ctr_inflation_fsm.hpp"
#include "vertirta/wind_acoustics/istanbul_receptors.hpp"

namespace vertirta::mavlink {

namespace {

[[nodiscard]] bool is_finite_f32(float v) noexcept {
  return std::isfinite(static_cast<double>(v));
}

[[nodiscard]] std::vector<uint8_t> to_vec(std::span<const uint8_t> buf, std::size_t n) {
  return std::vector<uint8_t>(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
}

}  // namespace

OffboardRtaShield::OffboardRtaShield(
    const ShieldConfig& cfg,
    const rta::MmapCbvf4DLut* cbvf_lut,
    const rta::Cel2DLut* cel_lut) noexcept
    : cfg_(cfg), cbvf_filter_(cbvf_lut, cel_lut) {
  // Precompute LTBX ARP, Kartal Hospital helipad, and LTFJ CTR western wall in local NED.
  const core::Vec3Enu ltbx_enu = core::geodetic_to_local_enu(
      {gnss_denied::kVerifiedLtbxArpLatDeg, gnss_denied::kVerifiedLtbxArpLonDeg, core::kKartalDatum.h_hae_m},
      core::kKartalDatum);
  ltbx_arp_ned_ = core::enu_to_ned(ltbx_enu);

  const rta::HazardPoint hosp = wind_acoustics::build_kartal_hospital_helipad_hazard();
  hospital_ned_ = hosp.pos_ned;

  const core::Vec3Enu ltfj_enu = core::geodetic_to_local_enu(
      {core::kKartalDatum.lat_deg, cfg_.ltfj_max_lon_deg, core::kKartalDatum.h_hae_m},
      core::kKartalDatum);
  ltfj_wall_east_m_ = core::enu_to_ned(ltfj_enu).y;
}

void OffboardRtaShield::reset() noexcept {
  state_ = VehicleTelemetryState{};
  stats_ = ShieldStats{};
  cbvf_filter_.reset();
  tx_seq_ = 0u;
}

bool OffboardRtaShield::is_deadman_expired(uint64_t now_us) const noexcept {
  if (!state_.heartbeat_seen) {
    return true;
  }
  if (now_us < state_.last_fcu_heartbeat_us) {
    return false;
  }
  return (now_us - state_.last_fcu_heartbeat_us) > cfg_.deadman_timeout_us;
}

bool OffboardRtaShield::violates_istanbul_nfz(const core::Vec3Ned& pos_ned) const noexcept {
  // 1. LTFJ CTR western wall (lon >= 29.1895 deg E or within safety buffer)
  const core::Vec3Enu enu = core::ned_to_enu(pos_ned);
  const core::GeodeticWgs84 geo = core::local_enu_to_geodetic(enu, core::kKartalDatum);
  if (geo.lon_deg >= cfg_.ltfj_max_lon_deg || pos_ned.y >= (ltfj_wall_east_m_ - cfg_.ltfj_buffer_m)) {
    return true;
  }

  // 2. Kartal Dr. Lutfi Kirdar Hospital hard Tier-2 helipad cylinder
  const double hosp_r = std::hypot(pos_ned.x - hospital_ned_.x, pos_ned.y - hospital_ned_.y);
  const double agl_m = -pos_ned.z;
  if (hosp_r <= cfg_.hospital_nfz_radius_m && agl_m <= cfg_.hospital_nfz_ceiling_agl_m) {
    return true;
  }

  // 3. LTBX Samandira Military CTR 3 NM cylinder (40.990556 N, 29.216111 E)
  const double ltbx_r = std::hypot(pos_ned.x - ltbx_arp_ned_.x, pos_ned.y - ltbx_arp_ned_.y);
  if (ltbx_r <= cfg_.ltbx_ctr_radius_m) {
    return true;
  }

  return false;
}

std::array<rta::HazardPoint, 4> OffboardRtaShield::build_active_hazards() const noexcept {
  std::array<rta::HazardPoint, 4> hazards{};

  // 0: LTFJ CTR western wall (real VertiRTA Istanbul hazard generator)
  hazards[0] = wind_acoustics::build_ltfj_ctr_wall_hazard(state_.pos_ned);

  // 1: Kartal Dr. Lutfi Kirdar Hospital active helipad (real VertiRTA Istanbul hazard generator)
  hazards[1] = wind_acoustics::build_kartal_hospital_helipad_hazard();

  // 2: LTBX CTR boundary point nearest to ownship
  {
    const double dn = state_.pos_ned.x - ltbx_arp_ned_.x;
    const double de = state_.pos_ned.y - ltbx_arp_ned_.y;
    const double r = std::hypot(dn, de);
    const double ux = (r > 1e-6) ? (dn / r) : -1.0;
    const double uy = (r > 1e-6) ? (de / r) : 0.0;
    rta::HazardPoint ltbx_hz{};
    ltbx_hz.pos_ned = {
        ltbx_arp_ned_.x + ux * cfg_.ltbx_ctr_radius_m,
        ltbx_arp_ned_.y + uy * cfg_.ltbx_ctr_radius_m,
        state_.pos_ned.z,
    };
    ltbx_hz.radius_h_m = 25.0;
    ltbx_hz.radius_z_m = 250.0;
    ltbx_hz.gamma = 1.5;
    ltbx_hz.kind = 3u;
    hazards[2] = ltbx_hz;
  }

  // 3: Terrain floor directly beneath ownship (protects min_agl_m against plunge commands)
  {
    rta::HazardPoint terrain_hz{};
    terrain_hz.pos_ned = {state_.pos_ned.x, state_.pos_ned.y, 0.0};
    terrain_hz.radius_h_m = 0.0;
    terrain_hz.radius_z_m = std::max(0.0, cfg_.min_agl_m - 3.0);
    terrain_hz.gamma = 1.6;
    terrain_hz.kind = 4u;
    hazards[3] = terrain_hz;
  }

  return hazards;
}

std::vector<uint8_t> OffboardRtaShield::make_statustext_packet(uint8_t severity, std::string_view text) {
  StatusText st{};
  st.severity = severity;
  st.set_text(text);
  std::array<uint8_t, kMaxFrameBytes> buf{};
  const std::size_t n = encode_statustext(st, tx_seq_++, cfg_.bridge_sysid, cfg_.bridge_compid, buf, false);
  ++stats_.statustext_emitted;
  return to_vec(buf, n);
}

std::vector<uint8_t> OffboardRtaShield::make_command_ack_packet(
    uint16_t command, uint8_t result, uint8_t target_sys, uint8_t target_comp) {
  CommandAck ack{};
  ack.command = command;
  ack.result = result;
  ack.progress = 0u;
  ack.result_param2 = 0;
  ack.target_system = target_sys;
  ack.target_component = target_comp;
  std::array<uint8_t, kMaxFrameBytes> buf{};
  const std::size_t n = encode_command_ack(ack, tx_seq_++, cfg_.bridge_sysid, cfg_.bridge_compid, buf, true);
  ++stats_.command_acks_emitted;
  return to_vec(buf, n);
}

std::vector<uint8_t> OffboardRtaShield::make_hover_setpoint_packet(const SetPositionTargetLocalNed& src) {
  SetPositionTargetLocalNed hover{};
  hover.time_boot_ms = src.time_boot_ms;
  hover.x = static_cast<float>(state_.pos_ned.x);
  hover.y = static_cast<float>(state_.pos_ned.y);
  hover.z = static_cast<float>(std::min(state_.pos_ned.z, -cfg_.min_agl_m));
  hover.vx = 0.0f;
  hover.vy = 0.0f;
  hover.vz = 0.0f;
  hover.afx = 0.0f;
  hover.afy = 0.0f;
  hover.afz = 0.0f;
  hover.yaw = 0.0f;
  hover.yaw_rate = 0.0f;
  hover.type_mask = kTypeMaskVelocityOnly;
  hover.target_system = src.target_system;
  hover.target_component = src.target_component;
  hover.coordinate_frame = kMavFrameLocalNed;

  std::array<uint8_t, kMaxFrameBytes> buf{};
  const std::size_t n = encode_set_position_target_local_ned(
      hover, tx_seq_++, cfg_.bridge_sysid, cfg_.bridge_compid, buf);
  return to_vec(buf, n);
}

ShieldStepOutput OffboardRtaShield::process_fcu_packet(
    std::span<const uint8_t> wire_bytes, uint64_t now_us) {
  ShieldStepOutput out{};
  const DecodeResult dec = decode_frame(wire_bytes);
  if (!dec.ok()) {
    ++stats_.fcu_rx_rejected;
    if (dec.error == DecodeError::kBadCrc) {
      ++stats_.crc_rejected;
    } else {
      ++stats_.malformed_rejected;
    }
    out.action = ShieldAction::kRejectedInvalidFrame;
    return out;
  }

  ++stats_.fcu_rx_valid;
  const MavlinkV2Frame& f = dec.frame;
  switch (f.msgid) {
    case kMsgIdHeartbeat: {
      const Heartbeat hb = decode_heartbeat(f);
      state_.heartbeat_seen = true;
      state_.last_fcu_heartbeat_us = now_us;
      state_.base_mode = hb.base_mode;
      state_.armed = (hb.base_mode & kMavModeFlagSafetyArmed) != 0u;
      state_.system_status = hb.system_status;
      break;
    }
    case kMsgIdSysStatus: {
      const SysStatus ss = decode_sys_status(f);
      state_.voltage_battery_mv = ss.voltage_battery;
      state_.battery_remaining_pct = ss.battery_remaining;
      if (ss.battery_remaining >= 0 && ss.battery_remaining <= 100) {
        state_.envelope.soc = static_cast<float>(ss.battery_remaining) * 0.01f;
      }
      break;
    }
    case kMsgIdAttitudeQuaternion: {
      const AttitudeQuaternion aq = decode_attitude_quaternion(f);
      if (is_finite_f32(aq.q1) && is_finite_f32(aq.q2) && is_finite_f32(aq.q3) && is_finite_f32(aq.q4)) {
        state_.quat_px4 = {aq.q1, aq.q2, aq.q3, aq.q4};
      }
      break;
    }
    case kMsgIdLocalPositionNed: {
      const LocalPositionNed lp = decode_local_position_ned(f);
      if (is_finite_f32(lp.x) && is_finite_f32(lp.y) && is_finite_f32(lp.z) &&
          is_finite_f32(lp.vx) && is_finite_f32(lp.vy) && is_finite_f32(lp.vz)) {
        state_.pos_ned = {lp.x, lp.y, lp.z};
        state_.vel_ned = {lp.vx, lp.vy, lp.vz};
      }
      break;
    }
    default:
      break;
  }

  out.action = ShieldAction::kForwardedNominal;
  out.to_offboard.emplace_back(wire_bytes.begin(), wire_bytes.begin() + static_cast<std::ptrdiff_t>(dec.bytes_consumed));
  return out;
}

ShieldStepOutput OffboardRtaShield::process_offboard_packet(
    std::span<const uint8_t> wire_bytes, uint64_t now_us) {
  ShieldStepOutput out{};
  const DecodeResult dec = decode_frame(wire_bytes);
  if (!dec.ok()) {
    if (dec.error == DecodeError::kBadCrc) {
      ++stats_.crc_rejected;
    } else {
      ++stats_.malformed_rejected;
    }
    out.action = ShieldAction::kRejectedInvalidFrame;
    return out;
  }

  ++stats_.offboard_rx_valid;
  const MavlinkV2Frame& f = dec.frame;

  // 1. Intercept COMMAND_LONG (hazardous disarm / flight termination checks).
  if (f.msgid == kMsgIdCommandLong) {
    const CommandLong cmd = decode_command_long(f);
    const bool is_airborne = state_.pos_ned.z < -cfg_.airborne_disarm_agl_m;
    const bool is_airborne_disarm =
        (cmd.command == kMavCmdComponentArmDisarm) &&
        is_finite_f32(cmd.param1) && (std::abs(cmd.param1) < 0.5f) &&
        is_airborne;
    const bool is_flight_termination = (cmd.command == kMavCmdDoFlightTermination);

    if (is_airborne_disarm || is_flight_termination) {
      ++stats_.commands_denied;
      out.action = ShieldAction::kDeniedCommand;
      out.to_offboard.push_back(
          make_command_ack_packet(cmd.command, kMavResultDenied, f.sysid, f.compid));
      out.to_offboard.push_back(make_statustext_packet(
          kMavSeverityCritical,
          is_airborne_disarm ? "RTA DENY: AIRBORNE DISARM BLOCKED"
                             : "RTA DENY: FLIGHT TERMINATION BLOCKED"));
      return out;
    }

    ++stats_.commands_passed;
    out.action = ShieldAction::kForwardedNominal;
    out.to_fcu.emplace_back(wire_bytes.begin(), wire_bytes.begin() + static_cast<std::ptrdiff_t>(dec.bytes_consumed));
    return out;
  }

  // 2. Intercept SET_POSITION_TARGET_LOCAL_NED through Deadman + CbvfSafetyFilter + NFZ Geofence.
  if (f.msgid == kMsgIdSetPositionTargetLocalNed) {
    const SetPositionTargetLocalNed raw = decode_set_position_target_local_ned(f);

    // 2a. Fail-closed deadman timeout (> 500 ms stale FCU heartbeat -> zero-velocity hover).
    if (is_deadman_expired(now_us)) {
      ++stats_.deadman_overrides;
      out.action = ShieldAction::kDeadmanHoverOverride;
      out.to_fcu.push_back(make_hover_setpoint_packet(raw));
      out.to_offboard.push_back(
          make_statustext_packet(kMavSeverityCritical, "RTA DEADMAN: FCU HB TIMEOUT -> ZERO-VEL HOVER"));
      return out;
    }

    // 2b. Reject non-finite payloads fail-closed into zero-velocity hover.
    const bool all_finite =
        is_finite_f32(raw.x) && is_finite_f32(raw.y) && is_finite_f32(raw.z) &&
        is_finite_f32(raw.vx) && is_finite_f32(raw.vy) && is_finite_f32(raw.vz) &&
        is_finite_f32(raw.afx) && is_finite_f32(raw.afy) && is_finite_f32(raw.afz) &&
        is_finite_f32(raw.yaw) && is_finite_f32(raw.yaw_rate);
    if (!all_finite) {
      ++stats_.setpoints_clamped;
      ++stats_.setpoints_clamped_plunge_speed;
      out.action = ShieldAction::kClampedSafeSetpoint;
      out.to_fcu.push_back(make_hover_setpoint_packet(raw));
      out.to_offboard.push_back(
          make_statustext_packet(kMavSeverityCritical, "RTA CLAMP: NON-FINITE SETPOINT -> HOVER"));
      return out;
    }

    SetPositionTargetLocalNed safe = raw;
    bool clamped = false;
    bool clamped_plunge_speed = false;
    bool clamped_nfz = false;
    bool clamped_cbvf = false;

    const bool use_pos = (raw.type_mask & (kTypeMaskIgnoreX | kTypeMaskIgnoreY | kTypeMaskIgnoreZ)) !=
                         (kTypeMaskIgnoreX | kTypeMaskIgnoreY | kTypeMaskIgnoreZ);
    const bool use_vel = (raw.type_mask & (kTypeMaskIgnoreVx | kTypeMaskIgnoreVy | kTypeMaskIgnoreVz)) !=
                         (kTypeMaskIgnoreVx | kTypeMaskIgnoreVy | kTypeMaskIgnoreVz);
    const bool use_acc = (raw.type_mask & (kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz)) !=
                         (kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz);

    // Derive desired velocity and nominal acceleration in NED.
    core::Vec3Ned v_cmd = state_.vel_ned;
    if (use_vel) {
      v_cmd = {
          ((raw.type_mask & kTypeMaskIgnoreVx) == 0u) ? static_cast<double>(raw.vx) : state_.vel_ned.x,
          ((raw.type_mask & kTypeMaskIgnoreVy) == 0u) ? static_cast<double>(raw.vy) : state_.vel_ned.y,
          ((raw.type_mask & kTypeMaskIgnoreVz) == 0u) ? static_cast<double>(raw.vz) : state_.vel_ned.z,
      };
    } else if (use_pos) {
      v_cmd = {
          ((raw.type_mask & kTypeMaskIgnoreX) == 0u) ? (static_cast<double>(raw.x) - state_.pos_ned.x) : 0.0,
          ((raw.type_mask & kTypeMaskIgnoreY) == 0u) ? (static_cast<double>(raw.y) - state_.pos_ned.y) : 0.0,
          ((raw.type_mask & kTypeMaskIgnoreZ) == 0u) ? (static_cast<double>(raw.z) - state_.pos_ned.z) : 0.0,
      };
    }

    core::Vec3Ned a_nom{
        cfg_.vel_kp_s * (v_cmd.x - state_.vel_ned.x),
        cfg_.vel_kp_s * (v_cmd.y - state_.vel_ned.y),
        cfg_.vel_kp_s * (v_cmd.z - state_.vel_ned.z),
    };
    if (use_acc) {
      if ((raw.type_mask & kTypeMaskIgnoreAfx) == 0u) a_nom.x = static_cast<double>(raw.afx);
      if ((raw.type_mask & kTypeMaskIgnoreAfy) == 0u) a_nom.y = static_cast<double>(raw.afy);
      if ((raw.type_mask & kTypeMaskIgnoreAfz) == 0u) a_nom.z = static_cast<double>(raw.afz);
    } else {
      // Saturate nominal velocity-tracker acceleration to the vehicle's feasible tilt-cone
      // so benign step-velocity commands don't trip actuator-cone clamping unless a hazard/envelope triggers.
      const double a_nom_xy_lim =
          0.90 * std::min(static_cast<double>(state_.envelope.a_xy_max) * 0.9238795,
                          core::params::kGravity * std::tan(static_cast<double>(state_.envelope.tilt_max_rad)));
      const double a_nom_xy = std::hypot(a_nom.x, a_nom.y);
      if (a_nom_xy > a_nom_xy_lim && a_nom_xy > 1e-6) {
        const double s = a_nom_xy_lim / a_nom_xy;
        a_nom.x *= s;
        a_nom.y *= s;
      }
      a_nom.z = std::clamp(a_nom.z, -3.0, 3.0);
    }

    // 2c. Run real VertiRTA CbvfSafetyFilter::step_100hz over active Istanbul NFZ + terrain hazards.
    const auto hazards = build_active_hazards();
    const double yaw_rate_nom =
        ((raw.type_mask & kTypeMaskIgnoreYawRate) == 0u) ? static_cast<double>(raw.yaw_rate) : 0.0;
    const rta::SafetyFilterOutput sf = cbvf_filter_.step_100hz(
        state_.pos_ned, state_.vel_ned, a_nom, yaw_rate_nom, state_.envelope, hazards);
    out.last_cbvf = sf;
    out.cbvf_evaluated = true;

    const core::Vec3Ned da = sf.a_safe_ned - a_nom;
    if (sf.used_hj_fallback || da.norm() > 0.35) {
      clamped = true;
      clamped_cbvf = true;
      // Project safe velocity from CBVF-filtered acceleration.
      v_cmd = {
          state_.vel_ned.x + sf.a_safe_ned.x * cfg_.lookahead_dt_s,
          state_.vel_ned.y + sf.a_safe_ned.y * cfg_.lookahead_dt_s,
          state_.vel_ned.z + sf.a_safe_ned.z * cfg_.lookahead_dt_s,
      };
    }

    // 2d. Velocity, descent/plunge, and altitude ceiling/floor envelope clamps.
    const double v_xy = std::hypot(v_cmd.x, v_cmd.y);
    if (v_xy > cfg_.max_horiz_speed_mps) {
      const double scale = cfg_.max_horiz_speed_mps / v_xy;
      v_cmd.x *= scale;
      v_cmd.y *= scale;
      clamped = true;
      clamped_plunge_speed = true;
    }
    if (v_cmd.z > cfg_.max_descent_rate_mps) {
      v_cmd.z = cfg_.max_descent_rate_mps;
      clamped = true;
      clamped_plunge_speed = true;
    } else if (v_cmd.z < -cfg_.max_ascent_rate_mps) {
      v_cmd.z = -cfg_.max_ascent_rate_mps;
      clamped = true;
      clamped_plunge_speed = true;
    }

    // Check target position Z against terrain floor / urban ceiling.
    if ((raw.type_mask & kTypeMaskIgnoreZ) == 0u) {
      const double z_min_ned = -cfg_.max_agl_m;
      const double z_max_ned = -cfg_.min_agl_m;
      if (static_cast<double>(raw.z) > z_max_ned) {
        safe.z = static_cast<float>(z_max_ned);
        v_cmd.z = std::min(v_cmd.z, 0.0);
        clamped = true;
        clamped_plunge_speed = true;
      } else if (static_cast<double>(raw.z) < z_min_ned) {
        safe.z = static_cast<float>(z_min_ned);
        v_cmd.z = std::max(v_cmd.z, 0.0);
        clamped = true;
        clamped_plunge_speed = true;
      }
    }
    // Prevent descent into floor when already near min_agl_m.
    if (state_.pos_ned.z + v_cmd.z * cfg_.lookahead_dt_s > -cfg_.min_agl_m) {
      v_cmd.z = std::min(v_cmd.z, (-cfg_.min_agl_m - state_.pos_ned.z) / cfg_.lookahead_dt_s);
      clamped = true;
      clamped_plunge_speed = true;
    }

    // 2e. Hard Istanbul NFZ geofence checks (LTFJ CTR, Kartal Hospital helipad, LTBX CTR).
    const core::Vec3Ned predicted_pos{
        state_.pos_ned.x + v_cmd.x * (2.0 * cfg_.lookahead_dt_s),
        state_.pos_ned.y + v_cmd.y * (2.0 * cfg_.lookahead_dt_s),
        state_.pos_ned.z + v_cmd.z * (2.0 * cfg_.lookahead_dt_s),
    };
    core::Vec3Ned target_pos = state_.pos_ned;
    if ((raw.type_mask & kTypeMaskIgnoreX) == 0u) target_pos.x = static_cast<double>(raw.x);
    if ((raw.type_mask & kTypeMaskIgnoreY) == 0u) target_pos.y = static_cast<double>(raw.y);
    if ((raw.type_mask & kTypeMaskIgnoreZ) == 0u) target_pos.z = static_cast<double>(safe.z);

    if ((use_pos && violates_istanbul_nfz(target_pos)) || violates_istanbul_nfz(predicted_pos)) {
      clamped = true;
      clamped_nfz = true;

      // Project velocity strictly outward from any threatened NFZ boundary.
      const double ltfj_safe_east = ltfj_wall_east_m_ - cfg_.ltfj_buffer_m;
      if (predicted_pos.y >= ltfj_safe_east || target_pos.y >= ltfj_safe_east) {
        if (v_cmd.y > 0.0) {
          v_cmd.y = -std::min(2.0, std::abs(v_cmd.y));  // Westward escape away from LTFJ CTR
        }
        target_pos.y = std::min(target_pos.y, ltfj_safe_east - 5.0);
      }

      const double hosp_dn = predicted_pos.x - hospital_ned_.x;
      const double hosp_de = predicted_pos.y - hospital_ned_.y;
      const double hosp_r = std::hypot(hosp_dn, hosp_de);
      const double hosp_t_r = std::hypot(target_pos.x - hospital_ned_.x, target_pos.y - hospital_ned_.y);
      if (hosp_r <= cfg_.hospital_nfz_radius_m || hosp_t_r <= cfg_.hospital_nfz_radius_m) {
        const double dn0 = state_.pos_ned.x - hospital_ned_.x;
        const double de0 = state_.pos_ned.y - hospital_ned_.y;
        const double r0 = std::max(1.0, std::hypot(dn0, de0));
        const double un = dn0 / r0;
        const double ue = de0 / r0;
        const double radial_v = v_cmd.x * un + v_cmd.y * ue;
        if (radial_v < 0.0) {
          // Strip inward component and push gently outward.
          v_cmd.x -= (radial_v - 1.0) * un;
          v_cmd.y -= (radial_v - 1.0) * ue;
        }
        target_pos.x = hospital_ned_.x + un * (cfg_.hospital_nfz_radius_m + 15.0);
        target_pos.y = hospital_ned_.y + ue * (cfg_.hospital_nfz_radius_m + 15.0);
      }

      const double ltbx_dn = predicted_pos.x - ltbx_arp_ned_.x;
      const double ltbx_de = predicted_pos.y - ltbx_arp_ned_.y;
      const double ltbx_r = std::hypot(ltbx_dn, ltbx_de);
      const double ltbx_t_r = std::hypot(target_pos.x - ltbx_arp_ned_.x, target_pos.y - ltbx_arp_ned_.y);
      if (ltbx_r <= cfg_.ltbx_ctr_radius_m || ltbx_t_r <= cfg_.ltbx_ctr_radius_m) {
        const double dn0 = state_.pos_ned.x - ltbx_arp_ned_.x;
        const double de0 = state_.pos_ned.y - ltbx_arp_ned_.y;
        const double r0 = std::max(1.0, std::hypot(dn0, de0));
        const double un = dn0 / r0;
        const double ue = de0 / r0;
        const double radial_v = v_cmd.x * un + v_cmd.y * ue;
        if (radial_v < 0.0) {
          v_cmd.x -= (radial_v - 1.0) * un;
          v_cmd.y -= (radial_v - 1.0) * ue;
        }
        target_pos.x = ltbx_arp_ned_.x + un * (cfg_.ltbx_ctr_radius_m + 25.0);
        target_pos.y = ltbx_arp_ned_.y + ue * (cfg_.ltbx_ctr_radius_m + 25.0);
      }

      // Final speed clamp after NFZ outward projection.
      const double v_xy2 = std::hypot(v_cmd.x, v_cmd.y);
      if (v_xy2 > cfg_.max_horiz_speed_mps) {
        const double s2 = cfg_.max_horiz_speed_mps / v_xy2;
        v_cmd.x *= s2;
        v_cmd.y *= s2;
      }
    }

    if (!clamped) {
      ++stats_.setpoints_passed;
      out.action = ShieldAction::kForwardedNominal;
      out.to_fcu.emplace_back(wire_bytes.begin(), wire_bytes.begin() + static_cast<std::ptrdiff_t>(dec.bytes_consumed));
      return out;
    }

    ++stats_.setpoints_clamped;
    if (clamped_cbvf) ++stats_.setpoints_clamped_cbvf;
    if (clamped_nfz) ++stats_.setpoints_clamped_nfz;
    if (clamped_plunge_speed) ++stats_.setpoints_clamped_plunge_speed;

    // Emit safe clamped velocity-controlled setpoint (guaranteeing FCU executes safe projected velocity).
    safe.x = static_cast<float>(target_pos.x);
    safe.y = static_cast<float>(target_pos.y);
    safe.z = static_cast<float>(std::clamp(target_pos.z, -cfg_.max_agl_m, -cfg_.min_agl_m));
    safe.vx = static_cast<float>(v_cmd.x);
    safe.vy = static_cast<float>(v_cmd.y);
    safe.vz = static_cast<float>(std::clamp(v_cmd.z, -cfg_.max_ascent_rate_mps, cfg_.max_descent_rate_mps));
    safe.afx = static_cast<float>(sf.a_safe_ned.x);
    safe.afy = static_cast<float>(sf.a_safe_ned.y);
    safe.afz = static_cast<float>(sf.a_safe_ned.z);
    safe.yaw_rate = static_cast<float>(sf.yaw_rate_safe);
    safe.type_mask = kTypeMaskVelocityOnly;
    safe.coordinate_frame = kMavFrameLocalNed;

    std::array<uint8_t, kMaxFrameBytes> buf{};
    const std::size_t n = encode_set_position_target_local_ned(
        safe, tx_seq_++, cfg_.bridge_sysid, cfg_.bridge_compid, buf);
    out.action = ShieldAction::kClampedSafeSetpoint;
    out.to_fcu.push_back(to_vec(buf, n));

    const char* alert_txt = "RTA CLAMP: CBVF SAFETY PROJECTION APPLIED";
    if (clamped_nfz) {
      alert_txt = "RTA CLAMP: ISTANBUL NFZ BREACH BLOCKED";
    } else if (clamped_plunge_speed) {
      alert_txt = "RTA CLAMP: PLUNGE/OVERSPEED ENVELOPE LIMIT";
    }
    out.to_offboard.push_back(make_statustext_packet(kMavSeverityWarning, alert_txt));
    return out;
  }

  // Other valid MAVLink v2 messages (e.g., offboard HEARTBEAT) are forwarded intact.
  out.action = ShieldAction::kForwardedNominal;
  out.to_fcu.emplace_back(wire_bytes.begin(), wire_bytes.begin() + static_cast<std::ptrdiff_t>(dec.bytes_consumed));
  return out;
}

}  // namespace vertirta::mavlink
