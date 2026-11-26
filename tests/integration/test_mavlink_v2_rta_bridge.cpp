// SPDX-License-Identifier: Apache-2.0
// Catch2 verification suite for Zero-GPL Native C++20 MAVLink v2 Codec, Offboard RTA Shield, & Live UDP Loopback.

#include <netinet/in.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <thread>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "vertirta/mavlink/mavlink_v2_codec.hpp"
#include "vertirta/mavlink/offboard_rta_shield.hpp"
#include "vertirta/mavlink/udp_bridge.hpp"

using namespace vertirta::mavlink;

TEST_CASE("MAVLink v2 golden CRC-16/MCRF4XX, CRC_EXTRA, and trailing-zero truncation", "[mavlink_v2]") {
  // 1. Verify official common.xml CRC_EXTRA lookup for all 8 supported messages.
  REQUIRE(crc_extra_for_msgid(kMsgIdHeartbeat) == 50u);
  REQUIRE(crc_extra_for_msgid(kMsgIdSysStatus) == 124u);
  REQUIRE(crc_extra_for_msgid(kMsgIdAttitudeQuaternion) == 246u);
  REQUIRE(crc_extra_for_msgid(kMsgIdLocalPositionNed) == 185u);
  REQUIRE(crc_extra_for_msgid(kMsgIdCommandLong) == 152u);
  REQUIRE(crc_extra_for_msgid(kMsgIdCommandAck) == 143u);
  REQUIRE(crc_extra_for_msgid(kMsgIdSetPositionTargetLocalNed) == 143u);
  REQUIRE(crc_extra_for_msgid(kMsgIdStatusText) == 83u);
  REQUIRE_FALSE(crc_extra_for_msgid(999999u).has_value());

  // 2. Trailing-zero truncation rule: all-zero payload MUST retain exactly 1 byte (0x00).
  const std::array<uint8_t, 9> all_zeros{};
  REQUIRE(truncated_payload_length(all_zeros) == 1u);

  // 3. Golden HEARTBEAT wire-byte verification (verified independently against MAVLink v2 X.25 spec):
  //    custom_mode=0, type=2, autopilot=12, base_mode=0x80, system_status=4, mavlink_version=3
  //    seq=7, sysid=1, compid=1, msgid=0 -> 9 non-zero-tail bytes -> total 21 wire bytes.
  Heartbeat hb{};
  hb.custom_mode = 0u;
  hb.type = 2u;
  hb.autopilot = 12u;
  hb.base_mode = kMavModeFlagSafetyArmed;
  hb.system_status = 4u;
  hb.mavlink_version = 3u;

  std::array<uint8_t, kMaxFrameBytes> wire{};
  const std::size_t n = encode_heartbeat(hb, 7u, 1u, 1u, wire);
  REQUIRE(n == 21u);
  REQUIRE(wire[0] == 0xFDu);
  REQUIRE(wire[1] == 9u);
  REQUIRE(wire[2] == 0u);
  REQUIRE(wire[3] == 0u);
  REQUIRE(wire[4] == 7u);
  REQUIRE(wire[5] == 1u);
  REQUIRE(wire[6] == 1u);
  REQUIRE(wire[7] == 0u);
  REQUIRE(wire[8] == 0u);
  REQUIRE(wire[9] == 0u);
  // Payload bytes [10..18]:
  REQUIRE(wire[10] == 0x00u);
  REQUIRE(wire[14] == 2u);
  REQUIRE(wire[15] == 12u);
  REQUIRE(wire[16] == 0x80u);
  REQUIRE(wire[17] == 4u);
  REQUIRE(wire[18] == 3u);

  // Independent X.25 CRC-16 check over wire[1..18] + CRC_EXTRA=50:
  uint16_t expected_crc = crc16_init();
  for (std::size_t i = 1u; i < 19u; ++i) {
    crc16_accumulate(wire[i], expected_crc);
  }
  crc16_accumulate(50u, expected_crc);
  REQUIRE(wire[19] == static_cast<uint8_t>(expected_crc & 0xFFu));
  REQUIRE(wire[20] == static_cast<uint8_t>((expected_crc >> 8u) & 0xFFu));

  const DecodeResult dec = decode_frame(std::span<const uint8_t>(wire.data(), n));
  REQUIRE(dec.ok());
  REQUIRE(dec.bytes_consumed == n);
  const Heartbeat hb2 = decode_heartbeat(dec.frame);
  REQUIRE(hb2.custom_mode == 0u);
  REQUIRE(hb2.type == 2u);
  REQUIRE(hb2.autopilot == 12u);
  REQUIRE(hb2.base_mode == kMavModeFlagSafetyArmed);
  REQUIRE(hb2.system_status == 4u);
  REQUIRE(hb2.mavlink_version == 3u);

  // 4. Corrupted CRC & bad STX & truncated frame MUST fail closed.
  wire[19] ^= 0x01u;
  REQUIRE(decode_frame(std::span<const uint8_t>(wire.data(), n)).error == DecodeError::kBadCrc);
  wire[19] ^= 0x01u;
  wire[0] = 0xFEu;  // MAVLink v1 STX rejected
  REQUIRE(decode_frame(std::span<const uint8_t>(wire.data(), n)).error == DecodeError::kBadMagic);
  wire[0] = 0xFDu;
  REQUIRE(decode_frame(std::span<const uint8_t>(wire.data(), n - 1u)).error == DecodeError::kTruncatedFrame);
}

TEST_CASE("MAVLink v2 round-trip and trailing-zero zero-extension for all 8 message types", "[mavlink_v2]") {
  std::array<uint8_t, kMaxFrameBytes> wire{};

  // #1 SYS_STATUS (with trailing zeros truncated and zero-extended back)
  {
    SysStatus ss{};
    ss.onboard_control_sensors_present = 0x12345678u;
    ss.voltage_battery = 40320u;
    ss.current_battery = 920;
    ss.battery_remaining = 0;  // trailing zero -> truncated!
    const std::size_t n = encode_sys_status(ss, 1u, 1u, 1u, wire);
    REQUIRE(n < kHeaderBytes + kPayloadLenSysStatus + kChecksumBytes);
    const DecodeResult dec = decode_frame(std::span<const uint8_t>(wire.data(), n));
    REQUIRE(dec.ok());
    const SysStatus out = decode_sys_status(dec.frame);
    REQUIRE(out.onboard_control_sensors_present == 0x12345678u);
    REQUIRE(out.voltage_battery == 40320u);
    REQUIRE(out.current_battery == 920);
    REQUIRE(out.battery_remaining == 0);
  }

  // #31 ATTITUDE_QUATERNION (v2 extension fields)
  {
    AttitudeQuaternion aq{};
    aq.time_boot_ms = 42000u;
    aq.q1 = 0.9659f;
    aq.q2 = 0.0f;
    aq.q3 = 0.0f;
    aq.q4 = 0.2588f;
    aq.yawspeed = 0.15f;
    aq.repr_offset_q = {1.0f, 0.0f, 0.0f, 0.0f};
    const std::size_t n = encode_attitude_quaternion(aq, 2u, 1u, 1u, wire, true);
    const DecodeResult dec = decode_frame(std::span<const uint8_t>(wire.data(), n));
    REQUIRE(dec.ok());
    const AttitudeQuaternion out = decode_attitude_quaternion(dec.frame);
    REQUIRE(out.time_boot_ms == 42000u);
    REQUIRE(out.q1 == Catch::Approx(0.9659f).margin(1e-5f));
    REQUIRE(out.q4 == Catch::Approx(0.2588f).margin(1e-5f));
    REQUIRE(out.repr_offset_q[0] == Catch::Approx(1.0f));
  }

  // #32 LOCAL_POSITION_NED
  {
    LocalPositionNed lp{12345u, 500.5f, -400.25f, -60.0f, 8.5f, -1.25f, 0.5f};
    const std::size_t n = encode_local_position_ned(lp, 3u, 1u, 1u, wire);
    const DecodeResult dec = decode_frame(std::span<const uint8_t>(wire.data(), n));
    REQUIRE(dec.ok());
    const LocalPositionNed out = decode_local_position_ned(dec.frame);
    REQUIRE(out.time_boot_ms == 12345u);
    REQUIRE(out.x == Catch::Approx(500.5f));
    REQUIRE(out.y == Catch::Approx(-400.25f));
    REQUIRE(out.z == Catch::Approx(-60.0f));
    REQUIRE(out.vx == Catch::Approx(8.5f));
  }

  // #76 COMMAND_LONG & #77 COMMAND_ACK
  {
    CommandLong cmd{};
    cmd.command = kMavCmdComponentArmDisarm;
    cmd.param1 = 1.0f;
    cmd.target_system = 1u;
    cmd.target_component = 1u;
    const std::size_t n = encode_command_long(cmd, 4u, 255u, 190u, wire);
    const DecodeResult dec = decode_frame(std::span<const uint8_t>(wire.data(), n));
    REQUIRE(dec.ok());
    const CommandLong out = decode_command_long(dec.frame);
    REQUIRE(out.command == kMavCmdComponentArmDisarm);
    REQUIRE(out.param1 == Catch::Approx(1.0f));

    CommandAck ack{kMavCmdComponentArmDisarm, kMavResultDenied, 0u, -1, 255u, 190u};
    const std::size_t na = encode_command_ack(ack, 5u, 1u, 191u, wire, true);
    const DecodeResult deca = decode_frame(std::span<const uint8_t>(wire.data(), na));
    REQUIRE(deca.ok());
    const CommandAck ack_out = decode_command_ack(deca.frame);
    REQUIRE(ack_out.command == kMavCmdComponentArmDisarm);
    REQUIRE(ack_out.result == kMavResultDenied);
    REQUIRE(ack_out.result_param2 == -1);
    REQUIRE(ack_out.target_system == 255u);
  }

  // #84 SET_POSITION_TARGET_LOCAL_NED & #253 STATUSTEXT
  {
    SetPositionTargetLocalNed sp{};
    sp.time_boot_ms = 999u;
    sp.vx = 6.0f;
    sp.vy = -2.0f;
    sp.vz = -0.5f;
    sp.type_mask = kTypeMaskVelocityOnly;
    const std::size_t n = encode_set_position_target_local_ned(sp, 6u, 255u, 190u, wire);
    const DecodeResult dec = decode_frame(std::span<const uint8_t>(wire.data(), n));
    REQUIRE(dec.ok());
    const SetPositionTargetLocalNed out = decode_set_position_target_local_ned(dec.frame);
    REQUIRE(out.vx == Catch::Approx(6.0f));
    REQUIRE(out.vy == Catch::Approx(-2.0f));
    REQUIRE(out.type_mask == kTypeMaskVelocityOnly);

    StatusText st{};
    st.severity = kMavSeverityCritical;
    st.set_text("RTA TEST ALERT");
    const std::size_t ns = encode_statustext(st, 7u, 1u, 191u, wire, false);
    const DecodeResult decs = decode_frame(std::span<const uint8_t>(wire.data(), ns));
    REQUIRE(decs.ok());
    const StatusText stout = decode_statustext(decs.frame);
    REQUIRE(stout.severity == kMavSeverityCritical);
    REQUIRE(stout.text_view() == "RTA TEST ALERT");
  }
}

TEST_CASE("OffboardRtaShield enforces deadman, CBVF/plunge/NFZ clamping, and airborne disarm denial", "[mavlink_rta_shield]") {
  OffboardRtaShield shield;
  std::array<uint8_t, kMaxFrameBytes> buf{};

  // Feed fresh FCU HEARTBEAT + LOCAL_POSITION_NED (60m AGL, west of LTFJ).
  auto feed_fcu = [&](uint64_t t_us, float x, float y, float z) {
    Heartbeat hb{};
    hb.base_mode = kMavModeFlagSafetyArmed;
    const std::size_t nhb = encode_heartbeat(hb, 1u, 1u, 1u, buf);
    REQUIRE(shield.process_fcu_packet(std::span<const uint8_t>(buf.data(), nhb), t_us).action ==
            ShieldAction::kForwardedNominal);

    LocalPositionNed lp{static_cast<uint32_t>(t_us / 1000u), x, y, z, 2.0f, 0.0f, 0.0f};
    const std::size_t nlp = encode_local_position_ned(lp, 2u, 1u, 1u, buf);
    REQUIRE(shield.process_fcu_packet(std::span<const uint8_t>(buf.data(), nlp), t_us).action ==
            ShieldAction::kForwardedNominal);
  };

  feed_fcu(1'000'000u, 500.0f, -400.0f, -60.0f);

  // 1. Nominal setpoint passes cleanly.
  {
    SetPositionTargetLocalNed sp{};
    sp.vx = 5.0f;
    sp.vy = -1.0f;
    sp.vz = 0.0f;
    sp.type_mask = kTypeMaskVelocityOnly;
    const std::size_t n = encode_set_position_target_local_ned(sp, 10u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'050'000u);
    REQUIRE(out.action == ShieldAction::kForwardedNominal);
    REQUIRE(out.to_fcu.size() == 1u);
    REQUIRE(out.cbvf_evaluated);
  }

  // 2. Plunge & overspeed setpoint clamped + STATUSTEXT emitted.
  {
    SetPositionTargetLocalNed sp{};
    sp.vx = 60.0f;
    sp.vz = 25.0f;
    sp.type_mask = kTypeMaskVelocityOnly;
    const std::size_t n = encode_set_position_target_local_ned(sp, 11u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'060'000u);
    REQUIRE(out.action == ShieldAction::kClampedSafeSetpoint);
    REQUIRE(out.to_fcu.size() == 1u);
    REQUIRE(out.to_offboard.size() == 1u);
    const DecodeResult dec = decode_frame(out.to_fcu[0]);
    REQUIRE(dec.ok());
    const SetPositionTargetLocalNed clamped = decode_set_position_target_local_ned(dec.frame);
    REQUIRE(std::hypot(clamped.vx, clamped.vy) <= shield.config().max_horiz_speed_mps + 1e-3);
    REQUIRE(clamped.vz <= shield.config().max_descent_rate_mps + 1e-3);
  }

  // 3. Airborne disarm (z = -60m < -2.0m) DENIED with COMMAND_ACK (MAV_RESULT_DENIED) + STATUSTEXT CRITICAL.
  {
    CommandLong disarm{};
    disarm.command = kMavCmdComponentArmDisarm;
    disarm.param1 = 0.0f;
    const std::size_t n = encode_command_long(disarm, 12u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'070'000u);
    REQUIRE(out.action == ShieldAction::kDeniedCommand);
    REQUIRE(out.to_fcu.empty());
    REQUIRE(out.to_offboard.size() == 2u);
    const DecodeResult ack_dec = decode_frame(out.to_offboard[0]);
    REQUIRE(ack_dec.ok());
    REQUIRE(decode_command_ack(ack_dec.frame).result == kMavResultDenied);
    const DecodeResult st_dec = decode_frame(out.to_offboard[1]);
    REQUIRE(st_dec.ok());
    REQUIRE(decode_statustext(st_dec.frame).severity == kMavSeverityCritical);
  }

  // 4. Flight termination DENIED unconditionally.
  {
    CommandLong fterm{};
    fterm.command = kMavCmdDoFlightTermination;
    fterm.param1 = 1.0f;
    const std::size_t n = encode_command_long(fterm, 13u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'080'000u);
    REQUIRE(out.action == ShieldAction::kDeniedCommand);
    REQUIRE(out.to_fcu.empty());
  }

  // 5. LTFJ CTR eastern breach blocked.
  feed_fcu(1'100'000u, 500.0f, 95.0f, -60.0f);
  {
    SetPositionTargetLocalNed breach{};
    breach.vx = 0.0f;
    breach.vy = 20.0f;
    breach.vz = 0.0f;
    breach.type_mask = kTypeMaskVelocityOnly;
    const std::size_t n = encode_set_position_target_local_ned(breach, 14u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'120'000u);
    REQUIRE(out.action == ShieldAction::kClampedSafeSetpoint);
    const DecodeResult dec = decode_frame(out.to_fcu[0]);
    REQUIRE(dec.ok());
    REQUIRE(decode_set_position_target_local_ned(dec.frame).vy <= 0.0f);
  }

  // 6. Deadman heartbeat timeout (> 500 ms stale) forces fail-closed zero-velocity hover + CRITICAL STATUSTEXT.
  {
    SetPositionTargetLocalNed sp{};
    sp.vx = 10.0f;
    sp.vy = 5.0f;
    sp.vz = 1.0f;
    sp.type_mask = kTypeMaskVelocityOnly;
    const std::size_t n = encode_set_position_target_local_ned(sp, 15u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'700'000u);  // +600 ms stale!
    REQUIRE(out.action == ShieldAction::kDeadmanHoverOverride);
    REQUIRE(out.to_fcu.size() == 1u);
    const DecodeResult dec = decode_frame(out.to_fcu[0]);
    REQUIRE(dec.ok());
    const SetPositionTargetLocalNed hov = decode_set_position_target_local_ned(dec.frame);
    REQUIRE(hov.vx == Catch::Approx(0.0f));
    REQUIRE(hov.vy == Catch::Approx(0.0f));
    REQUIRE(hov.vz == Catch::Approx(0.0f));
    REQUIRE(hov.type_mask == kTypeMaskVelocityOnly);
    REQUIRE(out.to_offboard.size() == 1u);
    REQUIRE(decode_statustext(decode_frame(out.to_offboard[0]).frame).severity == kMavSeverityCritical);
  }

  // 7. Grounded disarm (z = -0.4 m, NOT airborne) is permitted.
  feed_fcu(2'000'000u, 500.0f, -400.0f, -0.4f);
  {
    CommandLong disarm{};
    disarm.command = kMavCmdComponentArmDisarm;
    disarm.param1 = 0.0f;
    const std::size_t n = encode_command_long(disarm, 16u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 2'010'000u);
    REQUIRE(out.action == ShieldAction::kForwardedNominal);
    REQUIRE(out.to_fcu.size() == 1u);
  }
}

TEST_CASE("Live 3-endpoint POSIX UDP loopback stress proof (>= 2,000 packets, 0 drops, 100% interception)", "[mavlink_udp]") {
  const UdpLoopbackStressReport rep = run_udp_loopback_stress_proof(2100u, "");
  REQUIRE(rep.passed);
  REQUIRE(rep.total_offboard_packets_sent == 2100u);
  REQUIRE(rep.valid_packets_dropped == 0u);
  REQUIRE(rep.fcu_unsafe_packets_leaked == 0u);
  REQUIRE(rep.fcu_corrupted_packets_leaked == 0u);
  REQUIRE(rep.shield_stats.crc_rejected == rep.corrupted_crc_sent);
  REQUIRE(rep.shield_stats.deadman_overrides == rep.deadman_timeout_sent);
  REQUIRE(rep.shield_stats.commands_denied == rep.airborne_disarm_sent + rep.flight_term_sent);
  REQUIRE(rep.planner_acks_denied_received == rep.airborne_disarm_sent + rep.flight_term_sent);
  REQUIRE(rep.latency_us.sample_count >= 2000u);
  REQUIRE(rep.latency_us.p99_us < 10'000.0);  // Well under 10 ms even under ASan/UBSan
  REQUIRE(rep.report_sha256.size() == 64u);
}

TEST_CASE("OffboardRtaShield rejects non-LOCAL_NED frames, FORCE_SET bit, NaN/Inf disarm, pre-heartbeat & expired-deadman disarm, and binds FCU socket for LAN hosts",
          "[mavlink_rta_shield][safety_hardening]") {
  const float qnan = std::numeric_limits<float>::quiet_NaN();
  const float pinf = std::numeric_limits<float>::infinity();
  const float ninf = -std::numeric_limits<float>::infinity();
  std::array<uint8_t, kMaxFrameBytes> buf{};

  // 1. Pre-heartbeat disarm on brand-new shield MUST fail closed (unknown altitude == protected/airborne!).
  {
    OffboardRtaShield cold_shield;
    CommandLong disarm{};
    disarm.command = kMavCmdComponentArmDisarm;
    disarm.param1 = 0.0f;
    const std::size_t n = encode_command_long(disarm, 1u, 255u, 190u, buf);
    const auto out = cold_shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 100'000u);
    REQUIRE(out.action == ShieldAction::kDeniedCommand);
    REQUIRE(out.to_fcu.empty());
    REQUIRE(out.to_offboard.size() == 2u);
    REQUIRE(decode_command_ack(decode_frame(out.to_offboard[0]).frame).result == kMavResultDenied);
  }

  // 2. Heartbeat-only without LOCAL_POSITION_NED MUST still deny disarm fail-closed.
  {
    OffboardRtaShield hb_only_shield;
    Heartbeat hb{};
    hb.base_mode = kMavModeFlagSafetyArmed;
    const std::size_t nhb = encode_heartbeat(hb, 1u, 1u, 1u, buf);
    REQUIRE(hb_only_shield.process_fcu_packet(std::span<const uint8_t>(buf.data(), nhb), 1'000'000u).action ==
            ShieldAction::kForwardedNominal);

    CommandLong disarm{};
    disarm.command = kMavCmdComponentArmDisarm;
    disarm.param1 = 0.0f;
    const std::size_t n = encode_command_long(disarm, 2u, 255u, 190u, buf);
    const auto out = hb_only_shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'010'000u);
    REQUIRE(out.action == ShieldAction::kDeniedCommand);
    REQUIRE(out.to_fcu.empty());
  }

  OffboardRtaShield shield;
  auto feed_fcu = [&](uint64_t t_us, float x, float y, float z) {
    Heartbeat hb{};
    hb.base_mode = kMavModeFlagSafetyArmed;
    const std::size_t nhb = encode_heartbeat(hb, 1u, 1u, 1u, buf);
    REQUIRE(shield.process_fcu_packet(std::span<const uint8_t>(buf.data(), nhb), t_us).action ==
            ShieldAction::kForwardedNominal);

    LocalPositionNed lp{static_cast<uint32_t>(t_us / 1000u), x, y, z, 2.0f, 0.0f, 0.0f};
    const std::size_t nlp = encode_local_position_ned(lp, 2u, 1u, 1u, buf);
    REQUIRE(shield.process_fcu_packet(std::span<const uint8_t>(buf.data(), nlp), t_us).action ==
            ShieldAction::kForwardedNominal);
  };

  // 3. Airborne & grounded NaN/Inf param1 disarm MUST be denied (never bypassing disarm block!).
  feed_fcu(1'000'000u, 500.0f, -400.0f, -60.0f);
  for (const float bad_p1 : {qnan, pinf, ninf}) {
    CommandLong disarm{};
    disarm.command = kMavCmdComponentArmDisarm;
    disarm.param1 = bad_p1;
    const std::size_t n = encode_command_long(disarm, 3u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 1'020'000u);
    REQUIRE(out.action == ShieldAction::kDeniedCommand);
    REQUIRE(out.to_fcu.empty());
  }

  // 4. Expired deadman after a grounded packet MUST default fail-closed (deny disarm!).
  feed_fcu(2'000'000u, 500.0f, -400.0f, -0.3f);
  {
    // Even while freshly grounded, NaN param1 is still rejected fail-closed!
    CommandLong nan_ground{};
    nan_ground.command = kMavCmdComponentArmDisarm;
    nan_ground.param1 = qnan;
    const std::size_t n_nan = encode_command_long(nan_ground, 4u, 255u, 190u, buf);
    REQUIRE(shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n_nan), 2'010'000u).action ==
            ShieldAction::kDeniedCommand);

    // After > 500 ms deadman expiry, valid param1=0.0f disarm is DENIED because ground state is stale!
    CommandLong stale_disarm{};
    stale_disarm.command = kMavCmdComponentArmDisarm;
    stale_disarm.param1 = 0.0f;
    const std::size_t n_stale = encode_command_long(stale_disarm, 5u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n_stale), 2'600'000u);
    REQUIRE(out.action == ShieldAction::kDeniedCommand);
    REQUIRE(out.to_fcu.empty());
  }

  // 5. Non-LOCAL_NED setpoints (BODY_NED=8, LOCAL_OFFSET_NED=7, GLOBAL=0) clamped fail-closed to LOCAL_NED hover!
  feed_fcu(3'000'000u, 500.0f, 95.0f, -60.0f);  // Right next to LTFJ CTR western wall
  for (const uint8_t bad_frame : {uint8_t{8u}, uint8_t{7u}, uint8_t{9u}, uint8_t{0u}}) {
    SetPositionTargetLocalNed sp{};
    sp.vx = 5.0f;
    sp.vy = 0.0f;
    sp.vz = 0.0f;
    sp.type_mask = kTypeMaskVelocityOnly;
    sp.coordinate_frame = bad_frame;
    const std::size_t n = encode_set_position_target_local_ned(sp, 6u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 3'010'000u);
    REQUIRE(out.action == ShieldAction::kClampedSafeSetpoint);
    REQUIRE(out.to_fcu.size() == 1u);
    const DecodeResult dec = decode_frame(out.to_fcu[0]);
    REQUIRE(dec.ok());
    const SetPositionTargetLocalNed safe = decode_set_position_target_local_ned(dec.frame);
    REQUIRE(safe.coordinate_frame == kMavFrameLocalNed);
    REQUIRE((safe.type_mask & kTypeMaskForce) == 0u);
    REQUIRE(safe.vx == Catch::Approx(0.0f));
    REQUIRE(safe.vy == Catch::Approx(0.0f));
    REQUIRE(safe.vz == Catch::Approx(0.0f));
    REQUIRE(out.to_offboard.size() == 1u);
    REQUIRE(decode_statustext(decode_frame(out.to_offboard[0]).frame).severity == kMavSeverityCritical);
  }

  // 6. FORCE_SET bit (0x0200) MUST be rejected/cleared fail-closed into LOCAL_NED zero-velocity hover!
  feed_fcu(3'100'000u, 500.0f, -400.0f, -60.0f);
  {
    SetPositionTargetLocalNed sp{};
    sp.vx = 4.0f;
    sp.vy = 0.0f;
    sp.vz = 0.0f;
    sp.afx = 2.0f;
    sp.type_mask = static_cast<uint16_t>(kTypeMaskVelocityOnly | kTypeMaskForce);
    sp.coordinate_frame = kMavFrameLocalNed;
    const std::size_t n = encode_set_position_target_local_ned(sp, 7u, 255u, 190u, buf);
    const auto out = shield.process_offboard_packet(std::span<const uint8_t>(buf.data(), n), 3'110'000u);
    REQUIRE(out.action == ShieldAction::kClampedSafeSetpoint);
    REQUIRE(out.to_fcu.size() == 1u);
    const DecodeResult dec = decode_frame(out.to_fcu[0]);
    REQUIRE(dec.ok());
    const SetPositionTargetLocalNed safe = decode_set_position_target_local_ned(dec.frame);
    REQUIRE(safe.coordinate_frame == kMavFrameLocalNed);
    REQUIRE((safe.type_mask & kTypeMaskForce) == 0u);
    REQUIRE(safe.vx == Catch::Approx(0.0f));
    REQUIRE(safe.vy == Catch::Approx(0.0f));
    REQUIRE(safe.vz == Catch::Approx(0.0f));
  }

  // 7. Non-loopback FCU host selects INADDR_ANY (0.0.0.0) bind while loopback preserves INADDR_LOOPBACK.
  {
    const auto lb = select_fcu_socket_bind_addr_host_order("127.0.0.1");
    REQUIRE(lb.has_value());
    REQUIRE(*lb == static_cast<uint32_t>(INADDR_LOOPBACK));

    const auto lb2 = select_fcu_socket_bind_addr_host_order("127.0.1.1");
    REQUIRE(lb2.has_value());
    REQUIRE(*lb2 == static_cast<uint32_t>(INADDR_LOOPBACK));

    const auto lan = select_fcu_socket_bind_addr_host_order("192.168.1.50");
    REQUIRE(lan.has_value());
    REQUIRE(*lan == static_cast<uint32_t>(INADDR_ANY));

    const auto lan2 = select_fcu_socket_bind_addr_host_order("10.42.0.2");
    REQUIRE(lan2.has_value());
    REQUIRE(*lan2 == static_cast<uint32_t>(INADDR_ANY));

    REQUIRE_FALSE(select_fcu_socket_bind_addr_host_order("not-an-ipv4").has_value());

    // Live proxy smoke check: invalid host returns 2; non-loopback LAN IP binds cleanly and exits 0 on stop.
    ShieldConfig proxy_cfg{};
    std::atomic<bool> stop_immediate{true};
    REQUIRE(run_live_udp_proxy(0u, "not-an-ipv4", 14540u, proxy_cfg, &stop_immediate) == 2);
    REQUIRE(run_live_udp_proxy(0u, "192.0.2.1", 14540u, proxy_cfg, &stop_immediate) == 0);
    REQUIRE(run_live_udp_proxy(0u, "127.0.0.1", 14540u, proxy_cfg, &stop_immediate) == 0);
  }
}
