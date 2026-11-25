// SPDX-License-Identifier: Apache-2.0
#include "vertirta/mavlink/mavlink_v2_codec.hpp"

#include <algorithm>
#include <cstring>

namespace vertirta::mavlink {

namespace {

inline void put_u16_le(uint8_t* dst, uint16_t v) noexcept {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8u) & 0xFFu);
}

inline void put_i16_le(uint8_t* dst, int16_t v) noexcept {
  uint16_t u = 0u;
  std::memcpy(&u, &v, sizeof(u));
  put_u16_le(dst, u);
}

inline void put_u32_le(uint8_t* dst, uint32_t v) noexcept {
  dst[0] = static_cast<uint8_t>(v & 0xFFu);
  dst[1] = static_cast<uint8_t>((v >> 8u) & 0xFFu);
  dst[2] = static_cast<uint8_t>((v >> 16u) & 0xFFu);
  dst[3] = static_cast<uint8_t>((v >> 24u) & 0xFFu);
}

inline void put_i32_le(uint8_t* dst, int32_t v) noexcept {
  uint32_t u = 0u;
  std::memcpy(&u, &v, sizeof(u));
  put_u32_le(dst, u);
}

inline void put_f32_le(uint8_t* dst, float v) noexcept {
  uint32_t u = 0u;
  std::memcpy(&u, &v, sizeof(u));
  put_u32_le(dst, u);
}

[[nodiscard]] inline uint16_t get_u16_le(const uint8_t* src) noexcept {
  return static_cast<uint16_t>(
      static_cast<uint16_t>(src[0]) |
      static_cast<uint16_t>(static_cast<uint16_t>(src[1]) << 8u));
}

[[nodiscard]] inline int16_t get_i16_le(const uint8_t* src) noexcept {
  const uint16_t u = get_u16_le(src);
  int16_t v = 0;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

[[nodiscard]] inline uint32_t get_u32_le(const uint8_t* src) noexcept {
  return static_cast<uint32_t>(src[0]) |
         (static_cast<uint32_t>(src[1]) << 8u) |
         (static_cast<uint32_t>(src[2]) << 16u) |
         (static_cast<uint32_t>(src[3]) << 24u);
}

[[nodiscard]] inline int32_t get_i32_le(const uint8_t* src) noexcept {
  const uint32_t u = get_u32_le(src);
  int32_t v = 0;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

[[nodiscard]] inline float get_f32_le(const uint8_t* src) noexcept {
  const uint32_t u = get_u32_le(src);
  float v = 0.0f;
  std::memcpy(&v, &u, sizeof(v));
  return v;
}

template <std::size_t N>
[[nodiscard]] std::array<uint8_t, N> zero_extended_payload(const MavlinkV2Frame& frame) noexcept {
  std::array<uint8_t, N> buf{};
  const std::size_t copy_n = std::min<std::size_t>(frame.payload_len, N);
  if (copy_n > 0u) {
    std::memcpy(buf.data(), frame.payload.data(), copy_n);
  }
  return buf;
}

}  // namespace

void StatusText::set_text(std::string_view msg) noexcept {
  text.fill('\0');
  const std::size_t n = std::min<std::size_t>(msg.size(), text.size());
  if (n > 0u) {
    std::memcpy(text.data(), msg.data(), n);
  }
}

std::string_view StatusText::text_view() const noexcept {
  std::size_t n = 0u;
  while (n < text.size() && text[n] != '\0') {
    ++n;
  }
  return std::string_view(text.data(), n);
}

uint16_t crc16_buffer(std::span<const uint8_t> data, uint16_t seed) noexcept {
  uint16_t crc = seed;
  for (uint8_t b : data) {
    crc16_accumulate(b, crc);
  }
  return crc;
}

std::optional<uint8_t> crc_extra_for_msgid(uint32_t msgid) noexcept {
  switch (msgid) {
    case kMsgIdHeartbeat:
      return kCrcExtraHeartbeat;
    case kMsgIdSysStatus:
      return kCrcExtraSysStatus;
    case kMsgIdAttitudeQuaternion:
      return kCrcExtraAttitudeQuaternion;
    case kMsgIdLocalPositionNed:
      return kCrcExtraLocalPositionNed;
    case kMsgIdCommandLong:
      return kCrcExtraCommandLong;
    case kMsgIdCommandAck:
      return kCrcExtraCommandAck;
    case kMsgIdSetPositionTargetLocalNed:
      return kCrcExtraSetPositionTargetLocalNed;
    case kMsgIdStatusText:
      return kCrcExtraStatusText;
    default:
      return std::nullopt;
  }
}

uint8_t truncated_payload_length(std::span<const uint8_t> full_payload) noexcept {
  if (full_payload.empty()) {
    return 0u;
  }
  std::size_t len = std::min<std::size_t>(full_payload.size(), kMaxPayloadBytes);
  while (len > 1u && full_payload[len - 1u] == 0u) {
    --len;
  }
  return static_cast<uint8_t>(len);
}

std::size_t encode_frame(
    uint8_t seq,
    uint8_t sysid,
    uint8_t compid,
    uint32_t msgid,
    std::span<const uint8_t> full_payload,
    std::span<uint8_t> out_buf,
    uint8_t compat_flags) noexcept {
  const auto crc_extra = crc_extra_for_msgid(msgid);
  if (!crc_extra.has_value() || msgid > 0xFFFFFFu || full_payload.size() > kMaxPayloadBytes) {
    return 0u;
  }

  const uint8_t trunc_len = truncated_payload_length(full_payload);
  const std::size_t total_bytes = kHeaderBytes + static_cast<std::size_t>(trunc_len) + kChecksumBytes;
  if (out_buf.size() < total_bytes) {
    return 0u;
  }

  out_buf[0] = kMavlinkV2Stx;
  out_buf[1] = trunc_len;
  out_buf[2] = 0u;  // incompat_flags (unsigned)
  out_buf[3] = compat_flags;
  out_buf[4] = seq;
  out_buf[5] = sysid;
  out_buf[6] = compid;
  out_buf[7] = static_cast<uint8_t>(msgid & 0xFFu);
  out_buf[8] = static_cast<uint8_t>((msgid >> 8u) & 0xFFu);
  out_buf[9] = static_cast<uint8_t>((msgid >> 16u) & 0xFFu);

  if (trunc_len > 0u) {
    std::memcpy(out_buf.data() + kHeaderBytes, full_payload.data(), trunc_len);
  }

  uint16_t crc = crc16_init();
  for (std::size_t i = 1u; i < kHeaderBytes + static_cast<std::size_t>(trunc_len); ++i) {
    crc16_accumulate(out_buf[i], crc);
  }
  crc16_accumulate(*crc_extra, crc);

  put_u16_le(out_buf.data() + kHeaderBytes + static_cast<std::size_t>(trunc_len), crc);
  return total_bytes;
}

DecodeResult decode_frame(std::span<const uint8_t> wire_bytes) noexcept {
  DecodeResult res{};
  if (wire_bytes.size() < kHeaderBytes + kChecksumBytes) {
    res.error = DecodeError::kTooShort;
    return res;
  }
  if (wire_bytes[0] != kMavlinkV2Stx) {
    res.error = DecodeError::kBadMagic;
    return res;
  }

  const uint8_t len = wire_bytes[1];
  const uint8_t incompat = wire_bytes[2];
  const uint8_t compat = wire_bytes[3];
  const uint8_t seq = wire_bytes[4];
  const uint8_t sysid = wire_bytes[5];
  const uint8_t compid = wire_bytes[6];
  const uint32_t msgid =
      static_cast<uint32_t>(wire_bytes[7]) |
      (static_cast<uint32_t>(wire_bytes[8]) << 8u) |
      (static_cast<uint32_t>(wire_bytes[9]) << 16u);

  if ((incompat & static_cast<uint8_t>(~kIncompatFlagSigned)) != 0u) {
    res.error = DecodeError::kUnsupportedIncompatFlags;
    return res;
  }
  const bool is_signed = (incompat & kIncompatFlagSigned) != 0u;
  const std::size_t expected_total =
      kHeaderBytes + static_cast<std::size_t>(len) + kChecksumBytes +
      (is_signed ? kSignatureBytes : 0u);
  if (wire_bytes.size() < expected_total) {
    res.error = DecodeError::kTruncatedFrame;
    return res;
  }

  const auto crc_extra = crc_extra_for_msgid(msgid);
  if (!crc_extra.has_value()) {
    res.error = DecodeError::kUnknownMsgId;
    return res;
  }

  uint16_t computed_crc = crc16_init();
  for (std::size_t i = 1u; i < kHeaderBytes + static_cast<std::size_t>(len); ++i) {
    crc16_accumulate(wire_bytes[i], computed_crc);
  }
  crc16_accumulate(*crc_extra, computed_crc);

  const uint16_t wire_crc = get_u16_le(wire_bytes.data() + kHeaderBytes + static_cast<std::size_t>(len));
  if (computed_crc != wire_crc) {
    res.error = DecodeError::kBadCrc;
    return res;
  }

  res.error = DecodeError::kNone;
  res.bytes_consumed = expected_total;
  res.frame.payload_len = len;
  res.frame.incompat_flags = incompat;
  res.frame.compat_flags = compat;
  res.frame.seq = seq;
  res.frame.sysid = sysid;
  res.frame.compid = compid;
  res.frame.msgid = msgid;
  res.frame.payload.fill(0u);
  if (len > 0u) {
    std::memcpy(res.frame.payload.data(), wire_bytes.data() + kHeaderBytes, len);
  }
  res.frame.checksum = wire_crc;
  res.frame.is_signed = is_signed;
  if (is_signed) {
    std::memcpy(
        res.frame.signature.data(),
        wire_bytes.data() + kHeaderBytes + static_cast<std::size_t>(len) + kChecksumBytes,
        kSignatureBytes);
  }
  return res;
}

std::size_t encode_heartbeat(
    const Heartbeat& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept {
  std::array<uint8_t, kPayloadLenHeartbeat> p{};
  put_u32_le(p.data() + 0u, msg.custom_mode);
  p[4] = msg.type;
  p[5] = msg.autopilot;
  p[6] = msg.base_mode;
  p[7] = msg.system_status;
  p[8] = msg.mavlink_version;
  return encode_frame(seq, sysid, compid, kMsgIdHeartbeat, p, out_buf);
}

Heartbeat decode_heartbeat(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenHeartbeat>(frame);
  Heartbeat msg{};
  msg.custom_mode = get_u32_le(p.data() + 0u);
  msg.type = p[4];
  msg.autopilot = p[5];
  msg.base_mode = p[6];
  msg.system_status = p[7];
  msg.mavlink_version = p[8];
  return msg;
}

std::size_t encode_sys_status(
    const SysStatus& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept {
  std::array<uint8_t, kPayloadLenSysStatus> p{};
  put_u32_le(p.data() + 0u, msg.onboard_control_sensors_present);
  put_u32_le(p.data() + 4u, msg.onboard_control_sensors_enabled);
  put_u32_le(p.data() + 8u, msg.onboard_control_sensors_health);
  put_u16_le(p.data() + 12u, msg.load);
  put_u16_le(p.data() + 14u, msg.voltage_battery);
  put_i16_le(p.data() + 16u, msg.current_battery);
  put_u16_le(p.data() + 18u, msg.drop_rate_comm);
  put_u16_le(p.data() + 20u, msg.errors_comm);
  put_u16_le(p.data() + 22u, msg.errors_count1);
  put_u16_le(p.data() + 24u, msg.errors_count2);
  put_u16_le(p.data() + 26u, msg.errors_count3);
  put_u16_le(p.data() + 28u, msg.errors_count4);
  p[30] = static_cast<uint8_t>(msg.battery_remaining);
  return encode_frame(seq, sysid, compid, kMsgIdSysStatus, p, out_buf);
}

SysStatus decode_sys_status(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenSysStatus>(frame);
  SysStatus msg{};
  msg.onboard_control_sensors_present = get_u32_le(p.data() + 0u);
  msg.onboard_control_sensors_enabled = get_u32_le(p.data() + 4u);
  msg.onboard_control_sensors_health = get_u32_le(p.data() + 8u);
  msg.load = get_u16_le(p.data() + 12u);
  msg.voltage_battery = get_u16_le(p.data() + 14u);
  msg.current_battery = get_i16_le(p.data() + 16u);
  msg.drop_rate_comm = get_u16_le(p.data() + 18u);
  msg.errors_comm = get_u16_le(p.data() + 20u);
  msg.errors_count1 = get_u16_le(p.data() + 22u);
  msg.errors_count2 = get_u16_le(p.data() + 24u);
  msg.errors_count3 = get_u16_le(p.data() + 26u);
  msg.errors_count4 = get_u16_le(p.data() + 28u);
  msg.battery_remaining = static_cast<int8_t>(p[30]);
  return msg;
}

std::size_t encode_attitude_quaternion(
    const AttitudeQuaternion& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf,
    bool include_extensions) noexcept {
  std::array<uint8_t, kPayloadLenAttitudeQuaternionExt> p{};
  put_u32_le(p.data() + 0u, msg.time_boot_ms);
  put_f32_le(p.data() + 4u, msg.q1);
  put_f32_le(p.data() + 8u, msg.q2);
  put_f32_le(p.data() + 12u, msg.q3);
  put_f32_le(p.data() + 16u, msg.q4);
  put_f32_le(p.data() + 20u, msg.rollspeed);
  put_f32_le(p.data() + 24u, msg.pitchspeed);
  put_f32_le(p.data() + 28u, msg.yawspeed);
  if (include_extensions) {
    for (std::size_t i = 0u; i < 4u; ++i) {
      put_f32_le(p.data() + 32u + 4u * i, msg.repr_offset_q[i]);
    }
    return encode_frame(seq, sysid, compid, kMsgIdAttitudeQuaternion, p, out_buf);
  }
  return encode_frame(
      seq, sysid, compid, kMsgIdAttitudeQuaternion,
      std::span<const uint8_t>(p.data(), kPayloadLenAttitudeQuaternionBase), out_buf);
}

AttitudeQuaternion decode_attitude_quaternion(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenAttitudeQuaternionExt>(frame);
  AttitudeQuaternion msg{};
  msg.time_boot_ms = get_u32_le(p.data() + 0u);
  msg.q1 = get_f32_le(p.data() + 4u);
  msg.q2 = get_f32_le(p.data() + 8u);
  msg.q3 = get_f32_le(p.data() + 12u);
  msg.q4 = get_f32_le(p.data() + 16u);
  msg.rollspeed = get_f32_le(p.data() + 20u);
  msg.pitchspeed = get_f32_le(p.data() + 24u);
  msg.yawspeed = get_f32_le(p.data() + 28u);
  for (std::size_t i = 0u; i < 4u; ++i) {
    msg.repr_offset_q[i] = get_f32_le(p.data() + 32u + 4u * i);
  }
  return msg;
}

std::size_t encode_local_position_ned(
    const LocalPositionNed& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept {
  std::array<uint8_t, kPayloadLenLocalPositionNed> p{};
  put_u32_le(p.data() + 0u, msg.time_boot_ms);
  put_f32_le(p.data() + 4u, msg.x);
  put_f32_le(p.data() + 8u, msg.y);
  put_f32_le(p.data() + 12u, msg.z);
  put_f32_le(p.data() + 16u, msg.vx);
  put_f32_le(p.data() + 20u, msg.vy);
  put_f32_le(p.data() + 24u, msg.vz);
  return encode_frame(seq, sysid, compid, kMsgIdLocalPositionNed, p, out_buf);
}

LocalPositionNed decode_local_position_ned(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenLocalPositionNed>(frame);
  LocalPositionNed msg{};
  msg.time_boot_ms = get_u32_le(p.data() + 0u);
  msg.x = get_f32_le(p.data() + 4u);
  msg.y = get_f32_le(p.data() + 8u);
  msg.z = get_f32_le(p.data() + 12u);
  msg.vx = get_f32_le(p.data() + 16u);
  msg.vy = get_f32_le(p.data() + 20u);
  msg.vz = get_f32_le(p.data() + 24u);
  return msg;
}

std::size_t encode_command_long(
    const CommandLong& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept {
  std::array<uint8_t, kPayloadLenCommandLong> p{};
  put_f32_le(p.data() + 0u, msg.param1);
  put_f32_le(p.data() + 4u, msg.param2);
  put_f32_le(p.data() + 8u, msg.param3);
  put_f32_le(p.data() + 12u, msg.param4);
  put_f32_le(p.data() + 16u, msg.param5);
  put_f32_le(p.data() + 20u, msg.param6);
  put_f32_le(p.data() + 24u, msg.param7);
  put_u16_le(p.data() + 28u, msg.command);
  p[30] = msg.target_system;
  p[31] = msg.target_component;
  p[32] = msg.confirmation;
  return encode_frame(seq, sysid, compid, kMsgIdCommandLong, p, out_buf);
}

CommandLong decode_command_long(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenCommandLong>(frame);
  CommandLong msg{};
  msg.param1 = get_f32_le(p.data() + 0u);
  msg.param2 = get_f32_le(p.data() + 4u);
  msg.param3 = get_f32_le(p.data() + 8u);
  msg.param4 = get_f32_le(p.data() + 12u);
  msg.param5 = get_f32_le(p.data() + 16u);
  msg.param6 = get_f32_le(p.data() + 20u);
  msg.param7 = get_f32_le(p.data() + 24u);
  msg.command = get_u16_le(p.data() + 28u);
  msg.target_system = p[30];
  msg.target_component = p[31];
  msg.confirmation = p[32];
  return msg;
}

std::size_t encode_command_ack(
    const CommandAck& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf,
    bool include_extensions) noexcept {
  std::array<uint8_t, kPayloadLenCommandAckExt> p{};
  put_u16_le(p.data() + 0u, msg.command);
  p[2] = msg.result;
  if (include_extensions) {
    p[3] = msg.progress;
    put_i32_le(p.data() + 4u, msg.result_param2);
    p[8] = msg.target_system;
    p[9] = msg.target_component;
    return encode_frame(seq, sysid, compid, kMsgIdCommandAck, p, out_buf);
  }
  return encode_frame(
      seq, sysid, compid, kMsgIdCommandAck,
      std::span<const uint8_t>(p.data(), kPayloadLenCommandAckBase), out_buf);
}

CommandAck decode_command_ack(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenCommandAckExt>(frame);
  CommandAck msg{};
  msg.command = get_u16_le(p.data() + 0u);
  msg.result = p[2];
  msg.progress = p[3];
  msg.result_param2 = get_i32_le(p.data() + 4u);
  msg.target_system = p[8];
  msg.target_component = p[9];
  return msg;
}

std::size_t encode_set_position_target_local_ned(
    const SetPositionTargetLocalNed& msg, uint8_t seq, uint8_t sysid, uint8_t compid,
    std::span<uint8_t> out_buf) noexcept {
  std::array<uint8_t, kPayloadLenSetPositionTargetLocalNed> p{};
  put_u32_le(p.data() + 0u, msg.time_boot_ms);
  put_f32_le(p.data() + 4u, msg.x);
  put_f32_le(p.data() + 8u, msg.y);
  put_f32_le(p.data() + 12u, msg.z);
  put_f32_le(p.data() + 16u, msg.vx);
  put_f32_le(p.data() + 20u, msg.vy);
  put_f32_le(p.data() + 24u, msg.vz);
  put_f32_le(p.data() + 28u, msg.afx);
  put_f32_le(p.data() + 32u, msg.afy);
  put_f32_le(p.data() + 36u, msg.afz);
  put_f32_le(p.data() + 40u, msg.yaw);
  put_f32_le(p.data() + 44u, msg.yaw_rate);
  put_u16_le(p.data() + 48u, msg.type_mask);
  p[50] = msg.target_system;
  p[51] = msg.target_component;
  p[52] = msg.coordinate_frame;
  return encode_frame(seq, sysid, compid, kMsgIdSetPositionTargetLocalNed, p, out_buf);
}

SetPositionTargetLocalNed decode_set_position_target_local_ned(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenSetPositionTargetLocalNed>(frame);
  SetPositionTargetLocalNed msg{};
  msg.time_boot_ms = get_u32_le(p.data() + 0u);
  msg.x = get_f32_le(p.data() + 4u);
  msg.y = get_f32_le(p.data() + 8u);
  msg.z = get_f32_le(p.data() + 12u);
  msg.vx = get_f32_le(p.data() + 16u);
  msg.vy = get_f32_le(p.data() + 20u);
  msg.vz = get_f32_le(p.data() + 24u);
  msg.afx = get_f32_le(p.data() + 28u);
  msg.afy = get_f32_le(p.data() + 32u);
  msg.afz = get_f32_le(p.data() + 36u);
  msg.yaw = get_f32_le(p.data() + 40u);
  msg.yaw_rate = get_f32_le(p.data() + 44u);
  msg.type_mask = get_u16_le(p.data() + 48u);
  msg.target_system = p[50];
  msg.target_component = p[51];
  msg.coordinate_frame = p[52];
  return msg;
}

std::size_t encode_statustext(
    const StatusText& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf,
    bool include_extensions) noexcept {
  std::array<uint8_t, kPayloadLenStatusTextExt> p{};
  p[0] = msg.severity;
  std::memcpy(p.data() + 1u, msg.text.data(), msg.text.size());
  if (include_extensions) {
    put_u16_le(p.data() + 51u, msg.id);
    p[53] = msg.chunk_seq;
    return encode_frame(seq, sysid, compid, kMsgIdStatusText, p, out_buf);
  }
  return encode_frame(
      seq, sysid, compid, kMsgIdStatusText,
      std::span<const uint8_t>(p.data(), kPayloadLenStatusTextBase), out_buf);
}

StatusText decode_statustext(const MavlinkV2Frame& frame) noexcept {
  const auto p = zero_extended_payload<kPayloadLenStatusTextExt>(frame);
  StatusText msg{};
  msg.severity = p[0];
  std::memcpy(msg.text.data(), p.data() + 1u, msg.text.size());
  msg.id = get_u16_le(p.data() + 51u);
  msg.chunk_seq = p[53];
  return msg;
}

}  // namespace vertirta::mavlink
