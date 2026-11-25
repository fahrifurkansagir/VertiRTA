// SPDX-License-Identifier: Apache-2.0
// Clean-room Apache-2.0 native C++20 MAVLink v2 (STX=0xFD) wire codec.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace vertirta::mavlink {

inline constexpr uint8_t kMavlinkV2Stx = 0xFDu;
inline constexpr std::size_t kHeaderBytes = 10u;
inline constexpr std::size_t kChecksumBytes = 2u;
inline constexpr std::size_t kSignatureBytes = 13u;
inline constexpr std::size_t kMaxPayloadBytes = 255u;
inline constexpr std::size_t kMaxFrameBytes =
    kHeaderBytes + kMaxPayloadBytes + kChecksumBytes + kSignatureBytes;  // 280 B

inline constexpr uint8_t kIncompatFlagSigned = 0x01u;

// Official MAVLink common.xml message IDs.
inline constexpr uint32_t kMsgIdHeartbeat = 0u;
inline constexpr uint32_t kMsgIdSysStatus = 1u;
inline constexpr uint32_t kMsgIdAttitudeQuaternion = 31u;
inline constexpr uint32_t kMsgIdLocalPositionNed = 32u;
inline constexpr uint32_t kMsgIdCommandLong = 76u;
inline constexpr uint32_t kMsgIdCommandAck = 77u;
inline constexpr uint32_t kMsgIdSetPositionTargetLocalNed = 84u;
inline constexpr uint32_t kMsgIdStatusText = 253u;

// Official MAVLink common.xml CRC_EXTRA seeds.
inline constexpr uint8_t kCrcExtraHeartbeat = 50u;
inline constexpr uint8_t kCrcExtraSysStatus = 124u;
inline constexpr uint8_t kCrcExtraAttitudeQuaternion = 246u;
inline constexpr uint8_t kCrcExtraLocalPositionNed = 185u;
inline constexpr uint8_t kCrcExtraCommandLong = 152u;
inline constexpr uint8_t kCrcExtraCommandAck = 143u;
inline constexpr uint8_t kCrcExtraSetPositionTargetLocalNed = 143u;
inline constexpr uint8_t kCrcExtraStatusText = 83u;

// Official MAVLink base / v2-extension payload sizes [bytes].
inline constexpr std::size_t kPayloadLenHeartbeat = 9u;
inline constexpr std::size_t kPayloadLenSysStatus = 31u;
inline constexpr std::size_t kPayloadLenAttitudeQuaternionBase = 32u;
inline constexpr std::size_t kPayloadLenAttitudeQuaternionExt = 48u;
inline constexpr std::size_t kPayloadLenLocalPositionNed = 28u;
inline constexpr std::size_t kPayloadLenCommandLong = 33u;
inline constexpr std::size_t kPayloadLenCommandAckBase = 3u;
inline constexpr std::size_t kPayloadLenCommandAckExt = 10u;
inline constexpr std::size_t kPayloadLenSetPositionTargetLocalNed = 53u;
inline constexpr std::size_t kPayloadLenStatusTextBase = 51u;
inline constexpr std::size_t kPayloadLenStatusTextExt = 54u;

// MAVLink enums used by the RTA Shield.
inline constexpr uint8_t kMavModeFlagSafetyArmed = 0x80u;
inline constexpr uint8_t kMavFrameLocalNed = 1u;

inline constexpr uint16_t kMavCmdDoFlightTermination = 185u;
inline constexpr uint16_t kMavCmdComponentArmDisarm = 400u;

inline constexpr uint8_t kMavResultAccepted = 0u;
inline constexpr uint8_t kMavResultDenied = 2u;

inline constexpr uint8_t kMavSeverityEmergency = 0u;
inline constexpr uint8_t kMavSeverityAlert = 1u;
inline constexpr uint8_t kMavSeverityCritical = 2u;
inline constexpr uint8_t kMavSeverityError = 3u;
inline constexpr uint8_t kMavSeverityWarning = 4u;
inline constexpr uint8_t kMavSeverityNotice = 5u;
inline constexpr uint8_t kMavSeverityInfo = 6u;

// POSITION_TARGET_TYPEMASK bitflags.
inline constexpr uint16_t kTypeMaskIgnoreX = 0x0001u;
inline constexpr uint16_t kTypeMaskIgnoreY = 0x0002u;
inline constexpr uint16_t kTypeMaskIgnoreZ = 0x0004u;
inline constexpr uint16_t kTypeMaskIgnoreVx = 0x0008u;
inline constexpr uint16_t kTypeMaskIgnoreVy = 0x0010u;
inline constexpr uint16_t kTypeMaskIgnoreVz = 0x0020u;
inline constexpr uint16_t kTypeMaskIgnoreAfx = 0x0040u;
inline constexpr uint16_t kTypeMaskIgnoreAfy = 0x0080u;
inline constexpr uint16_t kTypeMaskIgnoreAfz = 0x0100u;
inline constexpr uint16_t kTypeMaskForce = 0x0200u;
inline constexpr uint16_t kTypeMaskIgnoreYaw = 0x0400u;
inline constexpr uint16_t kTypeMaskIgnoreYawRate = 0x0800u;
inline constexpr uint16_t kTypeMaskVelocityOnly =
    kTypeMaskIgnoreX | kTypeMaskIgnoreY | kTypeMaskIgnoreZ |
    kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz |
    kTypeMaskIgnoreYaw | kTypeMaskIgnoreYawRate;  // 0x0DC7

struct Heartbeat {
  uint32_t custom_mode{0u};
  uint8_t type{2u};             // MAV_TYPE_QUADROTOR / VTOL
  uint8_t autopilot{12u};       // MAV_AUTOPILOT_PX4
  uint8_t base_mode{0u};
  uint8_t system_status{4u};    // MAV_STATE_ACTIVE
  uint8_t mavlink_version{3u};
};

struct SysStatus {
  uint32_t onboard_control_sensors_present{0u};
  uint32_t onboard_control_sensors_enabled{0u};
  uint32_t onboard_control_sensors_health{0u};
  uint16_t load{0u};
  uint16_t voltage_battery{0u};    // mV (UINT16_MAX if unknown)
  int16_t current_battery{-1};     // cA (-1 if unknown)
  uint16_t drop_rate_comm{0u};
  uint16_t errors_comm{0u};
  uint16_t errors_count1{0u};
  uint16_t errors_count2{0u};
  uint16_t errors_count3{0u};
  uint16_t errors_count4{0u};
  int8_t battery_remaining{-1};    // % (-1 if unknown)
};

struct AttitudeQuaternion {
  uint32_t time_boot_ms{0u};
  float q1{1.0f};
  float q2{0.0f};
  float q3{0.0f};
  float q4{0.0f};
  float rollspeed{0.0f};
  float pitchspeed{0.0f};
  float yawspeed{0.0f};
  std::array<float, 4> repr_offset_q{0.0f, 0.0f, 0.0f, 0.0f};
};

struct LocalPositionNed {
  uint32_t time_boot_ms{0u};
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  float vx{0.0f};
  float vy{0.0f};
  float vz{0.0f};
};

struct CommandLong {
  float param1{0.0f};
  float param2{0.0f};
  float param3{0.0f};
  float param4{0.0f};
  float param5{0.0f};
  float param6{0.0f};
  float param7{0.0f};
  uint16_t command{0u};
  uint8_t target_system{1u};
  uint8_t target_component{1u};
  uint8_t confirmation{0u};
};

struct CommandAck {
  uint16_t command{0u};
  uint8_t result{0u};
  uint8_t progress{0u};
  int32_t result_param2{0};
  uint8_t target_system{0u};
  uint8_t target_component{0u};
};

struct SetPositionTargetLocalNed {
  uint32_t time_boot_ms{0u};
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};
  float vx{0.0f};
  float vy{0.0f};
  float vz{0.0f};
  float afx{0.0f};
  float afy{0.0f};
  float afz{0.0f};
  float yaw{0.0f};
  float yaw_rate{0.0f};
  uint16_t type_mask{0u};
  uint8_t target_system{1u};
  uint8_t target_component{1u};
  uint8_t coordinate_frame{kMavFrameLocalNed};
};

struct StatusText {
  uint8_t severity{kMavSeverityInfo};
  std::array<char, 50> text{};
  uint16_t id{0u};
  uint8_t chunk_seq{0u};

  void set_text(std::string_view msg) noexcept;
  [[nodiscard]] std::string_view text_view() const noexcept;
};

struct MavlinkV2Frame {
  uint8_t payload_len{0u};
  uint8_t incompat_flags{0u};
  uint8_t compat_flags{0u};
  uint8_t seq{0u};
  uint8_t sysid{0u};
  uint8_t compid{0u};
  uint32_t msgid{0u};
  std::array<uint8_t, kMaxPayloadBytes> payload{};
  uint16_t checksum{0u};
  bool is_signed{false};
  std::array<uint8_t, kSignatureBytes> signature{};
};

enum class DecodeError : uint8_t {
  kNone = 0u,
  kTooShort = 1u,
  kBadMagic = 2u,
  kTruncatedFrame = 3u,
  kUnsupportedIncompatFlags = 4u,
  kUnknownMsgId = 5u,
  kBadCrc = 6u,
};

struct DecodeResult {
  DecodeError error{DecodeError::kNone};
  MavlinkV2Frame frame{};
  std::size_t bytes_consumed{0u};

  [[nodiscard]] bool ok() const noexcept { return error == DecodeError::kNone; }
};

// X.25 CRC-16/MCRF4XX (seed 0xFFFF, reflected polynomial 0x8408).
[[nodiscard]] constexpr uint16_t crc16_init() noexcept { return 0xFFFFu; }

constexpr void crc16_accumulate(uint8_t data, uint16_t& crc) noexcept {
  uint8_t tmp = static_cast<uint8_t>(data ^ static_cast<uint8_t>(crc & 0xFFu));
  tmp = static_cast<uint8_t>(tmp ^ static_cast<uint8_t>(tmp << 4u));
  crc = static_cast<uint16_t>(
      (crc >> 8u) ^
      (static_cast<uint16_t>(tmp) << 8u) ^
      (static_cast<uint16_t>(tmp) << 3u) ^
      (static_cast<uint16_t>(tmp) >> 4u));
}

[[nodiscard]] uint16_t crc16_buffer(std::span<const uint8_t> data, uint16_t seed = 0xFFFFu) noexcept;

[[nodiscard]] std::optional<uint8_t> crc_extra_for_msgid(uint32_t msgid) noexcept;

// Compute truncated MAVLink v2 payload length (strips trailing 0x00, keeping >= 1 byte if all zeros).
[[nodiscard]] uint8_t truncated_payload_length(std::span<const uint8_t> full_payload) noexcept;

// Encode a generic MAVLink v2 frame into out_buf (size >= kMaxFrameBytes). Returns total wire bytes or 0 on error.
[[nodiscard]] std::size_t encode_frame(
    uint8_t seq,
    uint8_t sysid,
    uint8_t compid,
    uint32_t msgid,
    std::span<const uint8_t> full_payload,
    std::span<uint8_t> out_buf,
    uint8_t compat_flags = 0u) noexcept;

// Fail-closed MAVLink v2 frame decoder (validates STX=0xFD, length, incompat_flags, CRC_EXTRA, CRC-16).
[[nodiscard]] DecodeResult decode_frame(std::span<const uint8_t> wire_bytes) noexcept;

// Typed payload packers/unpackers (exact common.xml wire-sorted LE field order + zero-extension on decode).
[[nodiscard]] std::size_t encode_heartbeat(
    const Heartbeat& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept;
[[nodiscard]] Heartbeat decode_heartbeat(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_sys_status(
    const SysStatus& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept;
[[nodiscard]] SysStatus decode_sys_status(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_attitude_quaternion(
    const AttitudeQuaternion& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf,
    bool include_extensions = false) noexcept;
[[nodiscard]] AttitudeQuaternion decode_attitude_quaternion(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_local_position_ned(
    const LocalPositionNed& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept;
[[nodiscard]] LocalPositionNed decode_local_position_ned(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_command_long(
    const CommandLong& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept;
[[nodiscard]] CommandLong decode_command_long(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_command_ack(
    const CommandAck& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf,
    bool include_extensions = true) noexcept;
[[nodiscard]] CommandAck decode_command_ack(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_set_position_target_local_ned(
    const SetPositionTargetLocalNed& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf) noexcept;
[[nodiscard]] SetPositionTargetLocalNed decode_set_position_target_local_ned(const MavlinkV2Frame& frame) noexcept;

[[nodiscard]] std::size_t encode_statustext(
    const StatusText& msg, uint8_t seq, uint8_t sysid, uint8_t compid, std::span<uint8_t> out_buf,
    bool include_extensions = false) noexcept;
[[nodiscard]] StatusText decode_statustext(const MavlinkV2Frame& frame) noexcept;

}  // namespace vertirta::mavlink
