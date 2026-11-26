// SPDX-License-Identifier: Apache-2.0
// Zero-GPL Native C++20 MAVLink v2 Offboard RTA Shield integrating VertiRTA CbvfSafetyFilter + Istanbul NFZs.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/mavlink/mavlink_v2_codec.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"

namespace vertirta::mavlink {

struct ShieldConfig {
  uint8_t bridge_sysid{1u};
  uint8_t bridge_compid{191u};              // MAV_COMP_ID_ONBOARD_COMPUTER
  uint64_t deadman_timeout_us{500'000u};    // 500 ms fail-closed heartbeat timeout
  double max_horiz_speed_mps{18.0};
  double max_ascent_rate_mps{4.0};
  double max_descent_rate_mps{3.0};
  double min_agl_m{10.0};
  double max_agl_m{150.0};
  double airborne_disarm_agl_m{2.0};
  double ltfj_max_lon_deg{29.1895};
  double ltfj_buffer_m{30.0};
  double ltbx_ctr_radius_m{5556.0};         // 3 NM verified DHMI AIP LTBX CTR
  double hospital_nfz_radius_m{165.0};      // Kartal Dr. Lutfi Kirdar Hospital hard cylinder
  double hospital_nfz_ceiling_agl_m{167.0};
  double lookahead_dt_s{0.50};
  double vel_kp_s{2.0};
};

struct VehicleTelemetryState {
  bool heartbeat_seen{false};
  uint64_t last_fcu_heartbeat_us{0u};
  bool local_position_seen{false};
  uint64_t last_local_position_us{0u};
  bool armed{false};
  uint8_t base_mode{0u};
  uint8_t system_status{0u};
  core::Vec3Ned pos_ned{0.0, 0.0, -50.0};
  core::Vec3Ned vel_ned{0.0, 0.0, 0.0};
  core::QuatPx4FrdNed quat_px4{1.0, 0.0, 0.0, 0.0};
  uint16_t voltage_battery_mv{40320u};
  int8_t battery_remaining_pct{80};
  core::CapabilityEnvelope envelope{};
};

struct ShieldStats {
  uint64_t fcu_rx_valid{0u};
  uint64_t fcu_rx_rejected{0u};
  uint64_t offboard_rx_valid{0u};
  uint64_t crc_rejected{0u};
  uint64_t malformed_rejected{0u};
  uint64_t setpoints_passed{0u};
  uint64_t setpoints_clamped{0u};
  uint64_t setpoints_clamped_cbvf{0u};
  uint64_t setpoints_clamped_nfz{0u};
  uint64_t setpoints_clamped_plunge_speed{0u};
  uint64_t deadman_overrides{0u};
  uint64_t commands_passed{0u};
  uint64_t commands_denied{0u};
  uint64_t statustext_emitted{0u};
  uint64_t command_acks_emitted{0u};
};

enum class ShieldAction : uint8_t {
  kRejectedInvalidFrame = 0u,
  kForwardedNominal = 1u,
  kClampedSafeSetpoint = 2u,
  kDeadmanHoverOverride = 3u,
  kDeniedCommand = 4u,
};

struct ShieldStepOutput {
  ShieldAction action{ShieldAction::kRejectedInvalidFrame};
  std::vector<std::vector<uint8_t>> to_fcu{};
  std::vector<std::vector<uint8_t>> to_offboard{};
  rta::SafetyFilterOutput last_cbvf{};
  bool cbvf_evaluated{false};
};

class OffboardRtaShield {
 public:
  explicit OffboardRtaShield(
      const ShieldConfig& cfg = ShieldConfig{},
      const rta::MmapCbvf4DLut* cbvf_lut = nullptr,
      const rta::Cel2DLut* cel_lut = nullptr) noexcept;

  OffboardRtaShield(const OffboardRtaShield&) = delete;
  OffboardRtaShield& operator=(const OffboardRtaShield&) = delete;

  void reset() noexcept;

  // Ingest raw MAVLink v2 wire bytes arriving from the FCU (Pixhawk/PX4).
  // Updates vehicle state, rejects corrupted frames fail-closed, forwards valid telemetry to offboard.
  [[nodiscard]] ShieldStepOutput process_fcu_packet(
      std::span<const uint8_t> wire_bytes, uint64_t now_us);

  // Intercept raw MAVLink v2 wire bytes arriving from an untrusted Offboard/LLM/GCS planner.
  // Enforces CRC integrity, deadman timeout, CbvfSafetyFilter + NFZ clamping, and COMMAND_LONG safety.
  [[nodiscard]] ShieldStepOutput process_offboard_packet(
      std::span<const uint8_t> wire_bytes, uint64_t now_us);

  [[nodiscard]] bool is_deadman_expired(uint64_t now_us) const noexcept;
  [[nodiscard]] const VehicleTelemetryState& state() const noexcept { return state_; }
  [[nodiscard]] const ShieldStats& stats() const noexcept { return stats_; }
  [[nodiscard]] const ShieldConfig& config() const noexcept { return cfg_; }

  // Helper exposing Istanbul NFZ check (LTFJ CTR, LTBX CTR, Kartal Dr. Lutfi Kirdar Hospital).
  [[nodiscard]] bool violates_istanbul_nfz(const core::Vec3Ned& pos_ned) const noexcept;

 private:
  [[nodiscard]] std::array<rta::HazardPoint, 4> build_active_hazards() const noexcept;
  [[nodiscard]] std::vector<uint8_t> make_statustext_packet(uint8_t severity, std::string_view text);
  [[nodiscard]] std::vector<uint8_t> make_command_ack_packet(
      uint16_t command, uint8_t result, uint8_t target_sys, uint8_t target_comp);
  [[nodiscard]] std::vector<uint8_t> make_hover_setpoint_packet(
      const SetPositionTargetLocalNed& src);

  ShieldConfig cfg_{};
  VehicleTelemetryState state_{};
  ShieldStats stats_{};
  rta::CbvfSafetyFilter cbvf_filter_;
  core::Vec3Ned ltbx_arp_ned_{};
  core::Vec3Ned hospital_ned_{};
  double ltfj_wall_east_m_{126.0};
  uint8_t tx_seq_{0u};
};

}  // namespace vertirta::mavlink
