// SPDX-License-Identifier: Apache-2.0
// OpenCyphal v1.0 Reactive CAN-FD Node & <20 ms Dual-FCC Failover FSM (DR-08 Sec 2.2-2.3).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vertirta_cyphal/o1heap_arena.hpp"
#include "vertirta_cyphal/safety_envelope_codec.h"

namespace vertirta::cyphal {

enum class FccRole : std::uint8_t {
  PRIMARY = 0u,
  SECONDARY = 1u,
};

enum class FccState : std::uint8_t {
  INIT = 0u,
  ACTIVE = 1u,
  STANDBY = 2u,
  FAILOVER_ACTIVE = 3u,
  FAULT_LOCKOUT = 4u,
};

enum class CyphalPriority : std::uint8_t {
  EXCEPTIONAL = 0u,
  IMMEDIATE = 1u,
  FAST = 2u,
  HIGH = 3u,
  NOMINAL = 4u,
  LOW = 5u,
  SLOW = 6u,
  OPTIONAL = 7u,
};

struct CanFdFrame {
  std::uint32_t ext_id{0u};  // 29-bit OpenCyphal v1.0 extended CAN-FD identifier
  std::uint8_t len{0u};      // 0..64 B
  std::uint8_t data[VERTIRTA_CANFD_MTU_BYTES]{};
};

// OpenCyphal/CAN v1.0 29-bit message identifier helpers (Cyphal Spec Sec 4.2.1.2).
[[nodiscard]] std::uint32_t encode_cyphal_msg_can_id(
    CyphalPriority priority,
    std::uint16_t subject_id,
    std::uint8_t src_node_id) noexcept;

[[nodiscard]] bool decode_cyphal_msg_can_id(
    std::uint32_t ext_id,
    CyphalPriority& out_priority,
    std::uint16_t& out_subject_id,
    std::uint8_t& out_src_node_id) noexcept;

[[nodiscard]] std::uint8_t make_single_frame_tail_byte(std::uint8_t transfer_id) noexcept;
[[nodiscard]] bool is_valid_single_frame_tail_byte(std::uint8_t tail) noexcept;

// Deterministic reactive OpenCyphal/CAN-FD dual-FCC supervision node backed by O1HeapDtcmArena.
class CyphalFccNode {
 public:
  static constexpr std::uint8_t kPrimaryNodeId = 10u;
  static constexpr std::uint8_t kSecondaryNodeId = 11u;
  static constexpr std::uint32_t kWatchdogTimeoutUs = VERTIRTA_FAILOVER_WATCHDOG_4FRAME_US;
  static constexpr std::uint32_t kHeartbeatPeriodUs = 1000000u;
  static constexpr std::size_t kMaxQueuedTxFrames = 32u;

  CyphalFccNode(FccRole role, std::uint8_t node_id, O1HeapDtcmArena& arena) noexcept;
  ~CyphalFccNode() noexcept;

  CyphalFccNode(const CyphalFccNode&) = delete;
  CyphalFccNode& operator=(const CyphalFccNode&) = delete;

  void reset(std::uint32_t now_us) noexcept;

  // Primary (or Secondary after failover) publishes 400 Hz SafetyEnvelope.1.0 (Subject-ID 1024).
  [[nodiscard]] bool publish_safety_envelope(
      const vertirta_flight_SafetyEnvelope_1_0& env,
      CyphalPriority priority = CyphalPriority::FAST) noexcept;

  // Both nodes publish standard OpenCyphal 1 Hz Heartbeat.1.0 (Subject-ID 7509).
  [[nodiscard]] bool publish_heartbeat(
      std::uint32_t uptime_s,
      std::uint8_t health = 0u,
      std::uint8_t mode = 0u,
      std::uint8_t vendor_code = 0u) noexcept;

  // Processes an incoming CAN-FD frame; rejects corrupted-CRC babbling-idiot frames without
  // refreshing the Secondary supervision watchdog.
  [[nodiscard]] bool on_rx_frame(const CanFdFrame& frame, std::uint32_t now_us) noexcept;

  // Advances 400 Hz supervision timers & triggers Secondary failover on >= 10.50 ms silence.
  // Returns true on the exact tick when Secondary transitions STANDBY -> FAILOVER_ACTIVE.
  [[nodiscard]] bool tick(
      std::uint32_t now_us,
      const vertirta_flight_SafetyEnvelope_1_0& local_env) noexcept;

  // Pops highest-priority queued TX frame from the O1HeapDtcmArena-backed TX queue.
  [[nodiscard]] bool pop_tx_frame(CanFdFrame& out_frame) noexcept;

  [[nodiscard]] FccRole role() const noexcept { return role_; }
  [[nodiscard]] std::uint8_t node_id() const noexcept { return node_id_; }
  [[nodiscard]] FccState state() const noexcept { return state_; }
  [[nodiscard]] std::uint32_t rx_valid_count() const noexcept { return rx_valid_count_; }
  [[nodiscard]] std::uint32_t rx_crc_reject_count() const noexcept { return rx_crc_reject_count_; }
  [[nodiscard]] std::uint32_t takeover_latency_us() const noexcept { return takeover_latency_us_; }
  [[nodiscard]] std::uint32_t modeled_total_handover_us() const noexcept;
  [[nodiscard]] std::size_t tx_queue_depth() const noexcept { return tx_count_; }
  [[nodiscard]] const vertirta_flight_SafetyEnvelope_1_0& last_rx_envelope() const noexcept {
    return last_rx_envelope_;
  }

 private:
  struct TxNode {
    TxNode* next;
    CanFdFrame frame;
  };

  [[nodiscard]] bool enqueue_tx(const CanFdFrame& frame) noexcept;
  void clear_tx_queue() noexcept;

  FccRole role_;
  std::uint8_t node_id_;
  FccState state_{FccState::INIT};
  O1HeapDtcmArena& arena_;

  TxNode* tx_head_{nullptr};
  std::size_t tx_count_{0u};

  std::uint8_t safety_tid_{0u};
  std::uint8_t heartbeat_tid_{0u};
  std::uint32_t last_heartbeat_us_{0u};

  bool primary_seen_{false};
  std::uint32_t last_valid_primary_rx_us_{0u};
  std::uint32_t rx_valid_count_{0u};
  std::uint32_t rx_crc_reject_count_{0u};
  std::uint32_t takeover_latency_us_{0u};
  vertirta_flight_SafetyEnvelope_1_0 last_rx_envelope_{};
};

}  // namespace vertirta::cyphal
