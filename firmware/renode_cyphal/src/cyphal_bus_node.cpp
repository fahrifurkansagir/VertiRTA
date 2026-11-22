// SPDX-License-Identifier: Apache-2.0
#include "vertirta_cyphal/cyphal_bus_node.hpp"

#include <cstring>

namespace vertirta::cyphal {

namespace {
// ISO 11898-1:2015 CAN-FD DLC ceiling rounding (0..8, 12, 16, 20, 24, 32, 48, 64).
std::uint8_t round_up_canfd_len(std::size_t raw_len) noexcept {
  static constexpr std::uint8_t kDlcSizes[] = {8u, 12u, 16u, 20u, 24u, 32u, 48u, 64u};
  if (raw_len <= 8u) {
    return static_cast<std::uint8_t>(raw_len);
  }
  for (std::uint8_t sz : kDlcSizes) {
    if (raw_len <= sz) {
      return sz;
    }
  }
  return VERTIRTA_CANFD_MTU_BYTES;
}
}  // namespace

std::uint32_t encode_cyphal_msg_can_id(
    CyphalPriority priority,
    std::uint16_t subject_id,
    std::uint8_t src_node_id) noexcept {
  const std::uint32_t prio = (static_cast<std::uint32_t>(priority) & 0x7u) << 26u;
  const std::uint32_t reserved_r = 0x3u << 21u;  // Cyphal/CAN v1.0 fixed reserved bits [22:21] = 11
  const std::uint32_t subj = (static_cast<std::uint32_t>(subject_id) & 0x1FFFu) << 8u;
  const std::uint32_t src = static_cast<std::uint32_t>(src_node_id) & 0x7Fu;
  return prio | reserved_r | subj | src;
}

bool decode_cyphal_msg_can_id(
    std::uint32_t ext_id,
    CyphalPriority& out_priority,
    std::uint16_t& out_subject_id,
    std::uint8_t& out_src_node_id) noexcept {
  // Reject service frames (bit 25 == 1) and anonymous frames (bit 24 == 1).
  if ((ext_id & (3u << 24u)) != 0u) {
    return false;
  }
  // Verify Cyphal/CAN v1.0 reserved bits [22:21] == 0b11.
  if (((ext_id >> 21u) & 0x3u) != 0x3u) {
    return false;
  }
  out_priority = static_cast<CyphalPriority>((ext_id >> 26u) & 0x7u);
  out_subject_id = static_cast<std::uint16_t>((ext_id >> 8u) & 0x1FFFu);
  out_src_node_id = static_cast<std::uint8_t>(ext_id & 0x7Fu);
  return true;
}

std::uint8_t make_single_frame_tail_byte(std::uint8_t transfer_id) noexcept {
  // Single-frame Cyphal v1.0 transfer: SOT=1 (0x80), EOT=1 (0x40), Toggle=1 (0x20).
  return static_cast<std::uint8_t>(0xE0u | (transfer_id & 0x1Fu));
}

bool is_valid_single_frame_tail_byte(std::uint8_t tail) noexcept {
  return (tail & 0xE0u) == 0xE0u;
}

CyphalFccNode::CyphalFccNode(
    FccRole role,
    std::uint8_t node_id,
    O1HeapDtcmArena& arena) noexcept
    : role_(role), node_id_(node_id), arena_(arena) {
  reset(0u);
}

CyphalFccNode::~CyphalFccNode() noexcept {
  clear_tx_queue();
}

void CyphalFccNode::clear_tx_queue() noexcept {
  while (tx_head_ != nullptr) {
    TxNode* cur = tx_head_;
    tx_head_ = cur->next;
    arena_.deallocate(cur);
  }
  tx_count_ = 0u;
}

void CyphalFccNode::reset(std::uint32_t now_us) noexcept {
  clear_tx_queue();
  state_ = (role_ == FccRole::PRIMARY) ? FccState::ACTIVE : FccState::STANDBY;
  safety_tid_ = 0u;
  heartbeat_tid_ = 0u;
  last_heartbeat_us_ = now_us;
  primary_seen_ = false;
  last_valid_primary_rx_us_ = now_us;
  rx_valid_count_ = 0u;
  rx_crc_reject_count_ = 0u;
  takeover_latency_us_ = 0u;
  std::memset(&last_rx_envelope_, 0, sizeof(last_rx_envelope_));
}

bool CyphalFccNode::enqueue_tx(const CanFdFrame& frame) noexcept {
  if (tx_count_ >= kMaxQueuedTxFrames) {
    return false;
  }
  void* mem = arena_.allocate(sizeof(TxNode));
  if (mem == nullptr) {
    return false;
  }
  auto* node = static_cast<TxNode*>(mem);
  node->frame = frame;
  node->next = nullptr;

  // Priority-ordered insertion by 29-bit CAN arbitration ID (lower numeric ID = higher priority).
  if ((tx_head_ == nullptr) || (frame.ext_id < tx_head_->frame.ext_id)) {
    node->next = tx_head_;
    tx_head_ = node;
  } else {
    TxNode* cur = tx_head_;
    while ((cur->next != nullptr) && (cur->next->frame.ext_id <= frame.ext_id)) {
      cur = cur->next;
    }
    node->next = cur->next;
    cur->next = node;
  }
  ++tx_count_;
  return true;
}

bool CyphalFccNode::pop_tx_frame(CanFdFrame& out_frame) noexcept {
  if (tx_head_ == nullptr) {
    return false;
  }
  TxNode* node = tx_head_;
  tx_head_ = node->next;
  out_frame = node->frame;
  arena_.deallocate(node);
  --tx_count_;
  return true;
}

bool CyphalFccNode::publish_safety_envelope(
    const vertirta_flight_SafetyEnvelope_1_0& env,
    CyphalPriority priority) noexcept {
  CanFdFrame frame{};
  frame.ext_id = encode_cyphal_msg_can_id(
      priority, VERTIRTA_SAFETY_ENVELOPE_SUBJECT_ID, node_id_);

  std::size_t payload_size = VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES;
  const std::int32_t rc = vertirta_flight_SafetyEnvelope_1_0_serialize_(
      &env, frame.data, &payload_size);
  if ((rc != VERTIRTA_CODEC_OK) ||
      (payload_size != VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES)) {
    return false;
  }

  // Cyphal/CAN single-frame CAN-FD rule: 48 B payload + zero padding + 1 tail byte at end of DLC.
  const std::uint8_t wire_len = round_up_canfd_len(payload_size + VERTIRTA_CYPHAL_TAIL_BYTE_SIZE);
  for (std::size_t i = payload_size; i < static_cast<std::size_t>(wire_len - 1u); ++i) {
    frame.data[i] = 0u;
  }
  frame.data[wire_len - 1u] = make_single_frame_tail_byte(safety_tid_);
  safety_tid_ = static_cast<std::uint8_t>((safety_tid_ + 1u) & 0x1Fu);
  frame.len = wire_len;

  return enqueue_tx(frame);
}

bool CyphalFccNode::publish_heartbeat(
    std::uint32_t uptime_s,
    std::uint8_t health,
    std::uint8_t mode,
    std::uint8_t vendor_code) noexcept {
  vertirta_uavcan_node_Heartbeat_1_0 hb{};
  hb.uptime = uptime_s;
  hb.health = health;
  hb.mode = mode;
  hb.vendor_specific_status_code = vendor_code;

  CanFdFrame frame{};
  frame.ext_id = encode_cyphal_msg_can_id(
      CyphalPriority::NOMINAL, VERTIRTA_HEARTBEAT_SUBJECT_ID, node_id_);

  std::size_t payload_size = VERTIRTA_HEARTBEAT_EXTENT_BYTES;
  const std::int32_t rc = vertirta_uavcan_node_Heartbeat_1_0_serialize_(
      &hb, frame.data, &payload_size);
  if ((rc != VERTIRTA_CODEC_OK) || (payload_size != VERTIRTA_HEARTBEAT_EXTENT_BYTES)) {
    return false;
  }

  frame.data[VERTIRTA_HEARTBEAT_EXTENT_BYTES] = make_single_frame_tail_byte(heartbeat_tid_);
  heartbeat_tid_ = static_cast<std::uint8_t>((heartbeat_tid_ + 1u) & 0x1Fu);
  frame.len = static_cast<std::uint8_t>(VERTIRTA_HEARTBEAT_EXTENT_BYTES + 1u);  // exact 8 B
  return enqueue_tx(frame);
}

bool CyphalFccNode::on_rx_frame(const CanFdFrame& frame, std::uint32_t now_us) noexcept {
  CyphalPriority prio = CyphalPriority::NOMINAL;
  std::uint16_t subject_id = 0u;
  std::uint8_t src_node = 0u;
  if (!decode_cyphal_msg_can_id(frame.ext_id, prio, subject_id, src_node)) {
    return false;
  }
  if (src_node == node_id_) {
    return false;  // Ignore self-loopback frames
  }
  if (subject_id != VERTIRTA_SAFETY_ENVELOPE_SUBJECT_ID) {
    return true;  // Non-supervision subject (e.g., Heartbeat 7509)
  }

  if ((frame.len < (VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES + VERTIRTA_CYPHAL_TAIL_BYTE_SIZE)) ||
      (frame.len > VERTIRTA_CANFD_MTU_BYTES)) {
    ++rx_crc_reject_count_;
    return false;
  }
  if (!is_valid_single_frame_tail_byte(frame.data[frame.len - 1u])) {
    ++rx_crc_reject_count_;
    return false;
  }

  vertirta_flight_SafetyEnvelope_1_0 decoded{};
  std::size_t buf_len = static_cast<std::size_t>(frame.len - 1u);
  const std::int32_t rc = vertirta_flight_SafetyEnvelope_1_0_deserialize_(
      &decoded, frame.data, &buf_len);
  if (rc != VERTIRTA_CODEC_OK) {
    // Babbling-idiot corrupted CRC-32C or non-finite wrench -> strictly reject; DO NOT reset watchdog!
    ++rx_crc_reject_count_;
    return false;
  }

  if ((role_ == FccRole::SECONDARY) && (src_node == kPrimaryNodeId)) {
    last_rx_envelope_ = decoded;
    ++rx_valid_count_;
    if (state_ == FccState::STANDBY) {
      primary_seen_ = true;
      last_valid_primary_rx_us_ = now_us;
    }
  }
  return true;
}

bool CyphalFccNode::tick(
    std::uint32_t now_us,
    const vertirta_flight_SafetyEnvelope_1_0& local_env) noexcept {
  // Periodic 1 Hz OpenCyphal Heartbeat.1.0 on fixed Subject-ID 7509.
  if (static_cast<std::uint32_t>(now_us - last_heartbeat_us_) >= kHeartbeatPeriodUs) {
    last_heartbeat_us_ = now_us;
    const std::uint8_t health = (state_ == FccState::FAILOVER_ACTIVE) ? 1u : 0u;
    const std::uint8_t vendor_code = static_cast<std::uint8_t>(state_);
    (void)publish_heartbeat(now_us / 1000000u, health, 0u, vendor_code);
  }

  if (role_ == FccRole::PRIMARY) {
    if (state_ == FccState::ACTIVE) {
      (void)publish_safety_envelope(local_env, CyphalPriority::FAST);
    }
    return false;
  }

  // Secondary Safety Monitor (Node 11) watchdog & deterministic <20 ms takeover FSM.
  if (state_ == FccState::STANDBY) {
    const std::uint32_t silence_us = static_cast<std::uint32_t>(now_us - last_valid_primary_rx_us_);
    if (silence_us >= kWatchdogTimeoutUs) {
      state_ = FccState::FAILOVER_ACTIVE;
      takeover_latency_us_ = silence_us;

      vertirta_flight_SafetyEnvelope_1_0 handover = local_env;
      handover.cel_latched = 1u;
      handover.t_us = now_us;
      (void)publish_safety_envelope(handover, CyphalPriority::EXCEPTIONAL);
      return true;
    }
    return false;
  }

  if (state_ == FccState::FAILOVER_ACTIVE) {
    vertirta_flight_SafetyEnvelope_1_0 active_env = local_env;
    active_env.cel_latched = 1u;
    active_env.t_us = now_us;
    (void)publish_safety_envelope(active_env, CyphalPriority::FAST);
  }
  return false;
}

std::uint32_t CyphalFccNode::modeled_total_handover_us() const noexcept {
  const std::uint32_t watchdog = (takeover_latency_us_ > 0u) ? takeover_latency_us_
                                                             : kWatchdogTimeoutUs;
  return watchdog + VERTIRTA_FAILOVER_BUS_HANDOVER_US + VERTIRTA_FAILOVER_SW_ALLOWANCE_US +
         VERTIRTA_FAILOVER_FIRST_FRAME_US;
}

}  // namespace vertirta::cyphal
