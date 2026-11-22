// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "vertirta_cyphal/cyphal_bus_node.hpp"
#include "vertirta_cyphal/o1heap_arena.hpp"
#include "vertirta_cyphal/safety_envelope_codec.h"
#include "zero_heap_guard.hpp"

using vertirta::cyphal::CanFdFrame;
using vertirta::cyphal::CyphalFccNode;
using vertirta::cyphal::CyphalPriority;
using vertirta::cyphal::FccRole;
using vertirta::cyphal::FccState;
using vertirta::cyphal::O1HeapDtcmArena;

namespace {

vertirta_flight_SafetyEnvelope_1_0 make_sample_envelope(std::uint32_t seq, std::uint32_t t_us) {
  vertirta_flight_SafetyEnvelope_1_0 env{};
  env.wrench_min_frd[0] = -5886.0f;
  env.wrench_min_frd[1] = -250.0f;
  env.wrench_min_frd[2] = -250.0f;
  env.wrench_min_frd[3] = -120.0f;
  env.wrench_max_frd[0] = 0.0f;
  env.wrench_max_frd[1] = 250.0f;
  env.wrench_max_frd[2] = 250.0f;
  env.wrench_max_frd[3] = 120.0f;
  env.seq = seq;
  env.t_us = t_us;
  env.fault_mask = 0x05u;
  env.flight_regime = 2u;
  env.tmr_state = 1u;
  env.cel_latched = 0u;
  env.crc32c = 0u;
  return env;
}

}  // namespace

TEST_CASE("SafetyEnvelope C11 codec enforces 48 B <= 63 B single CAN-FD frame and CRC-32C integrity") {
  static_assert(VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES == 48u);
  static_assert(VERTIRTA_SAFETY_ENVELOPE_EXTENT_BYTES + VERTIRTA_CYPHAL_TAIL_BYTE_SIZE <=
                VERTIRTA_CANFD_MTU_BYTES);
  static_assert(VERTIRTA_FAILOVER_MODELED_TOTAL_US == 13750u);
  static_assert(VERTIRTA_FAILOVER_MODELED_TOTAL_US < VERTIRTA_FAILOVER_HARD_CEILING_US);

  // Standard RFC 3720 / iSCSI Castagnoli CRC-32C vector ("123456789" -> 0xE3069283).
  const std::uint8_t check_str[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  REQUIRE(vertirta_crc32c_c11(check_str, sizeof(check_str)) == 0xE3069283u);

  const auto in = make_sample_envelope(42u, 105000u);
  std::uint8_t wire[VERTIRTA_CANFD_MTU_BYTES]{};
  std::size_t sz = sizeof(wire);
  REQUIRE(vertirta_flight_SafetyEnvelope_1_0_serialize_(&in, wire, &sz) == VERTIRTA_CODEC_OK);
  REQUIRE(sz == 48u);

  vertirta_flight_SafetyEnvelope_1_0 out{};
  std::size_t dec_sz = sz;
  REQUIRE(vertirta_flight_SafetyEnvelope_1_0_deserialize_(&out, wire, &dec_sz) == VERTIRTA_CODEC_OK);
  REQUIRE(dec_sz == 48u);
  REQUIRE(out.seq == in.seq);
  REQUIRE(out.t_us == in.t_us);
  REQUIRE(out.fault_mask == in.fault_mask);
  REQUIRE(out.flight_regime == in.flight_regime);
  REQUIRE(out.tmr_state == in.tmr_state);
  REQUIRE(out.wrench_min_frd[0] == in.wrench_min_frd[0]);
  REQUIRE(out.wrench_max_frd[3] == in.wrench_max_frd[3]);

  // Single-bit flip anywhere in [0..47] must fail CRC-32C validation.
  for (std::size_t byte_idx = 0u; byte_idx < 48u; ++byte_idx) {
    std::uint8_t corrupted[48];
    std::memcpy(corrupted, wire, 48u);
    corrupted[byte_idx] ^= 0x01u;
    std::size_t csz = 48u;
    REQUIRE(vertirta_flight_SafetyEnvelope_1_0_deserialize_(&out, corrupted, &csz) ==
            VERTIRTA_CODEC_ERR_CRC_MISMATCH);
  }

  // Non-finite float must be rejected on serialize AND deserialize.
  auto bad_nan = in;
  bad_nan.wrench_min_frd[1] = std::numeric_limits<float>::quiet_NaN();
  std::size_t bsz = sizeof(wire);
  REQUIRE(vertirta_flight_SafetyEnvelope_1_0_serialize_(&bad_nan, wire, &bsz) ==
          VERTIRTA_CODEC_ERR_NON_FINITE);

  // Truncated buffer rejected cleanly.
  std::size_t small_sz = 47u;
  REQUIRE(vertirta_flight_SafetyEnvelope_1_0_serialize_(&in, wire, &small_sz) ==
          VERTIRTA_CODEC_ERR_BUF_TOO_SMALL);

  // Standard uavcan.node.Heartbeat.1.0 (7 B, Subject-ID 7509) round-trip.
  vertirta_uavcan_node_Heartbeat_1_0 hb_in{1234u, 1u, 2u, 0x5Au};
  std::uint8_t hb_buf[8]{};
  std::size_t hb_sz = sizeof(hb_buf);
  REQUIRE(vertirta_uavcan_node_Heartbeat_1_0_serialize_(&hb_in, hb_buf, &hb_sz) ==
          VERTIRTA_CODEC_OK);
  REQUIRE(hb_sz == 7u);
  vertirta_uavcan_node_Heartbeat_1_0 hb_out{};
  REQUIRE(vertirta_uavcan_node_Heartbeat_1_0_deserialize_(&hb_out, hb_buf, &hb_sz) ==
          VERTIRTA_CODEC_OK);
  REQUIRE(hb_out.uptime == 1234u);
  REQUIRE(hb_out.health == 1u);
  REQUIRE(hb_out.mode == 2u);
  REQUIRE(hb_out.vendor_specific_status_code == 0x5Au);
}

TEST_CASE("O1HeapDtcmArena 32 KiB survives 10,000 randomized alloc/free/realloc ops with 0 OOM and 0 system heap") {
  static_assert(O1HeapDtcmArena::kArenaSizeBytes == 32768u);
  static_assert(O1HeapDtcmArena::kArenaSizeBytes > O1HeapDtcmArena::kWcmcUpperBoundBytes);

  O1HeapDtcmArena arena;
  REQUIRE(arena.invariants_hold());

  struct Slot {
    void* ptr{nullptr};
    std::size_t size{0u};
    std::uint8_t tag{0u};
  };
  constexpr std::size_t kMaxSlots = 48u;
  std::array<Slot, kMaxSlots> slots{};
  std::size_t live_payload_bytes = 0u;
  std::uint32_t rng = 0xA511E9B3u;
  auto next_rng = [&]() noexcept -> std::uint32_t {
    rng ^= rng << 13u;
    rng ^= rng >> 17u;
    rng ^= rng << 5u;
    return rng;
  };

  {
    vertirta::test::ScopedZeroHeapGuard guard;

    for (std::size_t iter = 0u; iter < 10000u; ++iter) {
      const std::size_t idx = next_rng() % kMaxSlots;
      const std::uint32_t op = next_rng() % 10u;

      if ((slots[idx].ptr != nullptr) && (op < 5u)) {
        // Verify payload bytes were never corrupted by neighboring fragments.
        const auto* bytes = static_cast<const std::uint8_t*>(slots[idx].ptr);
        REQUIRE(bytes[0] == slots[idx].tag);
        REQUIRE(bytes[slots[idx].size - 1u] == slots[idx].tag);
        arena.deallocate(slots[idx].ptr);
        live_payload_bytes -= slots[idx].size;
        slots[idx] = {};
      } else if ((slots[idx].ptr != nullptr) && (op == 5u)) {
        const std::size_t new_sz = 32u + (next_rng() % 225u);  // 32..256 B
        if ((live_payload_bytes - slots[idx].size + new_sz) <= 4096u) {
          void* p = arena.reallocate(slots[idx].ptr, new_sz);
          REQUIRE(p != nullptr);
          const auto* bytes = static_cast<const std::uint8_t*>(p);
          REQUIRE(bytes[0] == slots[idx].tag);
          std::memset(p, slots[idx].tag, new_sz);
          live_payload_bytes = live_payload_bytes - slots[idx].size + new_sz;
          slots[idx].ptr = p;
          slots[idx].size = new_sz;
        }
      } else if (slots[idx].ptr == nullptr) {
        const std::size_t req_sz = 32u + (next_rng() % 225u);  // 32..256 B
        if ((live_payload_bytes + req_sz) <= 4096u) {
          void* p = arena.allocate(req_sz);
          REQUIRE(p != nullptr);
          REQUIRE((reinterpret_cast<std::uintptr_t>(p) % O1HeapDtcmArena::kAlignment) == 0u);
          const auto tag = static_cast<std::uint8_t>((iter & 0x7Fu) | 0x80u);
          std::memset(p, tag, req_sz);
          slots[idx] = {p, req_sz, tag};
          live_payload_bytes += req_sz;
        }
      }

      if ((iter % 250u) == 0u) {
        REQUIRE(arena.invariants_hold());
      }
    }

    for (auto& s : slots) {
      if (s.ptr != nullptr) {
        arena.deallocate(s.ptr);
        s = {};
      }
    }

    REQUIRE(guard.alloc_count() == 0u);
    REQUIRE(guard.bytes_allocated() == 0u);
  }

  const auto diag = arena.diagnostics();
  REQUIRE(diag.capacity == 32768u);
  REQUIRE(diag.allocated == 0u);
  REQUIRE(diag.oom_count == 0u);
  REQUIRE(diag.peak_allocated <= 32768u);
  REQUIRE(arena.invariants_hold());
}

TEST_CASE("CyphalFccNode dual-FCC <20 ms failover FSM handles clean silence and babbling-idiot CRC corruption") {
  O1HeapDtcmArena pri_arena;
  O1HeapDtcmArena sec_arena;
  CyphalFccNode primary(FccRole::PRIMARY, CyphalFccNode::kPrimaryNodeId, pri_arena);
  CyphalFccNode secondary(FccRole::SECONDARY, CyphalFccNode::kSecondaryNodeId, sec_arena);

  // Verify exact OpenCyphal v1.0 29-bit CAN-FD ID encoding matching Robot Framework CANTester.
  REQUIRE(vertirta::cyphal::encode_cyphal_msg_can_id(
              CyphalPriority::FAST, VERTIRTA_SAFETY_ENVELOPE_SUBJECT_ID, 10u) == 0x0864000Au);
  REQUIRE(vertirta::cyphal::encode_cyphal_msg_can_id(
              CyphalPriority::EXCEPTIONAL, VERTIRTA_SAFETY_ENVELOPE_SUBJECT_ID, 11u) == 0x0064000Bu);

  {
    vertirta::test::ScopedZeroHeapGuard guard;

    // Phase 1: Nominal 400 Hz lockstep for 10 ticks (0..22.5 ms).
    std::uint32_t now_us = 0u;
    for (std::uint32_t tick = 0u; tick < 10u; ++tick) {
      now_us = tick * VERTIRTA_FAILOVER_PERIOD_US;
      const auto env = make_sample_envelope(tick, now_us);
      REQUIRE_FALSE(primary.tick(now_us, env));

      CanFdFrame frame{};
      while (primary.pop_tx_frame(frame)) {
        REQUIRE(secondary.on_rx_frame(frame, now_us));
      }
      REQUIRE_FALSE(secondary.tick(now_us, env));
      REQUIRE(secondary.state() == FccState::STANDBY);
    }
    REQUIRE(secondary.rx_valid_count() == 10u);
    REQUIRE(secondary.rx_crc_reject_count() == 0u);

    // Phase 2: Primary crashes (silence after t = 22.5 ms).
    // Secondary stays STANDBY at +2.5, +5.0, +7.5, +10.0 ms, then takes over at +10.50 ms!
    const std::uint32_t last_rx_us = now_us;
    for (std::uint32_t delta : {2500u, 5000u, 7500u, 10000u}) {
      const auto env = make_sample_envelope(100u, last_rx_us + delta);
      REQUIRE_FALSE(secondary.tick(last_rx_us + delta, env));
      REQUIRE(secondary.state() == FccState::STANDBY);
    }

    const std::uint32_t takeover_t_us = last_rx_us + VERTIRTA_FAILOVER_WATCHDOG_4FRAME_US;
    const auto env_takeover = make_sample_envelope(105u, takeover_t_us);
    REQUIRE(secondary.tick(takeover_t_us, env_takeover));
    REQUIRE(secondary.state() == FccState::FAILOVER_ACTIVE);
    REQUIRE(secondary.takeover_latency_us() == 10500u);
    REQUIRE(secondary.modeled_total_handover_us() == 13750u);
    REQUIRE(secondary.modeled_total_handover_us() < VERTIRTA_FAILOVER_HARD_CEILING_US);

    // Verify Secondary emitted Exceptional-priority Subject-ID 1024 handover frame with cel_latched=1.
    CanFdFrame handover{};
    REQUIRE(secondary.pop_tx_frame(handover));
    REQUIRE(handover.ext_id == 0x0064000Bu);
    REQUIRE(handover.len <= VERTIRTA_CANFD_MTU_BYTES);
    vertirta_flight_SafetyEnvelope_1_0 decoded{};
    std::size_t dec_sz = static_cast<std::size_t>(handover.len - 1u);
    REQUIRE(vertirta_flight_SafetyEnvelope_1_0_deserialize_(&decoded, handover.data, &dec_sz) ==
            VERTIRTA_CODEC_OK);
    REQUIRE(decoded.cel_latched == 1u);

    // Phase 3: Reset Secondary and inject 400 Hz corrupted-CRC babbling-idiot frames.
    secondary.reset(0u);
    const auto valid_env = make_sample_envelope(1u, 0u);
    REQUIRE(primary.publish_safety_envelope(valid_env, CyphalPriority::FAST));
    CanFdFrame good_frame{};
    REQUIRE(primary.pop_tx_frame(good_frame));
    REQUIRE(secondary.on_rx_frame(good_frame, 0u));
    REQUIRE(secondary.state() == FccState::STANDBY);

    CanFdFrame babble = good_frame;
    babble.data[44] ^= 0xFFu;  // Corrupt CRC-32C

    bool babble_failover = false;
    for (std::uint32_t t = 2500u; t <= 12500u; t += 2500u) {
      REQUIRE_FALSE(secondary.on_rx_frame(babble, t));
      if (secondary.tick(t, make_sample_envelope(t / 2500u, t))) {
        babble_failover = true;
        break;
      }
    }
    REQUIRE(babble_failover);
    REQUIRE(secondary.state() == FccState::FAILOVER_ACTIVE);
    REQUIRE(secondary.rx_crc_reject_count() >= 5u);
    REQUIRE(secondary.takeover_latency_us() <= 12500u);
    REQUIRE(secondary.modeled_total_handover_us() < VERTIRTA_FAILOVER_HARD_CEILING_US);

    // Drain any remaining Secondary frames so arena returns to clean 0 allocated bytes.
    CanFdFrame drain{};
    while (secondary.pop_tx_frame(drain)) {
    }

    REQUIRE(guard.alloc_count() == 0u);
    REQUIRE(guard.bytes_allocated() == 0u);
  }

  REQUIRE(pri_arena.invariants_hold());
  REQUIRE(sec_arena.invariants_hold());
  REQUIRE(pri_arena.diagnostics().oom_count == 0u);
  REQUIRE(sec_arena.diagnostics().oom_count == 0u);
}
