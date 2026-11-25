// SPDX-License-Identifier: Apache-2.0
// POSIX UDP Loopback Bridge & Live 3-Endpoint Stress Proof Harness (ephemeral OS-assigned ports on 127.0.0.1).
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "vertirta/mavlink/offboard_rta_shield.hpp"

namespace vertirta::mavlink {

struct LatencyStatsUs {
  double min_us{0.0};
  double p50_us{0.0};
  double p95_us{0.0};
  double p99_us{0.0};
  double max_us{0.0};
  double mean_us{0.0};
  std::size_t sample_count{0u};
};

struct UdpLoopbackStressReport {
  bool passed{false};
  uint16_t fcu_port{0u};
  uint16_t bridge_fcu_port{0u};
  uint16_t bridge_offboard_port{0u};
  uint16_t planner_port{0u};

  uint32_t total_offboard_packets_sent{0u};
  uint32_t nominal_sent{0u};
  uint32_t plunge_overspeed_sent{0u};
  uint32_t ltfj_nfz_sent{0u};
  uint32_t hospital_nfz_sent{0u};
  uint32_t ltbx_nfz_sent{0u};
  uint32_t airborne_disarm_sent{0u};
  uint32_t flight_term_sent{0u};
  uint32_t corrupted_crc_sent{0u};
  uint32_t deadman_timeout_sent{0u};

  uint64_t fcu_valid_packets_received{0u};
  uint64_t fcu_unsafe_packets_leaked{0u};
  uint64_t fcu_corrupted_packets_leaked{0u};
  uint64_t planner_acks_denied_received{0u};
  uint64_t planner_statustext_alerts_received{0u};
  uint64_t valid_packets_dropped{0u};

  ShieldStats shield_stats{};
  LatencyStatsUs latency_us{};
  double elapsed_ms{0.0};
  std::string report_sha256{};
};

// Run a live 3-endpoint localhost POSIX UDP stress proof:
//   MockFCU (ephemeral 127.0.0.1:0) <-> VertiRTA MavlinkRtaBridge (ephemeral 127.0.0.1:0) <-> AdversarialOffboardPlanner
// Optionally writes a SHA-256 sealed JSON report if out_json_path is non-empty.
[[nodiscard]] UdpLoopbackStressReport run_udp_loopback_stress_proof(
    uint32_t total_offboard_packets = 2400u,
    const std::string& out_json_path = "");

// Run a live long-running UDP proxy bridging an external Offboard planner port to a target FCU UDP host:port.
int run_live_udp_proxy(
    uint16_t listen_offboard_port,
    const std::string& fcu_host,
    uint16_t fcu_port,
    const ShieldConfig& cfg,
    std::atomic<bool>* stop_flag = nullptr);

}  // namespace vertirta::mavlink
