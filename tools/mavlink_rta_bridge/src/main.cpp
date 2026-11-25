// SPDX-License-Identifier: Apache-2.0
// VertiRTA Zero-GPL Native C++20 MAVLink v2 UDP/Offboard RTA Shield Bridge CLI.

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "vertirta/mavlink/udp_bridge.hpp"

namespace {

std::atomic<bool> g_stop_requested{false};

void handle_signal(int) noexcept {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

void print_usage() {
  std::cout
      << "VertiRTA Zero-GPL MAVLink v2 UDP/Offboard RTA Shield Bridge\n"
      << "Usage:\n"
      << "  vertirta_mavlink_rta_bridge --self-test-udp-proof <out.json> [--packets <N>=2500]\n"
      << "  vertirta_mavlink_rta_bridge --listen-offboard-port <PORT> --fcu-host <IP> --fcu-port <PORT>\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string proof_out_path;
  uint32_t packets = 2500u;
  uint16_t listen_port = 0u;
  std::string fcu_host = "127.0.0.1";
  uint16_t fcu_port = 0u;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--self-test-udp-proof" && i + 1 < argc) {
      proof_out_path = argv[++i];
    } else if (arg == "--packets" && i + 1 < argc) {
      packets = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--listen-offboard-port" && i + 1 < argc) {
      listen_port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--fcu-host" && i + 1 < argc) {
      fcu_host = argv[++i];
    } else if (arg == "--fcu-port" && i + 1 < argc) {
      fcu_port = static_cast<uint16_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      print_usage();
      return 2;
    }
  }

  if (!proof_out_path.empty()) {
    const auto rep = vertirta::mavlink::run_udp_loopback_stress_proof(packets, proof_out_path);
    std::cout << "MAVLink v2 UDP Loopback Stress Proof: " << (rep.passed ? "PASS" : "FAIL")
              << " | packets=" << rep.total_offboard_packets_sent
              << " | dropped=" << rep.valid_packets_dropped
              << " | unsafe_leaked=" << rep.fcu_unsafe_packets_leaked
              << " | crc_rejected=" << rep.shield_stats.crc_rejected
              << " | clamped=" << rep.shield_stats.setpoints_clamped
              << " | denied=" << rep.shield_stats.commands_denied
              << " | deadman=" << rep.shield_stats.deadman_overrides
              << " | p50=" << rep.latency_us.p50_us << "us"
              << " | p95=" << rep.latency_us.p95_us << "us"
              << " | p99=" << rep.latency_us.p99_us << "us"
              << " | max=" << rep.latency_us.max_us << "us"
              << " | sha256=" << rep.report_sha256 << "\n";
    return rep.passed ? 0 : 1;
  }

  if (listen_port == 0u || fcu_port == 0u) {
    print_usage();
    return 2;
  }

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  vertirta::mavlink::ShieldConfig cfg{};
  return vertirta::mavlink::run_live_udp_proxy(
      listen_port, fcu_host, fcu_port, cfg, &g_stop_requested);
}
