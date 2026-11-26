// SPDX-License-Identifier: Apache-2.0
#include "vertirta/mavlink/udp_bridge.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <thread>

#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace vertirta::mavlink {

namespace {

using SteadyClock = std::chrono::steady_clock;

[[nodiscard]] uint64_t steady_now_us() noexcept {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(SteadyClock::now().time_since_epoch()).count());
}

class UdpSocket {
 public:
  UdpSocket() noexcept = default;
  ~UdpSocket() noexcept { close_fd(); }

  UdpSocket(const UdpSocket&) = delete;
  UdpSocket& operator=(const UdpSocket&) = delete;

  bool bind_ipv4(uint32_t addr_host_order, uint16_t port = 0u) noexcept {
    close_fd();
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) {
      return false;
    }

    int rcvbuf = 2 * 1024 * 1024;
    (void)::setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));
    int sndbuf = 2 * 1024 * 1024;
    (void)::setsockopt(fd_, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(addr_host_order);

    if (::bind(fd_, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) != 0) {
      close_fd();
      return false;
    }

    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&bound), &len) != 0) {
      close_fd();
      return false;
    }
    port_ = ntohs(bound.sin_port);
    return true;
  }

  bool bind_loopback(uint16_t port = 0u) noexcept {
    return bind_ipv4(INADDR_LOOPBACK, port);
  }

  bool bind_any(uint16_t port = 0u) noexcept {
    return bind_ipv4(INADDR_ANY, port);
  }

  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] uint16_t port() const noexcept { return port_; }

  bool send_to(std::span<const uint8_t> data, const sockaddr_in& dst) const noexcept {
    if (fd_ < 0 || data.empty()) {
      return false;
    }
    const ssize_t sent = ::sendto(
        fd_, data.data(), data.size(), 0, reinterpret_cast<const sockaddr*>(&dst), sizeof(dst));
    return sent == static_cast<ssize_t>(data.size());
  }

  [[nodiscard]] ssize_t recv_from(
      std::span<uint8_t> buf, sockaddr_in& src, int timeout_ms) const noexcept {
    if (fd_ < 0) {
      return -1;
    }
    pollfd pfd{};
    pfd.fd = fd_;
    pfd.events = POLLIN;
    const int pr = ::poll(&pfd, 1, timeout_ms);
    if (pr <= 0) {
      return 0;
    }
    socklen_t slen = sizeof(src);
    return ::recvfrom(fd_, buf.data(), buf.size(), 0, reinterpret_cast<sockaddr*>(&src), &slen);
  }

 private:
  void close_fd() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
      port_ = 0u;
    }
  }

  int fd_{-1};
  uint16_t port_{0u};
};

[[nodiscard]] sockaddr_in make_loopback_addr(uint16_t port) noexcept {
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_port = htons(port);
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  return a;
}

[[nodiscard]] LatencyStatsUs compute_latency_stats(std::vector<double> samples_us) {
  LatencyStatsUs s{};
  if (samples_us.empty()) {
    return s;
  }
  std::sort(samples_us.begin(), samples_us.end());
  const std::size_t n = samples_us.size();
  auto pct = [&](double q) -> double {
    const double idx = q * static_cast<double>(n - 1u);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = std::min(n - 1u, lo + 1u);
    const double frac = idx - static_cast<double>(lo);
    return samples_us[lo] * (1.0 - frac) + samples_us[hi] * frac;
  };
  s.sample_count = n;
  s.min_us = samples_us.front();
  s.p50_us = pct(0.50);
  s.p95_us = pct(0.95);
  s.p99_us = pct(0.99);
  s.max_us = samples_us.back();
  const double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
  s.mean_us = sum / static_cast<double>(n);
  return s;
}

[[nodiscard]] std::string sha256_hex(const std::string& input) {
  uint8_t digest[32]{};
  rta::compute_sha256(input.data(), input.size(), digest);
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (uint8_t b : digest) {
    oss << std::setw(2) << static_cast<unsigned>(b);
  }
  return oss.str();
}

[[nodiscard]] std::string format_proof_json(const UdpLoopbackStressReport& r, bool with_sha) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);
  oss << "{\n";
  oss << "  \"schema_version\": \"vertirta.mavlink_v2_rta_bridge_proof/v1.0.0\",\n";
  oss << "  \"license\": \"Apache-2.0 (Zero-GPL clean-room C++20)\",\n";
  oss << "  \"transport\": \"POSIX_UDP_LOOPBACK_127_0_0_1_EPHEMERAL\",\n";
  oss << "  \"passed\": " << (r.passed ? "true" : "false") << ",\n";
  oss << "  \"ephemeral_ports\": {\n";
  oss << "    \"mock_fcu_port\": " << r.fcu_port << ",\n";
  oss << "    \"bridge_fcu_port\": " << r.bridge_fcu_port << ",\n";
  oss << "    \"bridge_offboard_port\": " << r.bridge_offboard_port << ",\n";
  oss << "    \"adversarial_planner_port\": " << r.planner_port << "\n";
  oss << "  },\n";
  oss << "  \"workload_breakdown\": {\n";
  oss << "    \"total_offboard_packets_sent\": " << r.total_offboard_packets_sent << ",\n";
  oss << "    \"nominal_safe_setpoints\": " << r.nominal_sent << ",\n";
  oss << "    \"plunge_overspeed_attacks\": " << r.plunge_overspeed_sent << ",\n";
  oss << "    \"ltfj_ctr_nfz_breach_attacks\": " << r.ltfj_nfz_sent << ",\n";
  oss << "    \"kartal_hospital_nfz_breach_attacks\": " << r.hospital_nfz_sent << ",\n";
  oss << "    \"ltbx_ctr_nfz_breach_attacks\": " << r.ltbx_nfz_sent << ",\n";
  oss << "    \"airborne_disarm_attacks\": " << r.airborne_disarm_sent << ",\n";
  oss << "    \"flight_termination_attacks\": " << r.flight_term_sent << ",\n";
  oss << "    \"corrupted_crc_spoofed_frames\": " << r.corrupted_crc_sent << ",\n";
  oss << "    \"deadman_heartbeat_timeout_setpoints\": " << r.deadman_timeout_sent << "\n";
  oss << "  },\n";
  oss << "  \"safety_verification\": {\n";
  oss << "    \"fcu_valid_packets_received\": " << r.fcu_valid_packets_received << ",\n";
  oss << "    \"fcu_unsafe_packets_leaked\": " << r.fcu_unsafe_packets_leaked << ",\n";
  oss << "    \"fcu_corrupted_packets_leaked\": " << r.fcu_corrupted_packets_leaked << ",\n";
  oss << "    \"valid_packets_dropped\": " << r.valid_packets_dropped << ",\n";
  oss << "    \"planner_acks_denied_received\": " << r.planner_acks_denied_received << ",\n";
  oss << "    \"planner_statustext_alerts_received\": " << r.planner_statustext_alerts_received << ",\n";
  const uint64_t unsafe_sent = r.plunge_overspeed_sent + r.ltfj_nfz_sent + r.hospital_nfz_sent
                             + r.ltbx_nfz_sent + r.airborne_disarm_sent + r.flight_term_sent
                             + r.deadman_timeout_sent;
  const double unsafe_interception_pct =
      (unsafe_sent > 0U && r.fcu_unsafe_packets_leaked <= unsafe_sent)
          ? 100.0 * static_cast<double>(unsafe_sent - r.fcu_unsafe_packets_leaked)
                / static_cast<double>(unsafe_sent)
          : 0.0;
  const double crc_rejection_pct =
      (r.corrupted_crc_sent > 0U && r.fcu_corrupted_packets_leaked <= r.corrupted_crc_sent)
          ? 100.0 * static_cast<double>(r.corrupted_crc_sent - r.fcu_corrupted_packets_leaked)
                / static_cast<double>(r.corrupted_crc_sent)
          : 0.0;
  oss << "    \"unsafe_interception_rate_pct\": " << unsafe_interception_pct << ",\n";
  oss << "    \"corrupted_crc_rejection_rate_pct\": " << crc_rejection_pct << "\n";
  oss << "  },\n";
  oss << "  \"shield_counters\": {\n";
  oss << "    \"fcu_rx_valid\": " << r.shield_stats.fcu_rx_valid << ",\n";
  oss << "    \"offboard_rx_valid\": " << r.shield_stats.offboard_rx_valid << ",\n";
  oss << "    \"crc_rejected\": " << r.shield_stats.crc_rejected << ",\n";
  oss << "    \"malformed_rejected\": " << r.shield_stats.malformed_rejected << ",\n";
  oss << "    \"setpoints_passed\": " << r.shield_stats.setpoints_passed << ",\n";
  oss << "    \"setpoints_clamped\": " << r.shield_stats.setpoints_clamped << ",\n";
  oss << "    \"setpoints_clamped_cbvf\": " << r.shield_stats.setpoints_clamped_cbvf << ",\n";
  oss << "    \"setpoints_clamped_nfz\": " << r.shield_stats.setpoints_clamped_nfz << ",\n";
  oss << "    \"setpoints_clamped_plunge_speed\": " << r.shield_stats.setpoints_clamped_plunge_speed << ",\n";
  oss << "    \"deadman_overrides\": " << r.shield_stats.deadman_overrides << ",\n";
  oss << "    \"commands_passed\": " << r.shield_stats.commands_passed << ",\n";
  oss << "    \"commands_denied\": " << r.shield_stats.commands_denied << ",\n";
  oss << "    \"statustext_emitted\": " << r.shield_stats.statustext_emitted << ",\n";
  oss << "    \"command_acks_emitted\": " << r.shield_stats.command_acks_emitted << "\n";
  oss << "  },\n";
  oss << "  \"latency_us\": {\n";
  oss << "    \"sample_count\": " << r.latency_us.sample_count << ",\n";
  oss << "    \"min_us\": " << r.latency_us.min_us << ",\n";
  oss << "    \"p50_us\": " << r.latency_us.p50_us << ",\n";
  oss << "    \"p95_us\": " << r.latency_us.p95_us << ",\n";
  oss << "    \"p99_us\": " << r.latency_us.p99_us << ",\n";
  oss << "    \"max_us\": " << r.latency_us.max_us << ",\n";
  oss << "    \"mean_us\": " << r.latency_us.mean_us << "\n";
  oss << "  },\n";
  oss << "  \"elapsed_ms\": " << r.elapsed_ms;
  if (with_sha) {
    oss << ",\n  \"report_sha256\": \"" << r.report_sha256 << "\"\n";
  } else {
    oss << "\n";
  }
  oss << "}\n";
  return oss.str();
}

}  // namespace

UdpLoopbackStressReport run_udp_loopback_stress_proof(
    uint32_t total_offboard_packets,
    const std::string& out_json_path) {
  const auto t_start = SteadyClock::now();
  UdpLoopbackStressReport report{};
  const uint32_t n_packets = std::max<uint32_t>(2000u, total_offboard_packets);

  UdpSocket fcu_sock;
  UdpSocket bridge_fcu_sock;
  UdpSocket bridge_offboard_sock;
  UdpSocket planner_sock;
  if (!fcu_sock.bind_loopback(0u) ||
      !bridge_fcu_sock.bind_loopback(0u) ||
      !bridge_offboard_sock.bind_loopback(0u) ||
      !planner_sock.bind_loopback(0u)) {
    return report;
  }

  report.fcu_port = fcu_sock.port();
  report.bridge_fcu_port = bridge_fcu_sock.port();
  report.bridge_offboard_port = bridge_offboard_sock.port();
  report.planner_port = planner_sock.port();

  const sockaddr_in fcu_addr = make_loopback_addr(fcu_sock.port());
  const sockaddr_in bridge_fcu_addr = make_loopback_addr(bridge_fcu_sock.port());
  const sockaddr_in bridge_offboard_addr = make_loopback_addr(bridge_offboard_sock.port());
  const sockaddr_in planner_addr = make_loopback_addr(planner_sock.port());

  ShieldConfig cfg{};
  cfg.deadman_timeout_us = 500'000u;  // Exact 500 ms deadman spec
  OffboardRtaShield shield(cfg);

  std::atomic<bool> stop_bridge{false};
  std::atomic<bool> stop_fcu{false};
  std::atomic<bool> pause_fcu_heartbeat{false};
  std::atomic<int> fcu_scenario_zone{0};  // 0=nominal, 1=near LTFJ, 2=near Hospital, 3=near LTBX

  std::vector<double> shield_latencies_us;
  shield_latencies_us.reserve(n_packets + 128u);

  std::atomic<uint64_t> fcu_rx_valid{0u};
  std::atomic<uint64_t> fcu_unsafe_leaked{0u};
  std::atomic<uint64_t> fcu_corrupt_leaked{0u};

  // Endpoint 1: MockFCU thread (receives & audits every packet from bridge, emits live telemetry).
  std::thread fcu_thread([&]() {
    std::array<uint8_t, kMaxFrameBytes> rx_buf{};
    std::array<uint8_t, kMaxFrameBytes> tx_buf{};
    uint8_t fcu_seq = 0u;
    auto last_telem = SteadyClock::now() - std::chrono::milliseconds(100);

    while (!stop_fcu.load(std::memory_order_relaxed)) {
      // Emit periodic FCU telemetry at ~50 Hz unless heartbeat is intentionally suspended for deadman test.
      const auto now = SteadyClock::now();
      if (!pause_fcu_heartbeat.load(std::memory_order_relaxed) &&
          (now - last_telem) >= std::chrono::milliseconds(20)) {
        last_telem = now;

        Heartbeat hb{};
        hb.base_mode = kMavModeFlagSafetyArmed;
        hb.system_status = 4u;
        const std::size_t n_hb = encode_heartbeat(hb, fcu_seq++, 1u, 1u, tx_buf);
        (void)fcu_sock.send_to(std::span<const uint8_t>(tx_buf.data(), n_hb), bridge_fcu_addr);

        LocalPositionNed lp{};
        lp.time_boot_ms = static_cast<uint32_t>(steady_now_us() / 1000u);
        const int zone = fcu_scenario_zone.load(std::memory_order_relaxed);
        if (zone == 1) {
          lp.x = 500.0f; lp.y = 90.0f; lp.z = -60.0f;  // ~36 m west of LTFJ wall
        } else if (zone == 2) {
          lp.x = 3200.0f; lp.y = -1165.0f; lp.z = -60.0f;  // ~175 m south of Kartal Hospital
        } else if (zone == 3) {
          lp.x = 6020.0f; lp.y = 2365.0f; lp.z = -60.0f;  // ~5590 m south of LTBX ARP (just outside 5556 m CTR)
        } else {
          lp.x = 500.0f; lp.y = -400.0f; lp.z = -60.0f;  // Nominal coastal corridor
        }
        lp.vx = 4.0f; lp.vy = 0.0f; lp.vz = 0.0f;
        const std::size_t n_lp = encode_local_position_ned(lp, fcu_seq++, 1u, 1u, tx_buf);
        (void)fcu_sock.send_to(std::span<const uint8_t>(tx_buf.data(), n_lp), bridge_fcu_addr);
      }

      sockaddr_in src{};
      const ssize_t n_rx = fcu_sock.recv_from(rx_buf, src, 2);
      if (n_rx > 0) {
        const DecodeResult dec = decode_frame(std::span<const uint8_t>(rx_buf.data(), static_cast<std::size_t>(n_rx)));
        if (!dec.ok()) {
          fcu_corrupt_leaked.fetch_add(1u, std::memory_order_relaxed);
          continue;
        }
        fcu_rx_valid.fetch_add(1u, std::memory_order_relaxed);

        // Audit every received command on the FCU side: zero unsafe commands may EVER leak through!
        if (dec.frame.msgid == kMsgIdCommandLong) {
          const CommandLong cmd = decode_command_long(dec.frame);
          if (cmd.command == kMavCmdDoFlightTermination ||
              (cmd.command == kMavCmdComponentArmDisarm &&
               (!std::isfinite(cmd.param1) || std::abs(cmd.param1 - 1.0f) >= 0.5f))) {
            fcu_unsafe_leaked.fetch_add(1u, std::memory_order_relaxed);
          }
        } else if (dec.frame.msgid == kMsgIdSetPositionTargetLocalNed) {
          const SetPositionTargetLocalNed sp = decode_set_position_target_local_ned(dec.frame);
          const double v_xy = std::hypot(static_cast<double>(sp.vx), static_cast<double>(sp.vy));
          const bool bad_speed = (v_xy > cfg.max_horiz_speed_mps + 0.05) ||
                                 (static_cast<double>(sp.vz) > cfg.max_descent_rate_mps + 0.05) ||
                                 (static_cast<double>(sp.vz) < -cfg.max_ascent_rate_mps - 0.05);
          const bool bad_alt = ((sp.type_mask & kTypeMaskIgnoreZ) == 0u) &&
                               (static_cast<double>(sp.z) > -cfg.min_agl_m + 0.05);
          const bool bad_frame_or_force =
              (sp.coordinate_frame != kMavFrameLocalNed) || ((sp.type_mask & kTypeMaskForce) != 0u);
          if (!std::isfinite(sp.vx) || !std::isfinite(sp.vy) || !std::isfinite(sp.vz) ||
              bad_speed || bad_alt || bad_frame_or_force) {
            fcu_unsafe_leaked.fetch_add(1u, std::memory_order_relaxed);
          }
        }
      }
    }
  });

  // Endpoint 2: VertiRTA MavlinkRtaBridge thread (real POSIX poll over both UDP sockets).
  std::thread bridge_thread([&]() {
    std::array<uint8_t, kMaxFrameBytes> rx_buf{};
    pollfd pfds[2]{};
    pfds[0].fd = bridge_fcu_sock.fd();
    pfds[0].events = POLLIN;
    pfds[1].fd = bridge_offboard_sock.fd();
    pfds[1].events = POLLIN;

    while (!stop_bridge.load(std::memory_order_relaxed)) {
      const int pr = ::poll(pfds, 2, 2);
      if (pr <= 0) {
        continue;
      }

      if ((pfds[0].revents & POLLIN) != 0) {
        sockaddr_in src{};
        socklen_t slen = sizeof(src);
        const ssize_t n_rx = ::recvfrom(
            bridge_fcu_sock.fd(), rx_buf.data(), rx_buf.size(), 0,
            reinterpret_cast<sockaddr*>(&src), &slen);
        if (n_rx > 0) {
          const auto out = shield.process_fcu_packet(
              std::span<const uint8_t>(rx_buf.data(), static_cast<std::size_t>(n_rx)), steady_now_us());
          for (const auto& pkt : out.to_offboard) {
            (void)bridge_offboard_sock.send_to(pkt, planner_addr);
          }
        }
      }

      if ((pfds[1].revents & POLLIN) != 0) {
        sockaddr_in src{};
        socklen_t slen = sizeof(src);
        const ssize_t n_rx = ::recvfrom(
            bridge_offboard_sock.fd(), rx_buf.data(), rx_buf.size(), 0,
            reinterpret_cast<sockaddr*>(&src), &slen);
        if (n_rx > 0) {
          const auto t0 = SteadyClock::now();
          const auto out = shield.process_offboard_packet(
              std::span<const uint8_t>(rx_buf.data(), static_cast<std::size_t>(n_rx)), steady_now_us());
          for (const auto& pkt : out.to_fcu) {
            (void)bridge_fcu_sock.send_to(pkt, fcu_addr);
          }
          for (const auto& pkt : out.to_offboard) {
            (void)bridge_offboard_sock.send_to(pkt, planner_addr);
          }
          const auto t1 = SteadyClock::now();
          const double dt_us =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
          shield_latencies_us.push_back(dt_us);
        }
      }
    }
  });

  // Helper to drain planner RX socket (counts denied ACKs & STATUSTEXT alerts).
  uint64_t planner_acks_denied = 0u;
  uint64_t planner_statustext = 0u;
  auto drain_planner_rx = [&](int timeout_ms) {
    std::array<uint8_t, kMaxFrameBytes> rx_buf{};
    sockaddr_in src{};
    while (true) {
      const ssize_t n_rx = planner_sock.recv_from(rx_buf, src, timeout_ms);
      if (n_rx <= 0) {
        break;
      }
      timeout_ms = 0;  // drain remaining queued datagrams immediately
      const DecodeResult dec = decode_frame(std::span<const uint8_t>(rx_buf.data(), static_cast<std::size_t>(n_rx)));
      if (!dec.ok()) {
        continue;
      }
      if (dec.frame.msgid == kMsgIdCommandAck) {
        const CommandAck ack = decode_command_ack(dec.frame);
        if (ack.result == kMavResultDenied) {
          ++planner_acks_denied;
        }
      } else if (dec.frame.msgid == kMsgIdStatusText) {
        ++planner_statustext;
      }
    }
  };

  // Wait briefly until bridge receives initial FCU HEARTBEAT + LOCAL_POSITION_NED.
  std::this_thread::sleep_for(std::chrono::milliseconds(60));
  drain_planner_rx(0);

  // Also inject one SYS_STATUS + ATTITUDE_QUATERNION telemetry burst from FCU to exercise all telemetry codecs.
  {
    std::array<uint8_t, kMaxFrameBytes> tx_buf{};
    SysStatus ss{};
    ss.voltage_battery = 40320u;
    ss.battery_remaining = 82;
    const std::size_t n1 = encode_sys_status(ss, 200u, 1u, 1u, tx_buf);
    (void)fcu_sock.send_to(std::span<const uint8_t>(tx_buf.data(), n1), bridge_fcu_addr);

    AttitudeQuaternion aq{};
    aq.time_boot_ms = 1000u;
    aq.q1 = 1.0f;
    const std::size_t n2 = encode_attitude_quaternion(aq, 201u, 1u, 1u, tx_buf, false);
    (void)fcu_sock.send_to(std::span<const uint8_t>(tx_buf.data(), n2), bridge_fcu_addr);
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
  }

  // Endpoint 3: AdversarialOffboardPlanner driving >= 2,400 live UDP packets across all attack vectors.
  std::array<uint8_t, kMaxFrameBytes> tx_buf{};
  uint8_t p_seq = 0u;

  const uint32_t deadman_pkts = 40u;
  const uint32_t active_pkts = n_packets - deadman_pkts;

  auto send_and_pace = [&](std::size_t len, uint32_t idx) {
    (void)planner_sock.send_to(std::span<const uint8_t>(tx_buf.data(), len), bridge_offboard_addr);
    if ((idx % 16u) == 0u) {
      drain_planner_rx(0);
      std::this_thread::sleep_for(std::chrono::microseconds(150));
    }
  };

  // Phase A: Nominal, Plunge/Overspeed, Airborne Disarm, Flight Termination, Corrupted CRC (zone 0).
  const uint32_t phase_a = (active_pkts * 65u) / 100u;
  const uint32_t phase_b = (active_pkts * 15u) / 100u;  // LTFJ zone 1
  const uint32_t phase_c = (active_pkts * 10u) / 100u;  // Hospital zone 2
  const uint32_t phase_d = active_pkts - phase_a - phase_b - phase_c;  // LTBX zone 3

  for (uint32_t i = 0u; i < phase_a; ++i) {
    const uint32_t slot = i % 20u;
    if (slot < 10u) {
      // Nominal safe velocity setpoint in coastal corridor.
      SetPositionTargetLocalNed sp{};
      sp.time_boot_ms = i;
      sp.vx = 5.0f;
      sp.vy = -1.0f;
      sp.vz = 0.0f;
      sp.type_mask = kTypeMaskVelocityOnly;
      const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
      send_and_pace(n, i);
      ++report.nominal_sent;
    } else if (slot < 14u) {
      // Aggressive plunge + overspeed attack.
      SetPositionTargetLocalNed sp{};
      sp.time_boot_ms = i;
      sp.x = 500.0f;
      sp.y = -400.0f;
      sp.z = +40.0f;  // Underground plunge!
      sp.vx = 55.0f;  // Severe overspeed!
      sp.vy = 0.0f;
      sp.vz = 25.0f;  // 25 m/s downward plunge!
      sp.type_mask = kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz | kTypeMaskIgnoreYaw;
      const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
      send_and_pace(n, i);
      ++report.plunge_overspeed_sent;
    } else if (slot < 16u) {
      // Airborne disarm attack (MAV_CMD_COMPONENT_ARM_DISARM=400, param1=0 while at 60m AGL).
      CommandLong cmd{};
      cmd.command = kMavCmdComponentArmDisarm;
      cmd.param1 = 0.0f;
      const std::size_t n = encode_command_long(cmd, p_seq++, 255u, 190u, tx_buf);
      send_and_pace(n, i);
      ++report.airborne_disarm_sent;
    } else if (slot == 16u) {
      // Flight termination attack (MAV_CMD_DO_FLIGHTTERMINATION=185).
      CommandLong cmd{};
      cmd.command = kMavCmdDoFlightTermination;
      cmd.param1 = 1.0f;
      const std::size_t n = encode_command_long(cmd, p_seq++, 255u, 190u, tx_buf);
      send_and_pace(n, i);
      ++report.flight_term_sent;
    } else {
      // Corrupted-CRC spoofed frame (flip CRC low byte).
      SetPositionTargetLocalNed sp{};
      sp.time_boot_ms = i;
      sp.vx = 40.0f;
      sp.type_mask = kTypeMaskVelocityOnly;
      const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
      tx_buf[n - 1u] ^= 0x5Au;
      send_and_pace(n, i);
      ++report.corrupted_crc_sent;
    }
  }

  // Phase B: LTFJ CTR eastern wall breach vectors (zone 1).
  fcu_scenario_zone.store(1, std::memory_order_relaxed);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  for (uint32_t i = 0u; i < phase_b; ++i) {
    SetPositionTargetLocalNed sp{};
    sp.time_boot_ms = phase_a + i;
    sp.x = 500.0f;
    sp.y = 450.0f;  // Deep inside LTFJ CTR!
    sp.z = -60.0f;
    sp.vx = 0.0f;
    sp.vy = 20.0f;  // Fast eastward breach into LTFJ!
    sp.vz = 0.0f;
    sp.type_mask = kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz | kTypeMaskIgnoreYaw;
    const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
    send_and_pace(n, i);
    ++report.ltfj_nfz_sent;
  }

  // Phase C: Kartal Dr. Lutfi Kirdar Hospital helipad NFZ breach vectors (zone 2).
  fcu_scenario_zone.store(2, std::memory_order_relaxed);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  for (uint32_t i = 0u; i < phase_c; ++i) {
    SetPositionTargetLocalNed sp{};
    sp.time_boot_ms = phase_a + phase_b + i;
    sp.x = 3375.0f;
    sp.y = -1165.0f;  // Directly on Kartal Hospital helipad!
    sp.z = -60.0f;
    sp.vx = 18.0f;
    sp.vy = 0.0f;
    sp.vz = 0.0f;
    sp.type_mask = kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz | kTypeMaskIgnoreYaw;
    const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
    send_and_pace(n, i);
    ++report.hospital_nfz_sent;
  }

  // Phase D: LTBX CTR 3 NM breach vectors (zone 3).
  fcu_scenario_zone.store(3, std::memory_order_relaxed);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  for (uint32_t i = 0u; i < phase_d; ++i) {
    SetPositionTargetLocalNed sp{};
    sp.time_boot_ms = phase_a + phase_b + phase_c + i;
    sp.x = 7500.0f;
    sp.y = 2365.0f;  // Deep inside LTBX 3 NM CTR!
    sp.z = -60.0f;
    sp.vx = 20.0f;
    sp.vy = 0.0f;
    sp.vz = 0.0f;
    sp.type_mask = kTypeMaskIgnoreAfx | kTypeMaskIgnoreAfy | kTypeMaskIgnoreAfz | kTypeMaskIgnoreYaw;
    const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
    send_and_pace(n, i);
    ++report.ltbx_nfz_sent;
  }

  // Phase E: Deadman heartbeat timeout (> 500 ms FCU heartbeat loss -> fail-closed zero-velocity hover).
  pause_fcu_heartbeat.store(true, std::memory_order_relaxed);
  std::this_thread::sleep_for(std::chrono::milliseconds(540));
  for (uint32_t i = 0u; i < deadman_pkts; ++i) {
    SetPositionTargetLocalNed sp{};
    sp.time_boot_ms = active_pkts + i;
    sp.vx = 10.0f;
    sp.vy = 5.0f;
    sp.vz = 1.0f;
    sp.type_mask = kTypeMaskVelocityOnly;
    const std::size_t n = encode_set_position_target_local_ned(sp, p_seq++, 255u, 190u, tx_buf);
    send_and_pace(n, i);
    ++report.deadman_timeout_sent;
  }

  // Drain remaining in-flight UDP packets cleanly before tearing down threads.
  std::this_thread::sleep_for(std::chrono::milliseconds(80));
  drain_planner_rx(20);

  stop_bridge.store(true, std::memory_order_relaxed);
  bridge_thread.join();
  stop_fcu.store(true, std::memory_order_relaxed);
  fcu_thread.join();
  drain_planner_rx(5);

  report.total_offboard_packets_sent =
      report.nominal_sent + report.plunge_overspeed_sent + report.ltfj_nfz_sent +
      report.hospital_nfz_sent + report.ltbx_nfz_sent + report.airborne_disarm_sent +
      report.flight_term_sent + report.corrupted_crc_sent + report.deadman_timeout_sent;

  report.shield_stats = shield.stats();
  report.fcu_valid_packets_received = fcu_rx_valid.load(std::memory_order_relaxed);
  report.fcu_unsafe_packets_leaked = fcu_unsafe_leaked.load(std::memory_order_relaxed);
  report.fcu_corrupted_packets_leaked = fcu_corrupt_leaked.load(std::memory_order_relaxed);
  report.planner_acks_denied_received = planner_acks_denied;
  report.planner_statustext_alerts_received = planner_statustext;

  const uint64_t expected_valid_offboard =
      static_cast<uint64_t>(report.total_offboard_packets_sent - report.corrupted_crc_sent);
  const uint64_t expected_to_fcu =
      static_cast<uint64_t>(report.nominal_sent + report.plunge_overspeed_sent + report.ltfj_nfz_sent +
                            report.hospital_nfz_sent + report.ltbx_nfz_sent + report.deadman_timeout_sent);
  const uint64_t expected_denied =
      static_cast<uint64_t>(report.airborne_disarm_sent + report.flight_term_sent);
  const uint64_t expected_clamped =
      static_cast<uint64_t>(report.plunge_overspeed_sent + report.ltfj_nfz_sent +
                            report.hospital_nfz_sent + report.ltbx_nfz_sent);

  const uint64_t dropped_rx =
      (expected_valid_offboard > report.shield_stats.offboard_rx_valid)
          ? (expected_valid_offboard - report.shield_stats.offboard_rx_valid)
          : 0u;
  const uint64_t dropped_fcu =
      (expected_to_fcu > report.fcu_valid_packets_received)
          ? (expected_to_fcu - report.fcu_valid_packets_received)
          : 0u;
  report.valid_packets_dropped = dropped_rx + dropped_fcu;

  report.latency_us = compute_latency_stats(std::move(shield_latencies_us));
  report.elapsed_ms =
      std::chrono::duration<double, std::milli>(SteadyClock::now() - t_start).count();

  report.passed =
      (report.total_offboard_packets_sent >= 2000u) &&
      (report.valid_packets_dropped == 0u) &&
      (report.fcu_unsafe_packets_leaked == 0u) &&
      (report.fcu_corrupted_packets_leaked == 0u) &&
      (report.shield_stats.crc_rejected == report.corrupted_crc_sent) &&
      (report.shield_stats.commands_denied == expected_denied) &&
      (report.shield_stats.setpoints_clamped == expected_clamped) &&
      (report.shield_stats.deadman_overrides == report.deadman_timeout_sent) &&
      (report.shield_stats.setpoints_passed == report.nominal_sent) &&
      (report.planner_acks_denied_received == expected_denied);

  const std::string body_no_sha = format_proof_json(report, false);
  report.report_sha256 = sha256_hex(body_no_sha);

  if (!out_json_path.empty()) {
    const std::filesystem::path p(out_json_path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
    ofs << format_proof_json(report, true);
  }

  return report;
}

std::optional<uint32_t> select_fcu_socket_bind_addr_host_order(
    const std::string& fcu_host) noexcept {
  in_addr addr{};
  if (::inet_pton(AF_INET, fcu_host.c_str(), &addr) != 1) {
    return std::nullopt;
  }
  const uint32_t host_ip = ntohl(addr.s_addr);
  if ((host_ip & 0xFF000000u) == 0x7F000000u) {
    return static_cast<uint32_t>(INADDR_LOOPBACK);
  }
  return static_cast<uint32_t>(INADDR_ANY);
}

int run_live_udp_proxy(
    uint16_t listen_offboard_port,
    const std::string& fcu_host,
    uint16_t fcu_port,
    const ShieldConfig& cfg,
    std::atomic<bool>* stop_flag) {
  sockaddr_in fcu_addr{};
  fcu_addr.sin_family = AF_INET;
  fcu_addr.sin_port = htons(fcu_port);
  if (::inet_pton(AF_INET, fcu_host.c_str(), &fcu_addr.sin_addr) != 1) {
    return 2;
  }
  const auto fcu_bind_addr = select_fcu_socket_bind_addr_host_order(fcu_host);
  if (!fcu_bind_addr.has_value()) {
    return 2;
  }

  UdpSocket offboard_sock;
  UdpSocket fcu_sock;
  const bool fcu_bound = (*fcu_bind_addr == static_cast<uint32_t>(INADDR_LOOPBACK))
                             ? fcu_sock.bind_loopback(0u)
                             : fcu_sock.bind_any(0u);
  if (!offboard_sock.bind_loopback(listen_offboard_port) || !fcu_bound) {
    return 1;
  }

  sockaddr_in last_planner_addr{};
  bool has_planner_addr = false;
  OffboardRtaShield shield(cfg);

  std::array<uint8_t, kMaxFrameBytes> rx_buf{};
  pollfd pfds[2]{};
  pfds[0].fd = fcu_sock.fd();
  pfds[0].events = POLLIN;
  pfds[1].fd = offboard_sock.fd();
  pfds[1].events = POLLIN;

  while (stop_flag == nullptr || !stop_flag->load(std::memory_order_relaxed)) {
    const int pr = ::poll(pfds, 2, 10);
    if (pr <= 0) {
      continue;
    }
    if ((pfds[0].revents & POLLIN) != 0) {
      sockaddr_in src{};
      socklen_t slen = sizeof(src);
      const ssize_t n_rx = ::recvfrom(
          fcu_sock.fd(), rx_buf.data(), rx_buf.size(), 0,
          reinterpret_cast<sockaddr*>(&src), &slen);
      if (n_rx > 0) {
        const auto out = shield.process_fcu_packet(
            std::span<const uint8_t>(rx_buf.data(), static_cast<std::size_t>(n_rx)), steady_now_us());
        if (has_planner_addr) {
          for (const auto& pkt : out.to_offboard) {
            (void)offboard_sock.send_to(pkt, last_planner_addr);
          }
        }
      }
    }
    if ((pfds[1].revents & POLLIN) != 0) {
      sockaddr_in src{};
      socklen_t slen = sizeof(src);
      const ssize_t n_rx = ::recvfrom(
          offboard_sock.fd(), rx_buf.data(), rx_buf.size(), 0,
          reinterpret_cast<sockaddr*>(&src), &slen);
      if (n_rx > 0) {
        last_planner_addr = src;
        has_planner_addr = true;
        const auto out = shield.process_offboard_packet(
            std::span<const uint8_t>(rx_buf.data(), static_cast<std::size_t>(n_rx)), steady_now_us());
        for (const auto& pkt : out.to_fcu) {
          (void)fcu_sock.send_to(pkt, fcu_addr);
        }
        for (const auto& pkt : out.to_offboard) {
          (void)offboard_sock.send_to(pkt, last_planner_addr);
        }
      }
    }
  }
  return 0;
}

}  // namespace vertirta::mavlink
