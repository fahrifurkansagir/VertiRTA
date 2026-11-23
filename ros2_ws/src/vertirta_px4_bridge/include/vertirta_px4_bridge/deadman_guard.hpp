// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace vertirta::px4_bridge {

// Pure C++20 fail-closed 100 ms staleness deadman guard for /vertirta/internal/rta_cmd.
// Prevents VertiRtaMode from holding stale acceleration commands if vertirta_rta_node drops out.
inline constexpr uint64_t kRtaCmdDeadmanTimeoutNs = 100'000'000ULL;

struct DeadmanFallbackOutput {
  std::array<float, 3> accel_ned{0.0f, 0.0f, 0.0f};
  float yawspeed_sp{0.0f};
  float vz_limit_mps{1.5f};
  bool deadman_triggered{false};
};

[[nodiscard]] inline bool is_rta_cmd_stale(uint64_t now_ns, uint64_t last_rx_ns) noexcept {
  if (last_rx_ns == 0ULL) {
    return true;
  }
  return (now_ns > last_rx_ns) && ((now_ns - last_rx_ns) > kRtaCmdDeadmanTimeoutNs);
}

[[nodiscard]] inline DeadmanFallbackOutput apply_rta_deadman_fallback(
    uint64_t now_ns,
    uint64_t last_rx_ns,
    bool cel_latched,
    const std::array<float, 3>& a_in,
    float yawspeed_in,
    float vz_limit_in) noexcept {
  DeadmanFallbackOutput out{};
  if (!is_rta_cmd_stale(now_ns, last_rx_ns)) {
    out.accel_ned = a_in;
    out.yawspeed_sp = yawspeed_in;
    out.vz_limit_mps = vz_limit_in;
    out.deadman_triggered = false;
    return out;
  }
  out.deadman_triggered = true;
  out.accel_ned = {0.0f, 0.0f, 0.0f};
  out.yawspeed_sp = cel_latched ? -2.2f : 0.0f;
  out.vz_limit_mps = std::min(vz_limit_in, 1.5f);
  return out;
}

}  // namespace vertirta::px4_bridge
