// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vertirta::uspace {

struct TandemQueueMetrics {
  uint32_t n_aircraft{0U};
  double mean_airborne_hold_min{0.0};
  double p95_airborne_hold_min{0.0};
  double throughput_flt_per_hr{0.0};
  double hub_fato_utilization{0.0};
  double hub_stand_utilization{0.0};
  double bbs_sno_blocking_prob{0.0};
  bool stable{false};
};

// Deterministic closed-network BBS-SNO tandem vertiport queue evaluation across 4 coastal vertiports
// (2 hubs with 1.25x skew + 2 spokes; M/G/1 FATO Cv^2=0.15 feeding M/G/c/c stands Cv^2=0.20).
[[nodiscard]] TandemQueueMetrics evaluate_tandem_fleet_queue(
    uint32_t n_aircraft,
    uint32_t op_fatos_hub,
    uint32_t chilled_stands,
    double wake_s,
    double turnaround_min) noexcept;

}  // namespace vertirta::uspace
