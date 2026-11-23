// SPDX-License-Identifier: Apache-2.0
#include "vertirta/uspace/vertiport_queue.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace vertirta::uspace {

namespace {

// Deterministic SplitMix64 PRNG for zero-variance reproducible discrete-event queueing.
class DetRng {
 public:
  explicit DetRng(uint64_t seed) noexcept : state_(seed) {}

  [[nodiscard]] double uniform01() noexcept {
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31U);
    return static_cast<double>(z >> 11U) * (1.0 / 9007199254740992.0);
  }

  // Samples positive duration with target mean and squared coefficient of variation Cv^2.
  [[nodiscard]] double sample_cv2(double mean, double cv2) noexcept {
    // 12-uniform Irwin-Hall approximate normal N(0,1)
    double sum = 0.0;
    for (int i = 0; i < 12; ++i) {
      sum += uniform01();
    }
    const double z = sum - 6.0;
    const double sigma = std::sqrt(std::max(0.0, cv2));
    return std::max(0.25 * mean, mean * (1.0 + sigma * z));
  }

 private:
  uint64_t state_;
};

struct VertiportState {
  uint32_t op_fatos{1U};
  uint32_t stands{4U};
  std::array<double, 4U> fato_free_s{};
  std::array<double, 8U> stand_free_s{};
  double fato_busy_s{0.0};
  double stand_busy_s{0.0};
  uint32_t arrivals{0U};
  uint32_t blocked_bbs_sno{0U};
};

}  // namespace

TandemQueueMetrics evaluate_tandem_fleet_queue(
    uint32_t n_aircraft,
    uint32_t op_fatos_hub,
    uint32_t chilled_stands,
    double wake_s,
    double turnaround_min) noexcept {
  TandemQueueMetrics out{};
  out.n_aircraft = n_aircraft;
  if (n_aircraft == 0U || op_fatos_hub == 0U || chilled_stands == 0U) {
    out.stable = false;
    return out;
  }

  constexpr uint32_t kVertiports = 4U;
  constexpr double kWarmupS = 1800.0;
  constexpr double kHorizonS = 14400.0;  // 4 hours measured window
  constexpr double kEndTimeS = kWarmupS + kHorizonS;
  constexpr double kMeanFlightS = 330.0;  // 5.5 min mean Anatolian coastal leg (DR-11 Sec 5)
  constexpr double kFatoCv2 = 0.15;
  constexpr double kStandCv2 = 0.20;

  // Exact stationary 1.25x hub-to-spoke skew [5/18, 4/18, 4/18, 5/18] (hub/spoke = 1.25) with zero self-loops:
  //   From Hub 0 (MLT):   {0.00, 0.34, 0.34, 0.32} -> CDF {0.00, 0.34, 0.68, 1.00}
  //   From Spoke 1 (DRG): {0.425, 0.00, 0.15, 0.425} -> CDF {0.425, 0.425, 0.575, 1.00}
  //   From Spoke 2 (CVZ): {0.425, 0.15, 0.00, 0.425} -> CDF {0.425, 0.575, 0.575, 1.00}
  //   From Hub 3 (KRT):   {0.32, 0.34, 0.34, 0.00} -> CDF {0.32, 0.66, 1.00, 1.00}
  const std::array<double, kVertiports> init_cdf = {5.0 / 18.0, 9.0 / 18.0, 13.0 / 18.0, 1.0};
  const std::array<std::array<double, kVertiports>, kVertiports> trans_cdf = {{
      {{0.000, 0.340, 0.680, 1.000}},
      {{0.425, 0.425, 0.575, 1.000}},
      {{0.425, 0.575, 0.575, 1.000}},
      {{0.320, 0.660, 1.000, 1.000}},
  }};
  std::array<VertiportState, kVertiports> vps{};
  for (uint32_t v = 0U; v < kVertiports; ++v) {
    const bool is_hub = (v == 0U || v == 3U);
    vps[v].op_fatos = std::clamp(is_hub ? op_fatos_hub : 1U, 1U, 4U);
    vps[v].stands = std::clamp(chilled_stands, 1U, 8U);
  }

  DetRng rng(20261123ULL ^ (static_cast<uint64_t>(n_aircraft) << 16U));
  const double mean_turn_s = std::max(60.0, turnaround_min * 60.0);

  // Per-aircraft 2-phase chronological discrete-event state (ARRIVAL vs DEPARTURE)
  enum class EventKind : uint8_t { ARRIVAL = 0, DEPARTURE = 1 };
  struct AircraftEvent {
    double time_s{0.0};
    EventKind kind{EventKind::ARRIVAL};
    uint32_t vp_idx{0U};
    uint32_t stand_idx{0U};
  };

  std::vector<AircraftEvent> events(n_aircraft);
  for (uint32_t a = 0U; a < n_aircraft; ++a) {
    events[a].time_s =
        (static_cast<double>(a) / static_cast<double>(n_aircraft)) * (kMeanFlightS + mean_turn_s);
    events[a].kind = EventKind::ARRIVAL;
    const double u = rng.uniform01();
    uint32_t d = 3U;
    for (uint32_t v = 0U; v < kVertiports; ++v) {
      if (u <= init_cdf[v]) {
        d = v;
        break;
      }
    }
    events[a].vp_idx = d;
  }

  std::vector<double> hold_samples_min{};
  hold_samples_min.reserve(1024U);
  uint32_t completed_flights = 0U;

  // Chronological event loop: always pop earliest event across all aircraft
  for (uint32_t iter = 0U; iter < 100000U; ++iter) {
    uint32_t a_min = 0U;
    double t_now = events[0].time_s;
    for (uint32_t a = 1U; a < n_aircraft; ++a) {
      if (events[a].time_s < t_now) {
        t_now = events[a].time_s;
        a_min = a;
      }
    }
    if (t_now >= kEndTimeS) {
      break;
    }

    auto& ev = events[a_min];
    auto& vp = vps[ev.vp_idx];

    if (ev.kind == EventKind::ARRIVAL) {
      // Earliest free FATO
      uint32_t f_best = 0U;
      for (uint32_t f = 1U; f < vp.op_fatos; ++f) {
        if (vp.fato_free_s[f] < vp.fato_free_s[f_best]) {
          f_best = f;
        }
      }
      // Earliest free charging stand (BBS-SNO: landing holds airborne until stand is confirmed)
      uint32_t s_best = 0U;
      for (uint32_t s = 1U; s < vp.stands; ++s) {
        if (vp.stand_free_s[s] < vp.stand_free_s[s_best]) {
          s_best = s;
        }
      }

      const double t_fato_ready = std::max(t_now, vp.fato_free_s[f_best]);
      const bool bbs_blocked = (vp.stand_free_s[s_best] > t_fato_ready);
      const double t_touchdown = std::max(t_fato_ready, vp.stand_free_s[s_best]);
      const double hold_s = t_touchdown - t_now;

      const double ldg_wake = rng.sample_cv2(wake_s, kFatoCv2);
      const double stand_dur = rng.sample_cv2(mean_turn_s, kStandCv2);

      // Occupy FATO strictly for landing wake; stand until turnaround completes
      vp.fato_free_s[f_best] = t_touchdown + ldg_wake;
      vp.stand_free_s[s_best] = t_touchdown + stand_dur;

      if (t_now >= kWarmupS) {
        ++completed_flights;
        ++vp.arrivals;
        if (bbs_blocked) {
          ++vp.blocked_bbs_sno;
        }
        hold_samples_min.push_back(hold_s / 60.0);
        vp.fato_busy_s += ldg_wake;
        vp.stand_busy_s += stand_dur;
      }

      ev.kind = EventKind::DEPARTURE;
      ev.time_s = t_touchdown + stand_dur;
      ev.stand_idx = s_best;
    } else {
      // DEPARTURE event at actual turnaround completion time t_now
      uint32_t f_dep = 0U;
      for (uint32_t f = 1U; f < vp.op_fatos; ++f) {
        if (vp.fato_free_s[f] < vp.fato_free_s[f_dep]) {
          f_dep = f;
        }
      }
      const double t_dep_start = std::max(t_now, vp.fato_free_s[f_dep]);
      const double dep_wake = rng.sample_cv2(wake_s, kFatoCv2);
      vp.fato_free_s[f_dep] = t_dep_start + dep_wake;
      if (t_dep_start > vp.stand_free_s[ev.stand_idx]) {
        vp.stand_free_s[ev.stand_idx] = t_dep_start;
      }
      if (t_now >= kWarmupS) {
        vp.fato_busy_s += dep_wake;
      }

      // Schedule next corridor leg using exact stationary 1.25x hub-skew Markov chain
      const double leg_s = rng.sample_cv2(kMeanFlightS, 0.08);
      const auto& row = trans_cdf[ev.vp_idx];
      const double u = rng.uniform01();
      uint32_t next_d = (ev.vp_idx + 1U) % kVertiports;
      for (uint32_t v = 0U; v < kVertiports; ++v) {
        if (v != ev.vp_idx && u <= row[v]) {
          next_d = v;
          break;
        }
      }
      ev.kind = EventKind::ARRIVAL;
      ev.vp_idx = next_d;
      ev.time_s = t_dep_start + leg_s;
    }
  }

  if (hold_samples_min.empty()) {
    out.stable = false;
    return out;
  }

  double sum_hold = 0.0;
  for (double h : hold_samples_min) {
    sum_hold += h;
  }
  std::sort(hold_samples_min.begin(), hold_samples_min.end());
  const std::size_t p95_idx = std::min(
      hold_samples_min.size() - 1U,
      static_cast<std::size_t>(std::floor(0.95 * static_cast<double>(hold_samples_min.size()))));

  out.mean_airborne_hold_min = sum_hold / static_cast<double>(hold_samples_min.size());
  out.p95_airborne_hold_min = hold_samples_min[p95_idx];
  out.throughput_flt_per_hr = static_cast<double>(completed_flights) / (kHorizonS / 3600.0);

  double max_hub_fato_u = 0.0;
  double max_hub_stand_u = 0.0;
  uint32_t total_arr = 0U;
  uint32_t total_blk = 0U;
  for (uint32_t v = 0U; v < kVertiports; ++v) {
    const double u_fato = vps[v].fato_busy_s / (kHorizonS * static_cast<double>(vps[v].op_fatos));
    const double u_stand = vps[v].stand_busy_s / (kHorizonS * static_cast<double>(vps[v].stands));
    if (v == 0U || v == 3U) {
      max_hub_fato_u = std::max(max_hub_fato_u, u_fato);
      max_hub_stand_u = std::max(max_hub_stand_u, u_stand);
    }
    total_arr += vps[v].arrivals;
    total_blk += vps[v].blocked_bbs_sno;
  }

  out.hub_fato_utilization = max_hub_fato_u;
  out.hub_stand_utilization = max_hub_stand_u;
  out.bbs_sno_blocking_prob = (total_arr > 0U)
      ? (static_cast<double>(total_blk) / static_cast<double>(total_arr))
      : 0.0;
  out.stable = (out.mean_airborne_hold_min <= 2.0) &&
               (out.hub_stand_utilization < 0.96) &&
               (out.hub_fato_utilization < 0.95) &&
               (out.bbs_sno_blocking_prob < 0.40);
  return out;
}

}  // namespace vertirta::uspace
