#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "vertirta/copilot/rta_state_contract.hpp"

namespace vertirta::stl {

inline constexpr std::size_t kMaxWindowSamples = 601U;  // 0..600 s at 1 Hz (zero heap)
inline constexpr std::size_t kMaxPolyFacets = 24U;
inline constexpr std::size_t kMaxZones = 6U;  // LTFJ, LTBX, DYN_NFZ_0..3

struct HalfSpace2D {
  double ax{0.0};
  double ay{0.0};
  double b{0.0};  // Inside polygon iff ax * E + ay * N <= b
};

struct ZoneHalfSpaces {
  std::array<HalfSpace2D, kMaxPolyFacets> facets{};
  uint8_t count{0U};
  bool active{false};
};

struct Sample {
  uint32_t idx{0U};
  float val{0.0f};
};

// Fixed-capacity circular monotonic deque (Lemire's O(1)-amortized sliding min/max wedge).
// DR-6 Critical Erratum #4 fix: ALWAYS evict expired head_ entries BEFORE pushing tail_
// so a full window (Cap=601, span=600, tau=601) NEVER wraps tail_ onto head_.
template <std::size_t Cap, bool IsMin>
class FixedMonotonicWedge {
  static_assert(Cap >= 2U, "FixedMonotonicWedge requires Cap >= 2");

 public:
  void reset() noexcept {
    head_ = 0U;
    tail_ = 0U;
    count_ = 0U;
  }

  // Push sample at arrival tick tau; returns exact extremum over [tau - span_ba, tau].
  [[nodiscard]] float push_and_query(uint32_t tau, float v, uint32_t span_ba) noexcept {
    if (std::isnan(v)) {
      v = -std::numeric_limits<float>::infinity();
    }
    const uint32_t eff_span = (span_ba < Cap) ? span_ba : static_cast<uint32_t>(Cap - 1U);
    const uint32_t min_idx = (tau >= eff_span) ? (tau - eff_span) : 0U;

    // 1. Evict expired front entries FIRST so count_ drops below Cap before pushing tail_.
    while (count_ > 0U && buf_[head_].idx < min_idx) {
      head_ = (head_ + 1U) % Cap;
      --count_;
    }

    // 2. Pop dominated back entries to maintain monotonicity.
    while (count_ > 0U) {
      const std::size_t back_idx = (tail_ + Cap - 1U) % Cap;
      const Sample& back = buf_[back_idx];
      const bool dominated = IsMin ? (back.val >= v) : (back.val <= v);
      if (!dominated) {
        break;
      }
      tail_ = back_idx;
      --count_;
    }

    // 3. Defensive capacity guard (only reachable on non-monotonic duplicate tau).
    if (count_ == Cap) {
      head_ = (head_ + 1U) % Cap;
      --count_;
    }

    // 4. Push new sample at tail_.
    buf_[tail_] = Sample{tau, v};
    tail_ = (tail_ + 1U) % Cap;
    ++count_;

    return buf_[head_].val;
  }

  [[nodiscard]] std::size_t size() const noexcept { return count_; }

 private:
  std::array<Sample, Cap> buf_{};
  std::size_t head_{0U};
  std::size_t tail_{0U};
  std::size_t count_{0U};
};

struct RosiBound {
  float inf{-std::numeric_limits<float>::infinity()};
  float sup{+std::numeric_limits<float>::infinity()};

  [[nodiscard]] constexpr bool definitively_violated() const noexcept { return sup < 0.0f; }
  [[nodiscard]] constexpr bool definitively_satisfied() const noexcept { return inf >= 0.0f; }
};

// Bounded-memory online U_[a, b](phi1, phi2) + real-time RoSI tracking for mission start (t = 0).
template <std::size_t Cap = kMaxWindowSamples>
class OnlineBoundedUntil {
  static_assert(Cap >= 2U, "OnlineBoundedUntil requires Cap >= 2");

 public:
  void configure(uint32_t a, uint32_t b) noexcept {
    b_ = (b < Cap) ? b : static_cast<uint32_t>(Cap - 1U);
    a_ = (a <= b_) ? a : b_;
    ring_len_ = b_ + 1U;
    live_min_r1_ = +std::numeric_limits<float>::infinity();
    live_best_u_ = -std::numeric_limits<float>::infinity();
    mission_locked_ = false;
    mission_rho_ = -std::numeric_limits<float>::infinity();
    ring1_.fill(0.0f);
    ring2_.fill(0.0f);
  }

  // Feed sample at tick tau; emits exact pastified rho for t = tau - b once tau >= b.
  bool step(uint32_t tau, float r1, float r2, float& out_exact_rho, RosiBound& out_rosi) noexcept {
    if (std::isnan(r1)) {
      r1 = -std::numeric_limits<float>::infinity();
    }
    if (std::isnan(r2)) {
      r2 = -std::numeric_limits<float>::infinity();
    }
    const std::size_t slot = static_cast<std::size_t>(tau % ring_len_);
    ring1_[slot] = r1;
    ring2_[slot] = r2;

    if (tau == 0U) {
      live_min_r1_ = r1;
      live_best_u_ =
          (a_ == 0U) ? std::min(r1, r2) : -std::numeric_limits<float>::infinity();
    } else if (tau <= b_) {
      live_min_r1_ = std::min(live_min_r1_, r1);
      if (tau >= a_) {
        live_best_u_ = std::max(live_best_u_, std::min(live_min_r1_, r2));
      }
    }
    if (tau == b_ && !mission_locked_) {
      mission_locked_ = true;
      mission_rho_ = live_best_u_;
    }

    out_rosi.inf = live_best_u_;
    out_rosi.sup = (tau < b_) ? std::max(live_best_u_, live_min_r1_) : live_best_u_;

    if (tau < b_) {
      return false;
    }
    const uint32_t t = tau - b_;
    const uint32_t s = t + a_;

    float g_pref = +std::numeric_limits<float>::infinity();
    for (uint32_t j = t; j < s; ++j) {
      g_pref = std::min(g_pref, ring1_[static_cast<std::size_t>(j % ring_len_)]);
    }
    float run_min = +std::numeric_limits<float>::infinity();
    float u_0h = -std::numeric_limits<float>::infinity();
    for (uint32_t k = s; k <= tau; ++k) {
      run_min = std::min(run_min, ring1_[static_cast<std::size_t>(k % ring_len_)]);
      u_0h = std::max(u_0h, std::min(run_min, ring2_[static_cast<std::size_t>(k % ring_len_)]));
    }
    out_exact_rho = std::min(g_pref, u_0h);
    return true;
  }

  [[nodiscard]] bool mission_locked() const noexcept { return mission_locked_; }
  [[nodiscard]] float mission_rho() const noexcept { return mission_rho_; }

 private:
  uint32_t a_{0U};
  uint32_t b_{100U};
  uint32_t ring_len_{101U};
  std::array<float, Cap> ring1_{};
  std::array<float, Cap> ring2_{};
  float live_min_r1_{+std::numeric_limits<float>::infinity()};
  float live_best_u_{-std::numeric_limits<float>::infinity()};
  bool mission_locked_{false};
  float mission_rho_{-std::numeric_limits<float>::infinity()};
};

struct StlSignalSample {
  double east_m{0.0};
  double north_m{0.0};
  float agl_m{0.0f};
  float soc_pct{0.0f};
  float groundspeed_mps{0.0f};
  float ood_var{0.0f};
};

struct StlTickOutput {
  RosiBound rosi{};
  float mission_rho{-std::numeric_limits<float>::infinity()};
  float streaming_rho{-std::numeric_limits<float>::infinity()};
  bool mission_complete{false};
  bool violated{false};
};

class StlMonitorEngine {
 public:
  void set_zone_halfspaces(copilot::ZoneId zone, std::span<const HalfSpace2D> facets) noexcept;
  void configure(const copilot::BoundedStlFormula& formula) noexcept;
  [[nodiscard]] StlTickOutput step(uint32_t tau, const StlSignalSample& s) noexcept;

 private:
  static constexpr std::size_t kMaxAlwaysSlots = 8U;
  static constexpr std::size_t kMaxEventuallySlots = 4U;
  static constexpr std::size_t kMaxUntilSlots = 4U;

  struct TemporalTrack {
    float mission_acc{0.0f};
    bool started{false};
    bool locked{false};
    uint8_t slot{0xFFU};
  };

  [[nodiscard]] float eval_outside_zone(copilot::ZoneId zone, double east_m,
                                        double north_m) const noexcept;

  copilot::BoundedStlFormula formula_{};
  std::array<ZoneHalfSpaces, kMaxZones> zones_{};
  std::array<FixedMonotonicWedge<kMaxWindowSamples, true>, kMaxAlwaysSlots> always_wedges_{};
  std::array<FixedMonotonicWedge<kMaxWindowSamples, false>, kMaxEventuallySlots>
      eventually_wedges_{};
  std::array<OnlineBoundedUntil<kMaxWindowSamples>, kMaxUntilSlots> until_ops_{};
  std::array<TemporalTrack, copilot::kMaxStlNodes> tracks_{};
  bool configured_{false};
};

}  // namespace vertirta::stl
