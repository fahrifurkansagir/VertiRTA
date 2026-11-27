#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"

namespace vertirta::rta {

struct StalenessPolicy {
  double delta_r_stale_m{0.0};
  double v_cap_mps{22.0};
  bool divert_to_slz{false};
};

// Deterministic ESDF / ADS-B / Remote-ID staleness escalation (DR-2 Section 4.3):
//   age <= 0.25 s: delta_r = 0
//   0.25 < age <= 1.5 s: delta_r = v_unc*(age-0.25) + 0.5*a_unc*(age-0.25)^2
//   age > 1.0 s: v_cap = 8.0 m/s
//   age > 2.0 s: divert_to_slz = true
[[nodiscard]] StalenessPolicy compute_staleness_policy(double age_s) noexcept;

// Official DHMI AIP LTFJ (polygon + 5 NM arc incl. NW J2 pinch) & LTBX (5 NM arc) CTR walls.
class AirspaceExclusion {
 public:
  // Positive margin [m] = outside CTR (safe); negative = inside CTR (violation).
  [[nodiscard]] static double signed_margin_ltfj_m(
      double lat_deg,
      double lon_deg,
      double alt_amsl_m) noexcept;

  [[nodiscard]] static double signed_margin_ltbx_m(
      double lat_deg,
      double lon_deg,
      double alt_amsl_m) noexcept;

  // Generate up to 2 active CTR wall HazardPoints in NED when near LTFJ/LTBX boundaries.
  [[nodiscard]] static std::size_t extract_ctr_hazards(
      double lat_deg,
      double lon_deg,
      double alt_amsl_m,
      const core::Vec3Ned& own_pos_ned,
      std::span<HazardPoint> out) noexcept;
};

struct IntruderTrack {
  uint32_t track_id{0};
  uint64_t last_meas_ns{0};
  std::array<double, 6> x_enu{};   // [p_E, p_N, p_U, v_E, v_N, v_U]
  std::array<double, 36> P{};      // 6x6 covariance (row-major)
  double r_sep_h_m{12.0};
  double r_sep_z_m{6.0};
  bool active{false};
};

// Fixed-capacity 6-state Constant-Velocity KF for ADS-B / Remote-ID traffic (up to 16 tracks, 0 heap).
class IntruderTrackerKf {
 public:
  static constexpr std::size_t kMaxTracks = 16;

  IntruderTrackerKf() noexcept = default;
  void clear() noexcept;

  void update_measurement(
      uint32_t track_id,
      const core::Vec3Enu& pos_enu,
      const core::Vec3Enu& vel_enu,
      uint64_t meas_ns) noexcept;

  // Forward-predict active tracks to now_ns (incorporating latency & staleness inflation) into HazardPoints.
  [[nodiscard]] std::size_t extract_hazards(
      uint64_t now_ns,
      std::span<HazardPoint> out) const noexcept;

  // Read-only spatial nearest-neighbor association against active, non-expired tracks
  // forward-predicted to now_ns. Excludes any track IDs in excluded_ids for 1-to-1 frame matching.
  [[nodiscard]] std::optional<uint32_t> match_nearest_track_id(
      const core::Vec3Enu& pos_enu,
      uint64_t now_ns,
      double max_gate_m,
      std::span<const uint32_t> excluded_ids = {},
      double* matched_dist_m_out = nullptr) const noexcept;

  // Read-only lookup for an active track by ID (returns nullptr if inactive/absent).
  [[nodiscard]] const IntruderTrack* find_track(uint32_t track_id) const noexcept;

  [[nodiscard]] std::size_t active_count() const noexcept;

 private:
  std::array<IntruderTrack, kMaxTracks> tracks_{};
};

}  // namespace vertirta::rta
