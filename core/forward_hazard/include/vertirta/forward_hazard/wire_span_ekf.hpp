#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/frames.hpp"
#include "vertirta/forward_hazard/forward_hazard_provider.hpp"

namespace vertirta::forward_hazard {

struct WireSpanEkfConfig {
  double interp_plane_chi2_gate{9.21};  // 2-DOF chi^2 p=0.01
  double interp_plane_sigma_rad{0.015};
  double radar_range_sigma_m{0.35};
  double radar_angle_sigma_rad{0.025};
  double grazing_parallax_min_deg{3.0};
  double grazing_standoff_range_m{35.0};
  double grazing_standoff_extra_radius_m{1.50};
  double wire_clearance_radius_m{0.35};
  double track_timeout_s{2.0};
  uint32_t min_vision_hits_for_standoff{3};
  uint32_t max_segments_per_span{4};
};

// 6-state thin-wire catenary span estimate in local NED:
//   x = [p_mid_n, p_mid_e, p_mid_d, dir_az_rad, dir_el_rad, sag_m]
struct WireSpanTrack {
  uint32_t id{0};
  uint64_t last_update_ns{0};
  uint8_t voltage_class{0};

  // 6-state mean and diagonal/coupled 6x6 covariance (row-major).
  std::array<double, 6> x{};
  std::array<double, 36> P{};
  double half_length_m{20.0};

  // Multi-frame vision bearing/plane accumulator for >3 deg SVD grazing parallax fallback.
  bool has_anchor_obs{false};
  core::Vec3Ned anchor_cam_pos_ned{};
  core::Vec3Ned anchor_ray_unit_ned{};
  core::Vec3Ned anchor_plane_n_ned{};
  double max_parallax_deg{0.0};

  uint32_t vision_hits{0};
  uint32_t radar_hits{0};
  bool radar_locked{false};
  bool svd_triangulated{false};
  bool standoff_fallback{false};
  bool active{false};
};

// Fixed-capacity 6-state wire-span EKF + >3 deg grazing-angle SVD parallax fallback (zero heap).
class WireSpanEkfBank {
 public:
  static constexpr std::size_t kMaxSpanTracks = 32;

  explicit WireSpanEkfBank(const WireSpanEkfConfig& cfg = WireSpanEkfConfig{}) noexcept;

  void clear() noexcept;

  [[nodiscard]] const WireSpanEkfConfig& config() const noexcept { return cfg_; }

  // Prune stale tracks older than track_timeout_s.
  void prune_stale(uint64_t now_ns) noexcept;

  // Associate & update with a 3D camera interpretation plane in NED (chi^2 <= 9.21 Mahalanobis gate).
  // Also accumulates multi-frame baseline parallax for >3 deg SVD triangulation when radar drops out.
  void ingest_vision_plane(
      uint64_t stamp_ns,
      const core::Vec3Ned& cam_pos_ned,
      const core::Vec3Ned& ray_mid_unit_ned,
      const core::Vec3Ned& plane_normal_unit_ned,
      uint8_t voltage_class) noexcept;

  // Fuse a static 77 GHz radar glint/Bragg point in local NED into matching wire span EKF (or spawn new span).
  void ingest_radar_wire_point(
      uint64_t stamp_ns,
      const core::Vec3Ned& radar_pt_ned,
      const core::Vec3Ned& wire_tangent_hint_ned,
      double snr_db) noexcept;

  // Tessellate active tracks into piecewise catenary WireCapsules (<=64 total, R_veh=2.10 m, prioritized by TTC).
  void emit_wire_capsules(
      const core::Vec3Ned& own_pos_ned,
      const core::Vec3Ned& own_vel_ned,
      rta::WireCapsuleTable& out_table,
      uint32_t& out_active_spans,
      bool& out_grazing_parallax_active,
      bool& out_standoff_emitted) const noexcept;

  [[nodiscard]] std::size_t active_track_count() const noexcept;

 private:
  [[nodiscard]] std::size_t allocate_slot(uint64_t stamp_ns) noexcept;

  WireSpanEkfConfig cfg_{};
  std::array<WireSpanTrack, kMaxSpanTracks> tracks_{};
  uint32_t next_id_{1};
};

}  // namespace vertirta::forward_hazard
