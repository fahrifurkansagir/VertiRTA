#include "vertirta/forward_hazard/wire_span_ekf.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::forward_hazard {

namespace {

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

[[nodiscard]] core::Vec3Ned normalize_ned(const core::Vec3Ned& v) noexcept {
  const double n = v.norm();
  if (!std::isfinite(n) || n < 1e-9) {
    return {0.0, 1.0, 0.0};
  }
  return v * (1.0 / n);
}

[[nodiscard]] core::Vec3Ned cross_ned(const core::Vec3Ned& a, const core::Vec3Ned& b) noexcept {
  return {
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

[[nodiscard]] core::Vec3Ned span_dir_unit(double az_rad, double el_rad) noexcept {
  const double ce = std::cos(el_rad);
  return {ce * std::cos(az_rad), ce * std::sin(az_rad), std::sin(el_rad)};
}

// 2-view closed-form SVD/normal-equation triangulation between two camera rays (p0, d0) and (p1, d1).
[[nodiscard]] bool triangulate_two_rays_ned(
    const core::Vec3Ned& p0,
    const core::Vec3Ned& d0,
    const core::Vec3Ned& p1,
    const core::Vec3Ned& d1,
    core::Vec3Ned& out_pt_ned,
    double& out_parallax_deg) noexcept {
  const double cos_ang = std::clamp(d0.dot(d1), -1.0, 1.0);
  out_parallax_deg = std::acos(cos_ang) * kRadToDeg;
  const double denom = 1.0 - cos_ang * cos_ang;
  if (denom < 1e-5) {
    return false;
  }
  const core::Vec3Ned w0 = p0 - p1;
  const double b = cos_ang;
  const double d = d0.dot(w0);
  const double e = d1.dot(w0);
  const double s = (b * e - d) / denom;
  const double t = (e - b * d) / denom;
  if (s <= 3.0 || t <= 3.0 || s > 250.0 || t > 250.0) {
    return false;
  }
  const core::Vec3Ned q0 = p0 + d0 * s;
  const core::Vec3Ned q1 = p1 + d1 * t;
  out_pt_ned = (q0 + q1) * 0.5;
  return out_pt_ned.is_finite();
}

void init_covariance(std::array<double, 36>& P, double pos_var, double ang_var, double sag_var) noexcept {
  P.fill(0.0);
  P[0 * 6 + 0] = pos_var;
  P[1 * 6 + 1] = pos_var;
  P[2 * 6 + 2] = pos_var;
  P[3 * 6 + 3] = ang_var;
  P[4 * 6 + 4] = ang_var;
  P[5 * 6 + 5] = sag_var;
}

}  // namespace

WireSpanEkfBank::WireSpanEkfBank(const WireSpanEkfConfig& cfg) noexcept : cfg_(cfg) {}

void WireSpanEkfBank::clear() noexcept {
  for (auto& tr : tracks_) {
    tr = WireSpanTrack{};
  }
  next_id_ = 1u;
}

void WireSpanEkfBank::prune_stale(uint64_t now_ns) noexcept {
  const uint64_t timeout_ns = static_cast<uint64_t>(std::max(0.1, cfg_.track_timeout_s) * 1e9);
  for (auto& tr : tracks_) {
    if (!tr.active) {
      continue;
    }
    if (now_ns > tr.last_update_ns && (now_ns - tr.last_update_ns) > timeout_ns) {
      tr = WireSpanTrack{};
    }
  }
}

std::size_t WireSpanEkfBank::allocate_slot(uint64_t stamp_ns) noexcept {
  std::size_t oldest_idx = 0;
  uint64_t oldest_ns = UINT64_MAX;
  for (std::size_t i = 0; i < kMaxSpanTracks; ++i) {
    if (!tracks_[i].active) {
      return i;
    }
    if (tracks_[i].last_update_ns < oldest_ns) {
      oldest_ns = tracks_[i].last_update_ns;
      oldest_idx = i;
    }
  }
  (void)stamp_ns;
  return oldest_idx;
}

void WireSpanEkfBank::ingest_vision_plane(
    uint64_t stamp_ns,
    const core::Vec3Ned& cam_pos_ned,
    const core::Vec3Ned& ray_mid_unit_ned_in,
    const core::Vec3Ned& plane_normal_unit_ned_in,
    uint8_t voltage_class) noexcept {
  if (!cam_pos_ned.is_finite() || !ray_mid_unit_ned_in.is_finite() ||
      !plane_normal_unit_ned_in.is_finite()) {
    return;
  }
  const core::Vec3Ned ray_u = normalize_ned(ray_mid_unit_ned_in);
  const core::Vec3Ned n_plane = normalize_ned(plane_normal_unit_ned_in);

  // 1. Find best matching active wire track via 2-DOF interpretation-plane Mahalanobis gate (chi^2 <= 9.21).
  std::size_t best_idx = kMaxSpanTracks;
  double best_chi2 = cfg_.interp_plane_chi2_gate;

  for (std::size_t i = 0; i < kMaxSpanTracks; ++i) {
    const WireSpanTrack& tr = tracks_[i];
    if (!tr.active) {
      continue;
    }
    const core::Vec3Ned p_mid{tr.x[0], tr.x[1], tr.x[2]};
    const core::Vec3Ned rel = p_mid - cam_pos_ned;
    const double range = std::max(5.0, rel.norm());
    if (rel.dot(ray_u) <= 0.0) {
      continue;  // Behind camera
    }

    // Residual 1: angular distance of span midpoint from interpretation plane.
    const double r_pos = n_plane.dot(rel) / range;
    const double var_pos = (n_plane.x * n_plane.x * tr.P[0] +
                            n_plane.y * n_plane.y * tr.P[7] +
                            n_plane.z * n_plane.z * tr.P[14]) /
                               (range * range) +
                           cfg_.interp_plane_sigma_rad * cfg_.interp_plane_sigma_rad;

    // Residual 2: orthogonality of wire span direction with interpretation plane normal.
    const core::Vec3Ned dir = span_dir_unit(tr.x[3], tr.x[4]);
    const double r_dir = n_plane.dot(dir);
    const double var_dir = tr.P[21] + tr.P[28] + 0.04;

    const double chi2 = (r_pos * r_pos) / std::max(var_pos, 1e-6) +
                        (r_dir * r_dir) / std::max(var_dir, 1e-6);
    if (chi2 <= best_chi2) {
      best_chi2 = chi2;
      best_idx = i;
    }
  }

  if (best_idx < kMaxSpanTracks) {
    WireSpanTrack& tr = tracks_[best_idx];
    tr.last_update_ns = stamp_ns;
    tr.vision_hits += 1u;
    tr.voltage_class = voltage_class;

    // Scalar EKF projection update pulling span midpoint onto the observed interpretation plane.
    const core::Vec3Ned p_mid{tr.x[0], tr.x[1], tr.x[2]};
    const core::Vec3Ned rel = p_mid - cam_pos_ned;
    const double range = std::max(5.0, rel.norm());
    const double innov = -n_plane.dot(rel);  // Want n_plane . (p_mid - cam_pos) == 0
    const double h_p_ht = n_plane.x * n_plane.x * tr.P[0] +
                          n_plane.y * n_plane.y * tr.P[7] +
                          n_plane.z * n_plane.z * tr.P[14];
    const double r_meas = (cfg_.interp_plane_sigma_rad * range) * (cfg_.interp_plane_sigma_rad * range);
    const double s_inv = 1.0 / std::max(h_p_ht + r_meas, 1e-6);
    const double k0 = tr.P[0] * n_plane.x * s_inv;
    const double k1 = tr.P[7] * n_plane.y * s_inv;
    const double k2 = tr.P[14] * n_plane.z * s_inv;

    tr.x[0] += k0 * innov;
    tr.x[1] += k1 * innov;
    tr.x[2] += k2 * innov;
    tr.P[0] = std::max(0.04, (1.0 - k0 * n_plane.x) * tr.P[0]);
    tr.P[7] = std::max(0.04, (1.0 - k1 * n_plane.y) * tr.P[7]);
    tr.P[14] = std::max(0.04, (1.0 - k2 * n_plane.z) * tr.P[14]);

    // If radar is NOT locked (grazing >20 deg cliff), test >3 deg multi-frame SVD parallax triangulation.
    if (!tr.radar_locked) {
      if (!tr.has_anchor_obs) {
        tr.has_anchor_obs = true;
        tr.anchor_cam_pos_ned = cam_pos_ned;
        tr.anchor_ray_unit_ned = ray_u;
        tr.anchor_plane_n_ned = n_plane;
      } else {
        core::Vec3Ned tri_pt{};
        double parallax_deg = 0.0;
        const bool ok_tri = triangulate_two_rays_ned(
            tr.anchor_cam_pos_ned,
            tr.anchor_ray_unit_ned,
            cam_pos_ned,
            ray_u,
            tri_pt,
            parallax_deg);
        tr.max_parallax_deg = std::max(tr.max_parallax_deg, parallax_deg);
        if (ok_tri && parallax_deg >= cfg_.grazing_parallax_min_deg) {
          tr.x[0] = tri_pt.x;
          tr.x[1] = tri_pt.y;
          tr.x[2] = tri_pt.z;
          // Wire direction is orthogonal to both camera interpretation plane normals (n0 x n1).
          const core::Vec3Ned dir_est = cross_ned(tr.anchor_plane_n_ned, n_plane);
          if (dir_est.norm() > 1e-4) {
            const core::Vec3Ned du = normalize_ned(dir_est);
            tr.x[3] = std::atan2(du.y, du.x);
            tr.x[4] = std::asin(std::clamp(du.z, -0.95, 0.95));
          }
          init_covariance(tr.P, 4.0, 0.08, 0.25);
          tr.svd_triangulated = true;
          tr.standoff_fallback = false;
        } else if (tr.vision_hits >= cfg_.min_vision_hits_for_standoff && !tr.svd_triangulated) {
          // Conservative horizon standoff capsule before 3 deg parallax is accumulated.
          const core::Vec3Ned standoff_pt = cam_pos_ned + ray_u * cfg_.grazing_standoff_range_m;
          tr.x[0] = standoff_pt.x;
          tr.x[1] = standoff_pt.y;
          tr.x[2] = standoff_pt.z;
          tr.standoff_fallback = true;
        }
      }
    }
    return;
  }

  // 2. Spawn a new vision-only hypothesis track at conservative standoff distance.
  const std::size_t slot = allocate_slot(stamp_ns);
  WireSpanTrack& tr = tracks_[slot];
  tr = WireSpanTrack{};
  tr.id = next_id_++;
  tr.last_update_ns = stamp_ns;
  tr.voltage_class = voltage_class;
  const core::Vec3Ned init_pt = cam_pos_ned + ray_u * cfg_.grazing_standoff_range_m;
  const core::Vec3Ned tangent = normalize_ned(cross_ned(n_plane, ray_u));
  tr.x = {
      init_pt.x,
      init_pt.y,
      init_pt.z,
      std::atan2(tangent.y, tangent.x),
      std::asin(std::clamp(tangent.z, -0.95, 0.95)),
      0.6,  // Nominal mid-span catenary sag [m]
  };
  init_covariance(tr.P, 100.0, 0.25, 0.5);
  tr.has_anchor_obs = true;
  tr.anchor_cam_pos_ned = cam_pos_ned;
  tr.anchor_ray_unit_ned = ray_u;
  tr.anchor_plane_n_ned = n_plane;
  tr.vision_hits = 1u;
  tr.active = true;
}

void WireSpanEkfBank::ingest_radar_wire_point(
    uint64_t stamp_ns,
    const core::Vec3Ned& radar_pt_ned,
    const core::Vec3Ned& wire_tangent_hint_ned,
    double snr_db) noexcept {
  if (!radar_pt_ned.is_finite() || !std::isfinite(snr_db)) {
    return;
  }

  // Associate against closest active span (within 18 m of midpoint or standoff ray).
  std::size_t best_idx = kMaxSpanTracks;
  double best_dist = 18.0;

  for (std::size_t i = 0; i < kMaxSpanTracks; ++i) {
    const WireSpanTrack& tr = tracks_[i];
    if (!tr.active) {
      continue;
    }
    if (!tr.radar_locked && tr.has_anchor_obs) {
      // Distance from radar point to the vision bearing ray (allows locking vision-only track at true range!).
      const core::Vec3Ned rel = radar_pt_ned - tr.anchor_cam_pos_ned;
      const double proj = rel.dot(tr.anchor_ray_unit_ned);
      if (proj > 3.0) {
        const core::Vec3Ned perp = rel - tr.anchor_ray_unit_ned * proj;
        const double ray_dist = perp.norm();
        if (ray_dist < best_dist) {
          best_dist = ray_dist;
          best_idx = i;
        }
      }
    } else {
      const core::Vec3Ned p_mid{tr.x[0], tr.x[1], tr.x[2]};
      const double d = (radar_pt_ned - p_mid).norm();
      if (d < best_dist) {
        best_dist = d;
        best_idx = i;
      }
    }
  }

  const double r_var = std::max(0.08, cfg_.radar_range_sigma_m * cfg_.radar_range_sigma_m *
                                          std::pow(10.0, std::clamp((15.0 - snr_db) / 20.0, -0.5, 1.0)));

  if (best_idx < kMaxSpanTracks) {
    WireSpanTrack& tr = tracks_[best_idx];
    tr.last_update_ns = stamp_ns;
    tr.radar_hits += 1u;

    if (!tr.radar_locked) {
      // First radar range lock snaps standoff hypothesis directly to true 3D radar range.
      tr.x[0] = radar_pt_ned.x;
      tr.x[1] = radar_pt_ned.y;
      tr.x[2] = radar_pt_ned.z;
      tr.P[0] = r_var;
      tr.P[7] = r_var;
      tr.P[14] = r_var;
      tr.radar_locked = true;
      tr.standoff_fallback = false;
    } else {
      // 3D position EKF Kalman gain update.
      const double k0 = tr.P[0] / (tr.P[0] + r_var);
      const double k1 = tr.P[7] / (tr.P[7] + r_var);
      const double k2 = tr.P[14] / (tr.P[14] + r_var);
      tr.x[0] += k0 * (radar_pt_ned.x - tr.x[0]);
      tr.x[1] += k1 * (radar_pt_ned.y - tr.x[1]);
      tr.x[2] += k2 * (radar_pt_ned.z - tr.x[2]);
      tr.P[0] = std::max(0.02, (1.0 - k0) * tr.P[0]);
      tr.P[7] = std::max(0.02, (1.0 - k1) * tr.P[7]);
      tr.P[14] = std::max(0.02, (1.0 - k2) * tr.P[14]);
    }
    return;
  }

  // Spawn a radar-initialized wire span track.
  const std::size_t slot = allocate_slot(stamp_ns);
  WireSpanTrack& tr = tracks_[slot];
  tr = WireSpanTrack{};
  tr.id = next_id_++;
  tr.last_update_ns = stamp_ns;
  const core::Vec3Ned tangent = normalize_ned(wire_tangent_hint_ned);
  tr.x = {
      radar_pt_ned.x,
      radar_pt_ned.y,
      radar_pt_ned.z,
      std::atan2(tangent.y, tangent.x),
      std::asin(std::clamp(tangent.z, -0.95, 0.95)),
      0.5,
  };
  init_covariance(tr.P, r_var, 0.15, 0.35);
  tr.radar_hits = 1u;
  tr.radar_locked = true;
  tr.active = true;
}

void WireSpanEkfBank::emit_wire_capsules(
    const core::Vec3Ned& own_pos_ned,
    const core::Vec3Ned& own_vel_ned,
    rta::WireCapsuleTable& out_table,
    uint32_t& out_active_spans,
    bool& out_grazing_parallax_active,
    bool& out_standoff_emitted) const noexcept {
  out_table.clear();
  out_active_spans = 0u;
  out_grazing_parallax_active = false;
  out_standoff_emitted = false;

  struct RankedTrack {
    std::size_t idx{0};
    double priority_ttc_s{1e9};
  };
  std::array<RankedTrack, kMaxSpanTracks> ranked{};
  std::size_t n_ranked = 0;

  for (std::size_t i = 0; i < kMaxSpanTracks; ++i) {
    const WireSpanTrack& tr = tracks_[i];
    if (!tr.active) {
      continue;
    }
    // Emit tracks that have radar lock, SVD triangulation, or >= min_vision_hits standoff fallback.
    if (!tr.radar_locked && !tr.svd_triangulated && !tr.standoff_fallback) {
      continue;
    }
    const core::Vec3Ned p_mid{tr.x[0], tr.x[1], tr.x[2]};
    const core::Vec3Ned rel = p_mid - own_pos_ned;
    const double dist = std::max(1.0, rel.norm());
    const double v_close = std::max(0.5, own_vel_ned.dot(rel) / dist);
    ranked[n_ranked++] = {i, dist / v_close};
  }

  // Sort ascending by time-to-collision (closest/highest-threat spans get capsule slots first).
  std::sort(ranked.begin(), ranked.begin() + static_cast<std::ptrdiff_t>(n_ranked),
            [](const RankedTrack& a, const RankedTrack& b) noexcept {
              return a.priority_ttc_s < b.priority_ttc_s;
            });

  const uint32_t n_seg = std::clamp<uint32_t>(cfg_.max_segments_per_span, 1u, 4u);
  for (std::size_t r = 0; r < n_ranked; ++r) {
    const WireSpanTrack& tr = tracks_[ranked[r].idx];
    if (out_table.size() >= rta::WireCapsuleTable::kMaxCapsules) {
      break;
    }

    out_active_spans += 1u;
    if (tr.svd_triangulated && !tr.radar_locked) {
      out_grazing_parallax_active = true;
    }
    if (tr.standoff_fallback && !tr.radar_locked && !tr.svd_triangulated) {
      out_standoff_emitted = true;
    }

    const core::Vec3Ned p_mid{tr.x[0], tr.x[1], tr.x[2]};
    const core::Vec3Ned dir = span_dir_unit(tr.x[3], tr.x[4]);
    const double sag = std::clamp(tr.x[5], 0.0, 4.0);
    const double sigma_pos = std::sqrt(std::max({tr.P[0], tr.P[7], tr.P[14], 0.04}));

    // Canonical airframe radius R_veh = 2.10 m + wire clearance + bounded uncertainty margin.
    double cap_radius_m = kCanonicalVehicleRadiusM + cfg_.wire_clearance_radius_m +
                          std::min(2.0, 3.0 * sigma_pos);
    if (tr.standoff_fallback && !tr.radar_locked) {
      cap_radius_m = std::max(cap_radius_m,
                              kCanonicalVehicleRadiusM + cfg_.grazing_standoff_extra_radius_m);
    }

    // Piecewise parabolic catenary tessellation (up to 4 segments per span).
    for (uint32_t s = 0; s < n_seg; ++s) {
      if (out_table.size() >= rta::WireCapsuleTable::kMaxCapsules) {
        break;
      }
      const double u0 = -1.0 + 2.0 * static_cast<double>(s) / static_cast<double>(n_seg);
      const double u1 = -1.0 + 2.0 * static_cast<double>(s + 1u) / static_cast<double>(n_seg);
      // Catenary sag: lowest (+Z_NED) at midpoint u=0, higher (-Z_NED) at towers u=+-1.
      const core::Vec3Ned q0_ned{
          p_mid.x + dir.x * (u0 * tr.half_length_m),
          p_mid.y + dir.y * (u0 * tr.half_length_m),
          p_mid.z + dir.z * (u0 * tr.half_length_m) - sag * (u0 * u0),
      };
      const core::Vec3Ned q1_ned{
          p_mid.x + dir.x * (u1 * tr.half_length_m),
          p_mid.y + dir.y * (u1 * tr.half_length_m),
          p_mid.z + dir.z * (u1 * tr.half_length_m) - sag * (u1 * u1),
      };
      const uint32_t cap_id = tr.id * 10u + s;
      (void)out_table.add_capsule(
          core::ned_to_enu(q0_ned),
          core::ned_to_enu(q1_ned),
          cap_radius_m,
          cap_id);
    }
  }
}

std::size_t WireSpanEkfBank::active_track_count() const noexcept {
  std::size_t n = 0;
  for (const auto& tr : tracks_) {
    if (tr.active) {
      ++n;
    }
  }
  return n;
}

}  // namespace vertirta::forward_hazard
