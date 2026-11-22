#include "vertirta/rta/intruder_kf.hpp"

#include <algorithm>
#include <cmath>

#include "vertirta/core/geodesy.hpp"

namespace vertirta::rta {

namespace {

constexpr double kLtfjArpLat = 40.898333;
constexpr double kLtfjArpLon = 29.309167;
constexpr double kLtfjRadiusM = 9260.0;
constexpr double kLtfjCeilAmslM = 762.0;

// LTFJ NW polygon extension vertices adjoining Samandira CTR (AIP Turkey AD 2 LTFJ / DR-4/5):
// Pulls CTR western edge 603 m west of 29.200 E to J2 (40.912786 N, 29.192122 E).
constexpr double kLtfjJ1Lat = 40.987500;
constexpr double kLtfjJ1Lon = 29.325000;
constexpr double kLtfjJ2Lat = 40.912786;
constexpr double kLtfjJ2Lon = 29.192122;
constexpr double kLtfjWestJoinLat = 40.898333;
constexpr double kLtfjWestJoinLon = 29.199133;  // Western tangent point on 5 NM arc

constexpr double kLtbxArpLat = 40.990556;
constexpr double kLtbxArpLon = 29.216111;
constexpr double kLtbxRadiusM = 9260.0;
constexpr double kLtbxCeilAmslM = 609.6;

struct Pt2 {
  double x{0.0};
  double y{0.0};
};

[[nodiscard]] Pt2 ll_to_local_xy(double lat_deg, double lon_deg) noexcept {
  const core::Vec3Enu e = core::geodetic_to_local_enu(
      core::GeodeticWgs84{lat_deg, lon_deg, core::amsl_to_hae(12.0, lat_deg, lon_deg)});
  return {e.x, e.y};
}

// Exact signed distance to a CCW convex polygon in 2D (negative inside, positive outside).
[[nodiscard]] double signed_dist_convex_poly(const Pt2& p, const Pt2* poly, std::size_t n) noexcept {
  double min_dist2 = 1e30;
  bool inside = true;
  for (std::size_t i = 0; i < n; ++i) {
    const Pt2& a = poly[i];
    const Pt2& b = poly[(i + 1u) % n];
    const double ex = b.x - a.x;
    const double ey = b.y - a.y;
    const double px = p.x - a.x;
    const double py = p.y - a.y;
    // CCW edge cross product: < 0 means outside
    if (ex * py - ey * px < 0.0) {
      inside = false;
    }
    const double e2 = ex * ex + ey * ey;
    const double t = (e2 > 1e-12) ? std::clamp((px * ex + py * ey) / e2, 0.0, 1.0) : 0.0;
    const double dx = px - t * ex;
    const double dy = py - t * ey;
    min_dist2 = std::min(min_dist2, dx * dx + dy * dy);
  }
  const double d = std::sqrt(min_dist2);
  return inside ? -d : +d;
}

}  // namespace

StalenessPolicy compute_staleness_policy(double age_s) noexcept {
  StalenessPolicy pol{};
  if (!std::isfinite(age_s) || age_s <= 0.25) {
    return pol;
  }
  constexpr double kVUnc = 15.0;  // [m/s] worst-case unmonitored intruder speed
  constexpr double kAUnc = 3.0;   // [m/s^2] worst-case unmonitored intruder accel
  const double dt_eff = std::min(age_s, 1.5) - 0.25;
  pol.delta_r_stale_m = kVUnc * dt_eff + 0.5 * kAUnc * dt_eff * dt_eff;
  if (age_s > 1.0) {
    pol.v_cap_mps = 8.0;
  }
  if (age_s > 2.0) {
    pol.divert_to_slz = true;
  }
  return pol;
}

double AirspaceExclusion::signed_margin_ltfj_m(
    double lat_deg,
    double lon_deg,
    double alt_amsl_m) noexcept {
  const Pt2 p = ll_to_local_xy(lat_deg, lon_deg);
  const Pt2 arp = ll_to_local_xy(kLtfjArpLat, kLtfjArpLon);
  const double sd_arc = std::hypot(p.x - arp.x, p.y - arp.y) - kLtfjRadiusM;

  // CCW order: ARP -> J1 (N) -> J2 (NW pinch) -> WestJoin (W)
  const Pt2 nw_poly[4] = {
      arp,
      ll_to_local_xy(kLtfjJ1Lat, kLtfjJ1Lon),
      ll_to_local_xy(kLtfjJ2Lat, kLtfjJ2Lon),
      ll_to_local_xy(kLtfjWestJoinLat, kLtfjWestJoinLon),
  };
  const double sd_poly = signed_dist_convex_poly(p, nw_poly, 4u);
  const double sd_horiz = std::min(sd_arc, sd_poly);

  if (alt_amsl_m > kLtfjCeilAmslM) {
    return std::max(sd_horiz, alt_amsl_m - kLtfjCeilAmslM);
  }
  return sd_horiz;
}

double AirspaceExclusion::signed_margin_ltbx_m(
    double lat_deg,
    double lon_deg,
    double alt_amsl_m) noexcept {
  const Pt2 p = ll_to_local_xy(lat_deg, lon_deg);
  const Pt2 arp = ll_to_local_xy(kLtbxArpLat, kLtbxArpLon);
  const double sd_horiz = std::hypot(p.x - arp.x, p.y - arp.y) - kLtbxRadiusM;
  if (alt_amsl_m > kLtbxCeilAmslM) {
    return std::max(sd_horiz, alt_amsl_m - kLtbxCeilAmslM);
  }
  return sd_horiz;
}

std::size_t AirspaceExclusion::extract_ctr_hazards(
    double lat_deg,
    double lon_deg,
    double alt_amsl_m,
    const core::Vec3Ned& own_pos_ned,
    std::span<HazardPoint> out) noexcept {
  std::size_t n = 0;
  const Pt2 p = ll_to_local_xy(lat_deg, lon_deg);

  const double m_ltfj = signed_margin_ltfj_m(lat_deg, lon_deg, alt_amsl_m);
  if (n < out.size() && m_ltfj < 500.0) {
    const Pt2 arp = ll_to_local_xy(kLtfjArpLat, kLtfjArpLon);
    const double dx = arp.x - p.x;
    const double dy = arp.y - p.y;
    const double d = std::max(1.0, std::hypot(dx, dy));
    // Place virtual wall point along direction toward LTFJ CTR at distance max(m_ltfj, 0)
    const double wall_e = (dx / d) * std::max(m_ltfj, 0.5);
    const double wall_n = (dy / d) * std::max(m_ltfj, 0.5);
    out[n++] = HazardPoint{
        .pos_ned = {own_pos_ned.x + wall_n, own_pos_ned.y + wall_e, own_pos_ned.z},
        .vel_ned = {},
        .accel_ned = {},
        .radius_h_m = 0.0,
        .radius_z_m = 700.0,
        .gamma = 1.5,
        .kind = 3u,
    };
  }

  const double m_ltbx = signed_margin_ltbx_m(lat_deg, lon_deg, alt_amsl_m);
  if (n < out.size() && m_ltbx < 500.0) {
    const Pt2 arp = ll_to_local_xy(kLtbxArpLat, kLtbxArpLon);
    const double dx = arp.x - p.x;
    const double dy = arp.y - p.y;
    const double d = std::max(1.0, std::hypot(dx, dy));
    const double wall_e = (dx / d) * std::max(m_ltbx, 0.5);
    const double wall_n = (dy / d) * std::max(m_ltbx, 0.5);
    out[n++] = HazardPoint{
        .pos_ned = {own_pos_ned.x + wall_n, own_pos_ned.y + wall_e, own_pos_ned.z},
        .vel_ned = {},
        .accel_ned = {},
        .radius_h_m = 0.0,
        .radius_z_m = 600.0,
        .gamma = 1.5,
        .kind = 3u,
    };
  }
  return n;
}

void IntruderTrackerKf::clear() noexcept {
  for (auto& tr : tracks_) {
    tr = IntruderTrack{};
  }
}

void IntruderTrackerKf::update_measurement(
    uint32_t track_id,
    const core::Vec3Enu& pos_enu,
    const core::Vec3Enu& vel_enu,
    uint64_t meas_ns) noexcept {
  if (!std::isfinite(pos_enu.x) || !std::isfinite(pos_enu.y) || !std::isfinite(pos_enu.z) ||
      !std::isfinite(vel_enu.x) || !std::isfinite(vel_enu.y) || !std::isfinite(vel_enu.z)) {
    return;
  }

  // Locate existing track or allocate free/oldest slot
  std::size_t slot = kMaxTracks;
  std::size_t oldest_slot = 0;
  uint64_t oldest_ts = UINT64_MAX;
  for (std::size_t i = 0; i < kMaxTracks; ++i) {
    if (tracks_[i].active && tracks_[i].track_id == track_id) {
      slot = i;
      break;
    }
    if (!tracks_[i].active && slot == kMaxTracks) {
      slot = i;
    }
    if (tracks_[i].last_meas_ns < oldest_ts) {
      oldest_ts = tracks_[i].last_meas_ns;
      oldest_slot = i;
    }
  }
  if (slot == kMaxTracks) {
    slot = oldest_slot;
  }

  IntruderTrack& tr = tracks_[slot];
  if (!tr.active || tr.track_id != track_id || meas_ns <= tr.last_meas_ns) {
    tr = IntruderTrack{};
    tr.track_id = track_id;
    tr.last_meas_ns = meas_ns;
    tr.x_enu = {pos_enu.x, pos_enu.y, pos_enu.z, vel_enu.x, vel_enu.y, vel_enu.z};
    tr.P.fill(0.0);
    tr.P[0 * 6 + 0] = 4.0;
    tr.P[1 * 6 + 1] = 4.0;
    tr.P[2 * 6 + 2] = 2.0;
    tr.P[3 * 6 + 3] = 1.0;
    tr.P[4 * 6 + 4] = 1.0;
    tr.P[5 * 6 + 5] = 0.5;
    tr.active = true;
    return;
  }

  // 6-state CV-KF Predict + Measurement Update (decoupled 3x 2-state axes for exact speed & symmetry)
  const double dt = std::clamp(static_cast<double>(meas_ns - tr.last_meas_ns) * 1e-9, 0.001, 5.0);
  const double z_pos[3] = {pos_enu.x, pos_enu.y, pos_enu.z};
  const double z_vel[3] = {vel_enu.x, vel_enu.y, vel_enu.z};
  constexpr double kQa = 2.0;   // Process accel spectral density [m^2/s^4]
  constexpr double kRp = 2.25;  // ADS-B/Remote-ID position variance [m^2]
  constexpr double kRv = 0.25;  // Velocity variance [(m/s)^2]

  for (int a = 0; a < 3; ++a) {
    const std::size_t ip = static_cast<std::size_t>(a);
    const std::size_t iv = static_cast<std::size_t>(a + 3);

    // Predict
    const double xp = tr.x_enu[ip] + dt * tr.x_enu[iv];
    const double xv = tr.x_enu[iv];
    double p00 = tr.P[ip * 6 + ip] + 2.0 * dt * tr.P[ip * 6 + iv] + dt * dt * tr.P[iv * 6 + iv] +
                 0.25 * kQa * dt * dt * dt * dt;
    double p01 = tr.P[ip * 6 + iv] + dt * tr.P[iv * 6 + iv] + 0.5 * kQa * dt * dt * dt;
    double p11 = tr.P[iv * 6 + iv] + kQa * dt * dt;

    // Sequential scalar update on position
    const double sp = p00 + kRp;
    const double kp0 = p00 / sp;
    const double kp1 = p01 / sp;
    const double yp = z_pos[a] - xp;
    const double xp1 = xp + kp0 * yp;
    const double xv1 = xv + kp1 * yp;
    const double np00 = (1.0 - kp0) * p00;
    const double np01 = (1.0 - kp0) * p01;
    const double np11 = p11 - kp1 * p01;

    // Sequential scalar update on velocity
    const double sv = np11 + kRv;
    const double kv0 = np01 / sv;
    const double kv1 = np11 / sv;
    const double yv = z_vel[a] - xv1;
    tr.x_enu[ip] = xp1 + kv0 * yv;
    tr.x_enu[iv] = xv1 + kv1 * yv;
    tr.P[ip * 6 + ip] = std::max(0.05, np00 - kv0 * np01);
    tr.P[ip * 6 + iv] = np01 - kv0 * np11;
    tr.P[iv * 6 + ip] = tr.P[ip * 6 + iv];
    tr.P[iv * 6 + iv] = std::max(0.02, (1.0 - kv1) * np11);
  }
  tr.last_meas_ns = meas_ns;
}

std::size_t IntruderTrackerKf::extract_hazards(
    uint64_t now_ns,
    std::span<HazardPoint> out) const noexcept {
  std::size_t n = 0;
  for (std::size_t i = 0; i < kMaxTracks && n < out.size(); ++i) {
    const IntruderTrack& tr = tracks_[i];
    if (!tr.active) {
      continue;
    }
    const double dt = (now_ns >= tr.last_meas_ns)
                          ? (static_cast<double>(now_ns - tr.last_meas_ns) * 1e-9)
                          : 0.0;
    if (dt > 5.0) {
      continue;  // Expired ghost track
    }
    const StalenessPolicy stale = compute_staleness_policy(dt);

    const core::Vec3Enu p_pred{
        tr.x_enu[0] + dt * tr.x_enu[3],
        tr.x_enu[1] + dt * tr.x_enu[4],
        tr.x_enu[2] + dt * tr.x_enu[5],
    };
    const core::Vec3Enu v_pred{tr.x_enu[3], tr.x_enu[4], tr.x_enu[5]};

    // Propagate position variance P_pp(t_now) = P_pp + 2*dt*P_pv + dt^2*P_vv + Q
    constexpr double kQa = 2.0;
    const double var_e = tr.P[0] + 2.0 * dt * tr.P[3] + dt * dt * tr.P[21] + 0.25 * kQa * dt * dt * dt * dt;
    const double var_n = tr.P[7] + 2.0 * dt * tr.P[10] + dt * dt * tr.P[28] + 0.25 * kQa * dt * dt * dt * dt;
    const double var_u = tr.P[14] + 2.0 * dt * tr.P[17] + dt * dt * tr.P[35] + 0.25 * kQa * dt * dt * dt * dt;

    const double a_h = tr.r_sep_h_m + 3.0 * std::sqrt(std::max(var_e, var_n)) + stale.delta_r_stale_m;
    const double a_v = tr.r_sep_z_m + 3.0 * std::sqrt(std::max(0.0, var_u)) + 0.5 * stale.delta_r_stale_m;

    out[n++] = HazardPoint{
        .pos_ned = core::enu_to_ned(p_pred),
        .vel_ned = core::enu_to_ned(v_pred),
        .accel_ned = {},
        .radius_h_m = std::max(0.0, a_h - 4.5),
        .radius_z_m = std::max(0.0, a_v - 3.0),
        .gamma = 1.4,
        .kind = 2u,
    };
  }
  return n;
}

std::size_t IntruderTrackerKf::active_count() const noexcept {
  std::size_t c = 0;
  for (const auto& tr : tracks_) {
    if (tr.active) {
      ++c;
    }
  }
  return c;
}

}  // namespace vertirta::rta
