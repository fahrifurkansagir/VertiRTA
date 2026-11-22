#include "vertirta/rta/wire_capsules.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::rta {

bool WireCapsuleTable::add_capsule(
    const core::Vec3Enu& p0_enu,
    const core::Vec3Enu& p1_enu,
    double radius_m,
    uint32_t id) noexcept {
  if (count_ >= kMaxCapsules || !std::isfinite(radius_m) || radius_m <= 0.0 ||
      !std::isfinite(p0_enu.x) || !std::isfinite(p0_enu.y) || !std::isfinite(p0_enu.z) ||
      !std::isfinite(p1_enu.x) || !std::isfinite(p1_enu.y) || !std::isfinite(p1_enu.z)) {
    return false;
  }
  WireCapsule cap{};
  cap.p0_enu[0] = static_cast<float>(p0_enu.x);
  cap.p0_enu[1] = static_cast<float>(p0_enu.y);
  cap.p0_enu[2] = static_cast<float>(p0_enu.z);
  cap.p1_enu[0] = static_cast<float>(p1_enu.x);
  cap.p1_enu[1] = static_cast<float>(p1_enu.y);
  cap.p1_enu[2] = static_cast<float>(p1_enu.z);
  cap.radius_m = static_cast<float>(radius_m);
  cap.id = id;
  capsules_[count_++] = cap;
  return true;
}

WireQueryResult WireCapsuleTable::eval_capsule(
    const WireCapsule& cap,
    const core::Vec3Enu& pos_enu) noexcept {
  const core::Vec3Enu p0{cap.p0_enu[0], cap.p0_enu[1], cap.p0_enu[2]};
  const core::Vec3Enu p1{cap.p1_enu[0], cap.p1_enu[1], cap.p1_enu[2]};
  const core::Vec3Enu d = p1 - p0;
  const double d2 = d.x * d.x + d.y * d.y + d.z * d.z;

  double t = 0.0;
  if (d2 > 1e-12) {
    const core::Vec3Enu pm0 = pos_enu - p0;
    t = std::clamp((pm0.x * d.x + pm0.y * d.y + pm0.z * d.z) / d2, 0.0, 1.0);
  }
  const core::Vec3Enu q = p0 + d * t;
  const core::Vec3Enu r = pos_enu - q;
  constexpr double kEps2 = 1e-8;  // epsilon = 1e-4 m (DR-2 Section 4.2)
  const double rho = std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z + kEps2);

  WireQueryResult out{};
  out.signed_dist_m = rho - static_cast<double>(cap.radius_m);
  out.closest_pt_enu = q;
  out.grad_unit = {r.x / rho, r.y / rho, r.z / rho};
  out.id = cap.id;
  out.valid = true;
  return out;
}

bool WireCapsuleTable::query_nearest_single(
    const core::Vec3Enu& pos_enu,
    WireQueryResult& out) const noexcept {
  if (count_ == 0u) {
    return false;
  }
  bool found = false;
  double best = 1e30;
  for (std::size_t i = 0; i < count_; ++i) {
    const WireQueryResult q = eval_capsule(capsules_[i], pos_enu);
    if (q.signed_dist_m < best) {
      best = q.signed_dist_m;
      out = q;
      found = true;
    }
  }
  return found;
}

std::size_t WireCapsuleTable::extract_nearest_hazards(
    const core::Vec3Enu& pos_enu,
    std::span<HazardPoint> out) const noexcept {
  const std::size_t k = std::min<std::size_t>({out.size(), count_, 4u});
  if (k == 0u) {
    return 0u;
  }

  std::array<WireQueryResult, 4> top{};
  std::array<double, 4> top_rad{};
  std::size_t n = 0;
  for (std::size_t i = 0; i < count_; ++i) {
    const WireQueryResult q = eval_capsule(capsules_[i], pos_enu);
    if (n < k) {
      top[n] = q;
      top_rad[n] = static_cast<double>(capsules_[i].radius_m);
      ++n;
    } else {
      std::size_t worst = 0;
      for (std::size_t j = 1; j < n; ++j) {
        if (top[j].signed_dist_m > top[worst].signed_dist_m) {
          worst = j;
        }
      }
      if (q.signed_dist_m < top[worst].signed_dist_m) {
        top[worst] = q;
        top_rad[worst] = static_cast<double>(capsules_[i].radius_m);
      }
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    out[i] = HazardPoint{
        .pos_ned = core::enu_to_ned(top[i].closest_pt_enu),
        .vel_ned = {},
        .accel_ned = {},
        .radius_h_m = std::max(0.0, top_rad[i] - 4.5),
        .radius_z_m = std::max(0.0, top_rad[i] - 3.0),
        .gamma = 1.5,
        .kind = 1u,
    };
  }
  return n;
}

}  // namespace vertirta::rta
