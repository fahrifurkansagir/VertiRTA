#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"

namespace vertirta::rta {

// Analytical 3D thin-wire capsule (48 B each -> 64 x 48 B = 3.00 KiB static table).
struct WireCapsule {
  float p0_enu[3]{};
  float p1_enu[3]{};
  float radius_m{3.5f};  // R_wire + R_veh + 3*sqrt(sigma_prior^2 + sigma_range^2) + sigma_sag
  uint32_t id{0};
  uint32_t flags{0};
  float reserved{0.0f};
};
static_assert(sizeof(WireCapsule) <= 48, "WireCapsule exceeds 48 B budget");

struct WireQueryResult {
  double signed_dist_m{1e6};
  core::Vec3Enu closest_pt_enu{};
  core::Vec3Enu grad_unit{};
  uint32_t id{0};
  bool valid{false};
};

// Decoupled O(1)-per-segment analytic thin-wire table (up to 64 capsules, zero heap).
class WireCapsuleTable {
 public:
  static constexpr std::size_t kMaxCapsules = 64;

  WireCapsuleTable() noexcept = default;
  void clear() noexcept { count_ = 0; }

  [[nodiscard]] bool add_capsule(
      const core::Vec3Enu& p0_enu,
      const core::Vec3Enu& p1_enu,
      double radius_m,
      uint32_t id) noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return count_; }

  // Exact O(1) distance & smooth C^1 gradient (epsilon = 1e-4 m) for a single capsule.
  [[nodiscard]] static WireQueryResult eval_capsule(
      const WireCapsule& cap,
      const core::Vec3Enu& pos_enu) noexcept;

  [[nodiscard]] bool query_nearest_single(
      const core::Vec3Enu& pos_enu,
      WireQueryResult& out) const noexcept;

  [[nodiscard]] std::size_t extract_nearest_hazards(
      const core::Vec3Enu& pos_enu,
      std::span<HazardPoint> out) const noexcept;

 private:
  std::array<WireCapsule, kMaxCapsules> capsules_{};
  std::size_t count_{0};
};

}  // namespace vertirta::rta
