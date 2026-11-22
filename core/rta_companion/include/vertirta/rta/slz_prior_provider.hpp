#pragma once

#include <array>
#include <cstdint>

#include "vertirta/core/frames.hpp"

namespace vertirta::rta {

// Canonical SLZ rejection bitmask bits (shared with Plan 2 Co-Pilot & Plan 3 Active Visual SLZ).
namespace slz_reason {
inline constexpr uint32_t kCtrViolation = 1u << 0;
inline constexpr uint32_t kRoughnessExceeded = 1u << 1;  // sigma_z_m > 0.15 m
inline constexpr uint32_t kSlopeExceeded = 1u << 2;      // theta_max_deg > 8.0 deg
inline constexpr uint32_t kRadiusTooSmall = 1u << 3;     // clearance_r_m < 5.0 m
inline constexpr uint32_t kEnergyUnreachable = 1u << 4;  // E_req > 0.85 * E_rem
inline constexpr uint32_t kObstacleClearance = 1u << 5;  // ESDF / wire conflict
inline constexpr uint32_t kStepExceeded = 1u << 6;       // step_max_m > 0.25 m
}  // namespace slz_reason

// Canonical Plan 1A Task 9 / Plan 3 SLZ prior record (zero-heap, fixed-size).
struct SlzPrior {
  char id[32]{};
  double lat_deg{40.8860};
  double lon_deg{29.1880};
  double elev_amsl_m{12.0};
  core::Vec3Enu center_enu{};
  float p_safe{0.95f};
  float slz_ood_variance{0.01f};
  float theta_max_deg{1.5f};
  float sigma_z_m{0.05f};
  float step_max_m{0.05f};
  float clearance_r_m{15.0f};
  float h_dyn_max{0.0f};
  float eta_s{0.0f};
  float soc_margin_pct{0.0f};
  uint8_t rank{0};
  uint32_t reason_mask{0};
};

// Canonical ranked SLZ snapshot returned by ISlzPriorProvider (v1.0 geometric OR v2.0 active visual).
struct SlzProviderSnapshot {
  std::array<SlzPrior, 8> priors{};
  uint8_t count{0};
  uint64_t stamp_ns{0};
  bool stale{true};
  uint8_t active_decision_state{0};
  float gimbal_pitch_rad{0.0f};
  float gimbal_yaw_rad{0.0f};
};

// Canonical Plan 1A / Plan 3 polymorphic zero-heap SLZ provider interface.
class ISlzPriorProvider {
 public:
  virtual ~ISlzPriorProvider() = default;

  [[nodiscard]] virtual SlzProviderSnapshot query_priors(
      const core::Vec3Enu& vehicle_enu,
      float soc_pct,
      uint64_t now_ns) const noexcept = 0;
};

}  // namespace vertirta::rta
