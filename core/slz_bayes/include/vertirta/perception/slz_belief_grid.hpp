// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/slz_prior_provider.hpp"

namespace vertirta::perception {

inline constexpr uint32_t kSlzGridDim = 160u;
inline constexpr uint32_t kSlzGridCells = kSlzGridDim * kSlzGridDim;
inline constexpr float kSlzGridResolutionM = 0.5f;

// Cache-line-aligned 64 B cell storing cell-local shifted 3D moments dp = p_W - c_ij
// (eliminates float32 catastrophic cancellation at Maltepe +/-3500 m ENU coordinates),
// static semantic log-odds, asymmetric dynamic hazard density, inverse-variance precision,
// and explicit OOD variance floor.
struct alignas(64) SlzCell {
  float s[3]{0.0f, 0.0f, 0.0f};            // sum(dp) in cell-local shifted frame [12 B]
  float M[6]{0.0f, 0.0f, 0.0f,             // upper-tri sum(dp * dp^T): xx,xy,xz,yy,yz,zz [24 B]
             0.0f, 0.0f, 0.0f};
  uint32_t N{0u};                          // accumulated 3D points [4 B]
  float L{0.0f};                           // semantic log-odds clipped to [-8, +8] [4 B]
  float h_dyn{0.0f};                       // dynamic hazard density [0, 1] [4 B]
  float lambda_prec{0.0f};                 // inverse-variance precision Lambda_k [4 B]
  float ood_ema{0.0f};                     // filtered OOD energy excess [4 B]
  float sigma2_floor{0.0f};                // explicit OOD variance floor [4 B]
  uint32_t last_ts_ms{0u};                 // last update timestamp (ms) [4 B]
};

static_assert(sizeof(SlzCell) == 64u, "SlzCell must be exactly 64 bytes (1 cache line)");
static_assert(alignof(SlzCell) == 64u, "SlzCell must be 64-byte aligned");
static_assert(std::is_trivially_copyable_v<SlzCell>, "SlzCell must be trivially copyable");
static_assert(sizeof(SlzCell) * kSlzGridCells == 1638400u,
              "160x160 SlzBeliefGrid must be exactly 1,638,400 B (1.5625 MiB)");

// Exact integer lattice disk cell count (CountCircularDiskCells(5.0f, 0.5f) == 317u).
[[nodiscard]] uint32_t CountCircularDiskCells(float r_m, float res_m) noexcept;

struct SlzPointObservation {
  core::Vec3Enu point_enu{};
  float p_safe_obs{0.5f};       // P(y in {0, 1} | u) in [1e-3, 1 - 1e-3]
  float p_dyn_obs{0.0f};        // P(y == 3 | u) dynamic person/vehicle probability [0, 1]
  float e_tilde_ood{-12.0f};    // boundary-compensated free energy E_tilde(u)
  float cos_incidence{1.0f};    // cos(alpha_inc) in (0, 1]
};

struct SlzGridConfig {
  float tau_static_s{30.0f};
  float tau_dyn_s{5.0f};
  float tau_var_s{30.0f};
  float tau_ood_decay_s{1.5f};
  float tau_ood{-8.0f};
  float ood_beta{0.5f};
  float ood_kappa{2.0f};
  float alpha_ood{0.05f};
  float ood_var_floor_min{0.12f};  // strictly >= 0.08 to prevent OOD precision collapse
  float sigma2_base_rgbd{0.02f};
  float lambda_min{1.0f};
  uint32_t min_points_plane_fit{12u};
  float theta_valid_deg{8.0f};
  float sigma_z_valid_m{0.15f};
  float step_valid_m{0.25f};
  float p_safe_valid{0.85f};
  float h_dyn_valid{0.15f};
};

// Robocentric 160x160 @ 0.5 m (80 m x 80 m, 1.5625 MiB static ring buffer) 2.5D Bayesian SLZ grid.
// Zero dynamic memory allocations in UpdateFrame and EvaluateDisk.
class SlzBeliefGrid {
 public:
  explicit SlzBeliefGrid(SlzGridConfig cfg = {}) noexcept;

  void Reset() noexcept;
  void SetOrigin(const core::Vec3Enu& origin_enu) noexcept;

  // Integrates one frame of 3D point observations with semantic, dynamic-hazard, and OOD updates.
  // viewpoint_novelty in [0.05, 1.0] discounts repeated static-pose frames.
  // extra_geo_var_floor >= 0 adds Mod B monocular altitude variance floor.
  void UpdateFrame(
      std::span<const SlzPointObservation> obs,
      const core::Vec3Enu& vehicle_enu,
      double timestamp_s,
      float viewpoint_novelty = 0.5f,
      float extra_geo_var_floor = 0.0f) noexcept;

  // Evaluates a circular rotor-clearance disk (default R = 5.0 m => 317 cells) around center_enu.
  [[nodiscard]] rta::SlzPrior EvaluateDisk(
      const core::Vec3Enu& center_enu,
      float r_m = 5.0f) const noexcept;

  [[nodiscard]] bool has_origin() const noexcept { return origin_initialized_; }
  [[nodiscard]] core::Vec3Enu origin_enu() const noexcept;
  [[nodiscard]] const SlzCell* cell_at_world(double x_enu, double y_enu) const noexcept;
  [[nodiscard]] const SlzGridConfig& config() const noexcept { return cfg_; }

 private:
  void ShiftOriginTo(int32_t new_origin_ix, int32_t new_origin_iy) noexcept;
  [[nodiscard]] bool WorldToGlobalIdx(double x_enu, double y_enu,
                                      int32_t& gx, int32_t& gy) const noexcept;
  [[nodiscard]] bool IsGlobalInWindow(int32_t gx, int32_t gy) const noexcept;
  [[nodiscard]] static uint32_t WrapSlot(int32_t g) noexcept;
  [[nodiscard]] std::size_t SlotIndex(int32_t gx, int32_t gy) const noexcept;

  struct LocalPlaneFit {
    bool valid{false};
    float theta_deg{90.0f};
    float sigma_z_m{1.0f};
    double mean_x{0.0};
    double mean_y{0.0};
    double mean_z{0.0};
    double nx{0.0};
    double ny{0.0};
    double nz{1.0};
    uint32_t n_pts{0u};
  };

  [[nodiscard]] LocalPlaneFit FitWindow3x3(int32_t gx, int32_t gy) const noexcept;

  SlzGridConfig cfg_{};
  bool origin_initialized_{false};
  int32_t origin_ix_{0};
  int32_t origin_iy_{0};
  double origin_z_m_{0.0};
  std::array<SlzCell, kSlzGridCells> cells_{};
};

}  // namespace vertirta::perception
