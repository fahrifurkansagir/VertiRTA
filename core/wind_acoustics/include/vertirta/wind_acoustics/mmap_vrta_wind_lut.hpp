// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/wind_acoustics/vrta_wind_format.hpp"

namespace vertirta::wind_acoustics {

struct WindRomSample {
  core::Vec3Enu mean_wind_enu{};
  core::Vec3Enu turb_gust_enu{};
  core::Vec3Enu total_wind_enu{};
  core::Vec3Enu sigma_enu{};
  double sigma_rms_mps{0.0};
  double eps_rom_p95_mps{2.5};
  double l_turb_m{30.0};
  uint8_t tier_used{0};
  bool in_bounds{false};
  bool valid{false};
};

struct CbvfWindBuffer {
  double delta_radius_h_m{0.0};
  double delta_radius_z_m{0.0};
  double effective_disturbance_h_mps{0.0};
  double effective_disturbance_z_mps{0.0};
};

// Zero-copy Apache-2.0 POSIX mmap() reader for pure numeric VRTAWND1 POD-RBF wind tables.
// Verifies FIPS 180-4 SHA-256 and LTFJ longitude lock (< 29.1895 E) on load(); hot-path
// `update_inflow` and `query_enu` are strictly noexcept and zero-heap (< 5 us target).
class MmapVrtaWindLut {
 public:
  static constexpr double kConservativeFallbackEpsMps = 2.5;
  static constexpr double kConservativeFallbackSigmaMps = 1.5;

  MmapVrtaWindLut() noexcept = default;
  ~MmapVrtaWindLut() noexcept;

  MmapVrtaWindLut(const MmapVrtaWindLut&) = delete;
  MmapVrtaWindLut& operator=(const MmapVrtaWindLut&) = delete;
  MmapVrtaWindLut(MmapVrtaWindLut&& other) noexcept;
  MmapVrtaWindLut& operator=(MmapVrtaWindLut&& other) noexcept;

  [[nodiscard]] bool load(std::string_view path, bool verify_sha256 = true) noexcept;
  void unload() noexcept;

  [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }
  [[nodiscard]] const VrtaWndHeader& header() const noexcept { return header_; }
  [[nodiscard]] const std::array<double, kVrtaWndNumModes>& modal_amplitudes() const noexcept {
    return modal_amp_;
  }

  // Evaluate 48-center multiquadric RBF + affine polynomial at meteorological inflow (dir, u_ref).
  void update_inflow(double wind_dir_rad, double u_ref_mps) noexcept;

  // Hierarchical Tier-1 -> Tier-0 4x4x4 Morton-brick trilinear lookup + 8-mode FMA.
  // Optional deterministic OU gust step when `dt_s > 0.0 && ou_state != nullptr && rng_state != nullptr`.
  [[nodiscard]] bool query_enu(
      const core::Vec3Enu& pos_enu,
      WindRomSample& out,
      double dt_s = 0.0,
      core::Vec3Enu* ou_state = nullptr,
      uint64_t* rng_state = nullptr) const noexcept;

 private:
  [[nodiscard]] bool eval_grid_trilinear(
      const VrtaWndGridDesc& grid,
      const core::Vec3Enu& pos_enu,
      WindRomSample& out,
      uint8_t tier_id) const noexcept;

  void* mapped_{nullptr};
  std::size_t mapped_size_{0};
  int fd_{-1};
  bool loaded_{false};
  VrtaWndHeader header_{};
  const VrtaWndNode64* nodes_{nullptr};
  std::array<double, kVrtaWndNumModes> modal_amp_{};
  double active_uref_mps_{10.0};
};

// Conservatively map ROM residual `eps_rom_p95` + turbulence `sigma` into CBVF hazard radii.
[[nodiscard]] CbvfWindBuffer compute_cbvf_wind_inflation(
    const WindRomSample& sample,
    double v_horiz_mps,
    double a_xy_max_mps2,
    double horizon_s = 2.0,
    double sigma_multiplier = 2.0) noexcept;

// Zero-heap copy + monotonic inflation of HazardPoint span for frozen CbvfSafetyFilter::step_100hz.
// Returns number of hazards written into `out_hazards` (capped to `out_hazards.size()`).
[[nodiscard]] std::size_t inflate_hazards_with_wind_rom(
    std::span<const rta::HazardPoint> hazards_in,
    const WindRomSample& sample,
    double v_horiz_mps,
    double a_xy_max_mps2,
    std::span<rta::HazardPoint> out_hazards,
    double horizon_s = 2.0) noexcept;

}  // namespace vertirta::wind_acoustics
