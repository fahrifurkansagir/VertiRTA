#pragma once

#include <array>
#include <cstdint>

#include "vertirta/core/frames.hpp"

namespace vertirta::rta {

struct DrydenScales {
  double L_u{152.5};
  double L_v{152.5};
  double L_w{30.0};
  double sigma_u{2.65};
  double sigma_v{2.65};
  double sigma_w{1.543};
};

// MIL-F-8785C low-altitude (<1000 ft) exact 1st-order H_u(z) & 2nd-order H_v(z), H_w(z)
// Tustin discrete turbulence filter scaled by sqrt(pi/dt) (DR-2 Section 1.5).
class DrydenTurbulenceFilter {
 public:
  explicit DrydenTurbulenceFilter(uint64_t seed = 20261122ULL) noexcept;

  void reseed(uint64_t seed) noexcept;
  void reset() noexcept;

  [[nodiscard]] static DrydenScales compute_scales(double h_agl_m, double w20_mps) noexcept;

  // Exact analytical Parseval variance ratio (Var_discrete / sigma^2) for [u, v, w].
  [[nodiscard]] static std::array<double, 3> theoretical_variance_ratios(
      double dt_s,
      double h_agl_m,
      double v_air_mps) noexcept;

  // Advance with caller-supplied unit-variance Gaussian draws eta[0..2] ~ N(0, 1).
  [[nodiscard]] core::Vec3Ned step(
      double dt_s,
      double h_agl_m,
      double v_air_mps,
      double w20_mps,
      const double eta[3]) noexcept;

  // Advance using internal deterministic xoshiro256** + Box-Muller RNG.
  [[nodiscard]] core::Vec3Ned step_seeded(
      double dt_s,
      double h_agl_m,
      double v_air_mps,
      double w20_mps) noexcept;

 private:
  uint64_t next_u64() noexcept;
  double next_normal() noexcept;

  std::array<uint64_t, 4> rng_state_{};
  bool has_spare_{false};
  double spare_{0.0};

  double u_prev_{0.0};
  double eta_u_prev_{0.0};
  double v_prev1_{0.0};
  double v_prev2_{0.0};
  double eta_v_prev1_{0.0};
  double eta_v_prev2_{0.0};
  double w_prev1_{0.0};
  double w_prev2_{0.0};
  double eta_w_prev1_{0.0};
  double eta_w_prev2_{0.0};
};

}  // namespace vertirta::rta
