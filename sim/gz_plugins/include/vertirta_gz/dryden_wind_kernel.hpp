#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "vertirta_gz/coaxial_physics_kernel.hpp"

namespace vertirta_gz {

struct DrydenScales {
  double L_u{152.5};      // [m]
  double L_v{152.5};      // [m]
  double L_w{30.0};       // [m]
  double sigma_u{2.65};   // [m/s]
  double sigma_v{2.65};   // [m/s]
  double sigma_w{1.543};  // [m/s]
};

// Pure C++ MIL-F-8785C low-altitude Tustin discrete Dryden turbulence kernel (DR-2 Section 1.5)
// with deterministic xoshiro256** + Box-Muller RNG and open-water coastal drag helper.
// Note (DR-4 Erratum #3): Never applies inverted urban shear U(z)=W20*(z/6)^0.30.
class DrydenWindKernel {
 public:
  explicit DrydenWindKernel(uint64_t seed = 1ULL) noexcept { reseed(seed); }

  void reseed(uint64_t seed) noexcept {
    // SplitMix64 seeding of xoshiro256**
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
      z += 0x9E3779B97F4A7C15ULL;
      uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      rng_state_[i] = x ^ (x >> 31);
    }
    has_spare_gauss_ = false;
    spare_gauss_ = 0.0;
    reset_filter_state();
  }

  void reset_filter_state() noexcept {
    u_prev_ = 0.0;
    eta_u_prev_ = 0.0;
    v_prev1_ = v_prev2_ = 0.0;
    eta_v_prev1_ = eta_v_prev2_ = 0.0;
    w_prev1_ = w_prev2_ = 0.0;
    eta_w_prev1_ = eta_w_prev2_ = 0.0;
  }

  // Exact MIL-F-8785C low-altitude (<1000 ft) scale lengths and intensities (DR-2 Table 4)
  [[nodiscard]] static DrydenScales compute_scales(double h_agl_m, double w20_mps) noexcept {
    const double h_m = std::clamp(std::isfinite(h_agl_m) ? h_agl_m : 30.0, 3.0, 304.8);
    const double w20 = std::max(0.0, std::isfinite(w20_mps) ? w20_mps : 0.0);
    const double h_ft = h_m * 3.28084;
    const double denom = 0.177 + 0.000823 * h_ft;

    DrydenScales s{};
    s.sigma_w = 0.1 * w20;
    s.sigma_u = s.sigma_w / std::pow(denom, 0.4);
    s.sigma_v = s.sigma_u;
    s.L_w = h_m;
    s.L_u = (h_ft / std::pow(denom, 1.2)) * 0.3048;
    s.L_v = s.L_u;
    return s;
  }

  // Advance 1st-order H_u(z) and 2nd-order H_v(z), H_w(z) Tustin difference equations
  // given unit-variance white Gaussian draws eta[0..2] ~ N(0, 1).
  [[nodiscard]] Vec3D step(
      double dt_s,
      double h_agl_m,
      double v_air_mps,
      double w20_mps,
      const std::array<double, 3>& eta) noexcept {
    if (!std::isfinite(dt_s) || dt_s <= 1e-6) {
      return {u_prev_, v_prev1_, w_prev1_};
    }
    const double eta_u = std::isfinite(eta[0]) ? eta[0] : 0.0;
    const double eta_v = std::isfinite(eta[1]) ? eta[1] : 0.0;
    const double eta_w = std::isfinite(eta[2]) ? eta[2] : 0.0;

    const DrydenScales sc = compute_scales(h_agl_m, w20_mps);
    const double v_eff = std::max(std::isfinite(v_air_mps) ? v_air_mps : 3.0, 3.0);
    const double noise_scale = std::sqrt(kPi / dt_s);
    constexpr double kSqrt3 = 1.7320508075688772935;

    // Longitudinal H_u(z) — 1st-order Tustin
    const double tau_u = sc.L_u / v_eff;
    const double a_u = (2.0 * tau_u) / dt_s;
    const double k_u = sc.sigma_u * std::sqrt((2.0 * sc.L_u) / (kPi * v_eff));
    const double u_k =
        -((1.0 - a_u) / (1.0 + a_u)) * u_prev_ +
        (k_u / (1.0 + a_u)) * noise_scale * (eta_u + eta_u_prev_);
    u_prev_ = u_k;
    eta_u_prev_ = eta_u;

    // Lateral H_v(z) — exact 2nd-order Tustin
    const double tau_v = sc.L_v / v_eff;
    const double a_v = (2.0 * tau_v) / dt_s;
    const double k_v = sc.sigma_v * std::sqrt(sc.L_v / (kPi * v_eff));
    const double d0_v = (1.0 + a_v) * (1.0 + a_v);
    const double v_k =
        -(2.0 * (1.0 - a_v * a_v) / d0_v) * v_prev1_ -
        (((1.0 - a_v) * (1.0 - a_v)) / d0_v) * v_prev2_ +
        (k_v / d0_v) * noise_scale *
            ((1.0 + kSqrt3 * a_v) * eta_v + 2.0 * eta_v_prev1_ + (1.0 - kSqrt3 * a_v) * eta_v_prev2_);
    v_prev2_ = v_prev1_;
    v_prev1_ = v_k;
    eta_v_prev2_ = eta_v_prev1_;
    eta_v_prev1_ = eta_v;

    // Vertical H_w(z) — exact 2nd-order Tustin
    const double tau_w = sc.L_w / v_eff;
    const double a_w = (2.0 * tau_w) / dt_s;
    const double k_w = sc.sigma_w * std::sqrt(sc.L_w / (kPi * v_eff));
    const double d0_w = (1.0 + a_w) * (1.0 + a_w);
    const double w_k =
        -(2.0 * (1.0 - a_w * a_w) / d0_w) * w_prev1_ -
        (((1.0 - a_w) * (1.0 - a_w)) / d0_w) * w_prev2_ +
        (k_w / d0_w) * noise_scale *
            ((1.0 + kSqrt3 * a_w) * eta_w + 2.0 * eta_w_prev1_ + (1.0 - kSqrt3 * a_w) * eta_w_prev2_);
    w_prev2_ = w_prev1_;
    w_prev1_ = w_k;
    eta_w_prev2_ = eta_w_prev1_;
    eta_w_prev1_ = eta_w;

    return {u_k, v_k, w_k};
  }

  // Deterministic seeded turbulence sample in ENU [m/s]
  [[nodiscard]] Vec3D step_seeded(double dt_s, double h_agl_m, double v_air_mps, double w20_mps) noexcept {
    const std::array<double, 3> eta = {next_normal(), next_normal(), next_normal()};
    return step(dt_s, h_agl_m, v_air_mps, w20_mps, eta);
  }

  // Flat-plate quadratic aerodynamic gust/drag force on fuselage in world ENU [N]
  [[nodiscard]] static Vec3D compute_fuselage_drag_force_enu(
      Vec3D v_wind_minus_body_enu,
      double cd_a_xy_m2 = 0.48,
      double cd_a_z_m2 = 1.20,
      double rho_air = 1.225) noexcept {
    const double v_xy = std::hypot(v_wind_minus_body_enu.x, v_wind_minus_body_enu.y);
    const double fx = 0.5 * rho_air * cd_a_xy_m2 * v_xy * v_wind_minus_body_enu.x;
    const double fy = 0.5 * rho_air * cd_a_xy_m2 * v_xy * v_wind_minus_body_enu.y;
    const double fz = 0.5 * rho_air * cd_a_z_m2 * std::abs(v_wind_minus_body_enu.z) * v_wind_minus_body_enu.z;
    return {fx, fy, fz};
  }

 private:
  static uint64_t rotl(uint64_t x, int k) noexcept { return (x << k) | (x >> (64 - k)); }

  uint64_t next_u64() noexcept {
    const uint64_t result = rotl(rng_state_[1] * 5ULL, 7) * 9ULL;
    const uint64_t t = rng_state_[1] << 17;
    rng_state_[2] ^= rng_state_[0];
    rng_state_[3] ^= rng_state_[1];
    rng_state_[1] ^= rng_state_[2];
    rng_state_[0] ^= rng_state_[3];
    rng_state_[2] ^= t;
    rng_state_[3] = rotl(rng_state_[3], 45);
    return result;
  }

  double next_uniform01() noexcept {
    // 53-bit IEEE-754 uniform in (0, 1]
    return (static_cast<double>(next_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }

  double next_normal() noexcept {
    if (has_spare_gauss_) {
      has_spare_gauss_ = false;
      return spare_gauss_;
    }
    const double u1 = std::max(1e-15, next_uniform01());
    const double u2 = next_uniform01();
    const double mag = std::sqrt(-2.0 * std::log(u1));
    const double ang = 2.0 * kPi * u2;
    spare_gauss_ = mag * std::sin(ang);
    has_spare_gauss_ = true;
    return mag * std::cos(ang);
  }

  std::array<uint64_t, 4> rng_state_{};
  bool has_spare_gauss_{false};
  double spare_gauss_{0.0};

  double u_prev_{0.0};
  double eta_u_prev_{0.0};
  double v_prev1_{0.0}, v_prev2_{0.0};
  double eta_v_prev1_{0.0}, eta_v_prev2_{0.0};
  double w_prev1_{0.0}, w_prev2_{0.0};
  double eta_w_prev1_{0.0}, eta_w_prev2_{0.0};
};

}  // namespace vertirta_gz
