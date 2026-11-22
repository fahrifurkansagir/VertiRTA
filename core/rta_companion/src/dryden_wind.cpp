#include "vertirta/rta/dryden_wind.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::rta {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrt3 = 1.7320508075688772935;

inline uint64_t rotl64(uint64_t x, int k) noexcept {
  return (x << k) | (x >> (64 - k));
}

// Exact discrete Parseval variance of y[k] = -a1 y[k-1] - a2 y[k-2] + b0 x[k] + b1 x[k-1] + b2 x[k-2]
// driven by unit white noise x[k] ~ N(0, 1).
[[nodiscard]] double arma2_unit_variance(
    double a1, double a2, double b0, double b1, double b2) noexcept {
  const double h0 = b0;
  const double h1 = b1 - a1 * h0;
  const double h2 = b2 - a1 * h1 - a2 * h0;
  const double denom = (1.0 - a2) * ((1.0 + a2) * (1.0 + a2) - a1 * a1);
  if (std::abs(denom) < 1e-18) {
    return 0.0;
  }
  const double num = (1.0 + a2) * (b0 * h0 + b1 * h1 + b2 * h2 - a2 * b2 * h0) -
                     a1 * (1.0 - a2) * (b1 * h0 + b2 * h1);
  return num / denom;
}

}  // namespace

DrydenTurbulenceFilter::DrydenTurbulenceFilter(uint64_t seed) noexcept {
  reseed(seed);
}

void DrydenTurbulenceFilter::reseed(uint64_t seed) noexcept {
  uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
  for (int i = 0; i < 4; ++i) {
    z += 0x9E3779B97F4A7C15ULL;
    uint64_t x = z;
    x = (x ^ (x >> 30u)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27u)) * 0x94D049BB133111EBULL;
    rng_state_[static_cast<std::size_t>(i)] = x ^ (x >> 31u);
  }
  has_spare_ = false;
  spare_ = 0.0;
  reset();
}

void DrydenTurbulenceFilter::reset() noexcept {
  u_prev_ = 0.0;
  eta_u_prev_ = 0.0;
  v_prev1_ = 0.0;
  v_prev2_ = 0.0;
  eta_v_prev1_ = 0.0;
  eta_v_prev2_ = 0.0;
  w_prev1_ = 0.0;
  w_prev2_ = 0.0;
  eta_w_prev1_ = 0.0;
  eta_w_prev2_ = 0.0;
}

DrydenScales DrydenTurbulenceFilter::compute_scales(double h_agl_m, double w20_mps) noexcept {
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

std::array<double, 3> DrydenTurbulenceFilter::theoretical_variance_ratios(
    double dt_s,
    double h_agl_m,
    double v_air_mps) noexcept {
  const DrydenScales sc = compute_scales(h_agl_m, 10.0);  // Ratio is independent of W20
  const double v_eff = std::max(std::isfinite(v_air_mps) ? v_air_mps : 3.0, 3.0);
  const double noise_scale = std::sqrt(kPi / dt_s);

  // Longitudinal H_u(z)
  const double a_u = (2.0 * sc.L_u / v_eff) / dt_s;
  const double ku_n = (sc.sigma_u * std::sqrt((2.0 * sc.L_u) / (kPi * v_eff)) / (1.0 + a_u)) * noise_scale;
  const double a1_u = (1.0 - a_u) / (1.0 + a_u);
  const double var_u = arma2_unit_variance(a1_u, 0.0, ku_n, ku_n, 0.0);

  // Lateral H_v(z)
  const double a_v = (2.0 * sc.L_v / v_eff) / dt_s;
  const double d0_v = (1.0 + a_v) * (1.0 + a_v);
  const double kv_n = (sc.sigma_v * std::sqrt(sc.L_v / (kPi * v_eff)) / d0_v) * noise_scale;
  const double var_v = arma2_unit_variance(
      2.0 * (1.0 - a_v * a_v) / d0_v,
      ((1.0 - a_v) * (1.0 - a_v)) / d0_v,
      kv_n * (1.0 + kSqrt3 * a_v),
      kv_n * 2.0,
      kv_n * (1.0 - kSqrt3 * a_v));

  // Vertical H_w(z)
  const double a_w = (2.0 * sc.L_w / v_eff) / dt_s;
  const double d0_w = (1.0 + a_w) * (1.0 + a_w);
  const double kw_n = (sc.sigma_w * std::sqrt(sc.L_w / (kPi * v_eff)) / d0_w) * noise_scale;
  const double var_w = arma2_unit_variance(
      2.0 * (1.0 - a_w * a_w) / d0_w,
      ((1.0 - a_w) * (1.0 - a_w)) / d0_w,
      kw_n * (1.0 + kSqrt3 * a_w),
      kw_n * 2.0,
      kw_n * (1.0 - kSqrt3 * a_w));

  return {
      var_u / (sc.sigma_u * sc.sigma_u),
      var_v / (sc.sigma_v * sc.sigma_v),
      var_w / (sc.sigma_w * sc.sigma_w),
  };
}

core::Vec3Ned DrydenTurbulenceFilter::step(
    double dt_s,
    double h_agl_m,
    double v_air_mps,
    double w20_mps,
    const double eta[3]) noexcept {
  if (!std::isfinite(dt_s) || dt_s <= 1e-6 || eta == nullptr) {
    return {u_prev_, v_prev1_, w_prev1_};
  }
  const double eta_u = std::isfinite(eta[0]) ? eta[0] : 0.0;
  const double eta_v = std::isfinite(eta[1]) ? eta[1] : 0.0;
  const double eta_w = std::isfinite(eta[2]) ? eta[2] : 0.0;

  const DrydenScales sc = compute_scales(h_agl_m, w20_mps);
  const double v_eff = std::max(std::isfinite(v_air_mps) ? v_air_mps : 3.0, 3.0);
  const double noise_scale = std::sqrt(kPi / dt_s);

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

core::Vec3Ned DrydenTurbulenceFilter::step_seeded(
    double dt_s,
    double h_agl_m,
    double v_air_mps,
    double w20_mps) noexcept {
  const double eta[3] = {next_normal(), next_normal(), next_normal()};
  return step(dt_s, h_agl_m, v_air_mps, w20_mps, eta);
}

uint64_t DrydenTurbulenceFilter::next_u64() noexcept {
  const uint64_t result = rotl64(rng_state_[1] * 5ULL, 7) * 9ULL;
  const uint64_t t = rng_state_[1] << 17u;
  rng_state_[2] ^= rng_state_[0];
  rng_state_[3] ^= rng_state_[1];
  rng_state_[1] ^= rng_state_[2];
  rng_state_[0] ^= rng_state_[3];
  rng_state_[2] ^= t;
  rng_state_[3] = rotl64(rng_state_[3], 45);
  return result;
}

double DrydenTurbulenceFilter::next_normal() noexcept {
  if (has_spare_) {
    has_spare_ = false;
    return spare_;
  }
  const double u1 = std::max(1e-15, (static_cast<double>(next_u64() >> 11u) + 0.5) * (1.0 / 9007199254740992.0));
  const double u2 = (static_cast<double>(next_u64() >> 11u) + 0.5) * (1.0 / 9007199254740992.0);
  const double mag = std::sqrt(-2.0 * std::log(u1));
  const double ang = 2.0 * kPi * u2;
  spare_ = mag * std::sin(ang);
  has_spare_ = true;
  return mag * std::cos(ang);
}

}  // namespace vertirta::rta
