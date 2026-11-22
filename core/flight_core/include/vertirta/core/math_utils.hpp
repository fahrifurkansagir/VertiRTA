#pragma once

#include <cmath>
#include <cstdint>

namespace vertirta::core {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 6.28318530717958647692;
inline constexpr double kDegToRad = kPi / 180.0;
inline constexpr double kRadToDeg = 180.0 / kPi;
inline constexpr double kInvSqrt2 = 0.70710678118654752440;

[[nodiscard]] inline bool is_finite_f(float v) noexcept {
  return std::isfinite(v);
}

[[nodiscard]] inline bool is_finite_d(double v) noexcept {
  return std::isfinite(v);
}

[[nodiscard]] inline constexpr float clamp_f_raw(float v, float lo, float hi) noexcept {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

[[nodiscard]] inline constexpr double clamp_d_raw(double v, double lo, double hi) noexcept {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Freestanding NaN-safe clamps (zero <algorithm> dependency for bare-metal NuttX/STM32H7).
[[nodiscard]] inline float clamp_f(float v, float lo, float hi) noexcept {
  if (!is_finite_f(v)) {
    return lo;
  }
  return clamp_f_raw(v, lo, hi);
}

[[nodiscard]] inline double clamp_d(double v, double lo, double hi) noexcept {
  if (!is_finite_d(v)) {
    return lo;
  }
  return clamp_d_raw(v, lo, hi);
}

[[nodiscard]] inline constexpr float median3_f(float a, float b, float c) noexcept {
  if (a > b) {
    const float t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    b = c;
    if (a > b) {
      b = a;
    }
  }
  return b;
}

[[nodiscard]] inline constexpr double median3_d(double a, double b, double c) noexcept {
  if (a > b) {
    const double t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    b = c;
    if (a > b) {
      b = a;
    }
  }
  return b;
}

[[nodiscard]] inline double wrap_pi_d(double rad) noexcept {
  if (!is_finite_d(rad)) {
    return 0.0;
  }
  double r = std::fmod(rad + kPi, kTwoPi);
  if (r < 0.0) {
    r += kTwoPi;
  }
  return r - kPi;
}

}  // namespace vertirta::core
