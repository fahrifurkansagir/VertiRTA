#pragma once

#include <cmath>
#include <type_traits>

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

struct NedTag {};
struct EnuTag {};
struct FrdTag {};
struct FluTag {};

template <typename FrameTag>
struct Vec3 {
  double x{0.0};
  double y{0.0};
  double z{0.0};

  [[nodiscard]] constexpr Vec3 operator+(const Vec3& o) const noexcept {
    return {x + o.x, y + o.y, z + o.z};
  }
  [[nodiscard]] constexpr Vec3 operator-(const Vec3& o) const noexcept {
    return {x - o.x, y - o.y, z - o.z};
  }
  [[nodiscard]] constexpr Vec3 operator-() const noexcept {
    return {-x, -y, -z};
  }
  [[nodiscard]] constexpr Vec3 operator*(double s) const noexcept {
    return {x * s, y * s, z * s};
  }

  template <typename OtherTag, typename = std::enable_if_t<!std::is_same_v<FrameTag, OtherTag>>>
  Vec3 operator+(const Vec3<OtherTag>&) const = delete;

  template <typename OtherTag, typename = std::enable_if_t<!std::is_same_v<FrameTag, OtherTag>>>
  Vec3 operator-(const Vec3<OtherTag>&) const = delete;

  [[nodiscard]] constexpr double dot(const Vec3& o) const noexcept {
    return x * o.x + y * o.y + z * o.z;
  }
  [[nodiscard]] constexpr double norm_sq() const noexcept {
    return dot(*this);
  }
  [[nodiscard]] double norm() const noexcept {
    return std::sqrt(norm_sq());
  }
  [[nodiscard]] bool is_finite() const noexcept {
    return is_finite_d(x) && is_finite_d(y) && is_finite_d(z);
  }
};

using Vec3Ned = Vec3<NedTag>;
using Vec3Enu = Vec3<EnuTag>;
using Vec3Frd = Vec3<FrdTag>;
using Vec3Flu = Vec3<FluTag>;

// PX4 Hamilton scalar-first active rotation R_{FRD -> NED}
struct QuatPx4FrdNed {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

// ROS 2 REP-105 Hamilton scalar-last active rotation R_{FLU -> ENU}
struct QuatRosFluEnu {
  double x{0.0};
  double y{0.0};
  double z{0.0};
  double w{1.0};
};

[[nodiscard]] constexpr Vec3Ned enu_to_ned(Vec3Enu v) noexcept {
  return {v.y, v.x, -v.z};
}
[[nodiscard]] constexpr Vec3Enu ned_to_enu(Vec3Ned v) noexcept {
  return {v.y, v.x, -v.z};
}
[[nodiscard]] constexpr Vec3Frd flu_to_frd(Vec3Flu v) noexcept {
  return {v.x, -v.y, -v.z};
}
[[nodiscard]] constexpr Vec3Flu frd_to_flu(Vec3Frd v) noexcept {
  return {v.x, -v.y, -v.z};
}

// Closed-form sandwich product q_{NED->ENU} (x) q_PX4 (x) q_{FLU->FRD} (DR-1 Section 1.1)
[[nodiscard]] constexpr QuatRosFluEnu px4_to_ros_quat(QuatPx4FrdNed q) noexcept {
  return {
      kInvSqrt2 * (q.x + q.y),
      kInvSqrt2 * (q.x - q.y),
      kInvSqrt2 * (q.w - q.z),
      kInvSqrt2 * (q.w + q.z),
  };
}

[[nodiscard]] constexpr QuatPx4FrdNed ros_to_px4_quat(QuatRosFluEnu q) noexcept {
  return {
      kInvSqrt2 * (q.w + q.z),
      kInvSqrt2 * (q.x + q.y),
      kInvSqrt2 * (q.x - q.y),
      kInvSqrt2 * (q.w - q.z),
  };
}

}  // namespace vertirta::core
