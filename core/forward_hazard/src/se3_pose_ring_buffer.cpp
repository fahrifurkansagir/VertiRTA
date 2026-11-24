#include "vertirta/forward_hazard/se3_pose_ring_buffer.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::forward_hazard {

void Se3PoseRingBuffer::clear() noexcept {
  head_ = 0;
  count_ = 0;
}

core::QuatPx4FrdNed Se3PoseRingBuffer::normalize_quat(const core::QuatPx4FrdNed& q) noexcept {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (!std::isfinite(n) || n < 1e-12) {
    return {1.0, 0.0, 0.0, 0.0};
  }
  const double inv_n = 1.0 / n;
  return {q.w * inv_n, q.x * inv_n, q.y * inv_n, q.z * inv_n};
}

core::QuatPx4FrdNed Se3PoseRingBuffer::slerp_quat(
    const core::QuatPx4FrdNed& q0_in,
    const core::QuatPx4FrdNed& q1_in,
    double alpha) noexcept {
  const core::QuatPx4FrdNed q0 = normalize_quat(q0_in);
  core::QuatPx4FrdNed q1 = normalize_quat(q1_in);
  const double t = std::clamp(alpha, 0.0, 1.0);

  double dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z;
  if (dot < 0.0) {
    dot = -dot;
    q1 = {-q1.w, -q1.x, -q1.y, -q1.z};
  }

  if (dot > 0.999995) {
    return normalize_quat({
        q0.w + t * (q1.w - q0.w),
        q0.x + t * (q1.x - q0.x),
        q0.y + t * (q1.y - q0.y),
        q0.z + t * (q1.z - q0.z),
    });
  }

  const double theta = std::acos(std::clamp(dot, -1.0, 1.0));
  const double sin_theta = std::sin(theta);
  const double w0 = std::sin((1.0 - t) * theta) / sin_theta;
  const double w1 = std::sin(t * theta) / sin_theta;
  return normalize_quat({
      w0 * q0.w + w1 * q1.w,
      w0 * q0.x + w1 * q1.x,
      w0 * q0.y + w1 * q1.y,
      w0 * q0.z + w1 * q1.z,
  });
}

core::Vec3Ned Se3PoseRingBuffer::rotate_frd_to_ned(
    const core::QuatPx4FrdNed& q_in,
    const core::Vec3Frd& v) noexcept {
  const core::QuatPx4FrdNed q = normalize_quat(q_in);
  // Active Hamilton rotation v' = q * (0, v) * q*
  const double tx = 2.0 * (q.y * v.z - q.z * v.y);
  const double ty = 2.0 * (q.z * v.x - q.x * v.z);
  const double tz = 2.0 * (q.x * v.y - q.y * v.x);
  return {
      v.x + q.w * tx + (q.y * tz - q.z * ty),
      v.y + q.w * ty + (q.z * tx - q.x * tz),
      v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

core::Vec3Frd Se3PoseRingBuffer::rotate_ned_to_frd(
    const core::QuatPx4FrdNed& q_in,
    const core::Vec3Ned& v) noexcept {
  const core::QuatPx4FrdNed q = normalize_quat(q_in);
  // Inverse rotation uses conjugate q* = (w, -x, -y, -z)
  const double qx = -q.x;
  const double qy = -q.y;
  const double qz = -q.z;
  const double tx = 2.0 * (qy * v.z - qz * v.y);
  const double ty = 2.0 * (qz * v.x - qx * v.z);
  const double tz = 2.0 * (qx * v.y - qy * v.x);
  return {
      v.x + q.w * tx + (qy * tz - qz * ty),
      v.y + q.w * ty + (qz * tx - qx * tz),
      v.z + q.w * tz + (qx * ty - qy * tx),
  };
}

const TimedPoseSe3& Se3PoseRingBuffer::at_logical(std::size_t idx) const noexcept {
  return buf_[(head_ + idx) % kCapacity];
}

bool Se3PoseRingBuffer::push(const TimedPoseSe3& sample) noexcept {
  if (!sample.valid || !sample.pos_ned.is_finite() || !sample.vel_ned.is_finite()) {
    return false;
  }
  if (count_ > 0) {
    const TimedPoseSe3& last = at_logical(count_ - 1u);
    if (sample.stamp_ns < last.stamp_ns) {
      return false;
    }
    if (sample.stamp_ns == last.stamp_ns) {
      TimedPoseSe3 normalized = sample;
      normalized.q_frd_to_ned = normalize_quat(sample.q_frd_to_ned);
      buf_[(head_ + count_ - 1u) % kCapacity] = normalized;
      return true;
    }
  }

  TimedPoseSe3 normalized = sample;
  normalized.q_frd_to_ned = normalize_quat(sample.q_frd_to_ned);
  if (count_ < kCapacity) {
    buf_[(head_ + count_) % kCapacity] = normalized;
    ++count_;
  } else {
    buf_[head_] = normalized;
    head_ = (head_ + 1u) % kCapacity;
  }
  return true;
}

bool Se3PoseRingBuffer::latest(TimedPoseSe3& out) const noexcept {
  if (count_ == 0) {
    return false;
  }
  out = at_logical(count_ - 1u);
  return true;
}

bool Se3PoseRingBuffer::query(
    uint64_t query_stamp_ns,
    TimedPoseSe3& out,
    uint64_t max_extrap_ns) const noexcept {
  out = TimedPoseSe3{};
  if (count_ == 0) {
    return false;
  }

  const TimedPoseSe3& oldest = at_logical(0u);
  const TimedPoseSe3& newest = at_logical(count_ - 1u);

  if (query_stamp_ns < oldest.stamp_ns) {
    if ((oldest.stamp_ns - query_stamp_ns) > max_extrap_ns) {
      return false;
    }
    const double dt_s = -1e-9 * static_cast<double>(oldest.stamp_ns - query_stamp_ns);
    out = oldest;
    out.stamp_ns = query_stamp_ns;
    out.pos_ned = oldest.pos_ned + oldest.vel_ned * dt_s;
    out.valid = true;
    return true;
  }

  if (query_stamp_ns >= newest.stamp_ns) {
    if ((query_stamp_ns - newest.stamp_ns) > max_extrap_ns) {
      return false;
    }
    const double dt_s = 1e-9 * static_cast<double>(query_stamp_ns - newest.stamp_ns);
    out = newest;
    out.stamp_ns = query_stamp_ns;
    out.pos_ned = newest.pos_ned + newest.vel_ned * dt_s;
    out.valid = true;
    return true;
  }

  // Binary search in strictly increasing logical timestamps [0, count_ - 1].
  std::size_t lo = 0;
  std::size_t hi = count_ - 1u;
  while (lo + 1u < hi) {
    const std::size_t mid = lo + (hi - lo) / 2u;
    if (at_logical(mid).stamp_ns <= query_stamp_ns) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const TimedPoseSe3& p0 = at_logical(lo);
  const TimedPoseSe3& p1 = at_logical(hi);
  const uint64_t span_ns = p1.stamp_ns - p0.stamp_ns;
  const double alpha = (span_ns > 0ULL)
                           ? static_cast<double>(query_stamp_ns - p0.stamp_ns) /
                                 static_cast<double>(span_ns)
                           : 0.0;

  out.stamp_ns = query_stamp_ns;
  out.pos_ned = p0.pos_ned + (p1.pos_ned - p0.pos_ned) * alpha;
  out.vel_ned = p0.vel_ned + (p1.vel_ned - p0.vel_ned) * alpha;
  out.q_frd_to_ned = slerp_quat(p0.q_frd_to_ned, p1.q_frd_to_ned, alpha);
  out.valid = true;
  return true;
}

}  // namespace vertirta::forward_hazard
