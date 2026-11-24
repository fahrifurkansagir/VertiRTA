#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/forward_hazard/forward_hazard_provider.hpp"

namespace vertirta::forward_hazard {

// Fixed-capacity 200 Hz x 2.0 s = 400-sample zero-heap SE(3) pose ring buffer.
// Performs exact quaternion SLERP + linear NED position/velocity interpolation
// to time-align asynchronous 30 Hz forward camera and 15-20 Hz 77 GHz radar frames.
class Se3PoseRingBuffer {
 public:
  static constexpr std::size_t kCapacity = 400;  // 200 Hz * 2.0 s
  static constexpr uint64_t kDefaultMaxExtrapNs = 25'000'000ULL;  // 25 ms

  Se3PoseRingBuffer() noexcept = default;

  void clear() noexcept;

  // Push strictly monotonic pose sample (drops non-finite or out-of-order timestamps).
  bool push(const TimedPoseSe3& sample) noexcept;

  [[nodiscard]] std::size_t size() const noexcept { return count_; }
  [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
  [[nodiscard]] bool latest(TimedPoseSe3& out) const noexcept;

  // Query interpolated SE(3) pose at query_stamp_ns.
  // Rejects queries requiring extrapolation > max_extrap_ns or older than buffer horizon.
  [[nodiscard]] bool query(
      uint64_t query_stamp_ns,
      TimedPoseSe3& out,
      uint64_t max_extrap_ns = kDefaultMaxExtrapNs) const noexcept;

  // Rotate a vector from body FRD into local NED using active Hamilton quaternion R_{FRD -> NED}.
  [[nodiscard]] static core::Vec3Ned rotate_frd_to_ned(
      const core::QuatPx4FrdNed& q_frd_to_ned,
      const core::Vec3Frd& v_frd) noexcept;

  // Rotate a vector from local NED into body FRD using transpose R_{NED -> FRD}.
  [[nodiscard]] static core::Vec3Frd rotate_ned_to_frd(
      const core::QuatPx4FrdNed& q_frd_to_ned,
      const core::Vec3Ned& v_ned) noexcept;

  [[nodiscard]] static core::QuatPx4FrdNed normalize_quat(const core::QuatPx4FrdNed& q) noexcept;

  [[nodiscard]] static core::QuatPx4FrdNed slerp_quat(
      const core::QuatPx4FrdNed& q0,
      const core::QuatPx4FrdNed& q1,
      double alpha) noexcept;

 private:
  [[nodiscard]] const TimedPoseSe3& at_logical(std::size_t idx) const noexcept;

  std::array<TimedPoseSe3, kCapacity> buf_{};
  std::size_t head_{0};   // Index of oldest sample when full, or 0 before wrap
  std::size_t count_{0};
};

}  // namespace vertirta::forward_hazard
