#pragma once

#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"

namespace vertirta::core {

// Dual-channel 400 Hz Motor FDI Observer (DR-1 Section 3.5):
// - Channel 1: ESC RPM innovation (isolates electrical/ESC cut in 35-60 ms)
// - Channel 2: Sensorless 4-DOF INDI wrench residual + cosine similarity (isolates blade loss in 65-85 ms)
class MotorFdiObserver {
 public:
  MotorFdiObserver() noexcept;

  void reset() noexcept;

  [[nodiscard]] uint8_t step_400hz(
      const Vec3Frd& gyro_f,
      const Vec3Frd& dgyro_f,
      float accel_z_f_frd,
      const float omega_hat[8],
      const float omega_esc[8],
      const bool esc_valid[8],
      float v_horiz_mps,
      float v_term_v,
      uint8_t current_mask) noexcept;

  [[nodiscard]] uint16_t last_isolation_latency_ms() const noexcept {
    return last_isolation_latency_ms_;
  }

 private:
  uint16_t esc_fault_ticks_[8]{};
  uint16_t wrench_fault_ticks_[8]{};
  uint16_t last_isolation_latency_ms_{0};
};

}  // namespace vertirta::core
