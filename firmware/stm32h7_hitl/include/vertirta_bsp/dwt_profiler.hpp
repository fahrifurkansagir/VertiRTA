// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace vertirta::bsp {

struct CycleSummary {
  uint32_t count{0u};
  uint32_t bracket_overhead_cycles{0u};
  uint32_t min_cycles{0u};
  uint32_t p50_cycles{0u};
  uint32_t p99_cycles{0u};
  uint32_t p999_cycles{0u};
  uint32_t max_cycles{0u};
  uint32_t fpscr_sticky_or{0u};  // Aggregated FPSCR[4:0] (ES0392 Erratum 2.1.6)
  uint32_t fpscr_nonzero_ticks{0u};
};

// Reads serialized DWT->CYCCNT (issues DSB + ISB before register read).
[[nodiscard]] uint32_t read_dwt_cyccnt_serialized() noexcept;

// Computes wrap-safe unsigned cycle delta minus calibrated empty-bracket overhead.
[[nodiscard]] constexpr uint32_t compute_cycle_delta(
    uint32_t t0, uint32_t t1, uint32_t overhead) noexcept {
  const uint32_t raw = t1 - t0;
  return (raw > overhead) ? (raw - overhead) : 0u;
}

// Reads and clears Cortex-M7 FPU sticky exception flags FPSCR[4:0] (ES0392 Erratum 2.1.6:
// FPU interrupt is unmapped on NVIC line 81, so every 400 Hz tick must poll/clear FPSCR).
[[nodiscard]] uint32_t poll_and_clear_fpscr_flags() noexcept;

// Atomic GPIO BSRR toggle on PB0 (NUCLEO-H743ZI2 green LED / external oscilloscope trigger).
void scope_gpio_set(bool high) noexcept;

// Zero-heap, zero-recursion profiler storing up to 10,000 raw uint32_t cycle samples in .axi_sram (40 KB).
class DwtCycleProfiler {
 public:
  static constexpr std::size_t kMaxSamples = 10000u;

  void reset() noexcept;
  void calibrate_overhead(uint32_t iters = 1024u) noexcept;
  void record_sample(uint32_t t0, uint32_t t1, uint32_t fpscr_flags) noexcept;
  [[nodiscard]] CycleSummary finalize_summary() noexcept;

  [[nodiscard]] uint32_t bracket_overhead() const noexcept { return bracket_overhead_; }
  [[nodiscard]] std::size_t sample_count() const noexcept { return count_; }

 private:
  static void heapsort_u32(uint32_t* data, std::size_t n) noexcept;

  uint32_t bracket_overhead_{0u};
  std::size_t count_{0u};
  uint32_t fpscr_sticky_or_{0u};
  uint32_t fpscr_nonzero_ticks_{0u};
};

}  // namespace vertirta::bsp
