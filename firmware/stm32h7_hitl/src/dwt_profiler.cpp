// SPDX-License-Identifier: Apache-2.0
#include "vertirta_bsp/dwt_profiler.hpp"

#include "vertirta_bsp/stm32h7_regs.hpp"

namespace vertirta::bsp {

namespace {

// 10,000 raw 32-bit cycle deltas (40 KB) placed in D1 AXI SRAM (.axi_sram) so zero DTCM/heap is spent.
#if defined(__arm__) || defined(__thumb__)
__attribute__((section(".axi_sram"), aligned(32)))
#endif
uint32_t g_cycle_samples[DwtCycleProfiler::kMaxSamples]{};

}  // namespace

uint32_t read_dwt_cyccnt_serialized() noexcept {
  regs::dsb();
  regs::isb();
  return regs::reg32(regs::kDwtCyccnt);
}

uint32_t poll_and_clear_fpscr_flags() noexcept {
#if defined(__arm__) && defined(__VFP_FP__) && !defined(__SOFTFP__)
  uint32_t fpscr = 0u;
  __asm__ volatile("vmrs %0, fpscr" : "=r"(fpscr));
  const uint32_t flags = fpscr & regs::kFpscrExceptionMask;
  if (flags != 0u) {
    const uint32_t cleared = fpscr & ~regs::kFpscrExceptionMask;
    __asm__ volatile("vmsr fpscr, %0" :: "r"(cleared));
  }
  return flags;
#else
  return 0u;
#endif
}

void scope_gpio_set(bool high) noexcept {
  // Atomic BSRR write on GPIOB pin 0 (bit 0 = set, bit 16 = reset).
  regs::reg32(regs::kGpioBBase + regs::kGpioBsrrOff) = high ? (1u << 0u) : (1u << 16u);
}

void DwtCycleProfiler::reset() noexcept {
  count_ = 0u;
  fpscr_sticky_or_ = 0u;
  fpscr_nonzero_ticks_ = 0u;
}

void DwtCycleProfiler::calibrate_overhead(uint32_t iters) noexcept {
  if (iters == 0u) {
    bracket_overhead_ = 0u;
    return;
  }
  uint64_t acc = 0u;
  for (uint32_t i = 0u; i < iters; ++i) {
    const uint32_t t0 = read_dwt_cyccnt_serialized();
    const uint32_t t1 = read_dwt_cyccnt_serialized();
    acc += static_cast<uint64_t>(t1 - t0);
  }
  bracket_overhead_ = static_cast<uint32_t>(acc / iters);
}

void DwtCycleProfiler::record_sample(uint32_t t0, uint32_t t1, uint32_t fpscr_flags) noexcept {
  const uint32_t delta = compute_cycle_delta(t0, t1, bracket_overhead_);
  if (count_ < kMaxSamples) {
    g_cycle_samples[count_++] = delta;
  }
  if (fpscr_flags != 0u) {
    fpscr_sticky_or_ |= fpscr_flags;
    ++fpscr_nonzero_ticks_;
  }
}

void DwtCycleProfiler::heapsort_u32(uint32_t* data, std::size_t n) noexcept {
  if (n < 2u) {
    return;
  }
  auto sift_down = [&](std::size_t start, std::size_t end) noexcept {
    std::size_t root = start;
    while (2u * root + 1u <= end) {
      std::size_t child = 2u * root + 1u;
      std::size_t swap_idx = root;
      if (data[swap_idx] < data[child]) {
        swap_idx = child;
      }
      if (child + 1u <= end && data[swap_idx] < data[child + 1u]) {
        swap_idx = child + 1u;
      }
      if (swap_idx == root) {
        return;
      }
      const uint32_t tmp = data[root];
      data[root] = data[swap_idx];
      data[swap_idx] = tmp;
      root = swap_idx;
    }
  };

  for (std::size_t i = n / 2u; i > 0u; --i) {
    sift_down(i - 1u, n - 1u);
  }
  for (std::size_t end = n - 1u; end > 0u; --end) {
    const uint32_t tmp = data[end];
    data[end] = data[0];
    data[0] = tmp;
    sift_down(0u, end - 1u);
  }
}

CycleSummary DwtCycleProfiler::finalize_summary() noexcept {
  CycleSummary s{};
  s.count = static_cast<uint32_t>(count_);
  s.bracket_overhead_cycles = bracket_overhead_;
  s.fpscr_sticky_or = fpscr_sticky_or_;
  s.fpscr_nonzero_ticks = fpscr_nonzero_ticks_;
  if (count_ == 0u) {
    return s;
  }
  heapsort_u32(g_cycle_samples, count_);
  const std::size_t last = count_ - 1u;
  s.min_cycles = g_cycle_samples[0];
  s.p50_cycles = g_cycle_samples[(last * 50u) / 100u];
  s.p99_cycles = g_cycle_samples[(last * 99u) / 100u];
  s.p999_cycles = g_cycle_samples[(last * 999u) / 1000u];
  s.max_cycles = g_cycle_samples[last];
  return s;
}

}  // namespace vertirta::bsp
