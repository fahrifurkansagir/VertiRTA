// SPDX-License-Identifier: Apache-2.0
// Deterministic O(1) Half-Fit Memory Arena modeled after MIT o1heap v3.0.0 (DR-08 Sec 2.2).
// Strictly isolated inside firmware/renode_cyphal/ (adds 0 B to frozen core/flight_core/).
#pragma once

#include <cstddef>
#include <cstdint>

namespace vertirta::cyphal {

struct O1HeapDiagnostics {
  std::size_t capacity{0};
  std::size_t allocated{0};
  std::size_t peak_allocated{0};
  std::size_t peak_request_size{0};
  std::size_t oom_count{0};
};

// Static 32 KiB DTCM arena for OpenCyphal/libcanard CAN-FD TX/RX framing.
// Half-Fit WCMC bound for M=4,096 B live, l=32 B, n=256 B: H_b = 30,095 B < 32,768 B.
// On 32-bit Cortex-M7: O1HEAP_ALIGNMENT = 8 B, fragment header = 2*sizeof(void*) = 8 B.
class O1HeapDtcmArena {
 public:
  static constexpr std::size_t kArenaSizeBytes = 32768u;
  static constexpr std::size_t kWcmcUpperBoundBytes = 30095u;
  static constexpr std::size_t kAlignment = 2u * sizeof(void*);
  static constexpr std::size_t kHeaderBytes = 2u * sizeof(void*);
  static constexpr std::size_t kMinBlockBytes = 4u * sizeof(void*);
  static constexpr std::size_t kNumBins = 16u;

  static_assert(kArenaSizeBytes > kWcmcUpperBoundBytes,
                "32 KiB arena must exceed Half-Fit WCMC bound (30,095 B)");
  static_assert((kArenaSizeBytes % kMinBlockBytes) == 0u,
                "Arena size must be a multiple of minimum fragment size");

  O1HeapDtcmArena() noexcept;

  O1HeapDtcmArena(const O1HeapDtcmArena&) = delete;
  O1HeapDtcmArena& operator=(const O1HeapDtcmArena&) = delete;

  void reset() noexcept;

  [[nodiscard]] void* allocate(std::size_t amount) noexcept;
  void deallocate(void* pointer) noexcept;
  [[nodiscard]] void* reallocate(void* pointer, std::size_t new_amount) noexcept;

  [[nodiscard]] O1HeapDiagnostics diagnostics() const noexcept;
  [[nodiscard]] bool invariants_hold() const noexcept;

 private:
  struct FragmentHeader {
    FragmentHeader* prev_phys;
    std::uintptr_t size_and_used;  // bit 0 = used; upper bits = total fragment size in bytes
  };
  static_assert(sizeof(FragmentHeader) == kHeaderBytes,
                "o1heap v3.0.0 fragment overhead must equal 2 * sizeof(void*)");

  struct FreeLinks {
    FragmentHeader* next_free;
    FragmentHeader* prev_free;
  };
  static_assert(sizeof(FragmentHeader) + sizeof(FreeLinks) == kMinBlockBytes,
                "Minimum fragment must hold header + doubly-linked free pointers");

  static constexpr std::uintptr_t kUsedBit = 1u;

  [[nodiscard]] static std::size_t block_size(const FragmentHeader* hdr) noexcept {
    return static_cast<std::size_t>(hdr->size_and_used & ~kUsedBit);
  }
  [[nodiscard]] static bool is_used(const FragmentHeader* hdr) noexcept {
    return (hdr->size_and_used & kUsedBit) != 0u;
  }
  [[nodiscard]] static FreeLinks* free_links(FragmentHeader* hdr) noexcept {
    return reinterpret_cast<FreeLinks*>(reinterpret_cast<std::uint8_t*>(hdr) + kHeaderBytes);
  }
  [[nodiscard]] static const FreeLinks* free_links(const FragmentHeader* hdr) noexcept {
    return reinterpret_cast<const FreeLinks*>(
        reinterpret_cast<const std::uint8_t*>(hdr) + kHeaderBytes);
  }
  [[nodiscard]] static std::uint32_t floor_log2_u32(std::uint32_t v) noexcept;
  [[nodiscard]] static std::uint32_t ceil_log2_u32(std::uint32_t v) noexcept;
  [[nodiscard]] static std::uint32_t bin_index_for_size(std::size_t total_bytes) noexcept;

  [[nodiscard]] FragmentHeader* next_phys(FragmentHeader* hdr) noexcept;
  [[nodiscard]] const FragmentHeader* next_phys(const FragmentHeader* hdr) const noexcept;
  void push_free(FragmentHeader* hdr) noexcept;
  void unlink_free(FragmentHeader* hdr) noexcept;

  alignas(kAlignment) std::uint8_t storage_[kArenaSizeBytes]{};
  FragmentHeader* bins_[kNumBins]{};
  std::uint32_t nonempty_mask_{0u};
  O1HeapDiagnostics diag_{};
};

}  // namespace vertirta::cyphal
