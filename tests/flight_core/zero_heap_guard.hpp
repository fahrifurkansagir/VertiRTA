#pragma once

#include <cstddef>

namespace vertirta::test {

// Standalone-TU zero-heap guard (tracks C++ operator new/new[] and, outside ASan,
// linker-wrapped malloc/calloc/realloc). Never defined inline in production headers.
class ScopedZeroHeapGuard {
 public:
  ScopedZeroHeapGuard() noexcept;
  ~ScopedZeroHeapGuard() noexcept;

  ScopedZeroHeapGuard(const ScopedZeroHeapGuard&) = delete;
  ScopedZeroHeapGuard& operator=(const ScopedZeroHeapGuard&) = delete;

  [[nodiscard]] std::size_t alloc_count() const noexcept;
  [[nodiscard]] std::size_t bytes_allocated() const noexcept;

 private:
  std::size_t start_count_{0};
  std::size_t start_bytes_{0};
};

}  // namespace vertirta::test
