#include "zero_heap_guard.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace {

thread_local int t_guard_depth = 0;
std::atomic<std::size_t> g_alloc_count{0};
std::atomic<std::size_t> g_alloc_bytes{0};

inline void record_alloc(std::size_t bytes) noexcept {
  if (t_guard_depth > 0) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
}

}  // namespace

#if defined(VERTIRTA_WRAP_MALLOC) && !defined(__SANITIZE_ADDRESS__)
extern "C" {
void* __real_malloc(std::size_t size);
void* __real_calloc(std::size_t nmemb, std::size_t size);
void* __real_realloc(void* ptr, std::size_t size);

void* __wrap_malloc(std::size_t size) {
  record_alloc(size);
  return __real_malloc(size);
}

void* __wrap_calloc(std::size_t nmemb, std::size_t size) {
  record_alloc(nmemb * size);
  return __real_calloc(nmemb, size);
}

void* __wrap_realloc(void* ptr, std::size_t size) {
  record_alloc(size);
  return __real_realloc(ptr, size);
}
}  // extern "C"
#endif

// C++ global operator new overrides (only compiled into standalone test binary, never headers).
#if !defined(__SANITIZE_ADDRESS__)
void* operator new(std::size_t size) {
  record_alloc(size);
#if defined(VERTIRTA_WRAP_MALLOC)
  void* p = __real_malloc(size == 0 ? 1 : size);
#else
  void* p = std::malloc(size == 0 ? 1 : size);
#endif
  if (!p) {
    throw std::bad_alloc();
  }
  return p;
}

void* operator new[](std::size_t size) {
  return ::operator new(size);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  record_alloc(size);
#if defined(VERTIRTA_WRAP_MALLOC)
  return __real_malloc(size == 0 ? 1 : size);
#else
  return std::malloc(size == 0 ? 1 : size);
#endif
}

void* operator new[](std::size_t size, const std::nothrow_t& nt) noexcept {
  return ::operator new(size, nt);
}

void operator delete(void* ptr) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
  std::free(ptr);
}

#if __cplusplus >= 201703L
void* operator new(std::size_t size, std::align_val_t align) {
  record_alloc(size);
  const std::size_t a = static_cast<std::size_t>(align);
  const std::size_t adj = ((size == 0 ? 1 : size) + a - 1) & ~(a - 1);
  void* p = std::aligned_alloc(a, adj);
  if (!p) {
    throw std::bad_alloc();
  }
  return p;
}

void* operator new[](std::size_t size, std::align_val_t align) {
  return ::operator new(size, align);
}

void operator delete(void* ptr, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete(void* ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}

void operator delete[](void* ptr, std::size_t, std::align_val_t) noexcept {
  std::free(ptr);
}
#endif
#endif

namespace vertirta::test {

ScopedZeroHeapGuard::ScopedZeroHeapGuard() noexcept
    : start_count_(g_alloc_count.load(std::memory_order_relaxed)),
      start_bytes_(g_alloc_bytes.load(std::memory_order_relaxed)) {
  ++t_guard_depth;
}

ScopedZeroHeapGuard::~ScopedZeroHeapGuard() noexcept {
  --t_guard_depth;
}

std::size_t ScopedZeroHeapGuard::alloc_count() const noexcept {
  return g_alloc_count.load(std::memory_order_relaxed) - start_count_;
}

std::size_t ScopedZeroHeapGuard::bytes_allocated() const noexcept {
  return g_alloc_bytes.load(std::memory_order_relaxed) - start_bytes_;
}

}  // namespace vertirta::test
