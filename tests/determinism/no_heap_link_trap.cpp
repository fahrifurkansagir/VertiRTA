// Standalone non-inline link-time zero-heap trap TU (DR-3 Erratum #1 lock).
// NEVER move into a header and NEVER mark global operator new/new[] `inline`.
// Compiled with -ffunction-sections and linked with -Wl,--gc-sections and
// per-target -Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc so any transitive
// dynamic allocation inside libvertirta_flight_core fails at link time.

#include <cstddef>
#include <new>

extern "C" [[noreturn]] void vertirta_forbidden_heap_allocation_link_trap() noexcept;

void* operator new(std::size_t) {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new[](std::size_t) {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new(std::size_t, const std::nothrow_t&) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new[](std::size_t, const std::nothrow_t&) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new(std::size_t, std::align_val_t) {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new[](std::size_t, std::align_val_t) {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new(std::size_t, std::align_val_t, const std::nothrow_t&) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* operator new[](std::size_t, std::align_val_t, const std::nothrow_t&) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

extern "C" {

void* __wrap_malloc(std::size_t) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* __wrap_calloc(std::size_t, std::size_t) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

void* __wrap_realloc(void*, std::size_t) noexcept {
  vertirta_forbidden_heap_allocation_link_trap();
}

}  // extern "C"
