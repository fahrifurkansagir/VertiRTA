// SPDX-License-Identifier: Apache-2.0
#include "vertirta_cyphal/o1heap_arena.hpp"

#include <cstring>

namespace vertirta::cyphal {

O1HeapDtcmArena::O1HeapDtcmArena() noexcept {
  reset();
}

void O1HeapDtcmArena::reset() noexcept {
  for (std::size_t i = 0u; i < kNumBins; ++i) {
    bins_[i] = nullptr;
  }
  nonempty_mask_ = 0u;
  diag_.capacity = kArenaSizeBytes;
  diag_.allocated = 0u;
  diag_.peak_allocated = 0u;
  diag_.peak_request_size = 0u;
  diag_.oom_count = 0u;

  auto* root = reinterpret_cast<FragmentHeader*>(storage_);
  root->prev_phys = nullptr;
  root->size_and_used = static_cast<std::uintptr_t>(kArenaSizeBytes);
  push_free(root);
}

std::uint32_t O1HeapDtcmArena::floor_log2_u32(std::uint32_t v) noexcept {
  if (v <= 1u) {
    return 0u;
  }
  return 31u - static_cast<std::uint32_t>(__builtin_clz(v));
}

std::uint32_t O1HeapDtcmArena::ceil_log2_u32(std::uint32_t v) noexcept {
  if (v <= 1u) {
    return 0u;
  }
  return 32u - static_cast<std::uint32_t>(__builtin_clz(v - 1u));
}

std::uint32_t O1HeapDtcmArena::bin_index_for_size(std::size_t total_bytes) noexcept {
  const std::uint32_t blocks = static_cast<std::uint32_t>(total_bytes / kMinBlockBytes);
  const std::uint32_t b = floor_log2_u32(blocks);
  return (b < kNumBins) ? b : static_cast<std::uint32_t>(kNumBins - 1u);
}

O1HeapDtcmArena::FragmentHeader* O1HeapDtcmArena::next_phys(FragmentHeader* hdr) noexcept {
  return reinterpret_cast<FragmentHeader*>(
      reinterpret_cast<std::uint8_t*>(hdr) + block_size(hdr));
}

const O1HeapDtcmArena::FragmentHeader* O1HeapDtcmArena::next_phys(
    const FragmentHeader* hdr) const noexcept {
  return reinterpret_cast<const FragmentHeader*>(
      reinterpret_cast<const std::uint8_t*>(hdr) + block_size(hdr));
}

void O1HeapDtcmArena::push_free(FragmentHeader* hdr) noexcept {
  const std::uint32_t bin = bin_index_for_size(block_size(hdr));
  FragmentHeader* head = bins_[bin];
  FreeLinks* links = free_links(hdr);
  links->prev_free = nullptr;
  links->next_free = head;
  if (head != nullptr) {
    free_links(head)->prev_free = hdr;
  }
  bins_[bin] = hdr;
  nonempty_mask_ |= (1u << bin);
}

void O1HeapDtcmArena::unlink_free(FragmentHeader* hdr) noexcept {
  const std::uint32_t bin = bin_index_for_size(block_size(hdr));
  FreeLinks* links = free_links(hdr);
  FragmentHeader* prev = links->prev_free;
  FragmentHeader* next = links->next_free;
  if (prev != nullptr) {
    free_links(prev)->next_free = next;
  } else {
    bins_[bin] = next;
  }
  if (next != nullptr) {
    free_links(next)->prev_free = prev;
  }
  links->prev_free = nullptr;
  links->next_free = nullptr;
  if (bins_[bin] == nullptr) {
    nonempty_mask_ &= ~(1u << bin);
  }
}

void* O1HeapDtcmArena::allocate(std::size_t amount) noexcept {
  if (amount == 0u) {
    return nullptr;
  }
  if (amount > diag_.peak_request_size) {
    diag_.peak_request_size = amount;
  }
  if (amount > (kArenaSizeBytes - kHeaderBytes)) {
    ++diag_.oom_count;
    return nullptr;
  }

  // Round up total fragment size to a multiple of kMinBlockBytes (>= kMinBlockBytes).
  const std::size_t raw = kHeaderBytes + amount;
  std::size_t need = ((raw + kMinBlockBytes - 1u) / kMinBlockBytes) * kMinBlockBytes;
  if (need < kMinBlockBytes) {
    need = kMinBlockBytes;
  }
  if (need > kArenaSizeBytes) {
    ++diag_.oom_count;
    return nullptr;
  }

  const std::uint32_t need_blocks = static_cast<std::uint32_t>(need / kMinBlockBytes);
  const std::uint32_t min_bin = ceil_log2_u32(need_blocks);
  if (min_bin >= kNumBins) {
    ++diag_.oom_count;
    return nullptr;
  }

  const std::uint32_t candidate_mask = nonempty_mask_ & (~0u << min_bin);
  if (candidate_mask == 0u) {
    ++diag_.oom_count;
    return nullptr;
  }

  // Strictly O(1): every fragment in bin >= min_bin has size >= 2^min_bin * kMinBlockBytes >= need.
  const std::uint32_t chosen_bin = static_cast<std::uint32_t>(__builtin_ctz(candidate_mask));
  FragmentHeader* blk = bins_[chosen_bin];
  unlink_free(blk);

  const std::size_t total = block_size(blk);
  const std::size_t leftover = total - need;
  const auto* arena_end = reinterpret_cast<const FragmentHeader*>(storage_ + kArenaSizeBytes);

  if (leftover >= kMinBlockBytes) {
    FragmentHeader* old_next = next_phys(blk);
    blk->size_and_used = static_cast<std::uintptr_t>(need) | kUsedBit;
    FragmentHeader* tail = next_phys(blk);
    tail->prev_phys = blk;
    tail->size_and_used = static_cast<std::uintptr_t>(leftover);
    if (old_next < arena_end) {
      old_next->prev_phys = tail;
    }
    push_free(tail);
  } else {
    need = total;
    blk->size_and_used = static_cast<std::uintptr_t>(total) | kUsedBit;
  }

  diag_.allocated += need;
  if (diag_.allocated > diag_.peak_allocated) {
    diag_.peak_allocated = diag_.allocated;
  }
  return reinterpret_cast<std::uint8_t*>(blk) + kHeaderBytes;
}

void O1HeapDtcmArena::deallocate(void* pointer) noexcept {
  if (pointer == nullptr) {
    return;
  }
  auto* byte_ptr = static_cast<std::uint8_t*>(pointer);
  if ((byte_ptr < (storage_ + kHeaderBytes)) || (byte_ptr >= (storage_ + kArenaSizeBytes))) {
    return;
  }

  auto* blk = reinterpret_cast<FragmentHeader*>(byte_ptr - kHeaderBytes);
  if (!is_used(blk)) {
    return;
  }

  std::size_t sz = block_size(blk);
  diag_.allocated -= sz;
  blk->size_and_used = static_cast<std::uintptr_t>(sz);

  const auto* arena_end = reinterpret_cast<const FragmentHeader*>(storage_ + kArenaSizeBytes);

  // Coalesce with physically next fragment in O(1).
  FragmentHeader* nxt = next_phys(blk);
  if ((nxt < arena_end) && !is_used(nxt)) {
    unlink_free(nxt);
    sz += block_size(nxt);
    blk->size_and_used = static_cast<std::uintptr_t>(sz);
    FragmentHeader* after = next_phys(blk);
    if (after < arena_end) {
      after->prev_phys = blk;
    }
  }

  // Coalesce with physically previous fragment in O(1).
  FragmentHeader* prv = blk->prev_phys;
  if ((prv != nullptr) && !is_used(prv)) {
    unlink_free(prv);
    sz += block_size(prv);
    prv->size_and_used = static_cast<std::uintptr_t>(sz);
    blk = prv;
    FragmentHeader* after = next_phys(blk);
    if (after < arena_end) {
      after->prev_phys = blk;
    }
  }

  push_free(blk);
}

void* O1HeapDtcmArena::reallocate(void* pointer, std::size_t new_amount) noexcept {
  if (pointer == nullptr) {
    return allocate(new_amount);
  }
  if (new_amount == 0u) {
    deallocate(pointer);
    return nullptr;
  }
  auto* byte_ptr = static_cast<std::uint8_t*>(pointer);
  auto* blk = reinterpret_cast<FragmentHeader*>(byte_ptr - kHeaderBytes);
  const std::size_t old_payload = block_size(blk) - kHeaderBytes;
  if (new_amount <= old_payload) {
    return pointer;
  }
  void* fresh = allocate(new_amount);
  if (fresh == nullptr) {
    return nullptr;
  }
  std::memcpy(fresh, pointer, old_payload);
  deallocate(pointer);
  return fresh;
}

O1HeapDiagnostics O1HeapDtcmArena::diagnostics() const noexcept {
  return diag_;
}

bool O1HeapDtcmArena::invariants_hold() const noexcept {
  const auto* arena_begin = reinterpret_cast<const FragmentHeader*>(storage_);
  const auto* arena_end = reinterpret_cast<const FragmentHeader*>(storage_ + kArenaSizeBytes);

  const FragmentHeader* cur = arena_begin;
  const FragmentHeader* prev = nullptr;
  std::size_t used_sum = 0u;
  std::size_t total_sum = 0u;
  std::size_t phys_free_count = 0u;
  bool prev_was_free = false;

  while (cur < arena_end) {
    const std::size_t sz = block_size(cur);
    if ((sz < kMinBlockBytes) || ((sz % kMinBlockBytes) != 0u)) {
      return false;
    }
    if ((total_sum + sz) > kArenaSizeBytes) {
      return false;
    }
    if (cur->prev_phys != prev) {
      return false;
    }
    const bool used = is_used(cur);
    if (used) {
      used_sum += sz;
      prev_was_free = false;
    } else {
      if (prev_was_free) {
        return false;  // Immediate coalescing invariant violated
      }
      ++phys_free_count;
      prev_was_free = true;
    }
    total_sum += sz;
    prev = cur;
    cur = next_phys(cur);
  }

  if ((cur != arena_end) || (total_sum != kArenaSizeBytes) || (used_sum != diag_.allocated)) {
    return false;
  }

  std::size_t bin_free_count = 0u;
  for (std::uint32_t b = 0u; b < kNumBins; ++b) {
    const bool mask_bit = ((nonempty_mask_ >> b) & 1u) != 0u;
    const FragmentHeader* node = bins_[b];
    if (mask_bit != (node != nullptr)) {
      return false;
    }
    const FragmentHeader* expected_prev = nullptr;
    while (node != nullptr) {
      if ((node < arena_begin) || (node >= arena_end) || is_used(node)) {
        return false;
      }
      if (bin_index_for_size(block_size(node)) != b) {
        return false;
      }
      const FreeLinks* links = free_links(node);
      if (links->prev_free != expected_prev) {
        return false;
      }
      expected_prev = node;
      node = links->next_free;
      ++bin_free_count;
      if (bin_free_count > (kArenaSizeBytes / kMinBlockBytes)) {
        return false;  // Cycle guard
      }
    }
  }

  return bin_free_count == phys_free_count;
}

}  // namespace vertirta::cyphal
