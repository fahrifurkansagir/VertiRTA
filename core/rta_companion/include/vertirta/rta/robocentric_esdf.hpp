#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"

namespace vertirta::rta {

// Explicitly zero-initialized SlidingConfig (fixes upstream hku-mars/ROG-Map Issue #15).
struct SlidingConfig {
  int32_t origin_ix{0};
  int32_t origin_iy{0};
  int32_t origin_iz{0};
  int32_t safe_margin_i{0};
};

// Bit-packed 16 B FIESTA ESDF voxel (DR-2 Section 4.1 Config B):
//   - dll_packed (8 B): 3 x 18-bit doubly-linked-list indices (head [0..17], prev [18..35], next [36..53]; 0x3FFFF = NULL)
//   - coc_idx    (4 B): closest occupied cell local index (0x3FFFF = NONE)
//   - dist_mm    (2 B): Euclidean distance in mm (0..65535)
//   - occ_cnt    (1 B): 0..8 child 0.5 m occupancy counter hysteresis
//   - flags      (1 B): bit0=occupied, bit1=in_insert_q, bit2=in_delete_q, bit3=in_update_q
struct EsdfCell {
  static constexpr uint32_t kNullIdx = 0x3FFFFu;

  uint64_t dll_packed{0x003FFFFFFFFFFFULL};
  uint32_t coc_idx{kNullIdx};
  uint16_t dist_mm{65535u};
  uint8_t occ_cnt{0};
  uint8_t flags{0};

  [[nodiscard]] uint32_t dll_head() const noexcept { return static_cast<uint32_t>(dll_packed & 0x3FFFFULL); }
  [[nodiscard]] uint32_t dll_prev() const noexcept { return static_cast<uint32_t>((dll_packed >> 18u) & 0x3FFFFULL); }
  [[nodiscard]] uint32_t dll_next() const noexcept { return static_cast<uint32_t>((dll_packed >> 36u) & 0x3FFFFULL); }
  void set_dll(uint32_t head, uint32_t prev, uint32_t next) noexcept {
    dll_packed = (static_cast<uint64_t>(head & 0x3FFFFu)) |
                 (static_cast<uint64_t>(prev & 0x3FFFFu) << 18u) |
                 (static_cast<uint64_t>(next & 0x3FFFFu) << 36u);
  }
};
static_assert(sizeof(EsdfCell) == 16, "EsdfCell must be exactly 16 bytes");

// Dual-Resolution Zero-Heap Robocentric Map (3.094 MiB <= 3.15 MiB static budget):
//   - 0.5 m occupancy log-odds ring buffer: 128 x 128 x 64 = 1.00 MiB
//   - 1.0 m bit-packed FIESTA ESDF ring buffer: 64 x 64 x 32 = 131,072 x 16 B = 2.00 MiB
//   - 3 x 8,192 incremental BFS queues = 96 KiB
class RobocentricEsdfMap {
 public:
  static constexpr int32_t kOccNx = 128;
  static constexpr int32_t kOccNy = 128;
  static constexpr int32_t kOccNz = 64;
  static constexpr std::size_t kOccTotal =
      static_cast<std::size_t>(kOccNx) * kOccNy * kOccNz;  // 1,048,576 B

  static constexpr int32_t kEsdfNx = 64;
  static constexpr int32_t kEsdfNy = 64;
  static constexpr int32_t kEsdfNz = 32;
  static constexpr std::size_t kEsdfTotal =
      static_cast<std::size_t>(kEsdfNx) * kEsdfNy * kEsdfNz;  // 131,072

  static constexpr std::size_t kQueueCap = 8192;
  static constexpr double kOccResM = 0.5;
  static constexpr double kEsdfResM = 1.0;

  RobocentricEsdfMap() noexcept;

  void reset(const core::Vec3Enu& center_enu) noexcept;
  void slide_to(const core::Vec3Enu& new_center_enu) noexcept;

  void insert_point_cloud(
      const core::Vec3Enu& sensor_enu,
      std::span<const core::Vec3Enu> points_enu,
      uint64_t stamp_ns) noexcept;

  // 8-corner trilinear O(1) distance & unit gradient query in ENU.
  [[nodiscard]] bool query_dist_grad(
      const core::Vec3Enu& pos_enu,
      double& dist_m_out,
      core::Vec3Enu& grad_unit_out) const noexcept;

  // Extract up to 4 angularly separated nearest ESDF obstacle points in NED.
  [[nodiscard]] std::size_t extract_nearest_hazards(
      const core::Vec3Enu& pos_enu,
      std::span<HazardPoint> out) const noexcept;

  [[nodiscard]] uint64_t last_update_ns() const noexcept { return last_update_ns_; }
  [[nodiscard]] double age_s(uint64_t now_ns) const noexcept {
    return (now_ns >= last_update_ns_) ? (static_cast<double>(now_ns - last_update_ns_) * 1e-9) : 0.0;
  }
  [[nodiscard]] const SlidingConfig& config() const noexcept { return cfg_; }

 private:
  [[nodiscard]] static constexpr std::size_t occ_local_index(int32_t gx, int32_t gy, int32_t gz) noexcept {
    const uint32_t lx = static_cast<uint32_t>(gx) & 127u;
    const uint32_t ly = static_cast<uint32_t>(gy) & 127u;
    const uint32_t lz = static_cast<uint32_t>(gz) & 63u;
    return (static_cast<std::size_t>(lx) << 13u) | (static_cast<std::size_t>(ly) << 6u) | lz;
  }

  [[nodiscard]] static constexpr uint32_t esdf_local_index(int32_t ex, int32_t ey, int32_t ez) noexcept {
    const uint32_t lx = static_cast<uint32_t>(ex) & 63u;
    const uint32_t ly = static_cast<uint32_t>(ey) & 63u;
    const uint32_t lz = static_cast<uint32_t>(ez) & 31u;
    return (lx << 11u) | (ly << 5u) | lz;
  }

  [[nodiscard]] bool in_esdf_window(int32_t ex, int32_t ey, int32_t ez) const noexcept {
    return ex >= cfg_.origin_ix && ex < cfg_.origin_ix + kEsdfNx &&
           ey >= cfg_.origin_iy && ey < cfg_.origin_iy + kEsdfNy &&
           ez >= cfg_.origin_iz && ez < cfg_.origin_iz + kEsdfNz;
  }

  void decode_esdf_global(uint32_t local_idx, int32_t& ex, int32_t& ey, int32_t& ez) const noexcept;
  void clear_esdf_cell(uint32_t idx) noexcept;
  void mark_occ_voxel(int32_t gx, int32_t gy, int32_t gz, bool occupied) noexcept;
  void process_queues() noexcept;

  SlidingConfig cfg_{};
  uint64_t last_update_ns_{0};
  uint32_t insert_head_{0}, insert_tail_{0};
  uint32_t delete_head_{0}, delete_tail_{0};
  uint32_t update_head_{0}, update_tail_{0};

  std::array<int8_t, kOccTotal> occ_logodds_{};
  std::array<EsdfCell, kEsdfTotal> esdf_{};
  std::array<uint32_t, kQueueCap> insert_q_{};
  std::array<uint32_t, kQueueCap> delete_q_{};
  std::array<uint32_t, kQueueCap> update_q_{};
};

static_assert(sizeof(RobocentricEsdfMap) <= 3300000, "RobocentricEsdfMap must fit <= 3.15 MiB");

}  // namespace vertirta::rta
