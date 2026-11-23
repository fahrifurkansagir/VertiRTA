// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace vertirta::wind_acoustics {

inline constexpr std::array<char, 8> kVrtaWndMagic{'V', 'R', 'T', 'A', 'W', 'N', 'D', '1'};
inline constexpr uint32_t kVrtaWndVersion = 1u;
inline constexpr std::size_t kVrtaWndHeaderBytes = 12288u;
inline constexpr std::size_t kVrtaWndNodeBytes = 64u;
inline constexpr uint32_t kVrtaWndNumModes = 8u;
inline constexpr uint32_t kVrtaWndNumRbfCenters = 48u;
inline constexpr double kLtfjMaxAllowedLonDeg = 29.1895;

// Single cache-line (64 B) quantized POD-RBF wind surrogate node stored inside 4x4x4 Morton bricks.
// Channel layout:
//   ch[0..2]   : mean velocity u0 (E, N, U)
//   ch[3..26]  : 8 spatial POD modes x 3 velocity components (mode m at 3 + 3*m + c)
//   ch[27..29] : anisotropic turbulence RMS (sigma_E, sigma_N, sigma_U)
//   ch[30]     : calibrated LODO reconstruction residual eps_rom_p95
//   ch[31]     : integral turbulence length scale L_turb
struct alignas(64) VrtaWndNode64 {
  std::array<int16_t, 32> ch{};
};
static_assert(sizeof(VrtaWndNode64) == kVrtaWndNodeBytes);
static_assert(alignof(VrtaWndNode64) == 64u);

struct VrtaWndGridDesc {
  uint32_t nx{0};
  uint32_t ny{0};
  uint32_t nz{0};
  uint32_t brick_count{0};
  double origin_enu[3]{0.0, 0.0, 0.0};
  double spacing_m[3]{1.0, 1.0, 1.0};
  uint64_t node_offset{0};
  uint64_t node_count{0};
};
static_assert(sizeof(VrtaWndGridDesc) == 80u);

struct VrtaWndHeader {
  char magic[8]{};
  uint32_t version{0};
  uint32_t flags{0};
  uint32_t num_modes{0};
  uint32_t num_rbf_centers{0};
  double datum_lat_deg{0.0};
  double datum_lon_deg{0.0};
  double datum_alt_hae_m{0.0};
  double max_lon_deg{0.0};
  double vel_scale_mps{0.0};
  double mode_scale_mps[kVrtaWndNumModes]{};
  double sigma_scale_mps{0.0};
  double eps_scale_mps{0.0};
  double l_turb_scale_m{0.0};
  double rbf_shape_c{0.0};
  VrtaWndGridDesc tier0{};
  VrtaWndGridDesc tier1{};
  double rbf_centers_ux[kVrtaWndNumRbfCenters]{};
  double rbf_centers_uy[kVrtaWndNumRbfCenters]{};
  double rbf_weights[kVrtaWndNumRbfCenters][kVrtaWndNumModes]{};
  double rbf_poly_weights[kVrtaWndNumModes][3]{};
  uint64_t total_nodes{0};
  uint64_t payload_bytes{0};
  uint8_t payload_sha256[32]{};
  uint8_t reserved[7888]{};
};

static_assert(sizeof(VrtaWndHeader) == kVrtaWndHeaderBytes);
static_assert(offsetof(VrtaWndHeader, tier0) == 160u);
static_assert(offsetof(VrtaWndHeader, tier1) == 240u);
static_assert(offsetof(VrtaWndHeader, rbf_centers_ux) == 320u);
static_assert(offsetof(VrtaWndHeader, rbf_centers_uy) == 704u);
static_assert(offsetof(VrtaWndHeader, rbf_weights) == 1088u);
static_assert(offsetof(VrtaWndHeader, rbf_poly_weights) == 4160u);
static_assert(offsetof(VrtaWndHeader, total_nodes) == 4352u);
static_assert(offsetof(VrtaWndHeader, payload_bytes) == 4360u);
static_assert(offsetof(VrtaWndHeader, payload_sha256) == 4368u);
static_assert(offsetof(VrtaWndHeader, reserved) == 4400u);

// 4x4x4 local brick Morton Z-order index [0..63] interleaving 2 bits each of (lx, ly, lz).
[[nodiscard]] constexpr uint32_t morton_encode_4x4x4(
    uint32_t lx,
    uint32_t ly,
    uint32_t lz) noexcept {
  const uint32_t x0 = lx & 1u;
  const uint32_t x1 = (lx >> 1u) & 1u;
  const uint32_t y0 = ly & 1u;
  const uint32_t y1 = (ly >> 1u) & 1u;
  const uint32_t z0 = lz & 1u;
  const uint32_t z1 = (lz >> 1u) & 1u;
  return x0 | (y0 << 1u) | (z0 << 2u) | (x1 << 3u) | (y1 << 4u) | (z1 << 5u);
}

[[nodiscard]] constexpr uint64_t morton_swizzle_index(
    uint32_t ix,
    uint32_t iy,
    uint32_t iz,
    uint32_t nx,
    uint32_t ny) noexcept {
  const uint64_t bx = ix >> 2u;
  const uint64_t by = iy >> 2u;
  const uint64_t bz = iz >> 2u;
  const uint64_t nbx = nx >> 2u;
  const uint64_t nby = ny >> 2u;
  const uint64_t brick = (bz * nby + by) * nbx + bx;
  return brick * 64ull + static_cast<uint64_t>(morton_encode_4x4x4(ix & 3u, iy & 3u, iz & 3u));
}

}  // namespace vertirta::wind_acoustics
