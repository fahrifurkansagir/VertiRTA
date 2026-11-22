#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vertirta::core {

// Canonical naturally-aligned little-endian POD .vrta binary schema (Plan 1A Task 5 contract).
// Shared byte-for-byte with Plan 1B vrta_reader.py (Cesium :5180) and Blender 5.2 LTS cinema.
struct alignas(8) VrtaFileHeader {
  char magic[8]{'V', 'R', 'T', 'A', '0', '0', '0', '1'};
  uint32_t schema_ver{1};
  uint32_t reserved0{0};
  double datum_lat_deg{40.8860};
  double datum_lon_deg{29.1880};
  double datum_amsl_m{12.0};
  double egm2008_n_m{36.7};
  uint8_t lut_sha256[32]{};
  char git_sha[40]{};
  uint8_t reserved1[4]{};
  uint32_t header_crc32c{0};
};
static_assert(sizeof(VrtaFileHeader) == 128, "VrtaFileHeader must be exactly 128 bytes");
static_assert(std::is_trivially_copyable_v<VrtaFileHeader>, "VrtaFileHeader must be POD");
static_assert(offsetof(VrtaFileHeader, header_crc32c) == 124, "header_crc32c offset mismatch");

struct alignas(8) VrtaFrame100Hz {
  uint64_t seq{0};
  uint64_t t_sim_us{0};
  double pos_enu[3]{};
  float vel_enu[3]{};
  float q_frd_ned[4]{1.0f, 0.0f, 0.0f, 0.0f};
  float omega_frd[3]{};
  float a_nom_enu[3]{};
  float a_safe_enu[3]{};
  float rotor_rpm[8]{};
  float v_term_v{0.0f};
  float soc_pct{0.0f};
  float cbvf_min_v{0.0f};
  float ltfj_margin_m{0.0f};
  float ltbx_margin_m{0.0f};
  float chi2_residual{0.0f};
  float agl_m{0.0f};
  uint8_t active_motor_mask{0xFF};
  uint8_t flight_regime{0};
  uint8_t tmr_state{0};
  uint8_t cel_latched{0};
  uint32_t flags{0};
  uint8_t pad[80]{};
  uint32_t crc32c{0};
};
static_assert(sizeof(VrtaFrame100Hz) == 256, "VrtaFrame100Hz must be exactly 256 bytes");
static_assert(std::is_trivially_copyable_v<VrtaFrame100Hz>, "VrtaFrame100Hz must be POD");
static_assert(offsetof(VrtaFrame100Hz, crc32c) == 252, "crc32c offset mismatch");

// Standard RFC 3720 / iSCSI Castagnoli CRC32C (polynomial 0x82F63B78).
[[nodiscard]] uint32_t crc32c(const void* data, std::size_t len) noexcept;

[[nodiscard]] VrtaFileHeader make_default_vrta_header(
    const char* git_sha_40 = nullptr,
    const uint8_t* lut_sha256_32 = nullptr) noexcept;

void finalize_vrta_header(VrtaFileHeader& hdr) noexcept;
[[nodiscard]] bool verify_vrta_header(const VrtaFileHeader& hdr) noexcept;

void finalize_vrta_frame(VrtaFrame100Hz& frame) noexcept;
[[nodiscard]] bool verify_vrta_frame(const VrtaFrame100Hz& frame) noexcept;

}  // namespace vertirta::core
