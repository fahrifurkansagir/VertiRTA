#include "vertirta/core/vrta_recorder.hpp"

#include <cstring>

namespace vertirta::core {

uint32_t crc32c(const void* data, std::size_t len) noexcept {
  const auto* p = static_cast<const uint8_t*>(data);
  uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < len; ++i) {
    crc ^= static_cast<uint32_t>(p[i]);
    for (int bit = 0; bit < 8; ++bit) {
      const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(crc & 1u));
      crc = (crc >> 1) ^ (0x82F63B78u & mask);
    }
  }
  return ~crc;
}

VrtaFileHeader make_default_vrta_header(
    const char* git_sha_40,
    const uint8_t* lut_sha256_32) noexcept {
  VrtaFileHeader hdr{};
  if (lut_sha256_32 != nullptr) {
    std::memcpy(hdr.lut_sha256, lut_sha256_32, 32);
  }
  if (git_sha_40 != nullptr) {
    for (std::size_t i = 0; i < 40 && git_sha_40[i] != '\0'; ++i) {
      hdr.git_sha[i] = git_sha_40[i];
    }
  }
  finalize_vrta_header(hdr);
  return hdr;
}

void finalize_vrta_header(VrtaFileHeader& hdr) noexcept {
  hdr.header_crc32c = crc32c(&hdr, offsetof(VrtaFileHeader, header_crc32c));
}

bool verify_vrta_header(const VrtaFileHeader& hdr) noexcept {
  static constexpr char kMagic[8] = {'V', 'R', 'T', 'A', '0', '0', '0', '1'};
  if (std::memcmp(hdr.magic, kMagic, 8) != 0) {
    return false;
  }
  if (hdr.schema_ver != 1u || hdr.reserved0 != 0u) {
    return false;
  }
  return hdr.header_crc32c == crc32c(&hdr, offsetof(VrtaFileHeader, header_crc32c));
}

void finalize_vrta_frame(VrtaFrame100Hz& frame) noexcept {
  frame.crc32c = crc32c(&frame, offsetof(VrtaFrame100Hz, crc32c));
}

bool verify_vrta_frame(const VrtaFrame100Hz& frame) noexcept {
  return frame.crc32c == crc32c(&frame, offsetof(VrtaFrame100Hz, crc32c));
}

}  // namespace vertirta::core
