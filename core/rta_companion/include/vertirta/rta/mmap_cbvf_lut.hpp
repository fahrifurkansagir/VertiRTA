#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vertirta::rta {

struct CbvfQueryResult {
  double value_m{0.0};
  std::array<double, 4> grad{};  // [dV/dr, dV/dvr, dV/dz, dV/dvz]
  bool valid{false};
};

// Freestanding FIPS 180-4 SHA-256 helper (zero OpenSSL dependency).
void compute_sha256(const void* data, std::size_t len, uint8_t out_32[32]) noexcept;

// Zero-copy mmap + mlock 4D Fault-Degraded CBVF LUT reader (exact 11,313,864 B .vrta_lut).
// Guards !loaded_, regime_idx >= 4, and non-finite inputs before any indexing (DR-2 Erratum #4).
class MmapCbvf4DLut {
 public:
  static constexpr std::size_t kHeaderBytes = 4096;
  static constexpr uint32_t kNumRegimes = 4;
  static constexpr uint32_t kN0 = 41;
  static constexpr uint32_t kN1 = 29;
  static constexpr uint32_t kN2 = 41;
  static constexpr uint32_t kN3 = 29;
  static constexpr std::size_t kElementsPerRegime =
      static_cast<std::size_t>(kN0) * kN1 * kN2 * kN3;  // 1,413,721
  static constexpr std::size_t kPayloadBytes = kNumRegimes * kElementsPerRegime * sizeof(int16_t);
  static constexpr std::size_t kTotalBytes = kHeaderBytes + kPayloadBytes;  // 11,313,864 B

  MmapCbvf4DLut() noexcept = default;
  ~MmapCbvf4DLut() noexcept;

  MmapCbvf4DLut(const MmapCbvf4DLut&) = delete;
  MmapCbvf4DLut& operator=(const MmapCbvf4DLut&) = delete;
  MmapCbvf4DLut(MmapCbvf4DLut&& other) noexcept;
  MmapCbvf4DLut& operator=(MmapCbvf4DLut&& other) noexcept;

  [[nodiscard]] bool load(std::string_view path, bool verify_sha256 = true) noexcept;
  void unload() noexcept;

  [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

  [[nodiscard]] CbvfQueryResult eval(
      uint8_t regime_idx,
      double r,
      double vr,
      double z,
      double vz) const noexcept;

  [[nodiscard]] double eps_grid(uint8_t regime_idx) const noexcept {
    return (regime_idx < kNumRegimes) ? eps_grid_[regime_idx] : 0.0;
  }
  [[nodiscard]] double v_cap_f(uint8_t regime_idx) const noexcept {
    return (regime_idx < kNumRegimes) ? v_cap_f_[regime_idx] : 22.0;
  }
  [[nodiscard]] double d_ar_max() const noexcept { return d_ar_max_; }
  [[nodiscard]] double d_az_max() const noexcept { return d_az_max_; }
  [[nodiscard]] double horizon_s() const noexcept { return horizon_s_; }
  [[nodiscard]] const std::array<uint8_t, 32>& payload_sha256() const noexcept {
    return payload_sha256_;
  }

 private:
  void* mapped_{nullptr};
  std::size_t mapped_size_{0};
  int fd_{-1};
  bool loaded_{false};

  const int16_t* data_{nullptr};
  std::array<uint32_t, 4> dims_{kN0, kN1, kN2, kN3};
  std::array<double, 4> lo_{-2.0, -24.0, -5.0, -12.0};
  std::array<double, 4> hi_{58.0, 16.0, 55.0, 8.0};
  std::array<double, 4> inv_dx_{};
  std::array<double, 4> eps_grid_{};
  std::array<double, 4> v_cap_f_{22.0, 18.0, 14.0, 14.0};
  std::array<uint8_t, 32> payload_sha256_{};
  double scale_{65.0 / 65534.0};
  double offset_{7.5};
  double horizon_s_{6.0};
  double d_ar_max_{1.75};
  double d_az_max_{0.75};
};

}  // namespace vertirta::rta
