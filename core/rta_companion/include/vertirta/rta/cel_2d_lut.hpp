#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace vertirta::rta {

struct Cel2DQuery {
  double value_m{0.0};
  double dv_dz{0.0};
  double dv_dvz{0.0};
  double a_net{0.97};
  double v_td_max{2.0};
  double d_z_max{0.50};
  uint8_t tier_idx{2};
  bool valid{false};
};

// 2D CEL Vertical Braking LUT reader (exact 11,230 B .vrta_cel2d: 4,096 B VRTACEL2 header + 3x41x29 int16).
class Cel2DLut {
 public:
  static constexpr std::size_t kHeaderBytes = 4096;
  static constexpr uint32_t kNumTiers = 3;
  static constexpr uint32_t kNz = 41;
  static constexpr uint32_t kNv = 29;
  static constexpr std::size_t kElementsPerTier = static_cast<std::size_t>(kNz) * kNv;  // 1,189
  static constexpr std::size_t kPayloadBytes = kNumTiers * kElementsPerTier * sizeof(int16_t);  // 7,134 B
  static constexpr std::size_t kTotalBytes = kHeaderBytes + kPayloadBytes;  // 11,230 B

  Cel2DLut() noexcept = default;
  [[nodiscard]] bool load(std::string_view path, bool verify_sha256 = true) noexcept;
  [[nodiscard]] bool is_loaded() const noexcept { return loaded_; }

  [[nodiscard]] uint8_t select_tier(float v_term_v) const noexcept;
  [[nodiscard]] Cel2DQuery eval(float v_term_v, double z_agl, double vz_up) const noexcept;

  [[nodiscard]] double a_net(uint8_t tier) const noexcept {
    return (tier < kNumTiers) ? a_net_tiers_[tier] : a_net_tiers_[2];
  }
  [[nodiscard]] double v_td_max() const noexcept { return v_td_max_; }
  [[nodiscard]] double d_z_max() const noexcept { return d_z_max_; }

 private:
  bool loaded_{false};
  std::array<int16_t, kNumTiers * kElementsPerTier> data_{};
  std::array<uint32_t, 2> dims_{kNz, kNv};
  std::array<double, 2> lo_{-2.0, -12.0};
  std::array<double, 2> hi_{58.0, 4.0};
  std::array<double, 2> inv_dx_{};
  std::array<double, kNumTiers> tw_tiers_{1.25, 1.20, 1.15};
  std::array<double, kNumTiers> v_term_thresh_{358.4, 348.0, 338.0};
  std::array<double, kNumTiers> a_net_tiers_{1.9516625, 1.46133, 0.9709975};
  std::array<double, kNumTiers> eps_grid_{};
  std::array<uint8_t, 32> payload_sha256_{};
  double scale_{100.0 / 65534.0};
  double offset_{10.0};
  double v_td_max_{2.0};
  double d_z_max_{0.50};
};

}  // namespace vertirta::rta
