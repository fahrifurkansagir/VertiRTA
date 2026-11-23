// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "vertirta/wind_acoustics/gz_urban_wind_adapter.hpp"
#include "vertirta/wind_acoustics/mmap_vrta_wind_lut.hpp"
#include "vrta_wind_test_fixture.hpp"

using namespace vertirta::wind_acoustics;
using vertirta::wind_acoustics::testing::ScopedVrtaWindFixture;

TEST_CASE(
    "MmapVrtaWindLut verifies SHA-256, rejects 1-bit tamper & LTFJ breach, runs <5us lookup, and inflates CBVF via eps_rom_p95") {
  static_assert(sizeof(VrtaWndNode64) == 64u);
  static_assert(alignof(VrtaWndNode64) == 64u);
  static_assert(sizeof(VrtaWndHeader) == 12288u);

  // 0) Morton 4x4x4 bijection onto [0..63]
  std::array<bool, 64> seen{};
  for (uint32_t lx = 0; lx < 4u; ++lx) {
    for (uint32_t ly = 0; ly < 4u; ++ly) {
      for (uint32_t lz = 0; lz < 4u; ++lz) {
        const uint32_t idx = morton_encode_4x4x4(lx, ly, lz);
        REQUIRE(idx < 64u);
        seen[idx] = true;
      }
    }
  }
  for (bool v : seen) {
    REQUIRE(v);
  }

  // 1) Self-contained RAII fixture loads with SHA-256 verification; 1-bit tamper & LTFJ breach fail closed
  ScopedVrtaWindFixture clean_fix;
  ScopedVrtaWindFixture::Options tamper_opt{};
  tamper_opt.tamper_payload_bit = true;
  ScopedVrtaWindFixture tampered_fix(tamper_opt);

  ScopedVrtaWindFixture::Options ltfj_opt{};
  ltfj_opt.max_lon_deg = 29.2057;  // Erroneous DR-09 ISTMarina inside LTFJ CTR
  ScopedVrtaWindFixture ltfj_bad_fix(ltfj_opt);

  MmapVrtaWindLut lut;
  REQUIRE(lut.load(tampered_fix.path_str(), /*verify_sha256=*/true) == false);
  REQUIRE_FALSE(lut.is_loaded());

  REQUIRE(lut.load(ltfj_bad_fix.path_str(), /*verify_sha256=*/true) == false);
  REQUIRE_FALSE(lut.is_loaded());

  REQUIRE(lut.load(clean_fix.path_str(), /*verify_sha256=*/true) == true);
  REQUIRE(lut.is_loaded());
  REQUIRE(lut.header().max_lon_deg < kLtfjMaxAllowedLonDeg);
  // Verify best-effort mlock() state transfers cleanly across move semantics & clears on unload.
  {
    MmapVrtaWindLut moved_lut;
    REQUIRE(moved_lut.load(clean_fix.path_str(), /*verify_sha256=*/true));
    const bool locked_before = moved_lut.is_mlocked();
    MmapVrtaWindLut dst_lut(std::move(moved_lut));
    REQUIRE_FALSE(moved_lut.is_loaded());
    REQUIRE_FALSE(moved_lut.is_mlocked());
    REQUIRE(dst_lut.is_loaded());
    REQUIRE(dst_lut.is_mlocked() == locked_before);
    dst_lut.unload();
    REQUIRE_FALSE(dst_lut.is_mlocked());
  }

  // 2) Query valid ENU inside Tier-1 rooftop nest and Tier-0 corridor
  lut.update_inflow(225.0 * vertirta::core::kPi / 180.0, 14.0);

  WindRomSample s1{};
  REQUIRE(lut.query_enu({-25.0, 15.0, 36.0}, s1));
  REQUIRE(s1.valid);
  REQUIRE(s1.tier_used == 1u);
  REQUIRE(s1.in_bounds);
  REQUIRE(s1.mean_wind_enu.is_finite());
  REQUIRE(s1.sigma_rms_mps > 0.0);
  REQUIRE(s1.eps_rom_p95_mps > 0.0);

  WindRomSample s0{};
  REQUIRE(lut.query_enu({-1400.0, 1200.0, 80.0}, s0));
  REQUIRE(s0.valid);
  REQUIRE(s0.tier_used == 0u);
  REQUIRE(s0.in_bounds);
  REQUIRE(s0.mean_wind_enu.is_finite());
  REQUIRE(s0.eps_rom_p95_mps > 0.0);

  // Deterministic OU gust realization reproduces bit-for-bit from identical seed
  vertirta::core::Vec3Enu ou_a{};
  vertirta::core::Vec3Enu ou_b{};
  uint64_t seed_a = 20261123ull;
  uint64_t seed_b = 20261123ull;
  WindRomSample sg_a{};
  WindRomSample sg_b{};
  REQUIRE(lut.query_enu({-25.0, 15.0, 36.0}, sg_a, 0.01, &ou_a, &seed_a));
  REQUIRE(lut.query_enu({-25.0, 15.0, 36.0}, sg_b, 0.01, &ou_b, &seed_b));
  REQUIRE(sg_a.total_wind_enu.x == sg_b.total_wind_enu.x);
  REQUIRE(sg_a.total_wind_enu.y == sg_b.total_wind_enu.y);
  REQUIRE(sg_a.total_wind_enu.z == sg_b.total_wind_enu.z);

  // Differential shear adapter returns finite gradients
  const auto shear = query_differential_shear(lut, {-25.0, 15.0, 36.0}, 6.84, 2.2, 0.5);
  REQUIRE(shear.valid);
  REQUIRE(vertirta::core::is_finite_d(shear.dw_z_dy_roll_per_s));
  REQUIRE(vertirta::core::is_finite_d(shear.dw_z_dx_pitch_per_s));

  // 3) Query NaN / Inf coordinates -> fail-closed conservative fallback
  WindRomSample s_nan{};
  REQUIRE_FALSE(lut.query_enu({std::numeric_limits<double>::quiet_NaN(), 0.0, 40.0}, s_nan));
  REQUIRE_FALSE(s_nan.valid);
  REQUIRE(s_nan.eps_rom_p95_mps >= MmapVrtaWindLut::kConservativeFallbackEpsMps);

  WindRomSample s_inf{};
  REQUIRE_FALSE(lut.query_enu({0.0, std::numeric_limits<double>::infinity(), 40.0}, s_inf));
  REQUIRE_FALSE(s_inf.valid);
  REQUIRE(s_inf.eps_rom_p95_mps >= MmapVrtaWindLut::kConservativeFallbackEpsMps);

  // 4) Strictly monotonic CBVF hazard inflation wrt eps_rom_p95
  WindRomSample s_low = s1;
  s_low.eps_rom_p95_mps = 0.20;
  WindRomSample s_high = s1;
  s_high.eps_rom_p95_mps = 1.80;

  std::array<vertirta::rta::HazardPoint, 1> base_hz{};
  base_hz[0].pos_ned = {50.0, -30.0, -40.0};
  base_hz[0].radius_h_m = 10.0;
  base_hz[0].radius_z_m = 8.0;
  base_hz[0].kind = 0u;

  std::array<vertirta::rta::HazardPoint, 1> out_low{};
  std::array<vertirta::rta::HazardPoint, 1> out_high{};
  REQUIRE(inflate_hazards_with_wind_rom(base_hz, s_low, 15.0, 6.87, out_low) == 1u);
  REQUIRE(inflate_hazards_with_wind_rom(base_hz, s_high, 15.0, 6.87, out_high) == 1u);
  REQUIRE(out_low[0].radius_h_m > base_hz[0].radius_h_m);
  REQUIRE(out_high[0].radius_h_m > out_low[0].radius_h_m);
  REQUIRE(out_high[0].radius_z_m > out_low[0].radius_z_m);

  // 5) Latency benchmark: 100,000 flight-path queries (chunked into 1,000 x 100 for exact p99/mean)
  constexpr int kChunks = 1000;
  constexpr int kPerChunk = 100;
  std::vector<double> chunk_us(kChunks);
  double sink = 0.0;
  WindRomSample sample{};

  for (int c = 0; c < kChunks; ++c) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kPerChunk; ++i) {
      const double frac = static_cast<double>((c * kPerChunk + i) % 1000) / 1000.0;
      const vertirta::core::Vec3Enu p{-70.0 + 120.0 * frac, -60.0 + 110.0 * frac, 20.0 + 50.0 * frac};
      (void)lut.query_enu(p, sample);
      sink += sample.mean_wind_enu.x;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    chunk_us[static_cast<std::size_t>(c)] = (ns / static_cast<double>(kPerChunk)) * 1e-3;
  }
  REQUIRE(vertirta::core::is_finite_d(sink));
  std::sort(chunk_us.begin(), chunk_us.end());
  const double mean_us =
      std::accumulate(chunk_us.begin(), chunk_us.end(), 0.0) / static_cast<double>(kChunks);
  const double p99_us = chunk_us[static_cast<std::size_t>(kChunks * 99 / 100)];
  INFO("MmapVrtaWindLut lookup latency: mean=" << mean_us << " us, p99=" << p99_us << " us");
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_UNDEFINED__)
  REQUIRE(mean_us < 5.0);
  REQUIRE(p99_us < 5.0);
#else
  REQUIRE(mean_us < 50.0);
  REQUIRE(p99_us < 80.0);
#endif
}
