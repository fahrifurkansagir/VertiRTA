// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vertirta/core/math_utils.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"
#include "vertirta/wind_acoustics/vrta_wind_format.hpp"

namespace vertirta::wind_acoustics::testing {

// Self-contained RAII C++ VRTAWND1 fixture generator writing to a unique per-process/per-test
// temp file (zero Python/numpy dependency and zero parallel `ctest -j` temp-file races).
class ScopedVrtaWindFixture {
 public:
  struct Options {
    double max_lon_deg{29.1892};
    double eps_rom_base_mps{0.45};
    double downdraft_peak_mps{-3.8};
    bool tamper_payload_bit{false};
  };

  ScopedVrtaWindFixture() : ScopedVrtaWindFixture(Options{}) {}

  explicit ScopedVrtaWindFixture(const Options& opt) {
    static std::atomic<uint64_t> counter{0};
    const uint64_t seq = counter.fetch_add(1u, std::memory_order_relaxed);
    const std::string fname = "vertirta_vrtawnd1_test_" + std::to_string(::getpid()) + "_" +
                              std::to_string(seq) + ".vrtawnd1";
    path_ = std::filesystem::temp_directory_path() / fname;
    write_fixture(path_, opt);
  }

  ~ScopedVrtaWindFixture() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  ScopedVrtaWindFixture(const ScopedVrtaWindFixture&) = delete;
  ScopedVrtaWindFixture& operator=(const ScopedVrtaWindFixture&) = delete;

  [[nodiscard]] std::string path_str() const { return path_.string(); }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  static void fill_grid(
      const VrtaWndGridDesc& grid,
      const VrtaWndHeader& hdr,
      const Options& opt,
      std::vector<VrtaWndNode64>& nodes) {
    for (uint32_t iz = 0; iz < grid.nz; ++iz) {
      const double z = grid.origin_enu[2] + static_cast<double>(iz) * grid.spacing_m[2];
      for (uint32_t iy = 0; iy < grid.ny; ++iy) {
        const double y = grid.origin_enu[1] + static_cast<double>(iy) * grid.spacing_m[1];
        for (uint32_t ix = 0; ix < grid.nx; ++ix) {
          const double x = grid.origin_enu[0] + static_cast<double>(ix) * grid.spacing_m[0];
          const uint64_t local_idx = morton_swizzle_index(ix, iy, iz, grid.nx, grid.ny);
          auto& node = nodes[grid.node_offset + local_idx];

          const double r2 = (x + 40.0) * (x + 40.0) + (y - 20.0) * (y - 20.0);
          const double wake = std::exp(-r2 / (180.0 * 180.0)) *
                              std::exp(-((z - 40.0) * (z - 40.0)) / (30.0 * 30.0));
          const double u0_x = 1.2 * (1.0 - 0.3 * wake);
          const double u0_y = 0.8 * (1.0 - 0.25 * wake);
          const double u0_z = opt.downdraft_peak_mps * wake;

          auto q16 = [](double v, double s) -> int16_t {
            const double q = std::clamp(std::rint(v / s), -32767.0, 32767.0);
            return static_cast<int16_t>(q);
          };

          node.ch[0] = q16(u0_x, hdr.vel_scale_mps);
          node.ch[1] = q16(u0_y, hdr.vel_scale_mps);
          node.ch[2] = q16(u0_z, hdr.vel_scale_mps);

          // Mode 0 aligns with East inflow, Mode 1 aligns with North inflow, Modes 2..7 wake harmonics
          const double log_z = std::clamp(std::log((std::max(z, 2.0) + 0.1) / 0.1) / 4.6, 0.4, 1.5);
          node.ch[3] = q16(0.75 * log_z * (1.0 - 0.35 * wake), hdr.mode_scale_mps[0]);
          node.ch[4] = q16(0.05 * wake, hdr.mode_scale_mps[0]);
          node.ch[5] = q16(-0.18 * wake, hdr.mode_scale_mps[0]);

          node.ch[6] = q16(-0.04 * wake, hdr.mode_scale_mps[1]);
          node.ch[7] = q16(0.75 * log_z * (1.0 - 0.30 * wake), hdr.mode_scale_mps[1]);
          node.ch[8] = q16(-0.15 * wake, hdr.mode_scale_mps[1]);

          for (uint32_t m = 2u; m < kVrtaWndNumModes; ++m) {
            const double phase = 0.002 * (x + y) + 0.4 * static_cast<double>(m);
            node.ch[3u + 3u * m] = q16(0.08 * std::sin(phase) * wake, hdr.mode_scale_mps[m]);
            node.ch[4u + 3u * m] = q16(0.08 * std::cos(phase) * wake, hdr.mode_scale_mps[m]);
            node.ch[5u + 3u * m] = q16(-0.06 * wake, hdr.mode_scale_mps[m]);
          }

          node.ch[27] = q16(0.70 + 1.20 * wake, hdr.sigma_scale_mps);
          node.ch[28] = q16(0.60 + 1.00 * wake, hdr.sigma_scale_mps);
          node.ch[29] = q16(0.45 + 0.85 * wake, hdr.sigma_scale_mps);
          node.ch[30] = q16(opt.eps_rom_base_mps + 0.65 * wake, hdr.eps_scale_mps);
          node.ch[31] = q16(24.0 + 0.25 * z, hdr.l_turb_scale_m);
        }
      }
    }
  }

  static void write_fixture(const std::filesystem::path& p, const Options& opt) {
    VrtaWndHeader hdr{};
    std::memcpy(hdr.magic, kVrtaWndMagic.data(), kVrtaWndMagic.size());
    hdr.version = kVrtaWndVersion;
    hdr.flags = 0x03u;
    hdr.num_modes = kVrtaWndNumModes;
    hdr.num_rbf_centers = kVrtaWndNumRbfCenters;
    hdr.datum_lat_deg = 40.8860;
    hdr.datum_lon_deg = 29.1880;
    hdr.datum_alt_hae_m = 48.7;
    hdr.max_lon_deg = opt.max_lon_deg;
    hdr.vel_scale_mps = 40.0 / 32767.0;
    for (uint32_t m = 0u; m < kVrtaWndNumModes; ++m) {
      hdr.mode_scale_mps[m] = 1.0 / 32767.0;
    }
    hdr.sigma_scale_mps = 15.0 / 32767.0;
    hdr.eps_scale_mps = 10.0 / 32767.0;
    hdr.l_turb_scale_m = 200.0 / 32767.0;
    hdr.rbf_shape_c = 4.0;

    // Tier-0: 8x8x8 coastal corridor west of +100 m E
    hdr.tier0.nx = 8u;
    hdr.tier0.ny = 8u;
    hdr.tier0.nz = 8u;
    hdr.tier0.brick_count = 8u;
    hdr.tier0.origin_enu[0] = -2700.0;
    hdr.tier0.origin_enu[1] = -400.0;
    hdr.tier0.origin_enu[2] = 0.0;
    hdr.tier0.spacing_m[0] = 400.0;
    hdr.tier0.spacing_m[1] = 600.0;
    hdr.tier0.spacing_m[2] = 25.0;
    hdr.tier0.node_offset = 0u;
    hdr.tier0.node_count = 512u;

    // Tier-1: 8x8x8 rooftop SLZ nest
    hdr.tier1.nx = 8u;
    hdr.tier1.ny = 8u;
    hdr.tier1.nz = 8u;
    hdr.tier1.brick_count = 8u;
    hdr.tier1.origin_enu[0] = -80.0;
    hdr.tier1.origin_enu[1] = -80.0;
    hdr.tier1.origin_enu[2] = 0.0;
    hdr.tier1.spacing_m[0] = 20.0;
    hdr.tier1.spacing_m[1] = 20.0;
    hdr.tier1.spacing_m[2] = 12.0;
    hdr.tier1.node_offset = 512u;
    hdr.tier1.node_count = 512u;

    const double speeds[3] = {6.0, 12.0, 18.0};
    uint32_t k = 0u;
    for (int d = 0; d < 16; ++d) {
      const double ang = (static_cast<double>(d) * 22.5) * core::kPi / 180.0;
      for (int s = 0; s < 3; ++s) {
        hdr.rbf_centers_ux[k] = speeds[s] * std::cos(ang);
        hdr.rbf_centers_uy[k] = speeds[s] * std::sin(ang);
        ++k;
      }
    }
    // Affine linear modes 0/1 respond directly to (ux, uy)
    hdr.rbf_poly_weights[0][1] = 1.0;
    hdr.rbf_poly_weights[1][2] = 1.0;

    hdr.total_nodes = hdr.tier0.node_count + hdr.tier1.node_count;
    hdr.payload_bytes = hdr.total_nodes * kVrtaWndNodeBytes;

    std::vector<VrtaWndNode64> nodes(hdr.total_nodes);
    fill_grid(hdr.tier0, hdr, opt, nodes);
    fill_grid(hdr.tier1, hdr, opt, nodes);

    rta::compute_sha256(nodes.data(), static_cast<std::size_t>(hdr.payload_bytes), hdr.payload_sha256);

    if (opt.tamper_payload_bit && !nodes.empty()) {
      nodes[0].ch[0] ^= static_cast<int16_t>(0x0001);
    }

    std::ofstream ofs(p, std::ios::binary | std::ios::trunc);
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(VrtaWndHeader));
    ofs.write(reinterpret_cast<const char*>(nodes.data()), static_cast<std::streamsize>(hdr.payload_bytes));
    ofs.close();
  }

  std::filesystem::path path_{};
};

}  // namespace vertirta::wind_acoustics::testing
