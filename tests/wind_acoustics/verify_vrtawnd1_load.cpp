// SPDX-License-Identifier: Apache-2.0
// Standalone Python -> C++ VRTAWND1 binary ABI parity checker invoked by pytest.
#include <cmath>
#include <iostream>
#include <string_view>

#include "vertirta/core/math_utils.hpp"
#include "vertirta/wind_acoustics/mmap_vrta_wind_lut.hpp"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: verify_vrtawnd1_load <path.vrtawnd1>\n";
    return 2;
  }

  vertirta::wind_acoustics::MmapVrtaWindLut lut;
  if (!lut.load(std::string_view(argv[1]), /*verify_sha256=*/true)) {
    std::cerr << "FAIL: MmapVrtaWindLut::load rejected " << argv[1] << "\n";
    return 1;
  }

  lut.update_inflow(225.0 * vertirta::core::kPi / 180.0, 14.0);

  vertirta::wind_acoustics::WindRomSample s_tier1{};
  if (!lut.query_enu({-20.0, 10.0, 36.0}, s_tier1) || !s_tier1.valid || s_tier1.tier_used != 1u ||
      !s_tier1.mean_wind_enu.is_finite() || s_tier1.sigma_rms_mps <= 0.0 ||
      s_tier1.eps_rom_p95_mps <= 0.0) {
    std::cerr << "FAIL: Tier-1 query failed or returned non-physical values\n";
    return 1;
  }

  vertirta::wind_acoustics::WindRomSample s_tier0{};
  if (!lut.query_enu({-1200.0, 800.0, 60.0}, s_tier0) || !s_tier0.valid ||
      s_tier0.tier_used != 0u || !s_tier0.mean_wind_enu.is_finite() ||
      s_tier0.eps_rom_p95_mps <= 0.0) {
    std::cerr << "FAIL: Tier-0 query failed or returned non-physical values\n";
    return 1;
  }

  std::cout << "VRTAWND1_OK nodes=" << lut.header().total_nodes
            << " max_lon=" << lut.header().max_lon_deg
            << " t1_wind=(" << s_tier1.mean_wind_enu.x << "," << s_tier1.mean_wind_enu.y << ","
            << s_tier1.mean_wind_enu.z << ")"
            << " eps_p95=" << s_tier1.eps_rom_p95_mps << "\n";
  return 0;
}
