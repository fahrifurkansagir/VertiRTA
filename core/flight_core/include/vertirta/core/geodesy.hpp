#pragma once

#include <cstddef>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

struct GeodeticWgs84 {
  double lat_deg{0.0};
  double lon_deg{0.0};
  double h_hae_m{0.0};
};

struct Vec3Ecef {
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

// Canonical Kartal Coastal Vertiport ENU Datum Origin (40.8860 N, 29.1880 E, 12.0 m AMSL + 36.7 m EGM2008)
inline constexpr GeodeticWgs84 kKartalDatum{40.8860, 29.1880, 48.7};

// WGS84 ellipsoid constants
inline constexpr double kWgs84A = 6378137.0;
inline constexpr double kWgs84F = 1.0 / 298.257223563;
inline constexpr double kWgs84E2 = kWgs84F * (2.0 - kWgs84F);

// Freestanding constexpr 4x4 bilinear EGM2008 geoid patch over Istanbul Anatolian Coast
// [lat: 40.75..41.15 deg N, lon: 29.00..29.45 deg E], anchored to DHMI AIP LTFJ/LTBA/LTFM
// undulation records (+36.58 m LTFJ, +36.70 m Kartal, +36.88 m LTBA, +37.19 m LTFM).
[[nodiscard]] constexpr double egm2008_undulation_m(double lat_deg, double lon_deg) noexcept {
  constexpr double kLatMin = 40.75;
  constexpr double kLatMax = 41.15;
  constexpr double kLonMin = 29.00;
  constexpr double kLonMax = 29.45;
  constexpr double kGrid[4][4] = {
      {36.68, 36.54, 36.40, 36.26},  // 40.7500 N
      {36.87, 36.73, 36.59, 36.45},  // 40.8833 N
      {37.06, 36.92, 36.78, 36.64},  // 41.0167 N
      {37.25, 37.11, 36.97, 36.83},  // 41.1500 N
  };

  const double u_lat = clamp_d_raw((lat_deg - kLatMin) / ((kLatMax - kLatMin) / 3.0), 0.0, 2.999999999);
  const double u_lon = clamp_d_raw((lon_deg - kLonMin) / ((kLonMax - kLonMin) / 3.0), 0.0, 2.999999999);
  const std::size_t i = static_cast<std::size_t>(u_lat);
  const std::size_t j = static_cast<std::size_t>(u_lon);
  const double fl = u_lat - static_cast<double>(i);
  const double fn = u_lon - static_cast<double>(j);

  const double n00 = kGrid[i][j];
  const double n01 = kGrid[i][j + 1];
  const double n10 = kGrid[i + 1][j];
  const double n11 = kGrid[i + 1][j + 1];
  const double n0 = n00 * (1.0 - fn) + n01 * fn;
  const double n1 = n10 * (1.0 - fn) + n11 * fn;
  return n0 * (1.0 - fl) + n1 * fl;
}

[[nodiscard]] constexpr double amsl_to_hae(double h_amsl_m, double lat_deg, double lon_deg) noexcept {
  return h_amsl_m + egm2008_undulation_m(lat_deg, lon_deg);
}

[[nodiscard]] constexpr double hae_to_amsl(double h_hae_m, double lat_deg, double lon_deg) noexcept {
  return h_hae_m - egm2008_undulation_m(lat_deg, lon_deg);
}

[[nodiscard]] Vec3Ecef geodetic_to_ecef(const GeodeticWgs84& geo) noexcept;
[[nodiscard]] GeodeticWgs84 ecef_to_geodetic(const Vec3Ecef& ecef) noexcept;

[[nodiscard]] Vec3Enu geodetic_to_local_enu(
    const GeodeticWgs84& geo,
    const GeodeticWgs84& datum = kKartalDatum) noexcept;

[[nodiscard]] GeodeticWgs84 local_enu_to_geodetic(
    const Vec3Enu& enu,
    const GeodeticWgs84& datum = kKartalDatum) noexcept;

}  // namespace vertirta::core
