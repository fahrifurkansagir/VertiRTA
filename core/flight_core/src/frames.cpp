#include "vertirta/core/geodesy.hpp"

#include <cmath>

namespace vertirta::core {

Vec3Ecef geodetic_to_ecef(const GeodeticWgs84& geo) noexcept {
  const double lat = geo.lat_deg * kDegToRad;
  const double lon = geo.lon_deg * kDegToRad;
  const double slat = std::sin(lat);
  const double clat = std::cos(lat);
  const double slon = std::sin(lon);
  const double clon = std::cos(lon);
  const double nu = kWgs84A / std::sqrt(1.0 - kWgs84E2 * slat * slat);
  return {
      (nu + geo.h_hae_m) * clat * clon,
      (nu + geo.h_hae_m) * clat * slon,
      (nu * (1.0 - kWgs84E2) + geo.h_hae_m) * slat,
  };
}

GeodeticWgs84 ecef_to_geodetic(const Vec3Ecef& ecef) noexcept {
  const double b = kWgs84A * (1.0 - kWgs84F);
  const double ep2 = (kWgs84A * kWgs84A - b * b) / (b * b);
  const double p = std::hypot(ecef.x, ecef.y);
  const double lon = std::atan2(ecef.y, ecef.x);

  if (p < 1e-9) {
    const double lat_deg = (ecef.z >= 0.0) ? 90.0 : -90.0;
    return {lat_deg, 0.0, std::abs(ecef.z) - b};
  }

  const double theta = std::atan2(ecef.z * kWgs84A, p * b);
  const double st = std::sin(theta);
  const double ct = std::cos(theta);
  double lat = std::atan2(
      ecef.z + ep2 * b * st * st * st,
      p - kWgs84E2 * kWgs84A * ct * ct * ct);

  // Two Bowring refinement iterations for sub-millimeter closure
  for (int iter = 0; iter < 2; ++iter) {
    const double slat = std::sin(lat);
    const double nu = kWgs84A / std::sqrt(1.0 - kWgs84E2 * slat * slat);
    lat = std::atan2(ecef.z + kWgs84E2 * nu * slat, p);
  }

  const double slat = std::sin(lat);
  const double clat = std::cos(lat);
  const double nu = kWgs84A / std::sqrt(1.0 - kWgs84E2 * slat * slat);
  const double h = (p / clat) - nu;
  return {lat * kRadToDeg, lon * kRadToDeg, h};
}

Vec3Enu geodetic_to_local_enu(const GeodeticWgs84& geo, const GeodeticWgs84& datum) noexcept {
  const Vec3Ecef p = geodetic_to_ecef(geo);
  const Vec3Ecef p0 = geodetic_to_ecef(datum);
  const double dx = p.x - p0.x;
  const double dy = p.y - p0.y;
  const double dz = p.z - p0.z;

  const double lat0 = datum.lat_deg * kDegToRad;
  const double lon0 = datum.lon_deg * kDegToRad;
  const double slat = std::sin(lat0);
  const double clat = std::cos(lat0);
  const double slon = std::sin(lon0);
  const double clon = std::cos(lon0);

  return {
      -slon * dx + clon * dy,
      -slat * clon * dx - slat * slon * dy + clat * dz,
      clat * clon * dx + clat * slon * dy + slat * dz,
  };
}

GeodeticWgs84 local_enu_to_geodetic(const Vec3Enu& enu, const GeodeticWgs84& datum) noexcept {
  const Vec3Ecef p0 = geodetic_to_ecef(datum);
  const double lat0 = datum.lat_deg * kDegToRad;
  const double lon0 = datum.lon_deg * kDegToRad;
  const double slat = std::sin(lat0);
  const double clat = std::cos(lat0);
  const double slon = std::sin(lon0);
  const double clon = std::cos(lon0);

  const Vec3Ecef p{
      p0.x - slon * enu.x - slat * clon * enu.y + clat * clon * enu.z,
      p0.y + clon * enu.x - slat * slon * enu.y + clat * slon * enu.z,
      p0.z + clat * enu.y + slat * enu.z,
  };
  return ecef_to_geodetic(p);
}

}  // namespace vertirta::core
