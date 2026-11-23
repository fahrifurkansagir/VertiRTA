// SPDX-License-Identifier: Apache-2.0
#include "vertirta/uspace/corridor_network.hpp"

#include <cmath>
#include <limits>

namespace vertirta::uspace {

namespace {

[[nodiscard]] core::Vec3Enu wgs84_amsl_to_kartal_enu(
    double lat_deg,
    double lon_deg,
    double alt_amsl_m) noexcept {
  const double h_hae = core::amsl_to_hae(alt_amsl_m, lat_deg, lon_deg);
  return core::geodetic_to_local_enu({lat_deg, lon_deg, h_hae}, core::kKartalDatum);
}

}  // namespace

CorridorNetwork make_anatolian_corridor_network() noexcept {
  CorridorNetwork net{};
  net.datum_lat_deg = core::kKartalDatum.lat_deg;  // 40.8860 N
  net.datum_lon_deg = core::kKartalDatum.lon_deg;  // 29.1880 E
  net.datum_alt_amsl_m = 12.0;
  net.egm2008_n0_m = 36.7;
  net.eastern_geofence_lon_deg = 29.1895;  // Strictly west of LTFJ NW J2 pinch (29.19212 E)
  net.cruise_ceiling_agl_m = 120.0;
  net.odd_ceiling_agl_m = 150.0;
  net.lane_se_agl_m = 70.0;
  net.lane_nw_agl_m = 105.0;

  net.vertiports[0] = VertiportSpec{
      .code = "VPT-MLT",
      .name = "Maltepe_Orhangazi_Park",
      .lat_deg = 40.94016,
      .lon_deg = 29.10790,
      .alt_amsl_m = 5.0,
      .pos_enu = wgs84_amsl_to_kartal_enu(40.94016, 29.10790, 5.0),
      .operational_fatos = 2U,
      .has_dedicated_fato_emg = true,
      .chilled_charger_stands = 4U,
      .tlof_size_m = 8.0,
      .fato_size_m = 12.0,
      .safety_area_m = 24.2,
      .is_hub = true,
  };

  net.vertiports[1] = VertiportSpec{
      .code = "VPT-DRG",
      .name = "Dragos_Sahil_Park",
      .lat_deg = 40.91686,
      .lon_deg = 29.13822,
      .alt_amsl_m = 6.0,
      .pos_enu = wgs84_amsl_to_kartal_enu(40.91686, 29.13822, 6.0),
      .operational_fatos = 1U,
      .has_dedicated_fato_emg = true,
      .chilled_charger_stands = 4U,
      .tlof_size_m = 8.0,
      .fato_size_m = 12.0,
      .safety_area_m = 24.2,
      .is_hub = false,
  };

  net.vertiports[2] = VertiportSpec{
      .code = "VPT-CVZ",
      .name = "Cevizli_Coastal_Hub",
      .lat_deg = 40.90380,
      .lon_deg = 29.15450,
      .alt_amsl_m = 8.0,
      .pos_enu = wgs84_amsl_to_kartal_enu(40.90380, 29.15450, 8.0),
      .operational_fatos = 1U,
      .has_dedicated_fato_emg = true,
      .chilled_charger_stands = 4U,
      .tlof_size_m = 8.0,
      .fato_size_m = 12.0,
      .safety_area_m = 24.2,
      .is_hub = false,
  };

  net.vertiports[3] = VertiportSpec{
      .code = "VPT-KRT",
      .name = "Kartal_Coastal_Vertiport",
      .lat_deg = 40.88600,
      .lon_deg = 29.18800,
      .alt_amsl_m = 12.0,
      .pos_enu = wgs84_amsl_to_kartal_enu(40.88600, 29.18800, 12.0),
      .operational_fatos = 2U,
      .has_dedicated_fato_emg = true,
      .chilled_charger_stands = 4U,
      .tlof_size_m = 8.0,
      .fato_size_m = 12.0,
      .safety_area_m = 24.2,
      .is_hub = true,
  };

  // Active hospital helipad arc-second aligned with Plan 5A (40.91639 N, 29.17417 E)
  net.hospital_helipad = HospitalHelipadZone{
      .name = "Kartal_Dr_Lutfi_Kirdar_Hospital_Helipad",
      .lat_deg = 40.91639,
      .lon_deg = 29.17417,
      .alt_amsl_m = 45.0,
      .radius_m = 150.0,
      .ceiling_agl_m = 150.0,
      .pos_enu = wgs84_amsl_to_kartal_enu(40.91639, 29.17417, 45.0),
  };

  return net;
}

double compute_siso_wake_spacing_s(
    double crosswind_mps,
    double total_wind_mps,
    bool ltfj_heavy_arrival) noexcept {
  if (ltfj_heavy_arrival) {
    return 150.0;  // ICAO heavy wake buffer [120..180 s]
  }
  if (std::abs(total_wind_mps) >= 12.0) {
    return 90.0;  // Strong coastal Lodos/Poyraz gusts
  }
  if (std::abs(crosswind_mps) < 1.5) {
    return 75.0;  // Calm / stagnant wake persistence
  }
  return 60.0;  // Moderate crosswind flushing >= 1.5 m/s
}

bool is_point_inside_corridor_odd(
    const CorridorNetwork& net,
    double lat_deg,
    double lon_deg,
    double alt_agl_m) noexcept {
  if (lon_deg >= net.eastern_geofence_lon_deg || lon_deg < 29.0950) {
    return false;
  }
  if (lat_deg < 40.8750 || lat_deg > 40.9450) {
    return false;
  }
  if (alt_agl_m < 0.0 || alt_agl_m > net.cruise_ceiling_agl_m) {
    return false;
  }
  const core::Vec3Enu p_enu = wgs84_amsl_to_kartal_enu(lat_deg, lon_deg, net.datum_alt_amsl_m + alt_agl_m);
  const double dx = p_enu.x - net.hospital_helipad.pos_enu.x;
  const double dy = p_enu.y - net.hospital_helipad.pos_enu.y;
  const double dist_h = std::hypot(dx, dy);
  if (dist_h < net.hospital_helipad.radius_m && alt_agl_m <= net.hospital_helipad.ceiling_agl_m) {
    return false;
  }
  return true;
}

uint32_t nearest_fato_emg_vertiport(
    const CorridorNetwork& net,
    const core::Vec3Enu& pos_enu) noexcept {
  uint32_t best_idx = 0U;
  double best_d2 = std::numeric_limits<double>::max();
  for (uint32_t i = 0U; i < static_cast<uint32_t>(net.vertiports.size()); ++i) {
    const auto& vp = net.vertiports[i];
    if (!vp.has_dedicated_fato_emg) {
      continue;
    }
    const double dx = pos_enu.x - vp.pos_enu.x;
    const double dy = pos_enu.y - vp.pos_enu.y;
    const double d2 = dx * dx + dy * dy;
    if (d2 < best_d2) {
      best_d2 = d2;
      best_idx = i;
    }
  }
  return best_idx;
}

}  // namespace vertirta::uspace
