// SPDX-License-Identifier: Apache-2.0
#include "vertirta/wind_acoustics/istanbul_receptors.hpp"

#include "vertirta/core/geodesy.hpp"
#include "vertirta/wind_acoustics/vrta_wind_format.hpp"

namespace vertirta::wind_acoustics {

namespace {

struct RawReceptorSeed {
  const char* name;
  ReceptorCategory category;
  double lat_deg;
  double lon_deg;
  double alt_amsl_m;
  double g_factor;
};

// 13 landmark hospitals & coastal schools strictly west of 29.1895 E; remaining 51 receptors
// populate the Dragos-Cevizli-Maltepe-west Kartal coastal residential & park grid (< 29.1888 E).
constexpr RawReceptorSeed kLandmarkSeeds[13] = {
    {"Kartal_Dr_Lutfi_Kirdar_Sehir_Hastanesi", ReceptorCategory::kHospital,
     kKartalHospitalLatDeg, kKartalHospitalLonDeg, kKartalHospitalAltAmslM, 0.2},
    {"Kosuyolu_Yuksek_Ihtisas_Hastanesi", ReceptorCategory::kHospital,
     40.92280, 29.17650, 55.0, 0.2},
    {"Sureyyapasa_Gogus_Hastaliklari_Hastanesi", ReceptorCategory::kHospital,
     40.93520, 29.15200, 78.0, 0.25},
    {"Maltepe_Devlet_Hastanesi_Coastal", ReceptorCategory::kHospital,
     40.92750, 29.14150, 32.0, 0.2},
    {"Dragos_Coastal_Primary_School", ReceptorCategory::kSchool,
     40.91420, 29.14620, 18.0, 0.2},
    {"Cevizli_Anatolian_High_School", ReceptorCategory::kSchool,
     40.90540, 29.16050, 22.0, 0.2},
    {"Kartal_West_Technical_School", ReceptorCategory::kSchool,
     40.89250, 29.18120, 16.0, 0.2},
    {"Maltepe_Sahil_Secondary_School", ReceptorCategory::kSchool,
     40.92340, 29.13180, 14.0, 0.2},
    {"Orhangazi_Coastal_Campus", ReceptorCategory::kSchool,
     40.93480, 29.11850, 12.0, 0.2},
    {"Dragos_Hill_Academy", ReceptorCategory::kSchool,
     40.91880, 29.14890, 48.0, 0.25},
    {"Atalar_Waterfront_School", ReceptorCategory::kSchool,
     40.89810, 29.17140, 15.0, 0.2},
    {"Maltepe_Orhangazi_Park_Core", ReceptorCategory::kCoastalPark,
     40.94016, 29.10790, 5.0, 0.35},
    {"Dragos_Sahil_Park_Core", ReceptorCategory::kCoastalPark,
     40.91686, 29.13822, 6.0, 0.35},
};

[[nodiscard]] IstanbulReceptor make_receptor(
    uint16_t id,
    const char* name,
    ReceptorCategory cat,
    double lat_deg,
    double lon_deg,
    double alt_amsl_m,
    double g_factor) noexcept {
  IstanbulReceptor r{};
  r.id = id;
  r.name = name;
  r.category = cat;
  r.lat_deg = lat_deg;
  r.lon_deg = lon_deg;
  r.alt_amsl_m = alt_amsl_m;
  r.ground_factor_g = g_factor;

  const double h_hae = core::amsl_to_hae(alt_amsl_m, lat_deg, lon_deg);
  r.pos_enu = core::geodetic_to_local_enu({lat_deg, lon_deg, h_hae}, core::kKartalDatum);

  // Turkiye 2022 Environmental Noise Regulation / WHO exterior thresholds [dBA]
  switch (cat) {
    case ReceptorCategory::kHospital:
      r.thresh_day_dba = 55.0;
      r.thresh_evening_dba = 50.0;
      r.thresh_night_dba = 45.0;
      r.soft_weight = 18.0;
      break;
    case ReceptorCategory::kSchool:
      r.thresh_day_dba = 60.0;
      r.thresh_evening_dba = 55.0;
      r.thresh_night_dba = 45.0;
      r.soft_weight = 3.5;
      break;
    case ReceptorCategory::kResidential:
      r.thresh_day_dba = 65.0;
      r.thresh_evening_dba = 60.0;
      r.thresh_night_dba = 55.0;
      r.soft_weight = 0.6;
      break;
    case ReceptorCategory::kCoastalPark:
      r.thresh_day_dba = 65.0;
      r.thresh_evening_dba = 60.0;
      r.thresh_night_dba = 55.0;
      r.soft_weight = 0.3;
      break;
  }
  return r;
}

}  // namespace

std::array<IstanbulReceptor, kNumIstanbulReceptors> build_istanbul_64_receptors() noexcept {
  std::array<IstanbulReceptor, kNumIstanbulReceptors> out{};

  for (std::size_t i = 0; i < 13u; ++i) {
    const auto& s = kLandmarkSeeds[i];
    out[i] = make_receptor(
        static_cast<uint16_t>(i), s.name, s.category, s.lat_deg, s.lon_deg, s.alt_amsl_m,
        s.g_factor);
  }

  // Remaining 51 receptors: 17 along-shore coastal stations x 3 inland tiers spanning
  // west Kartal (40.887 N, 29.185 E) -> Cevizli -> Dragos -> Maltepe (40.941 N, 29.110 E).
  // Strictly lon_deg <= 29.1885 < 29.1895 E (zero LTFJ CTR overlap).
  std::size_t idx = 13u;
  for (int seg = 0; seg < 17; ++seg) {
    const double t = static_cast<double>(seg) / 16.0;
    const double base_lat = 40.8875 + t * (40.9405 - 40.8875);
    const double base_lon = 29.1845 + t * (29.1110 - 29.1845);

    for (int tier = 0; tier < 3; ++tier) {
      const double lat = base_lat + static_cast<double>(tier) * 0.0018;
      const double lon = base_lon + static_cast<double>(tier) * 0.0016;
      const double clamped_lon = (lon < 29.1885) ? lon : 29.1885;
      const double alt_amsl = 8.0 + static_cast<double>(tier) * 14.0;
      const ReceptorCategory cat =
          (tier == 0 && (seg % 3 == 0)) ? ReceptorCategory::kCoastalPark
                                        : ReceptorCategory::kResidential;
      const double g_val = (cat == ReceptorCategory::kCoastalPark) ? 0.30 : 0.20;
      out[idx] = make_receptor(
          static_cast<uint16_t>(idx),
          (cat == ReceptorCategory::kCoastalPark) ? "Coastal_Park_Frontage"
                                                  : "Coastal_Residential_Frontage",
          cat, lat, clamped_lon, alt_amsl, g_val);
      ++idx;
    }
  }

  return out;
}

std::array<GroundReceptor, kNumIstanbulReceptors> extract_ground_receptors(
    const std::array<IstanbulReceptor, kNumIstanbulReceptors>& catalog) noexcept {
  std::array<GroundReceptor, kNumIstanbulReceptors> out{};
  for (std::size_t i = 0; i < kNumIstanbulReceptors; ++i) {
    out[i] = GroundReceptor{catalog[i].pos_enu, catalog[i].ground_factor_g};
  }
  return out;
}

rta::HazardPoint build_kartal_hospital_helipad_hazard() noexcept {
  const double h_hae = core::amsl_to_hae(
      kKartalHospitalAltAmslM, kKartalHospitalLatDeg, kKartalHospitalLonDeg);
  const core::Vec3Enu enu = core::geodetic_to_local_enu(
      {kKartalHospitalLatDeg, kKartalHospitalLonDeg, h_hae}, core::kKartalDatum);

  rta::HazardPoint hz{};
  hz.pos_ned = core::enu_to_ned(enu);
  hz.vel_ned = {0.0, 0.0, 0.0};
  hz.accel_ned = {0.0, 0.0, 0.0};
  hz.radius_h_m = kKartalHospitalHelipadRadiusHM;
  hz.radius_z_m = kKartalHospitalHelipadRadiusZM;
  hz.gamma = 1.4;
  hz.kind = 3u;  // Hard CTR / Helipad airspace exclusion cylinder
  return hz;
}

rta::HazardPoint build_ltfj_ctr_wall_hazard(const core::Vec3Ned& aircraft_pos_ned) noexcept {
  const core::Vec3Enu ac_enu = core::ned_to_enu(aircraft_pos_ned);
  const core::GeodeticWgs84 ac_geo = core::local_enu_to_geodetic(ac_enu, core::kKartalDatum);
  const double wall_lat = ac_geo.lat_deg;
  const double wall_lon = kLtfjMaxAllowedLonDeg;
  const core::Vec3Enu wall_enu =
      core::geodetic_to_local_enu({wall_lat, wall_lon, ac_geo.h_hae_m}, core::kKartalDatum);

  rta::HazardPoint hz{};
  hz.pos_ned = core::enu_to_ned(wall_enu);
  hz.vel_ned = {0.0, 0.0, 0.0};
  hz.accel_ned = {0.0, 0.0, 0.0};
  hz.radius_h_m = 25.0;
  hz.radius_z_m = 200.0;
  hz.gamma = 1.5;
  hz.kind = 3u;  // Hard LTFJ CTR western wall
  return hz;
}

}  // namespace vertirta::wind_acoustics
