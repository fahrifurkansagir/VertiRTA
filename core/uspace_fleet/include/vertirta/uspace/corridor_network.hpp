// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"

namespace vertirta::uspace {

struct VertiportSpec {
  std::string_view code{};
  std::string_view name{};
  double lat_deg{0.0};
  double lon_deg{0.0};
  double alt_amsl_m{0.0};
  core::Vec3Enu pos_enu{};
  uint32_t operational_fatos{1U};
  bool has_dedicated_fato_emg{true};
  uint32_t chilled_charger_stands{4U};
  double tlof_size_m{8.0};
  double fato_size_m{12.0};
  double safety_area_m{24.2};
  bool is_hub{false};
};

struct HospitalHelipadZone {
  std::string_view name{};
  double lat_deg{0.0};
  double lon_deg{0.0};
  double alt_amsl_m{0.0};
  double radius_m{150.0};
  double ceiling_agl_m{150.0};
  core::Vec3Enu pos_enu{};
};

struct CorridorNetwork {
  double datum_lat_deg{40.8860};
  double datum_lon_deg{29.1880};
  double datum_alt_amsl_m{12.0};
  double egm2008_n0_m{36.7};
  double eastern_geofence_lon_deg{29.1895};
  double cruise_ceiling_agl_m{120.0};
  double odd_ceiling_agl_m{150.0};
  double lane_se_agl_m{70.0};
  double lane_nw_agl_m{105.0};
  std::array<VertiportSpec, 4U> vertiports{};
  HospitalHelipadZone hospital_helipad{};
};

// Returns the frozen Anatolian 4-vertiport coastal corridor strictly west of LTFJ NW J2 (29.1895 E),
// anchored to canonical Kartal datum (40.8860 N, 29.1880 E, 12.0 m AMSL, EGM2008 N0 = 36.7 m),
// with Kartal Dr. Lutfi Kirdar Sehir Hastanesi active helipad separation cylinder.
[[nodiscard]] CorridorNetwork make_anatolian_corridor_network() noexcept;

// Wind-conditioned SISO FATO wake-separation spacing [s]:
//   - LTFJ heavy arrival overlap: 150 s (120-180 s ICAO heavy-wake buffer)
//   - Strong coastal gusts (>= 12 m/s): 90 s
//   - Calm stagnant crosswind (< 1.5 m/s): 75 s
//   - Moderate crosswind flushing (>= 1.5 m/s, < 12 m/s total): 60 s
[[nodiscard]] double compute_siso_wake_spacing_s(
    double crosswind_mps,
    double total_wind_mps,
    bool ltfj_heavy_arrival) noexcept;

// Checks whether a WGS84 position lies strictly inside the Anatolian corridor ODD
// (west of 29.1895 E LTFJ NW J2 margin, <= 120 m cruise ceiling, outside Hospital helipad cylinder).
[[nodiscard]] bool is_point_inside_corridor_odd(
    const CorridorNetwork& net,
    double lat_deg,
    double lon_deg,
    double alt_agl_m) noexcept;

// Nearest vertiport index [0..3] with dedicated FATO-EMG for emergency diversion.
[[nodiscard]] uint32_t nearest_fato_emg_vertiport(
    const CorridorNetwork& net,
    const core::Vec3Enu& pos_enu) noexcept;

}  // namespace vertirta::uspace
