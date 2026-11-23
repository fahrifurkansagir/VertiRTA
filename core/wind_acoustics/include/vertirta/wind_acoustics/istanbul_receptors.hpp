// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/wind_acoustics/aeroacoustics.hpp"

namespace vertirta::wind_acoustics {

inline constexpr std::size_t kNumIstanbulReceptors = 64u;

// Kartal Dr. Lutfi Kirdar Sehir Hastanesi active helipad (arc-second aligned with Plan 5B).
inline constexpr double kKartalHospitalLatDeg = 40.91639;
inline constexpr double kKartalHospitalLonDeg = 29.17417;
inline constexpr double kKartalHospitalAltAmslM = 42.0;
inline constexpr double kKartalHospitalHelipadRadiusHM = 160.0;  // >= 150 m Tier-2 hard cylinder
inline constexpr double kKartalHospitalHelipadRadiusZM = 125.0;  // >= 120 m Tier-2 hard cylinder

enum class ReceptorCategory : uint8_t {
  kHospital = 0,
  kSchool = 1,
  kResidential = 2,
  kCoastalPark = 3,
};

enum class TimeOfDayPeriod : uint8_t {
  kDay = 0,
  kEvening = 1,
  kNight = 2,
};

struct IstanbulReceptor {
  uint16_t id{0};
  const char* name{""};
  ReceptorCategory category{ReceptorCategory::kResidential};
  double lat_deg{40.8860};
  double lon_deg{29.1880};
  double alt_amsl_m{12.0};
  core::Vec3Enu pos_enu{};
  double ground_factor_g{0.2};
  double thresh_day_dba{65.0};
  double thresh_evening_dba{60.0};
  double thresh_night_dba{55.0};
  double soft_weight{1.0};

  [[nodiscard]] double threshold_dba(TimeOfDayPeriod period) const noexcept {
    switch (period) {
      case TimeOfDayPeriod::kDay:
        return thresh_day_dba;
      case TimeOfDayPeriod::kEvening:
        return thresh_evening_dba;
      case TimeOfDayPeriod::kNight:
        return thresh_night_dba;
    }
    return thresh_night_dba;
  }
};

// Build canonical 64-receptor Anatolian coastal catalog strictly west of 29.1895 deg E,
// anchored to kKartalDatum (40.8860 N, 29.1880 E, 12.0 m AMSL, EGM2008 N = +36.7 m).
[[nodiscard]] std::array<IstanbulReceptor, kNumIstanbulReceptors>
build_istanbul_64_receptors() noexcept;

// Extract lightweight GroundReceptor view for AeroacousticEngine::propagate_to_receptors.
[[nodiscard]] std::array<GroundReceptor, kNumIstanbulReceptors> extract_ground_receptors(
    const std::array<IstanbulReceptor, kNumIstanbulReceptors>& catalog) noexcept;

// Hard Tier-2 airspace separation cylinder around Kartal Dr. Lutfi Kirdar Hospital active helipad.
// Strictly separate from Tier-1 soft noise cost and fed into frozen CbvfSafetyFilter::step_100hz.
[[nodiscard]] rta::HazardPoint build_kartal_hospital_helipad_hazard() noexcept;

// Nearest hard Tier-2 LTFJ CTR western wall boundary (lon = 29.1895 deg E) for current aircraft pos.
[[nodiscard]] rta::HazardPoint build_ltfj_ctr_wall_hazard(
    const core::Vec3Ned& aircraft_pos_ned) noexcept;

}  // namespace vertirta::wind_acoustics
