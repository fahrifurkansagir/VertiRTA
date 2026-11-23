// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/uspace/corridor_network.hpp"
#include "vertirta/uspace/thermal_turnaround.hpp"
#include "vertirta/uspace/vertiport_queue.hpp"

using namespace vertirta::uspace;

TEST_CASE("Corridor network enforces frozen Kartal datum, <29.1895 E J2 margin, and Hospital helipad") {
  const auto net = make_anatolian_corridor_network();
  REQUIRE(net.datum_lat_deg == 40.8860);
  REQUIRE(net.datum_lon_deg == 29.1880);
  REQUIRE_THAT(net.egm2008_n0_m, Catch::Matchers::WithinAbs(36.7, 1e-6));
  REQUIRE(net.eastern_geofence_lon_deg == 29.1895);
  REQUIRE(net.cruise_ceiling_agl_m == 120.0);
  REQUIRE(net.vertiports.size() == 4U);

  for (const auto& vp : net.vertiports) {
    REQUIRE(vp.lon_deg < net.eastern_geofence_lon_deg);  // Never slices Kartal or enters LTFJ NW J2
    REQUIRE(vp.tlof_size_m >= 8.0);
    REQUIRE(vp.fato_size_m >= 12.0);
    REQUIRE(vp.safety_area_m >= 24.2);
    REQUIRE(vp.has_dedicated_fato_emg);
    REQUIRE(vp.chilled_charger_stands == 4U);
  }
  REQUIRE(net.hospital_helipad.radius_m >= 150.0);
  REQUIRE(net.hospital_helipad.lon_deg < net.eastern_geofence_lon_deg);
  REQUIRE_THAT(net.hospital_helipad.lat_deg, Catch::Matchers::WithinAbs(40.91639, 1e-5));
  REQUIRE_THAT(net.hospital_helipad.lon_deg, Catch::Matchers::WithinAbs(29.17417, 1e-5));

  // Verify ODD checks reject LTFJ breach (29.2057 E ISTMarina) and Hospital helipad cylinder
  REQUIRE(is_point_inside_corridor_odd(net, 40.9038, 29.1545, 70.0));
  REQUIRE_FALSE(is_point_inside_corridor_odd(net, 40.8846, 29.2057, 70.0));
  REQUIRE_FALSE(is_point_inside_corridor_odd(net, 40.91639, 29.17417, 70.0));
  REQUIRE_FALSE(is_point_inside_corridor_odd(net, 40.9038, 29.1545, 130.0));

  REQUIRE(compute_siso_wake_spacing_s(2.0, 5.0, false) == 60.0);
  REQUIRE(compute_siso_wake_spacing_s(0.5, 2.0, false) == 75.0);
  REQUIRE(compute_siso_wake_spacing_s(4.0, 18.0, false) == 90.0);
  REQUIRE(compute_siso_wake_spacing_s(2.0, 5.0, true) >= 120.0);
}

TEST_CASE("68 kg 112S12P pack with 15 C active chiller prevents 35 C summer thermal ratchet and stabilizes N=20 fleet") {
  ThermalPackParams pack{};
  REQUIRE_THAT(pack.pack_mass_kg, Catch::Matchers::WithinAbs(68.0, 1e-3));
  REQUIRE_THAT(pack.c_th_j_per_k, Catch::Matchers::WithinAbs(70720.0, 200.0));
  REQUIRE(pack.dispatch_reserve_soc_pct == 25.0);
  REQUIRE(pack.emergency_floor_soc_pct == 20.0);

  const auto chilled = simulate_multicycle_turnaround(
      pack, CoolingMode::ACTIVE_LIQUID_CHILLER_15C, /*ambient_c=*/35.0, /*eol=*/true, /*cycles=*/5U);
  const auto air_only = simulate_multicycle_turnaround(
      pack, CoolingMode::FORCED_AIR_ONLY, /*ambient_c=*/35.0, /*eol=*/true, /*cycles=*/5U);

  REQUIRE(chilled.peak_charge_temp_c < 41.0);
  REQUIRE(chilled.total_turnaround_min < 13.0);
  REQUIRE(air_only.peak_charge_temp_c >= 44.0);
  REQUIRE(air_only.total_turnaround_min > chilled.total_turnaround_min + 4.0);

  const auto q20_chilled = evaluate_tandem_fleet_queue(
      /*n_aircraft=*/20U, /*op_fatos_hub=*/2U, /*chilled_stands=*/4U, /*wake_s=*/75.0, chilled.total_turnaround_min);
  const auto q20_air = evaluate_tandem_fleet_queue(
      /*n_aircraft=*/20U, /*op_fatos_hub=*/1U, /*chilled_stands=*/3U, /*wake_s=*/75.0, air_only.total_turnaround_min);
  CAPTURE(q20_chilled.mean_airborne_hold_min,
          q20_chilled.throughput_flt_per_hr,
          q20_chilled.hub_fato_utilization,
          q20_chilled.hub_stand_utilization,
          q20_chilled.bbs_sno_blocking_prob);
  REQUIRE(q20_chilled.stable);
  REQUIRE(q20_chilled.mean_airborne_hold_min <= 2.0);
  REQUIRE(q20_chilled.throughput_flt_per_hr >= 55.0);
  REQUIRE_FALSE(q20_air.stable);
}
