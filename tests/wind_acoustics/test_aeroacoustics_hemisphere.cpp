// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_all.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <vector>

#include "vertirta/wind_acoustics/aeroacoustics.hpp"
#include "vertirta/wind_acoustics/istanbul_receptors.hpp"

using namespace vertirta::wind_acoustics;

TEST_CASE(
    "AeroacousticEngine matches physical X8 D=1.50m vs Lift+Cruise ordering, sideline dipole, ISO 9613-1/2, and <1ms budget") {
  // 1) Canonical X8-240 constants strictly locked to config/vehicle_x8_240.yaml (zero DR-09 1.65 m drift)
  REQUIRE(kX8RotorDiameterM == Catch::Approx(1.50).margin(1e-9));
  REQUIRE(kX8CoaxZOverD == Catch::Approx(0.233333).margin(1e-4));
  REQUIRE(kX8HoverRpm == Catch::Approx(1450.0).margin(1.0));

  // 2) ISO 9613-1:1993 atmospheric absorption at 20 C, 70% RH, 101.325 kPa within +-5% of ISO table
  const double a_63 = compute_iso9613_1_alpha_db_per_km(63.0, 20.0, 70.0, 101.325);
  const double a_1k = compute_iso9613_1_alpha_db_per_km(1000.0, 20.0, 70.0, 101.325);
  const double a_4k = compute_iso9613_1_alpha_db_per_km(4000.0, 20.0, 70.0, 101.325);
  const double a_8k = compute_iso9613_1_alpha_db_per_km(8000.0, 20.0, 70.0, 101.325);
  REQUIRE(a_63 == Catch::Approx(0.09).epsilon(0.10));
  REQUIRE(a_1k == Catch::Approx(4.98).epsilon(0.05));
  REQUIRE(a_4k == Catch::Approx(23.1).epsilon(0.05));
  REQUIRE(a_8k == Catch::Approx(77.6).epsilon(0.05));

  // IEC 61672-1 A-weighting sanity: 1 kHz == 0 dB, 100 Hz ~ -19.1 dB
  REQUIRE(compute_iec61672_a_weight_db(1000.0) == Catch::Approx(0.0).margin(0.05));
  REQUIRE(compute_iec61672_a_weight_db(100.0) == Catch::Approx(-19.14).margin(0.2));

  AeroacousticEngine engine(20.0, 70.0, 101.325);
  const vertirta::core::Vec3Enu ac_pos{0.0, 0.0, 120.0};  // 120 m AGL
  const GroundReceptor overhead{{0.0, 0.0, 0.0}, 0.2};
  const GroundReceptor sideline_150m{{0.0, 150.0, 0.0}, 0.2};  // 150 m lateral for East heading

  // 3) Overhead 120 m AGL physical ordering:
  //    LC_Cruise_38mps < LC_Hover_8p1 < X8_Hover_zD0233 < X8_Tight_zD012
  AcousticFlightState st_lc_cruise{};
  st_lc_cruise.airframe = AcousticAirframeKind::kLiftCruise_8p1;
  st_lc_cruise.airspeed_mps = 38.0;
  st_lc_cruise.heading_enu_rad = 0.0;  // Flying East
  st_lc_cruise.lift_rotors_locked = true;
  st_lc_cruise.lift_rpm = 0.0;
  st_lc_cruise.lift_thrust_total_n = 0.0;
  st_lc_cruise.pusher_rpm = 1850.0;
  st_lc_cruise.pusher_thrust_n = 185.0;  // L/D ~ 13 cruise thrust

  AcousticFlightState st_lc_hover{};
  st_lc_hover.airframe = AcousticAirframeKind::kLiftCruise_8p1;
  st_lc_hover.airspeed_mps = 0.0;
  st_lc_hover.lift_rpm = 1600.0;
  st_lc_hover.lift_thrust_total_n = 2353.596;

  AcousticFlightState st_x8_hover{};
  st_x8_hover.airframe = AcousticAirframeKind::kCoaxialX8_240;
  st_x8_hover.airspeed_mps = 0.0;
  st_x8_hover.lift_rpm = kX8HoverRpm;
  st_x8_hover.lift_thrust_total_n = 2353.596;
  st_x8_hover.coax_z_over_d = kX8CoaxZOverD;

  AcousticFlightState st_x8_tight = st_x8_hover;
  st_x8_tight.coax_z_over_d = 0.12;

  AcousticHemisphere180 h_lc_c{}, h_lc_h{}, h_x8_h{}, h_x8_t{};
  engine.synthesize_hemisphere(st_lc_cruise, h_lc_c);
  engine.synthesize_hemisphere(st_lc_hover, h_lc_h);
  engine.synthesize_hemisphere(st_x8_hover, h_x8_h);
  engine.synthesize_hemisphere(st_x8_tight, h_x8_t);

  const double dba_lc_cruise = engine.propagate_single_dba(h_lc_c, ac_pos, {38.0, 0.0, 0.0}, overhead);
  const double dba_lc_hover = engine.propagate_single_dba(h_lc_h, ac_pos, {0.0, 0.0, 0.0}, overhead);
  const double dba_x8_hover = engine.propagate_single_dba(h_x8_h, ac_pos, {0.0, 0.0, 0.0}, overhead);
  const double dba_x8_tight = engine.propagate_single_dba(h_x8_t, ac_pos, {0.0, 0.0, 0.0}, overhead);

  INFO("120m AGL overhead dBA: LC_Cruise=" << dba_lc_cruise << " < LC_Hover=" << dba_lc_hover
       << " < X8_Hover(z/D=0.233)=" << dba_x8_hover << " < X8_Tight(z/D=0.12)=" << dba_x8_tight);

  REQUIRE(dba_x8_hover >= 68.0);
  REQUIRE(dba_x8_hover <= 82.0);
  REQUIRE(dba_lc_cruise < dba_lc_hover);
  REQUIRE(dba_lc_hover < dba_x8_hover);
  REQUIRE(dba_x8_hover < dba_x8_tight);
  REQUIRE((dba_x8_hover - dba_lc_cruise) >= 12.0);

  // 4) Pusher 3D directivity: LC_Cruise_38mps at 120 m AGL has higher dBA at 150 m sideline than overhead
  const double dba_lc_sideline =
      engine.propagate_single_dba(h_lc_c, ac_pos, {38.0, 0.0, 0.0}, sideline_150m);
  INFO("LC_Cruise 120m AGL: overhead=" << dba_lc_cruise << " dBA vs 150m sideline=" << dba_lc_sideline << " dBA");
  REQUIRE(dba_lc_sideline > dba_lc_cruise);

  // 5) Shallow-descent coaxial BVI trigger (-6 deg @ 10 m/s) increases SPL over pure vertical hover
  AcousticFlightState st_x8_bvi = st_x8_hover;
  st_x8_bvi.airspeed_mps = 10.0;
  st_x8_bvi.flight_path_angle_deg = -6.0;
  AcousticHemisphere180 h_x8_bvi{};
  engine.synthesize_hemisphere(st_x8_bvi, h_x8_bvi);
  const double dba_x8_bvi = engine.propagate_single_dba(h_x8_bvi, ac_pos, {9.94, 0.0, -1.045}, overhead);
  INFO("X8 shallow-descent BVI (-6 deg @ 10 m/s)=" << dba_x8_bvi << " dBA vs hover=" << dba_x8_hover << " dBA");
  REQUIRE(dba_x8_bvi > dba_x8_hover + 1.5);

  // 6) End-to-end 180-node hemisphere synthesis + 64-receptor propagation latency p99 < 1.0 ms (Release)
  const auto catalog = build_istanbul_64_receptors();
  const auto g_receptors = extract_ground_receptors(catalog);
  std::array<double, kNumIstanbulReceptors> out_dba{};
  AcousticHemisphere180 bench_hemi{};

  constexpr int kRuns = 1000;
  std::vector<double> run_ms(kRuns);
  double sink = 0.0;
  for (int i = 0; i < kRuns; ++i) {
    st_x8_bvi.heading_enu_rad = 0.005 * static_cast<double>(i);
    const auto t0 = std::chrono::steady_clock::now();
    engine.synthesize_hemisphere(st_x8_bvi, bench_hemi);
    engine.propagate_to_receptors(bench_hemi, {-1200.0, 2200.0, 120.0}, {10.0, 5.0, -1.0}, g_receptors, out_dba);
    const auto t1 = std::chrono::steady_clock::now();
    run_ms[static_cast<std::size_t>(i)] =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) * 1e-6;
    sink += out_dba[0];
  }
  REQUIRE(sink > 0.0);
  std::sort(run_ms.begin(), run_ms.end());
  const double p50_ms = run_ms[kRuns / 2];
  const double p99_ms = run_ms[kRuns * 99 / 100];
  INFO("180-node + 64-receptor AeroacousticEngine latency: p50=" << p50_ms << " ms, p99=" << p99_ms << " ms");
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_UNDEFINED__)
  REQUIRE(p99_ms < 1.0);
#else
  REQUIRE(p99_ms < 10.0);
#endif
}
