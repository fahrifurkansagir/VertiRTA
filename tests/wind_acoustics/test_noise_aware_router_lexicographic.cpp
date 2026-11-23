// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_all.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/cel_2d_lut.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"
#include "vertirta/wind_acoustics/istanbul_receptors.hpp"
#include "vertirta/wind_acoustics/noise_aware_router.hpp"
#include "vrta_wind_test_fixture.hpp"

using namespace vertirta::wind_acoustics;
using vertirta::wind_acoustics::testing::ScopedVrtaWindFixture;

namespace {

uint64_t next_rng(uint64_t& s) noexcept {
  s ^= s << 13u;
  s ^= s >> 7u;
  s ^= s << 17u;
  return s;
}

double uniform_range(uint64_t& s, double lo, double hi) noexcept {
  const double u = static_cast<double>(next_rng(s) & 0xffffffffull) / 4294967295.0;
  return lo + u * (hi - lo);
}

}  // namespace

TEST_CASE(
    "64 Istanbul receptors stay <29.1895E, Kartal Hospital helipad is Tier-2 hard barrier, and 500-run stress yields 0 breaches") {
  // 1) Verify all 64 receptors stay strictly west of LTFJ CTR J2 (29.1895 E) and Kartal Hospital is present
  const auto receptors = build_istanbul_64_receptors();
  REQUIRE(receptors.size() == 64u);
  bool found_kartal_hospital = false;
  for (const auto& r : receptors) {
    REQUIRE(r.lon_deg < kLtfjMaxAllowedLonDeg);
    REQUIRE(r.pos_enu.is_finite());
    if (std::strstr(r.name, "Kartal_Dr_Lutfi_Kirdar") != nullptr) {
      found_kartal_hospital = true;
      REQUIRE(r.lat_deg == Catch::Approx(kKartalHospitalLatDeg).margin(1e-6));
      REQUIRE(r.lon_deg == Catch::Approx(kKartalHospitalLonDeg).margin(1e-6));
      REQUIRE(r.category == ReceptorCategory::kHospital);
    }
  }
  REQUIRE(found_kartal_hospital);

  const auto helipad_hz = build_kartal_hospital_helipad_hazard();
  REQUIRE(helipad_hz.kind == 3u);
  REQUIRE(helipad_hz.radius_h_m >= 150.0);
  REQUIRE(helipad_hz.radius_z_m >= 120.0);

  // Load frozen RTA CBVF + CEL LUTs from repo fixtures and self-contained RAII VRTAWND1 fixture
  const std::string repo_root(VERTIRTA_REPO_ROOT);
  vertirta::rta::MmapCbvf4DLut cbvf_lut;
  vertirta::rta::Cel2DLut cel_lut;
  REQUIRE(cbvf_lut.load(repo_root + "/tests/fixtures/quick_test.vrta_lut", /*verify_sha256=*/true));
  REQUIRE(cel_lut.load(repo_root + "/tests/fixtures/quick_test.vrta_cel2d", /*verify_sha256=*/true));

  ScopedVrtaWindFixture wind_fix;
  MmapVrtaWindLut wind_lut;
  REQUIRE(wind_lut.load(wind_fix.path_str(), /*verify_sha256=*/true));
  wind_lut.update_inflow(225.0 * vertirta::core::kPi / 180.0, 14.0);

  const vertirta::core::CapabilityEnvelope env_nominal =
      vertirta::core::CoaxEffectiveness::compute_envelope(
          0x00u, 0u, 0x07u, 403.2f, 92.0f, 0.80f, 0.0f, 0.0f, 0.0f, 0u);
  const vertirta::core::CapabilityEnvelope env_oei =
      vertirta::core::CoaxEffectiveness::compute_envelope(
          0x01u, 0u, 0x07u, 385.0f, 130.0f, 0.65f, 0.0f, 0.0f, 0.0f, 5u);

  AcousticFlightState ac_nominal{};
  ac_nominal.airframe = AcousticAirframeKind::kCoaxialX8_240;
  ac_nominal.airspeed_mps = 14.0;
  ac_nominal.lift_rpm = kX8HoverRpm * 1.05;
  ac_nominal.lift_thrust_total_n = 2400.0;

  // 2) Tier-1 quiet-mode routing measurably lowers Kartal Hospital dBA along nominal corridor
  //    vs noise-agnostic straight path (starting 450 m south-west of hospital, heading north)
  const auto hosp_ned = helipad_hz.pos_ned;
  auto simulate_corridor = [&](bool quiet_mode) -> double {
    vertirta::rta::CbvfSafetyFilter sf(&cbvf_lut, &cel_lut);
    NoiseRouterConfig cfg{};
    cfg.quiet_mode_enabled = quiet_mode;
    cfg.period = TimeOfDayPeriod::kNight;  // 45 dBA hospital threshold activates quiet-mode deflection
    NoiseAwareRouter router(&sf, &wind_lut, cfg);

    // Start 480 m south, 220 m west of Kartal Hospital at 75 m AGL, flying North at 12 m/s
    vertirta::core::Vec3Ned pos{hosp_ned.x - 480.0, hosp_ned.y - 220.0, -75.0};
    vertirta::core::Vec3Ned vel{12.0, 0.0, 0.0};
    double peak_hosp_dba = 0.0;

    for (int tick = 0; tick < 800; ++tick) {
      // Gentle speed-hold tracking acceleration along north corridor
      const vertirta::core::Vec3Ned a_track{0.5 * (12.0 - vel.x), 0.0, 0.0};
      const auto out = router.step_100hz(pos, vel, a_track, 0.0, env_nominal, ac_nominal, {});
      REQUIRE(out.rta_out.a_safe_ned.is_finite());
      REQUIRE(out.total_hazards_fed_to_rta <= kMaxRouterActiveHazards);
      peak_hosp_dba = std::max(peak_hosp_dba, out.kartal_hospital_dba);
      vel = vel + out.rta_out.a_safe_ned * 0.01;
      pos = pos + vel * 0.01;
    }
    return peak_hosp_dba;
  };

  const double peak_dba_agnostic = simulate_corridor(/*quiet_mode=*/false);
  const double peak_dba_quiet = simulate_corridor(/*quiet_mode=*/true);
  INFO("Corridor peak Kartal Hospital dBA: agnostic=" << peak_dba_agnostic << " vs quiet=" << peak_dba_quiet);
  REQUIRE(peak_dba_quiet < peak_dba_agnostic - 0.5);

  // 3) Adversarial override test: OEI + coastal downdraft forces immediate 10 ms (1-tick) noise shedding
  {
    vertirta::rta::CbvfSafetyFilter sf(&cbvf_lut, &cel_lut);
    NoiseRouterConfig cfg{};
    cfg.quiet_mode_enabled = true;
    cfg.period = TimeOfDayPeriod::kNight;
    NoiseAwareRouter router(&sf, &wind_lut, cfg);

    // Place aircraft 220 m south of Kartal Hospital helipad approaching at 14 m/s under OEI
    const vertirta::core::Vec3Ned pos_adv{hosp_ned.x - 220.0, hosp_ned.y, -70.0};
    const vertirta::core::Vec3Ned vel_adv{14.0, 0.0, 0.5};
    const vertirta::core::Vec3Ned a_track{1.5, 0.0, 0.0};

    const auto out = router.step_100hz(pos_adv, vel_adv, a_track, 0.0, env_oei, ac_nominal, {});
    REQUIRE(out.noise_preference_shed == true);  // 100% shed within 1 tick (10 ms)
    REQUIRE(out.a_ref_soft_ned.x == Catch::Approx(a_track.x));
    REQUIRE(out.a_ref_soft_ned.y == Catch::Approx(a_track.y));
    REQUIRE(out.rta_out.a_safe_ned.is_finite());
    REQUIRE(out.rta_out.qp_exit_flag >= 0);
    REQUIRE(out.total_hazards_fed_to_rta <= kMaxRouterActiveHazards);
  }

  // 4) 500-scenario Monte Carlo Lodos/Poyraz VRTAWND1 + OEI + hospital-helipad proximity stress runs
  uint64_t rng = 0x20261123a5a5ull;
  int helipad_breaches = 0;
  int ctr_breaches = 0;
  int non_finite_count = 0;
  int qp_deadlocks = 0;

  for (int sc = 0; sc < 500; ++sc) {
    // Alternate Lodos SW (200..250 deg) and Poyraz NE (20..70 deg) at 8..18 m/s
    const double dir_deg =
        (sc % 2 == 0) ? uniform_range(rng, 200.0, 250.0) : uniform_range(rng, 20.0, 70.0);
    const double u_ref = uniform_range(rng, 8.0, 18.0);
    wind_lut.update_inflow(dir_deg * vertirta::core::kPi / 180.0, u_ref);

    vertirta::rta::CbvfSafetyFilter sf(&cbvf_lut, &cel_lut);
    NoiseRouterConfig cfg{};
    cfg.quiet_mode_enabled = true;
    cfg.period = static_cast<TimeOfDayPeriod>(sc % 3);
    NoiseAwareRouter router(&sf, &wind_lut, cfg);

    // Spawn safely outside Kartal Hospital helipad cylinder (>= 240 m radial clearance) & west of LTFJ CTR
    const double angle = uniform_range(rng, 0.0, 2.0 * vertirta::core::kPi);
    const double start_r = uniform_range(rng, 250.0, 420.0);
    vertirta::core::Vec3Ned pos{
        hosp_ned.x + start_r * std::cos(angle),
        std::min(hosp_ned.y + start_r * std::sin(angle), -120.0),
        -uniform_range(rng, 55.0, 115.0),
    };
    // Deliberately point initial velocity toward the hospital helipad or east toward LTFJ CTR
    const double spd = uniform_range(rng, 6.0, 13.0);
    vertirta::core::Vec3Ned vel{
        -spd * std::cos(angle),
        (sc % 4 == 0) ? 8.0 : -spd * std::sin(angle),
        uniform_range(rng, -0.5, 0.5),
    };

    const auto& env = (sc % 3 == 0) ? env_oei : env_nominal;
    for (int step = 0; step < 30; ++step) {
      const vertirta::core::Vec3Ned a_track{
          -0.6 * std::cos(angle), (sc % 4 == 0) ? 1.5 : -0.6 * std::sin(angle), 0.0};
      const auto out = router.step_100hz(pos, vel, a_track, 0.0, env, ac_nominal, {});
      if (!out.rta_out.a_safe_ned.is_finite()) {
        ++non_finite_count;
        break;
      }
      if (out.rta_out.qp_exit_flag < 0 && !out.rta_out.used_hj_fallback) {
        ++qp_deadlocks;
      }

      // Integrate 100 Hz kinematic step (0.01 s)
      vel = vel + out.rta_out.a_safe_ned * 0.01;
      pos = pos + vel * 0.01;

      const double d_hosp = std::hypot(pos.x - hosp_ned.x, pos.y - hosp_ned.y);
      const double alt_agl = -pos.z;
      if (d_hosp < kKartalHospitalHelipadRadiusHM && alt_agl < kKartalHospitalHelipadRadiusZM) {
        ++helipad_breaches;
        break;
      }
      const auto geo =
          vertirta::core::local_enu_to_geodetic(vertirta::core::ned_to_enu(pos), vertirta::core::kKartalDatum);
      if (geo.lon_deg >= kLtfjMaxAllowedLonDeg) {
        ++ctr_breaches;
        break;
      }
    }
  }

  REQUIRE(helipad_breaches == 0);
  REQUIRE(ctr_breaches == 0);
  REQUIRE(non_finite_count == 0);
  REQUIRE(qp_deadlocks == 0);
}

TEST_CASE("NoiseAwareRouter caches 180-node acoustic hemisphere across steady ticks and refreshes on >2% RPM or >2 m/s shift") {
  const std::string repo_root(VERTIRTA_REPO_ROOT);
  vertirta::rta::MmapCbvf4DLut cbvf_lut;
  vertirta::rta::Cel2DLut cel_lut;
  REQUIRE(cbvf_lut.load(repo_root + "/tests/fixtures/quick_test.vrta_lut", true));
  REQUIRE(cel_lut.load(repo_root + "/tests/fixtures/quick_test.vrta_cel2d", true));
  vertirta::rta::CbvfSafetyFilter sf(&cbvf_lut, &cel_lut);
  NoiseAwareRouter router(&sf, nullptr, NoiseRouterConfig{});

  const vertirta::core::CapabilityEnvelope env =
      vertirta::core::CoaxEffectiveness::compute_envelope(
          0x00u, 0u, 0x07u, 403.2f, 92.0f, 0.80f, 0.0f, 0.0f, 0.0f, 0u);
  const vertirta::core::Vec3Ned pos{0.0, -300.0, -80.0};
  const vertirta::core::Vec3Ned vel{12.0, 0.0, 0.0};

  AcousticFlightState ac{};
  ac.airframe = AcousticAirframeKind::kCoaxialX8_240;
  ac.airspeed_mps = 12.0;
  ac.lift_rpm = kX8HoverRpm;
  ac.lift_thrust_total_n = 2353.596;

  // 100 steady-state ticks with small (<1% RPM, <0.8 m/s speed) jitter -> strictly 1 synthesis
  for (int tick = 0; tick < 100; ++tick) {
    AcousticFlightState s = ac;
    s.lift_rpm = ac.lift_rpm * (1.0 + 0.008 * std::sin(0.1 * static_cast<double>(tick)));
    s.airspeed_mps = ac.airspeed_mps + 0.5 * std::cos(0.1 * static_cast<double>(tick));
    const auto out = router.step_100hz(pos, vel, {0.0, 0.0, 0.0}, 0.0, env, s, {});
    REQUIRE(out.hemisphere_recomputed == (tick == 0));
  }
  REQUIRE(router.hemisphere_recompute_count() == 1u);

  // > 2% RPM shift triggers immediate cache refresh
  AcousticFlightState ac_rpm = ac;
  ac_rpm.lift_rpm = ac.lift_rpm * 1.035;
  const auto out_rpm = router.step_100hz(pos, vel, {0.0, 0.0, 0.0}, 0.0, env, ac_rpm, {});
  REQUIRE(out_rpm.hemisphere_recomputed == true);
  REQUIRE(router.hemisphere_recompute_count() == 2u);

  // > 2 m/s airspeed shift triggers immediate cache refresh
  AcousticFlightState ac_spd = ac_rpm;
  ac_spd.airspeed_mps = ac_rpm.airspeed_mps + 2.6;
  const auto out_spd = router.step_100hz(pos, vel, {0.0, 0.0, 0.0}, 0.0, env, ac_spd, {});
  REQUIRE(out_spd.hemisphere_recomputed == true);
  REQUIRE(router.hemisphere_recompute_count() == 3u);
}
