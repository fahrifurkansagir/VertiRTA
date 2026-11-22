#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

#include "vertirta_gz/coaxial_physics_kernel.hpp"
#include "vertirta_gz/dryden_wind_kernel.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

std::string read_text_file(const std::string& path) {
  std::ifstream ifs(path);
  REQUIRE(ifs.is_open());
  std::ostringstream oss;
  oss << ifs.rdbuf();
  return oss.str();
}

std::string repo_root() {
#ifdef VERTIRTA_REPO_ROOT
  return std::string(VERTIRTA_REPO_ROOT);
#else
  return ".";
#endif
}

}  // namespace

TEST_CASE("Coaxial kernel fixes climb inflow sign (v_iu + v_air.z) and narrows wake skew chi") {
  vertirta_gz::CoaxialPhysicsKernel kernel;
  std::array<double, 8> cmds{};
  cmds.fill(151.8);
  kernel.set_rotor_speeds_direct(cmds);
  kernel.set_motor_commands(cmds);

  // Level forward flight at 10 m/s (+X FLU)
  const auto level_out = kernel.step(0.0025, {10.0, 0.0, 0.0});
  // Climbing forward flight at 10 m/s forward, +5 m/s climb (+Z FLU)
  kernel.set_rotor_speeds_direct(cmds);
  const auto climb_out = kernel.step(0.0025, {10.0, 0.0, 5.0});

  for (int arm = 0; arm < 4; ++arm) {
    REQUIRE(climb_out.v_iu_mps[arm] > 0.0);
    // DR-1 erratum fix: climb inflow adds +v_air.z -> total downward axial flow exceeds v_iu
    REQUIRE(climb_out.v_axial_down_mps[arm] > climb_out.v_iu_mps[arm] + 4.9);
    REQUIRE(climb_out.v_axial_down_mps[arm] > level_out.v_axial_down_mps[arm]);
    // Narrower wake skew angle chi when climbing than in level flight
    REQUIRE(climb_out.wake_skew_chi_rad[arm] < level_out.wake_skew_chi_rad[arm]);
  }

  // At 22 m/s cruise, upper wake clears lower disk -> eta_T approaches 1.0
  kernel.set_rotor_speeds_direct(cmds);
  const auto cruise_out = kernel.step(0.0025, {22.0, 0.0, 0.0});
  REQUIRE(cruise_out.eta_t[4] > 0.97);
}

TEST_CASE("Lower rotor applies 0.85*k_T0 thrust, 0.96*k_Q0 swirl recovery, and +17.6% unblanking") {
  vertirta_gz::CoaxialPhysicsKernel kernel;
  kernel.set_v_term_override(403.2);  // Nominal 400 V bus

  std::array<double, 8> max_cmds{};
  max_cmds.fill(240.0);
  kernel.set_rotor_speeds_direct(max_cmds);
  kernel.set_motor_commands(max_cmds);

  const auto hover = kernel.step(0.0025, {0.0, 0.0, 0.0});
  const double k_t0 = kernel.params().k_t0;
  const double k_q0 = kernel.params().k_q0;

  // Upper rotors: 795.4 N @ 240 rad/s, eta_t = 1.0, eta_q = 1.0
  REQUIRE_THAT(hover.eta_t[0], WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(hover.eta_q[0], WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(hover.thrust_n[0], WithinAbs(k_t0 * 240.0 * 240.0, 1e-3));
  REQUIRE_THAT(hover.thrust_n[0], WithinAbs(795.40, 0.25));

  // Lower rotors: 0.85 * k_T0 (676.1 N) and 0.96 * k_Q0 swirl recovery
  REQUIRE_THAT(hover.eta_t[4], WithinAbs(0.85, 1e-6));
  REQUIRE_THAT(hover.eta_q[4], WithinAbs(0.96, 1e-6));
  REQUIRE_THAT(hover.thrust_n[4], WithinAbs(0.85 * k_t0 * 240.0 * 240.0, 1e-3));
  REQUIRE_THAT(hover.thrust_n[4], WithinAbs(676.09, 0.25));
  REQUIRE_THAT(std::abs(hover.torque_z_flu[4]), WithinAbs(0.96 * k_q0 * 240.0 * 240.0, 1e-3));

  // Balanced counter-rotating pairs produce zero net yaw torque in symmetric hover
  double net_yaw = 0.0;
  for (int i = 0; i < 8; ++i) {
    net_yaw += hover.torque_z_flu[i];
  }
  REQUIRE_THAT(net_yaw, WithinAbs(0.0, 1e-6));

  // Upper motor M0 fails -> lower motor M4 unblanks (+17.647%: 676.09 N -> 795.40 N)
  kernel.set_motor_fault(0, true);
  std::array<double, 8> w_after = max_cmds;
  w_after[0] = 0.0;
  kernel.set_rotor_speeds_direct(w_after);
  const auto unblanked = kernel.step(0.0025, {0.0, 0.0, 0.0});
  REQUIRE_THAT(unblanked.eta_t[4], WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(unblanked.eta_q[4], WithinAbs(1.0, 1e-6));
  REQUIRE_THAT(unblanked.thrust_n[4] / hover.thrust_n[4], WithinRel(1.0 / 0.85, 1e-5));
  REQUIRE_THAT(unblanked.thrust_n[4], WithinAbs(795.40, 0.25));
}

TEST_CASE("Fault-off wires tau_coast_ and preserves -179 rad/s^2 hover coast-down vs powered brake") {
  vertirta_gz::CoaxialPhysicsKernel k_nom;
  vertirta_gz::CoaxialPhysicsKernel k_fast_coast;
  vertirta_gz::CoaxialPhysicsKernel k_powered_brake;

  auto fast_params = k_fast_coast.params();
  fast_params.tau_coast_s = 0.15;
  k_fast_coast.set_params(fast_params);

  std::array<double, 8> w_hov{};
  w_hov.fill(151.8);
  k_nom.set_rotor_speeds_direct(w_hov);
  k_fast_coast.set_rotor_speeds_direct(w_hov);
  k_powered_brake.set_rotor_speeds_direct(w_hov);

  k_nom.set_motor_commands(w_hov);
  k_fast_coast.set_motor_commands(w_hov);
  auto brake_cmd = w_hov;
  brake_cmd[0] = 0.0;
  k_powered_brake.set_motor_commands(brake_cmd);

  k_nom.set_motor_fault(0, true);
  k_fast_coast.set_motor_fault(0, true);

  // Single tick check: nominal coast-down deceleration ~ -179 rad/s^2 (DR-1 Section 3.5)
  const auto s1_nom = k_nom.step(0.0025, {0.0, 0.0, 0.0});
  const auto s1_fast = k_fast_coast.step(0.0025, {0.0, 0.0, 0.0});
  const auto s1_brake = k_powered_brake.step(0.0025, {0.0, 0.0, 0.0});

  const double dw_nom = (s1_nom.omega_rad_s[0] - 151.8) / 0.0025;
  const double dw_fast = (s1_fast.omega_rad_s[0] - 151.8) / 0.0025;
  const double dw_brake = (s1_brake.omega_rad_s[0] - 151.8) / 0.0025;

  REQUIRE_THAT(dw_nom, WithinAbs(-179.0, 5.0));
  // tau_coast_ is genuinely wired: shorter tau_coast_ produces faster coast-down
  REQUIRE(dw_fast < dw_nom - 100.0);
  // Powered brake hits -alpha_max (-487.7 rad/s^2), much faster than unpowered coast-down
  REQUIRE_THAT(dw_brake, WithinAbs(-487.7, 1.0));
  REQUIRE(dw_nom > dw_brake);

  // Coast over 3 s: smoothly decays toward 0 without going negative
  for (int k = 0; k < 1200; ++k) {
    (void)k_nom.step(0.0025, {0.0, 0.0, 0.0});
  }
  REQUIRE(k_nom.rotor(0).omega >= 0.0);
  REQUIRE(k_nom.rotor(0).omega < 10.0);
}

TEST_CASE("BLDC K_e,bus=1.36 voltage ceiling preserves 2-arm T/W >= 1.25 at 358.4 V and >= 1.12 at 338 V") {
  constexpr double kWeightN = 240.0 * 9.80665;
  vertirta_gz::CoaxialPhysicsKernel kernel;

  const double w_358 = kernel.solve_bldc_voltage_limit(358.4);
  const double w_338 = kernel.solve_bldc_voltage_limit(338.0);
  REQUIRE(w_358 >= 240.0);
  REQUIRE(w_338 > 226.5);

  // 2-arm full thrust (2 upper + 2 lower = 2 * (1 + 0.85) * k_T0 * min(w_cap, 240)^2)
  const auto two_arm_tw = [&](double v_term) {
    const double w = std::min(kernel.params().omega_max_nom, kernel.solve_bldc_voltage_limit(v_term));
    const double t_pair = (1.0 + kernel.params().eta_coax_hover) * kernel.params().k_t0 * w * w;
    return (2.0 * t_pair) / kWeightN;
  };
  REQUIRE(two_arm_tw(358.4) >= 1.249);
  REQUIRE(two_arm_tw(338.0) >= 1.120);

  // Coupled 1-RC battery step sags voltage below OCV and discharges SoC monotonically under hover load
  kernel.clear_v_term_override();
  auto p50 = kernel.params();
  p50.soc_init = 0.50;
  kernel.set_params(p50);
  kernel.reset();
  std::array<double, 8> hov{};
  hov.fill(151.8);
  kernel.set_rotor_speeds_direct(hov);
  kernel.set_motor_commands(hov);
  const double soc_init = kernel.soc();
  const double voc_init = vertirta_gz::CoaxialPhysicsKernel::open_circuit_voltage(soc_init);
  const auto out = kernel.step(0.01, {0.0, 0.0, 0.0});
  REQUIRE(std::isfinite(out.v_term_v));
  REQUIRE(out.p_elec_w > 28000.0);
  REQUIRE(out.p_elec_w < 45000.0);
  REQUIRE(out.i_pack_a > 60.0);
  REQUIRE(out.v_term_v < voc_init);
  REQUIRE(out.v_term_v > 380.0);
  REQUIRE(kernel.soc() < soc_init);
}

TEST_CASE("MIL-F-8785C Tustin Dryden kernel matches Table 4 scale lengths, preserves variance, and is deterministic") {
  const auto p30 = vertirta_gz::DrydenWindKernel::compute_scales(30.0, 15.43);
  const auto p60 = vertirta_gz::DrydenWindKernel::compute_scales(60.0, 15.43);
  const auto p120 = vertirta_gz::DrydenWindKernel::compute_scales(120.0, 15.43);

  REQUIRE_THAT(p30.L_u, WithinAbs(152.5, 0.5));
  REQUIRE_THAT(p60.L_u, WithinAbs(219.7, 0.5));
  REQUIRE_THAT(p120.L_u, WithinAbs(275.0, 0.5));
  REQUIRE_THAT(p30.sigma_u, WithinAbs(2.65, 0.03));
  REQUIRE_THAT(p120.sigma_u, WithinAbs(2.03, 0.03));
  REQUIRE_THAT(p60.sigma_w, WithinAbs(1.543, 0.005));

  // Exact Parseval impulse-response variance preservation >= 99.5% @ 100 Hz (h = 60 m, V = 15 m/s)
  vertirta_gz::DrydenWindKernel imp_kernel(1ULL);
  double parseval_u = 0.0, parseval_v = 0.0, parseval_w = 0.0;
  for (int k = 0; k < 20000; ++k) {
    const std::array<double, 3> impulse = (k == 0) ? std::array<double, 3>{1.0, 1.0, 1.0}
                                                   : std::array<double, 3>{0.0, 0.0, 0.0};
    const auto h = imp_kernel.step(0.01, 60.0, 15.0, 15.43, impulse);
    parseval_u += h.x * h.x;
    parseval_v += h.y * h.y;
    parseval_w += h.z * h.z;
  }
  REQUIRE_THAT(parseval_u / (p60.sigma_u * p60.sigma_u), WithinAbs(1.0, 0.005));
  REQUIRE_THAT(parseval_v / (p60.sigma_v * p60.sigma_v), WithinAbs(1.0, 0.005));
  REQUIRE_THAT(parseval_w / (p60.sigma_w * p60.sigma_w), WithinAbs(1.0, 0.005));

  // Determinism check (same seed -> exact 0-ULP match)
  vertirta_gz::DrydenWindKernel d1(42ULL);
  vertirta_gz::DrydenWindKernel d2(42ULL);
  for (int i = 0; i < 200; ++i) {
    const auto g1 = d1.step_seeded(0.01, 60.0, 15.0, 15.43);
    const auto g2 = d2.step_seeded(0.01, 60.0, 15.0, 15.43);
    REQUIRE(g1.x == g2.x);
    REQUIRE(g1.y == g2.y);
    REQUIRE(g1.z == g2.z);
  }

  // Empirical Monte Carlo std check across 300k ticks @ 100 Hz (h = 30 m, V = 22 m/s, W20 = 15.43 m/s)
  vertirta_gz::DrydenWindKernel stat_kernel(1337ULL);
  double sum_u2 = 0.0, sum_v2 = 0.0, sum_w2 = 0.0;
  constexpr int kWarmup = 5000;
  constexpr int kSamples = 300000;
  for (int i = 0; i < kWarmup + kSamples; ++i) {
    const auto g = stat_kernel.step_seeded(0.01, 30.0, 22.0, 15.43);
    if (i >= kWarmup) {
      sum_u2 += g.x * g.x;
      sum_v2 += g.y * g.y;
      sum_w2 += g.z * g.z;
    }
  }
  const double std_u = std::sqrt(sum_u2 / kSamples);
  const double std_v = std::sqrt(sum_v2 / kSamples);
  const double std_w = std::sqrt(sum_w2 / kSamples);

  REQUIRE_THAT(std_u, WithinRel(p30.sigma_u, 0.05));
  REQUIRE_THAT(std_v, WithinRel(p30.sigma_v, 0.05));
  REQUIRE_THAT(std_w, WithinRel(p30.sigma_w, 0.05));
}

TEST_CASE("model.sdf defines lidar_link 60 m, both frame tags, 3 IMUs, 2 baros, navsat, mag, and 8 rotors") {
  const std::string sdf = read_text_file(repo_root() + "/sim/models/vertirta_x8_240/model.sdf");

  REQUIRE(sdf.find("<model name=\"vertirta_x8_240\">") != std::string::npos);
  REQUIRE(sdf.find("<mass>240.0</mass>") != std::string::npos);
  REQUIRE(sdf.find("<ixx>85.0</ixx>") != std::string::npos);
  REQUIRE(sdf.find("<iyy>95.0</iyy>") != std::string::npos);
  REQUIRE(sdf.find("<izz>140.0</izz>") != std::string::npos);

  // Explicit lidar_link with gpu_lidar max=60.0 and both gz_frame_id + frame_id
  const auto lidar_link_pos = sdf.find("<link name=\"lidar_link\">");
  REQUIRE(lidar_link_pos != std::string::npos);
  const auto lidar_link_end = sdf.find("</link>", lidar_link_pos);
  REQUIRE(lidar_link_end != std::string::npos);
  const std::string lidar_block = sdf.substr(lidar_link_pos, lidar_link_end - lidar_link_pos);
  REQUIRE(lidar_block.find("type=\"gpu_lidar\"") != std::string::npos);
  REQUIRE(lidar_block.find("<max>60.0</max>") != std::string::npos);
  REQUIRE(lidar_block.find("<gz_frame_id>lidar_link</gz_frame_id>") != std::string::npos);
  REQUIRE(lidar_block.find("<frame_id>lidar_link</frame_id>") != std::string::npos);

  // 3 IMUs, 2 barometers, navsat, magnetometer
  for (const char* sensor : {
           "name=\"imu_sensor_0\"",
           "name=\"imu_sensor_1\"",
           "name=\"imu_sensor_2\"",
           "name=\"air_pressure_sensor_0\"",
           "name=\"air_pressure_sensor_1\"",
           "name=\"navsat_sensor\"",
           "name=\"magnetometer_sensor\"",
       }) {
    REQUIRE(sdf.find(sensor) != std::string::npos);
  }

  // All 8 coaxial rotor links and custom propulsion plugin
  for (int i = 0; i < 8; ++i) {
    REQUIRE(sdf.find("<link name=\"rotor_" + std::to_string(i) + "\">") != std::string::npos);
  }
  REQUIRE(sdf.find("libvertirta_coaxial_propulsion.so") != std::string::npos);
  REQUIRE(sdf.find("<ke_bus>1.36</ke_bus>") != std::string::npos);
}

TEST_CASE("kartal_coast.sdf locks 400 Hz physics, Kartal datum, AMSL elevation (no +36.7 m double-add), and no duplicate global plugins") {
  const std::string world = read_text_file(repo_root() + "/sim/worlds/kartal_coast.sdf");

  REQUIRE(world.find("<world name=\"kartal_coast\">") != std::string::npos);
  REQUIRE(world.find("<max_step_size>0.0025</max_step_size>") != std::string::npos);
  REQUIRE(world.find("<real_time_update_rate>400</real_time_update_rate>") != std::string::npos);
  REQUIRE(world.find("<latitude_deg>40.8860</latitude_deg>") != std::string::npos);
  REQUIRE(world.find("<longitude_deg>29.1880</longitude_deg>") != std::string::npos);

  // Review Focus #5: elevation must be AMSL (12.0 m), NOT HAE (48.7 m), so PX4 EKF2 + geodesy::amsl_to_hae() never double-adds +36.7 m
  std::smatch elev_match;
  REQUIRE(std::regex_search(world, elev_match, std::regex(R"(<elevation>([0-9.]+)</elevation>)")));
  const double elev_m = std::stod(elev_match[1].str());
  REQUIRE_THAT(elev_m, WithinAbs(12.0, 1e-6));
  REQUIRE_THAT(elev_m + 36.7, WithinAbs(48.7, 1e-6));
  REQUIRE(world.find("36.7") != std::string::npos);
  REQUIRE(world.find("AMSL") != std::string::npos);

  // No duplicate global Gazebo plugins already loaded by PX4 server.config
  for (const char* forbidden_plugin : {
           "gz-sim-physics-system",
           "gz-sim-user-commands-system",
           "gz-sim-scene-broadcaster-system",
           "gz-sim-contact-system",
           "gz-sim-imu-system",
           "gz-sim-air-pressure-system",
           "gz-sim-navsat-system",
           "gz-sim-magnetometer-system",
           "gz-sim-wind-effects-system",
           "gz-sim-sensors-system",
       }) {
    REQUIRE(world.find(forbidden_plugin) == std::string::npos);
  }
}

TEST_CASE("Gazebo Harmonic plugins enable velocity checks and use gz::sim::worldPose (no silent ECM no-op)") {
  const std::string coax_src = read_text_file(repo_root() + "/sim/gz_plugins/src/vertirta_coaxial_propulsion.cpp");
  const std::string wind_src = read_text_file(repo_root() + "/sim/gz_plugins/src/vertirta_dryden_wind.cpp");

  for (const std::string& src : {coax_src, wind_src}) {
    REQUIRE(src.find("#include <gz/sim/Util.hh>") != std::string::npos);
    REQUIRE(src.find("EnableVelocityChecks(ecm, true)") != std::string::npos);
    REQUIRE(src.find("gz::sim::worldPose(") != std::string::npos);
    REQUIRE(src.find("WorldLinearVelocity(ecm)") != std::string::npos);
    REQUIRE(src.find("Component<gz::sim::components::WorldPose>") == std::string::npos);
    REQUIRE(src.find("Component<gz::sim::components::WorldLinearVelocity>") == std::string::npos);
  }
}
