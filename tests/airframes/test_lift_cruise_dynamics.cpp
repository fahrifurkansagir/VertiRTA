// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "vertirta/airframes/coaxial_x8_adapter.hpp"
#include "vertirta/airframes/lift_cruise_dynamics.hpp"

using namespace vertirta::airframes;

TEST_CASE("Canonical X8-240 adapter locks D=1.50m z/D=0.233 and rejects DR-09 parameter drift") {
  CoaxialX8AirframeAdapter x8;
  REQUIRE(x8.actuator_count() == 8u);
  REQUIRE(x8.control_count() == 4u);
  REQUIRE(x8.mass_kg() == Catch::Approx(240.0).margin(1e-6));
  REQUIRE(x8.rotor_diameter_m() == Catch::Approx(1.50).margin(1e-6));
  REQUIRE(x8.coax_z_over_d() == Catch::Approx(0.35 / 1.50).margin(1e-6));
  REQUIRE(x8.max_airspeed_mps() == Catch::Approx(22.0).margin(1e-6));

  AirframeState6Dof st{};
  st.pos_ned = {0.0, 0.0, -40.0};
  const double hover_t = x8.mass_kg() * vertirta::core::params::kGravity / 8.0;
  for (uint8_t i = 0; i < 8u; ++i) {
    st.u_act[i] = hover_t;
  }

  EffectivenessMatrix eff{};
  x8.compute_effectiveness(st, {0.0, 0.0, 0.0}, 0u, 0.0025, eff);
  REQUIRE(eff.n_actuators == 8u);
  REQUIRE(eff.n_controls == 4u);
  REQUIRE(eff.tau_act_s[0] == Catch::Approx(0.085).margin(1e-6));
  REQUIRE(eff.u_max[0] > 750.0);  // Upper rotor at 403.2 V (~795 N)

  const auto deriv = x8.eval_dynamics(st, {0.0, 0.0, 0.0}, 0u);
  REQUIRE(std::abs(deriv.vel_dot_ned.z) < 0.05);
  REQUIRE(deriv.omega_dot_frd.norm() < 1e-6);
}

TEST_CASE("8+1 Lift+Cruise passes <=79.2% T_max hover OEI while 4+1 drops to rank 3 and recovers >=14 m/s") {
  LiftCruiseDynamics lc8(LiftCruiseLayout::kEightPlusOne);
  REQUIRE(lc8.actuator_count() == 12u);
  REQUIRE(lc8.control_count() == 5u);
  REQUIRE(lc8.mass_kg() == Catch::Approx(240.0).margin(1e-6));
  REQUIRE(lc8.cruise_airspeed_mps() == Catch::Approx(38.0).margin(1e-6));
  REQUIRE(lc8.max_airspeed_mps() == Catch::Approx(46.3).margin(0.2));  // 90 kt Lift+Cruise V_NE
  REQUIRE(lc8.stall_speed_1g_mps() == Catch::Approx(21.83).margin(0.15));

  // Worst-case outboard lift rotor #0 failure in hover (V_a = 0 m/s):
  // Exact 4-DOF trim retains > 20.8% dynamic thrust margin (peak <= 79.2% T_max).
  for (uint8_t failed = 0; failed < 8u; ++failed) {
    const auto oei = lc8.audit_oei_trim_feasibility(/*v_airspeed_mps=*/0.0, failed);
    REQUIRE(oei.trim_feasible_4dof == true);
    REQUIRE(oei.effective_rank == 4u);
    REQUIRE(oei.peak_thrust_ratio <= 0.793);
    REQUIRE(oei.dynamic_margin_ratio >= 0.207);
    REQUIRE(oei.u_trim[failed] == Catch::Approx(0.0).margin(1e-9));
  }

  // Outboard rotor #0 specifically hits the closed-form minimax ~79.07%-79.15% ceiling (~430.4 N)
  const auto oei_outboard = lc8.audit_oei_trim_feasibility(0.0, 0u);
  REQUIRE(oei_outboard.peak_thrust_n == Catch::Approx(430.4).margin(1.0));
  REQUIRE(oei_outboard.peak_thrust_ratio <= 0.792);

  // Verify 4+1 compat layout collapses to rank 3 in hover, regains 4-DOF forward trim at >= 14 m/s
  LiftCruiseDynamics lc4(LiftCruiseLayout::kFourPlusOne);
  REQUIRE(lc4.actuator_count() == 8u);
  const auto oei4_hover = lc4.audit_oei_trim_feasibility(0.0, 0u);
  REQUIRE(oei4_hover.trim_feasible_4dof == false);
  REQUIRE(oei4_hover.effective_rank == 3u);

  const auto oei4_trans = lc4.audit_oei_trim_feasibility(14.0, 0u);
  REQUIRE(oei4_trans.trim_feasible_4dof == true);
  REQUIRE(oei4_trans.effective_rank == 4u);
  REQUIRE(oei4_trans.peak_thrust_ratio <= 0.86);
  REQUIRE(oei4_trans.dynamic_margin_ratio >= 0.14);
}

TEST_CASE("Lift+Cruise 6-DOF aerodynamics couples relative wind and sustains 38 m/s wing-borne cruise") {
  LiftCruiseDynamics lc8(LiftCruiseLayout::kEightPlusOne);

  // Wing-borne cruise trim check at 38 m/s, lift rotors stopped
  AirframeState6Dof st{};
  st.pos_ned = {0.0, 0.0, -100.0};
  // Pitch up slightly to generate trim C_L ~ 0.511 at 38 m/s (q = 884.45 Pa, S = 5.20 m^2)
  const double theta = 0.056;  // ~3.2 deg trim AoA
  st.q_frd_ned = {std::cos(0.5 * theta), 0.0, std::sin(0.5 * theta), 0.0};
  st.vel_ned = {38.0, 0.0, 0.0};
  st.u_act[8] = 200.0;  // Pusher cruise thrust ~199.1 N

  const auto deriv_calm = lc8.eval_dynamics(st, {0.0, 0.0, 0.0}, 0u);
  REQUIRE(deriv_calm.airspeed_mps == Catch::Approx(38.0).margin(0.1));
  REQUIRE(std::abs(deriv_calm.vel_dot_ned.z) < 0.8);
  REQUIRE(std::abs(deriv_calm.vel_dot_ned.x) < 0.8);

  // Headwind along -N increases relative airspeed to ~48 m/s and produces upward lift excess
  const auto deriv_headwind = lc8.eval_dynamics(st, {-10.0, 0.0, 0.0}, 0u);
  REQUIRE(deriv_headwind.airspeed_mps == Catch::Approx(48.0).margin(0.2));
  REQUIRE(deriv_headwind.vel_dot_ned.z < deriv_calm.vel_dot_ned.z - 2.0);  // More upward (-z) accel

  // Effectiveness keeps physical lift-rotor columns non-zero during transition (V_a = 21 m/s)
  st.vel_ned = {21.0, 0.0, 0.0};
  EffectivenessMatrix eff{};
  lc8.compute_effectiveness(st, {0.0, 0.0, 0.0}, 0u, 0.0025, eff);
  REQUIRE(std::abs(eff.B[0][0]) > 0.01);  // Outboard lift rotor roll effectiveness active
  REQUIRE(std::abs(eff.B[0][9]) > 0.05);  // Aileron roll effectiveness active at 21 m/s
  REQUIRE(std::abs(eff.B[1][10]) > 0.05); // Elevator pitch effectiveness active
  REQUIRE(eff.tau_act_s[9] == Catch::Approx(0.020).margin(1e-6));  // 50 rad/s servo lag
  REQUIRE(eff.tau_act_s[0] == Catch::Approx(0.085).margin(1e-6));  // Lift rotor lag
}
