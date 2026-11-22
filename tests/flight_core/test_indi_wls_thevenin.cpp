#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/motor_fdi.hpp"

using Catch::Matchers::WithinAbs;
using namespace vertirta::core;

TEST_CASE("Coaxial unblanking and K_e=1.36 Thevenin sag hold 2-arm T/W >= 1.12 to 338 V") {
  double B[4][8]{};
  float umax[8]{};
  CoaxEffectiveness::build_B(0x00, 0.0f, B, umax, 400.0f);
  REQUIRE_THAT(umax[0], WithinAbs(795.4f, 1.5f));
  REQUIRE_THAT(umax[4], WithinAbs(676.1f, 1.5f));

  // Upper motor M0 fails -> lower coaxial partner M4 unblanks (+17.6% to 795.4 N)
  CoaxEffectiveness::build_B(0x01, 0.0f, B, umax, 400.0f);
  REQUIRE_THAT(umax[0], WithinAbs(0.0f, 1e-6f));
  REQUIRE_THAT(umax[4], WithinAbs(795.4f, 1.5f));

  // Voltage-headroom winding K_e,bus = 1.36 V*s/rad preserves T/W=1.25 to 358.4 V & >=1.12 to 338 V
  REQUIRE(CoaxEffectiveness::two_arm_tw_at_vterm(358.4f) >= 1.249f);
  REQUIRE(CoaxEffectiveness::two_arm_tw_at_vterm(338.0f) >= 1.120f);

  // 1-RC Thevenin battery at 20% SoC under 30 s Regime-3 hover (51.05 kW) + 5 s flare (62.85 kW)
  TheveninBattery1RC batt(0.20f);
  BatteryState st{};
  for (int k = 0; k < 3000; ++k) {
    st = batt.step(0.01f, 51050.0f);
  }
  REQUIRE(st.v_term_v >= 352.0f);
  for (int k = 0; k < 500; ++k) {
    st = batt.step(0.01f, 62850.0f);
  }
  REQUIRE(st.v_term_v >= 344.0f);  // Includes real coulomb SoC depletion (20% -> ~17.3%) + RC sag
  REQUIRE(CoaxEffectiveness::two_arm_tw_at_vterm(st.v_term_v) >= 1.12f);  // Safely above 338 V floor
}

TEST_CASE("f64/QR WLS never fails Cholesky across 10k ticks and commands Option B1b +/-2.2 rad/s spin") {
  IndiWlsController ctrl;
  const uint8_t regimes[4] = {0x00, 0x01, 0x21, 0x11};  // Nominal, M0, M0+M5, M0+M4 (Arm 0 dual)

  for (int r = 0; r < 4; ++r) {
    ctrl.reset();
    for (int k = 0; k < 2500; ++k) {
      IndiInput in{};
      in.q_frd_ned = {1.0, 0.0, 0.0, 0.0};
      in.gyro_frd = {0.02 * std::sin(0.01 * k), -0.02 * std::cos(0.01 * k), 0.0};
      in.accel_des_ned = {0.2 * std::sin(0.005 * k), -0.2 * std::cos(0.005 * k), 0.0};
      in.yaw_rate_des = 0.0;
      in.v_horiz_mps = 5.0f;
      in.v_term_v = 390.0f;
      in.fault_mask = regimes[r];

      float omega_cmd[8]{};
      const WlsResult res = ctrl.step_400hz(in, omega_cmd);
      REQUIRE_FALSE(res.cholesky_failed);
      for (int i = 0; i < 8; ++i) {
        REQUIRE(is_finite_f(omega_cmd[i]));
        if (regimes[r] & (1u << i)) {
          REQUIRE_THAT(omega_cmd[i], WithinAbs(0.0f, 1e-6f));
        } else {
          REQUIRE(omega_cmd[i] >= 0.0f);
          REQUIRE(omega_cmd[i] <= 240.01f);
        }
      }
    }
  }

  // Verify Option B1b active flat spin sign & phase lead Delta_psi = atan(Omega_z * tau_m):
  // Arm 0 (M0+M4 = 0x11) -> CCW -2.2 rad/s, phase lead -10.59 deg
  // Arm 2 (M2+M6 = 0x44) -> CW  +2.2 rad/s, phase lead +10.59 deg
  IndiInput in_arm0{};
  in_arm0.fault_mask = 0x11;
  in_arm0.v_term_v = 380.0f;
  float omega_cmd[8]{};
  ctrl.reset();
  (void)ctrl.step_400hz(in_arm0, omega_cmd);
  REQUIRE_THAT(ctrl.spin_rate_ref_rad_s(), WithinAbs(-2.2f, 1e-4f));
  REQUIRE_THAT(ctrl.phase_lead_deg(), WithinAbs(-10.59f, 0.15f));

  IndiInput in_arm2{};
  in_arm2.fault_mask = 0x44;
  in_arm2.v_term_v = 380.0f;
  ctrl.reset();
  (void)ctrl.step_400hz(in_arm2, omega_cmd);
  REQUIRE_THAT(ctrl.spin_rate_ref_rad_s(), WithinAbs(2.2f, 1e-4f));
  REQUIRE_THAT(ctrl.phase_lead_deg(), WithinAbs(10.59f, 0.15f));
}

TEST_CASE("Dual-channel Motor FDI isolates ESC cut <60 ms, blade loss <85 ms, and ignores 45 N*m gust") {
  // Channel 1: ESC RPM drop > 12 rad/s on M0 -> latches inside 25..60 ms
  {
    MotorFdiObserver fdi;
    float w_hat[8], w_esc[8];
    bool esc_ok[8];
    for (int i = 0; i < 8; ++i) {
      w_hat[i] = 151.8f;
      w_esc[i] = 151.8f;
      esc_ok[i] = true;
    }
    uint8_t mask = 0x00;
    for (int k = 0; k < 24; ++k) {  // 24 ticks = 60 ms
      w_esc[0] = 151.8f - 4.5f * static_cast<float>(k);  // Rapid coast-down
      mask = fdi.step_400hz({0, 0, 0}, {0, 0, 0}, -9.80665f, w_hat, w_esc, esc_ok, 0.0f, 400.0f, mask);
    }
    REQUIRE((mask & 0x01u) != 0u);
    REQUIRE(fdi.last_isolation_latency_ms() <= 60);
  }

  // Channel 2: Sensorless blade fracture on M0 (ESC still spins at 151.8, 0 thrust from M0) -> <85 ms
  {
    MotorFdiObserver fdi;
    float w_hat[8], w_esc[8];
    bool esc_ok[8];
    for (int i = 0; i < 8; ++i) {
      w_hat[i] = 151.8f;
      w_esc[i] = 151.8f;
      esc_ok[i] = true;
    }
    // Losing M0 (T0=318.2 N, d=0.9546, c_m,u=+0.0759) causes residual -b_0 * T0:
    // tau_x_actual - tau_x_hat = +303.7 N*m (dOmega_x = +3.573 rad/s^2),
    // tau_y_actual - tau_y_hat = -303.7 N*m (dOmega_y = -3.197 rad/s^2),
    // tau_z_actual - tau_z_hat = -24.16 N*m (dOmega_z = -0.1726 rad/s^2),
    // F_z_actual - F_z_hat = -318.2 N -> a_z = -(W - 318.2)/m = -8.481 m/s^2.
    const Vec3Frd domega_m0_loss{303.75 / 85.0, -303.75 / 95.0, -24.16 / 140.0};
    const float az_m0_loss = static_cast<float>(-(2353.596 - 318.2) / 240.0);
    uint8_t mask = 0x00;
    for (int k = 0; k < 34; ++k) {  // 34 ticks = 85 ms
      mask = fdi.step_400hz({0, 0, 0}, domega_m0_loss, az_m0_loss, w_hat, w_esc, esc_ok, 0.0f, 400.0f, mask);
    }
    REQUIRE((mask & 0x01u) != 0u);
    REQUIRE(fdi.last_isolation_latency_ms() <= 85);
  }

  // Gust immunity: 45 N*m roll/pitch coastal gust residual must NOT trigger motor fault
  {
    MotorFdiObserver fdi;
    float w_hat[8], w_esc[8];
    bool esc_ok[8];
    for (int i = 0; i < 8; ++i) {
      w_hat[i] = 151.8f;
      w_esc[i] = 151.8f;
      esc_ok[i] = true;
    }
    const Vec3Frd domega_gust{45.0 / 85.0, -45.0 / 95.0, 2.0 / 140.0};
    uint8_t mask = 0x00;
    for (int k = 0; k < 100; ++k) {
      mask = fdi.step_400hz({0, 0, 0}, domega_gust, -9.80665f, w_hat, w_esc, esc_ok, 0.0f, 400.0f, mask);
    }
    REQUIRE(mask == 0x00u);
  }
}
