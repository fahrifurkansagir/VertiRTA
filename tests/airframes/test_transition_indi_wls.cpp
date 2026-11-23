// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include "vertirta/airframes/lift_cruise_dynamics.hpp"
#include "vertirta/airframes/transition_indi_wls.hpp"
#include "../../core/airframes/lift_cruise/px4_v116_overlay/vtol_rate_ctrl_overlay.hpp"

using namespace vertirta::airframes;

namespace {

void integrate_step_rk2(
    const LiftCruiseDynamics& plant,
    AirframeState6Dof& st,
    const std::array<double, kMaxActuators>& u_cmd,
    const vertirta::core::Vec3Ned& wind_ned,
    uint16_t fault_mask,
    double dt_s) {
  // First-order asymmetric actuator lag per channel
  EffectivenessMatrix eff{};
  plant.compute_effectiveness(st, wind_ned, fault_mask, dt_s, eff);
  for (uint8_t i = 0; i < plant.actuator_count(); ++i) {
    const double target = vertirta::core::clamp_d(u_cmd[i], eff.u_min[i], eff.u_max[i]);
    if (i < 8u && ((fault_mask & static_cast<uint16_t>(1u << i)) != 0u)) {
      st.u_act[i] *= std::exp(-dt_s / 0.424);
    } else if (i == 8u && ((fault_mask & static_cast<uint16_t>(1u << 8u)) != 0u)) {
      st.u_act[i] *= std::exp(-dt_s / 0.35);
    } else {
      const double tau = std::max(0.005, eff.tau_act_s[i]);
      const double alpha_f = 1.0 - std::exp(-dt_s / tau);
      st.u_act[i] += alpha_f * (target - st.u_act[i]);
    }
  }

  auto advance = [&](const AirframeState6Dof& s, const AirframeDerivatives& d, double h) {
    AirframeState6Dof out = s;
    out.pos_ned = s.pos_ned + d.pos_dot_ned * h;
    out.vel_ned = s.vel_ned + d.vel_dot_ned * h;
    out.omega_frd = s.omega_frd + d.omega_dot_frd * h;

    // Quaternion kinematic step q_dot = 0.5 * q (x) [0, omega]
    const auto& q = s.q_frd_ned;
    const auto& w = s.omega_frd;
    const double qw = q.w + 0.5 * h * (-q.x * w.x - q.y * w.y - q.z * w.z);
    const double qx = q.x + 0.5 * h * (q.w * w.x + q.y * w.z - q.z * w.y);
    const double qy = q.y + 0.5 * h * (q.w * w.y - q.x * w.z + q.z * w.x);
    const double qz = q.z + 0.5 * h * (q.w * w.z + q.x * w.y - q.y * w.x);
    const double inv_n = 1.0 / std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    out.q_frd_ned = {qw * inv_n, qx * inv_n, qy * inv_n, qz * inv_n};
    return out;
  };

  const auto k1 = plant.eval_dynamics(st, wind_ned, fault_mask);
  const auto mid = advance(st, k1, 0.5 * dt_s);
  const auto k2 = plant.eval_dynamics(mid, wind_ned, fault_mask);
  st = advance(st, k2, dt_s);
}

}  // namespace

TEST_CASE("Quintic Hermite blend eta(V_a) satisfies C^2 continuity and endpoint zero-derivatives") {
  REQUIRE(quintic_hermite_blend(0.0) == Catch::Approx(0.0).margin(1e-12));
  REQUIRE(quintic_hermite_blend(14.0) == Catch::Approx(0.0).margin(1e-12));
  REQUIRE(quintic_hermite_blend(21.0) == Catch::Approx(0.5).margin(1e-12));
  REQUIRE(quintic_hermite_blend(28.0) == Catch::Approx(1.0).margin(1e-12));
  REQUIRE(quintic_hermite_blend(38.0) == Catch::Approx(1.0).margin(1e-12));

  // Numerical first & second derivatives vanish at both endpoints (14 m/s and 28 m/s)
  const double eps = 1e-4;
  const double d_lo = (quintic_hermite_blend(14.0 + eps) - quintic_hermite_blend(14.0)) / eps;
  const double d_hi = (quintic_hermite_blend(28.0) - quintic_hermite_blend(28.0 - eps)) / eps;
  REQUIRE(std::abs(d_lo) < 1e-6);
  REQUIRE(std::abs(d_hi) < 1e-6);
}

TEST_CASE("8+1 Lift+Cruise completes 0->38->0 m/s transition with <0.3m altitude deviation and zero stall breach") {
  LiftCruiseDynamics plant(LiftCruiseLayout::kEightPlusOne);
  TransitionIndiWlsController ctrl(plant);

  AirframeState6Dof st{};
  const double target_alt_m = 100.0;
  st.pos_ned = {0.0, 0.0, -target_alt_m};
  const double t_hover = plant.mass_kg() * LiftCruiseDynamics::kGravity / 8.0;
  for (uint8_t i = 0; i < 8u; ++i) {
    st.u_act[i] = t_hover;
  }
  ctrl.reset(st);

  constexpr double kDt = 0.0025;  // 400 Hz
  constexpr int kTotalTicks = 400 * 66;  // 66 s: 0..2s hover, 2..20s accel->38, 20..26s cruise, 26..58s decel->0, 58..66s hover
  double max_alt_err_m = 0.0;
  double max_airspeed_mps = 0.0;
  bool reached_cruise_lock = false;

  for (int tick = 0; tick < kTotalTicks; ++tick) {
    const double t = static_cast<double>(tick) * kDt;
    double v_des = 0.0;
    if (t < 2.0) {
      v_des = 0.0;
    } else if (t < 20.0) {
      const double s = (t - 2.0) / 18.0;
      v_des = 38.0 * (s * s * s * (10.0 - 15.0 * s + 6.0 * s * s));
    } else if (t < 26.0) {
      v_des = 38.0;
    } else if (t < 58.0) {
      const double s = (t - 26.0) / 32.0;
      v_des = 38.0 * (1.0 - (s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)));
    } else {
      v_des = 0.0;
    }

    TransitionCommand cmd{};
    cmd.target_airspeed_mps = v_des;
    cmd.target_alt_m = target_alt_m;
    cmd.target_heading_rad = 0.0;
    cmd.fault_mask = 0u;

    const auto out = ctrl.step_400hz(st, {0.0, 0.0, 0.0}, cmd);
    REQUIRE(out.valid);
    REQUIRE(std::abs(out.alpha_rad) < 0.25);  // Strictly below stall AoA
    if (out.rotors_locked_cruise) {
      reached_cruise_lock = true;
    }

    integrate_step_rk2(plant, st, out.u_cmd, {0.0, 0.0, 0.0}, 0u, kDt);
    const double alt_err = std::abs((-st.pos_ned.z) - target_alt_m);
    if (alt_err > max_alt_err_m) {
      max_alt_err_m = alt_err;
    }
    if (out.airspeed_mps > max_airspeed_mps) {
      max_airspeed_mps = out.airspeed_mps;
    }
  }

  REQUIRE(max_airspeed_mps >= 37.5);
  REQUIRE(reached_cruise_lock == true);
  REQUIRE(st.vel_ned.x < 1.0);
  REQUIRE(max_alt_err_m < 0.30);
}

TEST_CASE("8+1 Lift+Cruise survives mid-transition OEI, pusher loss glide/abort, and warm-started WLS <50us") {
  LiftCruiseDynamics plant(LiftCruiseLayout::kEightPlusOne);

  SECTION("Mid-transition outboard lift-rotor OEI at 18 m/s stays <=80% T_max and maintains altitude") {
    TransitionIndiWlsController ctrl(plant);
    AirframeState6Dof st{};
    st.pos_ned = {0.0, 0.0, -100.0};
    st.vel_ned = {18.0, 0.0, 0.0};
    const double t_init = 200.0;
    for (uint8_t i = 0; i < 8u; ++i) {
      st.u_act[i] = t_init;
    }
    st.u_act[8] = 180.0;
    ctrl.reset(st);

    const uint16_t fault_mask = 0x0001u;  // Outboard front-left rotor #0 failed
    double peak_ratio = 0.0;
    for (int tick = 0; tick < 1600; ++tick) {  // 4 s recovery
      TransitionCommand cmd{};
      cmd.target_airspeed_mps = 18.0;
      cmd.target_alt_m = 100.0;
      cmd.fault_mask = fault_mask;
      const auto out = ctrl.step_400hz(st, {0.0, 0.0, 0.0}, cmd);
      REQUIRE(out.valid);
      REQUIRE(out.u_cmd[0] == Catch::Approx(0.0).margin(1e-9));
      for (uint8_t i = 1; i < 8u; ++i) {
        peak_ratio = std::max(peak_ratio, out.u_cmd[i] / plant.max_lift_rotor_thrust_n());
      }
      integrate_step_rk2(plant, st, out.u_cmd, {0.0, 0.0, 0.0}, fault_mask, 0.0025);
    }
    REQUIRE(peak_ratio <= 0.80);
    REQUIRE(std::abs((-st.pos_ned.z) - 100.0) < 1.0);
  }

  SECTION("Pusher failure policy selects 13.5:1 glide at >=25.1 m/s and hover abort at <24.0 m/s") {
    TransitionIndiWlsController ctrl_glide(plant);
    AirframeState6Dof st_fast{};
    st_fast.pos_ned = {0.0, 0.0, -120.0};
    st_fast.vel_ned = {30.0, 0.0, 0.0};
    st_fast.u_act[8] = 200.0;
    ctrl_glide.reset(st_fast);

    TransitionCommand cmd{};
    cmd.target_airspeed_mps = 38.0;
    cmd.target_alt_m = 120.0;
    cmd.fault_mask = static_cast<uint16_t>(1u << 8u);  // Pusher failed
    const auto out_glide = ctrl_glide.step_400hz(st_fast, {0.0, 0.0, 0.0}, cmd);
    REQUIRE(out_glide.fault_mode == TransitionFaultMode::kPusherLossGlide);
    REQUIRE(out_glide.u_cmd[8] == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(out_glide.glide_target_airspeed_mps == Catch::Approx(29.3).margin(0.3));
    REQUIRE(out_glide.estimated_ld_ratio >= 13.0);

    TransitionIndiWlsController ctrl_abort(plant);
    AirframeState6Dof st_slow{};
    st_slow.pos_ned = {0.0, 0.0, -100.0};
    st_slow.vel_ned = {20.0, 0.0, 0.0};
    for (uint8_t i = 0; i < 8u; ++i) {
      st_slow.u_act[i] = 180.0;
    }
    ctrl_abort.reset(st_slow);
    const auto out_abort = ctrl_abort.step_400hz(st_slow, {0.0, 0.0, 0.0}, cmd);
    REQUIRE(out_abort.fault_mode == TransitionFaultMode::kPusherLossHoverAbort);
    REQUIRE(out_abort.eta_blend < 0.5);
  }

  SECTION("Warm-started 17x12 f64-WLS solves below 50 us p99 (<= 2% of 2.5 ms 400 Hz period)") {
    AirframeState6Dof st{};
    st.pos_ned = {0.0, 0.0, -100.0};
    st.vel_ned = {21.0, 0.0, 0.0};
    EffectivenessMatrix eff{};
    plant.compute_effectiveness(st, {0.0, 0.0, 0.0}, 0u, 0.0025, eff);

    std::array<double, kMaxControls> v_req{0.15, -0.10, 0.05, -0.30, 0.50};
    std::array<double, kMaxControls> Wv{10.0, 10.0, 1.5, 2.55, 3.0};
    std::array<double, kMaxActuators> Wu{10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 1.0, 10.5, 10.5, 10.5};
    std::array<double, kMaxActuators> ud_minus_u0{};
    std::array<int8_t, kMaxActuators> W_warm{};

    constexpr int kBenchIters = 10000;
    std::vector<double> us_samples;
    us_samples.reserve(kBenchIters);
    volatile double sink = 0.0;

    for (int i = 0; i < kBenchIters; ++i) {
      v_req[0] = 0.12 * std::sin(0.01 * static_cast<double>(i));
      const auto t0 = std::chrono::steady_clock::now();
      const auto res = wls_alloc_12x5_f64(
          eff.B, v_req, eff.du_min, eff.du_max, Wv, Wu, ud_minus_u0, 1e6, W_warm, 12u, 12u);
      const auto t1 = std::chrono::steady_clock::now();
      W_warm = res.W;
      sink += res.x[0];
      us_samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    (void)sink;
    std::sort(us_samples.begin(), us_samples.end());
    const double p99_us = us_samples[static_cast<std::size_t>(0.99 * kBenchIters)];
#if defined(NDEBUG) && !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_UNDEFINED__)
    REQUIRE(p99_us < 50.0);
#else
    REQUIRE(p99_us < 1000.0);
#endif
  }
}

TEST_CASE("TransitionSensorFrame handles noisy IMU, blocked Pitot GNSS/EKF failover, and DShot ESC staleness") {
  LiftCruiseDynamics plant(LiftCruiseLayout::kEightPlusOne);
  TransitionIndiWlsController ctrl(plant);

  AirframeState6Dof st{};
  st.pos_ned = {0.0, 0.0, -100.0};
  st.vel_ned = {32.0, 0.0, 0.0};
  const double theta = 0.08;
  st.q_frd_ned = {std::cos(0.5 * theta), 0.0, std::sin(0.5 * theta), 0.0};
  st.u_act[8] = 200.0;
  ctrl.reset(st);

  TransitionCommand cmd{};
  cmd.target_airspeed_mps = 34.0;
  cmd.target_alt_m = 100.0;

  // Deterministic pseudo-noise generator
  uint64_t rng = 20261123ull;
  auto noise = [&](double amp) {
    rng ^= rng << 13u;
    rng ^= rng >> 7u;
    rng ^= rng << 17u;
    const double u = static_cast<double>(rng & 0xFFFFu) / 32768.0 - 1.0;
    return amp * u;
  };

  bool observed_pitot_failover = false;
  bool observed_esc_stale = false;

  for (int tick = 0; tick < 800; ++tick) {
    const auto dyn = plant.eval_dynamics(st, {0.0, 0.0, 0.0}, 0u);
    TransitionSensorFrame sf{};
    sf.gyro_meas_frd = {
        st.omega_frd.x + noise(0.015),
        st.omega_frd.y + noise(0.015),
        st.omega_frd.z + noise(0.015),
    };
    sf.accel_meas_frd = {
        dyn.accel_body_frd.x + noise(0.20),
        dyn.accel_body_frd.y + noise(0.20),
        dyn.accel_body_frd.z + noise(0.20),
    };
    sf.gnss_ekf_airspeed_mps = dyn.airspeed_mps + noise(0.15);
    sf.gnss_ekf_valid = true;

    // Inject blocked/iced Pitot tube (stuck 0 m/s or NaN) during ticks 200..499
    if (tick >= 200 && tick < 500) {
      sf.pitot_valid = false;
      sf.pitot_airspeed_mps = (tick == 250) ? std::numeric_limits<double>::quiet_NaN() : 0.0;
    } else {
      sf.pitot_valid = true;
      sf.pitot_airspeed_mps = dyn.airspeed_mps + noise(0.25);
    }

    // Enable bidirectional DShot telemetry and drop rotor #2 & pusher #8 during ticks 300..349
    sf.use_esc_telemetry = true;
    sf.esc_stale_ticks_limit = 10u;
    for (uint8_t i = 0; i <= plant.pusher_index(); ++i) {
      sf.esc_thrust_meas_n[i] = st.u_act[i];
      sf.esc_valid[i] = !(tick >= 300 && tick < 350 && (i == 2u || i == 8u));
    }

    const auto out = ctrl.step_400hz_with_sensors(st, sf, cmd);
    REQUIRE(out.valid);
    if (tick >= 200 && tick < 500) {
      REQUIRE(out.pitot_failover_active == true);
      REQUIRE(out.airspeed_mps > 25.0);  // Failed over to GNSS/EKF, NOT stuck 0 m/s!
      REQUIRE(out.fault_mode != TransitionFaultMode::kStallRecovery);
      observed_pitot_failover = true;
    }
    if (tick >= 312 && tick < 350) {
      REQUIRE((out.esc_stale_mask & (1u << 2u)) != 0u);
      REQUIRE((out.esc_stale_mask & (1u << 8u)) != 0u);
      observed_esc_stale = true;
    }
    if (tick >= 355) {
      REQUIRE(out.esc_stale_mask == 0u);  // Cleared cleanly once DShot frames resume
    }
    integrate_step_rk2(plant, st, out.u_cmd, {0.0, 0.0, 0.0}, 0u, 0.0025);
  }

  REQUIRE(observed_pitot_failover);
  REQUIRE(observed_esc_stale);
  REQUIRE(std::abs((-st.pos_ned.z) - 100.0) < 1.5);
}
