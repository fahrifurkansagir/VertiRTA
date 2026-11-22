// Freestanding zero-heap runtime test exercising EVERY component of libvertirta_flight_core:
// frames, geodesy, TmrVoter, CoaxEffectiveness, TheveninBattery1RC, IndiWlsController,
// MotorFdiObserver, StaticDaqpSolver<5,32>, CoaxialPlant6Dof, and VrtaRecorder (128B/256B POD).
// Linked with non-inline no_heap_link_trap.cpp and per-target -Wl,--wrap=malloc/calloc/realloc.

#include <cmath>
#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/daqp_qp.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta/core/vrta_recorder.hpp"

namespace {

using vertirta::core::BatteryState;
using vertirta::core::CapabilityEnvelope;
using vertirta::core::CoaxEffectiveness;
using vertirta::core::CoaxialPlant6Dof;
using vertirta::core::DaqpSolveResult;
using vertirta::core::FlightRegime;
using vertirta::core::GeodeticWgs84;
using vertirta::core::ImuSampleFrd;
using vertirta::core::IndiInput;
using vertirta::core::IndiWlsController;
using vertirta::core::MotorFdiObserver;
using vertirta::core::PlantState;
using vertirta::core::QuatPx4FrdNed;
using vertirta::core::StaticDaqpSolver;
using vertirta::core::TheveninBattery1RC;
using vertirta::core::TmrConfig;
using vertirta::core::TmrState;
using vertirta::core::TmrVoteOutput;
using vertirta::core::TmrVoter;
using vertirta::core::Vec3Enu;
using vertirta::core::Vec3Frd;
using vertirta::core::Vec3Ned;
using vertirta::core::VrtaFileHeader;
using vertirta::core::VrtaFrame100Hz;
using vertirta::core::WlsResult;

// Static BSS arenas mirror embedded flight-controller allocation (zero heap, zero stack bloat).
StaticDaqpSolver<5, 32> g_daqp_solver;
IndiWlsController g_indi_ctrl{true};
CoaxialPlant6Dof g_plant;
MotorFdiObserver g_fdi;
TmrVoter g_tmr{TmrConfig{}};
TheveninBattery1RC g_batt{0.80f};

bool near(double a, double b, double tol) noexcept {
  return std::fabs(a - b) <= tol;
}

int test_frames_and_geodesy() noexcept {
  const Vec3Enu enu{120.5, -340.25, 45.0};
  const Vec3Enu round_enu = vertirta::core::ned_to_enu(vertirta::core::enu_to_ned(enu));
  if (!near(enu.x, round_enu.x, 1e-12) || !near(enu.y, round_enu.y, 1e-12) ||
      !near(enu.z, round_enu.z, 1e-12)) {
    return 1;
  }

  const QuatPx4FrdNed q_in{0.9238795325112867, 0.1, -0.2, 0.3249196962329063};
  const QuatPx4FrdNed q_out = vertirta::core::ros_to_px4_quat(vertirta::core::px4_to_ros_quat(q_in));
  if (!near(q_in.w, q_out.w, 1e-12) || !near(q_in.x, q_out.x, 1e-12) ||
      !near(q_in.y, q_out.y, 1e-12) || !near(q_in.z, q_out.z, 1e-12)) {
    return 2;
  }

  const double n_kartal = vertirta::core::egm2008_undulation_m(40.8860, 29.1880);
  if (!(n_kartal > 36.4 && n_kartal < 37.0)) {
    return 3;
  }

  const GeodeticWgs84 g0{40.9125, 29.1460, 120.0};
  const Vec3Enu local = vertirta::core::geodetic_to_local_enu(g0);
  const GeodeticWgs84 g1 = vertirta::core::local_enu_to_geodetic(local);
  if (!near(g0.lat_deg, g1.lat_deg, 1e-7) || !near(g0.lon_deg, g1.lon_deg, 1e-7) ||
      !near(g0.h_hae_m, g1.h_hae_m, 1e-3)) {
    return 4;
  }
  return 0;
}

int test_tmr_400hz_isolation_and_nan_guard() noexcept {
  g_tmr.reset();
  const Vec3Frd nominal_gyro{0.01, -0.02, 0.005};
  const Vec3Frd nominal_accel{0.0, 0.0, -9.80665};

  TmrVoteOutput out{};
  for (uint64_t tick = 0; tick < 20; ++tick) {
    ImuSampleFrd imus[3]{
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
    };
    out = g_tmr.step(imus, nominal_gyro, true);
  }
  if (out.state != TmrState::TMR_3OF3 || out.health_mask != 0x07U) {
    return 5;
  }

  for (uint64_t tick = 20; tick < 36; ++tick) {
    ImuSampleFrd imus[3]{
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {nominal_gyro, nominal_accel, tick * 2500ULL, true},
        {{1.25, -0.80, 0.50}, nominal_accel, tick * 2500ULL, true},
    };
    out = g_tmr.step(imus, nominal_gyro, true);
  }
  if (out.state != TmrState::DUPLEX_2OF2 || out.health_mask != 0x03U || out.isolation_latency_ms > 20U) {
    return 6;
  }

  ImuSampleFrd nan_imus[3]{
      {nominal_gyro, nominal_accel, 90000ULL, true},
      {{NAN, 0.0, 0.0}, nominal_accel, 90000ULL, true},
      {{1.25, -0.80, 0.50}, nominal_accel, 90000ULL, true},
  };
  out = g_tmr.step(nan_imus, nominal_gyro, true);
  if (!out.gyro_voted.is_finite() || !out.accel_voted.is_finite()) {
    return 7;
  }
  return 0;
}

int test_coax_envelope_battery_indi_fdi_and_plant() noexcept {
  // Verify coaxial regime classification & upper-loss lower-rotor unblanking (+17.6%)
  if (CoaxEffectiveness::classify_regime(0x00U) != FlightRegime::NOMINAL ||
      CoaxEffectiveness::classify_regime(0x01U) != FlightRegime::SINGLE_LOSS ||
      CoaxEffectiveness::classify_regime(0x11U) != FlightRegime::SAME_ARM_ACTIVE_SPIN) {
    return 10;
  }
  double B[4][8]{};
  float u_max[8]{};
  CoaxEffectiveness::build_B(0x01U, 0.0f, B, u_max, 395.0f);
  if (!(u_max[0] == 0.0f && u_max[4] > 780.0f)) {
    return 11;
  }

  g_batt.reset(0.80f);
  const BatteryState bstate = g_batt.step(0.0025f, 38000.0f);
  if (!(bstate.v_term_v > 350.0f && bstate.v_term_v < 475.0f && !bstate.voltage_collapse)) {
    return 12;
  }

  const CapabilityEnvelope env = CoaxEffectiveness::compute_envelope(
      0x01U, 0U, 0x07U, bstate.v_term_v, bstate.i_pack_a, bstate.soc, 0.0f, 0.0f, 0.5f, 40U);
  if (env.regime != FlightRegime::SINGLE_LOSS || env.t_max_n <= 0.0f) {
    return 13;
  }

  // Closed-loop 400 Hz INDI+WLS + MotorFdiObserver + CoaxialPlant6Dof (1M fault on rotor 0)
  g_indi_ctrl.reset();
  g_fdi.reset();
  g_plant.reset({0.0, 0.0, -35.0}, {0.0, 0.0, 0.0}, 0.80f);

  float omega_cmd[8]{};
  float omega_hat[8]{};
  float omega_esc[8]{};
  bool esc_valid[8]{true, true, true, true, true, true, true, true};
  uint8_t detected_mask = 0x00U;

  for (int tick = 0; tick < 120; ++tick) {
    const uint8_t phys_fault = (tick >= 10) ? 0x01U : 0x00U;
    const PlantState& st = g_plant.state();
    for (int i = 0; i < 8; ++i) {
      omega_esc[i] = static_cast<float>(st.rotor_omega[i]);
    }
    IndiInput in{};
    in.q_frd_ned = st.q_frd_ned;
    in.gyro_frd = st.omega_frd;
    in.accel_des_ned = {0.0, 0.0, 0.0};
    in.yaw_rate_des = 0.0;
    in.v_horiz_mps = 0.0f;
    in.v_term_v = st.battery.v_term_v;
    in.fault_mask = detected_mask;
    in.esc_omega_rad_s = nullptr;
    in.esc_valid = nullptr;

    const WlsResult wls = g_indi_ctrl.step_400hz(in, omega_cmd);
    if (wls.cholesky_failed) {
      return 14;
    }
    g_indi_ctrl.get_omega_hat(omega_hat);
    detected_mask = g_fdi.step_400hz(
        g_indi_ctrl.filtered_gyro_frd(),
        g_indi_ctrl.filtered_dgyro_frd(),
        -9.80665f,
        omega_hat,
        omega_esc,
        esc_valid,
        0.0f,
        st.battery.v_term_v,
        detected_mask);
    (void)g_plant.step_400hz(omega_cmd, phys_fault, {1.0, -0.5, 0.0});
  }
  if ((detected_mask & 0x01U) == 0U || !g_plant.state().pos_ned.is_finite()) {
    return 15;
  }
  return 0;
}

int test_daqp_solver_and_vrta_recorder() noexcept {
  g_daqp_solver.reset_warm_start();
  const double H_diag[5] = {1.0, 1.0, 1.0, 100.0, 100.0};
  const double f[5] = {-2.0, 1.0, -0.5, 0.0, 0.0};
  const double A[5] = {1.0, 1.0, 0.0, -1.0, 0.0};
  const double x_lo[5] = {-5.0, -5.0, -5.0, 0.0, 0.0};
  const double x_hi[5] = {5.0, 5.0, 5.0, 10.0, 10.0};
  const double b_lo[1] = {-1e6};
  const double b_hi[1] = {0.5};

  const DaqpSolveResult qp = g_daqp_solver.solve(5, 1, H_diag, f, A, x_lo, x_hi, b_lo, b_hi, false);
  if (qp.used_fallback || qp.exit_flag < 1 || !near(qp.x[0] + qp.x[1] - qp.x[3], 0.5, 1e-3)) {
    return 20;
  }

  // Verify RFC 3720 CRC32C check vector ("123456789" -> 0xE3069283) + 128 B / 256 B POD framing
  const char check_vec[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  if (vertirta::core::crc32c(check_vec, 9) != 0xE3069283U) {
    return 21;
  }

  VrtaFileHeader hdr = vertirta::core::make_default_vrta_header("0123456789abcdef0123456789abcdef01234567");
  vertirta::core::finalize_vrta_header(hdr);
  if (!vertirta::core::verify_vrta_header(hdr)) {
    return 22;
  }

  VrtaFrame100Hz frame{};
  frame.seq = 1U;
  frame.t_sim_us = 10000U;
  frame.pos_enu[0] = 10.0;
  frame.pos_enu[1] = -20.0;
  frame.pos_enu[2] = 35.0;
  frame.active_motor_mask = 0xFEU;
  frame.flight_regime = static_cast<uint8_t>(FlightRegime::SINGLE_LOSS);
  vertirta::core::finalize_vrta_frame(frame);
  if (!vertirta::core::verify_vrta_frame(frame)) {
    return 23;
  }
  return 0;
}

}  // namespace

int main() {
  if (const int rc = test_frames_and_geodesy(); rc != 0) {
    return rc;
  }
  if (const int rc = test_tmr_400hz_isolation_and_nan_guard(); rc != 0) {
    return rc;
  }
  if (const int rc = test_coax_envelope_battery_indi_fdi_and_plant(); rc != 0) {
    return rc;
  }
  if (const int rc = test_daqp_solver_and_vrta_recorder(); rc != 0) {
    return rc;
  }
  return 0;
}
