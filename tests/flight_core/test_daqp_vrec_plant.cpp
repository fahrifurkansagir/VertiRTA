#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/core/daqp_qp.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/vrta_recorder.hpp"
#include "zero_heap_guard.hpp"

using Catch::Matchers::WithinAbs;
using namespace vertirta::core;
using vertirta::test::ScopedZeroHeapGuard;

namespace {

double quat_tilt_deg(const QuatPx4FrdNed& q) {
  // Body -Z_FRD rotated into NED has z-component -(1 - 2*(x^2 + y^2))
  const double cos_tilt = clamp_d(1.0 - 2.0 * (q.x * q.x + q.y * q.y), -1.0, 1.0);
  return std::acos(cos_tilt) * kRadToDeg;
}

struct SimSummary {
  double max_tilt_deg{0.0};
  double touchdown_vz_mps{0.0};
  double touchdown_tilt_deg{0.0};
  bool landed{false};
  uint64_t frame_hash{0};
  VrtaFrame100Hz last_frame{};
};

SimSummary run_same_arm_descent(bool allow_active_spin) {
  CoaxialPlant6Dof plant;
  plant.reset({0.0, 0.0, -35.0}, {0.0, 0.0, 0.0}, 0.50f);

  IndiWlsController ctrl(allow_active_spin);
  ctrl.reset(params::kOmegaHover);

  SimSummary s{};
  // 64-bit FNV-1a digest over serialized 256 B VrtaFrame100Hz stream for 0-ULP determinism check
  uint64_t h = 14695981039346656037ULL;
  uint64_t seq = 0;

  constexpr int kTotalTicks = 25 * 400;  // 25 s @ 400 Hz (covers full 35 m scheduled descent & flare)
  for (int k = 0; k < kTotalTicks; ++k) {
    const double t = static_cast<double>(k) * 0.0025;
    const uint8_t fault_mask = (t >= 1.0) ? 0x11u : 0x00u;  // Same-arm M0+M4 cut at t=1.0 s
    // 5 m/s coastal gust toward FR Arm 0 (+N, +E)
    const Vec3Ned wind_ned{3.5355, 3.5355, 0.0};

    const PlantState st = plant.state();
    const double agl = -st.pos_ned.z;

    // Scheduled Regime-3 vertical sink profile: <=2.5 m/s >10m, <=1.5 <5m, <=0.8 touchdown
    double vz_des_down = 0.0;
    if (fault_mask != 0u) {
      if (agl > 10.0) {
        vz_des_down = 2.2;
      } else if (agl > 5.0) {
        vz_des_down = 1.5;
      } else if (agl > 1.0) {
        vz_des_down = 0.9;
      } else {
        vz_des_down = 0.5;
      }
    }
    const double az_cmd = clamp_d(2.5 * (vz_des_down - st.vel_ned.z), -2.2, 3.0);

    IndiInput in{};
    in.q_frd_ned = st.q_frd_ned;
    in.gyro_frd = st.omega_frd;
    in.accel_des_ned = {0.0, 0.0, az_cmd};
    in.yaw_rate_des = 0.0;
    in.v_horiz_mps = static_cast<float>(std::hypot(st.vel_ned.x, st.vel_ned.y));
    in.v_term_v = st.battery.v_term_v;
    in.fault_mask = fault_mask;

    float omega_cmd[8]{};
    (void)ctrl.step_400hz(in, omega_cmd);
    const PlantState next = plant.step_400hz(omega_cmd, fault_mask, wind_ned);

    const double tilt = quat_tilt_deg(next.q_frd_ned);
    if (tilt > s.max_tilt_deg) {
      s.max_tilt_deg = tilt;
    }

    if (k % 4 == 0) {
      VrtaFrame100Hz fr{};
      fr.seq = seq++;
      fr.t_sim_us = static_cast<uint64_t>(k) * 2500ULL;
      const Vec3Enu pos_enu = ned_to_enu(next.pos_ned);
      const Vec3Enu vel_enu = ned_to_enu(next.vel_ned);
      fr.pos_enu[0] = pos_enu.x;
      fr.pos_enu[1] = pos_enu.y;
      fr.pos_enu[2] = pos_enu.z;
      fr.vel_enu[0] = static_cast<float>(vel_enu.x);
      fr.vel_enu[1] = static_cast<float>(vel_enu.y);
      fr.vel_enu[2] = static_cast<float>(vel_enu.z);
      fr.q_frd_ned[0] = static_cast<float>(next.q_frd_ned.w);
      fr.q_frd_ned[1] = static_cast<float>(next.q_frd_ned.x);
      fr.q_frd_ned[2] = static_cast<float>(next.q_frd_ned.y);
      fr.q_frd_ned[3] = static_cast<float>(next.q_frd_ned.z);
      fr.v_term_v = next.battery.v_term_v;
      fr.soc_pct = next.battery.soc * 100.0f;
      fr.agl_m = static_cast<float>(-next.pos_ned.z);
      fr.active_motor_mask = static_cast<uint8_t>(~fault_mask);
      fr.flight_regime = static_cast<uint8_t>(CoaxEffectiveness::classify_regime(fault_mask));
      fr.cel_latched = (fault_mask == 0x11u) ? 1u : 0u;
      finalize_vrta_frame(fr);
      REQUIRE(verify_vrta_frame(fr));

      const auto* bytes = reinterpret_cast<const uint8_t*>(&fr);
      for (std::size_t b = 0; b < sizeof(fr); ++b) {
        h ^= bytes[b];
        h *= 1099511628211ULL;
      }
      s.last_frame = fr;
    }

    if (next.on_ground && t > 2.0) {
      s.landed = true;
      s.touchdown_vz_mps = std::abs(next.touchdown_vz_mps);
      s.touchdown_tilt_deg = tilt;
      break;
    }
  }
  s.frame_hash = h;
  return s;
}

}  // namespace

TEST_CASE("StaticDaqpSolver fits <=6400 B, solves n=5,m=28 QP with 0 heap allocs, and guards NaN/infeasible") {
  static_assert(sizeof(StaticDaqpSolver<6, 32>) <= 6400, "StaticDaqpSolver exceeds 6.4 KiB static budget");
  static_assert(!std::is_copy_constructible_v<StaticDaqpSolver<6, 32>>, "Must be non-copyable (self-pointers)");
  static_assert(!std::is_copy_assignable_v<StaticDaqpSolver<6, 32>>, "Must be non-copy-assignable");
  static_assert(!std::is_move_constructible_v<StaticDaqpSolver<6, 32>>, "Must be non-movable (self-pointers)");
  static_assert(!std::is_move_assignable_v<StaticDaqpSolver<6, 32>>, "Must be non-move-assignable");
  static_assert(!std::is_copy_constructible_v<StaticDaqpSolver<5, 32>>, "Must be non-copyable");
  static_assert(!std::is_move_constructible_v<StaticDaqpSolver<5, 32>>, "Must be non-movable");
  StaticDaqpSolver<6, 32> solver;

  // Feasible 5-var QP: min 0.5 * sum((x_i - target_i)^2) s.t. x_0 + x_1 <= 1.0, box [-5, 5], 23 extra slack bounds
  const double H_diag[5] = {1.0, 1.0, 2.0, 1.0, 1e4};
  const double f[5] = {-2.0, -2.0, 0.0, 0.0, 1e3};  // Unconstrained optimum at x0=2, x1=2 -> active constraint x0+x1<=1
  const double x_lo[5] = {-5.0, -5.0, -5.0, -1.0, 0.0};
  const double x_hi[5] = {+5.0, +5.0, +5.0, +1.0, 100.0};
  double A[23 * 5]{};
  double b_lo[23]{};
  double b_hi[23]{};
  A[0] = 1.0;
  A[1] = 1.0;
  b_lo[0] = -1e6;
  b_hi[0] = 1.0;
  for (int r = 1; r < 23; ++r) {
    A[r * 5 + (r % 3)] = 1.0;
    b_lo[r] = -10.0;
    b_hi[r] = +10.0;
  }

  DaqpSolveResult res{};
  {
    ScopedZeroHeapGuard guard;
    res = solver.solve(5, 23, H_diag, f, A, x_lo, x_hi, b_lo, b_hi, false);
    REQUIRE(guard.alloc_count() == 0);
  }
  REQUIRE_FALSE(res.used_fallback);
  REQUIRE_THAT(res.x[0], WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(res.x[1], WithinAbs(0.5, 1e-6));
  REQUIRE_THAT(res.x[4], WithinAbs(0.0, 1e-6));

  // Warm-start re-solve -> 0 heap allocs
  {
    ScopedZeroHeapGuard guard;
    const DaqpSolveResult res2 = solver.solve(5, 23, H_diag, f, A, x_lo, x_hi, b_lo, b_hi, true);
    REQUIRE(guard.alloc_count() == 0);
    REQUIRE_FALSE(res2.used_fallback);
  }

  // NaN input -> deterministic fallback without crash
  double f_nan[5] = {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, 0.0};
  const DaqpSolveResult res_nan = solver.solve(5, 23, H_diag, f_nan, A, x_lo, x_hi, b_lo, b_hi, false);
  REQUIRE(res_nan.used_fallback);

  // Infeasible contradictory box bounds (x0 >= 4 and x0 <= -4) -> pre-check fallback
  const double x_lo_bad[5] = {4.0, -5.0, -5.0, -1.0, 0.0};
  const double x_hi_bad[5] = {-4.0, 5.0, 5.0, 1.0, 100.0};
  const DaqpSolveResult res_inf = solver.solve(5, 23, H_diag, f, A, x_lo_bad, x_hi_bad, b_lo, b_hi, false);
  REQUIRE(res_inf.used_fallback);

  // Genuine general-constraint infeasibility entering daqp_ldp (x0+x1 <= 0 AND x0+x1 >= 2, valid box)
  const double A_conflict[2 * 5] = {
      1.0, 1.0, 0.0, 0.0, 0.0,
      1.0, 1.0, 0.0, 0.0, 0.0,
  };
  const double b_lo_conflict[2] = {-1e6, 2.0};
  const double b_hi_conflict[2] = {0.0, +1e6};
  {
    ScopedZeroHeapGuard guard;
    const DaqpSolveResult res_gen_inf =
        solver.solve(5, 2, H_diag, f, A_conflict, x_lo, x_hi, b_lo_conflict, b_hi_conflict, false);
    REQUIRE(guard.alloc_count() == 0);
    REQUIRE(res_gen_inf.used_fallback);
    REQUIRE(res_gen_inf.iterations >= 1);
    REQUIRE(res_gen_inf.exit_flag < 0);
  }
}

TEST_CASE("Vrta POD header/frame layouts match exact 128B/256B contract and RFC 3720 CRC32C") {
  static_assert(sizeof(VrtaFileHeader) == 128 && std::is_trivially_copyable_v<VrtaFileHeader>);
  static_assert(sizeof(VrtaFrame100Hz) == 256 && std::is_trivially_copyable_v<VrtaFrame100Hz>);
  static_assert(offsetof(VrtaFileHeader, datum_lat_deg) == 16);
  static_assert(offsetof(VrtaFileHeader, lut_sha256) == 48);
  static_assert(offsetof(VrtaFileHeader, git_sha) == 80);
  static_assert(offsetof(VrtaFileHeader, header_crc32c) == 124);
  static_assert(offsetof(VrtaFrame100Hz, pos_enu) == 16);
  static_assert(offsetof(VrtaFrame100Hz, rotor_rpm) == 104);
  static_assert(offsetof(VrtaFrame100Hz, active_motor_mask) == 164);
  static_assert(offsetof(VrtaFrame100Hz, flags) == 168);
  static_assert(offsetof(VrtaFrame100Hz, crc32c) == 252);

  // RFC 3720 Castagnoli CRC32C standard check vector
  const uint8_t check_vec[9] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
  REQUIRE(crc32c(check_vec, 9) == 0xE3069283u);

  VrtaFileHeader hdr = make_default_vrta_header("0123456789abcdef0123456789abcdef01234567");
  REQUIRE(verify_vrta_header(hdr));
  hdr.datum_lat_deg += 0.001;
  REQUIRE_FALSE(verify_vrta_header(hdr));
}

TEST_CASE("6-DOF Same-Arm M0+M4 loss capsizes under Option A and lands safely with 0-ULP replay under Option B1b") {
  const SimSummary opt_a = run_same_arm_descent(false);
  REQUIRE(opt_a.max_tilt_deg > 60.0);  // Uncontrolled unilateral capsize verified!

  const SimSummary opt_b1 = run_same_arm_descent(true);
  REQUIRE(opt_b1.landed);
  REQUIRE(opt_b1.touchdown_vz_mps <= 0.85);
  REQUIRE(opt_b1.touchdown_tilt_deg <= 15.0);

  // Bit-exact deterministic rerun (0-ULP identical serialized .vrta frames)
  const SimSummary opt_b2 = run_same_arm_descent(true);
  REQUIRE(opt_b1.frame_hash == opt_b2.frame_hash);
}
