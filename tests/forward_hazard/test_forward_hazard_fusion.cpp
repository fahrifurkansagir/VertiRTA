#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>

#include "vertirta/forward_hazard/forward_hazard_fusion.hpp"
#include "vertirta/forward_hazard/horizon_split_stack.hpp"
#include "vertirta/forward_hazard/onnx_forward_detectors.hpp"
#include "vertirta/forward_hazard/se3_pose_ring_buffer.hpp"
#include "vertirta/forward_hazard/wire_span_ekf.hpp"

using Catch::Approx;
using namespace vertirta::forward_hazard;

namespace {

constexpr double kPi = 3.14159265358979323846;

vertirta::core::QuatPx4FrdNed yaw_quat(double yaw_rad) {
  return {std::cos(0.5 * yaw_rad), 0.0, 0.0, std::sin(0.5 * yaw_rad)};
}

}  // namespace

TEST_CASE("HorizonSplitStackPreprocessor 1280x320 -> 640x640 round-trip and nadir lock",
          "[forward_hazard][split_stack]") {
  HorizonSplitStackPreprocessor pre{};

  // Verify locked nadir SLZ constants are strictly separate from 45 deg forward camera.
  STATIC_REQUIRE(ForwardCameraIntrinsics::kLockedNadirSlzWidthPx == 512u);
  STATIC_REQUIRE(ForwardCameraIntrinsics::kLockedNadirSlzHeightPx == 512u);
  STATIC_REQUIRE(ForwardCameraIntrinsics::kLockedNadirSlzHfovDeg == 80.0);
  REQUIRE(pre.config().intrinsics.width_px == 1280u);
  REQUIRE(pre.config().intrinsics.height_px == 720u);
  REQUIRE(pre.config().intrinsics.hfov_deg == Approx(45.0));
  REQUIRE(pre.config().intrinsics.fx_px == Approx(1545.1));

  // Verify < 0.5 px round-trip across left half, right half, and 16 px overlap seam under IMU roll/pitch.
  const double roll_rad = 8.0 * kPi / 180.0;
  const double pitch_rad = -3.5 * kPi / 180.0;

  const double test_u[] = {120.0, 510.0, 636.0, 644.0, 820.0, 1180.0};
  const double test_v[] = {290.0, 340.0, 355.0, 360.0, 380.0, 420.0};

  for (int i = 0; i < 6; ++i) {
    const auto st = pre.native_to_stacked(test_u[i], test_v[i], roll_rad, pitch_rad, false);
    REQUIRE(st.valid);
    const auto rt = pre.stacked_to_native(st.x_stack, st.y_stack, roll_rad, pitch_rad);
    REQUIRE(rt.valid);
    REQUIRE(std::hypot(rt.u_px - test_u[i], rt.v_px - test_v[i]) < 0.5);
  }

  // Seam pixel at u=640 (with zero roll/pitch) lies in the 16 px overlap seam and maps to both halves cleanly.
  const auto seam_left = pre.native_to_stacked(640.0, 360.0, 0.0, 0.0, false);
  const auto seam_right = pre.native_to_stacked(640.0, 360.0, 0.0, 0.0, true);
  REQUIRE(seam_left.valid);
  REQUIRE(seam_right.valid);
  REQUIRE(seam_left.in_overlap_seam);
  REQUIRE(seam_right.in_overlap_seam);
  REQUIRE(seam_left.half_id == 0u);
  REQUIRE(seam_right.half_id == 1u);
  const auto back_l = pre.stacked_to_native(seam_left.x_stack, seam_left.y_stack, 0.0, 0.0);
  const auto back_r = pre.stacked_to_native(seam_right.x_stack, seam_right.y_stack, 0.0, 0.0);
  REQUIRE(std::abs(back_l.u_px - 640.0) < 1e-6);
  REQUIRE(std::abs(back_r.u_px - 640.0) < 1e-6);
}

TEST_CASE("Se3PoseRingBuffer 200 Hz SLERP accuracy and > 25 ms extrapolation rejection",
          "[forward_hazard][pose_ring]") {
  Se3PoseRingBuffer ring{};
  const uint64_t dt_ns = 5'000'000ULL;  // 200 Hz

  for (int i = 0; i <= 420; ++i) {
    const double t = 0.005 * static_cast<double>(i);
    TimedPoseSe3 p{};
    p.stamp_ns = 1'000'000'000ULL + static_cast<uint64_t>(i) * dt_ns;
    p.pos_ned = {10.0 * t, 2.0 * t, -40.0};
    p.vel_ned = {10.0, 2.0, 0.0};
    p.q_frd_to_ned = yaw_quat(0.4 * t);
    p.valid = true;
    REQUIRE(ring.push(p));
  }
  REQUIRE(ring.size() == Se3PoseRingBuffer::kCapacity);

  // Midpoint interpolation between i=300 (t=1.500 s) and i=301 (t=1.505 s) -> t=1.5025 s
  const uint64_t q_ns = 1'000'000'000ULL + 300ULL * dt_ns + 2'500'000ULL;
  TimedPoseSe3 out{};
  REQUIRE(ring.query(q_ns, out));
  REQUIRE(out.pos_ned.x == Approx(15.025).margin(1e-4));
  REQUIRE(out.pos_ned.y == Approx(3.005).margin(1e-4));

  const auto q_exp = yaw_quat(0.4 * 1.5025);
  const double dot_q = std::abs(out.q_frd_to_ned.w * q_exp.w + out.q_frd_to_ned.x * q_exp.x +
                                out.q_frd_to_ned.y * q_exp.y + out.q_frd_to_ned.z * q_exp.z);
  const double ang_err_rad = 2.0 * std::acos(std::min(1.0, dot_q));
  REQUIRE(ang_err_rad < 1e-4);

  // Reject extrapolation > 25 ms past newest timestamp (t=2.100 s)
  const uint64_t stale_future = 1'000'000'000ULL + 420ULL * dt_ns + 30'000'000ULL;
  REQUIRE_FALSE(ring.query(stale_future, out));
}

TEST_CASE("OnnxForwardDetectors enforces TU116 enable_fp16=false default and descent SLZ phase yield",
          "[forward_hazard][onnx]") {
  OnnxForwardDetectors det{};
  REQUIRE_FALSE(det.config().enable_fp16);
  REQUIRE(det.config().enable_cuda_graph);
  REQUIRE(det.config().vram_budget_mib <= 300u);

  ForwardSensorBundle b{};
  b.camera_healthy = true;
  b.wire_line_count = 2;
  // Two duplicate detections across the 16 px split-stack seam should merge into 1.
  b.wire_lines[0] = {600.0, 360.0, 680.0, 360.0, 0.80, 1u};
  b.wire_lines[1] = {602.0, 361.0, 682.0, 361.0, 0.92, 1u};

  ForwardDetectorRawOutput out{};
  REQUIRE(det.infer_from_bundle(b, out));
  REQUIRE_FALSE(out.phase_yielded_to_slz);
  REQUIRE(out.wire_line_count == 1u);
  REQUIRE(out.wire_lines[0].score == Approx(0.92));
  REQUIRE(out.estimated_vram_mib <= 300u);

  // During descent, yield GPU entirely to locked 512x512 PIDNet-S nadir SLZ.
  det.set_flight_phase(InferenceFlightPhase::kDescentNadirSlz);
  REQUIRE(det.infer_from_bundle(b, out));
  REQUIRE(out.phase_yielded_to_slz);
  REQUIRE(out.estimated_vram_mib == 0u);
}

TEST_CASE("ForwardHazardFusion fuses 30 Hz vision + 77 GHz radar, handles grazing cliff + 15 m/s cap, and rescues hovering drone",
          "[forward_hazard][fusion]") {
  ForwardHazardFusion fusion{};
  REQUIRE(fusion.config().vehicle_radius_m == Approx(2.10));
  REQUIRE(fusion.config().v_max_nominal_mps == Approx(22.0));
  REQUIRE(fusion.config().v_max_radar_degraded_mps == Approx(15.0));

  HorizonSplitStackPreprocessor cam{};
  // Ground-truth horizontal transmission wire at N=80 m, E in [-20, +20] m, D=-40 m (same altitude as aircraft).
  const vertirta::core::Vec3Ned wire_p0{80.0, -20.0, -40.0};
  const vertirta::core::Vec3Ned wire_p1{80.0, +20.0, -40.0};
  const vertirta::core::Vec3Ned hover_drone_ned{55.0, 4.0, -41.0};

  // Scenario A: Healthy radar + vision + hovering drone inside static Doppler gate rescued by D-FINE-N.
  ForwardHazardOutput out{};
  for (int k = 0; k < 5; ++k) {
    const uint64_t t_ns = 1'000'000'000ULL + static_cast<uint64_t>(k) * 33'333'333ULL;
    TimedPoseSe3 pose{};
    pose.stamp_ns = t_ns;
    pose.pos_ned = {static_cast<double>(k) * 0.5, 0.0, -40.0};
    pose.vel_ned = {15.0, 0.0, 0.0};
    pose.q_frd_to_ned = {1.0, 0.0, 0.0, 0.0};
    pose.valid = true;
    fusion.push_pose_200hz(pose);

    double u0 = 0, v0 = 0, u1 = 0, v1 = 0;
    const vertirta::core::Vec3Frd r0_frd{wire_p0.x - pose.pos_ned.x, wire_p0.y - pose.pos_ned.y, wire_p0.z - pose.pos_ned.z};
    const vertirta::core::Vec3Frd r1_frd{wire_p1.x - pose.pos_ned.x, wire_p1.y - pose.pos_ned.y, wire_p1.z - pose.pos_ned.z};
    REQUIRE(cam.project_frd_to_pixel(r0_frd, u0, v0));
    REQUIRE(cam.project_frd_to_pixel(r1_frd, u1, v1));

    double ud = 0, vd = 0;
    const vertirta::core::Vec3Frd drone_frd{
        hover_drone_ned.x - pose.pos_ned.x,
        hover_drone_ned.y - pose.pos_ned.y,
        hover_drone_ned.z - pose.pos_ned.z};
    REQUIRE(cam.project_frd_to_pixel(drone_frd, ud, vd));

    ForwardSensorBundle b{};
    b.frame_stamp_ns = t_ns;
    b.radar_stamp_ns = t_ns;
    b.radar_healthy = true;
    b.camera_healthy = true;
    b.wire_line_count = 1;
    b.wire_lines[0] = {u0, v0, u1, v1, 0.90, 1u};
    b.intruder_box_count = 1;
    b.intruder_boxes[0] = {ud, vd, 28.0, 20.0, 0.88, 0u};

    b.radar_point_count = 2;
    // Wire normal-incidence glint at midpoint (N=80, E=0, D=-40)
    b.radar_points[0] = {80.0 - pose.pos_ned.x, 0.0, 0.0, -15.0, 0.05, 19.5, 4.2,
                         RadarDopplerClass::kStaticOrHoverCandidate, true};
    // Hovering drone with |v_res| = 0.15 m/s (inside static Doppler gate!)
    const double r_drone = std::sqrt(drone_frd.x * drone_frd.x + drone_frd.y * drone_frd.y + drone_frd.z * drone_frd.z);
    b.radar_points[1] = {r_drone, std::atan2(drone_frd.y, drone_frd.x), std::asin(drone_frd.z / r_drone),
                         -14.85, 0.15, 16.0, -14.0,
                         RadarDopplerClass::kStaticOrHoverCandidate, true};

    REQUIRE(fusion.step(b, out));
  }

  REQUIRE_FALSE(out.radar_degraded);
  REQUIRE(out.recommended_v_max_mps == Approx(22.0));
  REQUIRE(out.active_wire_spans == 1u);
  REQUIRE(out.active_wire_capsules == 4u);
  REQUIRE(out.hover_intruders_rescued == 1u);
  REQUIRE(out.active_intruders == 1u);

  // Verify wire localization converged within < 0.6 m of true N=80 m wire midpoint!
  vertirta::rta::WireQueryResult qres{};
  REQUIRE(out.wire_capsules.query_nearest_single({0.0, 80.0, 40.0}, qres));
  REQUIRE(std::abs(qres.closest_pt_enu.y - 80.0) < 0.6);

  // Scenario B: >20 deg grazing radar dropout (-45 dBsm cliff) -> standoff before 3 deg parallax,
  // then SVD triangulation after > 3 deg lateral parallax, AND mandatory <= 15.0 m/s speed cap!
  fusion.reset();
  const vertirta::core::Vec3Ned crane_wire_mid{60.0, 0.0, -40.0};

  // Phase B1: 3 frames with small lateral motion (< 3 deg parallax) and ZERO radar points -> standoff capsule + 15 m/s cap.
  for (int k = 0; k < 4; ++k) {
    const uint64_t t_ns = 2'000'000'000ULL + static_cast<uint64_t>(k) * 33'333'333ULL;
    TimedPoseSe3 pose{};
    pose.stamp_ns = t_ns;
    pose.pos_ned = {0.0, 0.2 * static_cast<double>(k), -40.0};
    pose.vel_ned = {14.0, 0.5, 0.0};
    pose.q_frd_to_ned = {1.0, 0.0, 0.0, 0.0};
    pose.valid = true;
    fusion.push_pose_200hz(pose);

    double u0 = 0, v0 = 0, u1 = 0, v1 = 0;
    REQUIRE(cam.project_frd_to_pixel({crane_wire_mid.x, -10.0 - pose.pos_ned.y, 0.0}, u0, v0));
    REQUIRE(cam.project_frd_to_pixel({crane_wire_mid.x, +10.0 - pose.pos_ned.y, 0.0}, u1, v1));

    ForwardSensorBundle b{};
    b.frame_stamp_ns = t_ns;
    b.radar_healthy = true;
    b.camera_healthy = true;
    b.wire_line_count = 1;
    b.wire_lines[0] = {u0, v0, u1, v1, 0.85, 2u};  // smooth crane cable, 0 radar returns
    REQUIRE(fusion.step(b, out));
  }
  REQUIRE(out.radar_degraded);
  REQUIRE(out.standoff_capsule_emitted);
  REQUIRE(out.recommended_v_max_mps <= 15.0);
  REQUIRE(out.active_wire_capsules >= 1u);

  // Phase B2: Lateral shift by 6 m (> 5.7 deg parallax at 60 m) -> > 3 deg SVD triangulation engages!
  const uint64_t t_tri = 2'200'000'000ULL;
  TimedPoseSe3 pose_tri{};
  pose_tri.stamp_ns = t_tri;
  pose_tri.pos_ned = {0.0, 6.0, -40.0};
  pose_tri.vel_ned = {14.0, 3.0, 0.0};
  pose_tri.q_frd_to_ned = {1.0, 0.0, 0.0, 0.0};
  pose_tri.valid = true;
  fusion.push_pose_200hz(pose_tri);

  double u0 = 0, v0 = 0, u1 = 0, v1 = 0;
  REQUIRE(cam.project_frd_to_pixel({crane_wire_mid.x, -10.0 - 6.0, 0.0}, u0, v0));
  REQUIRE(cam.project_frd_to_pixel({crane_wire_mid.x, +10.0 - 6.0, 0.0}, u1, v1));
  ForwardSensorBundle b_tri{};
  b_tri.frame_stamp_ns = t_tri;
  b_tri.radar_healthy = true;
  b_tri.camera_healthy = true;
  b_tri.wire_line_count = 1;
  b_tri.wire_lines[0] = {u0, v0, u1, v1, 0.88, 2u};
  REQUIRE(fusion.step(b_tri, out));
  REQUIRE(out.grazing_parallax_active);
  REQUIRE(out.recommended_v_max_mps <= 15.0);
  REQUIRE(out.wire_capsules.query_nearest_single({0.0, 60.0, 40.0}, qres));
  REQUIRE(std::abs(qres.closest_pt_enu.y - 60.0) < 3.0);
}

TEST_CASE("ForwardHazardFusion stress caps strictly at <= 64 WireCapsules and <= 16 IntruderTracks",
          "[forward_hazard][caps]") {
  STATIC_REQUIRE(sizeof(vertirta::rta::WireCapsule) <= 48u);
  STATIC_REQUIRE(vertirta::rta::WireCapsuleTable::kMaxCapsules == 64u);
  STATIC_REQUIRE(vertirta::rta::IntruderTrackerKf::kMaxTracks == 16u);

  ForwardHazardFusion fusion{};
  TimedPoseSe3 pose{};
  pose.stamp_ns = 5'000'000'000ULL;
  pose.pos_ned = {0.0, 0.0, -50.0};
  pose.vel_ned = {15.0, 0.0, 0.0};
  pose.q_frd_to_ned = {1.0, 0.0, 0.0, 0.0};
  pose.valid = true;
  fusion.push_pose_200hz(pose);

  ForwardSensorBundle b{};
  b.frame_stamp_ns = pose.stamp_ns;
  b.radar_stamp_ns = pose.stamp_ns;
  b.radar_healthy = true;
  b.camera_healthy = true;

  // Inject 40 widely separated static wire radar points + 30 dynamic radar intruders.
  std::size_t idx = 0;
  for (int i = 0; i < 40 && idx < kMaxRadarDetections; ++i) {
    b.radar_points[idx++] = {
        30.0 + 3.0 * static_cast<double>(i),
        -0.30 + 0.015 * static_cast<double>(i),
        0.0,
        -15.0,
        0.0,
        18.0,
        3.0,
        RadarDopplerClass::kStaticOrHoverCandidate,
        true,
    };
  }
  for (int i = 0; i < 30 && idx < kMaxRadarDetections; ++i) {
    b.radar_points[idx++] = {
        40.0 + 2.5 * static_cast<double>(i),
        -0.25 + 0.017 * static_cast<double>(i),
        0.02,
        -19.0,
        -4.0,
        15.0,
        -10.0,
        RadarDopplerClass::kDynamicIntruder,
        true,
    };
  }
  b.radar_point_count = idx;

  ForwardHazardOutput out{};
  REQUIRE(fusion.step(b, out));
  REQUIRE(out.wire_capsules.size() <= 64u);
  REQUIRE(out.intruder_tracker.active_count() <= 16u);
  REQUIRE(out.active_wire_capsules == out.wire_capsules.size());
  REQUIRE(out.active_intruders == out.intruder_tracker.active_count());
}
