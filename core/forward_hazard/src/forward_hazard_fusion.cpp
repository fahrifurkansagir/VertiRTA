#include "vertirta/forward_hazard/forward_hazard_fusion.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::forward_hazard {

namespace {

[[nodiscard]] core::Vec3Frd spherical_radar_to_frd(
    double range_m,
    double az_rad,
    double el_rad) noexcept {
  const double ce = std::cos(el_rad);
  return {
      range_m * ce * std::cos(az_rad),
      range_m * ce * std::sin(az_rad),
      range_m * std::sin(el_rad),
  };
}

}  // namespace

ForwardHazardFusion::ForwardHazardFusion(const ForwardFusionConfig& cfg) noexcept
    : cfg_(cfg),
      split_stack_(cfg.split_stack),
      detectors_(cfg.onnx, cfg.split_stack),
      wire_ekf_(cfg.wire_ekf) {}

void ForwardHazardFusion::reset() noexcept {
  pose_ring_.clear();
  wire_ekf_.clear();
  intruder_kf_.clear();
  next_radar_track_id_ = 1000u;
}

void ForwardHazardFusion::push_pose_200hz(const TimedPoseSe3& pose) noexcept {
  (void)pose_ring_.push(pose);
}

bool ForwardHazardFusion::step(
    const ForwardSensorBundle& bundle,
    ForwardHazardOutput& out) noexcept {
  out = ForwardHazardOutput{};
  out.stamp_ns = bundle.frame_stamp_ns;
  out.recommended_v_max_mps = cfg_.v_max_nominal_mps;

  TimedPoseSe3 cam_pose{};
  if (!pose_ring_.query(bundle.frame_stamp_ns, cam_pose)) {
    if (!pose_ring_.latest(cam_pose)) {
      return false;
    }
  }

  TimedPoseSe3 radar_pose = cam_pose;
  if (bundle.radar_stamp_ns > 0ULL) {
    (void)pose_ring_.query(bundle.radar_stamp_ns, radar_pose);
  }

  wire_ekf_.prune_stale(bundle.frame_stamp_ns);

  // 1. Run M-LSD-WireSeg + D-FINE-N (or yield gracefully to locked 512x512 PIDNet-S during descent).
  ForwardDetectorRawOutput det{};
  (void)detectors_.infer_from_bundle(bundle, det);

  // 2. Ingest 30 Hz thin-wire vision interpretation planes into 6-state wire-span EKF.
  for (std::size_t i = 0; i < det.wire_line_count; ++i) {
    const WireLineDetection2D& line = det.wire_lines[i];
    core::Vec3Frd n_frd{};
    if (!split_stack_.segment_to_interp_plane_frd(
            line.u0_px, line.v0_px, line.u1_px, line.v1_px, n_frd)) {
      continue;
    }
    const double mid_u = 0.5 * (line.u0_px + line.u1_px);
    const double mid_v = 0.5 * (line.v0_px + line.v1_px);
    const core::Vec3Frd ray_frd = split_stack_.pixel_to_unit_ray_frd(mid_u, mid_v);
    const core::Vec3Ned ray_ned = Se3PoseRingBuffer::rotate_frd_to_ned(cam_pose.q_frd_to_ned, ray_frd);
    const core::Vec3Ned n_ned = Se3PoseRingBuffer::rotate_frd_to_ned(cam_pose.q_frd_to_ned, n_frd);
    wire_ekf_.ingest_vision_plane(
        bundle.frame_stamp_ns,
        cam_pose.pos_ned,
        ray_ned,
        n_ned,
        line.voltage_class);
  }

  // 3. Ingest 15-20 Hz 77 GHz mmWave radar points:
  //    - Dynamic Doppler returns -> IntruderTrackerKf (<=16 tracks).
  //    - Static-gate returns matched to D-FINE-N drone/bird box -> rescued into IntruderTrackerKf!
  //    - Remaining static-gate returns -> WireSpanEkfBank (3D wire glint/Bragg fusion).
  const std::size_t n_rad = std::min(bundle.radar_point_count, kMaxRadarDetections);
  uint32_t valid_radar_pts = 0u;
  uint32_t hover_rescued = 0u;
  uint32_t dynamic_idx = 0u;

  if (bundle.radar_healthy) {
    for (std::size_t i = 0; i < n_rad; ++i) {
      const RadarPointDetection& rp = bundle.radar_points[i];
      if (!rp.valid || !std::isfinite(rp.range_m) || rp.range_m < 2.0 ||
          rp.snr_db < cfg_.min_radar_snr_db) {
        continue;
      }
      ++valid_radar_pts;

      const core::Vec3Frd pt_frd = spherical_radar_to_frd(rp.range_m, rp.az_rad, rp.el_rad);
      const core::Vec3Ned pt_ned =
          radar_pose.pos_ned + Se3PoseRingBuffer::rotate_frd_to_ned(radar_pose.q_frd_to_ned, pt_frd);

      // Check whether this static-gate radar return projects onto a D-FINE-N drone/bird/aircraft box.
      bool matched_dfine_intruder = false;
      double proj_u = 0.0;
      double proj_v = 0.0;
      if (split_stack_.project_frd_to_pixel(pt_frd, proj_u, proj_v)) {
        for (std::size_t b = 0; b < det.intruder_box_count; ++b) {
          const IntruderBoxDetection2D& box = det.intruder_boxes[b];
          if (box.class_id == 3u) {
            continue;  // Ignore static pylon boxes for intruder tracker
          }
          const double du = std::abs(proj_u - box.cx_px);
          const double dv = std::abs(proj_v - box.cy_px);
          if (du <= std::max(0.6 * box.width_px, cfg_.dfine_hover_assoc_px) &&
              dv <= std::max(0.6 * box.height_px, cfg_.dfine_hover_assoc_px)) {
            matched_dfine_intruder = true;
            break;
          }
        }
      }

      if (rp.doppler_class == RadarDopplerClass::kDynamicIntruder || matched_dfine_intruder) {
        if (rp.doppler_class == RadarDopplerClass::kStaticOrHoverCandidate && matched_dfine_intruder) {
          ++hover_rescued;
        }
        const core::Vec3Frd u_frd = pt_frd * (1.0 / std::max(rp.range_m, 1.0));
        const core::Vec3Ned u_ned = Se3PoseRingBuffer::rotate_frd_to_ned(radar_pose.q_frd_to_ned, u_frd);
        const core::Vec3Ned vel_est_ned = u_ned * rp.residual_doppler_mps;
        const uint32_t trk_id = 1000u + (dynamic_idx++ % static_cast<uint32_t>(rta::IntruderTrackerKf::kMaxTracks));
        intruder_kf_.update_measurement(
            trk_id,
            core::ned_to_enu(pt_ned),
            core::ned_to_enu(vel_est_ned),
            bundle.radar_stamp_ns > 0ULL ? bundle.radar_stamp_ns : bundle.frame_stamp_ns);
      } else {
        // Static wire glint/Bragg point -> horizontal lateral tangent hint in NED.
        const core::Vec3Ned lateral_hint =
            Se3PoseRingBuffer::rotate_frd_to_ned(radar_pose.q_frd_to_ned, {0.0, 1.0, 0.0});
        wire_ekf_.ingest_radar_wire_point(
            bundle.radar_stamp_ns > 0ULL ? bundle.radar_stamp_ns : bundle.frame_stamp_ns,
            pt_ned,
            lateral_hint,
            rp.snr_db);
      }
    }
  }

  // 4. Emit piecewise catenary WireCapsules (<=64, R_veh=2.10 m) and IntruderTrackerKf copy (<=16).
  uint32_t active_spans = 0u;
  bool grazing_parallax = false;
  bool standoff_emitted = false;
  wire_ekf_.emit_wire_capsules(
      cam_pose.pos_ned,
      cam_pose.vel_ned,
      out.wire_capsules,
      active_spans,
      grazing_parallax,
      standoff_emitted);

  out.intruder_tracker = intruder_kf_;
  out.active_wire_spans = active_spans;
  out.active_wire_capsules = static_cast<uint32_t>(out.wire_capsules.size());
  out.active_intruders = static_cast<uint32_t>(out.intruder_tracker.active_count());
  out.hover_intruders_rescued = hover_rescued;
  out.grazing_parallax_active = grazing_parallax;
  out.standoff_capsule_emitted = standoff_emitted;

  // 5. Mandatory radar-degraded / grazing-angle urban-clutter CBVF speed cap (<= 15.0 m/s -> 35.8 m stop).
  const bool vision_saw_wires_without_radar = (det.wire_line_count > 0u && valid_radar_pts == 0u);
  out.radar_degraded = (!bundle.radar_healthy) || vision_saw_wires_without_radar ||
                       grazing_parallax || standoff_emitted;
  if (out.radar_degraded) {
    out.recommended_v_max_mps = std::min(out.recommended_v_max_mps, cfg_.v_max_radar_degraded_mps);
  }

  return true;
}

}  // namespace vertirta::forward_hazard
