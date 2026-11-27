#include "vertirta/forward_hazard/forward_hazard_fusion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>

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
  //    - Dynamic Doppler returns -> IntruderTrackerKf (<=16 tracks) via 1-to-1 spatial association.
  //    - Static-gate returns matched to D-FINE-N drone/bird box -> rescued into IntruderTrackerKf!
  //    - Remaining static-gate returns -> WireSpanEkfBank (3D wire glint/Bragg fusion).
  const std::size_t n_rad = std::min(bundle.radar_point_count, kMaxRadarDetections);
  uint32_t valid_radar_pts = 0u;
  uint32_t hover_rescued = 0u;

  struct PendingRadarIntruder {
    core::Vec3Enu pos_enu{};
    core::Vec3Enu vel_enu{};
    uint32_t track_id{0u};
    bool assigned{false};
  };
  std::array<PendingRadarIntruder, kMaxRadarDetections> pending_intruders{};
  std::size_t pending_count = 0u;
  const uint64_t radar_ts = (bundle.radar_stamp_ns > 0ULL) ? bundle.radar_stamp_ns : bundle.frame_stamp_ns;

  if (bundle.radar_healthy) {
    for (std::size_t i = 0; i < n_rad; ++i) {
      const RadarPointDetection& rp = bundle.radar_points[i];
      if (!rp.valid ||
          !std::isfinite(rp.range_m) || rp.range_m < 2.0 || rp.range_m > 2000.0 ||
          !std::isfinite(rp.az_rad) || !std::isfinite(rp.el_rad) ||
          !std::isfinite(rp.snr_db) || rp.snr_db < cfg_.min_radar_snr_db ||
          !std::isfinite(rp.residual_doppler_mps)) {
        continue;
      }

      const core::Vec3Frd pt_frd = spherical_radar_to_frd(rp.range_m, rp.az_rad, rp.el_rad);
      const core::Vec3Ned pt_ned =
          radar_pose.pos_ned + Se3PoseRingBuffer::rotate_frd_to_ned(radar_pose.q_frd_to_ned, pt_frd);
      if (!pt_frd.is_finite() || !pt_ned.is_finite()) {
        continue;
      }
      ++valid_radar_pts;

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
        if (vel_est_ned.is_finite() && pending_count < kMaxRadarDetections) {
          pending_intruders[pending_count++] = PendingRadarIntruder{
              .pos_enu = core::ned_to_enu(pt_ned),
              .vel_enu = core::ned_to_enu(vel_est_ned),
              .track_id = 0u,
              .assigned = false,
          };
        }
      } else {
        // Static wire glint/Bragg point -> horizontal lateral tangent hint in NED.
        const core::Vec3Ned lateral_hint =
            Se3PoseRingBuffer::rotate_frd_to_ned(radar_pose.q_frd_to_ned, {0.0, 1.0, 0.0});
        wire_ekf_.ingest_radar_wire_point(
            radar_ts,
            pt_ned,
            lateral_hint,
            rp.snr_db);
      }
    }

    // One-to-one globally-closest spatial nearest-neighbor association against active KF tracks.
    const double gate_m = (std::isfinite(cfg_.radar_assoc_gate_m) && cfg_.radar_assoc_gate_m > 0.0)
                              ? cfg_.radar_assoc_gate_m
                              : 15.0;
    std::array<uint32_t, rta::IntruderTrackerKf::kMaxTracks> claimed_ids{};
    std::size_t claimed_count = 0u;

    while (claimed_count < rta::IntruderTrackerKf::kMaxTracks) {
      std::size_t best_det_idx = kMaxRadarDetections;
      uint32_t best_trk_id = 0u;
      double best_dist_m = gate_m + 1.0;

      const std::span<const uint32_t> excluded(claimed_ids.data(), claimed_count);
      for (std::size_t d = 0; d < pending_count; ++d) {
        if (pending_intruders[d].assigned) {
          continue;
        }
        double dist_m = 0.0;
        const std::optional<uint32_t> cand = intruder_kf_.match_nearest_track_id(
            pending_intruders[d].pos_enu, radar_ts, gate_m, excluded, &dist_m);
        if (cand.has_value() && dist_m < best_dist_m) {
          best_dist_m = dist_m;
          best_det_idx = d;
          best_trk_id = *cand;
        }
      }

      if (best_det_idx >= pending_count) {
        break;
      }
      pending_intruders[best_det_idx].track_id = best_trk_id;
      pending_intruders[best_det_idx].assigned = true;
      claimed_ids[claimed_count++] = best_trk_id;
    }

    for (std::size_t d = 0; d < pending_count; ++d) {
      if (!pending_intruders[d].assigned) {
        if (next_radar_track_id_ < 1000u) {
          next_radar_track_id_ = 1000u;
        }
        pending_intruders[d].track_id = next_radar_track_id_++;
        pending_intruders[d].assigned = true;
      }
      intruder_kf_.update_measurement(
          pending_intruders[d].track_id,
          pending_intruders[d].pos_enu,
          pending_intruders[d].vel_enu,
          radar_ts);
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
