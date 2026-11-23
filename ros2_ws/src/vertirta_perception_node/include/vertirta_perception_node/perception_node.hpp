// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/perception/active_inspection_policy.hpp"
#include "vertirta/perception/slz_belief_grid.hpp"
#include "vertirta/perception/slz_geometry.hpp"
#include "vertirta/perception/slz_ort_engine.hpp"
#include "vertirta/perception/visual_slz_prior_provider.hpp"
#include "vertirta/rta/slz_prior_provider.hpp"

namespace vertirta::ros {

inline constexpr uint64_t kDefaultMaxRgbdSyncNs = 2'000'000ULL;  // |t_rgb - t_depth| <= 2 ms

// Overflow-safe RGB-D hardware timestamp synchronization gate (|t_rgb - t_depth| <= max_skew_ns).
[[nodiscard]] inline bool CheckRgbdTimestampSync(
    uint64_t rgb_stamp_ns,
    uint64_t depth_stamp_ns,
    uint64_t max_skew_ns = kDefaultMaxRgbdSyncNs) noexcept {
  if (rgb_stamp_ns == 0ULL || depth_stamp_ns == 0ULL) {
    return false;
  }
  const uint64_t diff =
      (rgb_stamp_ns >= depth_stamp_ns) ? (rgb_stamp_ns - depth_stamp_ns)
                                       : (depth_stamp_ns - rgb_stamp_ns);
  return diff <= max_skew_ns;
}

struct RgbDecodeStats {
  bool ok{false};
  float saturated_pixel_frac{0.0f};
  bool glare_warning{false};
};

// Decodes 8-bit rgb8/bgr8 image into ImageNet-normalized NCHW 3x512x512 float buffer (zero heap).
// Tracks specular sun-glare / over-exposure clipping fraction (R,G,B >= 252).
[[nodiscard]] inline RgbDecodeStats DecodeRgbImageToNchw512(
    std::span<const uint8_t> raw_bytes,
    uint32_t width,
    uint32_t height,
    uint32_t step_bytes,
    bool is_bgr,
    std::span<float> out_nchw_512,
    float glare_warn_frac = 0.20f) noexcept {
  RgbDecodeStats stats{};
  if (width == 0u || height == 0u || step_bytes < width * 3u ||
      raw_bytes.size() < static_cast<std::size_t>(step_bytes) * static_cast<std::size_t>(height) ||
      out_nchw_512.size() != perception::kSlzRgbElements) {
    return stats;
  }

  constexpr float kInv255 = 1.0f / 255.0f;
  constexpr float kMeanR = 0.485f;
  constexpr float kMeanG = 0.456f;
  constexpr float kMeanB = 0.406f;
  constexpr float kInvStdR = 1.0f / 0.229f;
  constexpr float kInvStdG = 1.0f / 0.224f;
  constexpr float kInvStdB = 1.0f / 0.225f;
  constexpr std::size_t kHW = perception::kSlzPixels;

  float* r_plane = out_nchw_512.data();
  float* g_plane = out_nchw_512.data() + kHW;
  float* b_plane = out_nchw_512.data() + 2u * kHW;

  const uint32_t r_off = is_bgr ? 2u : 0u;
  const uint32_t g_off = 1u;
  const uint32_t b_off = is_bgr ? 0u : 2u;
  std::size_t sat_count = 0u;

  for (uint32_t y = 0u; y < perception::kSlzInputHeight; ++y) {
    const uint32_t sy = static_cast<uint32_t>(
        (static_cast<uint64_t>(y) * height) / perception::kSlzInputHeight);
    const std::size_t src_row = static_cast<std::size_t>(sy) * step_bytes;
    const std::size_t dst_row = static_cast<std::size_t>(y) * perception::kSlzInputWidth;
    for (uint32_t x = 0u; x < perception::kSlzInputWidth; ++x) {
      const uint32_t sx = static_cast<uint32_t>(
          (static_cast<uint64_t>(x) * width) / perception::kSlzInputWidth);
      const std::size_t p = src_row + static_cast<std::size_t>(sx) * 3u;
      const uint8_t r8 = raw_bytes[p + r_off];
      const uint8_t g8 = raw_bytes[p + g_off];
      const uint8_t b8 = raw_bytes[p + b_off];
      if (r8 >= 252u && g8 >= 252u && b8 >= 252u) {
        ++sat_count;
      }
      const std::size_t dst_idx = dst_row + x;
      r_plane[dst_idx] = (static_cast<float>(r8) * kInv255 - kMeanR) * kInvStdR;
      g_plane[dst_idx] = (static_cast<float>(g8) * kInv255 - kMeanG) * kInvStdG;
      b_plane[dst_idx] = (static_cast<float>(b8) * kInv255 - kMeanB) * kInvStdB;
    }
  }

  stats.ok = true;
  stats.saturated_pixel_frac = static_cast<float>(sat_count) / static_cast<float>(kHW);
  stats.glare_warning = (stats.saturated_pixel_frac >= glare_warn_frac);
  return stats;
}

// Decodes 32FC1 (meters) or 16UC1 (millimeters) depth image into 512x512 float meters (zero heap).
[[nodiscard]] inline bool DecodeDepthImageToFloat512(
    std::span<const uint8_t> raw_bytes,
    uint32_t width,
    uint32_t height,
    uint32_t step_bytes,
    bool is_16uc1_mm,
    std::span<float> out_depth_512) noexcept {
  const uint32_t bpp = is_16uc1_mm ? 2u : 4u;
  if (width == 0u || height == 0u || step_bytes < width * bpp ||
      raw_bytes.size() < static_cast<std::size_t>(step_bytes) * static_cast<std::size_t>(height) ||
      out_depth_512.size() != perception::kSlzPixels) {
    return false;
  }

  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  for (uint32_t y = 0u; y < perception::kSlzInputHeight; ++y) {
    const uint32_t sy = static_cast<uint32_t>(
        (static_cast<uint64_t>(y) * height) / perception::kSlzInputHeight);
    const std::size_t src_row = static_cast<std::size_t>(sy) * step_bytes;
    const std::size_t dst_row = static_cast<std::size_t>(y) * perception::kSlzInputWidth;
    for (uint32_t x = 0u; x < perception::kSlzInputWidth; ++x) {
      const uint32_t sx = static_cast<uint32_t>(
          (static_cast<uint64_t>(x) * width) / perception::kSlzInputWidth);
      const std::size_t p = src_row + static_cast<std::size_t>(sx) * bpp;
      float z_m = kNaN;
      if (is_16uc1_mm) {
        uint16_t mm = 0u;
        std::memcpy(&mm, raw_bytes.data() + p, sizeof(uint16_t));
        z_m = (mm > 0u) ? (static_cast<float>(mm) * 0.001f) : kNaN;
      } else {
        std::memcpy(&z_m, raw_bytes.data() + p, sizeof(float));
      }
      out_depth_512[dst_row + x] = z_m;
    }
  }
  return true;
}

struct SlzPerceptionOptions {
  perception::SlzOrtConfig ort_cfg{};
  perception::SlzGridConfig grid_cfg{};
  perception::ActiveInspectionConfig policy_cfg{};
  perception::CameraIntrinsics intrinsics{};
  uint64_t staleness_warn_ns{250'000'000ULL};  // 250 ms staleness contract
  uint64_t max_rgbd_sync_ns{kDefaultMaxRgbdSyncNs};
  uint32_t backproject_stride{2u};             // 256x256 = 65,536 pts/frame (>=12 pts per 0.5 m cell @ 22 m AGL)
  float soc_reserve_pct{12.0f};
};

[[nodiscard]] inline SlzPerceptionOptions ResolvePerceptionOptions(SlzPerceptionOptions opts) {
  if (!opts.ort_cfg.ood_calibration_path.empty()) {
    const bool loaded =
        perception::LoadOodCalibrationJson(opts.ort_cfg.ood_calibration_path, opts.ort_cfg);
    if (!loaded && !opts.ort_cfg.allow_mock_fallback) {
      throw std::runtime_error(
          "SlzPerceptionCore: failed to load ood_calibration_path: " +
          opts.ort_cfg.ood_calibration_path);
    }
  }
  opts.grid_cfg.tau_ood = opts.ort_cfg.tau_ood;
  return opts;
}

// Pure C++20 Active Visual SLZ Perception Core (zero ROS 2 dependency when VERTIRTA_WITH_ROS2 is unset).
// Wraps real SlzOrtRunner -> Free-Energy + Boundary Compensation -> BackprojectDepthRgbd ->
// SlzBeliefGrid (1.5625 MiB) -> VisualSlzPriorProvider (ISlzPriorProvider) with zero hot-loop heap allocs.
class SlzPerceptionCore {
 public:
  explicit SlzPerceptionCore(SlzPerceptionOptions opts = {})
      : opts_(ResolvePerceptionOptions(std::move(opts))),
        ort_runner_(opts_.ort_cfg),
        belief_grid_(opts_.grid_cfg),
        provider_(opts_.policy_cfg, opts_.staleness_warn_ns),
        rgb_buf_(perception::kSlzRgbElements, 0.25f),
        depth_buf_(perception::kSlzPixels, 22.0f),
        logits_buf_(perception::kSlzLogitsElements, 0.0f),
        normals_buf_(perception::kSlzNormalsElements, 0.0f),
        bdry_buf_(perception::kSlzBdryElements, 0.0f),
        e_raw_buf_(perception::kSlzPixels, 0.0f),
        e_tilde_buf_(perception::kSlzPixels, 0.0f) {
    const uint32_t s = std::max(1u, opts_.backproject_stride);
    const std::size_t max_pts =
        ((static_cast<std::size_t>(opts_.intrinsics.width) + s - 1u) / s) *
        ((static_cast<std::size_t>(opts_.intrinsics.height) + s - 1u) / s);
    pts_buf_.resize(max_pts);
    pix_idx_buf_.resize(max_pts);
    obs_buf_.resize(max_pts);

    // Seed default canonical Kartal Coastal Vertiport candidate (west of LTFJ J2)
    rta::SlzPrior kartal{};
    const char kId[] = "Kartal_Coastal_Vertiport";
    std::memcpy(kartal.id, kId, sizeof(kId));
    kartal.lat_deg = 40.8860;
    kartal.lon_deg = 29.1880;
    kartal.elev_amsl_m = 12.0;
    kartal.center_enu = {0.0, 0.0, 0.0};
    kartal.clearance_r_m = 15.0f;
    kartal.eta_s = 12.0f;
    kartal.rank = 0u;
    kartal.reason_mask = perception::SlzReasonBits::REASON_NONE;
    catalog_[0] = kartal;
    catalog_count_ = 1u;
  }

  SlzPerceptionCore(const SlzPerceptionCore&) = delete;
  SlzPerceptionCore& operator=(const SlzPerceptionCore&) = delete;

  void SetCatalogCandidates(std::span<const rta::SlzPrior> candidates) noexcept {
    const std::size_t n = std::min(candidates.size(), catalog_.size());
    for (std::size_t i = 0u; i < n; ++i) {
      catalog_[i] = candidates[i];
    }
    catalog_count_ = static_cast<uint8_t>(n);
  }

  void SetCameraIntrinsics(const perception::CameraIntrinsics& intr) noexcept {
    if (intr.fx > 1e-6f && intr.fy > 1e-6f && intr.width > 0u && intr.height > 0u) {
      // Rescale intrinsics to 512x512 network/depth working resolution if CameraInfo differs.
      const float sx =
          static_cast<float>(perception::kSlzInputWidth) / static_cast<float>(intr.width);
      const float sy =
          static_cast<float>(perception::kSlzInputHeight) / static_cast<float>(intr.height);
      opts_.intrinsics = intr;
      opts_.intrinsics.fx = intr.fx * sx;
      opts_.intrinsics.fy = intr.fy * sy;
      opts_.intrinsics.cx = intr.cx * sx;
      opts_.intrinsics.cy = intr.cy * sy;
      opts_.intrinsics.width = perception::kSlzInputWidth;
      opts_.intrinsics.height = perception::kSlzInputHeight;
    }
  }

  void UpdateVehicleEnvelope(
      float soc_pct,
      float wind_mps,
      core::FlightRegime regime) noexcept {
    soc_pct_ = std::clamp(soc_pct, 0.0f, 100.0f);
    wind_mps_ = std::max(0.0f, wind_mps);
    regime_ = regime;
  }

  // Runs full zero-heap RGB-D inference -> Free-Energy OOD -> Backproject -> SlzBeliefGrid -> VisualSlzPriorProvider
  [[nodiscard]] bool ProcessRgbdFrame(
      std::span<const float> rgb_nchw_512,
      std::span<const float> depth_512,
      const perception::CameraPoseEnu& cam_pose,
      const core::Vec3Enu& vehicle_enu,
      float agl_m,
      perception::DepthMode depth_mode,
      uint64_t frame_stamp_ns,
      uint64_t now_ns) noexcept {
    if (rgb_nchw_512.size() != perception::kSlzRgbElements ||
        depth_512.size() != perception::kSlzPixels) {
      return false;
    }

    if (!ort_runner_.Run(rgb_nchw_512, logits_buf_, normals_buf_, bdry_buf_)) {
      return false;
    }
    if (!perception::ComputeFreeEnergy(logits_buf_, opts_.ort_cfg.t_ood, e_raw_buf_)) {
      return false;
    }
    if (!perception::ApplyUpsampledBoundaryCompensation(
            e_raw_buf_, bdry_buf_, opts_.ort_cfg.kappa_b, e_tilde_buf_)) {
      return false;
    }

    const uint32_t stride = std::max(1u, opts_.backproject_stride);
    const std::size_t n_pts = perception::BackprojectDepthRgbd(
        depth_512, opts_.intrinsics, cam_pose, pts_buf_, pix_idx_buf_, stride);

    const float inv_t = 1.0f / std::max(opts_.ort_cfg.t_prob, 1e-3f);
    constexpr std::size_t kHW = perception::kSlzPixels;

    for (std::size_t k = 0u; k < n_pts; ++k) {
      const std::size_t p = pix_idx_buf_[k];
      float lmax = logits_buf_[p];
      for (uint32_t c = 1u; c < perception::kSlzNumClasses; ++c) {
        lmax = std::max(lmax, logits_buf_[c * kHW + p]);
      }
      float exps[perception::kSlzNumClasses]{};
      float sum_exp = 0.0f;
      for (uint32_t c = 0u; c < perception::kSlzNumClasses; ++c) {
        exps[c] = std::exp((logits_buf_[c * kHW + p] - lmax) * inv_t);
        sum_exp += exps[c];
      }
      const float inv_sum = (sum_exp > 1e-12f) ? (1.0f / sum_exp) : 0.0f;
      const float p_safe = std::clamp((exps[0] + exps[1]) * inv_sum, 1e-3f, 1.0f - 1e-3f);
      const float p_dyn = std::clamp(exps[3] * inv_sum, 0.0f, 1.0f);

      perception::SlzPointObservation o{};
      o.point_enu = pts_buf_[k];
      o.p_safe_obs = p_safe;
      o.p_dyn_obs = p_dyn;
      o.e_tilde_ood = e_tilde_buf_[p];
      o.cos_incidence = 1.0f;
      obs_buf_[k] = o;
    }

    const double ts_s = static_cast<double>(frame_stamp_ns) * 1e-9;
    if (!belief_grid_.has_origin()) {
      belief_grid_.SetOrigin(vehicle_enu);
    }
    belief_grid_.UpdateFrame(
        std::span<const perception::SlzPointObservation>(obs_buf_.data(), n_pts),
        vehicle_enu,
        ts_s,
        /*viewpoint_novelty=*/0.8f,
        /*extra_geo_var_floor=*/(depth_mode == perception::DepthMode::MONO_FALLBACK) ? 0.10f : 0.0f);

    perception::EnergyGateInput energy{};
    energy.soc_current_pct = soc_pct_;
    energy.soc_reserve_pct = opts_.soc_reserve_pct;
    energy.wind_mps = wind_mps_;
    energy.regime = regime_;

    last_vehicle_enu_ = vehicle_enu;
    last_decision_ = provider_.UpdateFromGrid(
        belief_grid_,
        std::span<const rta::SlzPrior>(catalog_.data(), catalog_count_),
        vehicle_enu,
        agl_m,
        depth_mode,
        energy,
        frame_stamp_ns);
    latest_snapshot_ = provider_.query_priors(vehicle_enu, soc_pct_, now_ns);
    return true;
  }

  // Deterministic host-testable mock frame injection exercising the full ORT+Grid+Provider pipeline
  bool InjectMockFrame(
      uint64_t stamp_ns,
      uint64_t now_ns,
      bool inject_ood_tarp = false,
      perception::DepthMode depth_mode = perception::DepthMode::RGBD_PRIMARY,
      float agl_m = 22.0f) noexcept {
    if (inject_ood_tarp) {
      // Uniform logits -> E_raw = -ln(5) = -1.61 > tau_ood (-8.0) -> triggers OOD variance floor
      const float ood_logits[perception::kSlzNumClasses] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
      ort_runner_.SetMockOutputs(ood_logits, 0.0f);
    } else {
      // High-confidence class-1 Safe_Flat_Roof -> E_raw ~ -12.0 < tau_ood (-8.0)
      const float clean_logits[perception::kSlzNumClasses] = {-1.0f, 12.0f, -3.0f, -4.0f, -4.0f};
      ort_runner_.SetMockOutputs(clean_logits, 0.0f);
    }

    std::fill(depth_buf_.begin(), depth_buf_.end(), agl_m);
    const core::Vec3Enu veh{0.0, 0.0, static_cast<double>(agl_m)};
    const auto cam_pose = perception::MakeGimbalCameraPoseEnu(veh, 0.0, 1.5707963267948966);
    return ProcessRgbdFrame(
        rgb_buf_, depth_buf_, cam_pose, veh, agl_m, depth_mode, stamp_ns, now_ns);
  }

  void CheckStaleness(uint64_t now_ns) noexcept {
    latest_snapshot_ = provider_.query_priors(last_vehicle_enu_, soc_pct_, now_ns);
  }

  [[nodiscard]] const rta::SlzProviderSnapshot& LatestSnapshot() const noexcept {
    return latest_snapshot_;
  }

  [[nodiscard]] const perception::ActiveInspectionDecision& LastDecision() const noexcept {
    return last_decision_;
  }

  [[nodiscard]] const rta::ISlzPriorProvider& PriorProvider() const noexcept { return provider_; }
  [[nodiscard]] const perception::SlzOrtRunner& OrtRunner() const noexcept { return ort_runner_; }
  [[nodiscard]] const perception::SlzBeliefGrid& BeliefGrid() const noexcept { return belief_grid_; }
  [[nodiscard]] const SlzPerceptionOptions& Options() const noexcept { return opts_; }

 private:
  SlzPerceptionOptions opts_{};
  perception::SlzOrtRunner ort_runner_;
  perception::SlzBeliefGrid belief_grid_;
  perception::VisualSlzPriorProvider provider_;

  std::array<rta::SlzPrior, 8> catalog_{};
  uint8_t catalog_count_{0u};
  core::Vec3Enu last_vehicle_enu_{0.0, 0.0, 22.0};
  float soc_pct_{65.0f};
  float wind_mps_{4.0f};
  core::FlightRegime regime_{core::FlightRegime::NOMINAL};
  rta::SlzProviderSnapshot latest_snapshot_{};
  perception::ActiveInspectionDecision last_decision_{};

  std::vector<float> rgb_buf_;
  std::vector<float> depth_buf_;
  std::vector<float> logits_buf_;
  std::vector<float> normals_buf_;
  std::vector<float> bdry_buf_;
  std::vector<float> e_raw_buf_;
  std::vector<float> e_tilde_buf_;
  std::vector<core::Vec3Enu> pts_buf_;
  std::vector<uint32_t> pix_idx_buf_;
  std::vector<perception::SlzPointObservation> obs_buf_;
};

#ifndef VERTIRTA_WITH_ROS2
// Host-testable alias matching Plan 3 Task 6's SlzPerceptionNode API without requiring ROS 2 installed
using SlzPerceptionNode = SlzPerceptionCore;
#endif

}  // namespace vertirta::ros
