// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
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

struct SlzPerceptionOptions {
  perception::SlzOrtConfig ort_cfg{};
  perception::SlzGridConfig grid_cfg{};
  perception::ActiveInspectionConfig policy_cfg{};
  perception::CameraIntrinsics intrinsics{};
  uint64_t staleness_warn_ns{250'000'000ULL};  // 250 ms staleness contract
  uint32_t backproject_stride{2u};             // 256x256 = 65,536 pts/frame (>=12 pts per 0.5 m cell @ 22 m AGL)
  float soc_reserve_pct{12.0f};
};

// Pure C++20 Active Visual SLZ Perception Core (zero ROS 2 dependency when VERTIRTA_WITH_ROS2 is unset).
// Wraps real SlzOrtRunner -> Free-Energy + Boundary Compensation -> BackprojectDepthRgbd ->
// SlzBeliefGrid (1.5625 MiB) -> VisualSlzPriorProvider (ISlzPriorProvider) with zero hot-loop heap allocs.
class SlzPerceptionCore {
 public:
  explicit SlzPerceptionCore(SlzPerceptionOptions opts = {})
      : opts_(std::move(opts)),
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
