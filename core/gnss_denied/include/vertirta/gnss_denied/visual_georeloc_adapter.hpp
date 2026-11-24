#pragma once

#include <array>
#include <cstdint>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"

namespace vertirta::gnss_denied {

enum class OrthoMapTier : uint8_t {
  kLocalHighResOrtho05m = 0,   // Local 0.5-1.0 m ortho MBTiles cache (primary LTFJ/LTBX localization)
  kSentinel2GrossBlunder10m = 1,  // 10 m Sentinel-2 (~45 m CEP50 gross-blunder only; NEVER tightens corridor P)
};

enum class GeoRelocRejectReason : uint8_t {
  kAccepted = 0,
  kTooFewInliers = 1,
  kHighReprojRmse = 2,
  kLowConvexHullCoverage = 3,
  kMahalanobisGateFailed = 4,
  kSentinel2GrossBlunderOnly = 5,
  kMegaLocAwaitingCorroboration = 6,
  kInvalidInput = 7,
};

struct VisualGeoRelocConfig {
  // 01-DR08-13 Override #6: TU116 enable_fp16=false default
  bool enable_fp16{false};
  uint32_t min_ransac_inliers{25};
  double max_reproj_rmse_px{2.5};
  double min_convex_hull_ratio{0.25};
  double mahalanobis_chi2_2dof_gate{9.21};  // p = 0.01
  double megaloc_trigger_sigma_m{50.0};
  double megaloc_corroborate_radius_m{15.0};
  double ortho_05m_meas_sigma_m{2.0};
  double sentinel2_cep50_sigma_m{45.0};
};

// Candidate match produced by XFeat+LighterGlue (Apache-2.0) or MegaLoc (MIT) against local ortho tiles.
struct VisualMatchObservation {
  uint64_t stamp_ns{0};
  core::GeodeticWgs84 matched_geo{};
  uint32_t ransac_inliers{0};
  double reproj_rmse_px{0.0};
  double convex_hull_ratio{0.0};
  double yaw_derotate_rad{0.0};
  double radalt_agl_m{40.0};
  OrthoMapTier map_tier{OrthoMapTier::kLocalHighResOrtho05m};
  bool from_megaloc{false};
  bool valid{true};
};

// Symmetric 2x2 horizontal covariance matrix P_pos = [[p_nn, p_ne], [p_ne, p_ee]] in NED [m^2].
struct Cov2D {
  double p_nn{4.0};
  double p_ne{0.0};
  double p_ee{4.0};

  [[nodiscard]] double max_eigenvalue() const noexcept;
  [[nodiscard]] double min_eigenvalue() const noexcept;
  [[nodiscard]] double sigma_max() const noexcept;
  [[nodiscard]] double directional_variance(double n_north, double n_east) const noexcept;
};

struct GeoRelocDecision {
  bool accepted{false};
  GeoRelocRejectReason reason{GeoRelocRejectReason::kInvalidInput};
  bool megaloc_requested{false};
  bool megaloc_relocked{false};
  double mahalanobis_chi2{0.0};
  core::Vec3Ned updated_pos_ned{};
  Cov2D updated_cov{};
};

// 0.5-1 Hz Visual Geo-Registration Adapter: XFeat+LighterGlue (Apache-2.0) + MegaLoc (MIT).
// Uses pure vertirta::core::geodesy (zero libproj/GDAL) and SPD Joseph-form Kalman covariance updates.
class VisualGeoRelocAdapter {
 public:
  explicit VisualGeoRelocAdapter(const VisualGeoRelocConfig& cfg = VisualGeoRelocConfig{}) noexcept;

  void reset() noexcept;

  [[nodiscard]] const VisualGeoRelocConfig& config() const noexcept { return cfg_; }

  // Convert geodetic ortho coordinate to local Kartal NED strictly via pure vertirta::core::geodesy.
  [[nodiscard]] static core::Vec3Ned geodetic_to_kartal_ned(const core::GeodeticWgs84& geo) noexcept;

  // Apply 3-gate acceptance filter (inliers>=25 & RMSE<2.5 px, convex hull>0.25, Mahalanobis<=9.21)
  // and Joseph-form Kalman update P+ = (I - KH) P- (I - KH)^T + K R K^T.
  [[nodiscard]] GeoRelocDecision fuse_observation(
      const core::Vec3Ned& prior_pos_ned,
      const Cov2D& prior_cov,
      const VisualMatchObservation& obs) noexcept;

  [[nodiscard]] static constexpr bool compiled_with_onnxruntime() noexcept {
#ifdef VERTIRTA_HAS_ONNXRUNTIME
    return true;
#else
    return false;
#endif
  }

 private:
  VisualGeoRelocConfig cfg_{};
  bool has_pending_megaloc_{false};
  core::Vec3Ned pending_megaloc_ned_{};
};

}  // namespace vertirta::gnss_denied
