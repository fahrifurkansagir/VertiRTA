#include "vertirta/gnss_denied/visual_georeloc_adapter.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::gnss_denied {

double Cov2D::max_eigenvalue() const noexcept {
  const double tr = p_nn + p_ee;
  const double diff = p_nn - p_ee;
  const double disc = std::sqrt(diff * diff + 4.0 * p_ne * p_ne);
  return std::max(0.0, 0.5 * (tr + disc));
}

double Cov2D::min_eigenvalue() const noexcept {
  const double tr = p_nn + p_ee;
  const double diff = p_nn - p_ee;
  const double disc = std::sqrt(diff * diff + 4.0 * p_ne * p_ne);
  return std::max(0.0, 0.5 * (tr - disc));
}

double Cov2D::sigma_max() const noexcept {
  return std::sqrt(max_eigenvalue());
}

double Cov2D::directional_variance(double n_north, double n_east) const noexcept {
  const double v = n_north * n_north * p_nn +
                   2.0 * n_north * n_east * p_ne +
                   n_east * n_east * p_ee;
  return std::max(0.0, v);
}

VisualGeoRelocAdapter::VisualGeoRelocAdapter(const VisualGeoRelocConfig& cfg) noexcept : cfg_(cfg) {}

void VisualGeoRelocAdapter::reset() noexcept {
  has_pending_megaloc_ = false;
  pending_megaloc_ned_ = {};
}

core::Vec3Ned VisualGeoRelocAdapter::geodetic_to_kartal_ned(const core::GeodeticWgs84& geo) noexcept {
  // Pure vertirta::core::geodesy projection (ZERO libproj/GDAL dependency).
  const core::Vec3Enu enu = core::geodetic_to_local_enu(geo, core::kKartalDatum);
  return core::enu_to_ned(enu);
}

GeoRelocDecision VisualGeoRelocAdapter::fuse_observation(
    const core::Vec3Ned& prior_pos_ned,
    const Cov2D& prior_cov,
    const VisualMatchObservation& obs) noexcept {
  GeoRelocDecision out{};
  out.updated_pos_ned = prior_pos_ned;
  out.updated_cov = prior_cov;

  const bool prior_cov_ok =
      std::isfinite(prior_cov.p_nn) && std::isfinite(prior_cov.p_ne) && std::isfinite(prior_cov.p_ee) &&
      prior_cov.p_nn >= 0.0 && prior_cov.p_ee >= 0.0 &&
      (prior_cov.p_nn * prior_cov.p_ee - prior_cov.p_ne * prior_cov.p_ne) >= -1e-12;
  if (!obs.valid || !prior_pos_ned.is_finite() || !prior_cov_ok ||
      !std::isfinite(obs.matched_geo.lat_deg) || !std::isfinite(obs.matched_geo.lon_deg) ||
      !std::isfinite(obs.matched_geo.h_hae_m) ||
      !std::isfinite(obs.reproj_rmse_px) || !std::isfinite(obs.convex_hull_ratio)) {
    out.reason = GeoRelocRejectReason::kInvalidInput;
    return out;
  }

  const double prior_sigma = prior_cov.sigma_max();
  if (!std::isfinite(prior_sigma)) {
    out.reason = GeoRelocRejectReason::kInvalidInput;
    return out;
  }
  out.megaloc_requested = (prior_sigma > cfg_.megaloc_trigger_sigma_m);

  // Sentinel-2 10 m scale chasm lock (~45 m CEP50): GROSS-BLUNDER ONLY — NEVER tighten LTFJ/LTBX corridor P!
  if (obs.map_tier == OrthoMapTier::kSentinel2GrossBlunder10m &&
      prior_sigma <= cfg_.sentinel2_cep50_sigma_m) {
    out.reason = GeoRelocRejectReason::kSentinel2GrossBlunderOnly;
    return out;
  }

  // Gate 1: RANSAC inliers >= 25 AND reprojection RMSE < 2.5 px.
  if (obs.ransac_inliers < cfg_.min_ransac_inliers) {
    out.reason = GeoRelocRejectReason::kTooFewInliers;
    return out;
  }
  if (obs.reproj_rmse_px >= cfg_.max_reproj_rmse_px) {
    out.reason = GeoRelocRejectReason::kHighReprojRmse;
    return out;
  }

  // Gate 2: Spatial convex-hull coverage > 0.25 (blocks single-roof / narrow-road false locks).
  if (obs.convex_hull_ratio <= cfg_.min_convex_hull_ratio) {
    out.reason = GeoRelocRejectReason::kLowConvexHullCoverage;
    return out;
  }

  const core::Vec3Ned meas_ned = geodetic_to_kartal_ned(obs.matched_geo);
  const double meas_sigma = (obs.map_tier == OrthoMapTier::kSentinel2GrossBlunder10m)
                                ? cfg_.sentinel2_cep50_sigma_m
                                : cfg_.ortho_05m_meas_sigma_m;
  const double r_var = meas_sigma * meas_sigma;

  // Innovation covariance S = P^- + R (since H = I_2)
  const double s_nn = prior_cov.p_nn + r_var;
  const double s_ne = prior_cov.p_ne;
  const double s_ee = prior_cov.p_ee + r_var;
  const double det_s = s_nn * s_ee - s_ne * s_ne;
  if (!std::isfinite(det_s) || det_s <= 1e-9) {
    out.reason = GeoRelocRejectReason::kInvalidInput;
    return out;
  }
  const double inv_det = 1.0 / det_s;
  const double sinv_nn = s_ee * inv_det;
  const double sinv_ne = -s_ne * inv_det;
  const double sinv_ee = s_nn * inv_det;

  const double yn = meas_ned.x - prior_pos_ned.x;
  const double ye = meas_ned.y - prior_pos_ned.y;
  const double chi2 = yn * yn * sinv_nn + 2.0 * yn * ye * sinv_ne + ye * ye * sinv_ee;
  if (!std::isfinite(chi2)) {
    out.reason = GeoRelocRejectReason::kInvalidInput;
    return out;
  }
  out.mahalanobis_chi2 = chi2;

  // Gate 3: 2D Mahalanobis innovation gate (chi^2 <= 9.21), bypassed ONLY by 2-keyframe corroborated MegaLoc.
  if (obs.from_megaloc && out.megaloc_requested) {
    if (!has_pending_megaloc_) {
      has_pending_megaloc_ = true;
      pending_megaloc_ned_ = meas_ned;
      out.reason = GeoRelocRejectReason::kMegaLocAwaitingCorroboration;
      return out;
    }
    const double d_corrob = std::hypot(meas_ned.x - pending_megaloc_ned_.x,
                                       meas_ned.y - pending_megaloc_ned_.y);
    if (d_corrob > cfg_.megaloc_corroborate_radius_m) {
      pending_megaloc_ned_ = meas_ned;
      out.reason = GeoRelocRejectReason::kMegaLocAwaitingCorroboration;
      return out;
    }
    // Validated 2-keyframe MegaLoc lost-in-space re-init!
    has_pending_megaloc_ = false;
    out.megaloc_relocked = true;
  } else if (chi2 > cfg_.mahalanobis_chi2_2dof_gate) {
    out.reason = GeoRelocRejectReason::kMahalanobisGateFailed;
    return out;
  }

  // Exact 2x2 Kalman gain K = P^- S^{-1}
  const double k00 = prior_cov.p_nn * sinv_nn + prior_cov.p_ne * sinv_ne;
  const double k01 = prior_cov.p_nn * sinv_ne + prior_cov.p_ne * sinv_ee;
  const double k10 = prior_cov.p_ne * sinv_nn + prior_cov.p_ee * sinv_ne;
  const double k11 = prior_cov.p_ne * sinv_ne + prior_cov.p_ee * sinv_ee;

  out.updated_pos_ned.x = prior_pos_ned.x + k00 * yn + k01 * ye;
  out.updated_pos_ned.y = prior_pos_ned.y + k10 * yn + k11 * ye;
  out.updated_pos_ned.z = prior_pos_ned.z;  // Vertical owned by radalt/baro lock

  // Numerically positive-definite Joseph-form update: P^+ = (I - K) P^- (I - K)^T + K R K^T
  const double a00 = 1.0 - k00;
  const double a01 = -k01;
  const double a10 = -k10;
  const double a11 = 1.0 - k11;

  // M = A * P^-
  const double m00 = a00 * prior_cov.p_nn + a01 * prior_cov.p_ne;
  const double m01 = a00 * prior_cov.p_ne + a01 * prior_cov.p_ee;
  const double m10 = a10 * prior_cov.p_nn + a11 * prior_cov.p_ne;
  const double m11 = a10 * prior_cov.p_ne + a11 * prior_cov.p_ee;

  // A * P^- * A^T + K * (r_var * I) * K^T
  const double p_nn_new = m00 * a00 + m01 * a01 + r_var * (k00 * k00 + k01 * k01);
  const double p_ne_new = 0.5 * ((m00 * a10 + m01 * a11) + (m10 * a00 + m11 * a01)) +
                          r_var * (k00 * k10 + k01 * k11);
  const double p_ee_new = m10 * a10 + m11 * a11 + r_var * (k10 * k10 + k11 * k11);

  if (!std::isfinite(out.updated_pos_ned.x) || !std::isfinite(out.updated_pos_ned.y) ||
      !std::isfinite(p_nn_new) || !std::isfinite(p_ne_new) || !std::isfinite(p_ee_new)) {
    out.updated_pos_ned = prior_pos_ned;
    out.updated_cov = prior_cov;
    out.reason = GeoRelocRejectReason::kInvalidInput;
    return out;
  }

  out.updated_cov.p_nn = std::max(0.25, p_nn_new);
  out.updated_cov.p_ne = p_ne_new;
  out.updated_cov.p_ee = std::max(0.25, p_ee_new);
  out.accepted = true;
  out.reason = GeoRelocRejectReason::kAccepted;
  return out;
}

}  // namespace vertirta::gnss_denied
