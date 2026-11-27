#include "vertirta/rta/cbvf_safety_filter.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::rta {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kCosPiOver8 = 0.9238795325112867;
constexpr double kCdR = 0.001225;   // 0.5 * rho * A_fp / m = 0.5*1.225*0.48/240
constexpr double kCdZ = 0.00245;    // Vertical parasitic drag coeff
constexpr double kDvrMax = 0.50;    // Horizontal velocity uncertainty [m/s]
constexpr double kDvzMax = 0.35;    // Vertical velocity uncertainty [m/s]
constexpr double kClutterInflateM = 1.50;
constexpr double kBrtFallbackThreshM = -0.25;

[[nodiscard]] inline bool is_finite_vec3(const core::Vec3Ned& v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[nodiscard]] inline double safe_clamp(double v, double lo, double hi, double fallback) noexcept {
  if (!std::isfinite(v)) {
    return fallback;
  }
  return std::clamp(v, lo, hi);
}

[[nodiscard]] inline double sanitize_scalar(double v, double lo, double hi, double fallback = 0.0) noexcept {
  if (!std::isfinite(v)) {
    return fallback;
  }
  return std::clamp(v, lo, hi);
}

[[nodiscard]] uint8_t select_lut_regime(const core::CapabilityEnvelope& env) noexcept {
  const bool low_vbat = !std::isfinite(env.v_term_v) || !std::isfinite(env.soc) ||
                        (env.v_term_v < 365.0f) || (env.soc < 0.15f);
  switch (env.regime) {
    case core::FlightRegime::NOMINAL:
      return low_vbat ? 3u : 0u;
    case core::FlightRegime::SINGLE_LOSS:
      return low_vbat ? 3u : 1u;
    case core::FlightRegime::CROSS_DUAL:
    case core::FlightRegime::SAME_ARM_ACTIVE_SPIN:
    case core::FlightRegime::CRITICAL_DESCENT:
    default:
      return 2u;
  }
}

struct EvaluatedHazard {
  double v_eff{1e6};
  double v_raw{1e6};
  std::array<double, 4> grad{};
  double er_n{1.0};
  double er_e{0.0};
  double s_z{1.0};
  double r_clamped{0.5};
  double v_r{0.0};
  double v_perp2{0.0};
  double vz_rel{0.0};
  double a_obs_r{0.0};
  double a_obs_z_up{0.0};
  double gamma{1.2};
  bool valid{false};
};

// Analytic conservative fallback when no .vrta_lut is bound (used only if caller passes nullptr).
[[nodiscard]] CbvfQueryResult analytic_cbvf_fallback(
    double r, double vr, double z, double vz, double a_r_eff, double a_z_eff) noexcept {
  const double v_close_r = std::max(0.0, kDvrMax - vr);
  const double t_br = std::min(6.0, v_close_r / std::max(a_r_eff, 0.5));
  const double sr = (r - 4.5) - (v_close_r * t_br - 0.5 * a_r_eff * t_br * t_br);

  const double v_close_z = std::max(0.0, kDvzMax - vz);
  const double t_bz = std::min(6.0, v_close_z / std::max(a_z_eff, 0.5));
  const double sz = (z - 3.0) - (v_close_z * t_bz - 0.5 * a_z_eff * t_bz * t_bz);

  const bool use_r = (sr >= sz);
  CbvfQueryResult out{};
  out.value_m = (use_r ? sr : sz) - 0.1733;
  if (use_r) {
    out.grad = {1.0, (v_close_r > 0.0) ? t_br : 0.0, 0.0, 0.0};
  } else {
    out.grad = {0.0, 0.0, 1.0, (v_close_z > 0.0) ? t_bz : 0.0};
  }
  out.valid = true;
  return out;
}

}  // namespace

CbvfSafetyFilter::CbvfSafetyFilter(const MmapCbvf4DLut* cbvf_lut, const Cel2DLut* cel_lut) noexcept
    : cbvf_lut_(cbvf_lut), cel_lut_(cel_lut) {}

void CbvfSafetyFilter::reset() noexcept {
  cel_latched_ = false;
  solver_.reset_warm_start();
}

core::Vec3Ned CbvfSafetyFilter::project_to_actuator_cone(
    const core::Vec3Ned& w_dir,
    double a_xy_max,
    double a_d_min,
    double a_d_max,
    double tilt_max_rad) const noexcept {
  if (!is_finite_vec3(w_dir) || !std::isfinite(a_xy_max) || !std::isfinite(a_d_min) ||
      !std::isfinite(a_d_max) || !std::isfinite(tilt_max_rad)) {
    return {0.0, 0.0, 0.0};
  }
  const double safe_xy = safe_clamp(a_xy_max, 0.5, 14.0, 0.5);
  const double safe_d_min = std::min(a_d_min, a_d_max);
  const double safe_d_max = std::max(a_d_min, a_d_max);
  const double tan_t = std::tan(safe_clamp(tilt_max_rad, 0.05, 0.85, 0.08));
  const double w_xy = std::hypot(w_dir.x, w_dir.y);
  const double ux = (std::isfinite(w_xy) && w_xy > 1e-9) ? (w_dir.x / w_xy) : 0.0;
  const double uy = (std::isfinite(w_xy) && w_xy > 1e-9) ? (w_dir.y / w_xy) : 0.0;

  // Maximize w_xy * r_xy + w_dir.z * a_D over a_D in [a_d_min, a_d_max],
  // 0 <= r_xy <= min(a_xy_max * cos(pi/8), (-a_D + g) * tan(tilt_max)).
  // Piecewise-linear concave objective -> optimum lies at one of at most 3 vertices!
  const double r_disk = safe_xy * kCosPiOver8;
  const double a_d_kink = safe_clamp(core::params::kGravity - r_disk / tan_t, safe_d_min, safe_d_max, 0.0);
  const double cand_ad[3] = {safe_d_min, a_d_kink, safe_d_max};

  double best_val = -1e30;
  core::Vec3Ned best{0.0, 0.0, 0.0};
  for (double ad : cand_ad) {
    const double r_cone = std::max(0.0, (-ad + core::params::kGravity) * tan_t);
    const double r_max = std::min(r_disk, r_cone);
    const double val = w_xy * r_max + w_dir.z * ad;
    if (std::isfinite(val) && val > best_val) {
      best_val = val;
      best = {ux * r_max, uy * r_max, ad};
    }
  }
  return is_finite_vec3(best) ? best : core::Vec3Ned{0.0, 0.0, 0.0};
}

SafetyFilterOutput CbvfSafetyFilter::step_100hz(
    const core::Vec3Ned& pos_ned,
    const core::Vec3Ned& vel_ned,
    const core::Vec3Ned& a_nom_ned,
    double yaw_rate_nom,
    const core::CapabilityEnvelope& env,
    std::span<const HazardPoint> hazards) noexcept {
  SafetyFilterOutput out{};

  // 1. Irreversible CEL latch on SAME_ARM_ACTIVE_SPIN or CRITICAL_DESCENT.
  if (env.regime == core::FlightRegime::SAME_ARM_ACTIVE_SPIN ||
      env.regime == core::FlightRegime::CRITICAL_DESCENT) {
    cel_latched_ = true;
  }
  out.cel_latched = cel_latched_;

  // Fail-closed immediately if ego state is non-finite: bounded level safe hover.
  if (!is_finite_vec3(pos_ned) || !is_finite_vec3(vel_ned)) {
    out.a_safe_ned = {0.0, 0.0, 0.0};
    out.yaw_rate_safe = 0.0;
    out.slack = 0.0;
    out.used_hj_fallback = true;
    out.qp_exit_flag = 1;
    out.qp_iters = 0;
    return out;
  }

  // Sanitize nominal commands & envelope limits (std::clamp(NaN, lo, hi) returns NaN!).
  const core::Vec3Ned a_nom_safe{
      sanitize_scalar(a_nom_ned.x, -1e4, 1e4, 0.0),
      sanitize_scalar(a_nom_ned.y, -1e4, 1e4, 0.0),
      sanitize_scalar(a_nom_ned.z, -1e4, 1e4, 0.0),
  };
  const double yaw_nom_safe = sanitize_scalar(yaw_rate_nom, -100.0, 100.0, 0.0);

  const double a_xy_max = safe_clamp(static_cast<double>(env.a_xy_max), 0.5, 14.0, 0.5);
  const double tilt_max = safe_clamp(static_cast<double>(env.tilt_max_rad), 0.08, 0.70, 0.08);
  const double a_up_raw = std::isfinite(env.t_max_n)
                              ? (static_cast<double>(env.t_max_n) / core::params::kMassKg - core::params::kGravity)
                              : 0.8;
  const double a_up_max = safe_clamp(a_up_raw, 0.8, 30.0, 0.8);
  const double a_d_min = -a_up_max;
  const double a_d_max = 4.0;

  const float v_term_safe = std::isfinite(env.v_term_v) ? env.v_term_v : 330.0f;
  const bool spin_meas_valid = std::isfinite(env.spin_rate_meas);
  const double spin_rate_meas = spin_meas_valid ? static_cast<double>(env.spin_rate_meas) : 0.0;
  const double spin_rate_ref = safe_clamp(static_cast<double>(env.spin_rate_ref), -10.0, 10.0, 0.0);

  // 2. Latched CEL mode: 2D vertical braking barrier (Cel2DLut) + Option B1b spin authority.
  if (cel_latched_) {
    const double z_agl = std::max(0.0, -pos_ned.z);
    const double vz_up = -vel_ned.z;
    double a_up_cmd = -a_nom_safe.z;
    double a_net_avail = a_up_max - 0.50;

    if (cel_lut_ != nullptr && cel_lut_->is_loaded()) {
      const Cel2DQuery cq = cel_lut_->eval(v_term_safe, z_agl, vz_up);
      if (cq.valid && std::isfinite(cq.value_m) && std::isfinite(cq.dv_dz) &&
          std::isfinite(cq.dv_dvz) && std::isfinite(cq.d_z_max)) {
        out.cel_v_2d = cq.value_m;
        out.min_cbvf_v = cq.value_m;
        a_net_avail = cq.a_net;
        constexpr double kGammaCel = 1.5;
        const double lhs = cq.dv_dz * vz_up + cq.dv_dvz * a_up_cmd - std::abs(cq.dv_dvz) * cq.d_z_max;
        if (cq.value_m < 0.0 || lhs < -kGammaCel * cq.value_m) {
          const double denom = std::max(cq.dv_dvz, 0.05);
          const double a_up_req =
              (-kGammaCel * cq.value_m - cq.dv_dz * vz_up + std::abs(cq.dv_dvz) * cq.d_z_max) / denom;
          if (std::isfinite(a_up_req)) {
            a_up_cmd = std::max(a_up_cmd, a_up_req);
          }
        }
      }
    }

    // Scheduled Regime-3 sink profile: <=2.5 >10m, <=1.5 <5m, <=0.8 touchdown
    const double vz_sink_lim = (z_agl > 10.0) ? 2.5 : ((z_agl > 5.0) ? 1.5 : ((z_agl > 1.0) ? 0.9 : 0.6));
    if (-vz_up > vz_sink_lim) {
      a_up_cmd = std::max(a_up_cmd, 2.5 * (-vz_up - vz_sink_lim));
    } else if (spin_meas_valid && std::abs(spin_rate_meas) < 1.5 && z_agl > 12.0 && -vz_up < 2.2) {
      // Controlled collective unload during high-AGL Option B1b spin-up opens differential yaw headroom.
      a_up_cmd = std::min(a_up_cmd, -0.6);
    }
    (void)a_net_avail;

    const double a_up_safe = safe_clamp(a_up_cmd, -a_d_max, a_up_max, 0.0);
    // Keep body level during spin-up; gentle horizontal damping only once flat spin is established.
    const bool spin_locked = spin_meas_valid && (std::abs(spin_rate_meas) >= 1.5);
    const double a_n = spin_locked ? safe_clamp(-0.25 * vel_ned.x, -0.5, 0.5, 0.0) : 0.0;
    const double a_e = spin_locked ? safe_clamp(-0.25 * vel_ned.y, -0.5, 0.5, 0.0) : 0.0;
    out.a_safe_ned = {a_n, a_e, -a_up_safe};
    out.yaw_rate_safe = spin_rate_ref;
    out.qp_exit_flag = 1;
    out.qp_iters = 0;
    return out;
  }

  // 3. CSFL mode: evaluate up to 15 closest/most-critical hazards against 4D CBVF LUT.
  const uint8_t regime_idx = select_lut_regime(env);
  const double d_ar_raw = (cbvf_lut_ != nullptr && cbvf_lut_->is_loaded()) ? cbvf_lut_->d_ar_max() : 1.75;
  const double d_az_raw = (cbvf_lut_ != nullptr && cbvf_lut_->is_loaded()) ? cbvf_lut_->d_az_max() : 0.75;
  const double v_cap_raw = (cbvf_lut_ != nullptr && cbvf_lut_->is_loaded()) ? cbvf_lut_->v_cap_f(regime_idx) : 16.0;
  const double d_ar = safe_clamp(d_ar_raw, 0.0, 10.0, 1.75);
  const double d_az = safe_clamp(d_az_raw, 0.0, 10.0, 0.75);
  const double v_cap = safe_clamp(v_cap_raw, 1.0, 40.0, 16.0);

  constexpr std::size_t kMaxActiveHazards = 15;
  std::array<EvaluatedHazard, kMaxActiveHazards> evals{};
  std::size_t n_eval = 0;

  const double p_up = -pos_ned.z;
  const double v_up = -vel_ned.z;

  for (const HazardPoint& h : hazards) {
    if (!is_finite_vec3(h.pos_ned)) {
      continue;
    }
    const double d_n = pos_ned.x - h.pos_ned.x;
    const double d_e = pos_ned.y - h.pos_ned.y;
    const double r_raw = std::hypot(d_n, d_e);
    if (!std::isfinite(r_raw)) {
      continue;
    }
    const double r_clamped = std::max(r_raw, 0.5);
    const double er_n = (r_raw > 1e-6) ? (d_n / r_raw) : 1.0;
    const double er_e = (r_raw > 1e-6) ? (d_e / r_raw) : 0.0;

    // Sanitize hazard kinematics & parameters so non-finite metadata never poisons CBVF or drops a nearby obstacle.
    const double h_vx = sanitize_scalar(h.vel_ned.x, -1e4, 1e4, 0.0);
    const double h_vy = sanitize_scalar(h.vel_ned.y, -1e4, 1e4, 0.0);
    const double h_vz = sanitize_scalar(h.vel_ned.z, -1e4, 1e4, 0.0);
    const double h_ax = sanitize_scalar(h.accel_ned.x, -1e3, 1e3, 0.0);
    const double h_ay = sanitize_scalar(h.accel_ned.y, -1e3, 1e3, 0.0);
    const double h_az = sanitize_scalar(h.accel_ned.z, -1e3, 1e3, 0.0);
    const double h_rad_h = safe_clamp(h.radius_h_m, 0.0, 1e5, 0.0);
    const double h_rad_z = safe_clamp(h.radius_z_m, 0.0, 1e5, 0.0);
    const double h_gamma = safe_clamp(h.gamma, 0.2, 5.0, 1.2);

    const double v_rel_n = vel_ned.x - h_vx;
    const double v_rel_e = vel_ned.y - h_vy;
    const double v_r = er_n * v_rel_n + er_e * v_rel_e;
    const double v_perp2_raw = v_rel_n * v_rel_n + v_rel_e * v_rel_e - v_r * v_r;
    const double v_perp2 = std::isfinite(v_perp2_raw) ? std::max(0.0, v_perp2_raw) : 0.0;

    // Exact overhead-intruder sign preservation (DR-2 Erratum #2 & Review Focus #2):
    // p_up - p_obs_up < 0 (intruder overhead) => s_z = -1 => climbing (+v_up) gives vz_rel < 0 (closing).
    const double p_obs_up = -h.pos_ned.z;
    const double v_obs_up = -h_vz;
    const double dz_signed = p_up - p_obs_up;
    if (!std::isfinite(dz_signed) || !std::isfinite(v_r)) {
      continue;
    }
    const double s_z = (dz_signed >= 0.0) ? 1.0 : -1.0;
    const double z_eff = std::abs(dz_signed) - h_rad_z;
    const double vz_rel = s_z * (v_up - v_obs_up);
    const double r_eff = r_clamped - h_rad_h;
    if (!std::isfinite(z_eff) || !std::isfinite(vz_rel) || !std::isfinite(r_eff)) {
      continue;
    }

    CbvfQueryResult q{};
    if (z_eff < -5.0) {
      // Tall vertical barrier / CTR wall (e.g., LTFJ 762 m ceiling): vertical climb-over is non-viable;
      // enforce exact horizontal radial braking CBVF so LUT z-clamping at -5 m never hallucinates climb escape.
      const double a_r_eff = std::max(0.5, a_xy_max - d_ar);
      const double horizon = (cbvf_lut_ != nullptr && cbvf_lut_->is_loaded()) ? cbvf_lut_->horizon_s() : 6.0;
      const double eps = (cbvf_lut_ != nullptr && cbvf_lut_->is_loaded()) ? cbvf_lut_->eps_grid(regime_idx) : 0.15;
      const double v_close_r = std::max(0.0, kDvrMax - v_r);
      const double t_br = std::min(horizon, v_close_r / a_r_eff);
      q.value_m = (r_eff - 4.5) - (v_close_r * t_br - 0.5 * a_r_eff * t_br * t_br) - eps;
      q.grad = {1.0, (v_close_r > 0.0) ? t_br : 0.0, 0.0, 0.0};
      q.valid = true;
    } else if (cbvf_lut_ != nullptr && cbvf_lut_->is_loaded()) {
      q = cbvf_lut_->eval(regime_idx, r_eff, v_r, z_eff, vz_rel);
    } else {
      q = analytic_cbvf_fallback(r_eff, v_r, z_eff, vz_rel, std::max(0.5, a_xy_max - d_ar), std::max(0.5, a_up_max - d_az));
    }
    if (!q.valid || !std::isfinite(q.value_m) ||
        !std::isfinite(q.grad[0]) || !std::isfinite(q.grad[1]) ||
        !std::isfinite(q.grad[2]) || !std::isfinite(q.grad[3])) {
      continue;
    }

    EvaluatedHazard eh{};
    eh.v_raw = q.value_m;
    eh.v_eff = q.value_m;
    eh.grad = q.grad;
    eh.er_n = er_n;
    eh.er_e = er_e;
    eh.s_z = s_z;
    eh.r_clamped = r_clamped;
    eh.v_r = v_r;
    eh.v_perp2 = v_perp2;
    eh.vz_rel = vz_rel;
    eh.a_obs_r = er_n * h_ax + er_e * h_ay;
    eh.a_obs_z_up = -h_az;
    eh.gamma = h_gamma;
    eh.valid = true;

    if (n_eval < kMaxActiveHazards) {
      evals[n_eval++] = eh;
    } else {
      // Keep the 15 most-critical (smallest v_raw) hazards without heap allocation.
      std::size_t max_idx = 0;
      for (std::size_t j = 1; j < kMaxActiveHazards; ++j) {
        if (evals[j].v_raw > evals[max_idx].v_raw) {
          max_idx = j;
        }
      }
      if (eh.v_raw < evals[max_idx].v_raw) {
        evals[max_idx] = eh;
      }
    }
  }

  // 4. Opposing-gradient clutter inflation (delta_clutter = 1.5 m when e_r,i . e_r,j < -0.5).
  bool inflated_mask[kMaxActiveHazards]{};
  for (std::size_t i = 0; i < n_eval; ++i) {
    if (evals[i].v_raw > 12.0) {
      continue;
    }
    for (std::size_t j = i + 1; j < n_eval; ++j) {
      if (evals[j].v_raw > 12.0) {
        continue;
      }
      const double dot_er = evals[i].er_n * evals[j].er_n + evals[i].er_e * evals[j].er_e;
      if (dot_er < -0.5) {
        inflated_mask[i] = true;
        inflated_mask[j] = true;
        out.clutter_inflated = true;
      }
    }
  }
  for (std::size_t i = 0; i < n_eval; ++i) {
    if (inflated_mask[i]) {
      evals[i].v_eff -= kClutterInflateM;
    }
  }

  // Identify most-critical hazard k*
  std::size_t k_star = 0;
  double min_v = 1e6;
  for (std::size_t i = 0; i < n_eval; ++i) {
    if (evals[i].v_eff < min_v) {
      min_v = evals[i].v_eff;
      k_star = i;
    }
  }
  out.active_hazards = static_cast<uint16_t>(n_eval);
  out.min_cbvf_v = min_v;
  if (n_eval > 0) {
    out.grad_min = evals[k_star].grad;
    out.last_mapped_r_min = evals[k_star].r_clamped;
    out.last_mapped_vz_min = evals[k_star].vz_rel;
  }

  // 5. BRT-outside deterministic fallback (min_V < -0.25 m) -> closed-form u*_HJ cone projection.
  if (n_eval > 0 && min_v < kBrtFallbackThreshM) {
    const auto& hs = evals[k_star];
    // Escape direction maximizing grad(V) . g(x) u = [V_vr*er_n, V_vr*er_e, -s_z*V_vz]
    double wx = hs.grad[1] * hs.er_n;
    double wy = hs.grad[1] * hs.er_e;
    double wz = -hs.s_z * hs.grad[3];
    if (std::hypot(wx, wy) < 1e-6 && std::abs(wz) < 1e-6) {
      // Degenerate zero velocity gradient: push along spatial gradient (away from hazard)
      wx = hs.grad[0] * hs.er_n;
      wy = hs.grad[0] * hs.er_e;
      wz = -hs.s_z * hs.grad[2];
    }
    out.a_safe_ned = project_to_actuator_cone({wx, wy, wz}, a_xy_max, a_d_min, a_d_max, tilt_max);
    out.yaw_rate_safe = safe_clamp(yaw_nom_safe, -1.5, 1.5, 0.0);
    out.used_hj_fallback = true;
    out.qp_exit_flag = 1;
    out.qp_iters = 0;
    return out;
  }

  // 6. Pack 5-var QP into StaticDaqpSolver<5, 32>:
  //    u = [a_N, a_E, a_D, yaw_rate, slack]
  //    General rows: (<=14 if v_cap active else <=15) CBVF rows + 1 optional v_cap row + 12 actuator polytope rows <= 27
  //    Total rows: n(5) + m_general(<=27) <= 32.
  const double v_h = std::hypot(vel_ned.x, vel_ned.y);
  const bool need_vcap_row = std::isfinite(v_h) && (v_h > v_cap + 0.1) && (v_h > 1e-6);
  const std::size_t max_haz_rows = need_vcap_row ? 14u : 15u;
  const std::size_t n_haz_rows = std::min(n_eval, max_haz_rows);

  const double H_diag[5] = {1.0, 1.0, 2.0, 1.0, 1e4};
  const double f[5] = {
      -a_nom_safe.x,
      -a_nom_safe.y,
      -2.0 * a_nom_safe.z,
      -yaw_nom_safe,
      1e3,
  };
  const double x_lo[5] = {-a_xy_max, -a_xy_max, a_d_min, -1.5, 0.0};
  const double x_hi[5] = {+a_xy_max, +a_xy_max, a_d_max, +1.5, 200.0};

  double A[27 * 5]{};
  double b_lo[27]{};
  double b_hi[27]{};
  int row = 0;

  // 6a. CBVF hazard rows
  for (std::size_t i = 0; i < n_haz_rows; ++i) {
    const auto& eh = evals[i];
    const double vr_g = eh.grad[0];
    const double vvr_g = eh.grad[1];
    const double vz_g = eh.grad[2];
    const double vvz_g = eh.grad[3];

    const double lf_v =
        vr_g * eh.v_r +
        vvr_g * (eh.v_perp2 / eh.r_clamped - kCdR * eh.v_r * std::abs(eh.v_r) - eh.a_obs_r) +
        vz_g * eh.vz_rel -
        vvz_g * (kCdZ * eh.vz_rel * std::abs(eh.vz_rel) + eh.s_z * eh.a_obs_z_up);

    const double d_term =
        std::abs(vr_g) * kDvrMax + std::abs(vvr_g) * d_ar +
        std::abs(vz_g) * kDvzMax + std::abs(vvz_g) * d_az;

    const double rhs = -lf_v + d_term - eh.gamma * eh.v_eff;
    if (!std::isfinite(rhs)) {
      continue;
    }

    // L_g V * u + slack >= rhs, where d(vz_rel)/d(a_D) = -s_z
    A[row * 5 + 0] = vvr_g * eh.er_n;
    A[row * 5 + 1] = vvr_g * eh.er_e;
    A[row * 5 + 2] = -eh.s_z * vvz_g;
    A[row * 5 + 3] = 0.0;
    A[row * 5 + 4] = 1.0;
    b_lo[row] = rhs;
    b_hi[row] = +1e20;
    ++row;
  }

  // 6b. Optional degraded v_cap_f speed-cap braking row (DR-2 Erratum #3)
  if (need_vcap_row) {
    const double vh_n = vel_ned.x / v_h;
    const double vh_e = vel_ned.y / v_h;
    // vh_hat . a_xy - slack <= -1.5 * (v_h - v_cap)
    A[row * 5 + 0] = vh_n;
    A[row * 5 + 1] = vh_e;
    A[row * 5 + 2] = 0.0;
    A[row * 5 + 3] = 0.0;
    A[row * 5 + 4] = -1.0;
    b_lo[row] = -1e20;
    b_hi[row] = -1.5 * (v_h - v_cap);
    ++row;
  }

  // 6c. 8-gon inscribed horizontal acceleration disk (3.8% radial conservatism)
  const double r_disk = a_xy_max * kCosPiOver8;
  for (int j = 0; j < 8; ++j) {
    const double ang = static_cast<double>(j) * (0.25 * kPi);
    A[row * 5 + 0] = std::cos(ang);
    A[row * 5 + 1] = std::sin(ang);
    A[row * 5 + 2] = 0.0;
    A[row * 5 + 3] = 0.0;
    A[row * 5 + 4] = 0.0;
    b_lo[row] = -1e20;
    b_hi[row] = r_disk;
    ++row;
  }

  // 6d. 4 tilt-thrust cone coupling rows: +/- a_N, +/- a_E <= (-a_D + g) * tan(tilt_max)
  //     => +/- a_N + tan(tilt_max) * a_D <= g * tan(tilt_max)
  const double tan_t = std::tan(tilt_max);
  const double g_tan_t = core::params::kGravity * tan_t;
  const double tilt_signs[4][2] = {{+1.0, 0.0}, {-1.0, 0.0}, {0.0, +1.0}, {0.0, -1.0}};
  for (int j = 0; j < 4; ++j) {
    A[row * 5 + 0] = tilt_signs[j][0];
    A[row * 5 + 1] = tilt_signs[j][1];
    A[row * 5 + 2] = tan_t;
    A[row * 5 + 3] = 0.0;
    A[row * 5 + 4] = 0.0;
    b_lo[row] = -1e20;
    b_hi[row] = g_tan_t;
    ++row;
  }

  const core::DaqpSolveResult res = solver_.solve(5, row, H_diag, f, A, x_lo, x_hi, b_lo, b_hi, true);
  out.qp_exit_flag = res.exit_flag;
  out.qp_iters = res.iterations;

  const bool qp_finite = std::isfinite(res.x[0]) && std::isfinite(res.x[1]) &&
                         std::isfinite(res.x[2]) && std::isfinite(res.x[3]) &&
                         std::isfinite(res.x[4]);
  if (!res.used_fallback && qp_finite) {
    out.a_safe_ned = {res.x[0], res.x[1], res.x[2]};
    out.yaw_rate_safe = res.x[3];
    out.slack = res.x[4];
    out.used_hj_fallback = false;
  } else {
    // Non-optimal or non-finite QP -> deterministic u*_HJ cone fallback
    core::Vec3Ned w_dir{-vel_ned.x, -vel_ned.y, -1.0};
    if (n_eval > 0) {
      const auto& hs = evals[k_star];
      w_dir = {hs.grad[1] * hs.er_n, hs.grad[1] * hs.er_e, -hs.s_z * hs.grad[3]};
    }
    out.a_safe_ned = project_to_actuator_cone(w_dir, a_xy_max, a_d_min, a_d_max, tilt_max);
    out.yaw_rate_safe = safe_clamp(yaw_nom_safe, -1.5, 1.5, 0.0);
    out.used_hj_fallback = true;
  }

  // Final defense-in-depth finiteness postcondition guarantee.
  if (!is_finite_vec3(out.a_safe_ned) || !std::isfinite(out.yaw_rate_safe)) {
    out.a_safe_ned = {0.0, 0.0, 0.0};
    out.yaw_rate_safe = 0.0;
    out.slack = 0.0;
    out.used_hj_fallback = true;
  }

  return out;
}

}  // namespace vertirta::rta
