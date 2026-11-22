// SPDX-License-Identifier: Apache-2.0
#include "vertirta/perception/slz_belief_grid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "vertirta/perception/active_inspection_policy.hpp"

namespace vertirta::perception {

namespace {

constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr std::size_t kMaxDiskCells = 512u;

struct MinEigen3x3 {
  double lambda_min{0.0};
  double nx{0.0};
  double ny{0.0};
  double nz{1.0};
};

// Exact cyclic Jacobi eigendecomposition for 3x3 symmetric covariance matrices.
// Zero heap allocations; guarantees orthonormal eigenvectors and exact smallest eigenvalue.
[[nodiscard]] MinEigen3x3 SolveSymmetric3x3MinEigen(
    double xx, double xy, double xz,
    double yy, double yz, double zz) noexcept {
  double a[3][3] = {
      {xx, xy, xz},
      {xy, yy, yz},
      {xz, yz, zz},
  };
  double v[3][3] = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
  };

  constexpr int kPairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
  for (int sweep = 0; sweep < 15; ++sweep) {
    double max_off = 0.0;
    for (int k = 0; k < 3; ++k) {
      const int p = kPairs[k][0];
      const int q = kPairs[k][1];
      const double apq = a[p][q];
      const double abs_apq = std::fabs(apq);
      if (abs_apq > max_off) {
        max_off = abs_apq;
      }
      if (abs_apq < 1e-14) {
        continue;
      }

      const double tau = (a[q][q] - a[p][p]) / (2.0 * apq);
      const double t = (tau >= 0.0)
                           ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                           : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
      const double c = 1.0 / std::sqrt(1.0 + t * t);
      const double s = t * c;

      const double app = a[p][p];
      const double aqq = a[q][q];
      a[p][p] = app - t * apq;
      a[q][q] = aqq + t * apq;
      a[p][q] = 0.0;
      a[q][p] = 0.0;

      for (int r = 0; r < 3; ++r) {
        if (r != p && r != q) {
          const double arp = a[r][p];
          const double arq = a[r][q];
          a[r][p] = c * arp - s * arq;
          a[p][r] = a[r][p];
          a[r][q] = s * arp + c * arq;
          a[q][r] = a[r][q];
        }
        const double vrp = v[r][p];
        const double vrq = v[r][q];
        v[r][p] = c * vrp - s * vrq;
        v[r][q] = s * vrp + c * vrq;
      }
    }
    if (max_off < 1e-13) {
      break;
    }
  }

  int min_i = 0;
  if (a[1][1] < a[min_i][min_i]) min_i = 1;
  if (a[2][2] < a[min_i][min_i]) min_i = 2;

  double nx = v[0][min_i];
  double ny = v[1][min_i];
  double nz = v[2][min_i];
  const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
  if (norm > 1e-12) {
    nx /= norm;
    ny /= norm;
    nz /= norm;
  } else {
    nx = 0.0;
    ny = 0.0;
    nz = 1.0;
  }
  if (nz < 0.0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }

  return {std::max(0.0, a[min_i][min_i]), nx, ny, nz};
}

[[nodiscard]] float QuantileStack(
    std::array<float, kMaxDiskCells>& arr,
    std::size_t count,
    float q) noexcept {
  if (count == 0u) {
    return 0.0f;
  }
  const float q_clamped = std::clamp(q, 0.0f, 1.0f);
  const std::size_t idx = static_cast<std::size_t>(
      q_clamped * static_cast<float>(count - 1u) + 0.5f);
  const std::size_t safe_idx = std::min(idx, count - 1u);
  auto nth = arr.begin() + static_cast<std::ptrdiff_t>(safe_idx);
  std::nth_element(arr.begin(), nth, arr.begin() + static_cast<std::ptrdiff_t>(count));
  return *nth;
}

[[nodiscard]] float Sigmoid(float x) noexcept {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

[[nodiscard]] float ComputeOodFloor(
    float e_tilde,
    const SlzGridConfig& cfg) noexcept {
  const float excess = e_tilde - cfg.tau_ood;
  if (excess <= 0.0f) {
    return 0.0f;
  }
  const float scaled = std::min(excess / std::max(cfg.ood_kappa, 1e-3f), 20.0f);
  const float sp = std::log1p(std::exp(scaled)) - 0.69314718f;
  const float floor_v = cfg.ood_var_floor_min + cfg.alpha_ood * sp * sp;
  return std::clamp(floor_v, cfg.ood_var_floor_min, 1.0f);
}

}  // namespace

uint32_t CountCircularDiskCells(float r_m, float res_m) noexcept {
  if (!std::isfinite(r_m) || !std::isfinite(res_m) || r_m < 0.0f || res_m <= 1e-6f) {
    return 0u;
  }
  const int32_t r_cells = static_cast<int32_t>(std::floor(static_cast<double>(r_m) / static_cast<double>(res_m)));
  const double r_sq = static_cast<double>(r_m) * static_cast<double>(r_m) + 1e-9;
  const double res_d = static_cast<double>(res_m);
  uint32_t count = 0u;
  for (int32_t dy = -r_cells; dy <= r_cells; ++dy) {
    const double y = static_cast<double>(dy) * res_d;
    for (int32_t dx = -r_cells; dx <= r_cells; ++dx) {
      const double x = static_cast<double>(dx) * res_d;
      if (x * x + y * y <= r_sq) {
        ++count;
      }
    }
  }
  return count;
}

SlzBeliefGrid::SlzBeliefGrid(SlzGridConfig cfg) noexcept : cfg_(cfg) {
  Reset();
}

void SlzBeliefGrid::Reset() noexcept {
  origin_initialized_ = false;
  origin_ix_ = 0;
  origin_iy_ = 0;
  origin_z_m_ = 0.0;
  cells_.fill(SlzCell{});
}

void SlzBeliefGrid::SetOrigin(const core::Vec3Enu& origin_enu) noexcept {
  if (!origin_enu.is_finite()) {
    return;
  }
  const int32_t new_ix = static_cast<int32_t>(std::floor(origin_enu.x / kSlzGridResolutionM));
  const int32_t new_iy = static_cast<int32_t>(std::floor(origin_enu.y / kSlzGridResolutionM));
  if (!origin_initialized_) {
    origin_ix_ = new_ix;
    origin_iy_ = new_iy;
    origin_z_m_ = origin_enu.z;
    origin_initialized_ = true;
    cells_.fill(SlzCell{});
    return;
  }
  ShiftOriginTo(new_ix, new_iy);
}

core::Vec3Enu SlzBeliefGrid::origin_enu() const noexcept {
  return {
      (static_cast<double>(origin_ix_) + 0.5) * kSlzGridResolutionM,
      (static_cast<double>(origin_iy_) + 0.5) * kSlzGridResolutionM,
      origin_z_m_,
  };
}

uint32_t SlzBeliefGrid::WrapSlot(int32_t g) noexcept {
  const int32_t dim = static_cast<int32_t>(kSlzGridDim);
  int32_t m = g % dim;
  if (m < 0) {
    m += dim;
  }
  return static_cast<uint32_t>(m);
}

std::size_t SlzBeliefGrid::SlotIndex(int32_t gx, int32_t gy) const noexcept {
  return static_cast<std::size_t>(WrapSlot(gy)) * kSlzGridDim + WrapSlot(gx);
}

bool SlzBeliefGrid::WorldToGlobalIdx(
    double x_enu, double y_enu,
    int32_t& gx, int32_t& gy) const noexcept {
  if (!std::isfinite(x_enu) || !std::isfinite(y_enu)) {
    return false;
  }
  gx = static_cast<int32_t>(std::floor(x_enu / kSlzGridResolutionM));
  gy = static_cast<int32_t>(std::floor(y_enu / kSlzGridResolutionM));
  return true;
}

bool SlzBeliefGrid::IsGlobalInWindow(int32_t gx, int32_t gy) const noexcept {
  if (!origin_initialized_) {
    return false;
  }
  constexpr int32_t kHalf = static_cast<int32_t>(kSlzGridDim / 2u);
  return (gx >= origin_ix_ - kHalf) && (gx < origin_ix_ + kHalf) &&
         (gy >= origin_iy_ - kHalf) && (gy < origin_iy_ + kHalf);
}

void SlzBeliefGrid::ShiftOriginTo(int32_t new_origin_ix, int32_t new_origin_iy) noexcept {
  const int32_t dx = new_origin_ix - origin_ix_;
  const int32_t dy = new_origin_iy - origin_iy_;
  if (dx == 0 && dy == 0) {
    return;
  }

  constexpr int32_t kDim = static_cast<int32_t>(kSlzGridDim);
  constexpr int32_t kHalf = kDim / 2;
  if (std::abs(dx) >= kDim || std::abs(dy) >= kDim) {
    cells_.fill(SlzCell{});
    origin_ix_ = new_origin_ix;
    origin_iy_ = new_origin_iy;
    return;
  }

  // Clear columns scrolling into the robocentric ring window.
  if (dx > 0) {
    for (int32_t c = origin_ix_ + kHalf; c < new_origin_ix + kHalf; ++c) {
      const uint32_t sx = WrapSlot(c);
      for (uint32_t sy = 0; sy < kSlzGridDim; ++sy) {
        cells_[static_cast<std::size_t>(sy) * kSlzGridDim + sx] = SlzCell{};
      }
    }
  } else if (dx < 0) {
    for (int32_t c = new_origin_ix - kHalf; c < origin_ix_ - kHalf; ++c) {
      const uint32_t sx = WrapSlot(c);
      for (uint32_t sy = 0; sy < kSlzGridDim; ++sy) {
        cells_[static_cast<std::size_t>(sy) * kSlzGridDim + sx] = SlzCell{};
      }
    }
  }

  // Clear rows scrolling into the robocentric ring window.
  if (dy > 0) {
    for (int32_t r = origin_iy_ + kHalf; r < new_origin_iy + kHalf; ++r) {
      const uint32_t sy = WrapSlot(r);
      const std::size_t row_off = static_cast<std::size_t>(sy) * kSlzGridDim;
      for (uint32_t sx = 0; sx < kSlzGridDim; ++sx) {
        cells_[row_off + sx] = SlzCell{};
      }
    }
  } else if (dy < 0) {
    for (int32_t r = new_origin_iy - kHalf; r < origin_iy_ - kHalf; ++r) {
      const uint32_t sy = WrapSlot(r);
      const std::size_t row_off = static_cast<std::size_t>(sy) * kSlzGridDim;
      for (uint32_t sx = 0; sx < kSlzGridDim; ++sx) {
        cells_[row_off + sx] = SlzCell{};
      }
    }
  }

  origin_ix_ = new_origin_ix;
  origin_iy_ = new_origin_iy;
}

const SlzCell* SlzBeliefGrid::cell_at_world(double x_enu, double y_enu) const noexcept {
  int32_t gx = 0;
  int32_t gy = 0;
  if (!WorldToGlobalIdx(x_enu, y_enu, gx, gy) || !IsGlobalInWindow(gx, gy)) {
    return nullptr;
  }
  return &cells_[SlotIndex(gx, gy)];
}

void SlzBeliefGrid::UpdateFrame(
    std::span<const SlzPointObservation> obs,
    const core::Vec3Enu& vehicle_enu,
    double timestamp_s,
    float viewpoint_novelty,
    float extra_geo_var_floor) noexcept {
  if (obs.empty() || !std::isfinite(timestamp_s)) {
    return;
  }

  if (!origin_initialized_) {
    core::Vec3Enu init_origin = vehicle_enu;
    if (!init_origin.is_finite()) {
      init_origin = obs[0].point_enu;
    }
    // Anchor z_ref to first finite observed point so cell-local dz stays small (~O(1 m)).
    for (const auto& o : obs) {
      if (o.point_enu.is_finite()) {
        init_origin.z = o.point_enu.z;
        break;
      }
    }
    SetOrigin(init_origin);
  } else if (vehicle_enu.is_finite()) {
    int32_t vx = 0;
    int32_t vy = 0;
    if (WorldToGlobalIdx(vehicle_enu.x, vehicle_enu.y, vx, vy)) {
      ShiftOriginTo(vx, vy);
    }
  }

  const float rho_view = std::clamp(
      std::isfinite(viewpoint_novelty) ? viewpoint_novelty : 0.05f, 0.05f, 1.0f);
  const float geo_extra = std::max(0.0f, std::isfinite(extra_geo_var_floor) ? extra_geo_var_floor : 0.0f);

  // Offset by +1 ms so last_ts_ms == 0u remains an unambiguous uninitialized sentinel.
  const double ts_ms_d = std::max(0.0, timestamp_s) * 1000.0;
  const uint32_t frame_ts_ms = static_cast<uint32_t>(std::min(ts_ms_d, 4.0e9) + 0.5) + 1u;

  for (const auto& o : obs) {
    if (!o.point_enu.is_finite()) {
      continue;
    }
    int32_t gx = 0;
    int32_t gy = 0;
    if (!WorldToGlobalIdx(o.point_enu.x, o.point_enu.y, gx, gy) || !IsGlobalInWindow(gx, gy)) {
      continue;
    }

    SlzCell& cell = cells_[SlotIndex(gx, gy)];

    // 1) Cell-local shifted 3D moments dp = p_W - c_ij (eliminates Maltepe +/-3500 m cancellation).
    const double cell_cx = (static_cast<double>(gx) + 0.5) * kSlzGridResolutionM;
    const double cell_cy = (static_cast<double>(gy) + 0.5) * kSlzGridResolutionM;
    const float dx = static_cast<float>(o.point_enu.x - cell_cx);
    const float dy = static_cast<float>(o.point_enu.y - cell_cy);
    const float dz = static_cast<float>(o.point_enu.z - origin_z_m_);

    if (cell.N < 1000000u) {
      cell.s[0] += dx;
      cell.s[1] += dy;
      cell.s[2] += dz;
      cell.M[0] += dx * dx;
      cell.M[1] += dx * dy;
      cell.M[2] += dx * dz;
      cell.M[3] += dy * dy;
      cell.M[4] += dy * dz;
      cell.M[5] += dz * dz;
      ++cell.N;
    }

    const float p_safe = std::clamp(
        std::isfinite(o.p_safe_obs) ? o.p_safe_obs : 0.001f, 1e-3f, 1.0f - 1e-3f);
    const float p_dyn = std::clamp(
        std::isfinite(o.p_dyn_obs) ? o.p_dyn_obs : 1.0f, 0.0f, 1.0f);
    const float e_tilde = std::isfinite(o.e_tilde_ood) ? o.e_tilde_ood : 0.0f;
    const float cos_inc = std::clamp(
        std::isfinite(o.cos_incidence) ? o.cos_incidence : 0.5f, 0.05f, 1.0f);

    const float l_obs = std::log(p_safe / (1.0f - p_safe));
    const float rel_exp = std::min(cfg_.ood_beta * (e_tilde - cfg_.tau_ood), 20.0f);
    const float w_rel = cos_inc / (1.0f + std::exp(rel_exp));
    const float w_eff = (p_safe < 0.5f) ? (2.0f * w_rel) : w_rel;

    const float ood_excess = e_tilde - cfg_.tau_ood;
    const float sp_arg = std::min(std::max(ood_excess, 0.0f) / std::max(cfg_.ood_kappa, 1e-3f), 20.0f);
    const float sp_val = std::log1p(std::exp(sp_arg));
    const float sigma2_obs =
        cfg_.sigma2_base_rgbd + geo_extra + cfg_.alpha_ood * sp_val * sp_val;

    const bool first_hit_this_frame = (cell.last_ts_ms != frame_ts_ms);
    if (first_hit_this_frame) {
      if (cell.last_ts_ms != 0u && frame_ts_ms > cell.last_ts_ms) {
        const float dt_s = static_cast<float>(frame_ts_ms - cell.last_ts_ms) * 1e-3f;
        cell.L *= std::exp(-dt_s / std::max(cfg_.tau_static_s, 1e-3f));
        cell.h_dyn *= std::exp(-dt_s / std::max(cfg_.tau_dyn_s, 1e-3f));
        cell.lambda_prec *= std::exp(-dt_s / std::max(cfg_.tau_var_s, 1e-3f));
        const float ood_decay = std::exp(-dt_s / std::max(cfg_.tau_ood_decay_s, 1e-3f));
        cell.ood_ema *= ood_decay;
        cell.sigma2_floor *= ood_decay;
      }
      cell.last_ts_ms = frame_ts_ms;
      cell.L = std::clamp(cell.L + w_eff * l_obs, -8.0f, 8.0f);
      cell.lambda_prec += rho_view / std::max(sigma2_obs, 1e-4f);
    } else if (p_safe < 0.5f) {
      // Conservative hazard fusion if any pixel inside the same cell observes an obstacle.
      cell.L = std::clamp(cell.L + 0.25f * w_eff * l_obs, -8.0f, 8.0f);
    }

    // 2) Asymmetric dynamic hazard tracking (instant attack, tau_dyn = 5 s decay).
    cell.h_dyn = std::max(cell.h_dyn, p_dyn);

    // 3) DR-7 Erratum #2: Explicit OOD variance floor + Lambda_k cap/reset on OOD trigger.
    // Prevents additive precision collapse when a new OOD tarp appears on a clean roof
    // and keeps sigma2 >= 0.08 across repeated OOD frames.
    if (ood_excess > 0.0f) {
      const float floor_v = ComputeOodFloor(e_tilde, cfg_);
      cell.ood_ema = std::min(30.0f, std::max(ood_excess, 0.35f * cell.ood_ema + 0.65f * ood_excess));
      cell.sigma2_floor = std::max(cell.sigma2_floor, floor_v);
      const float lambda_cap = 1.0f / std::max(cell.sigma2_floor, cfg_.ood_var_floor_min);
      cell.lambda_prec = std::min(cell.lambda_prec, lambda_cap);
    } else if (geo_extra > 0.0f) {
      // Mod B monocular altitude variance floor also bounds optimistic precision.
      cell.sigma2_floor = std::max(cell.sigma2_floor, geo_extra);
    }
  }
}

SlzBeliefGrid::LocalPlaneFit SlzBeliefGrid::FitWindow3x3(int32_t gx, int32_t gy) const noexcept {
  LocalPlaneFit fit{};
  if (!IsGlobalInWindow(gx, gy)) {
    return fit;
  }

  uint32_t n_win = 0u;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double m_xx = 0.0;
  double m_xy = 0.0;
  double m_xz = 0.0;
  double m_yy = 0.0;
  double m_yz = 0.0;
  double m_zz = 0.0;

  for (int32_t dy = -1; dy <= 1; ++dy) {
    for (int32_t dx = -1; dx <= 1; ++dx) {
      const int32_t nx_idx = gx + dx;
      const int32_t ny_idx = gy + dy;
      if (!IsGlobalInWindow(nx_idx, ny_idx)) {
        continue;
      }
      const SlzCell& c = cells_[SlotIndex(nx_idx, ny_idx)];
      if (c.N == 0u) {
        continue;
      }

      // Parallel-axis shift from neighbor cell frame to center cell (gx, gy) frame.
      // Offsets ox, oy are strictly in {-0.5, 0.0, +0.5} m => zero catastrophic cancellation.
      const double ox = static_cast<double>(dx) * kSlzGridResolutionM;
      const double oy = static_cast<double>(dy) * kSlzGridResolutionM;
      const double nd = static_cast<double>(c.N);
      const double sx = static_cast<double>(c.s[0]);
      const double sy = static_cast<double>(c.s[1]);
      const double sz = static_cast<double>(c.s[2]);

      sum_x += sx + nd * ox;
      sum_y += sy + nd * oy;
      sum_z += sz;

      m_xx += static_cast<double>(c.M[0]) + 2.0 * ox * sx + nd * ox * ox;
      m_xy += static_cast<double>(c.M[1]) + ox * sy + oy * sx + nd * ox * oy;
      m_xz += static_cast<double>(c.M[2]) + ox * sz;
      m_yy += static_cast<double>(c.M[3]) + 2.0 * oy * sy + nd * oy * oy;
      m_yz += static_cast<double>(c.M[4]) + oy * sz;
      m_zz += static_cast<double>(c.M[5]);
      n_win += c.N;
    }
  }

  fit.n_pts = n_win;
  if (n_win < cfg_.min_points_plane_fit) {
    return fit;  // Conservative invalid: theta = 90 deg, sigma_z = 1.0 m.
  }

  const double inv_n = 1.0 / static_cast<double>(n_win);
  const double mx = sum_x * inv_n;
  const double my = sum_y * inv_n;
  const double mz = sum_z * inv_n;

  const double cov_xx = std::max(0.0, m_xx * inv_n - mx * mx);
  const double cov_xy = m_xy * inv_n - mx * my;
  const double cov_xz = m_xz * inv_n - mx * mz;
  const double cov_yy = std::max(0.0, m_yy * inv_n - my * my);
  const double cov_yz = m_yz * inv_n - my * mz;
  const double cov_zz = std::max(0.0, m_zz * inv_n - mz * mz);

  const MinEigen3x3 eig = SolveSymmetric3x3MinEigen(
      cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz);

  const double nz_clamped = std::clamp(eig.nz, -1.0, 1.0);
  fit.theta_deg = static_cast<float>(std::acos(nz_clamped) * kRadToDeg);
  fit.sigma_z_m = static_cast<float>(std::sqrt(std::max(0.0, eig.lambda_min)));
  fit.nx = eig.nx;
  fit.ny = eig.ny;
  fit.nz = eig.nz;
  fit.mean_x = (static_cast<double>(gx) + 0.5) * kSlzGridResolutionM + mx;
  fit.mean_y = (static_cast<double>(gy) + 0.5) * kSlzGridResolutionM + my;
  fit.mean_z = origin_z_m_ + mz;
  fit.valid = std::isfinite(fit.theta_deg) && std::isfinite(fit.sigma_z_m);
  return fit;
}

rta::SlzPrior SlzBeliefGrid::EvaluateDisk(
    const core::Vec3Enu& center_enu,
    float r_m) const noexcept {
  rta::SlzPrior out{};
  out.center_enu = center_enu;
  out.p_safe = 0.0f;
  out.slz_ood_variance = 1.0f;
  out.theta_max_deg = 90.0f;
  out.sigma_z_m = 1.0f;
  out.step_max_m = 1.0f;
  out.clearance_r_m = 0.0f;
  out.h_dyn_max = 1.0f;
  out.reason_mask = REASON_CLEARANCE_INSUFF | REASON_SLOPE_EXCEEDED | REASON_OOD_UNCERTAIN;

  if (!origin_initialized_ || !center_enu.is_finite() || !std::isfinite(r_m) || r_m <= 0.0f) {
    return out;
  }

  int32_t cgx = 0;
  int32_t cgy = 0;
  if (!WorldToGlobalIdx(center_enu.x, center_enu.y, cgx, cgy)) {
    return out;
  }

  const int32_t r_cells = std::min<int32_t>(
      12, static_cast<int32_t>(std::floor(static_cast<double>(r_m) / kSlzGridResolutionM)));
  const double r_sq = static_cast<double>(r_m) * static_cast<double>(r_m) + 1e-9;
  const double center_x = (static_cast<double>(cgx) + 0.5) * kSlzGridResolutionM;
  const double center_y = (static_cast<double>(cgy) + 0.5) * kSlzGridResolutionM;

  std::array<float, kMaxDiskCells> psafe_vals{};
  std::array<float, kMaxDiskCells> theta_vals{};
  std::array<float, kMaxDiskCells> sigmaz_vals{};
  std::array<float, kMaxDiskCells> sigma2_vals{};

  struct CellMeanRecord {
    double x{0.0};
    double y{0.0};
    double z{0.0};
    float dist_m{0.0f};
  };
  std::array<CellMeanRecord, kMaxDiskCells> mean_records{};

  std::size_t total_disk_cells = 0u;
  std::size_t valid_geom_cells = 0u;
  std::size_t observed_cells = 0u;
  std::size_t mean_rec_count = 0u;

  float max_h_dyn = 0.0f;
  float min_viol_r = r_m + kSlzGridResolutionM;

  // Disk-wide shifted sums (in center cell frame) for global disk plane & parapet step_max_m.
  uint64_t disk_n_pts = 0u;
  double d_sx = 0.0, d_sy = 0.0, d_sz = 0.0;
  double d_mxx = 0.0, d_mxy = 0.0, d_mxz = 0.0, d_myy = 0.0, d_myz = 0.0, d_mzz = 0.0;

  for (int32_t dy = -r_cells; dy <= r_cells; ++dy) {
    const double oy = static_cast<double>(dy) * kSlzGridResolutionM;
    for (int32_t dx = -r_cells; dx <= r_cells; ++dx) {
      const double ox = static_cast<double>(dx) * kSlzGridResolutionM;
      const double dist_sq = ox * ox + oy * oy;
      if (dist_sq > r_sq) {
        continue;
      }
      if (total_disk_cells >= kMaxDiskCells) {
        break;
      }

      const float cell_dist_m = static_cast<float>(std::sqrt(dist_sq));
      const int32_t gx = cgx + dx;
      const int32_t gy = cgy + dy;

      if (!IsGlobalInWindow(gx, gy)) {
        psafe_vals[total_disk_cells] = 0.0f;
        theta_vals[total_disk_cells] = 90.0f;
        sigmaz_vals[total_disk_cells] = 1.0f;
        sigma2_vals[total_disk_cells] = 1.0f;
        max_h_dyn = std::max(max_h_dyn, 0.0f);
        min_viol_r = std::min(min_viol_r, cell_dist_m);
        ++total_disk_cells;
        continue;
      }

      const SlzCell& c = cells_[SlotIndex(gx, gy)];
      const float p_safe_c = (c.last_ts_ms != 0u) ? Sigmoid(c.L) : 0.0f;
      const float prec_var = 1.0f / std::max(c.lambda_prec, cfg_.lambda_min);
      const float sigma2_c = (c.last_ts_ms != 0u)
                                 ? std::max(prec_var, c.sigma2_floor)
                                 : 1.0f;
      const float h_dyn_c = c.h_dyn;
      if (c.last_ts_ms != 0u) {
        ++observed_cells;
      }

      const LocalPlaneFit fit = FitWindow3x3(gx, gy);
      psafe_vals[total_disk_cells] = p_safe_c;
      theta_vals[total_disk_cells] = fit.valid ? fit.theta_deg : 90.0f;
      sigmaz_vals[total_disk_cells] = fit.valid ? fit.sigma_z_m : 1.0f;
      sigma2_vals[total_disk_cells] = sigma2_c;
      max_h_dyn = std::max(max_h_dyn, h_dyn_c);

      if (fit.valid) {
        ++valid_geom_cells;
      }

      if (c.N > 0u) {
        const double nd = static_cast<double>(c.N);
        const double sx = static_cast<double>(c.s[0]);
        const double sy = static_cast<double>(c.s[1]);
        const double sz = static_cast<double>(c.s[2]);

        d_sx += sx + nd * ox;
        d_sy += sy + nd * oy;
        d_sz += sz;

        d_mxx += static_cast<double>(c.M[0]) + 2.0 * ox * sx + nd * ox * ox;
        d_mxy += static_cast<double>(c.M[1]) + ox * sy + oy * sx + nd * ox * oy;
        d_mxz += static_cast<double>(c.M[2]) + ox * sz;
        d_myy += static_cast<double>(c.M[3]) + 2.0 * oy * sy + nd * oy * oy;
        d_myz += static_cast<double>(c.M[4]) + oy * sz;
        d_mzz += static_cast<double>(c.M[5]);
        disk_n_pts += c.N;

        const double inv_cn = 1.0 / nd;
        mean_records[mean_rec_count++] = {
            ox + sx * inv_cn,
            oy + sy * inv_cn,
            sz * inv_cn,
            cell_dist_m,
        };
      }

      const bool cell_viol = (!fit.valid) ||
                             (fit.theta_deg > cfg_.theta_valid_deg) ||
                             (fit.sigma_z_m > cfg_.sigma_z_valid_m) ||
                             (p_safe_c < cfg_.p_safe_valid) ||
                             (h_dyn_c >= cfg_.h_dyn_valid);
      if (cell_viol) {
        min_viol_r = std::min(min_viol_r, cell_dist_m);
      }

      ++total_disk_cells;
    }
  }

  if (total_disk_cells == 0u || observed_cells == 0u) {
    return out;
  }

  // Sparse-cell conservative guard (DR-7 Review Focus #2): <12 total points or <50% valid cells.
  if (disk_n_pts < cfg_.min_points_plane_fit || valid_geom_cells * 2u < total_disk_cells) {
    out.theta_max_deg = 90.0f;
    out.sigma_z_m = 1.0f;
    out.step_max_m = 1.0f;
    out.p_safe = 0.0f;
    out.h_dyn_max = max_h_dyn;
    out.slz_ood_variance = QuantileStack(sigma2_vals, total_disk_cells, 0.90f);
    out.clearance_r_m = 0.0f;
    out.reason_mask = REASON_SLOPE_EXCEEDED | REASON_ROUGHNESS_EXCEEDED |
                      REASON_CLEARANCE_INSUFF | REASON_OOD_UNCERTAIN;
    return out;
  }

  // Global disk plane fit (in double, shifted to center_x, center_y, origin_z_m_).
  const double inv_dn = 1.0 / static_cast<double>(disk_n_pts);
  const double dmx = d_sx * inv_dn;
  const double dmy = d_sy * inv_dn;
  const double dmz = d_sz * inv_dn;

  const double dcov_xx = std::max(0.0, d_mxx * inv_dn - dmx * dmx);
  const double dcov_xy = d_mxy * inv_dn - dmx * dmy;
  const double dcov_xz = d_mxz * inv_dn - dmx * dmz;
  const double dcov_yy = std::max(0.0, d_myy * inv_dn - dmy * dmy);
  const double dcov_yz = d_myz * inv_dn - dmy * dmz;
  const double dcov_zz = std::max(0.0, d_mzz * inv_dn - dmz * dmz);

  const MinEigen3x3 disk_eig = SolveSymmetric3x3MinEigen(
      dcov_xx, dcov_xy, dcov_xz, dcov_yy, dcov_yz, dcov_zz);
  const float theta_disk_deg =
      static_cast<float>(std::acos(std::clamp(disk_eig.nz, -1.0, 1.0)) * kRadToDeg);
  const float sigmaz_disk_m = static_cast<float>(std::sqrt(std::max(0.0, disk_eig.lambda_min)));

  // Parapet / step vertical deviation across all populated disk cells.
  float step_max_m = 0.0f;
  for (std::size_t i = 0; i < mean_rec_count; ++i) {
    const double proj =
        (mean_records[i].x - dmx) * disk_eig.nx +
        (mean_records[i].y - dmy) * disk_eig.ny +
        (mean_records[i].z - dmz) * disk_eig.nz;
    const float dev = static_cast<float>(std::fabs(proj));
    if (dev > step_max_m) {
      step_max_m = dev;
    }
    if (dev > cfg_.step_valid_m) {
      min_viol_r = std::min(min_viol_r, mean_records[i].dist_m);
    }
  }

  const float q05_psafe = QuantileStack(psafe_vals, total_disk_cells, 0.05f);
  const float q95_theta = QuantileStack(theta_vals, total_disk_cells, 0.95f);
  const float q95_sigmaz = QuantileStack(sigmaz_vals, total_disk_cells, 0.95f);
  const float q90_sigma2 = QuantileStack(sigma2_vals, total_disk_cells, 0.90f);

  out.center_enu = {center_x + dmx, center_y + dmy, origin_z_m_ + dmz};
  out.p_safe = std::clamp(std::min(q05_psafe, 1.0f - max_h_dyn), 0.0f, 1.0f);
  out.h_dyn_max = std::clamp(max_h_dyn, 0.0f, 1.0f);
  out.theta_max_deg = std::max(theta_disk_deg, q95_theta);
  out.sigma_z_m = std::max(sigmaz_disk_m, q95_sigmaz);
  out.step_max_m = step_max_m;
  out.slz_ood_variance = std::clamp(q90_sigma2, 0.0f, 1.0f);
  out.clearance_r_m = (min_viol_r > r_m)
                          ? r_m
                          : std::max(0.0f, min_viol_r - 0.5f * kSlzGridResolutionM);

  uint32_t mask = REASON_NONE;
  if (out.sigma_z_m > cfg_.sigma_z_valid_m) mask |= REASON_ROUGHNESS_EXCEEDED;
  if (out.theta_max_deg > cfg_.theta_valid_deg) mask |= REASON_SLOPE_EXCEEDED;
  if (out.clearance_r_m < r_m) mask |= REASON_CLEARANCE_INSUFF;
  if (out.p_safe < cfg_.p_safe_valid) mask |= REASON_OBSTACLE_CLEARANCE;
  if (out.step_max_m > cfg_.step_valid_m) mask |= REASON_STEP_EXCEEDED;
  if (out.slz_ood_variance >= 0.08f) mask |= REASON_OOD_UNCERTAIN;
  if (out.h_dyn_max >= cfg_.h_dyn_valid) mask |= REASON_DYNAMIC_HAZARD;
  out.reason_mask = mask;
  return out;
}

}  // namespace vertirta::perception
