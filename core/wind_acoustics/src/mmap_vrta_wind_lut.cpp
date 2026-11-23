// SPDX-License-Identifier: Apache-2.0
#include "vertirta/wind_acoustics/mmap_vrta_wind_lut.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vertirta/core/math_utils.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace vertirta::wind_acoustics {

namespace {

[[nodiscard]] bool validate_grid_desc(const VrtaWndGridDesc& g, uint64_t total_nodes) noexcept {
  if (g.nx == 0u || g.ny == 0u || g.nz == 0u) {
    return false;
  }
  if ((g.nx & 3u) != 0u || (g.ny & 3u) != 0u || (g.nz & 3u) != 0u) {
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (!core::is_finite_d(g.origin_enu[i]) || !core::is_finite_d(g.spacing_m[i]) ||
        g.spacing_m[i] <= 0.0) {
      return false;
    }
  }
  const uint64_t expected_nodes =
      static_cast<uint64_t>(g.nx) * static_cast<uint64_t>(g.ny) * static_cast<uint64_t>(g.nz);
  const uint32_t expected_bricks = (g.nx >> 2u) * (g.ny >> 2u) * (g.nz >> 2u);
  if (g.node_count != expected_nodes || g.brick_count != expected_bricks) {
    return false;
  }
  if (g.node_offset > total_nodes || g.node_count > total_nodes - g.node_offset) {
    return false;
  }
  return true;
}

[[nodiscard]] bool point_inside_grid(
    const VrtaWndGridDesc& g,
    const core::Vec3Enu& p) noexcept {
  const double x_max = g.origin_enu[0] + static_cast<double>(g.nx - 1u) * g.spacing_m[0];
  const double y_max = g.origin_enu[1] + static_cast<double>(g.ny - 1u) * g.spacing_m[1];
  const double z_max = g.origin_enu[2] + static_cast<double>(g.nz - 1u) * g.spacing_m[2];
  return (p.x >= g.origin_enu[0] && p.x <= x_max && p.y >= g.origin_enu[1] && p.y <= y_max &&
          p.z >= g.origin_enu[2] && p.z <= z_max);
}

// Deterministic zero-heap SplitMix64 uniform [0, 1) -> Irwin-Hall approximate N(0, 1)
[[nodiscard]] uint64_t splitmix64_next(uint64_t& state) noexcept {
  uint64_t z = (state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30u)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27u)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31u);
}

[[nodiscard]] double normal_sample_det(uint64_t& state) noexcept {
  // Sum of 12 uniforms minus 6.0 has exact mean 0 and variance 1.0 (zero log/trig calls)
  constexpr double kInv2Pow53 = 1.0 / 9007199254740992.0;
  double acc = -6.0;
  for (int i = 0; i < 12; ++i) {
    const uint64_t u53 = splitmix64_next(state) >> 11u;
    acc += static_cast<double>(u53) * kInv2Pow53;
  }
  return acc;
}

}  // namespace

MmapVrtaWindLut::~MmapVrtaWindLut() noexcept {
  unload();
}

MmapVrtaWindLut::MmapVrtaWindLut(MmapVrtaWindLut&& other) noexcept
    : mapped_(other.mapped_),
      mapped_size_(other.mapped_size_),
      fd_(other.fd_),
      loaded_(other.loaded_),
      header_(other.header_),
      nodes_(other.nodes_),
      modal_amp_(other.modal_amp_),
      active_uref_mps_(other.active_uref_mps_) {
  other.mapped_ = nullptr;
  other.mapped_size_ = 0;
  other.fd_ = -1;
  other.loaded_ = false;
  other.nodes_ = nullptr;
}

MmapVrtaWindLut& MmapVrtaWindLut::operator=(MmapVrtaWindLut&& other) noexcept {
  if (this != &other) {
    unload();
    mapped_ = other.mapped_;
    mapped_size_ = other.mapped_size_;
    fd_ = other.fd_;
    loaded_ = other.loaded_;
    header_ = other.header_;
    nodes_ = other.nodes_;
    modal_amp_ = other.modal_amp_;
    active_uref_mps_ = other.active_uref_mps_;
    other.mapped_ = nullptr;
    other.mapped_size_ = 0;
    other.fd_ = -1;
    other.loaded_ = false;
    other.nodes_ = nullptr;
  }
  return *this;
}

void MmapVrtaWindLut::unload() noexcept {
  if (mapped_ != nullptr && mapped_ != MAP_FAILED && mapped_size_ > 0u) {
    ::munmap(mapped_, mapped_size_);
  }
  if (fd_ >= 0) {
    ::close(fd_);
  }
  mapped_ = nullptr;
  mapped_size_ = 0u;
  fd_ = -1;
  loaded_ = false;
  nodes_ = nullptr;
  modal_amp_.fill(0.0);
}

bool MmapVrtaWindLut::load(std::string_view path, bool verify_sha256) noexcept {
  unload();
  if (path.empty() || path.size() >= 1024u) {
    return false;
  }

  char path_buf[1024];
  std::memcpy(path_buf, path.data(), path.size());
  path_buf[path.size()] = '\0';

  const int fd = ::open(path_buf, O_RDONLY);
  if (fd < 0) {
    return false;
  }

  struct stat st{};
  if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(kVrtaWndHeaderBytes)) {
    ::close(fd);
    return false;
  }

  const std::size_t file_size = static_cast<std::size_t>(st.st_size);
  void* ptr = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (ptr == MAP_FAILED) {
    ::close(fd);
    return false;
  }

  fd_ = fd;
  mapped_ = ptr;
  mapped_size_ = file_size;

  VrtaWndHeader hdr{};
  std::memcpy(&hdr, ptr, sizeof(VrtaWndHeader));

  if (std::memcmp(hdr.magic, kVrtaWndMagic.data(), kVrtaWndMagic.size()) != 0 ||
      hdr.version != kVrtaWndVersion || hdr.num_modes != kVrtaWndNumModes ||
      hdr.num_rbf_centers != kVrtaWndNumRbfCenters) {
    unload();
    return false;
  }

  // Enforce LTFJ CTR hard western longitude lock (< 29.1895 deg E) on every loaded LUT
  if (!core::is_finite_d(hdr.max_lon_deg) || hdr.max_lon_deg >= kLtfjMaxAllowedLonDeg) {
    unload();
    return false;
  }
  if (!core::is_finite_d(hdr.rbf_shape_c) || hdr.rbf_shape_c <= 0.0 ||
      !core::is_finite_d(hdr.vel_scale_mps) || hdr.vel_scale_mps <= 0.0 ||
      !core::is_finite_d(hdr.sigma_scale_mps) || hdr.sigma_scale_mps <= 0.0 ||
      !core::is_finite_d(hdr.eps_scale_mps) || hdr.eps_scale_mps <= 0.0 ||
      !core::is_finite_d(hdr.l_turb_scale_m) || hdr.l_turb_scale_m <= 0.0) {
    unload();
    return false;
  }

  if (hdr.total_nodes == 0u ||
      hdr.payload_bytes != hdr.total_nodes * static_cast<uint64_t>(kVrtaWndNodeBytes) ||
      file_size != kVrtaWndHeaderBytes + static_cast<std::size_t>(hdr.payload_bytes)) {
    unload();
    return false;
  }

  if (!validate_grid_desc(hdr.tier0, hdr.total_nodes) ||
      !validate_grid_desc(hdr.tier1, hdr.total_nodes)) {
    unload();
    return false;
  }

  const auto* payload_ptr = static_cast<const uint8_t*>(ptr) + kVrtaWndHeaderBytes;
  if (verify_sha256) {
    uint8_t digest[32]{};
    rta::compute_sha256(payload_ptr, static_cast<std::size_t>(hdr.payload_bytes), digest);
    uint8_t diff = 0u;
    for (std::size_t i = 0; i < 32u; ++i) {
      diff |= static_cast<uint8_t>(digest[i] ^ hdr.payload_sha256[i]);
    }
    if (diff != 0u) {
      unload();
      return false;
    }
  }

  header_ = hdr;
  nodes_ = reinterpret_cast<const VrtaWndNode64*>(payload_ptr);
  loaded_ = true;

  // Initialize modal amplitudes to a clean nominal coastal breeze (SW 225 deg, 10 m/s)
  update_inflow(225.0 * core::kPi / 180.0, 10.0);
  return true;
}

void MmapVrtaWindLut::update_inflow(double wind_dir_rad, double u_ref_mps) noexcept {
  if (!loaded_ || !core::is_finite_d(wind_dir_rad) || !core::is_finite_d(u_ref_mps)) {
    modal_amp_.fill(0.0);
    return;
  }
  const double u_clamped = std::clamp(u_ref_mps, 0.0, 40.0);
  active_uref_mps_ = u_clamped;
  const double ux = u_clamped * std::cos(wind_dir_rad);
  const double uy = u_clamped * std::sin(wind_dir_rad);
  const double c2 = header_.rbf_shape_c * header_.rbf_shape_c;

  for (uint32_t m = 0u; m < kVrtaWndNumModes; ++m) {
    modal_amp_[m] = header_.rbf_poly_weights[m][0] + header_.rbf_poly_weights[m][1] * ux +
                    header_.rbf_poly_weights[m][2] * uy;
  }

  for (uint32_t k = 0u; k < kVrtaWndNumRbfCenters; ++k) {
    const double dx = ux - header_.rbf_centers_ux[k];
    const double dy = uy - header_.rbf_centers_uy[k];
    const double phi = std::sqrt(dx * dx + dy * dy + c2);
    for (uint32_t m = 0u; m < kVrtaWndNumModes; ++m) {
      modal_amp_[m] += phi * header_.rbf_weights[k][m];
    }
  }
}

bool MmapVrtaWindLut::eval_grid_trilinear(
    const VrtaWndGridDesc& grid,
    const core::Vec3Enu& pos_enu,
    WindRomSample& out,
    uint8_t tier_id) const noexcept {
  const double fx_raw = (pos_enu.x - grid.origin_enu[0]) / grid.spacing_m[0];
  const double fy_raw = (pos_enu.y - grid.origin_enu[1]) / grid.spacing_m[1];
  const double fz_raw = (pos_enu.z - grid.origin_enu[2]) / grid.spacing_m[2];

  const double max_x = static_cast<double>(grid.nx - 1u);
  const double max_y = static_cast<double>(grid.ny - 1u);
  const double max_z = static_cast<double>(grid.nz - 1u);

  const bool in_bounds = (fx_raw >= 0.0 && fx_raw <= max_x && fy_raw >= 0.0 && fy_raw <= max_y &&
                          fz_raw >= 0.0 && fz_raw <= max_z);

  const double fx = std::clamp(fx_raw, 0.0, max_x);
  const double fy = std::clamp(fy_raw, 0.0, max_y);
  const double fz = std::clamp(fz_raw, 0.0, max_z);

  const uint32_t ix0 = std::min(static_cast<uint32_t>(fx), grid.nx - 2u);
  const uint32_t iy0 = std::min(static_cast<uint32_t>(fy), grid.ny - 2u);
  const uint32_t iz0 = std::min(static_cast<uint32_t>(fz), grid.nz - 2u);
  const uint32_t ix1 = ix0 + 1u;
  const uint32_t iy1 = iy0 + 1u;
  const uint32_t iz1 = iz0 + 1u;

  const double tx = fx - static_cast<double>(ix0);
  const double ty = fy - static_cast<double>(iy0);
  const double tz = fz - static_cast<double>(iz0);
  const double ox = 1.0 - tx;
  const double oy = 1.0 - ty;
  const double oz = 1.0 - tz;

  const double weights[8] = {
      ox * oy * oz,
      tx * oy * oz,
      ox * ty * oz,
      tx * ty * oz,
      ox * oy * tz,
      tx * oy * tz,
      ox * ty * tz,
      tx * ty * tz,
  };
  const uint32_t c_ix[8] = {ix0, ix1, ix0, ix1, ix0, ix1, ix0, ix1};
  const uint32_t c_iy[8] = {iy0, iy0, iy1, iy1, iy0, iy0, iy1, iy1};
  const uint32_t c_iz[8] = {iz0, iz0, iz0, iz0, iz1, iz1, iz1, iz1};

  // Pre-multiply mode_scale * modal_amp for single-pass FMA across the 8 Morton corners
  double amp_scaled[kVrtaWndNumModes];
  for (uint32_t m = 0u; m < kVrtaWndNumModes; ++m) {
    amp_scaled[m] = header_.mode_scale_mps[m] * modal_amp_[m];
  }

  double vx = 0.0;
  double vy = 0.0;
  double vz = 0.0;
  double sx = 0.0;
  double sy = 0.0;
  double sz = 0.0;
  double eps = 0.0;
  double lt = 0.0;

  const VrtaWndNode64* base = nodes_ + grid.node_offset;
  for (int c = 0; c < 8; ++c) {
    const double w = weights[c];
    const uint64_t idx = morton_swizzle_index(c_ix[c], c_iy[c], c_iz[c], grid.nx, grid.ny);
    const auto& ch = base[idx].ch;

    double nvx = static_cast<double>(ch[0]) * header_.vel_scale_mps;
    double nvy = static_cast<double>(ch[1]) * header_.vel_scale_mps;
    double nvz = static_cast<double>(ch[2]) * header_.vel_scale_mps;

    for (uint32_t m = 0u; m < kVrtaWndNumModes; ++m) {
      const double a = amp_scaled[m];
      const std::size_t off = 3u + 3u * m;
      nvx += a * static_cast<double>(ch[off]);
      nvy += a * static_cast<double>(ch[off + 1u]);
      nvz += a * static_cast<double>(ch[off + 2u]);
    }

    vx += w * nvx;
    vy += w * nvy;
    vz += w * nvz;
    sx += w * (static_cast<double>(ch[27]) * header_.sigma_scale_mps);
    sy += w * (static_cast<double>(ch[28]) * header_.sigma_scale_mps);
    sz += w * (static_cast<double>(ch[29]) * header_.sigma_scale_mps);
    eps += w * (static_cast<double>(ch[30]) * header_.eps_scale_mps);
    lt += w * (static_cast<double>(ch[31]) * header_.l_turb_scale_m);
  }

  out.mean_wind_enu = {vx, vy, vz};
  out.turb_gust_enu = {0.0, 0.0, 0.0};
  out.total_wind_enu = out.mean_wind_enu;
  out.sigma_enu = {std::max(0.0, sx), std::max(0.0, sy), std::max(0.0, sz)};
  out.sigma_rms_mps = std::sqrt(
      (out.sigma_enu.x * out.sigma_enu.x + out.sigma_enu.y * out.sigma_enu.y +
       out.sigma_enu.z * out.sigma_enu.z) /
      3.0);
  out.eps_rom_p95_mps = std::max(0.0, eps);
  out.l_turb_m = std::max(1.0, lt);
  out.tier_used = tier_id;
  out.in_bounds = in_bounds;
  out.valid = out.mean_wind_enu.is_finite() && core::is_finite_d(out.eps_rom_p95_mps);
  return out.valid;
}

bool MmapVrtaWindLut::query_enu(
    const core::Vec3Enu& pos_enu,
    WindRomSample& out,
    double dt_s,
    core::Vec3Enu* ou_state,
    uint64_t* rng_state) const noexcept {
  out = WindRomSample{};
  out.eps_rom_p95_mps = kConservativeFallbackEpsMps;
  out.sigma_rms_mps = kConservativeFallbackSigmaMps;
  out.sigma_enu = {kConservativeFallbackSigmaMps, kConservativeFallbackSigmaMps,
                   kConservativeFallbackSigmaMps};

  if (!loaded_ || nodes_ == nullptr || !pos_enu.is_finite()) {
    out.valid = false;
    return false;
  }

  const bool use_tier1 = ((header_.flags & 0x01u) != 0u) && point_inside_grid(header_.tier1, pos_enu);
  const bool ok = use_tier1 ? eval_grid_trilinear(header_.tier1, pos_enu, out, 1u)
                            : eval_grid_trilinear(header_.tier0, pos_enu, out, 0u);
  if (!ok) {
    out.eps_rom_p95_mps = kConservativeFallbackEpsMps;
    out.valid = false;
    return false;
  }

  // Optional deterministic Ornstein-Uhlenbeck turbulence gust realization
  if (dt_s > 0.0 && core::is_finite_d(dt_s) && ou_state != nullptr && rng_state != nullptr) {
    const double v_adv = std::max(std::max(active_uref_mps_, out.mean_wind_enu.norm()), 4.0);
    const double tau_s = std::clamp(out.l_turb_m / v_adv, 0.2, 20.0);
    const double phi = std::exp(-dt_s / tau_s);
    const double gain = std::sqrt(std::max(0.0, 1.0 - phi * phi));

    ou_state->x = phi * ou_state->x + gain * out.sigma_enu.x * normal_sample_det(*rng_state);
    ou_state->y = phi * ou_state->y + gain * out.sigma_enu.y * normal_sample_det(*rng_state);
    ou_state->z = phi * ou_state->z + gain * out.sigma_enu.z * normal_sample_det(*rng_state);
    out.turb_gust_enu = *ou_state;
    out.total_wind_enu = out.mean_wind_enu + out.turb_gust_enu;
  }

  return true;
}

CbvfWindBuffer compute_cbvf_wind_inflation(
    const WindRomSample& sample,
    double v_horiz_mps,
    double a_xy_max_mps2,
    double horizon_s,
    double sigma_multiplier) noexcept {
  CbvfWindBuffer buf{};
  const double eps =
      (sample.valid && core::is_finite_d(sample.eps_rom_p95_mps))
          ? std::max(0.0, sample.eps_rom_p95_mps)
          : MmapVrtaWindLut::kConservativeFallbackEpsMps;
  const double sig_h = (sample.valid && sample.sigma_enu.is_finite())
                           ? std::hypot(sample.sigma_enu.x, sample.sigma_enu.y)
                           : MmapVrtaWindLut::kConservativeFallbackSigmaMps;
  const double sig_z = (sample.valid && core::is_finite_d(sample.sigma_enu.z))
                           ? std::max(0.0, sample.sigma_enu.z)
                           : MmapVrtaWindLut::kConservativeFallbackSigmaMps;

  const double k_sig = std::max(0.0, sigma_multiplier);
  const double t_h = std::clamp(core::is_finite_d(horizon_s) ? horizon_s : 2.0, 0.25, 6.0);
  const double a_eff = std::max(core::is_finite_d(a_xy_max_mps2) ? a_xy_max_mps2 : 4.0, 1.5);
  const double v_eff = std::clamp(core::is_finite_d(v_horiz_mps) ? std::abs(v_horiz_mps) : 0.0, 0.0, 45.0);

  // Conservative velocity uncertainty d_v = k_sig * sigma + eps_rom_p95 mapped to stopping-distance
  // & drift margin: delta_r = d_v * t_h + (2*v_eff*d_v + d_v^2) / (2*a_eff) * 0.25 (strictly monotonic)
  const double dv_h = k_sig * sig_h + eps;
  const double dv_z = k_sig * sig_z + eps;

  buf.effective_disturbance_h_mps = dv_h;
  buf.effective_disturbance_z_mps = dv_z;
  buf.delta_radius_h_m = dv_h * t_h + 0.125 * (2.0 * v_eff * dv_h + dv_h * dv_h) / a_eff;
  buf.delta_radius_z_m = dv_z * (0.75 * t_h) + 0.10 * (dv_z * dv_z) / a_eff;
  return buf;
}

std::size_t inflate_hazards_with_wind_rom(
    std::span<const rta::HazardPoint> hazards_in,
    const WindRomSample& sample,
    double v_horiz_mps,
    double a_xy_max_mps2,
    std::span<rta::HazardPoint> out_hazards,
    double horizon_s) noexcept {
  const CbvfWindBuffer buf =
      compute_cbvf_wind_inflation(sample, v_horiz_mps, a_xy_max_mps2, horizon_s);
  const std::size_t count = std::min(hazards_in.size(), out_hazards.size());
  for (std::size_t i = 0; i < count; ++i) {
    out_hazards[i] = hazards_in[i];
    out_hazards[i].radius_h_m += buf.delta_radius_h_m;
    out_hazards[i].radius_z_m += buf.delta_radius_z_m;
  }
  return count;
}

}  // namespace vertirta::wind_acoustics
