#include "vertirta/rta/cel_2d_lut.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>

#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace vertirta::rta {

namespace {

// Exact 136 B binary header matching reachability/solve_cel_2d_lut.py (<8sII2I2f2f2f3f3f3f3f32s2f)
struct RawCel2DHeader {
  char magic[8];
  uint32_t version;
  uint32_t num_tiers;
  uint32_t dims[2];
  float lo[2];
  float hi[2];
  float scale;
  float offset;
  float tw_tiers[3];
  float v_term_thresh[3];
  float a_net_tiers[3];
  float eps_grid[3];
  uint8_t payload_sha256[32];
  float v_td_max;
  float d_z_max;
};
static_assert(sizeof(RawCel2DHeader) == 136, "RawCel2DHeader size mismatch");

}  // namespace

bool Cel2DLut::load(std::string_view path, bool verify_sha256) noexcept {
  loaded_ = false;
  std::ifstream in(std::string(path), std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    return false;
  }
  const std::streamsize sz = in.tellg();
  if (sz != static_cast<std::streamsize>(kTotalBytes)) {
    return false;
  }
  in.seekg(0, std::ios::beg);

  uint8_t hdr_buf[kHeaderBytes]{};
  if (!in.read(reinterpret_cast<char*>(hdr_buf), static_cast<std::streamsize>(kHeaderBytes))) {
    return false;
  }

  RawCel2DHeader hdr{};
  std::memcpy(&hdr, hdr_buf, sizeof(hdr));
  if (std::memcmp(hdr.magic, "VRTACEL2", 8) != 0 || hdr.version != 1u ||
      hdr.num_tiers != kNumTiers || hdr.dims[0] != kNz || hdr.dims[1] != kNv) {
    return false;
  }
  if (!std::isfinite(hdr.scale) || hdr.scale <= 0.0f || !std::isfinite(hdr.offset) ||
      !std::isfinite(hdr.lo[0]) || !std::isfinite(hdr.hi[0]) || hdr.hi[0] <= hdr.lo[0] ||
      !std::isfinite(hdr.lo[1]) || !std::isfinite(hdr.hi[1]) || hdr.hi[1] <= hdr.lo[1]) {
    return false;
  }

  if (!in.read(reinterpret_cast<char*>(data_.data()), static_cast<std::streamsize>(kPayloadBytes))) {
    return false;
  }

  if (verify_sha256) {
    uint8_t digest[32]{};
    compute_sha256(data_.data(), kPayloadBytes, digest);
    if (std::memcmp(digest, hdr.payload_sha256, 32) != 0) {
      return false;
    }
  }

  dims_ = {hdr.dims[0], hdr.dims[1]};
  lo_ = {static_cast<double>(hdr.lo[0]), static_cast<double>(hdr.lo[1])};
  hi_ = {static_cast<double>(hdr.hi[0]), static_cast<double>(hdr.hi[1])};
  inv_dx_[0] = static_cast<double>(dims_[0] - 1u) / (hi_[0] - lo_[0]);
  inv_dx_[1] = static_cast<double>(dims_[1] - 1u) / (hi_[1] - lo_[1]);
  scale_ = static_cast<double>(hdr.scale);
  offset_ = static_cast<double>(hdr.offset);
  for (std::size_t t = 0; t < kNumTiers; ++t) {
    tw_tiers_[t] = static_cast<double>(hdr.tw_tiers[t]);
    v_term_thresh_[t] = static_cast<double>(hdr.v_term_thresh[t]);
    a_net_tiers_[t] = static_cast<double>(hdr.a_net_tiers[t]);
    eps_grid_[t] = static_cast<double>(hdr.eps_grid[t]);
  }
  std::memcpy(payload_sha256_.data(), hdr.payload_sha256, 32);
  v_td_max_ = std::isfinite(hdr.v_td_max) ? static_cast<double>(hdr.v_td_max) : 2.0;
  d_z_max_ = std::isfinite(hdr.d_z_max) ? static_cast<double>(hdr.d_z_max) : 0.50;
  loaded_ = true;
  return true;
}

uint8_t Cel2DLut::select_tier(float v_term_v) const noexcept {
  if (!std::isfinite(v_term_v)) {
    return 2u;
  }
  const double vt = static_cast<double>(v_term_v);
  if (vt >= v_term_thresh_[0]) {
    return 0u;
  }
  if (vt >= v_term_thresh_[1]) {
    return 1u;
  }
  return 2u;
}

Cel2DQuery Cel2DLut::eval(float v_term_v, double z_agl, double vz_up) const noexcept {
  if (!loaded_ || !std::isfinite(v_term_v) || !std::isfinite(z_agl) || !std::isfinite(vz_up)) {
    return Cel2DQuery{};
  }
  const uint8_t tier = select_tier(v_term_v);
  const double uz = std::clamp((z_agl - lo_[0]) * inv_dx_[0], 0.0, static_cast<double>(dims_[0] - 1u) - 1e-9);
  const double uv = std::clamp((vz_up - lo_[1]) * inv_dx_[1], 0.0, static_cast<double>(dims_[1] - 1u) - 1e-9);
  const auto iz = static_cast<std::size_t>(uz);
  const auto iv = static_cast<std::size_t>(uv);
  const double fz = uz - static_cast<double>(iz);
  const double fv = uv - static_cast<double>(iv);
  const double oz = 1.0 - fz;
  const double ov = 1.0 - fv;

  const std::size_t base = static_cast<std::size_t>(tier) * kElementsPerTier;
  const double c00 = static_cast<double>(data_[base + iz * kNv + iv]) * scale_ + offset_;
  const double c01 = static_cast<double>(data_[base + iz * kNv + (iv + 1u)]) * scale_ + offset_;
  const double c10 = static_cast<double>(data_[base + (iz + 1u) * kNv + iv]) * scale_ + offset_;
  const double c11 = static_cast<double>(data_[base + (iz + 1u) * kNv + (iv + 1u)]) * scale_ + offset_;

  const double a0 = c00 * ov + c01 * fv;
  const double a1 = c10 * ov + c11 * fv;

  Cel2DQuery out{};
  out.value_m = (a0 * oz + a1 * fz) - eps_grid_[tier];
  out.dv_dz = (a1 - a0) * inv_dx_[0];
  out.dv_dvz = ((c01 - c00) * oz + (c11 - c10) * fz) * inv_dx_[1];
  out.a_net = a_net_tiers_[tier];
  out.v_td_max = v_td_max_;
  out.d_z_max = d_z_max_;
  out.tier_idx = tier;
  out.valid = true;
  return out;
}

}  // namespace vertirta::rta
