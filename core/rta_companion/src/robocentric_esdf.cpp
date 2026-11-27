#include "vertirta/rta/robocentric_esdf.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vertirta::rta {

namespace {

constexpr uint8_t kFlagOccupied = 1u << 0;
constexpr uint8_t kFlagInInsertQ = 1u << 1;
constexpr uint8_t kFlagInDeleteQ = 1u << 2;
constexpr uint8_t kFlagInUpdateQ = 1u << 3;
constexpr uint16_t kMaxDistMm = 16000u;  // 16 m wave propagation cap keeps < 2 ms budget
constexpr double kMaxCoordM = 1.0e7;     // +/-10,000 km local ENU bound prevents int32 overflow UB

constexpr int32_t kNbDx[6] = {+1, -1, 0, 0, 0, 0};
constexpr int32_t kNbDy[6] = {0, 0, +1, -1, 0, 0};
constexpr int32_t kNbDz[6] = {0, 0, 0, 0, +1, -1};

[[nodiscard]] inline bool is_valid_coord(double v) noexcept {
  return std::isfinite(v) && std::abs(v) <= kMaxCoordM;
}

[[nodiscard]] inline bool is_valid_enu(const core::Vec3Enu& p) noexcept {
  return is_valid_coord(p.x) && is_valid_coord(p.y) && is_valid_coord(p.z);
}

[[nodiscard]] inline int32_t floor_div2(int32_t v) noexcept {
  return (v >= 0) ? (v >> 1) : -(((-v) + 1) >> 1);
}

[[nodiscard]] inline uint16_t dist_cells_mm(int32_t dx, int32_t dy, int32_t dz) noexcept {
  const double d_mm = std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz)) * 1000.0;
  return (d_mm >= 65534.0) ? 65535u : static_cast<uint16_t>(std::lround(d_mm));
}

}  // namespace

RobocentricEsdfMap::RobocentricEsdfMap() noexcept {
  reset({0.0, 0.0, 0.0});
}

void RobocentricEsdfMap::reset(const core::Vec3Enu& center_enu) noexcept {
  const core::Vec3Enu safe_center =
      is_valid_enu(center_enu) ? center_enu : core::Vec3Enu{0.0, 0.0, 0.0};
  cfg_ = SlidingConfig{};
  cfg_.origin_ix = static_cast<int32_t>(std::floor(safe_center.x)) - (kEsdfNx / 2);
  cfg_.origin_iy = static_cast<int32_t>(std::floor(safe_center.y)) - (kEsdfNy / 2);
  cfg_.origin_iz = static_cast<int32_t>(std::floor(safe_center.z)) - (kEsdfNz / 2);
  cfg_.safe_margin_i = 0;

  occ_logodds_.fill(-20);
  for (auto& c : esdf_) {
    c = EsdfCell{};
    c.dist_mm = kMaxDistMm;
  }
  insert_head_ = insert_tail_ = 0;
  delete_head_ = delete_tail_ = 0;
  update_head_ = update_tail_ = 0;
  last_update_ns_ = 0;
}

void RobocentricEsdfMap::decode_esdf_global(
    uint32_t local_idx,
    int32_t& ex,
    int32_t& ey,
    int32_t& ez) const noexcept {
  const uint32_t lx = (local_idx >> 11u) & 63u;
  const uint32_t ly = (local_idx >> 5u) & 63u;
  const uint32_t lz = local_idx & 31u;

  const uint32_t ox = static_cast<uint32_t>(cfg_.origin_ix) & 63u;
  const uint32_t oy = static_cast<uint32_t>(cfg_.origin_iy) & 63u;
  const uint32_t oz = static_cast<uint32_t>(cfg_.origin_iz) & 31u;

  ex = cfg_.origin_ix + static_cast<int32_t>((lx - ox) & 63u);
  ey = cfg_.origin_iy + static_cast<int32_t>((ly - oy) & 63u);
  ez = cfg_.origin_iz + static_cast<int32_t>((lz - oz) & 31u);
}

void RobocentricEsdfMap::clear_esdf_cell(uint32_t idx) noexcept {
  EsdfCell& cell = esdf_[idx];
  cell.dll_packed = 0x003FFFFFFFFFFFULL;
  cell.coc_idx = EsdfCell::kNullIdx;
  cell.dist_mm = kMaxDistMm;
  cell.occ_cnt = 0;
  cell.flags = 0;
}

void RobocentricEsdfMap::slide_to(const core::Vec3Enu& new_center_enu) noexcept {
  if (!is_valid_enu(new_center_enu)) {
    return;
  }
  const int32_t new_ox = static_cast<int32_t>(std::floor(new_center_enu.x)) - (kEsdfNx / 2);
  const int32_t new_oy = static_cast<int32_t>(std::floor(new_center_enu.y)) - (kEsdfNy / 2);
  const int32_t new_oz = static_cast<int32_t>(std::floor(new_center_enu.z)) - (kEsdfNz / 2);

  const int32_t dx = new_ox - cfg_.origin_ix;
  const int32_t dy = new_oy - cfg_.origin_iy;
  const int32_t dz = new_oz - cfg_.origin_iz;
  if (dx == 0 && dy == 0 && dz == 0) {
    return;
  }

  // If shifted by more than half window, fast full reset.
  if (std::abs(dx) >= kEsdfNx || std::abs(dy) >= kEsdfNy || std::abs(dz) >= kEsdfNz) {
    reset(new_center_enu);
    return;
  }

  // Zero-copy ring buffer wrap: clear ONLY the newly exposed boundary slabs!
  auto clear_slab_x = [&](int32_t ex) {
    for (int32_t ey = new_oy; ey < new_oy + kEsdfNy; ++ey) {
      for (int32_t ez = new_oz; ez < new_oz + kEsdfNz; ++ez) {
        clear_esdf_cell(esdf_local_index(ex, ey, ez));
        for (int ox = 0; ox < 2; ++ox) {
          for (int oy = 0; oy < 2; ++oy) {
            for (int oz = 0; oz < 2; ++oz) {
              occ_logodds_[occ_local_index(2 * ex + ox, 2 * ey + oy, 2 * ez + oz)] = -20;
            }
          }
        }
      }
    }
  };
  auto clear_slab_y = [&](int32_t ey) {
    for (int32_t ex = new_ox; ex < new_ox + kEsdfNx; ++ex) {
      for (int32_t ez = new_oz; ez < new_oz + kEsdfNz; ++ez) {
        clear_esdf_cell(esdf_local_index(ex, ey, ez));
        for (int ox = 0; ox < 2; ++ox) {
          for (int oy = 0; oy < 2; ++oy) {
            for (int oz = 0; oz < 2; ++oz) {
              occ_logodds_[occ_local_index(2 * ex + ox, 2 * ey + oy, 2 * ez + oz)] = -20;
            }
          }
        }
      }
    }
  };
  auto clear_slab_z = [&](int32_t ez) {
    for (int32_t ex = new_ox; ex < new_ox + kEsdfNx; ++ex) {
      for (int32_t ey = new_oy; ey < new_oy + kEsdfNy; ++ey) {
        clear_esdf_cell(esdf_local_index(ex, ey, ez));
        for (int ox = 0; ox < 2; ++ox) {
          for (int oy = 0; oy < 2; ++oy) {
            for (int oz = 0; oz < 2; ++oz) {
              occ_logodds_[occ_local_index(2 * ex + ox, 2 * ey + oy, 2 * ez + oz)] = -20;
            }
          }
        }
      }
    }
  };

  cfg_.origin_ix = new_ox;
  cfg_.origin_iy = new_oy;
  cfg_.origin_iz = new_oz;

  if (dx > 0) {
    for (int32_t ex = new_ox + kEsdfNx - dx; ex < new_ox + kEsdfNx; ++ex) {
      clear_slab_x(ex);
    }
  } else if (dx < 0) {
    for (int32_t ex = new_ox; ex < new_ox - dx; ++ex) {
      clear_slab_x(ex);
    }
  }
  if (dy > 0) {
    for (int32_t ey = new_oy + kEsdfNy - dy; ey < new_oy + kEsdfNy; ++ey) {
      clear_slab_y(ey);
    }
  } else if (dy < 0) {
    for (int32_t ey = new_oy; ey < new_oy - dy; ++ey) {
      clear_slab_y(ey);
    }
  }
  if (dz > 0) {
    for (int32_t ez = new_oz + kEsdfNz - dz; ez < new_oz + kEsdfNz; ++ez) {
      clear_slab_z(ez);
    }
  } else if (dz < 0) {
    for (int32_t ez = new_oz; ez < new_oz - dz; ++ez) {
      clear_slab_z(ez);
    }
  }
}

void RobocentricEsdfMap::mark_occ_voxel(int32_t gx, int32_t gy, int32_t gz, bool occupied) noexcept {
  const int32_t ex = floor_div2(gx);
  const int32_t ey = floor_div2(gy);
  const int32_t ez = floor_div2(gz);
  if (!in_esdf_window(ex, ey, ez)) {
    return;
  }

  const std::size_t oidx = occ_local_index(gx, gy, gz);
  const bool was_occ = occ_logodds_[oidx] > 0;
  const int32_t next_lo = occupied ? std::min<int32_t>(100, occ_logodds_[oidx] + 45)
                                   : std::max<int32_t>(-100, occ_logodds_[oidx] - 25);
  occ_logodds_[oidx] = static_cast<int8_t>(next_lo);
  const bool now_occ = occ_logodds_[oidx] > 0;
  if (was_occ == now_occ) {
    return;
  }

  // Hysteresis counter on parent 1.0 m ESDF cell (occ_cnt in 0..8):
  // queue insert ONLY on 0->1, delete ONLY on 1->0!
  const uint32_t eidx = esdf_local_index(ex, ey, ez);
  EsdfCell& cell = esdf_[eidx];
  if (now_occ) {
    if (cell.occ_cnt == 0u) {
      cell.occ_cnt = 1u;
      cell.flags |= kFlagOccupied;
      cell.dist_mm = 0u;
      cell.coc_idx = eidx;
      if ((cell.flags & kFlagInInsertQ) == 0u && ((insert_tail_ - insert_head_) < kQueueCap)) {
        cell.flags |= kFlagInInsertQ;
        insert_q_[insert_tail_++ & (kQueueCap - 1u)] = eidx;
      }
    } else if (cell.occ_cnt < 8u) {
      ++cell.occ_cnt;
    }
  } else if (cell.occ_cnt > 0u) {
    --cell.occ_cnt;
    if (cell.occ_cnt == 0u) {
      cell.flags &= static_cast<uint8_t>(~kFlagOccupied);
      if ((cell.flags & kFlagInDeleteQ) == 0u && ((delete_tail_ - delete_head_) < kQueueCap)) {
        cell.flags |= kFlagInDeleteQ;
        delete_q_[delete_tail_++ & (kQueueCap - 1u)] = eidx;
      }
    }
  }
}

void RobocentricEsdfMap::process_queues() noexcept {
  // 1. Drain delete_q (invalidate removed obstacles & re-seed from valid neighbors)
  while (delete_head_ != delete_tail_) {
    const uint32_t idx = delete_q_[delete_head_++ & (kQueueCap - 1u)];
    EsdfCell& cell = esdf_[idx];
    cell.flags &= static_cast<uint8_t>(~kFlagInDeleteQ);
    if ((cell.flags & kFlagOccupied) != 0u) {
      continue;
    }
    cell.dist_mm = kMaxDistMm;
    cell.coc_idx = EsdfCell::kNullIdx;

    int32_t ex = 0, ey = 0, ez = 0;
    decode_esdf_global(idx, ex, ey, ez);
    for (int d = 0; d < 6; ++d) {
      const int32_t nx = ex + kNbDx[d];
      const int32_t ny = ey + kNbDy[d];
      const int32_t nz = ez + kNbDz[d];
      if (!in_esdf_window(nx, ny, nz)) {
        continue;
      }
      const uint32_t nidx = esdf_local_index(nx, ny, nz);
      EsdfCell& nb = esdf_[nidx];
      if (nb.coc_idx != EsdfCell::kNullIdx && nb.coc_idx != idx &&
          (nb.flags & kFlagInUpdateQ) == 0u && ((update_tail_ - update_head_) < kQueueCap)) {
        nb.flags |= kFlagInUpdateQ;
        update_q_[update_tail_++ & (kQueueCap - 1u)] = nidx;
      }
    }
  }

  // 2. Drain insert_q into update_q
  while (insert_head_ != insert_tail_) {
    const uint32_t idx = insert_q_[insert_head_++ & (kQueueCap - 1u)];
    EsdfCell& cell = esdf_[idx];
    cell.flags &= static_cast<uint8_t>(~kFlagInInsertQ);
    if ((cell.flags & kFlagOccupied) == 0u) {
      continue;
    }
    if ((cell.flags & kFlagInUpdateQ) == 0u && ((update_tail_ - update_head_) < kQueueCap)) {
      cell.flags |= kFlagInUpdateQ;
      update_q_[update_tail_++ & (kQueueCap - 1u)] = idx;
    }
  }

  // 3. Incremental FIESTA BFS wavefront propagation (bounded iterations & radius <= 16 m)
  uint32_t ops = 0;
  constexpr uint32_t kMaxWaveOps = 65536u;
  while (update_head_ != update_tail_ && ops < kMaxWaveOps) {
    ++ops;
    const uint32_t idx = update_q_[update_head_++ & (kQueueCap - 1u)];
    EsdfCell& cell = esdf_[idx];
    cell.flags &= static_cast<uint8_t>(~kFlagInUpdateQ);

    const uint32_t coc = cell.coc_idx;
    if (coc == EsdfCell::kNullIdx || cell.dist_mm >= kMaxDistMm) {
      continue;
    }

    int32_t ex = 0, ey = 0, ez = 0;
    int32_t cx = 0, cy = 0, cz = 0;
    decode_esdf_global(idx, ex, ey, ez);
    decode_esdf_global(coc, cx, cy, cz);

    for (int d = 0; d < 6; ++d) {
      const int32_t nx = ex + kNbDx[d];
      const int32_t ny = ey + kNbDy[d];
      const int32_t nz = ez + kNbDz[d];
      if (!in_esdf_window(nx, ny, nz)) {
        continue;
      }
      const uint32_t nidx = esdf_local_index(nx, ny, nz);
      EsdfCell& nb = esdf_[nidx];
      if ((nb.flags & kFlagOccupied) != 0u) {
        continue;
      }
      const uint16_t cand_mm = dist_cells_mm(nx - cx, ny - cy, nz - cz);
      if (cand_mm < nb.dist_mm && cand_mm < kMaxDistMm) {
        nb.dist_mm = cand_mm;
        nb.coc_idx = coc;
        if ((nb.flags & kFlagInUpdateQ) == 0u && ((update_tail_ - update_head_) < kQueueCap)) {
          nb.flags |= kFlagInUpdateQ;
          update_q_[update_tail_++ & (kQueueCap - 1u)] = nidx;
        }
      }
    }
  }
}

void RobocentricEsdfMap::insert_point_cloud(
    const core::Vec3Enu& sensor_enu,
    std::span<const core::Vec3Enu> points_enu,
    uint64_t stamp_ns) noexcept {
  (void)sensor_enu;
  for (const core::Vec3Enu& pt : points_enu) {
    if (!is_valid_enu(pt)) {
      continue;
    }
    const int32_t gx = static_cast<int32_t>(std::floor(pt.x / kOccResM));
    const int32_t gy = static_cast<int32_t>(std::floor(pt.y / kOccResM));
    const int32_t gz = static_cast<int32_t>(std::floor(pt.z / kOccResM));
    mark_occ_voxel(gx, gy, gz, true);
  }
  process_queues();
  last_update_ns_ = stamp_ns;
}

bool RobocentricEsdfMap::query_dist_grad(
    const core::Vec3Enu& pos_enu,
    double& dist_m_out,
    core::Vec3Enu& grad_unit_out) const noexcept {
  if (!is_valid_enu(pos_enu)) {
    return false;
  }

  // Cell centers at (ex + 0.5, ey + 0.5, ez + 0.5)
  const double ux = pos_enu.x - 0.5;
  const double uy = pos_enu.y - 0.5;
  const double uz = pos_enu.z - 0.5;
  const int32_t ex = static_cast<int32_t>(std::floor(ux));
  const int32_t ey = static_cast<int32_t>(std::floor(uy));
  const int32_t ez = static_cast<int32_t>(std::floor(uz));

  if (!in_esdf_window(ex, ey, ez) || !in_esdf_window(ex + 1, ey + 1, ez + 1)) {
    return false;
  }

  const double fx = std::clamp(ux - static_cast<double>(ex), 0.0, 1.0);
  const double fy = std::clamp(uy - static_cast<double>(ey), 0.0, 1.0);
  const double fz = std::clamp(uz - static_cast<double>(ez), 0.0, 1.0);

  double c[2][2][2]{};
  for (int ix = 0; ix < 2; ++ix) {
    for (int iy = 0; iy < 2; ++iy) {
      for (int iz = 0; iz < 2; ++iz) {
        const uint32_t idx = esdf_local_index(ex + ix, ey + iy, ez + iz);
        c[ix][iy][iz] = static_cast<double>(esdf_[idx].dist_mm) * 0.001;
      }
    }
  }

  const double ox = 1.0 - fx, oy = 1.0 - fy, oz = 1.0 - fz;
  const double c00 = c[0][0][0] * oz + c[0][0][1] * fz;
  const double c01 = c[0][1][0] * oz + c[0][1][1] * fz;
  const double c10 = c[1][0][0] * oz + c[1][0][1] * fz;
  const double c11 = c[1][1][0] * oz + c[1][1][1] * fz;

  const double c0 = c00 * oy + c01 * fy;
  const double c1 = c10 * oy + c11 * fy;
  dist_m_out = c0 * ox + c1 * fx;

  const double gx = c1 - c0;
  const double gy = (c01 - c00) * ox + (c11 - c10) * fx;
  const double gz =
      ((c[0][0][1] - c[0][0][0]) * oy + (c[0][1][1] - c[0][1][0]) * fy) * ox +
      ((c[1][0][1] - c[1][0][0]) * oy + (c[1][1][1] - c[1][1][0]) * fy) * fx;

  const double gnorm = std::sqrt(gx * gx + gy * gy + gz * gz);
  if (gnorm > 1e-6) {
    grad_unit_out = {gx / gnorm, gy / gnorm, gz / gnorm};
  } else {
    grad_unit_out = {0.0, 0.0, 1.0};
  }
  return true;
}

std::size_t RobocentricEsdfMap::extract_nearest_hazards(
    const core::Vec3Enu& pos_enu,
    std::span<HazardPoint> out) const noexcept {
  if (out.empty() || !is_valid_enu(pos_enu)) {
    return 0;
  }
  double dist = 0.0;
  core::Vec3Enu grad{};
  if (!query_dist_grad(pos_enu, dist, grad) || dist >= static_cast<double>(kMaxDistMm) * 0.001 - 0.1) {
    return 0;
  }
  // Surface obstacle point in ENU: pos - dist * grad_unit
  const core::Vec3Enu obs_enu = pos_enu - grad * dist;
  out[0] = HazardPoint{
      .pos_ned = core::enu_to_ned(obs_enu),
      .vel_ned = {},
      .accel_ned = {},
      .radius_h_m = 0.0,
      .radius_z_m = 0.0,
      .gamma = 1.2,
      .kind = 0u,
  };
  return 1u;
}

}  // namespace vertirta::rta
