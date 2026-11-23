// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>

#include "vertirta/core/frames.hpp"
#include "vertirta/wind_acoustics/mmap_vrta_wind_lut.hpp"

namespace vertirta::wind_acoustics {

struct DifferentialWindShear {
  WindRomSample center{};
  double dw_z_dy_roll_per_s{0.0};   // Vertical wind gradient across wingspan [1/s]
  double dw_z_dx_pitch_per_s{0.0};  // Vertical wind gradient nose-to-tail [1/s]
  double du_horiz_dz_per_s{0.0};    // Vertical shear of horizontal wind [1/s]
  bool valid{false};
};

// Zero-heap differential shear probe across wingtips, nose/tail, and vertical boom span
// for coupling urban coastal wakes into 6-DOF or Gazebo Lift+Cruise / X8-240 dynamics.
[[nodiscard]] inline DifferentialWindShear query_differential_shear(
    const MmapVrtaWindLut& lut,
    const core::Vec3Enu& pos_enu,
    double wingspan_m,
    double tail_arm_m,
    double heading_enu_rad,
    double dz_probe_m = 2.0) noexcept {
  DifferentialWindShear out{};
  if (!lut.is_loaded() || !pos_enu.is_finite() || wingspan_m <= 0.1 || tail_arm_m <= 0.1 ||
      dz_probe_m <= 0.05) {
    out.center.eps_rom_p95_mps = MmapVrtaWindLut::kConservativeFallbackEpsMps;
    return out;
  }

  if (!lut.query_enu(pos_enu, out.center)) {
    return out;
  }

  const double c = std::cos(heading_enu_rad);
  const double s = std::sin(heading_enu_rad);
  // Forward unit vector in ENU: (c, s, 0); left wingtip unit vector: (-s, c, 0)
  const double half_b = 0.5 * wingspan_m;
  const core::Vec3Enu p_left{pos_enu.x - s * half_b, pos_enu.y + c * half_b, pos_enu.z};
  const core::Vec3Enu p_right{pos_enu.x + s * half_b, pos_enu.y - c * half_b, pos_enu.z};
  const core::Vec3Enu p_nose{pos_enu.x + c * tail_arm_m, pos_enu.y + s * tail_arm_m, pos_enu.z};
  const core::Vec3Enu p_tail{pos_enu.x - c * tail_arm_m, pos_enu.y - s * tail_arm_m, pos_enu.z};
  const core::Vec3Enu p_up{pos_enu.x, pos_enu.y, pos_enu.z + dz_probe_m};
  const core::Vec3Enu p_dn{pos_enu.x, pos_enu.y, pos_enu.z - dz_probe_m};

  WindRomSample sl{}, sr{}, sn{}, st{}, su{}, sd{};
  const bool ok_lr = lut.query_enu(p_left, sl) && lut.query_enu(p_right, sr);
  const bool ok_nt = lut.query_enu(p_nose, sn) && lut.query_enu(p_tail, st);
  const bool ok_ud = lut.query_enu(p_up, su) && lut.query_enu(p_dn, sd);

  if (ok_lr) {
    out.dw_z_dy_roll_per_s = (sl.mean_wind_enu.z - sr.mean_wind_enu.z) / wingspan_m;
  }
  if (ok_nt) {
    out.dw_z_dx_pitch_per_s = (sn.mean_wind_enu.z - st.mean_wind_enu.z) / (2.0 * tail_arm_m);
  }
  if (ok_ud) {
    const double uh_up = std::hypot(su.mean_wind_enu.x, su.mean_wind_enu.y);
    const double uh_dn = std::hypot(sd.mean_wind_enu.x, sd.mean_wind_enu.y);
    out.du_horiz_dz_per_s = (uh_up - uh_dn) / (2.0 * dz_probe_m);
  }
  out.valid = ok_lr && ok_nt && ok_ud;
  return out;
}

}  // namespace vertirta::wind_acoustics
