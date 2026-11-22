#pragma once

#include <array>
#include <cstdint>
#include <span>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/daqp_qp.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cel_2d_lut.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace vertirta::rta {

struct HazardPoint {
  core::Vec3Ned pos_ned{};
  core::Vec3Ned vel_ned{};
  core::Vec3Ned accel_ned{};
  double radius_h_m{0.0};  // Extra horizontal inflation beyond LUT R_safe (e.g., ellipsoid / stale / wire)
  double radius_z_m{0.0};  // Extra vertical inflation beyond LUT Z_safe
  double gamma{1.2};       // Class-K decay rate [1/s]
  uint8_t kind{0};         // 0=ESDF, 1=WIRE, 2=INTRUDER, 3=CTR, 4=TERRAIN
};

struct SafetyFilterOutput {
  core::Vec3Ned a_safe_ned{};
  double yaw_rate_safe{0.0};
  double slack{0.0};
  double min_cbvf_v{1e6};
  std::array<double, 4> grad_min{};
  double cel_v_2d{1e6};
  double last_mapped_r_min{1e6};
  double last_mapped_vz_min{0.0};
  int qp_exit_flag{1};
  int qp_iters{0};
  uint16_t active_hazards{0};
  bool used_hj_fallback{false};
  bool cel_latched{false};
  bool clutter_inflated{false};
};

// 100 Hz Zero-Heap HJI-CBVF-QP Safety Filter with:
//   - StaticDaqpSolver<5, 32> (15 hazard rows + 12 actuator polytope rows + optional v_cap row <= 32)
//   - Overhead intruder s_z sign preservation & r >= 0.5 m singularity clamp (DR-2 Erratum #2)
//   - Opposing-gradient clutter inflation (+1.5 m when e_r,i . e_r,j < -0.5)
//   - Degraded v_cap_f speed-cap damping (DR-2 Erratum #3)
//   - Closed-form u*_HJ actuator-cone fallback when min_V < -0.25 m or non-optimal QP
//   - Irreversible SAME_ARM_ACTIVE_SPIN / CRITICAL_DESCENT CEL latch onto Cel2DLut
class CbvfSafetyFilter {
 public:
  CbvfSafetyFilter(const MmapCbvf4DLut* cbvf_lut, const Cel2DLut* cel_lut) noexcept;

  // Non-copyable & non-movable (embeds StaticDaqpSolver<5, 32> with internal self-pointers).
  CbvfSafetyFilter(const CbvfSafetyFilter&) = delete;
  CbvfSafetyFilter& operator=(const CbvfSafetyFilter&) = delete;
  CbvfSafetyFilter(CbvfSafetyFilter&&) = delete;
  CbvfSafetyFilter& operator=(CbvfSafetyFilter&&) = delete;
  ~CbvfSafetyFilter() = default;

  void reset() noexcept;

  [[nodiscard]] bool is_cel_latched() const noexcept { return cel_latched_; }

  [[nodiscard]] SafetyFilterOutput step_100hz(
      const core::Vec3Ned& pos_ned,
      const core::Vec3Ned& vel_ned,
      const core::Vec3Ned& a_nom_ned,
      double yaw_rate_nom,
      const core::CapabilityEnvelope& env,
      std::span<const HazardPoint> hazards) noexcept;

 private:
  [[nodiscard]] core::Vec3Ned project_to_actuator_cone(
      const core::Vec3Ned& w_dir,
      double a_xy_max,
      double a_d_min,
      double a_d_max,
      double tilt_max_rad) const noexcept;

  const MmapCbvf4DLut* cbvf_lut_{nullptr};
  const Cel2DLut* cel_lut_{nullptr};
  core::StaticDaqpSolver<5, 32> solver_{};
  bool cel_latched_{false};
};

}  // namespace vertirta::rta
