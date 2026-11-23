// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/wind_acoustics/aeroacoustics.hpp"
#include "vertirta/wind_acoustics/istanbul_receptors.hpp"
#include "vertirta/wind_acoustics/mmap_vrta_wind_lut.hpp"

namespace vertirta::wind_acoustics {

inline constexpr std::size_t kMaxRouterActiveHazards = 14u;  // Strictly <= 15 RTA QP hazard slots

struct NoiseRouterConfig {
  TimeOfDayPeriod period{TimeOfDayPeriod::kDay};
  bool quiet_mode_enabled{true};
  double lambda_noise{0.85};          // Tier-1 smooth softplus noise cost weight
  double softplus_beta_dba{2.0};      // Smooth hinge sharpness [dBA]
  double lookahead_s{4.5};            // Tier-1 kinematic lookahead for lateral/vertical noise avoidance
  double max_soft_accel_delta_mps2{2.2};
  double downdraft_shed_mps{-3.0};    // Shed 100% of soft noise preference below this vertical wind
};

struct NoiseRouterOutput {
  core::Vec3Ned a_ref_soft_ned{};     // Tier-1 soft outer planner output
  rta::SafetyFilterOutput rta_out{};  // Tier-2 hard CbvfSafetyFilter output
  WindRomSample wind_sample{};
  double max_receptor_dba{0.0};
  double kartal_hospital_dba{0.0};
  double tier1_noise_cost{0.0};
  uint16_t total_hazards_fed_to_rta{0};
  bool noise_preference_shed{false};  // True when Tier-2 safety / OEI / downdraft sheds Tier-1 noise
};

// Strictly two-tier lexicographic safety-subordinated noise-aware router:
//   Tier 1 (Soft Outer Planner): smooth softplus noise cost over 64 Istanbul receptors; adds ZERO
//     constraints to CbvfSafetyFilter's QP.
//   Tier 2 (Hard RTA Safety Filter): combines Kartal Dr. Lutfi Kirdar Hospital active helipad
//     cylinder (kind=3) + LTFJ CTR western wall (kind=3) + corridor hazards inflated by VRTAWND1
//     eps_rom_p95 (total <= 14 <= 15 slots), and projects through frozen CbvfSafetyFilter::step_100hz.
class NoiseAwareRouter {
 public:
  NoiseAwareRouter(
      rta::CbvfSafetyFilter* safety_filter,
      const MmapVrtaWindLut* wind_lut = nullptr,
      NoiseRouterConfig cfg = {}) noexcept;

  [[nodiscard]] const std::array<IstanbulReceptor, kNumIstanbulReceptors>& receptors() const noexcept {
    return receptors_;
  }
  [[nodiscard]] const NoiseRouterConfig& config() const noexcept { return cfg_; }
  void set_config(const NoiseRouterConfig& cfg) noexcept { cfg_ = cfg; }

  // Zero-heap 100 Hz two-tier lexicographic step.
  [[nodiscard]] NoiseRouterOutput step_100hz(
      const core::Vec3Ned& pos_ned,
      const core::Vec3Ned& vel_ned,
      const core::Vec3Ned& a_track_ned,
      double yaw_rate_nom,
      const core::CapabilityEnvelope& env,
      const AcousticFlightState& acoustic_state,
      std::span<const rta::HazardPoint> corridor_hazards) noexcept;

 private:
  [[nodiscard]] double eval_softplus_noise_cost(
      const AcousticHemisphere180& hemi,
      const core::Vec3Enu& pos_enu,
      const core::Vec3Enu& vel_enu,
      double* out_max_dba = nullptr,
      double* out_hospital_dba = nullptr) const noexcept;

  rta::CbvfSafetyFilter* safety_filter_{nullptr};
  const MmapVrtaWindLut* wind_lut_{nullptr};
  NoiseRouterConfig cfg_{};
  AeroacousticEngine acoustics_{20.0, 70.0, 101.325};
  std::array<IstanbulReceptor, kNumIstanbulReceptors> receptors_{};
  std::array<GroundReceptor, kNumIstanbulReceptors> ground_receptors_{};
};

}  // namespace vertirta::wind_acoustics
