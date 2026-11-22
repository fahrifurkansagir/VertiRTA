// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <span>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/rta/slz_prior_provider.hpp"

namespace vertirta::perception {

enum class DepthMode : uint8_t {
  RGBD_PRIMARY = 0,
  MONO_FALLBACK = 1,
};

enum class SlzDecisionState : uint8_t {
  CANDIDATE_EVAL = 0,
  APPROACH_ORBIT = 1,
  ACTIVE_INSPECT_ORBIT = 2,
  COMMIT_TOUCHDOWN = 3,
  REJECT_DIVERT_NEXT = 4,
};

// Harmonized SLZ rejection bitmask — bit-exact with both
// core/rta_companion/include/vertirta/rta/slz_prior_provider.hpp (bits 0..6)
// and ros2_ws/src/vertirta_msgs/msg/SlzPrior.msg (bits 0..10).
enum SlzReasonBits : uint32_t {
  REASON_NONE               = 0u,
  REASON_CTR_VIOLATION      = 1u << 0,   // 1    (rta::slz_reason::kCtrViolation)
  REASON_ROUGHNESS_EXCEEDED = 1u << 1,   // 2    (rta::slz_reason::kRoughnessExceeded)
  REASON_SLOPE_EXCEEDED     = 1u << 2,   // 4    (rta::slz_reason::kSlopeExceeded)
  REASON_CLEARANCE_INSUFF   = 1u << 3,   // 8    (rta::slz_reason::kRadiusTooSmall)
  REASON_RADIUS_TOO_SMALL   = 1u << 3,   // 8    alias
  REASON_ENERGY_UNREACHABLE = 1u << 4,   // 16   (rta::slz_reason::kEnergyUnreachable)
  REASON_ENERGY_INSUFF      = 1u << 4,   // 16   alias
  REASON_OBSTACLE_CLEARANCE = 1u << 5,   // 32   (rta::slz_reason::kObstacleClearance)
  REASON_LOW_SEM_SAFETY     = 1u << 5,   // 32   alias (surface / semantic hazard)
  REASON_STEP_EXCEEDED      = 1u << 6,   // 64   (rta::slz_reason::kStepExceeded)
  REASON_OOD_UNCERTAIN      = 1u << 7,   // 128
  REASON_MONO_ALT_VETO      = 1u << 8,   // 256
  REASON_ORBIT_TIMEOUT      = 1u << 9,   // 512
  REASON_DYNAMIC_HAZARD     = 1u << 10,  // 1024
};

static_assert(REASON_CTR_VIOLATION == rta::slz_reason::kCtrViolation);
static_assert(REASON_ROUGHNESS_EXCEEDED == rta::slz_reason::kRoughnessExceeded);
static_assert(REASON_SLOPE_EXCEEDED == rta::slz_reason::kSlopeExceeded);
static_assert(REASON_CLEARANCE_INSUFF == rta::slz_reason::kRadiusTooSmall);
static_assert(REASON_ENERGY_UNREACHABLE == rta::slz_reason::kEnergyUnreachable);
static_assert(REASON_OBSTACLE_CLEARANCE == rta::slz_reason::kObstacleClearance);
static_assert(REASON_STEP_EXCEEDED == rta::slz_reason::kStepExceeded);

struct EnergyGateInput {
  float soc_current_pct{50.0f};
  float soc_reserve_pct{20.0f};
  float wind_mps{0.0f};
  core::FlightRegime regime{core::FlightRegime::NOMINAL};
};

struct ActiveInspectSetpoint {
  core::Vec3Enu target_pos_enu{};
  core::Vec3Enu look_at_tlof_enu{};
  float desired_speed_mps{3.3f};
  float arc_progress_rad{0.0f};
  float elapsed_s{0.0f};
  float gimbal_pitch_rad{0.0f};  // depression angle from horizontal [rad], e.g. atan2(22, 15)
  float gimbal_yaw_rad{0.0f};    // ENU yaw pointing at TLOF center [rad]
};

struct ActiveInspectionDecision {
  uint64_t timestamp_ns{0u};
  DepthMode depth_mode{DepthMode::RGBD_PRIMARY};
  SlzDecisionState state{SlzDecisionState::CANDIDATE_EVAL};
  uint32_t reason_mask{REASON_NONE};
  rta::SlzPrior candidate{};
  std::optional<ActiveInspectSetpoint> inspect_setpoint{};
};

// Decoupled V2.1 forward-looking thin-wire / obstacle hazard contract
// (kept strictly separate from 512x512 nadir SLZ semantic segmentation).
struct ForwardWireHazard {
  core::Vec3Enu start_enu{};
  core::Vec3Enu end_enu{};
  float confidence{0.0f};
  float min_clearance_m{0.0f};
};

class IForwardHazardProvider {
 public:
  virtual ~IForwardHazardProvider() = default;
  [[nodiscard]] virtual std::span<const ForwardWireHazard> QueryForwardHazards(
      uint64_t now_ns) const noexcept = 0;
};

class NullForwardHazardProvider final : public IForwardHazardProvider {
 public:
  [[nodiscard]] std::span<const ForwardWireHazard> QueryForwardHazards(
      uint64_t /*now_ns*/) const noexcept override {
    return {};
  }
};

struct ActiveInspectionConfig {
  float r_slz_min_m{5.0f};
  float p_safe_commit{0.85f};
  float p_safe_screen{0.55f};
  float p_safe_abort{0.40f};
  float theta_commit_deg{8.0f};
  float theta_screen_deg{10.0f};
  float sigma_z_commit_m{0.15f};
  float sigma_z_screen_m{0.22f};
  float step_max_m{0.25f};
  float h_dyn_commit_max{0.15f};
  float h_dyn_abort_max{0.30f};
  float tau_commit{0.08f};
  float mono_alt_veto_m{30.0f};
  float z_inspect_agl_m{22.0f};
  float r_inspect_m{15.0f};
  float arrival_agl_tol_m{2.5f};
  float arrival_r_tol_m{3.0f};
  float arc_total_rad{2.6179939f};  // 150 deg
  float max_budget_s{12.0f};
  float approach_timeout_s{25.0f};
  float approach_speed_mps{4.5f};
  float soc_safety_mult{1.25f};
};

// Deterministic 5-state 2-phase active SLZ inspection policy (zero heap, noexcept).
// Phase 1 (APPROACH_ORBIT) absorbs cruise-to-22 m AGL descent before arming the 12 s
// active orbit timer (Phase 2: ACTIVE_INSPECT_ORBIT).
class ActiveInspectionPolicy {
 public:
  using Config = ActiveInspectionConfig;

  // Locked VertiRTA-X8-240 22 kWh, R = 0.75 m coaxial hover (~40.5 kW -> 0.614% SoC / 12 s).
  static constexpr float kBaseSoc12s = 0.614f;

  explicit ActiveInspectionPolicy(Config cfg = Config{}) noexcept;

  [[nodiscard]] float EstimateOrbitDeltaSocPct(const EnergyGateInput& e) const noexcept;

  [[nodiscard]] ActiveInspectionDecision Step(
      const rta::SlzPrior& candidate,
      const core::Vec3Enu& vehicle_enu,
      float agl_m,
      DepthMode mode,
      const EnergyGateInput& energy,
      uint64_t now_ns) noexcept;

  void Reset() noexcept;

  [[nodiscard]] SlzDecisionState state() const noexcept { return state_; }
  [[nodiscard]] const Config& config() const noexcept { return cfg_; }

 private:
  [[nodiscard]] bool SameCandidate(const rta::SlzPrior& c) const noexcept;
  void LatchCandidateIdentity(const rta::SlzPrior& c) noexcept;
  [[nodiscard]] bool IsAtOrbitRing(
      const core::Vec3Enu& vehicle_enu,
      const core::Vec3Enu& tlof_enu,
      float agl_m) const noexcept;
  [[nodiscard]] uint32_t CollectRejectReasonMask(
      const rta::SlzPrior& c,
      bool mono_hi,
      bool var_ok) const noexcept;
  [[nodiscard]] ActiveInspectionDecision Reject(
      ActiveInspectionDecision& out,
      uint32_t mask) noexcept;
  [[nodiscard]] ActiveInspectSetpoint ComputeApproachSetpoint(
      const core::Vec3Enu& vehicle_enu,
      const core::Vec3Enu& tlof_enu,
      float agl_m,
      float approach_elapsed_s) const noexcept;
  [[nodiscard]] ActiveInspectSetpoint ComputeOrbitSetpoint(
      const core::Vec3Enu& tlof_enu,
      float elapsed_s) const noexcept;

  Config cfg_{};
  SlzDecisionState state_{SlzDecisionState::CANDIDATE_EVAL};
  char active_id_[32]{};
  core::Vec3Enu active_center_{};
  bool has_active_{false};
  uint64_t approach_start_ns_{0u};
  uint64_t orbit_start_ns_{0u};
  float start_azimuth_rad_{0.0f};
};

}  // namespace vertirta::perception
