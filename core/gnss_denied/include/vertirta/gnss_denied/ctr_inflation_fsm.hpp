#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/gnss_denied/visual_georeloc_adapter.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/intruder_kf.hpp"

namespace vertirta::gnss_denied {

// Verified DHMI AIP Turkey LTBX ARP (40d59m26sN 029d12m58sE = 40.990556 N, 29.216111 E)
// Strictly supersedes DR-10 stale 40d59m56sN draft typo (01-DR08-13 Override #3).
inline constexpr double kVerifiedLtfjArpLatDeg = 40.898333;
inline constexpr double kVerifiedLtfjArpLonDeg = 29.309167;
inline constexpr double kVerifiedLtbxArpLatDeg = 40.990556;
inline constexpr double kVerifiedLtbxArpLonDeg = 29.216111;
inline constexpr double kCanonicalMaxLonNearJ2Deg = 29.1895;

enum class GnssDeniedNavState : uint8_t {
  kBoundedGnssDeniedCruise = 0,       // sigma_pos < 25 m (3sigma < 75 m)
  kDegradedCaution = 1,               // 25 <= sigma_pos < 65 m (12-15 m/s cap + westward bias)
  kDeterministicVisualSlzAbort = 2,   // sigma_pos >= 65 m (latched before 80 m CBVF infeasibility floor)
};

struct CtrInflationConfig {
  double b_map_m{2.0};
  double q_rw_m2_per_s{0.04};          // VIO random-walk diffusion [m^2/s]
  double kappa_drift_fraction{0.008};  // 0.8% open-loop systematic drift vs distance traveled
  double sigma_caution_m{25.0};        // 1sigma caution entry (3sigma = 75 m)
  double sigma_abort_m{65.0};          // 1sigma visual-SLZ abort latch (3sigma = 195 m < 80 m floor)
  double nominal_3sigma_target_m{25.0};
  double fsm_hysteresis_m{5.0};
  double v_max_nominal_mps{22.0};
  double v_max_caution_mps{13.5};      // Within [12.0, 15.0] m/s per DR-10
  double max_r_dot_infl_mps{6.0};      // Analytic clamp on outward wall expansion velocity
  double ctr_base_activation_m{500.0};
  double ltfj_arp_lat_deg{kVerifiedLtfjArpLatDeg};
  double ltfj_arp_lon_deg{kVerifiedLtfjArpLonDeg};
  double ltbx_arp_lat_deg{kVerifiedLtbxArpLatDeg};
  double ltbx_arp_lon_deg{kVerifiedLtbxArpLonDeg};
};

struct CtrInflationStatus {
  GnssDeniedNavState nav_state{GnssDeniedNavState::kBoundedGnssDeniedCruise};
  double sigma_pos_m{2.0};
  double three_sigma_pos_m{6.0};
  bool nominal_3sigma_met{true};
  bool visual_slz_abort_latched{false};
  bool westward_bias_active{false};
  double v_max_cap_mps{22.0};

  double ltfj_nominal_margin_m{1e6};
  double ltbx_nominal_margin_m{1e6};
  double ltfj_inflated_margin_m{1e6};
  double ltbx_inflated_margin_m{1e6};
  double ltfj_delta_r_m{0.0};
  double ltbx_delta_r_m{0.0};
  double ltfj_r_dot_infl_mps{0.0};  // Exact -dh/dt Lie-derivative outward wall expansion rate [m/s]
  double ltbx_r_dot_infl_mps{0.0};
  std::size_t emitted_ctr_hazards{0};
};

// Dynamic 3sigma + dh/dt CBVF CTR Wall Inflation & 3-State GNSS-Denied Navigation FSM.
// Couples strictly via frozen public vertirta::rta::HazardPoint (kind=3, radius_h_m + vel_ned)
// into unmodified CbvfSafetyFilter::step_100hz (ZERO edits to core/rta_companion/).
class CtrInflationFsm {
 public:
  explicit CtrInflationFsm(const CtrInflationConfig& cfg = CtrInflationConfig{}) noexcept;

  void reset(double sigma_init_m = 2.0) noexcept;

  // Propagate 2x2 horizontal covariance P_pos with diffusion q_rw*dt + systematic drift (kappa*d_ol)^2.
  void propagate_open_loop(double dt_s, double horiz_speed_mps) noexcept;

  // Apply an accepted Joseph-form covariance update from VisualGeoRelocAdapter (resets open-loop d_ol).
  void apply_reloc_update(const Cov2D& updated_cov) noexcept;

  // Build up to 2 inflated LTFJ/LTBX CTR HazardPoints (kind=3) with BOTH:
  //   - h.radius_h_m = 3*sqrt(er^T P_pos er) + b_map_m
  //   - h.vel_ned    = +r_dot_infl * er   (injects exact dh/dt = -r_dot_infl into frozen CbvfSafetyFilter!)
  [[nodiscard]] CtrInflationStatus evaluate_and_build_hazards(
      double dt_s,
      const core::Vec3Ned& own_pos_ned,
      double alt_amsl_m,
      std::span<rta::HazardPoint> out_hazards) noexcept;

  [[nodiscard]] const Cov2D& covariance() const noexcept { return cov_; }
  [[nodiscard]] const CtrInflationConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] double open_loop_distance_m() const noexcept { return open_loop_dist_m_; }

 private:
  CtrInflationConfig cfg_{};
  Cov2D cov_{};
  Cov2D cov_rate_{};  // Analytic dP/dt [m^2/s] from last propagate_open_loop step
  double open_loop_dist_m_{0.0};
  double prev_ltfj_delta_r_m_{0.0};
  double prev_ltbx_delta_r_m_{0.0};
  bool has_prev_delta_{false};
  GnssDeniedNavState nav_state_{GnssDeniedNavState::kBoundedGnssDeniedCruise};
  bool abort_latched_{false};
};

}  // namespace vertirta::gnss_denied
