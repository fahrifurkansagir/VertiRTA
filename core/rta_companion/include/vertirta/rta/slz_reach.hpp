#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/rta/robocentric_esdf.hpp"
#include "vertirta/rta/slz_prior_provider.hpp"
#include "vertirta/rta/wire_capsules.hpp"

namespace vertirta::rta {

// Deterministic v1.0 DEM/ESDF/CTR/Energy Reachable SLZ Ranker implementing ISlzPriorProvider.
// Enforces hard gates:
//   - LTFJ (incl. NW J2 pinch & lon < 29.1900 E) & LTBX CTR margin > 0
//   - sigma_z_m <= 0.15 m, theta_max_deg <= 8.0 deg, clearance_r_m >= 5.0 m, step_max_m <= 0.25 m
//   - Fault-degraded energy reachability E_req <= 0.85 * E_rem (400 V / 22 kWh pack)
//   - Optional ESDF & thin-wire clearance checks
class SlzReachRanker final : public ISlzPriorProvider {
 public:
  static constexpr std::size_t kMaxCandidates = 32;

  SlzReachRanker() noexcept = default;

  void clear() noexcept { count_ = 0; }
  void load_default_istanbul_priors() noexcept;

  [[nodiscard]] bool add_candidate(const SlzPrior& cand) noexcept;
  void set_flight_regime(core::FlightRegime regime) noexcept { regime_ = regime; }
  void set_wind_enu(const core::Vec3Enu& wind_enu) noexcept { wind_enu_ = wind_enu; }
  void set_esdf(const RobocentricEsdfMap* esdf) noexcept { esdf_ = esdf; }
  void set_wires(const WireCapsuleTable* wires) noexcept { wires_ = wires; }

  [[nodiscard]] SlzProviderSnapshot query_priors(
      const core::Vec3Enu& vehicle_enu,
      float soc_pct,
      uint64_t now_ns) const noexcept override;

  [[nodiscard]] uint32_t evaluate_candidate(
      const SlzPrior& cand,
      const core::Vec3Enu& vehicle_enu,
      float soc_pct,
      float& eta_s_out,
      float& soc_margin_out) const noexcept;

 private:
  std::array<SlzPrior, kMaxCandidates> candidates_{};
  std::size_t count_{0};
  core::FlightRegime regime_{core::FlightRegime::NOMINAL};
  core::Vec3Enu wind_enu_{};
  const RobocentricEsdfMap* esdf_{nullptr};
  const WireCapsuleTable* wires_{nullptr};
};

}  // namespace vertirta::rta
