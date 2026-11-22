#include "vertirta/rta/slz_reach.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vertirta/core/geodesy.hpp"
#include "vertirta/rta/intruder_kf.hpp"

namespace vertirta::rta {

namespace {

constexpr double kPackEnergyWh = 21772.8;  // 112S12P 403.2 V * 54 Ah (~22 kWh)
constexpr double kMaxLonNearJ2 = 29.1900;  // Strict west lock against LTFJ J2 pinch (29.192122 E)

double cruise_power_w(core::FlightRegime regime) noexcept {
  switch (regime) {
    case core::FlightRegime::NOMINAL:
      return 21400.0;  // DR-1 Section 4.3 @ 15 m/s
    case core::FlightRegime::SINGLE_LOSS:
      return 24130.0;
    case core::FlightRegime::CROSS_DUAL:
      return 26860.0;
    case core::FlightRegime::SAME_ARM_ACTIVE_SPIN:
    case core::FlightRegime::CRITICAL_DESCENT:
    default:
      return 51050.0;
  }
}

double hover_power_w(core::FlightRegime regime) noexcept {
  switch (regime) {
    case core::FlightRegime::NOMINAL:
      return 36230.0;
    case core::FlightRegime::SINGLE_LOSS:
      return 37770.0;
    case core::FlightRegime::CROSS_DUAL:
      return 39320.0;
    case core::FlightRegime::SAME_ARM_ACTIVE_SPIN:
    case core::FlightRegime::CRITICAL_DESCENT:
    default:
      return 51050.0;
  }
}

}  // namespace

void SlzReachRanker::load_default_istanbul_priors() noexcept {
  clear();

  auto make_pad = [](const char* name, double lat, double lon, double amsl,
                     float r_m, float slope_deg, float rough_m) -> SlzPrior {
    SlzPrior p{};
    std::strncpy(p.id, name, sizeof(p.id) - 1u);
    p.lat_deg = lat;
    p.lon_deg = lon;
    p.elev_amsl_m = amsl;
    p.center_enu = core::geodetic_to_local_enu(
        core::GeodeticWgs84{lat, lon, core::amsl_to_hae(amsl, lat, lon)});
    p.p_safe = 0.98f;
    p.slz_ood_variance = 0.01f;
    p.theta_max_deg = slope_deg;
    p.sigma_z_m = rough_m;
    p.step_max_m = 0.05f;
    p.clearance_r_m = r_m;
    return p;
  };

  // Canonical Kartal-Dragos-Maltepe coastal pads from config/istanbul_ltfj_airspace.yaml
  (void)add_candidate(make_pad("Kartal_Coastal_Vertiport", 40.8860, 29.1880, 12.0, 15.0f, 1.5f, 0.05f));
  (void)add_candidate(make_pad("Dragos_Sahil_Park", 40.91686, 29.13822, 6.0, 18.0f, 2.0f, 0.06f));
  (void)add_candidate(make_pad("Maltepe_Orhangazi_Park", 40.94016, 29.10790, 5.0, 25.0f, 1.0f, 0.04f));
}

bool SlzReachRanker::add_candidate(const SlzPrior& cand) noexcept {
  if (count_ >= kMaxCandidates) {
    return false;
  }
  SlzPrior c = cand;
  c.center_enu = core::geodetic_to_local_enu(
      core::GeodeticWgs84{c.lat_deg, c.lon_deg, core::amsl_to_hae(c.elev_amsl_m, c.lat_deg, c.lon_deg)});
  candidates_[count_++] = c;
  return true;
}

uint32_t SlzReachRanker::evaluate_candidate(
    const SlzPrior& cand,
    const core::Vec3Enu& vehicle_enu,
    float soc_pct,
    float& eta_s_out,
    float& soc_margin_out) const noexcept {
  uint32_t mask = 0u;

  // 1. Hard LTFJ (incl. NW J2 pinch & lon < 29.1900 E) & LTBX CTR exclusion
  const double m_ltfj = AirspaceExclusion::signed_margin_ltfj_m(cand.lat_deg, cand.lon_deg, cand.elev_amsl_m + 30.0);
  const double m_ltbx = AirspaceExclusion::signed_margin_ltbx_m(cand.lat_deg, cand.lon_deg, cand.elev_amsl_m + 30.0);
  if (m_ltfj <= 0.0 || m_ltbx <= 0.0 || cand.lon_deg >= kMaxLonNearJ2) {
    mask |= slz_reason::kCtrViolation;
  }

  // 2. Geometry & terrain surface locks (DR-3 Erratum #4: sigma_z <= 0.15 m, theta <= 8 deg, R >= 5.0 m)
  if (!std::isfinite(cand.sigma_z_m) || cand.sigma_z_m > 0.15f) {
    mask |= slz_reason::kRoughnessExceeded;
  }
  if (!std::isfinite(cand.theta_max_deg) || cand.theta_max_deg > 8.0f) {
    mask |= slz_reason::kSlopeExceeded;
  }
  if (!std::isfinite(cand.clearance_r_m) || cand.clearance_r_m < 5.0f) {
    mask |= slz_reason::kRadiusTooSmall;
  }
  if (!std::isfinite(cand.step_max_m) || cand.step_max_m > 0.25f) {
    mask |= slz_reason::kStepExceeded;
  }

  // 3. Optional ESDF & wire obstacle clearance check at candidate pad
  if (esdf_ != nullptr) {
    double dist = 0.0;
    core::Vec3Enu grad{};
    if (esdf_->query_dist_grad(cand.center_enu, dist, grad) && dist < 3.5) {
      mask |= slz_reason::kObstacleClearance;
    }
  }
  if (wires_ != nullptr && wires_->size() > 0u) {
    WireQueryResult wq{};
    if (wires_->query_nearest_single(cand.center_enu, wq) && wq.signed_dist_m < 1.5) {
      mask |= slz_reason::kObstacleClearance;
    }
  }

  // 4. Fault-degraded 3D energy reachability (E_req <= 0.85 * E_rem, 22 kWh HV pack)
  const double dx = cand.center_enu.x - vehicle_enu.x;
  const double dy = cand.center_enu.y - vehicle_enu.y;
  const double dist_h = std::hypot(dx, dy);
  const double v_air = (regime_ == core::FlightRegime::SAME_ARM_ACTIVE_SPIN) ? 4.0 : 15.0;
  const double wind_along = (dist_h > 1.0) ? ((dx * wind_enu_.x + dy * wind_enu_.y) / dist_h) : 0.0;
  const double v_gnd = std::max(2.0, v_air + wind_along);
  const double t_cruise_s = dist_h / v_gnd;
  const double t_flare_s = 30.0;

  const double e_req_wh =
      (cruise_power_w(regime_) * t_cruise_s + hover_power_w(regime_) * t_flare_s) / 3600.0;
  const double soc_usable = std::max(0.0, static_cast<double>(soc_pct) - 10.0) / 100.0;
  const double e_rem_wh = soc_usable * kPackEnergyWh;
  if (e_req_wh > 0.85 * e_rem_wh) {
    mask |= slz_reason::kEnergyUnreachable;
  }

  eta_s_out = static_cast<float>(t_cruise_s + t_flare_s);
  soc_margin_out = static_cast<float>(static_cast<double>(soc_pct) - (e_req_wh / kPackEnergyWh) * 100.0);
  return mask;
}

SlzProviderSnapshot SlzReachRanker::query_priors(
    const core::Vec3Enu& vehicle_enu,
    float soc_pct,
    uint64_t now_ns) const noexcept {
  SlzProviderSnapshot snap{};
  snap.stamp_ns = now_ns;
  snap.stale = false;

  struct ScoredCand {
    SlzPrior prior{};
    double score{0.0};
  };
  std::array<ScoredCand, kMaxCandidates> valid{};
  std::size_t n_valid = 0;

  for (std::size_t i = 0; i < count_; ++i) {
    SlzPrior p = candidates_[i];
    float eta_s = 0.0f;
    float soc_margin = 0.0f;
    const uint32_t mask = evaluate_candidate(p, vehicle_enu, soc_pct, eta_s, soc_margin);
    p.eta_s = eta_s;
    p.soc_margin_pct = soc_margin;
    p.reason_mask = mask;
    if (mask != 0u) {
      continue;
    }
    // Higher score = shorter ETA, larger clearance radius, higher safety confidence & SoC margin
    const double sc =
        100.0 * static_cast<double>(p.p_safe) +
        0.5 * static_cast<double>(p.clearance_r_m) +
        0.5 * static_cast<double>(p.soc_margin_pct) -
        0.4 * static_cast<double>(p.eta_s);
    valid[n_valid++] = ScoredCand{p, sc};
  }

  // Deterministic descending sort (tie-break lexicographically by id)
  std::sort(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(n_valid),
            [](const ScoredCand& a, const ScoredCand& b) {
              if (std::abs(a.score - b.score) > 1e-6) {
                return a.score > b.score;
              }
              return std::strcmp(a.prior.id, b.prior.id) < 0;
            });

  const std::size_t out_n = std::min<std::size_t>(n_valid, snap.priors.size());
  snap.count = static_cast<uint8_t>(out_n);
  for (std::size_t i = 0; i < out_n; ++i) {
    snap.priors[i] = valid[i].prior;
    snap.priors[i].rank = static_cast<uint8_t>(i + 1u);
  }
  return snap;
}

}  // namespace vertirta::rta
