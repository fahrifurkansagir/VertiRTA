// SPDX-License-Identifier: Apache-2.0
#include "vertirta/uspace/thermal_turnaround.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::uspace {

namespace {

[[nodiscard]] double arrhenius_resistance_scale(double temp_c) noexcept {
  constexpr double kEaOverR = 1600.0;  // K (NMC811 high-energy cell activation factor)
  const double t_k = std::clamp(temp_c + 273.15, 253.15, 363.15);
  return std::exp(kEaOverR * (1.0 / t_k - 1.0 / 298.15));
}

[[nodiscard]] double soc_resistance_scale(double soc) noexcept {
  const double s = std::clamp(soc, 0.05, 0.99);
  if (s < 0.20) {
    return 1.0 + 1.8 * (0.20 - s);
  }
  if (s > 0.85) {
    return 1.0 + 1.2 * (s - 0.85);
  }
  return 1.0;
}

}  // namespace

double charge_thermal_derate_factor(double temp_c) noexcept {
  if (temp_c <= 40.0) {
    return 1.0;
  }
  if (temp_c <= 45.0) {
    return 1.0 - 0.55 * ((temp_c - 40.0) / 5.0);
  }
  if (temp_c <= 50.0) {
    return 0.45 - 0.33 * ((temp_c - 45.0) / 5.0);
  }
  return 0.0;  // Thermal safety inhibit > 50 C
}

TurnaroundResult simulate_multicycle_turnaround(
    const ThermalPackParams& pack,
    CoolingMode mode,
    double ambient_c,
    bool eol,
    uint32_t cycles) noexcept {
  TurnaroundResult out{};
  const uint32_t n_cycles = std::max(1U, cycles);

  const double r_eol = eol ? 1.35 : 1.0;
  const double c_eol = eol ? 0.85 : 1.0;
  const double cap_as = pack.capacity_ah_bol * c_eol * 3600.0;
  const double c_th = std::max(1000.0, pack.c_th_j_per_k);

  const double h_stand = (mode == CoolingMode::ACTIVE_LIQUID_CHILLER_15C) ? 260.0 : 45.0;
  const double t_coolant = (mode == CoolingMode::ACTIVE_LIQUID_CHILLER_15C) ? 15.0 : ambient_c;
  constexpr double kHFlight = 55.0;  // Airborne convective cooling [W/K]
  constexpr double kDt = 1.0;

  double temp_c = ambient_c;
  double soc = pack.target_depart_soc;
  double v_rc = 0.0;

  double last_charge_s = 0.0;
  double last_charge_energy_j = 0.0;
  double global_peak_charge_c = ambient_c;
  bool inhibit_hit = false;

  for (uint32_t cyc = 0U; cyc < n_cycles; ++cyc) {
    // 1) Corridor flight discharge leg (~9 min)
    const int flight_steps = static_cast<int>(std::round(pack.mission_flight_s / kDt));
    for (int s = 0; s < flight_steps; ++s) {
      const double voc = static_cast<double>(core::TheveninBattery1RC::voc_from_soc(static_cast<float>(soc)));
      const double r_scale = r_eol * arrhenius_resistance_scale(temp_c) * soc_resistance_scale(soc);
      const double r0 = pack.r0_bol_ohm * r_scale;
      const double r1 = pack.r1_bol_ohm * r_scale;
      const double tau1 = std::max(1.0, r1 * pack.c1_bol_f);

      const double v_eff = std::max(300.0, voc - v_rc);
      const double disc = std::max(0.0, v_eff * v_eff - 4.0 * r0 * pack.mission_avg_power_w);
      const double i_dis = (v_eff - std::sqrt(disc)) / std::max(2.0 * r0, 1e-6);

      v_rc += ((i_dis * r1 - v_rc) / tau1) * kDt;
      soc = std::clamp(soc - (i_dis * kDt) / cap_as, 0.05, 0.99);

      const double q_gen = i_dis * i_dis * r0 + (v_rc * v_rc) / std::max(r1, 1e-6);
      const double q_cool = kHFlight * (temp_c - ambient_c);
      temp_c += ((q_gen - q_cool) / c_th) * kDt;
    }

    // 2) Dock & pax disembark + pre-cool phase (half of ground overhead = 90 s)
    const int pre_cool_steps = static_cast<int>(std::round(pack.dock_and_board_overhead_min * 30.0 / kDt));
    for (int s = 0; s < pre_cool_steps; ++s) {
      const double r_scale = r_eol * arrhenius_resistance_scale(temp_c);
      const double tau1 = std::max(1.0, pack.r1_bol_ohm * r_scale * pack.c1_bol_f);
      v_rc *= std::exp(-kDt / tau1);
      const double q_cool = h_stand * (temp_c - t_coolant);
      temp_c += (-q_cool / c_th) * kDt;
    }

    // 3) Fast charging to target_depart_soc with 80 kW / 465.9 V clamp & 4-zone thermal derate
    double charge_s = 0.0;
    double charge_energy_j = 0.0;
    constexpr double kMaxChargeS = 2400.0;  // 40 min safety cap

    while (soc < pack.target_depart_soc && charge_s < kMaxChargeS) {
      const double voc = static_cast<double>(core::TheveninBattery1RC::voc_from_soc(static_cast<float>(soc)));
      const double r_scale = r_eol * arrhenius_resistance_scale(temp_c) * soc_resistance_scale(soc);
      const double r0 = pack.r0_bol_ohm * r_scale;
      const double r1 = pack.r1_bol_ohm * r_scale;
      const double tau1 = std::max(1.0, r1 * pack.c1_bol_f);

      const double derate = charge_thermal_derate_factor(temp_c);
      if (derate <= 0.0) {
        inhibit_hit = true;
      }

      const double p_allowed = pack.p_charger_max_w * derate;
      const double i_from_power = p_allowed / std::max(voc + v_rc, 320.0);
      const double i_from_vclamp = std::max(0.0, (pack.v_max_charge_v - voc - v_rc) / std::max(r0, 1e-6));
      const double i_chg = std::min(i_from_power, i_from_vclamp);

      v_rc += ((i_chg * r1 - v_rc) / tau1) * kDt;
      soc = std::clamp(soc + (i_chg * kDt) / cap_as, 0.05, 0.99);

      const double q_gen = i_chg * i_chg * r0 + (v_rc * v_rc) / std::max(r1, 1e-6);
      const double q_cool = h_stand * (temp_c - t_coolant);
      temp_c += ((q_gen - q_cool) / c_th) * kDt;

      if (temp_c > global_peak_charge_c) {
        global_peak_charge_c = temp_c;
      }
      charge_energy_j += i_chg * (voc + v_rc + i_chg * r0) * kDt;
      charge_s += kDt;
    }

    // 4) Pax boarding & undock (second half of ground overhead = 90 s)
    const int board_steps = pre_cool_steps;
    for (int s = 0; s < board_steps; ++s) {
      const double r_scale = r_eol * arrhenius_resistance_scale(temp_c);
      const double tau1 = std::max(1.0, pack.r1_bol_ohm * r_scale * pack.c1_bol_f);
      v_rc *= std::exp(-kDt / tau1);
      const double q_cool = h_stand * (temp_c - t_coolant);
      temp_c += (-q_cool / c_th) * kDt;
    }

    last_charge_s = charge_s;
    last_charge_energy_j = charge_energy_j;
  }

  out.peak_charge_temp_c = global_peak_charge_c;
  out.final_pack_temp_c = temp_c;
  out.charge_duration_min = last_charge_s / 60.0;
  out.total_turnaround_min = out.charge_duration_min + pack.dock_and_board_overhead_min;
  out.mean_charge_power_kw = (last_charge_s > 0.0) ? ((last_charge_energy_j / last_charge_s) / 1000.0) : 0.0;
  out.end_soc = soc;
  out.cycles_completed = n_cycles;
  out.thermal_inhibit_hit = inhibit_hit;
  return out;
}

}  // namespace vertirta::uspace
