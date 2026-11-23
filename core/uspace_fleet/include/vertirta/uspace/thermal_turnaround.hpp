// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "vertirta/core/coax_envelope.hpp"

namespace vertirta::uspace {

enum class CoolingMode : uint8_t {
  ACTIVE_LIQUID_CHILLER_15C = 0,
  FORCED_AIR_ONLY = 1,
};

// Canonical 68.0 kg 112S12P high-energy pack locked to config/vehicle_x8_240.yaml
// with U-Space dispatch reserves layered above airborne reserve.
struct ThermalPackParams {
  double pack_mass_kg{68.0};
  double specific_heat_j_per_kg_k{1040.0};
  double c_th_j_per_k{70720.0};  // 68.0 * 1040.0 J/K
  double r0_bol_ohm{core::params::kBattR0};
  double r1_bol_ohm{core::params::kBattR1};
  double c1_bol_f{core::params::kBattC1};
  double capacity_ah_bol{core::params::kBattCapAh};
  double v_nom_v{403.2};
  double v_max_charge_v{465.9};
  double p_charger_max_w{80000.0};
  double dispatch_reserve_soc_pct{25.0};
  double emergency_floor_soc_pct{20.0};
  double airborne_reserve_soc_pct{10.0};
  double mission_flight_s{330.0};           // 5.5 min mean Anatolian coastal leg (DR-11 Sec 5)
  double mission_avg_power_w{39000.0};      // Canonical 36.23 kW hover class + climb/approach margin
  double target_depart_soc{0.85};
  double dock_and_board_overhead_min{3.0};  // 1.5 min dock/pre-cool + 1.5 min board/undock
};

struct TurnaroundResult {
  double peak_charge_temp_c{0.0};
  double final_pack_temp_c{0.0};
  double charge_duration_min{0.0};
  double total_turnaround_min{0.0};
  double mean_charge_power_kw{0.0};
  double end_soc{0.0};
  uint32_t cycles_completed{0U};
  bool thermal_inhibit_hit{false};
};

// 4-zone C-rate thermal derating multiplier:
//   <= 40 C: 1.00 | 40-45 C: linear 1.00->0.45 | 45-50 C: linear 0.45->0.12 | > 50 C: 0.0 (inhibit)
[[nodiscard]] double charge_thermal_derate_factor(double temp_c) noexcept;

// Simulates consecutive flight-charge cycles with 1-RC Thevenin + Arrhenius electro-thermal ODE.
[[nodiscard]] TurnaroundResult simulate_multicycle_turnaround(
    const ThermalPackParams& pack,
    CoolingMode mode,
    double ambient_c,
    bool eol,
    uint32_t cycles) noexcept;

}  // namespace vertirta::uspace
