// SPDX-License-Identifier: Apache-2.0
#include "vertirta/forensics/ercs_scorer.hpp"

#include <algorithm>

namespace vertirta::forensics {

namespace {

// CDR (EU) 2020/2034 barrier sum (0..18) -> ERCS probability index (0..9).
// Matches DR-12 examples: sum 10 -> 5, sum 2 -> 1.
uint8_t map_barrier_sum_to_probability_index(uint8_t sum) noexcept {
  static constexpr std::array<uint8_t, 19U> kMap = {
      0U, 0U, 1U, 1U, 2U, 2U, 3U, 3U, 4U, 4U, 5U, 5U, 6U, 6U, 7U, 7U, 8U, 8U, 9U};
  const std::size_t idx = std::min<std::size_t>(sum, 18U);
  return kMap[idx];
}

std::string map_cictt_category(FaultDomain domain) {
  switch (domain) {
    case FaultDomain::ROTOR_ACTUATOR_OR_INVERTER:
    case FaultDomain::FLIGHT_CONTROL_OR_SOFTWARE:
    case FaultDomain::DATALINK_OR_USPACE_CMU:
    case FaultDomain::NAVIGATION_GNSS_VIO:
      // CICTT taxonomy lock: rotor, actuator, inverter, flight control, SW, and datalink
      // faults are coded SCF-NP (non-powerplant), NEVER SCF-PP.
      return "SCF-NP";
    case FaultDomain::PROPULSION_POWERPLANT_ONLY:
      return "SCF-PP";
    case FaultDomain::AIRBORNE_SEPARATION_LOSS:
      return "MAC";
    case FaultDomain::BATTERY_THERMAL_FIRE:
      return "F-NI";
    case FaultDomain::CRITICAL_ENERGY_RESERVE:
      return "FUEL";
    case FaultDomain::LOW_ALTITUDE_OR_OBSTACLE:
      return "LALT";
    case FaultDomain::ADVERSE_WIND_TURBULENCE:
      return "WSTRW";
  }
  return "SCF-NP";
}

uint32_t map_ata_jasc_chapter(FaultDomain domain) noexcept {
  switch (domain) {
    case FaultDomain::ROTOR_ACTUATOR_OR_INVERTER:
      return 62U;
    case FaultDomain::FLIGHT_CONTROL_OR_SOFTWARE:
      return 27U;
    case FaultDomain::DATALINK_OR_USPACE_CMU:
      return 23U;
    case FaultDomain::NAVIGATION_GNSS_VIO:
      return 34U;
    case FaultDomain::PROPULSION_POWERPLANT_ONLY:
      return 71U;
    case FaultDomain::AIRBORNE_SEPARATION_LOSS:
      return 22U;
    case FaultDomain::BATTERY_THERMAL_FIRE:
      return 26U;
    case FaultDomain::CRITICAL_ENERGY_RESERVE:
      return 24U;
    case FaultDomain::LOW_ALTITUDE_OR_OBSTACLE:
      return 22U;
    case FaultDomain::ADVERSE_WIND_TURBULENCE:
      return 31U;
  }
  return 27U;
}

// Dynamic CDR (EU) 2020/2034 severity axis (A, E, I, M, S, X) honoring canonical
// vehicle occupant capacity (1 pax X8-240 vs 4-6 pax Lift+Cruise 8+1).
char map_severity_letter(KeyRiskArea kra, uint32_t occupant_capacity) noexcept {
  if (!is_high_energy_key_risk_area(kra)) {
    return (kra == KeyRiskArea::OTHER_INJURY_LOW_ENERGY) ? 'E' : 'A';
  }
  if (kra == KeyRiskArea::MAC_HIGH_ENERGY) {
    return (occupant_capacity >= 2U) ? 'S' : 'M';
  }
  // High-energy LOC-I / CFIT / Fire over urban corridor:
  // 1-pax X8-240 maps to 'I' (single-occupant + 3rd-party coastal band),
  // 2-19 pax Lift+Cruise maps to 'M' ("small certified aircraft 2-19 potential fatalities").
  return (occupant_capacity >= 2U) ? 'M' : 'I';
}

}  // namespace

std::string to_string(Cir2026Trigger trigger) {
  switch (trigger) {
    case Cir2026Trigger::FATO_INCURSION:
      return "FATO_INCURSION";
    case Cir2026Trigger::USPACE_AIRBORNE_ENCOUNTER:
      return "USPACE_AIRBORNE_ENCOUNTER";
    case Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT:
      return "CRITICAL_ENERGY_OR_ENVELOPE_PROT";
    case Cir2026Trigger::PART_IS_INFOSEC_ANOMALY:
      return "PART_IS_INFOSEC_ANOMALY";
    case Cir2026Trigger::UAS_CMU_TECHNICAL_FAULT:
      return "UAS_CMU_TECHNICAL_FAULT";
  }
  return "CRITICAL_ENERGY_OR_ENVELOPE_PROT";
}

bool is_high_energy_key_risk_area(KeyRiskArea kra) noexcept {
  switch (kra) {
    case KeyRiskArea::LOC_I_HIGH_ENERGY:
    case KeyRiskArea::MAC_HIGH_ENERGY:
    case KeyRiskArea::CFIT_HIGH_ENERGY:
    case KeyRiskArea::FIRE_THERMAL_HIGH_ENERGY:
      return true;
    case KeyRiskArea::GROUND_DAMAGE_LOW_ENERGY:
    case KeyRiskArea::EXCURSION_LOW_ENERGY:
    case KeyRiskArea::OTHER_INJURY_LOW_ENERGY:
      return false;
  }
  return true;
}

ErcsResult score_ercs_cdr_2020_2034(const ErcsInput& input) {
  ErcsResult res{};
  res.cictt_category = map_cictt_category(input.fault_domain);
  res.ata_jasc_chapter = map_ata_jasc_chapter(input.fault_domain);

  // Per CDR (EU) 2020/2034: barriers prior to the stopping barrier did not prevent
  // progression and contribute 0. Barrier #8 (Low-Energy, weight <= 1) is strictly
  // 0 (N/A) on high-energy Key Risk Areas (MAC / LOC-I / CFIT / Fire).
  const std::size_t start_idx =
      (input.stopping_barrier_idx >= 1U && input.stopping_barrier_idx <= 8U)
          ? static_cast<std::size_t>(input.stopping_barrier_idx - 1U)
          : 0U;

  uint8_t sum = 0U;
  uint8_t effective_b8 = 0U;
  for (std::size_t i = start_idx; i < 8U; ++i) {
    uint8_t w = input.barriers[i];
    if (i == 7U) {
      w = is_high_energy_key_risk_area(input.key_risk_area) ? 0U : std::min<uint8_t>(w, 1U);
      effective_b8 = w;
    } else {
      w = std::min<uint8_t>(w, 5U);
    }
    sum = static_cast<uint8_t>(std::min<uint32_t>(18U, static_cast<uint32_t>(sum) + w));
  }

  res.effective_barrier8_score = effective_b8;
  res.effective_barrier_sum = sum;
  res.probability_index = map_barrier_sum_to_probability_index(sum);
  res.severity_letter = map_severity_letter(input.key_risk_area, input.occupant_capacity);
  res.ercs_matrix_code = std::string(1U, res.severity_letter) + std::to_string(res.probability_index);

  // Hard statutory invariant: NEVER auto-classify Accident or Serious Incident.
  res.occurrence_class_id_431 = OccurrenceClassDraft::DRAFT_INCIDENT_ONLY;
  res.auto_classified_accident_or_serious_incident = false;
  res.requires_safety_manager_review = true;
  res.reporting_status = "DRAFT_INTERNAL_PRE_SUBMISSION";
  res.immediate_escalation_queue =
      (res.probability_index <= 2U && is_high_energy_key_risk_area(input.key_risk_area));

  return res;
}

}  // namespace vertirta::forensics
