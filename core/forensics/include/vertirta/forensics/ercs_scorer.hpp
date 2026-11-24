// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace vertirta::forensics {

enum class Cir2026Trigger : uint8_t {
  FATO_INCURSION = 0U,
  USPACE_AIRBORNE_ENCOUNTER = 1U,
  CRITICAL_ENERGY_OR_ENVELOPE_PROT = 2U,
  PART_IS_INFOSEC_ANOMALY = 3U,
  UAS_CMU_TECHNICAL_FAULT = 4U,
};

enum class FaultDomain : uint8_t {
  ROTOR_ACTUATOR_OR_INVERTER = 0U,
  FLIGHT_CONTROL_OR_SOFTWARE = 1U,
  DATALINK_OR_USPACE_CMU = 2U,
  NAVIGATION_GNSS_VIO = 3U,
  PROPULSION_POWERPLANT_ONLY = 4U,
  AIRBORNE_SEPARATION_LOSS = 5U,
  BATTERY_THERMAL_FIRE = 6U,
  CRITICAL_ENERGY_RESERVE = 7U,
  LOW_ALTITUDE_OR_OBSTACLE = 8U,
  ADVERSE_WIND_TURBULENCE = 9U,
};

enum class KeyRiskArea : uint8_t {
  LOC_I_HIGH_ENERGY = 0U,
  MAC_HIGH_ENERGY = 1U,
  CFIT_HIGH_ENERGY = 2U,
  FIRE_THERMAL_HIGH_ENERGY = 3U,
  GROUND_DAMAGE_LOW_ENERGY = 4U,
  EXCURSION_LOW_ENERGY = 5U,
  OTHER_INJURY_LOW_ENERGY = 6U,
};

// Hard regulatory lock (EU 376/2014 & ICAO Annex 13): software NEVER auto-classifies
// Accident or Serious Incident (statutory SIA prerogative). Draft Incident only.
enum class OccurrenceClassDraft : uint8_t {
  DRAFT_INCIDENT_ONLY = 0U,
  DRAFT_NOT_DETERMINED = 1U,
};

struct ErcsInput {
  Cir2026Trigger trigger{Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT};
  FaultDomain fault_domain{FaultDomain::ROTOR_ACTUATOR_OR_INVERTER};
  KeyRiskArea key_risk_area{KeyRiskArea::LOC_I_HIGH_ENERGY};
  uint32_t occupant_capacity{1U};  // Canonical X8-240 1-pax lock (Lift+Cruise = 4..6)
  // 1..8 = explicit CDR (EU) 2020/2034 stopping barrier index (barriers prior to it score 0);
  // 0 = sum all provided barrier weights directly (with Barrier #8 high-energy clamp still enforced).
  uint8_t stopping_barrier_idx{0U};
  std::array<uint8_t, 8U> barriers{0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
};

struct ErcsResult {
  std::string cictt_category{"SCF-NP"};
  char severity_letter{'I'};
  uint8_t probability_index{5U};
  std::string ercs_matrix_code{"I5"};
  uint8_t effective_barrier_sum{0U};
  uint8_t effective_barrier8_score{0U};
  OccurrenceClassDraft occurrence_class_id_431{OccurrenceClassDraft::DRAFT_INCIDENT_ONLY};
  bool auto_classified_accident_or_serious_incident{false};
  bool immediate_escalation_queue{false};
  bool requires_safety_manager_review{true};
  std::string reporting_status{"DRAFT_INTERNAL_PRE_SUBMISSION"};
  uint32_t ata_jasc_chapter{27U};
};

[[nodiscard]] std::string to_string(Cir2026Trigger trigger);
[[nodiscard]] bool is_high_energy_key_risk_area(KeyRiskArea kra) noexcept;
[[nodiscard]] ErcsResult score_ercs_cdr_2020_2034(const ErcsInput& input);

}  // namespace vertirta::forensics
