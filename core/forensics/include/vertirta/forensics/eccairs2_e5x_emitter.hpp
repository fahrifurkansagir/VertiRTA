// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>

#include "vertirta/forensics/ercs_scorer.hpp"

namespace vertirta::forensics {

struct E5xOccurrenceDraft {
  std::string file_number{"VRT-2026-0001"};
  std::string reporting_entity{"VERTIRTA_RESEARCH_TESTBED"};
  std::string state_area_id_454{"TR"};
  std::string fir_uir_id_16{"LTBB"};
  std::string ats_unit_id_372{"LTFJ_TWR"};
  std::string location_name_id_440{"Kartal_Sahil_Istanbul"};
  std::string utc_date_time_id_477{"2026-11-24T09:00:00Z"};
  std::string local_date_time_id_433{"2026-11-24T12:00:00+03:00"};
  uint32_t evtol_flight_phase_id_121{10602U};  // Final approach (rotorcraft ADREP mapping)
  std::string aircraft_model{"VertiRTA-X8-240"};
  uint32_t mtow_kg{240U};
  std::string headline_raw{};
  std::string narrative_en{};
  std::string narrative_tr{};
  Cir2026Trigger cir_trigger{Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT};
  ErcsResult ercs{};
  std::string vrta_file_sha256{};
  bool sidecar_chain_verified{false};
  bool numeric_grounding_verified{false};
  uint32_t shgm_deadline_hours{72U};
};

// Emits deterministic XSD-valid ECCAIRS2 / E5X XML with strictly ascending <ATTRIBUTES> IDs,
// Article 16 PII-stripped Headline (ID 601), draft/Incident ID 431 lock, parallel EN+TR
// Entity 22 narratives, and explicit omission of N/A injury count IDs.
[[nodiscard]] std::string emit_eccairs2_e5x_xml(const E5xOccurrenceDraft& draft);

}  // namespace vertirta::forensics
