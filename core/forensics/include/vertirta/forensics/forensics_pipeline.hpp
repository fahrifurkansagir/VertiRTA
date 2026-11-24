// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "vertirta/forensics/ercs_scorer.hpp"
#include "vertirta/forensics/narrative_grounding.hpp"

namespace vertirta::forensics {

struct ForensicsPipelineRequest {
  std::filesystem::path vrta_path{};
  std::filesystem::path sidecar_path{};
  std::string sidecar_public_key_hex{};
  std::filesystem::path out_e5x_xml_path{};
  std::filesystem::path out_audit_json_path{};
  std::string file_number{"VRT-2026-0001"};
  std::string raw_headline{"Envelope protection event along Istanbul coastal corridor"};
  std::string narrative_template_en{};
  std::string narrative_template_tr{};
  std::string utc_date_time{"2026-11-24T09:00:00Z"};
  std::string local_date_time{"2026-11-24T12:00:00+03:00"};
  ErcsInput ercs_input{};
};

struct ForensicsPipelineResult {
  bool ok{false};
  std::size_t frames_verified{0U};
  std::size_t sidecar_events_joined{0U};
  std::string vrta_sha256_hex{};
  ErcsResult ercs{};
  GroundedTelemetrySlots grounded_slots{};
  std::string error_message{};
};

[[nodiscard]] ForensicsPipelineResult run_forensics_pipeline(const ForensicsPipelineRequest& req);

}  // namespace vertirta::forensics
