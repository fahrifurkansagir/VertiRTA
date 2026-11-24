// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <string>

#include "vertirta/forensics/forensics_pipeline.hpp"

int main(int argc, char** argv) {
  vertirta::forensics::ForensicsPipelineRequest req{};
  for (int i = 1; i + 1 < argc; ++i) {
    const std::string arg = argv[i];
    const std::string val = argv[i + 1];
    if (arg == "--vrta") {
      req.vrta_path = val;
      ++i;
    } else if (arg == "--sidecar") {
      req.sidecar_path = val;
      ++i;
    } else if (arg == "--pubkey") {
      req.sidecar_public_key_hex = val;
      ++i;
    } else if (arg == "--out-e5x") {
      req.out_e5x_xml_path = val;
      ++i;
    } else if (arg == "--out-audit") {
      req.out_audit_json_path = val;
      ++i;
    } else if (arg == "--headline") {
      req.raw_headline = val;
      ++i;
    } else if (arg == "--file-number") {
      req.file_number = val;
      ++i;
    }
  }

  if (req.vrta_path.empty() || req.sidecar_path.empty() || req.sidecar_public_key_hex.empty() ||
      req.out_e5x_xml_path.empty()) {
    std::cerr << "Usage: vertirta_forensics_cli --vrta <flight.vrta> --sidecar <events.jsonl> "
                 "--pubkey <64-hex> --out-e5x <draft.e5x.xml> [--out-audit <audit.json>]\n";
    return 2;
  }

  req.ercs_input.barriers = {1U, 2U, 2U, 1U, 2U, 1U, 1U, 0U};
  const auto res = vertirta::forensics::run_forensics_pipeline(req);
  if (!res.ok) {
    std::cerr << "vertirta_forensics_cli error: " << res.error_message << "\n";
    return 1;
  }
  std::cout << "OK frames=" << res.frames_verified
            << " events=" << res.sidecar_events_joined
            << " cictt=" << res.ercs.cictt_category
            << " ercs=" << res.ercs.ercs_matrix_code << "\n";
  return 0;
}
