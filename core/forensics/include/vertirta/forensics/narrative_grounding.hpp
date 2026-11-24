// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <string>
#include <vector>

namespace vertirta::forensics {

struct GroundedTelemetrySlots {
  std::map<std::string, std::string> values{};
};

struct GroundingVerificationResult {
  bool passed{false};
  std::string rendered_text{};
  std::vector<std::string> hallucinated_numeric_tokens{};
  std::vector<std::string> unknown_slots{};
  std::string error_message{};
};

[[nodiscard]] bool is_allowlisted_narrative_slot(const std::string& slot_name) noexcept;

// Renders a zero-digit GBNF template containing allowlisted {{SLOT}} placeholders
// and enforces 0.0% numeric hallucination fail-closed.
[[nodiscard]] GroundingVerificationResult render_and_verify_grounded_narrative(
    const std::string& template_text,
    const GroundedTelemetrySlots& slots);

// Independent post-render check verifying every numeric literal in rendered_text
// originates strictly from the supplied grounded telemetry slots.
[[nodiscard]] GroundingVerificationResult verify_numeric_grounding(
    const std::string& rendered_text,
    const GroundedTelemetrySlots& slots);

// Regulation (EU) 376/2014 Article 16 de-identification for Headline (ID 601).
[[nodiscard]] std::string sanitize_article16_headline(const std::string& raw_headline);

}  // namespace vertirta::forensics
