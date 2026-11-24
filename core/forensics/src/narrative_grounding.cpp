// SPDX-License-Identifier: Apache-2.0
#include "vertirta/forensics/narrative_grounding.hpp"

#include <cctype>
#include <set>
#include <unordered_set>

namespace vertirta::forensics {

namespace {

const std::unordered_set<std::string>& allowlisted_slots() {
  static const std::unordered_set<std::string> kSlots = {
      "EVENT_SEQ",
      "T_SIM_S",
      "MIN_LTFJ_MARGIN_M",
      "MIN_LTBX_MARGIN_M",
      "SOC_PCT",
      "V_TERM_V",
      "AGL_M",
      "PEAK_CHI2",
      "ROTOR_MASK_HEX",
      "SNR_STRATUM",
      "CONFORMAL_SET_SIZE",
      "ERCS_SCORE",
      "CICTT_CODE",
      "OCCURRENCE_PHASE",
      "FLIGHT_REGIME",
  };
  return kSlots;
}

bool is_ident_char(char c) noexcept {
  const unsigned char uc = static_cast<unsigned char>(c);
  return (std::isalnum(uc) != 0) || c == '_';
}

bool is_digit_char(char c) noexcept {
  return std::isdigit(static_cast<unsigned char>(c)) != 0;
}

bool is_email_char(char c) noexcept {
  const unsigned char uc = static_cast<unsigned char>(c);
  return std::isalnum(uc) != 0 || c == '.' || c == '_' || c == '%' || c == '+' || c == '-';
}

// Extracts standalone numeric tokens (e.g. "1200", "42.5"), skipping digits attached to
// identifiers (e.g. "X8-240", "S2_Hover", "I5", "0xFE").
std::vector<std::string> extract_numeric_literals(const std::string& text) {
  std::vector<std::string> nums;
  const std::size_t n = text.size();
  std::size_t i = 0U;
  while (i < n) {
    if (!is_digit_char(text[i])) {
      ++i;
      continue;
    }
    const std::size_t start = i;
    while (i < n && is_digit_char(text[i])) {
      ++i;
    }
    if (i + 1U < n && text[i] == '.' && is_digit_char(text[i + 1U])) {
      ++i;
      while (i < n && is_digit_char(text[i])) {
        ++i;
      }
    }
    const bool left_ident = (start > 0U) && is_ident_char(text[start - 1U]);
    const bool right_ident = (i < n) && is_ident_char(text[i]);
    if (!left_ident && !right_ident) {
      nums.push_back(text.substr(start, i - start));
    }
  }
  return nums;
}

// Extracts ANY digit run in template prose outside {{SLOT}} placeholders (strict zero-digit GBNF).
std::vector<std::string> extract_any_digit_runs(const std::string& prose) {
  std::vector<std::string> runs;
  const std::size_t n = prose.size();
  std::size_t i = 0U;
  while (i < n) {
    if (!is_digit_char(prose[i])) {
      ++i;
      continue;
    }
    const std::size_t start = i;
    while (i < n && is_digit_char(prose[i])) {
      ++i;
    }
    if (i + 1U < n && prose[i] == '.' && is_digit_char(prose[i + 1U])) {
      ++i;
      while (i < n && is_digit_char(prose[i])) {
        ++i;
      }
    }
    runs.push_back(prose.substr(start, i - start));
  }
  return runs;
}

bool is_capitalized_name_word(const std::string& w) {
  if (w.size() < 2U) {
    return false;
  }
  if (std::isupper(static_cast<unsigned char>(w[0])) == 0) {
    return false;
  }
  for (std::size_t i = 1U; i < w.size(); ++i) {
    if (std::islower(static_cast<unsigned char>(w[i])) == 0) {
      return false;
    }
  }
  return true;
}

bool is_crew_title(const std::string& w) {
  static const std::unordered_set<std::string> kTitles = {
      "Pilot", "Captain", "Kaptan", "Operator", "PIC", "FO", "Controller"};
  return kTitles.find(w) != kTitles.end();
}

bool is_registration_mark(const std::string& token) {
  // Turkish civil registration TC-XXXXX (3..6 uppercase alphanumeric)
  if (token.size() >= 6U && token.size() <= 9U && token.rfind("TC-", 0U) == 0U) {
    for (std::size_t i = 3U; i < token.size(); ++i) {
      const unsigned char uc = static_cast<unsigned char>(token[i]);
      if (std::isupper(uc) == 0 && std::isdigit(uc) == 0) {
        return false;
      }
    }
    return true;
  }
  // US N-number registration N12345[A-Z]{0,2}
  if (token.size() >= 2U && token.size() <= 7U && token[0] == 'N' && is_digit_char(token[1])) {
    std::size_t i = 1U;
    while (i < token.size() && is_digit_char(token[i])) {
      ++i;
    }
    while (i < token.size() && std::isupper(static_cast<unsigned char>(token[i])) != 0) {
      ++i;
    }
    return i == token.size();
  }
  return false;
}

bool looks_like_email(const std::string& tok) {
  const std::size_t at = tok.find('@');
  if (at == std::string::npos || at == 0U || at + 3U >= tok.size()) {
    return false;
  }
  const std::size_t dot = tok.find('.', at + 1U);
  if (dot == std::string::npos || dot + 2U > tok.size()) {
    return false;
  }
  for (std::size_t i = 0U; i < tok.size(); ++i) {
    if (i == at) {
      continue;
    }
    if (!is_email_char(tok[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool is_allowlisted_narrative_slot(const std::string& slot_name) noexcept {
  return allowlisted_slots().find(slot_name) != allowlisted_slots().end();
}

GroundingVerificationResult verify_numeric_grounding(
    const std::string& rendered_text,
    const GroundedTelemetrySlots& slots) {
  GroundingVerificationResult res{};
  res.rendered_text = rendered_text;

  std::set<std::string> grounded_nums;
  for (const auto& [key, val] : slots.values) {
    (void)key;
    for (const auto& tok : extract_numeric_literals(val)) {
      grounded_nums.insert(tok);
    }
  }

  for (const auto& tok : extract_numeric_literals(rendered_text)) {
    if (grounded_nums.find(tok) == grounded_nums.end()) {
      res.hallucinated_numeric_tokens.push_back(tok);
    }
  }

  res.passed = res.hallucinated_numeric_tokens.empty();
  if (!res.passed) {
    res.error_message = "ungrounded numeric tokens detected in rendered narrative";
  }
  return res;
}

GroundingVerificationResult render_and_verify_grounded_narrative(
    const std::string& template_text,
    const GroundedTelemetrySlots& slots) {
  GroundingVerificationResult res{};
  std::string prose_only;
  prose_only.reserve(template_text.size());
  std::string rendered;
  rendered.reserve(template_text.size() + 64U);

  std::size_t i = 0U;
  while (i < template_text.size()) {
    if (i + 1U < template_text.size() && template_text[i] == '{' && template_text[i + 1U] == '{') {
      const std::size_t close = template_text.find("}}", i + 2U);
      if (close == std::string::npos) {
        res.error_message = "unclosed '{{' placeholder in narrative template";
        return res;
      }
      const std::string slot_name = template_text.substr(i + 2U, close - (i + 2U));
      if (!is_allowlisted_narrative_slot(slot_name)) {
        res.unknown_slots.push_back(slot_name);
      } else {
        const auto it = slots.values.find(slot_name);
        if (it == slots.values.end()) {
          res.unknown_slots.push_back(slot_name);
        } else {
          rendered += it->second;
        }
      }
      prose_only.push_back(' ');
      i = close + 2U;
    } else if (template_text[i] == '{' || template_text[i] == '}') {
      res.error_message = "stray brace delimiter in narrative template";
      return res;
    } else {
      prose_only.push_back(template_text[i]);
      rendered.push_back(template_text[i]);
      ++i;
    }
  }

  // Zero-digit GBNF invariant: NO raw digits may appear anywhere in prose outside {{SLOT}}.
  res.hallucinated_numeric_tokens = extract_any_digit_runs(prose_only);

  if (!res.unknown_slots.empty()) {
    res.passed = false;
    res.error_message = "unknown or missing telemetry slot(s)";
    return res;
  }
  if (!res.hallucinated_numeric_tokens.empty()) {
    res.passed = false;
    res.error_message = "raw digit(s) forbidden outside allowlisted {{SLOT}} placeholders";
    return res;
  }

  const auto post_check = verify_numeric_grounding(rendered, slots);
  if (!post_check.passed) {
    return post_check;
  }

  res.rendered_text = std::move(rendered);
  res.passed = true;
  return res;
}

std::string sanitize_article16_headline(const std::string& raw_headline) {
  // Deterministic linear-time Article 16 de-identification (emails, crew names,
  // aircraft registrations, and >= 7-digit ID/phone numbers).
  std::vector<std::string> tokens;
  std::vector<bool> is_sep;
  const std::size_t n = raw_headline.size();
  std::size_t i = 0U;
  while (i < n) {
    const char ch = raw_headline[i];
    if (std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == '(' || ch == ')' ||
        ch == ',' || ch == ';') {
      tokens.emplace_back(1U, ch);
      is_sep.push_back(true);
      ++i;
    } else {
      const std::size_t start = i;
      while (i < n) {
        const char c = raw_headline[i];
        if (std::isspace(static_cast<unsigned char>(c)) != 0 || c == '(' || c == ')' ||
            c == ',' || c == ';') {
          break;
        }
        ++i;
      }
      tokens.push_back(raw_headline.substr(start, i - start));
      is_sep.push_back(false);
    }
  }

  std::string out;
  out.reserve(raw_headline.size() + 32U);
  for (std::size_t idx = 0U; idx < tokens.size(); ++idx) {
    if (is_sep[idx]) {
      out += tokens[idx];
      continue;
    }
    const std::string& tok = tokens[idx];
    if (looks_like_email(tok)) {
      out += "[REDACTED_EMAIL]";
      continue;
    }
    if (is_registration_mark(tok)) {
      out += "[REDACTED_REG]";
      continue;
    }
    bool all_digits = (tok.size() >= 7U);
    if (all_digits) {
      for (const char c : tok) {
        if (!is_digit_char(c)) {
          all_digits = false;
          break;
        }
      }
    }
    if (all_digits) {
      out += "[REDACTED_ID]";
      continue;
    }
    if (is_crew_title(tok)) {
      out += tok;
      // Consume up to 3 following capitalized personal name tokens separated by spaces
      std::size_t look = idx + 1U;
      int redacted_names = 0;
      while (redacted_names < 3 && look + 1U < tokens.size() && is_sep[look] &&
             tokens[look] == " " && !is_sep[look + 1U] &&
             is_capitalized_name_word(tokens[look + 1U])) {
        look += 2U;
        ++redacted_names;
      }
      if (redacted_names > 0) {
        out += " [REDACTED_PII]";
        idx = look - 1U;
      }
      continue;
    }
    out += tok;
  }
  return out;
}

}  // namespace vertirta::forensics
