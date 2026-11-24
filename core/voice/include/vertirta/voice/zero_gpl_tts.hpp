// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "vertirta/voice/voice_types.hpp"

namespace vertirta::voice {

struct ZeroGplTtsConfig {
  std::uint32_t sherpa_major{2U};
  std::uint32_t sherpa_minor{0U};
  std::uint32_t sherpa_patch{0U};
  std::string en_voice_id{"en_US-ljspeech-medium"};
  std::string tr_voice_id{"tr_TR-ema-lightning-apache2"};
  std::uint32_t sample_rate_hz{kVoiceSampleRateHz};
  std::uint32_t num_threads{2U};
  double ttfa_budget_ms{kTtsTtfaBudgetMs};
};

struct TtsSynthesisResult {
  bool ok{false};
  TtsBackend backend{TtsBackend::DETERMINISTIC_PHONEME_SYNTH};
  std::string voice_id_used{};
  std::string error_reason{};
  double ttfa_ms{0.0};
  std::vector<std::string> clauses{};
  std::vector<std::uint32_t> phoneme_codepoints{};
  std::vector<float> pcm_16khz{};
};

[[nodiscard]] ZeroGplTtsConfig make_default_tts_config();

// Hard-enforces sherpa-onnx >= 2.0.0 (PR #4055, commit 0b36476 removing GPL-3.0 espeak-ng/piper-phonemize).
[[nodiscard]] constexpr bool check_sherpa_onnx_semver(
    std::uint32_t major,
    std::uint32_t minor,
    std::uint32_t patch) noexcept {
  (void)minor;
  (void)patch;
  return major >= 2U;
}

// Strict zero-GPL / zero-NC voice & engine allowlist/denylist (Erratum #8 / DR-12).
[[nodiscard]] bool is_tts_voice_license_allowed(std::string_view voice_or_engine_id) noexcept;

// Pure Apache-2.0 C++20 Turkish rule-based G2P phonemizer (zero espeak-ng dependency).
// Converts UTF-8 Turkish aviation text + numerals into IPA/Unicode phoneme codepoints
// suitable for sherpa-onnx >= 2.0.0 `phoneme_codepoints` direct VITS injection.
class TurkishG2pPhonemizer {
 public:
  [[nodiscard]] static std::vector<std::uint32_t> text_to_phoneme_codepoints(
      std::string_view utf8_text);
};

// Clause-streaming Zero-GPL TTS synthesizer (< 250 ms TTFA).
class ZeroGplSherpaTts {
 public:
  explicit ZeroGplSherpaTts(ZeroGplTtsConfig cfg = make_default_tts_config());

  [[nodiscard]] const ZeroGplTtsConfig& config() const noexcept { return cfg_; }
  [[nodiscard]] TtsSynthesisResult synthesize_clause_stream(
      std::string_view utf8_text,
      VoiceLang lang) const;

 private:
  ZeroGplTtsConfig cfg_{};
};

}  // namespace vertirta::voice
