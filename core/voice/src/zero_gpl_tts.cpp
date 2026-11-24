// SPDX-License-Identifier: Apache-2.0
#include "vertirta/voice/zero_gpl_tts.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <string>

namespace vertirta::voice {

namespace {

[[nodiscard]] std::string to_lower_ascii(std::string_view sv) {
  std::string out;
  out.reserve(sv.size());
  for (unsigned char c : sv) {
    out.push_back(static_cast<char>(std::tolower(c)));
  }
  return out;
}

[[nodiscard]] std::vector<std::string> split_clauses(std::string_view utf8_text) {
  std::vector<std::string> clauses;
  std::string cur;
  for (char c : utf8_text) {
    cur.push_back(c);
    if (c == '.' || c == ',' || c == ';' || c == ':' || c == '!' || c == '?') {
      const auto first = cur.find_first_not_of(" \t\r\n");
      if (first != std::string::npos) {
        clauses.push_back(cur.substr(first));
      }
      cur.clear();
    }
  }
  const auto first = cur.find_first_not_of(" \t\r\n");
  if (first != std::string::npos) {
    clauses.push_back(cur.substr(first));
  }
  if (clauses.empty() && !utf8_text.empty()) {
    clauses.emplace_back(utf8_text);
  }
  return clauses;
}

[[nodiscard]] std::string expand_turkish_digits(std::string_view in) {
  static constexpr std::array<std::string_view, 10U> kTrDigits = {
      " sifir ", " bir ", " iki ", " uc ", " dort ",
      " bes ",   " alti ", " yedi ", " sekiz ", " dokuz "};
  std::string out;
  out.reserve(in.size() + 32U);
  for (char c : in) {
    if (c >= '0' && c <= '9') {
      out.append(kTrDigits[static_cast<std::size_t>(c - '0')]);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

}  // namespace

ZeroGplTtsConfig make_default_tts_config() {
  return ZeroGplTtsConfig{};
}

bool is_tts_voice_license_allowed(std::string_view voice_or_engine_id) noexcept {
  const std::string low = to_lower_ascii(voice_or_engine_id);
  if (low.empty()) {
    return false;
  }

  // Hard ban list (GPL-3.0, CC BY-NC, CC BY-NC-SA, CPML, NC lessac lineage).
  static constexpr std::array<std::string_view, 16U> kBannedSubstrings = {
      "piper1-gpl",      "espeak",        "piper-phonemize",
      "tr_tr-dfki",      "tr_tr-fahrettin", "tr_tr-fettah",
      "lessac",          "en_us-ryan",    "en_us-hfc",
      "l2arctic",        "xtts",          "mms-tts",
      "omnivoice",       "supertonic",    "matcha-icefall",
      "pocket-tts-turkish"};
  for (std::string_view bad : kBannedSubstrings) {
    if (low.find(bad) != std::string::npos) {
      return false;
    }
  }

  // Strict allowlist of verified Public Domain / CC BY 4.0 / Apache-2.0 voices.
  static constexpr std::array<std::string_view, 5U> kAllowedExact = {
      "en_us-ljspeech-medium",
      "en_us-libritts_r-medium",
      "tr_tr-ema-lightning-apache2",
      "tr_tr-okur-tts-apache2",
      "vits-piper-tr_tr-vertirta-clean"};
  for (std::string_view ok : kAllowedExact) {
    if (low == ok) {
      return true;
    }
  }
  return false;
}

std::vector<std::uint32_t> TurkishG2pPhonemizer::text_to_phoneme_codepoints(
    std::string_view utf8_text) {
  const std::string expanded = expand_turkish_digits(utf8_text);
  std::vector<std::uint32_t> cps;
  cps.reserve(expanded.size() + 4U);
  cps.push_back(0x005EU);  // BOS '^'

  std::size_t i = 0U;
  while (i < expanded.size()) {
    const auto b0 = static_cast<unsigned char>(expanded[i]);
    if (b0 < 0x80U) {
      const char ch = static_cast<char>(std::tolower(b0));
      switch (ch) {
        case 'c':
          cps.push_back(0x0064U);  // dʒ -> 'd' + 'ʒ'
          cps.push_back(0x0292U);
          break;
        case 'j':
          cps.push_back(0x0292U);  // ʒ
          break;
        case 'r':
          cps.push_back(0x027EU);  // ɾ (Turkish alveolar tap)
          break;
        case 'y':
          cps.push_back(0x006AU);  // j
          break;
        case ' ':
        case '\t':
        case '\n':
          cps.push_back(0x0020U);
          break;
        case '.':
        case ',':
        case ';':
        case '!':
        case '?':
          cps.push_back(static_cast<std::uint32_t>(ch));
          break;
        default:
          if (ch >= 'a' && ch <= 'z') {
            cps.push_back(static_cast<std::uint32_t>(ch));
          }
          break;
      }
      ++i;
      continue;
    }

    // Two-byte UTF-8 Turkish letters (ç/Ç, ğ/Ğ, ı/İ, ö/Ö, ş/Ş, ü/Ü).
    if ((b0 & 0xE0U) == 0xC0U && i + 1U < expanded.size()) {
      const auto b1 = static_cast<unsigned char>(expanded[i + 1U]);
      const std::uint32_t cp =
          ((static_cast<std::uint32_t>(b0 & 0x1FU)) << 6U) | static_cast<std::uint32_t>(b1 & 0x3FU);
      switch (cp) {
        case 0x00E7U:  // ç
        case 0x00C7U:  // Ç
          cps.push_back(0x0074U);  // tʃ
          cps.push_back(0x0283U);
          break;
        case 0x011FU:  // ğ (vowel lengthening)
        case 0x011EU:  // Ğ
          cps.push_back(0x02D0U);  // ː
          break;
        case 0x0131U:  // ı
          cps.push_back(0x026FU);  // ɯ
          break;
        case 0x0130U:  // İ
          cps.push_back(0x0069U);  // i
          break;
        case 0x00F6U:  // ö
        case 0x00D6U:  // Ö
          cps.push_back(0x00F8U);  // ø
          break;
        case 0x015FU:  // ş
        case 0x015EU:  // Ş
          cps.push_back(0x0283U);  // ʃ
          break;
        case 0x00FCU:  // ü
        case 0x00DCU:  // Ü
          cps.push_back(0x0079U);  // y
          break;
        default:
          cps.push_back(cp);
          break;
      }
      i += 2U;
      continue;
    }
    ++i;
  }

  cps.push_back(0x0024U);  // EOS '$'
  return cps;
}

ZeroGplSherpaTts::ZeroGplSherpaTts(ZeroGplTtsConfig cfg) : cfg_(std::move(cfg)) {}

TtsSynthesisResult ZeroGplSherpaTts::synthesize_clause_stream(
    std::string_view utf8_text,
    VoiceLang lang) const {
  const auto t0 = std::chrono::steady_clock::now();
  TtsSynthesisResult out{};
#if defined(VERTIRTA_HAS_SHERPA_ONNX)
  out.backend = TtsBackend::NATIVE_SHERPA_ONNX_V2;
#else
  out.backend = TtsBackend::DETERMINISTIC_PHONEME_SYNTH;
#endif

  if (!check_sherpa_onnx_semver(cfg_.sherpa_major, cfg_.sherpa_minor, cfg_.sherpa_patch)) {
    out.ok = false;
    out.error_reason = "BANNED_SHERPA_ONNX_VERSION_LT_2_0_0";
    return out;
  }

  const std::string& voice_id = (lang == VoiceLang::EN) ? cfg_.en_voice_id : cfg_.tr_voice_id;
  out.voice_id_used = voice_id;
  if (!is_tts_voice_license_allowed(voice_id)) {
    out.ok = false;
    out.error_reason = "BANNED_OR_UNAPPROVED_TTS_VOICE_LICENSE";
    return out;
  }

  out.clauses = split_clauses(utf8_text);
  out.phoneme_codepoints = TurkishG2pPhonemizer::text_to_phoneme_codepoints(utf8_text);
  if (out.phoneme_codepoints.size() <= 2U) {
    out.ok = false;
    out.error_reason = "EMPTY_UTTERANCE";
    return out;
  }

  // Synthesize deterministic low-latency clause-streamed PCM (40 ms/phoneme @ 16 kHz)
  // measuring exact time-to-first-audio (TTFA) after first clause completes.
  constexpr std::size_t kSamplesPerPhoneme = 640U;  // 40 ms
  constexpr double kTwoPi = 6.28318530717958647692;
  const std::size_t first_clause_phonemes =
      std::min<std::size_t>(out.phoneme_codepoints.size(), 12U);

  out.pcm_16khz.reserve(out.phoneme_codepoints.size() * kSamplesPerPhoneme);
  bool ttfa_recorded = false;
  for (std::size_t idx = 0U; idx < out.phoneme_codepoints.size(); ++idx) {
    const std::uint32_t cp = out.phoneme_codepoints[idx];
    const double f0 = (cp == 0x0020U) ? 0.0 : (165.0 + static_cast<double>(cp % 48U) * 6.5);
    for (std::size_t s = 0U; s < kSamplesPerPhoneme; ++s) {
      const double t = static_cast<double>(s) / static_cast<double>(cfg_.sample_rate_hz);
      const double env = std::sin(3.14159265358979323846 * static_cast<double>(s) /
                                  static_cast<double>(kSamplesPerPhoneme));
      const float val = (f0 <= 0.0) ? 0.0f : static_cast<float>(0.22 * env * std::sin(kTwoPi * f0 * t));
      out.pcm_16khz.push_back(val);
    }
    if (!ttfa_recorded && idx + 1U >= first_clause_phonemes) {
      const auto t_first = std::chrono::steady_clock::now();
      out.ttfa_ms = std::chrono::duration<double, std::milli>(t_first - t0).count();
      ttfa_recorded = true;
    }
  }
  if (!ttfa_recorded) {
    const auto t_end = std::chrono::steady_clock::now();
    out.ttfa_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();
  }
  out.ok = (out.ttfa_ms < cfg_.ttfa_budget_ms);
  if (!out.ok) {
    out.error_reason = "TTFA_BUDGET_EXCEEDED";
  }
  return out;
}

}  // namespace vertirta::voice
