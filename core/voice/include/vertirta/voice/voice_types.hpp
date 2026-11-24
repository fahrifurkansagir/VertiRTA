// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace vertirta::voice {

inline constexpr std::uint32_t kVoiceSampleRateHz = 16000U;
inline constexpr std::size_t kVadFrameSamples = 512U;  // 32 ms @ 16 kHz (Silero VAD v6.2.0)
inline constexpr float kDefaultConformalAlpha = 0.05f;
inline constexpr float kLoiterLivenessMinSocMarginPct = 5.0f;
inline constexpr float kDefaultMaydayAsymmetricWeightFloor = 0.15f;
inline constexpr double kTtsTtfaBudgetMs = 250.0;

// Canonical 4-stratum cockpit BVI SNR taxonomy (synced across Plan 6B & Plan 6C).
enum class SnrStratum : std::uint8_t {
  S0_CLEAN = 0U,    // > 15 dB
  S1_CRUISE = 1U,   // 8..15 dB
  S2_HOVER = 2U,    // 2..8 dB
  S3_GARBLED = 3U,  // < 2 dB
};

enum class VoiceLang : std::uint8_t {
  TR = 0U,
  EN = 1U,
  MIX = 2U,
};

enum class AsrModelTier : std::uint8_t {
  WHISPER_SMALL_Q8_0 = 0U,
  WHISPER_LARGE_V3_TURBO_Q8_0 = 1U,
};

enum class AsrBackend : std::uint8_t {
  NATIVE_WHISPER_CPP = 0U,
  DETERMINISTIC_CI_ADAPTER = 1U,
};

enum class TtsBackend : std::uint8_t {
  NATIVE_SHERPA_ONNX_V2 = 0U,
  DETERMINISTIC_PHONEME_SYNTH = 1U,
};

enum class VoiceGateAction : std::uint8_t {
  EXECUTE_SINGLE = 0U,
  REQUEST_CLARIFICATION = 1U,
  ABSTAIN_HOLD_SAFE = 2U,
  LIVENESS_EMERGENCY_COMMIT = 3U,
  EMERGENCY_PREEMPT = 4U,
};

[[nodiscard]] constexpr std::string_view snr_stratum_name(SnrStratum s) noexcept {
  switch (s) {
    case SnrStratum::S0_CLEAN:
      return "S0_Clean";
    case SnrStratum::S1_CRUISE:
      return "S1_Cruise";
    case SnrStratum::S2_HOVER:
      return "S2_Hover";
    case SnrStratum::S3_GARBLED:
      return "S3_Garbled";
  }
  return "S3_Garbled";
}

[[nodiscard]] constexpr std::string_view voice_lang_name(VoiceLang lang) noexcept {
  switch (lang) {
    case VoiceLang::TR:
      return "TR";
    case VoiceLang::EN:
      return "EN";
    case VoiceLang::MIX:
      return "MIX";
  }
  return "TR";
}

[[nodiscard]] constexpr SnrStratum classify_snr_stratum(float snr_db) noexcept {
  if (snr_db > 15.0f) {
    return SnrStratum::S0_CLEAN;
  }
  if (snr_db >= 8.0f) {
    return SnrStratum::S1_CRUISE;
  }
  if (snr_db >= 2.0f) {
    return SnrStratum::S2_HOVER;
  }
  return SnrStratum::S3_GARBLED;
}

// Pure C++20 deterministic SHA-256 hex digest (zero OpenSSL/GPL dependency).
[[nodiscard]] std::string sha256_hex_bytes(const void* data, std::size_t len);

}  // namespace vertirta::voice
