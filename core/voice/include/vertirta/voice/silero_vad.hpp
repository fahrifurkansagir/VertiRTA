// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vertirta/voice/voice_types.hpp"

namespace vertirta::voice {

struct SileroVadConfig {
  std::uint32_t sample_rate_hz{kVoiceSampleRateHz};
  std::size_t frame_samples{kVadFrameSamples};
  float speech_prob_threshold{0.50f};
  std::uint32_t pre_speech_pad_ms{300U};
  std::uint32_t hangover_ms{450U};
  float max_utterance_s{12.0f};
};

struct SpeechSegment {
  std::vector<float> pcm_16khz{};
  float duration_s{0.0f};
  float estimated_snr_db{0.0f};
  SnrStratum stratum{SnrStratum::S3_GARBLED};
  float mean_speech_prob{0.0f};
  std::string audio_sha256_hex{};
  bool truncated_at_max_duration{false};
};

// Frame-gated voice activity detector modeling Silero VAD v6.2.0 (MIT) 32 ms windowing,
// pre-speech lookback, trailing hangover, real-time RMS SNR estimation, and SHA-256 PCM digest.
class SileroVadGate {
 public:
  explicit SileroVadGate(SileroVadConfig cfg = {});

  [[nodiscard]] const SileroVadConfig& config() const noexcept { return cfg_; }

  // Estimate per-frame speech probability in [0, 1] for a 512-sample 16 kHz frame.
  [[nodiscard]] float score_frame(std::span<const float> frame_512) const noexcept;

  // Segment an utterance buffer into speech audio, SNR estimate, stratum, and SHA-256 digest.
  [[nodiscard]] SpeechSegment extract_utterance(std::span<const float> pcm_16khz) const;

 private:
  SileroVadConfig cfg_{};
};

}  // namespace vertirta::voice
