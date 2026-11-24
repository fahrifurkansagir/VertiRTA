// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "vertirta/voice/voice_types.hpp"

namespace vertirta::voice {

struct WhisperDecodePolicy {
  AsrModelTier fast_model_tier{AsrModelTier::WHISPER_SMALL_Q8_0};
  AsrModelTier verify_model_tier{AsrModelTier::WHISPER_LARGE_V3_TURBO_Q8_0};
  bool translate{false};             // HARD LOCK: never translate TR->EN inside ASR
  bool no_context{true};             // HARD LOCK: zero cross-turn hallucination carry-over
  bool carry_initial_prompt{true};   // Preserve aviation lexicon prompt
  std::uint32_t min_audio_ctx{256U};
  std::uint32_t max_audio_ctx{384U};
  std::uint32_t nbest_k{4U};
  float escalate_plog_threshold{-0.35f};
  float escalate_snr_db_threshold{8.0f};
  float escalate_slot_prob_threshold{0.80f};
  std::string initial_prompt{
      "Kartal Sahil, Dragos Plaza, Maltepe Park, Cevizli Helipad, "
      "LTFJ, LTBX, MAYDAY, PAN-PAN, irtifa, divert, hold loiter, abort return"};
};

struct AsrHypothesis {
  std::string text_utf8{};
  float mean_token_plog{-0.20f};  // whisper_token_data::plog mean
  float normalized_weight{0.0f};
  float min_safety_slot_prob{1.0f};
  bool contains_emergency_keyword{false};
};

struct AsrTranscriptionResult {
  AsrBackend backend{AsrBackend::DETERMINISTIC_CI_ADAPTER};
  AsrModelTier tier_used{AsrModelTier::WHISPER_SMALL_Q8_0};
  bool escalated_to_turbo{false};
  std::uint32_t audio_ctx_used{256U};
  VoiceLang detected_lang{VoiceLang::TR};
  float nospeech_prob{0.0f};
  float nbest_shannon_entropy_norm{0.0f};
  float min_safety_slot_prob{1.0f};
  float mayday_acoustic_weight{0.0f};
  double decode_latency_ms{0.0};
  std::vector<AsrHypothesis> nbest{};

  [[nodiscard]] std::string top_transcript() const {
    return nbest.empty() ? std::string{} : nbest.front().text_utf8;
  }
};

[[nodiscard]] WhisperDecodePolicy make_default_whisper_policy();

// Bounded Whisper audio_ctx = clamp(round_up_64(ceil(dur_s / 30 * 1500) + 128), 256, 384).
[[nodiscard]] std::uint32_t compute_bounded_audio_ctx(
    float duration_s,
    std::uint32_t min_ctx = 256U,
    std::uint32_t max_ctx = 384U) noexcept;

// Normalized Shannon entropy H / log(K) in [0, 1] over K N-best weights.
[[nodiscard]] float compute_normalized_nbest_entropy(std::span<const float> weights) noexcept;

// Detect TR/EN aviation emergency keywords (MAYDAY, PAN-PAN, ACIL, MOTOR ARIZASI).
[[nodiscard]] bool detect_emergency_phrase(std::string_view text_utf8) noexcept;

// Dual-tier whisper.cpp (MIT) ASR orchestrator (small Q8_0 fast path -> large-v3-turbo verify path).
class DualTierWhisperAsr {
 public:
  explicit DualTierWhisperAsr(WhisperDecodePolicy policy = make_default_whisper_policy());

  [[nodiscard]] const WhisperDecodePolicy& policy() const noexcept { return policy_; }

  // Run dual-tier decoding over segmented audio; deterministic hint text is accepted for CI/unit testing
  // when native whisper.cpp model weights are not loaded on disk.
  [[nodiscard]] AsrTranscriptionResult transcribe(
      std::span<const float> pcm_16khz,
      float estimated_snr_db,
      std::string_view ci_hint_transcript = {},
      float ci_hint_plog = -0.15f) const;

 private:
  WhisperDecodePolicy policy_{};
};

}  // namespace vertirta::voice
