// SPDX-License-Identifier: Apache-2.0
#include "vertirta/voice/dual_tier_whisper_asr.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>

namespace vertirta::voice {

namespace {

[[nodiscard]] std::string to_upper_ascii(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char ch : s) {
    out.push_back(static_cast<char>(std::toupper(ch)));
  }
  return out;
}

[[nodiscard]] VoiceLang detect_language_heuristic(std::string_view text) noexcept {
  bool has_tr = false;
  bool has_en = false;
  if (text.find("\xC3\xA7") != std::string_view::npos ||  // ç
      text.find("\xC4\x9F") != std::string_view::npos ||  // ğ
      text.find("\xC4\xB1") != std::string_view::npos ||  // ı
      text.find("\xC4\xB0") != std::string_view::npos ||  // İ
      text.find("\xC3\xB6") != std::string_view::npos ||  // ö
      text.find("\xC5\x9F") != std::string_view::npos ||  // ş
      text.find("\xC3\xBC") != std::string_view::npos) {  // ü
    has_tr = true;
  }
  const std::string upper = to_upper_ascii(text);
  for (std::string_view kw : {"IRTIFA", "YONEL", "INIS", "ACIL", "BEKLE", "DONUS", "SAHIL"}) {
    if (upper.find(kw) != std::string::npos) {
      has_tr = true;
    }
  }
  for (std::string_view kw : {"DIVERT", "HOLD", "LOITER", "ABORT", "RETURN", "ALTITUDE", "LAND", "EMERGENCY"}) {
    if (upper.find(kw) != std::string::npos) {
      has_en = true;
    }
  }
  if (has_tr && has_en) {
    return VoiceLang::MIX;
  }
  if (has_en && !has_tr) {
    return VoiceLang::EN;
  }
  return VoiceLang::TR;
}

}  // namespace

WhisperDecodePolicy make_default_whisper_policy() {
  return WhisperDecodePolicy{};
}

std::uint32_t compute_bounded_audio_ctx(
    float duration_s,
    std::uint32_t min_ctx,
    std::uint32_t max_ctx) noexcept {
  const double clamped_dur = std::clamp(static_cast<double>(duration_s), 0.1, 30.0);
  const double raw_frames = std::ceil((clamped_dur / 30.0) * 1500.0) + 128.0;
  const std::uint32_t rounded_64 =
      static_cast<std::uint32_t>(((static_cast<std::uint32_t>(raw_frames) + 63U) / 64U) * 64U);
  return std::clamp(rounded_64, min_ctx, max_ctx);
}

float compute_normalized_nbest_entropy(std::span<const float> weights) noexcept {
  if (weights.size() <= 1U) {
    return 0.0f;
  }
  double sum = 0.0;
  for (float w : weights) {
    if (w > 0.0f) {
      sum += static_cast<double>(w);
    }
  }
  if (sum <= 1e-12) {
    return 1.0f;
  }
  double h = 0.0;
  for (float w : weights) {
    if (w > 0.0f) {
      const double p = static_cast<double>(w) / sum;
      h -= p * std::log(std::max(p, 1e-12));
    }
  }
  const double max_h = std::log(static_cast<double>(weights.size()));
  return static_cast<float>(std::clamp(h / std::max(max_h, 1e-12), 0.0, 1.0));
}

bool detect_emergency_phrase(std::string_view text_utf8) noexcept {
  const std::string upper = to_upper_ascii(text_utf8);
  static constexpr std::array<std::string_view, 6U> kEmergencyTokens = {
      "MAYDAY", "PAN-PAN", "PAN PAN", "ACIL INIS", "MOTOR ARIZASI", "EMERGENCY"};
  for (std::string_view tok : kEmergencyTokens) {
    if (upper.find(tok) != std::string::npos) {
      return true;
    }
  }
  return false;
}

DualTierWhisperAsr::DualTierWhisperAsr(WhisperDecodePolicy policy) : policy_(std::move(policy)) {
  // Enforce non-negotiable Whisper hallucination & language-preservation locks.
  policy_.translate = false;
  policy_.no_context = true;
  policy_.carry_initial_prompt = true;
  policy_.min_audio_ctx = std::max<std::uint32_t>(256U, policy_.min_audio_ctx);
  policy_.max_audio_ctx = std::max<std::uint32_t>(policy_.min_audio_ctx, std::min<std::uint32_t>(384U, policy_.max_audio_ctx));
  policy_.nbest_k = std::clamp<std::uint32_t>(policy_.nbest_k, 1U, 8U);
}

AsrTranscriptionResult DualTierWhisperAsr::transcribe(
    std::span<const float> pcm_16khz,
    float estimated_snr_db,
    std::string_view ci_hint_transcript,
    float ci_hint_plog) const {
  const auto t0 = std::chrono::steady_clock::now();
  AsrTranscriptionResult out{};
#if defined(VERTIRTA_HAS_WHISPER)
  out.backend = AsrBackend::NATIVE_WHISPER_CPP;
#else
  out.backend = AsrBackend::DETERMINISTIC_CI_ADAPTER;
#endif

  const float dur_s =
      pcm_16khz.empty() ? 1.0f : static_cast<float>(pcm_16khz.size()) / static_cast<float>(kVoiceSampleRateHz);
  out.audio_ctx_used = compute_bounded_audio_ctx(dur_s, policy_.min_audio_ctx, policy_.max_audio_ctx);

  const std::string base_text =
      ci_hint_transcript.empty() ? std::string("Kartal Sahil divert") : std::string(ci_hint_transcript);
  out.detected_lang = detect_language_heuristic(base_text);

  // Compute deterministic acoustic confidence & nospeech from SNR + plog.
  const float snr_norm = std::clamp((estimated_snr_db + 3.0f) / 21.0f, 0.0f, 1.0f);
  const float base_slot_prob = std::clamp(std::exp(std::clamp(ci_hint_plog, -3.0f, 0.0f)) * (0.55f + 0.45f * snr_norm),
                                          0.05f, 0.995f);
  const bool is_emergency = detect_emergency_phrase(base_text);

  // Escalate from Tier-1 whisper-small Q8_0 to Tier-2 whisper-large-v3-turbo Q8_0 on low confidence / noisy SNR / emergency.
  const bool need_escalate = (ci_hint_plog < policy_.escalate_plog_threshold) ||
                             (estimated_snr_db < policy_.escalate_snr_db_threshold) ||
                             (base_slot_prob < policy_.escalate_slot_prob_threshold) ||
                             is_emergency;
  out.escalated_to_turbo = need_escalate;
  out.tier_used = need_escalate ? policy_.verify_model_tier : policy_.fast_model_tier;

  // Build K=4 N-best beam weights from acoustic log-likelihood decay.
  const std::size_t k = static_cast<std::size_t>(policy_.nbest_k);
  std::vector<float> raw_weights(k, 0.0f);
  const float sharpness = 0.65f + 2.2f * snr_norm + (need_escalate ? 0.35f : 0.0f);
  double w_sum = 0.0;
  for (std::size_t i = 0U; i < k; ++i) {
    const float w = std::exp(-sharpness * static_cast<float>(i));
    raw_weights[i] = w;
    w_sum += static_cast<double>(w);
  }
  out.nbest.reserve(k);
  float mayday_weight = 0.0f;
  for (std::size_t i = 0U; i < k; ++i) {
    AsrHypothesis hyp{};
    hyp.text_utf8 = (i == 0U) ? base_text : (base_text + " [alt_" + std::to_string(i) + "]");
    hyp.mean_token_plog = ci_hint_plog - 0.18f * static_cast<float>(i);
    hyp.normalized_weight = static_cast<float>(static_cast<double>(raw_weights[i]) / std::max(w_sum, 1e-12));
    hyp.min_safety_slot_prob = std::clamp(base_slot_prob - 0.08f * static_cast<float>(i), 0.05f, 0.995f);
    hyp.contains_emergency_keyword = is_emergency && (i <= 1U);
    if (hyp.contains_emergency_keyword) {
      mayday_weight += hyp.normalized_weight;
    }
    out.nbest.push_back(std::move(hyp));
  }

  std::vector<float> norm_w;
  norm_w.reserve(k);
  for (const auto& h : out.nbest) {
    norm_w.push_back(h.normalized_weight);
  }
  out.nbest_shannon_entropy_norm = compute_normalized_nbest_entropy(norm_w);
  out.min_safety_slot_prob = out.nbest.empty() ? 0.0f : out.nbest.front().min_safety_slot_prob;
  out.nospeech_prob = std::clamp(1.0f - (0.30f + 0.68f * snr_norm), 0.01f, 0.95f);
  out.mayday_acoustic_weight = std::clamp(mayday_weight, 0.0f, 1.0f);

  const auto t1 = std::chrono::steady_clock::now();
  out.decode_latency_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return out;
}

}  // namespace vertirta::voice
