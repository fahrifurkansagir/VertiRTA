// SPDX-License-Identifier: Apache-2.0
#include "vertirta/voice/silero_vad.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vertirta::voice {

SileroVadGate::SileroVadGate(SileroVadConfig cfg) : cfg_(cfg) {
  if (cfg_.sample_rate_hz == 0U) {
    cfg_.sample_rate_hz = kVoiceSampleRateHz;
  }
  if (cfg_.frame_samples == 0U) {
    cfg_.frame_samples = kVadFrameSamples;
  }
}

float SileroVadGate::score_frame(std::span<const float> frame_512) const noexcept {
  if (frame_512.empty()) {
    return 0.0f;
  }
  double energy_sum = 0.0;
  std::size_t zcr_count = 0U;
  for (std::size_t i = 0U; i < frame_512.size(); ++i) {
    const double x = static_cast<double>(frame_512[i]);
    energy_sum += x * x;
    if (i > 0U && ((frame_512[i] >= 0.0f) != (frame_512[i - 1U] >= 0.0f))) {
      ++zcr_count;
    }
  }
  const double rms = std::sqrt(energy_sum / static_cast<double>(frame_512.size()));
  const double zcr = static_cast<double>(zcr_count) / static_cast<double>(frame_512.size());

  // Speech energy + human voiced zero-crossing band (approx 0.015..0.28 at 16 kHz).
  const double energy_db = 20.0 * std::log10(std::max(rms, 1e-7));
  const double energy_logit = (energy_db + 36.0) / 6.0;
  const double zcr_penalty = (zcr < 0.008 || zcr > 0.35) ? -1.4 : 0.35;
  const double logit = energy_logit + zcr_penalty;
  const double prob = 1.0 / (1.0 + std::exp(-std::clamp(logit, -20.0, 20.0)));
  return static_cast<float>(prob);
}

SpeechSegment SileroVadGate::extract_utterance(std::span<const float> pcm_16khz) const {
  SpeechSegment seg{};
  if (pcm_16khz.empty()) {
    seg.audio_sha256_hex = sha256_hex_bytes(nullptr, 0U);
    return seg;
  }

  const std::size_t frame_n = cfg_.frame_samples;
  const std::size_t num_frames = pcm_16khz.size() / frame_n;
  if (num_frames == 0U) {
    seg.pcm_16khz.assign(pcm_16khz.begin(), pcm_16khz.end());
    seg.duration_s = static_cast<float>(seg.pcm_16khz.size()) / static_cast<float>(cfg_.sample_rate_hz);
    seg.mean_speech_prob = score_frame(pcm_16khz);
    seg.estimated_snr_db = 0.0f;
    seg.stratum = classify_snr_stratum(seg.estimated_snr_db);
    seg.audio_sha256_hex = sha256_hex_bytes(seg.pcm_16khz.data(), seg.pcm_16khz.size() * sizeof(float));
    return seg;
  }

  std::vector<float> frame_probs(num_frames, 0.0f);
  std::vector<double> frame_powers(num_frames, 0.0);
  std::size_t first_speech = num_frames;
  std::size_t last_speech = 0U;
  double speech_prob_sum = 0.0;
  std::size_t speech_frame_count = 0U;

  for (std::size_t f = 0U; f < num_frames; ++f) {
    const auto span = pcm_16khz.subspan(f * frame_n, frame_n);
    const float p = score_frame(span);
    frame_probs[f] = p;
    double pwr = 0.0;
    for (float v : span) {
      pwr += static_cast<double>(v) * static_cast<double>(v);
    }
    frame_powers[f] = pwr / static_cast<double>(frame_n);
    if (p >= cfg_.speech_prob_threshold) {
      if (first_speech == num_frames) {
        first_speech = f;
      }
      last_speech = f;
      speech_prob_sum += static_cast<double>(p);
      ++speech_frame_count;
    }
  }

  const std::size_t pre_frames =
      (static_cast<std::size_t>(cfg_.pre_speech_pad_ms) * cfg_.sample_rate_hz) / (1000U * frame_n);
  const std::size_t hang_frames =
      (static_cast<std::size_t>(cfg_.hangover_ms) * cfg_.sample_rate_hz) / (1000U * frame_n);

  std::size_t start_frame = 0U;
  std::size_t end_frame = num_frames - 1U;
  if (speech_frame_count > 0U) {
    start_frame = (first_speech > pre_frames) ? (first_speech - pre_frames) : 0U;
    end_frame = std::min(num_frames - 1U, last_speech + hang_frames);
    seg.mean_speech_prob = static_cast<float>(speech_prob_sum / static_cast<double>(speech_frame_count));
  } else {
    seg.mean_speech_prob =
        std::accumulate(frame_probs.begin(), frame_probs.end(), 0.0f) / static_cast<float>(num_frames);
  }

  const std::size_t start_sample = start_frame * frame_n;
  std::size_t end_sample = std::min(pcm_16khz.size(), (end_frame + 1U) * frame_n);
  const std::size_t max_samples =
      static_cast<std::size_t>(cfg_.max_utterance_s * static_cast<float>(cfg_.sample_rate_hz));
  if (end_sample - start_sample > max_samples) {
    end_sample = start_sample + max_samples;
    seg.truncated_at_max_duration = true;
  }

  seg.pcm_16khz.assign(pcm_16khz.begin() + static_cast<std::ptrdiff_t>(start_sample),
                       pcm_16khz.begin() + static_cast<std::ptrdiff_t>(end_sample));
  seg.duration_s = static_cast<float>(seg.pcm_16khz.size()) / static_cast<float>(cfg_.sample_rate_hz);

  // Estimate SNR from 85th-percentile active frame power vs 15th-percentile noise-floor frame power.
  std::vector<double> sorted_pwr = frame_powers;
  std::sort(sorted_pwr.begin(), sorted_pwr.end());
  const double noise_pwr = std::max(sorted_pwr[(num_frames * 15U) / 100U], 1e-12);
  const double sig_pwr = std::max(sorted_pwr[(num_frames * 85U) / 100U], noise_pwr);
  const double net_sig = std::max(sig_pwr - noise_pwr, 1e-12);
  seg.estimated_snr_db = static_cast<float>(std::clamp(10.0 * std::log10(net_sig / noise_pwr), -10.0, 40.0));
  seg.stratum = classify_snr_stratum(seg.estimated_snr_db);
  seg.audio_sha256_hex =
      sha256_hex_bytes(seg.pcm_16khz.data(), seg.pcm_16khz.size() * sizeof(float));
  return seg;
}

}  // namespace vertirta::voice
