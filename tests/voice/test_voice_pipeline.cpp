// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "vertirta/voice/dual_tier_whisper_asr.hpp"
#include "vertirta/voice/silero_vad.hpp"
#include "vertirta/voice/voice_pipeline.hpp"
#include "vertirta/voice/zero_gpl_tts.hpp"

using namespace vertirta::voice;

namespace {

std::vector<float> make_test_speech_pcm(float duration_s, float speech_amp, float noise_amp) {
  const std::size_t n = static_cast<std::size_t>(duration_s * static_cast<float>(kVoiceSampleRateHz));
  std::vector<float> pcm(n, 0.0f);
  constexpr double kTwoPi = 6.28318530717958647692;
  for (std::size_t i = 0U; i < n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kVoiceSampleRateHz);
    const bool in_speech = (t >= 0.35 && t <= static_cast<double>(duration_s) - 0.35);
    const double noise =
        static_cast<double>(noise_amp) * std::sin(kTwoPi * 114.6 * t) * 0.6 +
        static_cast<double>(noise_amp) * std::sin(kTwoPi * 229.2 * t + 0.7) * 0.4;
    const double speech = in_speech
                              ? (static_cast<double>(speech_amp) *
                                 (0.65 * std::sin(kTwoPi * 320.0 * t) + 0.35 * std::sin(kTwoPi * 690.0 * t)))
                              : 0.0;
    pcm[i] = static_cast<float>(speech + noise);
  }
  return pcm;
}

}  // namespace

TEST_CASE("DualTierWhisperAsr enforces audio_ctx [256, 384] rounding, no_context, translate=false, and K=4 N-best") {
  REQUIRE(compute_bounded_audio_ctx(1.2f) == 256U);
  REQUIRE(compute_bounded_audio_ctx(3.0f) == 320U);
  REQUIRE(compute_bounded_audio_ctx(6.0f) == 384U);
  REQUIRE(compute_bounded_audio_ctx(25.0f) == 384U);

  const WhisperDecodePolicy pol = make_default_whisper_policy();
  REQUIRE_FALSE(pol.translate);
  REQUIRE(pol.no_context);
  REQUIRE(pol.carry_initial_prompt);
  REQUIRE(pol.nbest_k == 4U);
  REQUIRE(pol.fast_model_tier == AsrModelTier::WHISPER_SMALL_Q8_0);
  REQUIRE(pol.verify_model_tier == AsrModelTier::WHISPER_LARGE_V3_TURBO_Q8_0);

  DualTierWhisperAsr asr(pol);
  const auto clean_pcm = make_test_speech_pcm(2.0f, 0.25f, 0.01f);
  const auto fast_res = asr.transcribe(clean_pcm, 18.0f, "Kartal Sahil divert", -0.10f);
  REQUIRE_FALSE(fast_res.escalated_to_turbo);
  REQUIRE(fast_res.tier_used == AsrModelTier::WHISPER_SMALL_Q8_0);
  REQUIRE(fast_res.nbest.size() == 4U);
  REQUIRE(fast_res.audio_ctx_used >= 256U);
  REQUIRE(fast_res.audio_ctx_used <= 384U);

  const auto noisy_res = asr.transcribe(clean_pcm, 3.5f, "MAYDAY motor arizasi Kartal Sahil", -0.55f);
  REQUIRE(noisy_res.escalated_to_turbo);
  REQUIRE(noisy_res.tier_used == AsrModelTier::WHISPER_LARGE_V3_TURBO_Q8_0);
  REQUIRE(noisy_res.mayday_acoustic_weight >= 0.15f);
}

TEST_CASE("ZeroGplSherpaTts enforces sherpa-onnx >= 2.0.0, bans GPL & NC voices, and maps Turkish G2P < 250 ms TTFA") {
  REQUIRE(is_tts_voice_license_allowed("en_US-ljspeech-medium"));
  REQUIRE(is_tts_voice_license_allowed("en_US-libritts_r-medium"));
  REQUIRE(is_tts_voice_license_allowed("tr_TR-ema-lightning-apache2"));
  REQUIRE(is_tts_voice_license_allowed("tr_TR-okur-tts-apache2"));
  REQUIRE(is_tts_voice_license_allowed("vits-piper-tr_TR-vertirta-clean"));

  // Hard license bans (Erratum #8 / DR-12)
  REQUIRE_FALSE(is_tts_voice_license_allowed("tr_TR-dfki-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("tr_TR-fahrettin-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("tr_TR-fettah-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("en_US-lessac-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("en_US-ryan-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("xtts-v2"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("mms-tts-tur"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("piper1-gpl"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("espeak-ng"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("pocket-tts-turkish"));

  REQUIRE(check_sherpa_onnx_semver(2, 0, 0));
  REQUIRE(check_sherpa_onnx_semver(2, 1, 3));
  REQUIRE_FALSE(check_sherpa_onnx_semver(1, 12, 9));

  const auto cps = TurkishG2pPhonemizer::text_to_phoneme_codepoints("Kartal Sahil'e y\xC3\xB6nel, irtifa 80 metre.");
  REQUIRE(cps.size() >= 16U);

  ZeroGplSherpaTts tts(make_default_tts_config());
  const auto res_tr = tts.synthesize_clause_stream("Kartal Sahil'e y\xC3\xB6nel, irtifa 80 metre.", VoiceLang::TR);
  REQUIRE(res_tr.ok);
  REQUIRE(res_tr.ttfa_ms < 250.0);
  REQUIRE(res_tr.clauses.size() == 2U);
  REQUIRE_FALSE(res_tr.pcm_16khz.empty());

  // Attempting to run sherpa-onnx < 2.0.0 must fail closed.
  auto bad_cfg = make_default_tts_config();
  bad_cfg.sherpa_major = 1U;
  ZeroGplSherpaTts bad_tts(bad_cfg);
  const auto bad_res = bad_tts.synthesize_clause_stream("Divert Kartal.", VoiceLang::EN);
  REQUIRE_FALSE(bad_res.ok);
  REQUIRE(bad_res.error_reason == "BANNED_SHERPA_ONNX_VERSION_LT_2_0_0");
}

TEST_CASE("VoiceAtcPipeline produces deterministic SHA-256 audio digest, VAD SNR, ASR, Mondrian gate, and TTS readback") {
  VoiceAtcPipeline pipe;
  const auto pcm = make_test_speech_pcm(2.0f, 0.30f, 0.015f);
  std::vector<VoiceCandidateScore> cands{
      {.id = "DIVERT_SLZ:Kartal_Sahil", .prob = 0.91f, .z3_sat = true},
      {.id = "DIVERT_SLZ:Dragos_Plaza", .prob = 0.06f, .z3_sat = true},
  };
  VoiceRtaLivenessState rta{.soc_pct = 55.0f, .soc_reserve_floor_pct = 12.0f, .seq = 4200U, .t_sim_us = 42'000'000U};

  const auto rec = pipe.process_turn(pcm, cands, rta, "Kartal Sahil'e y\xC3\xB6nel", -0.10f);
  REQUIRE(rec.seq == 4200U);
  REQUIRE(rec.t_sim_us == 42'000'000U);
  REQUIRE(rec.vad_segment.audio_sha256_hex.size() == 64U);
  REQUIRE(rec.gate.decision == VoiceGateAction::EXECUTE_SINGLE);
  REQUIRE(rec.gate.selected_action_id == "DIVERT_SLZ:Kartal_Sahil");
  REQUIRE(rec.tts.ok);
  REQUIRE(rec.tts.ttfa_ms < 250.0);
}
