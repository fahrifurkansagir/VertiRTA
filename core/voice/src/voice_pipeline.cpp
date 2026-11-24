// SPDX-License-Identifier: Apache-2.0
#include "vertirta/voice/voice_pipeline.hpp"

namespace vertirta::voice {

namespace {

[[nodiscard]] std::string format_readback(const MondrianGateDecision& d) {
  const bool is_en = (d.lang == VoiceLang::EN);
  switch (d.decision) {
    case VoiceGateAction::EMERGENCY_PREEMPT:
      return is_en ? ("Mayday acknowledged, immediate emergency divert " + d.selected_action_id + ".")
                   : ("Acil durum alindi, derhal guvenli yonlendirme " + d.selected_action_id + ".");
    case VoiceGateAction::LIVENESS_EMERGENCY_COMMIT:
      return is_en ? ("Low reserve margin, committing safest verified corridor " + d.selected_action_id + ".")
                   : ("Dusuk batarya rezervi, en guvenli dogrulanmis rota " + d.selected_action_id + ".");
    case VoiceGateAction::EXECUTE_SINGLE:
      return is_en ? ("Verified and executing " + d.selected_action_id + ".")
                   : ("Dogrulandi, uygulaniyor " + d.selected_action_id + ".");
    case VoiceGateAction::REQUEST_CLARIFICATION:
      return is_en ? ("Confirm target: " + d.clarify_option_a + " or " + d.clarify_option_b + "?")
                   : ("Hedefi dogrulayin: " + d.clarify_option_a + " veya " + d.clarify_option_b + "?");
    case VoiceGateAction::ABSTAIN_HOLD_SAFE:
      return is_en ? "Holding safe loiter, repeat command."
                   : "Guvenli beklemede, komutu tekrarlayin.";
  }
  return "Holding safe loiter.";
}

}  // namespace

VoiceAtcPipeline::VoiceAtcPipeline(
    SileroVadConfig vad_cfg,
    WhisperDecodePolicy asr_cfg,
    MondrianGateConfig gate_cfg,
    ZeroGplTtsConfig tts_cfg)
    : vad_(vad_cfg),
      asr_(std::move(asr_cfg)),
      gate_(gate_cfg),
      tts_(std::move(tts_cfg)) {}

VoiceTurnAuditRecord VoiceAtcPipeline::process_turn(
    std::span<const float> pcm_16khz,
    std::span<const VoiceCandidateScore> z3_verified_candidates,
    const VoiceRtaLivenessState& rta_state,
    std::string_view ci_hint_transcript,
    float ci_hint_plog) const {
  VoiceTurnAuditRecord rec{};
  rec.seq = rta_state.seq;
  rec.t_sim_us = rta_state.t_sim_us;
  rec.vad_segment = vad_.extract_utterance(pcm_16khz);
  rec.asr = asr_.transcribe(rec.vad_segment.pcm_16khz, rec.vad_segment.estimated_snr_db,
                            ci_hint_transcript, ci_hint_plog);

  VoiceTurnSignals sig{};
  sig.snr_db = rec.vad_segment.estimated_snr_db;
  sig.lang = rec.asr.detected_lang;
  sig.nbest_shannon_entropy_norm = rec.asr.nbest_shannon_entropy_norm;
  sig.min_safety_slot_prob = rec.asr.min_safety_slot_prob;
  sig.nospeech_prob = rec.asr.nospeech_prob;
  sig.mayday_acoustic_weight = rec.asr.mayday_acoustic_weight;

  rec.gate = gate_.evaluate(sig, z3_verified_candidates, rta_state);
  rec.readback_text_utf8 = format_readback(rec.gate);
  rec.tts = tts_.synthesize_clause_stream(rec.readback_text_utf8, rec.gate.lang);
  return rec;
}

}  // namespace vertirta::voice
