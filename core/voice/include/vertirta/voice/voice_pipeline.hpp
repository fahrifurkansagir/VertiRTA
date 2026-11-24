// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "vertirta/voice/dual_tier_whisper_asr.hpp"
#include "vertirta/voice/mondrian_voice_gate.hpp"
#include "vertirta/voice/silero_vad.hpp"
#include "vertirta/voice/zero_gpl_tts.hpp"

namespace vertirta::voice {

struct VoiceTurnAuditRecord {
  std::uint64_t seq{0U};
  std::uint64_t t_sim_us{0U};
  SpeechSegment vad_segment{};
  AsrTranscriptionResult asr{};
  MondrianGateDecision gate{};
  TtsSynthesisResult tts{};
  std::string readback_text_utf8{};
};

// End-to-end offline cockpit voice pipeline:
//   16 kHz PCM -> SileroVadGate -> DualTierWhisperAsr -> MondrianVoiceConformalGate -> ZeroGplSherpaTts.
class VoiceAtcPipeline {
 public:
  VoiceAtcPipeline(
      SileroVadConfig vad_cfg = {},
      WhisperDecodePolicy asr_cfg = make_default_whisper_policy(),
      MondrianGateConfig gate_cfg = make_default_mondrian_config(),
      ZeroGplTtsConfig tts_cfg = make_default_tts_config());

  [[nodiscard]] VoiceTurnAuditRecord process_turn(
      std::span<const float> pcm_16khz,
      std::span<const VoiceCandidateScore> z3_verified_candidates,
      const VoiceRtaLivenessState& rta_state,
      std::string_view ci_hint_transcript = {},
      float ci_hint_plog = -0.15f) const;

 private:
  SileroVadGate vad_;
  DualTierWhisperAsr asr_;
  MondrianVoiceConformalGate gate_;
  ZeroGplSherpaTts tts_;
};

}  // namespace vertirta::voice
