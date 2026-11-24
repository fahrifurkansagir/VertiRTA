// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

#include "vertirta_voice_node/voice_node_core.hpp"

using namespace vertirta::voice_node;

TEST_CASE("VoiceNodeCore end-to-end turn executes VAD, dual-tier ASR, Mondrian gate, Z3 verify, TTS < 250 ms, and sidecar (seq, t_sim_us)") {
  REQUIRE(std::string(kTopicAudio16kHz) == "/vertirta/voice/audio_16khz");
  REQUIRE(std::string(kTopicRtaStateSnapshot) == "/vertirta/rta/state_snapshot");
  REQUIRE(std::string(kTopicVoiceDslCommand) == "/vertirta/voice/dsl_command");
  REQUIRE(std::string(kTopicVoiceTtsReadback) == "/vertirta/voice/tts_readback");
  REQUIRE(std::string(kTopicForensicsSidecarEvent) == "/vertirta/forensics/sidecar_event");

  VoiceNodeCore core{};
  auto snap = vertirta::copilot::make_reference_400v_sml_snapshot();
  snap.seq_id = 4200ULL;
  snap.timestamp_ms = 42000ULL;  // 42,000,000 us
  core.submit_rta_snapshot(snap);

  AudioTurnRequest req{};
  req.turn_id = 1U;
  req.snr_db = 14.0f;  // S1_Cruise
  req.vad_speech_prob = 0.94f;
  req.audio_rms = 0.22f;
  req.lang = vertirta::copilot::LangCode::TR;
  req.transcript_hint = "Kartal Sahil inis yap";
  req.dsl_candidates[0] = {
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", 0.96f};
  req.dsl_candidates[1] = {"ACT=HOLD_LOITER;DUR=60;AGL=60\nLNG=TR;MOD=BRIEF", 0.03f};
  req.candidate_count = 2U;

  const VoiceTurnOutcome out = core.process_turn(req);
  REQUIRE(out.vad_triggered);
  REQUIRE(out.snr_stratum == SnrStratum::S1_Cruise);
  REQUIRE(out.asr_tier == AsrTier::FAST_WHISPER_SMALL_Q8);
  REQUIRE(out.conformal_set_size == 1U);
  REQUIRE(out.z3_verified_sat);
  REQUIRE(out.dsl_dispatched);
  REQUIRE(out.tts_ttfa_ms < 250.0);
  REQUIRE(out.tts_readback_utf8.size() > 10U);
  REQUIRE(out.sidecar_seq == 4200ULL);
  REQUIRE(out.sidecar_t_sim_us == 42'000'000ULL);
  REQUIRE(out.sidecar_jsonl.find("\"seq\":4200") != std::string::npos);
  REQUIRE(out.sidecar_jsonl.find("\"t_sim_us\":42000000") != std::string::npos);
  REQUIRE(out.sidecar_jsonl.find("\"snr_stratum\":\"S1_Cruise\"") != std::string::npos);
  REQUIRE(out.sidecar_jsonl.find("\"event_sha256\":\"") != std::string::npos);
}

TEST_CASE("VoiceNodeCore MAYDAY in S3_Garbled (w_m >= 0.15) preempts clarification and logs EMERGENCY_PREEMPT") {
  VoiceNodeCore core{};
  auto snap = vertirta::copilot::make_reference_400v_sml_snapshot();
  snap.seq_id = 5100ULL;
  snap.timestamp_ms = 51000ULL;
  core.submit_rta_snapshot(snap);

  AudioTurnRequest req{};
  req.turn_id = 2U;
  req.snr_db = -3.5f;  // S3_Garbled
  req.vad_speech_prob = 0.88f;
  req.audio_rms = 0.45f;
  req.lang = vertirta::copilot::LangCode::EN;
  req.mayday_posterior = 0.21f;  // >= 0.15 threshold
  req.transcript_hint = "MAYDAY MAYDAY rotor vibration immediate landing";
  req.dsl_candidates[0] = {"ACT=HOLD_LOITER;DUR=60;AGL=60\nLNG=EN;MOD=BRIEF", 0.42f};
  req.dsl_candidates[1] = {
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=EN;MOD=BRIEF", 0.36f};
  req.dsl_candidates[2] = {
      "ACT=ABORT_RETURN;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=EN;MOD=BRIEF", 0.22f};
  req.candidate_count = 3U;

  const VoiceTurnOutcome out = core.process_turn(req);
  REQUIRE(out.snr_stratum == SnrStratum::S3_Garbled);
  REQUIRE(out.asr_tier == AsrTier::HEAVY_WHISPER_LARGE_V3_TURBO_Q8);
  REQUIRE(out.emergency_preempted);
  REQUIRE_FALSE(out.clarification_required);
  REQUIRE(out.dsl_dispatched);
  REQUIRE(out.tts_ttfa_ms < 250.0);
  REQUIRE(out.sidecar_jsonl.find("\"event_kind\":\"EMERGENCY_PREEMPT\"") != std::string::npos);
  REQUIRE(out.sidecar_jsonl.find("\"snr_stratum\":\"S3_Garbled\"") != std::string::npos);
}

TEST_CASE("VoiceNodeCore < 5.0% SoC triggers LIVENESS_EMERGENCY_COMMIT and bounded SPSC decouples 100 Hz RTA path") {
  VoiceNodeCore core{};
  auto snap = vertirta::copilot::make_reference_400v_sml_snapshot();
  snap.seq_id = 6000ULL;
  snap.timestamp_ms = 60000ULL;
  snap.soc_pct = 4.2f;  // < 5.0% critical SoC
  core.submit_rta_snapshot(snap);

  AudioTurnRequest low_soc_req{};
  low_soc_req.turn_id = 3U;
  low_soc_req.snr_db = 4.0f;  // S2_Hover
  low_soc_req.vad_speech_prob = 0.91f;
  low_soc_req.audio_rms = 0.30f;
  low_soc_req.lang = vertirta::copilot::LangCode::TR;
  low_soc_req.transcript_hint = "nereye inelim";
  low_soc_req.dsl_candidates[0] = {
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", 0.45f};
  low_soc_req.dsl_candidates[1] = {"ACT=HOLD_LOITER;DUR=60;AGL=60\nLNG=TR;MOD=BRIEF", 0.40f};
  low_soc_req.candidate_count = 2U;

  const auto out = core.process_turn(low_soc_req);
  REQUIRE(out.liveness_emergency_commit);
  REQUIRE_FALSE(out.clarification_required);
  REQUIRE(out.dsl_dispatched);
  REQUIRE(out.sidecar_jsonl.find("\"event_kind\":\"LIVENESS_EMERGENCY_COMMIT\"") != std::string::npos);

  // Concurrent TSan stress: 100 Hz RTA snapshot writer + non-blocking SPSC audio ingress + async worker
  core.start_worker();
  std::atomic<bool> writer_done{false};
  std::thread rta_writer([&]() {
    auto s = vertirta::copilot::make_reference_400v_sml_snapshot();
    for (uint64_t seq = 1ULL; seq <= 200ULL; ++seq) {
      s.seq_id = 7000ULL + seq;
      s.timestamp_ms = (7000ULL + seq) * 10ULL;
      core.submit_rta_snapshot(s);
      if ((seq % 25ULL) == 0ULL) {
        AudioTurnRequest q{};
        q.turn_id = static_cast<uint32_t>(seq);
        q.snr_db = 22.0f;
        q.vad_speech_prob = 0.95f;
        q.audio_rms = 0.20f;
        q.lang = vertirta::copilot::LangCode::EN;
        q.transcript_hint = "Divert to Kartal Park";
        q.dsl_candidates[0] = {
            "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=EN;MOD=BRIEF", 0.97f};
        q.candidate_count = 1U;
        (void)core.try_enqueue_audio_turn(q);
      }
    }
    writer_done.store(true, std::memory_order_release);
  });

  rta_writer.join();
  REQUIRE(writer_done.load(std::memory_order_acquire));
  std::vector<VoiceTurnOutcome> collected{};
  for (int wait_i = 0; wait_i < 200 && collected.size() < 4U; ++wait_i) {
    VoiceTurnOutcome item{};
    while (core.try_pop_outcome(item)) {
      collected.push_back(item);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  core.stop_worker();
  REQUIRE(collected.size() >= 4U);
  for (const auto& item : collected) {
    REQUIRE(item.sidecar_seq >= 7001ULL);
    REQUIRE(item.tts_ttfa_ms < 250.0);
  }
}
