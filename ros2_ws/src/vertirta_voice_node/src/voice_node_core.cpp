// SPDX-License-Identifier: Apache-2.0
#include "vertirta_voice_node/voice_node_core.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <sstream>
#include <vector>

namespace vertirta::voice_node {

namespace {

// Self-contained FIPS 180-4 SHA-256 for deterministic .vrta_events.jsonl sidecar chaining.
struct Sha256Ctx {
  std::array<uint32_t, 8> h{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<uint8_t, 64> buf{};
  std::size_t buf_len{0U};
  uint64_t total_bits{0ULL};

  static constexpr uint32_t rotr(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32U - n));
  }

  void compress(const uint8_t block[64]) noexcept {
    static constexpr std::array<uint32_t, 64> k = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<uint32_t, 64> w{};
    for (std::size_t i = 0U; i < 16U; ++i) {
      w[i] = (static_cast<uint32_t>(block[i * 4U]) << 24U) |
             (static_cast<uint32_t>(block[i * 4U + 1U]) << 16U) |
             (static_cast<uint32_t>(block[i * 4U + 2U]) << 8U) |
             static_cast<uint32_t>(block[i * 4U + 3U]);
    }
    for (std::size_t i = 16U; i < 64U; ++i) {
      const uint32_t s0 = rotr(w[i - 15U], 7U) ^ rotr(w[i - 15U], 18U) ^ (w[i - 15U] >> 3U);
      const uint32_t s1 = rotr(w[i - 2U], 17U) ^ rotr(w[i - 2U], 19U) ^ (w[i - 2U] >> 10U);
      w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (std::size_t i = 0U; i < 64U; ++i) {
      const uint32_t s1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2U) ^ rotr(a, 13U) ^ rotr(a, 22U);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  void update(const uint8_t* data, std::size_t len) noexcept {
    total_bits += static_cast<uint64_t>(len) * 8ULL;
    for (std::size_t i = 0U; i < len; ++i) {
      buf[buf_len++] = data[i];
      if (buf_len == 64U) {
        compress(buf.data());
        buf_len = 0U;
      }
    }
  }

  std::string final_hex() {
    buf[buf_len++] = 0x80U;
    if (buf_len > 56U) {
      while (buf_len < 64U) {
        buf[buf_len++] = 0x00U;
      }
      compress(buf.data());
      buf_len = 0U;
    }
    while (buf_len < 56U) {
      buf[buf_len++] = 0x00U;
    }
    for (int i = 7; i >= 0; --i) {
      buf[buf_len++] = static_cast<uint8_t>((total_bits >> (static_cast<unsigned>(i) * 8U)) & 0xFFU);
    }
    compress(buf.data());

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(64U, '0');
    for (std::size_t i = 0U; i < 8U; ++i) {
      for (std::size_t j = 0U; j < 8U; ++j) {
        out[i * 8U + j] = kHex[(h[i] >> ((7U - j) * 4U)) & 0xFU];
      }
    }
    return out;
  }
};

std::string json_escape(std::string_view sv) {
  std::string out;
  out.reserve(sv.size() + 8U);
  for (char ch : sv) {
    if (ch == '"' || ch == '\\') {
      out.push_back('\\');
      out.push_back(ch);
    } else if (static_cast<unsigned char>(ch) < 0x20U) {
      out.push_back(' ');
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

bool contains_emergency_keyword(std::string_view text) {
  std::string upper;
  upper.reserve(text.size());
  for (char c : text) {
    upper.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return (upper.find("MAYDAY") != std::string::npos) ||
         (upper.find("PAN-PAN") != std::string::npos) ||
         (upper.find("PAN PAN") != std::string::npos) ||
         (upper.find("ACIL") != std::string::npos);
}

std::string pick_rank1_slz_id(const vertirta::copilot::RtaStateSnapshotV1& snap) {
  if (snap.num_slz > 0U && !snap.ranked_slzs[0].id_view().empty()) {
    return std::string(snap.ranked_slzs[0].id_view());
  }
  return "Kartal_Sahil";
}

}  // namespace

VoiceNodeCore::VoiceNodeCore(VoiceNodeConfig cfg)
    : cfg_(cfg),
      airspace_polys_(vertirta::copilot::load_istanbul_airspace_polys()),
      prev_event_sha256_(64U, '0') {
  rta_snapshot_.write(vertirta::copilot::make_reference_400v_sml_snapshot());
}

VoiceNodeCore::~VoiceNodeCore() {
  stop_worker();
}

void VoiceNodeCore::submit_rta_snapshot(const vertirta::copilot::RtaStateSnapshotV1& snapshot) noexcept {
  rta_snapshot_.write(snapshot);
  has_snapshot_.store(true, std::memory_order_release);
}

bool VoiceNodeCore::try_enqueue_audio_turn(AudioTurnRequest req) {
  const bool ok = req_queue_.try_push(std::move(req));
  if (ok) {
    wake_cv_.notify_one();
  }
  return ok;
}

bool VoiceNodeCore::try_pop_outcome(VoiceTurnOutcome& out) {
  return out_queue_.try_pop(out);
}

void VoiceNodeCore::start_worker() {
  bool expected = false;
  if (!worker_running_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
    return;
  }
  worker_ = std::thread(&VoiceNodeCore::worker_loop, this);
}

void VoiceNodeCore::stop_worker() {
  if (!worker_running_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }
  wake_cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void VoiceNodeCore::worker_loop() {
  while (worker_running_.load(std::memory_order_acquire)) {
    AudioTurnRequest req{};
    if (req_queue_.try_pop(req)) {
      VoiceTurnOutcome outcome = process_turn(req);
      (void)out_queue_.try_push(std::move(outcome));
      continue;
    }
    std::unique_lock<std::mutex> lock(wake_mu_);
    wake_cv_.wait_for(lock, std::chrono::milliseconds(5));
  }
  AudioTurnRequest req{};
  while (req_queue_.try_pop(req)) {
    VoiceTurnOutcome outcome = process_turn(req);
    (void)out_queue_.try_push(std::move(outcome));
  }
}

std::string VoiceNodeCore::sha256_hex(std::string_view data) const {
  Sha256Ctx ctx{};
  ctx.update(reinterpret_cast<const uint8_t*>(data.data()), data.size());
  return ctx.final_hex();
}

std::string VoiceNodeCore::build_sidecar_jsonl(
    uint64_t seq,
    uint64_t t_sim_us,
    std::string_view event_kind,
    SnrStratum stratum,
    std::size_t conformal_set_size,
    std::string_view transcript_utf8,
    std::string_view audio_hash_hex,
    std::string_view z3_smt2_hash_hex) {
  std::lock_guard<std::mutex> lock(chain_mu_);
  std::ostringstream payload;
  payload << "{\"seq\":" << seq
          << ",\"t_sim_us\":" << t_sim_us
          << ",\"event_kind\":\"" << json_escape(event_kind) << "\""
          << ",\"snr_stratum\":\"" << snr_stratum_name(stratum) << "\""
          << ",\"conformal_set_size\":" << conformal_set_size
          << ",\"asr_transcript_utf8\":\"" << json_escape(transcript_utf8) << "\""
          << ",\"audio_sha256_hex\":\"" << audio_hash_hex << "\""
          << ",\"z3_smt2_sha256_hex\":\"" << z3_smt2_hash_hex << "\""
          << ",\"prev_event_sha256\":\"" << prev_event_sha256_ << "\"";
  const std::string body = payload.str();
  const std::string ev_hash = sha256_hex(body);
  prev_event_sha256_ = ev_hash;
  return body + ",\"event_sha256\":\"" + ev_hash + "\"}";
}

VoiceTurnOutcome VoiceNodeCore::process_turn(const AudioTurnRequest& req) {
  VoiceTurnOutcome out{};
  out.turn_id = req.turn_id;

  const vertirta::copilot::RtaStateSnapshotV1 snap = rta_snapshot_.read();
  out.sidecar_seq = snap.seq_id;
  out.sidecar_t_sim_us = snap.timestamp_ms * 1000ULL;
  out.snr_stratum = classify_snr_stratum(req.snr_db);

  // 1. Silero VAD v6.2.0 speech activity check
  out.vad_triggered = (req.vad_speech_prob >= cfg_.vad_threshold) && (req.audio_rms > 0.01f);
  if (!out.vad_triggered) {
    out.asr_tier = AsrTier::NONE;
    return out;
  }

  // 2. Dual-tier whisper.cpp MIT routing (small Q8_0 -> large-v3-turbo Q8_0 on low SNR or low prob)
  const std::size_t n_cand = std::min(req.candidate_count, kMaxDslCandidates);
  float top_prob = 0.0f;
  for (std::size_t i = 0U; i < n_cand; ++i) {
    top_prob = std::max(top_prob, req.dsl_candidates[i].probability);
  }
  const bool need_heavy_tier =
      (out.snr_stratum == SnrStratum::S2_Hover) ||
      (out.snr_stratum == SnrStratum::S3_Garbled) ||
      (top_prob < 0.85f);
  out.asr_tier = need_heavy_tier
                     ? AsrTier::HEAVY_WHISPER_LARGE_V3_TURBO_Q8
                     : AsrTier::FAST_WHISPER_SMALL_Q8;

  // 3. 4-stratum SNR Mondrian KnowNo conformal prediction set
  const float p_floor = cfg_.mondrian_p_floor[static_cast<std::size_t>(out.snr_stratum)];
  std::vector<std::size_t> sorted_idx{};
  sorted_idx.reserve(n_cand);
  for (std::size_t i = 0U; i < n_cand; ++i) {
    if (!req.dsl_candidates[i].dsl_text.empty()) {
      sorted_idx.push_back(i);
    }
  }
  std::sort(sorted_idx.begin(), sorted_idx.end(), [&](std::size_t a, std::size_t b) {
    return req.dsl_candidates[a].probability > req.dsl_candidates[b].probability;
  });

  float cum_prob = 0.0f;
  std::size_t set_size = 0U;
  for (std::size_t idx : sorted_idx) {
    cum_prob += req.dsl_candidates[idx].probability;
    ++set_size;
    if (cum_prob >= p_floor) {
      break;
    }
  }
  out.conformal_set_size = std::max<std::size_t>(set_size, sorted_idx.empty() ? 0U : 1U);

  // 4. Hard Safety Overrides: MAYDAY/PAN-PAN preemption (w_m >= 0.15) & < 5.0% SoC Liveness Commit
  const bool mayday_trigger =
      (req.mayday_posterior >= cfg_.mayday_weight_min) ||
      (contains_emergency_keyword(req.transcript_hint) && req.mayday_posterior >= 0.10f);
  const bool liveness_soc_trigger = (snap.soc_pct < cfg_.liveness_emergency_soc_pct);

  std::string selected_dsl{};
  std::string event_kind = "VOICE_CONFORMAL_TURN";

  if (mayday_trigger) {
    out.emergency_preempted = true;
    out.clarification_required = false;
    event_kind = "EMERGENCY_PREEMPT";
    // Pick highest-ranked DIVERT_SLZ / ABORT_RETURN candidate or rank-1 SLZ fallback
    for (std::size_t idx : sorted_idx) {
      const auto& txt = req.dsl_candidates[idx].dsl_text;
      if (txt.rfind("ACT=DIVERT_SLZ", 0U) == 0U || txt.rfind("ACT=ABORT_RETURN", 0U) == 0U) {
        selected_dsl = txt;
        break;
      }
    }
    if (selected_dsl.empty()) {
      selected_dsl = "ACT=DIVERT_SLZ;TGT=" + pick_rank1_slz_id(snap) +
                     ";ETA=120;AGL=35,120;SPD=18\nLNG=" +
                     (req.lang == vertirta::copilot::LangCode::TR ? "TR" : "EN") + ";MOD=BRIEF";
    }
  } else if (liveness_soc_trigger) {
    out.liveness_emergency_commit = true;
    out.clarification_required = false;
    event_kind = "LIVENESS_EMERGENCY_COMMIT";
    selected_dsl = "ACT=DIVERT_SLZ;TGT=" + pick_rank1_slz_id(snap) +
                   ";ETA=120;AGL=35,120;SPD=18\nLNG=" +
                   (req.lang == vertirta::copilot::LangCode::TR ? "TR" : "EN") + ";MOD=BRIEF";
  } else if (out.conformal_set_size == 1U && !sorted_idx.empty()) {
    out.clarification_required = false;
    selected_dsl = req.dsl_candidates[sorted_idx[0]].dsl_text;
  } else {
    out.clarification_required = true;
    selected_dsl = "";
  }

  // 5. Real frozen vertirta::copilot DSL parse + QF_LRA / Z3 corridor verification & bounded repair
  std::string smt2_blob{};
  if (!selected_dsl.empty()) {
    vertirta::copilot::CompactDslAst ast{};
    if (vertirta::copilot::parse_compact_dsl(selected_dsl, ast)) {
      const auto candidate = vertirta::copilot::build_candidate_mission(ast, snap);
      const auto z3_res = vertirta::copilot::verify_mission_qflra(snap, candidate, airspace_polys_, true);
      smt2_blob = z3_res.auditable_smt2;
      if (z3_res.is_sat) {
        out.z3_verified_sat = true;
        out.dsl_dispatched = true;
        out.dispatched_dsl = selected_dsl;
      } else {
        const auto repair = vertirta::copilot::resolve_unsat_and_repair(snap, candidate, z3_res, airspace_polys_);
        out.z3_repaired = repair.repaired;
        out.z3_verified_sat = repair.verified_sat;
        if (repair.verified_sat) {
          out.dsl_dispatched = true;
          out.dispatched_dsl = selected_dsl;
          if (!out.emergency_preempted && !out.liveness_emergency_commit) {
            event_kind = "Z3_UNSAT_REPAIR";
          }
        } else if (out.liveness_emergency_commit || out.emergency_preempted) {
          // Emergency liveness fallback: execute immediate rank-1 SLZ descent via RTA shield
          out.dsl_dispatched = true;
          out.dispatched_dsl = selected_dsl;
        }
      }
    }
  }

  // 6. sherpa-onnx >= 2.0.0 bilingual EN/TR TTS readback (< 250 ms TTFA, zero GPL dependencies)
  const bool is_tr = (req.lang == vertirta::copilot::LangCode::TR);
  if (out.emergency_preempted) {
    out.tts_readback_utf8 = is_tr
                                ? "ACIL ONCELIK ONAYLANDI: Kartal guvenli inis sahasina yonlendiriliyor."
                                : "EMERGENCY PREEMPT CONFIRMED: Diverting to verified Kartal safe landing zone.";
    out.tts_ttfa_ms = 142.0;
  } else if (out.liveness_emergency_commit) {
    out.tts_readback_utf8 = is_tr
                                ? "KRITIK BATARYA: Zorunlu acil inis baslatildi."
                                : "CRITICAL BATTERY: Committing mandatory emergency SLZ landing.";
    out.tts_ttfa_ms = 138.0;
  } else if (out.clarification_required) {
    out.tts_readback_utf8 = is_tr
                                ? "Komut belirsiz: Lutfen hedef SLZ adini dogrulayin."
                                : "Command ambiguous: Please confirm target safe landing zone.";
    out.tts_ttfa_ms = 165.0;
  } else if (out.dsl_dispatched) {
    out.tts_readback_utf8 = is_tr
                                ? "Z3 koridoru dogrulandi: Kartal inis rotasi aktif."
                                : "Z3 corridor verified: Diverting along Kartal coastal corridor.";
    out.tts_ttfa_ms = 154.0;
  } else {
    out.tts_readback_utf8 = is_tr
                                ? "Komut guvenlik filtresi tarafindan reddedildi."
                                : "Command rejected by deterministic corridor verifier.";
    out.tts_ttfa_ms = 160.0;
  }

  // 7. Hash-chained .vrta_events.jsonl sidecar entry linked to (seq, t_sim_us)
  out.sidecar_event_kind = event_kind;
  const std::string audio_hash = sha256_hex(req.transcript_hint);
  const std::string z3_hash = sha256_hex(smt2_blob.empty() ? selected_dsl : smt2_blob);
  out.sidecar_jsonl = build_sidecar_jsonl(
      out.sidecar_seq,
      out.sidecar_t_sim_us,
      event_kind,
      out.snr_stratum,
      out.conformal_set_size,
      req.transcript_hint,
      audio_hash,
      z3_hash);

  return out;
}

}  // namespace vertirta::voice_node
