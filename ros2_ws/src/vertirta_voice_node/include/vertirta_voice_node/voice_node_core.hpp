// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

#include "vertirta/copilot/rta_state_contract.hpp"
#include "vertirta/copilot/z3_corridor_verifier.hpp"

namespace vertirta::voice_node {

// Canonical M3 Voice Node ROS 2 topics (Plan 6B Task 5)
inline constexpr const char* kTopicAudio16kHz = "/vertirta/voice/audio_16khz";
inline constexpr const char* kTopicRtaStateSnapshot = "/vertirta/rta/state_snapshot";
inline constexpr const char* kTopicVoiceDslCommand = "/vertirta/voice/dsl_command";
inline constexpr const char* kTopicVoiceTtsReadback = "/vertirta/voice/tts_readback";
inline constexpr const char* kTopicForensicsSidecarEvent = "/vertirta/forensics/sidecar_event";

inline constexpr std::size_t kVoiceQueueCapacity = 16U;
inline constexpr std::size_t kMaxDslCandidates = 4U;

enum class SnrStratum : uint8_t {
  S0_Clean = 0U,
  S1_Cruise = 1U,
  S2_Hover = 2U,
  S3_Garbled = 3U,
};

enum class AsrTier : uint8_t {
  NONE = 0U,
  FAST_WHISPER_SMALL_Q8 = 1U,
  HEAVY_WHISPER_LARGE_V3_TURBO_Q8 = 2U,
};

[[nodiscard]] constexpr const char* snr_stratum_name(SnrStratum s) noexcept {
  switch (s) {
    case SnrStratum::S0_Clean:
      return "S0_Clean";
    case SnrStratum::S1_Cruise:
      return "S1_Cruise";
    case SnrStratum::S2_Hover:
      return "S2_Hover";
    case SnrStratum::S3_Garbled:
      return "S3_Garbled";
  }
  return "S3_Garbled";
}

[[nodiscard]] constexpr SnrStratum classify_snr_stratum(float snr_db) noexcept {
  if (snr_db >= 20.0f) {
    return SnrStratum::S0_Clean;
  }
  if (snr_db >= 10.0f) {
    return SnrStratum::S1_Cruise;
  }
  if (snr_db >= 0.0f) {
    return SnrStratum::S2_Hover;
  }
  return SnrStratum::S3_Garbled;
}

struct VoiceNodeConfig {
  float vad_threshold{0.50f};
  float mayday_weight_min{0.15f};
  float liveness_emergency_soc_pct{5.0f};
  double tts_ttfa_budget_ms{250.0};
  int asr_audio_ctx{256};
  bool asr_translate{false};
  bool asr_no_context{true};
  bool asr_carry_initial_prompt{true};
  // Per-stratum Mondrian calibrated probability thresholds (1 - q_hat)
  std::array<float, 4> mondrian_p_floor{0.82f, 0.85f, 0.88f, 0.92f};
};

struct DslCandidateSpec {
  std::string dsl_text{};
  float probability{0.0f};
};

struct AudioTurnRequest {
  uint32_t turn_id{0U};
  float snr_db{15.0f};
  float vad_speech_prob{0.0f};
  float audio_rms{0.0f};
  float mayday_posterior{0.0f};
  vertirta::copilot::LangCode lang{vertirta::copilot::LangCode::TR};
  std::string transcript_hint{};
  std::array<DslCandidateSpec, kMaxDslCandidates> dsl_candidates{};
  std::size_t candidate_count{0U};
};

struct VoiceTurnOutcome {
  uint32_t turn_id{0U};
  bool vad_triggered{false};
  SnrStratum snr_stratum{SnrStratum::S0_Clean};
  AsrTier asr_tier{AsrTier::NONE};
  std::size_t conformal_set_size{0U};
  bool emergency_preempted{false};
  bool liveness_emergency_commit{false};
  bool clarification_required{false};
  bool z3_verified_sat{false};
  bool z3_repaired{false};
  bool dsl_dispatched{false};
  double tts_ttfa_ms{0.0};
  std::string dispatched_dsl{};
  std::string tts_readback_utf8{};
  uint64_t sidecar_seq{0ULL};
  uint64_t sidecar_t_sim_us{0ULL};
  std::string sidecar_event_kind{};
  std::string sidecar_jsonl{};
};

// Zero-heap, TSan-clean seqlock isolating 100 Hz RTA snapshot writes from async voice turns.
template <typename T>
class alignas(64) LockFreeRtaSeqlock {
  static_assert(std::is_trivially_copyable_v<T>, "Seqlock requires trivially copyable POD");
  static_assert(std::atomic<uint64_t>::is_always_lock_free, "64-bit atomics must be lock-free");

 public:
  static constexpr std::size_t kWords = (sizeof(T) + sizeof(uint64_t) - 1U) / sizeof(uint64_t);

  LockFreeRtaSeqlock() noexcept {
    for (auto& w : words_) {
      w.store(0ULL, std::memory_order_relaxed);
    }
  }

  void write(const T& value) noexcept {
    std::array<uint64_t, kWords> buf{};
    std::memcpy(static_cast<void*>(buf.data()), static_cast<const void*>(&value), sizeof(T));
    const uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1ULL, std::memory_order_release);
    for (std::size_t i = 0U; i < kWords; ++i) {
      words_[i].store(buf[i], std::memory_order_release);
    }
    seq_.store(s + 2ULL, std::memory_order_release);
  }

  [[nodiscard]] T read() const noexcept {
    std::array<uint64_t, kWords> buf{};
    while (true) {
      const uint64_t s1 = seq_.load(std::memory_order_acquire);
      if ((s1 & 1ULL) != 0ULL) {
        continue;
      }
      for (std::size_t i = 0U; i < kWords; ++i) {
        buf[i] = words_[i].load(std::memory_order_acquire);
      }
      const uint64_t s2 = seq_.load(std::memory_order_acquire);
      if (s1 == s2) {
        T out{};
        std::memcpy(static_cast<void*>(&out), static_cast<const void*>(buf.data()), sizeof(T));
        return out;
      }
    }
  }

 private:
  alignas(64) std::atomic<uint64_t> seq_{0ULL};
  alignas(64) std::array<std::atomic<uint64_t>, kWords> words_{};
};

// Bounded single-producer single-consumer ring buffer for non-blocking turn submission.
template <typename T, std::size_t Capacity>
class BoundedSpscQueue {
  static_assert((Capacity & (Capacity - 1U)) == 0U && Capacity >= 2U, "Capacity must be power of 2");

 public:
  [[nodiscard]] bool try_push(T item) {
    const std::size_t h = head_.load(std::memory_order_relaxed);
    const std::size_t next_h = (h + 1U) & (Capacity - 1U);
    if (next_h == tail_.load(std::memory_order_acquire)) {
      return false;
    }
    slots_[h] = std::move(item);
    head_.store(next_h, std::memory_order_release);
    return true;
  }

  [[nodiscard]] bool try_pop(T& out) {
    const std::size_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) {
      return false;
    }
    out = std::move(slots_[t]);
    tail_.store((t + 1U) & (Capacity - 1U), std::memory_order_release);
    return true;
  }

 private:
  alignas(64) std::atomic<std::size_t> head_{0U};
  alignas(64) std::atomic<std::size_t> tail_{0U};
  std::array<T, Capacity> slots_{};
};

class VoiceNodeCore {
 public:
  explicit VoiceNodeCore(VoiceNodeConfig cfg = {});
  ~VoiceNodeCore();

  VoiceNodeCore(const VoiceNodeCore&) = delete;
  VoiceNodeCore& operator=(const VoiceNodeCore&) = delete;

  // Wait-free, zero-heap ingress safe to call directly from 100 Hz RTA callback thread.
  void submit_rta_snapshot(const vertirta::copilot::RtaStateSnapshotV1& snapshot) noexcept;

  // Non-blocking bounded enqueue from audio ingress thread.
  [[nodiscard]] bool try_enqueue_audio_turn(AudioTurnRequest req);
  [[nodiscard]] bool try_pop_outcome(VoiceTurnOutcome& out);

  // Synchronous processing entry point (also invoked by the decoupled worker thread).
  [[nodiscard]] VoiceTurnOutcome process_turn(const AudioTurnRequest& req);

  void start_worker();
  void stop_worker();

  [[nodiscard]] const VoiceNodeConfig& config() const noexcept { return cfg_; }

 private:
  void worker_loop();
  [[nodiscard]] std::string sha256_hex(std::string_view data) const;
  [[nodiscard]] std::string build_sidecar_jsonl(
      uint64_t seq,
      uint64_t t_sim_us,
      std::string_view event_kind,
      SnrStratum stratum,
      std::size_t conformal_set_size,
      std::string_view transcript_utf8,
      std::string_view audio_hash_hex,
      std::string_view z3_smt2_hash_hex);

  VoiceNodeConfig cfg_{};
  vertirta::copilot::AirspacePolys airspace_polys_{};
  LockFreeRtaSeqlock<vertirta::copilot::RtaStateSnapshotV1> rta_snapshot_{};
  std::atomic<bool> has_snapshot_{false};

  BoundedSpscQueue<AudioTurnRequest, kVoiceQueueCapacity> req_queue_{};
  BoundedSpscQueue<VoiceTurnOutcome, kVoiceQueueCapacity> out_queue_{};

  std::mutex chain_mu_{};
  std::string prev_event_sha256_{};

  std::mutex wake_mu_{};
  std::condition_variable wake_cv_{};
  std::atomic<bool> worker_running_{false};
  std::thread worker_{};
};

}  // namespace vertirta::voice_node
