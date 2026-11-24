// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "vertirta/core/vrta_recorder.hpp"

namespace vertirta::forensics {

enum class SidecarSignerBackend : uint8_t {
  LIBSODIUM_ED25519 = 0U,
  SIM_SHA256_TEST_SIGNER = 1U,
};

enum class SidecarEventKind : uint8_t {
  VOICE_CONFORMAL_TURN = 0U,
  Z3_UNSAT_REPAIR = 1U,
  RTA_ENVELOPE_PROTECTION = 2U,
  MOTOR_FDI_FAULT = 3U,
  GNSS_SPOOF_ALERT = 4U,
  USPACE_PREEMPTION = 5U,
  PART_IS_CYBER_ANOMALY = 6U,
  CRITICAL_ENERGY_RESERVE = 7U,
};

struct SidecarEvent {
  uint64_t seq{0U};
  uint64_t t_sim_us{0U};
  SidecarEventKind kind{SidecarEventKind::VOICE_CONFORMAL_TURN};
  std::string snr_stratum{"S0_Clean"};
  uint32_t conformal_set_size{1U};
  std::string asr_transcript_utf8{};
  std::string audio_sha256_hex{};
  std::string z3_smt2_sha256_hex{};
  std::vector<std::string> z3_unsat_core{};
  std::string gsn_node_id{};

  // Cryptographic chaining fields populated by VrtaEventsSidecarWriter
  std::string prev_event_sha256{};
  std::string event_sha256{};
  std::string signature_hex{};
  std::string signer_backend{};
};

struct SidecarChainVerification {
  bool valid{false};
  std::vector<SidecarEvent> events{};
  std::string error_message{};
};

struct SidecarFrameJoinResult {
  bool valid{false};
  std::size_t matched_events{0U};
  std::string error_message{};
};

[[nodiscard]] SidecarSignerBackend active_sidecar_signer_backend() noexcept;
[[nodiscard]] std::string to_string(SidecarSignerBackend backend);
[[nodiscard]] std::string to_string(SidecarEventKind kind);
[[nodiscard]] bool parse_sidecar_event_kind(const std::string& s, SidecarEventKind& out) noexcept;
[[nodiscard]] bool check_forensics_libsodium_min_version(
    unsigned int major,
    unsigned int minor,
    unsigned int patch) noexcept;
[[nodiscard]] std::string sha256_hex_bytes(const void* data, std::size_t len);
[[nodiscard]] std::string sha256_hex_string(const std::string& data);

// EUROCAE ED-155A append-only .vrta_events.jsonl sidecar writer linked by (seq, t_sim_us).
// Never modifies frozen 128 B VrtaFileHeader or 256 B VrtaFrame100Hz binary schemas.
class VrtaEventsSidecarWriter {
 public:
  explicit VrtaEventsSidecarWriter(uint64_t deterministic_key_seed = 0x56525441464F5231ULL);

  [[nodiscard]] std::string public_key_hex() const;
  [[nodiscard]] std::string last_event_sha256() const;
  [[nodiscard]] SidecarSignerBackend backend() const noexcept;

  [[nodiscard]] std::string append_serialized(const SidecarEvent& event);
  [[nodiscard]] bool append_to_file(
      const std::filesystem::path& sidecar_path,
      const SidecarEvent& event,
      bool fsync_after_write = true);

 private:
  std::string prev_hash_hex_{};
  std::string pub_key_hex_{};
  std::vector<uint8_t> secret_key_{};
  std::vector<uint8_t> public_key_{};
  SidecarSignerBackend backend_{SidecarSignerBackend::SIM_SHA256_TEST_SIGNER};
};

class VrtaEventsSidecarReader {
 public:
  [[nodiscard]] static SidecarChainVerification verify_chain(
      const std::string& ndjson_text,
      const std::string& public_key_hex);

  [[nodiscard]] static SidecarChainVerification verify_file(
      const std::filesystem::path& sidecar_path,
      const std::string& public_key_hex);

  [[nodiscard]] static SidecarFrameJoinResult join_with_vrta_frames(
      const std::vector<vertirta::core::VrtaFrame100Hz>& frames,
      const std::vector<SidecarEvent>& events);
};

}  // namespace vertirta::forensics
