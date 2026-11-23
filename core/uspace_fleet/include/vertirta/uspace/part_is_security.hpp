// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace vertirta::uspace {

// Mandatory 40-byte Part-IS (EU) 2023/203 application-layer security header closing the
// OMG DDS Security 1.2 AES-GCM-GMAC replay gap (RFC 9374 DET + monotonic nonce + TAI clock).
struct alignas(8) VertiRtaSecureHeader {
  uint8_t det[16]{};
  uint64_t monotonic_nonce{0ULL};
  uint64_t tai_ns{0ULL};
  uint32_t key_epoch{0U};
  uint32_t flags_and_mode{0U};
};
static_assert(sizeof(VertiRtaSecureHeader) == 40U, "VertiRtaSecureHeader must be exactly 40 bytes");

enum class ReplayStatus : uint8_t {
  ACCEPT = 0,
  REJECT_REPLAY = 1,
  REJECT_TAI_SKEW = 2,
  REJECT_INVALID_NONCE = 3,
};

enum class PartIsBackend : uint8_t {
  LIBSODIUM_AES256GCM_ED25519 = 1,
  SIM_HMAC_SHA256_BACKEND = 2,
};

enum class CyberState : uint8_t {
  SEC_NOMINAL = 0,
  SEC_SUSPECT_PEER = 1,
  SEC_USSP_QUARANTINE = 2,
  SEC_AUTONOMOUS_RTB = 3,
};

// Enforces mandatory CVE pin: libsodium >= 1.0.21 (rejects <= 1.0.20 Ed25519 subgroup defect).
[[nodiscard]] constexpr bool check_libsodium_min_version(
    uint32_t major,
    uint32_t minor,
    uint32_t patch) noexcept {
  if (major != 1U) {
    return major > 1U;
  }
  if (minor != 0U) {
    return minor > 0U;
  }
  return patch >= 21U;
}

// Returns active runtime crypto backend (LIBSODIUM_AES256GCM_ED25519 when built with libsodium >= 1.0.21,
// explicitly labeled SIM_HMAC_SHA256_BACKEND for local builds without libsodium).
[[nodiscard]] PartIsBackend active_part_is_backend() noexcept;

// RFC 6479 O(1) 128-packet sliding anti-replay window + +/-250 ms TAI freshness gate.
class ReplayWindow128 {
 public:
  static constexpr uint64_t kMaxTaiSkewNs = 250'000'000ULL;

  void reset() noexcept;

  [[nodiscard]] ReplayStatus check_and_accept(
      uint64_t nonce,
      uint64_t packet_tai_ns,
      uint64_t now_tai_ns) noexcept;

  [[nodiscard]] uint64_t highest_nonce() const noexcept { return highest_nonce_; }

 private:
  uint64_t highest_nonce_{0ULL};
  uint64_t mask_lo_{0ULL};
  uint64_t mask_hi_{0ULL};
  bool initialized_{false};
};

struct SecuredPacket {
  static constexpr std::size_t kMaxPayloadBytes = 256U;
  static constexpr std::size_t kTagBytes = 32U;
  static constexpr std::size_t kSigBytes = 64U;

  VertiRtaSecureHeader header{};
  std::array<uint8_t, kMaxPayloadBytes> payload{};
  std::size_t payload_len{0U};
  std::array<uint8_t, kTagBytes> auth_tag{};
  std::array<uint8_t, kSigBytes> ed25519_sig{};
  bool has_ed25519_sig{false};
};

// Mode B authenticator/verifier:
//   - 20 Hz V2V kinematics: AEAD AES-256-GCM (or deterministic HMAC-SHA256 fallback)
//   - 1 Hz Remote ID & Priority-0 / OIV alerts: adds Ed25519 detached signature
class PartIsCryptoContext {
 public:
  explicit PartIsCryptoContext(uint32_t key_epoch = 1U) noexcept;

  [[nodiscard]] PartIsBackend backend() const noexcept;
  [[nodiscard]] uint32_t key_epoch() const noexcept { return key_epoch_; }

  [[nodiscard]] bool seal_mode_b_v2v(
      const uint8_t* payload,
      std::size_t len,
      uint64_t nonce,
      uint64_t tai_ns,
      SecuredPacket& out_pkt) const noexcept;

  [[nodiscard]] bool seal_mode_b_priority0_signed(
      const uint8_t* payload,
      std::size_t len,
      uint64_t nonce,
      uint64_t tai_ns,
      SecuredPacket& out_pkt) const noexcept;

  [[nodiscard]] bool verify_mode_b(
      const SecuredPacket& pkt,
      uint64_t now_tai_ns) noexcept;

  void reset_replay_window() noexcept { replay_window_.reset(); }

 private:
  uint32_t key_epoch_{1U};
  std::array<uint8_t, 32U> symmetric_key_{};
  std::array<uint8_t, 32U> sign_pk_{};
  std::array<uint8_t, 64U> sign_sk_{};
  ReplayWindow128 replay_window_{};
};

struct PeerAnomalyAction {
  uint32_t peer_id{0U};
  bool demote_to_non_cooperative{false};
  bool strip_priority0_rights{false};
};

struct UsspCheckAction {
  bool drop_ussp_directives{false};
  bool emit_is_or_230_draft{false};
};

struct RtbAction {
  bool divert_to_nearest_fato_emg{false};
  double elapsed_quarantine_s{0.0};
};

// Part-IS (EU) 2023/203 IS.I.OR.220 / .230 Cyber-Resilience Fail-Safe FSM (RTO <= 10.0 s).
class CyberResilienceFsm {
 public:
  static constexpr double kMaxRtoSeconds = 10.0;
  static constexpr double kMahalanobisThresholdSigma = 3.0;
  static constexpr double kEasternGeofenceLonDeg = 29.1895;
  static constexpr double kMaxCorridorAglM = 120.0;

  void reset() noexcept;

  [[nodiscard]] CyberState state() const noexcept { return state_; }
  [[nodiscard]] uint32_t suspect_peer_id() const noexcept { return suspect_peer_id_; }

  [[nodiscard]] PeerAnomalyAction on_peer_kinematic_anomaly(
      uint32_t peer_id,
      double mahalanobis_sigma) noexcept;

  [[nodiscard]] UsspCheckAction on_ussp_command_check(
      bool sig_valid,
      double cmd_lon_deg,
      double cmd_agl_m,
      double t_s) noexcept;

  [[nodiscard]] RtbAction step_time(
      double t_s,
      bool ussp_auth_restored) noexcept;

 private:
  CyberState state_{CyberState::SEC_NOMINAL};
  uint32_t suspect_peer_id_{0U};
  double quarantine_start_s_{0.0};
  bool quarantine_active_{false};
};

}  // namespace vertirta::uspace
