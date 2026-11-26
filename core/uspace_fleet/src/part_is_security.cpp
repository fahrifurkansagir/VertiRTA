// SPDX-License-Identifier: Apache-2.0
#include "vertirta/uspace/part_is_security.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
#include <sodium.h>
#endif

namespace vertirta::uspace {

namespace {

// Self-contained FIPS 180-4 SHA-256 + RFC 2104 HMAC-SHA256 (used for deterministic
// SIM_HMAC_SHA256_BACKEND in zero-dependency local builds AND for key derivation).
class Sha256 {
 public:
  Sha256() noexcept { reset(); }

  void reset() noexcept {
    h_[0] = 0x6a09e667U;
    h_[1] = 0xbb67ae85U;
    h_[2] = 0x3c6ef372U;
    h_[3] = 0xa54ff53aU;
    h_[4] = 0x510e527fU;
    h_[5] = 0x9b05688cU;
    h_[6] = 0x1f83d9abU;
    h_[7] = 0x5be0cd19U;
    buf_len_ = 0U;
    total_bits_ = 0ULL;
  }

  void update(const uint8_t* data, std::size_t len) noexcept {
    for (std::size_t i = 0U; i < len; ++i) {
      buf_[buf_len_++] = data[i];
      if (buf_len_ == 64U) {
        compress(buf_.data());
        total_bits_ += 512ULL;
        buf_len_ = 0U;
      }
    }
  }

  void finalize(uint8_t digest[32]) noexcept {
    total_bits_ += static_cast<uint64_t>(buf_len_) * 8ULL;
    buf_[buf_len_++] = 0x80U;
    if (buf_len_ > 56U) {
      while (buf_len_ < 64U) {
        buf_[buf_len_++] = 0x00U;
      }
      compress(buf_.data());
      buf_len_ = 0U;
    }
    while (buf_len_ < 56U) {
      buf_[buf_len_++] = 0x00U;
    }
    for (int i = 7; i >= 0; --i) {
      buf_[buf_len_++] = static_cast<uint8_t>((total_bits_ >> (static_cast<unsigned>(i) * 8U)) & 0xFFU);
    }
    compress(buf_.data());
    for (std::size_t i = 0U; i < 8U; ++i) {
      digest[i * 4U + 0U] = static_cast<uint8_t>((h_[i] >> 24U) & 0xFFU);
      digest[i * 4U + 1U] = static_cast<uint8_t>((h_[i] >> 16U) & 0xFFU);
      digest[i * 4U + 2U] = static_cast<uint8_t>((h_[i] >> 8U) & 0xFFU);
      digest[i * 4U + 3U] = static_cast<uint8_t>(h_[i] & 0xFFU);
    }
  }

 private:
  static constexpr uint32_t rotr(uint32_t x, uint32_t n) noexcept {
    return (x >> n) | (x << (32U - n));
  }

  void compress(const uint8_t block[64]) noexcept {
    static constexpr uint32_t kRound[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

    uint32_t w[64]{};
    for (std::size_t i = 0U; i < 16U; ++i) {
      w[i] = (static_cast<uint32_t>(block[i * 4U + 0U]) << 24U) |
             (static_cast<uint32_t>(block[i * 4U + 1U]) << 16U) |
             (static_cast<uint32_t>(block[i * 4U + 2U]) << 8U) |
             static_cast<uint32_t>(block[i * 4U + 3U]);
    }
    for (std::size_t i = 16U; i < 64U; ++i) {
      const uint32_t s0 = rotr(w[i - 15U], 7U) ^ rotr(w[i - 15U], 18U) ^ (w[i - 15U] >> 3U);
      const uint32_t s1 = rotr(w[i - 2U], 17U) ^ rotr(w[i - 2U], 19U) ^ (w[i - 2U] >> 10U);
      w[i] = w[i - 16U] + s0 + w[i - 7U] + s1;
    }

    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], hh = h_[7];

    for (std::size_t i = 0U; i < 64U; ++i) {
      const uint32_t s1 = rotr(e, 6U) ^ rotr(e, 11U) ^ rotr(e, 25U);
      const uint32_t ch = (e & f) ^ ((~e) & g);
      const uint32_t t1 = hh + s1 + ch + kRound[i] + w[i];
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

    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += hh;
  }

  uint32_t h_[8]{};
  std::array<uint8_t, 64U> buf_{};
  std::size_t buf_len_{0U};
  uint64_t total_bits_{0ULL};
};

#if !defined(VERTIRTA_HAS_LIBSODIUM) || !VERTIRTA_HAS_LIBSODIUM
void hmac_sha256(
    const uint8_t key[32],
    const uint8_t* aad,
    std::size_t aad_len,
    const uint8_t* msg,
    std::size_t msg_len,
    uint8_t out[32]) noexcept {
  uint8_t ipad[64]{};
  uint8_t opad[64]{};
  for (std::size_t i = 0U; i < 64U; ++i) {
    const uint8_t kb = (i < 32U) ? key[i] : 0x00U;
    ipad[i] = static_cast<uint8_t>(kb ^ 0x36U);
    opad[i] = static_cast<uint8_t>(kb ^ 0x5CU);
  }
  uint8_t inner[32]{};
  Sha256 s1{};
  s1.update(ipad, 64U);
  s1.update(aad, aad_len);
  s1.update(msg, msg_len);
  s1.finalize(inner);

  Sha256 s2{};
  s2.update(opad, 64U);
  s2.update(inner, 32U);
  s2.finalize(out);
}

[[nodiscard]] bool ct_equal(const uint8_t* a, const uint8_t* b, std::size_t n) noexcept {
  uint8_t diff = 0U;
  for (std::size_t i = 0U; i < n; ++i) {
    diff |= static_cast<uint8_t>(a[i] ^ b[i]);
  }
  return diff == 0U;
}
#else
[[nodiscard]] bool verify_runtime_libsodium_version() noexcept {
  unsigned int maj = 0U, min = 0U, patch = 0U;
  if (std::sscanf(sodium_version_string(), "%u.%u.%u", &maj, &min, &patch) < 3) {
    return false;
  }
  return check_libsodium_min_version(maj, min, patch);
}
#endif

}  // namespace

PartIsBackend active_part_is_backend() noexcept {
#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  return PartIsBackend::LIBSODIUM_AES256GCM_ED25519;
#else
  return PartIsBackend::SIM_HMAC_SHA256_BACKEND;
#endif
}

void ReplayWindow128::reset() noexcept {
  highest_nonce_ = 0ULL;
  mask_lo_ = 0ULL;
  mask_hi_ = 0ULL;
  initialized_ = false;
}

ReplayStatus ReplayWindow128::check_and_accept(
    uint64_t nonce,
    uint64_t packet_tai_ns,
    uint64_t now_tai_ns) noexcept {
  if (nonce == 0ULL) {
    return ReplayStatus::REJECT_INVALID_NONCE;
  }
  const uint64_t skew_ns = (packet_tai_ns >= now_tai_ns)
      ? (packet_tai_ns - now_tai_ns)
      : (now_tai_ns - packet_tai_ns);
  if (skew_ns > kMaxTaiSkewNs) {
    return ReplayStatus::REJECT_TAI_SKEW;
  }

  if (!initialized_) {
    highest_nonce_ = nonce;
    mask_lo_ = 1ULL;
    mask_hi_ = 0ULL;
    initialized_ = true;
    return ReplayStatus::ACCEPT;
  }

  if (nonce > highest_nonce_) {
    const uint64_t shift = nonce - highest_nonce_;
    if (shift >= 128ULL) {
      mask_lo_ = 1ULL;
      mask_hi_ = 0ULL;
    } else if (shift >= 64ULL) {
      mask_hi_ = mask_lo_ << (shift - 64ULL);
      mask_lo_ = 1ULL;
    } else {
      mask_hi_ = (mask_hi_ << shift) | (mask_lo_ >> (64ULL - shift));
      mask_lo_ = (mask_lo_ << shift) | 1ULL;
    }
    highest_nonce_ = nonce;
    return ReplayStatus::ACCEPT;
  }

  const uint64_t diff = highest_nonce_ - nonce;
  if (diff >= 128ULL) {
    return ReplayStatus::REJECT_REPLAY;
  }
  if (diff < 64ULL) {
    const uint64_t bit = 1ULL << diff;
    if ((mask_lo_ & bit) != 0ULL) {
      return ReplayStatus::REJECT_REPLAY;
    }
    mask_lo_ |= bit;
  } else {
    const uint64_t bit = 1ULL << (diff - 64ULL);
    if ((mask_hi_ & bit) != 0ULL) {
      return ReplayStatus::REJECT_REPLAY;
    }
    mask_hi_ |= bit;
  }
  return ReplayStatus::ACCEPT;
}

PartIsCryptoContext::PartIsCryptoContext(uint32_t key_epoch) noexcept : key_epoch_(key_epoch) {
  // Deterministic epoch-bound key material for reproducible SIL/CI verification
  uint8_t seed_msg[24] = {'V', 'E', 'R', 'T', 'I', 'R', 'T', 'A', '-', 'P', 'A', 'R',
                          'T', '-', 'I', 'S', '-', 'M', 'O', 'D', 'E', 'B', 0U, 0U};
  seed_msg[22] = static_cast<uint8_t>((key_epoch >> 8U) & 0xFFU);
  seed_msg[23] = static_cast<uint8_t>(key_epoch & 0xFFU);

  Sha256 h{};
  h.update(seed_msg, sizeof(seed_msg));
  h.finalize(symmetric_key_.data());

  uint8_t sign_seed[32]{};
  seed_msg[0] = 'E';
  Sha256 h2{};
  h2.update(seed_msg, sizeof(seed_msg));
  h2.finalize(sign_seed);

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  if (sodium_init() >= 0 && verify_runtime_libsodium_version()) {
    crypto_sign_ed25519_seed_keypair(sign_pk_.data(), sign_sk_.data(), sign_seed);
  }
#else
  std::memcpy(sign_pk_.data(), sign_seed, 32U);
  std::memcpy(sign_sk_.data(), sign_seed, 32U);
  std::memcpy(sign_sk_.data() + 32U, symmetric_key_.data(), 32U);
#endif
}

PartIsBackend PartIsCryptoContext::backend() const noexcept {
  return active_part_is_backend();
}

bool PartIsCryptoContext::seal_mode_b_v2v(
    const uint8_t* payload,
    std::size_t len,
    uint64_t nonce,
    uint64_t tai_ns,
    SecuredPacket& out_pkt) const noexcept {
  if (payload == nullptr || len == 0U || len > SecuredPacket::kMaxPayloadBytes || nonce == 0ULL) {
    return false;
  }
  out_pkt = SecuredPacket{};
  for (std::size_t i = 0U; i < 16U; ++i) {
    out_pkt.header.det[i] = static_cast<uint8_t>(0x20U ^ symmetric_key_[i]);
  }
  out_pkt.header.monotonic_nonce = nonce;
  out_pkt.header.tai_ns = tai_ns;
  out_pkt.header.key_epoch = key_epoch_;
  out_pkt.header.flags_and_mode = 0x00000001U;  // Mode B V2V AEAD
  out_pkt.payload_len = len;
  out_pkt.has_ed25519_sig = false;

  const auto* aad = reinterpret_cast<const uint8_t*>(&out_pkt.header);

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  if (!verify_runtime_libsodium_version()) {
    return false;
  }
  if (crypto_aead_aes256gcm_is_available() != 0) {
    uint8_t npub[crypto_aead_aes256gcm_NPUBBYTES]{};
    std::memcpy(npub, &nonce, sizeof(nonce));
    std::memcpy(npub + 8U, &key_epoch_, sizeof(key_epoch_));
    unsigned long long mac_len = 0ULL;
    const int rc = crypto_aead_aes256gcm_encrypt_detached(
        out_pkt.payload.data(),
        out_pkt.auth_tag.data(),
        &mac_len,
        payload,
        static_cast<unsigned long long>(len),
        aad,
        sizeof(VertiRtaSecureHeader),
        nullptr,
        npub,
        symmetric_key_.data());
    return rc == 0;
  }
  uint8_t xnpub[crypto_aead_xchacha20poly1305_ietf_NPUBBYTES]{};
  std::memcpy(xnpub, &nonce, sizeof(nonce));
  std::memcpy(xnpub + 8U, &tai_ns, sizeof(tai_ns));
  std::memcpy(xnpub + 16U, &key_epoch_, sizeof(key_epoch_));
  unsigned long long mac_len = 0ULL;
  const int rc = crypto_aead_xchacha20poly1305_ietf_encrypt_detached(
      out_pkt.payload.data(),
      out_pkt.auth_tag.data(),
      &mac_len,
      payload,
      static_cast<unsigned long long>(len),
      aad,
      sizeof(VertiRtaSecureHeader),
      nullptr,
      xnpub,
      symmetric_key_.data());
  return rc == 0;
#else
  std::memcpy(out_pkt.payload.data(), payload, len);
  hmac_sha256(symmetric_key_.data(), aad, sizeof(VertiRtaSecureHeader),
              out_pkt.payload.data(), len, out_pkt.auth_tag.data());
  return true;
#endif
}

bool PartIsCryptoContext::seal_mode_b_priority0_signed(
    const uint8_t* payload,
    std::size_t len,
    uint64_t nonce,
    uint64_t tai_ns,
    SecuredPacket& out_pkt) const noexcept {
  if (!seal_mode_b_v2v(payload, len, nonce, tai_ns, out_pkt)) {
    return false;
  }
  out_pkt.header.flags_and_mode = 0x00000003U;  // Mode B AEAD + Ed25519 Priority-0/RID
  const auto* aad = reinterpret_cast<const uint8_t*>(&out_pkt.header);

  // Re-seal tag over updated flags_and_mode
#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  if (!verify_runtime_libsodium_version()) {
    return false;
  }
  if (crypto_aead_aes256gcm_is_available() != 0) {
    uint8_t npub[crypto_aead_aes256gcm_NPUBBYTES]{};
    std::memcpy(npub, &nonce, sizeof(nonce));
    std::memcpy(npub + 8U, &key_epoch_, sizeof(key_epoch_));
    unsigned long long mac_len = 0ULL;
    if (crypto_aead_aes256gcm_encrypt_detached(
            out_pkt.payload.data(),
            out_pkt.auth_tag.data(),
            &mac_len,
            payload,
            static_cast<unsigned long long>(len),
            aad,
            sizeof(VertiRtaSecureHeader),
            nullptr,
            npub,
            symmetric_key_.data()) != 0) {
      return false;
    }
  } else {
    uint8_t xnpub[crypto_aead_xchacha20poly1305_ietf_NPUBBYTES]{};
    std::memcpy(xnpub, &nonce, sizeof(nonce));
    std::memcpy(xnpub + 8U, &tai_ns, sizeof(tai_ns));
    std::memcpy(xnpub + 16U, &key_epoch_, sizeof(key_epoch_));
    unsigned long long mac_len = 0ULL;
    if (crypto_aead_xchacha20poly1305_ietf_encrypt_detached(
            out_pkt.payload.data(),
            out_pkt.auth_tag.data(),
            &mac_len,
            payload,
            static_cast<unsigned long long>(len),
            aad,
            sizeof(VertiRtaSecureHeader),
            nullptr,
            xnpub,
            symmetric_key_.data()) != 0) {
      return false;
    }
  }
  unsigned long long sig_len = 0ULL;
  uint8_t sig_buf[sizeof(VertiRtaSecureHeader) + SecuredPacket::kTagBytes]{};
  std::memcpy(sig_buf, aad, sizeof(VertiRtaSecureHeader));
  std::memcpy(sig_buf + sizeof(VertiRtaSecureHeader), out_pkt.auth_tag.data(), SecuredPacket::kTagBytes);
  if (crypto_sign_ed25519_detached(
          out_pkt.ed25519_sig.data(), &sig_len, sig_buf, sizeof(sig_buf), sign_sk_.data()) != 0) {
    return false;
  }
  out_pkt.has_ed25519_sig = true;
  return true;
#else
  hmac_sha256(symmetric_key_.data(), aad, sizeof(VertiRtaSecureHeader),
              out_pkt.payload.data(), len, out_pkt.auth_tag.data());
  hmac_sha256(sign_pk_.data(), aad, sizeof(VertiRtaSecureHeader),
              out_pkt.auth_tag.data(), SecuredPacket::kTagBytes, out_pkt.ed25519_sig.data());
  hmac_sha256(sign_sk_.data() + 32U, out_pkt.ed25519_sig.data(), 32U,
              out_pkt.payload.data(), len, out_pkt.ed25519_sig.data() + 32U);
  out_pkt.has_ed25519_sig = true;
  return true;
#endif
}

bool PartIsCryptoContext::verify_mode_b(
    const SecuredPacket& pkt,
    uint64_t now_tai_ns) noexcept {
  if (pkt.payload_len == 0U || pkt.payload_len > SecuredPacket::kMaxPayloadBytes) {
    return false;
  }
  if (pkt.header.key_epoch != key_epoch_) {
    return false;
  }
  // Enforce authenticated header mode bits: bit 0 (0x01U) = Mode B AEAD, bit 1 (0x02U) = Ed25519 signature required.
  // Never trust unauthenticated struct boolean pkt.has_ed25519_sig alone (blocks Mode-B signature downgrade!).
  if ((pkt.header.flags_and_mode & 0x01U) == 0U) {
    return false;
  }
  const bool header_requires_sig = (pkt.header.flags_and_mode & 0x02U) != 0U;
  if (header_requires_sig != pkt.has_ed25519_sig) {
    return false;
  }

  const auto* aad = reinterpret_cast<const uint8_t*>(&pkt.header);
  bool crypto_ok = false;

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  if (!verify_runtime_libsodium_version()) {
    return false;
  }
  std::array<uint8_t, SecuredPacket::kMaxPayloadBytes> plain{};
  if (crypto_aead_aes256gcm_is_available() != 0) {
    uint8_t npub[crypto_aead_aes256gcm_NPUBBYTES]{};
    std::memcpy(npub, &pkt.header.monotonic_nonce, sizeof(pkt.header.monotonic_nonce));
    std::memcpy(npub + 8U, &pkt.header.key_epoch, sizeof(pkt.header.key_epoch));
    crypto_ok = (crypto_aead_aes256gcm_decrypt_detached(
                     plain.data(),
                     nullptr,
                     pkt.payload.data(),
                     static_cast<unsigned long long>(pkt.payload_len),
                     pkt.auth_tag.data(),
                     aad,
                     sizeof(VertiRtaSecureHeader),
                     npub,
                     symmetric_key_.data()) == 0);
  } else {
    uint8_t xnpub[crypto_aead_xchacha20poly1305_ietf_NPUBBYTES]{};
    std::memcpy(xnpub, &pkt.header.monotonic_nonce, sizeof(pkt.header.monotonic_nonce));
    std::memcpy(xnpub + 8U, &pkt.header.tai_ns, sizeof(pkt.header.tai_ns));
    std::memcpy(xnpub + 16U, &pkt.header.key_epoch, sizeof(pkt.header.key_epoch));
    crypto_ok = (crypto_aead_xchacha20poly1305_ietf_decrypt_detached(
                     plain.data(),
                     nullptr,
                     pkt.payload.data(),
                     static_cast<unsigned long long>(pkt.payload_len),
                     pkt.auth_tag.data(),
                     aad,
                     sizeof(VertiRtaSecureHeader),
                     xnpub,
                     symmetric_key_.data()) == 0);
  }
  if (crypto_ok && header_requires_sig) {
    uint8_t sig_buf[sizeof(VertiRtaSecureHeader) + SecuredPacket::kTagBytes]{};
    std::memcpy(sig_buf, aad, sizeof(VertiRtaSecureHeader));
    std::memcpy(sig_buf + sizeof(VertiRtaSecureHeader), pkt.auth_tag.data(), SecuredPacket::kTagBytes);
    crypto_ok = (crypto_sign_ed25519_verify_detached(
                     pkt.ed25519_sig.data(), sig_buf, sizeof(sig_buf), sign_pk_.data()) == 0);
  }
#else
  uint8_t expected_tag[32]{};
  hmac_sha256(symmetric_key_.data(), aad, sizeof(VertiRtaSecureHeader),
              pkt.payload.data(), pkt.payload_len, expected_tag);
  crypto_ok = ct_equal(expected_tag, pkt.auth_tag.data(), 32U);
  if (crypto_ok && header_requires_sig) {
    uint8_t expected_sig[64]{};
    hmac_sha256(sign_pk_.data(), aad, sizeof(VertiRtaSecureHeader),
                pkt.auth_tag.data(), SecuredPacket::kTagBytes, expected_sig);
    hmac_sha256(sign_sk_.data() + 32U, expected_sig, 32U,
                pkt.payload.data(), pkt.payload_len, expected_sig + 32U);
    crypto_ok = ct_equal(expected_sig, pkt.ed25519_sig.data(), 64U);
  }
#endif

  if (!crypto_ok) {
    return false;
  }

  // Only advance replay window after cryptographic authenticity passes!
  return replay_window_.check_and_accept(
             pkt.header.monotonic_nonce, pkt.header.tai_ns, now_tai_ns) == ReplayStatus::ACCEPT;
}

void CyberResilienceFsm::reset() noexcept {
  state_ = CyberState::SEC_NOMINAL;
  suspect_peer_id_ = 0U;
  quarantine_start_s_ = 0.0;
  quarantine_active_ = false;
}

PeerAnomalyAction CyberResilienceFsm::on_peer_kinematic_anomaly(
    uint32_t peer_id,
    double mahalanobis_sigma) noexcept {
  PeerAnomalyAction act{};
  act.peer_id = peer_id;
  if (!std::isfinite(mahalanobis_sigma) || mahalanobis_sigma < 0.0 ||
      mahalanobis_sigma > kMahalanobisThresholdSigma) {
    suspect_peer_id_ = peer_id;
    act.demote_to_non_cooperative = true;
    act.strip_priority0_rights = true;
    if (state_ == CyberState::SEC_NOMINAL) {
      state_ = CyberState::SEC_SUSPECT_PEER;
    }
  }
  return act;
}

UsspCheckAction CyberResilienceFsm::on_ussp_command_check(
    bool sig_valid,
    double cmd_lon_deg,
    double cmd_agl_m,
    double t_s) noexcept {
  UsspCheckAction act{};
  const bool non_finite_input =
      (!std::isfinite(cmd_lon_deg)) || (!std::isfinite(cmd_agl_m)) || (!std::isfinite(t_s));
  const bool semantic_violation =
      (!sig_valid) || non_finite_input ||
      (cmd_lon_deg < -180.0) || (cmd_lon_deg > 180.0) ||
      (cmd_lon_deg >= kEasternGeofenceLonDeg) ||
      (cmd_agl_m > kMaxCorridorAglM) || (cmd_agl_m < 0.0);
  if (semantic_violation) {
    if (!quarantine_active_) {
      quarantine_start_s_ = std::isfinite(t_s) ? t_s : 0.0;
      quarantine_active_ = true;
    }
    state_ = CyberState::SEC_USSP_QUARANTINE;
    act.drop_ussp_directives = true;
    act.emit_is_or_230_draft = true;
  }
  return act;
}

RtbAction CyberResilienceFsm::step_time(double t_s, bool ussp_auth_restored) noexcept {
  RtbAction act{};
  if (state_ == CyberState::SEC_USSP_QUARANTINE && quarantine_active_) {
    if (!std::isfinite(t_s)) {
      state_ = CyberState::SEC_AUTONOMOUS_RTB;
      act.elapsed_quarantine_s = kMaxRtoSeconds;
      act.divert_to_nearest_fato_emg = true;
      return act;
    }
    if (ussp_auth_restored) {
      state_ = CyberState::SEC_NOMINAL;
      quarantine_active_ = false;
      return act;
    }
    act.elapsed_quarantine_s = std::max(0.0, t_s - quarantine_start_s_);
    if (act.elapsed_quarantine_s >= kMaxRtoSeconds) {
      state_ = CyberState::SEC_AUTONOMOUS_RTB;
      act.divert_to_nearest_fato_emg = true;
    }
  } else if (state_ == CyberState::SEC_AUTONOMOUS_RTB) {
    act.divert_to_nearest_fato_emg = true;
    act.elapsed_quarantine_s = std::isfinite(t_s)
                                   ? std::max(kMaxRtoSeconds, t_s - quarantine_start_s_)
                                   : kMaxRtoSeconds;
  }
  return act;
}

}  // namespace vertirta::uspace
