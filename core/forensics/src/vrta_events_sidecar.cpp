// SPDX-License-Identifier: Apache-2.0
#include "vertirta/forensics/vrta_events_sidecar.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <unordered_map>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
#include <sodium.h>
#endif

namespace vertirta::forensics {

namespace {

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
      buf_[buf_len_++] =
          static_cast<uint8_t>((total_bits_ >> (static_cast<unsigned>(i) * 8U)) & 0xFFU);
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
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
        0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
        0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
        0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
        0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
        0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
        0xc67178f2U};

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

    uint32_t a = h_[0];
    uint32_t b = h_[1];
    uint32_t c = h_[2];
    uint32_t d = h_[3];
    uint32_t e = h_[4];
    uint32_t f = h_[5];
    uint32_t g = h_[6];
    uint32_t hh = h_[7];

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

std::string bytes_to_hex(const uint8_t* data, std::size_t len) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(len * 2U);
  for (std::size_t i = 0U; i < len; ++i) {
    out.push_back(kHex[(data[i] >> 4U) & 0x0FU]);
    out.push_back(kHex[data[i] & 0x0FU]);
  }
  return out;
}

bool hex_to_bytes(const std::string& hex, std::vector<uint8_t>& out) {
  if ((hex.size() % 2U) != 0U) {
    return false;
  }
  out.clear();
  out.reserve(hex.size() / 2U);
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return 10 + (c - 'a');
    }
    if (c >= 'A' && c <= 'F') {
      return 10 + (c - 'A');
    }
    return -1;
  };
  for (std::size_t i = 0U; i < hex.size(); i += 2U) {
    const int hi = nibble(hex[i]);
    const int lo = nibble(hex[i + 1U]);
    if (hi < 0 || lo < 0) {
      return false;
    }
    out.push_back(static_cast<uint8_t>((hi << 4) | lo));
  }
  return true;
}

void hmac_sha256_32(
    const uint8_t key[32],
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
  s1.update(msg, msg_len);
  s1.finalize(inner);

  Sha256 s2{};
  s2.update(opad, 64U);
  s2.update(inner, 32U);
  s2.finalize(out);
}

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
bool verify_runtime_libsodium_ge_1_0_21() noexcept {
  unsigned int maj = 0U;
  unsigned int min = 0U;
  unsigned int patch = 0U;
  if (std::sscanf(sodium_version_string(), "%u.%u.%u", &maj, &min, &patch) < 3) {
    return false;
  }
  return check_forensics_libsodium_min_version(maj, min, patch);
}
#endif

std::string json_escape(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 8U);
  for (const unsigned char ch : raw) {
    switch (ch) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (ch < 0x20U) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(ch));
          out += buf;
        } else {
          out.push_back(static_cast<char>(ch));
        }
        break;
    }
  }
  return out;
}

std::string build_canonical_payload_json(const SidecarEvent& ev) {
  std::ostringstream oss;
  oss << "{\"seq\":" << ev.seq
      << ",\"t_sim_us\":" << ev.t_sim_us
      << ",\"kind\":\"" << to_string(ev.kind) << "\""
      << ",\"snr_stratum\":\"" << json_escape(ev.snr_stratum) << "\""
      << ",\"conformal_set_size\":" << ev.conformal_set_size
      << ",\"asr_transcript_utf8\":\"" << json_escape(ev.asr_transcript_utf8) << "\""
      << ",\"audio_sha256_hex\":\"" << json_escape(ev.audio_sha256_hex) << "\""
      << ",\"z3_smt2_sha256_hex\":\"" << json_escape(ev.z3_smt2_sha256_hex) << "\""
      << ",\"z3_unsat_core\":[";
  for (std::size_t i = 0U; i < ev.z3_unsat_core.size(); ++i) {
    if (i > 0U) {
      oss << ",";
    }
    oss << "\"" << json_escape(ev.z3_unsat_core[i]) << "\"";
  }
  oss << "],\"gsn_node_id\":\"" << json_escape(ev.gsn_node_id) << "\""
      << ",\"prev_event_sha256\":\"" << json_escape(ev.prev_event_sha256) << "\"}";
  return oss.str();
}

// Strict minimal flat-object JSON parser tailored to SidecarEvent NDJSON lines.
class FlatJsonParser {
 public:
  explicit FlatJsonParser(const std::string& text) : s_(text) {}

  bool parse_event(SidecarEvent& out, std::string& err) {
    skip_ws();
    if (!consume('{')) {
      err = "expected '{'";
      return false;
    }
    skip_ws();
    bool first = true;
    while (pos_ < s_.size() && s_[pos_] != '}') {
      if (!first) {
        if (!consume(',')) {
          err = "expected ','";
          return false;
        }
      }
      first = false;
      skip_ws();
      std::string key;
      if (!parse_string(key, err)) {
        return false;
      }
      skip_ws();
      if (!consume(':')) {
        err = "expected ':'";
        return false;
      }
      skip_ws();
      if (key == "seq") {
        if (!parse_uint64(out.seq, err)) {
          return false;
        }
      } else if (key == "t_sim_us") {
        if (!parse_uint64(out.t_sim_us, err)) {
          return false;
        }
      } else if (key == "conformal_set_size") {
        uint64_t val = 0U;
        if (!parse_uint64(val, err)) {
          return false;
        }
        out.conformal_set_size = static_cast<uint32_t>(val);
      } else if (key == "kind") {
        std::string ks;
        if (!parse_string(ks, err) || !parse_sidecar_event_kind(ks, out.kind)) {
          err = "invalid kind";
          return false;
        }
      } else if (key == "snr_stratum") {
        if (!parse_string(out.snr_stratum, err)) {
          return false;
        }
      } else if (key == "asr_transcript_utf8") {
        if (!parse_string(out.asr_transcript_utf8, err)) {
          return false;
        }
      } else if (key == "audio_sha256_hex") {
        if (!parse_string(out.audio_sha256_hex, err)) {
          return false;
        }
      } else if (key == "z3_smt2_sha256_hex") {
        if (!parse_string(out.z3_smt2_sha256_hex, err)) {
          return false;
        }
      } else if (key == "gsn_node_id") {
        if (!parse_string(out.gsn_node_id, err)) {
          return false;
        }
      } else if (key == "prev_event_sha256") {
        if (!parse_string(out.prev_event_sha256, err)) {
          return false;
        }
      } else if (key == "event_sha256") {
        if (!parse_string(out.event_sha256, err)) {
          return false;
        }
      } else if (key == "signature_hex") {
        if (!parse_string(out.signature_hex, err)) {
          return false;
        }
      } else if (key == "signer_backend") {
        if (!parse_string(out.signer_backend, err)) {
          return false;
        }
      } else if (key == "z3_unsat_core") {
        if (!parse_string_array(out.z3_unsat_core, err)) {
          return false;
        }
      } else {
        err = "unexpected field: " + key;
        return false;
      }
      skip_ws();
    }
    if (!consume('}')) {
      err = "expected '}'";
      return false;
    }
    skip_ws();
    if (pos_ != s_.size()) {
      err = "trailing characters after JSON object";
      return false;
    }
    return true;
  }

 private:
  void skip_ws() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_])) != 0) {
      ++pos_;
    }
  }

  bool consume(char expected) {
    if (pos_ < s_.size() && s_[pos_] == expected) {
      ++pos_;
      return true;
    }
    return false;
  }

  bool parse_uint64(uint64_t& val, std::string& err) {
    if (pos_ >= s_.size() || std::isdigit(static_cast<unsigned char>(s_[pos_])) == 0) {
      err = "expected unsigned integer";
      return false;
    }
    uint64_t acc = 0U;
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_])) != 0) {
      const uint64_t digit = static_cast<uint64_t>(s_[pos_] - '0');
      acc = acc * 10ULL + digit;
      ++pos_;
    }
    val = acc;
    return true;
  }

  bool parse_string(std::string& out, std::string& err) {
    if (!consume('"')) {
      err = "expected '\"'";
      return false;
    }
    out.clear();
    while (pos_ < s_.size()) {
      const char ch = s_[pos_++];
      if (ch == '"') {
        return true;
      }
      if (ch == '\\') {
        if (pos_ >= s_.size()) {
          err = "unterminated escape";
          return false;
        }
        const char esc = s_[pos_++];
        switch (esc) {
          case '"':
            out.push_back('"');
            break;
          case '\\':
            out.push_back('\\');
            break;
          case '/':
            out.push_back('/');
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          default:
            err = "unsupported escape";
            return false;
        }
      } else {
        out.push_back(ch);
      }
    }
    err = "unterminated string";
    return false;
  }

  bool parse_string_array(std::vector<std::string>& out, std::string& err) {
    if (!consume('[')) {
      err = "expected '['";
      return false;
    }
    out.clear();
    skip_ws();
    bool first = true;
    while (pos_ < s_.size() && s_[pos_] != ']') {
      if (!first && !consume(',')) {
        err = "expected ',' in array";
        return false;
      }
      first = false;
      skip_ws();
      std::string elem;
      if (!parse_string(elem, err)) {
        return false;
      }
      out.push_back(elem);
      skip_ws();
    }
    if (!consume(']')) {
      err = "expected ']'";
      return false;
    }
    return true;
  }

  const std::string& s_;
  std::size_t pos_{0U};
};

}  // namespace

bool check_forensics_libsodium_min_version(
    unsigned int major,
    unsigned int minor,
    unsigned int patch) noexcept {
  if (major != 1U) {
    return major > 1U;
  }
  if (minor != 0U) {
    return minor > 0U;
  }
  return patch >= 21U;
}

SidecarSignerBackend active_sidecar_signer_backend() noexcept {
#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  return SidecarSignerBackend::LIBSODIUM_ED25519;
#else
  return SidecarSignerBackend::SIM_SHA256_TEST_SIGNER;
#endif
}

std::string to_string(SidecarSignerBackend backend) {
  switch (backend) {
    case SidecarSignerBackend::LIBSODIUM_ED25519:
      return "LIBSODIUM_ED25519";
    case SidecarSignerBackend::SIM_SHA256_TEST_SIGNER:
      return "SIM_SHA256_TEST_SIGNER";
  }
  return "SIM_SHA256_TEST_SIGNER";
}

std::string to_string(SidecarEventKind kind) {
  switch (kind) {
    case SidecarEventKind::VOICE_CONFORMAL_TURN:
      return "VOICE_CONFORMAL_TURN";
    case SidecarEventKind::Z3_UNSAT_REPAIR:
      return "Z3_UNSAT_REPAIR";
    case SidecarEventKind::RTA_ENVELOPE_PROTECTION:
      return "RTA_ENVELOPE_PROTECTION";
    case SidecarEventKind::MOTOR_FDI_FAULT:
      return "MOTOR_FDI_FAULT";
    case SidecarEventKind::GNSS_SPOOF_ALERT:
      return "GNSS_SPOOF_ALERT";
    case SidecarEventKind::USPACE_PREEMPTION:
      return "USPACE_PREEMPTION";
    case SidecarEventKind::PART_IS_CYBER_ANOMALY:
      return "PART_IS_CYBER_ANOMALY";
    case SidecarEventKind::CRITICAL_ENERGY_RESERVE:
      return "CRITICAL_ENERGY_RESERVE";
  }
  return "VOICE_CONFORMAL_TURN";
}

bool parse_sidecar_event_kind(const std::string& s, SidecarEventKind& out) noexcept {
  if (s == "VOICE_CONFORMAL_TURN") {
    out = SidecarEventKind::VOICE_CONFORMAL_TURN;
    return true;
  }
  if (s == "Z3_UNSAT_REPAIR") {
    out = SidecarEventKind::Z3_UNSAT_REPAIR;
    return true;
  }
  if (s == "RTA_ENVELOPE_PROTECTION") {
    out = SidecarEventKind::RTA_ENVELOPE_PROTECTION;
    return true;
  }
  if (s == "MOTOR_FDI_FAULT") {
    out = SidecarEventKind::MOTOR_FDI_FAULT;
    return true;
  }
  if (s == "GNSS_SPOOF_ALERT") {
    out = SidecarEventKind::GNSS_SPOOF_ALERT;
    return true;
  }
  if (s == "USPACE_PREEMPTION") {
    out = SidecarEventKind::USPACE_PREEMPTION;
    return true;
  }
  if (s == "PART_IS_CYBER_ANOMALY") {
    out = SidecarEventKind::PART_IS_CYBER_ANOMALY;
    return true;
  }
  if (s == "CRITICAL_ENERGY_RESERVE") {
    out = SidecarEventKind::CRITICAL_ENERGY_RESERVE;
    return true;
  }
  return false;
}

std::string sha256_hex_bytes(const void* data, std::size_t len) {
  Sha256 sha{};
  sha.update(static_cast<const uint8_t*>(data), len);
  uint8_t digest[32]{};
  sha.finalize(digest);
  return bytes_to_hex(digest, 32U);
}

std::string sha256_hex_string(const std::string& data) {
  return sha256_hex_bytes(data.data(), data.size());
}

VrtaEventsSidecarWriter::VrtaEventsSidecarWriter(uint64_t deterministic_key_seed)
    : prev_hash_hex_(64U, '0'), backend_(active_sidecar_signer_backend()) {
  uint8_t seed_bytes[32]{};
  {
    Sha256 sha{};
    sha.update(reinterpret_cast<const uint8_t*>(&deterministic_key_seed), sizeof(deterministic_key_seed));
    const char domain_tag[] = "VERTIRTA_ED155A_SIDECAR_KEY_V1";
    sha.update(reinterpret_cast<const uint8_t*>(domain_tag), sizeof(domain_tag) - 1U);
    sha.finalize(seed_bytes);
  }

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  if (sodium_init() >= 0 && verify_runtime_libsodium_ge_1_0_21()) {
    public_key_.resize(crypto_sign_PUBLICKEYBYTES);
    secret_key_.resize(crypto_sign_SECRETKEYBYTES);
    crypto_sign_seed_keypair(public_key_.data(), secret_key_.data(), seed_bytes);
    pub_key_hex_ = bytes_to_hex(public_key_.data(), public_key_.size());
    backend_ = SidecarSignerBackend::LIBSODIUM_ED25519;
    return;
  }
#endif
  // Explicit deterministic test-only HMAC-SHA256 signer when native libsodium >= 1.0.21 is absent.
  backend_ = SidecarSignerBackend::SIM_SHA256_TEST_SIGNER;
  public_key_.assign(seed_bytes, seed_bytes + 32U);
  secret_key_.assign(seed_bytes, seed_bytes + 32U);
  pub_key_hex_ = bytes_to_hex(public_key_.data(), public_key_.size());
}

std::string VrtaEventsSidecarWriter::public_key_hex() const { return pub_key_hex_; }

std::string VrtaEventsSidecarWriter::last_event_sha256() const { return prev_hash_hex_; }

SidecarSignerBackend VrtaEventsSidecarWriter::backend() const noexcept { return backend_; }

std::string VrtaEventsSidecarWriter::append_serialized(const SidecarEvent& event) {
  SidecarEvent sealed = event;
  sealed.prev_event_sha256 = prev_hash_hex_;
  const std::string canonical_payload = build_canonical_payload_json(sealed);
  sealed.event_sha256 = sha256_hex_string(canonical_payload);
  sealed.signer_backend = to_string(backend_);

#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
  if (backend_ == SidecarSignerBackend::LIBSODIUM_ED25519) {
    uint8_t sig[crypto_sign_BYTES]{};
    unsigned long long sig_len = 0ULL;
    crypto_sign_detached(
        sig,
        &sig_len,
        reinterpret_cast<const unsigned char*>(sealed.event_sha256.data()),
        static_cast<unsigned long long>(sealed.event_sha256.size()),
        secret_key_.data());
    sealed.signature_hex = bytes_to_hex(sig, static_cast<std::size_t>(sig_len));
  } else {
    uint8_t mac[32]{};
    hmac_sha256_32(
        secret_key_.data(),
        reinterpret_cast<const uint8_t*>(sealed.event_sha256.data()),
        sealed.event_sha256.size(),
        mac);
    sealed.signature_hex = bytes_to_hex(mac, 32U);
  }
#else
  uint8_t mac[32]{};
  hmac_sha256_32(
      secret_key_.data(),
      reinterpret_cast<const uint8_t*>(sealed.event_sha256.data()),
      sealed.event_sha256.size(),
      mac);
  sealed.signature_hex = bytes_to_hex(mac, 32U);
#endif

  prev_hash_hex_ = sealed.event_sha256;

  // Append cryptographic seal fields onto canonical payload object.
  std::string line = canonical_payload;
  line.pop_back();  // strip trailing '}'
  line += ",\"event_sha256\":\"" + sealed.event_sha256 + "\"";
  line += ",\"signature_hex\":\"" + sealed.signature_hex + "\"";
  line += ",\"signer_backend\":\"" + sealed.signer_backend + "\"}";
  return line;
}

bool VrtaEventsSidecarWriter::append_to_file(
    const std::filesystem::path& sidecar_path,
    const SidecarEvent& event,
    bool fsync_after_write) {
  const std::string line = append_serialized(event) + "\n";
#if defined(__unix__) || defined(__APPLE__)
  const int fd = ::open(sidecar_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) {
    return false;
  }
  const char* ptr = line.data();
  std::size_t remaining = line.size();
  while (remaining > 0U) {
    const ssize_t written = ::write(fd, ptr, remaining);
    if (written <= 0) {
      ::close(fd);
      return false;
    }
    ptr += written;
    remaining -= static_cast<std::size_t>(written);
  }
  if (fsync_after_write) {
    (void)::fsync(fd);
  }
  ::close(fd);
  return true;
#else
  std::ofstream ofs(sidecar_path, std::ios::binary | std::ios::app);
  if (!ofs.is_open()) {
    return false;
  }
  ofs.write(line.data(), static_cast<std::streamsize>(line.size()));
  if (fsync_after_write) {
    ofs.flush();
  }
  return ofs.good();
#endif
}

SidecarChainVerification VrtaEventsSidecarReader::verify_chain(
    const std::string& ndjson_text,
    const std::string& public_key_hex) {
  SidecarChainVerification out{};
  std::vector<uint8_t> pub_bytes;
  if (!hex_to_bytes(public_key_hex, pub_bytes) || pub_bytes.size() != 32U) {
    out.error_message = "invalid public_key_hex length";
    return out;
  }

  std::string expected_prev(64U, '0');
  bool has_prev_time = false;
  uint64_t prev_seq = 0U;
  uint64_t prev_t_us = 0U;

  std::istringstream iss(ndjson_text);
  std::string line;
  std::size_t line_no = 0U;
  while (std::getline(iss, line)) {
    ++line_no;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    SidecarEvent ev{};
    std::string parse_err;
    FlatJsonParser parser(line);
    if (!parser.parse_event(ev, parse_err)) {
      out.error_message = "line " + std::to_string(line_no) + ": " + parse_err;
      return out;
    }
    if (has_prev_time && (ev.seq < prev_seq || ev.t_sim_us < prev_t_us)) {
      out.error_message = "line " + std::to_string(line_no) + ": non-monotonic (seq, t_sim_us)";
      return out;
    }
    if (ev.prev_event_sha256 != expected_prev) {
      out.error_message = "line " + std::to_string(line_no) + ": broken prev_event_sha256 chain";
      return out;
    }
    const std::string recomputed = sha256_hex_string(build_canonical_payload_json(ev));
    if (recomputed != ev.event_sha256) {
      out.error_message = "line " + std::to_string(line_no) + ": event_sha256 mismatch";
      return out;
    }

    std::vector<uint8_t> sig_bytes;
    if (!hex_to_bytes(ev.signature_hex, sig_bytes)) {
      out.error_message = "line " + std::to_string(line_no) + ": invalid signature_hex";
      return out;
    }

    if (ev.signer_backend == "LIBSODIUM_ED25519") {
#if defined(VERTIRTA_HAS_LIBSODIUM) && VERTIRTA_HAS_LIBSODIUM
      if (sodium_init() < 0 || !verify_runtime_libsodium_ge_1_0_21() ||
          sig_bytes.size() != crypto_sign_BYTES) {
        out.error_message = "line " + std::to_string(line_no) + ": libsodium verify unavailable";
        return out;
      }
      if (crypto_sign_verify_detached(
              sig_bytes.data(),
              reinterpret_cast<const unsigned char*>(ev.event_sha256.data()),
              static_cast<unsigned long long>(ev.event_sha256.size()),
              pub_bytes.data()) != 0) {
        out.error_message = "line " + std::to_string(line_no) + ": Ed25519 signature invalid";
        return out;
      }
#else
      out.error_message = "line " + std::to_string(line_no) + ": built without libsodium";
      return out;
#endif
    } else if (ev.signer_backend == "SIM_SHA256_TEST_SIGNER") {
      if (sig_bytes.size() != 32U) {
        out.error_message = "line " + std::to_string(line_no) + ": invalid sim signature length";
        return out;
      }
      uint8_t expected_mac[32]{};
      hmac_sha256_32(
          pub_bytes.data(),
          reinterpret_cast<const uint8_t*>(ev.event_sha256.data()),
          ev.event_sha256.size(),
          expected_mac);
      uint8_t diff = 0U;
      for (std::size_t i = 0U; i < 32U; ++i) {
        diff |= static_cast<uint8_t>(expected_mac[i] ^ sig_bytes[i]);
      }
      if (diff != 0U) {
        out.error_message = "line " + std::to_string(line_no) + ": sim signature invalid";
        return out;
      }
    } else {
      out.error_message = "line " + std::to_string(line_no) + ": unknown signer_backend";
      return out;
    }

    expected_prev = ev.event_sha256;
    prev_seq = ev.seq;
    prev_t_us = ev.t_sim_us;
    has_prev_time = true;
    out.events.push_back(std::move(ev));
  }

  out.valid = true;
  return out;
}

SidecarChainVerification VrtaEventsSidecarReader::verify_file(
    const std::filesystem::path& sidecar_path,
    const std::string& public_key_hex) {
  std::ifstream ifs(sidecar_path, std::ios::binary);
  if (!ifs.is_open()) {
    SidecarChainVerification out{};
    out.error_message = "cannot open sidecar file";
    return out;
  }
  const std::string content(
      (std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return verify_chain(content, public_key_hex);
}

SidecarFrameJoinResult VrtaEventsSidecarReader::join_with_vrta_frames(
    const std::vector<vertirta::core::VrtaFrame100Hz>& frames,
    const std::vector<SidecarEvent>& events) {
  SidecarFrameJoinResult res{};
  std::unordered_map<uint64_t, uint64_t> seq_to_time;
  seq_to_time.reserve(frames.size());
  for (const auto& fr : frames) {
    if (!vertirta::core::verify_vrta_frame(fr)) {
      res.error_message = "corrupt VrtaFrame100Hz CRC-32C during join";
      return res;
    }
    seq_to_time[fr.seq] = fr.t_sim_us;
  }
  for (const auto& ev : events) {
    const auto it = seq_to_time.find(ev.seq);
    if (it == seq_to_time.end() || it->second != ev.t_sim_us) {
      res.error_message = "sidecar event (seq, t_sim_us) does not match any verified .vrta frame";
      return res;
    }
    ++res.matched_events;
  }
  res.valid = true;
  return res;
}

}  // namespace vertirta::forensics
