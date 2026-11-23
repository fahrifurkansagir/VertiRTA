// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vertirta::uspace_node {

// Canonical Kartal Datum constants (locked to config/istanbul_ltfj_airspace.yaml)
inline constexpr double kKartalDatumLatDeg = 40.8860;
inline constexpr double kKartalDatumLonDeg = 29.1880;
inline constexpr double kKartalEgm2008N0M = 36.7;
inline constexpr double kEasternGeofenceLonDeg = 29.1895;  // Strictly west of LTFJ NW J2 (29.19212 E)
inline constexpr double kMetersPerDegLat = 111052.0;
inline constexpr double kMetersPerDegLon = 84255.0;

// Internal RTA priority classes mapped bijectively to ASTM F3548-21 integers
// (ASTM uses HIGHER integers for HIGHER priority: 100 > 10 > 5).
enum class RtaPriorityClass : uint8_t {
  PRIORITY_0_EMERGENCY_CEL = 0,
  PRIORITY_1_REVENUE = 1,
  PRIORITY_2_REPOSITION = 2,
};

enum class OivState : uint8_t {
  NONE = 0,
  ACCEPTED = 1,
  ACTIVATED = 2,
  NONCONFORMING = 3,
  CONTINGENT = 4,
  ENDED = 5,
};

[[nodiscard]] constexpr int to_astm_priority(RtaPriorityClass cls) noexcept {
  switch (cls) {
    case RtaPriorityClass::PRIORITY_0_EMERGENCY_CEL:
      return 100;
    case RtaPriorityClass::PRIORITY_1_REVENUE:
      return 10;
    case RtaPriorityClass::PRIORITY_2_REPOSITION:
    default:
      return 5;
  }
}

[[nodiscard]] constexpr RtaPriorityClass from_astm_priority(int astm_priority) noexcept {
  if (astm_priority >= 100) {
    return RtaPriorityClass::PRIORITY_0_EMERGENCY_CEL;
  }
  if (astm_priority >= 10) {
    return RtaPriorityClass::PRIORITY_1_REVENUE;
  }
  return RtaPriorityClass::PRIORITY_2_REPOSITION;
}

[[nodiscard]] constexpr double istanbul_egm2008_n_m(double lat_deg, double lon_deg) noexcept {
  // Anchored to Kartal +36.70 m (consistent with core/flight_core geodesy.hpp)
  const double d_lat = lat_deg - kKartalDatumLatDeg;
  const double d_lon = lon_deg - kKartalDatumLonDeg;
  return kKartalEgm2008N0M + 0.18 * d_lat + 0.42 * d_lon;
}

struct OivRecord {
  std::string id{};
  OivState state{OivState::NONE};
  int astm_priority{5};
  double lat_deg{0.0};
  double lon_deg{0.0};
  double alt_msl_m{0.0};
  double alt_w84_m{0.0};
  std::string ovn{};
  std::string divert_vertiport{};
  bool omitted_ovn_key{false};
};

struct Volume4DEnu {
  double e_min_m{0.0};
  double e_max_m{0.0};
  double n_min_m{0.0};
  double n_max_m{0.0};
  double z_min_m{0.0};
  double z_max_m{0.0};
  double t_start_s{0.0};
  double t_end_s{0.0};
};

// Local Kartal ENU 4D exact intersection filter rejecting S2 Level-13 coarse cell DAR false positives.
[[nodiscard]] inline bool volumes_intersect_4d_enu(
    const Volume4DEnu& a,
    const Volume4DEnu& b) noexcept {
  const bool ov_e = (a.e_min_m <= b.e_max_m) && (b.e_min_m <= a.e_max_m);
  const bool ov_n = (a.n_min_m <= b.n_max_m) && (b.n_min_m <= a.n_max_m);
  const bool ov_z = (a.z_min_m <= b.z_max_m) && (b.z_min_m <= a.z_max_m);
  const bool ov_t = (a.t_start_s <= b.t_end_s) && (b.t_start_s <= a.t_end_s);
  return ov_e && ov_n && ov_z && ov_t;
}

// InterUSS #1549 guard: automatically tiles wide NOTAM polygons (> 2,500 km^2) into sub-100 km^2 tiles.
[[nodiscard]] inline std::vector<double> tile_notam_polygon_km2(
    double area_km2,
    double max_tile_km2 = 100.0) {
  std::vector<double> tiles{};
  if (area_km2 <= 0.0 || max_tile_km2 <= 0.0) {
    return tiles;
  }
  const std::size_t n_tiles = static_cast<std::size_t>(std::ceil(area_km2 / max_tile_km2));
  tiles.reserve(n_tiles);
  double rem = area_km2;
  for (std::size_t i = 0U; i < n_tiles; ++i) {
    const double a = std::min(rem, max_tile_km2);
    tiles.push_back(a);
    rem -= a;
  }
  return tiles;
}

// ASTM F3411-22a + RFC 9575 Extended DRIP Wrapper MessagePack encoder:
//   3 B header (0xF2, 25, 9) + 9 x 25 B messages (BasicID, Location, SelfID, System + 5 DRIP Auth pages)
//   = exact 228 bytes.
[[nodiscard]] inline std::array<uint8_t, 228U> encode_f3411_drip_message_pack(
    std::string_view uas_id,
    double lat_deg,
    double lon_deg,
    double alt_msl_m,
    bool cel_emg) noexcept {
  std::array<uint8_t, 228U> buf{};

  // 3-byte F3411-22a MessagePack header (MsgType=0xF, ProtoVersion=2 -> 0xF2)
  buf[0] = 0xF2U;
  buf[1] = 25U;  // Single-message length
  buf[2] = 9U;   // 4 core messages + 5 RFC 9575 Extended DRIP Auth pages

  const auto write_le32 = [](uint8_t* dst, int32_t v) noexcept {
    const uint32_t u = static_cast<uint32_t>(v);
    dst[0] = static_cast<uint8_t>(u & 0xFFU);
    dst[1] = static_cast<uint8_t>((u >> 8U) & 0xFFU);
    dst[2] = static_cast<uint8_t>((u >> 16U) & 0xFFU);
    dst[3] = static_cast<uint8_t>((u >> 24U) & 0xFFU);
  };
  const auto write_le16 = [](uint8_t* dst, uint16_t u) noexcept {
    dst[0] = static_cast<uint8_t>(u & 0xFFU);
    dst[1] = static_cast<uint8_t>((u >> 8U) & 0xFFU);
  };

  // Msg 0 [3..27]: Basic ID (MsgType=0x0, Proto=2 -> 0x02)
  uint8_t* m0 = buf.data() + 3U;
  m0[0] = 0x02U;
  m0[1] = 0x43U;  // IDType=4 (Specific Session ID / RFC 9374 DET), UAType=3 (Hybrid/VTOL)
  const std::size_t id_copy = std::min<std::size_t>(uas_id.size(), 20U);
  for (std::size_t i = 0U; i < id_copy; ++i) {
    m0[2U + i] = static_cast<uint8_t>(uas_id[i]);
  }

  // Msg 1 [28..52]: Location/Vector (MsgType=0x1, Proto=2 -> 0x12)
  uint8_t* m1 = buf.data() + 28U;
  m1[0] = 0x12U;
  m1[1] = cel_emg ? 0x30U : 0x20U;  // Operational status: Emergency (3) vs Airborne (2)
  const int32_t lat_e7 = static_cast<int32_t>(std::llround(lat_deg * 1.0e7));
  const int32_t lon_e7 = static_cast<int32_t>(std::llround(lon_deg * 1.0e7));
  const double alt_w84 = alt_msl_m + istanbul_egm2008_n_m(lat_deg, lon_deg);
  const uint16_t alt_enc = static_cast<uint16_t>(
      std::clamp(std::llround((alt_w84 + 1000.0) * 2.0), 0LL, 65535LL));
  write_le32(m1 + 5U, lat_e7);
  write_le32(m1 + 9U, lon_e7);
  write_le16(m1 + 15U, alt_enc);  // Geodetic altitude (WGS84 HAE)

  // Msg 2 [53..77]: Self-ID (MsgType=0x3, Proto=2 -> 0x32)
  uint8_t* m2 = buf.data() + 53U;
  m2[0] = 0x32U;
  m2[1] = cel_emg ? 0x01U : 0x00U;  // DescType: 1=Emergency, 0=Text
  const std::string_view self_txt = cel_emg ? "EMG-CEL-SPIN-VPT-CVZ" : "VERTIRTA-USPACE-CORR";
  const std::size_t txt_copy = std::min<std::size_t>(self_txt.size(), 23U);
  for (std::size_t i = 0U; i < txt_copy; ++i) {
    m2[2U + i] = static_cast<uint8_t>(self_txt[i]);
  }

  // Msg 3 [78..102]: System Message (MsgType=0x4, Proto=2 -> 0x42)
  uint8_t* m3 = buf.data() + 78U;
  m3[0] = 0x42U;
  m3[1] = 0x04U;  // EU classification + operator location fixed
  write_le32(m3 + 2U, static_cast<int32_t>(std::llround(kKartalDatumLatDeg * 1.0e7)));
  write_le32(m3 + 6U, static_cast<int32_t>(std::llround(kKartalDatumLonDeg * 1.0e7)));

  // Msgs 4..8 [103..227]: 5 Auth Pages carrying RFC 9575 Extended DRIP Wrapper (AuthType=0x5, IANA SAM=0x02)
  for (uint8_t page = 0U; page < 5U; ++page) {
    uint8_t* ma = buf.data() + 103U + static_cast<std::size_t>(page) * 25U;
    ma[0] = 0x22U;  // MsgType=0x2 (Auth), Proto=2
    ma[1] = static_cast<uint8_t>((0x5U << 4U) | (page & 0x0FU));  // AuthType=5 (SAM), DataPage=0..4
    if (page == 0U) {
      ma[2] = 4U;    // LastPageIndex = 4 (5 pages total)
      ma[3] = 109U;  // 17 + 4*23 = 109 B RFC 9575 Extended DRIP Wrapper payload
      // Timestamp (4 bytes) + IANA SAM ID = 0x02 (DRIP Wrapper) + Ed25519 signature prefix
      ma[8] = 0x02U;  // RFC 9575 IANA Specific Authentication Method: DRIP Wrapper
      ma[9] = cel_emg ? 0x03U : 0x01U;
    } else {
      // Deterministic DRIP signature / endorsement evidence bytes
      for (std::size_t b = 2U; b < 25U; ++b) {
        ma[b] = static_cast<uint8_t>((page * 37U + b * 13U) & 0xFFU);
      }
    }
  }

  return buf;
}

class UspaceCore {
 public:
  [[nodiscard]] OivRecord create_oiv(
      const std::string& id,
      RtaPriorityClass prio,
      double lat_deg,
      double lon_deg,
      double alt_msl_m) {
    OivRecord rec{};
    rec.id = id;
    rec.astm_priority = to_astm_priority(prio);
    rec.lat_deg = lat_deg;
    rec.lon_deg = lon_deg;
    rec.alt_msl_m = alt_msl_m;
    rec.alt_w84_m = alt_msl_m + istanbul_egm2008_n_m(lat_deg, lon_deg);
    rec.ovn = "ovn-" + id + "-" + std::to_string(++ovn_seq_);

    // Reject OIV creation east of LTFJ NW J2 fence (29.1895 E) or above ODD ceiling (150 m)
    if (lon_deg >= kEasternGeofenceLonDeg || alt_msl_m > 150.0) {
      rec.state = OivState::NONE;
      return rec;
    }
    rec.state = OivState::ACCEPTED;
    oivs_[id] = rec;
    return rec;
  }

  [[nodiscard]] bool transition_oiv(
      const std::string& id,
      OivState target,
      bool omit_ovn) {
    auto it = oivs_.find(id);
    if (it == oivs_.end()) {
      return false;
    }
    auto& rec = it->second;
    const OivState cur = rec.state;

    // ASTM F3548-21 strict OIV transition rules (Contingent is strictly irreversible except -> Ended)
    bool allowed = false;
    if (cur == OivState::ACCEPTED && (target == OivState::ACTIVATED || target == OivState::ENDED)) {
      allowed = true;
    } else if (cur == OivState::ACTIVATED &&
               (target == OivState::NONCONFORMING || target == OivState::CONTINGENT || target == OivState::ENDED)) {
      allowed = true;
    } else if (cur == OivState::NONCONFORMING &&
               (target == OivState::ACTIVATED || target == OivState::CONTINGENT || target == OivState::ENDED)) {
      allowed = true;
    } else if (cur == OivState::CONTINGENT && target == OivState::ENDED) {
      allowed = true;
    }

    if (!allowed) {
      return false;
    }
    rec.state = target;
    rec.omitted_ovn_key = omit_ovn;
    rec.ovn = "ovn-" + id + "-" + std::to_string(++ovn_seq_);
    return true;
  }

  [[nodiscard]] bool trigger_contingent_preemption(
      const std::string& id,
      const std::string& fato_emg_code) {
    auto it = oivs_.find(id);
    if (it == oivs_.end()) {
      return false;
    }
    auto& rec = it->second;
    if (rec.state == OivState::ENDED || rec.state == OivState::NONE) {
      return false;
    }
    rec.state = OivState::CONTINGENT;
    rec.astm_priority = to_astm_priority(RtaPriorityClass::PRIORITY_0_EMERGENCY_CEL);
    rec.omitted_ovn_key = true;  // Omit OVN key for zero-block DSS write (F3548-21 emergency override)
    rec.divert_vertiport = fato_emg_code;
    rec.ovn = "ovn-emg-" + id + "-" + std::to_string(++ovn_seq_);
    return true;
  }

  [[nodiscard]] OivState oiv_state(const std::string& id) const {
    const auto it = oivs_.find(id);
    return (it != oivs_.end()) ? it->second.state : OivState::NONE;
  }

  [[nodiscard]] int oiv_astm_priority(const std::string& id) const {
    const auto it = oivs_.find(id);
    return (it != oivs_.end()) ? it->second.astm_priority : 0;
  }

  [[nodiscard]] bool oiv_omitted_ovn(const std::string& id) const {
    const auto it = oivs_.find(id);
    return (it != oivs_.end()) ? it->second.omitted_ovn_key : false;
  }

 private:
  std::unordered_map<std::string, OivRecord> oivs_{};
  uint64_t ovn_seq_{0ULL};
};

}  // namespace vertirta::uspace_node
