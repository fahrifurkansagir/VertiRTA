#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include "vertirta/rta/slz_prior_provider.hpp"

namespace vertirta::copilot {

inline constexpr uint32_t kRtaSnapshotSchemaVersion = 1U;
inline constexpr std::size_t kMaxSlzCandidates = 8U;
inline constexpr std::size_t kMaxDynamicNfz = 4U;
inline constexpr std::size_t kMaxStlNodes = 32U;

// Locked VertiRTA-X8-240 HV electrical & envelope constants (config/vehicle_x8_240.yaml)
inline constexpr float kHvPackNominalVolts = 400.0f;
inline constexpr float kHvPackEnergyKwh = 22.0f;
inline constexpr float kDefaultSocReserveFloorPct = 12.0f;
inline constexpr float kDefaultMinAglM = 30.0f;
inline constexpr float kDefaultMaxAglM = 150.0f;
inline constexpr float kDefaultMinPsafe = 0.85f;
inline constexpr float kDefaultMaxOodVar = 0.12f;
inline constexpr float kDefaultReachRadiusM = 15.0f;
inline constexpr double kKartalOriginLatDeg = 40.8860;
inline constexpr double kKartalOriginLonDeg = 29.1880;
inline constexpr double kKartalOriginAmslM = 12.0;
inline constexpr double kIstanbulEgm2008UndulationM = 36.7;
inline constexpr double kValidCoastalMaxLonDeg = 29.200;

enum class FaultMode : uint8_t {
  NOMINAL = 0,
  SINGLE_MOTOR_LOSS = 1,
  OPPOSITE_DUAL_LOSS = 2,
  SAME_ARM_DUAL_LOSS = 3,
  SENSOR_TMR_DEGRADED = 4,
};

struct EnuPoint {
  double east_m{0.0};
  double north_m{0.0};
  double up_m{0.0};  // Relative to 12.0 m AMSL canonical Kartal origin
};

struct AmsEnvelope {
  float ams_max_accel_mps2{3.2f};
  float ams_max_climb_mps{2.5f};
  float max_cruise_airspeed_mps{18.0f};
  float burn_rate_soc_pct_per_s{0.140f};
};

struct SlzCandidate {
  std::array<char, 32> id{};
  double lat_deg{kKartalOriginLatDeg};
  double lon_deg{kKartalOriginLonDeg};
  float elev_amsl_m{static_cast<float>(kKartalOriginAmslM)};
  EnuPoint enu{};
  float p_safe{0.0f};
  float slz_ood_variance{1.0f};
  float eta_s{0.0f};
  float soc_margin_pct{0.0f};
  uint8_t rank{0};
  bool west_of_29200e{false};

  [[nodiscard]] std::string_view id_view() const noexcept {
    std::size_t n = 0;
    while (n < id.size() && id[n] != '\0') {
      ++n;
    }
    return std::string_view{id.data(), n};
  }

  void set_id(std::string_view sv) noexcept {
    id.fill('\0');
    const std::size_t n = (sv.size() < id.size() - 1U) ? sv.size() : (id.size() - 1U);
    if (n > 0U) {
      std::memcpy(id.data(), sv.data(), n);
    }
  }
};

struct DynamicNfzCircle {
  std::array<char, 32> id{};
  EnuPoint center_enu{};
  float radius_m{0.0f};
  float floor_amsl_m{0.0f};
  float ceil_amsl_m{0.0f};
  bool active{false};

  void set_id(std::string_view sv) noexcept {
    id.fill('\0');
    const std::size_t n = (sv.size() < id.size() - 1U) ? sv.size() : (id.size() - 1U);
    if (n > 0U) {
      std::memcpy(id.data(), sv.data(), n);
    }
  }
};

struct RtaStateSnapshotV1 {
  uint32_t schema_version{kRtaSnapshotSchemaVersion};
  uint64_t seq_id{0};
  uint64_t timestamp_ms{0};
  FaultMode fault_mode{FaultMode::NOMINAL};
  uint8_t failed_motor_mask{0};
  bool shield_active{true};

  // Locked 400 V / 22 kWh HV pack metadata (DR-6 Erratum #1)
  float pack_nominal_volts{kHvPackNominalVolts};
  float pack_energy_kwh{kHvPackEnergyKwh};

  double lat_deg{kKartalOriginLatDeg};
  double lon_deg{kKartalOriginLonDeg};
  EnuPoint pos_enu{};
  float agl_m{0.0f};
  float amsl_m{static_cast<float>(kKartalOriginAmslM)};
  float groundspeed_mps{0.0f};
  float soc_pct{0.0f};
  float soc_reserve_floor_pct{kDefaultSocReserveFloorPct};
  float v_term_volts{368.5f};  // 400 V HV pack loaded terminal voltage (~350-380 V)
  float wind_mps{0.0f};
  float wind_from_deg{0.0f};

  AmsEnvelope ams{};

  bool ltfj_ctr_active{true};
  bool ltbx_ctr_active{true};

  uint8_t num_dyn_nfz{0};
  std::array<DynamicNfzCircle, kMaxDynamicNfz> dyn_nfzs{};

  uint8_t num_slz{0};
  std::array<SlzCandidate, kMaxSlzCandidates> ranked_slzs{};

  [[nodiscard]] const SlzCandidate* find_slz(std::string_view target_id) const noexcept {
    const std::size_t count = (num_slz <= kMaxSlzCandidates) ? num_slz : kMaxSlzCandidates;
    for (std::size_t i = 0; i < count; ++i) {
      if (ranked_slzs[i].id_view() == target_id) {
        return &ranked_slzs[i];
      }
    }
    return nullptr;
  }
};

// Zero-heap adapter populating RtaStateSnapshotV1 ranked SLZs from Plan 1A/3 ISlzPriorProvider.
inline void populate_slzs_from_provider(const vertirta::rta::SlzProviderSnapshot& src,
                                        RtaStateSnapshotV1& dst) noexcept {
  const std::size_t n = (src.count <= kMaxSlzCandidates) ? src.count : kMaxSlzCandidates;
  dst.num_slz = static_cast<uint8_t>(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto& p = src.priors[i];
    auto& out = dst.ranked_slzs[i];
    out = SlzCandidate{};
    std::size_t len = 0;
    while (len < sizeof(p.id) && p.id[len] != '\0') {
      ++len;
    }
    out.set_id(std::string_view{p.id, len});
    out.lat_deg = p.lat_deg;
    out.lon_deg = p.lon_deg;
    out.elev_amsl_m = static_cast<float>(p.elev_amsl_m);
    out.enu = EnuPoint{p.center_enu.x, p.center_enu.y, p.center_enu.z};
    out.p_safe = p.p_safe;
    out.slz_ood_variance = p.slz_ood_variance;
    out.eta_s = p.eta_s;
    out.soc_margin_pct = p.soc_margin_pct;
    out.rank = p.rank;
    out.west_of_29200e = (p.lon_deg < kValidCoastalMaxLonDeg);
  }
}

enum class ActionKind : uint8_t {
  NONE = 0,
  DIVERT_SLZ = 1,
  INSPECT_SLZ = 2,
  HOLD_LOITER = 3,
  ABORT_RETURN = 4,
  REQUEST_CLARIFICATION = 5,
  REJECT_UNSAFE = 6,
};

enum class LangCode : uint8_t {
  TR = 0,
  EN = 1,
  MIX = 2,
};

enum class ModeCode : uint8_t {
  BRIEF = 0,
  REJECT = 1,
  CLARIFY = 2,
};

enum class ClarifyReason : uint8_t {
  NONE = 0,
  AMBIG = 1,
  NO_SLOT = 2,
  LOW_ASR = 3,
};

enum class ViolationCode : uint8_t {
  NONE = 0,
  LTFJ_CTR = 1,
  LTBX_CTR = 2,
  DYN_NFZ = 3,
  SOC_RESERVE = 4,
  SLZ_UNSAFE = 5,
  MIN_AGL = 6,
  SHIELD_BYPASS = 7,
};

struct CompactDslAst {
  ActionKind action{ActionKind::NONE};
  std::array<char, 32> target_id_buf{};
  std::array<char, 32> opt_a_buf{};
  std::array<char, 32> opt_b_buf{};
  std::array<char, 32> fallback_buf{};
  int32_t eta_lim_s{0};
  int32_t duration_s{0};
  int32_t agl_min_m{35};
  int32_t agl_max_m{120};
  int32_t speed_max_mps{18};
  ClarifyReason clarify_reason{ClarifyReason::NONE};
  ViolationCode violation_code{ViolationCode::NONE};
  LangCode lang{LangCode::TR};
  ModeCode mode{ModeCode::BRIEF};
  bool valid{false};

  [[nodiscard]] std::string_view target_id() const noexcept { return buf_view(target_id_buf); }
  [[nodiscard]] std::string_view opt_a() const noexcept { return buf_view(opt_a_buf); }
  [[nodiscard]] std::string_view opt_b() const noexcept { return buf_view(opt_b_buf); }
  [[nodiscard]] std::string_view fallback_target() const noexcept { return buf_view(fallback_buf); }

  void set_target_id(std::string_view sv) noexcept { set_buf(target_id_buf, sv); }
  void set_opt_a(std::string_view sv) noexcept { set_buf(opt_a_buf, sv); }
  void set_opt_b(std::string_view sv) noexcept { set_buf(opt_b_buf, sv); }
  void set_fallback_target(std::string_view sv) noexcept { set_buf(fallback_buf, sv); }

 private:
  static std::string_view buf_view(const std::array<char, 32>& b) noexcept {
    std::size_t n = 0;
    while (n < b.size() && b[n] != '\0') {
      ++n;
    }
    return std::string_view{b.data(), n};
  }
  static void set_buf(std::array<char, 32>& b, std::string_view sv) noexcept {
    b.fill('\0');
    const std::size_t n = (sv.size() < b.size() - 1U) ? sv.size() : (b.size() - 1U);
    if (n > 0U) {
      std::memcpy(b.data(), sv.data(), n);
    }
  }
};

enum class StlNodeKind : uint8_t {
  EMPTY = 0,
  PRED_AGL_GE = 1,
  PRED_AGL_LE = 2,
  PRED_OUTSIDE_ZONE = 3,
  PRED_SOC_GE = 4,
  PRED_REACH_TARGET = 5,
  PRED_SPEED_LE = 6,
  PRED_OOD_LE = 7,
  OP_NOT = 8,
  OP_AND = 9,
  OP_OR = 10,
  OP_ALWAYS = 11,
  OP_EVENTUALLY = 12,
  OP_UNTIL = 13,
};

enum class ZoneId : uint8_t {
  LTFJ_CTR = 0,
  LTBX_CTR = 1,
  DYN_NFZ_0 = 2,
  DYN_NFZ_1 = 3,
  DYN_NFZ_2 = 4,
  DYN_NFZ_3 = 5,
};

struct StlAstNode {
  StlNodeKind kind{StlNodeKind::EMPTY};
  uint8_t left{0xFFU};
  uint8_t right{0xFFU};
  uint32_t t_a{0U};
  uint32_t t_b{0U};
  float param{0.0f};
  ZoneId zone{ZoneId::LTFJ_CTR};
};

struct BoundedStlFormula {
  std::array<StlAstNode, kMaxStlNodes> nodes{};
  uint8_t node_count{0};
  uint8_t root_index{0xFFU};
  EnuPoint target_enu{};
  bool includes_ltfj_exclusion{false};
  bool includes_ltbx_exclusion{false};
  float until_target_radius_m{0.0f};
  uint32_t horizon_s{0U};
};

[[nodiscard]] bool parse_compact_dsl(std::string_view text, CompactDslAst& out) noexcept;

[[nodiscard]] BoundedStlFormula synthesize_canonical_stl(
    const CompactDslAst& ast,
    const RtaStateSnapshotV1& st) noexcept;

[[nodiscard]] RtaStateSnapshotV1 make_reference_400v_sml_snapshot() noexcept;

}  // namespace vertirta::copilot
