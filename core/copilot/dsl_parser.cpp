#include "vertirta/copilot/rta_state_contract.hpp"

#include <charconv>

namespace vertirta::copilot {

namespace {

[[nodiscard]] bool is_valid_slz_id(std::string_view s) noexcept {
  return s == "Kartal_Sahil" || s == "Dragos_Plaza" || s == "Maltepe_Park" ||
         s == "Cevizli_Helipad";
}

[[nodiscard]] bool parse_int_exact(std::string_view s, int32_t min_v, int32_t max_v,
                                   int32_t& out) noexcept {
  if (s.empty() || s.size() > 4U) {
    return false;
  }
  if (s.size() > 1U && s[0] == '0') {
    return false;
  }
  for (char ch : s) {
    if (ch < '0' || ch > '9') {
      return false;
    }
  }
  int32_t val = 0;
  const auto res = std::from_chars(s.data(), s.data() + s.size(), val);
  if (res.ec != std::errc{} || res.ptr != s.data() + s.size()) {
    return false;
  }
  if (val < min_v || val > max_v) {
    return false;
  }
  out = val;
  return true;
}

[[nodiscard]] bool split_kv(std::string_view token, std::string_view expected_key,
                            std::string_view& val_out) noexcept {
  const std::size_t eq = token.find('=');
  if (eq == std::string_view::npos || token.substr(0, eq) != expected_key) {
    return false;
  }
  val_out = token.substr(eq + 1U);
  return !val_out.empty();
}

[[nodiscard]] bool next_field(std::string_view& rest, std::string_view& field) noexcept {
  if (rest.empty()) {
    return false;
  }
  const std::size_t sep = rest.find(';');
  if (sep == std::string_view::npos) {
    field = rest;
    rest = std::string_view{};
  } else {
    field = rest.substr(0, sep);
    rest = rest.substr(sep + 1U);
    if (rest.empty()) {
      return false;  // Reject trailing semicolon
    }
  }
  return !field.empty();
}

[[nodiscard]] bool parse_lang(std::string_view v, LangCode& out) noexcept {
  if (v == "TR") {
    out = LangCode::TR;
    return true;
  }
  if (v == "EN") {
    out = LangCode::EN;
    return true;
  }
  if (v == "MIX") {
    out = LangCode::MIX;
    return true;
  }
  return false;
}

[[nodiscard]] bool parse_mode(std::string_view v, ModeCode& out) noexcept {
  if (v == "BRIEF") {
    out = ModeCode::BRIEF;
    return true;
  }
  if (v == "REJECT") {
    out = ModeCode::REJECT;
    return true;
  }
  if (v == "CLARIFY") {
    out = ModeCode::CLARIFY;
    return true;
  }
  return false;
}

[[nodiscard]] bool parse_clarify_reason(std::string_view v, ClarifyReason& out) noexcept {
  if (v == "AMBIG") {
    out = ClarifyReason::AMBIG;
    return true;
  }
  if (v == "NO_SLOT") {
    out = ClarifyReason::NO_SLOT;
    return true;
  }
  if (v == "LOW_ASR") {
    out = ClarifyReason::LOW_ASR;
    return true;
  }
  return false;
}

[[nodiscard]] bool parse_violation_code(std::string_view v, ViolationCode& out) noexcept {
  if (v == "LTFJ_CTR") {
    out = ViolationCode::LTFJ_CTR;
    return true;
  }
  if (v == "LTBX_CTR") {
    out = ViolationCode::LTBX_CTR;
    return true;
  }
  if (v == "DYN_NFZ") {
    out = ViolationCode::DYN_NFZ;
    return true;
  }
  if (v == "SOC_RESERVE") {
    out = ViolationCode::SOC_RESERVE;
    return true;
  }
  if (v == "SLZ_UNSAFE") {
    out = ViolationCode::SLZ_UNSAFE;
    return true;
  }
  if (v == "MIN_AGL") {
    out = ViolationCode::MIN_AGL;
    return true;
  }
  if (v == "SHIELD_BYPASS") {
    out = ViolationCode::SHIELD_BYPASS;
    return true;
  }
  return false;
}

}  // namespace

bool parse_compact_dsl(std::string_view text, CompactDslAst& out) noexcept {
  out = CompactDslAst{};
  if (text.empty()) {
    return false;
  }
  // Reject non-ASCII bytes, CR, or any <think> / angle-bracket injection
  for (unsigned char c : text) {
    if (c == '\r' || c == '<' || c == '>' || (c < 0x20U && c != '\n') || c >= 0x7FU) {
      return false;
    }
  }
  if (text.back() == '\n') {
    text.remove_suffix(1U);
  }
  const std::size_t nl = text.find('\n');
  if (nl == std::string_view::npos) {
    return false;
  }
  // Must be strictly 2 lines (no extra newline / trailing garbage)
  if (text.find('\n', nl + 1U) != std::string_view::npos) {
    return false;
  }

  std::string_view line1 = text.substr(0, nl);
  std::string_view line2 = text.substr(nl + 1U);

  std::string_view field{};
  std::string_view val{};
  if (!next_field(line1, field) || !split_kv(field, "ACT", val)) {
    return false;
  }

  if (val == "DIVERT_SLZ" || val == "INSPECT_SLZ" || val == "ABORT_RETURN") {
    if (val == "DIVERT_SLZ") {
      out.action = ActionKind::DIVERT_SLZ;
    } else if (val == "INSPECT_SLZ") {
      out.action = ActionKind::INSPECT_SLZ;
    } else {
      out.action = ActionKind::ABORT_RETURN;
    }

    std::string_view tgt_v{}, eta_v{}, agl_v{}, spd_v{};
    if (!next_field(line1, field) || !split_kv(field, "TGT", tgt_v) || !is_valid_slz_id(tgt_v)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "ETA", eta_v) ||
        !parse_int_exact(eta_v, 1, 600, out.eta_lim_s)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "AGL", agl_v)) {
      return false;
    }
    const std::size_t comma = agl_v.find(',');
    if (comma == std::string_view::npos || agl_v.find(',', comma + 1U) != std::string_view::npos) {
      return false;
    }
    if (!parse_int_exact(agl_v.substr(0, comma), 10, 150, out.agl_min_m) ||
        !parse_int_exact(agl_v.substr(comma + 1U), 30, 150, out.agl_max_m)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "SPD", spd_v) ||
        !parse_int_exact(spd_v, 5, 30, out.speed_max_mps)) {
      return false;
    }
    if (!line1.empty()) {
      return false;
    }
    out.set_target_id(tgt_v);
  } else if (val == "HOLD_LOITER") {
    out.action = ActionKind::HOLD_LOITER;
    std::string_view dur_v{}, agl_v{};
    if (!next_field(line1, field) || !split_kv(field, "DUR", dur_v) ||
        !parse_int_exact(dur_v, 10, 300, out.duration_s)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "AGL", agl_v) ||
        !parse_int_exact(agl_v, 30, 150, out.agl_min_m)) {
      return false;
    }
    if (!line1.empty()) {
      return false;
    }
    out.agl_max_m = out.agl_min_m;
    out.eta_lim_s = out.duration_s;
    out.speed_max_mps = 5;
  } else if (val == "REQUEST_CLARIFICATION") {
    out.action = ActionKind::REQUEST_CLARIFICATION;
    std::string_view a_v{}, b_v{}, rsn_v{};
    if (!next_field(line1, field) || !split_kv(field, "A", a_v) || !is_valid_slz_id(a_v)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "B", b_v) || !is_valid_slz_id(b_v)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "RSN", rsn_v) ||
        !parse_clarify_reason(rsn_v, out.clarify_reason)) {
      return false;
    }
    if (!line1.empty()) {
      return false;
    }
    out.set_opt_a(a_v);
    out.set_opt_b(b_v);
    out.duration_s = 30;
    out.eta_lim_s = 30;
  } else if (val == "REJECT_UNSAFE") {
    out.action = ActionKind::REJECT_UNSAFE;
    std::string_view vio_v{}, fb_v{};
    if (!next_field(line1, field) || !split_kv(field, "VIO", vio_v) ||
        !parse_violation_code(vio_v, out.violation_code)) {
      return false;
    }
    if (!next_field(line1, field) || !split_kv(field, "FB", fb_v) ||
        (!is_valid_slz_id(fb_v) && fb_v != "HOLD_LOITER")) {
      return false;
    }
    if (!line1.empty()) {
      return false;
    }
    out.set_fallback_target(fb_v);
    out.eta_lim_s = 100;
  } else {
    return false;
  }

  // Parse meta-line: LNG=<TR|EN|MIX>;MOD=<BRIEF|REJECT|CLARIFY>
  std::string_view lng_v{}, mod_v{};
  if (!next_field(line2, field) || !split_kv(field, "LNG", lng_v) || !parse_lang(lng_v, out.lang)) {
    return false;
  }
  if (!next_field(line2, field) || !split_kv(field, "MOD", mod_v) || !parse_mode(mod_v, out.mode)) {
    return false;
  }
  if (!line2.empty()) {
    return false;
  }

  out.valid = true;
  return true;
}

RtaStateSnapshotV1 make_reference_400v_sml_snapshot() noexcept {
  RtaStateSnapshotV1 st{};
  st.schema_version = kRtaSnapshotSchemaVersion;
  st.seq_id = 4821U;
  st.timestamp_ms = 1795260240000ULL;
  st.fault_mode = FaultMode::SINGLE_MOTOR_LOSS;
  st.failed_motor_mask = 0x04U;  // Motor 3
  st.shield_active = true;
  st.pack_nominal_volts = kHvPackNominalVolts;
  st.pack_energy_kwh = kHvPackEnergyKwh;

  st.lat_deg = 40.89455;
  st.lon_deg = 29.17435;
  st.pos_enu = EnuPoint{-1150.0, 950.0, 65.0};
  st.agl_m = 65.0f;
  st.amsl_m = 77.2f;
  st.groundspeed_mps = 16.4f;
  st.soc_pct = 34.0f;
  st.soc_reserve_floor_pct = 12.0f;
  st.v_term_volts = 368.5f;  // 400 V / 22 kWh HV pack loaded voltage (NOT 44.2 V)
  st.wind_mps = 5.0f;
  st.wind_from_deg = 240.0f;

  st.ams.ams_max_accel_mps2 = 3.2f;
  st.ams.ams_max_climb_mps = 2.5f;
  st.ams.max_cruise_airspeed_mps = 18.0f;
  st.ams.burn_rate_soc_pct_per_s = 0.140f;

  st.ltfj_ctr_active = true;
  st.ltbx_ctr_active = true;
  st.num_dyn_nfz = 0U;

  st.num_slz = 4U;

  st.ranked_slzs[0].set_id("Kartal_Sahil");
  st.ranked_slzs[0].lat_deg = 40.88600;
  st.ranked_slzs[0].lon_deg = 29.18800;
  st.ranked_slzs[0].elev_amsl_m = 12.0f;
  st.ranked_slzs[0].enu = EnuPoint{0.0, 0.0, 0.0};
  st.ranked_slzs[0].p_safe = 0.94f;
  st.ranked_slzs[0].slz_ood_variance = 0.03f;
  st.ranked_slzs[0].eta_s = 100.0f;
  st.ranked_slzs[0].soc_margin_pct = 8.0f;
  st.ranked_slzs[0].rank = 1U;
  st.ranked_slzs[0].west_of_29200e = true;

  st.ranked_slzs[1].set_id("Cevizli_Helipad");
  st.ranked_slzs[1].lat_deg = 40.90350;
  st.ranked_slzs[1].lon_deg = 29.17200;
  st.ranked_slzs[1].elev_amsl_m = 18.0f;
  st.ranked_slzs[1].enu = EnuPoint{-1348.1, 1943.5, 5.6};
  st.ranked_slzs[1].p_safe = 0.89f;
  st.ranked_slzs[1].slz_ood_variance = 0.06f;
  st.ranked_slzs[1].eta_s = 75.4f;
  st.ranked_slzs[1].soc_margin_pct = 11.4f;
  st.ranked_slzs[1].rank = 2U;
  st.ranked_slzs[1].west_of_29200e = true;

  st.ranked_slzs[2].set_id("Dragos_Plaza");
  st.ranked_slzs[2].lat_deg = 40.90950;
  st.ranked_slzs[2].lon_deg = 29.14650;
  st.ranked_slzs[2].elev_amsl_m = 14.0f;
  st.ranked_slzs[2].enu = EnuPoint{-3496.4, 2610.6, 0.5};
  st.ranked_slzs[2].p_safe = 0.92f;
  st.ranked_slzs[2].slz_ood_variance = 0.04f;
  st.ranked_slzs[2].eta_s = 209.5f;
  st.ranked_slzs[2].soc_margin_pct = -7.3f;
  st.ranked_slzs[2].rank = 3U;
  st.ranked_slzs[2].west_of_29200e = true;

  st.ranked_slzs[3].set_id("Maltepe_Park");
  st.ranked_slzs[3].lat_deg = 40.92350;
  st.ranked_slzs[3].lon_deg = 29.12350;
  st.ranked_slzs[3].elev_amsl_m = 10.0f;
  st.ranked_slzs[3].enu = EnuPoint{-5433.0, 4166.5, -5.7};
  st.ranked_slzs[3].p_safe = 0.91f;
  st.ranked_slzs[3].slz_ood_variance = 0.05f;
  st.ranked_slzs[3].eta_s = 368.9f;
  st.ranked_slzs[3].soc_margin_pct = -29.6f;
  st.ranked_slzs[3].rank = 4U;
  st.ranked_slzs[3].west_of_29200e = true;

  return st;
}

}  // namespace vertirta::copilot
