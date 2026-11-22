#include "vertirta/copilot/llama_gbnf_engine.hpp"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace vertirta::copilot {

namespace {

[[nodiscard]] std::string display_slz_name(std::string_view raw) {
  std::string out(raw);
  for (char& ch : out) {
    if (ch == '_') {
      ch = ' ';
    }
  }
  return out;
}

[[nodiscard]] int first_failed_motor_1based(uint8_t mask) noexcept {
  for (int bit = 0; bit < 8; ++bit) {
    if ((mask & static_cast<uint8_t>(1U << bit)) != 0U) {
      return bit + 1;
    }
  }
  return 0;
}

[[nodiscard]] const char* fault_mode_name(FaultMode m) noexcept {
  switch (m) {
    case FaultMode::NOMINAL:
      return "NOMINAL";
    case FaultMode::SINGLE_MOTOR_LOSS:
      return "SINGLE_MOTOR_LOSS";
    case FaultMode::OPPOSITE_DUAL_LOSS:
      return "OPPOSITE_DUAL_LOSS";
    case FaultMode::SAME_ARM_DUAL_LOSS:
      return "SAME_ARM_DUAL_LOSS";
    case FaultMode::SENSOR_TMR_DEGRADED:
      return "SENSOR_TMR_DEGRADED";
  }
  return "UNKNOWN";
}

[[nodiscard]] bool is_ascii_digit(char c) noexcept {
  return c >= '0' && c <= '9';
}

[[nodiscard]] bool is_ascii_alpha(char c) noexcept {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

}  // namespace

std::string render_clarification_question(const RtaStateSnapshotV1& st,
                                          std::string_view opt_a,
                                          std::string_view opt_b,
                                          LangCode lang) {
  const SlzCandidate* a = st.find_slz(opt_a);
  const SlzCandidate* b = st.find_slz(opt_b);
  const int a_pct = (a != nullptr) ? static_cast<int>(std::lround(a->p_safe * 100.0f)) : 0;
  const int a_eta = (a != nullptr) ? static_cast<int>(std::lround(a->eta_s)) : 0;
  const int b_pct = (b != nullptr) ? static_cast<int>(std::lround(b->p_safe * 100.0f)) : 0;
  const int b_eta = (b != nullptr) ? static_cast<int>(std::lround(b->eta_s)) : 0;
  const int motor_id = first_failed_motor_1based(st.failed_motor_mask);

  std::ostringstream oss;
  if (lang == LangCode::EN) {
    if (motor_id > 0) {
      oss << "Motor " << motor_id << " fault; ";
    }
    oss << "divert to " << display_slz_name(opt_a) << " (A, " << a_pct << "% safe, " << a_eta
        << " s) or " << display_slz_name(opt_b) << " (B, " << b_pct << "% safe, " << b_eta
        << " s)?";
  } else {
    if (motor_id > 0) {
      oss << "Motor " << motor_id << " arizali; ";
    }
    oss << display_slz_name(opt_a) << " (A, %" << a_pct << " guvenli, " << a_eta << " sn) mi, "
        << display_slz_name(opt_b) << " (B, %" << b_pct << " guvenli, " << b_eta << " sn) mi?";
  }
  return oss.str();
}

std::string render_bilingual_response(const RtaStateSnapshotV1& st,
                                      const CompactDslAst& ast,
                                      const RepairDecision* repair,
                                      LangCode lang) {
  if (repair != nullptr && repair->repaired) {
    return (lang == LangCode::EN) ? repair->explanation_en : repair->explanation_tr;
  }

  if (ast.action == ActionKind::REQUEST_CLARIFICATION) {
    return render_clarification_question(st, ast.opt_a(), ast.opt_b(), lang);
  }

  const int motor_id = first_failed_motor_1based(st.failed_motor_mask);
  const int soc_int = static_cast<int>(std::lround(st.soc_pct));
  const int agl_int = static_cast<int>(std::lround(st.agl_m));
  const SlzCandidate* tgt = st.find_slz(ast.target_id());

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1);

  if (lang == LangCode::EN) {
    if (motor_id > 0) {
      oss << "Motor " << motor_id << " fault (" << fault_mode_name(st.fault_mode) << "). ";
    }
    oss << "Battery " << soc_int << "% (" << st.v_term_volts << " V), altitude " << agl_int
        << " m AGL. ";
    if (tgt != nullptr) {
      oss << "Verified diversion: " << display_slz_name(tgt->id_view()) << " ("
          << static_cast<int>(std::lround(tgt->p_safe * 100.0f)) << "% safe, "
          << static_cast<int>(std::lround(tgt->eta_s)) << " s ETA, +" << tgt->soc_margin_pct
          << "% SoC margin). Cleared outside LTFJ/LTBX.";
    } else {
      oss << "Holding safe loiter outside LTFJ/LTBX.";
    }
  } else {
    if (motor_id > 0) {
      oss << "Motor " << motor_id << " arizali (" << fault_mode_name(st.fault_mode) << "). ";
    }
    oss << "Batarya %" << soc_int << " (" << st.v_term_volts << " V), irtifa " << agl_int
        << " m AGL. ";
    if (tgt != nullptr) {
      oss << "Dogrulanmis rota: " << display_slz_name(tgt->id_view()) << " (%"
          << static_cast<int>(std::lround(tgt->p_safe * 100.0f)) << " guvenli, "
          << static_cast<int>(std::lround(tgt->eta_s)) << " sn, +%" << tgt->soc_margin_pct
          << " batarya marji). LTFJ/LTBX disi guvenli inis baslatildi.";
    } else {
      oss << "LTFJ/LTBX disinda guvenli bekleme (loiter) suruyor.";
    }
  }
  return oss.str();
}

bool verify_numeric_grounding(std::string_view text,
                              const RtaStateSnapshotV1& st,
                              double tol) noexcept {
  // Build fixed-capacity whitelist of valid snapshot telemetry values (zero heap needed via array).
  std::array<double, 96> allowed{};
  std::size_t n_allowed = 0U;
  auto push_val = [&](double v) noexcept {
    if (std::isfinite(v) && n_allowed < allowed.size()) {
      allowed[n_allowed++] = v;
    }
  };

  for (int bit = 0; bit < 8; ++bit) {
    if ((st.failed_motor_mask & static_cast<uint8_t>(1U << bit)) != 0U) {
      push_val(static_cast<double>(bit + 1));
    }
  }
  push_val(static_cast<double>(st.soc_pct));
  push_val(static_cast<double>(st.soc_reserve_floor_pct));
  push_val(static_cast<double>(st.v_term_volts));
  push_val(static_cast<double>(st.pack_nominal_volts));
  push_val(static_cast<double>(st.pack_energy_kwh));
  push_val(static_cast<double>(st.agl_m));
  push_val(static_cast<double>(st.amsl_m));
  push_val(static_cast<double>(st.groundspeed_mps));
  push_val(static_cast<double>(st.wind_mps));
  push_val(static_cast<double>(st.wind_from_deg));
  push_val(static_cast<double>(st.ams.ams_max_accel_mps2));
  push_val(static_cast<double>(st.ams.ams_max_climb_mps));
  push_val(static_cast<double>(st.ams.max_cruise_airspeed_mps));

  const std::size_t n_slz = (st.num_slz <= kMaxSlzCandidates) ? st.num_slz : kMaxSlzCandidates;
  for (std::size_t i = 0U; i < n_slz; ++i) {
    const auto& s = st.ranked_slzs[i];
    push_val(static_cast<double>(s.p_safe) * 100.0);
    push_val(static_cast<double>(s.eta_s));
    push_val(static_cast<double>(s.soc_margin_pct));
    push_val(std::abs(static_cast<double>(s.soc_margin_pct)));
    push_val(static_cast<double>(s.rank));
  }

  // Scan text for standalone numerical literals (skipping alphanumeric IDs like F3269).
  std::size_t i = 0U;
  while (i < text.size()) {
    if (is_ascii_digit(text[i])) {
      if (i > 0U && is_ascii_alpha(text[i - 1U])) {
        while (i < text.size() && (is_ascii_digit(text[i]) || is_ascii_alpha(text[i]))) {
          ++i;
        }
        continue;
      }
      const std::size_t start = i;
      while (i < text.size() && is_ascii_digit(text[i])) {
        ++i;
      }
      if (i + 1U < text.size() && text[i] == '.' && is_ascii_digit(text[i + 1U])) {
        ++i;
        while (i < text.size() && is_ascii_digit(text[i])) {
          ++i;
        }
      }
      // Copy token into small null-terminated stack buffer for std::strtod
      char buf[32]{};
      const std::size_t len = (i - start < sizeof(buf) - 1U) ? (i - start) : (sizeof(buf) - 1U);
      std::memcpy(buf, text.data() + start, len);
      char* endptr = nullptr;
      const double parsed = std::strtod(buf, &endptr);
      if (endptr == buf) {
        return false;
      }
      bool matched = false;
      for (std::size_t k = 0U; k < n_allowed; ++k) {
        if (std::abs(parsed - allowed[k]) <= tol) {
          matched = true;
          break;
        }
      }
      if (!matched) {
        return false;
      }
    } else {
      ++i;
    }
  }
  return true;
}

}  // namespace vertirta::copilot
