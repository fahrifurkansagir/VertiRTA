// SPDX-License-Identifier: Apache-2.0
#include "vertirta/forensics/forensics_pipeline.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

#include "vertirta/core/vrta_recorder.hpp"
#include "vertirta/forensics/eccairs2_e5x_emitter.hpp"
#include "vertirta/forensics/vrta_events_sidecar.hpp"

namespace vertirta::forensics {

namespace {

std::string fmt_fixed(double val, int decimals) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.*f", decimals, val);
  return std::string(buf);
}

std::string fmt_hex_u8(uint8_t val) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "0x%02X", static_cast<unsigned int>(val));
  return std::string(buf);
}

std::string flight_regime_name(uint8_t regime) {
  switch (regime) {
    case 0U:
      return "NOMINAL";
    case 1U:
      return "SINGLE_LOSS";
    case 2U:
      return "CROSS_DUAL";
    case 3U:
      return "SAME_ARM_ACTIVE_SPIN";
    default:
      return "CRITICAL_DESCENT";
  }
}

std::string json_escape_simple(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 8U);
  for (const char ch : raw) {
    if (ch == '"') {
      out += "\\\"";
    } else if (ch == '\\') {
      out += "\\\\";
    } else if (ch == '\n') {
      out += "\\n";
    } else {
      out.push_back(ch);
    }
  }
  return out;
}

}  // namespace

ForensicsPipelineResult run_forensics_pipeline(const ForensicsPipelineRequest& req) {
  ForensicsPipelineResult res{};

  // 1) Read and verify frozen VRTA0001 binary file (128 B header + N * 256 B frames).
  std::ifstream vrta_ifs(req.vrta_path, std::ios::binary);
  if (!vrta_ifs.is_open()) {
    res.error_message = "cannot open .vrta file: " + req.vrta_path.string();
    return res;
  }
  const std::vector<uint8_t> vrta_bytes(
      (std::istreambuf_iterator<char>(vrta_ifs)), std::istreambuf_iterator<char>());
  if (vrta_bytes.size() < sizeof(vertirta::core::VrtaFileHeader)) {
    res.error_message = ".vrta file shorter than 128 B header";
    return res;
  }
  const std::size_t payload_bytes = vrta_bytes.size() - sizeof(vertirta::core::VrtaFileHeader);
  if ((payload_bytes % sizeof(vertirta::core::VrtaFrame100Hz)) != 0U || payload_bytes == 0U) {
    res.error_message = ".vrta payload size is not a positive multiple of 256 B VrtaFrame100Hz";
    return res;
  }

  vertirta::core::VrtaFileHeader hdr{};
  std::memcpy(&hdr, vrta_bytes.data(), sizeof(hdr));
  if (!vertirta::core::verify_vrta_header(hdr)) {
    res.error_message = "invalid .vrta header magic/schema/CRC-32C";
    return res;
  }

  const std::size_t n_frames = payload_bytes / sizeof(vertirta::core::VrtaFrame100Hz);
  std::vector<vertirta::core::VrtaFrame100Hz> frames(n_frames);
  for (std::size_t i = 0U; i < n_frames; ++i) {
    std::memcpy(
        &frames[i],
        vrta_bytes.data() + sizeof(hdr) + i * sizeof(vertirta::core::VrtaFrame100Hz),
        sizeof(vertirta::core::VrtaFrame100Hz));
    if (!vertirta::core::verify_vrta_frame(frames[i])) {
      res.error_message = "corrupt CRC-32C in .vrta frame index " + std::to_string(i);
      return res;
    }
  }
  res.frames_verified = n_frames;
  res.vrta_sha256_hex = sha256_hex_bytes(vrta_bytes.data(), vrta_bytes.size());

  // 2) Verify append-only .vrta_events.jsonl sidecar and join on (seq, t_sim_us).
  const auto sidecar_ver =
      VrtaEventsSidecarReader::verify_file(req.sidecar_path, req.sidecar_public_key_hex);
  if (!sidecar_ver.valid) {
    res.error_message = "sidecar chain verification failed: " + sidecar_ver.error_message;
    return res;
  }
  const auto join_ver =
      VrtaEventsSidecarReader::join_with_vrta_frames(frames, sidecar_ver.events);
  if (!join_ver.valid) {
    res.error_message = "sidecar-to-.vrta frame join failed: " + join_ver.error_message;
    return res;
  }
  res.sidecar_events_joined = join_ver.matched_events;

  // 3) Deterministically extract grounded telemetry slots directly from verified .vrta frames.
  double min_ltfj_m = std::numeric_limits<double>::infinity();
  double min_ltbx_m = std::numeric_limits<double>::infinity();
  double min_soc_pct = std::numeric_limits<double>::infinity();
  double min_vterm_v = std::numeric_limits<double>::infinity();
  double peak_chi2 = 0.0;
  std::size_t key_idx = 0U;

  for (std::size_t i = 0U; i < frames.size(); ++i) {
    const auto& fr = frames[i];
    if (static_cast<double>(fr.ltfj_margin_m) < min_ltfj_m) {
      min_ltfj_m = static_cast<double>(fr.ltfj_margin_m);
      key_idx = i;
    }
    min_ltbx_m = std::min(min_ltbx_m, static_cast<double>(fr.ltbx_margin_m));
    min_soc_pct = std::min(min_soc_pct, static_cast<double>(fr.soc_pct));
    min_vterm_v = std::min(min_vterm_v, static_cast<double>(fr.v_term_v));
    peak_chi2 = std::max(peak_chi2, static_cast<double>(fr.chi2_residual));
  }
  const auto& key_frame = frames[key_idx];

  // 4) Score CDR (EU) 2020/2034 ERCS (SCF-NP, Barrier #8 = 0 on high energy, draft ID 431 ONLY).
  res.ercs = score_ercs_cdr_2020_2034(req.ercs_input);

  const uint32_t phase_code = (key_frame.cel_latched != 0U) ? 30704U : 10602U;
  std::string snr_stratum = "S0_Clean";
  uint32_t conf_set_size = 1U;
  if (!sidecar_ver.events.empty()) {
    snr_stratum = sidecar_ver.events.back().snr_stratum;
    conf_set_size = sidecar_ver.events.back().conformal_set_size;
  }

  GroundedTelemetrySlots slots{};
  slots.values["EVENT_SEQ"] = std::to_string(key_frame.seq);
  slots.values["T_SIM_S"] = fmt_fixed(static_cast<double>(key_frame.t_sim_us) * 1e-6, 2);
  slots.values["MIN_LTFJ_MARGIN_M"] = fmt_fixed(min_ltfj_m, 1);
  slots.values["MIN_LTBX_MARGIN_M"] = fmt_fixed(min_ltbx_m, 1);
  slots.values["SOC_PCT"] = fmt_fixed(min_soc_pct, 1);
  slots.values["V_TERM_V"] = fmt_fixed(min_vterm_v, 1);
  slots.values["AGL_M"] = fmt_fixed(static_cast<double>(key_frame.agl_m), 1);
  slots.values["PEAK_CHI2"] = fmt_fixed(peak_chi2, 1);
  slots.values["ROTOR_MASK_HEX"] = fmt_hex_u8(key_frame.active_motor_mask);
  slots.values["SNR_STRATUM"] = snr_stratum;
  slots.values["CONFORMAL_SET_SIZE"] = std::to_string(conf_set_size);
  slots.values["ERCS_SCORE"] = res.ercs.ercs_matrix_code;
  slots.values["CICTT_CODE"] = res.ercs.cictt_category;
  slots.values["OCCURRENCE_PHASE"] = std::to_string(phase_code);
  slots.values["FLIGHT_REGIME"] = flight_regime_name(key_frame.flight_regime);
  res.grounded_slots = slots;

  // 5) Render and verify zero-digit EN & TR narratives (0.0% numeric hallucination).
  const std::string tmpl_en =
      req.narrative_template_en.empty()
          ? "At frame {{EVENT_SEQ}} ({{T_SIM_S}} s), regime {{FLIGHT_REGIME}} recorded minimum "
            "LTFJ clearance {{MIN_LTFJ_MARGIN_M}} m and LTBX clearance {{MIN_LTBX_MARGIN_M}} m "
            "at {{AGL_M}} m AGL, {{SOC_PCT}}% SoC, {{V_TERM_V}} V, peak chi-square {{PEAK_CHI2}}, "
            "rotor mask {{ROTOR_MASK_HEX}}; voice stratum {{SNR_STRATUM}} set size "
            "{{CONFORMAL_SET_SIZE}}; category {{CICTT_CODE}}, draft ERCS {{ERCS_SCORE}}."
          : req.narrative_template_en;
  const std::string tmpl_tr =
      req.narrative_template_tr.empty()
          ? "{{EVENT_SEQ}} numarali cercevede ({{T_SIM_S}} sn), {{FLIGHT_REGIME}} rejiminde en "
            "kucuk LTFJ mesafesi {{MIN_LTFJ_MARGIN_M}} m, LTBX mesafesi {{MIN_LTBX_MARGIN_M}} m, "
            "AGL {{AGL_M}} m, batarya {{SOC_PCT}}% ({{V_TERM_V}} V), tepe ki-kare {{PEAK_CHI2}}, "
            "rotor maskesi {{ROTOR_MASK_HEX}} olculdu; ses katmani {{SNR_STRATUM}} kume boyutu "
            "{{CONFORMAL_SET_SIZE}}; kategori {{CICTT_CODE}}, taslak ERCS {{ERCS_SCORE}}."
          : req.narrative_template_tr;

  const auto en_res = render_and_verify_grounded_narrative(tmpl_en, slots);
  if (!en_res.passed) {
    res.error_message = "EN narrative grounding failed: " + en_res.error_message;
    return res;
  }
  const auto tr_res = render_and_verify_grounded_narrative(tmpl_tr, slots);
  if (!tr_res.passed) {
    res.error_message = "TR narrative grounding failed: " + tr_res.error_message;
    return res;
  }

  // 6) Emit draft XSD-valid ECCAIRS2 / E5X XML and audit report JSON.
  E5xOccurrenceDraft draft{};
  draft.file_number = req.file_number;
  draft.utc_date_time_id_477 = req.utc_date_time;
  draft.local_date_time_id_433 = req.local_date_time;
  draft.evtol_flight_phase_id_121 = phase_code;
  draft.headline_raw = req.raw_headline;
  draft.narrative_en = en_res.rendered_text;
  draft.narrative_tr = tr_res.rendered_text;
  draft.cir_trigger = req.ercs_input.trigger;
  draft.ercs = res.ercs;
  draft.vrta_file_sha256 = res.vrta_sha256_hex;
  draft.sidecar_chain_verified = true;
  draft.numeric_grounding_verified = true;

  const std::string xml = emit_eccairs2_e5x_xml(draft);
  {
    std::ofstream xml_ofs(req.out_e5x_xml_path, std::ios::binary | std::ios::trunc);
    if (!xml_ofs.is_open()) {
      res.error_message = "cannot write E5X XML output: " + req.out_e5x_xml_path.string();
      return res;
    }
    xml_ofs.write(xml.data(), static_cast<std::streamsize>(xml.size()));
    if (!xml_ofs.good()) {
      res.error_message = "failed writing E5X XML output";
      return res;
    }
  }

  if (!req.out_audit_json_path.empty()) {
    std::ostringstream js;
    js << "{\n"
       << "  \"schema_version\": \"vertirta.forensics_audit/v1.0.0\",\n"
       << "  \"compliance_claim\": \"NONE_RESEARCH_TESTBED_ONLY\",\n"
       << "  \"reporting_status\": \"" << res.ercs.reporting_status << "\",\n"
       << "  \"signer_backend\": \"" << to_string(active_sidecar_signer_backend()) << "\",\n"
       << "  \"frames_verified\": " << res.frames_verified << ",\n"
       << "  \"sidecar_events_joined\": " << res.sidecar_events_joined << ",\n"
       << "  \"vrta_sha256\": \"" << res.vrta_sha256_hex << "\",\n"
       << "  \"e5x_xml_sha256\": \"" << sha256_hex_string(xml) << "\",\n"
       << "  \"cictt_category\": \"" << res.ercs.cictt_category << "\",\n"
       << "  \"ercs_matrix_code\": \"" << res.ercs.ercs_matrix_code << "\",\n"
       << "  \"effective_barrier8_score\": " << static_cast<unsigned int>(res.ercs.effective_barrier8_score) << ",\n"
       << "  \"auto_classified_accident_or_serious_incident\": false,\n"
       << "  \"numeric_hallucination_count\": 0,\n"
       << "  \"sanitized_headline\": \"" << json_escape_simple(sanitize_article16_headline(req.raw_headline)) << "\"\n"
       << "}\n";
    const std::string json_text = js.str();
    std::ofstream json_ofs(req.out_audit_json_path, std::ios::binary | std::ios::trunc);
    if (!json_ofs.is_open()) {
      res.error_message = "cannot write audit JSON: " + req.out_audit_json_path.string();
      return res;
    }
    json_ofs.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
    if (!json_ofs.good()) {
      res.error_message = "failed writing audit JSON";
      return res;
    }
  }

  res.ok = true;
  return res;
}

}  // namespace vertirta::forensics
