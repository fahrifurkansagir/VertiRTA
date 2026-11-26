// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "vertirta/core/vrta_recorder.hpp"
#include "vertirta/forensics/eccairs2_e5x_emitter.hpp"
#include "vertirta/forensics/ercs_scorer.hpp"
#include "vertirta/forensics/forensics_pipeline.hpp"
#include "vertirta/forensics/narrative_grounding.hpp"
#include "vertirta/forensics/vrta_events_sidecar.hpp"

using namespace vertirta::forensics;

TEST_CASE("Frozen 256 B VRTA0001 wire format stays untouched while .vrta_events.jsonl chains (seq, t_sim_us) with SHA-256 + Ed25519") {
  static_assert(sizeof(vertirta::core::VrtaFileHeader) == 128U);
  static_assert(sizeof(vertirta::core::VrtaFrame100Hz) == 256U);

  VrtaEventsSidecarWriter writer;
  REQUIRE(writer.public_key_hex().size() == 64U);

  SidecarEvent ev1{};
  ev1.seq = 1200U;
  ev1.t_sim_us = 12'000'000U;
  ev1.kind = SidecarEventKind::VOICE_CONFORMAL_TURN;
  ev1.snr_stratum = "S2_Hover";
  ev1.conformal_set_size = 1U;
  ev1.asr_transcript_utf8 = "Kartal Sahil'e acil inis";
  ev1.audio_sha256_hex = std::string(64U, 'a');
  ev1.z3_smt2_sha256_hex = std::string(64U, 'b');
  ev1.gsn_node_id = "G3.2-VOICE-CONFORMAL";

  const std::string line1 = writer.append_serialized(ev1);
  SidecarEvent ev2 = ev1;
  ev2.seq = 1250U;
  ev2.t_sim_us = 12'500'000U;
  ev2.kind = SidecarEventKind::Z3_UNSAT_REPAIR;
  ev2.z3_unsat_core = {"LTFJ_CTR_WALL", "SOC_RESERVE_FLOOR"};
  const std::string line2 = writer.append_serialized(ev2);

  const auto verified =
      VrtaEventsSidecarReader::verify_chain(line1 + "\n" + line2 + "\n", writer.public_key_hex());
  REQUIRE(verified.valid);
  REQUIRE(verified.events.size() == 2U);
  REQUIRE(verified.events[0].prev_event_sha256 == std::string(64U, '0'));
  REQUIRE(verified.events[1].prev_event_sha256 == verified.events[0].event_sha256);
  REQUIRE(verified.events[1].z3_unsat_core.size() == 2U);
  REQUIRE(verified.events[1].z3_unsat_core[0] == "LTFJ_CTR_WALL");

  // Tampering a single byte must break chain verification
  std::string tampered = line1 + "\n" + line2 + "\n";
  const std::size_t pos = tampered.find("1250");
  REQUIRE(pos != std::string::npos);
  tampered[pos] = '9';
  REQUIRE_FALSE(VrtaEventsSidecarReader::verify_chain(tampered, writer.public_key_hex()).valid);

  // Non-monotonic (seq, t_sim_us) rollback must fail verification
  SidecarEvent ev_back = ev2;
  ev_back.seq = 1249U;
  ev_back.t_sim_us = 12'490'000U;
  const std::string line3 = writer.append_serialized(ev_back);
  REQUIRE_FALSE(VrtaEventsSidecarReader::verify_chain(
                    line1 + "\n" + line2 + "\n" + line3 + "\n", writer.public_key_hex())
                    .valid);
}

TEST_CASE("Sidecar .vrta_events.jsonl round-trips ASCII control characters via \\uXXXX escape decoding and rejects malformed \\u sequences") {
  VrtaEventsSidecarWriter writer;

  SidecarEvent ev{};
  ev.seq = 100U;
  ev.t_sim_us = 1'000'000U;
  ev.kind = SidecarEventKind::VOICE_CONFORMAL_TURN;
  ev.snr_stratum = "S3_Garbled";
  ev.conformal_set_size = 2U;
  // Include NUL (0x00), SOH (0x01), BEL (0x07), VT (0x0b), ESC (0x1b), US (0x1f), plus standard escapes.
  ev.asr_transcript_utf8 = std::string("ATC\x00\x01\x07\x0b\x1b\x1f\"\\line\n\t", 17U);
  ev.audio_sha256_hex = std::string(64U, '1');
  ev.z3_smt2_sha256_hex = std::string(64U, '2');
  ev.z3_unsat_core = {std::string("CORE\x1b[31m_WALL", 14U), std::string("BEL\x07", 4U)};
  ev.gsn_node_id = "G3.2-CTRL-ROUNDTRIP";

  const std::string line = writer.append_serialized(ev);
  REQUIRE(line.find("\\u0000") != std::string::npos);
  REQUIRE(line.find("\\u001b") != std::string::npos);
  REQUIRE(line.find("\\u001f") != std::string::npos);

  const auto verified =
      VrtaEventsSidecarReader::verify_chain(line + "\n", writer.public_key_hex());
  REQUIRE(verified.valid);
  REQUIRE(verified.events.size() == 1U);
  REQUIRE(verified.events[0].asr_transcript_utf8 == ev.asr_transcript_utf8);
  REQUIRE(verified.events[0].z3_unsat_core == ev.z3_unsat_core);

  // Malformed \uXXXX escapes MUST fail closed during verification.
  {
    std::string bad_hex = line;
    const std::size_t p = bad_hex.find("\\u001b");
    REQUIRE(p != std::string::npos);
    bad_hex.replace(p, 6U, "\\u00ZZ");
    REQUIRE_FALSE(VrtaEventsSidecarReader::verify_chain(bad_hex + "\n", writer.public_key_hex()).valid);
  }
  {
    std::string bad_surrogate = line;
    const std::size_t p = bad_surrogate.find("\\u001b");
    REQUIRE(p != std::string::npos);
    bad_surrogate.replace(p, 6U, "\\ud800");
    REQUIRE_FALSE(
        VrtaEventsSidecarReader::verify_chain(bad_surrogate + "\n", writer.public_key_hex()).valid);
  }
}

TEST_CASE("ERCS CDR (EU) 2020/2034 enforces SCF-NP, Barrier #8 = 0 on high-energy MAC/LOC-I, and draft/Incident ID 431 ONLY") {
  ErcsInput in{};
  in.trigger = Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT;
  in.fault_domain = FaultDomain::ROTOR_ACTUATOR_OR_INVERTER;
  in.key_risk_area = KeyRiskArea::LOC_I_HIGH_ENERGY;
  in.occupant_capacity = 1U;  // Canonical X8-240 1-pax lock (NOT stale 4-6 pax!)
  in.barriers = {1, 2, 2, 1, 2, 1, 1, /*barrier8_low_energy_requested=*/3};  // Must be clamped to 0!

  const ErcsResult r = score_ercs_cdr_2020_2034(in);
  REQUIRE(r.cictt_category == "SCF-NP");       // NEVER SCF-PP
  REQUIRE(r.cictt_category != "SCF-PP");
  REQUIRE(r.effective_barrier8_score == 0U);   // High-energy LOC-I forces Barrier #8 = 0 (N/A)
  REQUIRE(r.effective_barrier_sum == 10U);     // 1+2+2+1+2+1+1 + 0 = 10 -> prob_idx 5
  REQUIRE(r.probability_index == 5U);
  REQUIRE(r.occurrence_class_id_431 == OccurrenceClassDraft::DRAFT_INCIDENT_ONLY);
  REQUIRE_FALSE(r.auto_classified_accident_or_serious_incident);
  REQUIRE(r.reporting_status == "DRAFT_INTERNAL_PRE_SUBMISSION");
  REQUIRE(r.ercs_matrix_code.size() >= 2U);

  // Verify high-energy MAC late RTA recovery (stopping barrier #6 + #7 -> sum 2 -> prob 1 -> escalation)
  ErcsInput mac_late{};
  mac_late.trigger = Cir2026Trigger::USPACE_AIRBORNE_ENCOUNTER;
  mac_late.fault_domain = FaultDomain::AIRBORNE_SEPARATION_LOSS;
  mac_late.key_risk_area = KeyRiskArea::MAC_HIGH_ENERGY;
  mac_late.occupant_capacity = 4U;  // Lift+Cruise 4-6 pax scaling
  mac_late.stopping_barrier_idx = 6U;
  mac_late.barriers = {2, 2, 2, 2, 3, 1, 1, 1};
  const ErcsResult r_mac = score_ercs_cdr_2020_2034(mac_late);
  REQUIRE(r_mac.cictt_category == "MAC");
  REQUIRE(r_mac.effective_barrier8_score == 0U);
  REQUIRE(r_mac.effective_barrier_sum == 2U);
  REQUIRE(r_mac.probability_index == 1U);
  REQUIRE(r_mac.immediate_escalation_queue);
  REQUIRE(r_mac.occurrence_class_id_431 == OccurrenceClassDraft::DRAFT_INCIDENT_ONLY);
  REQUIRE_FALSE(r_mac.auto_classified_accident_or_serious_incident);

  // Low-energy ground event permits Barrier #8 (clamped to CDR weight 1)
  ErcsInput low_e{};
  low_e.trigger = Cir2026Trigger::FATO_INCURSION;
  low_e.fault_domain = FaultDomain::FLIGHT_CONTROL_OR_SOFTWARE;
  low_e.key_risk_area = KeyRiskArea::GROUND_DAMAGE_LOW_ENERGY;
  low_e.occupant_capacity = 1U;
  low_e.barriers = {0, 0, 0, 0, 2, 1, 1, 1};
  const ErcsResult r_low = score_ercs_cdr_2020_2034(low_e);
  REQUIRE(r_low.cictt_category == "SCF-NP");
  REQUIRE(r_low.effective_barrier8_score == 1U);
  REQUIRE(r_low.effective_barrier_sum == 5U);
}

TEST_CASE("No-digit GBNF narrative grounding enforces 0.0% numeric hallucination and Article 16 PII stripping") {
  GroundedTelemetrySlots slots{};
  slots.values["EVENT_SEQ"] = "1200";
  slots.values["MIN_LTFJ_MARGIN_M"] = "42.5";
  slots.values["SOC_PCT"] = "14.2";
  slots.values["ERCS_SCORE"] = "I5";

  const std::string clean_template_en =
      "At frame {{EVENT_SEQ}}, lateral boundary margin reached {{MIN_LTFJ_MARGIN_M}} m at {{SOC_PCT}}% SoC (draft ERCS {{ERCS_SCORE}}).";
  const auto ok = render_and_verify_grounded_narrative(clean_template_en, slots);
  REQUIRE(ok.passed);
  REQUIRE(ok.hallucinated_numeric_tokens.empty());
  REQUIRE(ok.rendered_text.find("42.5") != std::string::npos);

  // Any raw digit injected outside {{SLOT}} MUST be rejected fail-closed!
  const std::string hallucinated =
      "At frame {{EVENT_SEQ}}, speed hit 99.9 kt near Kartal.";
  const auto bad = render_and_verify_grounded_narrative(hallucinated, slots);
  REQUIRE_FALSE(bad.passed);
  REQUIRE(bad.hallucinated_numeric_tokens.size() == 1U);
  REQUIRE(bad.hallucinated_numeric_tokens[0] == "99.9");

  // Unknown slot injection must also fail closed
  const auto bad_slot = render_and_verify_grounded_narrative("Value {{FAKE_SLOT}} seen.", slots);
  REQUIRE_FALSE(bad_slot.passed);
  REQUIRE(bad_slot.unknown_slots.size() == 1U);

  // Article 16 PII stripping on Headline ID 601
  const std::string headline =
      sanitize_article16_headline("Pilot Ahmet Yilmaz (ahmet@example.com) diverted TC-VRT01 near Kartal");
  REQUIRE(headline.find("ahmet@example.com") == std::string::npos);
  REQUIRE(headline.find("Ahmet Yilmaz") == std::string::npos);
  REQUIRE(headline.find("TC-VRT01") == std::string::npos);
}

TEST_CASE("End-to-end binary .vrta + .vrta_events.jsonl pipeline emits strictly ascending E5X XML omitting N/A injuries") {
  namespace fs = std::filesystem;
  const fs::path tmp_dir = fs::temp_directory_path() / "vertirta_forensics_unit_e2e";
  fs::create_directories(tmp_dir);
  const fs::path vrta_path = tmp_dir / "flight.vrta";
  const fs::path sidecar_path = tmp_dir / "flight.vrta_events.jsonl";
  const fs::path e5x_path = tmp_dir / "occurrence_draft.e5x.xml";
  const fs::path audit_path = tmp_dir / "forensics_audit_report.json";

  // Write valid frozen 128 B header + 3 valid 256 B VRTA0001 frames
  {
    std::ofstream ofs(vrta_path, std::ios::binary | std::ios::trunc);
    REQUIRE(ofs.is_open());
    const auto hdr = vertirta::core::make_default_vrta_header("b124237000000000000000000000000000000000");
    ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    for (uint64_t i = 0U; i < 3U; ++i) {
      vertirta::core::VrtaFrame100Hz fr{};
      fr.seq = 1200U + i * 50U;
      fr.t_sim_us = 12'000'000U + i * 500'000U;
      fr.soc_pct = 14.5f - static_cast<float>(i) * 0.3f;
      fr.v_term_v = 368.4f;
      fr.ltfj_margin_m = 48.2f - static_cast<float>(i) * 2.5f;
      fr.ltbx_margin_m = 215.0f;
      fr.agl_m = 62.0f;
      fr.chi2_residual = 18.4f;
      fr.active_motor_mask = 0xFEU;  // 1M single-rotor loss
      fr.flight_regime = 1U;
      vertirta::core::finalize_vrta_frame(fr);
      ofs.write(reinterpret_cast<const char*>(&fr), sizeof(fr));
    }
  }

  // Write ED-155A append-only sidecar joined on (seq=1250, t_sim_us=12500000)
  VrtaEventsSidecarWriter writer;
  SidecarEvent ev{};
  ev.seq = 1250U;
  ev.t_sim_us = 12'500'000U;
  ev.kind = SidecarEventKind::RTA_ENVELOPE_PROTECTION;
  ev.snr_stratum = "S1_Cruise";
  ev.conformal_set_size = 1U;
  ev.asr_transcript_utf8 = "Kartal Sahil divert approved";
  ev.audio_sha256_hex = std::string(64U, 'c');
  ev.z3_smt2_sha256_hex = std::string(64U, 'd');
  ev.gsn_node_id = "G2.1-CBVF-LTFJ";
  REQUIRE(writer.append_to_file(sidecar_path, ev, /*fsync_after_write=*/true));

  ForensicsPipelineRequest req{};
  req.vrta_path = vrta_path;
  req.sidecar_path = sidecar_path;
  req.sidecar_public_key_hex = writer.public_key_hex();
  req.out_e5x_xml_path = e5x_path;
  req.out_audit_json_path = audit_path;
  req.raw_headline = "Pilot Mehmet Kaya (mehmet@ops.tr) single-rotor RTA envelope protection near Kartal";
  req.ercs_input.trigger = Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT;
  req.ercs_input.fault_domain = FaultDomain::ROTOR_ACTUATOR_OR_INVERTER;
  req.ercs_input.key_risk_area = KeyRiskArea::LOC_I_HIGH_ENERGY;
  req.ercs_input.occupant_capacity = 1U;
  req.ercs_input.barriers = {1, 2, 2, 1, 2, 1, 1, 1};

  const ForensicsPipelineResult res = run_forensics_pipeline(req);
  REQUIRE(res.ok);
  REQUIRE(res.frames_verified == 3U);
  REQUIRE(res.sidecar_events_joined == 1U);
  REQUIRE(res.ercs.cictt_category == "SCF-NP");
  REQUIRE(res.ercs.effective_barrier8_score == 0U);
  REQUIRE(res.ercs.occurrence_class_id_431 == OccurrenceClassDraft::DRAFT_INCIDENT_ONLY);
  REQUIRE(fs::exists(e5x_path));
  REQUIRE(fs::exists(audit_path));

  // Verify XML structure: strictly ascending <Attribute id="...">, PII stripped, N/A injury IDs omitted
  std::ifstream xml_ifs(e5x_path);
  const std::string xml((std::istreambuf_iterator<char>(xml_ifs)), std::istreambuf_iterator<char>());
  REQUIRE(xml.find("<Occurrence entityId=\"24\"") != std::string::npos);
  REQUIRE(xml.find("DRAFT_INTERNAL_PRE_SUBMISSION") != std::string::npos);
  REQUIRE(xml.find("INCIDENT_DRAFT_ONLY") != std::string::npos);
  REQUIRE(xml.find("SCF-NP") != std::string::npos);
  REQUIRE(xml.find("language=\"en\"") != std::string::npos);
  REQUIRE(xml.find("language=\"tr\"") != std::string::npos);
  REQUIRE(xml.find("mehmet@ops.tr") == std::string::npos);
  REQUIRE(xml.find("Mehmet Kaya") == std::string::npos);
  // N/A injury count IDs must NOT be zero-filled
  for (const char* na_id : {"id=\"114\"", "id=\"187\"", "id=\"262\"", "id=\"460\"", "id=\"462\"", "id=\"463\"", "id=\"469\"", "id=\"472\""}) {
    REQUIRE(xml.find(na_id) == std::string::npos);
  }

  // Strictly ascending attribute IDs inside Occurrence <ATTRIBUTES>
  const std::size_t attr_start = xml.find("<ATTRIBUTES>");
  const std::size_t attr_end = xml.find("</ATTRIBUTES>", attr_start);
  REQUIRE(attr_start != std::string::npos);
  REQUIRE(attr_end != std::string::npos);
  const std::string attr_block = xml.substr(attr_start, attr_end - attr_start);
  int prev_id = -1;
  int count = 0;
  std::size_t scan_pos = 0U;
  while ((scan_pos = attr_block.find("id=\"", scan_pos)) != std::string::npos) {
    scan_pos += 4U;
    const std::size_t quote_end = attr_block.find('"', scan_pos);
    REQUIRE(quote_end != std::string::npos);
    const int curr = std::stoi(attr_block.substr(scan_pos, quote_end - scan_pos));
    REQUIRE(curr > prev_id);
    prev_id = curr;
    ++count;
    scan_pos = quote_end + 1U;
  }
  REQUIRE(count >= 12);

  fs::remove_all(tmp_dir);
}
