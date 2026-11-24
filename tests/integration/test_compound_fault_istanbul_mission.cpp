// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "vertirta/benchmarks/compound_fault_runner.hpp"

using Catch::Approx;
using namespace vertirta::benchmarks;

namespace {

std::filesystem::path make_unique_work_dir(const char* tag) {
  static std::atomic<uint64_t> seq{0U};
  const auto id = seq.fetch_add(1U, std::memory_order_relaxed);
  const auto p = std::filesystem::temp_directory_path() /
                 ("vertirta_e1_" + std::string(tag) + "_" + std::to_string(id));
  std::error_code ec;
  std::filesystem::remove_all(p, ec);
  std::filesystem::create_directories(p, ec);
  return p;
}

}  // namespace

TEST_CASE("Scenario 1 (Compound Apex): Rotor#3 + 12m/s Wind + GNSS Spoof + Grazing Wire + Part-IS + S2 MAYDAY",
          "[integration][compound_fault][scenario1]") {
  const auto wdir = make_unique_work_dir("sc1");
  const auto rep = run_scenario_1_compound_apex(wdir);

  INFO("Failure reason: " << rep.failure_reason);
  REQUIRE(rep.passed);
  REQUIRE(rep.ticks_100hz == 1000U);
  REQUIRE(rep.wind_lut_sha256_verified);
  REQUIRE(rep.peak_wind_mps >= 8.0);
  REQUIRE(rep.min_agl_m >= 15.0);
  REQUIRE(rep.max_tilt_deg < 45.0);
  REQUIRE(rep.fault_mask_hex == "0x08");
  REQUIRE(rep.ltfj_ltbx_breaches == 0U);
  REQUIRE(rep.min_ltfj_margin_m > 0.0);
  REQUIRE(rep.min_ltbx_margin_m > 0.0);
  REQUIRE(rep.gnss_hypothesis == "H3_SLOW_RAMP_SPOOF");
  REQUIRE(rep.gnss_spoof_ttd_s >= 0.0);
  REQUIRE(rep.gnss_spoof_ttd_s <= 1.45);
  REQUIRE(rep.gnss_rollback_executed);
  REQUIRE(rep.wire_grazing_rcs_dbsm <= -45.0);
  REQUIRE(rep.wire_standoff_emitted);
  REQUIRE(rep.wire_recommended_v_max_mps <= 15.0);
  REQUIRE(rep.min_wire_clearance_m >= 6.0);
  REQUIRE(rep.part_is_spoof_rejected);
  REQUIRE(rep.part_is_cyber_state == "SEC_USSP_QUARANTINE");
  REQUIRE(rep.voice_action == "EMERGENCY_PREEMPT");
  REQUIRE(rep.voice_asr_tier == "WHISPER_LARGE_V3_TURBO_Q8_0");
  REQUIRE(rep.voice_tts_ttfa_ms < 250.0);
  REQUIRE(rep.vrta_frames_verified == 1000U);
  REQUIRE(rep.sidecar_events_joined == 4U);
  REQUIRE(rep.e5x_xml_generated);

  std::error_code ec;
  std::filesystem::remove_all(wdir, ec);
}

TEST_CASE("Scenario 2 (CEL + Fleet P0 Preemption + ED-155A/E5X): <500ms clear, SCF-NP, Barrier#8=0, ID 431 draft, 0% hallucination",
          "[integration][compound_fault][scenario2]") {
  const auto wdir = make_unique_work_dir("sc2");
  const auto rep = run_scenario_2_cel_fleet_forensics(wdir);

  INFO("Failure reason: " << rep.failure_reason);
  REQUIRE(rep.passed);
  REQUIRE(rep.cel_latched);
  REQUIRE(rep.fault_mask_hex == "0x11");
  REQUIRE(rep.fleet_agents == 20U);
  REQUIRE(rep.fleet_preemption_escape_ms < 500.0);
  REQUIRE(rep.fleet_airspace_breaches == 0U);
  REQUIRE(rep.ltfj_ltbx_breaches == 0U);
  REQUIRE(rep.vrta_frames_verified == 300U);
  REQUIRE(rep.sidecar_events_joined == 3U);
  REQUIRE(rep.ercs_cictt == "SCF-NP");
  REQUIRE(rep.ercs_barrier8_score == 0U);
  REQUIRE(rep.ercs_draft_incident_only);
  REQUIRE(rep.e5x_xml_generated);
  REQUIRE(rep.narrative_hallucinated_numbers == 0U);

  // Verify emitted E5X XML structural invariants directly on disk
  std::ifstream ifs(rep.e5x_xml_path);
  REQUIRE(ifs.is_open());
  const std::string xml((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  REQUIRE(xml.find("<Occurrence entityId=\"24\"") != std::string::npos);
  REQUIRE(xml.find("SCF-NP") != std::string::npos);
  REQUIRE(xml.find("INCIDENT_DRAFT_ONLY") != std::string::npos);
  REQUIRE(xml.find("DRAFT_INTERNAL_PRE_SUBMISSION") != std::string::npos);
  // Article 16 PII must be stripped from Headline ID 601
  REQUIRE(xml.find("ali.demir@example.org") == std::string::npos);
  REQUIRE(xml.find("Ali Demir") == std::string::npos);
  REQUIRE(xml.find("TC-VRT02") == std::string::npos);
  // Strictly ascending attribute IDs inside <ATTRIBUTES>
  const auto a0 = xml.find("<ATTRIBUTES>");
  const auto a1 = xml.find("</ATTRIBUTES>", a0);
  REQUIRE(a0 != std::string::npos);
  REQUIRE(a1 != std::string::npos);
  const std::string block = xml.substr(a0, a1 - a0);
  int prev_id = -1;
  std::size_t pos = 0U;
  int attr_count = 0;
  while ((pos = block.find("id=\"", pos)) != std::string::npos) {
    pos += 4U;
    const auto qend = block.find('"', pos);
    REQUIRE(qend != std::string::npos);
    const int cur = std::stoi(block.substr(pos, qend - pos));
    REQUIRE(cur > prev_id);
    prev_id = cur;
    ++attr_count;
    pos = qend + 1U;
  }
  REQUIRE(attr_count >= 12);

  std::error_code ec;
  std::filesystem::remove_all(wdir, ec);
}

TEST_CASE("Scenario 3 (Marmara Wave Glint + Spoof-vs-Jam + Visual Reloc): 0 glint FAR, <=1.4s spoof TTD, 3sigma < 25m, 0 CTR breaches",
          "[integration][compound_fault][scenario3]") {
  const auto wdir = make_unique_work_dir("sc3");
  const auto rep = run_scenario_3_marmara_glint_spoof_reloc(wdir);

  INFO("Failure reason: " << rep.failure_reason);
  REQUIRE(rep.passed);
  REQUIRE(rep.ticks_100hz == 2000U);
  REQUIRE(rep.gnss_water_glint_false_alarms == 0U);
  REQUIRE(rep.gnss_jam_discriminated);
  REQUIRE(rep.gnss_hypothesis == "H3_SLOW_RAMP_SPOOF");
  REQUIRE(rep.gnss_spoof_ttd_s >= 0.0);
  REQUIRE(rep.gnss_spoof_ttd_s <= 1.45);
  REQUIRE(rep.gnss_rollback_executed);
  REQUIRE(rep.reloc_accepted_count >= 5U);
  REQUIRE(rep.reloc_sentinel2_rejected_count >= 5U);
  REQUIRE(rep.final_3sigma_m < 25.0);
  REQUIRE(rep.nav_state == "BOUNDED_GNSS_DENIED_CRUISE");
  REQUIRE(rep.ltfj_ltbx_breaches == 0U);

  std::error_code ec;
  std::filesystem::remove_all(wdir, ec);
}

TEST_CASE("Scenario 4 (Low-SoC 4.2% < 5.0% Liveness + S3 Garbled Voice + Wire Grazing <=15 m/s Cap)",
          "[integration][compound_fault][scenario4]") {
  const auto wdir = make_unique_work_dir("sc4");
  const auto rep = run_scenario_4_low_soc_liveness_wire_cap(wdir);

  INFO("Failure reason: " << rep.failure_reason);
  REQUIRE(rep.passed);
  REQUIRE(rep.voice_snr_stratum == "S3_Garbled");
  REQUIRE(rep.voice_action == "LIVENESS_EMERGENCY_COMMIT");
  REQUIRE(rep.voice_tts_ttfa_ms < 250.0);
  REQUIRE(rep.wire_grazing_rcs_dbsm <= -45.0);
  REQUIRE(rep.wire_standoff_emitted);
  REQUIRE(rep.wire_recommended_v_max_mps <= 15.0);
  REQUIRE(rep.max_speed_mps <= 15.25);
  REQUIRE(rep.min_wire_clearance_m >= 6.0);
  REQUIRE(rep.ltfj_ltbx_breaches == 0U);

  std::error_code ec;
  std::filesystem::remove_all(wdir, ec);
}

TEST_CASE("Scenario 5 (Lift+Cruise 8+1 Hover-OEI + Transition Gust + Kartal Hospital Cylinder Avoidance)",
          "[integration][compound_fault][scenario5]") {
  const auto wdir = make_unique_work_dir("sc5");
  const auto rep = run_scenario_5_lift_cruise_oei_gust_hospital(wdir);

  INFO("Failure reason: " << rep.failure_reason);
  REQUIRE(rep.passed);
  REQUIRE(rep.airframe == "VertiRTA-LC-240-8p1");
  REQUIRE(rep.wind_lut_sha256_verified);
  REQUIRE(rep.oei_trim_feasible);
  REQUIRE(rep.oei_peak_thrust_ratio <= 0.793);
  REQUIRE(rep.max_speed_mps >= 28.0);
  REQUIRE(rep.min_agl_m >= 60.0);
  REQUIRE(rep.hospital_cylinder_breaches == 0U);
  REQUIRE(rep.min_hospital_horiz_dist_m >= 160.0);
  REQUIRE(rep.min_hospital_horiz_dist_m < 260.0);
  REQUIRE(rep.final_soc_pct > 50.0);
  REQUIRE(rep.final_soc_pct < 80.0);
  REQUIRE(rep.min_v_term_v > 350.0);
  REQUIRE(rep.max_kartal_hospital_dba > 10.0);
  REQUIRE(rep.ltfj_ltbx_breaches == 0U);

  std::error_code ec;
  std::filesystem::remove_all(wdir, ec);
}

TEST_CASE("Full 5-Scenario Compound Suite serializes pure UTF-8 LF JSON with SHA-256 and zero NaN/Inf",
          "[integration][compound_fault][suite_json]") {
  const auto wdir = make_unique_work_dir("suite");
  const auto suite = run_all_compound_scenarios(wdir);
  REQUIRE(suite.all_passed);
  REQUIRE(suite.scenarios_passed == 5U);
  REQUIRE(suite.scenarios_total == 5U);
  REQUIRE(suite.report_sha256.size() == 64U);

  const auto out_json = wdir / "compound_fault_mission_report.json";
  REQUIRE(write_compound_suite_json(suite, out_json));
  std::ifstream ifs(out_json, std::ios::binary);
  const std::string raw((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  REQUIRE(raw.find('\r') == std::string::npos);
  REQUIRE(raw.find("nan") == std::string::npos);
  REQUIRE(raw.find("NaN") == std::string::npos);
  REQUIRE(raw.find("inf") == std::string::npos);
  REQUIRE(raw.find("vertirta.compound_fault_mission_report/v4.0.0") != std::string::npos);
  REQUIRE(raw.find(suite.report_sha256) != std::string::npos);

  std::error_code ec;
  std::filesystem::remove_all(wdir, ec);
}
