// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vertirta::benchmarks {

// Telemetry & safety-gate report for one closed-loop 100 Hz compound-fault Istanbul corridor scenario.
struct CompoundScenarioReport {
  uint32_t scenario_id{0U};
  std::string scenario_name{};
  std::string airframe{"VertiRTA-X8-240"};
  bool passed{false};
  std::string failure_reason{};

  double duration_s{0.0};
  uint32_t ticks_100hz{0U};
  double wall_ms{0.0};

  // 6-DOF flight dynamics, INDI-WLS, and capability envelope
  double min_agl_m{0.0};
  double max_tilt_deg{0.0};
  double max_speed_mps{0.0};
  double final_soc_pct{0.0};
  double min_v_term_v{0.0};
  std::string fault_mask_hex{"0x00"};
  bool cel_latched{false};
  bool oei_trim_feasible{true};
  double oei_peak_thrust_ratio{0.0};

  // Istanbul coastal airspace, VRTAWND1 wind ROM, and Kartal Hospital acoustics
  double min_ltfj_margin_m{0.0};
  double min_ltbx_margin_m{0.0};
  uint32_t ltfj_ltbx_breaches{0U};
  double peak_wind_mps{0.0};
  bool wind_lut_sha256_verified{false};
  double min_hospital_horiz_dist_m{0.0};
  uint32_t hospital_cylinder_breaches{0U};
  double max_kartal_hospital_dba{0.0};

  // 77 GHz mmWave + ForwardHazardFusion wire grazing & velocity clamp
  double wire_grazing_rcs_dbsm{0.0};
  bool wire_standoff_emitted{false};
  double wire_recommended_v_max_mps{22.0};
  uint32_t active_wire_capsules{0U};
  double min_wire_clearance_m{0.0};

  // GNSS-denied anti-spoofing, Marmara glint immunity, and visual reloc covariance
  std::string gnss_hypothesis{"H0_NOMINAL"};
  double gnss_spoof_ttd_s{-1.0};
  uint32_t gnss_water_glint_false_alarms{0U};
  bool gnss_rollback_executed{false};
  bool gnss_jam_discriminated{false};
  uint32_t reloc_accepted_count{0U};
  uint32_t reloc_sentinel2_rejected_count{0U};
  double final_3sigma_m{0.0};
  std::string nav_state{"BOUNDED_GNSS_DENIED_CRUISE"};

  // Part-IS (EU) 2023/203 cyber resilience & U-space P0 CEL fleet preemption
  bool part_is_spoof_rejected{false};
  std::string part_is_cyber_state{"SEC_NOMINAL"};
  uint32_t fleet_agents{0U};
  double fleet_preemption_escape_ms{0.0};
  uint32_t fleet_airspace_breaches{0U};

  // Cockpit Voice ATC (Silero VAD + Whisper dual-tier + Mondrian KnowNo + Zero-GPL TTS)
  std::string voice_snr_stratum{"N/A"};
  std::string voice_action{"N/A"};
  std::string voice_asr_tier{"N/A"};
  double voice_tts_ttfa_ms{0.0};
  std::string voice_audio_sha256_prefix{};

  // EUROCAE ED-155A .vrta + .vrta_events.jsonl -> ECCAIRS2 E5X Forensics
  uint32_t vrta_frames_verified{0U};
  uint32_t sidecar_events_joined{0U};
  std::string ercs_cictt{"N/A"};
  std::string ercs_matrix_code{"N/A"};
  uint32_t ercs_barrier8_score{0U};
  bool ercs_draft_incident_only{false};
  bool e5x_xml_generated{false};
  uint32_t narrative_hallucinated_numbers{0U};
  std::string e5x_xml_path{};
};

struct CompoundSuiteReport {
  std::string schema_version{"vertirta.compound_fault_mission_report/v4.0.0"};
  std::string timestamp_utc{};
  std::string part_is_backend{};
  std::string forensics_signer_backend{};
  bool all_passed{false};
  uint32_t scenarios_passed{0U};
  uint32_t scenarios_total{5U};
  double total_wall_ms{0.0};
  std::vector<CompoundScenarioReport> scenarios{};
  std::string report_sha256{};
};

// Individual closed-loop 100 Hz Kartal-Maltepe multi-fault scenario runners.
[[nodiscard]] CompoundScenarioReport run_scenario_1_compound_apex(
    const std::filesystem::path& work_dir);

[[nodiscard]] CompoundScenarioReport run_scenario_2_cel_fleet_forensics(
    const std::filesystem::path& work_dir,
    const std::filesystem::path& optional_e5x_out_path = {});

[[nodiscard]] CompoundScenarioReport run_scenario_3_marmara_glint_spoof_reloc(
    const std::filesystem::path& work_dir);

[[nodiscard]] CompoundScenarioReport run_scenario_4_low_soc_liveness_wire_cap(
    const std::filesystem::path& work_dir);

[[nodiscard]] CompoundScenarioReport run_scenario_5_lift_cruise_oei_gust_hospital(
    const std::filesystem::path& work_dir);

// Run all 5 scenarios end-to-end and populate canonical suite metrics + SHA-256.
[[nodiscard]] CompoundSuiteReport run_all_compound_scenarios(
    const std::filesystem::path& work_dir,
    const std::filesystem::path& optional_e5x_out_path = {});

// Serialize deterministic pure UTF-8 LF JSON (NaN/Inf guarded) with SHA-256 digest.
[[nodiscard]] std::string serialize_compound_suite_json(const CompoundSuiteReport& rep);

[[nodiscard]] bool write_compound_suite_json(
    const CompoundSuiteReport& rep,
    const std::filesystem::path& out_path);

}  // namespace vertirta::benchmarks
