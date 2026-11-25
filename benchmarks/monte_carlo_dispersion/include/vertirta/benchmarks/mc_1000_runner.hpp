// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vertirta::benchmarks {

enum class AirframeKind : uint8_t {
  kCoaxialX8_240 = 0,
  kLiftCruise_240_8p1 = 1,
};

enum class FaultRegimeKind : uint8_t {
  kNominalTurbulent = 0,
  kSingleRotorLossCsfl = 1,
  kSameArmDualKillCel = 2,
  kGnssSlowRampSpoof = 3,
  kCompoundLowSocWindFault = 4,
};

constexpr std::size_t kNumAirframes = 2U;
constexpr std::size_t kNumRegimes = 5U;
constexpr std::size_t kStressGridMassLevels = 6U;
constexpr std::size_t kStressGridWindLevels = 6U;

[[nodiscard]] const char* airframe_name(AirframeKind kind) noexcept;
[[nodiscard]] const char* regime_name(FaultRegimeKind kind) noexcept;

struct FlightSampleConfig {
  uint32_t flight_id{0U};
  uint64_t seed{20261125ULL};
  AirframeKind airframe{AirframeKind::kCoaxialX8_240};
  FaultRegimeKind regime{FaultRegimeKind::kNominalTurbulent};

  double mass_kg{240.0};
  double cg_x_m{0.0};
  double cg_y_m{0.0};
  double wind_speed_mps{6.0};
  double wind_dir_rad{0.0};
  double gust_sigma_scale{1.0};
  double init_agl_m{55.0};
  double init_east_m{-200.0};
  double init_north_m{0.0};
  double init_vx_mps{5.0};
  double slz_offset_n_m{85.0};
  double init_soc{0.72};
  double fault_time_s{5.0};
  uint8_t fault_index{0U};  // Rotor 0..7 for OEI, arm 0..3 for X8 CEL, inner-boom 0..1 for LC dual
  double spoof_rate_mps{0.0};
  double imu_gyro_sigma_rads{0.012};
  double alt_noise_sigma_m{0.22};
  double max_flight_s{75.0};
};

struct FlightOutcome {
  uint32_t flight_id{0U};
  AirframeKind airframe{AirframeKind::kCoaxialX8_240};
  FaultRegimeKind regime{FaultRegimeKind::kNominalTurbulent};

  double mass_kg{240.0};
  double cg_offset_m{0.0};
  double wind_speed_mps{0.0};
  double init_agl_m{0.0};
  double init_soc{0.0};

  bool airspace_breach{false};
  bool ltfj_breach{false};
  bool ltbx_breach{false};
  bool hospital_breach{false};
  double min_ltfj_margin_m{1e6};
  double min_ltbx_margin_m{1e6};
  double min_hospital_dist_m{1e6};

  double touchdown_dx_m{0.0};  // North offset from commanded SLZ aimpoint [m]
  double touchdown_dy_m{0.0};  // East offset from commanded SLZ aimpoint [m]
  double touchdown_error_m{0.0};
  double touchdown_vz_mps{0.0};
  double min_pack_voltage_v{403.2};
  double final_soc{0.0};
  double max_tilt_deg{0.0};
  double max_wind_mps{0.0};
  double flight_duration_s{0.0};

  bool rta_intervened{false};
  bool cel_latched{false};
  bool spoof_detected{false};
  double spoof_ttd_s{-1.0};
  bool spoof_rollback_ok{false};
  double max_3sigma_m{0.0};

  bool reached_ground{false};
  bool all_finite{true};
  bool battery_collapse{false};
  bool safe_recovery{false};
  std::string failure_mode{"NONE"};
  double wall_us{0.0};
};

struct ConfidenceInterval95 {
  uint32_t count{0U};
  uint32_t total{0U};
  double rate{0.0};
  double wilson_lo{0.0};
  double wilson_hi{0.0};
  double clopper_pearson_lo{0.0};
  double clopper_pearson_hi{0.0};
  double clopper_pearson_one_sided_hi{0.0};
};

struct AggregateStats {
  uint32_t flights{0U};
  uint32_t safe_recoveries{0U};
  uint32_t airspace_breaches{0U};
  uint32_t rta_interventions{0U};
  uint32_t cel_latches{0U};

  ConfidenceInterval95 recovery_ci95{};
  ConfidenceInterval95 breach_ci95{};

  double cep50_m{0.0};
  double cep95_m{0.0};
  double mean_td_error_m{0.0};
  double max_td_error_m{0.0};

  double mean_dx_m{0.0};
  double mean_dy_m{0.0};
  double cov_xx_m2{0.0};
  double cov_xy_m2{0.0};
  double cov_yy_m2{0.0};

  double vz_p50_mps{0.0};
  double vz_p95_mps{0.0};
  double vz_p99_mps{0.0};
  double vz_max_mps{0.0};

  double min_pack_voltage_v{0.0};
  double max_tilt_deg{0.0};
  double min_ltfj_margin_m{0.0};
  double min_ltbx_margin_m{0.0};
  double min_hospital_dist_m{0.0};
  double mean_wall_us{0.0};
};

struct GnssSpoofSummary {
  uint32_t spoof_flights{0U};
  uint32_t detected_count{0U};
  uint32_t rollback_ok_count{0U};
  double mean_ttd_s{0.0};
  double p95_ttd_s{0.0};
  double max_ttd_s{0.0};
  double max_3sigma_m{0.0};
};

struct StressGridCell {
  uint32_t mass_idx{0U};
  uint32_t wind_idx{0U};
  double mass_overload_pct{0.0};
  double mass_kg{240.0};
  double wind_gust_mps{6.0};
  uint32_t flights{0U};
  uint32_t safe_recoveries{0U};
  double survival_rate{0.0};
  double mean_td_vz_mps{0.0};
  double max_td_vz_mps{0.0};
  double min_voltage_v{0.0};
  double mean_max_tilt_deg{0.0};
  std::string dominant_failure_mode{"NONE"};
};

struct FailureBoundarySummary {
  std::string stress_airframe{"VertiRTA-X8-240"};
  std::string stress_fault_mode{"SAME_ARM_DUAL_KILL_CEL"};
  uint32_t total_stress_flights{0U};
  uint32_t safe_stress_flights{0U};
  uint32_t failed_stress_flights{0U};
  uint32_t green_cells_100pct{0U};
  uint32_t transition_cells_partial{0U};
  uint32_t red_cells_0pct{0U};
  double max_100pct_safe_mass_overload_pct{0.0};
  double first_breakdown_mass_overload_pct{0.0};
  double first_breakdown_wind_mps{0.0};
  double zero_survival_mass_overload_pct{0.0};
};

struct MonteCarloCampaignConfig {
  uint64_t master_seed{20261125ULL};
  uint32_t flights_per_regime_per_airframe{100U};  // 100 x 5 x 2 = 1,000 operational flights
  uint32_t stress_seeds_per_cell{5U};              // 6 x 6 x 5 = 180 extreme boundary flights
  uint32_t num_threads{0U};                        // 0 = hardware_concurrency
};

struct MonteCarloCampaignReport {
  std::string schema_version{"vertirta.monte_carlo_1000_dispersion_report/v4.0.0"};
  std::string timestamp_utc{};
  uint64_t master_seed{20261125ULL};
  uint32_t operational_flights_total{0U};
  uint32_t stress_flights_total{0U};
  uint32_t threads_used{1U};
  double total_wall_ms{0.0};

  AggregateStats overall{};
  std::array<AggregateStats, kNumAirframes> by_airframe{};
  std::array<AggregateStats, kNumRegimes> by_regime{};
  std::array<std::array<AggregateStats, kNumRegimes>, kNumAirframes> by_airframe_regime{};
  GnssSpoofSummary gnss_spoof_summary{};

  std::array<double, kStressGridMassLevels> stress_mass_overload_pcts{0.0, 7.0, 14.0, 21.0, 28.0, 35.0};
  std::array<double, kStressGridWindLevels> stress_wind_gust_mps{6.0, 10.0, 14.0, 18.0, 22.0, 26.0};
  std::vector<StressGridCell> stress_matrix{};
  FailureBoundarySummary failure_boundary{};

  std::vector<FlightOutcome> operational_outcomes{};
  std::string report_sha256{};
};

[[nodiscard]] ConfidenceInterval95 compute_binomial_ci95(uint32_t k, uint32_t n) noexcept;
[[nodiscard]] FlightOutcome run_single_flight(const FlightSampleConfig& cfg) noexcept;
[[nodiscard]] MonteCarloCampaignReport run_monte_carlo_campaign(const MonteCarloCampaignConfig& cfg = {});
[[nodiscard]] std::string serialize_mc_report_json(const MonteCarloCampaignReport& rep);
[[nodiscard]] bool write_mc_report_json(
    const MonteCarloCampaignReport& rep,
    const std::filesystem::path& out_path);
[[nodiscard]] bool verify_mc_report_json_sha256(
    const std::filesystem::path& json_path,
    std::string* err_out = nullptr);

}  // namespace vertirta::benchmarks
