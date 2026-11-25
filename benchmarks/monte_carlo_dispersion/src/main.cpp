// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "vertirta/benchmarks/mc_1000_runner.hpp"

int main(int argc, char** argv) {
  vertirta::benchmarks::MonteCarloCampaignConfig cfg{};
  std::filesystem::path out_json = "artifacts/v4.0.0/monte_carlo_1000_dispersion_report.json";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out-json" && i + 1 < argc) {
      out_json = argv[++i];
    } else if (arg == "--seed" && i + 1 < argc) {
      cfg.master_seed = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--flights-per-stratum" && i + 1 < argc) {
      cfg.flights_per_regime_per_airframe = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--stress-seeds" && i + 1 < argc) {
      cfg.stress_seeds_per_cell = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--threads" && i + 1 < argc) {
      cfg.num_threads = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: vertirta_mc_1000_cli [--out-json PATH] [--seed U64] [--flights-per-stratum N] "
          "[--stress-seeds K] [--threads T]\n");
      return 0;
    }
  }

  const auto rep = vertirta::benchmarks::run_monte_carlo_campaign(cfg);
  if (!vertirta::benchmarks::write_mc_report_json(rep, out_json)) {
    std::fprintf(stderr, "ERROR: Failed to write Monte Carlo report JSON to %s\n",
                 out_json.string().c_str());
    return 2;
  }

  std::string verify_err;
  if (!vertirta::benchmarks::verify_mc_report_json_sha256(out_json, &verify_err)) {
    std::fprintf(stderr, "ERROR: Self-check SHA-256 verification failed: %s\n", verify_err.c_str());
    return 3;
  }

  std::printf(
      "VertiRTA MC-1000 Campaign Complete (seed=%llu, threads=%u, wall=%.1f ms)\n"
      "  Operational: %u/%u safe (%.2f%%, 95%% CP [%.2f%%, %.2f%%]) | Airspace breaches: %u (95%% 1-sided <= %.3f%%)\n"
      "  X8-240:      %u/%u safe (%.2f%%) | CEP50=%.2f m, CEP95=%.2f m | vz p95=%.2f p99=%.2f max=%.2f m/s\n"
      "  LC-240 8+1:  %u/%u safe (%.2f%%) | CEP50=%.2f m, CEP95=%.2f m | vz p95=%.2f p99=%.2f max=%.2f m/s\n"
      "  Stress 6x6:  %u/%u safe (%u green / %u transition / %u red) | first breakdown @ +%.0f%% mass, %.0f m/s wind\n"
      "  Report JSON: %s | sha256=%s\n",
      static_cast<unsigned long long>(rep.master_seed),
      rep.threads_used,
      rep.total_wall_ms,
      rep.overall.safe_recoveries,
      rep.overall.flights,
      rep.overall.recovery_ci95.rate * 100.0,
      rep.overall.recovery_ci95.clopper_pearson_lo * 100.0,
      rep.overall.recovery_ci95.clopper_pearson_hi * 100.0,
      rep.overall.airspace_breaches,
      rep.overall.breach_ci95.clopper_pearson_one_sided_hi * 100.0,
      rep.by_airframe[0].safe_recoveries,
      rep.by_airframe[0].flights,
      rep.by_airframe[0].recovery_ci95.rate * 100.0,
      rep.by_airframe[0].cep50_m,
      rep.by_airframe[0].cep95_m,
      rep.by_airframe[0].vz_p95_mps,
      rep.by_airframe[0].vz_p99_mps,
      rep.by_airframe[0].vz_max_mps,
      rep.by_airframe[1].safe_recoveries,
      rep.by_airframe[1].flights,
      rep.by_airframe[1].recovery_ci95.rate * 100.0,
      rep.by_airframe[1].cep50_m,
      rep.by_airframe[1].cep95_m,
      rep.by_airframe[1].vz_p95_mps,
      rep.by_airframe[1].vz_p99_mps,
      rep.by_airframe[1].vz_max_mps,
      rep.failure_boundary.safe_stress_flights,
      rep.failure_boundary.total_stress_flights,
      rep.failure_boundary.green_cells_100pct,
      rep.failure_boundary.transition_cells_partial,
      rep.failure_boundary.red_cells_0pct,
      rep.failure_boundary.first_breakdown_mass_overload_pct,
      rep.failure_boundary.first_breakdown_wind_mps,
      out_json.string().c_str(),
      rep.report_sha256.c_str());
  return 0;
}
