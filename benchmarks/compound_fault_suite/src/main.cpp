// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <filesystem>
#include <string>

#include "vertirta/benchmarks/compound_fault_runner.hpp"

int main(int argc, char** argv) {
  std::filesystem::path out_json = "artifacts/v4.0.0/compound_fault_mission_report.json";
  std::filesystem::path work_dir =
      std::filesystem::temp_directory_path() / "vertirta_compound_fault_cli_work";
  std::filesystem::path e5x_out{};

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out-json" && i + 1 < argc) {
      out_json = argv[++i];
    } else if (arg == "--work-dir" && i + 1 < argc) {
      work_dir = argv[++i];
    } else if (arg == "--emit-e5x" && i + 1 < argc) {
      e5x_out = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: vertirta_compound_fault_cli [--out-json PATH] [--work-dir DIR] [--emit-e5x PATH]\n");
      return 0;
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(work_dir, ec);

  const auto report = vertirta::benchmarks::run_all_compound_scenarios(work_dir, e5x_out);
  if (!vertirta::benchmarks::write_compound_suite_json(report, out_json)) {
    std::fprintf(stderr, "ERROR: Failed to write compound fault report JSON to %s\n",
                 out_json.string().c_str());
    return 2;
  }

  for (const auto& sc : report.scenarios) {
    std::printf("[Scenario %u] %-52s | %s | ticks=%u | wall=%.1f ms\n",
                sc.scenario_id,
                sc.scenario_name.c_str(),
                sc.passed ? "PASS" : "FAIL",
                sc.ticks_100hz,
                sc.wall_ms);
    if (!sc.passed) {
      std::fprintf(stderr, "  FAILURE REASON: %s\n", sc.failure_reason.c_str());
    }
  }
  std::printf("Compound Suite: %u/%u PASSED | total_wall=%.1f ms | sha256=%s\n",
              report.scenarios_passed,
              report.scenarios_total,
              report.total_wall_ms,
              report.report_sha256.c_str());
  return report.all_passed ? 0 : 1;
}
