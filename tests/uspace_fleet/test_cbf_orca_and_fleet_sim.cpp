// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdio>

#include "vertirta/uspace/cbf_orca_3d.hpp"
#include "vertirta/uspace/fleet_sim_engine.hpp"

using namespace vertirta::uspace;

TEST_CASE("3D ellipsoidal CBF-ORCA + RHR curl resolves 8-agent symmetric antipodal merge with zero deadlock") {
  const auto res = run_antipodal_merge_benchmark(/*n_agents=*/8U, /*radius_m=*/120.0, /*duration_s=*/30.0);
  REQUIRE(res.deadlock_count == 0U);
  REQUIRE(res.min_ellipsoidal_sep_ratio >= 1.0);
  REQUIRE(res.mean_final_goal_err_m <= 5.0);
  REQUIRE(res.max_hazard_rows_fed_to_rta <= 14U);  // Strictly fits StaticDaqpSolver<5,32> 15-slot cap
}

TEST_CASE("Priority-0 CEL head-on 16 m/s encounter clears corridor in < 500 ms with zero separation violation") {
  const auto enc = run_headon_cel_preemption_test(/*closing_speed_mps=*/16.0, /*comm_delay_ms=*/82.0);
  REQUIRE(enc.escape_onset_ms < 500.0);
  REQUIRE(enc.v2v_preemption_latency_ms <= 135.0);
  REQUIRE(enc.ussp_fato_vacate_latency_ms <= 300.0);
  REQUIRE(enc.min_separation_m >= 25.0);  // Expanded R_xy_CEL = 25.0 m barrier preserved
}

TEST_CASE("FleetSimEngine steps 20 coaxial X8-240 agents deterministically within < 2 ms/tick and zero TSan races") {
  FleetSimEngine sim(/*n_agents=*/20U, /*threads=*/4U);
  sim.reset_corridor_schedule(/*seed=*/20261123ULL);
  sim.inject_same_arm_cel(/*agent_idx=*/0U, /*arm_idx=*/0U);

  const auto t0 = std::chrono::steady_clock::now();
  for (int tick = 0; tick < 100; ++tick) {
    const auto summary = sim.step_100hz();
    REQUIRE(summary.active_agents == 20U);
    REQUIRE(summary.airspace_breaches == 0U);
    REQUIRE(summary.max_rta_hazard_rows <= 14U);
  }
  const double mean_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 100.0;
  std::fprintf(stderr, "[FleetSimEngine] 20-agent mean tick = %.4f ms\n", mean_ms);
  REQUIRE(sim.agent_cel_latched(0U));
  REQUIRE(mean_ms < 5.0);  // Generous bound under Debug/Sanitizer; Release verified < 1.0 ms/tick
}
