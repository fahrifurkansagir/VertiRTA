#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/rta/cel_2d_lut.hpp"
#include "vertirta/rta/dryden_wind.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace vertirta::rta;

namespace {

std::string fixture_path(const char* rel) {
  return std::string(VERTIRTA_SOURCE_DIR) + "/" + rel;
}

}  // namespace

TEST_CASE("MmapCbvf4DLut verifies SHA-256, guards NaN/Inf/regime>=4, and runs <2.0 us p99") {
  MmapCbvf4DLut lut;
  REQUIRE_FALSE(lut.is_loaded());
  REQUIRE_FALSE(lut.eval(0, 10.0, 0.0, 10.0, 0.0).valid);

  const std::string path = fixture_path("tests/fixtures/quick_test.vrta_lut");
  REQUIRE(lut.load(path));
  REQUIRE(lut.is_loaded());

  // DR-2 Erratum #4 guards: OOB regime, NaN, Inf must cleanly return valid=false (no UB/segfault)
  REQUIRE_FALSE(lut.eval(4, 10.0, 0.0, 10.0, 0.0).valid);
  REQUIRE_FALSE(lut.eval(255, 10.0, 0.0, 10.0, 0.0).valid);
  REQUIRE_FALSE(lut.eval(0, std::nan(""), 0.0, 10.0, 0.0).valid);
  REQUIRE_FALSE(lut.eval(0, 10.0, std::numeric_limits<double>::infinity(), 10.0, 0.0).valid);
  REQUIRE_FALSE(lut.eval(0, 10.0, 0.0, -std::numeric_limits<double>::infinity(), 0.0).valid);
  REQUIRE_FALSE(lut.eval(0, 10.0, 0.0, 10.0, std::nan("")).valid);

  // Valid query check: far away & stationary should have positive value; approaching closely should be smaller
  const CbvfQueryResult q_far = lut.eval(0, 30.0, 0.0, 30.0, 0.0);
  const CbvfQueryResult q_close = lut.eval(0, 4.0, -10.0, 2.0, -5.0);
  REQUIRE(q_far.valid);
  REQUIRE(q_close.valid);
  REQUIRE(q_far.value_m > 10.0);
  REQUIRE(q_close.value_m < q_far.value_m);
  REQUIRE(q_far.grad[0] >= -1e-6);  // dV/dr >= 0 (moving farther increases safety)

  // Degraded v_cap_f from header (DR-2 Erratum #3: 2M-Cross < 16.5 m/s)
  REQUIRE(lut.v_cap_f(0) > 16.0);
  REQUIRE(lut.v_cap_f(2) < 16.5);
  REQUIRE(lut.eps_grid(0) > 0.0);

  // Corrupt 1 payload byte in a temp copy -> SHA-256 check MUST reject load!
  const std::string bad_path = "/tmp/vertirta_corrupt_lut_test.vrta_lut";
  {
    std::ifstream in(path, std::ios::binary);
    std::vector<char> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    REQUIRE(buf.size() == 11313864u);
    buf[4096 + 100] ^= 0x55;
    std::ofstream out(bad_path, std::ios::binary | std::ios::trunc);
    out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  }
  MmapCbvf4DLut bad_lut;
  REQUIRE_FALSE(bad_lut.load(bad_path));
  std::remove(bad_path.c_str());

  // Warm up cache/TLB, then measure 10,000 queries (p99 < 2.0 us in Release / O2)
  volatile double sink = 0.0;
  for (int k = 0; k < 500; ++k) {
    const auto res = lut.eval(static_cast<uint8_t>(k & 3), 15.0, -5.0, 20.0, -1.0);
    sink += res.value_m;
  }
  std::vector<double> latencies_us;
  latencies_us.reserve(10000);
  for (int k = 0; k < 10000; ++k) {
    const double r = 2.0 + 40.0 * static_cast<double>(k % 100) / 100.0;
    const double vr = -15.0 + 20.0 * static_cast<double>((k / 100) % 100) / 100.0;
    const auto t0 = std::chrono::steady_clock::now();
    const auto res = lut.eval(static_cast<uint8_t>(k & 3), r, vr, 15.0, -1.5);
    const auto t1 = std::chrono::steady_clock::now();
    sink += res.value_m;
    latencies_us.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  (void)sink;
  std::sort(latencies_us.begin(), latencies_us.end());
  const double p99_us = latencies_us[9900];
#if !defined(__SANITIZE_ADDRESS__)
  REQUIRE(p99_us < 2.0);
#else
  REQUIRE(p99_us < 25.0);
#endif
}

TEST_CASE("Cel2DLut loads 11,230 B .vrta_cel2d with SHA-256 check, tier selection, and monotone braking") {
  Cel2DLut cel;
  REQUIRE_FALSE(cel.is_loaded());
  REQUIRE_FALSE(cel.eval(380.0f, 10.0, -2.0).valid);

  REQUIRE(cel.load(fixture_path("tests/fixtures/quick_test.vrta_cel2d")));
  REQUIRE(cel.is_loaded());
  REQUIRE_FALSE(cel.eval(380.0f, std::nan(""), -2.0).valid);
  REQUIRE_FALSE(cel.eval(380.0f, 10.0, std::numeric_limits<double>::infinity()).valid);

  // Voltage-sag tier monotonicity: V_tier0(370V) >= V_tier1(352V) >= V_tier2(340V)
  const Cel2DQuery q0 = cel.eval(370.0f, 10.0, -5.0);
  const Cel2DQuery q1 = cel.eval(352.0f, 10.0, -5.0);
  const Cel2DQuery q2 = cel.eval(340.0f, 10.0, -5.0);
  REQUIRE(q0.valid);
  REQUIRE(q1.valid);
  REQUIRE(q2.valid);
  REQUIRE(q0.tier_idx == 0);
  REQUIRE(q1.tier_idx == 1);
  REQUIRE(q2.tier_idx == 2);
  REQUIRE(q0.value_m >= q1.value_m - 0.05);
  REQUIRE(q1.value_m >= q2.value_m - 0.05);
  REQUIRE(q0.dv_dz > 0.5);   // Higher altitude is safer
  REQUIRE(q0.dv_dvz > 0.1);  // More upward velocity (less sink) is safer
}

TEST_CASE("DrydenTurbulenceFilter preserves >=99.5% theoretical MIL-F-8785C variance at 100 Hz") {
  // h = 60 m AGL, V_air = 22.0 m/s (cruise), W20 = 15.43 m/s (Moderate 30 kt, DR-2 Table 4)
  constexpr double kH = 60.0;
  constexpr double kVair = 22.0;
  constexpr double kW20 = 15.43;
  constexpr double kDt = 0.01;

  const DrydenScales sc = DrydenTurbulenceFilter::compute_scales(kH, kW20);
  REQUIRE_THAT(sc.sigma_w, WithinAbs(1.543, 1e-4));
  REQUIRE_THAT(sc.L_w, WithinAbs(60.0, 1e-4));
  REQUIRE_THAT(sc.L_u, WithinRel(219.7, 0.01));

  DrydenTurbulenceFilter f1(20261122ULL);
  DrydenTurbulenceFilter f2(20261122ULL);

  // Bit-identical seeded determinism check
  const auto w1 = f1.step_seeded(kDt, kH, kVair, kW20);
  const auto w2 = f2.step_seeded(kDt, kH, kVair, kW20);
  REQUIRE(w1.x == w2.x);
  REQUIRE(w1.y == w2.y);
  REQUIRE(w1.z == w2.z);

  // 400k-tick stationary variance verification (after 2000-tick burn-in; ~200 independent tau_u periods)
  f1.reseed(20261122ULL);
  for (int k = 0; k < 2000; ++k) {
    (void)f1.step_seeded(kDt, kH, kVair, kW20);
  }
  double su2 = 0.0, sv2 = 0.0, sw2 = 0.0;
  constexpr int kSamples = 400000;
  for (int k = 0; k < kSamples; ++k) {
    const auto g = f1.step_seeded(kDt, kH, kVair, kW20);
    su2 += g.x * g.x;
    sv2 += g.y * g.y;
    sw2 += g.z * g.z;
  }
  const double std_u = std::sqrt(su2 / kSamples);
  const double std_v = std::sqrt(sv2 / kSamples);
  const double std_w = std::sqrt(sw2 / kSamples);

  REQUIRE_THAT(std_u, WithinRel(sc.sigma_u, 0.04));
  REQUIRE_THAT(std_v, WithinRel(sc.sigma_v, 0.04));
  REQUIRE_THAT(std_w, WithinRel(sc.sigma_w, 0.03));

  // Analytical discrete Parseval variance preservation check (>= 99.5% of continuous sigma^2)
  const auto ratios = DrydenTurbulenceFilter::theoretical_variance_ratios(kDt, kH, kVair);
  REQUIRE(ratios[0] >= 0.995);
  REQUIRE(ratios[1] >= 0.995);
  REQUIRE(ratios[2] >= 0.995);
}
