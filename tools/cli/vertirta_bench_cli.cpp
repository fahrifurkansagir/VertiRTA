#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/cel_2d_lut.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace {

using namespace vertirta::core;
using namespace vertirta::rta;

inline uint64_t read_clock_ns(clockid_t clk) noexcept {
  struct timespec ts {};
  ::clock_gettime(clk, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

}  // namespace

int main(int argc, char** argv) {
  std::size_t ticks = 10000;
  std::string clock_name = "monotonic_raw";
  std::string out_path;
  std::string lut_path;
  std::string cel_path;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto next_val = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string{};
    };
    if (arg == "--ticks") {
      ticks = std::stoull(next_val());
    } else if (arg == "--clock") {
      clock_name = next_val();
    } else if (arg == "--out") {
      out_path = next_val();
    } else if (arg == "--lut") {
      lut_path = next_val();
    } else if (arg == "--cel-lut") {
      cel_path = next_val();
    }
  }

  clockid_t clk = CLOCK_MONOTONIC_RAW;
  if (clock_name == "thread_cputime") {
    clk = CLOCK_THREAD_CPUTIME_ID;
  } else if (clock_name == "monotonic_raw") {
    clk = CLOCK_MONOTONIC_RAW;
  } else {
    std::cerr << "Unsupported --clock '" << clock_name << "' (expected monotonic_raw|thread_cputime)\n";
    return 2;
  }

  if (ticks == 0) {
    ticks = 1;
  }

  MmapCbvf4DLut lut;
  Cel2DLut cel;
  for (const std::string& c : {lut_path, std::string("tests/fixtures/quick_test.vrta_lut"),
                               std::string("../tests/fixtures/quick_test.vrta_lut")}) {
    if (!c.empty() && lut.load(c)) {
      break;
    }
  }
  for (const std::string& c : {cel_path, std::string("tests/fixtures/quick_test.vrta_cel2d"),
                               std::string("../tests/fixtures/quick_test.vrta_cel2d")}) {
    if (!c.empty() && cel.load(c)) {
      break;
    }
  }

  CbvfSafetyFilter filter(lut.is_loaded() ? &lut : nullptr, cel.is_loaded() ? &cel : nullptr);
  IndiWlsController ctrl(true);
  ctrl.reset(params::kOmegaHover);

  const CapabilityEnvelope env =
      CoaxEffectiveness::compute_envelope(0x21, 0, 0x07, 385.0f, 102.0f, 0.55f, 0.0f, 0.0f, 0.0f, 0);
  const std::array<HazardPoint, 4> hazards = {{
      HazardPoint{.pos_ned = {24.0, 2.0, -30.0}, .vel_ned = {-1.0, 0.0, 0.0}, .kind = 0},
      HazardPoint{.pos_ned = {15.0, 14.0, -28.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 1},
      HazardPoint{.pos_ned = {-22.0, 12.0, -32.0}, .vel_ned = {2.0, -1.0, 0.0}, .kind = 2},
      HazardPoint{.pos_ned = {0.0, 50.0, -30.0}, .vel_ned = {0.0, 0.0, 0.0}, .radius_z_m = 700.0, .kind = 3},
  }};

  volatile double sink = 0.0;
  float omega_cmd[8]{};

  // Warm up caches & branch predictors
  for (int w = 0; w < 200; ++w) {
    const auto sf = filter.step_100hz({0.0, 0.0, -30.0}, {10.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, 0.0, env, hazards);
    IndiInput in{};
    in.accel_des_ned = sf.a_safe_ned;
    in.v_term_v = 385.0f;
    in.fault_mask = 0x21;
    (void)ctrl.step_400hz(in, omega_cmd);
    sink += sf.a_safe_ned.x + omega_cmd[0];
  }

  std::vector<double> samples_us;
  samples_us.reserve(ticks);
  double sum_us = 0.0;

  for (std::size_t k = 0; k < ticks; ++k) {
    const double dx = 0.01 * static_cast<double>(k % 100);
    const uint64_t t0 = read_clock_ns(clk);
    const auto sf = filter.step_100hz(
        {dx, 0.0, -30.0}, {10.0, 1.0, -0.2}, {1.0, 0.2, 0.0}, 0.05, env, hazards);
    IndiInput in{};
    in.accel_des_ned = sf.a_safe_ned;
    in.v_term_v = 385.0f;
    in.fault_mask = 0x21;
    (void)ctrl.step_400hz(in, omega_cmd);
    const uint64_t t1 = read_clock_ns(clk);

    const double dt_us = static_cast<double>(t1 - t0) * 1e-3;
    samples_us.push_back(dt_us);
    sum_us += dt_us;
    sink += sf.a_safe_ned.x + omega_cmd[2];
  }
  (void)sink;

  std::sort(samples_us.begin(), samples_us.end());
  auto pct = [&](double p) -> double {
    const std::size_t idx = std::min(
        samples_us.size() - 1u,
        static_cast<std::size_t>(std::floor(p * static_cast<double>(samples_us.size()))));
    return samples_us[idx];
  };

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(4)
      << "{\n"
      << "  \"schema_version\": \"vertirta.bench_cli/v1.0.0\",\n"
      << "  \"clock\": \"" << clock_name << "\",\n"
      << "  \"ticks\": " << ticks << ",\n"
      << "  \"lut_loaded\": " << (lut.is_loaded() ? "true" : "false") << ",\n"
      << "  \"mean_us\": " << (sum_us / static_cast<double>(ticks)) << ",\n"
      << "  \"p50_us\": " << pct(0.50) << ",\n"
      << "  \"p99_us\": " << pct(0.99) << ",\n"
      << "  \"p999_us\": " << pct(0.999) << ",\n"
      << "  \"max_us\": " << samples_us.back() << "\n"
      << "}\n";

  const std::string json_out = oss.str();
  if (!out_path.empty() && out_path != "/dev/null") {
    std::ofstream f(out_path, std::ios::trunc);
    if (!f.is_open()) {
      std::cerr << "Cannot open --out path: " << out_path << "\n";
      return 1;
    }
    f << json_out;
  }
  std::cout << json_out;
  return 0;
}
