#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta/core/vrta_recorder.hpp"
#include "vertirta/rta/cbvf_safety_filter.hpp"
#include "vertirta/rta/cel_2d_lut.hpp"
#include "vertirta/rta/dryden_wind.hpp"
#include "vertirta/rta/mmap_cbvf_lut.hpp"

namespace {

using namespace vertirta::core;
using namespace vertirta::rta;

struct ScenarioSpec {
  int run_id{0};
  std::string stratum{"S1"};
  std::string outcome_class{"NOMINAL"};
  std::string fault_topology{"FLT_NONE"};
  uint8_t fault_mask{0x00};
  int imu_fault_channel{-1};
  std::string imu_fault_mode{};
  uint64_t seed{20261122ULL};
  std::string exposure{"coastal_d"};
  double start_lat_deg{40.8860};
  double start_lon_deg{29.1880};
  double mean_wind_mps{5.0};
  double peak_gust_mps{3.0};
  double wind_at_60m_mps{6.0};
  double initial_soc_pct{50.0};
  double cruise_speed_mps{12.0};
};

struct ScenarioResult {
  ScenarioSpec spec{};
  bool passed{true};
  std::string violated_invariant{};
  std::string root_cause_class{};
  double min_cbvf_margin_m{100.0};
  double touchdown_vz_mps{0.0};
  double max_tilt_deg{0.0};
  double min_v_term_v{400.0};
  bool cel_latched{false};
  double mean_yaw_rate_rad_s{0.0};
  std::string vrta_path{};
};

// Lightweight zero-dependency JSON manifest parser tailored to lhs_sampler.py output.
class MiniJsonParser {
 public:
  explicit MiniJsonParser(std::string_view s) : s_(s) {}

  void skip_ws() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
      ++pos_;
    }
  }

  bool consume(char c) {
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  std::string parse_string() {
    skip_ws();
    std::string out;
    if (pos_ >= s_.size() || s_[pos_] != '"') {
      return out;
    }
    ++pos_;
    while (pos_ < s_.size() && s_[pos_] != '"') {
      if (s_[pos_] == '\\' && pos_ + 1 < s_.size()) {
        ++pos_;
      }
      out.push_back(s_[pos_++]);
    }
    if (pos_ < s_.size()) {
      ++pos_;
    }
    return out;
  }

  double parse_number() {
    skip_ws();
    const std::size_t start = pos_;
    while (pos_ < s_.size()) {
      const char ch = s_[pos_];
      if (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-' || ch == '+' ||
          ch == '.' || ch == 'e' || ch == 'E') {
        ++pos_;
      } else {
        break;
      }
    }
    if (pos_ == start) {
      return 0.0;
    }
    return std::stod(std::string(s_.substr(start, pos_ - start)));
  }

  void skip_value() {
    skip_ws();
    if (pos_ >= s_.size()) {
      return;
    }
    const char ch = s_[pos_];
    if (ch == '"') {
      (void)parse_string();
    } else if (ch == '{') {
      ++pos_;
      if (!consume('}')) {
        do {
          (void)parse_string();
          (void)consume(':');
          skip_value();
        } while (consume(','));
        (void)consume('}');
      }
    } else if (ch == '[') {
      ++pos_;
      if (!consume(']')) {
        do {
          skip_value();
        } while (consume(','));
        (void)consume(']');
      }
    } else {
      while (pos_ < s_.size() && s_[pos_] != ',' && s_[pos_] != '}' && s_[pos_] != ']' &&
             !std::isspace(static_cast<unsigned char>(s_[pos_]))) {
        ++pos_;
      }
    }
  }

  std::vector<ScenarioSpec> parse_manifest() {
    std::vector<ScenarioSpec> out;
    if (!consume('[')) {
      return out;
    }
    if (consume(']')) {
      return out;
    }
    do {
      ScenarioSpec sc{};
      if (!consume('{')) {
        break;
      }
      if (!consume('}')) {
        do {
          const std::string key = parse_string();
          (void)consume(':');
          skip_ws();
          if (key == "run_id") {
            sc.run_id = static_cast<int>(std::llround(parse_number()));
          } else if (key == "stratum") {
            sc.stratum = parse_string();
          } else if (key == "outcome_class") {
            sc.outcome_class = parse_string();
          } else if (key == "fault_topology") {
            sc.fault_topology = parse_string();
          } else if (key == "seed") {
            sc.seed = static_cast<uint64_t>(std::llround(parse_number()));
          } else if (key == "exposure") {
            sc.exposure = parse_string();
          } else if (key == "start_lat_deg") {
            sc.start_lat_deg = parse_number();
          } else if (key == "start_lon_deg") {
            sc.start_lon_deg = parse_number();
          } else if (key == "mean_wind_mps") {
            sc.mean_wind_mps = parse_number();
          } else if (key == "peak_gust_mps") {
            sc.peak_gust_mps = parse_number();
          } else if (key == "wind_at_60m_mps") {
            sc.wind_at_60m_mps = parse_number();
          } else if (key == "initial_soc_pct") {
            sc.initial_soc_pct = parse_number();
          } else if (key == "cruise_speed_mps") {
            sc.cruise_speed_mps = parse_number();
          } else if (key == "failed_motors") {
            if (consume('[') && !consume(']')) {
              do {
                const int m = static_cast<int>(std::llround(parse_number()));
                if (m >= 0 && m < 8) {
                  sc.fault_mask = static_cast<uint8_t>(sc.fault_mask | (1u << m));
                }
              } while (consume(','));
              (void)consume(']');
            }
          } else if (key == "imu_fault") {
            if (pos_ < s_.size() && s_[pos_] == '{') {
              ++pos_;
              if (!consume('}')) {
                do {
                  const std::string ik = parse_string();
                  (void)consume(':');
                  if (ik == "channel") {
                    sc.imu_fault_channel = static_cast<int>(std::llround(parse_number()));
                  } else if (ik == "mode") {
                    sc.imu_fault_mode = parse_string();
                  } else {
                    skip_value();
                  }
                } while (consume(','));
                (void)consume('}');
              }
            } else {
              skip_value();
            }
          } else {
            skip_value();
          }
        } while (consume(','));
        (void)consume('}');
      }
      out.push_back(std::move(sc));
    } while (consume(','));
    (void)consume(']');
    return out;
  }

 private:
  std::string_view s_;
  std::size_t pos_{0};
};

double quat_tilt_deg(const QuatPx4FrdNed& q) noexcept {
  const double c = clamp_d(1.0 - 2.0 * (q.x * q.x + q.y * q.y), -1.0, 1.0);
  return std::acos(c) * kRadToDeg;
}

// Signed distance [m] to LTFJ NW J2 pinch east wall (~29.192122 E at 40.9128 N).
double approx_ltfj_east_margin_m(double lat_deg, double lon_deg, double east_offset_m) noexcept {
  const double meters_per_deg_lon = 111320.0 * std::cos(lat_deg * kDegToRad);
  const double cur_lon = lon_deg + east_offset_m / meters_per_deg_lon;
  // Conservative distance to LTFJ J2 pinch wall (29.192122 E)
  return (29.192122 - cur_lon) * meters_per_deg_lon;
}

void write_vrta_trace(
    const std::string& path,
    const std::vector<VrtaFrame100Hz>& frames,
    const MmapCbvf4DLut* lut) {
  std::filesystem::path p(path);
  if (p.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(p.parent_path(), ec);
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return;
  }
  const uint8_t* sha = (lut != nullptr && lut->is_loaded()) ? lut->payload_sha256().data() : nullptr;
  VrtaFileHeader hdr = make_default_vrta_header("vertirta_v1_0_0_tier1_batch_cli_trace", sha);
  out.write(reinterpret_cast<const char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
  for (const auto& fr : frames) {
    out.write(reinterpret_cast<const char*>(&fr), static_cast<std::streamsize>(sizeof(fr)));
  }
}

ScenarioResult run_single_scenario(
    const ScenarioSpec& sc,
    const MmapCbvf4DLut* lut,
    const Cel2DLut* cel,
    bool use_hj_cbvf,
    bool use_indi_wls,
    const std::string& vrta_dir) {
  ScenarioResult res{};
  res.spec = sc;

  const bool is_cel = (sc.outcome_class == "CEL") ||
                      (CoaxEffectiveness::classify_regime(sc.fault_mask) == FlightRegime::SAME_ARM_ACTIVE_SPIN);
  const double init_agl = is_cel ? 35.0 : 40.0;
  const double init_vx = is_cel ? 0.0 : clamp_d(sc.cruise_speed_mps, 6.0, 22.0);
  const float init_soc = static_cast<float>(clamp_d(sc.initial_soc_pct / 100.0, 0.12, 0.98));

  CoaxialPlant6Dof plant;
  plant.reset({0.0, 0.0, -init_agl}, {init_vx, 0.0, 0.0}, init_soc);

  IndiWlsController ctrl(use_indi_wls);
  ctrl.reset(params::kOmegaHover);
  TmrVoter voter;
  DrydenTurbulenceFilter dryden(sc.seed);
  CbvfSafetyFilter filter(lut, cel);

  // Corridor obstacle directly ahead along +N track (tests braking vs HOCBF late braking)
  // plus LTFJ east CTR wall derived from scenario start_lon_deg.
  const double init_ltfj_margin = std::max(40.0, approx_ltfj_east_margin_m(sc.start_lat_deg, sc.start_lon_deg, 0.0));
  const std::array<HazardPoint, 2> hazards = {{
      HazardPoint{.pos_ned = {68.0, 0.0, -init_agl}, .vel_ned = {0.0, 0.0, 0.0}, .radius_z_m = 45.0, .kind = 0},
      HazardPoint{.pos_ned = {0.0, init_ltfj_margin, -init_agl}, .vel_ned = {0.0, 0.0, 0.0}, .radius_z_m = 700.0, .kind = 3},
  }};

  std::vector<VrtaFrame100Hz> trace;
  trace.reserve(is_cel ? 2200u : 800u);

  const int max_outer_ticks = is_cel ? 2400 : 700;  // 24 s CEL descent; 7 s CSFL braking/stabilization
  double yaw_rate_sum = 0.0;
  int yaw_rate_cnt = 0;
  bool landed = false;
  bool imu_isolated = (sc.imu_fault_channel < 0);
  double imu_fault_elapsed_s = 0.0;
  double last_chi2 = 0.0;
  uint8_t last_tmr_state = 0;
  uint8_t last_imu_mask = 0x07;

  // Wind heading directed toward Arm 0 (+N/+E) scaled from manifest mean_wind_mps
  const double wind_mag = clamp_d(sc.mean_wind_mps * 0.35 + sc.peak_gust_mps * 0.15, 0.5, 12.0);
  const Vec3Ned wind_base{wind_mag * kInvSqrt2, wind_mag * kInvSqrt2, 0.0};

  for (int outer = 0; outer < max_outer_ticks; ++outer) {
    const double t = static_cast<double>(outer) * 0.01;
    const bool fault_active = (t >= 1.0);
    const uint8_t fm = fault_active ? sc.fault_mask : 0x00u;

    const PlantState st0 = plant.state();
    const double agl0 = std::max(0.0, -st0.pos_ned.z);
    if (static_cast<double>(st0.battery.v_term_v) < res.min_v_term_v) {
      res.min_v_term_v = static_cast<double>(st0.battery.v_term_v);
    }

    const Vec3Ned gust = dryden.step_seeded(0.01, std::max(agl0, 5.0), std::max(st0.vel_ned.norm(), 4.0), sc.mean_wind_mps);
    const Vec3Ned wind_ned = wind_base + gust * 0.15;

    // Nominal guidance command
    Vec3Ned a_nom_ned{};
    if (is_cel && fault_active) {
      const double vz_des = (agl0 > 10.0) ? 2.2 : ((agl0 > 5.0) ? 1.5 : ((agl0 > 1.0) ? 0.9 : 0.5));
      a_nom_ned = {
          0.0,
          0.0,
          clamp_d(2.5 * (vz_des - st0.vel_ned.z), -2.2, 3.0),
      };
    } else {
      a_nom_ned = {
          clamp_d(1.2 * (init_vx - st0.vel_ned.x), -4.0, 4.0),
          clamp_d(-1.0 * st0.vel_ned.y, -3.0, 3.0),
          clamp_d(1.5 * (agl0 - init_agl) - 1.8 * st0.vel_ned.z, -3.0, 3.0),
      };
    }

    const CapabilityEnvelope env = CoaxEffectiveness::compute_envelope(
        fm,
        last_tmr_state,
        last_imu_mask,
        st0.battery.v_term_v,
        st0.battery.i_pack_a,
        st0.battery.soc,
        ctrl.spin_rate_ref_rad_s(),
        static_cast<float>(st0.omega_frd.z),
        static_cast<float>(last_chi2),
        0u);

    SafetyFilterOutput sf{};
    if (is_cel) {
      sf = filter.step_100hz(st0.pos_ned, st0.vel_ned, a_nom_ned, 0.0, env, {});
    } else if (use_hj_cbvf) {
      sf = filter.step_100hz(st0.pos_ned, st0.vel_ned, a_nom_ned, 0.0, env, hazards);
    } else {
      // Baseline naive relative-degree-2 HOCBF (no BRT lookahead, no v_cap_f damping)
      sf.a_safe_ned = a_nom_ned;
      const double dist_obs = std::hypot(st0.pos_ned.x - hazards[0].pos_ned.x, st0.pos_ned.y - hazards[0].pos_ned.y);
      const double h_val = dist_obs - 4.5;
      sf.min_cbvf_v = h_val;
      const double v_toward = (hazards[0].pos_ned.x - st0.pos_ned.x > 0.0) ? st0.vel_ned.x : -st0.vel_ned.x;
      // Naive exponential HOCBF with fixed low gain only brakes very late (< 18 m)
      if (h_val < 14.0 && v_toward > 0.0) {
        const double a_req = -std::min(static_cast<double>(env.a_xy_max), 0.35 * v_toward + 0.25 * (14.0 - h_val));
        sf.a_safe_ned.x = std::min(sf.a_safe_ned.x, a_req);
      }
      const double a_lim = static_cast<double>(env.a_xy_max);
      sf.a_safe_ned.x = clamp_d(sf.a_safe_ned.x, -a_lim, a_lim);
      sf.a_safe_ned.y = clamp_d(sf.a_safe_ned.y, -a_lim, a_lim);
    }

    if (sf.min_cbvf_v < res.min_cbvf_margin_m) {
      res.min_cbvf_margin_m = sf.min_cbvf_v;
    }
    if (sf.cel_latched) {
      res.cel_latched = true;
    }

    // 4 inner 400 Hz control & plant integration ticks
    for (int sub = 0; sub < 4; ++sub) {
      const PlantState st = plant.state();
      ImuSampleFrd imus[3]{};
      for (int c = 0; c < 3; ++c) {
        imus[c].gyro_rad_s = st.omega_frd;
        imus[c].accel_m_s2 = {0.0, 0.0, -params::kGravity};
        imus[c].t_us = static_cast<uint64_t>(outer) * 10000ULL + static_cast<uint64_t>(sub) * 2500ULL;
        imus[c].valid = true;
      }
      if (fault_active && sc.imu_fault_channel >= 0 && sc.imu_fault_channel < 3) {
        imu_fault_elapsed_s += 0.0025;
        if (sc.imu_fault_mode == "FREEZE") {
          imus[sc.imu_fault_channel].gyro_rad_s = {0.12, -0.12, 0.05};
        } else if (sc.imu_fault_mode == "BIAS_RAMP") {
          imus[sc.imu_fault_channel].gyro_rad_s.x += 0.08 + 0.5 * imu_fault_elapsed_s;
        } else {
          imus[sc.imu_fault_channel].gyro_rad_s.x += 0.15;
        }
      }
      const TmrVoteOutput vr = voter.step(imus, st.omega_frd, true);
      last_chi2 = vr.max_chi2_residual;
      last_tmr_state = static_cast<uint8_t>(vr.state);
      last_imu_mask = vr.health_mask;
      if (fault_active && sc.imu_fault_channel >= 0 &&
          (vr.health_mask & static_cast<uint8_t>(1u << sc.imu_fault_channel)) == 0u) {
        imu_isolated = true;
      }

      IndiInput in{};
      in.q_frd_ned = st.q_frd_ned;
      in.gyro_frd = vr.gyro_voted;
      in.accel_des_ned = sf.a_safe_ned;
      in.yaw_rate_des = sf.yaw_rate_safe;
      in.v_horiz_mps = static_cast<float>(std::hypot(st.vel_ned.x, st.vel_ned.y));
      in.v_term_v = st.battery.v_term_v;
      in.fault_mask = fm;

      float omega_cmd[8]{};
      (void)ctrl.step_400hz(in, omega_cmd);
      const PlantState nx = plant.step_400hz(omega_cmd, fm, wind_ned);
      const double tilt = quat_tilt_deg(nx.q_frd_ned);
      if (tilt > res.max_tilt_deg) {
        res.max_tilt_deg = tilt;
      }
      if (fault_active) {
        yaw_rate_sum += std::abs(nx.omega_frd.z);
        ++yaw_rate_cnt;
      }
      if (nx.on_ground && t > 2.0) {
        landed = true;
        res.touchdown_vz_mps = std::abs(nx.touchdown_vz_mps);
        break;
      }
    }

    const PlantState fin = plant.state();
    VrtaFrame100Hz fr{};
    fr.seq = static_cast<uint64_t>(outer);
    fr.t_sim_us = static_cast<uint64_t>(outer) * 10000ULL;
    const Vec3Enu pos_enu = ned_to_enu(fin.pos_ned);
    const Vec3Enu vel_enu = ned_to_enu(fin.vel_ned);
    fr.pos_enu[0] = pos_enu.x;
    fr.pos_enu[1] = pos_enu.y;
    fr.pos_enu[2] = pos_enu.z;
    fr.vel_enu[0] = static_cast<float>(vel_enu.x);
    fr.vel_enu[1] = static_cast<float>(vel_enu.y);
    fr.vel_enu[2] = static_cast<float>(vel_enu.z);
    fr.q_frd_ned[0] = static_cast<float>(fin.q_frd_ned.w);
    fr.q_frd_ned[1] = static_cast<float>(fin.q_frd_ned.x);
    fr.q_frd_ned[2] = static_cast<float>(fin.q_frd_ned.y);
    fr.q_frd_ned[3] = static_cast<float>(fin.q_frd_ned.z);
    fr.v_term_v = fin.battery.v_term_v;
    fr.soc_pct = fin.battery.soc * 100.0f;
    fr.cbvf_min_v = static_cast<float>(sf.min_cbvf_v);
    fr.ltfj_margin_m = static_cast<float>(approx_ltfj_east_margin_m(sc.start_lat_deg, sc.start_lon_deg, pos_enu.x));
    fr.chi2_residual = static_cast<float>(last_chi2);
    fr.agl_m = static_cast<float>(std::max(0.0, -fin.pos_ned.z));
    fr.active_motor_mask = static_cast<uint8_t>(~fm);
    fr.flight_regime = static_cast<uint8_t>(CoaxEffectiveness::classify_regime(fm));
    fr.tmr_state = last_tmr_state;
    fr.cel_latched = sf.cel_latched ? 1u : 0u;
    finalize_vrta_frame(fr);
    trace.push_back(fr);

    // Check safety invariants
    if (res.max_tilt_deg > 75.0) {
      res.passed = false;
      res.violated_invariant = "MAX_TILT_CAPSIZE";
      res.root_cause_class = "CONTROL_AUTHORITY_CLIFF";
      break;
    }
    if (!is_cel) {
      const double obs_dist = std::hypot(fin.pos_ned.x - hazards[0].pos_ned.x, fin.pos_ned.y - hazards[0].pos_ned.y);
      const double obs_dz = std::abs(fin.pos_ned.z - hazards[0].pos_ned.z);
      if (obs_dist < 4.5 && obs_dz < hazards[0].radius_z_m) {
        res.passed = false;
        res.violated_invariant = "CBVF_COLLISION_BREACH";
        res.root_cause_class = "QP_INFEASIBLE";
        break;
      }
      if (approx_ltfj_east_margin_m(sc.start_lat_deg, sc.start_lon_deg, pos_enu.x) < 0.0) {
        res.passed = false;
        res.violated_invariant = "CTR_LTFJ_BREACH";
        res.root_cause_class = "CTR_BARRIER_VIOLATION";
        break;
      }
    }
    if (fin.battery.voltage_collapse || fin.battery.v_term_v < 330.0f) {
      res.passed = false;
      res.violated_invariant = "BATTERY_VOLTAGE_COLLAPSE";
      res.root_cause_class = "ENERGY_RESERVE_EXHAUSTION";
      break;
    }
    if (landed) {
      if (res.touchdown_vz_mps > 2.0) {
        res.passed = false;
        res.violated_invariant = "TOUCHDOWN_VZ_EXCEEDED";
        res.root_cause_class = "CONTROL_AUTHORITY_CLIFF";
      }
      break;
    }
  }

  if (res.passed && is_cel && !landed) {
    res.passed = false;
    res.violated_invariant = "CEL_LANDING_TIMEOUT";
    res.root_cause_class = "SIM_TIMEOUT";
  }
  if (res.passed && sc.imu_fault_channel >= 0 && !imu_isolated) {
    res.passed = false;
    res.violated_invariant = "TMR_ISOLATION_TIMEOUT";
    res.root_cause_class = "TMR_ISOLATION_TIMEOUT";
  }

  res.mean_yaw_rate_rad_s = (yaw_rate_cnt > 0) ? (yaw_rate_sum / static_cast<double>(yaw_rate_cnt)) : 0.0;

  if (!res.passed && !vrta_dir.empty()) {
    std::ostringstream fn;
    fn << vrta_dir << "/run_" << std::setw(5) << std::setfill('0') << sc.run_id << ".vrta";
    res.vrta_path = fn.str();
    write_vrta_trace(res.vrta_path, trace, lut);
  }

  return res;
}

std::string json_escape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 4);
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string manifest_path;
  std::string out_path;
  std::string lut_path = "tests/fixtures/quick_test.vrta_lut";
  std::string cel_path = "tests/fixtures/quick_test.vrta_cel2d";
  std::string cbf_mode = "hj_cbvf";
  std::string alloc_mode = "indi_wls";
  std::string vrta_dir;
  bool no_vrta = false;
  int max_runs = 0;
  int n_threads = static_cast<int>(std::max(1u, std::min(12u, std::thread::hardware_concurrency())));

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto next_val = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string{};
    };
    if (arg == "--manifest") {
      manifest_path = next_val();
    } else if (arg == "--out") {
      out_path = next_val();
    } else if (arg == "--lut") {
      lut_path = next_val();
    } else if (arg == "--cel-lut") {
      cel_path = next_val();
    } else if (arg == "--cbf-mode") {
      cbf_mode = next_val();
    } else if (arg == "--alloc-mode") {
      alloc_mode = next_val();
    } else if (arg == "--vrta-dir") {
      vrta_dir = next_val();
    } else if (arg == "--no-vrta") {
      no_vrta = true;
    } else if (arg == "--max-runs") {
      max_runs = std::stoi(next_val());
    } else if (arg == "--threads") {
      n_threads = std::max(1, std::stoi(next_val()));
    }
  }

  if (manifest_path.empty() || out_path.empty()) {
    std::cerr << "Usage: vertirta_6dof_batch_cli --manifest <in.json> --out <results.json> "
              << "[--lut <lut>] [--cel-lut <cel>] [--cbf-mode hj_cbvf|hocbf] "
              << "[--alloc-mode indi_wls|stock_ca] [--vrta-dir <dir>] [--no-vrta] [--max-runs N]\n";
    return 2;
  }

  std::ifstream mf(manifest_path);
  if (!mf.is_open()) {
    std::cerr << "Cannot open manifest: " << manifest_path << "\n";
    return 1;
  }
  const std::string content((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
  MiniJsonParser parser(content);
  std::vector<ScenarioSpec> scenarios = parser.parse_manifest();
  if (max_runs > 0 && static_cast<std::size_t>(max_runs) < scenarios.size()) {
    scenarios.resize(static_cast<std::size_t>(max_runs));
  }

  MmapCbvf4DLut lut;
  Cel2DLut cel;
  (void)lut.load(lut_path);
  (void)cel.load(cel_path);

  if (no_vrta) {
    vrta_dir.clear();
  } else if (vrta_dir.empty()) {
    const std::filesystem::path op(out_path);
    vrta_dir = (op.has_parent_path() ? (op.parent_path() / "vrta_traces") : std::filesystem::path("vrta_traces")).string();
  }

  const bool use_hj_cbvf = (cbf_mode != "hocbf");
  const bool use_indi_wls = (alloc_mode != "stock_ca");

  std::vector<ScenarioResult> results(scenarios.size());
  std::atomic<std::size_t> next_idx{0};
  const int workers = std::min(n_threads, static_cast<int>(std::max<std::size_t>(1, scenarios.size())));

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(workers));
  for (int t = 0; t < workers; ++t) {
    pool.emplace_back([&]() {
      while (true) {
        const std::size_t idx = next_idx.fetch_add(1u, std::memory_order_relaxed);
        if (idx >= scenarios.size()) {
          break;
        }
        results[idx] = run_single_scenario(
            scenarios[idx],
            lut.is_loaded() ? &lut : nullptr,
            cel.is_loaded() ? &cel : nullptr,
            use_hj_cbvf,
            use_indi_wls,
            vrta_dir);
      }
    });
  }
  for (auto& th : pool) {
    th.join();
  }

  std::filesystem::path op(out_path);
  if (op.has_parent_path()) {
    std::error_code ec;
    std::filesystem::create_directories(op.parent_path(), ec);
  }
  std::ofstream out(out_path, std::ios::trunc);
  if (!out.is_open()) {
    std::cerr << "Cannot write results JSON: " << out_path << "\n";
    return 1;
  }

  out << std::fixed << std::setprecision(4);
  out << "{\n"
      << "  \"schema_version\": \"vertirta.tier1_batch/v1.0.0\",\n"
      << "  \"cbf_mode\": \"" << json_escape(cbf_mode) << "\",\n"
      << "  \"alloc_mode\": \"" << json_escape(alloc_mode) << "\",\n"
      << "  \"runs\": [\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    out << "    {"
        << "\"run_id\":" << r.spec.run_id
        << ",\"stratum\":\"" << json_escape(r.spec.stratum) << "\""
        << ",\"outcome_class\":\"" << json_escape(r.spec.outcome_class) << "\""
        << ",\"fault_topology\":\"" << json_escape(r.spec.fault_topology) << "\""
        << ",\"seed\":" << r.spec.seed
        << ",\"passed\":" << (r.passed ? "true" : "false")
        << ",\"violated_invariant\":"
        << (r.violated_invariant.empty() ? "null" : ("\"" + json_escape(r.violated_invariant) + "\""))
        << ",\"root_cause_class\":"
        << (r.root_cause_class.empty() ? "null" : ("\"" + json_escape(r.root_cause_class) + "\""))
        << ",\"min_cbvf_margin_m\":" << r.min_cbvf_margin_m
        << ",\"touchdown_vz_mps\":" << r.touchdown_vz_mps
        << ",\"max_tilt_deg\":" << r.max_tilt_deg
        << ",\"min_v_term_v\":" << r.min_v_term_v
        << ",\"cel_latched\":" << (r.cel_latched ? "true" : "false")
        << ",\"mean_yaw_rate_rad_s\":" << r.mean_yaw_rate_rad_s
        << ",\"vrta_path\":"
        << (r.vrta_path.empty() ? "null" : ("\"" + json_escape(r.vrta_path) + "\""))
        << "}" << ((i + 1 < results.size()) ? "," : "") << "\n";
  }
  out << "  ]\n}\n";
  return 0;
}
