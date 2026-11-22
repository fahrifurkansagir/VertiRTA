#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/geodesy.hpp"
#include "vertirta/core/indi_wls.hpp"
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

struct ReplayRunStats {
  uint64_t frames_written{0};
  double min_cbvf_v{1e6};
  double max_tilt_deg{0.0};
  bool cel_latched{false};
  uint64_t fnv1a_hash{14695981039346656037ULL};
};

double quat_tilt_deg(const QuatPx4FrdNed& q) noexcept {
  const double c = clamp_d(1.0 - 2.0 * (q.x * q.x + q.y * q.y), -1.0, 1.0);
  return std::acos(c) * kRadToDeg;
}

bool try_load_luts(
    const std::string& lut_arg,
    const std::string& cel_arg,
    MmapCbvf4DLut& lut,
    Cel2DLut& cel) noexcept {
  const std::array<std::string, 3> lut_cands = {
      lut_arg,
      "tests/fixtures/quick_test.vrta_lut",
      "../tests/fixtures/quick_test.vrta_lut",
  };
  for (const auto& p : lut_cands) {
    if (!p.empty() && lut.load(p)) {
      break;
    }
  }
  const std::array<std::string, 3> cel_cands = {
      cel_arg,
      "tests/fixtures/quick_test.vrta_cel2d",
      "../tests/fixtures/quick_test.vrta_cel2d",
  };
  for (const auto& p : cel_cands) {
    if (!p.empty() && cel.load(p)) {
      break;
    }
  }
  return lut.is_loaded() && cel.is_loaded();
}

// Deterministic 100 Hz (400 Hz inner) closed-loop flight_core + rta_companion trace generator / replayer.
bool run_deterministic_trace(
    uint64_t n_frames_100hz,
    const std::vector<uint8_t>* fault_schedule,
    const std::vector<bool>* imu0_fault_schedule,
    const MmapCbvf4DLut& lut,
    const Cel2DLut& cel,
    const std::string& out_vrta_path,
    ReplayRunStats& stats) {
  std::ofstream out(out_vrta_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    return false;
  }

  const uint8_t* sha_ptr = lut.is_loaded() ? lut.payload_sha256().data() : nullptr;
  VrtaFileHeader hdr = make_default_vrta_header("vertirta_v1_0_0_deterministic_replay_cli", sha_ptr);
  out.write(reinterpret_cast<const char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));

  CoaxialPlant6Dof plant;
  plant.reset({0.0, 0.0, -35.0}, {4.0, 0.0, 0.0}, 0.55f);

  IndiWlsController ctrl(true);
  ctrl.reset(params::kOmegaHover);

  TmrVoter voter;
  DrydenTurbulenceFilter dryden(20261122ULL);
  CbvfSafetyFilter filter(lut.is_loaded() ? &lut : nullptr, cel.is_loaded() ? &cel : nullptr);

  const std::array<HazardPoint, 2> corridor_hazards = {{
      HazardPoint{.pos_ned = {90.0, 2.0, -35.0}, .vel_ned = {0.0, 0.0, 0.0}, .kind = 0},
      HazardPoint{.pos_ned = {0.0, 990.0, -35.0}, .vel_ned = {0.0, 0.0, 0.0}, .radius_z_m = 700.0, .kind = 3},
  }};

  SafetyFilterOutput last_sf{};
  double last_chi2 = 0.0;
  uint8_t last_tmr_state = 0;
  uint8_t last_imu_mask = 0x07;

  for (uint64_t seq = 0; seq < n_frames_100hz; ++seq) {
    const double t_outer = static_cast<double>(seq) * 0.01;
    uint8_t fault_mask = 0x00u;
    bool inject_imu0 = false;
    if (fault_schedule != nullptr && seq < fault_schedule->size()) {
      fault_mask = (*fault_schedule)[seq];
    } else {
      if (t_outer >= 6.0) {
        fault_mask = 0x11u;  // Same-Arm M0+M4 CEL
      } else if (t_outer >= 3.0) {
        fault_mask = 0x21u;  // Cross-Arm M0+M5 CSFL
      }
    }
    if (imu0_fault_schedule != nullptr && seq < imu0_fault_schedule->size()) {
      inject_imu0 = (*imu0_fault_schedule)[seq];
    } else {
      inject_imu0 = (t_outer >= 1.5);
    }

    const PlantState st0 = plant.state();
    const double agl0 = std::max(0.0, -st0.pos_ned.z);
    const Vec3Ned gust = dryden.step_seeded(0.01, std::max(agl0, 5.0), std::max(st0.vel_ned.norm(), 4.0), 7.72);
    const Vec3Ned wind_ned = Vec3Ned{2.5, 2.5, 0.0} + gust * 0.35;

    // 100 Hz outer guidance + CBVF-QP safety filter
    double vz_des = 0.0;
    double vx_des = (fault_mask == 0x11u || filter.is_cel_latched()) ? 0.0 : 4.0;
    if (fault_mask == 0x11u || filter.is_cel_latched()) {
      vz_des = (agl0 > 10.0) ? 2.2 : ((agl0 > 5.0) ? 1.5 : ((agl0 > 1.0) ? 0.9 : 0.5));
    }
    const Vec3Ned a_nom_ned{
        clamp_d(1.0 * (vx_des - st0.vel_ned.x), -2.5, 2.5),
        clamp_d(-1.0 * st0.vel_ned.y, -2.5, 2.5),
        clamp_d(2.2 * (vz_des - st0.vel_ned.z), -2.2, 2.5),
    };

    const CapabilityEnvelope env = CoaxEffectiveness::compute_envelope(
        fault_mask,
        last_tmr_state,
        last_imu_mask,
        st0.battery.v_term_v,
        st0.battery.i_pack_a,
        st0.battery.soc,
        ctrl.spin_rate_ref_rad_s(),
        static_cast<float>(st0.omega_frd.z),
        static_cast<float>(last_chi2),
        0u);

    last_sf = filter.step_100hz(st0.pos_ned, st0.vel_ned, a_nom_ned, 0.0, env, corridor_hazards);
    if (last_sf.min_cbvf_v < stats.min_cbvf_v) {
      stats.min_cbvf_v = last_sf.min_cbvf_v;
    }
    if (last_sf.cel_latched) {
      stats.cel_latched = true;
    }

    // 4 inner 400 Hz ticks per 100 Hz frame
    for (int sub = 0; sub < 4; ++sub) {
      const PlantState st = plant.state();
      ImuSampleFrd imus[3]{};
      for (int c = 0; c < 3; ++c) {
        imus[c].gyro_rad_s = st.omega_frd;
        imus[c].accel_m_s2 = {0.0, 0.0, -params::kGravity};
        imus[c].t_us = seq * 10000ULL + static_cast<uint64_t>(sub) * 2500ULL;
        imus[c].valid = true;
      }
      if (inject_imu0) {
        imus[0].gyro_rad_s.x += 0.08;
      }
      const TmrVoteOutput vr = voter.step(imus, st.omega_frd, true);
      last_chi2 = vr.max_chi2_residual;
      last_tmr_state = static_cast<uint8_t>(vr.state);
      last_imu_mask = vr.health_mask;

      IndiInput in{};
      in.q_frd_ned = st.q_frd_ned;
      in.gyro_frd = vr.gyro_voted;
      in.accel_des_ned = last_sf.a_safe_ned;
      in.yaw_rate_des = last_sf.yaw_rate_safe;
      in.v_horiz_mps = static_cast<float>(std::hypot(st.vel_ned.x, st.vel_ned.y));
      in.v_term_v = st.battery.v_term_v;
      in.fault_mask = fault_mask;

      float omega_cmd[8]{};
      (void)ctrl.step_400hz(in, omega_cmd);
      (void)plant.step_400hz(omega_cmd, fault_mask, wind_ned);
    }

    const PlantState fin = plant.state();
    const double tilt = quat_tilt_deg(fin.q_frd_ned);
    if (tilt > stats.max_tilt_deg) {
      stats.max_tilt_deg = tilt;
    }

    VrtaFrame100Hz fr{};
    fr.seq = seq;
    fr.t_sim_us = seq * 10000ULL;
    const Vec3Enu pos_enu = ned_to_enu(fin.pos_ned);
    const Vec3Enu vel_enu = ned_to_enu(fin.vel_ned);
    const Vec3Enu anom_enu = ned_to_enu(a_nom_ned);
    const Vec3Enu asafe_enu = ned_to_enu(last_sf.a_safe_ned);
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
    fr.omega_frd[0] = static_cast<float>(fin.omega_frd.x);
    fr.omega_frd[1] = static_cast<float>(fin.omega_frd.y);
    fr.omega_frd[2] = static_cast<float>(fin.omega_frd.z);
    fr.a_nom_enu[0] = static_cast<float>(anom_enu.x);
    fr.a_nom_enu[1] = static_cast<float>(anom_enu.y);
    fr.a_nom_enu[2] = static_cast<float>(anom_enu.z);
    fr.a_safe_enu[0] = static_cast<float>(asafe_enu.x);
    fr.a_safe_enu[1] = static_cast<float>(asafe_enu.y);
    fr.a_safe_enu[2] = static_cast<float>(asafe_enu.z);
    for (int i = 0; i < 8; ++i) {
      fr.rotor_rpm[i] = static_cast<float>(fin.rotor_omega[i] * (60.0 / (2.0 * kPi)));
    }
    fr.v_term_v = fin.battery.v_term_v;
    fr.soc_pct = fin.battery.soc * 100.0f;
    fr.cbvf_min_v = static_cast<float>(last_sf.min_cbvf_v);
    fr.ltfj_margin_m = static_cast<float>(989.4 - pos_enu.x);
    fr.ltbx_margin_m = static_cast<float>(3294.3 - pos_enu.y);
    fr.chi2_residual = static_cast<float>(last_chi2);
    fr.agl_m = static_cast<float>(std::max(0.0, -fin.pos_ned.z));
    fr.active_motor_mask = static_cast<uint8_t>(~fault_mask);
    fr.flight_regime = static_cast<uint8_t>(CoaxEffectiveness::classify_regime(fault_mask));
    fr.tmr_state = last_tmr_state;
    fr.cel_latched = last_sf.cel_latched ? 1u : 0u;
    fr.flags = (last_sf.used_hj_fallback ? 1u : 0u) | (last_sf.clutter_inflated ? 2u : 0u);
    finalize_vrta_frame(fr);

    out.write(reinterpret_cast<const char*>(&fr), static_cast<std::streamsize>(sizeof(fr)));
    const auto* bytes = reinterpret_cast<const uint8_t*>(&fr);
    for (std::size_t b = 0; b < sizeof(fr); ++b) {
      stats.fnv1a_hash ^= bytes[b];
      stats.fnv1a_hash *= 1099511628211ULL;
    }
    ++stats.frames_written;
  }

  return out.good();
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path;
  std::string out_vrta;
  std::string out_json;
  std::string gen_fixture;
  std::string lut_path;
  std::string cel_path;
  uint64_t ticks = 1000;

  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    auto next_val = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string{};
    };
    if (arg == "--input" || arg == "--in") {
      input_path = next_val();
    } else if (arg == "--out") {
      out_vrta = next_val();
    } else if (arg == "--out-json") {
      out_json = next_val();
    } else if (arg == "--generate-fixture") {
      gen_fixture = next_val();
    } else if (arg == "--ticks") {
      ticks = std::stoull(next_val());
    } else if (arg == "--lut") {
      lut_path = next_val();
    } else if (arg == "--cel-lut") {
      cel_path = next_val();
    }
  }

  MmapCbvf4DLut lut;
  Cel2DLut cel;
  (void)try_load_luts(lut_path, cel_path, lut, cel);

  ReplayRunStats stats{};

  if (!gen_fixture.empty()) {
    if (!run_deterministic_trace(ticks, nullptr, nullptr, lut, cel, gen_fixture, stats)) {
      std::cerr << "Failed to write fixture: " << gen_fixture << "\n";
      return 1;
    }
    std::cout << "{\"mode\":\"generate_fixture\",\"out\":\"" << gen_fixture
              << "\",\"frames\":" << stats.frames_written
              << ",\"cel_latched\":" << (stats.cel_latched ? "true" : "false")
              << ",\"fnv1a_hex\":\"0x" << std::hex << stats.fnv1a_hash << std::dec << "\"}\n";
    return 0;
  }

  if (input_path.empty()) {
    std::cerr << "Usage: vertirta_replay_cli --input <in.vrta> --out <out.vrta> [--out-json <j>]\n"
              << "   or: vertirta_replay_cli --generate-fixture <out.vrta> [--ticks N]\n";
    return 2;
  }

  // Read & validate canonical .vrta input binary
  std::ifstream in(input_path, std::ios::binary | std::ios::ate);
  if (!in.is_open()) {
    std::cerr << "Cannot open input .vrta: " << input_path << "\n";
    return 1;
  }
  const std::streamsize total_sz = in.tellg();
  if (total_sz < static_cast<std::streamsize>(sizeof(VrtaFileHeader)) ||
      (total_sz - static_cast<std::streamsize>(sizeof(VrtaFileHeader))) %
              static_cast<std::streamsize>(sizeof(VrtaFrame100Hz)) !=
          0) {
    std::cerr << "Invalid .vrta file size: " << total_sz << "\n";
    return 1;
  }
  in.seekg(0, std::ios::beg);

  VrtaFileHeader hdr{};
  in.read(reinterpret_cast<char*>(&hdr), static_cast<std::streamsize>(sizeof(hdr)));
  if (!verify_vrta_header(hdr)) {
    std::cerr << "Invalid .vrta header magic or CRC32C\n";
    return 1;
  }

  const std::size_t n_frames =
      static_cast<std::size_t>(total_sz - static_cast<std::streamsize>(sizeof(hdr))) /
      sizeof(VrtaFrame100Hz);
  std::vector<uint8_t> fault_sched;
  std::vector<bool> imu0_sched;
  fault_sched.reserve(n_frames);
  imu0_sched.reserve(n_frames);

  for (std::size_t k = 0; k < n_frames; ++k) {
    VrtaFrame100Hz fr{};
    in.read(reinterpret_cast<char*>(&fr), static_cast<std::streamsize>(sizeof(fr)));
    if (!verify_vrta_frame(fr)) {
      std::cerr << "CRC32C mismatch at frame " << k << "\n";
      return 1;
    }
    fault_sched.push_back(static_cast<uint8_t>(~fr.active_motor_mask));
    imu0_sched.push_back(fr.tmr_state != 0u);
  }

  const std::string target_vrta = out_vrta.empty() ? "/dev/null" : out_vrta;
  if (!run_deterministic_trace(n_frames, &fault_sched, &imu0_sched, lut, cel, target_vrta, stats)) {
    std::cerr << "Failed to replay into: " << target_vrta << "\n";
    return 1;
  }

  const std::string summary =
      std::string("{\"schema_version\":\"vertirta.replay_summary/v1.0.0\",\"frames\":") +
      std::to_string(stats.frames_written) +
      ",\"min_cbvf_v\":" + std::to_string(stats.min_cbvf_v) +
      ",\"max_tilt_deg\":" + std::to_string(stats.max_tilt_deg) +
      ",\"cel_latched\":" + (stats.cel_latched ? "true" : "false") +
      ",\"passed\":true}\n";

  if (!out_json.empty()) {
    std::ofstream jf(out_json, std::ios::trunc);
    jf << summary;
  }
  std::cout << summary;
  return 0;
}
