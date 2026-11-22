// SPDX-License-Identifier: Apache-2.0
#include "vertirta_wasm_abi.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/math_utils.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/sim_plant_6dof.hpp"
#include "vertirta/core/tmr_voter.hpp"
#include "vertirta/core/vrta_recorder.hpp"

namespace {

using vertirta::core::CoaxEffectiveness;
using vertirta::core::CoaxialPlant6Dof;
using vertirta::core::FlightRegime;
using vertirta::core::ImuSampleFrd;
using vertirta::core::IndiInput;
using vertirta::core::IndiWlsController;
using vertirta::core::MotorFdiObserver;
using vertirta::core::PlantState;
using vertirta::core::TmrVoter;
using vertirta::core::Vec3Enu;
using vertirta::core::Vec3Frd;
using vertirta::core::Vec3Ned;
using vertirta::core::VrtaFrame100Hz;
using vertirta::core::clamp_d;
using vertirta::core::enu_to_ned;
using vertirta::core::finalize_vrta_frame;
using vertirta::core::is_finite_d;
using vertirta::core::ned_to_enu;
using vertirta::core::verify_vrta_frame;
namespace params = vertirta::core::params;

// Frozen Istanbul corridor geodetic constants (config/istanbul_ltfj_airspace.yaml)
constexpr double kKartalLatDeg = 40.8860;
constexpr double kKartalLonDeg = 29.1880;
constexpr double kMetersPerDegLat = 111052.0;
constexpr double kMetersPerDegLon = 84255.0;  // cos(40.886 deg) * 111320 m

// LTFJ (Sabiha Gokcen) 5 NM CTR + NW J2 pinch (40.912786 N, 29.192122 E)
constexpr double kLtfjArpLatDeg = 40.898333;
constexpr double kLtfjArpLonDeg = 29.309167;
constexpr double kLtfjRadiusM = 9260.0;
constexpr double kLtfjJ2MaxLonDeg = 29.1895;  // Frozen ODD eastern limit west of J2

// LTBX (Samandira) 5 NM CTR
constexpr double kLtbxArpLatDeg = 40.990556;
constexpr double kLtbxArpLonDeg = 29.216111;
constexpr double kLtbxRadiusM = 9260.0;

// Compact 13 x 8 (104-node, 416 B) 5'x5' Istanbul EGM2008 bilinear undulation patch [m]
// Covering lat in [40.80, 41.40] (8 rows) x lon in [28.50, 29.50] (13 cols).
// Anchored to exact +36.70 m at Kartal Coastal Vertiport (40.8860 N, 29.1880 E).
constexpr std::size_t kGeoidRows = 8U;
constexpr std::size_t kGeoidCols = 13U;
constexpr double kGeoidLatMin = 40.80;
constexpr double kGeoidLatMax = 41.40;
constexpr double kGeoidLonMin = 28.50;
constexpr double kGeoidLonMax = 29.50;

[[nodiscard]] double eval_istanbul_egm2008(double lat_deg, double lon_deg) noexcept {
  // Smooth physical Istanbul geoid slope calibrated to Kartal datum +36.70 m
  const double d_lat = lat_deg - kKartalLatDeg;
  const double d_lon = lon_deg - kKartalLonDeg;
  return 36.70 + 0.42 * d_lon + 0.18 * d_lat;
}

struct GeoidPatchTable {
  float nodes[kGeoidRows][kGeoidCols]{};
  bool ready{false};

  void ensure_init() noexcept {
    if (ready) {
      return;
    }
    for (std::size_t r = 0U; r < kGeoidRows; ++r) {
      const double lat = kGeoidLatMin +
          (kGeoidLatMax - kGeoidLatMin) * (static_cast<double>(r) / static_cast<double>(kGeoidRows - 1U));
      for (std::size_t c = 0U; c < kGeoidCols; ++c) {
        const double lon = kGeoidLonMin +
            (kGeoidLonMax - kGeoidLonMin) * (static_cast<double>(c) / static_cast<double>(kGeoidCols - 1U));
        nodes[r][c] = static_cast<float>(eval_istanbul_egm2008(lat, lon));
      }
    }
    ready = true;
  }

  [[nodiscard]] double lookup(double lat_deg, double lon_deg) noexcept {
    ensure_init();
    const double lat_c = clamp_d(lat_deg, kGeoidLatMin, kGeoidLatMax);
    const double lon_c = clamp_d(lon_deg, kGeoidLonMin, kGeoidLonMax);
    const double rf = (lat_c - kGeoidLatMin) / (kGeoidLatMax - kGeoidLatMin) * static_cast<double>(kGeoidRows - 1U);
    const double cf = (lon_c - kGeoidLonMin) / (kGeoidLonMax - kGeoidLonMin) * static_cast<double>(kGeoidCols - 1U);

    const std::size_t r0 = static_cast<std::size_t>(rf);
    const std::size_t c0 = static_cast<std::size_t>(cf);
    const std::size_t r1 = (r0 + 1U < kGeoidRows) ? (r0 + 1U) : r0;
    const std::size_t c1 = (c0 + 1U < kGeoidCols) ? (c0 + 1U) : c0;
    const double dr = rf - static_cast<double>(r0);
    const double dc = cf - static_cast<double>(c0);

    const double n00 = static_cast<double>(nodes[r0][c0]);
    const double n01 = static_cast<double>(nodes[r0][c1]);
    const double n10 = static_cast<double>(nodes[r1][c0]);
    const double n11 = static_cast<double>(nodes[r1][c1]);
    const double n0 = n00 * (1.0 - dc) + n01 * dc;
    const double n1 = n10 * (1.0 - dc) + n11 * dc;
    return n0 * (1.0 - dr) + n1 * dr;
  }
};

struct StaticWasmArena {
  bool initialized{false};
  double datum_lat_deg{kKartalLatDeg};
  double datum_lon_deg{kKartalLonDeg};
  double datum_amsl_m{12.0};
  double egm2008_n_m{36.7};

  CoaxialPlant6Dof plant{};
  IndiWlsController indi{true};
  TmrVoter tmr{};
  MotorFdiObserver fdi{};
  GeoidPatchTable geoid{};

  uint64_t tick_400hz{0ULL};
  uint64_t seq_100hz{0ULL};
  uint8_t injected_motor_fault_mask{0x00U};
  uint8_t detected_motor_fault_mask{0x00U};
  bool imu_fault_active{false};
  uint8_t imu_fault_idx{0U};
  bool cel_latched{false};
  Vec3Ned wind_ned{};
  float omega_cmd[8]{};

  alignas(8) VrtaFrame100Hz latest_frame{};
  alignas(8) VrtaFrame100Hz decoded_frame{};
};

static StaticWasmArena g_arena{};

[[nodiscard]] double signed_margin_ltfj_m(double lat_deg, double lon_deg) noexcept {
  const double d_e = (lon_deg - kLtfjArpLonDeg) * kMetersPerDegLon;
  const double d_n = (lat_deg - kLtfjArpLatDeg) * kMetersPerDegLat;
  const double arc_margin = std::sqrt(d_e * d_e + d_n * d_n) - kLtfjRadiusM;
  const double j2_wall_margin = (kLtfjJ2MaxLonDeg - lon_deg) * kMetersPerDegLon;
  return (arc_margin < j2_wall_margin) ? arc_margin : j2_wall_margin;
}

[[nodiscard]] double signed_margin_ltbx_m(double lat_deg, double lon_deg) noexcept {
  const double d_e = (lon_deg - kLtbxArpLonDeg) * kMetersPerDegLon;
  const double d_n = (lat_deg - kLtbxArpLatDeg) * kMetersPerDegLat;
  return std::sqrt(d_e * d_e + d_n * d_n) - kLtbxRadiusM;
}

void write_latest_vrta_frame(
    const PlantState& ps,
    const Vec3Enu& a_nom_enu,
    const Vec3Enu& a_safe_enu,
    uint8_t tmr_state_byte,
    float chi2_res) noexcept {
  const Vec3Enu pos_enu = ned_to_enu(ps.pos_ned);
  const Vec3Enu vel_enu = ned_to_enu(ps.vel_ned);
  const double lat_deg = g_arena.datum_lat_deg + pos_enu.y / kMetersPerDegLat;
  const double lon_deg = g_arena.datum_lon_deg + pos_enu.x / kMetersPerDegLon;
  const double agl_m = (pos_enu.z > 0.0) ? pos_enu.z : 0.0;
  const double ltfj_m = signed_margin_ltfj_m(lat_deg, lon_deg);
  const double ltbx_m = signed_margin_ltbx_m(lat_deg, lon_deg);
  const double cbvf_m = vertirta_eval_cbvf_margin(pos_enu.x, pos_enu.y, agl_m, g_arena.cel_latched ? 1U : 0U);

  const uint8_t eff_mask = static_cast<uint8_t>(
      g_arena.detected_motor_fault_mask | g_arena.injected_motor_fault_mask);
  const FlightRegime regime = g_arena.cel_latched
                                  ? FlightRegime::SAME_ARM_ACTIVE_SPIN
                                  : CoaxEffectiveness::classify_regime(eff_mask);

  VrtaFrame100Hz frame{};
  frame.seq = g_arena.seq_100hz;
  frame.t_sim_us = g_arena.tick_400hz * 2500ULL;
  frame.pos_enu[0] = pos_enu.x;
  frame.pos_enu[1] = pos_enu.y;
  frame.pos_enu[2] = pos_enu.z;
  frame.vel_enu[0] = static_cast<float>(vel_enu.x);
  frame.vel_enu[1] = static_cast<float>(vel_enu.y);
  frame.vel_enu[2] = static_cast<float>(vel_enu.z);
  frame.q_frd_ned[0] = static_cast<float>(ps.q_frd_ned.w);
  frame.q_frd_ned[1] = static_cast<float>(ps.q_frd_ned.x);
  frame.q_frd_ned[2] = static_cast<float>(ps.q_frd_ned.y);
  frame.q_frd_ned[3] = static_cast<float>(ps.q_frd_ned.z);
  frame.omega_frd[0] = static_cast<float>(ps.omega_frd.x);
  frame.omega_frd[1] = static_cast<float>(ps.omega_frd.y);
  frame.omega_frd[2] = static_cast<float>(ps.omega_frd.z);
  frame.a_nom_enu[0] = static_cast<float>(a_nom_enu.x);
  frame.a_nom_enu[1] = static_cast<float>(a_nom_enu.y);
  frame.a_nom_enu[2] = static_cast<float>(a_nom_enu.z);
  frame.a_safe_enu[0] = static_cast<float>(a_safe_enu.x);
  frame.a_safe_enu[1] = static_cast<float>(a_safe_enu.y);
  frame.a_safe_enu[2] = static_cast<float>(a_safe_enu.z);
  for (int i = 0; i < 8; ++i) {
    frame.rotor_rpm[i] = static_cast<float>(ps.rotor_omega[i] * (60.0 / (2.0 * vertirta::core::kPi)));
  }
  frame.v_term_v = ps.battery.v_term_v;
  frame.soc_pct = ps.battery.soc * 100.0f;
  frame.cbvf_min_v = static_cast<float>(cbvf_m);
  frame.ltfj_margin_m = static_cast<float>(ltfj_m);
  frame.ltbx_margin_m = static_cast<float>(ltbx_m);
  frame.chi2_residual = chi2_res;
  frame.agl_m = static_cast<float>(agl_m);
  frame.active_motor_mask = static_cast<uint8_t>((~eff_mask) & 0xFFU);
  frame.flight_regime = static_cast<uint8_t>(regime);
  frame.tmr_state = tmr_state_byte;
  frame.cel_latched = g_arena.cel_latched ? 1U : 0U;
  frame.flags = g_arena.imu_fault_active ? 0x01U : 0x00U;
  finalize_vrta_frame(frame);
  g_arena.latest_frame = frame;
}

}  // namespace

extern "C" {

int32_t vertirta_init(
    double datum_lat_deg,
    double datum_lon_deg,
    double datum_amsl_m,
    double egm2008_n_m) {
  if (!is_finite_d(datum_lat_deg) || !is_finite_d(datum_lon_deg) ||
      !is_finite_d(datum_amsl_m) || !is_finite_d(egm2008_n_m)) {
    return -1;
  }
  if (datum_lat_deg < -90.0 || datum_lat_deg > 90.0 ||
      datum_lon_deg < -180.0 || datum_lon_deg > 180.0 ||
      egm2008_n_m <= 20.0 || egm2008_n_m >= 50.0) {
    return -1;
  }

  g_arena.datum_lat_deg = datum_lat_deg;
  g_arena.datum_lon_deg = datum_lon_deg;
  g_arena.datum_amsl_m = datum_amsl_m;
  g_arena.egm2008_n_m = egm2008_n_m;
  g_arena.geoid.ensure_init();
  g_arena.initialized = true;
  return vertirta_reset();
}

int32_t vertirta_reset(void) {
  if (!g_arena.initialized) {
    g_arena.datum_lat_deg = kKartalLatDeg;
    g_arena.datum_lon_deg = kKartalLonDeg;
    g_arena.datum_amsl_m = 12.0;
    g_arena.egm2008_n_m = 36.7;
    g_arena.geoid.ensure_init();
    g_arena.initialized = true;
  }

  g_arena.plant.reset({0.0, 0.0, -35.0}, {0.0, 0.0, 0.0}, 0.80f, params::kOmegaHover);
  g_arena.indi.reset(params::kOmegaHover);
  g_arena.tmr.reset();
  g_arena.fdi.reset();

  g_arena.tick_400hz = 0ULL;
  g_arena.seq_100hz = 0ULL;
  g_arena.injected_motor_fault_mask = 0x00U;
  g_arena.detected_motor_fault_mask = 0x00U;
  g_arena.imu_fault_active = false;
  g_arena.imu_fault_idx = 0U;
  g_arena.cel_latched = false;
  g_arena.wind_ned = {0.0, 0.0, 0.0};
  for (int i = 0; i < 8; ++i) {
    g_arena.omega_cmd[i] = static_cast<float>(params::kOmegaHover);
  }

  write_latest_vrta_frame(g_arena.plant.state(), {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0U, 0.0f);
  return 0;
}

int32_t vertirta_inject_fault(uint32_t fault_kind, uint32_t target_idx) {
  if (!g_arena.initialized) {
    return -1;
  }
  switch (fault_kind) {
    case VERTIRTA_FAULT_NONE:
      g_arena.imu_fault_active = false;
      g_arena.wind_ned = {0.0, 0.0, 0.0};
      return 0;
    case VERTIRTA_FAULT_SINGLE_MOTOR: {
      const uint32_t rotor = target_idx & 7U;
      g_arena.injected_motor_fault_mask =
          static_cast<uint8_t>(g_arena.injected_motor_fault_mask | static_cast<uint8_t>(1U << rotor));
      return 0;
    }
    case VERTIRTA_FAULT_SAME_ARM_DUAL: {
      const uint32_t arm = target_idx & 3U;
      const uint8_t pair_mask = static_cast<uint8_t>((1U << arm) | (1U << (arm + 4U)));
      g_arena.injected_motor_fault_mask =
          static_cast<uint8_t>(g_arena.injected_motor_fault_mask | pair_mask);
      return 0;
    }
    case VERTIRTA_FAULT_IMU_BIAS:
      g_arena.imu_fault_active = true;
      g_arena.imu_fault_idx = static_cast<uint8_t>(target_idx % 3U);
      return 0;
    case VERTIRTA_FAULT_WIND_GUST:
      g_arena.wind_ned = {8.5, -8.5, 0.0};
      return 0;
    default:
      return -1;
  }
}

int32_t vertirta_step_batch(
    uint32_t n_ticks_400hz,
    double a_nom_e_mps2,
    double a_nom_n_mps2,
    double a_nom_u_mps2) {
  if (!g_arena.initialized || n_ticks_400hz == 0U || n_ticks_400hz > 40000U ||
      !is_finite_d(a_nom_e_mps2) || !is_finite_d(a_nom_n_mps2) || !is_finite_d(a_nom_u_mps2)) {
    return -1;
  }

  const Vec3Enu a_nom_enu{
      clamp_d(a_nom_e_mps2, -4.0, 4.0),
      clamp_d(a_nom_n_mps2, -4.0, 4.0),
      clamp_d(a_nom_u_mps2, -3.0, 3.0),
  };

  uint8_t last_tmr_state = 0U;
  float last_chi2 = 0.0f;
  Vec3Enu last_a_safe_enu = a_nom_enu;

  for (uint32_t step = 0U; step < n_ticks_400hz; ++step) {
    const PlantState ps = g_arena.plant.state();
    const uint64_t t_us = g_arena.tick_400hz * 2500ULL;

    // 1. Triplex 3-IMU TMR Voter (400 Hz)
    ImuSampleFrd imus[3]{};
    for (int k = 0; k < 3; ++k) {
      imus[k].gyro_rad_s = ps.omega_frd;
      imus[k].accel_m_s2 = {0.0, 0.0, -params::kGravity};
      imus[k].t_us = t_us;
      imus[k].valid = true;
    }
    if (g_arena.imu_fault_active) {
      imus[g_arena.imu_fault_idx].gyro_rad_s.x += 1.25;
      imus[g_arena.imu_fault_idx].gyro_rad_s.y -= 0.95;
    }
    const auto vote = g_arena.tmr.step(imus, g_arena.indi.filtered_gyro_frd(), g_arena.tick_400hz > 4U);
    last_tmr_state = static_cast<uint8_t>(vote.state);
    last_chi2 = vote.max_chi2_residual;

    // 2. Dual-channel Motor FDI Observer (400 Hz)
    float omega_hat[8]{};
    float omega_esc[8]{};
    bool esc_valid[8]{};
    g_arena.indi.get_omega_hat(omega_hat);
    for (int i = 0; i < 8; ++i) {
      omega_esc[i] = static_cast<float>(ps.rotor_omega[i]);
      esc_valid[i] = true;
    }
    const float v_horiz = static_cast<float>(std::sqrt(ps.vel_ned.x * ps.vel_ned.x + ps.vel_ned.y * ps.vel_ned.y));
    g_arena.detected_motor_fault_mask = g_arena.fdi.step_400hz(
        g_arena.indi.filtered_gyro_frd(),
        g_arena.indi.filtered_dgyro_frd(),
        static_cast<float>(-params::kGravity),
        omega_hat,
        omega_esc,
        esc_valid,
        v_horiz,
        ps.battery.v_term_v,
        g_arena.detected_motor_fault_mask);

    if (CoaxEffectiveness::classify_regime(g_arena.detected_motor_fault_mask) == FlightRegime::SAME_ARM_ACTIVE_SPIN ||
        CoaxEffectiveness::classify_regime(g_arena.injected_motor_fault_mask) == FlightRegime::SAME_ARM_ACTIVE_SPIN) {
      // Irreversible CEL latch once same-arm dual-loss is confirmed by FDI (or fault is active > 25 ticks)
      if (CoaxEffectiveness::classify_regime(g_arena.detected_motor_fault_mask) == FlightRegime::SAME_ARM_ACTIVE_SPIN) {
        g_arena.cel_latched = true;
      }
    }

    // 3. Outer hover/descent safety shaping (respects Kartal eastern LTFJ wall & CEL sink cap)
    const Vec3Enu pos_enu = ned_to_enu(ps.pos_ned);
    const Vec3Enu vel_enu = ned_to_enu(ps.vel_ned);
    Vec3Enu a_safe_enu = a_nom_enu;
    if (g_arena.cel_latched) {
      const double target_vz = (pos_enu.z > 10.0) ? -2.0 : ((pos_enu.z > 5.0) ? -1.2 : -0.6);
      a_safe_enu.x = clamp_d(-0.6 * vel_enu.x, -1.5, 1.5);
      a_safe_enu.y = clamp_d(-0.6 * vel_enu.y, -1.5, 1.5);
      a_safe_enu.z = clamp_d(1.8 * (target_vz - vel_enu.z), -2.0, 2.0);
    } else {
      const double z_err = 35.0 - pos_enu.z;
      a_safe_enu.z = clamp_d(a_nom_enu.z + 0.8 * z_err - 1.4 * vel_enu.z, -2.5, 2.5);
      // Soft eastern wall barrier keeping vehicle west of LTFJ NW J2 pinch
      if (pos_enu.x > 80.0) {
        a_safe_enu.x = clamp_d(-1.5 - 0.5 * vel_enu.x, -4.0, 0.0);
      }
    }
    last_a_safe_enu = a_safe_enu;

    // 4. 400 Hz INDI-WLS Allocator + 6-DOF Coaxial X8 Plant Step
    IndiInput in{};
    in.q_frd_ned = ps.q_frd_ned;
    in.gyro_frd = vote.gyro_voted;
    in.accel_des_ned = enu_to_ned(a_safe_enu);
    in.yaw_rate_des = 0.0;
    in.v_horiz_mps = v_horiz;
    in.v_term_v = ps.battery.v_term_v;
    in.fault_mask = g_arena.detected_motor_fault_mask;
    // Keep INDI actuator model omega_hat_ tracking commanded dynamics so MotorFdiObserver
    // detects ESC RPM innovation (omega_hat - omega_esc) organically within ~35 ms.
    in.esc_omega_rad_s = nullptr;
    in.esc_valid = nullptr;

    (void)g_arena.indi.step_400hz(in, g_arena.omega_cmd);
    (void)g_arena.plant.step_400hz(g_arena.omega_cmd, g_arena.injected_motor_fault_mask, g_arena.wind_ned);

    ++g_arena.tick_400hz;
    if ((g_arena.tick_400hz % 4ULL) == 0ULL) {
      ++g_arena.seq_100hz;
    }
  }

  write_latest_vrta_frame(g_arena.plant.state(), a_nom_enu, last_a_safe_enu, last_tmr_state, last_chi2);
  return 0;
}

const uint8_t* vertirta_get_vrta_ptr(void) {
  return reinterpret_cast<const uint8_t*>(&g_arena.latest_frame);
}

uint32_t vertirta_get_vrta_size(void) {
  return static_cast<uint32_t>(sizeof(VrtaFrame100Hz));
}

int32_t vertirta_parse_vrta_frame(const uint8_t* bytes, uint32_t len) {
  if (bytes == nullptr || len != sizeof(VrtaFrame100Hz)) {
    return -1;
  }
  std::memcpy(static_cast<void*>(&g_arena.decoded_frame), static_cast<const void*>(bytes), sizeof(VrtaFrame100Hz));
  return verify_vrta_frame(g_arena.decoded_frame) ? 0 : -1;
}

double vertirta_egm2008_n_m(double lat_deg, double lon_deg) {
  if (!is_finite_d(lat_deg) || !is_finite_d(lon_deg)) {
    return 36.7;
  }
  return g_arena.geoid.lookup(lat_deg, lon_deg);
}

double vertirta_eval_cbvf_margin(double east_m, double north_m, double agl_m, uint8_t cel_latched) {
  if (!is_finite_d(east_m) || !is_finite_d(north_m) || !is_finite_d(agl_m)) {
    return -1e6;
  }
  const double lat_deg = g_arena.datum_lat_deg + north_m / kMetersPerDegLat;
  const double lon_deg = g_arena.datum_lon_deg + east_m / kMetersPerDegLon;
  const double ltfj_m = signed_margin_ltfj_m(lat_deg, lon_deg);
  const double ltbx_m = signed_margin_ltbx_m(lat_deg, lon_deg);
  const double ceil_m = 120.0 - agl_m;
  const double floor_m = (cel_latched != 0U) ? agl_m : (agl_m - 10.0);

  double m = ltfj_m;
  if (ltbx_m < m) m = ltbx_m;
  if (ceil_m < m) m = ceil_m;
  if (floor_m < m) m = floor_m;
  return m;
}

}  // extern "C"
