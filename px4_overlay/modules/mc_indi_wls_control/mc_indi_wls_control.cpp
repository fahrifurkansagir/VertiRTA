#include "mc_indi_wls_control.hpp"

#include <cmath>
#include <px4_platform_common/log.h>

namespace vertirta::px4_module {

namespace {
// Exact 400 Hz prewarped 8.0 Hz Butterworth (zeta = 0.7071) biquad matching IndiWlsController
constexpr double kB0 = 0.00362168;
constexpr double kB1 = 0.00724336;
constexpr double kB2 = 0.00362168;
constexpr double kA1 = -1.82269493;
constexpr double kA2 = 0.83718165;
}  // namespace

McIndiWlsControl::McIndiWlsControl()
    : ModuleParams(nullptr),
      ScheduledWorkItem(MODULE_NAME, px4::wq_configurations::rate_ctrl) {
  _loop_perf = perf_alloc(PC_ELAPSED, MODULE_NAME ": cycle");
  _loop_interval_perf = perf_alloc(PC_INTERVAL, MODULE_NAME ": interval");
}

McIndiWlsControl::~McIndiWlsControl() {
  perf_free(_loop_perf);
  perf_free(_loop_interval_perf);
}

bool McIndiWlsControl::init() {
  _tmr_voter.reset();
  _indi_wls.reset(vertirta::core::params::kOmegaHover);
  _motor_fdi.reset();
  _battery_ecm.reset(0.80f);

  // Trigger synchronously on IMU0 400 Hz callback with a 2.5 ms fallback interval guard
  if (!_sensor_gyro_cb.registerCallback()) {
    PX4_WARN("mc_indi_wls_control: sensor_gyro[0] callback registration failed, using 400 Hz timer");
  }
  ScheduleOnInterval(kPeriodUs, kPeriodUs);
  return true;
}

float McIndiWlsControl::filter_accel_z(float az_frd) noexcept {
  const double x = vertirta::core::is_finite_f(az_frd) ? static_cast<double>(az_frd) : -vertirta::core::params::kGravity;
  if (!_az_lpf_init) {
    _az_x1 = _az_x2 = _az_y1 = _az_y2 = x;
    _az_lpf_init = true;
    return static_cast<float>(x);
  }
  const double y = kB0 * x + kB1 * _az_x1 + kB2 * _az_x2 - kA1 * _az_y1 - kA2 * _az_y2;
  _az_x2 = _az_x1;
  _az_x1 = x;
  _az_y2 = _az_y1;
  _az_y1 = y;
  return static_cast<float>(y);
}

void McIndiWlsControl::Run() {
  if (should_exit()) {
    _sensor_gyro_cb.unregisterCallback();
    ScheduleClear();
    exit_and_cleanup();
    return;
  }

  perf_begin(_loop_perf);
  perf_count(_loop_interval_perf);

  const hrt_abstime now_us = hrt_absolute_time();

  // 1. Ingest all 3 heterogeneous 400 Hz IMUs into freestanding TmrVoter
  vertirta::core::ImuSampleFrd imus[3]{};
  uint64_t latest_sample_us = now_us;
  for (int i = 0; i < 3; ++i) {
    sensor_gyro_s g{};
    sensor_accel_s a{};
    const bool has_g = _sensor_gyro_subs[i].copy(&g);
    const bool has_a = _sensor_accel_subs[i].copy(&a);
    const bool fresh_g = has_g && (now_us >= g.timestamp) && ((now_us - g.timestamp) <= kImuStaleUs);
    const bool fresh_a = has_a && (now_us >= a.timestamp) && ((now_us - a.timestamp) <= kImuStaleUs);

    imus[i].valid = fresh_g && fresh_a;
    imus[i].t_us = fresh_g ? g.timestamp_sample : 0ULL;
    if (imus[i].valid) {
      imus[i].gyro_rad_s = {static_cast<double>(g.x), static_cast<double>(g.y), static_cast<double>(g.z)};
      imus[i].accel_m_s2 = {static_cast<double>(a.x), static_cast<double>(a.y), static_cast<double>(a.z)};
      if (g.timestamp_sample > latest_sample_us || latest_sample_us == now_us) {
        latest_sample_us = g.timestamp_sample;
      }
    }
  }

  const vertirta::core::TmrVoteOutput tmr = _tmr_voter.step(imus, _model_gyro_pred, _model_gyro_valid);
  const float az_filtered = filter_accel_z(static_cast<float>(tmr.accel_voted.z));

  // 2. Refresh estimator, control-mode, ESC, battery, and companion RTA setpoint snapshots
  _vehicle_attitude_sub.update(&_att);
  _vehicle_local_position_sub.update(&_lpos);
  _vehicle_control_mode_sub.update(&_vcontrol_mode);
  _vehicle_status_sub.update(&_vstatus);
  _esc_status_sub.update(&_esc_status);
  _battery_status_sub.update(&_batt_status);
  _rta_setpoint_sub.update(&_rta_sp);
  _manual_control_sub.update(&_manual_sp);

  const bool armed = (_vstatus.arming_state == vehicle_status_s::ARMING_STATE_ARMED);
  if (armed && !_armed_prev) {
    _indi_wls.reset(vertirta::core::params::kOmegaIdle);
    _motor_fdi.reset();
    _fault_mask = 0x00;
    _az_lpf_init = false;
  }
  _armed_prev = armed;

  // Build ESC feedback array (rad/s) for INDI complementary observer & Channel-1 Motor FDI
  float omega_esc[8]{};
  bool esc_valid[8]{};
  const bool esc_fresh = (_esc_status.timestamp > 0ULL) &&
                         (now_us >= _esc_status.timestamp) &&
                         ((now_us - _esc_status.timestamp) <= 50000ULL);
  constexpr float kRpmToRadS = static_cast<float>(vertirta::core::kTwoPi / 60.0);
  for (int i = 0; i < 8; ++i) {
    if (esc_fresh && i < static_cast<int>(_esc_status.esc_count) &&
        (_esc_status.esc_online_flags & (1u << i)) != 0u) {
      const float rpm = static_cast<float>(_esc_status.esc[i].esc_rpm);
      omega_esc[i] = (rpm > 0.0f) ? (rpm * kRpmToRadS) : 0.0f;
      esc_valid[i] = true;
    } else {
      omega_esc[i] = 0.0f;
      esc_valid[i] = false;
    }
  }

  // 3. Determine active setpoint: 100 Hz companion RTA setpoint vs safety-pilot manual stabilized
  vertirta::core::Vec3Ned accel_des_ned{0.0, 0.0, 0.0};
  double yaw_rate_des = 0.0;
  const bool rta_fresh = (_rta_sp.valid != 0u) &&
                         (_rta_sp.timestamp > 0ULL) &&
                         (now_us >= _rta_sp.timestamp) &&
                         ((now_us - _rta_sp.timestamp) <= kRtaSetpointStaleUs);

  if (rta_fresh) {
    accel_des_ned = {
        static_cast<double>(_rta_sp.accel_ned[0]),
        static_cast<double>(_rta_sp.accel_ned[1]),
        static_cast<double>(_rta_sp.accel_ned[2]),
    };
    yaw_rate_des = static_cast<double>(_rta_sp.yawspeed_sp);
    // Optional SITL/HIL fault override bitmask (monotonic OR)
    _fault_mask |= _rta_sp.fault_override_mask;
  } else if (_vcontrol_mode.flag_control_manual_enabled && _manual_sp.valid) {
    // Safety-pilot stabilized override routed through the identical INDI+WLS loop (zero transient)
    const float roll_cmd = vertirta::core::clamp_f(_manual_sp.roll, -1.0f, 1.0f);
    const float pitch_cmd = vertirta::core::clamp_f(_manual_sp.pitch, -1.0f, 1.0f);
    const float thr_cmd = vertirta::core::clamp_f(_manual_sp.throttle, -1.0f, 1.0f);
    const float yaw_cmd = vertirta::core::clamp_f(_manual_sp.yaw, -1.0f, 1.0f);
    accel_des_ned = {
        static_cast<double>(-pitch_cmd * 4.5f),
        static_cast<double>(roll_cmd * 4.5f),
        static_cast<double>(-thr_cmd * 3.0f),
    };
    yaw_rate_des = static_cast<double>(yaw_cmd * 1.2f);
  }

  // 4. Execute 400 Hz S^2 Reduced-Attitude + Synchronized INDI + f64/QR WLS + Option B1b Spin
  const float v_horiz = static_cast<float>(std::hypot(
      static_cast<double>(_lpos.vx), static_cast<double>(_lpos.vy)));
  const float v_term_v = (_batt_status.timestamp > 0ULL &&
                          vertirta::core::is_finite_f(_batt_status.voltage_v) &&
                          _batt_status.voltage_v > 200.0f)
                             ? _batt_status.voltage_v
                             : _battery_ecm.state().v_term_v;

  vertirta::core::IndiInput indi_in{};
  indi_in.q_frd_ned = {
      static_cast<double>(_att.q[0]),
      static_cast<double>(_att.q[1]),
      static_cast<double>(_att.q[2]),
      static_cast<double>(_att.q[3]),
  };
  indi_in.gyro_frd = tmr.gyro_voted;
  indi_in.accel_des_ned = accel_des_ned;
  indi_in.yaw_rate_des = yaw_rate_des;
  indi_in.v_horiz_mps = v_horiz;
  indi_in.v_term_v = v_term_v;
  indi_in.fault_mask = _fault_mask;
  indi_in.esc_omega_rad_s = omega_esc;
  indi_in.esc_valid = esc_valid;

  float omega_cmd[8]{};
  if (armed) {
    (void)_indi_wls.step_400hz(indi_in, omega_cmd);
  }

  // 5. Publish normalized motor speed commands [0, 1] -> SIM_GZ_EC_MAX = 240 rad/s
  actuator_motors_s motors_msg{};
  motors_msg.timestamp = hrt_absolute_time();
  motors_msg.timestamp_sample = latest_sample_us;
  motors_msg.reversible_flags = 0;
  constexpr float kInvOmegaMax = static_cast<float>(1.0 / vertirta::core::params::kOmegaMax);
  for (int i = 0; i < 8; ++i) {
    const bool dead = ((_fault_mask & static_cast<uint8_t>(1u << i)) != 0u);
    if (!armed || dead) {
      motors_msg.control[i] = 0.0f;
    } else {
      motors_msg.control[i] = vertirta::core::clamp_f(omega_cmd[i] * kInvOmegaMax, 0.0f, 1.0f);
    }
  }
  for (int i = 8; i < actuator_motors_s::NUM_CONTROLS; ++i) {
    motors_msg.control[i] = NAN;
  }
  _actuator_motors_pub.publish(motors_msg);

  // 6. Dual-channel 400 Hz Motor FDI Observer (<60 ms ESC / <85 ms sensorless wrench)
  float omega_hat[8]{};
  _indi_wls.get_omega_hat(omega_hat);
  if (armed) {
    const uint8_t new_mask = _motor_fdi.step_400hz(
        _indi_wls.filtered_gyro_frd(),
        _indi_wls.filtered_dgyro_frd(),
        az_filtered,
        omega_hat,
        omega_esc,
        esc_valid,
        v_horiz,
        v_term_v,
        _fault_mask);
    _fault_mask |= new_mask;
  }

  // Update 1-step analytical gyro prediction for TmrVoter DUPLEX->SIMPLEX tie-breaker
  _model_gyro_pred = _indi_wls.filtered_gyro_frd() +
                     (_indi_wls.filtered_dgyro_frd() * 0.0025);
  _model_gyro_valid = armed;

  // 7. 100 Hz decimated 1-RC Thevenin battery step & 72 B Capability Envelope publish
  ++_tick_counter;
  if ((_tick_counter % kEnvelopeDecimation) == 0u) {
    double k_t[8]{};
    double k_q[8]{};
    vertirta::core::CoaxEffectiveness::effective_rotor_constants(_fault_mask, v_horiz, k_t, k_q);
    float thrusts[8]{};
    for (int i = 0; i < 8; ++i) {
      const float w = armed ? omega_hat[i] : 0.0f;
      thrusts[i] = static_cast<float>(k_t[i]) * w * w;
    }
    const float p_elec_w = vertirta::core::CoaxEffectiveness::compute_electrical_power_w(
        thrusts, _fault_mask, v_horiz);
    const vertirta::core::BatteryState batt = _battery_ecm.step(0.01f, p_elec_w);
    publish_capability_envelope(now_us, latest_sample_us, tmr, batt);
  }

  perf_end(_loop_perf);
}

void McIndiWlsControl::publish_capability_envelope(
    hrt_abstime now_us,
    uint64_t sample_ts_us,
    const vertirta::core::TmrVoteOutput& tmr,
    const vertirta::core::BatteryState& batt) noexcept {
  const uint16_t latency_ms =
      (_motor_fdi.last_isolation_latency_ms() > tmr.isolation_latency_ms)
          ? _motor_fdi.last_isolation_latency_ms()
          : tmr.isolation_latency_ms;

  const vertirta::core::CapabilityEnvelope env = vertirta::core::CoaxEffectiveness::compute_envelope(
      _fault_mask,
      static_cast<uint8_t>(tmr.state),
      tmr.health_mask,
      batt.v_term_v,
      batt.i_pack_a,
      batt.soc,
      _indi_wls.spin_rate_ref_rad_s(),
      static_cast<float>(tmr.gyro_voted.z),
      tmr.max_chi2_residual,
      latency_ms);

  vertirta_capability_envelope_s msg{};
  msg.timestamp = now_us;
  msg.timestamp_sample = sample_ts_us;
  msg.fault_bitmask = env.fault_mask;
  msg.tmr_state = env.tmr_state;
  msg.imu_health_mask = env.imu_mask;
  msg.flight_regime = static_cast<uint8_t>(env.regime);
  msg.t_max = env.t_max_n;
  msg.a_xy_max = env.a_xy_max;
  msg.tilt_max = env.tilt_max_rad;
  msg.vz_sink_max = env.vz_sink_max;
  msg.v_term = env.v_term_v;
  msg.i_pack = env.i_pack_a;
  msg.soc = env.soc;
  msg.p_hover_deg = env.p_hover_deg_w;
  msg.spin_rate_ref = env.spin_rate_ref;
  msg.spin_rate_meas = env.spin_rate_meas;
  msg.chi2_residual = env.chi2_residual;
  msg.fdi_latency_ms = env.fdi_latency_ms;
  msg.reserved_align[0] = 0u;
  msg.reserved_align[1] = 0u;
  msg.reserved_pad = 0u;
  _envelope_pub.publish(msg);
}

int McIndiWlsControl::task_spawn(int /*argc*/, char* /*argv*/[]) {
  auto* instance = new McIndiWlsControl();
  if (instance == nullptr) {
    PX4_ERR("alloc failed");
    return PX4_ERROR;
  }
  _object.store(instance);
  _task_id = task_id_is_work_queue;
  if (instance->init()) {
    return PX4_OK;
  }
  delete instance;
  _object.store(nullptr);
  _task_id = -1;
  return PX4_ERROR;
}

int McIndiWlsControl::custom_command(int /*argc*/, char* /*argv*/[]) {
  return print_usage("unknown command");
}

int McIndiWlsControl::print_status() {
  PX4_INFO(
      "VertiRTA 400 Hz INDI+WLS/TMR: fault_mask=0x%02x tmr_state=%u imu_mask=0x%02x v_term=%.1f V",
      static_cast<unsigned>(_fault_mask),
      static_cast<unsigned>(_tmr_voter.state()),
      static_cast<unsigned>(_tmr_voter.health_mask()),
      static_cast<double>(_battery_ecm.state().v_term_v));
  perf_print_counter(_loop_perf);
  perf_print_counter(_loop_interval_perf);
  return 0;
}

int McIndiWlsControl::print_usage(const char* reason) {
  if (reason != nullptr) {
    PX4_WARN("%s\n", reason);
  }
  PRINT_MODULE_DESCRIPTION(
      R"DESCR_STR(
### Description
VertiRTA-X8-240 400 Hz PX4-internal TMR FDIR + Reduced-Attitude INDI + f64/QR WLS + Motor FDI module.
)DESCR_STR");
  PRINT_MODULE_USAGE_NAME("mc_indi_wls_control", "controller");
  PRINT_MODULE_USAGE_COMMAND("start");
  PRINT_MODULE_USAGE_DEFAULT_COMMANDS();
  return 0;
}

}  // namespace vertirta::px4_module

extern "C" __EXPORT int mc_indi_wls_control_main(int argc, char* argv[]) {
  return vertirta::px4_module::McIndiWlsControl::main(argc, argv);
}
