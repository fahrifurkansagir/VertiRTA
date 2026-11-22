#pragma once

#include <cstdint>

#include <drivers/drv_hrt.h>
#include <lib/perf/perf_counter.h>
#include <px4_platform_common/defines.h>
#include <px4_platform_common/module.h>
#include <px4_platform_common/module_params.h>
#include <px4_platform_common/px4_work_queue/ScheduledWorkItem.hpp>
#include <uORB/Publication.hpp>
#include <uORB/Subscription.hpp>
#include <uORB/SubscriptionCallback.hpp>
#include <uORB/topics/actuator_motors.h>
#include <uORB/topics/battery_status.h>
#include <uORB/topics/esc_status.h>
#include <uORB/topics/manual_control_setpoint.h>
#include <uORB/topics/sensor_accel.h>
#include <uORB/topics/sensor_gyro.h>
#include <uORB/topics/vehicle_attitude.h>
#include <uORB/topics/vehicle_control_mode.h>
#include <uORB/topics/vehicle_local_position.h>
#include <uORB/topics/vehicle_status.h>
#include <uORB/topics/vertirta_capability_envelope.h>
#include <uORB/topics/vertirta_rta_setpoint.h>

#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/frames.hpp"
#include "vertirta/core/indi_wls.hpp"
#include "vertirta/core/math_utils.hpp"
#include "vertirta/core/motor_fdi.hpp"
#include "vertirta/core/tmr_voter.hpp"

namespace vertirta::px4_module {

// 400 Hz PX4 v1.16 internal flight-critical control module on wq:rate_ctrl.
// Freestanding C++17 (zero heap, zero STL algorithm header, zero exceptions/RTTI):
//   3-IMU TMR Voter -> S^2 Reduced-Attitude + 8 Hz Butterworth INDI -> f64/QR WLS
//   -> Motor FDI (<85 ms) -> 1-RC Thevenin Battery ECM -> 100 Hz 72 B Capability Envelope.
class McIndiWlsControl : public ModuleBase<McIndiWlsControl>,
                         public ModuleParams,
                         public px4::ScheduledWorkItem {
 public:
  McIndiWlsControl();
  ~McIndiWlsControl() override;

  static int task_spawn(int argc, char* argv[]);
  static int custom_command(int argc, char* argv[]);
  static int print_usage(const char* reason = nullptr);

  bool init();
  int print_status() override;

 private:
  static constexpr uint32_t kRateHz = 400u;
  static constexpr uint32_t kPeriodUs = 2500u;
  static constexpr uint32_t kEnvelopeDecimation = 4u;  // 400 Hz / 4 = 100 Hz
  static constexpr hrt_abstime kImuStaleUs = 10000ULL;
  static constexpr hrt_abstime kRtaSetpointStaleUs = 250000ULL;  // 250 ms staleness lock

  void Run() override;
  float filter_accel_z(float az_frd) noexcept;
  void publish_capability_envelope(
      hrt_abstime now_us,
      uint64_t sample_ts_us,
      const vertirta::core::TmrVoteOutput& tmr,
      const vertirta::core::BatteryState& batt) noexcept;

  // 3x IMU subscriptions (IMU 0..2 from patched GZBridge / Pixhawk 6X)
  uORB::SubscriptionCallbackWorkItem _sensor_gyro_cb{this, ORB_ID(sensor_gyro), 0};
  uORB::Subscription _sensor_gyro_subs[3]{
      uORB::Subscription{ORB_ID(sensor_gyro), 0},
      uORB::Subscription{ORB_ID(sensor_gyro), 1},
      uORB::Subscription{ORB_ID(sensor_gyro), 2},
  };
  uORB::Subscription _sensor_accel_subs[3]{
      uORB::Subscription{ORB_ID(sensor_accel), 0},
      uORB::Subscription{ORB_ID(sensor_accel), 1},
      uORB::Subscription{ORB_ID(sensor_accel), 2},
  };

  uORB::Subscription _vehicle_attitude_sub{ORB_ID(vehicle_attitude)};
  uORB::Subscription _vehicle_local_position_sub{ORB_ID(vehicle_local_position)};
  uORB::Subscription _vehicle_control_mode_sub{ORB_ID(vehicle_control_mode)};
  uORB::Subscription _vehicle_status_sub{ORB_ID(vehicle_status)};
  uORB::Subscription _esc_status_sub{ORB_ID(esc_status)};
  uORB::Subscription _battery_status_sub{ORB_ID(battery_status)};
  uORB::Subscription _rta_setpoint_sub{ORB_ID(vertirta_rta_setpoint)};
  uORB::Subscription _manual_control_sub{ORB_ID(manual_control_setpoint)};

  uORB::Publication<actuator_motors_s> _actuator_motors_pub{ORB_ID(actuator_motors)};
  uORB::Publication<vertirta_capability_envelope_s> _envelope_pub{
      ORB_ID(vertirta_capability_envelope)};

  // Zero-heap libvertirta_flight_core stateful flight kernels
  vertirta::core::TmrVoter _tmr_voter{};
  vertirta::core::IndiWlsController _indi_wls{true};
  vertirta::core::MotorFdiObserver _motor_fdi{};
  vertirta::core::TheveninBattery1RC _battery_ecm{0.80f};

  // Cached estimator & telemetry snapshots
  vehicle_attitude_s _att{};
  vehicle_local_position_s _lpos{};
  vehicle_control_mode_s _vcontrol_mode{};
  vehicle_status_s _vstatus{};
  esc_status_s _esc_status{};
  battery_status_s _batt_status{};
  vertirta_rta_setpoint_s _rta_sp{};
  manual_control_setpoint_s _manual_sp{};

  bool _armed_prev{false};
  uint8_t _fault_mask{0x00};
  uint32_t _tick_counter{0};
  vertirta::core::Vec3Frd _model_gyro_pred{};
  bool _model_gyro_valid{false};

  // Synchronized 8 Hz Butterworth biquad state for FRD body-Z specific force (Motor FDI)
  bool _az_lpf_init{false};
  double _az_x1{0.0};
  double _az_x2{0.0};
  double _az_y1{0.0};
  double _az_y2{0.0};

  perf_counter_t _loop_perf{nullptr};
  perf_counter_t _loop_interval_perf{nullptr};
};

}  // namespace vertirta::px4_module
