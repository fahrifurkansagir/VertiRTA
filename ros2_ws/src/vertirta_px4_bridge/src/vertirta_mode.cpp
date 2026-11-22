#include "vertirta_px4_bridge/vertirta_mode.hpp"

#include <algorithm>
#include <cmath>

namespace vertirta::px4_bridge {

namespace {

std::string scoped_topic(const std::string& ns, const std::string& topic) {
  if (ns.empty() || ns == "/") {
    return topic;
  }
  if (ns.front() == '/') {
    return ns + topic;
  }
  return "/" + ns + topic;
}

}  // namespace

VertiRtaMode::VertiRtaMode(rclcpp::Node& node, const std::string& vehicle_namespace)
    : px4_ros2::ModeBase(node, px4_ros2::ModeBase::Settings{"VertiRTA_RTA"}.preventArming(false)),
      node_(node),
      vehicle_namespace_(vehicle_namespace) {
  // Lock exact 100 Hz outer-loop setpoint rate (DR-2 Section 5.1)
  setSetpointUpdateRate(100.0f);

  // AttitudeSetpointType bypasses stock PX4 mc_pos_control static MPC_TILTMAX_AIR
  attitude_setpoint_ = std::make_shared<px4_ros2::AttitudeSetpointType>(*this);

  auto qos = rclcpp::SensorDataQoS();
  thrust_pub_ = node_.create_publisher<px4_msgs::msg::VehicleThrustSetpoint>(
      scoped_topic(vehicle_namespace_, "/fmu/in/vehicle_thrust_setpoint"), qos);
  rta_sp_pub_ = node_.create_publisher<px4_msgs::msg::VertirtaRtaSetpoint>(
      scoped_topic(vehicle_namespace_, "/fmu/in/vertirta_rta_setpoint"), qos);

  env_sub_ = node_.create_subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>(
      scoped_topic(vehicle_namespace_, "/fmu/out/vertirta_capability_envelope"),
      qos,
      [this](const px4_msgs::msg::VertirtaCapabilityEnvelope::SharedPtr msg) {
        latest_regime_.store(msg->flight_regime, std::memory_order_release);
        if (std::isfinite(msg->t_max) && msg->t_max > 100.0f) {
          latest_t_max_n_.store(msg->t_max, std::memory_order_release);
        }
        // Irreversible CEL latch: transient flicker back to NOMINAL must NEVER unlatch (Review Focus #3)
        if (msg->flight_regime == px4_msgs::msg::VertirtaCapabilityEnvelope::REGIME_SAME_ARM_ACTIVE_SPIN ||
            msg->flight_regime == px4_msgs::msg::VertirtaCapabilityEnvelope::REGIME_CRITICAL_DESCENT) {
          cel_latched_.store(true, std::memory_order_release);
        }
      });

  rta_cmd_sub_ = node_.create_subscription<px4_msgs::msg::VertirtaRtaSetpoint>(
      scoped_topic(vehicle_namespace_, "/vertirta/internal/rta_cmd"),
      qos,
      [this](const px4_msgs::msg::VertirtaRtaSetpoint::SharedPtr msg) {
        if (msg->valid == 0U) {
          return;
        }
        latest_ax_ned_.store(msg->accel_ned[0], std::memory_order_release);
        latest_ay_ned_.store(msg->accel_ned[1], std::memory_order_release);
        latest_az_ned_.store(msg->accel_ned[2], std::memory_order_release);
        latest_yaw_sp_.store(msg->yaw_sp, std::memory_order_release);
        latest_yawspeed_sp_.store(msg->yawspeed_sp, std::memory_order_release);
        latest_vz_limit_.store(msg->vz_limit_mps, std::memory_order_release);
        if (msg->cel_latched != 0U) {
          cel_latched_.store(true, std::memory_order_release);
        }
      });
}

void VertiRtaMode::onActivate() {
  RCLCPP_INFO(node_.get_logger(), "VertiRtaMode activated @ 100 Hz (ns='%s')", vehicle_namespace_.c_str());
}

void VertiRtaMode::onDeactivate() {
  RCLCPP_INFO(node_.get_logger(), "VertiRtaMode deactivated (cel_latched=%d)", static_cast<int>(is_cel_latched()));
}

void VertiRtaMode::accel_ned_to_attitude_thrust(
    const Eigen::Vector3f& a_safe_ned,
    float yaw_sp_rad,
    float t_max_n,
    Eigen::Quaternionf& q_frd_ned_out,
    Eigen::Vector3f& thrust_body_frd_out) noexcept {
  // Specific inertial force f_NED = m * (a_safe - [0, 0, g]^T); thrust points along -Z_FRD
  const Eigen::Vector3f f_ned(
      kMassKg * a_safe_ned.x(),
      kMassKg * a_safe_ned.y(),
      kMassKg * (a_safe_ned.z() - kGravityMps2));
  const float f_norm = std::max(f_ned.norm(), 1.0f);
  const Eigen::Vector3f b_z = (-f_ned / f_norm).normalized();

  // Tilt-prioritized frame construction preserving yaw_sp_rad in NED
  const Eigen::Vector3f x_c(std::cos(yaw_sp_rad), std::sin(yaw_sp_rad), 0.0f);
  Eigen::Vector3f b_y = b_z.cross(x_c);
  if (b_y.norm() < 1e-4f) {
    b_y = Eigen::Vector3f(-std::sin(yaw_sp_rad), std::cos(yaw_sp_rad), 0.0f);
  } else {
    b_y.normalize();
  }
  const Eigen::Vector3f b_x = b_y.cross(b_z).normalized();

  Eigen::Matrix3f R_frd_to_ned;
  R_frd_to_ned.col(0) = b_x;
  R_frd_to_ned.col(1) = b_y;
  R_frd_to_ned.col(2) = b_z;
  q_frd_ned_out = Eigen::Quaternionf(R_frd_to_ned).normalized();

  const float t_ref = std::max(t_max_n, kMassKg * kGravityMps2);
  const float thr_norm = std::clamp(f_norm / t_ref, 0.05f, 1.0f);
  thrust_body_frd_out = Eigen::Vector3f(0.0f, 0.0f, -thr_norm);
}

void VertiRtaMode::updateSetpoint(float /*dt_s*/) {
  const Eigen::Vector3f a_safe(
      latest_ax_ned_.load(std::memory_order_acquire),
      latest_ay_ned_.load(std::memory_order_acquire),
      latest_az_ned_.load(std::memory_order_acquire));
  const float yaw_sp = latest_yaw_sp_.load(std::memory_order_acquire);
  const float t_max = latest_t_max_n_.load(std::memory_order_acquire);

  if (is_cel_latched()) {
    // CEL Regime 3 (SAME_ARM_DUAL_LOSS): bypass full SO(3) attitude setpoint and stream
    // collective vertical thrust + Option B1b flat-spin command to PX4 mc_indi_wls_control.
    const float f_vert = std::max(0.0f, kMassKg * (kGravityMps2 - a_safe.z()));
    const float thr_norm = std::clamp(f_vert / std::max(t_max, 1.0f), 0.05f, 1.0f);

    px4_msgs::msg::VehicleThrustSetpoint thr{};
    thr.timestamp = node_.now().nanoseconds() / 1000ULL;
    thr.timestamp_sample = thr.timestamp;
    thr.xyz = {0.0f, 0.0f, -thr_norm};
    thrust_pub_->publish(thr);

    px4_msgs::msg::VertirtaRtaSetpoint sp{};
    sp.timestamp = thr.timestamp;
    sp.timestamp_sample = thr.timestamp;
    sp.accel_ned = {a_safe.x(), a_safe.y(), a_safe.z()};
    sp.thrust_des_n = f_vert;
    sp.yaw_sp = yaw_sp;
    sp.yawspeed_sp = latest_yawspeed_sp_.load(std::memory_order_acquire);
    sp.vz_limit_mps = latest_vz_limit_.load(std::memory_order_acquire);
    sp.control_mode = px4_msgs::msg::VertirtaRtaSetpoint::MODE_CEL_CONTROLLED_LANDING;
    sp.cel_latched = 1U;
    sp.valid = 1U;
    sp.reserved_pad = 0U;
    rta_sp_pub_->publish(sp);
    return;
  }

  Eigen::Quaternionf q_frd_ned;
  Eigen::Vector3f thrust_body_frd;
  accel_ned_to_attitude_thrust(a_safe, yaw_sp, t_max, q_frd_ned, thrust_body_frd);
  attitude_setpoint_->update(q_frd_ned, thrust_body_frd, latest_yawspeed_sp_.load(std::memory_order_acquire));

  // Sole publisher of /fmu/in/vertirta_rta_setpoint for PX4 internal mc_indi_wls_control (VERTIRTA_USE_INDI=1)
  px4_msgs::msg::VertirtaRtaSetpoint sp{};
  sp.timestamp = node_.now().nanoseconds() / 1000ULL;
  sp.timestamp_sample = sp.timestamp;
  sp.accel_ned = {a_safe.x(), a_safe.y(), a_safe.z()};
  sp.thrust_des_n = kMassKg * std::hypot(std::hypot(a_safe.x(), a_safe.y()), a_safe.z() - kGravityMps2);
  sp.yaw_sp = yaw_sp;
  sp.yawspeed_sp = latest_yawspeed_sp_.load(std::memory_order_acquire);
  sp.vz_limit_mps = latest_vz_limit_.load(std::memory_order_acquire);
  sp.control_mode = px4_msgs::msg::VertirtaRtaSetpoint::MODE_ACCEL_NED;
  sp.cel_latched = 0U;
  sp.valid = 1U;
  sp.reserved_pad = 0U;
  rta_sp_pub_->publish(sp);
}

}  // namespace vertirta::px4_bridge
