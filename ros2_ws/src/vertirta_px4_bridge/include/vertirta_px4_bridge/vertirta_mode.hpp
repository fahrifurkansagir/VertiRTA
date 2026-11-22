#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <Eigen/Geometry>
#include <px4_msgs/msg/vehicle_thrust_setpoint.hpp>
#include <px4_msgs/msg/vertirta_capability_envelope.hpp>
#include <px4_msgs/msg/vertirta_rta_setpoint.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/control/setpoint_types/experimental/attitude.hpp>
#include <rclcpp/rclcpp.hpp>

namespace vertirta::px4_bridge {

// 100 Hz px4-ros2-interface-lib ModeBase adapter (DR-2 Section 5.1 / Plan 1A Task 12):
// - CSFL (Regimes 0..2): emits AttitudeSetpointType (bypasses stock mc_pos_control static MPC_TILTMAX_AIR)
// - CEL (Regime 3 SAME_ARM_ACTIVE_SPIN): irreversibly latches cel_latched_ and emits
//   VehicleThrustSetpoint + /fmu/in/vertirta_rta_setpoint for Option B1b active flat spin.
class VertiRtaMode : public px4_ros2::ModeBase {
 public:
  explicit VertiRtaMode(rclcpp::Node& node, const std::string& vehicle_namespace = "");

  void onActivate() override;
  void onDeactivate() override;
  void updateSetpoint(float dt_s) override;

  [[nodiscard]] bool is_cel_latched() const noexcept {
    return cel_latched_.load(std::memory_order_acquire);
  }

  [[nodiscard]] uint8_t flight_regime() const noexcept {
    return latest_regime_.load(std::memory_order_acquire);
  }

  // Computes gravity-compensated tilt-prioritized NED quaternion + normalized FRD thrust_body
  // from safe 100 Hz CBVF-QP acceleration a_safe_ned (Review Focus #5 quaternion sign convention).
  static void accel_ned_to_attitude_thrust(
      const Eigen::Vector3f& a_safe_ned,
      float yaw_sp_rad,
      float t_max_n,
      Eigen::Quaternionf& q_frd_ned_out,
      Eigen::Vector3f& thrust_body_frd_out) noexcept;

 private:
  static constexpr float kMassKg = 240.0f;
  static constexpr float kGravityMps2 = 9.80665f;
  static constexpr float kNominalTmaxN = 5886.0f;

  rclcpp::Node& node_;
  std::string vehicle_namespace_;
  std::shared_ptr<px4_ros2::AttitudeSetpointType> attitude_setpoint_;
  rclcpp::Publisher<px4_msgs::msg::VehicleThrustSetpoint>::SharedPtr thrust_pub_;
  rclcpp::Publisher<px4_msgs::msg::VertirtaRtaSetpoint>::SharedPtr rta_sp_pub_;
  rclcpp::Subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>::SharedPtr env_sub_;
  rclcpp::Subscription<px4_msgs::msg::VertirtaRtaSetpoint>::SharedPtr rta_cmd_sub_;

  std::atomic<bool> cel_latched_{false};
  std::atomic<uint8_t> latest_regime_{0U};
  std::atomic<float> latest_t_max_n_{kNominalTmaxN};
  std::atomic<float> latest_ax_ned_{0.0f};
  std::atomic<float> latest_ay_ned_{0.0f};
  std::atomic<float> latest_az_ned_{0.0f};
  std::atomic<float> latest_yaw_sp_{0.0f};
  std::atomic<float> latest_yawspeed_sp_{0.0f};
  std::atomic<float> latest_vz_limit_{2.5f};
};

}  // namespace vertirta::px4_bridge
