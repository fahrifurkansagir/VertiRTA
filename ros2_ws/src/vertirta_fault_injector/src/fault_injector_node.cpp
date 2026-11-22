#include <cstdint>
#include <memory>
#include <string>

#include <px4_msgs/msg/vehicle_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>
#include <vertirta_msgs/srv/inject_fault.hpp>

namespace vertirta::fault_injector {

class VertiRtaFaultInjectorNode : public rclcpp::Node {
 public:
  VertiRtaFaultInjectorNode() : Node("vertirta_fault_injector") {
    auto qos = rclcpp::SensorDataQoS();
    cmd_pub_ = create_publisher<px4_msgs::msg::VehicleCommand>("/fmu/in/vehicle_command", qos);
    gz_mask_pub_ = create_publisher<std_msgs::msg::UInt32>("/vertirta/fault/motor_mask", 10);

    srv_ = create_service<vertirta_msgs::srv::InjectFault>(
        "/vertirta/inject_fault",
        [this](
            const std::shared_ptr<vertirta_msgs::srv::InjectFault::Request> req,
            std::shared_ptr<vertirta_msgs::srv::InjectFault::Response> res) {
          handle_inject(req, res);
        });
  }

 private:
  void handle_inject(
      const std::shared_ptr<vertirta_msgs::srv::InjectFault::Request>& req,
      const std::shared_ptr<vertirta_msgs::srv::InjectFault::Response>& res) {
    using Srv = vertirta_msgs::srv::InjectFault::Request;
    const uint64_t now_ns = static_cast<uint64_t>(now().nanoseconds());

    px4_msgs::msg::VehicleCommand cmd{};
    cmd.timestamp = now_ns / 1000ULL;
    cmd.command = px4_msgs::msg::VehicleCommand::VEHICLE_CMD_INJECT_FAILURE;
    cmd.param3 = static_cast<float>(req->instance_index);
    cmd.target_system = 1U;
    cmd.target_component = 1U;
    cmd.from_external = true;

    bool valid = true;
    switch (req->fault_kind) {
      case Srv::FAULT_MOTOR_OFF:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SYSTEM_MOTOR);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_OFF);
        if (req->instance_index >= 1U && req->instance_index <= 8U) {
          motor_fault_mask_ |= (1U << (req->instance_index - 1U));
        }
        break;
      case Srv::FAULT_MOTOR_RESTORE:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SYSTEM_MOTOR);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_OK);
        if (req->instance_index == 0U) {
          motor_fault_mask_ = 0U;
        } else if (req->instance_index <= 8U) {
          motor_fault_mask_ &= ~(1U << (req->instance_index - 1U));
        }
        break;
      case Srv::FAULT_IMU_STUCK:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SENSOR_GYRO);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_STUCK);
        break;
      case Srv::FAULT_IMU_WRONG:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SENSOR_GYRO);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_WRONG);
        break;
      case Srv::FAULT_IMU_OFF:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SENSOR_GYRO);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_OFF);
        break;
      case Srv::FAULT_GPS_OFF:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SENSOR_GPS);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_OFF);
        break;
      case Srv::FAULT_BARO_STUCK:
        cmd.param1 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_UNIT_SENSOR_BARO);
        cmd.param2 = static_cast<float>(px4_msgs::msg::VehicleCommand::FAILURE_TYPE_STUCK);
        break;
      default:
        valid = false;
        break;
    }

    if (!valid) {
      res->accepted = false;
      res->applied_stamp_ns = now_ns;
      res->message = "Unsupported fault_kind";
      return;
    }

    cmd_pub_->publish(cmd);
    if (req->use_gz_topic_fallback ||
        req->fault_kind == Srv::FAULT_MOTOR_OFF ||
        req->fault_kind == Srv::FAULT_MOTOR_RESTORE) {
      std_msgs::msg::UInt32 mask_msg{};
      mask_msg.data = motor_fault_mask_;
      gz_mask_pub_->publish(mask_msg);
    }

    res->accepted = true;
    res->applied_stamp_ns = now_ns;
    res->message = "Fault command dispatched via VEHICLE_CMD_INJECT_FAILURE";
  }

  uint32_t motor_fault_mask_{0U};
  rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr cmd_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr gz_mask_pub_;
  rclcpp::Service<vertirta_msgs::srv::InjectFault>::SharedPtr srv_;
};

}  // namespace vertirta::fault_injector

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::fault_injector::VertiRtaFaultInjectorNode>());
  rclcpp::shutdown();
  return 0;
}
