#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "vertirta_px4_bridge/vertirta_executor.hpp"
#include "vertirta_px4_bridge/vertirta_mode.hpp"

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);

  // Bug #201 & #191 mitigation: persistent node lifecycle (never restarted between SITL runs;
  // Gazebo uses monotonic reset: {model_only: true}) with configurable vehicle_namespace.
  auto node = std::make_shared<rclcpp::Node>("vertirta_px4_bridge");
  const std::string vehicle_namespace =
      node->declare_parameter<std::string>("vehicle_namespace", "");

  auto mode = std::make_unique<vertirta::px4_bridge::VertiRtaMode>(*node, vehicle_namespace);
  auto executor_mode = std::make_unique<vertirta::px4_bridge::VertiRtaExecutor>(
      *node, *mode, vehicle_namespace);
  if (!executor_mode->doRegister()) {
    RCLCPP_FATAL(node->get_logger(), "Failed to register VertiRtaExecutor with PX4 FMU");
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::executors::SingleThreadedExecutor ros_exec;
  ros_exec.add_node(node);
  ros_exec.spin();
  rclcpp::shutdown();
  return 0;
}
