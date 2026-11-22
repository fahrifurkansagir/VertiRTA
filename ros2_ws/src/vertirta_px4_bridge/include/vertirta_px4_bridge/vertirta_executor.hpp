#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <px4_ros2/components/mode_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <vertirta_msgs/action/execute_contingency.hpp>

#include "vertirta_px4_bridge/vertirta_mode.hpp"

namespace vertirta::px4_bridge {

// px4-ros2-interface-lib ModeExecutorBase hosting /vertirta/execute_contingency
// with explicit upstream bug mitigations (DR-2 / Plan 1A Task 12):
//   - Bug #201 (waitForFMU hang across PX4 restarts): persistent ROS 2 node + Gazebo
//     monotonic `reset: {model_only: true}` keeping UXRCE_DDS_SYNCT=1 monotonic.
//   - Bug #167 (active mode reschedule crash): deferred pending_state_ dispatch on a
//     non-reentrant timer tick — NEVER calls scheduleMode() inside completion/action callbacks.
//   - Bug #191 (multi-vehicle clash): isolated vehicle_namespace parameter per worker.
class VertiRtaExecutor : public px4_ros2::ModeExecutorBase {
 public:
  using ExecuteContingency = vertirta_msgs::action::ExecuteContingency;
  using GoalHandle = rclcpp_action::ServerGoalHandle<ExecuteContingency>;

  enum class State : uint8_t {
    kIdle = 0,
    kTakingOff = 1,
    kRunningRta = 2,
    kHoldLoiter = 3,
    kCelDeferredLanding = 4,
    kLanded = 5,
  };

  VertiRtaExecutor(
      rclcpp::Node& node,
      VertiRtaMode& owned_mode,
      const std::string& vehicle_namespace = "");

  void onActivate() override;
  void onDeactivate(DeactivateReason reason) override;

 private:
  void request_deferred_transition(State next_state) noexcept;
  void dispatch_pending_transitions();

  rclcpp_action::GoalResponse handle_goal(
      const rclcpp_action::GoalUUID& uuid,
      std::shared_ptr<const ExecuteContingency::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle);

  rclcpp::Node& node_;
  VertiRtaMode& mode_;
  std::string vehicle_namespace_;

  // Bug #167 mitigation: non-reentrant deferred state transitions driven by 20 Hz timer
  std::atomic<State> current_state_{State::kIdle};
  std::atomic<bool> has_pending_state_{false};
  std::atomic<State> pending_state_{State::kIdle};
  std::atomic<bool> cel_failsafes_deferred_{false};

  rclcpp::TimerBase::SharedPtr state_timer_;
  rclcpp_action::Server<ExecuteContingency>::SharedPtr action_server_;
  std::mutex goal_mutex_;
  std::shared_ptr<GoalHandle> active_goal_;
};

}  // namespace vertirta::px4_bridge
