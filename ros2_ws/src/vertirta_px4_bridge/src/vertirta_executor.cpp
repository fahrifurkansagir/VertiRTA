#include "vertirta_px4_bridge/vertirta_executor.hpp"

#include <chrono>

namespace vertirta::px4_bridge {

namespace {

std::string scoped_action(const std::string& ns, const std::string& action) {
  if (ns.empty() || ns == "/") {
    return action;
  }
  if (ns.front() == '/') {
    return ns + action;
  }
  return "/" + ns + action;
}

}  // namespace

VertiRtaExecutor::VertiRtaExecutor(
    rclcpp::Node& node,
    VertiRtaMode& owned_mode,
    const std::string& vehicle_namespace)
    : px4_ros2::ModeExecutorBase(
          node,
          px4_ros2::ModeExecutorBase::Settings{}.activate(
              px4_ros2::ModeExecutorBase::Settings::Activation::ActivateImmediately),
          owned_mode),
      node_(node),
      mode_(owned_mode),
      vehicle_namespace_(vehicle_namespace) {
  // Bug #191 mitigation: namespace-isolated action server per vehicle/worker
  action_server_ = rclcpp_action::create_server<ExecuteContingency>(
      &node_,
      scoped_action(vehicle_namespace_, "/vertirta/execute_contingency"),
      [this](const rclcpp_action::GoalUUID& uuid, std::shared_ptr<const ExecuteContingency::Goal> goal) {
        return handle_goal(uuid, goal);
      },
      [this](const std::shared_ptr<GoalHandle> gh) { return handle_cancel(gh); },
      [this](const std::shared_ptr<GoalHandle> gh) { handle_accepted(gh); });

  // Bug #167 mitigation: all scheduleMode() calls execute strictly from this deferred timer
  state_timer_ = node_.create_wall_timer(
      std::chrono::milliseconds(50),
      [this]() { dispatch_pending_transitions(); });
}

void VertiRtaExecutor::onActivate() {
  RCLCPP_INFO(
      node_.get_logger(),
      "VertiRtaExecutor activated (ns='%s', Bug #201 persistent node / model_only reset mode)",
      vehicle_namespace_.c_str());
  request_deferred_transition(State::kRunningRta);
}

void VertiRtaExecutor::onDeactivate(DeactivateReason reason) {
  RCLCPP_INFO(
      node_.get_logger(),
      "VertiRtaExecutor deactivated (reason=%d)",
      static_cast<int>(reason));
  current_state_.store(State::kIdle, std::memory_order_release);
}

void VertiRtaExecutor::request_deferred_transition(State next_state) noexcept {
  pending_state_.store(next_state, std::memory_order_release);
  has_pending_state_.store(true, std::memory_order_release);
}

void VertiRtaExecutor::dispatch_pending_transitions() {
  // Monitor irreversible CEL latch: call deferFailsafesSync(true, 60) once upon entry.
  // Note (DR-2 lock): deferFailsafesSync does NOT suppress attitude-limit failsafes, so
  // PX4 airframe 12050_gz_vertirta_x8_240 ALSO locks FD_FAIL_R=0 and FD_FAIL_P=0.
  if (mode_.is_cel_latched() && !cel_failsafes_deferred_.exchange(true, std::memory_order_acq_rel)) {
    const auto res = deferFailsafesSync(true, 60);
    RCLCPP_WARN(
        node_.get_logger(),
        "CEL latched: deferFailsafesSync(true, 60) returned %d (FD_FAIL_R/P=0 active)",
        static_cast<int>(res));
    request_deferred_transition(State::kCelDeferredLanding);
  }

  if (!has_pending_state_.exchange(false, std::memory_order_acq_rel)) {
    return;
  }

  const State target = pending_state_.load(std::memory_order_acquire);
  current_state_.store(target, std::memory_order_release);

  switch (target) {
    case State::kRunningRta:
    case State::kCelDeferredLanding:
      scheduleMode(ownedMode().id(), [this](px4_ros2::Result result) {
        if (result != px4_ros2::Result::Success) {
          // Bug #167 mitigation: never reschedule directly inside callback; defer to next tick
          request_deferred_transition(State::kHoldLoiter);
        }
      });
      break;
    case State::kHoldLoiter:
      scheduleMode(px4_ros2::ModeBase::kModeIDLoiter, [](px4_ros2::Result) {});
      break;
    case State::kLanded:
      land([](px4_ros2::Result) {});
      break;
    default:
      break;
  }
}

rclcpp_action::GoalResponse VertiRtaExecutor::handle_goal(
    const rclcpp_action::GoalUUID& /*uuid*/,
    std::shared_ptr<const ExecuteContingency::Goal> goal) {
  if (!goal) {
    return rclcpp_action::GoalResponse::REJECT;
  }
  // Fail-closed EASA AI Level 1/2A gate: non-Z3-verified diversions fall back to ACTION_HOLD_LOITER
  if (!goal->z3_verified &&
      goal->action_kind != ExecuteContingency::Goal::ACTION_HOLD_LOITER &&
      goal->action_kind != ExecuteContingency::Goal::ACTION_CEL_LAND_NOW) {
    RCLCPP_WARN(
        node_.get_logger(),
        "Rejecting unverified contingency goal for target '%s' -> enforcing ACTION_HOLD_LOITER fallback",
        goal->target_id.c_str());
    request_deferred_transition(State::kHoldLoiter);
    return rclcpp_action::GoalResponse::REJECT;
  }
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse VertiRtaExecutor::handle_cancel(
    const std::shared_ptr<GoalHandle> /*goal_handle*/) {
  request_deferred_transition(State::kHoldLoiter);
  return rclcpp_action::CancelResponse::ACCEPT;
}

void VertiRtaExecutor::handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    active_goal_ = goal_handle;
  }
  const auto goal = goal_handle->get_goal();

  // Bug #167 mitigation: request deferred transition instead of calling scheduleMode() here
  if (mode_.is_cel_latched() ||
      goal->action_kind == ExecuteContingency::Goal::ACTION_CEL_LAND_NOW) {
    request_deferred_transition(State::kCelDeferredLanding);
  } else if (goal->action_kind == ExecuteContingency::Goal::ACTION_HOLD_LOITER) {
    request_deferred_transition(State::kHoldLoiter);
  } else {
    request_deferred_transition(State::kRunningRta);
  }

  auto result = std::make_shared<ExecuteContingency::Result>();
  result->accepted = true;
  result->completed = true;
  result->final_regime = mode_.flight_regime();
  result->status_text = "Dispatched asynchronously via deferred ModeExecutorBase queue";
  goal_handle->succeed(result);
}

}  // namespace vertirta::px4_bridge
