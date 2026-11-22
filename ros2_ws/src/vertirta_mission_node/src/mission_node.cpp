// SPDX-License-Identifier: Apache-2.0
#include "vertirta_mission_node/mission_core.hpp"

#ifdef VERTIRTA_WITH_ROS2

#include <chrono>
#include <memory>
#include <string>

#include <px4_msgs/msg/vertirta_capability_envelope.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <vertirta_msgs/action/execute_contingency.hpp>
#include <vertirta_msgs/msg/rta_state_snapshot.hpp>

namespace vertirta::mission {

// Isolated ROS 2 Jazzy EASA AI Level 1/2A Contingency Co-Pilot Node (Plan 2 Task 8).
// Consumes /vertirta/rta_state_snapshot & /fmu/out/vertirta_capability_envelope via lock-free
// snapshot buffer -> async std::jthread co-pilot worker -> /vertirta/execute_contingency Action
// to ModeExecutorBase with 0 us jitter impact on the 100 Hz ASTM F3269 RTA thread.
class VertiRtaMissionNode : public rclcpp::Node {
 public:
  using ExecuteContingency = vertirta_msgs::action::ExecuteContingency;

  explicit VertiRtaMissionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : Node("vertirta_mission_node", options) {
    MissionCoreConfig cfg{};
    cfg.gguf_path = declare_parameter<std::string>("gguf_model_path", "");
    cfg.use_cuda = declare_parameter<bool>("use_cuda", false);
    cfg.conformal_q_hat = static_cast<float>(declare_parameter<double>("conformal_q_hat", 0.70));
    cfg.clarification_timeout_s = declare_parameter<double>("clarification_timeout_s", 15.0);

    core_ = std::make_unique<MissionCore>(copilot::make_reference_400v_sml_snapshot(), std::move(cfg));

    action_client_ = rclcpp_action::create_client<ExecuteContingency>(
        this, "/vertirta/execute_contingency");

    response_pub_ = create_publisher<std_msgs::msg::String>("/vertirta/copilot_response", 10);
    stl_rho_pub_ = create_publisher<std_msgs::msg::Float32>("/vertirta/stl_robustness", 10);

    core_->set_goal_dispatch_callback([this](const DispatchedContingencyGoal& goal) {
      dispatch_verified_action_goal(goal);
    });
    core_->set_response_callback([this](const MissionPipelineResult& res) {
      std_msgs::msg::String msg{};
      msg.data = res.response_text;
      response_pub_->publish(msg);
    });

    auto sensor_qos = rclcpp::SensorDataQoS();
    snapshot_sub_ = create_subscription<vertirta_msgs::msg::RtaStateSnapshot>(
        "/vertirta/rta_state_snapshot", 10,
        [this](const vertirta_msgs::msg::RtaStateSnapshot::SharedPtr msg) { on_rta_snapshot(*msg); });

    env_sub_ = create_subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>(
        "/fmu/out/vertirta_capability_envelope", sensor_qos,
        [this](const px4_msgs::msg::VertirtaCapabilityEnvelope::SharedPtr msg) { on_envelope(*msg); });

    utterance_sub_ = create_subscription<std_msgs::msg::String>(
        "/vertirta/copilot_utterance", 10,
        [this](const std_msgs::msg::String::SharedPtr msg) { core_->enqueue_utterance(msg->data); });

    // 10 Hz clarification countdown watchdog (15.0 s auto-commit to Rank-1 Z3-SAT SLZ)
    watchdog_timer_ = create_wall_timer(
        std::chrono::milliseconds(100),
        [this]() { core_->advance_clock_sec(0.1); });

    core_->start_async_worker();
  }

  ~VertiRtaMissionNode() override {
    if (core_) {
      core_->stop_async_worker();
    }
  }

 private:
  void on_rta_snapshot(const vertirta_msgs::msg::RtaStateSnapshot& msg) {
    copilot::RtaStateSnapshotV1 st = core_->current_snapshot();
    st.timestamp_ms = msg.stamp_ns / 1'000'000ULL;
    st.fault_mode = static_cast<copilot::FaultMode>(msg.flight_regime);
    st.failed_motor_mask = msg.fault_bitmask;
    st.shield_active = true;
    st.pack_nominal_volts = msg.pack_nominal_volts;
    st.pack_energy_kwh = msg.pack_energy_kwh;
    st.v_term_volts = msg.v_term_volts;
    st.soc_pct = msg.soc_pct;
    st.soc_reserve_floor_pct = msg.soc_reserve_floor_pct;
    st.pos_enu = {msg.pos_enu.x, msg.pos_enu.y, msg.pos_enu.z};
    st.agl_m = msg.agl_m;
    st.groundspeed_mps = static_cast<float>(std::hypot(msg.vel_enu.x, msg.vel_enu.y));

    const std::size_t n_slz = std::min<std::size_t>(msg.slz_priors.size(), copilot::kMaxSlzCandidates);
    st.num_slz = static_cast<uint8_t>(n_slz);
    for (std::size_t i = 0U; i < n_slz; ++i) {
      auto& dst = st.ranked_slzs[i];
      const auto& src = msg.slz_priors[i];
      dst.set_id(src.id);
      dst.lat_deg = src.lat_deg;
      dst.lon_deg = src.lon_deg;
      dst.elev_amsl_m = static_cast<float>(src.elev_amsl_m);
      dst.enu = {src.center_enu.x, src.center_enu.y, src.center_enu.z};
      dst.p_safe = src.p_safe;
      dst.slz_ood_variance = src.slz_ood_variance;
      dst.eta_s = src.eta_s;
      dst.soc_margin_pct = src.soc_margin_pct;
      dst.rank = src.rank;
      dst.west_of_29200e = (src.lon_deg < copilot::kValidCoastalMaxLonDeg);
    }
    core_->update_snapshot(st);

    // Step online quantitative STL monitor @ 1 Hz snapshot cadence
    stl::StlSignalSample sample{};
    sample.east_m = st.pos_enu.east_m;
    sample.north_m = st.pos_enu.north_m;
    sample.agl_m = st.agl_m;
    sample.soc_pct = st.soc_pct;
    sample.groundspeed_mps = st.groundspeed_mps;
    const auto stl_out = core_->step_stl_monitor(stl_tau_++, sample);

    std_msgs::msg::Float32 rho_msg{};
    rho_msg.data = stl_out.rosi.sup;
    stl_rho_pub_->publish(rho_msg);
  }

  void on_envelope(const px4_msgs::msg::VertirtaCapabilityEnvelope& msg) {
    copilot::RtaStateSnapshotV1 st = core_->current_snapshot();
    st.failed_motor_mask = msg.fault_bitmask;
    st.v_term_volts = msg.v_term;
    st.soc_pct = msg.soc * 100.0f;
    st.ams.ams_max_accel_mps2 = msg.a_xy_max;
    core_->update_snapshot(st);
  }

  void dispatch_verified_action_goal(const DispatchedContingencyGoal& goal) {
    if (!goal.z3_verified) {
      RCLCPP_ERROR(get_logger(), "Blocked unverified contingency goal (z3_verified=false)");
      return;
    }
    if (!action_client_->action_server_is_ready()) {
      RCLCPP_WARN(get_logger(), "ModeExecutorBase action server /vertirta/execute_contingency not ready");
      return;
    }
    ExecuteContingency::Goal ros_goal{};
    ros_goal.action_kind = goal.action_kind;
    ros_goal.target_id = goal.target_id;
    ros_goal.target_enu.x = goal.target_enu.east_m;
    ros_goal.target_enu.y = goal.target_enu.north_m;
    ros_goal.target_enu.z = goal.target_enu.up_m;
    ros_goal.eta_lim_s = goal.eta_lim_s;
    ros_goal.min_agl_m = goal.min_agl_m;
    ros_goal.max_agl_m = goal.max_agl_m;
    ros_goal.max_speed_mps = goal.max_speed_mps;
    ros_goal.loiter_duration_s = goal.loiter_duration_s;
    ros_goal.z3_verified = goal.z3_verified;
    ros_goal.z3_audit_hash = goal.z3_audit_hash;
    (void)action_client_->async_send_goal(ros_goal);
  }

  std::unique_ptr<MissionCore> core_;
  uint32_t stl_tau_{0U};
  rclcpp_action::Client<ExecuteContingency>::SharedPtr action_client_;
  rclcpp::Subscription<vertirta_msgs::msg::RtaStateSnapshot>::SharedPtr snapshot_sub_;
  rclcpp::Subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>::SharedPtr env_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr utterance_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr response_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr stl_rho_pub_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
};

}  // namespace vertirta::mission

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::mission::VertiRtaMissionNode>());
  rclcpp::shutdown();
  return 0;
}

#else

int main() {
  vertirta::mission::MissionCore core{};
  const auto res = core.submit_utterance("Kartal Sahil'e inis planla");
  return res.goal_dispatched ? 0 : 1;
}

#endif
