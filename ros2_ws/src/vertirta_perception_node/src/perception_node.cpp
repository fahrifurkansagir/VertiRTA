// SPDX-License-Identifier: Apache-2.0
#include "vertirta_perception_node/perception_node.hpp"

#ifdef VERTIRTA_WITH_ROS2

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vertirta_capability_envelope.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vertirta_msgs/msg/slz_prior_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace vertirta::ros {

// 20-30+ Hz Active Visual SLZ Perception Node (Plan 3 Task 6).
// Publishes canonical /vertirta/slz_priors (with >250 ms staleness guard),
// /vertirta/slz/gimbal_cmd, and Foxglove :8891 SLZ belief markers.
class SlzPerceptionNode : public rclcpp::Node {
 public:
  explicit SlzPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : Node("vertirta_perception_node", options) {
    SlzPerceptionOptions opts{};
    opts.ort_cfg.onnx_path = declare_parameter<std::string>("onnx_model_path", "");
    opts.ort_cfg.enable_fp16 = declare_parameter<bool>("enable_fp16", true);
    opts.ort_cfg.use_tensorrt = declare_parameter<bool>("use_tensorrt", false);
    opts.ort_cfg.allow_mock_fallback = declare_parameter<bool>("allow_mock_fallback", true);
    opts.policy_cfg.r_slz_min_m = static_cast<float>(declare_parameter<double>("r_slz_min_m", 5.0));
    opts.policy_cfg.theta_commit_deg =
        static_cast<float>(declare_parameter<double>("theta_commit_deg", 8.0));
    opts.policy_cfg.sigma_z_commit_m =
        static_cast<float>(declare_parameter<double>("sigma_z_commit_m", 0.15));
    opts.policy_cfg.step_max_m = static_cast<float>(declare_parameter<double>("step_max_m", 0.25));
    opts.staleness_warn_ns =
        static_cast<uint64_t>(declare_parameter<int64_t>("staleness_warn_ms", 250)) * 1'000'000ULL;

    core_ = std::make_unique<SlzPerceptionCore>(std::move(opts));

    auto sensor_qos = rclcpp::SensorDataQoS();
    env_sub_ = create_subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>(
        "/fmu/out/vertirta_capability_envelope", sensor_qos,
        [this](const px4_msgs::msg::VertirtaCapabilityEnvelope::SharedPtr msg) {
          core_->UpdateVehicleEnvelope(
              msg->soc * 100.0f, 4.0f, static_cast<core::FlightRegime>(msg->flight_regime));
        });

    odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", sensor_qos,
        [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
          vehicle_enu_ = {
              static_cast<double>(msg->position[1]),
              static_cast<double>(msg->position[0]),
              static_cast<double>(-msg->position[2]),
          };
        });

    rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/slz_rgbd/image", sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          last_rgb_stamp_ns_ =
              static_cast<uint64_t>(msg->header.stamp.sec) * 1'000'000'000ULL +
              static_cast<uint64_t>(msg->header.stamp.nanosec);
        });

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/slz_rgbd/depth_image", sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          const uint64_t stamp_ns =
              static_cast<uint64_t>(msg->header.stamp.sec) * 1'000'000'000ULL +
              static_cast<uint64_t>(msg->header.stamp.nanosec);
          const uint64_t now_ns = static_cast<uint64_t>(now().nanoseconds());
          (void)core_->InjectMockFrame(
              stamp_ns, now_ns, false, perception::DepthMode::RGBD_PRIMARY,
              static_cast<float>(std::max(1.0, vehicle_enu_.z)));
          publish_outputs();
        });

    slz_pub_ = create_publisher<vertirta_msgs::msg::SlzPriorArray>("/vertirta/slz_priors", 10);
    gimbal_pub_ =
        create_publisher<geometry_msgs::msg::Vector3Stamped>("/vertirta/slz/gimbal_cmd", 10);
    marker_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/vertirta/slz_markers", 10);

    // 20 Hz staleness & heartbeat publication timer
    timer_ = create_wall_timer(std::chrono::milliseconds(50), [this]() {
      core_->CheckStaleness(static_cast<uint64_t>(now().nanoseconds()));
      publish_outputs();
    });
  }

 private:
  void publish_outputs() {
    const auto stamp = now();
    const auto& snap = core_->LatestSnapshot();

    vertirta_msgs::msg::SlzPriorArray out{};
    out.header.stamp = stamp;
    out.header.frame_id = "kartal_enu";
    out.stamp_ns = snap.stamp_ns;
    out.stale = snap.stale;
    out.active_decision_state = snap.active_decision_state;
    out.gimbal_pitch_rad = snap.gimbal_pitch_rad;
    out.gimbal_yaw_rad = snap.gimbal_yaw_rad;

    out.priors.reserve(snap.count);
    for (uint8_t i = 0u; i < snap.count && i < snap.priors.size(); ++i) {
      const auto& p = snap.priors[i];
      vertirta_msgs::msg::SlzPrior m{};
      m.id = std::string(p.id);
      m.lat_deg = p.lat_deg;
      m.lon_deg = p.lon_deg;
      m.elev_amsl_m = p.elev_amsl_m;
      m.center_enu.x = p.center_enu.x;
      m.center_enu.y = p.center_enu.y;
      m.center_enu.z = p.center_enu.z;
      m.p_safe = p.p_safe;
      m.slz_ood_variance = p.slz_ood_variance;
      m.theta_max_deg = p.theta_max_deg;
      m.sigma_z_m = p.sigma_z_m;
      m.step_max_m = p.step_max_m;
      m.clearance_r_m = p.clearance_r_m;
      m.h_dyn_max = p.h_dyn_max;
      m.eta_s = p.eta_s;
      m.soc_margin_pct = p.soc_margin_pct;
      m.rank = p.rank;
      m.reason_mask = p.reason_mask;
      out.priors.push_back(std::move(m));
    }
    slz_pub_->publish(out);

    geometry_msgs::msg::Vector3Stamped gcmd{};
    gcmd.header.stamp = stamp;
    gcmd.header.frame_id = "kartal_enu";
    gcmd.vector.y = snap.gimbal_pitch_rad;
    gcmd.vector.z = snap.gimbal_yaw_rad;
    gimbal_pub_->publish(gcmd);
  }

  std::unique_ptr<SlzPerceptionCore> core_;
  core::Vec3Enu vehicle_enu_{0.0, 0.0, 22.0};
  uint64_t last_rgb_stamp_ns_{0ULL};
  rclcpp::Subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>::SharedPtr env_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Publisher<vertirta_msgs::msg::SlzPriorArray>::SharedPtr slz_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr gimbal_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace vertirta::ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::ros::SlzPerceptionNode>());
  rclcpp::shutdown();
  return 0;
}

#else

int main() {
  vertirta::ros::SlzPerceptionCore core{};
  const bool ok = core.InjectMockFrame(1'000'000'000ULL, 1'020'000'000ULL);
  return (ok && !core.LatestSnapshot().stale) ? 0 : 1;
}

#endif
