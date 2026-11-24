// SPDX-License-Identifier: Apache-2.0
#include "vertirta_forward_hazard_node/forward_hazard_core.hpp"

#include <iostream>
#include <memory>

#if defined(VERTIRTA_WITH_ROS2) && VERTIRTA_WITH_ROS2
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <vertirta_msgs/msg/intruder_array.hpp>
#include <vertirta_msgs/msg/wire_capsule_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace vertirta::forward_hazard_node {

class ForwardHazardRosNode final : public rclcpp::Node {
 public:
  ForwardHazardRosNode() : Node("vertirta_forward_hazard_node") {
    ForwardHazardConfig cfg{};
    cfg.r_veh_m = this->declare_parameter<double>("r_veh_m", kCanonicalVehicleRadiusM);
    cfg.v_max_nominal_mps = this->declare_parameter<double>("v_max_nominal_mps", 22.0);
    cfg.v_max_radar_degraded_mps = this->declare_parameter<double>("v_max_radar_degraded_mps", 15.0);
    cfg.radar_grazing_skew_deg = this->declare_parameter<double>("radar_grazing_skew_deg", 20.0);
    cfg.enable_fp16 = this->declare_parameter<bool>("enable_fp16", false);
    core_ = ForwardHazardNodeCore{cfg};

    const auto sensor_qos = rclcpp::SensorDataQoS();
    sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>(
        kTopicForwardCamera, sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) { on_camera(*msg); });
    sub_radar_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        kTopicMmWaveDetections, sensor_qos,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { on_radar(*msg); });
    sub_odom_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
        kTopicVehicleOdometry, sensor_qos,
        [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) { on_odom(*msg); });
    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        kTopicImu200Hz, sensor_qos,
        [this](const sensor_msgs::msg::Imu::SharedPtr msg) { on_imu(*msg); });

    pub_wires_ = this->create_publisher<vertirta_msgs::msg::WireCapsuleArray>(kTopicWireCapsules, 10);
    pub_intruders_ = this->create_publisher<vertirta_msgs::msg::IntruderArray>(kTopicIntruders, 10);
    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(kTopicForwardHazardMarkers, 10);
    pub_status_ = this->create_publisher<std_msgs::msg::String>(kTopicForwardHazardStatus, 10);
    pub_speed_cap_ = this->create_publisher<std_msgs::msg::Float32>(kTopicForwardSpeedCap, 10);
  }

 private:
  void on_odom(const px4_msgs::msg::VehicleOdometry& msg) {
    OdometrySample s{};
    s.stamp_ns = static_cast<uint64_t>(this->now().nanoseconds());
    // PX4 NED -> Kartal ENU
    s.pos_enu = {static_cast<double>(msg.position[1]), static_cast<double>(msg.position[0]),
                 -static_cast<double>(msg.position[2])};
    s.vel_enu = {static_cast<double>(msg.velocity[1]), static_cast<double>(msg.velocity[0]),
                 -static_cast<double>(msg.velocity[2])};
    core_.push_odometry(s);
  }

  void on_imu(const sensor_msgs::msg::Imu& msg) {
    ImuSample s{};
    s.stamp_ns = static_cast<uint64_t>(this->now().nanoseconds());
    s.roll_rate_rps = msg.angular_velocity.x;
    s.pitch_rate_rps = msg.angular_velocity.y;
    s.yaw_rate_rps = msg.angular_velocity.z;
    core_.push_imu(s);
  }

  void on_radar(const sensor_msgs::msg::PointCloud2& msg) {
    last_radar_stamp_ns_ = static_cast<uint64_t>(this->now().nanoseconds());
    radar_healthy_ = (msg.width * msg.height) >= 0U;
  }

  void on_camera(const sensor_msgs::msg::Image& /*msg*/) {
    ForwardSensorBatch batch{};
    batch.stamp_ns = static_cast<uint64_t>(this->now().nanoseconds());
    batch.radar_healthy = radar_healthy_ && ((batch.stamp_ns - last_radar_stamp_ns_) <= 250'000'000ULL);
    const auto out = core_.step(batch);

    vertirta_msgs::msg::WireCapsuleArray wmsg{};
    wmsg.header.stamp = this->now();
    wmsg.header.frame_id = "kartal_enu";
    wmsg.stamp_ns = out.wire_msg.stamp_ns;
    wmsg.stale = out.wire_msg.stale;
    wmsg.ids = out.wire_msg.ids;
    wmsg.radius_m = out.wire_msg.radius_m;
    wmsg.voltage_class = out.wire_msg.voltage_class;
    for (std::size_t i = 0U; i < out.wire_msg.ids.size(); ++i) {
      geometry_msgs::msg::Point p0{};
      p0.x = out.wire_msg.p0_enu[i].x;
      p0.y = out.wire_msg.p0_enu[i].y;
      p0.z = out.wire_msg.p0_enu[i].z;
      geometry_msgs::msg::Point p1{};
      p1.x = out.wire_msg.p1_enu[i].x;
      p1.y = out.wire_msg.p1_enu[i].y;
      p1.z = out.wire_msg.p1_enu[i].z;
      wmsg.p0_enu.push_back(p0);
      wmsg.p1_enu.push_back(p1);
    }
    pub_wires_->publish(wmsg);

    vertirta_msgs::msg::IntruderArray imsg{};
    imsg.header.stamp = this->now();
    imsg.header.frame_id = "kartal_enu";
    imsg.stamp_ns = out.intruder_msg.stamp_ns;
    imsg.stale = out.intruder_msg.stale;
    imsg.track_ids = out.intruder_msg.track_ids;
    imsg.radius_horiz_m = out.intruder_msg.radius_horiz_m;
    imsg.half_height_m = out.intruder_msg.half_height_m;
    imsg.age_s = out.intruder_msg.age_s;
    imsg.source_type = out.intruder_msg.source_type;
    for (std::size_t i = 0U; i < out.intruder_msg.track_ids.size(); ++i) {
      geometry_msgs::msg::Point p{};
      p.x = out.intruder_msg.pos_enu[i].x;
      p.y = out.intruder_msg.pos_enu[i].y;
      p.z = out.intruder_msg.pos_enu[i].z;
      geometry_msgs::msg::Vector3 v{};
      v.x = out.intruder_msg.vel_enu[i].x;
      v.y = out.intruder_msg.vel_enu[i].y;
      v.z = out.intruder_msg.vel_enu[i].z;
      imsg.pos_enu.push_back(p);
      imsg.vel_enu.push_back(v);
    }
    pub_intruders_->publish(imsg);

    std_msgs::msg::String st{};
    st.data = core_.to_status_json(out);
    pub_status_->publish(st);

    std_msgs::msg::Float32 cap{};
    cap.data = static_cast<float>(out.v_max_cap_mps);
    pub_speed_cap_->publish(cap);

    visualization_msgs::msg::MarkerArray mk{};
    pub_markers_->publish(mk);
  }

  ForwardHazardNodeCore core_{};
  uint64_t last_radar_stamp_ns_{0U};
  bool radar_healthy_{false};

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam_{};
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_radar_{};
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr sub_odom_{};
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_{};
  rclcpp::Publisher<vertirta_msgs::msg::WireCapsuleArray>::SharedPtr pub_wires_{};
  rclcpp::Publisher<vertirta_msgs::msg::IntruderArray>::SharedPtr pub_intruders_{};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_{};
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_speed_cap_{};
};

}  // namespace vertirta::forward_hazard_node

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::forward_hazard_node::ForwardHazardRosNode>());
  rclcpp::shutdown();
  return 0;
}
#else
int main() {
  using namespace vertirta::forward_hazard_node;
  ForwardHazardNodeCore core{};
  core.push_odometry(OdometrySample{1'000'000'000ULL, {0.0, 0.0, 45.0}, {12.0, 0.0, 0.0}, 0.0, 0.0, 0.0});
  ForwardSensorBatch batch{};
  batch.stamp_ns = 1'050'000'000ULL;
  batch.radar_healthy = true;
  const auto out = core.step(batch);
  std::cout << core.to_status_json(out) << "\n";
  return 0;
}
#endif
