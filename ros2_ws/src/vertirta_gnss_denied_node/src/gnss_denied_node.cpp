// SPDX-License-Identifier: Apache-2.0
#include "vertirta_gnss_denied_node/gnss_denied_core.hpp"

#include <iostream>
#include <memory>

#if defined(VERTIRTA_WITH_ROS2) && VERTIRTA_WITH_ROS2
#include <nav_msgs/msg/odometry.hpp>
#include <px4_msgs/msg/vehicle_gps_position.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/range.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace vertirta::gnss_denied_node {

class GnssDeniedRosNode final : public rclcpp::Node {
 public:
  GnssDeniedRosNode() : Node("vertirta_gnss_denied_node") {
    GnssDeniedConfig cfg{};
    cfg.ekf2_gps_delay_ms = this->declare_parameter<double>("ekf2_gps_delay_ms", 110.0);
    cfg.uxrce_dds_synct_enabled = (this->declare_parameter<int>("uxrce_dds_synct", 1) == 1);
    cfg.land_mcusum_kd = this->declare_parameter<double>("land_mcusum_kd", 0.25);
    cfg.land_mcusum_ht = this->declare_parameter<double>("land_mcusum_ht", 2.0);
    cfg.water_mcusum_kd = this->declare_parameter<double>("water_mcusum_kd", 0.35);
    cfg.water_mcusum_ht = this->declare_parameter<double>("water_mcusum_ht", 3.4);
    cfg.v_max_caution_mps = this->declare_parameter<double>("v_max_caution_mps", 13.5);
    core_ = GnssDeniedNodeCore{cfg};

    const auto sensor_qos = rclcpp::SensorDataQoS();
    sub_gps_ = this->create_subscription<px4_msgs::msg::VehicleGpsPosition>(
        kTopicVehicleGpsPosition, sensor_qos,
        [this](const px4_msgs::msg::VehicleGpsPosition::SharedPtr msg) { on_gps(*msg); });
    sub_odom_ = this->create_subscription<px4_msgs::msg::VehicleOdometry>(
        kTopicVehicleOdometry, sensor_qos,
        [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) { on_odom(*msg); });
    sub_left_ = this->create_subscription<sensor_msgs::msg::Image>(
        kTopicBellyStereoLeft, sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr /*msg*/) { ++stereo_frames_; });
    sub_right_ = this->create_subscription<sensor_msgs::msg::Image>(
        kTopicBellyStereoRight, sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr /*msg*/) {});
    sub_radalt_ = this->create_subscription<sensor_msgs::msg::Range>(
        kTopicRadaltRange, sensor_qos,
        [this](const sensor_msgs::msg::Range::SharedPtr msg) { last_agl_m_ = msg->range; });
    sub_water_ = this->create_subscription<std_msgs::msg::Float32>(
        kTopicSlzWaterFraction, 10,
        [this](const std_msgs::msg::Float32::SharedPtr msg) { water_fraction_ = msg->data; });

    pub_status_ = this->create_publisher<std_msgs::msg::String>(kTopicGnssDeniedStatus, 10);
    pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>(kTopicGnssDeniedOdometry, 10);
    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(kTopicCtrInflationMarkers, 10);
    pub_speed_cap_ = this->create_publisher<std_msgs::msg::Float32>(kTopicGnssSpeedCap, 10);
  }

 private:
  void on_odom(const px4_msgs::msg::VehicleOdometry& msg) {
    last_vio_pos_ = {static_cast<double>(msg.position[1]), static_cast<double>(msg.position[0]),
                     -static_cast<double>(msg.position[2])};
    last_vio_vel_ = {static_cast<double>(msg.velocity[1]), static_cast<double>(msg.velocity[0]),
                     -static_cast<double>(msg.velocity[2])};
  }

  void on_gps(const px4_msgs::msg::VehicleGpsPosition& msg) {
    const double now_s = this->now().seconds();
    const double dt_s = (last_step_s_ > 0.0) ? std::clamp(now_s - last_step_s_, 0.01, 0.5) : 0.1;
    last_step_s_ = now_s;

    GnssDeniedStepInput in{};
    in.t_s = now_s;
    in.dt_s = dt_s;
    in.gps_valid = (msg.fix_type >= 3U);
    in.timesync_ok = (msg.timestamp != 0U);
    in.cn0_dbhz = (msg.jamming_indicator > 80U) ? 20.0 : 42.0;
    in.hdop = static_cast<double>(msg.hdop);
    in.water_fraction = static_cast<double>(water_fraction_);
    const double lat_deg = static_cast<double>(msg.lat) * 1e-7;
    const double lon_deg = static_cast<double>(msg.lon) * 1e-7;
    in.pos_gnss_enu = {
        (lon_deg - kKartalDatumLonDeg) * 84255.0,
        (lat_deg - kKartalDatumLatDeg) * 111052.0,
        static_cast<double>(last_agl_m_)};
    in.vel_gnss_enu = {static_cast<double>(msg.vel_e_m_s), static_cast<double>(msg.vel_n_m_s),
                       -static_cast<double>(msg.vel_d_m_s)};
    in.pos_vio_enu = last_vio_pos_;
    in.vel_vio_enu = last_vio_vel_;

    const auto out = core_.step(in);

    std_msgs::msg::String st{};
    st.data = core_.to_status_json(out);
    pub_status_->publish(st);

    nav_msgs::msg::Odometry odom{};
    odom.header.stamp = this->now();
    odom.header.frame_id = "kartal_enu";
    odom.pose.pose.position.x = out.est_pos_enu.x;
    odom.pose.pose.position.y = out.est_pos_enu.y;
    odom.pose.pose.position.z = out.est_pos_enu.z;
    odom.pose.covariance[0] = out.sigma_pos_m * out.sigma_pos_m;
    odom.pose.covariance[7] = out.sigma_pos_m * out.sigma_pos_m;
    odom.twist.twist.linear.x = out.est_vel_enu.x;
    odom.twist.twist.linear.y = out.est_vel_enu.y;
    odom.twist.twist.linear.z = out.est_vel_enu.z;
    pub_odom_->publish(odom);

    std_msgs::msg::Float32 cap{};
    cap.data = static_cast<float>(out.v_max_cap_mps);
    pub_speed_cap_->publish(cap);

    visualization_msgs::msg::MarkerArray mk{};
    pub_markers_->publish(mk);
  }

  GnssDeniedNodeCore core_{};
  core::Vec3Enu last_vio_pos_{};
  core::Vec3Enu last_vio_vel_{};
  float water_fraction_{0.05f};
  float last_agl_m_{45.0f};
  double last_step_s_{0.0};
  uint64_t stereo_frames_{0U};

  rclcpp::Subscription<px4_msgs::msg::VehicleGpsPosition>::SharedPtr sub_gps_{};
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr sub_odom_{};
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_left_{};
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_right_{};
  rclcpp::Subscription<sensor_msgs::msg::Range>::SharedPtr sub_radalt_{};
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr sub_water_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_{};
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_{};
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_{};
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_speed_cap_{};
};

}  // namespace vertirta::gnss_denied_node

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::gnss_denied_node::GnssDeniedRosNode>());
  rclcpp::shutdown();
  return 0;
}
#else
int main() {
  using namespace vertirta::gnss_denied_node;
  GnssDeniedNodeCore core{};
  GnssDeniedStepInput in{};
  in.t_s = 1.0;
  in.dt_s = 0.1;
  in.pos_gnss_enu = {-500.0, 200.0, 45.0};
  in.pos_vio_enu = {-500.0, 200.0, 45.0};
  const auto out = core.step(in);
  std::cout << core.to_status_json(out) << "\n";
  return 0;
}
#endif
