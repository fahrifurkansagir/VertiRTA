#pragma once

#include <cstdint>
#include <string>

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vertirta_msgs/msg/intruder_array.hpp>
#include <vertirta_msgs/msg/slz_prior_array.hpp>
#include <vertirta_msgs/msg/wire_capsule_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace vertirta::map {

// 20 Hz ESDF, WireCapsule, Intruder, and SLZ Prior adapter node (Plan 1A Task 12 + Plan 3 hook).
// Datum lock (Review Focus #5): kartal_coast.sdf reports 12.0 m AMSL; EGM2008 (+36.7 m) is applied
// ONCE to obtain 48.7 m HAE when evaluating LTFJ (798.58 m HAE) / LTBX (646.35 m HAE) ceilings.
class VertiRtaMapNode : public rclcpp::Node {
 public:
  explicit VertiRtaMapNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  static constexpr uint64_t kVisualPriorStaleNs = 250'000'000ULL;  // 250 ms staleness threshold
  static constexpr double kKartalDatumAmslM = 12.0;
  static constexpr double kEgm2008KartalM = 36.7;

  void on_lidar(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void on_odom(const px4_msgs::msg::VehicleOdometry::SharedPtr msg);
  void on_external_slz_priors(const vertirta_msgs::msg::SlzPriorArray::SharedPtr msg);
  void on_gimbal_cmd(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg);
  void publish_20hz_map_state();

  double pos_enu_[3]{0.0, 0.0, 0.0};
  uint64_t last_lidar_stamp_ns_{0ULL};
  uint64_t last_visual_slz_ns_{0ULL};
  float gimbal_pitch_rad_{0.0f};
  float gimbal_yaw_rad_{0.0f};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<vertirta_msgs::msg::SlzPriorArray>::SharedPtr visual_slz_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr gimbal_sub_;

  rclcpp::Publisher<vertirta_msgs::msg::WireCapsuleArray>::SharedPtr wire_pub_;
  rclcpp::Publisher<vertirta_msgs::msg::IntruderArray>::SharedPtr intruder_pub_;
  rclcpp::Publisher<vertirta_msgs::msg::SlzPriorArray>::SharedPtr slz_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_20hz_;
};

}  // namespace vertirta::map
