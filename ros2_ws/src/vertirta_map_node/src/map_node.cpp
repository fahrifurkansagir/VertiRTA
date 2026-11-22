#include "vertirta_map_node/map_node.hpp"

#include <chrono>
#include <cmath>
#include <memory>

namespace vertirta::map {

VertiRtaMapNode::VertiRtaMapNode(const rclcpp::NodeOptions& options)
    : Node("vertirta_map_node", options) {
  auto sensor_qos = rclcpp::SensorDataQoS();

  lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/vertirta/lidar_16", sensor_qos,
      [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { on_lidar(msg); });

  odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
      "/fmu/out/vehicle_odometry", sensor_qos,
      [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) { on_odom(msg); });

  // Plan 3 visual SLZ provider override on canonical /vertirta/slz_priors
  // (ignore_local_publications prevents self-loop when geometric fallback publishes)
  rclcpp::SubscriptionOptions ext_slz_opts;
  ext_slz_opts.ignore_local_publications = true;
  visual_slz_sub_ = create_subscription<vertirta_msgs::msg::SlzPriorArray>(
      "/vertirta/slz_priors", 10,
      [this](const vertirta_msgs::msg::SlzPriorArray::SharedPtr msg) { on_external_slz_priors(msg); },
      ext_slz_opts);

  gimbal_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/vertirta/slz/gimbal_cmd", 10,
      [this](const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) { on_gimbal_cmd(msg); });

  wire_pub_ = create_publisher<vertirta_msgs::msg::WireCapsuleArray>("/vertirta/wire_capsules", 10);
  intruder_pub_ = create_publisher<vertirta_msgs::msg::IntruderArray>("/vertirta/intruders", 10);
  slz_pub_ = create_publisher<vertirta_msgs::msg::SlzPriorArray>("/vertirta/slz_priors", 10);
  marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/vertirta/map_markers", 10);

  timer_20hz_ = create_wall_timer(
      std::chrono::milliseconds(50),
      [this]() { publish_20hz_map_state(); });
}

void VertiRtaMapNode::on_lidar(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
  last_lidar_stamp_ns_ =
      static_cast<uint64_t>(msg->header.stamp.sec) * 1'000'000'000ULL +
      static_cast<uint64_t>(msg->header.stamp.nanosec);
}

void VertiRtaMapNode::on_odom(const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
  // PX4 NED -> Kartal local ENU: (E=y, N=x, U=-z)
  pos_enu_[0] = static_cast<double>(msg->position[1]);
  pos_enu_[1] = static_cast<double>(msg->position[0]);
  pos_enu_[2] = static_cast<double>(-msg->position[2]);
}

void VertiRtaMapNode::on_external_slz_priors(const vertirta_msgs::msg::SlzPriorArray::SharedPtr msg) {
  if (!msg->stale) {
    last_visual_slz_ns_ = msg->stamp_ns;
  }
}

void VertiRtaMapNode::on_gimbal_cmd(const geometry_msgs::msg::Vector3Stamped::SharedPtr msg) {
  gimbal_pitch_rad_ = static_cast<float>(msg->vector.y);
  gimbal_yaw_rad_ = static_cast<float>(msg->vector.z);
}

void VertiRtaMapNode::publish_20hz_map_state() {
  const auto stamp = now();
  const uint64_t now_ns = static_cast<uint64_t>(stamp.nanoseconds());

  // 1. Publish static coastal powerline wire capsule matching kartal_coast.sdf pylons
  vertirta_msgs::msg::WireCapsuleArray wires{};
  wires.header.stamp = stamp;
  wires.header.frame_id = "kartal_enu";
  wires.stamp_ns = now_ns;
  wires.stale = false;
  wires.ids = {1U};
  geometry_msgs::msg::Point p0{};
  p0.x = -25.0;
  p0.y = 55.0;
  p0.z = 24.0;
  geometry_msgs::msg::Point p1{};
  p1.x = -25.0;
  p1.y = 95.0;
  p1.z = 24.0;
  wires.p0_enu = {p0};
  wires.p1_enu = {p1};
  wires.radius_m = {3.5f};
  wires.voltage_class = {vertirta_msgs::msg::WireCapsuleArray::CLASS_DISTRIBUTION};
  wire_pub_->publish(wires);

  // 2. Publish intruder array heartbeat
  vertirta_msgs::msg::IntruderArray intruders{};
  intruders.header.stamp = stamp;
  intruders.header.frame_id = "kartal_enu";
  intruders.stamp_ns = now_ns;
  intruders.stale = false;
  intruder_pub_->publish(intruders);

  // 3. Geometric SLZ fallback when Plan 3 visual provider is absent or stale (>250 ms)
  const bool visual_fresh =
      (last_visual_slz_ns_ > 0ULL) && (now_ns >= last_visual_slz_ns_) &&
      ((now_ns - last_visual_slz_ns_) <= kVisualPriorStaleNs);
  if (!visual_fresh) {
    vertirta_msgs::msg::SlzPriorArray slz_arr{};
    slz_arr.header.stamp = stamp;
    slz_arr.header.frame_id = "kartal_enu";
    slz_arr.stamp_ns = now_ns;
    slz_arr.stale = (last_lidar_stamp_ns_ > 0ULL) && (now_ns > last_lidar_stamp_ns_) &&
                    ((now_ns - last_lidar_stamp_ns_) > kVisualPriorStaleNs);
    slz_arr.active_decision_state = vertirta_msgs::msg::SlzPriorArray::STATE_IDLE;
    slz_arr.gimbal_pitch_rad = gimbal_pitch_rad_;
    slz_arr.gimbal_yaw_rad = gimbal_yaw_rad_;

    vertirta_msgs::msg::SlzPrior kartal{};
    kartal.id = "Kartal_Coastal_Vertiport";
    kartal.lat_deg = 40.8860;
    kartal.lon_deg = 29.1880;
    kartal.elev_amsl_m = kKartalDatumAmslM;  // +36.7 m EGM2008 applied once downstream -> 48.7 m HAE
    kartal.center_enu.x = 0.0;
    kartal.center_enu.y = 0.0;
    kartal.center_enu.z = 0.0;
    kartal.p_safe = 0.96f;
    kartal.slz_ood_variance = 0.03f;
    kartal.theta_max_deg = 1.5f;
    kartal.sigma_z_m = 0.05f;
    kartal.step_max_m = 0.04f;
    kartal.clearance_r_m = 15.0f;
    kartal.h_dyn_max = 0.0f;
    const double dist = std::hypot(pos_enu_[0], pos_enu_[1]);
    kartal.eta_s = static_cast<float>(dist / 12.0);
    kartal.soc_margin_pct = 45.0f;
    kartal.rank = 1U;
    kartal.reason_mask = vertirta_msgs::msg::SlzPrior::REASON_NONE;
    slz_arr.priors = {kartal};
    slz_pub_->publish(slz_arr);
  }
}

}  // namespace vertirta::map

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::map::VertiRtaMapNode>());
  rclcpp::shutdown();
  return 0;
}
