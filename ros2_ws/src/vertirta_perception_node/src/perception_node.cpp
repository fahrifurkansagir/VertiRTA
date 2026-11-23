// SPDX-License-Identifier: Apache-2.0
#include "vertirta_perception_node/perception_node.hpp"

#ifdef VERTIRTA_WITH_ROS2

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vertirta_capability_envelope.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vertirta_msgs/msg/slz_prior_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace vertirta::ros {

// 20-30+ Hz Active Visual SLZ Perception Node (Plan 3 Task 6 / v3.5.1 hardened).
// Runs real SlzOrtRunner::Run + |t_rgb - t_depth| <= 2 ms sync gate + CameraInfo distortion,
// and publishes /vertirta/slz_priors (with >250 ms staleness guard), /vertirta/slz/gimbal_cmd,
// and Foxglove :8891 /vertirta/slz_markers belief disks.
class SlzPerceptionNode : public rclcpp::Node {
 public:
  explicit SlzPerceptionNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : Node("vertirta_perception_node", options),
        rgb_nchw_buf_(perception::kSlzRgbElements, 0.0f),
        depth_m_buf_(perception::kSlzPixels, 0.0f) {
    SlzPerceptionOptions opts{};
    opts.ort_cfg.onnx_path = declare_parameter<std::string>("onnx_model_path", "");
    opts.ort_cfg.ood_calibration_path = declare_parameter<std::string>("ood_calibration_path", "");
    opts.ort_cfg.enable_fp16 = declare_parameter<bool>("enable_fp16", false);
    opts.ort_cfg.use_tensorrt = declare_parameter<bool>("use_tensorrt", false);
    opts.ort_cfg.enable_cuda_graph = declare_parameter<bool>("enable_cuda_graph", true);
    opts.ort_cfg.arena_extend_strategy =
        declare_parameter<std::string>("arena_extend_strategy", "kSameAsRequested");
    opts.ort_cfg.cudnn_conv_algo_search =
        declare_parameter<std::string>("cudnn_conv_algo_search", "AUTO");
    opts.ort_cfg.allow_mock_fallback = declare_parameter<bool>("allow_mock_fallback", false);
    opts.ort_cfg.t_ood = static_cast<float>(declare_parameter<double>("t_ood", 1.0));
    opts.ort_cfg.t_prob = static_cast<float>(declare_parameter<double>("t_prob", 1.0839));
    opts.ort_cfg.kappa_b = static_cast<float>(declare_parameter<double>("kappa_b", 1.5));
    opts.ort_cfg.tau_ood = static_cast<float>(declare_parameter<double>("tau_ood", -2.6106));
    opts.policy_cfg.r_slz_min_m = static_cast<float>(declare_parameter<double>("r_slz_min_m", 5.0));
    opts.policy_cfg.theta_commit_deg =
        static_cast<float>(declare_parameter<double>("theta_commit_deg", 8.0));
    opts.policy_cfg.sigma_z_commit_m =
        static_cast<float>(declare_parameter<double>("sigma_z_commit_m", 0.15));
    opts.policy_cfg.step_max_m = static_cast<float>(declare_parameter<double>("step_max_m", 0.25));
    opts.staleness_warn_ns =
        static_cast<uint64_t>(declare_parameter<int64_t>("staleness_warn_ms", 250)) * 1'000'000ULL;
    const double sync_ms = std::max(0.1, declare_parameter<double>("max_rgb_depth_sync_ms", 2.0));
    opts.max_rgbd_sync_ns = static_cast<uint64_t>(std::llround(sync_ms * 1'000'000.0));
    max_rgbd_sync_ns_ = opts.max_rgbd_sync_ns;

    core_ = std::make_unique<SlzPerceptionCore>(std::move(opts));
    RCLCPP_INFO(
        get_logger(),
        "SlzPerceptionNode ready=%d mock=%d tau_ood=%.4f T_prob=%.4f sync_gate=%.2f ms",
        core_->OrtRunner().is_ready() ? 1 : 0,
        core_->OrtRunner().is_mock_backend() ? 1 : 0,
        core_->Options().ort_cfg.tau_ood,
        core_->Options().ort_cfg.t_prob,
        sync_ms);

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

    cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/slz_rgbd/camera_info", sensor_qos,
        [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
          if (msg->width == 0u || msg->height == 0u || msg->k[0] <= 1e-6 || msg->k[4] <= 1e-6) {
            return;
          }
          perception::CameraIntrinsics intr{};
          intr.fx = static_cast<float>(msg->k[0]);
          intr.fy = static_cast<float>(msg->k[4]);
          intr.cx = static_cast<float>(msg->k[2]);
          intr.cy = static_cast<float>(msg->k[5]);
          intr.width = msg->width;
          intr.height = msg->height;
          if (msg->d.size() >= 4u) {
            intr.k1 = static_cast<float>(msg->d[0]);
            intr.k2 = static_cast<float>(msg->d[1]);
            intr.p1 = static_cast<float>(msg->d[2]);
            intr.p2 = static_cast<float>(msg->d[3]);
            if (msg->d.size() >= 5u) {
              intr.k3 = static_cast<float>(msg->d[4]);
            }
          }
          core_->SetCameraIntrinsics(intr);
        });

    rgb_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/slz_rgbd/image", sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          const bool is_bgr = (msg->encoding == "bgr8");
          const bool is_rgb = (msg->encoding == "rgb8");
          if (!is_rgb && !is_bgr) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Unsupported RGB image encoding '%s' (expected rgb8 or bgr8)",
                msg->encoding.c_str());
            return;
          }
          const auto stats = DecodeRgbImageToNchw512(
              msg->data, msg->width, msg->height, msg->step, is_bgr, rgb_nchw_buf_);
          if (!stats.ok) {
            return;
          }
          last_rgb_stamp_ns_ =
              static_cast<uint64_t>(msg->header.stamp.sec) * 1'000'000'000ULL +
              static_cast<uint64_t>(msg->header.stamp.nanosec);
          has_rgb_frame_ = true;
          if (stats.glare_warning) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Specular glare over-exposure detected (%.1f%% saturated pixels)",
                static_cast<double>(stats.saturated_pixel_frac) * 100.0);
          }
        });

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/slz_rgbd/depth_image", sensor_qos,
        [this](const sensor_msgs::msg::Image::SharedPtr msg) {
          const uint64_t depth_stamp_ns =
              static_cast<uint64_t>(msg->header.stamp.sec) * 1'000'000'000ULL +
              static_cast<uint64_t>(msg->header.stamp.nanosec);
          if (!has_rgb_frame_ ||
              !CheckRgbdTimestampSync(last_rgb_stamp_ns_, depth_stamp_ns, max_rgbd_sync_ns_)) {
            ++rgbd_sync_drop_count_;
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Dropped unsynchronized RGB-D pair (|t_rgb - t_depth| > %.2f ms, drops=%llu)",
                static_cast<double>(max_rgbd_sync_ns_) * 1e-6,
                static_cast<unsigned long long>(rgbd_sync_drop_count_));
            return;
          }

          const bool is_16uc1 = (msg->encoding == "16UC1");
          const bool is_32fc1 = (msg->encoding == "32FC1");
          if (!is_32fc1 && !is_16uc1) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "Unsupported depth image encoding '%s' (expected 32FC1 or 16UC1)",
                msg->encoding.c_str());
            return;
          }
          if (!DecodeDepthImageToFloat512(
                  msg->data, msg->width, msg->height, msg->step, is_16uc1, depth_m_buf_)) {
            return;
          }

          const uint64_t now_ns = static_cast<uint64_t>(now().nanoseconds());
          const float agl_m = static_cast<float>(std::max(1.0, vehicle_enu_.z));
          const auto cam_pose = perception::MakeGimbalCameraPoseEnu(
              vehicle_enu_, last_gimbal_yaw_rad_, last_gimbal_pitch_rad_);
          if (!core_->ProcessRgbdFrame(
                  rgb_nchw_buf_,
                  depth_m_buf_,
                  cam_pose,
                  vehicle_enu_,
                  agl_m,
                  perception::DepthMode::RGBD_PRIMARY,
                  depth_stamp_ns,
                  now_ns)) {
            RCLCPP_ERROR_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "SlzPerceptionCore::ProcessRgbdFrame failed (ORT runner not ready or inference error)");
            return;
          }

          const auto& snap = core_->LatestSnapshot();
          if (std::isfinite(snap.gimbal_pitch_rad)) {
            last_gimbal_pitch_rad_ = static_cast<double>(snap.gimbal_pitch_rad);
          }
          if (std::isfinite(snap.gimbal_yaw_rad)) {
            last_gimbal_yaw_rad_ = static_cast<double>(snap.gimbal_yaw_rad);
          }
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

    visualization_msgs::msg::MarkerArray markers{};
    markers.markers.reserve(static_cast<std::size_t>(snap.count) * 2u);

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

      visualization_msgs::msg::Marker disk{};
      disk.header.stamp = stamp;
      disk.header.frame_id = "kartal_enu";
      disk.ns = "slz_clearance_disks";
      disk.id = static_cast<int32_t>(i);
      disk.type = visualization_msgs::msg::Marker::CYLINDER;
      disk.action = visualization_msgs::msg::Marker::ADD;
      disk.pose.position.x = p.center_enu.x;
      disk.pose.position.y = p.center_enu.y;
      disk.pose.position.z = p.center_enu.z + 0.1;
      disk.pose.orientation.w = 1.0;
      const double diam = std::max(1.0, static_cast<double>(p.clearance_r_m) * 2.0);
      disk.scale.x = diam;
      disk.scale.y = diam;
      disk.scale.z = 0.2;
      if (snap.stale) {
        disk.color.r = 0.5f;
        disk.color.g = 0.5f;
        disk.color.b = 0.5f;
      } else if (p.reason_mask == perception::SlzReasonBits::REASON_NONE && p.p_safe >= 0.85f) {
        disk.color.r = 0.1f;
        disk.color.g = 0.85f;
        disk.color.b = 0.25f;
      } else if (p.p_safe >= 0.50f) {
        disk.color.r = 0.95f;
        disk.color.g = 0.70f;
        disk.color.b = 0.10f;
      } else {
        disk.color.r = 0.90f;
        disk.color.g = 0.15f;
        disk.color.b = 0.15f;
      }
      disk.color.a = 0.55f;
      markers.markers.push_back(std::move(disk));
    }
    slz_pub_->publish(out);
    marker_pub_->publish(markers);

    geometry_msgs::msg::Vector3Stamped gcmd{};
    gcmd.header.stamp = stamp;
    gcmd.header.frame_id = "kartal_enu";
    gcmd.vector.y = snap.gimbal_pitch_rad;
    gcmd.vector.z = snap.gimbal_yaw_rad;
    gimbal_pub_->publish(gcmd);
  }

  std::unique_ptr<SlzPerceptionCore> core_;
  std::vector<float> rgb_nchw_buf_;
  std::vector<float> depth_m_buf_;
  core::Vec3Enu vehicle_enu_{0.0, 0.0, 22.0};
  double last_gimbal_yaw_rad_{0.0};
  double last_gimbal_pitch_rad_{1.5707963267948966};  // default nadir
  uint64_t last_rgb_stamp_ns_{0ULL};
  uint64_t max_rgbd_sync_ns_{kDefaultMaxRgbdSyncNs};
  uint64_t rgbd_sync_drop_count_{0ULL};
  bool has_rgb_frame_{false};
  rclcpp::Subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>::SharedPtr env_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
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
  vertirta::ros::SlzPerceptionOptions opts{};
  opts.ort_cfg.allow_mock_fallback = true;
  vertirta::ros::SlzPerceptionCore core{std::move(opts)};
  const bool ok = core.InjectMockFrame(1'000'000'000ULL, 1'020'000'000ULL);
  return (ok && !core.LatestSnapshot().stale) ? 0 : 1;
}

#endif
