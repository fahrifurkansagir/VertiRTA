// SPDX-License-Identifier: Apache-2.0
#include "vertirta/sim/gz_egl_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace vertirta::sim {

QuatWxyz ComputeGimbalQuaternionEnu(double pitch_down_rad, double yaw_rad) noexcept {
  if (!std::isfinite(pitch_down_rad) || !std::isfinite(yaw_rad)) {
    return QuatWxyz{1.0, 0.0, 0.0, 0.0};
  }
  const double half_yaw = 0.5 * yaw_rad;
  const double half_pitch = 0.5 * pitch_down_rad;
  const double cy = std::cos(half_yaw);
  const double sy = std::sin(half_yaw);
  const double cp = std::cos(half_pitch);
  const double sp = std::sin(half_pitch);

  // Intrinsic Z-Y rotation: q = q_z(yaw) * q_y(pitch_down).
  // In Gazebo FLU (+X optical axis, +Y left, +Z up), +pitch_down tilts +X down toward -Z.
  QuatWxyz q{
      cy * cp,
      -sy * sp,
      cy * sp,
      sy * cp,
  };

  const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (norm <= 1e-12 || !std::isfinite(norm)) {
    return QuatWxyz{1.0, 0.0, 0.0, 0.0};
  }
  const double inv_norm = 1.0 / norm;
  q.w *= inv_norm;
  q.x *= inv_norm;
  q.y *= inv_norm;
  q.z *= inv_norm;
  return q;
}

std::array<double, 3> RotateVectorByQuat(const QuatWxyz& q,
                                         double vx,
                                         double vy,
                                         double vz) noexcept {
  // v' = v + 2 * q_vec x (q_vec x v + w * v)
  const double tx = 2.0 * (q.y * vz - q.z * vy);
  const double ty = 2.0 * (q.z * vx - q.x * vz);
  const double tz = 2.0 * (q.x * vy - q.y * vx);

  return {
      vx + q.w * tx + (q.y * tz - q.z * ty),
      vy + q.w * ty + (q.z * tx - q.x * tz),
      vz + q.w * tz + (q.x * ty - q.y * tx),
  };
}

bool GzEglSensorBridge::Init(const std::string& world, const std::string& rig_model_name) {
  if (world.empty() || rig_model_name.empty()) {
    return false;
  }
  {
    std::scoped_lock lk(mtx_);
    set_pose_srv_ = "/world/" + world + "/set_pose/blocking";
    rig_model_name_ = rig_model_name;
    rgb_buf_.assign(kRgbBytes, 0u);
    depth_buf_.assign(kDepthPixels, 0.0f);
    rgb_stamp_ns_ = 0;
    depth_stamp_ns_ = 0;
    has_rgb_ = false;
    has_depth_ = false;
  }

#ifdef VERTIRTA_HAS_GZ_TRANSPORT
  const bool ok_rgb = node_.Subscribe("/slz_rgbd/image", &GzEglSensorBridge::OnRgb, this);
  const bool ok_depth = node_.Subscribe("/slz_rgbd/depth_image", &GzEglSensorBridge::OnDepth, this);
  const bool ok = ok_rgb && ok_depth;
  initialized_.store(ok);
  return ok;
#else
  initialized_.store(true);
  return true;
#endif
}

bool GzEglSensorBridge::StepSensorRigPose(double x,
                                          double y,
                                          double z,
                                          double qw,
                                          double qx,
                                          double qy,
                                          double qz) {
  if (!initialized_.load()) {
    return false;
  }
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(qw) ||
      !std::isfinite(qx) || !std::isfinite(qy) || !std::isfinite(qz)) {
    return false;
  }
  const double qnorm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  if (qnorm <= 1e-9 || !std::isfinite(qnorm)) {
    return false;
  }
  const double inv_q = 1.0 / qnorm;
  const double nqw = qw * inv_q;
  const double nqx = qx * inv_q;
  const double nqy = qy * inv_q;
  const double nqz = qz * inv_q;

  std::string srv_name;
  std::string model_name;
  {
    std::scoped_lock lk(mtx_);
    last_pose_enu_ = {x, y, z, nqw, nqx, nqy, nqz};
    srv_name = set_pose_srv_;
    model_name = rig_model_name_;
  }

#ifdef VERTIRTA_HAS_GZ_TRANSPORT
  gz::msgs::Pose req;
  req.set_name(model_name);
  req.mutable_position()->set_x(x);
  req.mutable_position()->set_y(y);
  req.mutable_position()->set_z(z);
  req.mutable_orientation()->set_w(nqw);
  req.mutable_orientation()->set_x(nqx);
  req.mutable_orientation()->set_y(nqy);
  req.mutable_orientation()->set_z(nqz);

  gz::msgs::Boolean rep;
  bool result = false;
  constexpr unsigned int kTimeoutMs = 200u;
  return node_.Request(srv_name, req, kTimeoutMs, rep, result) && result && rep.data();
#else
  (void)srv_name;
  (void)model_name;
  return true;
#endif
}

bool GzEglSensorBridge::CopyLatestRgbd(std::vector<uint8_t>& rgb_out,
                                       std::vector<float>& depth_out,
                                       uint64_t& stamp_ns_out) const {
  std::scoped_lock lk(mtx_);
  if (!has_rgb_ || !has_depth_) {
    return false;
  }
  const uint64_t skew_ns = (rgb_stamp_ns_ >= depth_stamp_ns_)
                               ? (rgb_stamp_ns_ - depth_stamp_ns_)
                               : (depth_stamp_ns_ - rgb_stamp_ns_);
  if (skew_ns > kMaxPairSkewNs) {
    return false;
  }
  rgb_out = rgb_buf_;
  depth_out = depth_buf_;
  stamp_ns_out = std::max(rgb_stamp_ns_, depth_stamp_ns_);
  return true;
}

bool GzEglSensorBridge::InjectTestRgbFrame(std::span<const uint8_t> rgb, uint64_t stamp_ns) {
  if (rgb.size() != kRgbBytes) {
    return false;
  }
  std::scoped_lock lk(mtx_);
  rgb_buf_.assign(rgb.begin(), rgb.end());
  rgb_stamp_ns_ = stamp_ns;
  has_rgb_ = true;
  return true;
}

bool GzEglSensorBridge::InjectTestDepthFrame(std::span<const float> depth, uint64_t stamp_ns) {
  if (depth.size() != kDepthPixels) {
    return false;
  }
  std::scoped_lock lk(mtx_);
  depth_buf_.assign(depth.begin(), depth.end());
  depth_stamp_ns_ = stamp_ns;
  has_depth_ = true;
  return true;
}

#ifdef VERTIRTA_HAS_GZ_TRANSPORT
namespace {
uint64_t ExtractStampNs(const gz::msgs::Image& msg) noexcept {
  if (!msg.has_header() || !msg.header().has_stamp()) {
    return 0ULL;
  }
  const auto& st = msg.header().stamp();
  if (st.sec() < 0 || st.nsec() < 0) {
    return 0ULL;
  }
  return static_cast<uint64_t>(st.sec()) * 1'000'000'000ULL + static_cast<uint64_t>(st.nsec());
}
}  // namespace

void GzEglSensorBridge::OnRgb(const gz::msgs::Image& msg) {
  if (msg.data().size() != kRgbBytes) {
    return;
  }
  std::scoped_lock lk(mtx_);
  if (rgb_buf_.size() != kRgbBytes) {
    rgb_buf_.resize(kRgbBytes);
  }
  std::memcpy(rgb_buf_.data(), msg.data().data(), kRgbBytes);
  rgb_stamp_ns_ = ExtractStampNs(msg);
  has_rgb_ = true;
}

void GzEglSensorBridge::OnDepth(const gz::msgs::Image& msg) {
  if (msg.data().size() != kDepthPixels * sizeof(float)) {
    return;
  }
  std::scoped_lock lk(mtx_);
  if (depth_buf_.size() != kDepthPixels) {
    depth_buf_.resize(kDepthPixels);
  }
  std::memcpy(depth_buf_.data(), msg.data().data(), kDepthPixels * sizeof(float));
  depth_stamp_ns_ = ExtractStampNs(msg);
  has_depth_ = true;
}
#endif

}  // namespace vertirta::sim
