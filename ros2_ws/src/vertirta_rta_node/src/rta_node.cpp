#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <thread>

#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vertirta_capability_envelope.hpp>
#include <px4_msgs/msg/vertirta_rta_setpoint.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vertirta_msgs/msg/intruder_array.hpp>
#include <vertirta_msgs/msg/rta_state_snapshot.hpp>
#include <vertirta_msgs/msg/slz_prior_array.hpp>
#include <vertirta_msgs/msg/wire_capsule_array.hpp>

#include "vertirta_rta_node/lockfree_transport.hpp"

namespace vertirta::ros {

// 4-Thread Real-Time Architecture (DR-2 Section 5.1 / Plan 1A Task 12):
//   Thread 1: DDS I/O via SingleThreadedExecutor (MutuallyExclusive) <-> SPSC rings
//   Thread 2: SCHED_FIFO 100 Hz zero-ROS zero-heap RT safety loop <-> Seqlock + SPSC
//   Thread 3: 20 Hz perception/map seqlock publisher (subscribing /vertirta/slz_priors)
//   Thread 4: Low-priority async .vrta recorder draining rt_to_disk_ring
class VertiRtaNode : public rclcpp::Node {
 public:
  explicit VertiRtaNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : Node("vertirta_rta_node", options) {
    vrta_log_path_ = declare_parameter<std::string>("vrta_log_path", "/tmp/vertirta_live.vrta");
    auto cb_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = cb_group;

    auto sensor_qos = rclcpp::SensorDataQoS();

    odom_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(
        "/fmu/out/vehicle_odometry", sensor_qos,
        [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) { on_odometry(*msg); },
        sub_opts);

    env_sub_ = create_subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>(
        "/fmu/out/vertirta_capability_envelope", sensor_qos,
        [this](const px4_msgs::msg::VertirtaCapabilityEnvelope::SharedPtr msg) { on_envelope(*msg); },
        sub_opts);

    slz_sub_ = create_subscription<vertirta_msgs::msg::SlzPriorArray>(
        "/vertirta/slz_priors", 10,
        [this](const vertirta_msgs::msg::SlzPriorArray::SharedPtr msg) { on_slz_priors(*msg); },
        sub_opts);

    wire_sub_ = create_subscription<vertirta_msgs::msg::WireCapsuleArray>(
        "/vertirta/wire_capsules", 10,
        [this](const vertirta_msgs::msg::WireCapsuleArray::SharedPtr msg) { latest_wire_stamp_ns_ = msg->stamp_ns; },
        sub_opts);

    intruder_sub_ = create_subscription<vertirta_msgs::msg::IntruderArray>(
        "/vertirta/intruders", 10,
        [this](const vertirta_msgs::msg::IntruderArray::SharedPtr msg) { latest_intruder_stamp_ns_ = msg->stamp_ns; },
        sub_opts);

    // Single-publisher routing: publish internal command to VertiRtaMode (sole /fmu/in/* publisher)
    rta_sp_pub_ = create_publisher<px4_msgs::msg::VertirtaRtaSetpoint>(
        "/vertirta/internal/rta_cmd", sensor_qos);
    snapshot_pub_ = create_publisher<vertirta_msgs::msg::RtaStateSnapshot>(
        "/vertirta/rta_state_snapshot", 10);

    // Thread 1 egress pump (100 Hz) + 1 Hz / on-fault RtaStateSnapshot publisher
    egress_timer_ = create_wall_timer(
        std::chrono::milliseconds(10),
        [this]() { pump_egress_and_snapshot(); },
        cb_group);

    // Launch isolated Thread 2 (SCHED_FIFO 100 Hz), Thread 3 (20 Hz Map Seqlock), Thread 4 (Async .vrta)
    running_.store(true, std::memory_order_release);
    rt_thread_ = std::thread([this]() { run_rt_loop_100hz(); });
    map_thread_ = std::thread([this]() { run_map_sync_20hz(); });
    disk_thread_ = std::thread([this]() { run_async_vrta_writer(); });
  }

  ~VertiRtaNode() override {
    running_.store(false, std::memory_order_release);
    if (rt_thread_.joinable()) rt_thread_.join();
    if (map_thread_.joinable()) map_thread_.join();
    if (disk_thread_.joinable()) disk_thread_.join();
  }

 private:
  static constexpr uint64_t kStaleWarnNs = 250'000'000ULL;    // 250 ms staleness inflation
  static constexpr uint64_t kStaleCapNs = 1'000'000'000ULL;   // 1000 ms -> 8.0 m/s speed cap
  static constexpr uint64_t kStaleDivertNs = 2'000'000'000ULL; // 2000 ms -> divert flag
  static constexpr float kStaleSpeedCapMps = 8.0f;

  void on_odometry(const px4_msgs::msg::VehicleOdometry& msg) {
    latest_tel_.pos_ned[0] = static_cast<double>(msg.position[0]);
    latest_tel_.pos_ned[1] = static_cast<double>(msg.position[1]);
    latest_tel_.pos_ned[2] = static_cast<double>(msg.position[2]);
    latest_tel_.vel_ned[0] = msg.velocity[0];
    latest_tel_.vel_ned[1] = msg.velocity[1];
    latest_tel_.vel_ned[2] = msg.velocity[2];
    latest_tel_.q_frd_ned[0] = msg.q[0];
    latest_tel_.q_frd_ned[1] = msg.q[1];
    latest_tel_.q_frd_ned[2] = msg.q[2];
    latest_tel_.q_frd_ned[3] = msg.q[3];
    latest_tel_.stamp_ns = msg.timestamp * 1000ULL;
    latest_tel_.seq = ++ingress_seq_;
    (void)transport_.dds_to_rt_ring.try_push(latest_tel_);
  }

  void on_envelope(const px4_msgs::msg::VertirtaCapabilityEnvelope& msg) {
    latest_tel_.flight_regime = msg.flight_regime;
    latest_tel_.fault_bitmask = msg.fault_bitmask;
    latest_tel_.tmr_state = msg.tmr_state;
    latest_tel_.imu_health_mask = msg.imu_health_mask;
    latest_tel_.t_max_n = msg.t_max;
    latest_tel_.a_xy_max = msg.a_xy_max;
    latest_tel_.tilt_max_rad = msg.tilt_max;
    latest_tel_.vz_sink_max = msg.vz_sink_max;
    latest_tel_.v_term_v = msg.v_term;
    latest_tel_.i_pack_a = msg.i_pack;
    latest_tel_.soc = msg.soc;
    latest_tel_.chi2_residual = msg.chi2_residual;
    latest_p_hover_w_ = msg.p_hover_deg;
    latest_spin_ref_ = msg.spin_rate_ref;
    latest_spin_meas_ = msg.spin_rate_meas;
    latest_tel_.seq = ++ingress_seq_;
    (void)transport_.dds_to_rt_ring.try_push(latest_tel_);
  }

  void on_slz_priors(const vertirta_msgs::msg::SlzPriorArray& msg) {
    cached_slz_priors_ = msg;
    latest_slz_stamp_ns_.store(msg.stamp_ns, std::memory_order_release);
    latest_slz_count_.store(static_cast<uint8_t>(msg.priors.size()), std::memory_order_release);
    latest_slz_stale_.store(msg.stale, std::memory_order_release);
  }

  // Thread 2: 100 Hz SCHED_FIFO real-time loop — strictly ZERO ROS calls, ZERO heap allocations
  void run_rt_loop_100hz() noexcept {
    (void)try_configure_sched_fifo(80);
    TelemetryIngressFrame tel{};
    bool cel_latched = false;
    uint64_t rt_seq = 0;

    while (running_.load(std::memory_order_acquire)) {
      const auto tick_start = std::chrono::steady_clock::now();
      (void)transport_.dds_to_rt_ring.pop_latest(tel);
      const PerceptionHazardSnapshot haz = transport_.map_to_rt_seqlock.read();

      // Irreversible CEL latch on SAME_ARM_ACTIVE_SPIN (regime == 3)
      if (tel.flight_regime == 3U) {
        cel_latched = true;
      }

      // Guard unsigned timestamp subtraction against clock jitter (Review Focus #4)
      const uint64_t age_ns =
          (tel.stamp_ns >= haz.stamp_ns && haz.stamp_ns > 0ULL) ? (tel.stamp_ns - haz.stamp_ns) : 0ULL;
      const bool stale_warn = haz.stale || (age_ns > kStaleWarnNs);
      const bool stale_cap = (age_ns > kStaleCapNs);
      const bool stale_divert = (age_ns > kStaleDivertNs);

      // Compute safe bounded acceleration (capped to 8.0 m/s speed floor when stale > 1 s)
      RtaCommandEgressFrame cmd{};
      cmd.seq = ++rt_seq;
      cmd.stamp_ns = tel.stamp_ns;
      cmd.cel_latched = cel_latched;
      cmd.perception_stale = stale_warn;
      cmd.divert_requested = stale_divert;
      cmd.valid = true;

      if (cel_latched) {
        cmd.control_mode = px4_msgs::msg::VertirtaRtaSetpoint::MODE_CEL_CONTROLLED_LANDING;
        const float agl_m = static_cast<float>(-tel.pos_ned[2]);
        cmd.vz_limit_mps = (agl_m > 10.0f) ? 2.5f : ((agl_m > 5.0f) ? 1.5f : 0.8f);
        cmd.accel_ned = {0.0f, 0.0f, (tel.vel_ned[2] > cmd.vz_limit_mps) ? -1.5f : 0.0f};
        cmd.yawspeed_sp = -2.2f;
      } else {
        cmd.control_mode = px4_msgs::msg::VertirtaRtaSetpoint::MODE_ACCEL_NED;
        const float v_xy = std::hypot(tel.vel_ned[0], tel.vel_ned[1]);
        float ax = 0.0f;
        float ay = 0.0f;
        if (stale_cap && v_xy > kStaleSpeedCapMps) {
          const float scale = -1.5f * (v_xy - kStaleSpeedCapMps) / v_xy;
          ax = tel.vel_ned[0] * scale;
          ay = tel.vel_ned[1] * scale;
        }
        cmd.accel_ned = {ax, ay, 0.0f};
        cmd.vz_limit_mps = tel.vz_sink_max;
      }

      (void)transport_.rt_to_dds_ring.try_push(cmd);

      VrtaLogRecordSlot rec{};
      rec.seq = cmd.seq;
      rec.t_sim_us = tel.stamp_ns / 1000ULL;
      rec.pos_enu[0] = tel.pos_ned[1];
      rec.pos_enu[1] = tel.pos_ned[0];
      rec.pos_enu[2] = -tel.pos_ned[2];
      rec.vel_enu[0] = tel.vel_ned[1];
      rec.vel_enu[1] = tel.vel_ned[0];
      rec.vel_enu[2] = -tel.vel_ned[2];
      rec.v_term_v = tel.v_term_v;
      rec.soc_pct = tel.soc * 100.0f;
      rec.chi2_residual = tel.chi2_residual;
      rec.agl_m = static_cast<float>(-tel.pos_ned[2]);
      rec.active_motor_mask = static_cast<uint8_t>(~tel.fault_bitmask);
      rec.flight_regime = tel.flight_regime;
      rec.tmr_state = tel.tmr_state;
      rec.cel_latched = cel_latched ? 1U : 0U;
      (void)transport_.rt_to_disk_ring.try_push(rec);

      std::this_thread::sleep_until(tick_start + std::chrono::milliseconds(10));
    }
  }

  // Thread 3: 20 Hz perception/ESDF Seqlock publisher (isolated from RT loop)
  void run_map_sync_20hz() noexcept {
    uint64_t ver = 0;
    while (running_.load(std::memory_order_acquire)) {
      PerceptionHazardSnapshot snap{};
      snap.version = ++ver;
      snap.stamp_ns = latest_slz_stamp_ns_.load(std::memory_order_acquire);
      snap.slz_count = latest_slz_count_.load(std::memory_order_acquire);
      snap.stale = latest_slz_stale_.load(std::memory_order_acquire);
      transport_.map_to_rt_seqlock.publish(snap);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  // RFC 3720 Castagnoli CRC32C (0x82F63B78) matching vrta_recorder.hpp & vrta_reader.py
  static uint32_t crc32c(const uint8_t* data, std::size_t len) noexcept {
    uint32_t crc = 0xFFFFFFFFU;
    for (std::size_t i = 0U; i < len; ++i) {
      crc ^= data[i];
      for (int bit = 0; bit < 8; ++bit) {
        crc = (crc & 1U) ? ((crc >> 1U) ^ 0x82F63B78U) : (crc >> 1U);
      }
    }
    return ~crc;
  }

  // Thread 4: Low-priority async canonical VRTA0001 (128 B header + 256 B CRC32C frames) writer
  void run_async_vrta_writer() {
    struct VrtaHdr128 {
      char magic[8];
      uint32_t schema_ver;
      uint32_t reserved0;
      double datum_lat_deg;
      double datum_lon_deg;
      double datum_amsl_m;
      double egm2008_n_m;
      uint8_t lut_sha256[32];
      char git_sha[40];
      uint8_t reserved1[4];
      uint32_t header_crc32c;
    };
    struct VrtaFrm256 {
      uint64_t seq;
      uint64_t t_sim_us;
      double pos_enu[3];
      float vel_enu[3];
      float q_frd_ned[4];
      float omega_frd[3];
      float a_nom_enu[3];
      float a_safe_enu[3];
      float rotor_rpm[8];
      float v_term_v;
      float soc_pct;
      float cbvf_min_v;
      float ltfj_margin_m;
      float ltbx_margin_m;
      float chi2_residual;
      float agl_m;
      uint8_t active_motor_mask;
      uint8_t flight_regime;
      uint8_t tmr_state;
      uint8_t cel_latched;
      uint32_t flags;
      uint8_t pad[80];
      uint32_t crc32c;
    };
    static_assert(sizeof(VrtaHdr128) == 128U, "Canonical VRTA header must be 128 B");
    static_assert(sizeof(VrtaFrm256) == 256U, "Canonical VRTA frame must be 256 B");

    std::ofstream ofs(vrta_log_path_, std::ios::binary | std::ios::trunc);
    if (ofs.is_open()) {
      VrtaHdr128 hdr{};
      std::memcpy(hdr.magic, "VRTA0001", 8U);
      hdr.schema_ver = 1U;
      hdr.datum_lat_deg = 40.8860;
      hdr.datum_lon_deg = 29.1880;
      hdr.datum_amsl_m = 12.0;
      hdr.egm2008_n_m = 36.7;
      hdr.header_crc32c = crc32c(reinterpret_cast<const uint8_t*>(&hdr), 124U);
      ofs.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    }

    VrtaLogRecordSlot slot{};
    while (running_.load(std::memory_order_acquire) || !transport_.rt_to_disk_ring.empty()) {
      if (transport_.rt_to_disk_ring.try_pop(slot)) {
        if (ofs.is_open()) {
          VrtaFrm256 frm{};
          frm.seq = slot.seq;
          frm.t_sim_us = slot.t_sim_us;
          frm.pos_enu[0] = slot.pos_enu[0];
          frm.pos_enu[1] = slot.pos_enu[1];
          frm.pos_enu[2] = slot.pos_enu[2];
          frm.vel_enu[0] = slot.vel_enu[0];
          frm.vel_enu[1] = slot.vel_enu[1];
          frm.vel_enu[2] = slot.vel_enu[2];
          frm.q_frd_ned[0] = 1.0f;
          frm.a_safe_enu[0] = slot.a_safe_enu[0];
          frm.a_safe_enu[1] = slot.a_safe_enu[1];
          frm.a_safe_enu[2] = slot.a_safe_enu[2];
          frm.v_term_v = slot.v_term_v;
          frm.soc_pct = slot.soc_pct;
          frm.cbvf_min_v = slot.cbvf_min_v;
          frm.ltfj_margin_m = slot.ltfj_margin_m;
          frm.ltbx_margin_m = slot.ltbx_margin_m;
          frm.chi2_residual = slot.chi2_residual;
          frm.agl_m = slot.agl_m;
          frm.active_motor_mask = slot.active_motor_mask;
          frm.flight_regime = slot.flight_regime;
          frm.tmr_state = slot.tmr_state;
          frm.cel_latched = slot.cel_latched;
          frm.crc32c = crc32c(reinterpret_cast<const uint8_t*>(&frm), 252U);
          ofs.write(reinterpret_cast<const char*>(&frm), sizeof(frm));
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }
  }

  // Thread 1 DDS egress (100 Hz setpoint + 1 Hz / on-fault 400 V / 22 kWh RtaStateSnapshot)
  void pump_egress_and_snapshot() {
    RtaCommandEgressFrame cmd{};
    if (!transport_.rt_to_dds_ring.pop_latest(cmd)) {
      return;
    }

    px4_msgs::msg::VertirtaRtaSetpoint sp{};
    sp.timestamp = now().nanoseconds() / 1000ULL;
    sp.timestamp_sample = cmd.stamp_ns / 1000ULL;
    sp.accel_ned = {cmd.accel_ned[0], cmd.accel_ned[1], cmd.accel_ned[2]};
    sp.thrust_des_n = cmd.thrust_des_n;
    sp.yaw_sp = cmd.yaw_sp;
    sp.yawspeed_sp = cmd.yawspeed_sp;
    sp.vz_limit_mps = cmd.vz_limit_mps;
    sp.control_mode = cmd.control_mode;
    sp.cel_latched = cmd.cel_latched ? 1U : 0U;
    sp.valid = cmd.valid ? 1U : 0U;
    sp.seq = static_cast<uint32_t>(cmd.seq);
    sp.reserved_pad = 0U;
    rta_sp_pub_->publish(sp);

    ++snapshot_decim_;
    const bool fault_changed = (latest_tel_.fault_bitmask != prev_snapshot_fault_mask_) ||
                               (cmd.cel_latched != prev_snapshot_cel_);
    if ((snapshot_decim_ % 100U) == 0U || fault_changed) {
      prev_snapshot_fault_mask_ = latest_tel_.fault_bitmask;
      prev_snapshot_cel_ = cmd.cel_latched;

      vertirta_msgs::msg::RtaStateSnapshot snap{};
      snap.header.stamp = now();
      snap.header.frame_id = "kartal_enu";
      snap.stamp_ns = static_cast<uint64_t>(now().nanoseconds());
      snap.flight_regime = latest_tel_.flight_regime;
      snap.fault_bitmask = latest_tel_.fault_bitmask;
      snap.tmr_state = latest_tel_.tmr_state;
      snap.imu_health_mask = latest_tel_.imu_health_mask;
      snap.cel_latched = cmd.cel_latched;
      snap.shield_active = true;  // Immutable true in flight (EASA AI Level 1/2A lock)
      snap.pos_enu.x = latest_tel_.pos_ned[1];
      snap.pos_enu.y = latest_tel_.pos_ned[0];
      snap.pos_enu.z = -latest_tel_.pos_ned[2];
      snap.vel_enu.x = latest_tel_.vel_ned[1];
      snap.vel_enu.y = latest_tel_.vel_ned[0];
      snap.vel_enu.z = -latest_tel_.vel_ned[2];
      snap.agl_m = static_cast<float>(-latest_tel_.pos_ned[2]);

      // Locked 400 V / 22 kWh HV pack telemetry contract (Plan 1A + Plan 2)
      snap.pack_nominal_volts = 400.0f;
      snap.pack_energy_kwh = 22.0f;
      snap.v_term_volts = latest_tel_.v_term_v;
      snap.i_pack_amps = latest_tel_.i_pack_a;
      snap.soc_pct = latest_tel_.soc * 100.0f;
      snap.soc_reserve_floor_pct = 12.0f;

      snap.t_max_n = latest_tel_.t_max_n;
      snap.a_xy_max_mps2 = latest_tel_.a_xy_max;
      snap.tilt_max_rad = latest_tel_.tilt_max_rad;
      snap.vz_sink_max_mps = latest_tel_.vz_sink_max;
      snap.p_hover_deg_w = latest_p_hover_w_;
      snap.spin_rate_ref_rad_s = latest_spin_ref_;
      snap.spin_rate_meas_rad_s = latest_spin_meas_;
      snap.chi2_residual = latest_tel_.chi2_residual;
      snap.cbvf_min_margin_m = cmd.cbvf_min_margin_m;
      snap.ltfj_margin_m = cmd.ltfj_margin_m;
      snap.ltbx_margin_m = cmd.ltbx_margin_m;
      snap.perception_stale = cmd.perception_stale;
      snap.slz_priors = cached_slz_priors_.priors;
      snapshot_pub_->publish(snap);
    }
  }

  FourThreadRtaTransport transport_{};
  std::atomic<bool> running_{false};
  std::thread rt_thread_;
  std::thread map_thread_;
  std::thread disk_thread_;

  std::string vrta_log_path_;
  TelemetryIngressFrame latest_tel_{};
  uint64_t ingress_seq_{0};
  uint64_t latest_wire_stamp_ns_{0};
  uint64_t latest_intruder_stamp_ns_{0};
  std::atomic<uint64_t> latest_slz_stamp_ns_{0};
  std::atomic<uint8_t> latest_slz_count_{0};
  std::atomic<bool> latest_slz_stale_{false};
  float latest_p_hover_w_{36230.0f};
  float latest_spin_ref_{0.0f};
  float latest_spin_meas_{0.0f};
  uint32_t snapshot_decim_{0};
  uint8_t prev_snapshot_fault_mask_{0};
  bool prev_snapshot_cel_{false};
  vertirta_msgs::msg::SlzPriorArray cached_slz_priors_{};

  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::VertirtaCapabilityEnvelope>::SharedPtr env_sub_;
  rclcpp::Subscription<vertirta_msgs::msg::SlzPriorArray>::SharedPtr slz_sub_;
  rclcpp::Subscription<vertirta_msgs::msg::WireCapsuleArray>::SharedPtr wire_sub_;
  rclcpp::Subscription<vertirta_msgs::msg::IntruderArray>::SharedPtr intruder_sub_;
  rclcpp::Publisher<px4_msgs::msg::VertirtaRtaSetpoint>::SharedPtr rta_sp_pub_;
  rclcpp::Publisher<vertirta_msgs::msg::RtaStateSnapshot>::SharedPtr snapshot_pub_;
  rclcpp::TimerBase::SharedPtr egress_timer_;
};

}  // namespace vertirta::ros

int main(int argc, char* argv[]) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<vertirta::ros::VertiRtaNode>();
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
