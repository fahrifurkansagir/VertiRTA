#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/actuators.pb.h>
#include <gz/msgs/uint32.pb.h>
#include <gz/msgs/vector3d.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/transport/Node.hh>

#include "vertirta_gz/coaxial_physics_kernel.hpp"

namespace vertirta_gz {

class VertirtaCoaxialPropulsionSystem
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate {
 public:
  void Configure(
      const gz::sim::Entity& entity,
      const std::shared_ptr<const sdf::Element>& sdf,
      gz::sim::EntityComponentManager& ecm,
      gz::sim::EventManager& /*eventMgr*/) override {
    model_ = gz::sim::Model(entity);
    if (!model_.Valid(ecm)) {
      return;
    }

    CoaxParams p{};
    if (sdf) {
      if (sdf->HasElement("k_t0")) p.k_t0 = sdf->Get<double>("k_t0");
      if (sdf->HasElement("k_q0")) p.k_q0 = sdf->Get<double>("k_q0");
      if (sdf->HasElement("eta_coax_hover")) p.eta_coax_hover = sdf->Get<double>("eta_coax_hover");
      if (sdf->HasElement("swirl_recovery")) p.swirl_recovery = sdf->Get<double>("swirl_recovery");
      if (sdf->HasElement("j_r")) p.j_r = sdf->Get<double>("j_r");
      if (sdf->HasElement("rotor_radius")) p.rotor_radius_m = sdf->Get<double>("rotor_radius");
      if (sdf->HasElement("coax_spacing")) p.coax_spacing_m = sdf->Get<double>("coax_spacing");
      if (sdf->HasElement("tau_up")) p.tau_up_s = sdf->Get<double>("tau_up");
      if (sdf->HasElement("tau_down")) p.tau_down_s = sdf->Get<double>("tau_down");
      if (sdf->HasElement("tau_coast")) p.tau_coast_s = sdf->Get<double>("tau_coast");
      if (sdf->HasElement("alpha_max")) p.alpha_max_rad_s2 = sdf->Get<double>("alpha_max");
      if (sdf->HasElement("omega_max")) p.omega_max_nom = sdf->Get<double>("omega_max");
      if (sdf->HasElement("ke_bus")) p.ke_bus_vs_rad = sdf->Get<double>("ke_bus");
      if (sdf->HasElement("r_eff")) p.r_eff_hot_ohm = sdf->Get<double>("r_eff");
      if (sdf->HasElement("command_topic")) command_topic_ = sdf->Get<std::string>("command_topic");
      if (sdf->HasElement("feedback_topic")) feedback_topic_ = sdf->Get<std::string>("feedback_topic");
      if (sdf->HasElement("wind_topic")) wind_topic_ = sdf->Get<std::string>("wind_topic");
      if (sdf->HasElement("fault_mask_topic")) fault_mask_topic_ = sdf->Get<std::string>("fault_mask_topic");
    }
    kernel_.set_params(p);
    kernel_.reset();

    base_link_entity_ = model_.LinkByName(ecm, "base_link");
    if (base_link_entity_ != gz::sim::kNullEntity) {
      // Required in gz-sim8 so Link::WorldLinearVelocity(ecm) is populated at runtime
      gz::sim::Link(base_link_entity_).EnableVelocityChecks(ecm, true);
    }
    for (int i = 0; i < 8; ++i) {
      rotor_links_[i] = model_.LinkByName(ecm, "rotor_" + std::to_string(i));
      rotor_joints_[i] = model_.JointByName(ecm, "rotor_" + std::to_string(i) + "_joint");
    }

    node_.Subscribe(command_topic_, &VertirtaCoaxialPropulsionSystem::on_actuator_command, this);
    node_.Subscribe(wind_topic_, &VertirtaCoaxialPropulsionSystem::on_wind_world, this);
    node_.Subscribe(fault_mask_topic_, &VertirtaCoaxialPropulsionSystem::on_fault_mask, this);
    feedback_pub_ = node_.Advertise<gz::msgs::Actuators>(feedback_topic_);
  }

  void PreUpdate(
      const gz::sim::UpdateInfo& info,
      gz::sim::EntityComponentManager& ecm) override {
    if (info.paused || !model_.Valid(ecm)) {
      return;
    }
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0.0) {
      return;
    }

    std::array<double, 8> cmds{};
    gz::math::Vector3d wind_world{0.0, 0.0, 0.0};
    uint32_t fault_mask = 0;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      cmds = latest_cmds_;
      wind_world = latest_wind_world_;
      fault_mask = latest_fault_mask_;
    }

    kernel_.set_motor_commands(cmds);
    kernel_.set_motor_fault_mask(fault_mask);

    // Compute body FLU airspeed: v_air_flu = R_{world->body} * (v_body_world - v_wind_world)
    Vec3D v_air_flu{0.0, 0.0, 0.0};
    if (base_link_entity_ != gz::sim::kNullEntity) {
      const gz::math::Pose3d base_pose = gz::sim::worldPose(base_link_entity_, ecm);
      const gz::sim::Link base_link(base_link_entity_);
      const auto vel_opt = base_link.WorldLinearVelocity(ecm);
      const gz::math::Vector3d v_body_world = vel_opt.has_value() ? *vel_opt : gz::math::Vector3d::Zero;
      const gz::math::Vector3d v_rel_world = v_body_world - wind_world;
      const gz::math::Vector3d v_body = base_pose.Rot().RotateVectorReverse(v_rel_world);
      v_air_flu = {v_body.X(), v_body.Y(), v_body.Z()};
    }

    const CoaxStepOutput out = kernel_.step(dt, v_air_flu);

    // Apply physical thrust (+Z_FLU) and reaction torque (Q_z_FLU) to each rotor link
    for (int i = 0; i < 8; ++i) {
      if (rotor_links_[i] != gz::sim::kNullEntity) {
        gz::sim::Link link(rotor_links_[i]);
        const gz::math::Pose3d link_pose = gz::sim::worldPose(rotor_links_[i], ecm);
        const gz::math::Quaterniond q_wb = link_pose.Rot();
        const gz::math::Vector3d f_world = q_wb.RotateVector(gz::math::Vector3d(0.0, 0.0, out.thrust_n[i]));
        const gz::math::Vector3d tau_world = q_wb.RotateVector(gz::math::Vector3d(0.0, 0.0, out.torque_z_flu[i]));
        link.AddWorldWrench(ecm, f_world, tau_world);
      }
      if (rotor_joints_[i] != gz::sim::kNullEntity) {
        const double vis_vel = static_cast<double>(kernel_.rotor(i).spin_sign) * out.omega_rad_s[i] / 10.0;
        if (auto* cmd = ecm.Component<gz::sim::components::JointVelocityCmd>(rotor_joints_[i])) {
          cmd->Data() = {vis_vel};
        } else {
          ecm.CreateComponent(rotor_joints_[i], gz::sim::components::JointVelocityCmd({vis_vel}));
        }
      }
    }

    // Publish motor speed feedback for PX4 GZMixingInterfaceESC -> esc_status
    gz::msgs::Actuators fb_msg;
    for (int i = 0; i < 8; ++i) {
      fb_msg.add_velocity(out.omega_rad_s[i]);
    }
    feedback_pub_.Publish(fb_msg);
  }

 private:
  void on_actuator_command(const gz::msgs::Actuators& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    const int n = std::min(8, msg.velocity_size());
    for (int i = 0; i < n; ++i) {
      latest_cmds_[i] = msg.velocity(i);
    }
  }

  void on_wind_world(const gz::msgs::Vector3d& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_wind_world_.Set(msg.x(), msg.y(), msg.z());
  }

  void on_fault_mask(const gz::msgs::UInt32& msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    latest_fault_mask_ = msg.data();
  }

  gz::sim::Model model_{gz::sim::kNullEntity};
  gz::sim::Entity base_link_entity_{gz::sim::kNullEntity};
  std::array<gz::sim::Entity, 8> rotor_links_{};
  std::array<gz::sim::Entity, 8> rotor_joints_{};

  CoaxialPhysicsKernel kernel_{};
  gz::transport::Node node_{};
  gz::transport::Node::Publisher feedback_pub_{};

  std::string command_topic_{"/model/vertirta_x8_240/command/motor_speed"};
  std::string feedback_topic_{"/model/vertirta_x8_240/telemetry/motor_speed"};
  std::string wind_topic_{"/vertirta/wind_world_enu"};
  std::string fault_mask_topic_{"/vertirta/fault/motor_mask"};

  std::mutex mutex_{};
  std::array<double, 8> latest_cmds_{};
  gz::math::Vector3d latest_wind_world_{0.0, 0.0, 0.0};
  uint32_t latest_fault_mask_{0};
};

}  // namespace vertirta_gz

GZ_ADD_PLUGIN(
    vertirta_gz::VertirtaCoaxialPropulsionSystem,
    gz::sim::System,
    vertirta_gz::VertirtaCoaxialPropulsionSystem::ISystemConfigure,
    vertirta_gz::VertirtaCoaxialPropulsionSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
    vertirta_gz::VertirtaCoaxialPropulsionSystem,
    "vertirta_gz::VertirtaCoaxialPropulsionSystem")
