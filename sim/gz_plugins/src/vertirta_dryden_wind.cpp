#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/vector3d.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

#include "vertirta_gz/dryden_wind_kernel.hpp"

namespace vertirta_gz {

class VertirtaDrydenWindSystem
    : public gz::sim::System,
      public gz::sim::ISystemConfigure,
      public gz::sim::ISystemPreUpdate {
 public:
  void Configure(
      const gz::sim::Entity& /*entity*/,
      const std::shared_ptr<const sdf::Element>& sdf,
      gz::sim::EntityComponentManager& /*ecm*/,
      gz::sim::EventManager& /*eventMgr*/) override {
    uint64_t seed = 42ULL;
    if (sdf) {
      if (sdf->HasElement("w20_mps")) w20_mps_ = sdf->Get<double>("w20_mps");
      if (sdf->HasElement("mean_wind_mps")) mean_wind_mps_ = sdf->Get<double>("mean_wind_mps");
      if (sdf->HasElement("mean_wind_dir_deg")) mean_wind_dir_deg_ = sdf->Get<double>("mean_wind_dir_deg");
      if (sdf->HasElement("seed")) seed = sdf->Get<uint64_t>("seed");
      if (sdf->HasElement("cd_a_xy_m2")) cd_a_xy_m2_ = sdf->Get<double>("cd_a_xy_m2");
      if (sdf->HasElement("cd_a_z_m2")) cd_a_z_m2_ = sdf->Get<double>("cd_a_z_m2");
      if (sdf->HasElement("target_model")) target_model_name_ = sdf->Get<std::string>("target_model");
      if (sdf->HasElement("target_link")) target_link_name_ = sdf->Get<std::string>("target_link");
      if (sdf->HasElement("wind_topic")) wind_topic_ = sdf->Get<std::string>("wind_topic");
    }
    kernel_.reseed(seed);
    wind_pub_ = node_.Advertise<gz::msgs::Vector3d>(wind_topic_);
  }

  void PreUpdate(
      const gz::sim::UpdateInfo& info,
      gz::sim::EntityComponentManager& ecm) override {
    if (info.paused) {
      return;
    }
    const double dt = std::chrono::duration<double>(info.dt).count();
    if (dt <= 0.0) {
      return;
    }

    if (target_link_entity_ == gz::sim::kNullEntity) {
      const gz::sim::Entity model_ent =
          ecm.EntityByComponents(gz::sim::components::Name(target_model_name_));
      if (model_ent != gz::sim::kNullEntity) {
        gz::sim::Model m(model_ent);
        target_link_entity_ = m.LinkByName(ecm, target_link_name_);
        if (target_link_entity_ != gz::sim::kNullEntity) {
          // Required in gz-sim8 so Link::WorldLinearVelocity(ecm) is populated at runtime
          gz::sim::Link(target_link_entity_).EnableVelocityChecks(ecm, true);
        }
      }
    }

    double h_agl = 30.0;
    gz::math::Vector3d v_body_enu{0.0, 0.0, 0.0};
    if (target_link_entity_ != gz::sim::kNullEntity) {
      const gz::math::Pose3d pose = gz::sim::worldPose(target_link_entity_, ecm);
      h_agl = std::max(3.0, pose.Pos().Z());
      const gz::sim::Link target_link(target_link_entity_);
      if (const auto vel_opt = target_link.WorldLinearVelocity(ecm); vel_opt.has_value()) {
        v_body_enu = *vel_opt;
      }
    }

    const double rad = mean_wind_dir_deg_ * (kPi / 180.0);
    const Vec3D mean_wind_enu{mean_wind_mps_ * std::cos(rad), mean_wind_mps_ * std::sin(rad), 0.0};
    const double v_air = std::hypot(
        v_body_enu.X() - mean_wind_enu.x,
        v_body_enu.Y() - mean_wind_enu.y);

    const Vec3D gust_enu = kernel_.step_seeded(dt, h_agl, v_air, w20_mps_);
    const Vec3D total_wind_enu{
        mean_wind_enu.x + gust_enu.x,
        mean_wind_enu.y + gust_enu.y,
        mean_wind_enu.z + gust_enu.z};

    // Publish world ENU wind for coaxial propulsion plugin airspeed calculation
    gz::msgs::Vector3d wmsg;
    wmsg.set_x(total_wind_enu.x);
    wmsg.set_y(total_wind_enu.y);
    wmsg.set_z(total_wind_enu.z);
    wind_pub_.Publish(wmsg);

    if (target_link_entity_ != gz::sim::kNullEntity) {
      const Vec3D v_rel_enu{
          total_wind_enu.x - v_body_enu.X(),
          total_wind_enu.y - v_body_enu.Y(),
          total_wind_enu.z - v_body_enu.Z()};
      const Vec3D f_drag = DrydenWindKernel::compute_fuselage_drag_force_enu(
          v_rel_enu, cd_a_xy_m2_, cd_a_z_m2_);
      gz::sim::Link link(target_link_entity_);
      link.AddWorldForce(ecm, gz::math::Vector3d(f_drag.x, f_drag.y, f_drag.z));
    }
  }

 private:
  DrydenWindKernel kernel_{42ULL};
  double w20_mps_{7.72};             // Light coastal baseline (15 kt); override to 15.43 / 23.15
  double mean_wind_mps_{4.0};        // [m/s] Marmara steady coastal breeze (no inverted urban shear)
  double mean_wind_dir_deg_{215.0};  // SW Lodos / NE Poyraz configurable direction
  double cd_a_xy_m2_{0.48};
  double cd_a_z_m2_{1.20};

  std::string target_model_name_{"vertirta_x8_240"};
  std::string target_link_name_{"base_link"};
  std::string wind_topic_{"/vertirta/wind_world_enu"};
  gz::sim::Entity target_link_entity_{gz::sim::kNullEntity};

  gz::transport::Node node_{};
  gz::transport::Node::Publisher wind_pub_{};
};

}  // namespace vertirta_gz

GZ_ADD_PLUGIN(
    vertirta_gz::VertirtaDrydenWindSystem,
    gz::sim::System,
    vertirta_gz::VertirtaDrydenWindSystem::ISystemConfigure,
    vertirta_gz::VertirtaDrydenWindSystem::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(
    vertirta_gz::VertirtaDrydenWindSystem,
    "vertirta_gz::VertirtaDrydenWindSystem")
