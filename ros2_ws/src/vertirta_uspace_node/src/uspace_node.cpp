// SPDX-License-Identifier: Apache-2.0
#include "vertirta_uspace_node/uspace_core.hpp"

#include <cstdio>
#include <string>

#if defined(VERTIRTA_HAS_RCLCPP) && VERTIRTA_HAS_RCLCPP
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>

namespace vertirta::uspace_node {

class VertirtaUspaceRosNode : public rclcpp::Node {
 public:
  VertirtaUspaceRosNode() : Node("vertirta_uspace_node") {
    declare_parameter<std::string>("uas_id", "VRTA-X8-01");
    declare_parameter<double>("eastern_geofence_lon_deg", kEasternGeofenceLonDeg);
    uas_id_ = get_parameter("uas_id").as_string();

    (void)core_.create_oiv(
        uas_id_, RtaPriorityClass::PRIORITY_1_REVENUE, 40.9038, 29.1545, 70.0);
    (void)core_.transition_oiv(uas_id_, OivState::ACTIVATED, false);

    timer_ = create_wall_timer(
        std::chrono::seconds(1),
        [this]() { on_rid_timer(); });
    RCLCPP_INFO(get_logger(), "vertirta_uspace_node online (ASTM F3548-21 + F3411-22a 228 B DRIP)");
  }

 private:
  void on_rid_timer() {
    const bool emg = (core_.oiv_state(uas_id_) == OivState::CONTINGENT);
    const auto pack = encode_f3411_drip_message_pack(uas_id_, 40.9038, 29.1545, 70.0, emg);
    RCLCPP_DEBUG(get_logger(), "Encoded F3411-22a DRIP MessagePack (%zu bytes)", pack.size());
  }

  std::string uas_id_{};
  UspaceCore core_{};
  rclcpp::TimerBase::SharedPtr timer_{};
};

}  // namespace vertirta::uspace_node

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::uspace_node::VertirtaUspaceRosNode>());
  rclcpp::shutdown();
  return 0;
}
#else
int main() {
  using namespace vertirta::uspace_node;
  UspaceCore core{};
  const auto oiv = core.create_oiv("VRTA-X8-01", RtaPriorityClass::PRIORITY_1_REVENUE, 40.9038, 29.1545, 70.0);
  const auto pack = encode_f3411_drip_message_pack("VRTA-X8-01", 40.9038, 29.1545, 70.0, false);
  std::printf("vertirta_uspace_node standalone check: oiv=%s w84=%.2f m drip_bytes=%zu\n",
              oiv.id.c_str(), oiv.alt_w84_m, pack.size());
  return (pack.size() == 228U) ? 0 : 1;
}
#endif
