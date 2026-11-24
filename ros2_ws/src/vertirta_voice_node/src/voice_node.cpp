// SPDX-License-Identifier: Apache-2.0
#include "vertirta_voice_node/voice_node_core.hpp"

#include <chrono>
#include <iostream>
#include <memory>

#if defined(VERTIRTA_WITH_ROS2) && VERTIRTA_WITH_ROS2
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <vertirta_msgs/msg/rta_state_snapshot.hpp>

namespace vertirta::voice_node {

class VoiceAtcRosNode final : public rclcpp::Node {
 public:
  VoiceAtcRosNode() : Node("vertirta_voice_node") {
    VoiceNodeConfig cfg{};
    cfg.mayday_weight_min = static_cast<float>(this->declare_parameter<double>("mayday_weight_min", 0.15));
    cfg.liveness_emergency_soc_pct =
        static_cast<float>(this->declare_parameter<double>("liveness_emergency_soc_pct", 5.0));
    cfg.tts_ttfa_budget_ms = this->declare_parameter<double>("tts_ttfa_budget_ms", 250.0);
    core_ = std::make_unique<VoiceNodeCore>(cfg);
    core_->start_worker();

    sub_audio_ = this->create_subscription<std_msgs::msg::String>(
        kTopicAudio16kHz, 10,
        [this](const std_msgs::msg::String::SharedPtr msg) { on_audio_turn(*msg); });
    sub_snapshot_ = this->create_subscription<vertirta_msgs::msg::RtaStateSnapshot>(
        kTopicRtaStateSnapshot, rclcpp::SensorDataQoS(),
        [this](const vertirta_msgs::msg::RtaStateSnapshot::SharedPtr msg) { on_snapshot(*msg); });

    pub_dsl_ = this->create_publisher<std_msgs::msg::String>(kTopicVoiceDslCommand, 10);
    pub_tts_ = this->create_publisher<std_msgs::msg::String>(kTopicVoiceTtsReadback, 10);
    pub_sidecar_ = this->create_publisher<std_msgs::msg::String>(kTopicForensicsSidecarEvent, 10);

    timer_ = this->create_wall_timer(std::chrono::milliseconds(20), [this]() { drain_outcomes(); });
  }

  ~VoiceAtcRosNode() override {
    if (core_) {
      core_->stop_worker();
    }
  }

 private:
  void on_snapshot(const vertirta_msgs::msg::RtaStateSnapshot& msg) {
    auto snap = vertirta::copilot::make_reference_400v_sml_snapshot();
    snap.seq_id = msg.stamp_ns / 10'000'000ULL;
    snap.timestamp_ms = msg.stamp_ns / 1'000'000ULL;
    snap.agl_m = msg.agl_m;
    snap.soc_pct = msg.soc_pct;
    snap.v_term_volts = msg.v_term_volts;
    snap.shield_active = msg.shield_active;
    core_->submit_rta_snapshot(snap);
  }

  void on_audio_turn(const std_msgs::msg::String& msg) {
    AudioTurnRequest req{};
    req.turn_id = ++turn_seq_;
    req.snr_db = 14.0f;
    req.vad_speech_prob = 0.92f;
    req.audio_rms = 0.20f;
    req.transcript_hint = msg.data;
    req.dsl_candidates[0] = {
        "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", 0.94f};
    req.candidate_count = 1U;
    (void)core_->try_enqueue_audio_turn(std::move(req));
  }

  void drain_outcomes() {
    VoiceTurnOutcome out{};
    while (core_->try_pop_outcome(out)) {
      if (out.dsl_dispatched) {
        std_msgs::msg::String dsl{};
        dsl.data = out.dispatched_dsl;
        pub_dsl_->publish(dsl);
      }
      if (!out.tts_readback_utf8.empty()) {
        std_msgs::msg::String tts{};
        tts.data = out.tts_readback_utf8;
        pub_tts_->publish(tts);
      }
      if (!out.sidecar_jsonl.empty()) {
        std_msgs::msg::String ev{};
        ev.data = out.sidecar_jsonl;
        pub_sidecar_->publish(ev);
      }
    }
  }

  std::unique_ptr<VoiceNodeCore> core_{};
  uint32_t turn_seq_{0U};
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_audio_{};
  rclcpp::Subscription<vertirta_msgs::msg::RtaStateSnapshot>::SharedPtr sub_snapshot_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_dsl_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_tts_{};
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_sidecar_{};
  rclcpp::TimerBase::SharedPtr timer_{};
};

}  // namespace vertirta::voice_node

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<vertirta::voice_node::VoiceAtcRosNode>());
  rclcpp::shutdown();
  return 0;
}
#else
int main() {
  using namespace vertirta::voice_node;
  VoiceNodeCore core{};
  AudioTurnRequest req{};
  req.turn_id = 1U;
  req.snr_db = 16.0f;
  req.vad_speech_prob = 0.95f;
  req.audio_rms = 0.25f;
  req.transcript_hint = "Kartal Sahil acil inis";
  req.dsl_candidates[0] = {
      "ACT=DIVERT_SLZ;TGT=Kartal_Sahil;ETA=120;AGL=35,120;SPD=18\nLNG=TR;MOD=BRIEF", 0.95f};
  req.candidate_count = 1U;
  const auto out = core.process_turn(req);
  std::cout << out.sidecar_jsonl << "\n";
  return out.dsl_dispatched ? 0 : 1;
}
#endif
