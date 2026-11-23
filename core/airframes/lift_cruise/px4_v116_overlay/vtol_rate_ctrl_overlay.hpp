// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vertirta::airframes::px4_v116 {

// Compile-time PX4 v1.16 internal rate-loop placement contract (01-DR08-13 Override #4 & Spec I5).
// 400 Hz Lift+Cruise INDI-WLS runs strictly inside PX4's `wq:rate_ctrl` work queue, receiving
// 100 Hz HJI-CBVF-filtered outer-loop acceleration/attitude setpoints over uXRCE-DDS.
struct VtolRateCtrlOverlayConfig {
  static constexpr const char* kWorkQueueName = "wq:rate_ctrl";
  static constexpr const char* kPx4Version = "v1.16";
  static constexpr const char* kGpsTopic = "vehicle_gps_position";
  static constexpr const char* kEscStatusTopic = "esc_status";
  static constexpr const char* kVtolStatusTopic = "vtol_vehicle_status";

  static constexpr int kUxrceDdsSynct = 1;
  static constexpr bool kOffboardDirectActuator400HzAllowed = false;
  static constexpr bool kRequireVtolAttControlPreflight = true;

  static constexpr uint16_t kInnerRateHz = 400u;
  static constexpr uint16_t kCompanionRtaHz = 100u;
  static constexpr float kVtArspBlendMps = 14.0f;
  static constexpr float kVtArspTransMps = 28.0f;
  static constexpr float kFwAirspdMinMps = 24.0f;
  static constexpr float kFwAirspdTrimMps = 38.0f;
  static constexpr float kFwAirspdMaxMps = 46.3f;
  static constexpr float kVtPsherSlew = 0.33f;
};

static_assert(!VtolRateCtrlOverlayConfig::kOffboardDirectActuator400HzAllowed,
              "400 Hz offboard direct_actuator over uXRCE-DDS is forbidden by 01-DR08-13 Override #4");
static_assert(VtolRateCtrlOverlayConfig::kUxrceDdsSynct == 1,
              "UXRCE_DDS_SYNCT must remain enabled (1) for PX4 v1.16 companion time synchronization");

}  // namespace vertirta::airframes::px4_v116
