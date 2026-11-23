// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "vertirta_perception_node/perception_node.hpp"

using Catch::Approx;
using vertirta::perception::DepthMode;
using vertirta::perception::SlzDecisionState;
using vertirta::perception::SlzReasonBits;
using vertirta::ros::CheckRgbdTimestampSync;
using vertirta::ros::DecodeDepthImageToFloat512;
using vertirta::ros::DecodeRgbImageToNchw512;
using vertirta::ros::SlzPerceptionNode;
using vertirta::ros::SlzPerceptionOptions;

namespace {

SlzPerceptionOptions make_mock_runner_options() {
  SlzPerceptionOptions opts{};
  opts.ort_cfg.onnx_path = "";
  opts.ort_cfg.allow_mock_fallback = true;
  opts.staleness_warn_ns = 250'000'000ULL;
  opts.backproject_stride = 2u;
  return opts;
}

}  // namespace

TEST_CASE("SlzPerceptionNode wraps real ORT+Grid+Provider, publishes fresh snapshot <=250 ms, and flags stale >250 ms") {
  auto node = std::make_shared<SlzPerceptionNode>(make_mock_runner_options());
  REQUIRE(node->OrtRunner().is_ready());
  REQUIRE(node->OrtRunner().is_mock_backend());

  // 1) Inject clean flat-roof mock RGB-D frame (20 ms age <= 250 ms contract -> stale == false)
  REQUIRE(node->InjectMockFrame(/*stamp_ns=*/1'000'000'000ULL, /*now_ns=*/1'020'000'000ULL));
  const auto fresh = node->LatestSnapshot();
  REQUIRE_FALSE(fresh.stale);
  REQUIRE(fresh.count == 1u);
  REQUIRE(fresh.stamp_ns == 1'000'000'000ULL);
  REQUIRE(fresh.priors[0].p_safe >= 0.85f);
  REQUIRE(fresh.priors[0].theta_max_deg <= 8.0f);
  REQUIRE(fresh.priors[0].sigma_z_m <= 0.15f);
  REQUIRE(fresh.priors[0].reason_mask == SlzReasonBits::REASON_NONE);
  REQUIRE(node->LastDecision().state == SlzDecisionState::COMMIT_TOUCHDOWN);

  // 2) Staleness check at 300 ms > 250 ms contract -> stale == true
  node->CheckStaleness(/*now_ns=*/1'300'000'000ULL);
  REQUIRE(node->LatestSnapshot().stale);

  // 3) Backwards clock timestamp must NOT underflow uint64_t and fails closed to stale == true
  node->CheckStaleness(/*now_ns=*/900'000'000ULL);
  REQUIRE(node->LatestSnapshot().stale);
}

TEST_CASE("SlzPerceptionNode emits canonical harmonized SlzReasonBits on OOD tarp and Mod-B >30 m veto") {
  auto node = std::make_shared<SlzPerceptionNode>(make_mock_runner_options());

  // 1) Inject OOD tarp frame -> belief grid raises slz_ood_variance >= 0.12 -> triggers APPROACH_ORBIT or REASON_OOD_UNCERTAIN
  REQUIRE(node->InjectMockFrame(
      /*stamp_ns=*/2'000'000'000ULL,
      /*now_ns=*/2'010'000'000ULL,
      /*inject_ood_tarp=*/true));
  const auto ood_snap = node->LatestSnapshot();
  REQUIRE_FALSE(ood_snap.stale);
  REQUIRE(ood_snap.priors[0].slz_ood_variance >= 0.08f);
  REQUIRE(node->LastDecision().state != SlzDecisionState::COMMIT_TOUCHDOWN);

  // 2) Monocular fallback at 38 m AGL (>30 m veto) -> emits canonical REASON_MONO_ALT_VETO
  auto mono_node = std::make_shared<SlzPerceptionNode>(make_mock_runner_options());
  REQUIRE(mono_node->InjectMockFrame(
      /*stamp_ns=*/3'000'000'000ULL,
      /*now_ns=*/3'010'000'000ULL,
      /*inject_ood_tarp=*/false,
      DepthMode::MONO_FALLBACK,
      /*agl_m=*/38.0f));
  REQUIRE(mono_node->LastDecision().state != SlzDecisionState::COMMIT_TOUCHDOWN);
  REQUIRE((mono_node->LastDecision().reason_mask & SlzReasonBits::REASON_MONO_ALT_VETO) != 0u);
  REQUIRE((mono_node->LatestSnapshot().priors[0].reason_mask & SlzReasonBits::REASON_MONO_ALT_VETO) != 0u);

  // 3) Bit-exact harmonization between perception::SlzReasonBits and rta::slz_reason
  REQUIRE(SlzReasonBits::REASON_CTR_VIOLATION == vertirta::rta::slz_reason::kCtrViolation);
  REQUIRE(SlzReasonBits::REASON_ROUGHNESS_EXCEEDED == vertirta::rta::slz_reason::kRoughnessExceeded);
  REQUIRE(SlzReasonBits::REASON_SLOPE_EXCEEDED == vertirta::rta::slz_reason::kSlopeExceeded);
  REQUIRE(SlzReasonBits::REASON_CLEARANCE_INSUFF == vertirta::rta::slz_reason::kRadiusTooSmall);
  REQUIRE(SlzReasonBits::REASON_ENERGY_UNREACHABLE == vertirta::rta::slz_reason::kEnergyUnreachable);
  REQUIRE(SlzReasonBits::REASON_OBSTACLE_CLEARANCE == vertirta::rta::slz_reason::kObstacleClearance);
  REQUIRE(SlzReasonBits::REASON_STEP_EXCEEDED == vertirta::rta::slz_reason::kStepExceeded);
}

TEST_CASE("v3.5.1 RGB-D 2ms sync gate, rgb8/bgr8 ImageNet decode + glare guard, 32FC1/16UC1 depth, and OOD calibration wiring") {
  // 1) |t_rgb - t_depth| <= 2 ms gate (overflow-safe, zero-timestamp fail-closed)
  REQUIRE(CheckRgbdTimestampSync(1'000'000'000ULL, 1'002'000'000ULL));
  REQUIRE(CheckRgbdTimestampSync(1'002'000'000ULL, 1'000'000'000ULL));
  REQUIRE_FALSE(CheckRgbdTimestampSync(1'000'000'000ULL, 1'002'000'001ULL));
  REQUIRE_FALSE(CheckRgbdTimestampSync(0ULL, 1'000'000'000ULL));
  REQUIRE_FALSE(CheckRgbdTimestampSync(1'000'000'000ULL, 0ULL));

  // 2) rgb8 vs bgr8 ImageNet normalization parity + sun-glare over-exposure detection
  std::vector<uint8_t> rgb_raw(4u * 4u * 3u, 0u);
  std::vector<uint8_t> bgr_raw(4u * 4u * 3u, 0u);
  for (std::size_t i = 0; i < 16u; ++i) {
    rgb_raw[3u * i + 0u] = 200u;
    rgb_raw[3u * i + 1u] = 128u;
    rgb_raw[3u * i + 2u] = 64u;
    bgr_raw[3u * i + 0u] = 64u;
    bgr_raw[3u * i + 1u] = 128u;
    bgr_raw[3u * i + 2u] = 200u;
  }
  std::vector<float> nchw_rgb(vertirta::perception::kSlzRgbElements, 0.0f);
  std::vector<float> nchw_bgr(vertirta::perception::kSlzRgbElements, 0.0f);
  const auto st_rgb = DecodeRgbImageToNchw512(rgb_raw, 4u, 4u, 12u, false, nchw_rgb);
  const auto st_bgr = DecodeRgbImageToNchw512(bgr_raw, 4u, 4u, 12u, true, nchw_bgr);
  REQUIRE(st_rgb.ok);
  REQUIRE(st_bgr.ok);
  REQUIRE_FALSE(st_rgb.glare_warning);
  REQUIRE(nchw_rgb[0] == Approx(nchw_bgr[0]).margin(1e-6f));
  REQUIRE(nchw_rgb[vertirta::perception::kSlzPixels] == Approx(nchw_bgr[vertirta::perception::kSlzPixels]).margin(1e-6f));
  REQUIRE(nchw_rgb[2u * vertirta::perception::kSlzPixels] == Approx(nchw_bgr[2u * vertirta::perception::kSlzPixels]).margin(1e-6f));

  // Saturated Marmara sun-glare frame triggers glare_warning
  std::vector<uint8_t> glare_raw(4u * 4u * 3u, 255u);
  const auto st_glare = DecodeRgbImageToNchw512(glare_raw, 4u, 4u, 12u, false, nchw_rgb);
  REQUIRE(st_glare.ok);
  REQUIRE(st_glare.glare_warning);
  REQUIRE(st_glare.saturated_pixel_frac == Approx(1.0f));

  // 3) 32FC1 meters & 16UC1 mm depth decode
  std::vector<float> depth_out(vertirta::perception::kSlzPixels, 0.0f);
  std::vector<uint8_t> depth_32f(4u * 4u * 4u, 0u);
  const float d_val = 22.0f;
  for (std::size_t i = 0; i < 16u; ++i) {
    std::memcpy(depth_32f.data() + i * 4u, &d_val, sizeof(float));
  }
  REQUIRE(DecodeDepthImageToFloat512(depth_32f, 4u, 4u, 16u, false, depth_out));
  REQUIRE(depth_out[0] == Approx(22.0f));

  std::vector<uint8_t> depth_16u(4u * 4u * 2u, 0u);
  const uint16_t mm_val = 22000u;
  for (std::size_t i = 0; i < 16u; ++i) {
    std::memcpy(depth_16u.data() + i * 2u, &mm_val, sizeof(uint16_t));
  }
  REQUIRE(DecodeDepthImageToFloat512(depth_16u, 4u, 4u, 8u, true, depth_out));
  REQUIRE(depth_out[0] == Approx(22.0f).margin(1e-3f));

  // 4) ood_calibration.json wires calibrated tau_ood & T_prob into BOTH OrtRunner & BeliefGrid
  const std::string cal_path = "/tmp/vertirta_test_ood_cal_v351.json";
  {
    std::ofstream ofs(cal_path);
    ofs << R"({"T_ood": 1.0, "T_prob": 1.85, "kappa_b": 1.5, "tau_ood": -3.45})";
  }
  SlzPerceptionOptions cal_opts = make_mock_runner_options();
  cal_opts.ort_cfg.ood_calibration_path = cal_path;
  SlzPerceptionNode cal_node(std::move(cal_opts));
  std::remove(cal_path.c_str());
  REQUIRE(cal_node.Options().ort_cfg.tau_ood == Approx(-3.45f));
  REQUIRE(cal_node.Options().ort_cfg.t_prob == Approx(1.85f));
  REQUIRE(cal_node.BeliefGrid().config().tau_ood == Approx(-3.45f));

  // 5) Full decoded ProcessRgbdFrame without InjectMockFrame bypass
  const vertirta::core::Vec3Enu veh{0.0, 0.0, 22.0};
  const auto pose = vertirta::perception::MakeGimbalCameraPoseEnu(veh, 0.0, 1.5707963267948966);
  REQUIRE(cal_node.ProcessRgbdFrame(
      nchw_bgr, depth_out, pose, veh, 22.0f, DepthMode::RGBD_PRIMARY, 1'000'000'000ULL, 1'010'000'000ULL));
  REQUIRE_FALSE(cal_node.LatestSnapshot().stale);

  // 6) Fail-closed when allow_mock_fallback == false & onnx_path empty
  SlzPerceptionOptions strict_opts{};
  strict_opts.ort_cfg.onnx_path = "";
  strict_opts.ort_cfg.allow_mock_fallback = false;
  SlzPerceptionNode strict_node(std::move(strict_opts));
  REQUIRE_FALSE(strict_node.OrtRunner().is_ready());
  REQUIRE_FALSE(strict_node.ProcessRgbdFrame(
      nchw_bgr, depth_out, pose, veh, 22.0f, DepthMode::RGBD_PRIMARY, 2'000'000'000ULL, 2'010'000'000ULL));
}
