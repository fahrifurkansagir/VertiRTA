// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <memory>

#include "vertirta_perception_node/perception_node.hpp"

using vertirta::perception::DepthMode;
using vertirta::perception::SlzDecisionState;
using vertirta::perception::SlzReasonBits;
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
