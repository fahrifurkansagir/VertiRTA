// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

#include "vertirta/perception/active_inspection_policy.hpp"
#include "vertirta/perception/slz_belief_grid.hpp"
#include "vertirta/perception/slz_geometry.hpp"
#include "vertirta/perception/slz_ort_engine.hpp"
#include "zero_heap_guard.hpp"

using Catch::Approx;
using vertirta::Vec3Enu;
using vertirta::perception::AnchorMonocularDepthAffine;
using vertirta::perception::ApplyUpsampledBoundaryCompensation;
using vertirta::perception::BackprojectDepthRgbd;
using vertirta::perception::CameraIntrinsics;
using vertirta::perception::ComputeFreeEnergy;
using vertirta::perception::CountCircularDiskCells;
using vertirta::perception::MakeGimbalCameraPoseEnu;
using vertirta::perception::OpticalAxisEnu;
using vertirta::perception::SlzBeliefGrid;
using vertirta::perception::SlzCell;
using vertirta::perception::SlzOrtConfig;
using vertirta::perception::SlzOrtRunner;
using vertirta::perception::SlzPointObservation;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Deterministic PRNG (xorshift32) for repeatable synthetic surface roughness.
float NextDeterministicSigned(uint32_t& state) noexcept {
  state ^= state << 13u;
  state ^= state >> 17u;
  state ^= state << 5u;
  const float u = static_cast<float>(state & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
  return 2.0f * u - 1.0f;
}

std::vector<SlzPointObservation> BuildRoofObservations(
    const Vec3Enu& center,
    float slope_deg,
    float roughness_m,
    float p_safe_obs,
    float p_dyn_obs,
    float e_tilde_ood,
    float parapet_step_m = 0.0f,
    double pitch_m = 0.25) {
  std::vector<SlzPointObservation> obs;
  const double tan_theta = std::tan(static_cast<double>(slope_deg) * kPi / 180.0);
  uint32_t rng = 0x12345678u;
  const int steps = static_cast<int>(std::ceil(6.5 / pitch_m));
  for (int iy = -steps; iy <= steps; ++iy) {
    for (int ix = -steps; ix <= steps; ++ix) {
      const double dx = static_cast<double>(ix) * pitch_m;
      const double dy = static_cast<double>(iy) * pitch_m;
      const double r = std::hypot(dx, dy);
      if (r > 6.5) {
        continue;
      }
      // Uniform [-sqrt(3)*rms, +sqrt(3)*rms] gives exact standard deviation == roughness_m.
      const double noise = static_cast<double>(roughness_m) * 1.7320508075688772 *
                           static_cast<double>(NextDeterministicSigned(rng));
      double dz = dx * tan_theta + noise;
      if (parapet_step_m > 0.0f && dx > 3.5 && dx < 4.5 && std::fabs(dy) < 1.5) {
        dz += static_cast<double>(parapet_step_m);
      }
      obs.push_back({
          {center.x + dx, center.y + dy, center.z + dz},
          p_safe_obs,
          p_dyn_obs,
          e_tilde_ood,
          0.95f,
      });
    }
  }
  return obs;
}

}  // namespace

TEST_CASE("SlzBeliefGrid memory budget, cell-local covariance, OOD floor, sparse guard & upsample",
          "[slz_bayes][grid]") {
  static_assert(sizeof(SlzCell) == 64u);
  REQUIRE(sizeof(SlzCell) * 160u * 160u == 1638400u);  // 1.5625 MiB
  REQUIRE(CountCircularDiskCells(5.0f, 0.5f) == 317u);

  // 1) Cell-local covariance avoids float32 catastrophic cancellation at Maltepe (-3500 m, +3200 m).
  // Use realistic 22 m AGL GSD (~10 cm pitch => 25 pts/cell, 225 pts per 3x3 window).
  SlzBeliefGrid grid;
  const Vec3Enu maltepe_roof{-3500.0, 3200.0, 15.0};
  const auto sloped_obs =
      BuildRoofObservations(maltepe_roof, 6.0f, 0.02f, 0.96f, 0.01f, -12.0f, 0.0f, 0.10);
  grid.UpdateFrame(sloped_obs, {maltepe_roof.x, maltepe_roof.y, maltepe_roof.z + 22.0}, 0.0, 0.8f);
  grid.UpdateFrame(sloped_obs, {maltepe_roof.x, maltepe_roof.y, maltepe_roof.z + 22.0}, 0.05, 0.8f);

  const auto c = grid.EvaluateDisk(maltepe_roof, 5.0f);
  REQUIRE(std::isfinite(c.theta_max_deg));
  REQUIRE(std::isfinite(c.sigma_z_m));
  REQUIRE(c.theta_max_deg == Approx(6.0f).margin(0.35f));
  REQUIRE(c.sigma_z_m < 0.04f);
  REQUIRE(c.clearance_r_m == Approx(5.0f).margin(0.01f));
  REQUIRE(c.p_safe > 0.90f);

  // Exact zero-noise plane check at Maltepe (-3500 m, +3200 m) proves zero cancellation error.
  SlzBeliefGrid exact_grid;
  const auto exact_obs =
      BuildRoofObservations(maltepe_roof, 6.0f, 0.0f, 0.96f, 0.0f, -12.0f, 0.0f, 0.25);
  exact_grid.UpdateFrame(exact_obs, {maltepe_roof.x, maltepe_roof.y, maltepe_roof.z + 22.0}, 0.0, 0.8f);
  const auto c_exact = exact_grid.EvaluateDisk(maltepe_roof, 5.0f);
  REQUIRE(c_exact.theta_max_deg == Approx(6.0f).margin(1e-3f));
  REQUIRE(c_exact.sigma_z_m < 1e-3f);

  // 2) Newly appearing OOD tarp must raise slz_ood_variance >= 0.08 even after 40 clean frames
  //    AND hold >= 0.08 even after 25 repeated OOD frames (DR-7 Erratum #2).
  SlzBeliefGrid ood_grid;
  const Vec3Enu roof_origin{0.0, 0.0, 12.0};
  const auto clean_obs = BuildRoofObservations(roof_origin, 1.0f, 0.01f, 0.97f, 0.0f, -12.0f);
  for (int k = 0; k < 40; ++k) {
    ood_grid.UpdateFrame(clean_obs, {0.0, 0.0, 34.0}, static_cast<double>(k) * 0.05, 0.25f);
  }
  const auto before_tarp = ood_grid.EvaluateDisk(roof_origin, 5.0f);
  REQUIRE(before_tarp.slz_ood_variance < 0.06f);

  const auto tarp_obs = BuildRoofObservations(roof_origin, 1.0f, 0.01f, 0.85f, 0.0f, +4.0f);
  for (int k = 0; k < 25; ++k) {
    ood_grid.UpdateFrame(tarp_obs, {0.0, 0.0, 34.0}, 2.0 + static_cast<double>(k) * 0.05, 0.5f);
  }
  const auto after_tarp = ood_grid.EvaluateDisk(roof_origin, 5.0f);
  REQUIRE(after_tarp.slz_ood_variance >= 0.08f);
  REQUIRE((after_tarp.reason_mask & vertirta::perception::REASON_OOD_UNCERTAIN) != 0u);

  // 3) Sparse <12 points remains conservatively invalid (> 80 deg).
  SlzBeliefGrid sparse;
  std::array<SlzPointObservation, 6> sparse_pts{};
  for (std::size_t i = 0; i < sparse_pts.size(); ++i) {
    sparse_pts[i] = {{static_cast<double>(i) * 0.1, 0.0, 0.0}, 0.99f, 0.0f, -12.0f, 1.0f};
  }
  sparse.UpdateFrame(sparse_pts, {0.0, 0.0, 22.0}, 0.0, 1.0f);
  const auto sparse_eval = sparse.EvaluateDisk({0.0, 0.0, 0.0}, 5.0f);
  REQUIRE(sparse_eval.theta_max_deg > 80.0f);
  REQUIRE(sparse_eval.clearance_r_m == Approx(0.0f));

  // 4) Parapet step height > 0.25 m caught via disk plane normal deviation.
  SlzBeliefGrid parapet_grid;
  const auto parapet_obs = BuildRoofObservations(roof_origin, 0.5f, 0.01f, 0.95f, 0.0f, -12.0f, 0.40f);
  parapet_grid.UpdateFrame(parapet_obs, {0.0, 0.0, 34.0}, 0.0, 0.8f);
  const auto parapet_eval = parapet_grid.EvaluateDisk(roof_origin, 5.0f);
  REQUIRE(parapet_eval.step_max_m > 0.25f);
  REQUIRE((parapet_eval.reason_mask & vertirta::perception::REASON_STEP_EXCEEDED) != 0u);

  // 5) Mod B affine anchoring uses exact roll+pitch slant z_AGL / (-d_Wz) & adds >0.04 variance.
  std::array<float, 16> disp{};
  std::array<float, 16> depth{};
  disp.fill(1.0f);
  const Vec3Enu opt{0.45, 0.20, -std::sqrt(1.0 - 0.45 * 0.45 - 0.20 * 0.20)};
  const float mono_var = AnchorMonocularDepthAffine(
      disp, opt, 22.0f, std::nullopt, std::nullopt, depth);
  REQUIRE(mono_var > 0.04f);
  REQUIRE(depth[0] == Approx(22.0f / static_cast<float>(-opt.z)).margin(0.1f));

  // 6) 64x64 -> 512x512 bilinear boundary compensation.
  std::vector<float> e_raw(512u * 512u, -4.0f);
  std::vector<float> b64(64u * 64u, 1.0f);
  std::vector<float> e_out(512u * 512u, 0.0f);
  REQUIRE(ApplyUpsampledBoundaryCompensation(e_raw, b64, 1.5f, e_out));
  REQUIRE(e_out[512u * 511u + 511u] == Approx(-5.5f).margin(1e-5f));
  REQUIRE(e_out[0] == Approx(-5.5f).margin(1e-5f));
}

TEST_CASE("Mod A RGB-D backprojection, dynamic hazard decay, scrolling & zero-heap hot loop",
          "[slz_bayes][geometry][zero_heap]") {
  // 1) Nadir RGB-D center pixel at 22 m depth from z=34 m hits z=12 m exactly.
  CameraIntrinsics intr{};
  intr.width = 16u;
  intr.height = 16u;
  intr.fx = 16.0f;
  intr.fy = 16.0f;
  intr.cx = 8.0f;
  intr.cy = 8.0f;
  std::array<float, 256> depth_img{};
  depth_img.fill(22.0f);
  const auto pose = MakeGimbalCameraPoseEnu({10.0, 20.0, 34.0}, 0.0, kPi * 0.5);
  const auto opt_ax = OpticalAxisEnu(pose);
  REQUIRE(opt_ax.z == Approx(-1.0).margin(1e-9));

  std::array<Vec3Enu, 256> pts{};
  std::array<uint32_t, 256> pix{};
  const std::size_t n_pts = BackprojectDepthRgbd(depth_img, intr, pose, pts, pix, 1u);
  REQUIRE(n_pts == 256u);
  const std::size_t center_idx = 8u * 16u + 8u;
  REQUIRE(pts[center_idx].x == Approx(10.0).margin(1e-6));
  REQUIRE(pts[center_idx].y == Approx(20.0).margin(1e-6));
  REQUIRE(pts[center_idx].z == Approx(12.0).margin(1e-6));

  // 2) Dynamic hazard instant attack and tau_dyn = 5 s decay.
  SlzBeliefGrid dyn_grid;
  const Vec3Enu pad{0.0, 0.0, 12.0};
  const auto ped_obs = BuildRoofObservations(pad, 0.5f, 0.01f, 0.90f, 0.90f, -12.0f);
  dyn_grid.UpdateFrame(ped_obs, {0.0, 0.0, 34.0}, 0.0, 0.5f);
  REQUIRE(dyn_grid.EvaluateDisk(pad, 5.0f).h_dyn_max == Approx(0.90f).margin(1e-3f));

  const auto clear_obs = BuildRoofObservations(pad, 0.5f, 0.01f, 0.95f, 0.0f, -12.0f);
  dyn_grid.UpdateFrame(clear_obs, {0.0, 0.0, 34.0}, 5.0, 0.5f);
  // After 1 time constant (5.0 s), 0.90 * exp(-1) ~= 0.3311.
  REQUIRE(dyn_grid.EvaluateDisk(pad, 5.0f).h_dyn_max == Approx(0.90f * std::exp(-1.0f)).margin(0.02f));

  // 3) Robocentric scrolling clears evicted cells across >80 m shift.
  dyn_grid.SetOrigin({200.0, 0.0, 12.0});
  REQUIRE(dyn_grid.cell_at_world(0.0, 0.0) == nullptr);
  const auto* fresh_cell = dyn_grid.cell_at_world(200.0, 0.0);
  REQUIRE(fresh_cell != nullptr);
  REQUIRE(fresh_cell->N == 0u);

  // 4) Pre-allocate runner buffers, then verify ZERO heap allocations across full hot loop.
  SlzOrtConfig ort_cfg{};
  SlzOrtRunner runner(ort_cfg);
  REQUIRE(runner.is_ready());
  REQUIRE(runner.is_mock_backend());

  std::vector<float> rgb(vertirta::perception::kSlzRgbElements, 0.5f);
  std::vector<float> logits(vertirta::perception::kSlzLogitsElements, 0.0f);
  std::vector<float> normals(vertirta::perception::kSlzNormalsElements, 0.0f);
  std::vector<float> bdry(vertirta::perception::kSlzBdryElements, 0.0f);
  std::vector<float> e_raw(vertirta::perception::kSlzPixels, 0.0f);
  std::vector<float> e_tilde(vertirta::perception::kSlzPixels, 0.0f);
  SlzBeliefGrid hot_grid;
  const auto hot_obs = BuildRoofObservations({50.0, 50.0, 12.0}, 2.0f, 0.01f, 0.95f, 0.0f, -12.0f);
  hot_grid.SetOrigin({50.0, 50.0, 12.0});

  {
    vertirta::test::ScopedZeroHeapGuard guard;
    REQUIRE(runner.Run(rgb, logits, normals, bdry));
    REQUIRE(ComputeFreeEnergy(logits, 1.0f, e_raw));
    REQUIRE(ApplyUpsampledBoundaryCompensation(e_raw, bdry, 1.5f, e_tilde));
    const std::size_t bp_n = BackprojectDepthRgbd(depth_img, intr, pose, pts, pix, 1u);
    REQUIRE(bp_n == 256u);
    hot_grid.UpdateFrame(hot_obs, {50.0, 50.0, 34.0}, 1.0, 0.5f);
    const auto res = hot_grid.EvaluateDisk({50.0, 50.0, 12.0}, 5.0f);
    REQUIRE(std::isfinite(res.theta_max_deg));
    REQUIRE(guard.alloc_count() == 0u);
    REQUIRE(guard.bytes_allocated() == 0u);
  }
}
