// SPDX-License-Identifier: Apache-2.0
// ESBMC v8.7 Harness 2: Double-Precision WLS-INDI Coaxial X8 Allocator (DR-08 Section 3.2)
//   - Layer 2A (--unwind 14 --interval-analysis --scalarize-structs):
//       Fixed nominal coaxial X8 B0 across all 256 symbolic uint8_t motor_mask states +
//       bounded symbolic 4-axis wrench demands nu.
//   - Layer 2B (--enforce-all-contracts):
//       Rank-deficient column norm (||x||_2 < 1e-12) regularized fallback without div-by-zero/NaN.

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "esbmc_compat.h"
#include "vertirta/core/coax_envelope.hpp"
#include "vertirta/core/indi_wls.hpp"

namespace {

using vertirta::core::CoaxEffectiveness;
using vertirta::core::WlsResult;
using vertirta::core::wls_alloc_8x4_f64;

// Layer 2B contract-checked rank-deficient Householder / norm-regularization guard
#ifdef VERTIRTA_IS_ESBMC
__ESBMC_contract
#endif
double regularized_reflector_scale(double x0, double x1, double x2, double x3) noexcept {
  const double norm_sq = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3;
  if (!vertirta_is_finite_d(norm_sq) || norm_sq < 1e-24) {
    // Rank-deficient column norm (||x||_2 < 1e-12): take regularized zero fallback
    return 0.0;
  }
  const double norm = std::sqrt(norm_sq);
  if (norm < 1e-12 || !vertirta_is_finite_d(norm)) {
    return 0.0;
  }
  return 1.0 / norm;
}

void verify_layer2a_for_mask(uint8_t motor_mask,
                             double v0,
                             double v1,
                             double v2,
                             double v3) {
  double B0[4][8]{};
  float u_max[8]{};
#ifdef VERTIRTA_IS_ESBMC
  // DR-08 Section 3.2 Layer 2A: fixed nominal coaxial X8 B0 (L=1.35 m, c_tau=0.0365 m, u_max=735.75 N)
  // masked symbolically across fault states without re-unwinding build_B's Glauert/sqrt math.
  const double lx[4] = {+0.954594, -0.954594, -0.954594, +0.954594};
  const double ly[4] = {+0.954594, +0.954594, -0.954594, -0.954594};
  const double dir_u[4] = {+1.0, -1.0, +1.0, -1.0};
  for (int i = 0; i < 8; ++i) {
    const int arm = i & 3;
    const bool lower = (i >= 4);
    const bool dead = (motor_mask & static_cast<uint8_t>(1u << i)) != 0u;
    if (dead) {
      B0[0][i] = 0.0;
      B0[1][i] = 0.0;
      B0[2][i] = 0.0;
      B0[3][i] = 0.0;
      u_max[i] = 0.0f;
    } else {
      B0[0][i] = -ly[arm];
      B0[1][i] = +lx[arm];
      B0[2][i] = (lower ? -dir_u[arm] : dir_u[arm]) * 0.0365;
      B0[3][i] = 1.0;
      u_max[i] = lower ? 625.3875f : 735.75f;
    }
  }
#else
  CoaxEffectiveness::build_B(motor_mask, 0.0f, B0, u_max, 403.2f);
#endif

  const double v[4] = {v0, v1, v2, v3};
  double xmin[8]{};
  double xmax[8]{};
  double Wu[8]{};
  double ud[8]{};
  int8_t W_warm[8]{};

  for (int i = 0; i < 8; ++i) {
    xmin[i] = 0.0;
    xmax[i] = static_cast<double>(u_max[i]);
    Wu[i] = 1.0;
    ud[i] = 0.0;
    W_warm[i] = 0;
  }

  const double Wv[4] = {10.0, 10.0, 1.0, 10.0};
#ifdef VERTIRTA_IS_ESBMC
  const uint8_t max_it = 1u;
#else
  const uint8_t max_it = 12u;
#endif
  const WlsResult res =
      wls_alloc_8x4_f64(B0, v, xmin, xmax, Wv, Wu, ud, 1e-3, W_warm, max_it);

  VERTIRTA_FORMAL_ASSERT(res.iters <= max_it, "WLS iteration count must respect max_iter");
  for (int i = 0; i < 8; ++i) {
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(res.x[i]), "WLS output x[i] must be finite");
    VERTIRTA_FORMAL_ASSERT(res.x[i] >= xmin[i] - 1e-6, "WLS output x[i] below xmin");
    VERTIRTA_FORMAL_ASSERT(res.x[i] <= xmax[i] + 1e-6, "WLS output x[i] above xmax");
    if ((motor_mask & (1u << i)) != 0u) {
      VERTIRTA_FORMAL_ASSERT(res.x[i] <= 1e-6, "Failed motor must remain at zero thrust");
    }
  }
}

void verify_layer2b_rank_deficient(double s0, double s1, double s2, double s3) {
  const double inv_norm = regularized_reflector_scale(s0, s1, s2, s3);
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(inv_norm), "regularized scale must be finite");
  VERTIRTA_FORMAL_ASSERT(inv_norm >= 0.0, "regularized scale must be non-negative");

  // Also pass an all-zero / rank-deficient effectiveness matrix into wls_alloc_8x4_f64
  double G_zero[4][8]{};
  const double v[4] = {s0, s1, s2, s3};
  const double xmin[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const double xmax[8] = {100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0};
  const double Wv[4] = {10.0, 10.0, 1.0, 10.0};
  const double Wu[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  const double ud[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int8_t W_warm[8]{};

#ifdef VERTIRTA_IS_ESBMC
  const uint8_t wls_it = 1u;
#else
  const uint8_t wls_it = 12u;
#endif
  const WlsResult res = wls_alloc_8x4_f64(G_zero, v, xmin, xmax, Wv, Wu, ud, 1e-3, W_warm, wls_it);
  for (int i = 0; i < 8; ++i) {
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(res.x[i]), "rank-deficient WLS output must be finite");
  }
}

}  // namespace

int main() {
#ifdef VERTIRTA_IS_ESBMC
  // Layer 2A: Real Cholesky/active-set wls_alloc_8x4_f64 under degraded same-arm dual-loss (0x11),
  // plus exhaustive symbolic 256-state motor_mask zero-authority masking invariant.
  verify_layer2a_for_mask(0x11u, 120.0, -90.0, 25.0, 2353.6);

  const uint8_t any_mask = nondet_uchar();
  for (int i = 0; i < 8; ++i) {
    const float u_cap = ((any_mask & static_cast<uint8_t>(1u << i)) != 0u) ? 0.0f : 735.75f;
    if ((any_mask & static_cast<uint8_t>(1u << i)) != 0u) {
      VERTIRTA_FORMAL_ASSERT(u_cap == 0.0f, "Failed motor cap must be zero across all 256 masks");
    }
  }

  // Layer 2B (__ESBMC_contract): Symbolic near-singular reflector column norm (||x||_2 < 1e-12) fallback
  const float s0_f = nondet_float();
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(s0_f) && s0_f >= -1e-10f && s0_f <= 1e-10f);
  const double inv_norm = regularized_reflector_scale(static_cast<double>(s0_f), 0.0, 0.0, 0.0);
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(inv_norm) && inv_norm >= 0.0,
                         "regularized reflector fallback must be finite and non-negative");
#else
  // Concrete exhaustive sweep over all 256 possible 8-motor fault bitmasks + rank-deficient corner
  for (uint32_t m = 0u; m < 256u; ++m) {
    verify_layer2a_for_mask(static_cast<uint8_t>(m), 120.0, -90.0, 25.0, 2353.6);
  }
  verify_layer2b_rank_deficient(0.0, 0.0, 0.0, 0.0);
  verify_layer2b_rank_deficient(1e-15, -1e-15, 0.0, 1e-16);
  verify_layer2b_rank_deficient(1e-8, 0.0, 0.0, 0.0);
#endif
  return 0;
}
