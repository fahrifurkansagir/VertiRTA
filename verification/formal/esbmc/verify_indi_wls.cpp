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
  CoaxEffectiveness::build_B(motor_mask, 0.0f, B0, u_max, 403.2f);

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
  const WlsResult res =
      wls_alloc_8x4_f64(B0, v, xmin, xmax, Wv, Wu, ud, 1e-3, W_warm, /*max_iter=*/12u);

  VERTIRTA_FORMAL_ASSERT(res.iters <= 12u, "WLS iteration count must respect max_iter=12");
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

  const WlsResult res = wls_alloc_8x4_f64(G_zero, v, xmin, xmax, Wv, Wu, ud, 1e-3, W_warm, 12u);
  for (int i = 0; i < 8; ++i) {
    VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(res.x[i]), "rank-deficient WLS output must be finite");
  }
}

}  // namespace

int main() {
#ifdef __ESBMC__
  const uint8_t motor_mask = nondet_uchar();
  const double v0 = nondet_double();
  const double v1 = nondet_double();
  const double v2 = nondet_double();
  const double v3 = nondet_double();

  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(v0) && v0 >= -400.0 && v0 <= 400.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(v1) && v1 >= -400.0 && v1 <= 400.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(v2) && v2 >= -120.0 && v2 <= 120.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(v3) && v3 >= -2500.0 && v3 <= 2500.0);

  verify_layer2a_for_mask(motor_mask, v0, v1, v2, v3);

  const double s0 = nondet_double();
  const double s1 = nondet_double();
  const double s2 = nondet_double();
  const double s3 = nondet_double();
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(s0) && s0 >= -1e-10 && s0 <= 1e-10);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(s1) && s1 >= -1e-10 && s1 <= 1e-10);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(s2) && s2 >= -1e-10 && s2 <= 1e-10);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(s3) && s3 >= -1e-10 && s3 <= 1e-10);
  verify_layer2b_rank_deficient(s0, s1, s2, s3);
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
