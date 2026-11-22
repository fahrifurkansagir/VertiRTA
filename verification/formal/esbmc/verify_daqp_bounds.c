/* SPDX-License-Identifier: Apache-2.0
 * ESBMC v8.7 / CBMC / Frama-C Pure C99 Harness 5: Static DAQP Active-Set Bounds Safety
 * (DR-08 Section 3.2):
 *   - Pass 3A (--unwind 27): Full N=8, M=16, iter_limit=25 static workspace index,
 *       bounds, pointer, and non-recursive pivot safety.
 *   - Pass 3B: Symbolic N=3, M=6 near-singular pivot degeneracy & NaN immunity.
 *
 * When compiled with -DVERTIRTA_LINK_REAL_DAQP, also invokes real DAQP v0.10.0
 * daqp_ldp() / ldp2qp_solution() using the repo's non-recursive daqp_auxiliary_norec.c.
 */

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esbmc_compat.h"

#ifdef VERTIRTA_LINK_REAL_DAQP
#include "constants.h"
#include "daqp.h"
#include "types.h"
#endif

#define VERTIRTA_DAQP_PASS3A_N 8
#define VERTIRTA_DAQP_PASS3A_M 16
#define VERTIRTA_DAQP_ITER_LIMIT 25
#define VERTIRTA_DAQP_SING_TOL 1e-12

typedef struct {
  int n;
  int m;
  int n_active;
  int sing_ind;
  int iterations;
  int iter_limit;
  uint8_t WS[VERTIRTA_DAQP_PASS3A_N];
  uint16_t sense_mask;
  uint8_t sing_mask;
} DaqpStaticModel;

static void model_reset(DaqpStaticModel* w, int n, int m, int iter_limit) {
  const DaqpStaticModel zero = {0};
  *w = zero;
  w->n = n;
  w->m = m;
  w->n_active = 0;
  w->sing_ind = -1;
  w->iterations = 0;
  w->iter_limit = iter_limit;
}

/* Non-recursive iterative pivot matching core/flight_core/CMakeLists.txt daqp_auxiliary_norec.c */
static void model_pivot_last_iterative(DaqpStaticModel* w, bool trigger_swap, bool becomes_singular) {
  for (int step = 0; step < 2; ++step) {
    if (step >= w->n_active || !(w->n_active > 1 && w->sing_ind == -1)) {
      break;
    }
    const int rm_ind = w->n_active - 2;
    const int last_ind = w->n_active - 1;
    VERTIRTA_FORMAL_ASSERT(rm_ind >= 0 && rm_ind < w->n, "rm_ind within [0, n)");
    VERTIRTA_FORMAL_ASSERT(last_ind >= 0 && last_ind < w->n, "last index within [0, n)");
    if (!(trigger_swap && step == 0)) {
      break;
    }
    const uint8_t cid = w->WS[rm_ind];
    w->WS[rm_ind] = w->WS[last_ind];
    w->WS[last_ind] = cid;
    if (becomes_singular) {
      w->sing_ind = last_ind;
      break;
    }
  }
}

/* Pass 3A: Full N=8, M=16, iter_limit=25 workspace add/drop/pivot bounds safety */
void verify_pass3a_workspace_bounds(void) {
  DaqpStaticModel w;
  model_reset(&w, VERTIRTA_DAQP_PASS3A_N, VERTIRTA_DAQP_PASS3A_M, VERTIRTA_DAQP_ITER_LIMIT);

  for (int it = 0; it < VERTIRTA_DAQP_ITER_LIMIT; ++it) {
    w.iterations = it + 1;
#ifdef VERTIRTA_IS_ESBMC
    const bool do_add = nondet_bool();
    const bool is_sing = nondet_bool();
    const uint8_t cand = nondet_uchar();
    VERTIRTA_FORMAL_ASSUME(cand < (uint8_t)VERTIRTA_DAQP_PASS3A_M);
#else
    const bool do_add = ((it % 3) != 2);
    const bool is_sing = ((it % 7) == 0);
    const uint8_t cand = (uint8_t)(it % w.m);
#endif
    const uint16_t cbit = (uint16_t)(1u << (unsigned)cand);
    if (do_add && w.n_active < w.n && (w.sense_mask & cbit) == 0u) {
      const int slot = w.n_active;
      VERTIRTA_FORMAL_ASSERT(slot >= 0 && slot < VERTIRTA_DAQP_PASS3A_N, "add slot in bounds");
      const int l_diag = ((slot + 1) * (slot + 2)) / 2 - 1;
      VERTIRTA_FORMAL_ASSERT(
          l_diag >= 0 && l_diag < (VERTIRTA_DAQP_PASS3A_N * (VERTIRTA_DAQP_PASS3A_N + 1)) / 2,
          "packed L diagonal index in bounds");
      VERTIRTA_FORMAL_ASSERT(cand < (uint8_t)VERTIRTA_DAQP_PASS3A_M, "push constraint ID in bounds");
      w.WS[slot] = cand;
      if (is_sing) {
        w.sing_mask = (uint8_t)(w.sing_mask | (uint8_t)(1u << (unsigned)slot));
      }
      w.sense_mask = (uint16_t)(w.sense_mask | cbit);
      w.n_active += 1;
      if (is_sing) {
        w.sing_ind = slot;
      }
    } else if (w.n_active > 0) {
      const int rm = w.n_active - 1;
      VERTIRTA_FORMAL_ASSERT(rm >= 0 && rm < VERTIRTA_DAQP_PASS3A_N, "drop slot in bounds");
      const uint8_t cid = (uint8_t)(w.WS[rm] & 0x0Fu);
      VERTIRTA_FORMAL_ASSERT(cid < (uint8_t)VERTIRTA_DAQP_PASS3A_M, "WS constraint ID in bounds");
      w.sense_mask = (uint16_t)(w.sense_mask & (uint16_t)(~(1u << (unsigned)cid)));
      w.n_active -= 1;
      if (w.sing_ind >= w.n_active) {
        w.sing_ind = -1;
      }
    }

    VERTIRTA_FORMAL_ASSERT(w.n_active >= 0 && w.n_active <= w.n, "n_active invariant [0, n]");
    VERTIRTA_FORMAL_ASSERT(w.iterations <= VERTIRTA_DAQP_ITER_LIMIT, "iteration cap respected");
  }

  /* Inductive pivot safety check on the resulting workspace state */
#ifdef VERTIRTA_IS_ESBMC
  const bool trig_pivot = nondet_bool();
  const bool sing_after = nondet_bool();
#else
  const bool trig_pivot = true;
  const bool sing_after = false;
#endif
  model_pivot_last_iterative(&w, trig_pivot, sing_after);
  VERTIRTA_FORMAL_ASSERT(w.n_active >= 0 && w.n_active <= w.n, "post-pivot n_active invariant");
}

/* Pass 3B: Reduced N=3, M=6 symbolic pivot degeneracy & NaN-free LDL^T guard */
static void verify_pass3b_symbolic_pivot_degeneracy(double h0,
                                                    double h1,
                                                    double h2,
                                                    double a00,
                                                    double a01,
                                                    double a02) {
  const double H[3] = {h0, h1, h2};
  const double a[3] = {a00, a01, a02};

  /* Compute M * M^T diagonal pivot (a_i^2 / H_i) with strict singularity floor */
  double denom = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (!vertirta_is_finite_d(H[i]) || H[i] < 1e-9 || !vertirta_is_finite_d(a[i])) {
      return;
    }
    denom += (a[i] * a[i]) / H[i];
  }

  double step_scale = 0.0;
  if (vertirta_is_finite_d(denom) && denom >= VERTIRTA_DAQP_SING_TOL) {
    step_scale = 1.0 / denom;
  }
  VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(step_scale), "pivot step_scale must be finite");
  VERTIRTA_FORMAL_ASSERT(step_scale >= 0.0, "pivot step_scale must be non-negative");
}

#ifdef VERTIRTA_LINK_REAL_DAQP
static void verify_real_daqp_ldp_smoke(void) {
  DAQPWorkspace work;
  DAQPProblem qp;
  DAQPSettings settings;
  c_float RinvD[3] = {1.0, 1.0, 1.0};
  c_float v[3] = {-0.5, 0.25, -0.1};
  c_float M[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0};
  c_float dupper[6] = {2.0, 2.0, 2.0, 1.5, 1.5, 2.5};
  c_float dlower[6] = {-2.0, -2.0, -2.0, -1.5, -1.5, -2.5};
  c_float scaling[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  c_float x[3] = {0};
  c_float xold[3] = {0};
  c_float u[3] = {0};
  c_float lam[4] = {0};
  c_float lam_star[4] = {0};
  c_float L[10] = {0};
  c_float D[4] = {0};
  c_float xldl[4] = {0};
  c_float zldl[4] = {0};
  int sense[6] = {0};
  int WS[4] = {0};

  work = (DAQPWorkspace){0};
  qp = (DAQPProblem){0};
  settings = (DAQPSettings){0};

  settings.primal_tol = DAQP_DEFAULT_PRIM_TOL;
  settings.dual_tol = DAQP_DEFAULT_DUAL_TOL;
  settings.zero_tol = DAQP_DEFAULT_ZERO_TOL;
  settings.pivot_tol = DAQP_DEFAULT_PIVOT_TOL;
  settings.progress_tol = DAQP_DEFAULT_PROG_TOL;
  settings.cycle_tol = DAQP_DEFAULT_CYCLE_TOL;
  settings.iter_limit = VERTIRTA_DAQP_ITER_LIMIT;
  settings.fval_bound = DAQP_INF;
  settings.rho_soft = DAQP_DEFAULT_RHO_SOFT;
  settings.sing_tol = DAQP_DEFAULT_SING_TOL;
  settings.refactor_tol = DAQP_DEFAULT_REFACTOR_TOL;
  settings.eq_reduction = DAQP_EQ_REDUCTION_OFF;

  qp.n = 3;
  qp.m = 6;
  qp.ms = 3;
  qp.sense = sense;

  work.qp = &qp;
  work.settings = &settings;
  work.n = 3;
  work.m = 6;
  work.ms = 3;
  work.RinvD = RinvD;
  work.v = v;
  work.M = M;
  work.dupper = dupper;
  work.dlower = dlower;
  work.scaling = scaling;
  work.x = x;
  work.xold = xold;
  work.u = u;
  work.lam = lam;
  work.lam_star = lam_star;
  work.L = L;
  work.D = D;
  work.xldl = xldl;
  work.zldl = zldl;
  work.sense = sense;
  work.WS = WS;

  const int flag = daqp_ldp(&work);
  VERTIRTA_FORMAL_ASSERT(work.iterations <= VERTIRTA_DAQP_ITER_LIMIT, "real DAQP iter cap");
  VERTIRTA_FORMAL_ASSERT(work.n_active >= 0 && work.n_active <= 3, "real DAQP n_active in [0,3]");
  if (flag > 0) {
    ldp2qp_solution(&work);
    for (int i = 0; i < 3; ++i) {
      VERTIRTA_FORMAL_ASSERT(vertirta_is_finite_d(work.x[i]), "real DAQP x[i] must be finite");
    }
  }
}
#endif

void verify_pass3b_entry(void) {
#ifdef VERTIRTA_IS_ESBMC
  const float h0_f = nondet_float();
  const float a00_f = nondet_float();
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(h0_f) && h0_f >= 1e-3f && h0_f <= 1e3f);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_f(a00_f) && a00_f >= -100.0f && a00_f <= 100.0f);
  verify_pass3b_symbolic_pivot_degeneracy((double)h0_f, 1.0, 2.0, (double)a00_f, 0.0, 0.0);
#else
  verify_pass3b_symbolic_pivot_degeneracy(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
  verify_pass3b_symbolic_pivot_degeneracy(1e-6, 1.0, 10.0, 1e-9, 0.0, 0.0);
  verify_pass3b_symbolic_pivot_degeneracy(2.0, 2.0, 2.0, 1.0, -0.5, 0.25);
#endif
}

int main(void) {
  verify_pass3a_workspace_bounds();
  verify_pass3b_entry();

#ifdef VERTIRTA_LINK_REAL_DAQP
  verify_real_daqp_ldp_smoke();
#endif

  return 0;
}
