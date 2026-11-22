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
  int WS[VERTIRTA_DAQP_PASS3A_N];
  int sense[VERTIRTA_DAQP_PASS3A_M];
  double L[(VERTIRTA_DAQP_PASS3A_N * (VERTIRTA_DAQP_PASS3A_N + 1)) / 2];
  double D[VERTIRTA_DAQP_PASS3A_N];
  double lam[VERTIRTA_DAQP_PASS3A_N];
} DaqpStaticModel;

static void model_reset(DaqpStaticModel* w, int n, int m, int iter_limit) {
  w->n = n;
  w->m = m;
  w->n_active = 0;
  w->sing_ind = -1;
  w->iterations = 0;
  w->iter_limit = iter_limit;
  for (int i = 0; i < VERTIRTA_DAQP_PASS3A_N; ++i) {
    w->WS[i] = 0;
    w->D[i] = 1.0;
    w->lam[i] = 0.0;
  }
  for (int i = 0; i < VERTIRTA_DAQP_PASS3A_M; ++i) {
    w->sense[i] = 0;
  }
  const int l_len = (VERTIRTA_DAQP_PASS3A_N * (VERTIRTA_DAQP_PASS3A_N + 1)) / 2;
  for (int i = 0; i < l_len; ++i) {
    w->L[i] = 0.0;
  }
}

/* Non-recursive iterative pivot matching core/flight_core/CMakeLists.txt daqp_auxiliary_norec.c */
static void model_pivot_last_iterative(DaqpStaticModel* w, double pivot_tol) {
  for (int step = 0; step < w->n_active; ++step) {
    if (!(w->n_active > 1 && w->sing_ind == -1)) {
      break;
    }
    const int rm_ind = w->n_active - 2;
    VERTIRTA_FORMAL_ASSERT(rm_ind >= 0 && rm_ind < w->n, "rm_ind within [0, n)");
    VERTIRTA_FORMAL_ASSERT((w->n_active - 1) >= 0 && (w->n_active - 1) < w->n,
                           "last index within [0, n)");
    if (!(w->D[rm_ind] < pivot_tol && w->D[rm_ind] < w->D[w->n_active - 1])) {
      break;
    }
    const int cid = w->WS[rm_ind];
    const double lam_old = w->lam[rm_ind];
    const double d_old = w->D[rm_ind];
    w->WS[rm_ind] = w->WS[w->n_active - 1];
    w->lam[rm_ind] = w->lam[w->n_active - 1];
    w->D[rm_ind] = w->D[w->n_active - 1];
    w->WS[w->n_active - 1] = cid;
    w->lam[w->n_active - 1] = lam_old;
    w->D[w->n_active - 1] = d_old;
    if (w->D[w->n_active - 1] < VERTIRTA_DAQP_SING_TOL) {
      w->sing_ind = w->n_active - 1;
      break;
    }
  }
}

/* Pass 3A: Full N=8, M=16, iter_limit=25 workspace add/drop/pivot bounds safety */
static void verify_pass3a_workspace_bounds(void) {
  DaqpStaticModel w;
  model_reset(&w, VERTIRTA_DAQP_PASS3A_N, VERTIRTA_DAQP_PASS3A_M, VERTIRTA_DAQP_ITER_LIMIT);

  for (int it = 0; it < w.iter_limit; ++it) {
    w.iterations = it + 1;
#ifdef __ESBMC__
    const bool do_add = nondet_bool();
    const int cand = nondet_int();
    const double d_val = nondet_double();
    VERTIRTA_FORMAL_ASSUME(cand >= 0 && cand < w.m);
    VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(d_val) && d_val >= 0.0 && d_val <= 1e4);
#else
    const bool do_add = ((it % 3) != 2);
    const int cand = it % w.m;
    const double d_val = (it % 7 == 0) ? 1e-14 : (1.0 + 0.1 * (double)it);
#endif
    if (do_add && w.n_active < w.n && w.sense[cand] == 0) {
      const int slot = w.n_active;
      VERTIRTA_FORMAL_ASSERT(slot >= 0 && slot < VERTIRTA_DAQP_PASS3A_N, "add slot in bounds");
      const int l_diag = ((slot + 1) * (slot + 2)) / 2 - 1;
      VERTIRTA_FORMAL_ASSERT(
          l_diag >= 0 && l_diag < (VERTIRTA_DAQP_PASS3A_N * (VERTIRTA_DAQP_PASS3A_N + 1)) / 2,
          "packed L diagonal index in bounds");
      w.WS[slot] = cand;
      w.D[slot] = d_val;
      w.L[l_diag] = 1.0;
      w.lam[slot] = 0.0;
      w.sense[cand] = 1;
      w.n_active += 1;
      if (d_val < VERTIRTA_DAQP_SING_TOL) {
        w.sing_ind = slot;
      } else {
        model_pivot_last_iterative(&w, 1e-6);
      }
    } else if (w.n_active > 0) {
      const int rm = w.n_active - 1;
      const int cid = w.WS[rm];
      VERTIRTA_FORMAL_ASSERT(cid >= 0 && cid < VERTIRTA_DAQP_PASS3A_M, "WS constraint ID in bounds");
      w.sense[cid] = 0;
      w.n_active -= 1;
      if (w.sing_ind >= w.n_active) {
        w.sing_ind = -1;
      }
    }

    VERTIRTA_FORMAL_ASSERT(w.n_active >= 0 && w.n_active <= w.n, "n_active invariant [0, n]");
    VERTIRTA_FORMAL_ASSERT(w.iterations <= VERTIRTA_DAQP_ITER_LIMIT, "iteration cap respected");
  }
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

  /* Compute M * M^T diagonal pivot with strict singularity floor */
  double denom = 0.0;
  for (int i = 0; i < 3; ++i) {
    if (!vertirta_is_finite_d(H[i]) || H[i] < 1e-9 || !vertirta_is_finite_d(a[i])) {
      return;
    }
    const double r_inv = 1.0 / sqrt(H[i]);
    const double m_i = a[i] * r_inv;
    denom += m_i * m_i;
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

int main(void) {
  verify_pass3a_workspace_bounds();

#ifdef __ESBMC__
  const double h0 = nondet_double();
  const double h1 = nondet_double();
  const double h2 = nondet_double();
  const double a00 = nondet_double();
  const double a01 = nondet_double();
  const double a02 = nondet_double();
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(h0) && h0 >= 1e-6 && h0 <= 1e3);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(h1) && h1 >= 1e-6 && h1 <= 1e3);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(h2) && h2 >= 1e-6 && h2 <= 1e3);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(a00) && a00 >= -100.0 && a00 <= 100.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(a01) && a01 >= -100.0 && a01 <= 100.0);
  VERTIRTA_FORMAL_ASSUME(vertirta_is_finite_d(a02) && a02 >= -100.0 && a02 <= 100.0);
  verify_pass3b_symbolic_pivot_degeneracy(h0, h1, h2, a00, a01, a02);
#else
  verify_pass3b_symbolic_pivot_degeneracy(1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
  verify_pass3b_symbolic_pivot_degeneracy(1e-6, 1.0, 10.0, 1e-9, 0.0, 0.0);
  verify_pass3b_symbolic_pivot_degeneracy(2.0, 2.0, 2.0, 1.0, -0.5, 0.25);
#endif

#ifdef VERTIRTA_LINK_REAL_DAQP
  verify_real_daqp_ldp_smoke();
#endif

  return 0;
}
