#include "vertirta/core/daqp_qp.hpp"

#include <cmath>
#include <cstring>
#include <new>

extern "C" {
#include "constants.h"
#include "daqp.h"
#include "types.h"
}

#include "vertirta/core/math_utils.hpp"

namespace vertirta::core {

namespace {

template <int N_MAX, int M_MAX>
struct DaqpStaticBlock {
  static constexpr int kGenMax = M_MAX - N_MAX;
  static constexpr int kWSMax = N_MAX + 1;
  static constexpr int kLSize = (kWSMax * (kWSMax + 1)) / 2;

  DAQPProblem qp{};
  DAQPWorkspace work{};
  DAQPSettings settings{};
  int last_n{0};
  int last_m_total{0};

  c_float RinvD[N_MAX]{};
  c_float v[N_MAX]{};
  c_float M[kGenMax * N_MAX]{};
  c_float dupper[M_MAX]{};
  c_float dlower[M_MAX]{};
  c_float scaling[M_MAX]{};

  c_float x[N_MAX]{};
  c_float xold[N_MAX]{};
  c_float u[N_MAX]{};
  c_float lam[kWSMax]{};
  c_float lam_star[kWSMax]{};
  c_float L[kLSize]{};
  c_float D[kWSMax]{};
  c_float xldl[kWSMax]{};
  c_float zldl[kWSMax]{};

  int sense[M_MAX]{};
  int WS[kWSMax]{};

  void bind() noexcept {
    std::memset(&qp, 0, sizeof(qp));
    std::memset(&work, 0, sizeof(work));
    std::memset(&settings, 0, sizeof(settings));

    settings.primal_tol = DAQP_DEFAULT_PRIM_TOL;
    settings.dual_tol = DAQP_DEFAULT_DUAL_TOL;
    settings.zero_tol = DAQP_DEFAULT_ZERO_TOL;
    settings.pivot_tol = DAQP_DEFAULT_PIVOT_TOL;
    settings.progress_tol = DAQP_DEFAULT_PROG_TOL;
    settings.cycle_tol = DAQP_DEFAULT_CYCLE_TOL;
    settings.iter_limit = 200;
    settings.fval_bound = DAQP_INF;
    settings.eps_prox = 0.0;
    settings.eta_prox = 0.0;
    settings.rho_soft = DAQP_DEFAULT_RHO_SOFT;
    settings.rel_subopt = 0.0;
    settings.abs_subopt = 0.0;
    settings.sing_tol = DAQP_DEFAULT_SING_TOL;
    settings.refactor_tol = DAQP_DEFAULT_REFACTOR_TOL;
    settings.time_limit = 0.0;
    settings.w_soft = 0.0;
    settings.eq_reduction = DAQP_EQ_REDUCTION_OFF;

    qp.sense = sense;

    work.qp = &qp;
    work.settings = &settings;
    work.Rinv = nullptr;
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
    work.prox_mask = nullptr;
    work.bnb = nullptr;
    work.avi = nullptr;
    work.eq = nullptr;
    work.timer = nullptr;
    work.Mu = nullptr;
    work.rho_ls = nullptr;
    work.rho_us = nullptr;
    work.w_ls = nullptr;
    work.w_us = nullptr;
  }
};

}  // namespace

template <int N_MAX, int M_MAX>
StaticDaqpSolver<N_MAX, M_MAX>::StaticDaqpSolver() noexcept {
  using Block = DaqpStaticBlock<N_MAX, M_MAX>;
  static_assert(sizeof(Block) <= kWorkspaceBytes, "Increase kWorkspaceBytes for DaqpStaticBlock");
  static_assert(alignof(Block) <= 8, "DaqpStaticBlock alignment exceeds 8");
  auto* blk = new (storage_) Block();
  blk->bind();
}

template <int N_MAX, int M_MAX>
void StaticDaqpSolver<N_MAX, M_MAX>::reset_warm_start() noexcept {
  auto* blk = reinterpret_cast<DaqpStaticBlock<N_MAX, M_MAX>*>(storage_);
  std::memset(blk->sense, 0, sizeof(blk->sense));
  blk->work.n_active = 0;
  blk->work.reuse_ind = 0;
  blk->last_n = 0;
  blk->last_m_total = 0;
}

template <int N_MAX, int M_MAX>
DaqpSolveResult StaticDaqpSolver<N_MAX, M_MAX>::solve(
    int n,
    int m_general,
    const double* H_diag,
    const double* f,
    const double* A_rowmajor,
    const double* x_lower,
    const double* x_upper,
    const double* b_lower,
    const double* b_upper,
    bool warm_start) noexcept {
  DaqpSolveResult out{};
  out.used_fallback = true;

  if (n <= 0 || n > N_MAX || m_general < 0 || (n + m_general) > M_MAX ||
      m_general > (M_MAX - N_MAX) || H_diag == nullptr || f == nullptr ||
      x_lower == nullptr || x_upper == nullptr) {
    out.exit_flag = DAQP_EXIT_INFEASIBLE;
    return out;
  }
  if (m_general > 0 && (A_rowmajor == nullptr || b_lower == nullptr || b_upper == nullptr)) {
    out.exit_flag = DAQP_EXIT_INFEASIBLE;
    return out;
  }

  // Validate finiteness and strict convexity / bound feasibility before touching DAQP.
  for (int i = 0; i < n; ++i) {
    if (!is_finite_d(H_diag[i]) || H_diag[i] <= 1e-12 || !is_finite_d(f[i]) ||
        !is_finite_d(x_lower[i]) || !is_finite_d(x_upper[i]) || x_lower[i] > x_upper[i] + 1e-9) {
      out.exit_flag = DAQP_EXIT_INFEASIBLE;
      return out;
    }
  }
  for (int r = 0; r < m_general; ++r) {
    if (!is_finite_d(b_lower[r]) || !is_finite_d(b_upper[r]) || b_lower[r] > b_upper[r] + 1e-9) {
      out.exit_flag = DAQP_EXIT_INFEASIBLE;
      return out;
    }
    for (int c = 0; c < n; ++c) {
      if (!is_finite_d(A_rowmajor[r * n + c])) {
        out.exit_flag = DAQP_EXIT_INFEASIBLE;
        return out;
      }
    }
  }

  auto* blk = reinterpret_cast<DaqpStaticBlock<N_MAX, M_MAX>*>(storage_);
  const int m_total = n + m_general;

  blk->qp.n = n;
  blk->qp.m = m_total;
  blk->qp.ms = n;
  blk->work.n = n;
  blk->work.m = m_total;
  blk->work.ms = n;
  blk->work.state = 0;

  // 1. Diagonal Hessian Cholesky inverse RinvD[i] = 1/sqrt(H_i) and LDP shift v[i] = f[i]*RinvD[i].
  // Simple bounds: x = RinvD*(u - v) => (x_lo / RinvD) + v <= u <= (x_hi / RinvD) + v.
  // DAQP stores normalized simple bounds in dlower/dupper with scaling[i] = 1 / RinvD[i] = sqrt(H_i).
  for (int i = 0; i < n; ++i) {
    const double sqrt_h = std::sqrt(H_diag[i]);
    const double rinv = 1.0 / sqrt_h;
    blk->RinvD[i] = rinv;
    blk->v[i] = f[i] * rinv;
    blk->scaling[i] = sqrt_h;
    blk->dlower[i] = (x_lower[i] <= -1e20) ? -DAQP_INF : (x_lower[i] * sqrt_h + blk->v[i]);
    blk->dupper[i] = (x_upper[i] >= +1e20) ? +DAQP_INF : (x_upper[i] * sqrt_h + blk->v[i]);
  }

  // 2. General constraints: b_lo <= A x <= b_hi => b_lo + A*RinvD*v <= (A*RinvD)*u <= b_hi + A*RinvD*v,
  // normalized row-by-row so ||M_row||_2 == 1.0.
  for (int r = 0; r < m_general; ++r) {
    double row_norm2 = 0.0;
    double mv = 0.0;
    for (int c = 0; c < n; ++c) {
      const double m_rc = A_rowmajor[r * n + c] * blk->RinvD[c];
      blk->M[r * n + c] = m_rc;
      row_norm2 += m_rc * m_rc;
      mv += m_rc * blk->v[c];
    }
    const double row_norm = std::sqrt(row_norm2);
    const int idx = n + r;
    if (row_norm < 1e-12) {
      // Degenerate zero row: check trivial feasibility 0 in [b_lo, b_hi].
      if (b_lower[r] > 1e-9 || b_upper[r] < -1e-9) {
        out.exit_flag = DAQP_EXIT_INFEASIBLE;
        return out;
      }
      blk->scaling[idx] = 1.0;
      blk->dlower[idx] = -DAQP_INF;
      blk->dupper[idx] = +DAQP_INF;
    } else {
      const double inv_norm = 1.0 / row_norm;
      for (int c = 0; c < n; ++c) {
        blk->M[r * n + c] *= inv_norm;
      }
      blk->scaling[idx] = inv_norm;
      blk->dlower[idx] = (b_lower[r] <= -1e20) ? -DAQP_INF : (b_lower[r] + mv) * inv_norm;
      blk->dupper[idx] = (b_upper[r] >= +1e20) ? +DAQP_INF : (b_upper[r] + mv) * inv_norm;
    }
  }

  // 3. Working-set initialization (cold start or active-set warm start).
  reset_daqp_workspace(&blk->work);
  if (!warm_start || n != blk->last_n || m_total != blk->last_m_total) {
    for (int i = 0; i < m_total; ++i) {
      blk->sense[i] = 0;
    }
  } else {
    for (int i = 0; i < m_total; ++i) {
      blk->sense[i] &= (DAQP_ACTIVE | DAQP_LOWER);
    }
    (void)daqp_activate_constraints(&blk->work);
  }
  blk->last_n = n;
  blk->last_m_total = m_total;

  // 4. Solve LDP via pure C99 DAQP dual active-set and recover primal QP solution.
  out.exit_flag = daqp_ldp(&blk->work);
  out.iterations = blk->work.iterations;
  if (out.exit_flag == DAQP_EXIT_OPTIMAL || out.exit_flag == DAQP_EXIT_SOFT_OPTIMAL) {
    ldp2qp_solution(&blk->work);
    bool all_finite = true;
    for (int i = 0; i < n; ++i) {
      out.x[i] = blk->work.x[i];
      if (!is_finite_d(out.x[i])) {
        all_finite = false;
      }
    }
    out.used_fallback = !all_finite;
  } else {
    out.used_fallback = true;
  }

  return out;
}

template class StaticDaqpSolver<5, 32>;
template class StaticDaqpSolver<6, 32>;

}  // namespace vertirta::core
