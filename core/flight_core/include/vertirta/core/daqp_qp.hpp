#pragma once

#include <cstddef>
#include <cstdint>

namespace vertirta::core {

struct DaqpSolveResult {
  double x[6]{};
  int exit_flag{0};
  int iterations{0};
  bool used_fallback{true};
};

// Zero-heap, freestanding C++17 wrapper over static C99 DAQP v0.10.0 LDP solver.
// Solves min 0.5 * sum(H_diag[i] * x[i]^2) + f^T x
//   s.t. x_lo <= x <= x_hi  (n simple bounds)
//        b_lo <= A x <= b_hi (m_general row-major linear inequalities, n + m_general <= M_MAX).
// Opaque aligned static buffer avoids leaking C99 DAQP headers/macros to consumers.
template <int N_MAX = 6, int M_MAX = 32>
class StaticDaqpSolver {
 public:
  static_assert(N_MAX >= 1 && N_MAX <= 6, "N_MAX must be in [1, 6]");
  static_assert(M_MAX >= N_MAX && M_MAX <= 32, "M_MAX must be in [N_MAX, 32]");

  StaticDaqpSolver() noexcept;
  // Non-copyable & non-movable: internal DAQPWorkspace stores self-pointers into storage_.
  StaticDaqpSolver(const StaticDaqpSolver&) = delete;
  StaticDaqpSolver& operator=(const StaticDaqpSolver&) = delete;
  StaticDaqpSolver(StaticDaqpSolver&&) = delete;
  StaticDaqpSolver& operator=(StaticDaqpSolver&&) = delete;
  ~StaticDaqpSolver() = default;

  void reset_warm_start() noexcept;

  [[nodiscard]] DaqpSolveResult solve(
      int n,
      int m_general,
      const double* H_diag,
      const double* f,
      const double* A_rowmajor,
      const double* x_lower,
      const double* x_upper,
      const double* b_lower,
      const double* b_upper,
      bool warm_start = true) noexcept;

 private:
  // Static storage sized for DAQPWorkspace + DAQPProblem + DAQPSettings + dense arrays (<= 6,144 B).
  static constexpr std::size_t kWorkspaceBytes = 5888;
  alignas(8) std::uint8_t storage_[kWorkspaceBytes]{};
};

extern template class StaticDaqpSolver<5, 32>;
extern template class StaticDaqpSolver<6, 32>;

}  // namespace vertirta::core
