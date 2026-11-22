/* SPDX-License-Identifier: Apache-2.0
 * Shared ESBMC v8.7 / Concrete Freestanding Compatibility Layer
 *
 * Allows every verification harness in verification/formal/esbmc/ to execute under:
 *   1) ESBMC v8.7 (--32 --floatbv --nan-check --overflow-check --bitwuzla) on x86_64 CI, AND
 *   2) Concrete local compilation + smoke execution under host GCC/G++ and arm-none-eabi-g++.
 *
 * NOTE: ESBMC verification is strictly Supplemental Engineering Verification (DO-333/DO-330
 * boundary: no off-the-shelf TQL-5/TQL-4 TQSP; Clang frontend vs GCC 14.2.1 -Os EOC
 * preservation FM.A-7[9] remains outside source-level BMC scope).
 */
#ifndef VERTIRTA_FORMAL_ESBMC_COMPAT_H
#define VERTIRTA_FORMAL_ESBMC_COMPAT_H

#ifdef __cplusplus
#include <cmath>
#include <cstdint>
#else
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#endif

#if defined(VERTIRTA_ESBMC_ANALYSIS) || defined(__ESBMC__) || defined(__ESBMC)

#define VERTIRTA_IS_ESBMC 1

/* ESBMC v8.7 automatically pre-includes esbmc_intrinsics.h, which declares
 * __ESBMC_assume, __ESBMC_assert, and nondet_{double,float,uchar,uint,int,bool}. */
#define VERTIRTA_FORMAL_ASSUME(cond) __ESBMC_assume(cond)
#define VERTIRTA_FORMAL_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

#else /* Concrete local compile & smoke execution mode */

static inline void vertirta_concrete_assume(bool cond) {
  if (!cond) {
    __builtin_trap();
  }
}

static inline void vertirta_concrete_assert(bool cond, const char* msg) {
  (void)msg;
  if (!cond) {
    __builtin_trap();
  }
}

#define __ESBMC_assume(cond) vertirta_concrete_assume(cond)
#define __ESBMC_assert(cond, msg) vertirta_concrete_assert((cond), (msg))
#define VERTIRTA_FORMAL_ASSUME(cond) vertirta_concrete_assume(cond)
#define VERTIRTA_FORMAL_ASSERT(cond, msg) vertirta_concrete_assert((cond), (msg))

static inline double nondet_double(void) { return 0.0; }
static inline float nondet_float(void) { return 0.0f; }
static inline uint8_t nondet_uchar(void) { return 0u; }
static inline unsigned int nondet_uint(void) { return 0u; }
static inline int nondet_int(void) { return 0; }
static inline bool nondet_bool(void) { return false; }

#endif /* VERTIRTA_ESBMC_ANALYSIS || __ESBMC__ */

static inline bool vertirta_is_finite_d(double x) {
#ifdef __cplusplus
  return std::isfinite(x);
#else
  return isfinite(x) != 0;
#endif
}

static inline bool vertirta_is_finite_f(float x) {
#ifdef __cplusplus
  return std::isfinite(x);
#else
  return isfinite(x) != 0;
#endif
}

#endif /* VERTIRTA_FORMAL_ESBMC_COMPAT_H */
