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

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __ESBMC__

#ifdef __cplusplus
extern "C" {
#endif
void __ESBMC_assume(bool cond);
void __ESBMC_assert(bool cond, const char* msg);
double nondet_double(void);
float nondet_float(void);
uint8_t nondet_uchar(void);
unsigned int nondet_uint(void);
int nondet_int(void);
bool nondet_bool(void);
#ifdef __cplusplus
}
#endif

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

#endif /* __ESBMC__ */

static inline bool vertirta_is_finite_d(double x) {
  return (x == x) && ((x - x) == 0.0);
}

static inline bool vertirta_is_finite_f(float x) {
  return (x == x) && ((x - x) == 0.0f);
}

#endif /* VERTIRTA_FORMAL_ESBMC_COMPAT_H */
