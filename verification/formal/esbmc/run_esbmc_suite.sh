#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# VertiRTA ESBMC v8.7 Bit-Precise Formal Verification Suite (Plan 4A Task 5 / DR-08 Section 3)
#
# HONEST CERTIFICATION BOUNDARY (DO-333 / ED-216 & DO-330):
#   This suite executes ESBMC v8.7 strictly as Supplemental Engineering Verification.
#   - No off-the-shelf DO-330 TQL-5 / TQL-4 Tool Qualification Support Pack (TQSP) exists.
#   - ESBMC parses C++17 via Clang/LLVM while target Cortex-M7 executes GCC 14.2.1 -Os object code,
#     leaving DO-333 Table FM.A-7[9] source-to-object preservation outside source-level BMC scope.
#   - Unwinding assertions remain strictly enabled to prevent vacuous bounded proofs.
#   - ESBMC --32 runs strictly on ubuntu-24.04 x86_64 (ARM64 esbmc-linux-armv8 fails --32).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODE="esbmc"
SOLVER="bitwuzla"
BUILD_DIR="${VERTIRTA_FORMAL_BUILD_DIR:-/tmp/vertirta_v3a_formal}"
WITNESS_DIR="${BUILD_DIR}/witnesses"
ESBMC_BIN="${ESBMC_BIN:-esbmc}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile-only)
      MODE="compile-only"
      shift
      ;;
    --solver)
      SOLVER="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      WITNESS_DIR="${BUILD_DIR}/witnesses"
      shift 2
      ;;
    --witness-dir)
      WITNESS_DIR="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${BUILD_DIR}" "${WITNESS_DIR}"

FC_INC="${REPO_ROOT}/core/flight_core/include"
FC_SRC="${REPO_ROOT}/core/flight_core/src"
RTA_INC="${REPO_ROOT}/core/rta_companion/include"
RTA_SRC="${REPO_ROOT}/core/rta_companion/src"

DAQP_INC="${BUILD_DIR}/cmake_core/_deps/daqp-src/include"
DAQP_SRC="${BUILD_DIR}/cmake_core/_deps/daqp-src/src"
DAQP_NOREC="${BUILD_DIR}/cmake_core/core/flight_core/daqp_auxiliary_norec.c"

ensure_daqp_sources() {
  if [[ ! -d "${DAQP_INC}" || ! -f "${DAQP_NOREC}" ]]; then
    cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}/cmake_core" -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DVERTIRTA_BUILD_COMPANION=ON \
      -DVERTIRTA_BUILD_TESTS=OFF \
      -DVERTIRTA_ENABLE_STRICT_FP=ON > /dev/null
  fi
  if [[ ! -d "${DAQP_INC}" || ! -f "${DAQP_NOREC}" ]]; then
    echo "ERROR: Missing CMake-generated DAQP sources in ${BUILD_DIR}/cmake_core" >&2
    exit 1
  fi
}

if [[ "${MODE}" == "compile-only" ]]; then
  echo "[VertiRTA-Formal] Running local concrete compile + smoke execution + Cortex-M7 syntax audit..."
  HOST_CXX="${CXX:-g++}"
  HOST_CC="${CC:-gcc}"
  COMMON_CXX_FLAGS=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fno-fast-math -I"${SCRIPT_DIR}" -I"${FC_INC}")
  COMMON_C_FLAGS=(-std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fno-fast-math -I"${SCRIPT_DIR}")

  # 1-4: Host concrete compile & execution of flight_core harnesses
  "${HOST_CXX}" "${COMMON_CXX_FLAGS[@]}" \
    "${SCRIPT_DIR}/verify_tmr_voter.cpp" "${FC_SRC}/tmr_voter.cpp" "${FC_SRC}/frames.cpp" \
    -o "${BUILD_DIR}/verify_tmr_voter" -lm
  "${BUILD_DIR}/verify_tmr_voter"

  "${HOST_CXX}" "${COMMON_CXX_FLAGS[@]}" \
    "${SCRIPT_DIR}/verify_indi_wls.cpp" "${FC_SRC}/indi_wls.cpp" "${FC_SRC}/coax_envelope.cpp" "${FC_SRC}/frames.cpp" \
    -o "${BUILD_DIR}/verify_indi_wls" -lm
  "${BUILD_DIR}/verify_indi_wls"

  "${HOST_CXX}" "${COMMON_CXX_FLAGS[@]}" \
    "${SCRIPT_DIR}/verify_motor_fdi.cpp" "${FC_SRC}/motor_fdi.cpp" "${FC_SRC}/coax_envelope.cpp" "${FC_SRC}/frames.cpp" \
    -o "${BUILD_DIR}/verify_motor_fdi" -lm
  "${BUILD_DIR}/verify_motor_fdi"

  "${HOST_CXX}" "${COMMON_CXX_FLAGS[@]}" \
    "${SCRIPT_DIR}/verify_coax_envelope.cpp" "${FC_SRC}/coax_envelope.cpp" "${FC_SRC}/frames.cpp" \
    -o "${BUILD_DIR}/verify_coax_envelope" -lm
  "${BUILD_DIR}/verify_coax_envelope"

  # 5: Pure C99 DAQP workspace bounds harness (standalone + real non-recursive DAQP v0.10.0)
  "${HOST_CC}" "${COMMON_C_FLAGS[@]}" \
    "${SCRIPT_DIR}/verify_daqp_bounds.c" \
    -o "${BUILD_DIR}/verify_daqp_bounds" -lm
  "${BUILD_DIR}/verify_daqp_bounds"

  ensure_daqp_sources

  # Compile C99 non-recursive DAQP v0.10.0 sources once for direct linkage (no static archive masking)
  "${HOST_CC}" -std=c99 -O2 -ffp-contract=off -fno-fast-math -I"${DAQP_INC}" \
    -c "${DAQP_SRC}/daqp.c" -o "${BUILD_DIR}/daqp.o"
  "${HOST_CC}" -std=c99 -O2 -ffp-contract=off -fno-fast-math -I"${DAQP_INC}" \
    -c "${DAQP_NOREC}" -o "${BUILD_DIR}/daqp_auxiliary_norec.o"
  "${HOST_CC}" -std=c99 -O2 -ffp-contract=off -fno-fast-math -I"${DAQP_INC}" \
    -c "${DAQP_SRC}/factorization.c" -o "${BUILD_DIR}/factorization.o"

  "${HOST_CC}" -std=c99 -O2 -ffp-contract=off -fno-fast-math -DVERTIRTA_LINK_REAL_DAQP \
    -I"${SCRIPT_DIR}" -I"${DAQP_INC}" \
    "${SCRIPT_DIR}/verify_daqp_bounds.c" \
    "${BUILD_DIR}/daqp.o" "${BUILD_DIR}/daqp_auxiliary_norec.o" "${BUILD_DIR}/factorization.o" \
    -o "${BUILD_DIR}/verify_daqp_bounds_real" -lm
  "${BUILD_DIR}/verify_daqp_bounds_real"

  # 6: CBVF filter + 4D LUT guard harness linked directly from exact live-ESBMC source list
  "${HOST_CXX}" -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -ffp-contract=off -fno-fast-math \
    -I"${SCRIPT_DIR}" -I"${FC_INC}" -I"${RTA_INC}" -I"${DAQP_INC}" \
    "${SCRIPT_DIR}/verify_cbvf_filter.cpp" \
    "${RTA_SRC}/cbvf_safety_filter.cpp" \
    "${RTA_SRC}/mmap_cbvf_lut.cpp" \
    "${RTA_SRC}/cel_2d_lut.cpp" \
    "${FC_SRC}/daqp_qp.cpp" \
    "${FC_SRC}/frames.cpp" \
    "${BUILD_DIR}/daqp.o" "${BUILD_DIR}/daqp_auxiliary_norec.o" "${BUILD_DIR}/factorization.o" \
    -o "${BUILD_DIR}/verify_cbvf_filter" -lm
  "${BUILD_DIR}/verify_cbvf_filter"

  # Cross-compile syntax + ABI parity check on Arm Cortex-M7 (arm-none-eabi-g++ 14.2.1)
  if command -v arm-none-eabi-g++ >/dev/null 2>&1 && command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    CM7_FLAGS=(-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -ffp-contract=off -Wall -Wextra -Wpedantic -Werror)
    for cpp_h in verify_tmr_voter.cpp verify_indi_wls.cpp verify_motor_fdi.cpp verify_coax_envelope.cpp; do
      arm-none-eabi-g++ -std=c++17 -fno-exceptions -fno-rtti "${CM7_FLAGS[@]}" \
        -I"${SCRIPT_DIR}" -I"${FC_INC}" -c "${SCRIPT_DIR}/${cpp_h}" -o "${BUILD_DIR}/${cpp_h}.cm7.o"
    done
    arm-none-eabi-gcc -std=c99 "${CM7_FLAGS[@]}" \
      -I"${SCRIPT_DIR}" -c "${SCRIPT_DIR}/verify_daqp_bounds.c" -o "${BUILD_DIR}/verify_daqp_bounds.cm7.o"
    arm-none-eabi-g++ -std=c++20 -fno-exceptions -fno-rtti "${CM7_FLAGS[@]}" \
      -I"${SCRIPT_DIR}" -I"${FC_INC}" -I"${RTA_INC}" -c "${SCRIPT_DIR}/verify_cbvf_filter.cpp" \
      -o "${BUILD_DIR}/verify_cbvf_filter.cm7.o"
    echo "[VertiRTA-Formal] Cortex-M7 arm-none-eabi compile check PASSED (6/6 harnesses)."
  fi

  echo "[VertiRTA-Formal] Concrete compile + execution smoke PASSED (6/6 harnesses)."
  exit 0
fi

# Live ESBMC v8.7 execution strictly requires x86_64 host (ARM64 build fails --32 libc headers)
ARCH="$(uname -m)"
if [[ "${ARCH}" != "x86_64" ]]; then
  echo "ERROR: ESBMC --32 verification MUST run on x86_64 (ubuntu-24.04), detected '${ARCH}'." >&2
  exit 2
fi

if ! command -v "${ESBMC_BIN}" >/dev/null 2>&1; then
  echo "ERROR: '${ESBMC_BIN}' not found on PATH. Install ESBMC v8.7 or pass --compile-only." >&2
  exit 127
fi

SOLVER_FLAG="--bitwuzla"
if [[ "${SOLVER}" == "z3" ]]; then
  SOLVER_FLAG="--z3"
elif [[ "${SOLVER}" != "bitwuzla" ]]; then
  echo "ERROR: Unsupported solver '${SOLVER}' (expected bitwuzla or z3)." >&2
  exit 1
fi

# Mandatory ESBMC v8.7 flags (DR-08 Section 3.1) — unwinding assertions stay strictly ON
ESBMC_BASE_FLAGS=(
  --32
  --floatbv
  --nan-check
  --overflow-check
  --unsigned-overflow-check
  --memory-leak-check
  --struct-fields-check
  "${SOLVER_FLAG}"
  -I"${SCRIPT_DIR}"
  -I"${FC_INC}"
)

echo "[VertiRTA-Formal] Running ESBMC v8.7 Supplemental Engineering Verification (${SOLVER_FLAG})..."

"${ESBMC_BIN}" "${ESBMC_BASE_FLAGS[@]}" --std=c++17 --unwind 5 \
  --witness-output "${WITNESS_DIR}/verify_tmr_voter.yaml" \
  "${SCRIPT_DIR}/verify_tmr_voter.cpp" "${FC_SRC}/tmr_voter.cpp" "${FC_SRC}/frames.cpp"

"${ESBMC_BIN}" "${ESBMC_BASE_FLAGS[@]}" --std=c++17 --unwind 14 \
  --interval-analysis --scalarize-structs --enforce-all-contracts \
  --witness-output "${WITNESS_DIR}/verify_indi_wls.yaml" \
  "${SCRIPT_DIR}/verify_indi_wls.cpp" "${FC_SRC}/indi_wls.cpp" "${FC_SRC}/coax_envelope.cpp" "${FC_SRC}/frames.cpp"

"${ESBMC_BIN}" "${ESBMC_BASE_FLAGS[@]}" --std=c++17 --unwind 10 \
  --witness-output "${WITNESS_DIR}/verify_motor_fdi.yaml" \
  "${SCRIPT_DIR}/verify_motor_fdi.cpp" "${FC_SRC}/motor_fdi.cpp" "${FC_SRC}/coax_envelope.cpp" "${FC_SRC}/frames.cpp"

"${ESBMC_BIN}" "${ESBMC_BASE_FLAGS[@]}" --std=c++17 --unwind 10 \
  --witness-output "${WITNESS_DIR}/verify_coax_envelope.yaml" \
  "${SCRIPT_DIR}/verify_coax_envelope.cpp" "${FC_SRC}/coax_envelope.cpp" "${FC_SRC}/frames.cpp"

"${ESBMC_BIN}" "${ESBMC_BASE_FLAGS[@]}" --unwind 27 \
  --witness-output "${WITNESS_DIR}/verify_daqp_bounds.yaml" \
  "${SCRIPT_DIR}/verify_daqp_bounds.c"

ensure_daqp_sources

"${ESBMC_BIN}" "${ESBMC_BASE_FLAGS[@]}" -I"${RTA_INC}" -I"${DAQP_INC}" --std=c++20 --unwind 12 \
  --witness-output "${WITNESS_DIR}/verify_cbvf_filter.yaml" \
  "${SCRIPT_DIR}/verify_cbvf_filter.cpp" \
  "${RTA_SRC}/cbvf_safety_filter.cpp" \
  "${RTA_SRC}/mmap_cbvf_lut.cpp" \
  "${RTA_SRC}/cel_2d_lut.cpp" \
  "${FC_SRC}/daqp_qp.cpp" \
  "${FC_SRC}/frames.cpp" \
  "${DAQP_SRC}/daqp.c" \
  "${DAQP_NOREC}" \
  "${DAQP_SRC}/factorization.c"

echo "[VertiRTA-Formal] ESBMC v8.7 Supplemental Engineering Verification PASSED (6/6 harnesses)."
