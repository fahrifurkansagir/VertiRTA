#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# VertiRTA MISRA-Aligned High-Integrity C++17 Static Profile Runner (Plan 4A Task 6 / DR-08 §3.5)
#
# Framing: Automated CI defect filter (GCC 14.2.1 + clang-tidy LLVM 22 + Cppcheck 2.22.0 +
# structural zero-heap/stack-DAG/isolation gates). NOT official MISRA C++:2023 compliance;
# see verification/formal/misra_profile/MISRA_GAP_REGISTER.md for all 5 unverifiable classes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

STRICT_TOOLS=0
CHECK_TOOLS_ONLY=0
BUILD_DIR="${VERTIRTA_STATIC_BUILD_DIR:-/tmp/vertirta_v3a_formal/static_cm7}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict-tools)
      STRICT_TOOLS=1
      shift
      ;;
    --check-tools)
      CHECK_TOOLS_ONLY=1
      shift
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cleanup() {
  if [[ "${BUILD_DIR}" == /tmp/vertirta_v3a_* ]]; then
    rm -rf "${BUILD_DIR}"
  fi
}
trap cleanup EXIT

HAS_CLANG_TIDY=0
HAS_CPPCHECK=0
HAS_ARM_GCC=0
command -v clang-tidy >/dev/null 2>&1 && HAS_CLANG_TIDY=1
command -v cppcheck >/dev/null 2>&1 && HAS_CPPCHECK=1
command -v arm-none-eabi-g++ >/dev/null 2>&1 && HAS_ARM_GCC=1

echo "[VertiRTA-Static] Tool inventory: arm-none-eabi-g++=${HAS_ARM_GCC}, clang-tidy=${HAS_CLANG_TIDY}, cppcheck=${HAS_CPPCHECK}"

if [[ "${STRICT_TOOLS}" -eq 1 ]]; then
  if [[ "${HAS_ARM_GCC}" -ne 1 || "${HAS_CLANG_TIDY}" -ne 1 || "${HAS_CPPCHECK}" -ne 1 ]]; then
    echo "ERROR: --strict-tools requires arm-none-eabi-g++, clang-tidy, and cppcheck on PATH." >&2
    exit 1
  fi
fi

if [[ "${CHECK_TOOLS_ONLY}" -eq 1 ]]; then
  echo "[VertiRTA-Static] --check-tools completed."
  exit 0
fi

# 1. Architecture isolation gate (ASTM F3269-21: 0 AI/LLM/vision imports in flight_core/rta_companion)
python3 "${REPO_ROOT}/scripts/ci/check_arch_isolation.py" --repo-root "${REPO_ROOT}"

# 2. Strict Cortex-M7 GCC 14.2.1 build + compile_commands.json + footprint/heap/stack-DAG gates
if [[ "${HAS_ARM_GCC}" -eq 1 ]]; then
  mkdir -p "${BUILD_DIR}"
  cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_SYSTEM_NAME=Generic \
    -DCMAKE_SYSTEM_PROCESSOR=arm \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_C_COMPILER=arm-none-eabi-gcc \
    -DCMAKE_CXX_COMPILER=arm-none-eabi-g++ \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_FLAGS="-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -fno-lto -fstack-usage -fcallgraph-info=su,da -Wstack-usage=4096 -Wvla -Werror" \
    -DCMAKE_CXX_FLAGS="-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Os -fno-lto -fstack-usage -fcallgraph-info=su,da -Wstack-usage=4096 -Wvla -Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion" \
    -DVERTIRTA_BUILD_COMPANION=OFF \
    -DVERTIRTA_BUILD_TESTS=OFF \
    -DVERTIRTA_ENABLE_STRICT_FP=ON > /dev/null
  cmake --build "${BUILD_DIR}" --target vertirta_flight_core > /dev/null

  ARCHIVE="${BUILD_DIR}/core/flight_core/libvertirta_flight_core.a"
  python3 "${REPO_ROOT}/scripts/ci/check_bare_metal_size.py" \
    --target "${ARCHIVE}" --size-bin arm-none-eabi-size \
    --max-flash-bytes 262144 --max-ram-bytes 131072
  python3 "${REPO_ROOT}/scripts/ci/check_no_heap_symbols.py" \
    --archive "${ARCHIVE}" --nm-bin arm-none-eabi-nm
  python3 "${REPO_ROOT}/scripts/ci/check_stack_usage.py" \
    --build-dir "${BUILD_DIR}/core/flight_core" \
    --max-frame-bytes 4096 --max-dag-bytes 16384
else
  echo "[VertiRTA-Static] WARN: arm-none-eabi-g++ unavailable locally; skipping Cortex-M7 build gate."
fi

FLIGHT_CORE_SRCS=(
  "${REPO_ROOT}/core/flight_core/src/frames.cpp"
  "${REPO_ROOT}/core/flight_core/src/tmr_voter.cpp"
  "${REPO_ROOT}/core/flight_core/src/coax_envelope.cpp"
  "${REPO_ROOT}/core/flight_core/src/indi_wls.cpp"
  "${REPO_ROOT}/core/flight_core/src/motor_fdi.cpp"
  "${REPO_ROOT}/core/flight_core/src/daqp_qp.cpp"
  "${REPO_ROOT}/core/flight_core/src/vrta_recorder.cpp"
  "${REPO_ROOT}/core/flight_core/src/sim_plant_6dof.cpp"
)

# 3. clang-tidy (LLVM 22 profile: explicit clang-analyzer-* opt-in, zero hicpp-*)
if [[ "${HAS_CLANG_TIDY}" -eq 1 ]]; then
  echo "[VertiRTA-Static] Running clang-tidy against core/flight_core/src/*.cpp..."
  mkdir -p "${BUILD_DIR}/clang_tidy_db"
  python3 -c '
import json, pathlib, shlex, subprocess, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
drop = {"-fstack-usage", "-fcallgraph-info=su,da", "-Wstack-usage=4096"}
# Discover Arm GNU Toolchain C++17 system include paths so Clang resolves <cstdint>/<cmath>/<array>
arm_inc_args = []
try:
    proc = subprocess.run(
        ["arm-none-eabi-g++", "-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard", "-E", "-x", "c++", "-", "-v"],
        input="", text=True, capture_output=True, check=False
    )
    in_inc = False
    for line in proc.stderr.splitlines():
        s = line.strip()
        if s.startswith("#include <...> search starts here:"):
            in_inc = True
            continue
        if in_inc:
            if s.startswith("End of search list."):
                break
            p = pathlib.Path(s).resolve()
            if p.is_dir():
                arm_inc_args.extend(["-isystem", str(p)])
except Exception:
    pass
entries = json.loads(src.read_text(encoding="utf-8"))
for e in entries:
    toks = shlex.split(e.get("command", ""))
    cleaned = [t for t in toks if t not in drop and not t.startswith("-fcallgraph-info=")]
    e["command"] = shlex.join(cleaned + arm_inc_args)
dst.write_text(json.dumps(entries, indent=2), encoding="utf-8")
' "${BUILD_DIR}/compile_commands.json" "${BUILD_DIR}/clang_tidy_db/compile_commands.json"
  clang-tidy \
    --config-file="${SCRIPT_DIR}/.clang-tidy" \
    -p "${BUILD_DIR}/clang_tidy_db" \
    --extra-arg=--target=arm-none-eabi \
    --extra-arg=-mcpu=cortex-m7 \
    --extra-arg=-Wno-unknown-warning-option \
    "${FLIGHT_CORE_SRCS[@]}"
else
  echo "[VertiRTA-Static] SKIP: clang-tidy not installed locally (enforced with --strict-tools in CI)."
fi

# 4. Cppcheck 2.22.0 (ARM EABI unsigned char via --platform=unix32 --funsigned-char per Cppcheck 2.19+ ABI)
if [[ "${HAS_CPPCHECK}" -eq 1 ]]; then
  echo "[VertiRTA-Static] Running cppcheck (--platform=unix32 --funsigned-char) against core/flight_core/src/*.cpp..."
  CPPCHECK_INC=(-I"${REPO_ROOT}/core/flight_core/include")
  if [[ -d "${BUILD_DIR}/_deps/daqp-src/include" ]]; then
    CPPCHECK_INC+=(-I"${BUILD_DIR}/_deps/daqp-src/include")
  fi
  cppcheck \
    --platform=unix32 \
    --funsigned-char \
    --std=c++17 \
    --enable=warning,portability \
    --inline-suppr \
    --suppress=missingIncludeSystem \
    --suppress=unusedFunction \
    --suppressions-list="${SCRIPT_DIR}/cppcheck_suppressions.txt" \
    --error-exitcode=1 \
    "${CPPCHECK_INC[@]}" \
    "${FLIGHT_CORE_SRCS[@]}"
  # Advisory style/performance audit (non-blocking on frozen v2.0.0 flight_core)
  cppcheck \
    --platform=unix32 \
    --funsigned-char \
    --std=c++17 \
    --enable=style,performance \
    --inline-suppr \
    --suppress=missingIncludeSystem \
    --suppress=unusedFunction \
    --suppressions-list="${SCRIPT_DIR}/cppcheck_suppressions.txt" \
    "${CPPCHECK_INC[@]}" \
    "${FLIGHT_CORE_SRCS[@]}" || true
else
  echo "[VertiRTA-Static] SKIP: cppcheck not installed locally (enforced with --strict-tools in CI)."
fi

echo "[VertiRTA-Static] MISRA-aligned static profile gate PASSED."
