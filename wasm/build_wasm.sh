#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Single-threaded Emscripten flat-C WASM build for libvertirta_flight_core (< 150 KiB).
# Strict Worker-only ABI with deterministic IEEE-754 flags (no threads, no RTTI/exceptions).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/apps/vertigcs/public/wasm}"
BUILD_DIR="${2:-/tmp/vertirta_v3b_wasm_emcc}"
MAX_WASM_BYTES=153600

if ! command -v em++ >/dev/null 2>&1; then
  echo "[VertiRTA WASM] ERROR: em++ (Emscripten SDK) not found in PATH." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

emcmake cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DVERTIRTA_REPO_ROOT="${REPO_ROOT}" \
  -DVERTIRTA_WASM_OUT_DIR="${OUT_DIR}"

cmake --build "${BUILD_DIR}" --target libvertirta_flight_core_wasm -j

WASM_FILE="${OUT_DIR}/libvertirta_flight_core.wasm"
MJS_FILE="${OUT_DIR}/libvertirta_flight_core.mjs"

if [[ ! -f "${WASM_FILE}" || ! -f "${MJS_FILE}" ]]; then
  echo "[VertiRTA WASM] ERROR: Expected outputs missing in ${OUT_DIR}" >&2
  exit 3
fi

WASM_SIZE="$(stat -c%s "${WASM_FILE}")"
echo "[VertiRTA WASM] Built ${WASM_FILE} (${WASM_SIZE} bytes; budget <= ${MAX_WASM_BYTES} bytes)"
if (( WASM_SIZE > MAX_WASM_BYTES )); then
  echo "[VertiRTA WASM] FAIL: WASM size ${WASM_SIZE} exceeds 150 KiB cap (${MAX_WASM_BYTES})" >&2
  exit 4
fi

COI_GUARD="coi""-serviceworker"
if grep -E "SharedArrayBuffer|${COI_GUARD}" "${MJS_FILE}" >/dev/null 2>&1; then
  echo "[VertiRTA WASM] FAIL: Forbidden threading/COI symbol detected in ${MJS_FILE}" >&2
  exit 5
fi

# Note: CMake target compiles with:
#   -Os -fno-rtti -fno-exceptions -ffp-contract=off -fno-fast-math
#   -sENVIRONMENT=worker -sMODULARIZE=1 -sEXPORT_ES6=1
#   -sEXPORTED_FUNCTIONS=@${SCRIPT_DIR}/exported_funcs.txt
echo "[VertiRTA WASM] PASS: Single-threaded flat-C WASM verified under 150 KiB."
