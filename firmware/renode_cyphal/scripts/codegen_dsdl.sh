#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Nunavut C11 Code Generation Script for VertiRTA OpenCyphal v1.0 DSDL (DR-08 Sec 2.2)
#
# CRITICAL LOCK: Strictly target Nunavut C11 (--target-language c --language-standard c11).
# Upstream Nunavut C++ code generation is explicitly marked experimental and subject to
# redesign/removal; C++ codegen is strictly forbidden in VertiRTA flight software.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DSDL_ROOT="${ROOT_DIR}/dsdl/vertirta"
OUT_DIR="${1:-${ROOT_DIR}/generated/c11}"
VERIFY_ONLY=0

if [[ "${1:-}" == "--verify-only" ]]; then
  VERIFY_ONLY=1
  OUT_DIR="${ROOT_DIR}/generated/c11"
fi

DSDL_FILE="${DSDL_ROOT}/flight/SafetyEnvelope.1.0.dsdl"
if [[ ! -f "${DSDL_FILE}" ]]; then
  echo "[codegen_dsdl] ERROR: Missing DSDL schema: ${DSDL_FILE}" >&2
  exit 1
fi

# Verify mandatory single-frame CAN-FD DSDL assertions before codegen.
grep -q "@assert _offset_ == {384}" "${DSDL_FILE}"
grep -q "@assert _offset_ <= {504}" "${DSDL_FILE}"
grep -q "FIXED_SUBJECT_ID" "${DSDL_FILE}"

if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
  echo "[codegen_dsdl] DSDL schema contract verified (48 B / 384 bits <= 504 bits, C11 target)."
  exit 0
fi

if ! command -v nnvg >/dev/null 2>&1; then
  echo "[codegen_dsdl] ERROR: Nunavut 'nnvg' not found on PATH. Install via: pip install nunavut" >&2
  echo "[codegen_dsdl] Note: Checked-in C11 reference codec in src/safety_envelope_codec.c remains usable offline." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
nnvg \
  --target-language c \
  --language-standard c11 \
  --target-endianness little \
  --enable-serialization-asserts \
  --outdir "${OUT_DIR}" \
  "${DSDL_ROOT}"

echo "[codegen_dsdl] Generated Nunavut C11 headers into ${OUT_DIR}"
