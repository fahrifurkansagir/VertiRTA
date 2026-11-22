#!/usr/bin/env bash
set -euo pipefail

# VertiRTA Apache-2.0 GGUF Q4_K_M Export Script (Qwen3-4B / Qwen3-8B ONLY)
# Usage:
#   ./ml/copilot/export_gguf.sh <model_tag: qwen3-4b|qwen3-8b> <merged_hf_dir> <out_dir> [llama_cpp_dir]

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <qwen3-4b|qwen3-8b> <merged_hf_dir> <out_dir> [llama_cpp_dir]" >&2
  exit 2
fi

MODEL_TAG="$1"
MERGED_HF_DIR="$2"
OUT_DIR="$3"
LLAMA_CPP_DIR="${4:-${LLAMA_CPP_DIR:-$HOME/llama.cpp}}"

case "${MODEL_TAG}" in
  qwen3-4b)
    OUT_NAME="Qwen3-4B-VertiRTA-Q4_K_M.gguf"
    ;;
  qwen3-8b)
    OUT_NAME="Qwen3-8B-VertiRTA-Q4_K_M.gguf"
    ;;
  *)
    echo "ERROR: License lock permits exporting ONLY Apache-2.0 qwen3-4b or qwen3-8b (got '${MODEL_TAG}')." >&2
    exit 1
    ;;
esac

if [[ ! -d "${MERGED_HF_DIR}" ]]; then
  echo "ERROR: Merged BF16 checkpoint directory not found: ${MERGED_HF_DIR}" >&2
  exit 1
fi

CONVERT_PY="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANT_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"
if [[ ! -f "${CONVERT_PY}" ]]; then
  echo "ERROR: Missing llama.cpp converter: ${CONVERT_PY}" >&2
  exit 1
fi
if [[ ! -x "${QUANT_BIN}" ]]; then
  echo "ERROR: Missing llama-quantize binary: ${QUANT_BIN}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
F16_GGUF="${OUT_DIR}/${MODEL_TAG}-f16.gguf"
Q4_GGUF="${OUT_DIR}/${OUT_NAME}"

python3 "${CONVERT_PY}" "${MERGED_HF_DIR}" --outfile "${F16_GGUF}" --outtype f16
"${QUANT_BIN}" "${F16_GGUF}" "${Q4_GGUF}" Q4_K_M
rm -f "${F16_GGUF}"

sha256sum "${Q4_GGUF}" > "${Q4_GGUF}.sha256"
echo "EXPORTED_GGUF=${Q4_GGUF}"
cat "${Q4_GGUF}.sha256"
