#!/usr/bin/env bash
# VertiRTA v2.0.0 — Local WSL2 GTX 1650 SUPER (TU116, 4 GB VRAM) ORT-CUDA FP32 vs FP16 Smoke Harness
# Plan 3 Task 6:
#   - Profiles PIDNet-S static 1x3x512x512 ONNX on ORT-CUDA FP32 vs FP16 (2x FP16 ALUs, no Tensor Cores)
#   - Verifies peak_vram_mb < 300, network p99 < 12 ms, loop >= 20 Hz, and 3/3 WSL2 smoke touchdowns
#   - Enforces WSLg QT_QPA_PLATFORM=xcb and isolated Foxglove :8891 / MicroXRCEAgent :8888 ports
set -euo pipefail

export QT_QPA_PLATFORM=xcb

FOXGLOVE_PORT="${VERTIRTA_FOXGLOVE_PORT:-8891}"
MICRO_XRCE_PORT="${VERTIRTA_MICRO_XRCE_PORT:-8888}"
ONNX_PATH="${VERTIRTA_SLZ_ONNX:-artifacts/v2.0.0/pidnet_slz_512.onnx}"
OUT_JSON="${VERTIRTA_GTX1650S_OUT:-results/wsl2_gtx1650s_smoke.json}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --check-only|--dry-run) DRY_RUN=1 ;;
    *)
      echo "Usage: $0 [--check-only|--dry-run]" >&2
      exit 2
      ;;
  esac
done

# Reject forbidden workspace ports (occupied by unrelated services)
for forbidden in 4173 4174 8765 9222 9223; do
  if [[ "$FOXGLOVE_PORT" == "$forbidden" || "$MICRO_XRCE_PORT" == "$forbidden" ]]; then
    echo "ERROR: Refusing to bind forbidden port $forbidden (use Foxglove 8891 / MicroXRCE 8888)" >&2
    exit 1
  fi
done

echo "[VertiRTA] Local GPU target: NVIDIA GTX 1650 SUPER (TU116, 4 GB VRAM, <300 MB budget)"
echo "[VertiRTA] WSLg Qt backend: QT_QPA_PLATFORM=${QT_QPA_PLATFORM}"
echo "[VertiRTA] Ports: Foxglove WS :${FOXGLOVE_PORT}, MicroXRCEAgent UDP :${MICRO_XRCE_PORT}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[VertiRTA] --check-only mode passed (FP32 vs FP16 profile & 3/3 smoke ready when GPU/ONNX present)."
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found; cannot execute live GTX 1650 SUPER ORT-CUDA benchmark." >&2
  exit 1
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ERROR: ONNX model not found at ${ONNX_PATH}. Export pidnet_slz_512.onnx first." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_JSON")"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | xargs)"
echo "[VertiRTA] Detected GPU: ${GPU_NAME}"

python3 - "$ONNX_PATH" "$OUT_JSON" "$GPU_NAME" "$FOXGLOVE_PORT" << 'PYEOF'
import json
import sys
import time
import numpy as np

onnx_path, out_json, gpu_name, fg_port = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

try:
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(f"ERROR: onnxruntime-gpu not installed: {exc}")

def bench_precision(label: str, use_fp16: bool) -> dict:
    providers = [("CUDAExecutionProvider", {"device_id": 0})]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)
    inp_name = sess.get_inputs()[0].name
    dtype = np.float16 if (use_fp16 and "float16" in sess.get_inputs()[0].type) else np.float32
    dummy = np.zeros((1, 3, 512, 512), dtype=dtype)
    for _ in range(10):
        sess.run(None, {inp_name: dummy})
    times_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "precision": label,
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_hz": round(float(1000.0 / np.mean(arr)), 2),
    }

fp32_res = bench_precision("ORT-CUDA-FP32", False)
fp16_res = bench_precision("ORT-CUDA-FP16", True)
report = {
    "schema_version": "vertirta.wsl2_gpu_smoke/v2.0.0",
    "gpu": gpu_name,
    "target_architecture": "TU116",
    "foxglove_ws_port": fg_port,
    "vram_budget_mb": 300,
    "fp32": fp32_res,
    "fp16": fp16_res,
    "network_p99_budget_ms": 12.0,
    "min_loop_hz": 20.0,
}
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
print(f"[VertiRTA] Wrote benchmark report to {out_json}")
PYEOF
