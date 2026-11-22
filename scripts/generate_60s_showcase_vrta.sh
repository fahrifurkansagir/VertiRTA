#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-command 60-second (6,000-frame @ 100 Hz) Kartal fault showcase .vrta generator,
# Castagnoli CRC-32C verifier, and clean CesiumJS :5180 health check.
#
# Usage:
#   bash scripts/generate_60s_showcase_vrta.sh
#   bash scripts/generate_60s_showcase_vrta.sh --out build/showcase/demo_60s.vrta --port 5180
#   bash scripts/generate_60s_showcase_vrta.sh --serve   # Keep :5180 running interactively
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_VRTA="${REPO_ROOT}/build/showcase/kartal_fault_showcase_60s.vrta"
TICKS=6000
PORT=5180
SERVE_FOREVER=0
REPLAY_BIN=""
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_VRTA="$2"
      shift 2
      ;;
    --ticks)
      TICKS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --bin)
      REPLAY_BIN="$2"
      shift 2
      ;;
    --serve)
      SERVE_FOREVER=1
      shift
      ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# Workspace port isolation guard (never bind 4173, 4174, 8765, 9222, 9223).
case "${PORT}" in
  4173|4174|8765|9222|9223)
    echo "ERROR: Port ${PORT} is forbidden by workspace port isolation policy." >&2
    exit 2
    ;;
esac

if [[ -z "${REPLAY_BIN}" ]]; then
  for cand in \
    "${REPO_ROOT}/build/release-linux-x64/tools/cli/vertirta_replay_cli" \
    "${REPO_ROOT}/build/dev-linux-x64/tools/cli/vertirta_replay_cli"; do
    if [[ -x "${cand}" ]]; then
      REPLAY_BIN="${cand}"
      break
    fi
  done
fi

if [[ -z "${REPLAY_BIN}" || ! -x "${REPLAY_BIN}" ]]; then
  echo "ERROR: vertirta_replay_cli not found. Build with: cmake --preset release-linux-x64 && cmake --build --preset release-linux-x64" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_VRTA}")"
cd "${REPO_ROOT}"

echo "=== 1. Generating ${TICKS}-frame (60.0 s @ 100 Hz) Kartal fault showcase .vrta ==="
"${REPLAY_BIN}" \
  --generate-fixture "${OUT_VRTA}" \
  --ticks "${TICKS}" \
  --lut "${REPO_ROOT}/tests/fixtures/quick_test.vrta_lut" \
  --cel-lut "${REPO_ROOT}/tests/fixtures/quick_test.vrta_cel2d"

echo "=== 2. Verifying Castagnoli CRC-32C & telemetry integrity via viz.cesium.vrta_reader ==="
"${PYTHON_BIN}" - "${OUT_VRTA}" "${TICKS}" << 'PYEOF'
import hashlib
import json
import sys
from pathlib import Path
from viz.cesium.vrta_reader import read_vrta_file

vrta_path = Path(sys.argv[1])
expected_ticks = int(sys.argv[2])
hdr, frames = read_vrta_file(vrta_path)

if len(frames) != expected_ticks:
    raise SystemExit(f"Frame count mismatch: got {len(frames)}, expected {expected_ticks}")

cel_frames = sum(1 for f in frames if f.cel_latched != 0)
min_cbvf = min(f.cbvf_min_v for f in frames)
min_vterm = min(f.v_term_v for f in frames)
sha256 = hashlib.sha256(vrta_path.read_bytes()).hexdigest()

print(json.dumps({
    "vrta_path": str(vrta_path),
    "size_bytes": vrta_path.stat().st_size,
    "frames": len(frames),
    "duration_s": round(len(frames) / 100.0, 2),
    "cel_latched_frames": cel_frames,
    "datum_lat_deg": hdr.datum_lat_deg,
    "datum_lon_deg": hdr.datum_lon_deg,
    "min_cbvf_v": round(min_cbvf, 4),
    "min_v_term_v": round(min_vterm, 2),
    "crc32c_verified": True,
    "sha256": sha256,
}, indent=2))
PYEOF

if [[ "${SERVE_FOREVER}" -eq 1 ]]; then
  echo "=== 3. Starting interactive CesiumJS Tactical Console on http://127.0.0.1:${PORT}/ ==="
  exec "${PYTHON_BIN}" -m viz.cesium.ws_server --host 127.0.0.1 --port "${PORT}" --vrta "${OUT_VRTA}"
fi

echo "=== 3. Running live CesiumJS server health check on http://127.0.0.1:${PORT}/ ==="
SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" -m viz.cesium.ws_server \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --vrta "${OUT_VRTA}" >/tmp/vertirta_showcase_ws.log 2>&1 &
SERVER_PID=$!

"${PYTHON_BIN}" - "${PORT}" << 'PYEOF'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
base = f"http://127.0.0.1:{port}"
last_err = None
for _ in range(30):
    try:
        with urllib.request.urlopen(f"{base}/", timeout=2.0) as resp:
            html = resp.read().decode("utf-8", errors="replace")
            assert resp.status == 200 and "VertiRTA" in html
        with urllib.request.urlopen(f"{base}/api/airspace.geojson", timeout=2.0) as resp:
            geo = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200 and geo.get("type") == "FeatureCollection"
            n_feat = len(geo.get("features", []))
        print(json.dumps({
            "cesium_http_status": 200,
            "port": port,
            "airspace_features": n_feat,
            "health_check": "PASS",
        }, indent=2))
        sys.exit(0)
    except Exception as exc:
        last_err = exc
        time.sleep(0.2)

raise SystemExit(f"CesiumJS health check failed on {base}: {last_err}")
PYEOF

echo "60s showcase .vrta generated, CRC-32C verified, and CesiumJS health check PASSED (server cleanly stopped)."
