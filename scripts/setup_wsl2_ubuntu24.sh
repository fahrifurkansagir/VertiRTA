#!/usr/bin/env bash
# Idempotent WSL2 Ubuntu 24.04 / Linux toolchain check & bootstrap for VertiRTA (Plan 1A).
# Usage:
#   ./scripts/setup_wsl2_ubuntu24.sh --check-only   # Read-only environment report
#   ./scripts/setup_wsl2_ubuntu24.sh --install-core # Install C++/CMake/Ninja/Pytest/ARM toolchain
set -euo pipefail

MODE="${1:---check-only}"

check_cmd() {
  local cmd="$1"
  if command -v "$cmd" >/dev/null 2>&1; then
    printf "  [OK]      %-20s %s\n" "$cmd" "$("$cmd" --version 2>&1 | head -n 1)"
  else
    printf "  [MISSING] %-20s\n" "$cmd"
  fi
}

echo "=== VertiRTA Linux/WSL2 Toolchain Status ==="
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release || true
for c in g++ clang++ cmake ninja python3 pytest arm-none-eabi-g++ git gz ros2 MicroXRCEAgent; do
  check_cmd "$c"
done

if [[ "$MODE" == "--check-only" ]]; then
  exit 0
fi

if [[ "$MODE" == "--install-core" ]]; then
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    python3-pytest python3-yaml python3-numpy \
    gcc-arm-none-eabi libnewlib-arm-none-eabi
  echo "Core C++20/C++17 & bare-metal ARM toolchain installed."
  exit 0
fi

echo "Unknown option: $MODE (use --check-only or --install-core)" >&2
exit 2
