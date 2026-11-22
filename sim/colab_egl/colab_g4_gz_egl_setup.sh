#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sensor-only Colab G4 Gazebo Harmonic (OGRE2 NVIDIA EGL) setup with < 2 min Drive .deb cache.
# Strictly zero flight-stack or middleware installs; pure headless sensor rendering only.
set -euo pipefail

CACHE_DIR="${VERTIRTA_GZ_CACHE_DIR:-/content/drive/MyDrive/vertirta_cache}"
WORLD_SDF="${1:-/content/vertirta/sim/colab_egl/istanbul_slz_egl.sdf}"
CODENAME="$(lsb_release -cs)" # Colab: jammy (22.04) or noble (24.04)
DEB_TAR="${CACHE_DIR}/gz_harmonic_${CODENAME}_debs.tar.zst"

# 1. Fast < 2 min Drive .deb cache restore (sensor-only gz-harmonic + transport/msgs)
if [[ -f "${DEB_TAR}" ]]; then
  echo "[VertiRTA-EGL] Restoring cached Gazebo Harmonic packages from ${DEB_TAR}..."
  rm -rf /tmp/gz_debs
  mkdir -p /tmp/gz_debs
  tar --zstd -xf "${DEB_TAR}" -C /tmp/gz_debs
  sudo dpkg -i /tmp/gz_debs/*.deb || sudo apt-get install -f -y --no-download
else
  echo "[VertiRTA-EGL] Cold install of Gazebo Harmonic for Ubuntu ${CODENAME}..."
  sudo apt-get update
  sudo apt-get install -y curl lsb-release gnupg zstd mesa-utils-extra
  sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable ${CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
  sudo apt-get update
  sudo apt-get clean
  sudo apt-get install -y --download-only \
    gz-harmonic libgz-transport13-dev libgz-msgs10-dev libgz-math7-dev
  mkdir -p "${CACHE_DIR}"
  (cd /var/cache/apt/archives && tar --zstd -cf "${DEB_TAR}" ./*.deb)
  sudo apt-get install -y \
    gz-harmonic libgz-transport13-dev libgz-msgs10-dev libgz-math7-dev
fi

# 2. Mandatory NVIDIA EGL environment & shared-library integrity guard (gz-sim#1272)
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export DISPLAY=""
export GZ_PARTITION="${GZ_PARTITION:-vertirta_colab_g4}"

EGL_NV_LIB="$(ldconfig -p | awk '/libEGL_nvidia\.so\.0/{print $NF; exit}')"
if [[ -z "${EGL_NV_LIB}" ]]; then
  echo "ERROR: libEGL_nvidia.so.0 not found in ldconfig cache!" >&2
  exit 1
fi
if ldd "${EGL_NV_LIB}" | grep -q "not found"; then
  echo "ERROR: Missing shared library dependency for ${EGL_NV_LIB}:" >&2
  ldd "${EGL_NV_LIB}" | grep "not found" >&2
  exit 1
fi
if ! eglinfo -B 2>/dev/null | grep -qiE "NVIDIA|EGL_EXT_device_base"; then
  echo "ERROR: No NVIDIA EGL hardware device reported by eglinfo -B!" >&2
  exit 1
fi

# 3. Launch headless OGRE2 EGL server and strictly block silent llvmpipe CPU fallback
if [[ ! -f "${WORLD_SDF}" ]]; then
  echo "ERROR: World SDF not found at ${WORLD_SDF}" >&2
  exit 1
fi

rm -f "${HOME}/.gz/rendering/ogre2.log"
DISPLAY="" gz sim -v 4 -s -r --headless-rendering --render-engine-server ogre2 \
  "${WORLD_SDF}" > /tmp/gz_server.log 2>&1 &
GZ_PID=$!
sleep 4

if ! kill -0 "${GZ_PID}" 2>/dev/null; then
  echo "ERROR: gz sim exited prematurely; inspect /tmp/gz_server.log" >&2
  exit 1
fi
if [[ ! -f "${HOME}/.gz/rendering/ogre2.log" ]]; then
  echo "ERROR: Expected ~/.gz/rendering/ogre2.log was not created!" >&2
  exit 1
fi
if ! grep -q "EGL Subsystem" "${HOME}/.gz/rendering/ogre2.log"; then
  echo "ERROR: OGRE2 did not initialize EGL Subsystem!" >&2
  exit 1
fi
if grep -qi "llvmpipe" "${HOME}/.gz/rendering/ogre2.log"; then
  echo "ERROR: Silent llvmpipe software rendering detected (gz-sim#1272)! Aborting." >&2
  kill "${GZ_PID}" 2>/dev/null || true
  exit 1
fi

echo "[VertiRTA-EGL] Hardware NVIDIA EGL verified (PID=${GZ_PID}). Checking RGB-D topics..."
gz topic -i --topic /slz_rgbd/image
gz topic -i --topic /slz_rgbd/depth_image
