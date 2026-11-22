"""Driver-Gated Blender 5.2 LTS / 4.5 LTS Blackwell sm_120 OptiX Environment Guard.

Enforces DR-5 / Plan 1B Task 8 Blackwell GPU locks on Google Colab G4
(NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7 ECC):
- Blender 4.2 LTS and 4.3 lack NVIDIA Blackwell (sm_120) Cycles OptiX kernels
  and are explicitly rejected.
- NVIDIA driver >= 575 -> Primary **Blender 5.2 LTS** (5.2.4, July 2028 support).
- NVIDIA driver 535..574 -> Automatic fallback **Blender 4.5 LTS** (4.5.16).
- NVIDIA driver < 535 -> Hard RuntimeError (unsupported for Cycles OptiX).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from typing import Any, Sequence

BLACKWELL_MIN_BLENDER_VERSION: tuple[int, int, int] = (4, 4, 0)
BLENDER_52_MIN_DRIVER: int = 575
BLENDER_45_MIN_DRIVER: int = 535

BLENDER_RELEASES: dict[str, dict[str, Any]] = {
    "5.2": {
        "series": "5.2",
        "version": "5.2.4",
        "version_tuple": (5, 2, 4),
        "min_driver": BLENDER_52_MIN_DRIVER,
        "archive": "blender-5.2.4-linux-x64.tar.xz",
        "url": "https://download.blender.org/release/Blender5.2/blender-5.2.4-linux-x64.tar.xz",
        "extract_dir": "blender-5.2.4-linux-x64",
    },
    "4.5": {
        "series": "4.5",
        "version": "4.5.16",
        "version_tuple": (4, 5, 16),
        "min_driver": BLENDER_45_MIN_DRIVER,
        "archive": "blender-4.5.16-linux-x64.tar.xz",
        "url": "https://download.blender.org/release/Blender4.5/blender-4.5.16-linux-x64.tar.xz",
        "extract_dir": "blender-4.5.16-linux-x64",
    },
}


def parse_nvidia_driver_major(nvidia_smi_output: str) -> int:
    """Extract NVIDIA driver major version from ``nvidia-smi`` CSV or text output."""
    text = str(nvidia_smi_output).strip()
    if not text:
        raise RuntimeError("Empty nvidia-smi output; cannot determine NVIDIA driver version")
    # Match version tokens such as 575.51.03 or 560.35.03
    match = re.search(r"\b(\d{3})\.\d+(?:\.\d+)?\b", text)
    if not match:
        raise RuntimeError(f"Unable to parse NVIDIA driver version from: {text!r}")
    return int(match.group(1))


def select_blender_release(driver_major: int) -> dict[str, Any]:
    """Select Blender 5.2 LTS (driver >= 575) or 4.5 LTS fallback (535..574)."""
    drv = int(driver_major)
    if drv >= BLENDER_52_MIN_DRIVER:
        return dict(BLENDER_RELEASES["5.2"])
    if drv >= BLENDER_45_MIN_DRIVER:
        return dict(BLENDER_RELEASES["4.5"])
    raise RuntimeError(
        f"NVIDIA driver {drv} is below minimum {BLENDER_45_MIN_DRIVER} required for "
        "Blender 4.5/5.2 LTS Blackwell sm_120 Cycles OptiX"
    )


def validate_blender_version_for_blackwell(version_tuple: Sequence[int]) -> bool:
    """Reject Blender < 4.4 (explicitly 4.2/4.3 lacking NVIDIA Blackwell sm_120 OptiX)."""
    if len(version_tuple) < 2:
        raise ValueError(f"Invalid Blender version tuple: {version_tuple!r}")
    major, minor = int(version_tuple[0]), int(version_tuple[1])
    patch = int(version_tuple[2]) if len(version_tuple) >= 3 else 0
    if (major, minor, patch) < BLACKWELL_MIN_BLENDER_VERSION:
        raise ValueError(
            f"Blender {major}.{minor}.{patch} lacks NVIDIA Blackwell (sm_120) Cycles OptiX "
            "kernels (4.2/4.3 unsupported); install Blender 5.2 LTS (driver >= 575) or "
            "4.5 LTS fallback (driver 535..574)."
        )
    return True


def build_colab_install_commands(driver_major: int, install_root: str = "/opt") -> list[str]:
    """Generate headless Linux x64 Colab G4 installation & OptiX verification commands."""
    rel = select_blender_release(driver_major)
    validate_blender_version_for_blackwell(rel["version_tuple"])
    tarball = f"/tmp/{rel['archive']}"
    bin_path = f"{install_root.rstrip('/')}/{rel['extract_dir']}/blender"
    return [
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader",
        f"wget -q {rel['url']} -O {tarball}",
        f"tar -xf {tarball} -C {install_root}",
        f"ln -sf {bin_path} /usr/local/bin/blender",
        (
            "blender -b --python-expr \""
            "import bpy; "
            "p = bpy.context.preferences.addons['cycles'].preferences; "
            "p.compute_device_type = 'OPTIX'; p.get_devices(); "
            "optix = [d.name for d in p.devices if d.type == 'OPTIX']; "
            "assert optix, 'No OPTIX Blackwell device detected!'; "
            "print('OPTIX_READY:', bpy.app.version_string, optix)\""
        ),
    ]


def probe_host_nvidia_driver_major() -> int:
    """Invoke nvidia-smi on the current host and return the driver major version."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True,
    )
    return parse_nvidia_driver_major(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VertiRTA Colab G4 driver-gated Blender 5.2/4.5 LTS OptiX checker"
    )
    parser.add_argument(
        "--driver-major",
        type=int,
        default=None,
        help="Explicit NVIDIA driver major version (defaults to probing nvidia-smi)",
    )
    parser.add_argument("--print-json", action="store_true", help="Emit JSON install plan")
    args = parser.parse_args(argv)
    driver_major = args.driver_major if args.driver_major is not None else probe_host_nvidia_driver_major()
    rel = select_blender_release(driver_major)
    cmds = build_colab_install_commands(driver_major)
    if args.print_json:
        print(json.dumps({"driver_major": driver_major, "release": rel, "commands": cmds}, indent=2))
    else:
        for cmd in cmds:
            print(cmd)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
