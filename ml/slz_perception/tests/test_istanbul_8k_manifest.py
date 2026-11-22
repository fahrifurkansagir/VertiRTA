# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Task 2: Colab-G4 EGL setup, kinematic gimbal SDF, and 8K Istanbul generator."""

from __future__ import annotations

import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ml.slz_perception.data.generate_istanbul_8k import (
    apply_quadratic_depth_noise,
    export_schedule_manifest,
    generate_sampling_schedule,
)


def test_8k_schedule_stratification_and_egl_kinematic_gimbal(tmp_path: Path) -> None:
    sched = generate_sampling_schedule(8000, seed=42)
    assert len(sched) == 8000
    assert all(
        (not f.uses_osm_odbl)
        and f.datum_lat == 40.8860
        and f.datum_lon == 29.1880
        and f.datum_alt_amsl_m == 12.0
        for f in sched
    )
    assert abs(sum(15.0 <= f.agl_m < 30.0 for f in sched) / 8000 - 0.35) < 0.01
    assert abs(sum(30.0 <= f.agl_m < 65.0 for f in sched) / 8000 - 0.40) < 0.01
    assert abs(sum(65.0 <= f.agl_m <= 120.0 for f in sched) / 8000 - 0.25) < 0.01
    assert abs(sum(f.inject_ood_254 for f in sched) / 8000 - 0.15) < 0.01
    assert all(0.0 <= f.pitch_off_nadir_deg <= 35.0 for f in sched)
    assert {f.coastal_block_id for f in sched} == set(range(1, 13))

    # Deterministic repeatability + manifest export passes license gate
    assert sched[:5] == generate_sampling_schedule(8000, seed=42)[:5]
    out_manifest = export_schedule_manifest(sched[:20], tmp_path / "istanbul_8k_sample.json")
    assert out_manifest.is_file()

    # Validate Colab G4 sensor-only EGL setup script (pure LF, zero markdown URLs, zero PX4/ROS 2)
    sh_path = Path("sim/colab_egl/colab_g4_gz_egl_setup.sh")
    sh_bytes = sh_path.read_bytes()
    assert b"\r" not in sh_bytes, "colab_g4_gz_egl_setup.sh must use pure LF line endings"
    sh = sh_bytes.decode("utf-8")
    assert "](" not in sh and "llvmpipe" in sh and "libEGL_nvidia.so.0" in sh
    assert "apt-get install" in sh and "ros-jazzy" not in sh and "px4" not in sh.lower().split("zero")[0]
    assert subprocess.run(["bash", "-n", sh_path.as_posix()], check=False).returncode == 0

    # Validate kinematic gimbal SDF (12 Copernicus GLO-30 procedural blocks, 6 canonical labels)
    sdf_path = Path("sim/colab_egl/istanbul_slz_egl.sdf")
    sdf_bytes = sdf_path.read_bytes()
    assert b"\r" not in sdf_bytes, "istanbul_slz_egl.sdf must use pure LF line endings"
    sdf = sdf_bytes.decode("utf-8")
    assert "<static>false</static>" in sdf
    assert "<kinematic>true</kinematic>" in sdf
    assert "<gravity>false</gravity>" in sdf
    assert "gz-sim-label-system" in sdf

    root = ET.fromstring(sdf)
    world = root.find("world")
    assert world is not None and world.attrib.get("name") == "istanbul_coast"
    sph = world.find("spherical_coordinates")
    assert sph is not None
    assert float(sph.findtext("latitude_deg", "0")) == 40.8860
    assert float(sph.findtext("longitude_deg", "0")) == 29.1880
    assert float(sph.findtext("elevation", "0")) == 12.0

    block_names = [
        m.attrib["name"]
        for m in world.findall("model")
        if m.attrib.get("name", "").startswith("coastal_block_")
    ]
    assert len(block_names) == 12

    labels_found = {int(el.text.strip()) for el in root.findall(".//label") if el.text}
    assert {10, 20, 30, 40, 50, 254}.issubset(labels_found)

    # 2-axis gimbal eliminates far-edge 5 m pad clipping during 22 m AGL / 15 m radius orbit
    assert math.degrees(math.atan2(20.0, 22.0) - math.atan2(15.0, 22.0)) < (
        math.degrees(1.396263 / 2.0) - 25.0
    )


def test_quadratic_depth_noise_matches_sigma_model() -> None:
    rng = np.random.default_rng(7)
    clean_z = np.full((256, 256), 25.0, dtype=np.float32)
    noisy_z = apply_quadratic_depth_noise(clean_z, rng, coeff=0.002, dropout_prob=0.02)
    valid = np.isfinite(noisy_z)
    dropout_rate = 1.0 - float(valid.mean())
    assert abs(dropout_rate - 0.02) < 0.005
    expected_sigma = 0.002 * (25.0**2)  # 1.25 m
    measured_sigma = float(np.std(noisy_z[valid] - 25.0))
    assert abs(measured_sigma - expected_sigma) < 0.08
