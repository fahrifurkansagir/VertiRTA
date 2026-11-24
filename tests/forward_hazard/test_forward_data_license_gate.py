# SPDX-License-Identifier: Apache-2.0
"""M3 Forward-Wire Dataset License Firewall & Canonical Config Lock Tests.

Enforces 01-DR08-13 Overrides #1-#7 & Plan 6A Section 0.2:
  - Decoupled 45 deg forward wire head allows ONLY:
      TTPLA (Apache-2.0), Mendeley TEIAS (CC BY 4.0), Purdue UAV (BSD-3),
      Gazebo_Istanbul_Catenary_Synth (Apache-2.0).
  - Hard-bans globally:
      BlenderProc (GPL-3.0), InsPLAD / DDOS / Drone-vs-Bird / UAVid (NC/DUA),
      STN PLAD / PAMPC (GPL-3.0), Ultralytics (AGPL-3.0), SegFormer /
      Deep Hough Transform / SuperPoint / SuperGlue (NC), DINOv3 (custom ITAR),
      OpenVINS / VINS-Fusion / HybVIO / ORB-SLAM3 (GPL-3.0), sqrtVINS (LGPL-3.0),
      DINOv2-SALAD (GPL-3.0), Sample4Geo (unlicensed).
  - Quarantines UAVScenes (alongside AeroScapes & ICG_Semantic_Drone) to
    zero-shot eval-only (HARD-BANNED from training manifests per Override #7).
  - Explicitly confirms TTPLA remains BANNED in the locked 512x512 nadir SLZ
    license gate (`ml/slz_perception/data/license_gate.py`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.slz_perception.data.license_gate import (  # noqa: E402
    DatasetEntry,
    LicenseViolationError,
    verify_training_manifest,
)

APPROVED_FORWARD_TRAIN_DATASETS: frozenset[str] = frozenset(
    {
        "ttpla",
        "mendeleyteias",
        "purdueuav",
        "gazeboistanbulcatenarysynth",
    }
)

QUARANTINED_EVAL_ONLY_DATASETS: frozenset[str] = frozenset(
    {
        "aeroscapes",
        "icgsemanticdrone",
        "uavscenes",
        "pldu",
        "pldm",
        "synthwires",
        "wireseghr",
        "spl",
        "sod4sb",
        "seraphim",
        "efficientloftr",
        "sentinel2",
    }
)

HARD_BANNED_GLOBAL: frozenset[str] = frozenset(
    {
        "blenderproc",
        "insplad",
        "ddos",
        "dronevsbird",
        "uavid",
        "stnplad",
        "pampc",
        "ultralytics",
        "segformer",
        "deephoughtransform",
        "superpoint",
        "superglue",
        "dinov3",
        "openvins",
        "vinsfusion",
        "hybvio",
        "orbslam3",
        "sqrtvins",
        "dinov2salad",
        "sample4geo",
    }
)


def _norm(name: str) -> str:
    # Normalize Turkish dotted/dotless I before stripping non-alphanumerics.
    s = name.replace("İ", "I").replace("ı", "i").replace("Ş", "S").replace("ş", "s")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def verify_forward_hazard_manifest(entries: Iterable[tuple[str, str, str]]) -> list[str]:
    """Validate (dataset_name, split, license) tuples for the decoupled 45 deg forward head."""
    approved: list[str] = []
    for name, split, lic in entries:
        key = _norm(name)
        sp = split.strip().lower()
        lic_low = lic.strip().lower()
        if any(tok in lic_low for tok in ("gpl", "agpl", "non-commercial", "nc", "odbl", "sharealike", "-sa")):
            if not (key in QUARANTINED_EVAL_ONLY_DATASETS and sp in {"eval", "zero_shot_eval", "eval_only"}):
                raise LicenseViolationError(f"Banned license {lic!r} for {name!r} in split {split!r}")
        if key in HARD_BANNED_GLOBAL:
            raise LicenseViolationError(f"Globally hard-banned dataset/model: {name!r}")
        if key in QUARANTINED_EVAL_ONLY_DATASETS:
            if sp not in {"eval", "zero_shot_eval", "eval_only"}:
                raise LicenseViolationError(
                    f"Quarantined eval-only dataset {name!r} forbidden in split {split!r}"
                )
            continue
        if key not in APPROVED_FORWARD_TRAIN_DATASETS:
            raise LicenseViolationError(f"Unapproved forward-hazard source: {name!r}")
        approved.append(name)
    return approved


def test_forward_manifest_allows_canonical_4_permissive_sources() -> None:
    manifest = [
        ("TTPLA", "train", "Apache-2.0"),
        ("Mendeley TEİAŞ", "train", "CC BY 4.0"),
        ("Purdue UAV", "train", "BSD-3-Clause"),
        ("Gazebo_Istanbul_Catenary_Synth", "train", "Apache-2.0"),
        ("UAVScenes", "zero_shot_eval", "CC BY-NC-SA 4.0"),
        ("AeroScapes", "zero_shot_eval", "CC BY-SA 4.0"),
    ]
    approved = verify_forward_hazard_manifest(manifest)
    assert len(approved) == 4


@pytest.mark.parametrize(
    "banned_name",
    [
        "BlenderProc",
        "InsPLAD",
        "DDOS",
        "Drone-vs-Bird",
        "UAVid",
        "STN PLAD",
        "PAMPC",
        "Ultralytics",
        "SegFormer",
        "Deep Hough Transform",
        "SuperPoint",
        "SuperGlue",
        "DINOv3",
        "OpenVINS",
        "VINS-Fusion",
        "HybVIO",
        "ORB-SLAM3",
        "sqrtVINS",
        "DINOv2-SALAD",
        "Sample4Geo",
    ],
)
def test_forward_manifest_hard_bans_gpl_nc_dua_everywhere(banned_name: str) -> None:
    for split in ("train", "val", "zero_shot_eval"):
        with pytest.raises(LicenseViolationError):
            verify_forward_hazard_manifest([(banned_name, split, "Apache-2.0")])


@pytest.mark.parametrize("eval_only", ["UAVScenes", "AeroScapes", "ICG_Semantic_Drone", "Sentinel-2"])
def test_quarantined_datasets_rejected_in_training_allowed_in_eval(eval_only: str) -> None:
    with pytest.raises(LicenseViolationError):
        verify_forward_hazard_manifest([(eval_only, "train", "CC BY 4.0")])
    # Allowed cleanly in zero-shot eval split
    approved = verify_forward_hazard_manifest([(eval_only, "zero_shot_eval", "CC BY-NC-SA 4.0")])
    assert approved == []


def test_ttpla_remains_strictly_banned_from_nadir_slz_training() -> None:
    """TTPLA is Apache-2.0 forward-45-deg ONLY; nadir 512x512 SLZ gate must reject it."""
    with pytest.raises(LicenseViolationError):
        verify_training_manifest(
            [
                DatasetEntry(
                    name="TTPLA",
                    split="train",
                    license="Apache-2.0",
                )
            ]
        )


def test_canonical_m3_yaml_enforces_all_dr08_13_overrides() -> None:
    cfg_path = REPO_ROOT / "config" / "forward_hazard_and_gnss_denied.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # Override #1: R_veh = 2.10 m, v_max = 22.0 m/s, degraded = 15.0 m/s
    assert cfg["vehicle_canonical_lock"]["r_veh_m"] == pytest.approx(2.10)
    assert cfg["vehicle_canonical_lock"]["v_max_nominal_mps"] == pytest.approx(22.0)
    assert cfg["vehicle_canonical_lock"]["v_max_radar_degraded_mps"] == pytest.approx(15.0)

    # Override #3 + DHMI AIP lock: verified LTBX ARP 40d59m26sN (40.990556 N), NOT 40.99889 N
    assert cfg["airspace_canonical_lock"]["ltbx_arp_lat_deg"] == pytest.approx(40.990556, abs=1e-6)
    assert cfg["airspace_canonical_lock"]["ltbx_arp_lon_deg"] == pytest.approx(29.216111, abs=1e-6)
    assert cfg["airspace_canonical_lock"]["max_lon_near_j2_deg"] == pytest.approx(29.1895, abs=1e-6)

    # Override #4: PX4 v1.16 vehicle_gps_position + EKF2_GPS_DELAY (NOT v1.18 vehicle_gnss)
    assert cfg["px4_v116_gnss_lock"]["px4_version"] == "v1.16"
    assert cfg["px4_v116_gnss_lock"]["gps_topic"] == "/fmu/out/vehicle_gps_position"
    assert cfg["px4_v116_gnss_lock"]["gps_delay_param"] == "EKF2_GPS_DELAY"

    # Override #6: TU116 enable_fp16=false default + locked 512x512 nadir SLZ camera
    assert cfg["forward_onnx_runtime"]["enable_fp16"] is False
    assert cfg["forward_camera_split_stack"]["nadir_slz_locked_width_px"] == 512
    assert cfg["forward_camera_split_stack"]["nadir_slz_locked_height_px"] == 512
    assert cfg["forward_camera_split_stack"]["forward_hfov_deg"] == pytest.approx(45.0)
    assert cfg["forward_camera_split_stack"]["forward_f_net_px"] == pytest.approx(1545.1)
    assert cfg["gnss_denied_vio"]["allow_gnss_feedback_into_vio"] is False
