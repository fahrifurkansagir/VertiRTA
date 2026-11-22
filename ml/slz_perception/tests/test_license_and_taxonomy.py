# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Task 1: Zero-ShareAlike 5-Source License Gate & Canonical 5-Class Taxonomy."""

from __future__ import annotations

import numpy as np
import pytest

from ml.slz_perception.data.license_gate import (
    DatasetEntry,
    LicenseViolationError,
    parse_oem_permissive_whitelist,
    verify_training_manifest,
)
from ml.slz_perception.data.remappers import assert_uavscenes_cmap_ids, remap_mask


@pytest.mark.parametrize(
    "name",
    [
        "RescueNet",
        "LoveDA",
        "DroneDeploy",
        "FlyAwareV2",
        "DINOv3",
        "SegFormer-B0",
        "Mid-Air",
        "UAVid",
        "Depth-Anything-V2-Large",
        "Depth-Anything-V2-Base",
        "Depth-Anything-V2-Giant",
        "AeroScapes",
        "ICG_Semantic_Drone",
        "UAVScenes",
    ],
)
def test_banned_and_eval_only_blocked_from_training(name: str, tmp_path) -> None:
    with pytest.raises(LicenseViolationError):
        verify_training_manifest(
            [
                DatasetEntry(
                    name=name,
                    split="train",
                    license="CC-BY-SA-4.0",
                    uses_osm_odbl=False,
                )
            ],
            tmp_path / "attr.html",
        )


def test_oem_whitelist_excludes_public_domain_blank_and_osm_odbl(tmp_path) -> None:
    html = tmp_path / "attr.html"
    html.write_text(
        "<table>"
        "<tr><td>christchurch</td><td>CC BY 4.0</td></tr>"
        "<tr><td>aachen</td><td>DL-DE-BY-2.0</td></tr>"
        "<tr><td>austin</td><td>Public Domain</td></tr>"
        "<tr><td>tokyo</td><td></td></tr>"
        "<tr><td>paris</td><td>CC BY-SA 4.0</td></tr>"
        "</table>",
        encoding="utf-8",
    )
    assert parse_oem_permissive_whitelist(html, expect_total=5) == {"christchurch", "aachen"}

    with pytest.raises(LicenseViolationError):
        verify_training_manifest(
            [
                DatasetEntry(
                    name="Gazebo_Istanbul_8K",
                    split="train",
                    license="Apache-2.0",
                    uses_osm_odbl=True,
                )
            ],
            html,
        )


def test_oem_100_entry_whitelist_and_clean_5_source_manifest(tmp_path) -> None:
    rows = []
    for i in range(35):
        rows.append(f"<tr><td>ccby_region_{i:02d}</td><td>CC BY 4.0</td></tr>")
    for city in ("aachen", "bielefeld", "dortmund", "dusseldorf", "koln", "munster"):
        rows.append(f"<tr><td>{city}</td><td>DL-DE-BY-2.0</td></tr>")
    for i in range(36):
        rows.append(f"<tr><td>ncsa_region_{i:02d}</td><td>CC BY-NC-SA 4.0</td></tr>")
    for i in range(12):
        rows.append(f"<tr><td>blank_region_{i:02d}</td><td></td></tr>")
    for i in range(6):
        rows.append(f"<tr><td>sa_region_{i:02d}</td><td>CC BY-SA 4.0</td></tr>")
    for i in range(5):
        rows.append(f"<tr><td>pd_region_{i:02d}</td><td>Public Domain</td></tr>")
    assert len(rows) == 100

    html = tmp_path / "oem_attribution_100.html"
    html.write_text("<table>" + "".join(rows) + "</table>", encoding="utf-8")
    whitelist = parse_oem_permissive_whitelist(html, expect_total=100)
    assert len(whitelist) == 41
    assert "aachen" in whitelist and "ccby_region_00" in whitelist
    assert "pd_region_00" not in whitelist and "blank_region_00" not in whitelist

    entries = [
        DatasetEntry(name="FLAIR-1", split="train", license="etalab-2.0", uses_osm_odbl=False),
        DatasetEntry(
            name="OpenEarthMap",
            split="train",
            license="CC BY 4.0 + DL-DE-BY-2.0",
            uses_osm_odbl=False,
            regions=("ccby_region_00", "aachen"),
        ),
        DatasetEntry(
            name="FloodNet-Supervised-v1.0",
            split="train",
            license="CDLA-Permissive-1.0",
            uses_osm_odbl=False,
        ),
        DatasetEntry(name="SkyScenes", split="train", license="MIT", uses_osm_odbl=False),
        DatasetEntry(
            name="Gazebo_Istanbul_8K",
            split="train",
            license="Apache-2.0",
            uses_osm_odbl=False,
        ),
        DatasetEntry(name="AeroScapes", split="eval", license="CC BY-SA 4.0", uses_osm_odbl=False),
        DatasetEntry(
            name="ICG_Semantic_Drone", split="eval", license="Non-Commercial", uses_osm_odbl=False
        ),
        DatasetEntry(
            name="UAVScenes", split="eval", license="CC BY-NC-SA 4.0", uses_osm_odbl=False
        ),
    ]
    report = verify_training_manifest(entries, html)
    assert report.zero_sharealike_verified is True
    assert len(report.approved_train_entries) == 5
    assert set(report.quarantined_eval_entries) == {
        "AeroScapes",
        "ICG_Semantic_Drone",
        "UAVScenes",
    }
    assert len(report.oem_whitelisted_regions) == 41

    # Unlisted OEM region must raise LicenseViolationError
    bad_oem = DatasetEntry(
        name="OpenEarthMap",
        split="train",
        license="CC BY 4.0",
        uses_osm_odbl=False,
        regions=("aachen", "pd_region_00"),
    )
    with pytest.raises(LicenseViolationError):
        verify_training_manifest([bad_oem], html)


def test_skyscenes_wall_filter_canonical_maps_and_uavscenes_guard() -> None:
    mask = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        remap_mask("SkyScenes", mask)

    # Vertical wall facing horizontal forward camera => |n_z| == 0.0 < 0.85 => class 2
    rot_wc_wall = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    remapped_wall = remap_mask(
        "SkyScenes",
        mask,
        depth_m=np.full((4, 4), 10.0, np.float32),
        cam_rot_wc=rot_wc_wall,
        intrinsics=(305.09, 305.09, 2.0, 2.0),
    )
    assert np.all(remapped_wall[1:3, 1:3] == 2)

    # Flat roof viewed nadir downward => |n_z| == 1.0 >= 0.85 => remains class 1
    rot_wc_nadir = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    remapped_roof = remap_mask(
        "SkyScenes",
        mask,
        depth_m=np.full((4, 4), 22.0, np.float32),
        cam_rot_wc=rot_wc_nadir,
        intrinsics=(305.09, 305.09, 2.0, 2.0),
    )
    assert np.all(remapped_roof == 1)

    assert remap_mask("FLAIR-1", np.array([[1, 2, 5, 14, 18, 19]], np.uint8)).tolist() == [
        [1, 0, 4, 255, 2, 255]
    ]
    assert remap_mask(
        "FloodNet-Supervised-v1.0", np.array([[0, 1, 2, 4, 7, 9]], np.uint8)
    ).tolist() == [[255, 2, 1, 0, 3, 4]]
    assert remap_mask(
        "OpenEarthMap", np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]], np.uint8)
    ).tolist() == [[255, 0, 4, 0, 0, 4, 4, 4, 1]]
    assert remap_mask(
        "Gazebo_Istanbul_8K", np.array([[0, 10, 20, 30, 40, 50, 254]], np.uint8)
    ).tolist() == [[255, 0, 1, 2, 3, 4, 254]]
    assert remap_mask("AeroScapes", np.array([[0, 1, 7, 9, 10, 11]], np.uint8)).tolist() == [
        [255, 3, 2, 4, 0, 255]
    ]
    assert remap_mask(
        "ICG_Semantic_Drone", np.array([[0, 1, 3, 9, 10, 15, 21, 23]], np.uint8)
    ).tolist() == [[255, 0, 4, 1, 2, 3, 0, 255]]
    assert remap_mask(
        "UAVScenes", np.array([[0, 1, 2, 4, 6, 7, 15, 20, 24, 25]], np.uint8)
    ).tolist() == [[255, 1, 0, 4, 2, 255, 2, 3, 3, 255]]

    with pytest.raises(AssertionError):
        assert_uavscenes_cmap_ids({1: "road"})

    valid_uavscenes_cmap = {
        0: "background",
        1: "roof",
        2: "dirt_road",
        3: "paved_road",
        4: "river",
        5: "pool",
        6: "bridge",
        9: "container",
        10: "airstrip",
        11: "traffic_barrier",
        13: "green_field",
        14: "wild_field",
        15: "solar_panel",
        16: "umbrella",
        17: "transparent_roof",
        18: "car_park",
        19: "paved_walk",
        20: "sedan",
        24: "truck",
    }
    assert_uavscenes_cmap_ids(valid_uavscenes_cmap)
