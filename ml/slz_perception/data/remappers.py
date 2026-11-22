# SPDX-License-Identifier: Apache-2.0
"""Canonical 5-class SLZ taxonomy remappers, SkyScenes depth-normal wall filter, and UAVScenes guard."""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

NUM_CANONICAL_CLASSES = 5
OOD_INDEX = 254
IGNORE_INDEX = 255

_TAXONOMY_PATH = Path(__file__).with_name("taxonomy.json")

_DATASET_ALIASES: dict[str, tuple[str, str]] = {
    "flair1": ("train_datasets_permissive_only", "FLAIR_1_etalab2"),
    "flair12": ("train_datasets_permissive_only", "FLAIR_1_etalab2"),
    "flair1etalab2": ("train_datasets_permissive_only", "FLAIR_1_etalab2"),
    "openearthmap": ("train_datasets_permissive_only", "OpenEarthMap_filtered_41_entries"),
    "openearthmap41": ("train_datasets_permissive_only", "OpenEarthMap_filtered_41_entries"),
    "openearthmapfiltered41entries": (
        "train_datasets_permissive_only",
        "OpenEarthMap_filtered_41_entries",
    ),
    "oem41": ("train_datasets_permissive_only", "OpenEarthMap_filtered_41_entries"),
    "floodnetsupervisedv10": ("train_datasets_permissive_only", "FloodNet_Supervised_v1_0"),
    "floodnetsupervised": ("train_datasets_permissive_only", "FloodNet_Supervised_v1_0"),
    "skyscenes": ("train_datasets_permissive_only", "SkyScenes_MIT"),
    "skyscenesmit": ("train_datasets_permissive_only", "SkyScenes_MIT"),
    "gazeboistanbul8k": ("train_datasets_permissive_only", "Gazebo_Istanbul_8K_Author"),
    "gazeboistanbul8kauthor": ("train_datasets_permissive_only", "Gazebo_Istanbul_8K_Author"),
    "aeroscape": ("eval_only_datasets_never_trained", "AeroScapes_CC_BY_SA_4"),
    "aeroscapes": ("eval_only_datasets_never_trained", "AeroScapes_CC_BY_SA_4"),
    "aeroscapesccbysa4": ("eval_only_datasets_never_trained", "AeroScapes_CC_BY_SA_4"),
    "icg": ("eval_only_datasets_never_trained", "ICG_Semantic_Drone_NC"),
    "icgsemanticdrone": ("eval_only_datasets_never_trained", "ICG_Semantic_Drone_NC"),
    "icgsemanticdronenc": ("eval_only_datasets_never_trained", "ICG_Semantic_Drone_NC"),
    "uavscenes": ("eval_only_datasets_never_trained", "UAVScenes_CC_BY_NC_SA_4"),
    "uavscenesccbyncsa4": ("eval_only_datasets_never_trained", "UAVScenes_CC_BY_NC_SA_4"),
}


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


@functools.lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    """Load and cache canonical 5-class taxonomy JSON."""
    return json.loads(_TAXONOMY_PATH.read_text(encoding="utf-8"))


def _build_lut(mapping: dict[str, int]) -> np.ndarray:
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for k_str, v in mapping.items():
        idx = int(k_str)
        if 0 <= idx < 256:
            lut[idx] = np.uint8(v)
    return lut


def compute_world_normals_from_depth(
    depth_m: np.ndarray,
    cam_rot_wc: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute unit surface normals in world ENU frame from a pinhole depth map.

    Args:
        depth_m: (H, W) float depth map in meters along camera optical +Z axis.
        cam_rot_wc: (3, 3) rotation matrix mapping camera optical frame to world ENU.
        intrinsics: (fx, fy, cx, cy) in pixels.

    Returns:
        (normals_w, valid_mask) where normals_w has shape (H, W, 3) float32 unit vectors
        and valid_mask has shape (H, W) bool.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    rot_wc = np.asarray(cam_rot_wc, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth_m must be 2D (H, W), got shape {depth.shape}")
    if rot_wc.shape != (3, 3):
        raise ValueError(f"cam_rot_wc must have shape (3, 3), got {rot_wc.shape}")
    if len(intrinsics) != 4:
        raise ValueError(f"intrinsics must be (fx, fy, cx, cy), got length {len(intrinsics)}")

    fx, fy, cx, cy = (float(x) for x in intrinsics)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Focal lengths must be positive, got fx={fx}, fy={fy}")

    h, w = depth.shape
    if h < 2 or w < 2:
        raise ValueError(f"depth_m must be at least 2x2 for finite differences, got {depth.shape}")

    u_grid, v_grid = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    finite_pos = np.isfinite(depth) & (depth > 0.0)
    z_safe = np.where(finite_pos, depth, np.float32(0.0))

    p_c = np.stack(
        [
            z_safe * (u_grid - np.float32(cx)) / np.float32(fx),
            z_safe * (v_grid - np.float32(cy)) / np.float32(fy),
            z_safe,
        ],
        axis=-1,
    )

    dp_dv = np.gradient(p_c, axis=0)
    dp_du = np.gradient(p_c, axis=1)
    # Cross product dP/dv x dP/du points toward camera (-Z_c) for fronto-parallel surfaces
    n_c = np.cross(dp_dv, dp_du)
    norms = np.linalg.norm(n_c, axis=-1, keepdims=True)
    valid = finite_pos & (norms[..., 0] > 1e-8)
    n_c_unit = np.where(valid[..., None], n_c / np.maximum(norms, np.float32(1e-8)), np.float32(0.0))

    n_w = np.einsum("ij,hwj->hwi", rot_wc, n_c_unit)
    w_norms = np.linalg.norm(n_w, axis=-1, keepdims=True)
    valid = valid & (w_norms[..., 0] > 1e-8) & np.all(np.isfinite(n_w), axis=-1)
    n_w_unit = np.where(valid[..., None], n_w / np.maximum(w_norms, np.float32(1e-8)), np.float32(0.0))
    return n_w_unit.astype(np.float32, copy=False), valid


def assert_uavscenes_cmap_ids(cmap_dict: dict[int, str]) -> None:
    """Assert that a UAVScenes cmap.py ID dictionary matches the true sparse 0..25 ID layout.

    Prevents silent label corruption from dense 0..18 indexing or misaligned class names.
    Raises AssertionError on any mismatch.
    """
    tax = load_taxonomy()["eval_only_datasets_never_trained"]["UAVScenes_CC_BY_NC_SA_4"]
    expected_keywords: dict[str, list[str]] = tax["expected_active_cmap_keywords"]
    sparse_gaps: set[int] = set(int(i) for i in tax["sparse_gap_ids"])

    normalized: dict[int, str] = {int(k): str(v).strip().lower() for k, v in cmap_dict.items()}

    missing = [int(k) for k in expected_keywords if int(k) not in normalized]
    assert not missing, (
        f"UAVScenes cmap_dict missing required active sparse IDs: {missing}"
    )

    for gap_id in sorted(sparse_gaps):
        if gap_id in normalized:
            label = normalized[gap_id]
            assert label in ("", "unused", "empty", "ignore", "none", "unlabeled"), (
                f"UAVScenes sparse gap ID {gap_id} unexpectedly mapped to active label '{label}'"
            )

    for id_str, keywords in expected_keywords.items():
        cid = int(id_str)
        actual_name = normalized[cid]
        assert any(kw in actual_name for kw in keywords), (
            f"UAVScenes cmap ID {cid} name '{actual_name}' does not match expected keywords {keywords}"
        )


def remap_mask(
    dataset: str,
    mask: np.ndarray,
    *,
    depth_m: np.ndarray | None = None,
    cam_rot_wc: np.ndarray | None = None,
    intrinsics: tuple[float, float, float, float] | None = None,
    skyscenes_nz_threshold: float = 0.85,
    oem_zero_indexed: bool = False,
    icg_one_indexed: bool = False,
) -> np.ndarray:
    """Remap a raw dataset semantic mask to VertiRTA canonical 5-class + {254, 255} IDs.

    SkyScenes strictly requires depth_m, cam_rot_wc, and intrinsics so vertical building
    facades (|n_z| < 0.85) are reassigned from 1 (Safe_Flat_Roof) to 2 (Hazard_Obstacle_Structure).
    """
    key = _normalize_key(dataset)
    if key not in _DATASET_ALIASES:
        raise ValueError(f"Unsupported or banned dataset for remapping: '{dataset}'")

    group, canonical_key = _DATASET_ALIASES[key]
    tax_entry = load_taxonomy()[group][canonical_key]

    raw = np.asarray(mask)
    if raw.ndim != 2:
        raise ValueError(f"mask must be 2D (H, W), got shape {raw.shape}")

    if canonical_key == "OpenEarthMap_filtered_41_entries":
        mapping_dict = (
            tax_entry["zero_indexed_loader_0_to_7"]
            if oem_zero_indexed
            else tax_entry["raw_pixel_values_0_to_8"]
        )
    elif canonical_key == "Gazebo_Istanbul_8K_Author":
        mapping_dict = tax_entry["sdf_label_ids"]
    else:
        mapping_dict = tax_entry["mapping"]

    raw_int = raw.astype(np.int64, copy=False)
    if canonical_key == "ICG_Semantic_Drone_NC" and icg_one_indexed:
        raw_int = raw_int - 1

    lut = _build_lut(mapping_dict)
    in_range = (raw_int >= 0) & (raw_int < 256)
    safe_idx = np.where(in_range, raw_int, 0)
    out = np.where(in_range, lut[safe_idx], np.uint8(IGNORE_INDEX)).astype(np.uint8, copy=False)

    if canonical_key == "SkyScenes_MIT":
        if depth_m is None or cam_rot_wc is None or intrinsics is None:
            raise ValueError(
                "SkyScenes remapping requires depth_m, cam_rot_wc, and intrinsics "
                "to filter vertical building facades (|n_z| < 0.85 -> class 2)."
            )
        if np.asarray(depth_m).shape != raw.shape:
            raise ValueError(
                f"SkyScenes depth_m shape {np.asarray(depth_m).shape} != mask shape {raw.shape}"
            )
        normals_w, valid_n = compute_world_normals_from_depth(depth_m, cam_rot_wc, intrinsics)
        is_building = out == np.uint8(1)
        is_facade_or_invalid = (~valid_n) | (np.abs(normals_w[..., 2]) < np.float32(skyscenes_nz_threshold))
        out = out.copy()
        out[is_building & is_facade_or_invalid] = np.uint8(2)

    return out
