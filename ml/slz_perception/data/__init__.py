# SPDX-License-Identifier: Apache-2.0
"""Permissive data governance, canonical 5-class SLZ taxonomy, and 8K generator."""

from ml.slz_perception.data.license_gate import (
    DatasetEntry,
    LicenseViolationError,
    ManifestReport,
    parse_oem_permissive_whitelist,
    verify_training_manifest,
)
from ml.slz_perception.data.remappers import (
    IGNORE_INDEX,
    NUM_CANONICAL_CLASSES,
    OOD_INDEX,
    assert_uavscenes_cmap_ids,
    compute_world_normals_from_depth,
    load_taxonomy,
    remap_mask,
)

__all__ = [
    "DatasetEntry",
    "IGNORE_INDEX",
    "LicenseViolationError",
    "ManifestReport",
    "NUM_CANONICAL_CLASSES",
    "OOD_INDEX",
    "assert_uavscenes_cmap_ids",
    "compute_world_normals_from_depth",
    "load_taxonomy",
    "parse_oem_permissive_whitelist",
    "remap_mask",
    "verify_training_manifest",
]
