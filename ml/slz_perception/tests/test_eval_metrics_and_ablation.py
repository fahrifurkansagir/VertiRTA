# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Plan 3 Task 7: Zero-shot real eval metrics, quarantine, and 60-seed ablation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ml.slz_perception.data.license_gate import LicenseViolationError
from ml.slz_perception.eval.aggregate_ablation_tables import (
    build_ablation_seed_manifest,
    check_gate5_pass,
    clopper_pearson_ci,
    merge_metrics_manifest,
)
from ml.slz_perception.eval.eval_zeroshot_real import (
    compute_free_energy_vs_maxlogit_ood,
    compute_monocular_slope_metrics,
    compute_slz_metrics,
    verify_eval_only_dataset,
)


def test_slz_metrics_balanced_60_seeds_and_gate5_checker(tmp_path: Path) -> None:
    m = compute_slz_metrics(
        np.array([[0, 1], [1, 3]], np.uint8),
        np.array([[0, 1], [2, 3]], np.uint8),
        np.array([[-10.0, -9.0], [-2.0, +3.0]], np.float32),
        np.array([[False, False], [True, True]]),
    )
    assert abs(m["false_safe_rate"] - 0.5) < 1e-6
    assert abs(m["ood_auroc"] - 1.0) < 1e-9
    assert abs(m["ood_fpr95"] - 0.0) < 1e-9
    assert 0.0 < m["miou"] < 1.0
    assert abs(m["slz_f1"] - (4.0 / 5.0)) < 1e-6

    seeds = build_ablation_seed_manifest(60)
    assert len(seeds) == 60
    assert [
        sum(s["bucket"] == b for s in seeds)
        for b in ("clean_roof", "dynamic_hazard", "ood_glare_fog")
    ] == [20, 20, 20]
    assert all(not s["uses_osm_odbl"] and s["datum_lat"] == 40.8860 and s["datum_lon"] == 29.1880 for s in seeds)
    assert len({s["seed"] for s in seeds}) == 60

    ok = {
        "aeroscape_miou": 0.65,
        "aeroscape_fsr": 0.03,
        "icg_miou": 0.63,
        "ood_auroc": 0.91,
        "active_safe_touchdown": 0.933,
        "active_false_commit": 0.017,
        "active_max_soc_pct": 0.95,
        "wsl2_peak_vram_mb": 238.0,
    }
    assert check_gate5_pass(ok) == []
    assert "active_false_commit" in check_gate5_pass(dict(ok, active_false_commit=0.05))
    assert "wsl2_peak_vram_mb" in check_gate5_pass(dict(ok, wsl2_peak_vram_mb=320.0))
    assert "active_max_soc_pct" in check_gate5_pass(dict(ok, active_max_soc_pct=1.25))
    assert "aeroscape_miou" in check_gate5_pass(dict(ok, aeroscape_miou=0.55))

    # Pure manifest merger writes only when explicit out_path is supplied
    merged = merge_metrics_manifest(
        v1_atlas={"csfl_n": 6500, "cel_n": 1500},
        v2a_copilot={"conformal_q_level": 0.955},
        v2b_slz={"gate5_failures": check_gate5_pass(ok), **ok},
        out_path=tmp_path / "merged_manifest.json",
    )
    assert merged["v2b_slz"]["gate5_failures"] == []
    assert json.loads((tmp_path / "merged_manifest.json").read_text(encoding="utf-8"))["v1_atlas"]["csfl_n"] == 6500


def test_eval_quarantine_uavscenes_cmap_and_ood_comparison() -> None:
    # Eval-only quarantined sets pass ONLY under split="eval" and fail under split="train"
    for ds in ("AeroScapes", "ICG_Semantic_Drone"):
        verify_eval_only_dataset(ds, split="eval")
        with pytest.raises(LicenseViolationError):
            verify_eval_only_dataset(ds, split="train")

    # UAVScenes strictly requires verified sparse cmap.py dictionary
    with pytest.raises(ValueError):
        verify_eval_only_dataset("UAVScenes", split="eval", uavscenes_cmap=None)
    with pytest.raises(AssertionError):
        verify_eval_only_dataset("UAVScenes", split="eval", uavscenes_cmap={1: "road"})
    with pytest.raises(LicenseViolationError):
        verify_eval_only_dataset("UAVScenes", split="train", uavscenes_cmap={1: "roof"})

    valid_cmap = {
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
    verify_eval_only_dataset("UAVScenes", split="eval", uavscenes_cmap=valid_cmap)

    # Free Energy vs MaxLogit comparison on synthetic logits
    rng = np.random.default_rng(42)
    logits = np.zeros((20, 20, 5), dtype=np.float32)
    ood_mask = np.zeros((20, 20), dtype=bool)
    logits[:10, :, 1] = 10.0 + rng.normal(0.0, 0.3, size=(10, 20))  # confident ID roof
    logits[10:, :, :] = rng.normal(0.0, 0.4, size=(10, 20, 5))  # low-logit OOD
    ood_mask[10:, :] = True
    bdry = np.zeros((20, 20), dtype=np.float32)
    ood_cmp = compute_free_energy_vs_maxlogit_ood(logits, bdry, ood_mask, t_ood=1.0, kappa_b=1.5)
    assert ood_cmp["free_energy_auroc"] > 0.99
    assert ood_cmp["maxlogit_auroc"] > 0.99
    assert ood_cmp["free_energy_fpr95"] < 0.05

    # Monocular Mod B LiDAR slope MAE & AbsRel
    mono = compute_monocular_slope_metrics(
        pred_slope_deg=np.array([5.5, 6.2, 4.8], dtype=np.float32),
        gt_lidar_slope_deg=np.array([5.0, 5.0, 5.0], dtype=np.float32),
        pred_depth_m=np.array([26.0, 24.0, 25.5], dtype=np.float32),
        gt_lidar_depth_m=np.array([25.0, 25.0, 25.0], dtype=np.float32),
    )
    assert abs(mono["slope_mae_deg"] - ((0.5 + 1.2 + 0.2) / 3.0)) < 1e-5
    assert abs(mono["depth_abs_rel"] - ((1.0 + 1.0 + 0.5) / 75.0)) < 1e-5

    # Clopper-Pearson exact 95% CI sanity
    lo, hi = clopper_pearson_ci(56, 60)
    assert 0.83 < lo < 0.9333 < hi < 0.99
    assert math.isclose(clopper_pearson_ci(0, 60)[0], 0.0, abs_tol=1e-12)
    assert math.isclose(clopper_pearson_ci(60, 60)[1], 1.0, abs_tol=1e-12)
