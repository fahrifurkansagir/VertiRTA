#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local Physical GTX 1650 SUPER (4 GB, TU116) PIDNet-S v2.1 Recovery Trainer & HUD Renderer.

Executes a 0.00 CU local CUDA FP16 AMP recovery training of Static-PAPPM PIDNet-S v2.1
(8,003,817 deploy parameters) to replace the collapsed Epoch-2 all-hazard G4 checkpoint
without touching G4's original historical ablation artifacts.

Key guarantees:
  - Strict Zero-ShareAlike startup license firewall (`FLAIR-1` OPEN LICENCE 2.0,
    `OpenEarthMap` permissive CC BY 4.0 / DL-DE-BY-2.0 whitelist via live `attribution.html`,
    `FloodNet-Supervised-v1.0` CDLA-Permissive-1.0, `SkyScenes` MIT).
  - Quarantined `AeroScapes` (CC BY-SA 4.0) is NEVER trained or calibrated — strictly
    post-freeze zero-shot evaluation only.
  - Reuses battle-tested loaders & calibration from `scripts/colab/run_l4_pidnet_distill.py`
    and fixed Pareto selector (`select_pareto_checkpoint(..., min_miou=0.25)`),
    `calibrate_safe_logit_bias_for_fsr(..., target_max_fsr=0.030)`,
    `ObliqueBalancedFocalFsrLoss`, and `apply_oblique_perspective_aug` from
    `scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py`.
  - Fits safely inside 4 GB TU116 VRAM (`teachers=False`, FP16 AMP + GradScaler,
    `batch_size=4`, `grad_accum=4` -> effective `bs=16`).
  - Exports static 1x3x512x512 Opset-17 ONNX (`pidnet_slz_v21_local_gtx_opset17.onnx`) +
    calibration JSON (`ood_calibration_v21_local_gtx.json`) + SHA-256 sealed metrics JSON
    with head-to-head comparison against L4 v2 and G4 v2.1 + optional H.264 HUD preview MP4
    and 6-scene contact sheet PNG outside git.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.slz_perception.data.remappers import (  # noqa: E402
    NUM_CANONICAL_CLASSES,
    OOD_INDEX,
    remap_mask,
)
from scripts.colab.g4_jobs.common import (  # noqa: E402
    query_nvidia_smi_snapshot,
    seed_everything,
    sha256_file,
    validate_job_license_manifest,
    write_canonical_json,
)
from scripts.colab.g4_jobs.job_pidnet_v21_xfeat import (  # noqa: E402
    _build_oblique_loss_and_xfeat,
    _export_and_check_onnx,
    _pareto_checkpoint_sort_key,
    _shutdown_loader,
    apply_oblique_perspective_aug,
    build_pidnet_xfeat_license_manifest,
    calibrate_safe_logit_bias_for_fsr,
    select_pareto_checkpoint,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scripts.colab.run_l4_pidnet_distill import SampleRecord

CANONICAL_CLASS_NAMES: tuple[str, ...] = (
    "Safe_Flat_Ground",
    "Safe_Flat_Roof",
    "Hazard_Obstacle_Structure",
    "Hazard_Dynamic_Human_Vehicle",
    "Hazard_Water_Vegetation",
)
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

DEFAULT_WORK_DIR = Path("/var/tmp/vertirta_local_pidnet_work")
DEFAULT_BLOB_DIR = Path("/var/tmp/vertirta_v400_blobs")
DEFAULT_CACHED_FLAIR_ZIP = Path("/var/tmp/vertirta_local_xfeat_cache/flair_1_toy_dataset.zip")
DEFAULT_OUT_JSON = (
    REPO_ROOT / "artifacts" / "v4.0.0" / "local_gtx1650s_pidnet_v21_recovery_metrics.json"
)
DEFAULT_L4_METRICS = REPO_ROOT / "artifacts" / "v3.5.1" / "l4_pidnet_v2_distill_metrics.json"
DEFAULT_G4_METRICS = REPO_ROOT / "artifacts" / "v4.0.0" / "g4_pidnet_v21_xfeat_metrics.json"

ONNX_FILENAME = "pidnet_slz_v21_local_gtx_opset17.onnx"
CAL_FILENAME = "ood_calibration_v21_local_gtx.json"

# Unified VertiRTA v1.0.0 5-class avionic RGB palette (shared verbatim with NASA drone HUD):
#   0: Safe_Flat_Ground              -> Emerald Green
#   1: Safe_Flat_Roof                -> Avionic Cyan (max contrast vs orange facade hazard)
#   2: Hazard_Obstacle_Structure     -> Vivid Orange
#   3: Hazard_Dynamic_Human_Vehicle  -> Magenta
#   4: Hazard_Water_Vegetation       -> Deep Cobalt Blue
SLZ_CLASS_RGB: np.ndarray = np.array(
    [
        [52, 208, 88],    # 0: Safe_Flat_Ground (Emerald Green)
        [42, 212, 228],   # 1: Safe_Flat_Roof (Avionic Cyan)
        [245, 130, 32],   # 2: Hazard_Obstacle_Structure (Vivid Orange)
        [225, 52, 185],   # 3: Hazard_Dynamic_Human_Vehicle (Magenta)
        [42, 98, 215],    # 4: Hazard_Water_Vegetation (Deep Cobalt Blue)
    ],
    dtype=np.uint8,
)

SLZ_SHORT_NAMES: tuple[str, ...] = (
    "Ground",
    "Roof",
    "Obstacle",
    "Dynamic",
    "Water/Veg",
)


def sanitize_artifact_path(path: Path | str | None) -> str | None:
    """Return a clean repo-relative, /var/tmp, or basename path with zero personal paths."""
    if path is None:
        return None
    p = Path(path)
    try:
        rel = p.resolve().relative_to(REPO_ROOT.resolve())
        return rel.as_posix()
    except Exception:
        pass
    posix = p.as_posix()
    if posix.startswith("/var/tmp/"):
        return posix
    return p.name


class DisabledTeachersStub:
    """Zero-VRAM teacher stub (`teachers=False`) compatible with `evaluate_permissive_val_and_ood`."""

    def __init__(self) -> None:
        self.enabled = False
        self.dino = None
        self.dav2 = None
        self.metadata: dict[str, Any] = {
            "enabled": False,
            "mode": "supervised_5class_boundary_and_skyscenes_metric_normals",
            "reason": "Fits comfortably within physical 4 GB GTX 1650 SUPER TU116 VRAM",
        }

    def infer(
        self,
        rgb_512: Any,
        gt_normals: Any,
        has_gt_normal: Any,
        *,
        use_bf16: bool = False,
    ) -> tuple[Any, Any]:
        del rgb_512, has_gt_normal, use_bf16
        return None, gt_normals


def _build_safe_bias_eval_wrapper(model: Any, safe_logit_bias_delta: float) -> Any:
    """Wrap a PIDNetSlzTrain model to add frozen `safe_logit_bias_delta` to classes {0, 1}."""
    import torch
    import torch.nn as nn

    from ml.slz_perception.models.pidnet_slz import PIDNetTrainOutput

    class SafeLogitBiasEvalWrapper(nn.Module):
        def __init__(self, inner: nn.Module, delta: float) -> None:
            super().__init__()
            self.inner = inner
            self.delta = float(delta)

        def forward(self, x: torch.Tensor) -> PIDNetTrainOutput:
            out = self.inner(x)
            if self.delta == 0.0:
                return out
            logits = out.seg_logits.clone()
            logits[:, 0:2, :, :] = logits[:, 0:2, :, :] + self.delta
            return PIDNetTrainOutput(
                seg_logits=logits,
                surf_normals=out.surf_normals,
                bdry_prob=out.bdry_prob,
                dino_feats=out.dino_feats,
            )

    return SafeLogitBiasEvalWrapper(model, safe_logit_bias_delta).eval()


def _prepare_flair1_with_cache(
    work_dir: Path,
    max_samples: int,
    cached_zip: Path | None = DEFAULT_CACHED_FLAIR_ZIP,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    """Prepare FLAIR-1 reusing cached official IGN archive when present (never deletes cache)."""
    from scripts.colab.run_l4_pidnet_distill import (
        FLAIR1_TOY_URL,
        SampleRecord,
        _assign_group_splits,
        _prepare_flair1,
        _save_pair_512,
    )

    if cached_zip is None or not cached_zip.is_file():
        return _prepare_flair1(work_dir, max_samples=max_samples)

    import tifffile

    zip_sha = sha256_file(cached_zip)
    zip_mb = round(cached_zip.stat().st_size / (1024.0 * 1024.0), 2)
    extract_dir = work_dir / "raw" / "flair1" / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[FLAIR-1] Reusing cached official IGN etalab-2.0 archive ({zip_mb} MB, sha256={zip_sha[:12]}...)",
        flush=True,
    )
    with zipfile.ZipFile(cached_zip, "r") as zf:
        zf.extractall(extract_dir)

    img_files = sorted(extract_dir.rglob("IMG_*.tif"))
    msk_by_name = {p.name: p for p in extract_dir.rglob("MSK_*.tif")}
    pairs: list[tuple[Path, Path, str]] = []
    for img_p in img_files:
        msk_name = img_p.name.replace("IMG_", "MSK_", 1)
        cand = msk_by_name.get(msk_name)
        if cand is not None and cand.is_file():
            zone = img_p.parent.parent.name
            pairs.append((img_p, cand, zone))

    if max_samples > 0 and len(pairs) > max_samples:
        stride = np.linspace(0, len(pairs) - 1, max_samples, dtype=int)
        pairs = [pairs[i] for i in stride]

    val_zones = _assign_group_splits([z for _, _, z in pairs], val_frac=0.15, seed=101)
    out_root = work_dir / "prepared" / "flair1"
    records: list[SampleRecord] = []
    for idx, (img_p, msk_p, zone) in enumerate(pairs):
        arr = tifffile.imread(str(img_p))
        if arr.ndim == 3 and arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = np.transpose(arr, (1, 2, 0))
        rgb = arr[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        msk_raw = tifffile.imread(str(msk_p))
        if msk_raw.ndim == 3:
            msk_raw = msk_raw[..., 0] if msk_raw.shape[2] <= 4 else msk_raw[0]
        msk_canon = remap_mask("FLAIR-1", msk_raw)

        split = "val" if zone in val_zones else "train"
        rgb_out = out_root / split / "rgb" / f"flair1_{idx:05d}.jpg"
        msk_out = out_root / split / "mask" / f"flair1_{idx:05d}.png"
        _save_pair_512(rgb, msk_canon, rgb_out, msk_out)
        records.append(
            SampleRecord(
                source="FLAIR-1",
                split=split,
                rgb_path=str(rgb_out),
                mask_path=str(msk_out),
                group=zone,
            )
        )

    shutil.rmtree(extract_dir, ignore_errors=True)
    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    print(f"[FLAIR-1] Prepared {len(records)} samples from cache (train={n_train}, val={n_val})", flush=True)
    return records, {
        "source": "FLAIR-1",
        "license": "OPEN LICENCE 2.0 (etalab-2.0)",
        "uses_osm_odbl": False,
        "archive_url": FLAIR1_TOY_URL,
        "reused_local_cache": True,
        "archive_mb": zip_mb,
        "archive_sha256": zip_sha,
        "train_samples": n_train,
        "val_samples": n_val,
        "unique_zones": len(set(r.group for r in records)),
    }


def _save_prepared_manifest(
    work_dir: Path,
    records: list[SampleRecord],
    dataset_provenance: list[dict[str, Any]],
) -> None:
    manifest_path = work_dir / "prepared_manifest.json"
    payload = {
        "records": [
            {
                "source": r.source,
                "split": r.split,
                "rgb_path": r.rgb_path,
                "mask_path": r.mask_path,
                "normal_path": r.normal_path,
                "group": r.group,
            }
            for r in records
        ],
        "dataset_provenance": dataset_provenance,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _load_prepared_manifest(work_dir: Path) -> tuple[list[SampleRecord], list[dict[str, Any]]] | None:
    from scripts.colab.run_l4_pidnet_distill import SampleRecord

    manifest_path = work_dir / "prepared_manifest.json"
    if not manifest_path.is_file():
        return None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    recs: list[SampleRecord] = []
    for item in raw.get("records", []):
        if not Path(item["rgb_path"]).is_file() or not Path(item["mask_path"]).is_file():
            return None
        recs.append(
            SampleRecord(
                source=str(item["source"]),
                split=str(item["split"]),
                rgb_path=str(item["rgb_path"]),
                mask_path=str(item["mask_path"]),
                normal_path=item.get("normal_path"),
                group=str(item.get("group", "")),
            )
        )
    if not recs:
        return None
    return recs, list(raw.get("dataset_provenance", []))


def build_head_to_head_comparison(
    *,
    local_raw_val: dict[str, Any],
    local_cal_val: dict[str, Any],
    local_ood: dict[str, Any],
    local_raw_aero: dict[str, Any],
    local_cal_aero: dict[str, Any],
    local_bias_delta: float,
    l4_metrics_path: Path = DEFAULT_L4_METRICS,
    g4_metrics_path: Path = DEFAULT_G4_METRICS,
) -> dict[str, Any]:
    """Assemble honest head-to-head table comparing L4 v2, G4 v2.1 (collapsed), and Local GTX recovery."""
    l4_row: dict[str, Any] = {"available": False}
    if l4_metrics_path.is_file():
        l4 = json.loads(l4_metrics_path.read_text(encoding="utf-8"))
        l4_val = l4["evaluation"]["permissive_validation"]
        l4_ood = l4["evaluation"]["ood_detection"]
        l4_aero = l4["evaluation"]["quarantined_zero_shot_aeroscapes"]
        l4_row = {
            "available": True,
            "source_artifact": sanitize_artifact_path(l4_metrics_path),
            "hardware": l4["hardware"]["gpu"]["name"],
            "cu_cost_category": "COLAB_L4",
            "teachers_enabled": bool(l4["teachers"]["enabled"]),
            "selected_epoch": l4["training"]["stage2_ood_oe"]["selected_epoch"],
            "val_miou": float(l4_val["miou_present_classes"]),
            "val_fsr": float(l4_val["false_safe_rate_fsr"]),
            "val_safe_precision": float(l4_val["safe_precision"]),
            "ood_auroc_comp": float(l4_ood["boundary_compensated_auroc"]),
            "aeroscapes_miou": float(l4_aero["miou_present_classes"]),
            "aeroscapes_fsr": float(l4_aero["false_safe_rate_fsr"]),
            "safe_logit_bias_delta": 0.0,
            "status": "HEALTHY_REFERENCE_L4_DISTILLED",
        }

    g4_row: dict[str, Any] = {"available": False}
    if g4_metrics_path.is_file():
        g4 = json.loads(g4_metrics_path.read_text(encoding="utf-8"))
        g4_val = g4["permissive_val_metrics"]
        g4_ood = g4["ood_metrics"]
        g4_aero = g4["quarantined_zero_shot_aeroscapes"]
        g4_hist = g4.get("pidnet_v21_training", {}).get("pareto_history", [])
        g4_best_ep = max(g4_hist, key=lambda r: float(r["val_miou"])) if g4_hist else None
        g4_row = {
            "available": True,
            "source_artifact": sanitize_artifact_path(g4_metrics_path),
            "hardware": g4["hardware"]["vram_config"]["gpu_name"],
            "cu_cost_category": "COLAB_G4_HISTORICAL_ABLATION_UNTOUCHED",
            "teachers_enabled": bool(g4["teachers"]["enabled"]),
            "selected_epoch": int(round(g4["pidnet_v21_training"]["pareto_selection"]["selected"]["epoch"])),
            "val_miou": float(g4_val["miou_present_classes"]),
            "val_fsr": float(g4_val["false_safe_rate_fsr"]),
            "val_safe_precision": float(g4_val["safe_precision"]),
            "ood_auroc_comp": float(g4_ood["boundary_compensated_auroc"]),
            "aeroscapes_miou": float(g4_aero["miou_present_classes"]),
            "aeroscapes_fsr": float(g4_aero["false_safe_rate_fsr"]),
            "safe_logit_bias_delta": 0.0,
            "unselected_peak_epoch_miou": (
                float(g4_best_ep["val_miou"]) if g4_best_ep is not None else None
            ),
            "status": "COLLAPSED_EPOCH2_ALL_HAZARD_SELECTOR_BUG",
        }

    local_raw_row = {
        "hardware": "NVIDIA GeForce GTX 1650 SUPER (4 GB, TU116)",
        "cu_cost_category": "LOCAL_0_00_CU",
        "teachers_enabled": False,
        "val_miou": float(local_raw_val["miou_present_classes"]),
        "val_fsr": float(local_raw_val["false_safe_rate_fsr"]),
        "val_safe_precision": float(local_raw_val["safe_precision"]),
        "ood_auroc_comp": float(local_ood.get("boundary_compensated_auroc", 0.0)),
        "aeroscapes_miou": float(local_raw_aero["miou_present_classes"]),
        "aeroscapes_fsr": float(local_raw_aero["false_safe_rate_fsr"]),
        "safe_logit_bias_delta": 0.0,
    }
    local_cal_row = {
        "hardware": "NVIDIA GeForce GTX 1650 SUPER (4 GB, TU116)",
        "cu_cost_category": "LOCAL_0_00_CU",
        "teachers_enabled": False,
        "val_miou": float(local_cal_val["miou_present_classes"]),
        "val_fsr": float(local_cal_val["false_safe_rate_fsr"]),
        "val_safe_precision": float(local_cal_val["safe_precision"]),
        "ood_auroc_comp": float(local_ood.get("boundary_compensated_auroc", 0.0)),
        "aeroscapes_miou": float(local_cal_aero["miou_present_classes"]),
        "aeroscapes_fsr": float(local_cal_aero["false_safe_rate_fsr"]),
        "safe_logit_bias_delta": float(local_bias_delta),
    }

    delta_vs_g4: dict[str, float] = {}
    if g4_row.get("available"):
        delta_vs_g4 = {
            "val_miou_raw_gain": round(local_raw_row["val_miou"] - g4_row["val_miou"], 6),
            "val_miou_cal_gain": round(local_cal_row["val_miou"] - g4_row["val_miou"], 6),
            "aeroscapes_miou_raw_gain": round(
                local_raw_row["aeroscapes_miou"] - g4_row["aeroscapes_miou"], 6
            ),
            "aeroscapes_miou_cal_gain": round(
                local_cal_row["aeroscapes_miou"] - g4_row["aeroscapes_miou"], 6
            ),
            "ood_auroc_gain": round(
                local_cal_row["ood_auroc_comp"] - g4_row["ood_auroc_comp"], 6
            ),
        }

    return {
        "l4_v2_distilled": l4_row,
        "g4_v21_collapsed_epoch2": g4_row,
        "local_gtx1650s_v21_recovery_raw": local_raw_row,
        "local_gtx1650s_v21_recovery_bias_calibrated": local_cal_row,
        "delta_local_vs_g4_collapsed": delta_vs_g4,
    }


def _select_balanced_preview_records(
    val_recs: list[SampleRecord],
    total_scenes: int = 36,
) -> list[SampleRecord]:
    """Select up to `total_scenes` permissive validation images balanced across all 4 sources."""
    by_src: dict[str, list[SampleRecord]] = {}
    for r in val_recs:
        by_src.setdefault(r.source, []).append(r)

    source_order = [
        "SkyScenes",
        "OpenEarthMap-41",
        "FloodNet-Supervised-v1.0",
        "FLAIR-1",
    ]
    ordered_keys = [s for s in source_order if s in by_src] + [
        s for s in sorted(by_src) if s not in source_order
    ]
    if not ordered_keys:
        return []

    per_src_quota = max(1, total_scenes // len(ordered_keys))
    buckets: dict[str, list[SampleRecord]] = {}
    leftovers: list[SampleRecord] = []
    for src in ordered_keys:
        items = sorted(by_src[src], key=lambda r: r.rgb_path)
        if len(items) <= per_src_quota:
            buckets[src] = list(items)
        else:
            idx = np.linspace(0, len(items) - 1, per_src_quota, dtype=int)
            chosen_set = set(idx.tolist())
            buckets[src] = [items[i] for i in idx]
            leftovers.extend([items[i] for i in range(len(items)) if i not in chosen_set])

    # Round-robin interleave so video transitions smoothly across diverse domains
    selected: list[SampleRecord] = []
    max_len = max(len(v) for v in buckets.values())
    for i in range(max_len):
        for src in ordered_keys:
            if i < len(buckets[src]):
                selected.append(buckets[src][i])

    if len(selected) < total_scenes and leftovers:
        need = total_scenes - len(selected)
        stride = np.linspace(0, len(leftovers) - 1, min(need, len(leftovers)), dtype=int)
        selected.extend([leftovers[i] for i in stride])

    return selected[:total_scenes]


def _colorize_slz_mask_rgb(mask_hw: np.ndarray) -> np.ndarray:
    """Convert canonical 0..4 SLZ mask (with 255 ignore) into HxWx3 RGB uint8 image."""
    h, w = mask_hw.shape
    out = np.full((h, w, 3), 32, dtype=np.uint8)
    valid = (mask_hw >= 0) & (mask_hw < len(SLZ_CLASS_RGB))
    out[valid] = SLZ_CLASS_RGB[mask_hw[valid]]
    return out


def _draw_hud_frame_rgb(
    rgb_512: np.ndarray,
    gt_mask_512: np.ndarray,
    pred_mask_512: np.ndarray,
    *,
    source_name: str,
    scene_idx: int,
    total_scenes: int,
    latency_ms: float,
    safe_bias_delta: float,
    e_tilde_mean: float,
    tau_ood: float,
    ood_pct: float,
    miou_scene: float,
    fsr_scene: float,
) -> np.ndarray:
    """Render crisp 1536x640 RGB HUD frame: [Input Aerial | PIDNet-S v2.1 Overlay | Ground Truth]."""
    import cv2

    pred_rgb = _colorize_slz_mask_rgb(pred_mask_512)
    gt_rgb = _colorize_slz_mask_rgb(gt_mask_512)
    overlay_rgb = np.clip(
        0.66 * rgb_512.astype(np.float32) + 0.34 * pred_rgb.astype(np.float32), 0, 255
    ).astype(np.uint8)

    header_h = 64
    footer_h = 64
    canvas = np.full((header_h + 512 + footer_h, 1536, 3), (18, 22, 28), dtype=np.uint8)
    canvas[header_h : header_h + 512, 0:512] = rgb_512
    canvas[header_h : header_h + 512, 512:1024] = overlay_rgb
    canvas[header_h : header_h + 512, 1024:1536] = gt_rgb

    # Subtle panel dividers
    canvas[header_h : header_h + 512, 511:513] = (80, 92, 108)
    canvas[header_h : header_h + 512, 1023:1025] = (80, 92, 108)

    # Convert to BGR for cv2 text/rect drawing, then back to RGB
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX

    title = (
        f"VertiRTA v1.0.0  |  PIDNet-S v2.1 Local GTX 1650 SUPER Validation  |  "
        f"Scene {scene_idx + 1:02d}/{total_scenes:02d}: {source_name}"
    )
    cv2.putText(bgr, title, (16, 26), font, 0.58, (240, 245, 250), 1, cv2.LINE_AA)

    sub = (
        f"ONNX Opset-17 (8.00M params)  |  bias={safe_bias_delta:+.3f}  |  "
        f"scene mIoU={miou_scene:.3f}  |  FSR={fsr_scene * 100.0:.2f}%  |  "
        f"E_tilde={e_tilde_mean:.2f} (tau={tau_ood:.2f}, OOD={ood_pct:.1f}%)  |  "
        f"infer={latency_ms:.1f} ms"
    )
    cv2.putText(bgr, sub, (16, 50), font, 0.48, (160, 225, 195), 1, cv2.LINE_AA)

    # Panel badges
    badges = (
        (10, "INPUT RGB (512x512)"),
        (522, "PIDNet-S v2.1 PRED OVERLAY (BIAS-CAL)"),
        (1034, "CANONICAL 5-CLASS GROUND TRUTH"),
    )
    for bx, btxt in badges:
        cv2.rectangle(bgr, (bx, header_h + 8), (bx + 320, header_h + 32), (16, 20, 26), -1)
        cv2.putText(bgr, btxt, (bx + 8, header_h + 25), font, 0.45, (235, 240, 245), 1, cv2.LINE_AA)

    # Footer legend + per-class predicted coverage %
    total_px = float(pred_mask_512.size)
    y_leg = header_h + 512 + 20
    x_cur = 18
    for cid, cname in enumerate(SLZ_SHORT_NAMES):
        pct = float((pred_mask_512 == cid).sum()) * 100.0 / total_px
        rgb_col = SLZ_CLASS_RGB[cid]
        bgr_col = (int(rgb_col[2]), int(rgb_col[1]), int(rgb_col[0]))
        cv2.rectangle(bgr, (x_cur, y_leg - 12), (x_cur + 16, y_leg + 4), bgr_col, -1)
        cv2.rectangle(bgr, (x_cur, y_leg - 12), (x_cur + 16, y_leg + 4), (220, 225, 230), 1)
        label = f"C{cid} {cname}: {pct:4.1f}%"
        cv2.putText(bgr, label, (x_cur + 22, y_leg + 2), font, 0.48, (235, 240, 245), 1, cv2.LINE_AA)
        x_cur += 295

    safe_pct = float(((pred_mask_512 == 0) | (pred_mask_512 == 1)).sum()) * 100.0 / total_px
    haz_pct = 100.0 - safe_pct
    summary = (
        f"Predicted Safe (Ground+Roof): {safe_pct:5.1f}%   |   "
        f"Predicted Hazard (Obstacle+Dynamic+Water/Veg): {haz_pct:5.1f}%   |   "
        f"Zero-ShareAlike Permissive Validation"
    )
    cv2.putText(bgr, summary, (18, header_h + 512 + 50), font, 0.46, (180, 200, 220), 1, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _find_ffmpeg_executable() -> str | None:
    """Locate ffmpeg on PATH or standard Windows WinGet link accessible from WSL2."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    for cand in (
        Path("/usr/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ):
        if cand.is_file():
            return str(cand)
    # Check Windows WinGet links from WSL2 without hardcoding any username
    mnt_users = Path("/mnt/c/Users")
    if mnt_users.is_dir():
        for user_dir in sorted(mnt_users.iterdir()):
            cand = user_dir / "AppData/Local/Microsoft/WinGet/Links/ffmpeg.exe"
            if cand.is_file():
                return str(cand)
    return None


def render_hud_preview_and_contact_sheet(
    *,
    onnx_path: Path,
    calibration: dict[str, Any],
    val_recs: list[SampleRecord],
    mp4_out: Path | None,
    grid_out: Path | None,
    work_dir: Path,
    num_scenes: int = 36,
    use_cuda: bool = True,
) -> dict[str, Any]:
    """Run exported ONNX (+ safe_logit_bias_delta) over 36 real permissive val scenes & render H.264 + 6-scene PNG."""
    import cv2
    import onnxruntime as ort
    from PIL import Image

    selected = _select_balanced_preview_records(val_recs, total_scenes=num_scenes)
    if not selected:
        raise RuntimeError("No permissive validation records available for HUD preview rendering.")

    providers: list[Any] = []
    avail = ort.get_available_providers()
    if use_cuda and "CUDAExecutionProvider" in avail:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": "0",
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv_use_max_workspace": "0",
                },
            )
        )
    providers.append("CPUExecutionProvider")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=providers)
    inp_name = sess.get_inputs()[0].name

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    safe_bias = float(calibration.get("safe_logit_bias_delta", 0.0))
    t_ood = float(calibration.get("T_ood", 1.0))
    kappa_b = float(calibration.get("kappa_b", 1.5))
    tau_ood = float(calibration.get("tau_ood", -2.5))

    # Warmup ORT session
    dummy = np.zeros((1, 3, 512, 512), dtype=np.float32)
    for _ in range(3):
        sess.run(None, {inp_name: dummy})

    hud_frames: list[np.ndarray] = []
    grid_candidates: list[dict[str, Any]] = []
    class_pixel_totals = np.zeros(NUM_CANONICAL_CLASSES, dtype=np.int64)
    latencies_ms: list[float] = []

    for idx, rec in enumerate(selected):
        rgb_512 = np.asarray(
            Image.open(rec.rgb_path).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        gt_512 = np.asarray(
            Image.open(rec.mask_path).resize((512, 512), Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        x = (rgb_512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        x = np.ascontiguousarray((x - mean) / std, dtype=np.float32)

        t0 = time.perf_counter_ns()
        seg_logits, _surf_normals, bdry_prob = sess.run(None, {inp_name: x})
        lat_ms = (time.perf_counter_ns() - t0) * 1e-6
        latencies_ms.append(lat_ms)

        raw_logits = seg_logits[0].astype(np.float32)  # (5, 512, 512)
        shifted_logits = raw_logits.copy()
        shifted_logits[0:2] = shifted_logits[0:2] + safe_bias
        pred_512 = np.argmax(shifted_logits, axis=0).astype(np.uint8)

        # Boundary-compensated Helmholtz Free Energy on raw logits
        scaled = raw_logits / t_ood
        lmax = np.max(scaled, axis=0)
        e_raw = -t_ood * (lmax + np.log(np.sum(np.exp(scaled - lmax[None, ...]), axis=0) + 1e-12))
        bdry_64 = np.clip(bdry_prob[0, 0].astype(np.float32), 0.0, 1.0)
        bdry_512 = cv2.resize(bdry_64, (512, 512), interpolation=cv2.INTER_LINEAR)
        e_tilde = e_raw - kappa_b * bdry_512
        ood_pct = float((e_tilde > tau_ood).mean() * 100.0)

        # Per-scene mIoU & FSR
        valid = (gt_512 >= 0) & (gt_512 < NUM_CANONICAL_CLASSES)
        conf = np.zeros((NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES), dtype=np.int64)
        if bool(valid.any()):
            flat = gt_512[valid].astype(np.int64) * NUM_CANONICAL_CLASSES + pred_512[valid].astype(np.int64)
            conf = np.bincount(flat, minlength=25).reshape(5, 5)
        ious = []
        for c in range(NUM_CANONICAL_CLASSES):
            gt_s = int(conf[c, :].sum())
            if gt_s > 0:
                tp = int(conf[c, c])
                union = gt_s + int(conf[:, c].sum()) - tp
                ious.append(float(tp) / float(max(1, union)))
        miou_scene = float(np.mean(ious)) if ious else 0.0
        haz_tot = int(conf[2:, :].sum())
        fsr_scene = float(conf[2:, :2].sum()) / float(max(1, haz_tot)) if haz_tot > 0 else 0.0

        for c in range(NUM_CANONICAL_CLASSES):
            class_pixel_totals[c] += int((pred_512 == c).sum())

        hud_rgb = _draw_hud_frame_rgb(
            rgb_512,
            gt_512,
            pred_512,
            source_name=rec.source,
            scene_idx=idx,
            total_scenes=len(selected),
            latency_ms=lat_ms,
            safe_bias_delta=safe_bias,
            e_tilde_mean=float(np.mean(e_tilde)),
            tau_ood=tau_ood,
            ood_pct=ood_pct,
            miou_scene=miou_scene,
            fsr_scene=fsr_scene,
        )
        hud_frames.append(hud_rgb)

        present_pred_classes = int(sum(1 for c in range(NUM_CANONICAL_CLASSES) if (pred_512 == c).mean() >= 0.015))
        grid_candidates.append(
            {
                "source": rec.source,
                "rgb": rgb_512,
                "pred": pred_512,
                "gt": gt_512,
                "miou": miou_scene,
                "fsr": fsr_scene,
                "diversity": present_pred_classes,
            }
        )

    # Build 6-scene contact sheet PNG (3 rows x 2 scenes, each scene = [RGB | Pred Overlay | GT])
    grid_sha: str | None = None
    if grid_out is not None:
        # Pick 6 diverse scenes across all 4 sources (2 SkyScenes, 2 OEM, 1 FloodNet, 1 FLAIR-1)
        by_src_cands: dict[str, list[dict[str, Any]]] = {}
        for item in grid_candidates:
            by_src_cands.setdefault(item["source"], []).append(item)
        for src in by_src_cands:
            by_src_cands[src].sort(key=lambda d: (d["diversity"], d["miou"]), reverse=True)

        chosen_6: list[dict[str, Any]] = []
        plan = [
            ("SkyScenes", 0),
            ("OpenEarthMap-41", 0),
            ("FloodNet-Supervised-v1.0", 0),
            ("FLAIR-1", 0),
            ("SkyScenes", 1),
            ("OpenEarthMap-41", 1),
        ]
        used_ids: set[int] = set()
        for src, rank in plan:
            pool = by_src_cands.get(src, [])
            if rank < len(pool):
                chosen_6.append(pool[rank])
                used_ids.add(id(pool[rank]))
        for item in sorted(grid_candidates, key=lambda d: (d["diversity"], d["miou"]), reverse=True):
            if len(chosen_6) >= 6:
                break
            if id(item) not in used_ids:
                chosen_6.append(item)
                used_ids.add(id(item))

        tile_sz = 320
        cell_w = tile_sz * 3 + 12  # [RGB | Pred Overlay | GT]
        cell_h = tile_sz + 36
        pad = 14
        top_banner = 62
        bot_legend = 48
        grid_w = pad * 3 + cell_w * 2
        grid_h = top_banner + pad * 4 + cell_h * 3 + bot_legend
        sheet = np.full((grid_h, grid_w, 3), (18, 22, 28), dtype=np.uint8)
        sheet_bgr = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            sheet_bgr,
            f"VertiRTA v1.0.0 | PIDNet-S v2.1 6-Scene Permissive Validation Grid (GTX 1650 SUPER, bias={safe_bias:+.3f})",
            (pad, 28),
            font,
            0.70,
            (242, 246, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            sheet_bgr,
            "Each cell: [Left: Input Aerial RGB]  |  [Center: PIDNet-S v2.1 Predicted Overlay]  |  [Right: Canonical 5-Class Ground Truth]",
            (pad, 50),
            font,
            0.52,
            (170, 215, 200),
            1,
            cv2.LINE_AA,
        )

        for idx6, item in enumerate(chosen_6[:6]):
            row = idx6 // 2
            col = idx6 % 2
            x0 = pad + col * (cell_w + pad)
            y0 = top_banner + pad + row * (cell_h + pad)

            rgb_s = cv2.resize(item["rgb"], (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)
            pred_col = _colorize_slz_mask_rgb(item["pred"])
            ov_s = np.clip(0.66 * item["rgb"].astype(np.float32) + 0.34 * pred_col.astype(np.float32), 0, 255).astype(np.uint8)
            ov_s = cv2.resize(ov_s, (tile_sz, tile_sz), interpolation=cv2.INTER_AREA)
            gt_s = cv2.resize(_colorize_slz_mask_rgb(item["gt"]), (tile_sz, tile_sz), interpolation=cv2.INTER_NEAREST)

            trip = np.full((tile_sz, cell_w, 3), (45, 52, 62), dtype=np.uint8)
            trip[:, 0:tile_sz] = rgb_s
            trip[:, tile_sz + 6 : tile_sz * 2 + 6] = ov_s
            trip[:, tile_sz * 2 + 12 : tile_sz * 3 + 12] = gt_s

            sheet_bgr[y0 + 28 : y0 + 28 + tile_sz, x0 : x0 + cell_w] = cv2.cvtColor(trip, cv2.COLOR_RGB2BGR)
            lbl = f"{item['source']}  |  mIoU={item['miou']:.3f}  |  FSR={item['fsr'] * 100.0:.2f}%"
            cv2.putText(sheet_bgr, lbl, (x0 + 4, y0 + 20), font, 0.54, (235, 240, 248), 1, cv2.LINE_AA)

        # Bottom legend
        y_leg = grid_h - 16
        x_cur = pad
        for cid, cname in enumerate(SLZ_SHORT_NAMES):
            rc = SLZ_CLASS_RGB[cid]
            bc = (int(rc[2]), int(rc[1]), int(rc[0]))
            cv2.rectangle(sheet_bgr, (x_cur, y_leg - 14), (x_cur + 18, y_leg + 2), bc, -1)
            cv2.rectangle(sheet_bgr, (x_cur, y_leg - 14), (x_cur + 18, y_leg + 2), (225, 230, 235), 1)
            cv2.putText(
                sheet_bgr,
                f"C{cid}: {CANONICAL_CLASS_NAMES[cid]}",
                (x_cur + 24, y_leg),
                font,
                0.50,
                (235, 240, 248),
                1,
                cv2.LINE_AA,
            )
            x_cur += 375

        grid_out.parent.mkdir(parents=True, exist_ok=True)
        if grid_out.suffix.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(grid_out), sheet_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            cv2.imwrite(str(grid_out), sheet_bgr)
        grid_sha = sha256_file(grid_out)
        print(f"[Preview] Wrote 6-scene contact sheet ({grid_w}x{grid_h}): {grid_out.name}", flush=True)

    # Encode universal H.264 yuv420p +faststart MP4 via ffmpeg
    mp4_sha: str | None = None
    mp4_bytes = 0
    ffmpeg_used: str | None = None
    if mp4_out is not None:
        mp4_out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_bin = _find_ffmpeg_executable()
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg executable not found for H.264 yuv420p +faststart encoding.")
        ffmpeg_used = Path(ffmpeg_bin).name

        frames_dir = work_dir / "_hud_frames_tmp"
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Write each scene frame once and duplicate softly (12 frames @ 15 fps = 0.8s/scene -> 28.8s video)
            hold_frames = 12
            frame_idx = 0
            for scene_rgb in hud_frames:
                scene_bgr = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR)
                base_png = frames_dir / f"frame_{frame_idx:05d}.png"
                cv2.imwrite(str(base_png), scene_bgr)
                frame_idx += 1
                for _ in range(hold_frames - 1):
                    dup_png = frames_dir / f"frame_{frame_idx:05d}.png"
                    try:
                        os.link(base_png, dup_png)
                    except OSError:
                        shutil.copy2(base_png, dup_png)
                    frame_idx += 1

            # Encode directly to local Linux tmp first, then copy to mp4_out (works cleanly for Windows/WSL2 paths)
            tmp_mp4 = work_dir / "local_gtx_pidnet_v21_hud_h264.mp4"
            in_pattern = str(frames_dir / "frame_%05d.png")
            out_arg = str(tmp_mp4)
            # If invoking Windows ffmpeg.exe from WSL2, convert paths with wslpath -w
            if ffmpeg_bin.lower().endswith(".exe"):
                in_pattern = subprocess.check_output(
                    ["wslpath", "-w", in_pattern], text=True
                ).strip()
                out_arg = subprocess.check_output(
                    ["wslpath", "-w", out_arg], text=True
                ).strip()

            cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                "15",
                "-i",
                in_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                "-preset",
                "medium",
                "-movflags",
                "+faststart",
                out_arg,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            shutil.copy2(tmp_mp4, mp4_out)
            tmp_mp4.unlink(missing_ok=True)
            mp4_bytes = int(mp4_out.stat().st_size)
            mp4_sha = sha256_file(mp4_out)
            print(
                f"[Preview] Encoded H.264 yuv420p +faststart MP4 ({mp4_bytes / (1024*1024):.2f} MB, "
                f"{len(hud_frames)} scenes, {frame_idx} frames @ 15 fps): {mp4_out.name}",
                flush=True,
            )
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)

    total_pred_px = int(class_pixel_totals.sum())
    class_dist_pct = {
        CANONICAL_CLASS_NAMES[c]: round(float(class_pixel_totals[c]) * 100.0 / float(max(1, total_pred_px)), 3)
        for c in range(NUM_CANONICAL_CLASSES)
    }
    return {
        "scenes_rendered": len(hud_frames),
        "sources_represented": sorted({r.source for r in selected}),
        "ort_providers_active": list(sess.get_providers()),
        "mean_inference_ms": round(float(np.mean(latencies_ms)), 3),
        "p50_inference_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "predicted_class_distribution_pct": class_dist_pct,
        "non_collapsed_multiclass_verified": bool(
            sum(1 for v in class_dist_pct.values() if v >= 1.0) >= 3
            and class_dist_pct["Hazard_Water_Vegetation"] < 85.0
        ),
        "video_codec": "h264_yuv420p_faststart",
        "ffmpeg_binary": ffmpeg_used,
        "mp4_filename": mp4_out.name if mp4_out is not None else None,
        "mp4_size_bytes": mp4_bytes,
        "mp4_sha256": mp4_sha,
        "grid_filename": grid_out.name if grid_out is not None else None,
        "grid_sha256": grid_sha,
    }


def _train_local_gtx_pidnet(
    args: argparse.Namespace,
    device: Any,
    license_audit: dict[str, Any],
    t0: float,
) -> dict[str, Any]:
    """Run live 0.00 CU local GTX 1650 SUPER FP16 AMP recovery training + calibration + export."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from PIL import Image

    from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain
    from ml.slz_perception.train.calibrate_ood import (
        calibrate_temperature_and_tau,
        compute_free_energy,
    )
    from ml.slz_perception.train.train_slz_distill import (
        apply_anomaly_mix,
        compute_boundary_target,
    )
    from scripts.colab.run_l4_pidnet_distill import (
        _prepare_floodnet,
        _prepare_openearthmap,
        _prepare_skyscenes,
        calibrate_dynamic_energy_margins,
        evaluate_permissive_val_and_ood,
        evaluate_quarantined_aeroscapes,
        fetch_or_load_oem_attribution_html,
        run_startup_license_gate,
    )

    work_dir = Path(args.work_dir)
    blob_dir = Path(args.blob_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    # 1) Live Zero-ShareAlike Startup License Gate BEFORE any dataset download
    attr_path, attr_meta = fetch_or_load_oem_attribution_html(
        work_dir / "openearthmap_attribution.html"
    )
    active_sources = ("FLAIR-1", "OpenEarthMap", "FloodNet-Supervised-v1.0", "SkyScenes")
    manifest_report, oem_regions = run_startup_license_gate(
        attr_path, active_sources, expect_oem_total=None, include_aeroscapes_eval=True
    )
    print(
        f"[LicenseGate] PASSED: zero_sharealike_verified={manifest_report.zero_sharealike_verified}, "
        f"train={list(manifest_report.approved_train_entries)}, oem_regions={len(oem_regions)}",
        flush=True,
    )

    # 2) Prepare all 4 real permissive datasets (reusing cached FLAIR-1 zip & optional prepared cache)
    t_data0 = time.perf_counter()
    loaded_prepared = _load_prepared_manifest(work_dir) if args.reuse_prepared else None
    if loaded_prepared is not None:
        all_records, dataset_provenance = loaded_prepared
        print(f"[Data] Reused {len(all_records)} prepared permissive samples from {work_dir}", flush=True)
    else:
        all_records = []
        dataset_provenance = []
        for prep_fn in (
            lambda: _prepare_flair1_with_cache(
                work_dir, max_samples=args.max_flair, cached_zip=args.cached_flair_zip
            ),
            lambda: _prepare_openearthmap(work_dir, frozenset(oem_regions), max_samples=args.max_oem),
            lambda: _prepare_floodnet(work_dir, max_samples=args.max_floodnet),
            lambda: _prepare_skyscenes(work_dir, max_samples=args.max_skyscenes),
        ):
            recs, prov = prep_fn()
            all_records.extend(recs)
            dataset_provenance.append(prov)
        _save_prepared_manifest(work_dir, all_records, dataset_provenance)
    data_prep_sec = round(time.perf_counter() - t_data0, 2)

    train_recs = [r for r in all_records if r.split == "train"]
    val_recs = [r for r in all_records if r.split == "val"]
    if not train_recs or not val_recs:
        raise RuntimeError(f"Insufficient split records: train={len(train_recs)}, val={len(val_recs)}")

    # 3) Dataset with real Oblique Perspective Augmentation (35% on nadir frames) + crop/flip/jitter
    mean_t = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    class LocalObliqueSlzDataset(Dataset):
        def __init__(self, items: list[SampleRecord], augment: bool) -> None:
            self.items = items
            self.augment = augment

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            rec = self.items[idx]
            rgb_pil = Image.open(rec.rgb_path).convert("RGB")
            msk_pil = Image.open(rec.mask_path)
            nrm_pil = Image.open(rec.normal_path).convert("RGB") if rec.normal_path else None
            hflip = False
            if self.augment:
                if rec.source != "SkyScenes" and random.random() < 0.35:
                    rgb_pil, msk_pil, nrm_pil = apply_oblique_perspective_aug(rgb_pil, msk_pil, nrm_pil)
                if random.random() < 0.5:
                    w, h = rgb_pil.size
                    scale = random.uniform(0.78, 1.0)
                    cw = max(64, int(round(w * scale)))
                    ch = max(64, int(round(h * scale)))
                    x0 = random.randint(0, max(0, w - cw))
                    y0 = random.randint(0, max(0, h - ch))
                    box = (x0, y0, x0 + cw, y0 + ch)
                    rgb_pil = rgb_pil.crop(box).resize((512, 512), Image.Resampling.BILINEAR)
                    msk_pil = msk_pil.crop(box).resize((512, 512), Image.Resampling.NEAREST)
                    if nrm_pil is not None:
                        nrm_pil = nrm_pil.crop(box).resize((512, 512), Image.Resampling.BILINEAR)
                if random.random() < 0.5:
                    hflip = True
                    rgb_pil = rgb_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    msk_pil = msk_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                    if nrm_pil is not None:
                        nrm_pil = nrm_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            rgb_np = np.asarray(rgb_pil, dtype=np.float32) / 255.0
            if self.augment and random.random() < 0.5:
                rgb_np = np.clip(
                    rgb_np * random.uniform(0.85, 1.15) + random.uniform(-0.06, 0.06), 0.0, 1.0
                )
            rgb_t = (torch.from_numpy(rgb_np).permute(2, 0, 1) - mean_t) / std_t
            msk_t = torch.from_numpy(np.asarray(msk_pil, dtype=np.int64))
            if nrm_pil is not None:
                n_np = (np.asarray(nrm_pil, dtype=np.float32) / 127.5) - 1.0
                if hflip:
                    n_np[..., 0] = -n_np[..., 0]
                n_t = torch.from_numpy(n_np).permute(2, 0, 1)
                has_gt_normal = True
            else:
                n_t = torch.zeros((3, 512, 512), dtype=torch.float32)
                has_gt_normal = False
            return {
                "rgb": rgb_t,
                "label": msk_t,
                "gt_normal": n_t,
                "has_gt_normal": has_gt_normal,
                "source": rec.source,
            }

    target_ratios = {
        "FLAIR-1": 0.15,
        "OpenEarthMap-41": 0.25,
        "FloodNet-Supervised-v1.0": 0.20,
        "SkyScenes": 0.40,
    }
    src_counts = Counter(r.source for r in train_recs)
    weights = [target_ratios.get(r.source, 0.1) / float(max(1, src_counts[r.source])) for r in train_recs]
    sampler_num_samples = len(train_recs)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=sampler_num_samples,
        replacement=True,
    )
    num_workers = int(args.num_workers)
    train_loader = DataLoader(
        LocalObliqueSlzDataset(train_recs, augment=True),
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=(sampler_num_samples >= int(args.batch_size)),
    )
    val_loader = DataLoader(
        LocalObliqueSlzDataset(val_recs, augment=False),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    print(
        f"[Data] train={len(train_recs)} {dict(src_counts)}, val={len(val_recs)} "
        f"{dict(Counter(r.source for r in val_recs))} | prep={data_prep_sec}s",
        flush=True,
    )

    # 4) Local GTX 1650 SUPER FP16 AMP Training (teachers=False, bs=4 x grad_accum=4 -> eff_bs=16)
    teachers = DisabledTeachersStub()
    ObliqueBalancedFocalFsrLoss, _ = _build_oblique_loss_and_xfeat()
    loss_fn = ObliqueBalancedFocalFsrLoss(gamma=2.0, fsr_margin=0.12).to(device)
    pidnet_train = PIDNetSlzTrain(num_classes=NUM_CANONICAL_CLASSES, dino_dim=1024).to(device)

    total_epochs = int(args.epochs)
    warmup_epochs = min(2, max(1, total_epochs // 8))
    stage2_start_epoch = max(2, (total_epochs * 2) // 3 + 1)
    grad_accum = max(1, int(args.grad_accum))
    base_lr = float(args.lr)
    min_lr = 2e-5
    use_fp16 = bool(device.type == "cuda")

    opt = torch.optim.AdamW(pidnet_train.parameters(), lr=base_lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats(device)

    pareto_history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_score_key: tuple[int, float, float] = (-1, -math.inf, -math.inf)
    dyn_m_in, dyn_m_out = -8.0, -2.5
    stage2_margins_calibrated = False
    t_train0 = time.perf_counter()

    try:
        for epoch in range(1, total_epochs + 1):
            ep_t0 = time.perf_counter()
            if epoch <= warmup_epochs:
                lr = base_lr * float(epoch) / float(warmup_epochs)
            else:
                prog = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * prog))
            for pg in opt.param_groups:
                pg["lr"] = lr

            is_stage2 = epoch >= stage2_start_epoch
            if is_stage2 and not stage2_margins_calibrated:
                # Calibrate 5-class Stage-2 free-energy margins from Stage-1 ID energy scale
                _, _, _, _, s1_ood = evaluate_permissive_val_and_ood(
                    pidnet_train,
                    val_loader,
                    teachers,
                    device,
                    use_bf16=False,
                    max_cal_pixels=50_000,
                    eval_normals=False,
                    eval_ood=True,
                )
                mean_e = float(s1_ood.get("mean_id_energy", -6.0))
                dyn_m_in, dyn_m_out = calibrate_dynamic_energy_margins(mean_e)
                stage2_margins_calibrated = True
                print(
                    f"[Stage2-Cal] Ep {epoch:02d} entering OOD OE: mean_id_energy={mean_e:.3f} "
                    f"-> m_in={dyn_m_in}, m_out={dyn_m_out}",
                    flush=True,
                )

            ramp = 0.5 * (1.0 - math.cos(math.pi * float(epoch) / float(max(1, total_epochs))))
            lambda_fsr = 0.15 + 0.45 * ramp

            pidnet_train.train()
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            n_steps = 0

            for step_idx, batch in enumerate(train_loader):
                rgb = batch["rgb"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)
                gt_n = batch["gt_normal"].to(device, non_blocking=True)
                has_n = batch["has_gt_normal"].to(device, non_blocking=True)

                if is_stage2:
                    rgb, lbl = apply_anomaly_mix(rgb, lbl, prob=0.18, ood_index=OOD_INDEX)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                    out = pidnet_train(rgb)

                # Compute losses strictly in FP32 outside autocast (numerically stable BCE + logsumexp)
                loss_seg, _ = loss_fn(out.seg_logits.float(), lbl, lambda_fsr=lambda_fsr)

                # Surface normal alignment strictly where exact SkyScenes metric 3D normals exist
                t_norm_mag = torch.linalg.norm(gt_n.float(), dim=1)
                n_mask = has_n.view(-1, 1, 1) & (t_norm_mag > 1e-4) & (lbl >= 0) & (lbl < NUM_CANONICAL_CLASSES)
                if bool(n_mask.any()):
                    cos_n = F.cosine_similarity(out.surf_normals.float(), gt_n.float(), dim=1, eps=1e-6)
                    loss_norm = (1.0 - cos_n[n_mask]).mean()
                else:
                    loss_norm = torch.tensor(0.0, device=device)

                bdry_tgt = compute_boundary_target(lbl, out_hw=(64, 64))
                loss_bdry = F.binary_cross_entropy(
                    out.bdry_prob.float().clamp(1e-5, 1.0 - 1e-5), bdry_tgt
                )

                loss_ood = torch.tensor(0.0, device=device)
                if is_stage2:
                    energy = -torch.logsumexp(out.seg_logits.float(), dim=1)
                    id_m = (lbl >= 0) & (lbl < NUM_CANONICAL_CLASSES)
                    ood_m = lbl == OOD_INDEX
                    if bool(id_m.any()):
                        loss_ood = loss_ood + F.relu(energy[id_m] - dyn_m_in).pow(2).mean()
                    if bool(ood_m.any()):
                        loss_ood = loss_ood + F.relu(dyn_m_out - energy[ood_m]).pow(2).mean()

                total_loss = loss_seg + 0.25 * loss_norm + 0.25 * loss_bdry + 0.03 * loss_ood
                scaled_loss = total_loss / float(grad_accum)
                scaler.scale(scaled_loss).backward()

                is_boundary = ((step_idx + 1) % grad_accum == 0) or (step_idx + 1 == len(train_loader))
                if is_boundary:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(pidnet_train.parameters(), 2.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                loss_sum += float(total_loss.detach().cpu().item())
                n_steps += 1

            val_m, _, _, _, ood_ep = evaluate_permissive_val_and_ood(
                pidnet_train,
                val_loader,
                teachers,
                device,
                use_bf16=False,
                max_cal_pixels=50_000,
                eval_normals=False,
                eval_ood=is_stage2,
            )
            v_miou = float(val_m["miou_present_classes"])
            v_fsr = float(val_m["false_safe_rate_fsr"])
            ep_sec = time.perf_counter() - ep_t0
            cand: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": round(loss_sum / float(max(1, n_steps)), 5),
                "lr": round(lr, 7),
                "val_miou": round(v_miou, 6),
                "val_fsr": round(v_fsr, 6),
                "epoch_sec": round(ep_sec, 2),
            }
            if is_stage2:
                cand["ood_auroc"] = round(float(ood_ep.get("boundary_compensated_auroc", 0.0)), 6)
            pareto_history.append(cand)

            score_key = _pareto_checkpoint_sort_key(cand, max_fsr=0.030, min_miou=0.25)
            if score_key > best_score_key:
                best_score_key = score_key
                best_state = copy.deepcopy(
                    {k: v.detach().cpu() for k, v in pidnet_train.state_dict().items()}
                )

            ood_str = f" | OOD_AUROC={cand['ood_auroc']:.4f}" if is_stage2 else ""
            print(
                f"[GTX-Train] Ep {epoch:02d}/{total_epochs} | lr={lr:.2e} | loss={cand['train_loss']:.4f} | "
                f"val_mIoU={v_miou:.4f} | val_FSR={v_fsr:.4f}{ood_str} | {ep_sec:.1f}s",
                flush=True,
            )

        train_wall_sec = round(time.perf_counter() - t_train0, 2)
        assert best_state is not None
        pidnet_train.load_state_dict(best_state)
        pareto_res = select_pareto_checkpoint(pareto_history, max_fsr=0.030, min_miou=0.25)
        print(
            f"[Pareto] Selected Ep {int(pareto_res['selected']['epoch'])} "
            f"(val_mIoU={pareto_res['selected']['val_miou']:.4f}, "
            f"val_FSR={pareto_res['selected']['val_fsr']:.4f}, "
            f"degenerate_rejected={pareto_res['degenerate_rejected_count']})",
            flush=True,
        )

        # 5) Full Raw Permissive Val + OOD + Post-hoc T_prob/tau_ood & Safe-Logit Bias Calibration
        raw_val_metrics, cal_logits, cal_labels, cal_bdry, ood_metrics = evaluate_permissive_val_and_ood(
            pidnet_train,
            val_loader,
            teachers,
            device,
            use_bf16=False,
            max_cal_pixels=600_000,
            eval_normals=True,
            eval_ood=True,
        )
        ood_cal = calibrate_temperature_and_tau(
            cal_logits, cal_labels, cal_bdry, T_ood=1.0, kappa_b=1.5, id_tpr_target=0.95
        )
        e_cal_tilde = compute_free_energy(cal_logits, T=ood_cal.T_ood) - ood_cal.kappa_b * cal_bdry.clamp(0.0, 1.0)
        ood_metrics["empirical_clean_id_tpr_at_tau_ood"] = round(
            float((e_cal_tilde <= ood_cal.tau_ood).float().mean().item()), 6
        )

        # Calibrate safe-logit bias strictly on permissive ID val split
        fsr_bias_cal = calibrate_safe_logit_bias_for_fsr(
            cal_logits, cal_labels, target_max_fsr=0.030
        )
        safe_bias_delta = float(fsr_bias_cal["safe_logit_bias_delta"])

        # Evaluate FULL permissive val split with frozen safe-logit bias applied
        biased_model = _build_safe_bias_eval_wrapper(pidnet_train, safe_bias_delta)
        cal_val_metrics, _, _, _, _ = evaluate_permissive_val_and_ood(
            biased_model,
            val_loader,
            teachers,
            device,
            use_bf16=False,
            max_cal_pixels=50_000,
            eval_normals=True,
            eval_ood=False,
        )
        print(
            f"[Val-Final] RAW: mIoU={raw_val_metrics['miou_present_classes']:.4f}, "
            f"FSR={raw_val_metrics['false_safe_rate_fsr']:.4f}, "
            f"safe_prec={raw_val_metrics['safe_precision']:.4f} | "
            f"BIAS({safe_bias_delta:+.4f}): mIoU={cal_val_metrics['miou_present_classes']:.4f}, "
            f"FSR={cal_val_metrics['false_safe_rate_fsr']:.4f}, "
            f"safe_prec={cal_val_metrics['safe_precision']:.4f} | "
            f"OOD_AUROC={ood_metrics['boundary_compensated_auroc']:.4f}",
            flush=True,
        )
    finally:
        _shutdown_loader(train_loader)
        _shutdown_loader(val_loader)

    # Save calibration JSON to /var/tmp/vertirta_v400_blobs/ood_calibration_v21_local_gtx.json
    ood_cal_dict: dict[str, Any] = {
        **ood_cal.to_dict(),
        "safe_logit_bias_delta": safe_bias_delta,
        "safe_logit_bias_calibration": fsr_bias_cal,
    }
    ood_cal_path = blob_dir / CAL_FILENAME
    write_canonical_json(ood_cal_path, ood_cal_dict)

    # 6) Strictly Quarantined Post-Freeze Zero-Shot AeroScapes Evaluation (Raw + Bias-Calibrated)
    aero_raw = evaluate_quarantined_aeroscapes(
        pidnet_train, work_dir, device, max_samples=int(args.max_aeroscapes), use_bf16=False
    )
    aero_cal = evaluate_quarantined_aeroscapes(
        _build_safe_bias_eval_wrapper(pidnet_train, safe_bias_delta),
        work_dir,
        device,
        max_samples=int(args.max_aeroscapes),
        use_bf16=False,
    )
    aero_cal["safe_logit_bias_delta_applied_from_permissive_id_val"] = safe_bias_delta

    peak_alloc_mib = round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)
    peak_reserved_mib = round(float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0), 2)

    # 7) Export static 1x3x512x512 Opset-17 ONNX (8,003,817 params, zero AdaptiveAvgPool2d)
    deploy_wrapper = PIDNetSlzDeployWrapper(pidnet_train.cpu().eval()).eval()
    onnx_out = blob_dir / ONNX_FILENAME
    pidnet_onnx = _export_and_check_onnx(
        deploy_wrapper,
        onnx_out,
        input_names=["rgb_nchw"],
        output_names=["seg_logits", "surf_normals", "bdry_prob"],
        sample_shape=(1, 3, 512, 512),
    )
    assert pidnet_onnx["deploy_parameters"] == 8003817, (
        f"Expected 8003817 deploy params, got {pidnet_onnx['deploy_parameters']}"
    )
    pidnet_onnx["path"] = sanitize_artifact_path(onnx_out)

    # Release PyTorch CUDA context before ORT CUDA preview rendering so 4 GB TU116 stays roomy
    del pidnet_train, opt, scaler, loss_fn
    gc.collect()
    torch.cuda.empty_cache()

    # 8) Optional Playable H.264 HUD Preview + 6-Scene Contact Sheet PNG
    preview_summary: dict[str, Any] | None = None
    if args.preview_mp4 is not None or args.preview_grid is not None:
        preview_summary = render_hud_preview_and_contact_sheet(
            onnx_path=onnx_out,
            calibration=ood_cal_dict,
            val_recs=val_recs,
            mp4_out=args.preview_mp4,
            grid_out=args.preview_grid,
            work_dir=work_dir,
            num_scenes=int(args.preview_scenes),
            use_cuda=True,
        )

    h2h = build_head_to_head_comparison(
        local_raw_val=raw_val_metrics,
        local_cal_val=cal_val_metrics,
        local_ood=ood_metrics,
        local_raw_aero=aero_raw,
        local_cal_aero=aero_cal,
        local_bias_delta=safe_bias_delta,
    )

    payload: dict[str, Any] = {
        "schema_version": "vertirta.local_gtx1650s_pidnet_v21_recovery_metrics/v4.0.0",
        "mode": "LIVE_LOCAL_GTX1650S_CUDA_FP16",
        "weights_status": "TRAINED_LOCAL_GTX1650S_RECOVERY_V21_ZERO_CU",
        "cu_consumed": 0.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "amp_dtype": "float16",
            "peak_cuda_allocated_mib": peak_alloc_mib,
            "peak_cuda_reserved_mib": peak_reserved_mib,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "nvidia_smi": query_nvidia_smi_snapshot(),
        },
        "timing_sec": {
            "data_prep_sec": data_prep_sec,
            "train_wall_sec": train_wall_sec,
            "total_wall_sec": round(time.perf_counter() - t0, 3),
        },
        "license_audit": {
            **license_audit,
            "oem_attribution_audit": attr_meta,
            "oem_permissive_region_count": len(oem_regions),
            "startup_gate_approved_train": list(manifest_report.approved_train_entries),
            "quarantined_eval_entries": list(manifest_report.quarantined_eval_entries),
        },
        "skyscenes_nested_tar_filter_active": True,
        "datasets": dataset_provenance,
        "teachers": teachers.metadata,
        "training": {
            "seed": int(args.seed),
            "batch_size_micro": int(args.batch_size),
            "grad_accum_steps": grad_accum,
            "effective_batch_size": int(args.batch_size) * grad_accum,
            "learning_rate_peak": base_lr,
            "learning_rate_min": min_lr,
            "epochs": total_epochs,
            "stage2_ood_start_epoch": stage2_start_epoch,
            "stage2_dynamic_m_in": dyn_m_in,
            "stage2_dynamic_m_out": dyn_m_out,
            "oblique_perspective_aug_prob": 0.35,
            "sampler_num_samples": sampler_num_samples,
            "pareto_selection": pareto_res,
            "pareto_history": pareto_history,
        },
        "permissive_val_metrics_raw": raw_val_metrics,
        "permissive_val_metrics_bias_calibrated": cal_val_metrics,
        "ood_metrics": ood_metrics,
        "ood_calibration": {
            **ood_cal_dict,
            "file": CAL_FILENAME,
            "path": sanitize_artifact_path(ood_cal_path),
            "sha256": sha256_file(ood_cal_path),
        },
        "quarantined_zero_shot_aeroscapes_raw": aero_raw,
        "quarantined_zero_shot_aeroscapes_bias_calibrated": aero_cal,
        "head_to_head_comparison": h2h,
        "onnx_export": pidnet_onnx,
        "hud_preview_summary": preview_summary,
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    write_canonical_json(args.out_json, payload)
    print(f"[DONE] Sealed recovery report written to {sanitize_artifact_path(args.out_json)}", flush=True)
    return payload


def _run_smoke_cpu(
    args: argparse.Namespace,
    license_audit: dict[str, Any],
    t0: float,
) -> dict[str, Any]:
    """Fast 0.00 CU CPU CI verification of loss, fixed Pareto selector, FSR bias cal, and Opset-17 ONNX."""
    import torch
    import torch.nn.functional as F

    from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain
    from ml.slz_perception.train.calibrate_ood import calibrate_temperature_and_tau
    from ml.slz_perception.train.train_slz_distill import compute_boundary_target

    device = torch.device("cpu")
    ObliqueBalancedFocalFsrLoss, _ = _build_oblique_loss_and_xfeat()
    pidnet_train = PIDNetSlzTrain(num_classes=NUM_CANONICAL_CLASSES, dino_dim=1024).to(device)
    loss_fn = ObliqueBalancedFocalFsrLoss(gamma=2.0, fsr_margin=0.12).to(device)
    opt = torch.optim.AdamW(pidnet_train.parameters(), lr=5e-4, weight_decay=1e-4)
    rng = np.random.default_rng(int(args.seed))

    losses: list[float] = []
    pidnet_train.train()
    for step in range(2):
        x_t = torch.from_numpy(rng.normal(0.0, 0.5, size=(2, 3, 512, 512)).astype(np.float32))
        y_t = torch.from_numpy(rng.integers(0, 5, size=(2, 512, 512), dtype=np.int64))
        opt.zero_grad(set_to_none=True)
        out = pidnet_train(x_t)
        loss_seg, _ = loss_fn(out.seg_logits.float(), y_t, lambda_fsr=0.25 + 0.1 * step)
        bdry_tgt = compute_boundary_target(y_t, out_hw=(64, 64))
        loss_bdry = F.binary_cross_entropy(out.bdry_prob.float().clamp(1e-5, 1.0 - 1e-5), bdry_tgt)
        loss = loss_seg + 0.25 * loss_bdry
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

    # Verify fixed Pareto selector rejects Epoch-2 collapsed checkpoint
    synthetic_history = [
        {"epoch": 2.0, "val_miou": 0.1696, "val_fsr": 0.0090},
        {"epoch": 18.0, "val_miou": 0.4501, "val_fsr": 0.0789},
    ]
    pareto_res = select_pareto_checkpoint(synthetic_history, max_fsr=0.030, min_miou=0.25)
    assert pareto_res["selected"]["epoch"] == 18.0
    assert pareto_res["degenerate_rejected_count"] == 1

    # Verify post-hoc OOD temperature + safe-logit FSR bias calibration on synthetic logits
    pidnet_train.eval()
    with torch.no_grad():
        x_val = torch.from_numpy(rng.normal(0.0, 0.5, size=(1, 3, 512, 512)).astype(np.float32))
        y_val = torch.from_numpy(rng.integers(0, 5, size=(1, 512, 512), dtype=np.int64))
        out_val = pidnet_train(x_val)
        bdry_up = F.interpolate(out_val.bdry_prob, size=(512, 512), mode="bilinear", align_corners=False)
        cal_logits = out_val.seg_logits.permute(0, 2, 3, 1).reshape(-1, 5)[::32]
        cal_labels = y_val.reshape(-1)[::32]
        cal_bdry = bdry_up.reshape(-1)[::32]

    ood_cal = calibrate_temperature_and_tau(cal_logits, cal_labels, cal_bdry, kappa_b=1.5)
    fsr_bias_cal = calibrate_safe_logit_bias_for_fsr(cal_logits, cal_labels, target_max_fsr=0.030)
    assert fsr_bias_cal["fsr_gate_met"] is True

    blob_dir = Path(args.blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    ood_cal_dict = {
        **ood_cal.to_dict(),
        "safe_logit_bias_delta": fsr_bias_cal["safe_logit_bias_delta"],
        "safe_logit_bias_calibration": fsr_bias_cal,
    }
    ood_cal_path = blob_dir / CAL_FILENAME
    write_canonical_json(ood_cal_path, ood_cal_dict)

    deploy_wrapper = PIDNetSlzDeployWrapper(pidnet_train.eval())
    onnx_out = blob_dir / ONNX_FILENAME
    pidnet_onnx = _export_and_check_onnx(
        deploy_wrapper,
        onnx_out,
        input_names=["rgb_nchw"],
        output_names=["seg_logits", "surf_normals", "bdry_prob"],
        sample_shape=(1, 3, 512, 512),
    )
    assert pidnet_onnx["deploy_parameters"] == 8003817
    pidnet_onnx["path"] = sanitize_artifact_path(onnx_out)

    payload: dict[str, Any] = {
        "schema_version": "vertirta.local_gtx1650s_pidnet_v21_recovery_metrics/v4.0.0",
        "mode": "SMOKE_CPU_ONLY",
        "weights_status": "SMOKE_CPU_UNTRAINED_CONTRACT_CHECK",
        "cu_consumed": 0.0,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": "cpu",
            "amp_dtype": "float32",
            "peak_cuda_allocated_mib": 0.0,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "license_audit": license_audit,
        "skyscenes_nested_tar_filter_active": True,
        "teachers": DisabledTeachersStub().metadata,
        "training": {
            "batch_size_micro": 2,
            "steps": 2,
            "loss_start": round(losses[0], 6),
            "loss_end": round(losses[-1], 6),
            "all_losses_finite": bool(all(math.isfinite(v) for v in losses)),
            "pareto_selection": pareto_res,
        },
        "ood_calibration": {
            **ood_cal_dict,
            "file": CAL_FILENAME,
            "path": sanitize_artifact_path(ood_cal_path),
            "sha256": sha256_file(ood_cal_path),
        },
        "onnx_export": pidnet_onnx,
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    write_canonical_json(args.out_json, payload)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="0.00 CU Local Physical GTX 1650 SUPER PIDNet-S v2.1 Recovery Training & HUD Preview."
    )
    parser.add_argument("--smoke-cpu", action="store_true", help="Fast CPU contract & ONNX smoke test.")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--epochs", type=int, default=18, help="Total training epochs (default: 18).")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size for 4 GB VRAM (default: 4).")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak AdamW learning rate (default: 1e-3).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2).")
    parser.add_argument("--max-flair", type=int, default=200)
    parser.add_argument("--max-oem", type=int, default=250)
    parser.add_argument("--max-floodnet", type=int, default=200)
    parser.add_argument("--max-skyscenes", type=int, default=240)
    parser.add_argument("--max-aeroscapes", type=int, default=350)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--blob-dir", type=Path, default=DEFAULT_BLOB_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--cached-flair-zip", type=Path, default=DEFAULT_CACHED_FLAIR_ZIP)
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse existing prepared_manifest.json inside --work-dir if present.",
    )
    parser.add_argument("--preview-mp4", type=Path, default=None, help="Optional H.264 yuv420p HUD MP4 output.")
    parser.add_argument("--preview-grid", type=Path, default=None, help="Optional 6-scene contact sheet PNG/JPG.")
    parser.add_argument(
        "--render-grid-only",
        action="store_true",
        help="Re-render --preview-grid from existing ONNX + prepared validation manifest without retraining.",
    )
    parser.add_argument("--preview-scenes", type=int, default=36)
    return parser


def run_recovery(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_everything(int(args.seed))

    train_sources, eval_only_sources, models = build_pidnet_xfeat_license_manifest()
    license_audit = validate_job_license_manifest(
        "local_gtx1650s_pidnet_v21_recovery", train_sources, eval_only_sources, models
    )

    if args.smoke_cpu:
        return _run_smoke_cpu(args, license_audit, t0)

    if getattr(args, "render_grid_only", False):
        if args.preview_grid is None:
            raise ValueError("--render-grid-only requires --preview-grid <path>.")
        work_dir = Path(args.work_dir)
        blob_dir = Path(args.blob_dir)
        onnx_path = blob_dir / ONNX_FILENAME
        cal_path = blob_dir / CAL_FILENAME
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing ONNX blob for grid render: {onnx_path}")
        if not cal_path.is_file():
            raise FileNotFoundError(f"Missing calibration JSON for grid render: {cal_path}")
        loaded = _load_prepared_manifest(work_dir)
        if loaded is None:
            raise FileNotFoundError(f"Missing prepared_manifest.json in {work_dir}")
        all_records, _ = loaded
        val_recs = [r for r in all_records if r.split == "val"]
        calibration = json.loads(cal_path.read_text(encoding="utf-8"))
        return render_hud_preview_and_contact_sheet(
            onnx_path=onnx_path,
            calibration=calibration,
            val_recs=val_recs,
            mp4_out=args.preview_mp4,
            grid_out=args.preview_grid,
            work_dir=work_dir,
            num_scenes=int(args.preview_scenes),
            use_cuda=True,
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for live local GTX 1650 SUPER recovery training; pass --smoke-cpu on CPU hosts."
        )
    device = torch.device("cuda:0")
    return _train_local_gtx_pidnet(args, device, license_audit, t0)


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    run_recovery(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
