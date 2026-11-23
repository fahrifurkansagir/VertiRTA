#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Colab L4 (24 GB) Zero-ShareAlike PIDNet-S Distillation & Local GTX 1650 SUPER Proof Runner.

Two reproducible execution modes:
  1) Colab L4 training mode (default):
       - Fetches live OpenEarthMap attribution.html and enforces `validate_training_startup_license()`
         STRICTLY BEFORE downloading any training pixels (CC BY 4.0 + DL-DE-BY-2.0 ONLY; zero
         ShareAlike, zero NC, zero ODbL, zero hard-banned datasets or teachers).
       - Prepares curated <=12 GB zero-ShareAlike datasets:
           * FLAIR-1 (OPEN LICENCE 2.0 / etalab-2.0)
           * OpenEarthMap permissive whitelist (CC BY 4.0 + DL-DE-BY-2.0 ONLY)
           * FloodNet-Supervised v1.0 (CDLA-Permissive-1.0)
           * SkyScenes (MIT, oblique+nadir with exact depth-normal facade filter |n_z|<0.85 -> 2)
         plus strictly quarantined zero-shot eval-only AeroScapes (CC BY-SA 4.0, NEVER trained
         or calibrated).
       - Runs 2-stage BF16 AMP distillation on NVIDIA L4 (24 GB) with frozen Apache-2.0 teachers
         (`facebook/dinov2-large` + `depth-anything/Depth-Anything-V2-Small-hf`) and Stage-2
         AnomalyMix Free-Energy OOD outlier exposure (`label=254`, `m_in=-23.0`, `m_out=-5.0`).
       - Fits post-hoc `ood_calibration.json` (`T_ood=1.0`, `T_prob in [0.5, 3.0]`, `kappa_b=1.5`,
         95%-ID-TPR `tau_ood`) strictly on permissive ID validation split.
       - Exports static 1x3x512x512 Opset-17 FP32 ONNX (`PIDNetSlzDeployWrapper`, zero
         `AdaptiveAvgPool2d`, zero `dino_proj`) and writes `l4_pidnet_distill_metrics.json`.
  2) Local physical GTX 1650 SUPER (TU116) proof mode (`--benchmark-local-onnx`):
       - Verifies SHA-256 parity with `l4_pidnet_distill_metrics.json`, runs isolated subprocess
         `onnxruntime-gpu` FP32, FP32+CUDA-Graph, and FP16 benchmarks with `IoBinding` and
         `cudaMemGetInfo` VRAM measurement, verifies finite non-degenerate trained predictions,
         and writes `local_gtx1650s_pidnet_trained_proof.json` retiring `UNTRAINED_WEIGHTS`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.slz_perception.data.license_gate import (  # noqa: E402
    ManifestReport,
    _HtmlTableRowParser,
    parse_oem_permissive_whitelist,
)
from ml.slz_perception.data.remappers import (  # noqa: E402
    IGNORE_INDEX,
    NUM_CANONICAL_CLASSES,
    OOD_INDEX,
    compute_world_normals_from_depth,
    remap_mask,
)
from ml.slz_perception.train.train_slz_distill import (  # noqa: E402
    L4TrainProfile,
    Stage1Config,
    Stage2Config,
    apply_anomaly_mix,
    build_l4_training_manifest,
    compute_boundary_target,
    compute_stage1_lr,
    compute_stage2_lr,
    pseudo_normals_from_inv_depth,
    train_one_step,
    validate_training_startup_license,
)

OEM_ATTRIBUTION_URL = "https://open-earth-map.org/attribution.html"
OEM_ZENODO_ZIP_URL = "https://zenodo.org/api/records/7223446/files/OpenEarthMap.zip/content"
FLAIR1_TOY_URL = (
    "https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/"
    "defi-ia/flair_data_1/flair_1_toy_dataset.zip"
)
DINOV2_REPO_ID = "facebook/dinov2-large"
DAV2_SMALL_REPO_ID = "depth-anything/Depth-Anything-V2-Small-hf"
FLOODNET_HF_REPO_ID = "torchgeo/floodnet"
SKYSCENES_HF_REPO_ID = "hoffman-lab/SkyScenes"
AEROSCAPES_HF_REPO_ID = "dronefreak/Aeroscapes"

CANONICAL_CLASS_NAMES: tuple[str, ...] = (
    "Safe_Flat_Ground",
    "Safe_Flat_Roof",
    "Hazard_Obstacle_Structure",
    "Hazard_Dynamic_Human_Vehicle",
    "Hazard_Water_Vegetation",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_region_token(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def filter_oem_region(image_name: str, oem_whitelist: set[str] | frozenset[str]) -> str | None:
    """Return canonical whitelisted OEM region if `image_name` belongs to CC BY 4.0 / DL-DE-BY-2.0."""
    stem = Path(image_name).stem.lower()
    region_slug = re.sub(r"_\d+$", "", stem)
    tok = _norm_region_token(region_slug)
    if not tok:
        return None
    lookup = {_norm_region_token(r): r for r in oem_whitelist}
    return lookup.get(tok)


def fetch_or_load_oem_attribution_html(
    dest_path: Path,
    local_html: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fetch live OpenEarthMap attribution.html (or copy local) and summarize license counts."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if local_html is not None and local_html.is_file():
        html_text = local_html.read_text(encoding="utf-8")
        source_origin = "local_file"
    else:
        req = urllib.request.Request(
            OEM_ATTRIBUTION_URL,
            headers={"User-Agent": "VertiRTA-SLZ-LicenseGate/3.1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            html_text = resp.read().decode("utf-8")
        source_origin = OEM_ATTRIBUTION_URL

    dest_path.write_text(html_text, encoding="utf-8", newline="\n")
    parser = _HtmlTableRowParser()
    parser.feed(html_text)
    parser.close()

    data_rows = [
        r for r in parser.data_rows if len(r) >= 2 and r[0].strip().lower() != "region"
    ]
    lic_counts = Counter(r[-1].strip() if r[-1].strip() else "<blank>" for r in data_rows)
    return dest_path, {
        "origin": source_origin,
        "sha256": sha256_file(dest_path),
        "total_region_rows": len(data_rows),
        "license_breakdown": dict(sorted(lic_counts.items())),
    }


def run_startup_license_gate(
    attribution_html: Path,
    active_sources: tuple[str, ...],
    *,
    expect_oem_total: int | None = None,
    include_aeroscapes_eval: bool = True,
) -> tuple[ManifestReport, tuple[str, ...]]:
    """Mandatory startup Zero-ShareAlike gate invoked before any training data download."""
    oem_regions = tuple(
        sorted(parse_oem_permissive_whitelist(attribution_html, expect_total=expect_oem_total))
    )
    if not oem_regions:
        raise RuntimeError("Parsed OpenEarthMap permissive whitelist is empty.")
    manifest = build_l4_training_manifest(
        oem_regions,
        active_sources,
        include_aeroscapes_eval=include_aeroscapes_eval,
    )
    report = validate_training_startup_license(
        manifest,
        attribution_html,
        expect_oem_total=expect_oem_total,
    )
    if not report.zero_sharealike_verified:
        raise RuntimeError("License gate did not confirm zero_sharealike_verified=True.")
    return report, oem_regions


def skyscenes_cam_rot_wc_enu(pitch_down_deg: float) -> np.ndarray:
    """Right-handed camera-optical-to-world-ENU rotation for SkyScenes downward pitch `P` deg.

    Camera optical frame: +X right (East), +Y image-down, +Z optical-forward (tilted `P` deg
    down from North). At nadir (P=90 deg), +Z = [0, 0, -1], +X = [1, 0, 0], +Y = [0, -1, 0].
    det(R) = +1.0 for all pitch angles.
    """
    p = math.radians(float(pitch_down_deg))
    c, s = math.cos(p), math.sin(p)
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -s, c],
            [0.0, -c, -s],
        ],
        dtype=np.float32,
    )
    return rot


def decode_carla_depth_meters(arr: np.ndarray, far_plane_m: float = 1000.0) -> np.ndarray:
    """Decode standard CARLA / SkyScenes 24-bit RGB depth PNG into metric depth (meters)."""
    if arr.ndim == 3 and arr.shape[2] >= 3:
        r = arr[..., 0].astype(np.float64)
        g = arr[..., 1].astype(np.float64)
        b = arr[..., 2].astype(np.float64)
        normalized = (r + g * 256.0 + b * 65536.0) / (256.0**3 - 1.0)
        return (normalized * float(far_plane_m)).astype(np.float32)
    raw = arr.astype(np.float32)
    if raw.max() > 1.0 and raw.max() <= 65535.0:
        return (raw / 65535.0) * float(far_plane_m)
    return raw * float(far_plane_m)


def compute_binary_auroc_and_fpr95(
    scores: np.ndarray,
    is_ood: np.ndarray,
) -> tuple[float, float]:
    """Exact rank-based AUROC and FPR@95%TPR (higher score => OOD positive)."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(is_ood, dtype=np.bool_).reshape(-1)
    finite = np.isfinite(s)
    s = s[finite]
    y = y[finite]

    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Sort descending by score; compute exact ROC curve.
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted, dtype=np.float64)
    fp = np.cumsum(~y_sorted, dtype=np.float64)

    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)

    # Prepend (0, 0) for exact trapezoidal AUC
    tpr_ext = np.concatenate(([0.0], tpr))
    fpr_ext = np.concatenate(([0.0], fpr))
    auroc = float(np.trapezoid(tpr_ext, fpr_ext) if hasattr(np, "trapezoid") else np.trapz(tpr_ext, fpr_ext))

    idx95 = int(np.searchsorted(tpr, 0.95, side="left"))
    idx95 = min(idx95, len(fpr) - 1)
    fpr95 = float(fpr[idx95])
    return auroc, fpr95


def compute_confusion_metrics(conf: np.ndarray) -> dict[str, Any]:
    """Compute exact per-class IoU, mIoU, pixel accuracy, Safe precision, and False-Safe Rate."""
    c = np.asarray(conf, dtype=np.int64)
    if c.shape != (NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES):
        raise ValueError(f"Expected 5x5 confusion matrix, got {c.shape}")

    per_class_iou: dict[str, float | None] = {}
    present_ious: list[float] = []
    all_ious: list[float] = []

    for idx, name in enumerate(CANONICAL_CLASS_NAMES):
        tp = int(c[idx, idx])
        gt_sum = int(c[idx, :].sum())
        pred_sum = int(c[:, idx].sum())
        union = gt_sum + pred_sum - tp
        if gt_sum == 0:
            per_class_iou[name] = None
        else:
            iou = float(tp) / float(max(1, union))
            per_class_iou[name] = round(iou, 6)
            present_ious.append(iou)
            all_ious.append(iou)

    total_pix = int(c.sum())
    correct_pix = int(np.trace(c))
    pix_acc = float(correct_pix) / float(max(1, total_pix))

    # Safety-critical False-Safe Rate (FSR): true hazard {2,3,4} predicted as safe {0,1}
    true_hazard_total = int(c[2:, :].sum())
    false_safe_pix = int(c[2:, :2].sum())
    fsr = float(false_safe_pix) / float(max(1, true_hazard_total))

    # Safe-class precision: among predicted safe {0,1}, fraction truly safe {0,1}
    pred_safe_total = int(c[:, :2].sum())
    true_safe_pred_safe = int(c[:2, :2].sum())
    safe_precision = float(true_safe_pred_safe) / float(max(1, pred_safe_total))

    return {
        "per_class_iou": per_class_iou,
        "miou_present_classes": round(float(np.mean(present_ious)) if present_ious else 0.0, 6),
        "num_present_classes": len(present_ious),
        "pixel_accuracy": round(pix_acc, 6),
        "false_safe_rate_fsr": round(fsr, 6),
        "safe_precision": round(safe_precision, 6),
        "total_eval_pixels": total_pix,
        "true_hazard_pixels": true_hazard_total,
        "false_safe_pixels": false_safe_pix,
        "confusion_matrix_5x5": c.tolist(),
    }


def query_gpu_info() -> dict[str, Any]:
    """Query live NVIDIA GPU metadata via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": int(float(parts[2])),
        "memory_free_before_mib": int(float(parts[3])),
        "compute_cap": parts[4],
    }


# ---------------------------------------------------------------------------
# Dataset preparation (Colab L4 side; strictly Zero-ShareAlike)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleRecord:
    """Single prepared 512x512 RGB + canonical mask (+ optional metric normal) sample."""

    source: str
    split: str
    rgb_path: str
    mask_path: str
    normal_path: str | None = None
    group: str = ""


def _save_pair_512(
    rgb_img: Any,
    mask_u8: np.ndarray,
    rgb_out: Path,
    mask_out: Path,
    *,
    normals_hw3: np.ndarray | None = None,
    normal_out: Path | None = None,
    size: int = 512,
) -> None:
    from PIL import Image

    rgb_out.parent.mkdir(parents=True, exist_ok=True)
    mask_out.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(rgb_img, np.ndarray):
        pil_rgb = Image.fromarray(rgb_img.astype(np.uint8, copy=False))
    else:
        pil_rgb = rgb_img.convert("RGB")

    if pil_rgb.size != (size, size):
        pil_rgb = pil_rgb.resize((size, size), resample=Image.Resampling.BILINEAR)
    pil_rgb.save(rgb_out, format="JPEG", quality=92)

    pil_mask = Image.fromarray(mask_u8.astype(np.uint8, copy=False))
    if pil_mask.size != (size, size):
        pil_mask = pil_mask.resize((size, size), resample=Image.Resampling.NEAREST)
    pil_mask.save(mask_out, format="PNG")

    if normals_hw3 is not None and normal_out is not None:
        normal_out.parent.mkdir(parents=True, exist_ok=True)
        # Quantize [-1, 1] unit normals to uint8 [0, 255] (zero normal -> 128)
        n_q = np.clip(np.round((normals_hw3 + 1.0) * 127.5), 0.0, 255.0).astype(np.uint8)
        pil_n = Image.fromarray(n_q)
        if pil_n.size != (size, size):
            pil_n = pil_n.resize((size, size), resample=Image.Resampling.BILINEAR)
        pil_n.save(normal_out, format="PNG")


def _assign_group_splits(
    groups: list[str], val_frac: float = 0.15, seed: int = 20261123
) -> set[str]:
    """Deterministic group-level train/val split preventing spatial leakage."""
    uniq = sorted(set(groups))
    if len(uniq) <= 1:
        return set()
    rng = random.Random(seed)
    shuffled = list(uniq)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return set(shuffled[:n_val])


def _prepare_flair1(
    work_dir: Path,
    max_samples: int,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    import tifffile

    raw_dir = work_dir / "raw" / "flair1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "flair_1_toy_dataset.zip"

    print(f"[FLAIR-1] Downloading official IGN etalab-2.0 archive: {FLAIR1_TOY_URL}", flush=True)
    urllib.request.urlretrieve(FLAIR1_TOY_URL, zip_path)
    zip_sha = sha256_file(zip_path)
    zip_mb = round(zip_path.stat().st_size / (1024.0 * 1024.0), 2)

    extract_dir = raw_dir / "extracted"
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink(missing_ok=True)

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
        pairs = pairs[:max_samples]

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
    print(f"[FLAIR-1] Prepared {len(records)} samples (train={n_train}, val={n_val})", flush=True)
    return records, {
        "source": "FLAIR-1",
        "license": "OPEN LICENCE 2.0 (etalab-2.0)",
        "uses_osm_odbl": False,
        "archive_url": FLAIR1_TOY_URL,
        "archive_mb": zip_mb,
        "archive_sha256": zip_sha,
        "train_samples": n_train,
        "val_samples": n_val,
        "unique_zones": len(set(r.group for r in records)),
    }


def _prepare_openearthmap(
    work_dir: Path,
    oem_whitelist: frozenset[str],
    max_samples: int,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import tifffile
    from remotezip import RemoteZip

    print(
        f"[OEM-41] Reading central directory via HTTP Range (RemoteZip) from Zenodo 7223446: "
        f"{OEM_ZENODO_ZIP_URL}",
        flush=True,
    )
    with RemoteZip(OEM_ZENODO_ZIP_URL, initial_buffer_size=1024 * 1024) as rz:
        names = rz.namelist()

    # Map (region_dir, filename) -> zip member for images/ and labels/
    img_members: dict[tuple[str, str], str] = {}
    lbl_members: list[tuple[str, str, str]] = []
    for n in names:
        if n.endswith("/") or "__macosx" in n.lower():
            continue
        parts = Path(n).parts
        if len(parts) < 3:
            continue
        parent = parts[-2].lower()
        region_dir = parts[-3]
        fname = parts[-1]
        if not fname.lower().endswith((".tif", ".tiff", ".png")):
            continue
        if parent == "images":
            img_members[(region_dir, fname)] = n
        elif parent == "labels":
            lbl_members.append((region_dir, fname, n))

    lbl_members.sort()
    total_pairs_seen = 0
    rejected_pairs = 0
    rejected_regions: set[str] = set()
    allowed_pairs: list[tuple[str, str, str, str]] = []
    for region_dir, fname, lbl_m in lbl_members:
        img_m = img_members.get((region_dir, fname))
        if img_m is None:
            continue
        total_pairs_seen += 1
        matched = filter_oem_region(region_dir, oem_whitelist)
        if matched is None:
            matched = filter_oem_region(fname, oem_whitelist)
        if matched is None:
            rejected_pairs += 1
            rejected_regions.add(region_dir.lower())
            continue
        allowed_pairs.append((matched, region_dir, img_m, lbl_m))

    if max_samples > 0 and len(allowed_pairs) > max_samples:
        stride = np.linspace(0, len(allowed_pairs) - 1, max_samples, dtype=int)
        allowed_pairs = [allowed_pairs[i] for i in stride]

    print(
        f"[OEM-41] Whitelist filter complete BEFORE pixel download: {len(allowed_pairs)} selected "
        f"(total_scanned={total_pairs_seen}, rejected_non_permissive={rejected_pairs}). "
        f"Streaming lossless GeoTIFF pairs via 4 parallel HTTP Range workers...",
        flush=True,
    )

    out_root = work_dir / "prepared" / "oem"
    (out_root / "all" / "rgb").mkdir(parents=True, exist_ok=True)
    (out_root / "all" / "mask").mkdir(parents=True, exist_ok=True)

    tls = threading.local()

    def _get_rz() -> Any:
        rz_inst = getattr(tls, "rz", None)
        if rz_inst is None:
            rz_inst = RemoteZip(OEM_ZENODO_ZIP_URL, initial_buffer_size=1024 * 1024)
            tls.rz = rz_inst
        return rz_inst

    def _fetch_one(job: tuple[int, str, str, str]) -> tuple[Path, Path, str, int, int, list[int]] | None:
        idx, matched, img_m, lbl_m = job
        for attempt in range(3):
            try:
                rz_inst = _get_rz()
                img_bytes = rz_inst.read(img_m)
                lbl_bytes = rz_inst.read(lbl_m)
                rgb_arr = tifffile.imread(io.BytesIO(img_bytes))
                raw_msk = tifffile.imread(io.BytesIO(lbl_bytes))
                if rgb_arr.ndim == 3 and rgb_arr.shape[0] <= 8 and rgb_arr.shape[0] < rgb_arr.shape[1]:
                    rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
                rgb = rgb_arr[..., :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                if raw_msk.ndim == 3:
                    raw_msk = raw_msk[..., 0] if raw_msk.shape[2] <= 4 else raw_msk[0]
                canon_msk = remap_mask("OpenEarthMap", raw_msk, oem_zero_indexed=False)
                if float((canon_msk != IGNORE_INDEX).mean()) < 0.05:
                    return None
                rgb_out = out_root / "all" / "rgb" / f"oem_{idx:05d}.jpg"
                msk_out = out_root / "all" / "mask" / f"oem_{idx:05d}.png"
                _save_pair_512(rgb, canon_msk, rgb_out, msk_out)
                return (
                    rgb_out,
                    msk_out,
                    matched,
                    int(raw_msk.min()),
                    int(raw_msk.max()),
                    np.unique(raw_msk).tolist() if idx == 0 else [],
                )
            except Exception:
                # Reset thread-local RemoteZip handle on transient socket error
                old = getattr(tls, "rz", None)
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
                tls.rz = None
                if attempt == 2:
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    accepted_regions: Counter[str] = Counter()
    oem_mask_min = 255
    oem_mask_max = 0
    staged: list[tuple[Path, Path, str]] = []
    jobs = [(i, m, im, lm) for i, (m, _rd, im, lm) in enumerate(allowed_pairs)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(_fetch_one, j) for j in jobs]
        done_cnt = 0
        for fut in as_completed(futs):
            done_cnt += 1
            res = fut.result()
            if res is not None:
                rgb_out, msk_out, matched, rmin, rmax, uniq0 = res
                oem_mask_min = min(oem_mask_min, rmin)
                oem_mask_max = max(oem_mask_max, rmax)
                if uniq0:
                    print(
                        f"[OEM-41] Frame0 region={matched} unique_raw={uniq0}",
                        flush=True,
                    )
                staged.append((rgb_out, msk_out, matched))
                accepted_regions[matched] += 1
            if done_cnt % 200 == 0 or done_cnt == len(jobs):
                print(
                    f"[OEM-41] Streamed {done_cnt}/{len(jobs)} pairs (saved={len(staged)})",
                    flush=True,
                )

    staged.sort(key=lambda t: str(t[0]))
    # Region-grouped 85/15 train/val split (zero geographic leakage)
    val_regions = _assign_group_splits([r for _, _, r in staged], val_frac=0.15, seed=202)
    records: list[SampleRecord] = []
    for rgb_p, msk_p, reg in staged:
        split = "val" if reg in val_regions else "train"
        records.append(
            SampleRecord(
                source="OpenEarthMap-41",
                split=split,
                rgb_path=str(rgb_p),
                mask_path=str(msk_p),
                group=reg,
            )
        )

    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    print(
        f"[OEM-41] Prepared {len(records)} lossless whitelisted samples "
        f"(train={n_train}, val={n_val}, accepted_regions={len(accepted_regions)}, "
        f"rejected_pairs={rejected_pairs}, raw_mask_range=[{oem_mask_min},{oem_mask_max}])",
        flush=True,
    )
    return records, {
        "source": "OpenEarthMap-41",
        "license": "CC BY 4.0 + DL-DE-BY-2.0 ONLY (strictly filtered via live attribution.html)",
        "uses_osm_odbl": False,
        "archive_url": OEM_ZENODO_ZIP_URL,
        "ingestion_method": "selective_http_range_remotezip_zero_non_permissive_bytes_downloaded",
        "total_labeled_pairs_scanned": total_pairs_seen,
        "accepted_pairs": len(records),
        "rejected_non_permissive_pairs": rejected_pairs,
        "accepted_region_count": len(accepted_regions),
        "accepted_regions": sorted(accepted_regions.keys()),
        "rejected_region_count": len(rejected_regions),
        "raw_mask_min": oem_mask_min,
        "raw_mask_max": oem_mask_max,
        "oem_zero_indexed": False,
        "train_samples": n_train,
        "val_samples": n_val,
    }


def _prepare_floodnet(
    work_dir: Path,
    max_samples: int,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from PIL import Image
    from remotezip import RemoteZip

    floodnet_url = f"https://huggingface.co/datasets/{FLOODNET_HF_REPO_ID}/resolve/main/floodnet.zip"
    print(
        f"[FloodNet] Reading CDLA-Permissive-1.0 central directory via RemoteZip: {floodnet_url}",
        flush=True,
    )
    with RemoteZip(floodnet_url, initial_buffer_size=1024 * 1024) as rz:
        names = rz.namelist()

    # Map labeled masks (*_lab.png) to their corresponding RGB image inside Labeled/*/image/
    img_by_key: dict[tuple[str, str], str] = {}
    masks: list[tuple[str, str, str]] = []
    for n in names:
        low = n.lower()
        if "__macosx" in low or n.endswith("/") or "vqa" in low:
            continue
        if "/labeled/" not in low:
            continue
        p = Path(n)
        group_tag = "flooded" if "/flooded/" in low else "non_flooded"
        if low.endswith("_lab.png") and "/mask/" in low:
            stem = p.name[:-8]
            masks.append((group_tag, stem, n))
        elif p.suffix.lower() in (".jpg", ".jpeg", ".png") and "/image/" in low:
            img_by_key[(group_tag, p.stem)] = n

    masks.sort()
    matched_jobs: list[tuple[int, str, str, str, str]] = []
    # Deterministic stratified 85/15 train/val split across Flooded & Non-Flooded labeled frames
    by_group: dict[str, list[tuple[str, str]]] = {"flooded": [], "non_flooded": []}
    for grp, stem, msk_m in masks:
        img_m = img_by_key.get((grp, stem))
        if img_m is not None:
            by_group[grp].append((img_m, msk_m))

    idx = 0
    for grp, grp_pairs in by_group.items():
        rng = random.Random(303 if grp == "flooded" else 304)
        shuffled = list(grp_pairs)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * 0.15))) if len(shuffled) > 1 else 0
        for pos, (img_m, msk_m) in enumerate(shuffled):
            split_tag = "val" if pos < n_val else "train"
            matched_jobs.append((idx, split_tag, grp, img_m, msk_m))
            idx += 1

    if max_samples > 0 and len(matched_jobs) > max_samples:
        matched_jobs = matched_jobs[:max_samples]

    print(
        f"[FloodNet] Streaming {len(matched_jobs)} labeled UAV pairs via 8 parallel Range workers...",
        flush=True,
    )
    out_root = work_dir / "prepared" / "floodnet"
    for sp in ("train", "val"):
        (out_root / sp / "rgb").mkdir(parents=True, exist_ok=True)
        (out_root / sp / "mask").mkdir(parents=True, exist_ok=True)

    tls = threading.local()

    def _get_rz() -> Any:
        inst = getattr(tls, "rz", None)
        if inst is None:
            inst = RemoteZip(floodnet_url, initial_buffer_size=1024 * 1024)
            tls.rz = inst
        return inst

    def _fetch_one(
        job: tuple[int, str, str, str, str]
    ) -> tuple[SampleRecord, int, int, list[int]] | None:
        j_idx, split_tag, grp, img_m, msk_m = job
        for attempt in range(3):
            try:
                rz_inst = _get_rz()
                pil_img = Image.open(io.BytesIO(rz_inst.read(img_m))).convert("RGB")
                raw_msk = np.asarray(Image.open(io.BytesIO(rz_inst.read(msk_m))))
                if raw_msk.ndim == 3:
                    raw_msk = raw_msk[..., 0]
                canon = remap_mask("FloodNet-Supervised-v1.0", raw_msk)
                rgb_out = out_root / split_tag / "rgb" / f"floodnet_{j_idx:05d}.jpg"
                msk_out = out_root / split_tag / "mask" / f"floodnet_{j_idx:05d}.png"
                _save_pair_512(pil_img, canon, rgb_out, msk_out)
                rec = SampleRecord(
                    source="FloodNet-Supervised-v1.0",
                    split=split_tag,
                    rgb_path=str(rgb_out),
                    mask_path=str(msk_out),
                    group=f"floodnet_{grp}_{split_tag}",
                )
                return (
                    rec,
                    int(raw_msk.min()),
                    int(raw_msk.max()),
                    np.unique(raw_msk).tolist() if j_idx == 0 else [],
                )
            except Exception:
                old = getattr(tls, "rz", None)
                if old is not None:
                    try:
                        old.close()
                    except Exception:
                        pass
                tls.rz = None
                if attempt == 2:
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    records: list[SampleRecord] = []
    raw_min, raw_max = 255, 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_fetch_one, j) for j in matched_jobs]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                rec, rmin, rmax, uniq0 = res
                raw_min = min(raw_min, rmin)
                raw_max = max(raw_max, rmax)
                if uniq0:
                    print(f"[FloodNet] Frame0 unique_raw={uniq0}", flush=True)
                records.append(rec)

    records.sort(key=lambda r: r.rgb_path)
    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    print(
        f"[FloodNet] Prepared {len(records)} UAV samples "
        f"(train={n_train}, val={n_val}, raw_mask_range=[{raw_min},{raw_max}])",
        flush=True,
    )
    return records, {
        "source": "FloodNet-Supervised-v1.0",
        "license": "CDLA-Permissive-1.0",
        "uses_osm_odbl": False,
        "hf_repo": FLOODNET_HF_REPO_ID,
        "ingestion_method": "selective_http_range_remotezip_labeled_only",
        "raw_mask_min": raw_min,
        "raw_mask_max": raw_max,
        "train_samples": n_train,
        "val_samples": n_val,
    }


def _decode_skyscenes_seg_ids(raw_arr: np.ndarray) -> np.ndarray:
    """Extract raw CARLA semantic class IDs (0..27) from SkyScenes Segment PNG."""
    if raw_arr.ndim == 2:
        return raw_arr.astype(np.uint8, copy=False)
    # CARLA raw semantic tag PNG stores class ID in R channel with G=B=0 (or identical channels)
    r = raw_arr[..., 0]
    if int(r.max()) <= 30:
        return r.astype(np.uint8, copy=False)
    # Fallback: SkyScenes sometimes stores Cityscapes/CARLA color-coded PNGs; map exact RGB palette
    carla_palette: dict[tuple[int, int, int], int] = {
        (0, 0, 0): 0,
        (70, 70, 70): 1,       # Building
        (100, 40, 40): 2,      # Fence
        (55, 90, 80): 3,       # Other
        (220, 20, 60): 4,      # Pedestrian
        (153, 153, 153): 5,    # Pole
        (157, 234, 50): 6,     # RoadLine
        (128, 64, 128): 7,     # Road
        (244, 35, 232): 8,     # Sidewalk
        (107, 142, 35): 9,     # Vegetation
        (0, 0, 142): 10,       # Vehicles / Car
        (102, 102, 156): 11,   # Wall
        (220, 220, 0): 12,     # TrafficSign
        (70, 130, 180): 13,    # Sky
        (81, 0, 81): 14,       # Ground
        (150, 100, 100): 15,   # Bridge
        (230, 150, 140): 16,   # RailTrack
        (180, 165, 180): 17,   # GuardRail
        (250, 170, 30): 18,    # TrafficLight
        (110, 190, 160): 19,   # Static
        (170, 120, 50): 20,    # Dynamic
        (45, 60, 150): 21,     # Water
        (145, 170, 100): 22,   # Terrain
        (0, 0, 70): 23,        # Truck
        (0, 60, 100): 24,      # Bus
        (0, 0, 230): 25,       # Motorcycle
        (119, 11, 32): 26,     # Bicycle
        (255, 0, 0): 27,       # Rider
    }
    packed = (
        raw_arr[..., 0].astype(np.int32) * 65536
        + raw_arr[..., 1].astype(np.int32) * 256
        + raw_arr[..., 2].astype(np.int32)
    )
    out = np.zeros(raw_arr.shape[:2], dtype=np.uint8)
    for (cr, cg, cb), cid in carla_palette.items():
        key = cr * 65536 + cg * 256 + cb
        out[packed == key] = np.uint8(cid)
    return out


def _prepare_skyscenes(
    work_dir: Path,
    max_samples: int,
) -> tuple[list[SampleRecord], dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image

    api = HfApi()
    all_files = set(api.list_repo_files(SKYSCENES_HF_REPO_ID, repo_type="dataset"))

    # Select diverse oblique + nadir viewpoints across distinct CARLA towns (strict town holdout)
    requested_specs: list[tuple[str, str, str, str, int]] = [
        ("H_35_P_45", "ClearNoon", "Town01", "train", 450),
        ("H_35_P_60", "ClearNoon", "Town02", "train", 450),
        ("H_60_P_90", "ClearNoon", "Town03", "train", 450),
        ("H_35_P_45", "ClearNoon", "Town05", "val", 250),
    ]
    # Fallback val town if Town05 absent
    selected_specs: list[tuple[str, str, str, str, int]] = []
    for hp, weather, town, split_tag, cap in requested_specs:
        img_rel = f"Images/{hp}/{weather}/{town}/{town}.tar.gz"
        seg_rel = f"Segment/{hp}/ClearNoon/{town}/{town}.tar.gz"
        dep_rel = f"Depth/{hp}/ClearNoon/{town}/{town}.tar.gz"
        if img_rel in all_files and seg_rel in all_files and dep_rel in all_files:
            selected_specs.append((hp, weather, town, split_tag, cap))
        elif split_tag == "val":
            for alt_town in ("Town04", "Town06", "Town07", "Town10HD"):
                if (
                    f"Images/{hp}/{weather}/{alt_town}/{alt_town}.tar.gz" in all_files
                    and f"Segment/{hp}/ClearNoon/{alt_town}/{alt_town}.tar.gz" in all_files
                    and f"Depth/{hp}/ClearNoon/{alt_town}/{alt_town}.tar.gz" in all_files
                ):
                    selected_specs.append((hp, weather, alt_town, split_tag, cap))
                    break

    raw_cache = work_dir / "raw" / "skyscenes"
    out_root = work_dir / "prepared" / "skyscenes"
    records: list[SampleRecord] = []
    facade_remapped_pixels = 0
    roof_kept_pixels = 0
    towns_used: list[str] = []
    idx = 0

    for hp, weather, town, split_tag, cap in selected_specs:
        if max_samples > 0 and len(records) >= max_samples:
            break
        per_shard_cap = cap
        if max_samples > 0:
            per_shard_cap = min(cap, max_samples - len(records))
        pitch_match = re.search(r"P_(\d+)", hp)
        pitch_deg = float(pitch_match.group(1)) if pitch_match else 45.0
        rot_wc = skyscenes_cam_rot_wc_enu(pitch_deg)
        if abs(float(np.linalg.det(rot_wc)) - 1.0) > 1e-4:
            raise RuntimeError(f"Invalid rotation determinant for pitch={pitch_deg}")

        raw_cache.mkdir(parents=True, exist_ok=True)
        print(f"[SkyScenes] Downloading MIT shard {hp}/{weather}/{town} ({split_tag})...", flush=True)
        img_tar = Path(
            hf_hub_download(
                SKYSCENES_HF_REPO_ID,
                f"Images/{hp}/{weather}/{town}/{town}.tar.gz",
                repo_type="dataset",
                local_dir=str(raw_cache),
            )
        )
        seg_tar = Path(
            hf_hub_download(
                SKYSCENES_HF_REPO_ID,
                f"Segment/{hp}/ClearNoon/{town}/{town}.tar.gz",
                repo_type="dataset",
                local_dir=str(raw_cache),
            )
        )
        dep_tar = Path(
            hf_hub_download(
                SKYSCENES_HF_REPO_ID,
                f"Depth/{hp}/ClearNoon/{town}/{town}.tar.gz",
                repo_type="dataset",
                local_dir=str(raw_cache),
            )
        )

        ext_img = raw_cache / "ext_img"
        ext_seg = raw_cache / "ext_seg"
        ext_dep = raw_cache / "ext_dep"
        for tpath, dst in ((img_tar, ext_img), (seg_tar, ext_seg), (dep_tar, ext_dep)):
            dst.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tpath, "r:*") as tf:
                tf.extractall(dst)
            tpath.unlink(missing_ok=True)

        img_map = {
            p.stem: p
            for p in ext_img.rglob("*")
            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        }
        seg_map = {
            p.stem: p
            for p in ext_seg.rglob("*")
            if p.is_file() and p.suffix.lower() == ".png"
        }
        dep_map = {
            p.stem: p
            for p in ext_dep.rglob("*")
            if p.is_file() and p.suffix.lower() == ".png"
        }

        common_stems = sorted(set(img_map) & set(seg_map) & set(dep_map))
        if len(common_stems) > per_shard_cap:
            # Uniformly stride across sequence for maximum spatial diversity
            stride = np.linspace(0, len(common_stems) - 1, per_shard_cap, dtype=int)
            common_stems = [common_stems[i] for i in stride]

        towns_used.append(f"{hp}/{town}({split_tag}:{len(common_stems)})")
        for stem in common_stems:
            pil_rgb = Image.open(img_map[stem]).convert("RGB")
            raw_seg = _decode_skyscenes_seg_ids(np.asarray(Image.open(seg_map[stem])))
            raw_dep = np.asarray(Image.open(dep_map[stem]))
            depth_m = decode_carla_depth_meters(raw_dep, far_plane_m=1000.0)

            # Resize high-res CARLA frames to 512x512 BEFORE gradient normal calc for speed & consistency
            if raw_seg.shape != (512, 512):
                pil_rgb = pil_rgb.resize((512, 512), resample=Image.Resampling.BILINEAR)
                raw_seg = np.asarray(
                    Image.fromarray(raw_seg).resize(
                        (512, 512), resample=Image.Resampling.NEAREST
                    )
                )
                depth_m = np.asarray(
                    Image.fromarray(depth_m).resize(
                        (512, 512), resample=Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                )

            h, w = raw_seg.shape
            # SkyScenes CARLA camera uses 90 deg horizontal FOV -> fx = fy = W / 2
            intr = (float(w) * 0.5, float(w) * 0.5, float(w) * 0.5, float(h) * 0.5)
            normals_w, valid_n = compute_world_normals_from_depth(depth_m, rot_wc, intr)
            canon = remap_mask(
                "SkyScenes",
                raw_seg,
                depth_m=depth_m,
                cam_rot_wc=rot_wc,
                intrinsics=intr,
                skyscenes_nz_threshold=0.85,
            )

            raw_bldg = raw_seg == 1
            roof_kept_pixels += int((raw_bldg & (canon == 1)).sum())
            facade_remapped_pixels += int((raw_bldg & (canon == 2)).sum())
            if idx == 0:
                p5, p50, p95 = np.percentile(depth_m, [5, 50, 95]).tolist()
                print(
                    f"[SkyScenes] Frame0 {hp}/{town} raw_seg_uniq={np.unique(raw_seg).tolist()[:18]} "
                    f"depth_m(p5/p50/p95)=({p5:.1f}/{p50:.1f}/{p95:.1f}) "
                    f"canon_uniq={np.unique(canon).tolist()}",
                    flush=True,
                )

            normals_out = np.where(valid_n[..., None], normals_w, np.float32(0.0))
            rgb_out = out_root / split_tag / "rgb" / f"sky_{idx:05d}.jpg"
            msk_out = out_root / split_tag / "mask" / f"sky_{idx:05d}.png"
            nrm_out = out_root / split_tag / "normal" / f"sky_{idx:05d}.png"
            _save_pair_512(
                pil_rgb, canon, rgb_out, msk_out, normals_hw3=normals_out, normal_out=nrm_out
            )
            records.append(
                SampleRecord(
                    source="SkyScenes",
                    split=split_tag,
                    rgb_path=str(rgb_out),
                    mask_path=str(msk_out),
                    normal_path=str(nrm_out),
                    group=f"{hp}_{town}",
                )
            )
            idx += 1

        shutil.rmtree(raw_cache, ignore_errors=True)

    n_train = sum(1 for r in records if r.split == "train")
    n_val = sum(1 for r in records if r.split == "val")
    total_bldg = roof_kept_pixels + facade_remapped_pixels
    facade_ratio = float(facade_remapped_pixels) / float(max(1, total_bldg))
    print(
        f"[SkyScenes] Prepared {len(records)} MIT frames (train={n_train}, val={n_val}, "
        f"facade_remapped_frac={facade_ratio:.3f})",
        flush=True,
    )
    return records, {
        "source": "SkyScenes",
        "license": "MIT",
        "uses_osm_odbl": False,
        "hf_repo": SKYSCENES_HF_REPO_ID,
        "shards": towns_used,
        "depth_normal_facade_filter_nz_threshold": 0.85,
        "building_roof_kept_pixels": roof_kept_pixels,
        "building_facade_remapped_to_class2_pixels": facade_remapped_pixels,
        "facade_remap_fraction_of_building": round(facade_ratio, 6),
        "train_samples": n_train,
        "val_samples": n_val,
    }


# ---------------------------------------------------------------------------
# PyTorch Dataset, Teachers, Training & Evaluation
# ---------------------------------------------------------------------------


def _build_torch_dataset(records: list[SampleRecord], is_train: bool) -> Any:
    import torch
    from PIL import Image
    from torch.utils.data import Dataset

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    class SlzPreparedDataset(Dataset):
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
                # Random resized crop (0.75..1.0 scale) preserves aerial geometry cleanly
                if random.random() < 0.7:
                    w, h = rgb_pil.size
                    scale = random.uniform(0.75, 1.0)
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
                # Mild brightness & contrast jitter
                gain = random.uniform(0.85, 1.15)
                bias = random.uniform(-0.06, 0.06)
                rgb_np = np.clip(rgb_np * gain + bias, 0.0, 1.0)

            rgb_t = (torch.from_numpy(rgb_np).permute(2, 0, 1) - mean) / std
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

    return SlzPreparedDataset(records, augment=is_train)


class FrozenApache2Teachers:
    """Frozen Apache-2.0 DINOv2-ViT-L/14 + Depth-Anything-V2-Small teacher bundle."""

    def __init__(self, device: Any, *, enabled: bool = True) -> None:
        import torch
        from transformers import AutoModel, AutoModelForDepthEstimation

        self.device = device
        self.enabled = enabled
        self.dino: Any = None
        self.dav2: Any = None
        self.metadata: dict[str, Any] = {
            "enabled": enabled,
            "dinov2_repo": DINOV2_REPO_ID,
            "dinov2_license": "Apache-2.0",
            "dav2_repo": DAV2_SMALL_REPO_ID,
            "dav2_license": "Apache-2.0",
        }
        if not enabled:
            return

        print(f"[Teachers] Loading frozen Apache-2.0 {DINOV2_REPO_ID}...", flush=True)
        self.dino = AutoModel.from_pretrained(DINOV2_REPO_ID).to(device).eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

        print(f"[Teachers] Loading frozen Apache-2.0 {DAV2_SMALL_REPO_ID}...", flush=True)
        self.dav2 = AutoModelForDepthEstimation.from_pretrained(DAV2_SMALL_REPO_ID).to(device).eval()
        for p in self.dav2.parameters():
            p.requires_grad_(False)

        self.metadata["dinov2_hidden_size"] = int(getattr(self.dino.config, "hidden_size", 1024))
        self.metadata["dinov2_num_register_tokens"] = int(
            getattr(self.dino.config, "num_register_tokens", 0)
        )
        self.metadata["dav2_model_type"] = str(getattr(self.dav2.config, "model_type", "depth_anything"))

    def infer(
        self,
        rgb_512: Any,
        gt_normals: Any,
        has_gt_normal: Any,
        *,
        use_bf16: bool = True,
    ) -> tuple[Any, Any]:
        """Return `(teacher_dino (B,1024,37,37), target_normals (B,3,512,512))`."""
        import torch
        import torch.nn.functional as F

        if not self.enabled or self.dino is None or self.dav2 is None:
            return None, gt_normals

        device_type = "cuda" if rgb_512.is_cuda else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
        ):
            rgb_518 = F.interpolate(
                rgb_512, size=(518, 518), mode="bilinear", align_corners=False
            )
            dino_out = self.dino(pixel_values=rgb_518).last_hidden_state
            n_reg = int(getattr(self.dino.config, "num_register_tokens", 0))
            patches = dino_out[:, 1 + n_reg :, :]  # (B, 1369, 1024)
            bsz, n_tok, dim = patches.shape
            grid = int(round(math.sqrt(n_tok)))
            teacher_dino = patches.transpose(1, 2).reshape(bsz, dim, grid, grid).float()

            inv_d = self.dav2(pixel_values=rgb_518).predicted_depth  # (B, 518, 518)
            inv_d_512 = F.interpolate(
                inv_d.unsqueeze(1).float(),
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            # Scale-normalize per-frame inverse depth before finite-difference pseudo-normals
            d_mean = inv_d_512.mean(dim=(2, 3), keepdim=True)
            d_std = inv_d_512.std(dim=(2, 3), keepdim=True).clamp(min=1e-4)
            pseudo_n = pseudo_normals_from_inv_depth(((inv_d_512 - d_mean) / d_std) * 2.0)

        # Prefer exact metric 3D normals where available (SkyScenes), else DAv2-Small pseudo-normals
        mask_gt = has_gt_normal.to(device=rgb_512.device, dtype=torch.bool).view(-1, 1, 1, 1)
        target_n = torch.where(mask_gt, gt_normals.to(rgb_512.device), pseudo_n)
        return teacher_dino, target_n


def evaluate_permissive_val_and_ood(
    model: Any,
    val_loader: Any,
    teachers: FrozenApache2Teachers,
    device: Any,
    *,
    use_bf16: bool = True,
    kappa_b: float = 1.5,
    max_cal_pixels: int = 250_000,
    eval_normals: bool = True,
    eval_ood: bool = True,
) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any]]:
    """Evaluate clean permissive validation mIoU/FSR/normals + deterministic AnomalyMix OOD."""
    import torch
    import torch.nn.functional as F

    from ml.slz_perception.train.calibrate_ood import compute_free_energy

    model.eval()
    conf = np.zeros((NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES), dtype=np.int64)
    normal_cos_sum = 0.0
    normal_cos_count = 0

    cal_logits_chunks: list[torch.Tensor] = []
    cal_labels_chunks: list[torch.Tensor] = []
    cal_bdry_chunks: list[torch.Tensor] = []

    ood_raw_scores: list[np.ndarray] = []
    ood_comp_scores: list[np.ndarray] = []
    ood_flags: list[np.ndarray] = []

    ood_gen = torch.Generator(device="cpu").manual_seed(20261123)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    n_batches = max(1, len(val_loader))
    pix_per_batch_budget = max(256, max_cal_pixels // n_batches)

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            gt_n = batch["gt_normal"].to(device, non_blocking=True)
            has_gt_n = batch["has_gt_normal"]

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(rgb)

            logits = out.seg_logits.float()
            normals = out.surf_normals.float()
            bdry_up = F.interpolate(
                out.bdry_prob.float(), size=(512, 512), mode="bilinear", align_corners=False
            ).squeeze(1)

            preds = logits.argmax(dim=1)
            valid = (lbl >= 0) & (lbl < NUM_CANONICAL_CLASSES)
            if bool(valid.any()):
                t_np = lbl[valid].cpu().numpy()
                p_np = preds[valid].cpu().numpy()
                idx_flat = t_np * NUM_CANONICAL_CLASSES + p_np
                conf += np.bincount(
                    idx_flat, minlength=NUM_CANONICAL_CLASSES * NUM_CANONICAL_CLASSES
                ).reshape(NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES)

                # Subsample clean ID pixels uniformly for exact post-hoc T_prob & tau_ood calibration
                v_logits = logits.permute(0, 2, 3, 1)[valid]
                v_labels = lbl[valid]
                v_bdry = bdry_up[valid]
                n_v = v_labels.shape[0]
                if n_v > pix_per_batch_budget:
                    step = max(1, n_v // pix_per_batch_budget)
                    v_logits = v_logits[::step]
                    v_labels = v_labels[::step]
                    v_bdry = v_bdry[::step]
                cal_logits_chunks.append(v_logits.cpu())
                cal_labels_chunks.append(v_labels.cpu())
                cal_bdry_chunks.append(v_bdry.cpu())

            # Normal alignment against DAv2-Small / metric SkyScenes target normals
            if eval_normals:
                _, target_n = teachers.infer(rgb, gt_n, has_gt_n, use_bf16=use_bf16)
                t_norm = torch.linalg.norm(target_n, dim=1)
                n_valid = valid & (t_norm > 1e-4)
                if bool(n_valid.any()):
                    cos_v = F.cosine_similarity(normals, target_n, dim=1, eps=1e-6)[n_valid]
                    normal_cos_sum += float(cos_v.sum().item())
                    normal_cos_count += int(cos_v.numel())

            # Deterministic held-out AnomalyMix OOD evaluation (separate forward pass, prob=0.5)
            if eval_ood:
                mix_rgb, mix_lbl = apply_anomaly_mix(
                    rgb, lbl, prob=0.5, ood_index=OOD_INDEX, generator=ood_gen
                )
                if bool((mix_lbl == OOD_INDEX).any()):
                    with torch.autocast(
                        device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16
                    ):
                        mix_out = model(mix_rgb)
                    m_logits = mix_out.seg_logits.float().permute(0, 2, 3, 1)
                    m_bdry = F.interpolate(
                        mix_out.bdry_prob.float(),
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                    e_raw = compute_free_energy(m_logits, T=1.0)
                    e_tilde = e_raw - float(kappa_b) * m_bdry

                    # Stride 4x4 across spatial dims for fast exact AUROC over ~1M+ pixels
                    m_lbl_s = mix_lbl[:, ::4, ::4]
                    e_raw_s = e_raw[:, ::4, ::4]
                    e_til_s = e_tilde[:, ::4, ::4]
                    keep = ((m_lbl_s >= 0) & (m_lbl_s < NUM_CANONICAL_CLASSES)) | (
                        m_lbl_s == OOD_INDEX
                    )
                    if bool(keep.any()):
                        ood_raw_scores.append(e_raw_s[keep].cpu().numpy())
                        ood_comp_scores.append(e_til_s[keep].cpu().numpy())
                        ood_flags.append((m_lbl_s[keep] == OOD_INDEX).cpu().numpy())

    val_metrics = compute_confusion_metrics(conf)
    mean_cos = normal_cos_sum / float(max(1, normal_cos_count))
    val_metrics["surface_normal_mean_cosine_sim"] = round(mean_cos, 6)
    val_metrics["surface_normal_mean_angle_deg"] = round(
        math.degrees(math.acos(max(-1.0, min(1.0, mean_cos)))), 3
    )

    if ood_raw_scores:
        raw_s = np.concatenate(ood_raw_scores)
        comp_s = np.concatenate(ood_comp_scores)
        flags = np.concatenate(ood_flags)
        auroc_raw, fpr95_raw = compute_binary_auroc_and_fpr95(raw_s, flags)
        auroc_comp, fpr95_comp = compute_binary_auroc_and_fpr95(comp_s, flags)
        ood_metrics = {
            "eval_protocol": "heldout_permissive_val_anomalymix_p0.5_stride4",
            "id_pixels_evaluated": int((~flags).sum()),
            "ood_pixels_evaluated": int(flags.sum()),
            "raw_energy_auroc": round(auroc_raw, 6),
            "raw_energy_fpr95": round(fpr95_raw, 6),
            "boundary_compensated_auroc": round(auroc_comp, 6),
            "boundary_compensated_fpr95": round(fpr95_comp, 6),
            "mean_id_energy": round(float(raw_s[~flags].mean()), 4),
            "mean_ood_energy": round(float(raw_s[flags].mean()), 4),
        }
    else:
        ood_metrics = {"eval_protocol": "none", "boundary_compensated_auroc": 0.0}

    cal_logits = torch.cat(cal_logits_chunks, dim=0)
    cal_labels = torch.cat(cal_labels_chunks, dim=0)
    cal_bdry = torch.cat(cal_bdry_chunks, dim=0)
    return val_metrics, cal_logits, cal_labels, cal_bdry, ood_metrics


def evaluate_quarantined_aeroscapes(
    model: Any,
    work_dir: Path,
    device: Any,
    *,
    max_samples: int = 350,
    use_bf16: bool = True,
) -> dict[str, Any]:
    """Strictly quarantined zero-shot evaluation on AeroScapes (CC BY-SA 4.0; NEVER trained)."""
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    eval_dir = work_dir / "quarantined_eval" / "aeroscapes"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[ZeroShot-Eval] Downloading quarantined AeroScapes val split ({AEROSCAPES_HF_REPO_ID}, "
        "CC BY-SA 4.0, strictly post-freeze eval only)...",
        flush=True,
    )
    snapshot_download(
        repo_id=AEROSCAPES_HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(eval_dir),
        allow_patterns=["data/images/val/*", "data/masks/val/*"],
    )

    img_dir = eval_dir / "data" / "images" / "val"
    msk_dir = eval_dir / "data" / "masks" / "val"
    imgs = sorted(img_dir.glob("*")) if img_dir.is_dir() else []
    pairs: list[tuple[Path, Path]] = []
    for ip in imgs:
        mp = msk_dir / f"{ip.stem}.png"
        if mp.is_file():
            pairs.append((ip, mp))
    if max_samples > 0 and len(pairs) > max_samples:
        stride = np.linspace(0, len(pairs) - 1, max_samples, dtype=int)
        pairs = [pairs[i] for i in stride]

    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    conf = np.zeros((NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES), dtype=np.int64)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    with torch.no_grad():
        for ip, mp in pairs:
            pil_rgb = Image.open(ip).convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
            raw_msk = np.asarray(
                Image.open(mp).resize((512, 512), Image.Resampling.NEAREST)
            )
            if raw_msk.ndim == 3:
                raw_msk = raw_msk[..., 0]
            canon = remap_mask("AeroScapes", raw_msk)

            rgb_np = np.asarray(pil_rgb, dtype=np.float32) / 255.0
            x = (torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device) - mean) / std
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16):
                out = model(x)
            pred = out.seg_logits.float().argmax(dim=1).squeeze(0).cpu().numpy()

            valid = (canon >= 0) & (canon < NUM_CANONICAL_CLASSES)
            if bool(valid.any()):
                idx_flat = canon[valid].astype(np.int64) * NUM_CANONICAL_CLASSES + pred[valid].astype(
                    np.int64
                )
                conf += np.bincount(
                    idx_flat, minlength=NUM_CANONICAL_CLASSES * NUM_CANONICAL_CLASSES
                ).reshape(NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES)

    shutil.rmtree(eval_dir, ignore_errors=True)
    metrics = compute_confusion_metrics(conf)
    metrics.update(
        {
            "dataset": "AeroScapes",
            "hf_repo": AEROSCAPES_HF_REPO_ID,
            "license": "CC BY-SA 4.0 (QUARANTINED_ZERO_SHOT_EVAL_ONLY)",
            "used_in_training": False,
            "used_in_calibration": False,
            "evaluated_images": len(pairs),
        }
    )
    print(
        f"[ZeroShot-Eval] AeroScapes ({len(pairs)} imgs): "
        f"mIoU(present)={metrics['miou_present_classes']:.4f}, "
        f"FSR={metrics['false_safe_rate_fsr']:.4f}",
        flush=True,
    )
    return metrics


def run_l4_training_pipeline(args: argparse.Namespace) -> int:
    """Execute full Colab L4 2-stage distillation, OOD calibration, eval, and ONNX export."""
    import onnx
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain
    from ml.slz_perception.train.calibrate_ood import calibrate_temperature_and_tau
    from ml.slz_perception.train.distill_losses import SlzDistillCriterion
    from ml.slz_perception.train.export_onnx_trt import export_static_onnx

    t_pipeline_start = time.perf_counter()
    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: Mandatory Zero-ShareAlike Startup License Gate BEFORE touching any dataset!
    attr_path, attr_meta = fetch_or_load_oem_attribution_html(
        work_dir / "openearthmap_attribution.html",
        local_html=args.oem_attribution_html,
    )
    active_sources = (
        "FLAIR-1",
        "OpenEarthMap",
        "FloodNet-Supervised-v1.0",
        "SkyScenes",
    )
    manifest_report, oem_regions = run_startup_license_gate(
        attr_path,
        active_sources,
        expect_oem_total=None,
        include_aeroscapes_eval=not args.skip_aeroscapes,
    )
    print(
        f"[LicenseGate] PASSED: zero_sharealike_verified={manifest_report.zero_sharealike_verified}, "
        f"train_sources={list(manifest_report.approved_train_entries)}, "
        f"oem_permissive_regions={len(oem_regions)}/{attr_meta['total_region_rows']}, "
        f"quarantined_eval={list(manifest_report.quarantined_eval_entries)}",
        flush=True,
    )
    if args.verify_only:
        return 0

    # Deterministic seeding & GPU telemetry
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    use_bf16 = bool(args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    gpu_info = query_gpu_info() if device.type == "cuda" else {"name": "CPU"}
    print(f"[Hardware] Device={device}, GPU={gpu_info.get('name')}, BF16={use_bf16}", flush=True)

    # Phase 1: Curated Zero-ShareAlike Data Preparation
    t_data0 = time.perf_counter()
    oem_whitelist = frozenset(oem_regions)
    all_records: list[SampleRecord] = []
    dataset_provenance: list[dict[str, Any]] = []

    flair_recs, flair_prov = _prepare_flair1(work_dir, max_samples=args.max_flair)
    all_records.extend(flair_recs)
    dataset_provenance.append(flair_prov)

    oem_recs, oem_prov = _prepare_openearthmap(work_dir, oem_whitelist, max_samples=args.max_oem)
    all_records.extend(oem_recs)
    dataset_provenance.append(oem_prov)

    flood_recs, flood_prov = _prepare_floodnet(work_dir, max_samples=args.max_floodnet)
    all_records.extend(flood_recs)
    dataset_provenance.append(flood_prov)

    sky_recs, sky_prov = _prepare_skyscenes(work_dir, max_samples=args.max_skyscenes)
    all_records.extend(sky_recs)
    dataset_provenance.append(sky_prov)

    data_prep_sec = round(time.perf_counter() - t_data0, 2)
    train_recs = [r for r in all_records if r.split == "train"]
    val_recs = [r for r in all_records if r.split == "val"]
    if not train_recs or not val_recs:
        raise RuntimeError(f"Insufficient split samples: train={len(train_recs)}, val={len(val_recs)}")

    # Target 60% real permissive / 40% synthetic permissive sampling weights
    target_source_ratios = {
        "FLAIR-1": 0.15,
        "OpenEarthMap-41": 0.25,
        "FloodNet-Supervised-v1.0": 0.20,
        "SkyScenes": 0.40,
    }
    src_counts = Counter(r.source for r in train_recs)
    sample_weights = [
        target_source_ratios.get(r.source, 0.1) / float(max(1, src_counts[r.source]))
        for r in train_recs
    ]
    epoch_samples = min(len(train_recs), int(args.epoch_samples)) if args.epoch_samples > 0 else len(train_recs)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=epoch_samples,
        replacement=True,
    )

    train_ds = _build_torch_dataset(train_recs, is_train=True)
    val_ds = _build_torch_dataset(val_recs, is_train=False)
    num_workers = int(args.num_workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    print(
        f"[Data] Total train={len(train_recs)} (epoch_draws={epoch_samples}), val={len(val_recs)} "
        f"in {data_prep_sec}s; train_by_source={dict(src_counts)}",
        flush=True,
    )

    # Phase 2: Frozen Apache-2.0 Teachers + Static-PAPPM PIDNet-S Student
    teachers = FrozenApache2Teachers(device, enabled=not args.no_teachers)
    model = PIDNetSlzTrain(num_classes=NUM_CANONICAL_CLASSES, dino_dim=1024).to(device)
    train_params = sum(p.numel() for p in model.parameters())
    deploy_params = sum(p.numel() for p in PIDNetSlzDeployWrapper(model).parameters())

    prof = L4TrainProfile(
        device=str(device),
        batch_size=int(args.batch_size),
        grad_accum=int(args.grad_accum),
        stage1_epochs=int(args.stage1_epochs),
        stage1_warmup_epochs=min(3, max(1, int(args.stage1_epochs) // 6)),
        stage1_lr_start=float(args.stage1_lr),
        stage1_lr_min=1e-5,
        stage2_epochs=int(args.stage2_epochs),
        stage2_lr_start=float(args.stage2_lr),
        stage2_lr_min=1e-5,
        anomaly_mix_prob=float(args.anomaly_mix_prob),
        amp_bf16=use_bf16,
        seed=seed,
    )
    s1_cfg = Stage1Config(
        epochs=prof.stage1_epochs,
        warmup_epochs=prof.stage1_warmup_epochs,
        batch_size=prof.batch_size * prof.grad_accum,
        lr_start=prof.stage1_lr_start,
        lr_min=prof.stage1_lr_min,
        amp_bf16=use_bf16,
    )
    s2_cfg = Stage2Config(
        epochs=prof.stage2_epochs,
        batch_size=prof.batch_size * prof.grad_accum,
        lr_start=prof.stage2_lr_start,
        lr_min=prof.stage2_lr_min,
        amp_bf16=use_bf16,
        anomaly_mix_prob=prof.anomaly_mix_prob,
    )

    # Stage 1: Foundation Distillation + Multi-Task Segmentation/Normals/Boundary
    t_train0 = time.perf_counter()
    crit_s1 = SlzDistillCriterion(
        lambda_bdry=s1_cfg.lambda_bdry,
        lambda_dino=s1_cfg.lambda_dino if teachers.enabled else 0.0,
        lambda_geom=s1_cfg.lambda_geom,
        lambda_ood=0.0,
    ).to(device)
    opt_s1 = torch.optim.AdamW(
        model.parameters(), lr=s1_cfg.lr_start, weight_decay=s1_cfg.weight_decay
    )

    stage1_history: list[dict[str, Any]] = []
    best_s1_miou = -1.0
    best_s1_state: dict[str, Any] | None = None

    print(
        f"[Stage1] Starting {s1_cfg.epochs} epochs (bs={prof.batch_size}x{prof.grad_accum}="
        f"{s1_cfg.batch_size}, lr={s1_cfg.lr_start}->{s1_cfg.lr_min}, BF16={use_bf16})...",
        flush=True,
    )
    for ep in range(s1_cfg.epochs):
        ep_t0 = time.perf_counter()
        lr = compute_stage1_lr(ep, s1_cfg)
        for pg in opt_s1.param_groups:
            pg["lr"] = lr

        loss_sum = 0.0
        n_steps = 0
        opt_s1.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            gt_n = batch["gt_normal"].to(device, non_blocking=True)
            has_gt_n = batch["has_gt_normal"]
            bdry_tgt = compute_boundary_target(lbl, out_hw=(64, 64))
            t_dino, t_norm = teachers.infer(rgb, gt_n, has_gt_n, use_bf16=use_bf16)

            is_accum_boundary = ((step_idx + 1) % prof.grad_accum == 0) or (
                step_idx + 1 == len(train_loader)
            )
            step_loss = train_one_step(
                model,
                crit_s1,
                opt_s1,
                rgb,
                lbl,
                target_normals=t_norm,
                target_bdry=bdry_tgt,
                teacher_dino=t_dino,
                use_bf16_amp=use_bf16,
                grad_accum_steps=prof.grad_accum,
                zero_grad=False,
                step_optimizer=is_accum_boundary,
            )
            if is_accum_boundary:
                opt_s1.zero_grad(set_to_none=True)
            loss_sum += step_loss
            n_steps += 1

        avg_loss = loss_sum / float(max(1, n_steps))
        ep_sec = time.perf_counter() - ep_t0
        rec_ep: dict[str, Any] = {
            "epoch": ep + 1,
            "lr": round(lr, 7),
            "train_loss": round(avg_loss, 5),
            "epoch_sec": round(ep_sec, 2),
        }
        if (ep + 1) % max(1, int(args.eval_every)) == 0 or (ep + 1) == s1_cfg.epochs:
            val_m, _, _, _, _ = evaluate_permissive_val_and_ood(
                model,
                val_loader,
                teachers,
                device,
                use_bf16=use_bf16,
                max_cal_pixels=50_000,
                eval_normals=False,
                eval_ood=False,
            )
            rec_ep["val_miou"] = val_m["miou_present_classes"]
            rec_ep["val_fsr"] = val_m["false_safe_rate_fsr"]
            if val_m["miou_present_classes"] > best_s1_miou:
                best_s1_miou = val_m["miou_present_classes"]
                best_s1_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[Stage1] Ep {ep + 1:02d}/{s1_cfg.epochs} | lr={lr:.2e} | loss={avg_loss:.4f} | "
                f"val_mIoU={val_m['miou_present_classes']:.4f} | val_FSR={val_m['false_safe_rate_fsr']:.4f} | "
                f"{ep_sec:.1f}s",
                flush=True,
            )
        else:
            print(
                f"[Stage1] Ep {ep + 1:02d}/{s1_cfg.epochs} | lr={lr:.2e} | loss={avg_loss:.4f} | {ep_sec:.1f}s",
                flush=True,
            )
        stage1_history.append(rec_ep)

    if best_s1_state is not None:
        model.load_state_dict(best_s1_state)

    # Stage 2: AnomalyMix Free-Energy OOD Outlier Exposure Fine-Tuning
    crit_s2 = SlzDistillCriterion(
        lambda_bdry=s2_cfg.lambda_bdry,
        lambda_dino=s2_cfg.lambda_dino if teachers.enabled else 0.0,
        lambda_geom=s2_cfg.lambda_geom,
        lambda_ood=s2_cfg.lambda_ood,
        m_in=s2_cfg.m_in,
        m_out=s2_cfg.m_out,
    ).to(device)
    opt_s2 = torch.optim.AdamW(
        model.parameters(), lr=s2_cfg.lr_start, weight_decay=s2_cfg.weight_decay
    )
    stage2_history: list[dict[str, Any]] = []
    best_s2_score = -1.0
    best_s2_state: dict[str, Any] | None = None

    print(
        f"[Stage2] Starting {s2_cfg.epochs} epochs OE (AnomalyMix p={s2_cfg.anomaly_mix_prob}, "
        f"m_in={s2_cfg.m_in}, m_out={s2_cfg.m_out}, lr={s2_cfg.lr_start}->{s2_cfg.lr_min})...",
        flush=True,
    )
    for ep in range(s2_cfg.epochs):
        ep_t0 = time.perf_counter()
        lr = compute_stage2_lr(ep, s2_cfg)
        for pg in opt_s2.param_groups:
            pg["lr"] = lr

        loss_sum = 0.0
        n_steps = 0
        opt_s2.zero_grad(set_to_none=True)
        for step_idx, batch in enumerate(train_loader):
            rgb = batch["rgb"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)
            gt_n = batch["gt_normal"].to(device, non_blocking=True)
            has_gt_n = batch["has_gt_normal"]

            mix_rgb, mix_lbl = apply_anomaly_mix(
                rgb, lbl, prob=s2_cfg.anomaly_mix_prob, ood_index=OOD_INDEX
            )
            bdry_tgt = compute_boundary_target(mix_lbl, out_hw=(64, 64))
            t_dino, t_norm = teachers.infer(mix_rgb, gt_n, has_gt_n, use_bf16=use_bf16)

            is_accum_boundary = ((step_idx + 1) % prof.grad_accum == 0) or (
                step_idx + 1 == len(train_loader)
            )
            step_loss = train_one_step(
                model,
                crit_s2,
                opt_s2,
                mix_rgb,
                mix_lbl,
                target_normals=t_norm,
                target_bdry=bdry_tgt,
                teacher_dino=t_dino,
                use_bf16_amp=use_bf16,
                grad_accum_steps=prof.grad_accum,
                zero_grad=False,
                step_optimizer=is_accum_boundary,
            )
            if is_accum_boundary:
                opt_s2.zero_grad(set_to_none=True)
            loss_sum += step_loss
            n_steps += 1

        avg_loss = loss_sum / float(max(1, n_steps))
        ep_sec = time.perf_counter() - ep_t0
        rec_ep = {
            "epoch": ep + 1,
            "lr": round(lr, 7),
            "train_loss": round(avg_loss, 5),
            "epoch_sec": round(ep_sec, 2),
        }
        if (ep + 1) % max(1, min(2, int(args.eval_every))) == 0 or (ep + 1) == s2_cfg.epochs:
            val_m, _, _, _, ood_m = evaluate_permissive_val_and_ood(
                model,
                val_loader,
                teachers,
                device,
                use_bf16=use_bf16,
                max_cal_pixels=50_000,
                eval_normals=False,
                eval_ood=True,
            )
            rec_ep["val_miou"] = val_m["miou_present_classes"]
            rec_ep["val_fsr"] = val_m["false_safe_rate_fsr"]
            rec_ep["ood_auroc"] = ood_m["boundary_compensated_auroc"]
            combo = val_m["miou_present_classes"] + 0.25 * ood_m["boundary_compensated_auroc"]
            if combo > best_s2_score:
                best_s2_score = combo
                best_s2_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[Stage2] Ep {ep + 1:02d}/{s2_cfg.epochs} | lr={lr:.2e} | loss={avg_loss:.4f} | "
                f"val_mIoU={val_m['miou_present_classes']:.4f} | val_FSR={val_m['false_safe_rate_fsr']:.4f} | "
                f"OOD_AUROC={ood_m['boundary_compensated_auroc']:.4f} | {ep_sec:.1f}s",
                flush=True,
            )
        stage2_history.append(rec_ep)

    if best_s2_state is not None:
        model.load_state_dict(best_s2_state)

    train_wall_sec = round(time.perf_counter() - t_train0, 2)

    # Phase 3: Final Frozen Evaluation & Post-Hoc Temperature + OOD Calibration (STRICTLY permissive val!)
    print("[Eval] Running final frozen permissive validation + OOD + post-hoc calibration...", flush=True)
    final_val_metrics, cal_logits, cal_labels, cal_bdry, final_ood_metrics = (
        evaluate_permissive_val_and_ood(
            model, val_loader, teachers, device, use_bf16=use_bf16, max_cal_pixels=250_000
        )
    )
    cal = calibrate_temperature_and_tau(
        cal_logits,
        cal_labels,
        cal_bdry,
        T_ood=1.0,
        kappa_b=1.5,
        id_tpr_target=0.95,
    )
    cal_path = cal.save_json(out_dir / "ood_calibration.json")
    cal_sha = sha256_file(cal_path)

    # Empirical clean-ID acceptance at calibrated tau_ood
    from ml.slz_perception.train.calibrate_ood import compute_free_energy

    e_cal_tilde = compute_free_energy(cal_logits, T=cal.T_ood) - cal.kappa_b * cal_bdry.clamp(0.0, 1.0)
    emp_id_tpr = float((e_cal_tilde <= cal.tau_ood).float().mean().item())
    final_ood_metrics["empirical_clean_id_tpr_at_tau_ood"] = round(emp_id_tpr, 6)

    # Phase 4: Quarantined Zero-Shot AeroScapes Eval (CC BY-SA 4.0, strictly post-freeze!)
    if not args.skip_aeroscapes:
        aeroscapes_metrics: dict[str, Any] | None = evaluate_quarantined_aeroscapes(
            model, work_dir, device, max_samples=int(args.max_aeroscapes), use_bf16=use_bf16
        )
    else:
        aeroscapes_metrics = None

    # Phase 5: Save Checkpoint & Export Static 1x3x512x512 Opset-17 ONNX (dropping dino_proj)
    ckpt_path = out_dir / "pidnet_slz_trained_l4.pt"
    model_cpu = model.cpu().eval()
    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "ood_calibration": cal.to_dict(),
            "stage1_best_miou": best_s1_miou,
            "final_val_miou": final_val_metrics["miou_present_classes"],
        },
        ckpt_path,
    )

    deploy = PIDNetSlzDeployWrapper(model_cpu).eval()
    onnx_path = export_static_onnx(
        deploy,
        out_dir / "pidnet_slz_trained_opset17.onnx",
        opset=17,
        simplify=False,
        input_hw=(512, 512),
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_op_types = sorted({n.op_type for n in onnx_model.graph.node})
    onnx_sha = sha256_file(onnx_path)
    onnx_size_mb = round(onnx_path.stat().st_size / (1024.0 * 1024.0), 3)

    peak_alloc_mib = (
        round(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0), 1)
        if device.type == "cuda"
        else 0.0
    )
    peak_reserved_mib = (
        round(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0), 1)
        if device.type == "cuda"
        else 0.0
    )
    total_wall_sec = round(time.perf_counter() - t_pipeline_start, 2)

    report: dict[str, Any] = {
        "schema_version": "vertirta.slz_l4_distill_metrics/v3.1.0",
        "weights_status": "TRAINED_COLAB_L4_DISTILLED_WEIGHTS",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "device": str(device),
            "gpu": gpu_info,
            "amp_bf16": use_bf16,
            "peak_cuda_allocated_mib": peak_alloc_mib,
            "peak_cuda_reserved_mib": peak_reserved_mib,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "timing_sec": {
            "data_prep_sec": data_prep_sec,
            "train_wall_sec": train_wall_sec,
            "total_wall_sec": total_wall_sec,
        },
        "license_audit": {
            "zero_sharealike_verified": manifest_report.zero_sharealike_verified,
            "approved_train_entries": list(manifest_report.approved_train_entries),
            "quarantined_eval_entries": list(manifest_report.quarantined_eval_entries),
            "oem_attribution_audit": attr_meta,
            "oem_permissive_region_count": len(oem_regions),
            "gazebo_istanbul_8k_status": (
                "EXCLUDED_FROM_L4_RUN_SEPARATE_GAZEBO_EGL_CAPTURE_STAGE "
                "(0 synthetic Gazebo frames ingested; all 4 active training sources "
                "verified inside locked zero-ShareAlike allowlist)"
            ),
        },
        "teachers": teachers.metadata,
        "datasets": {
            "sources": dataset_provenance,
            "total_train_samples": len(train_recs),
            "total_val_samples": len(val_recs),
            "train_samples_by_source": dict(src_counts),
            "val_samples_by_source": dict(Counter(r.source for r in val_recs)),
            "epoch_sampling_draws": epoch_samples,
            "target_sampling_weights": target_source_ratios,
        },
        "training": {
            "seed": seed,
            "batch_size_micro": prof.batch_size,
            "grad_accum_steps": prof.grad_accum,
            "effective_batch_size": prof.batch_size * prof.grad_accum,
            "train_parameters": int(train_params),
            "deploy_parameters": int(deploy_params),
            "stage1": {
                "epochs": s1_cfg.epochs,
                "lr_start": s1_cfg.lr_start,
                "lr_min": s1_cfg.lr_min,
                "lambda_bdry": s1_cfg.lambda_bdry,
                "lambda_dino": s1_cfg.lambda_dino if teachers.enabled else 0.0,
                "lambda_geom": s1_cfg.lambda_geom,
                "best_val_miou": round(best_s1_miou, 6),
                "history": stage1_history,
            },
            "stage2_ood_oe": {
                "epochs": s2_cfg.epochs,
                "lr_start": s2_cfg.lr_start,
                "lr_min": s2_cfg.lr_min,
                "lambda_ood": s2_cfg.lambda_ood,
                "m_in": s2_cfg.m_in,
                "m_out": s2_cfg.m_out,
                "anomaly_mix_prob": s2_cfg.anomaly_mix_prob,
                "history": stage2_history,
            },
        },
        "evaluation": {
            "permissive_validation": final_val_metrics,
            "ood_detection": final_ood_metrics,
            "quarantined_zero_shot_aeroscapes": aeroscapes_metrics,
        },
        "ood_calibration": {
            **cal.to_dict(),
            "file": "ood_calibration.json",
            "sha256": cal_sha,
        },
        "export": {
            "onnx_filename": onnx_path.name,
            "onnx_size_mb": onnx_size_mb,
            "onnx_sha256": onnx_sha,
            "opset_version": 17,
            "input_shape": [1, 3, 512, 512],
            "output_shapes": {
                "seg_logits": [1, 5, 512, 512],
                "surf_normals": [1, 3, 512, 512],
                "bdry_prob": [1, 1, 64, 64],
            },
            "deploy_parameters": int(deploy_params),
            "zero_adaptive_avg_pool2d": True,
            "zero_dino_proj_nodes": True,
            "zero_dynamic_shape_ops": ("Shape" not in onnx_op_types and "Gather" not in onnx_op_types),
            "onnx_checker_passed": True,
            "wsl2_staging_path": f"/tmp/vertirta_v31_blobs/{onnx_path.name}",
        },
    }

    metrics_path = out_dir / "l4_pidnet_distill_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"[DONE] Metrics written to {metrics_path} | ONNX={onnx_path.name} ({onnx_size_mb} MB, "
        f"sha256={onnx_sha[:16]}...) | val_mIoU={final_val_metrics['miou_present_classes']:.4f} | "
        f"val_FSR={final_val_metrics['false_safe_rate_fsr']:.4f} | "
        f"OOD_AUROC={final_ood_metrics['boundary_compensated_auroc']:.4f} | "
        f"peak_VRAM={peak_alloc_mib} MiB | wall={total_wall_sec}s",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Local physical GTX 1650 SUPER (TU116) proof mode
# ---------------------------------------------------------------------------


def _run_trained_onnx_sanity_worker(
    onnx_path: Path,
    precision: str,
    warmup: int,
    iters: int,
    *,
    cuda_graph: bool = False,
) -> dict[str, Any]:
    """Benchmark trained ONNX in isolated subprocess + validate non-degenerate trained predictions."""
    from scripts.run_local_gtx1650s_onnx_bench import (
        _benchmark_session,
        _register_nvidia_dll_dirs,
    )

    res = _benchmark_session(
        onnx_path,
        precision,
        warmup=warmup,
        iters=iters,
        cuda_graph=cuda_graph,
    )

    # Extra semantic & geometric sanity check on structured synthetic aerial test patch
    _register_nvidia_dll_dirs()
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=[("CUDAExecutionProvider", {"device_id": "0"})],
    )
    # Build deterministic multi-region synthetic aerial frame (dark asphalt + green veg + bright roof)
    img = np.zeros((1, 3, 512, 512), dtype=np.float32)
    img[:, :, :256, :256] = np.array([-0.8, -0.8, -0.7], dtype=np.float32).reshape(1, 3, 1, 1)
    img[:, :, :256, 256:] = np.array([-0.9, 0.7, -0.8], dtype=np.float32).reshape(1, 3, 1, 1)
    img[:, :, 256:, :] = np.array([1.2, 1.1, 1.0], dtype=np.float32).reshape(1, 3, 1, 1)
    logits, normals, bdry = sess.run(None, {"rgb_nchw": img})
    preds = np.argmax(logits[0], axis=0)
    uniq_classes, counts = np.unique(preds, return_counts=True)
    norms = np.linalg.norm(normals[0], axis=0)
    energy = -np.log(np.sum(np.exp(logits[0] - np.max(logits[0], axis=0, keepdims=True)), axis=0)) - np.max(
        logits[0], axis=0
    )
    res["trained_output_sanity"] = {
        "predicted_class_histogram": {int(k): int(v) for k, v in zip(uniq_classes, counts)},
        "distinct_predicted_classes": int(len(uniq_classes)),
        "normal_l2_norm_mean": round(float(norms.mean()), 6),
        "normal_l2_norm_max_err": round(float(np.abs(norms - 1.0).max()), 6),
        "bdry_prob_min": round(float(bdry.min()), 6),
        "bdry_prob_max": round(float(bdry.max()), 6),
        "free_energy_mean": round(float(energy.mean()), 4),
        "all_outputs_finite": bool(
            np.all(np.isfinite(logits)) and np.all(np.isfinite(normals)) and np.all(np.isfinite(bdry))
        ),
    }
    del sess
    gc.collect()
    return res


def run_local_gtx1650s_proof(args: argparse.Namespace) -> int:
    """Benchmark trained ONNX live on physical GTX 1650 SUPER and emit trained proof JSON."""
    from scripts.run_local_gtx1650s_onnx_bench import (
        _build_fp16_onnx,
        _collect_nvidia_dll_dirs,
        _query_gpu_info,
    )

    if args.worker_precision is not None:
        if args.worker_onnx is None:
            raise SystemExit("--worker-onnx required with --worker-precision")
        res = _run_trained_onnx_sanity_worker(
            args.worker_onnx,
            args.worker_precision,
            warmup=int(args.warmup),
            iters=int(args.iters),
            cuda_graph=bool(args.worker_cuda_graph),
        )
        print(json.dumps(res))
        return 0

    onnx_path = Path(args.benchmark_local_onnx).resolve()
    metrics_path = Path(args.metrics_json).resolve()
    out_path = Path(args.proof_out).resolve()

    if not onnx_path.is_file():
        raise FileNotFoundError(f"Trained ONNX file not found: {onnx_path}")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"L4 metrics JSON not found: {metrics_path}")

    l4_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_sha = str(l4_metrics["export"]["onnx_sha256"])
    actual_sha = sha256_file(onnx_path)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"ONNX SHA-256 mismatch! file={actual_sha} vs l4_metrics={expected_sha}"
        )

    import onnx
    import onnxruntime as ort

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    gpu = _query_gpu_info()

    def _spawn_worker(mode: str, model_file: Path, cuda_graph: bool) -> dict[str, Any]:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--benchmark-local-onnx",
            str(onnx_path),
            "--metrics-json",
            str(metrics_path),
            "--worker-precision",
            mode,
            "--worker-onnx",
            str(model_file),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
        if cuda_graph:
            cmd.append("--worker-cuda-graph")
        env = os.environ.copy()
        nv_dirs = _collect_nvidia_dll_dirs()
        if nv_dirs:
            env["PATH"] = os.pathsep.join(nv_dirs) + os.pathsep + env.get("PATH", "")
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker {mode} (cg={cuda_graph}) failed (rc={proc.returncode}):\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )
        for line in reversed(proc.stdout.splitlines()):
            s = line.strip()
            if s.startswith("{") and s.endswith("}"):
                return json.loads(s)
        raise RuntimeError(f"Worker {mode} produced no JSON output: {proc.stdout}")

    with tempfile.TemporaryDirectory(prefix="vertirta_tu116_trained_") as td:
        fp16_path = Path(td) / "pidnet_slz_trained_fp16.onnx"
        _build_fp16_onnx(onnx_path, fp16_path)

        fp32_res = _spawn_worker("fp32", onnx_path, cuda_graph=False)
        fp32_cg_res = _spawn_worker("fp32", onnx_path, cuda_graph=True)
        fp16_res = _spawn_worker("fp16", fp16_path, cuda_graph=False)

    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        pass

    val_eval = l4_metrics["evaluation"]["permissive_validation"]
    ood_eval = l4_metrics["evaluation"]["ood_detection"]
    aero_eval = l4_metrics["evaluation"].get("quarantined_zero_shot_aeroscapes")

    report: dict[str, Any] = {
        "schema_version": "vertirta.local_gpu_trained_proof/v3.1.0",
        "weights_status": "TRAINED_COLAB_L4_DISTILLED_WEIGHTS",
        "retires_baseline": "artifacts/v2.0.0/local_gtx1650s_pidnet_arch_bench.json",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "host_os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "runtime": {
            "python_version": platform.python_version(),
            "onnxruntime_version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
        },
        "gpu": gpu,
        "model": {
            "name": "PIDNetSlzDeployWrapper (Colab L4 distilled weights, static-PAPPM, zero dino_proj)",
            "onnx_filename": onnx_path.name,
            "onnx_sha256": actual_sha,
            "onnx_sha256_verified_against_l4_metrics": True,
            "l4_metrics_artifact": "artifacts/v3.1.0/l4_pidnet_distill_metrics.json",
            "onnx_size_mb": round(onnx_path.stat().st_size / (1024.0 * 1024.0), 3),
            "opset_version": 17,
            "input_shape": [1, 3, 512, 512],
            "output_shapes": {
                "seg_logits": [1, 5, 512, 512],
                "surf_normals": [1, 3, 512, 512],
                "bdry_prob": [1, 1, 64, 64],
            },
            "deploy_parameters": int(l4_metrics["export"]["deploy_parameters"]),
            "wsl2_staging_path": f"/tmp/vertirta_v31_blobs/{onnx_path.name}",
        },
        "trained_accuracy_summary": {
            "training_gpu": l4_metrics["hardware"]["gpu"]["name"],
            "zero_sharealike_verified": l4_metrics["license_audit"]["zero_sharealike_verified"],
            "permissive_val_miou": val_eval["miou_present_classes"],
            "permissive_val_fsr": val_eval["false_safe_rate_fsr"],
            "permissive_val_safe_precision": val_eval["safe_precision"],
            "permissive_val_per_class_iou": val_eval["per_class_iou"],
            "ood_raw_energy_auroc": ood_eval.get("raw_energy_auroc"),
            "ood_boundary_compensated_auroc": ood_eval.get("boundary_compensated_auroc"),
            "ood_boundary_compensated_fpr95": ood_eval.get("boundary_compensated_fpr95"),
            "ood_calibration": {
                "T_ood": l4_metrics["ood_calibration"]["T_ood"],
                "T_prob": l4_metrics["ood_calibration"]["T_prob"],
                "kappa_b": l4_metrics["ood_calibration"]["kappa_b"],
                "tau_ood": l4_metrics["ood_calibration"]["tau_ood"],
            },
            "quarantined_zero_shot_aeroscapes": (
                {
                    "miou_present_classes": aero_eval["miou_present_classes"],
                    "false_safe_rate_fsr": aero_eval["false_safe_rate_fsr"],
                    "evaluated_images": aero_eval["evaluated_images"],
                    "license": aero_eval["license"],
                }
                if aero_eval is not None
                else None
            ),
        },
        "benchmarks": {
            "ort_cuda_fp32": fp32_res,
            "ort_cuda_fp32_cuda_graph": fp32_cg_res,
            "ort_cuda_fp16": fp16_res,
        },
        "fp16_vs_fp32_speedup_p50": round(
            fp32_res["network_iobinding"]["p50_ms"]
            / max(fp16_res["network_iobinding"]["p50_ms"], 1e-6),
            3,
        ),
        "recommended_local_tu116_mode": (
            "ort_cuda_fp32_cuda_graph"
            if fp32_cg_res["network_iobinding"]["p99_ms"] <= fp32_res["network_iobinding"]["p99_ms"]
            else "ort_cuda_fp32"
        ),
        "all_local_gates_passed": bool(
            fp32_cg_res["gates"]["network_p50_lt_12ms"]
            and fp32_cg_res["gates"]["network_p99_lt_12ms"]
            and fp32_cg_res["gates"]["e2e_ge_20hz"]
            and fp32_cg_res["gates"]["vram_lt_300mb"]
            and fp32_cg_res["outputs_finite"]
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VertiRTA v3.1.0 Colab L4 PIDNet-S Distillation & Local GTX 1650 SUPER Proof."
    )
    # Mode B: Local physical GTX 1650 SUPER proof
    parser.add_argument("--benchmark-local-onnx", type=Path, default=None)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v3.1.0" / "l4_pidnet_distill_metrics.json",
    )
    parser.add_argument(
        "--proof-out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v3.1.0" / "local_gtx1650s_pidnet_trained_proof.json",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--worker-precision", choices=("fp32", "fp16"), default=None)
    parser.add_argument("--worker-onnx", type=Path, default=None)
    parser.add_argument("--worker-cuda-graph", action="store_true")

    # Mode A: Colab L4 training pipeline
    parser.add_argument("--oem-attribution-html", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/vertirta_slz_l4_work"))
    parser.add_argument("--out-dir", type=Path, default=Path("/content/vertirta_slz_l4_out"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=24)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage1-lr", type=float, default=8e-4)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--anomaly-mix-prob", type=float, default=0.15)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--epoch-samples", type=int, default=3200)
    parser.add_argument("--max-flair", type=int, default=600)
    parser.add_argument("--max-oem", type=int, default=1600)
    parser.add_argument("--max-floodnet", type=int, default=1500)
    parser.add_argument("--max-skyscenes", type=int, default=1600)
    parser.add_argument("--max-aeroscapes", type=int, default=350)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--no-teachers", action="store_true")
    parser.add_argument("--skip-aeroscapes", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.benchmark_local_onnx is not None:
        return run_local_gtx1650s_proof(args)
    return run_l4_training_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
