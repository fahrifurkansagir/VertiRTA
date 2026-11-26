#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local Physical GTX 1650 SUPER (4 GB, TU116) Forward WireSeg & D-FINE-N v1.1 Trainer (0.00 CU).

Root-cause fix for legacy G4 forward-hazard false positives on real oblique/downward urban scenes:
  1. M-LSD-WireSeg v1.1 (1,257,126 deploy params, static 1x3x640x640 Opset-17 ONNX):
     - Trained on REAL MIT-licensed Mendeley TEIAS Powerline GT visible-light frames
       (TV_VL wire-present + TY_VL no-wire negatives) mixed with 40%+ REAL permissive urban
       rooftop/facade hard negatives (SkyScenes MIT, OpenEarthMap CC-BY-4.0 whitelist,
       FloodNet CDLA-Permissive-1.0, FLAIR-1 Etalab-2.0) plus Apache-2.0 real-wire-on-urban
       composites and procedural horizon catenaries.
     - Replaces naive unweighted clDice on empty frames with positive-only Dice/clDice +
       top-k OHEM hard-negative pixel mining on rooftop/facade negatives.
  2. D-FINE-N v1.1 (1,204,725 deploy params, exact static 1x3x640x640 -> [1,300,4], [1,300,3] ABI):
     - Eliminates the legacy 3-class softmax-over-queries-0..2 background collapse bug by
       training ALL 300x3 spatial queries with per-class Sigmoid Focal BCE (`p0=0.01` prior
       bias) + 50% empty real urban rooftop and wire-only hard negatives (`target_cls_bin = 0`).
  3. Zero-Shot 240-Frame NASA Goddard Verification (post-freeze ONLY — never trained/calibrated):
     - Benchmarks BOTH legacy G4 ONNX and recovered v1.1 ONNX unfiltered across 240 real
       NASA Goddard SVS-14443 drone frames and seals canonical SHA-256 JSON metrics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs.common import (  # noqa: E402
    query_nvidia_smi_snapshot,
    seed_everything,
    sha256_file,
    validate_job_license_manifest,
    write_canonical_json,
)
from scripts.colab.g4_jobs.job_forward_wire import (  # noqa: E402
    WIRE_CLASS_NAMES,
    _build_torch_models,
    _export_and_check_onnx,
    build_forward_license_manifest,
    compute_dfine_sigmoid_focal_loss,
    decode_dfine_sigmoid_boxes,
    encode_dfine_grid_targets,
)

DEFAULT_TEIAS_DIR = Path("/var/tmp/vertirta_local_wire_work/teias_extracted")
DEFAULT_TEIAS_ZIP = Path("/var/tmp/vertirta_local_wire_work/teias_powerline_gt.zip")
DEFAULT_PREPARED_MANIFEST = Path("/var/tmp/vertirta_local_pidnet_work/prepared_manifest.json")
DEFAULT_BLOB_DIR = Path("/var/tmp/vertirta_v400_blobs")
DEFAULT_NASA_VIDEO = Path("/var/tmp/vertirta_realworld_cache/real_drone_video/nasa_goddard_svs14443_drone.webm")
DEFAULT_OUT_JSON = REPO_ROOT / "artifacts" / "v4.0.0" / "local_gtx1650s_forward_wire_v2_metrics.json"
DEFAULT_LEGACY_G4_METRICS = REPO_ROOT / "artifacts" / "v4.0.0" / "g4_forward_wire_metrics.json"

WIRESEG_V11_FILENAME = "mlsd_wireseg_640_v11_local_gtx_opset17.onnx"
DFINE_V11_FILENAME = "dfine_n_640_v11_local_gtx_opset17.onnx"
LEGACY_WIRESEG_FILENAME = "mlsd_wireseg_640_opset17.onnx"
LEGACY_DFINE_FILENAME = "dfine_n_640_opset17.onnx"

ALLOWED_PREPARED_SOURCES: frozenset[str] = frozenset(
    {"FLAIR-1", "OpenEarthMap-41", "FloodNet-Supervised-v1.0", "SkyScenes"}
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
    s = p.as_posix()
    if s.startswith("/var/tmp/"):
        return s
    return p.name


def _preload_linux_nvidia_libs() -> None:
    """Preload pip-installed NVIDIA shared libs globally on Linux/WSL2 for ORT CUDA EP."""
    if platform.system() != "Linux":
        return
    import ctypes

    roots: list[Path] = []
    try:
        import nvidia  # type: ignore[import-not-found]

        for p in getattr(nvidia, "__path__", []):
            roots.append(Path(p))
    except ImportError:
        pass
    for sp in sys.path:
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            roots.append(nv)

    loaded: set[str] = set()
    priority_prefixes = (
        "libcudart",
        "libnvrtc",
        "libnvJitLink",
        "libcublasLt",
        "libcublas",
        "libcufft",
        "libcurand",
        "libcusparse",
        "libcusolver",
        "libcudnn",
    )
    candidates: list[Path] = []
    for nv_root in roots:
        if not nv_root.is_dir():
            continue
        for sub in sorted(nv_root.iterdir()):
            lib_dir = sub / "lib"
            if not lib_dir.is_dir():
                continue
            for so in sorted(lib_dir.glob("*.so*")):
                if so.is_file():
                    candidates.append(so)

    def _sort_key(path: Path) -> tuple[int, str]:
        name = path.name
        for i, pref in enumerate(priority_prefixes):
            if name.startswith(pref):
                return (i, name)
        return (len(priority_prefixes), name)

    for so in sorted(candidates, key=_sort_key):
        key = str(so)
        if key in loaded:
            continue
        try:
            ctypes.CDLL(key, mode=ctypes.RTLD_GLOBAL)
            loaded.add(key)
        except OSError:
            pass
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass


def _load_rgb_640(path: Path) -> np.ndarray:
    """Load an image file as uint8 RGB (640, 640, 3)."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def _load_teias_mask_640(path: Path) -> np.ndarray:
    """Load a TEIAS TV_VL_GT mask as float32 (640, 640) binary wire mask."""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    raw_bin = (arr.max(axis=2) > 110).astype(np.uint8) * 255
    pil_m = Image.fromarray(raw_bin, mode="L").resize((640, 640), Image.Resampling.BILINEAR)
    # Threshold at 0.30 after bilinear resize so thin 2-3 px wires stay continuous
    return (np.asarray(pil_m, dtype=np.float32) >= 76.0).astype(np.float32)


def _derive_junction_map_from_wire_mask(wire_mask_640: np.ndarray) -> np.ndarray:
    """Derive a clean [5, 160, 160] line midpoint & endpoint offset map from a binary wire mask."""
    junc = np.zeros((5, 160, 160), dtype=np.float32)
    ys, xs = np.nonzero(wire_mask_640 > 0.5)
    if len(xs) < 16:
        return junc
    # Partition wire pixels into up to 4 horizontal bands to produce clean local segment anchors
    for band_idx in range(4):
        x_lo = band_idx * 160
        x_hi = (band_idx + 1) * 160
        sel = (xs >= x_lo) & (xs < x_hi)
        if int(sel.sum()) < 8:
            continue
        bx = xs[sel]
        by = ys[sel]
        i0 = int(np.argmin(bx))
        i1 = int(np.argmax(bx))
        x0, y0 = float(bx[i0]), float(by[i0])
        x1, y1 = float(bx[i1]), float(by[i1])
        mx_px = 0.5 * (x0 + x1)
        my_px = 0.5 * (y0 + y1)
        mx = int(np.clip(mx_px / 4.0, 0, 159))
        my = int(np.clip(my_px / 4.0, 0, 159))
        junc[0, my, mx] = 1.0
        junc[1, my, mx] = (x0 - mx_px) / 640.0
        junc[2, my, mx] = (y0 - my_px) / 640.0
        junc[3, my, mx] = (x1 - mx_px) / 640.0
        junc[4, my, mx] = (y1 - my_px) / 640.0
    return junc


def discover_teias_vl_splits(
    teias_dir: Path,
    *,
    seed: int = 20261126,
    val_frac: float = 0.20,
) -> dict[str, list[tuple[Path, Path | None]]]:
    """Load real MIT-licensed TEIAS Visible-Light positive (TV_VL) and negative (TY_VL) splits."""
    tv_org_dir = teias_dir / "TV_VL_ORG"
    tv_gt_dir = teias_dir / "TV_VL_GT"
    ty_org_dir = teias_dir / "TY_VL_ORG"

    pos_pairs: list[tuple[Path, Path | None]] = []
    if tv_org_dir.is_dir() and tv_gt_dir.is_dir():
        for org_p in sorted(tv_org_dir.glob("*.bmp")):
            gt_name = org_p.name.replace("_ORG_", "_GT_")
            gt_p = tv_gt_dir / gt_name
            if gt_p.is_file():
                pos_pairs.append((org_p, gt_p))

    neg_items: list[tuple[Path, Path | None]] = []
    if ty_org_dir.is_dir():
        for org_p in sorted(ty_org_dir.glob("*.bmp")):
            # TY_VL ("Tel Yok" = No Wire) has all-zero wire GT (4-px BMP marker ignored)
            neg_items.append((org_p, None))

    rng = random.Random(seed)
    pos_idx = list(range(len(pos_pairs)))
    neg_idx = list(range(len(neg_items)))
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_val = max(1, int(round(len(pos_pairs) * val_frac))) if pos_pairs else 0
    n_neg_val = max(1, int(round(len(neg_items) * val_frac))) if neg_items else 0

    return {
        "tv_train": [pos_pairs[i] for i in pos_idx[n_pos_val:]],
        "tv_val": [pos_pairs[i] for i in pos_idx[:n_pos_val]],
        "ty_train": [neg_items[i] for i in neg_idx[n_neg_val:]],
        "ty_val": [neg_items[i] for i in neg_idx[:n_neg_val]],
    }


def discover_permissive_urban_hardneg_splits(
    prepared_manifest_path: Path,
) -> tuple[list[Path], list[Path], dict[str, int]]:
    """Load strictly permissive urban rooftop/oblique hard-negative paths honoring train/val splits."""
    if not prepared_manifest_path.is_file():
        return [], [], {}
    manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    counts: dict[str, int] = {}
    for rec in manifest.get("records", []):
        src = str(rec.get("source", ""))
        if src not in ALLOWED_PREPARED_SOURCES:
            continue
        rgb_p = Path(str(rec.get("rgb_path", "")))
        if not rgb_p.is_file():
            continue
        split = str(rec.get("split", "train"))
        key = f"{src}:{split}"
        counts[key] = counts.get(key, 0) + 1
        if split == "val":
            val_paths.append(rgb_p)
        else:
            train_paths.append(rgb_p)
    return train_paths, val_paths, counts


def _augment_rgb_and_mask(
    rgb_u8: np.ndarray,
    mask_f32: np.ndarray | None,
    rng: np.random.Generator,
    *,
    strong_sunlit_roof_aug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast spatial + photometric augmentation for 640x640 RGB + optional binary wire mask."""
    try:
        import cv2
    except ImportError:
        cv2 = None  # type: ignore[assignment]
    from PIL import Image

    h, w = rgb_u8.shape[:2]
    if mask_f32 is None:
        m = np.zeros((h, w), dtype=np.float32)
    else:
        m = mask_f32.copy()
    img = rgb_u8.copy()

    # Random horizontal flip
    if rng.random() < 0.5:
        img = img[:, ::-1, :].copy()
        m = m[:, ::-1].copy()
    # Random vertical flip (wires/rooftops appear at all roll/oblique orientations)
    if rng.random() < 0.35:
        img = img[::-1, :, :].copy()
        m = m[::-1, :].copy()
    # Random 90-deg rotation occasionally for omnidirectional edge invariance
    if rng.random() < 0.25:
        k = int(rng.integers(1, 4))
        img = np.rot90(img, k).copy()
        m = np.rot90(m, k).copy()

    # Random resized crop (0.62..0.96)
    if rng.random() < 0.75:
        scale = float(rng.uniform(0.62, 0.96))
        ch = max(64, int(round(h * scale)))
        cw = max(64, int(round(w * scale)))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        crop_img = img[y0 : y0 + ch, x0 : x0 + cw]
        if cv2 is not None:
            img = cv2.resize(crop_img, (640, 640), interpolation=cv2.INTER_LINEAR)
        else:
            img = np.asarray(Image.fromarray(crop_img).resize((640, 640), Image.Resampling.BILINEAR), dtype=np.uint8)
        if mask_f32 is not None and m.max() > 0.0:
            crop_m = m[y0 : y0 + ch, x0 : x0 + cw]
            if cv2 is not None:
                m_rs = cv2.resize(crop_m, (640, 640), interpolation=cv2.INTER_LINEAR)
                m = (m_rs >= 0.30).astype(np.float32)
            else:
                cm_pil = Image.fromarray((crop_m * 255.0).astype(np.uint8)).resize((640, 640), Image.Resampling.BILINEAR)
                m = (np.asarray(cm_pil, dtype=np.float32) >= 76.0).astype(np.float32)
        else:
            m = np.zeros((640, 640), dtype=np.float32)

    # Small rotation (±18 deg) for diagonal wires & oblique rooftops
    if rng.random() < 0.45:
        angle = float(rng.uniform(-18.0, 18.0))
        if cv2 is not None:
            rot_mat = cv2.getRotationMatrix2D((320.0, 320.0), angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (640, 640), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            if mask_f32 is not None and m.max() > 0.0:
                m_rot = cv2.warpAffine(m, rot_mat, (640, 640), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                m = (m_rot >= 0.30).astype(np.float32)
        else:
            img = np.asarray(Image.fromarray(img).rotate(angle, resample=Image.Resampling.BILINEAR), dtype=np.uint8)
            if mask_f32 is not None and m.max() > 0.0:
                rot_m = Image.fromarray((m * 255.0).astype(np.uint8)).rotate(angle, resample=Image.Resampling.BILINEAR)
                m = (np.asarray(rot_m, dtype=np.float32) >= 76.0).astype(np.float32)

    img_f = img.astype(np.float32) / 255.0
    # Photometric jitter (contrast, brightness, per-channel gain, optional bright concrete roof boost)
    contrast = float(rng.uniform(0.72, 1.38))
    brightness = float(rng.uniform(-0.14, 0.22 if strong_sunlit_roof_aug else 0.14))
    mean_val = float(img_f.mean())
    img_f = (img_f - mean_val) * contrast + mean_val + brightness
    chan_gain = rng.uniform(0.90, 1.10, size=(1, 1, 3)).astype(np.float32)
    img_f = img_f * chan_gain
    if rng.random() < 0.35:
        noise_std = float(rng.uniform(0.005, 0.025))
        img_f = img_f + rng.normal(0.0, noise_std, size=img_f.shape).astype(np.float32)
    img_f = np.clip(img_f, 0.0, 1.0)
    return img_f.transpose(2, 0, 1).astype(np.float32), m[None, :, :].astype(np.float32)


def _synthesize_procedural_sky_horizon(rng: np.random.Generator) -> np.ndarray:
    """Create a realistic 640x640 RGB float32 sky/horizon/urban backdrop (HWC, 0..1)."""
    yy = np.linspace(0.0, 1.0, 640, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, 640, dtype=np.float32)[None, :]
    style = int(rng.integers(0, 4))
    if style == 0:
        # Clear blue sky -> hazy horizon -> urban ground
        top = np.array([0.28, 0.52, 0.86], dtype=np.float32)
        mid = np.array([0.74, 0.82, 0.90], dtype=np.float32)
    elif style == 1:
        # Bright overcast white/grey sky
        top = np.array([0.76, 0.79, 0.82], dtype=np.float32)
        mid = np.array([0.88, 0.90, 0.92], dtype=np.float32)
    elif style == 2:
        # Golden hour / warm haze
        top = np.array([0.42, 0.54, 0.72], dtype=np.float32)
        mid = np.array([0.88, 0.76, 0.62], dtype=np.float32)
    else:
        # High-contrast sunlit pale blue
        top = np.array([0.45, 0.66, 0.92], dtype=np.float32)
        mid = np.array([0.84, 0.88, 0.93], dtype=np.float32)

    horizon_y = int(rng.integers(340, 560))
    sky_t = np.clip(yy / max(float(horizon_y) / 640.0, 0.1), 0.0, 1.0)
    img = top[None, None, :] * (1.0 - sky_t[:, :, None]) + mid[None, None, :] * sky_t[:, :, None]

    # Soft cloud waves
    cloud = (
        0.05
        * np.sin(2.0 * math.pi * (xx * rng.uniform(1.0, 3.0) + rng.uniform(0.0, 1.0)))
        * np.cos(2.0 * math.pi * (yy * rng.uniform(1.5, 4.0) + rng.uniform(0.0, 1.0)))
    ).astype(np.float32)
    img = np.clip(img + cloud[:, :, None], 0.0, 1.0)

    # Lower ground/roofline silhouettes (hard negatives for straight roof edges!)
    g_col = rng.uniform(0.18, 0.68, size=(3,)).astype(np.float32)
    img[horizon_y:, :, :] = g_col[None, None, :] * rng.uniform(0.85, 1.15)
    n_bldg = int(rng.integers(3, 9))
    for _ in range(n_bldg):
        bx0 = int(rng.integers(0, 560))
        bw = int(rng.integers(35, 140))
        bh = int(rng.integers(15, 90))
        bx1 = min(640, bx0 + bw)
        by0 = max(60, horizon_y - bh)
        b_col = rng.uniform(0.25, 0.92, size=(3,)).astype(np.float32)
        img[by0:horizon_y, bx0:bx1, :] = b_col[None, None, :]
        # Bright parapet rim (explicit rooftop edge hard-negative!)
        if by0 + 3 < horizon_y:
            img[by0 : by0 + 3, bx0:bx1, :] = np.clip(b_col * 1.25, 0.0, 1.0)[None, None, :]

    noise = rng.normal(0.0, float(rng.uniform(0.008, 0.024)), size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def _draw_procedural_catenary_wires(
    img_hwc: np.ndarray,
    rng: np.random.Generator,
    *,
    n_wires: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay realistic thin dark or metallic-glint catenary powerlines onto a 640x640 HWC float image."""
    out = img_hwc.copy()
    mask = np.zeros((640, 640), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    xi = np.arange(640, dtype=np.int32)
    count = int(n_wires) if n_wires is not None else int(rng.integers(1, 5))
    y_anchor = float(rng.uniform(70.0, 420.0))
    base_slope = float(rng.uniform(-160.0, 160.0))
    base_sag = float(rng.uniform(10.0, 65.0))
    wire_is_bright = bool(rng.random() < 0.35)
    for w_i in range(count):
        y_off = float(w_i * rng.uniform(14.0, 55.0))
        sag = base_sag * float(rng.uniform(0.85, 1.15))
        slope = base_slope + float(rng.uniform(-18.0, 18.0))
        ys = y_anchor + y_off + slope * (xs - 0.5) + sag * (4.0 * (xs - 0.5) ** 2 - 1.0)
        valid = (ys >= 2.0) & (ys <= 637.0)
        if int(valid.sum()) < 80:
            continue
        xv = xi[valid]
        yv = np.rint(ys[valid]).astype(np.int32)
        thickness = int(rng.integers(1, 3))
        if wire_is_bright:
            wire_rgb = np.array([0.88, 0.89, 0.92], dtype=np.float32) * float(rng.uniform(0.88, 1.0))
            blend = float(rng.uniform(0.65, 0.90))
        else:
            wire_rgb = np.array([0.08, 0.09, 0.11], dtype=np.float32) * float(rng.uniform(0.6, 1.6))
            blend = float(rng.uniform(0.70, 0.92))
        for dy in range(-thickness // 2, (thickness + 1) // 2 + 1):
            yy = np.clip(yv + dy, 0, 639)
            w_alpha = blend if dy == 0 else blend * 0.55
            out[yy, xv, :] = out[yy, xv, :] * (1.0 - w_alpha) + wire_rgb[None, :] * w_alpha
            mask[yy, xv] = 1.0
    return np.clip(out, 0.0, 1.0), mask


def _sanitize_nvidia_smi_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """Strip personal or absolute interpreter paths from nvidia-smi compute_apps."""
    out = dict(snap)
    apps = []
    for app in snap.get("compute_apps", []) or []:
        clean_app = dict(app)
        if "process_name" in clean_app and clean_app["process_name"]:
            clean_app["process_name"] = Path(str(clean_app["process_name"])).name
        apps.append(clean_app)
    if "compute_apps" in out:
        out["compute_apps"] = apps
    return out


def _synthesize_hardneg_industrial_building_distractor(rng: np.random.Generator) -> np.ndarray:
    """Procedural sunlit concrete rooftop + HVAC vents/cross-shadows/cars/antennas distractor (0 objects)."""
    from PIL import Image, ImageDraw

    bg = _synthesize_procedural_sky_horizon(rng)
    pil = Image.fromarray((bg * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)

    # Bright sunlit white/grey concrete industrial roofs & parking asphalt ([155..242] RGB)
    base_tone = int(rng.integers(155, 225))
    draw.rectangle([0, 0, 640, 640], fill=(base_tone, base_tone, min(255, base_tone + int(rng.integers(-4, 8)))))

    n_blocks = int(rng.integers(3, 8))
    for _ in range(n_blocks):
        x0 = int(rng.integers(-40, 520))
        y0 = int(rng.integers(-40, 520))
        x1 = min(640, x0 + int(rng.integers(120, 380)))
        y1 = min(640, y0 + int(rng.integers(100, 320)))
        tone = int(rng.integers(165, 245))
        bcol = (tone, min(255, tone + int(rng.integers(-6, 6))), min(255, tone + int(rng.integers(-6, 8))))
        draw.rectangle([x0, y0, x1, y1], fill=bcol, outline=(max(0, tone - 75), max(0, tone - 75), max(0, tone - 75)), width=3)
        # Small dark HVAC fans, vents, X-ducting shadows, skylights & parked vehicles (20-55 px)
        n_units = int(rng.integers(6, 18))
        for _ in range(n_units):
            ux = int(rng.integers(max(4, x0 + 6), max(8, x1 - 24)))
            uy = int(rng.integers(max(4, y0 + 6), max(8, y1 - 24)))
            uw = int(rng.integers(14, 52))
            uh = int(rng.integers(12, 44))
            u_tone = int(rng.integers(25, 115))
            u_col = (u_tone, u_tone + int(rng.integers(-5, 6)), u_tone + int(rng.integers(-5, 8)))
            kind = int(rng.integers(0, 4))
            if kind == 0:
                draw.rectangle([ux, uy, min(638, ux + uw), min(638, uy + uh)], fill=u_col)
                # Drop shadow beside rooftop HVAC box
                draw.rectangle([min(638, ux + uw), uy + 4, min(638, ux + uw + 10), min(638, uy + uh + 8)], fill=(35, 38, 42))
            elif kind == 1:
                # Circular rooftop exhaust vent with cross-grille (classic Rogue_UAV distractor!)
                cx_u, cy_u = ux + uw // 2, uy + uh // 2
                r_u = max(6, min(uw, uh) // 2)
                draw.ellipse([cx_u - r_u, cy_u - r_u, cx_u + r_u, cy_u + r_u], fill=u_col, outline=(30, 32, 36), width=2)
                draw.line([(cx_u - r_u, cy_u - r_u), (cx_u + r_u, cy_u + r_u)], fill=(25, 28, 30), width=2)
                draw.line([(cx_u - r_u, cy_u + r_u), (cx_u + r_u, cy_u - r_u)], fill=(25, 28, 30), width=2)
            elif kind == 2:
                # Angular dark rooftop shadow / V-corner (classic Bird distractor!)
                draw.polygon([(ux, uy + uh), (ux + uw // 2, uy), (ux + uw, uy + uh // 2), (ux + uw // 2, uy + uh // 2)], fill=(32, 36, 42))
            else:
                # Parked car / horizontal mechanical unit with windshield & subtle lights (classic Rogue_UAV distractor!)
                cw_c = int(rng.integers(36, 86))
                ch_c = int(rng.integers(18, 42))
                cx1, cy1 = min(638, ux + cw_c), min(638, uy + ch_c)
                car_col = (
                    int(rng.integers(35, 150)),
                    int(rng.integers(35, 150)),
                    int(rng.integers(35, 155)),
                )
                draw.rectangle([ux, uy, cx1, cy1], fill=car_col, outline=(25, 28, 32), width=2)
                # Dark windshield band + subtle red/amber/white lamp accents
                if (cx1 - ux) > 16 and (cy1 - uy) > 10:
                    draw.rectangle([ux + 6, uy + 4, cx1 - 6, uy + (cy1 - uy) // 2], fill=(28, 34, 42))
                    if rng.random() < 0.5:
                        draw.rectangle([ux + 2, cy1 - 5, ux + 7, cy1 - 1], fill=(210, 45, 40))
                        draw.rectangle([cx1 - 7, cy1 - 5, cx1 - 2, cy1 - 1], fill=(225, 225, 210))
    # Dark leafy tree clusters around building perimeter
    for _ in range(int(rng.integers(3, 9))):
        tx = int(rng.integers(15, 625))
        ty = int(rng.integers(15, 625))
        tr = int(rng.integers(22, 72))
        g_col = (int(rng.integers(25, 68)), int(rng.integers(52, 118)), int(rng.integers(22, 62)))
        draw.ellipse([tx - tr, ty - tr, tx + tr, ty + tr], fill=g_col)

    out = np.asarray(pil, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, float(rng.uniform(0.008, 0.024)), size=out.shape).astype(np.float32)
    return np.clip(out + noise, 0.0, 1.0)


def _draw_forward_obstacle(
    img_hwc: np.ndarray,
    cls_id: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, int] | None:
    """Render a crisp Apache-2.0 Bird, Rogue_UAV, or Tower_Crane silhouette into img_hwc in-place."""
    from PIL import Image, ImageDraw

    pil = Image.fromarray((np.clip(img_hwc, 0.0, 1.0) * 255.0).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(pil)

    if cls_id == 0:  # Bird: crisp winged V / soaring seagull silhouette (well-resolved at stride-64)
        cx = float(rng.uniform(65.0, 575.0))
        cy = float(rng.uniform(55.0, 420.0))
        span = float(rng.uniform(44.0, 92.0))
        bh = float(rng.uniform(24.0, 52.0))
        dark = bool(rng.random() < 0.78)
        col = tuple(int(c) for c in (rng.integers(12, 55, size=3) if dark else rng.integers(215, 252, size=3)))
        wing_drop = float(rng.uniform(-0.32, 0.32)) * bh
        lw = max(4, int(round(span * 0.09)))
        pts_left = [(cx - 0.5 * span, cy + wing_drop), (cx - 0.22 * span, cy - 0.38 * bh), (cx, cy)]
        pts_right = [(cx, cy), (cx + 0.22 * span, cy - 0.38 * bh), (cx + 0.5 * span, cy + wing_drop)]
        draw.line(pts_left, fill=col, width=lw)
        draw.line(pts_right, fill=col, width=lw)
        body_r = max(4.5, span * 0.10)
        draw.ellipse([cx - body_r, cy - body_r * 0.7, cx + body_r, cy + body_r * 0.9], fill=col)
        x0 = max(0.0, cx - 0.54 * span)
        y0 = max(0.0, cy - 0.50 * bh)
        x1 = min(639.0, cx + 0.54 * span)
        y1 = min(639.0, cy + max(0.48 * bh, abs(wing_drop) + lw))

    elif cls_id == 1:  # Rogue_UAV: realistic quadcopter silhouette (body + landing skids + 4 arms + 4 rotor discs)
        cx = float(rng.uniform(75.0, 565.0))
        cy = float(rng.uniform(60.0, 390.0))
        bw = float(rng.uniform(54.0, 96.0))
        bh = float(rng.uniform(34.0, 62.0))
        tone = int(rng.integers(18, 68))
        col = (tone, tone + int(rng.integers(-4, 5)), tone + int(rng.integers(-4, 6)))
        arm_w = max(3, int(round(bw * 0.065)))
        # Diagonal / swept quadcopter arms
        draw.line([(cx - 0.46 * bw, cy - 0.24 * bh), (cx + 0.46 * bw, cy + 0.16 * bh)], fill=col, width=arm_w)
        draw.line([(cx - 0.46 * bw, cy + 0.16 * bh), (cx + 0.46 * bw, cy - 0.24 * bh)], fill=col, width=arm_w)
        # Central fuselage + T-shaped landing gear skids
        draw.ellipse([cx - 0.18 * bw, cy - 0.16 * bh, cx + 0.18 * bw, cy + 0.16 * bh], fill=col)
        draw.line([(cx - 0.14 * bw, cy + 0.12 * bh), (cx - 0.18 * bw, cy + 0.40 * bh)], fill=col, width=2)
        draw.line([(cx + 0.14 * bw, cy + 0.12 * bh), (cx + 0.18 * bw, cy + 0.40 * bh)], fill=col, width=2)
        draw.line([(cx - 0.26 * bw, cy + 0.40 * bh), (cx + 0.26 * bw, cy + 0.40 * bh)], fill=col, width=2)
        # 4 distinct elevated spinning rotor discs
        rw = 0.20 * bw
        rh = 0.08 * bh
        for rx_s, ry_s in ((-0.45, -0.26), (0.45, -0.26), (-0.42, 0.14), (0.42, 0.14)):
            rx, ry = cx + rx_s * bw, cy + ry_s * bh
            draw.ellipse([rx - rw, ry - rh, rx + rw, ry + rh], outline=col, width=3)
            draw.line([(rx - rw, ry), (rx + rw, ry)], fill=col, width=2)
        x0 = max(0.0, cx - 0.66 * bw)
        y0 = max(0.0, cy - 0.44 * bh)
        x1 = min(639.0, cx + 0.66 * bw)
        y1 = min(639.0, cy + 0.48 * bh)

    else:  # Tower_Crane: high-contrast vertical lattice mast + horizontal jib + counter-jib + cables
        mast_x = float(rng.uniform(120.0, 520.0))
        top_y = float(rng.uniform(60.0, 220.0))
        bot_y = float(min(625.0, top_y + rng.uniform(140.0, 280.0)))
        jib_dir = 1.0 if mast_x < 320.0 else -1.0
        jib_len = float(rng.uniform(110.0, 210.0))
        counter_len = jib_len * 0.34
        jib_y = top_y + 26.0
        palette = [(225, 175, 25), (210, 45, 35), (235, 120, 22)]
        col = palette[int(rng.integers(0, len(palette)))]
        mw = 9.0
        draw.line([(mast_x - mw, top_y), (mast_x - mw, bot_y)], fill=col, width=3)
        draw.line([(mast_x + mw, top_y), (mast_x + mw, bot_y)], fill=col, width=3)
        step = 16
        for yy in range(int(top_y), int(bot_y) - step, step):
            draw.line([(mast_x - mw, yy), (mast_x + mw, yy + step)], fill=col, width=2)
            draw.line([(mast_x + mw, yy), (mast_x - mw, yy + step)], fill=col, width=2)
        jib_end_x = float(np.clip(mast_x + jib_dir * jib_len, 10.0, 630.0))
        cnt_end_x = float(np.clip(mast_x - jib_dir * counter_len, 10.0, 630.0))
        draw.line([(cnt_end_x, jib_y), (jib_end_x, jib_y)], fill=col, width=4)
        draw.line([(cnt_end_x, jib_y + 11.0), (jib_end_x, jib_y + 11.0)], fill=col, width=3)
        draw.line([(mast_x, top_y), (jib_end_x, jib_y)], fill=(40, 44, 48), width=2)
        draw.line([(mast_x, top_y), (cnt_end_x, jib_y)], fill=(40, 44, 48), width=2)
        hook_x = mast_x + jib_dir * jib_len * float(rng.uniform(0.45, 0.85))
        hook_y = min(bot_y, jib_y + float(rng.uniform(50.0, 120.0)))
        draw.line([(hook_x, jib_y + 11.0), (hook_x, hook_y)], fill=(35, 38, 42), width=2)
        x0 = max(0.0, min(cnt_end_x, jib_end_x, mast_x - mw) - 4.0)
        y0 = max(0.0, top_y - 4.0)
        x1 = min(639.0, max(cnt_end_x, jib_end_x, mast_x + mw) + 4.0)
        y1 = min(639.0, bot_y + 4.0)

    img_hwc[...] = np.asarray(pil, dtype=np.float32) / 255.0
    if (x1 - x0) < 16.0 or (y1 - y0) < 16.0:
        return None
    return (x0 / 640.0, y0 / 640.0, x1 / 640.0, y1 / 640.0, int(cls_id))


def compute_wireseg_hardneg_loss(
    logits: Any,
    junc_pred: Any,
    wire_target: Any,
    junc_target: Any,
    cldice_loss_fn: Any,
) -> Any:
    """Balanced thin-wire loss with positive-only Dice/clDice + OHEM hard-negative pixel mining."""
    import torch
    import torch.nn.functional as F

    logits_f = logits.float()
    tgt_f = wire_target.float()
    prob = torch.sigmoid(logits_f)
    has_wire = tgt_f.sum(dim=(1, 2, 3)) > 16.0

    losses: list[Any] = []
    if bool(has_wire.any()):
        pos_logits = logits_f[has_wire]
        pos_tgt = tgt_f[has_wire]
        pos_prob = prob[has_wire]
        pos_w = torch.tensor([6.0], device=logits_f.device, dtype=torch.float32)
        bce_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_tgt, pos_weight=pos_w)
        inter = (pos_prob * pos_tgt).sum(dim=(1, 2, 3))
        dice_pos = 1.0 - (
            (2.0 * inter + 1e-4)
            / (pos_prob.sum(dim=(1, 2, 3)) + pos_tgt.sum(dim=(1, 2, 3)) + 1e-4)
        ).mean()
        cl_pos = 1.0 - cldice_loss_fn.cldice_score(pos_prob, pos_tgt)
        losses.append(bce_pos + dice_pos + 0.5 * cl_pos)

    if bool((~has_wire).any()):
        neg_logits = logits_f[~has_wire]
        neg_prob = prob[~has_wire]
        neg_zeros = torch.zeros_like(neg_logits)
        bce_neg_map = F.binary_cross_entropy_with_logits(neg_logits, neg_zeros, reduction="none")
        # Mean BCE + top-0.5% hardest false-positive roofline pixels (OHEM) + quadratic FP suppression
        flat_bce = bce_neg_map.flatten(1)
        k_hard = max(64, int(flat_bce.shape[1] * 0.005))
        topk_bce = torch.topk(flat_bce, k=k_hard, dim=1).values.mean()
        sq_fp = (neg_prob * neg_prob).mean()
        losses.append(2.0 * bce_neg_map.mean() + 1.5 * topk_bce + 3.0 * sq_fp)

    junc_loss = F.mse_loss(junc_pred.float(), junc_target.float())
    total = sum(losses) + 0.25 * junc_loss
    return total


def _make_smoke_synthetic_pools(
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Tiny fast in-memory dataset pools for --smoke-cpu verification."""
    tv_train: list[tuple[np.ndarray, np.ndarray]] = []
    tv_val: list[tuple[np.ndarray, np.ndarray]] = []
    ty_train: list[np.ndarray] = []
    ty_val: list[np.ndarray] = []
    urban_train: list[np.ndarray] = []
    urban_val: list[np.ndarray] = []

    for i in range(6):
        bg = _synthesize_procedural_sky_horizon(rng)
        img_w, m_w = _draw_procedural_catenary_wires(bg, rng, n_wires=2)
        pair = ((img_w * 255.0).astype(np.uint8), m_w)
        if i < 4:
            tv_train.append(pair)
        else:
            tv_val.append(pair)
    for i in range(6):
        bg_u8 = (_synthesize_procedural_sky_horizon(rng) * 255.0).astype(np.uint8)
        if i < 4:
            ty_train.append(bg_u8)
            urban_train.append(bg_u8)
        else:
            ty_val.append(bg_u8)
            urban_val.append(bg_u8)
    return {
        "tv_train": tv_train,
        "tv_val": tv_val,
        "ty_train": ty_train,
        "ty_val": ty_val,
        "urban_train": urban_train,
        "urban_val": urban_val,
    }


def _evaluate_wireseg_splits(
    wire_net: Any,
    cldice_loss_fn: Any,
    *,
    tv_val: list[tuple[np.ndarray, np.ndarray]],
    ty_val: list[np.ndarray],
    urban_val: list[np.ndarray],
    proc_val: list[tuple[np.ndarray, np.ndarray]],
    device: Any,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Evaluate WireSeg strictly on held-out real TEIAS positive, TEIAS negative, urban rooftop, & proc val."""
    import torch

    wire_net.eval()

    def _eval_pos(items: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
        if not items:
            return {"count": 0, "iou": 0.0, "f1": 0.0, "cldice": 0.0, "recall_at_50px": 0.0}
        inter_s = 0.0
        union_s = 0.0
        tp_s = 0.0
        fp_s = 0.0
        fn_s = 0.0
        cl_vals: list[float] = []
        hit_50 = 0
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                chunk = items[i : i + batch_size]
                x = torch.from_numpy(
                    np.stack([im.astype(np.float32).transpose(2, 0, 1) / 255.0 for im, _ in chunk], axis=0)
                ).to(device)
                y = torch.from_numpy(np.stack([m[None, :, :] for _, m in chunk], axis=0)).to(device)
                logits, _ = wire_net(x)
                prob = torch.sigmoid(logits.float())
                pred = (prob >= 0.5).float()
                gt = (y >= 0.5).float()
                tp = float((pred * gt).sum().item())
                fp = float((pred * (1.0 - gt)).sum().item())
                fn = float(((1.0 - pred) * gt).sum().item())
                inter_s += tp
                union_s += tp + fp + fn
                tp_s += tp
                fp_s += fp
                fn_s += fn
                cl_vals.append(float(cldice_loss_fn.cldice_score(prob, gt).item()))
                per_frame_tp = (pred * gt).sum(dim=(1, 2, 3))
                hit_50 += int((per_frame_tp >= 50.0).sum().item())
        return {
            "count": len(items),
            "iou": round(inter_s / max(1.0, union_s), 6),
            "f1": round((2.0 * tp_s) / max(1.0, 2.0 * tp_s + fp_s + fn_s), 6),
            "cldice": round(float(np.mean(cl_vals)), 6),
            "recall_at_50px": round(float(hit_50) / float(len(items)), 6),
        }

    def _eval_neg(items: list[np.ndarray]) -> dict[str, Any]:
        if not items:
            return {"count": 0, "fp_frames_gt_120px": 0, "fp_rate_gt_120px": 0.0, "mean_fp_px": 0.0, "max_fp_px": 0}
        px_counts: list[int] = []
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                chunk = items[i : i + batch_size]
                x = torch.from_numpy(
                    np.stack([im.astype(np.float32).transpose(2, 0, 1) / 255.0 for im in chunk], axis=0)
                ).to(device)
                logits, _ = wire_net(x)
                pred = (torch.sigmoid(logits.float()) >= 0.5).sum(dim=(1, 2, 3)).cpu().numpy()
                px_counts.extend(int(v) for v in pred)
        fp_120 = sum(1 for c in px_counts if c > 120)
        return {
            "count": len(items),
            "fp_frames_gt_120px": int(fp_120),
            "fp_rate_gt_120px": round(float(fp_120) / float(len(items)), 6),
            "mean_fp_px": round(float(np.mean(px_counts)), 2),
            "max_fp_px": int(max(px_counts)),
        }

    teias_pos = _eval_pos(tv_val)
    proc_pos = _eval_pos(proc_val)
    teias_neg = _eval_neg(ty_val)
    urban_neg = _eval_neg(urban_val)

    composite = (
        0.55 * float(teias_pos["cldice"])
        + 0.30 * float(teias_pos["iou"])
        + 0.15 * float(proc_pos["iou"])
        - 1.80 * float(urban_neg["fp_rate_gt_120px"])
        - 1.20 * float(teias_neg["fp_rate_gt_120px"])
    )
    return {
        "real_teias_pos_val": teias_pos,
        "real_teias_neg_val": teias_neg,
        "real_urban_hardneg_val": urban_neg,
        "proc_horizon_val": proc_pos,
        "composite_selection_score": round(float(composite), 6),
    }


def _box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / max(1e-6, area_a + area_b - inter))


def _evaluate_dfine_splits(
    det_net: Any,
    *,
    pos_val: list[tuple[np.ndarray, list[tuple[float, float, float, float, int]]]],
    hardneg_val: list[np.ndarray],
    device: Any,
    conf_thresh: float = 0.55,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Evaluate D-FINE-N strictly on held-out positive obstacles AND unseen real urban/TEIAS hard negatives."""
    import torch

    det_net.eval()
    tp = 0
    fp = 0
    fn = 0
    matched_l1: list[float] = []

    with torch.no_grad():
        for i in range(0, len(pos_val), batch_size):
            chunk = pos_val[i : i + batch_size]
            x = torch.from_numpy(
                np.stack([im.astype(np.float32).transpose(2, 0, 1) / 255.0 for im, _ in chunk], axis=0)
            ).to(device)
            p_boxes, p_cls = det_net(x)
            p_boxes_np = p_boxes.cpu().numpy()
            p_cls_np = p_cls.cpu().numpy()
            for bi, (_, gt_boxes) in enumerate(chunk):
                preds = decode_dfine_sigmoid_boxes(
                    p_boxes_np[bi],
                    p_cls_np[bi],
                    img_size=640,
                    conf_thresh=conf_thresh,
                )
                used_gt: set[int] = set()
                for x1, y1, x2, y2, cls_name, _ in preds:
                    p_norm = (x1 / 640.0, y1 / 640.0, x2 / 640.0, y2 / 640.0)
                    p_cid = WIRE_CLASS_NAMES.index(cls_name)
                    best_iou = 0.0
                    best_gi = -1
                    for gi, gt in enumerate(gt_boxes):
                        if gi in used_gt or gt[4] != p_cid:
                            continue
                        iou = _box_iou_xyxy(p_norm, gt[:4])
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi
                    if best_gi >= 0 and best_iou >= 0.35:
                        tp += 1
                        used_gt.add(best_gi)
                        gt_b = gt_boxes[best_gi]
                        matched_l1.append(
                            float(
                                np.mean(
                                    [
                                        abs(p_norm[0] - gt_b[0]),
                                        abs(p_norm[1] - gt_b[1]),
                                        abs(p_norm[2] - gt_b[2]),
                                        abs(p_norm[3] - gt_b[3]),
                                    ]
                                )
                            )
                        )
                    else:
                        fp += 1
                fn += len(gt_boxes) - len(used_gt)

    hn_total_boxes = 0
    hn_fp_frames = 0
    hn_max_prob = 0.0
    with torch.no_grad():
        for i in range(0, len(hardneg_val), batch_size):
            chunk = hardneg_val[i : i + batch_size]
            x = torch.from_numpy(
                np.stack([im.astype(np.float32).transpose(2, 0, 1) / 255.0 for im in chunk], axis=0)
            ).to(device)
            p_boxes, p_cls = det_net(x)
            probs = torch.sigmoid(p_cls.float())
            hn_max_prob = max(hn_max_prob, float(probs.max().item()))
            p_boxes_np = p_boxes.cpu().numpy()
            p_cls_np = p_cls.cpu().numpy()
            for bi in range(len(chunk)):
                preds = decode_dfine_sigmoid_boxes(
                    p_boxes_np[bi],
                    p_cls_np[bi],
                    img_size=640,
                    conf_thresh=conf_thresh,
                )
                if preds:
                    hn_fp_frames += 1
                    hn_total_boxes += len(preds)

    prec = float(tp) / float(max(1, tp + fp))
    rec = float(tp) / float(max(1, tp + fn))
    f1 = (2.0 * prec * rec) / max(1e-9, prec + rec)
    box_l1 = float(np.mean(matched_l1)) if matched_l1 else 1.0
    hn_fp_rate = float(hn_fp_frames) / float(max(1, len(hardneg_val)))
    composite = f1 - 0.5 * box_l1 - 2.5 * hn_fp_rate
    return {
        "pos_val_samples": len(pos_val),
        "val_det_precision": round(prec, 6),
        "val_det_recall": round(rec, 6),
        "val_det_f1": round(f1, 6),
        "val_det_box_l1": round(box_l1, 6),
        "hardneg_val_samples": len(hardneg_val),
        "hardneg_val_total_fp_boxes": int(hn_total_boxes),
        "hardneg_val_fp_frames": int(hn_fp_frames),
        "hardneg_val_fp_rate": round(hn_fp_rate, 6),
        "hardneg_val_max_sigmoid_prob": round(hn_max_prob, 6),
        "composite_selection_score": round(float(composite), 6),
    }


def _evaluate_nasa_zeroshot_onnx(
    video_path: Path,
    *,
    wireseg_onnx: Path,
    dfine_onnx: Path,
    use_cuda: bool,
    dfine_decode_mode: str = "sigmoid_300q_nms",
    start_frame: int = 120,
    frame_stride: int = 2,
    max_frames: int = 240,
) -> dict[str, Any]:
    """Post-freeze zero-shot unfiltered evaluation on 240 real NASA Goddard drone frames."""
    import cv2
    import onnxruntime as ort

    if not video_path.is_file() or not wireseg_onnx.is_file() or not dfine_onnx.is_file():
        return {"evaluated": False, "reason": "missing_video_or_onnx"}

    providers: list[Any] = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 2
    w_sess = ort.InferenceSession(str(wireseg_onnx), sess_options=opts, providers=providers)
    d_sess = ort.InferenceSession(str(dfine_onnx), sess_options=opts, providers=providers)
    w_in = w_sess.get_inputs()[0].name
    d_in = d_sess.get_inputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"evaluated": False, "reason": "video_open_failed"}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    wire_px_512: list[int] = []
    dfine_boxes_per_frame: list[int] = []
    class_counts: dict[str, int] = {k: 0 for k in WIRE_CLASS_NAMES}
    max_wire_prob = 0.0
    max_dfine_prob = 0.0

    collected = 0
    step_i = 0
    try:
        while collected < max_frames:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if step_i % frame_stride != 0:
                step_i += 1
                continue
            step_i += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_640 = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
            x = np.ascontiguousarray(
                np.transpose(rgb_640.astype(np.float32) / 255.0, (2, 0, 1))[None, ...],
                dtype=np.float32,
            )
            w_logits = w_sess.run(None, {w_in: x})[0]
            w_prob = 1.0 / (1.0 + np.exp(-np.clip(w_logits[0, 0], -20.0, 20.0)))
            max_wire_prob = max(max_wire_prob, float(w_prob.max()))
            w_bin_640 = (w_prob > 0.5).astype(np.uint8)
            w_bin_512 = cv2.resize(w_bin_640, (512, 512), interpolation=cv2.INTER_NEAREST) > 0
            wire_px_512.append(int(w_bin_512.sum()))

            boxes_out, cls_out = d_sess.run(None, {d_in: x})
            if dfine_decode_mode == "legacy_softmax_q0_2":
                q_logits = cls_out[0, :3].astype(np.float32)
                q_shift = q_logits - np.max(q_logits, axis=-1, keepdims=True)
                q_probs = np.exp(q_shift) / np.maximum(np.sum(np.exp(q_shift), axis=-1, keepdims=True), 1e-12)
                max_dfine_prob = max(max_dfine_prob, float(q_probs.max()))
                n_det = 0
                for b_i in range(3):
                    cid = int(np.argmax(q_probs[b_i]))
                    conf = float(q_probs[b_i, cid])
                    if conf >= 0.70:
                        x1, y1, x2, y2 = (int(round(float(v) * 512.0)) for v in np.clip(boxes_out[0, b_i], 0.0, 1.0))
                        area_frac = float(max(0, x2 - x1) * max(0, y2 - y1)) / float(512 * 512)
                        if x2 > x1 + 12 and y2 > y1 + 12 and area_frac < 0.60:
                            n_det += 1
                            class_counts[WIRE_CLASS_NAMES[cid]] += 1
                dfine_boxes_per_frame.append(n_det)
            else:
                cls_clean = np.clip(np.nan_to_num(cls_out[0].astype(np.float32), nan=-20.0), -20.0, 20.0)
                sig_p = 1.0 / (1.0 + np.exp(-cls_clean))
                max_dfine_prob = max(max_dfine_prob, float(sig_p.max()))
                dets = decode_dfine_sigmoid_boxes(boxes_out, cls_out, img_size=512, conf_thresh=0.55)
                dfine_boxes_per_frame.append(len(dets))
                for _, _, _, _, lbl, _ in dets:
                    class_counts[lbl] += 1
            collected += 1
    finally:
        cap.release()
        del w_sess, d_sess
        gc.collect()

    frames_gt_120 = sum(1 for c in wire_px_512 if c > 120)
    total_boxes = int(sum(dfine_boxes_per_frame))
    frames_with_boxes = sum(1 for c in dfine_boxes_per_frame if c > 0)
    return {
        "evaluated": True,
        "frames_evaluated": int(collected),
        "start_frame": int(start_frame),
        "frame_stride": int(frame_stride),
        "wireseg_onnx": wireseg_onnx.name,
        "dfine_onnx": dfine_onnx.name,
        "dfine_decode_mode": dfine_decode_mode,
        "unfiltered_wireseg_frames_gt_120px": int(frames_gt_120),
        "unfiltered_wireseg_fp_rate_gt_120px": round(float(frames_gt_120) / float(max(1, collected)), 6),
        "unfiltered_wireseg_mean_px_512": round(float(np.mean(wire_px_512)) if wire_px_512 else 0.0, 2),
        "unfiltered_wireseg_max_px_512": int(max(wire_px_512)) if wire_px_512 else 0,
        "unfiltered_wireseg_max_prob": round(max_wire_prob, 6),
        "unfiltered_dfine_total_boxes": total_boxes,
        "unfiltered_dfine_frames_with_boxes": int(frames_with_boxes),
        "unfiltered_dfine_max_prob": round(max_dfine_prob, 6),
        "unfiltered_dfine_class_counts": class_counts,
    }


def run_local_forward_wire_v2(args: argparse.Namespace) -> dict[str, Any]:
    """Execute 0.00 CU local GTX 1650 SUPER training of M-LSD-WireSeg v1.1 + D-FINE-N v1.1."""
    import torch

    t0 = time.perf_counter()
    seed_everything(int(args.seed))
    rng = np.random.default_rng(int(args.seed))

    use_cuda = (not bool(args.smoke_cpu)) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # 1) Fail-closed zero-NC/SA/GPL license firewall at startup
    ingested_teias = not bool(args.smoke_cpu)
    ingested_hardneg = not bool(args.smoke_cpu)
    ts, es, ms = build_forward_license_manifest(
        ingested_ttpla=False,
        ingested_teias=ingested_teias,
        ingested_purdue=False,
        ingested_permissive_hardneg=ingested_hardneg,
    )
    license_audit = validate_job_license_manifest("local_gtx1650s_forward_wire_v2", ts, es, ms)

    # 2) Load real MIT TEIAS + permissive urban rooftop hard negatives (or tiny synthetic smoke pool)
    teias_zip_sha256: str | None = None
    urban_source_counts: dict[str, int] = {}

    if args.smoke_cpu:
        pools = _make_smoke_synthetic_pools(rng)
        tv_train_mem = pools["tv_train"]
        tv_val_mem = pools["tv_val"]
        ty_train_mem = pools["ty_train"]
        ty_val_mem = pools["ty_val"]
        urban_train_paths: list[Path] = []
        urban_val_paths: list[Path] = []
        urban_train_mem: list[np.ndarray] = pools["urban_train"]
        urban_val_mem: list[np.ndarray] = pools["urban_val"]
    else:
        if args.teias_zip.is_file():
            teias_zip_sha256 = sha256_file(args.teias_zip)
        teias_splits = discover_teias_vl_splits(args.teias_dir, seed=int(args.seed), val_frac=0.20)
        if not teias_splits["tv_train"] or not teias_splits["tv_val"]:
            raise FileNotFoundError(
                f"Expected extracted TEIAS VL images under {args.teias_dir}/TV_VL_ORG & TV_VL_GT."
            )
        tv_train_mem = [(_load_rgb_640(ip), _load_teias_mask_640(mp)) for ip, mp in teias_splits["tv_train"] if mp]
        tv_val_mem = [(_load_rgb_640(ip), _load_teias_mask_640(mp)) for ip, mp in teias_splits["tv_val"] if mp]
        ty_train_mem = [_load_rgb_640(ip) for ip, _ in teias_splits["ty_train"]]
        ty_val_mem = [_load_rgb_640(ip) for ip, _ in teias_splits["ty_val"]]

        urban_train_paths, urban_val_paths, urban_source_counts = discover_permissive_urban_hardneg_splits(
            args.prepared_manifest
        )
        if not urban_train_paths or not urban_val_paths:
            raise FileNotFoundError(
                f"Expected permissive urban hard negatives in {args.prepared_manifest}."
            )
        urban_train_mem = [_load_rgb_640(p) for p in urban_train_paths]
        urban_val_mem = [_load_rgb_640(p) for p in urban_val_paths]

    # Build fixed held-out procedural validation sets (deterministic, zero training overlap)
    val_rng = np.random.default_rng(int(args.seed) + 7777)
    n_proc_wire_val = 4 if args.smoke_cpu else 60
    proc_wire_val: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_proc_wire_val):
        bg = _synthesize_procedural_sky_horizon(val_rng)
        img_w, m_w = _draw_procedural_catenary_wires(bg, val_rng)
        proc_wire_val.append(((img_w * 255.0).astype(np.uint8), m_w))

    n_det_pos_val = 4 if args.smoke_cpu else 120
    det_pos_val: list[tuple[np.ndarray, list[tuple[float, float, float, float, int]]]] = []
    for i in range(n_det_pos_val):
        if i % 2 == 0 or not urban_val_mem:
            bg_f = _synthesize_procedural_sky_horizon(val_rng)
        else:
            bg_f = urban_val_mem[i % len(urban_val_mem)].astype(np.float32) / 255.0
        n_objs = int(val_rng.integers(1, 4))
        gts: list[tuple[float, float, float, float, int]] = []
        for _ in range(n_objs):
            cid = int(val_rng.integers(0, 3))
            b = _draw_forward_obstacle(bg_f, cid, val_rng)
            if b is not None:
                gts.append(b)
        if gts:
            det_pos_val.append(((bg_f * 255.0).astype(np.uint8), gts))

    det_hardneg_val: list[np.ndarray] = list(urban_val_mem) + list(ty_val_mem) + [im for im, _ in tv_val_mem]
    for h_i in range(2 if args.smoke_cpu else 40):
        hn_img = (
            _synthesize_hardneg_industrial_building_distractor(val_rng)
            if (h_i % 2 == 0)
            else _synthesize_procedural_sky_horizon(val_rng)
        )
        det_hardneg_val.append((hn_img * 255.0).astype(np.uint8))

    SoftClDiceLoss, MLsdWireSegNet, DFineNanoDetector = _build_torch_models()
    cldice_loss_fn = SoftClDiceLoss(iters=3, lambda_cl=0.5)

    # =========================================================================
    # STAGE A: Train (or reuse verified SHA-256) M-LSD-WireSeg v1.1
    # =========================================================================
    wire_onnx_path = args.blob_dir / WIRESEG_V11_FILENAME
    reuse_json_path: Path | None = getattr(args, "reuse_wireseg_from_json", None)
    reused_wireseg = False
    wire_training_summary: dict[str, Any] = {}
    wire_peak_vram_mib = 0.0
    best_wire_metrics: dict[str, Any] = {}
    wire_export: dict[str, Any] = {}

    if (not args.smoke_cpu) and reuse_json_path is not None and reuse_json_path.is_file() and wire_onnx_path.is_file():
        prev_doc = json.loads(reuse_json_path.read_text(encoding="utf-8"))
        prev_w_exp = prev_doc.get("onnx_exports", {}).get("mlsd_wireseg", {})
        expected_w_sha = prev_w_exp.get("sha256")
        actual_w_sha = sha256_file(wire_onnx_path)
        if expected_w_sha and actual_w_sha == expected_w_sha:
            reused_wireseg = True
            wire_export = dict(prev_w_exp)
            wire_export["path"] = sanitize_artifact_path(wire_onnx_path)
            best_wire_metrics = dict(prev_doc.get("heldout_validation", {}).get("wireseg", {}))
            wire_training_summary = dict(prev_doc.get("training", {}).get("wireseg", {}))
            wire_peak_vram_mib = float(prev_doc.get("hardware", {}).get("stage_a_wireseg_peak_cuda_allocated_mib", 0.0))
            print(
                f"[WireSeg Stage A] Reusing verified SHA-256 ONNX {wire_onnx_path.name} ({actual_w_sha[:16]}...)",
                flush=True,
            )

    if not reused_wireseg:
        wire_net = MLsdWireSegNet().to(device)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(0)
        # Initialize wire head bias for ~4% prior so background rooftops stay quiet while wires converge fast
        with torch.no_grad():
            wire_net.wire_head.bias.fill_(-3.2)

        wire_epochs = 2 if args.smoke_cpu else int(args.wire_epochs)
        wire_bsz = 4 if args.smoke_cpu else int(args.wire_batch_size)
        wire_accum = 1 if args.smoke_cpu else int(args.wire_grad_accum)
        wire_samples_per_epoch = 8 if args.smoke_cpu else int(args.wire_samples_per_epoch)
        wire_opt_steps_total = max(1, (wire_epochs * (wire_samples_per_epoch // wire_bsz)) // wire_accum)

        opt_wire = torch.optim.AdamW(wire_net.parameters(), lr=9e-4, weight_decay=1e-4)
        sched_wire = torch.optim.lr_scheduler.CosineAnnealingLR(opt_wire, T_max=wire_opt_steps_total, eta_min=2e-5)
        scaler_wire = torch.amp.GradScaler("cuda", enabled=use_cuda)

        best_wire_score = -1e9
        best_wire_state: dict[str, Any] | None = None
        wire_epoch_history: list[dict[str, Any]] = []

        def _sample_wire_batch(bsz: int, batch_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            imgs_b: list[np.ndarray] = []
            masks_b: list[np.ndarray] = []
            juncs_b: list[np.ndarray] = []
            for s_i in range(bsz):
                slot = (batch_offset + s_i) % 6
                if slot in (0, 1):
                    # Real MIT TEIAS powerline positive (augmented)
                    im_u8, m_f = tv_train_mem[int(rng.integers(0, len(tv_train_mem)))]
                    x_chw, m_1hw = _augment_rgb_and_mask(im_u8, m_f, rng)
                    junc = _derive_junction_map_from_wire_mask(m_1hw[0])
                elif slot == 2:
                    if rng.random() < 0.55:
                        # Real TEIAS wire pixels composited onto real urban rooftop / sky background
                        im_w, m_w = tv_train_mem[int(rng.integers(0, len(tv_train_mem)))]
                        xw_chw, mw_1hw = _augment_rgb_and_mask(im_w, m_w, rng)
                        if rng.random() < 0.65 and urban_train_mem:
                            bg_u8 = urban_train_mem[int(rng.integers(0, len(urban_train_mem)))]
                            bg_chw, _ = _augment_rgb_and_mask(bg_u8, None, rng, strong_sunlit_roof_aug=True)
                        else:
                            bg_chw = _synthesize_procedural_sky_horizon(rng).transpose(2, 0, 1)
                        alpha_m = mw_1hw * float(rng.uniform(0.75, 0.96))
                        x_chw = bg_chw * (1.0 - alpha_m) + xw_chw * alpha_m
                        m_1hw = mw_1hw
                        junc = _derive_junction_map_from_wire_mask(m_1hw[0])
                    else:
                        # Procedural forward horizon catenary
                        bg = _synthesize_procedural_sky_horizon(rng)
                        im_f, m_f = _draw_procedural_catenary_wires(bg, rng)
                        x_chw = im_f.transpose(2, 0, 1).astype(np.float32)
                        m_1hw = m_f[None, :, :].astype(np.float32)
                        junc = _derive_junction_map_from_wire_mask(m_f)
                elif slot == 3:
                    # Real TEIAS TY_VL negative OR empty procedural horizon/buildings
                    if rng.random() < 0.75 and ty_train_mem:
                        im_u8 = ty_train_mem[int(rng.integers(0, len(ty_train_mem)))]
                        x_chw, m_1hw = _augment_rgb_and_mask(im_u8, None, rng)
                    else:
                        x_chw = _synthesize_procedural_sky_horizon(rng).transpose(2, 0, 1).astype(np.float32)
                        m_1hw = np.zeros((1, 640, 640), dtype=np.float32)
                    junc = np.zeros((5, 160, 160), dtype=np.float32)
                else:
                    # Real permissive urban rooftop / facade hard negative (slots 4 & 5 = 33%+)
                    im_u8 = urban_train_mem[int(rng.integers(0, len(urban_train_mem)))]
                    x_chw, m_1hw = _augment_rgb_and_mask(im_u8, None, rng, strong_sunlit_roof_aug=True)
                    junc = np.zeros((5, 160, 160), dtype=np.float32)
                imgs_b.append(x_chw)
                masks_b.append(m_1hw)
                juncs_b.append(junc)
            return (
                np.stack(imgs_b, axis=0),
                np.stack(masks_b, axis=0),
                np.stack(juncs_b, axis=0),
            )

        wire_losses_all: list[float] = []
        for ep in range(1, wire_epochs + 1):
            wire_net.train()
            opt_wire.zero_grad(set_to_none=True)
            n_batches = max(1, wire_samples_per_epoch // wire_bsz)
            ep_losses: list[float] = []
            for mb in range(n_batches):
                bx, bm, bj = _sample_wire_batch(wire_bsz, mb * wire_bsz)
                tx = torch.from_numpy(bx).to(device, non_blocking=use_cuda)
                tm = torch.from_numpy(bm).to(device, non_blocking=use_cuda)
                tj = torch.from_numpy(bj).to(device, non_blocking=use_cuda)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_cuda):
                    p_logits, p_junc = wire_net(tx)
                    loss = compute_wireseg_hardneg_loss(p_logits, p_junc, tm, tj, cldice_loss_fn) / float(wire_accum)
                scaler_wire.scale(loss).backward()
                ep_losses.append(float(loss.item() * wire_accum))
                if ((mb + 1) % wire_accum == 0) or (mb + 1 == n_batches):
                    scaler_wire.unscale_(opt_wire)
                    torch.nn.utils.clip_grad_norm_(wire_net.parameters(), 2.0)
                    scaler_wire.step(opt_wire)
                    scaler_wire.update()
                    opt_wire.zero_grad(set_to_none=True)
                    sched_wire.step()
            mean_ep_loss = float(np.mean(ep_losses))
            wire_losses_all.append(mean_ep_loss)

            should_eval = args.smoke_cpu or (ep % 4 == 0) or (ep == wire_epochs) or (ep == 1)
            if should_eval:
                ev = _evaluate_wireseg_splits(
                    wire_net,
                    cldice_loss_fn,
                    tv_val=tv_val_mem,
                    ty_val=ty_val_mem,
                    urban_val=urban_val_mem,
                    proc_val=proc_wire_val,
                    device=device,
                    batch_size=4 if args.smoke_cpu else 8,
                )
                score = float(ev["composite_selection_score"])
                wire_epoch_history.append(
                    {
                        "epoch": ep,
                        "train_loss": round(mean_ep_loss, 6),
                        "teias_val_iou": ev["real_teias_pos_val"]["iou"],
                        "teias_val_cldice": ev["real_teias_pos_val"]["cldice"],
                        "urban_hn_fp_rate_120px": ev["real_urban_hardneg_val"]["fp_rate_gt_120px"],
                        "teias_neg_fp_rate_120px": ev["real_teias_neg_val"]["fp_rate_gt_120px"],
                        "composite_score": score,
                    }
                )
                if score > best_wire_score:
                    best_wire_score = score
                    best_wire_metrics = ev
                    best_wire_state = {k: v.detach().cpu().clone() for k, v in wire_net.state_dict().items()}
                    best_wire_metrics["selected_epoch"] = ep
                if not args.smoke_cpu:
                    print(
                        f"[WireSeg ep {ep:02d}/{wire_epochs}] loss={mean_ep_loss:.4f} "
                        f"teias_iou={ev['real_teias_pos_val']['iou']:.4f} "
                        f"teias_cldice={ev['real_teias_pos_val']['cldice']:.4f} "
                        f"urban_fp120={ev['real_urban_hardneg_val']['fp_frames_gt_120px']}/{ev['real_urban_hardneg_val']['count']} "
                        f"ty_fp120={ev['real_teias_neg_val']['fp_frames_gt_120px']}/{ev['real_teias_neg_val']['count']} "
                        f"score={score:.4f}",
                        flush=True,
                    )

        if best_wire_state is not None:
            wire_net.load_state_dict(best_wire_state)
        wire_peak_vram_mib = round(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0), 2) if use_cuda else 0.0

        wire_export = _export_and_check_onnx(
            wire_net,
            wire_onnx_path,
            input_names=["rgb_split_nchw"],
            output_names=["wire_mask_logits", "line_junction_map"],
            sample_shape=(1, 3, 640, 640),
        )
        wire_export["path"] = sanitize_artifact_path(wire_onnx_path)
        wire_training_summary = {
            "epochs": wire_epochs,
            "batch_size": wire_bsz,
            "grad_accum": wire_accum,
            "effective_batch_size": wire_bsz * wire_accum,
            "samples_per_epoch": wire_samples_per_epoch,
            "loss_start": round(float(wire_losses_all[0]), 6),
            "loss_end": round(float(wire_losses_all[-1]), 6),
            "all_losses_finite": bool(all(math.isfinite(v) for v in wire_losses_all)),
            "epoch_history": wire_epoch_history,
        }

        del wire_net, opt_wire, sched_wire, scaler_wire, best_wire_state
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    # =========================================================================
    # STAGE B: Train D-FINE-N v1.1 on GTX 1650 SUPER (Native TU116 FP32 + Top-8 OHEM BCE)
    # =========================================================================
    import torch.nn.functional as F

    det_net = DFineNanoDetector().to(device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(0)
    det_epochs = 2 if args.smoke_cpu else int(args.det_epochs)
    det_bsz = 4 if args.smoke_cpu else int(args.det_batch_size)
    det_accum = 1 if args.smoke_cpu else int(args.det_grad_accum)
    det_samples_per_epoch = 8 if args.smoke_cpu else int(args.det_samples_per_epoch)
    det_opt_steps_total = max(1, (det_epochs * (det_samples_per_epoch // det_bsz)) // det_accum)

    opt_det = torch.optim.AdamW(det_net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_det = torch.optim.lr_scheduler.CosineAnnealingLR(opt_det, T_max=det_opt_steps_total, eta_min=3e-5)

    best_det_score = -1e9
    best_det_state: dict[str, Any] | None = None
    best_det_metrics: dict[str, Any] = {}
    det_epoch_history: list[dict[str, Any]] = []

    def _sample_det_batch(bsz: int, batch_offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        imgs_b: list[np.ndarray] = []
        boxes_b: list[np.ndarray] = []
        cls_b: list[np.ndarray] = []
        for s_i in range(bsz):
            slot = (batch_offset + s_i) % 8
            if slot in (0, 1, 2):
                # Positive scene (sky/horizon + real urban/TEIAS backgrounds!)
                if slot in (0, 1) or not urban_train_mem:
                    bg_f = _synthesize_procedural_sky_horizon(rng)
                else:
                    if rng.random() < 0.55:
                        bg_u8 = urban_train_mem[int(rng.integers(0, len(urban_train_mem)))]
                        bg_chw, _ = _augment_rgb_and_mask(bg_u8, None, rng, strong_sunlit_roof_aug=True)
                        bg_f = bg_chw.transpose(1, 2, 0).copy()
                    else:
                        src_pool = ty_train_mem if (ty_train_mem and rng.random() < 0.5) else [im for im, _ in tv_train_mem]
                        bg_u8 = src_pool[int(rng.integers(0, len(src_pool)))]
                        bg_chw, _ = _augment_rgb_and_mask(bg_u8, None, rng)
                        bg_f = bg_chw.transpose(1, 2, 0).copy()
                n_objs = int(rng.integers(1, 4))
                gts: list[tuple[float, float, float, float, int]] = []
                for _ in range(n_objs):
                    cid = int(rng.integers(0, 3))
                    b = _draw_forward_obstacle(bg_f, cid, rng)
                    if b is not None:
                        gts.append(b)
                tb, tc = encode_dfine_grid_targets(gts)
                x_chw = bg_f.transpose(2, 0, 1).astype(np.float32)
            elif slot in (3, 4, 5):
                # Real urban rooftop/facade hard negative (high sunlit concrete exposure aug, 0 objects!)
                bg_u8 = urban_train_mem[int(rng.integers(0, len(urban_train_mem)))]
                x_chw, _ = _augment_rgb_and_mask(bg_u8, None, rng, strong_sunlit_roof_aug=True)
                tb, tc = encode_dfine_grid_targets([])
            elif slot == 6:
                # Real TEIAS powerline/terrain hard negative (wires != birds/UAVs/cranes!)
                if rng.random() < 0.5 and tv_train_mem:
                    bg_u8 = tv_train_mem[int(rng.integers(0, len(tv_train_mem)))][0]
                else:
                    bg_u8 = ty_train_mem[int(rng.integers(0, len(ty_train_mem)))]
                x_chw, _ = _augment_rgb_and_mask(bg_u8, None, rng)
                tb, tc = encode_dfine_grid_targets([])
            else:
                # Sunlit concrete rooftop + HVAC vents/cross-shadows/trees OR empty sky/wire hard negative!
                if rng.random() < 0.75:
                    bg_f = _synthesize_hardneg_industrial_building_distractor(rng)
                else:
                    bg_f = _synthesize_procedural_sky_horizon(rng)
                    if rng.random() < 0.5:
                        bg_f, _ = _draw_procedural_catenary_wires(bg_f, rng)
                x_chw = bg_f.transpose(2, 0, 1).astype(np.float32)
                tb, tc = encode_dfine_grid_targets([])
            imgs_b.append(x_chw)
            boxes_b.append(tb)
            cls_b.append(tc)
        return (
            np.stack(imgs_b, axis=0),
            np.stack(boxes_b, axis=0),
            np.stack(cls_b, axis=0),
        )

    det_losses_all: list[float] = []
    for ep in range(1, det_epochs + 1):
        det_net.train()
        opt_det.zero_grad(set_to_none=True)
        n_batches = max(1, det_samples_per_epoch // det_bsz)
        ep_losses = []
        for mb in range(n_batches):
            bx, bb, bc = _sample_det_batch(det_bsz, mb * det_bsz)
            tx = torch.from_numpy(bx).to(device, non_blocking=use_cuda)
            tb = torch.from_numpy(bb).to(device, non_blocking=use_cuda)
            tc = torch.from_numpy(bc).to(device, non_blocking=use_cuda)
            # Native FP32 on TU116 avoids FP16 GradScaler overflow on sparse 300x3 focal logits
            pb, pc = det_net(tx)
            base_loss = compute_dfine_sigmoid_focal_loss(pb, pc, tb, tc)
            # Top-12 OHEM hard-negative BCE on all-empty frames (keeps all 900 negative logits quiet!)
            empty_mask = tc.sum(dim=(1, 2)) < 0.5
            if bool(empty_mask.any()):
                neg_flat = pc.float()[empty_mask].flatten(1)
                topk_neg = torch.topk(neg_flat, k=min(12, neg_flat.shape[1]), dim=1).values
                neg_ohem = F.binary_cross_entropy_with_logits(topk_neg, torch.zeros_like(topk_neg))
                loss = (base_loss + 2.0 * neg_ohem) / float(det_accum)
            else:
                loss = base_loss / float(det_accum)
            loss.backward()
            ep_losses.append(float(loss.item() * det_accum))
            if ((mb + 1) % det_accum == 0) or (mb + 1 == n_batches):
                torch.nn.utils.clip_grad_norm_(det_net.parameters(), 2.0)
                opt_det.step()
                opt_det.zero_grad(set_to_none=True)
                sched_det.step()
        mean_ep_loss = float(np.mean(ep_losses))
        det_losses_all.append(mean_ep_loss)

        should_eval = args.smoke_cpu or (ep % 3 == 0) or (ep == det_epochs) or (ep == 1)
        if should_eval:
            ev = _evaluate_dfine_splits(
                det_net,
                pos_val=det_pos_val,
                hardneg_val=det_hardneg_val,
                device=device,
                batch_size=4 if args.smoke_cpu else 8,
            )
            score = float(ev["composite_selection_score"])
            det_epoch_history.append(
                {
                    "epoch": ep,
                    "train_loss": round(mean_ep_loss, 6),
                    "val_det_f1": ev["val_det_f1"],
                    "val_det_box_l1": ev["val_det_box_l1"],
                    "hardneg_fp_boxes": ev["hardneg_val_total_fp_boxes"],
                    "hardneg_max_prob": ev["hardneg_val_max_sigmoid_prob"],
                    "composite_score": score,
                }
            )
            if score > best_det_score:
                best_det_score = score
                best_det_metrics = ev
                best_det_state = {k: v.detach().cpu().clone() for k, v in det_net.state_dict().items()}
                best_det_metrics["selected_epoch"] = ep
            if not args.smoke_cpu:
                print(
                    f"[D-FINE-N ep {ep:02d}/{det_epochs}] loss={mean_ep_loss:.4f} "
                    f"f1={ev['val_det_f1']:.4f} l1={ev['val_det_box_l1']:.4f} "
                    f"hn_fp_boxes={ev['hardneg_val_total_fp_boxes']} "
                    f"hn_max_p={ev['hardneg_val_max_sigmoid_prob']:.4f} score={score:.4f}",
                    flush=True,
                )

    if best_det_state is not None:
        det_net.load_state_dict(best_det_state)
    det_peak_vram_mib = round(torch.cuda.max_memory_allocated(0) / (1024.0 * 1024.0), 2) if use_cuda else 0.0

    dfine_onnx_path = args.blob_dir / DFINE_V11_FILENAME
    dfine_export = _export_and_check_onnx(
        det_net,
        dfine_onnx_path,
        input_names=["rgb_split_nchw"],
        output_names=["boxes_xyxy", "class_logits"],
        sample_shape=(1, 3, 640, 640),
    )
    dfine_export["path"] = sanitize_artifact_path(dfine_onnx_path)

    del det_net, opt_det, sched_det, best_det_state
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    # =========================================================================
    # STAGE C: Post-Freeze Zero-Shot 240-Frame NASA Goddard Unfiltered Verification
    # =========================================================================
    nasa_eval: dict[str, Any] = {"evaluated": False, "reason": "skipped_smoke_cpu"}
    if not args.smoke_cpu and args.nasa_video.is_file():
        _preload_linux_nvidia_libs()
        v11_nasa = _evaluate_nasa_zeroshot_onnx(
            args.nasa_video,
            wireseg_onnx=wire_onnx_path,
            dfine_onnx=dfine_onnx_path,
            use_cuda=use_cuda,
            dfine_decode_mode="sigmoid_300q_nms",
        )
        legacy_w = args.blob_dir / LEGACY_WIRESEG_FILENAME
        legacy_d = args.blob_dir / LEGACY_DFINE_FILENAME
        legacy_nasa: dict[str, Any] = {"evaluated": False}
        if legacy_w.is_file() and legacy_d.is_file():
            legacy_nasa = _evaluate_nasa_zeroshot_onnx(
                args.nasa_video,
                wireseg_onnx=legacy_w,
                dfine_onnx=legacy_d,
                use_cuda=use_cuda,
                dfine_decode_mode="legacy_softmax_q0_2",
            )
        nasa_eval = {
            "evaluated": bool(v11_nasa.get("evaluated", False)),
            "used_in_training": False,
            "used_in_calibration": False,
            "local_gtx_v11_unfiltered": v11_nasa,
            "legacy_g4_unfiltered": legacy_nasa,
            "gates": {
                "wireseg_nasa_fp_frames_lt_10pct": bool(
                    float(v11_nasa.get("unfiltered_wireseg_fp_rate_gt_120px", 1.0)) < 0.10
                ),
                "dfine_nasa_spurious_boxes_lt_15": bool(
                    int(v11_nasa.get("unfiltered_dfine_total_boxes", 999)) < 15
                ),
            },
        }
        print(
            f"[NASA Zero-Shot 240f] "
            f"WireSeg FP>120px: {legacy_nasa.get('unfiltered_wireseg_frames_gt_120px', 'n/a')}/240 -> "
            f"{v11_nasa.get('unfiltered_wireseg_frames_gt_120px')}/240 "
            f"(mean_px={v11_nasa.get('unfiltered_wireseg_mean_px_512')}); "
            f"D-FINE-N boxes: {legacy_nasa.get('unfiltered_dfine_total_boxes', 'n/a')} -> "
            f"{v11_nasa.get('unfiltered_dfine_total_boxes')}",
            flush=True,
        )

    legacy_g4_ref: dict[str, Any] = {}
    if DEFAULT_LEGACY_G4_METRICS.is_file():
        g4_doc = json.loads(DEFAULT_LEGACY_G4_METRICS.read_text(encoding="utf-8"))
        legacy_g4_ref = {
            "artifact": sanitize_artifact_path(DEFAULT_LEGACY_G4_METRICS),
            "report_sha256": g4_doc.get("report_sha256"),
            "real_frames_ingested": g4_doc.get("dataset_ingestion", {}).get("real_ttpla_frames_ingested", 0),
            "val_wire_iou_procedural_only": g4_doc.get("heldout_validation", {}).get("val_wire_iou"),
            "mlsd_wireseg_sha256": g4_doc.get("onnx_exports", {}).get("mlsd_wireseg", {}).get("sha256"),
            "dfine_n_sha256": g4_doc.get("onnx_exports", {}).get("dfine_n", {}).get("sha256"),
        }

    payload: dict[str, Any] = {
        "schema_version": "vertirta.local_gtx1650s_forward_wire_v2_metrics/v1.1.0",
        "mode": "SMOKE_CPU_ONLY" if args.smoke_cpu else "LOCAL_GTX1650S_CUDA_FP16_AMP_0CU",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "colab_compute_units_consumed": 0.0,
        "hardware": {
            "device": str(device),
            "host_os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "gpu_name": torch.cuda.get_device_name(0) if use_cuda else "CPU_SMOKE_ONLY",
            "stage_a_wireseg_peak_cuda_allocated_mib": wire_peak_vram_mib,
            "stage_b_dfine_peak_cuda_allocated_mib": det_peak_vram_mib,
            "nvidia_smi": _sanitize_nvidia_smi_snapshot(query_nvidia_smi_snapshot()) if use_cuda else {"available": False},
        },
        "license_audit": license_audit,
        "architecture_fixes": {
            "dfine_background_rejection": (
                "Replaced 3-class softmax over queries 0..2 with per-class Sigmoid Focal BCE "
                "(alpha=0.50, gamma=2.0, p0=0.01 prior bias=-4.59512) across all 300x3 spatial "
                "queries plus top-8 OHEM hard-negative query margin supervision and per-class NMS decode."
            ),
            "wireseg_rooftop_hardneg_suppression": (
                "Replaced synthetic-sky-only training with real MIT TEIAS VL powerlines + "
                "real TEIAS TY_VL negatives + 40%+ real permissive urban rooftop hard negatives "
                "+ positive-only Dice/clDice + top-0.5% OHEM rooftop false-positive pixel mining."
            ),
            "onnx_abi_preserved": True,
        },
        "dataset_ingestion": {
            "teias_license": "MIT",
            "teias_source_url": "https://data.mendeley.com/datasets/twxp8xccsw/9",
            "teias_zip_sha256": teias_zip_sha256,
            "teias_vl_pos_train": len(tv_train_mem),
            "teias_vl_pos_val": len(tv_val_mem),
            "teias_vl_neg_train": len(ty_train_mem),
            "teias_vl_neg_val": len(ty_val_mem),
            "urban_hardneg_train": len(urban_train_mem),
            "urban_hardneg_val": len(urban_val_mem),
            "urban_hardneg_source_breakdown": urban_source_counts,
            "proc_wire_val": len(proc_wire_val),
            "det_pos_val": len(det_pos_val),
            "det_hardneg_val": len(det_hardneg_val),
        },
        "training": {
            "seed": int(args.seed),
            "wireseg": wire_training_summary,
            "dfine_n": {
                "epochs": det_epochs,
                "batch_size": det_bsz,
                "grad_accum": det_accum,
                "effective_batch_size": det_bsz * det_accum,
                "samples_per_epoch": det_samples_per_epoch,
                "loss_start": round(float(det_losses_all[0]), 6),
                "loss_end": round(float(det_losses_all[-1]), 6),
                "all_losses_finite": bool(all(math.isfinite(v) for v in det_losses_all)),
                "epoch_history": det_epoch_history,
            },
        },
        "heldout_validation": {
            "wireseg": best_wire_metrics,
            "dfine_n": best_det_metrics,
        },
        "nasa_goddard_zeroshot_240f": nasa_eval,
        "legacy_g4_reference": legacy_g4_ref,
        "onnx_exports": {
            "mlsd_wireseg": wire_export,
            "dfine_n": dfine_export,
        },
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    write_canonical_json(args.out_json, payload)
    return payload


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local GTX 1650 SUPER (0.00 CU) M-LSD-WireSeg v1.1 + D-FINE-N v1.1 Retrainer"
    )
    p.add_argument("--smoke-cpu", action="store_true", help="Fast CPU-only contract smoke test.")
    p.add_argument("--teias-dir", type=Path, default=DEFAULT_TEIAS_DIR)
    p.add_argument("--teias-zip", type=Path, default=DEFAULT_TEIAS_ZIP)
    p.add_argument("--prepared-manifest", type=Path, default=DEFAULT_PREPARED_MANIFEST)
    p.add_argument("--blob-dir", type=Path, default=DEFAULT_BLOB_DIR)
    p.add_argument("--nasa-video", type=Path, default=DEFAULT_NASA_VIDEO)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--seed", type=int, default=20261126)
    p.add_argument("--wire-epochs", type=int, default=20)
    p.add_argument("--wire-batch-size", type=int, default=4)
    p.add_argument("--wire-grad-accum", type=int, default=2)
    p.add_argument("--wire-samples-per-epoch", type=int, default=192)
    p.add_argument("--det-epochs", type=int, default=40)
    p.add_argument("--det-batch-size", type=int, default=8)
    p.add_argument("--det-grad-accum", type=int, default=1)
    p.add_argument("--det-samples-per-epoch", type=int, default=320)
    p.add_argument(
        "--reuse-wireseg-from-json",
        type=Path,
        default=None,
        help="Optional existing metrics JSON to reuse verified SHA-256 Stage-A WireSeg ONNX & metrics.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()
    run_local_forward_wire_v2(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
