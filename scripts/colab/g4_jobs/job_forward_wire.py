#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Wave-1 G4 Job (0.20 VRAM): M-LSD-WireSeg (clDice) + D-FINE-N (640x640 Split-Stack).

Implements Plan 6C Task 3 & DR-10 §1.1–§1.3 (#329 hardened for live G4):
  - Decoupled 45 deg HFOV narrow-FOV forward horizon split-stack (1280x720 -> 1280x320
    -> Left 640x320 + Right 640x320 stacked vertically into canonical 1x3x640x640 RGB with
    16 px overlap metadata and baked static (u, v) coordinate channels inside the ONNX graph).
  - Live G4 mode ingests REAL TTPLA (Apache-2.0, LabelMe cable + tower polygons rasterized into
    640x640 wire masks & boxes) alongside Apache-2.0 procedural catenary/crane/UAV/bird frames,
    logs exact `real_ttpla_frames_ingested` vs `procedural_frames_ingested`, scopes `license_audit`
    STRICTLY to actually ingested sources, and computes real held-out 20% validation metrics
    (`val_wire_iou`, `val_cldice_score`, `val_det_box_l1`, `val_det_cls_acc`).
  - Exports static Opset-17 FP32 ONNX graphs (zero AdaptiveAvgPool2d, zero dynamic axes):
      * mlsd_wireseg_640_opset17.onnx: rgb_split_nchw [1,3,640,640]
          -> wire_mask_logits [1,1,640,640], line_junction_map [1,5,160,160]
      * dfine_n_640_opset17.onnx: rgb_split_nchw [1,3,640,640]
          -> boxes_xyxy [1,300,4], class_logits [1,300,3]  (Bird, Rogue_UAV, Tower_Crane)
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs.common import (  # noqa: E402
    WAVE_BUDGETS,
    build_standard_argparser,
    configure_torch_vram_fraction,
    query_nvidia_smi_snapshot,
    seed_everything,
    sha256_file,
    validate_job_license_manifest,
    write_canonical_json,
)

FORWARD_HFOV_DEG = 45.0
FORWARD_F_NET_PX = 1545.1
SPLIT_OVERLAP_PX = 16
WIRE_CLASS_NAMES: tuple[str, ...] = ("Bird", "Rogue_UAV", "Tower_Crane")
TTPLA_HF_CANDIDATES: tuple[str, ...] = (
    "R3ab/ttpla_dataset",
    "Francesco/ttpla",
)


def build_forward_license_manifest(
    *,
    ingested_ttpla: bool = False,
    ingested_teias: bool = False,
    ingested_purdue: bool = False,
    include_planned_dry_run: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Scope train_sources strictly to actually ingested corpora (zero phantom claims!)."""
    train_sources: list[dict[str, Any]] = []
    if ingested_ttpla or include_planned_dry_run:
        train_sources.append({"name": "TTPLA", "repo": "R3ab/ttpla_dataset", "license": "Apache-2.0"})
    if ingested_teias:
        train_sources.append({"name": "Mendeley_TEIAS_Powerline_GT", "repo": "twxp8xccsw/9", "license": "CC BY 4.0"})
    if ingested_purdue:
        train_sources.append({"name": "Purdue_UAV", "repo": "purdue-uav-air2air", "license": "BSD-3-Clause"})
    train_sources.append(
        {"name": "VertiRTA_Procedural_Catenary_Horizon", "repo": "internal/vertirta", "license": "Apache-2.0"}
    )
    eval_only_sources: list[dict[str, Any]] = []
    models = [
        {"name": "M-LSD-WireSeg", "repo": "internal/vertirta_mlsd_wireseg", "license": "Apache-2.0"},
        {"name": "D-FINE-N", "repo": "internal/vertirta_dfine_nano", "license": "Apache-2.0"},
    ]
    return train_sources, eval_only_sources, models


def horizon_split_stack_1280x720_to_640x640(
    img_hwc: np.ndarray,
    *,
    roll_rad: float = 0.0,
    horizon_row: int = 360,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Derotate 1280x720 frame by IMU roll, extract 1280x320 horizon band, stack L/R into 640x640."""
    if img_hwc.shape[:2] != (720, 1280):
        raise ValueError(f"Expected 1280x720 input, got {img_hwc.shape}")

    h, w = 720, 1280
    y0 = int(np.clip(horizon_row - 160, 0, h - 320))
    band = img_hwc[y0 : y0 + 320, :, :]

    if abs(roll_rad) > 1e-4:
        c, s = math.cos(-roll_rad), math.sin(-roll_rad)
        yy, xx = np.meshgrid(
            np.arange(320, dtype=np.float32) - 160.0,
            np.arange(1280, dtype=np.float32) - 640.0,
            indexing="ij",
        )
        src_x = np.clip(np.rint(c * xx - s * yy + 640.0).astype(np.int32), 0, w - 1)
        src_y = np.clip(np.rint(s * xx + c * yy + float(y0 + 160)).astype(np.int32), 0, h - 1)
        band = img_hwc[src_y, src_x]

    left_tile = band[:, 0:640, :]
    right_tile = band[:, 640:1280, :]
    stacked = np.concatenate([left_tile, right_tile], axis=0)
    meta = {
        "source_hw": [720, 1280],
        "horizon_y0": y0,
        "overlap_px": SPLIT_OVERLAP_PX,
        "roll_rad": float(roll_rad),
        "f_net_px": FORWARD_F_NET_PX,
    }
    return stacked, meta


def split_stack_to_native_xy(u_stack: float, v_stack: float, horizon_y0: int = 200) -> tuple[float, float]:
    """Exact inverse map from 640x640 split-stack coordinate back to native 1280x720 pixel."""
    if v_stack < 320.0:
        return (float(u_stack), float(horizon_y0) + float(v_stack))
    return (640.0 + float(u_stack), float(horizon_y0) + (float(v_stack) - 320.0))


def _rasterize_ttpla_labelme_pair(
    img_path: Path,
    json_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Convert one real TTPLA (Apache-2.0) image + LabelMe JSON into 640x640 wire mask & tower boxes."""
    from PIL import Image, ImageDraw

    try:
        pil_img = Image.open(img_path).convert("RGB")
        ann = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    orig_w, orig_h = pil_img.size
    if orig_w <= 0 or orig_h <= 0:
        return None
    pil_640 = pil_img.resize((640, 640), Image.Resampling.BILINEAR)
    img_chw = (np.asarray(pil_640, dtype=np.float32) / 255.0).transpose(2, 0, 1)

    mask_pil = Image.new("L", (640, 640), 0)
    draw = ImageDraw.Draw(mask_pil)
    junc_map = np.zeros((5, 160, 160), dtype=np.float32)
    boxes = np.zeros((300, 4), dtype=np.float32)
    cls_ids = np.zeros((300,), dtype=np.int64)
    sx = 640.0 / float(orig_w)
    sy = 640.0 / float(orig_h)

    box_idx = 0
    has_cable = False
    for shape in ann.get("shapes", []):
        lbl = str(shape.get("label", "")).strip().lower()
        pts = shape.get("points", [])
        if not pts:
            continue
        scaled = [(float(p[0]) * sx, float(p[1]) * sy) for p in pts if len(p) >= 2]
        if len(scaled) < 2:
            continue
        if "cable" in lbl or "wire" in lbl or "line" in lbl:
            has_cable = True
            if len(scaled) >= 3:
                draw.polygon(scaled, fill=255, outline=255)
            else:
                draw.line(scaled, fill=255, width=2)
            p0, p1 = scaled[0], scaled[-1]
            mx = int(np.clip(0.5 * (p0[0] + p1[0]) / 4.0, 0, 159))
            my = int(np.clip(0.5 * (p0[1] + p1[1]) / 4.0, 0, 159))
            junc_map[0, my, mx] = 1.0
            junc_map[1, my, mx] = (p0[0] - 0.5 * (p0[0] + p1[0])) / 640.0
            junc_map[2, my, mx] = (p0[1] - 0.5 * (p0[1] + p1[1])) / 640.0
            junc_map[3, my, mx] = (p1[0] - 0.5 * (p0[0] + p1[0])) / 640.0
            junc_map[4, my, mx] = (p1[1] - 0.5 * (p0[1] + p1[1])) / 640.0
        elif "tower" in lbl and box_idx < 300:
            xs = [p[0] for p in scaled]
            ys = [p[1] for p in scaled]
            boxes[box_idx] = [
                max(0.0, min(xs) / 640.0),
                max(0.0, min(ys) / 640.0),
                min(1.0, max(xs) / 640.0),
                min(1.0, max(ys) / 640.0),
            ]
            cls_ids[box_idx] = 2  # Tower_Crane class
            box_idx += 1

    if not has_cable:
        return None
    wire_mask = (np.asarray(mask_pil, dtype=np.float32) / 255.0)[None, :, :]
    return img_chw, wire_mask, junc_map, boxes, cls_ids


def ingest_real_ttpla_apache2(
    work_dir: Path,
    *,
    data_dir: Path | None = None,
    max_samples: int = 320,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    """Attempt real TTPLA (Apache-2.0) ingestion from local --data-dir or public HF/GitHub mirror."""
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    info: dict[str, Any] = {
        "source": "TTPLA",
        "license": "Apache-2.0",
        "origin": None,
        "frames_ingested": 0,
        "fallback_reason": None,
    }

    search_roots: list[Path] = []
    if data_dir is not None and data_dir.is_dir():
        search_roots.append(data_dir)

    if not search_roots:
        # Try HuggingFace datasets streaming/download of public Apache-2.0 TTPLA mirror
        ttpla_dir = work_dir / "ttpla_apache2"
        ttpla_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download

            for repo_id in TTPLA_HF_CANDIDATES:
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        repo_type="dataset",
                        local_dir=str(ttpla_dir),
                        allow_patterns=["*.jpg", "*.png", "*.json", "*LICENSE*"],
                        max_workers=4,
                    )
                    if list(ttpla_dir.rglob("*.json")):
                        search_roots.append(ttpla_dir)
                        info["origin"] = f"hf://{repo_id}"
                        break
                except Exception:
                    continue
        except Exception as exc:
            info["fallback_reason"] = f"hf_unavailable:{type(exc).__name__}"

        # Also try HuggingFace `datasets` parquet-backed TTPLA (zero zip/LabelMe hassle)
        if not search_roots:
            try:
                from datasets import load_dataset
                from PIL import Image, ImageDraw

                for repo_id in ("Francesco/ttpla", "R3ab/ttpla_dataset"):
                    try:
                        ds = load_dataset(repo_id, split="train", streaming=True)
                        for row in ds:
                            if len(records) >= max_samples:
                                break
                            img = row.get("image")
                            objs = row.get("objects") or {}
                            if img is None:
                                continue
                            pil_640 = img.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
                            img_chw = (np.asarray(pil_640, dtype=np.float32) / 255.0).transpose(2, 0, 1)
                            orig_w, orig_h = img.size
                            sx, sy = 640.0 / max(1, orig_w), 640.0 / max(1, orig_h)
                            mask_pil = Image.new("L", (640, 640), 0)
                            draw = ImageDraw.Draw(mask_pil)
                            junc_map = np.zeros((5, 160, 160), dtype=np.float32)
                            boxes = np.zeros((300, 4), dtype=np.float32)
                            cls_ids = np.zeros((300,), dtype=np.int64)
                            bboxes = objs.get("bbox", [])
                            cats = objs.get("category", [])
                            box_idx = 0
                            for bb, cat in zip(bboxes, cats):
                                if len(bb) < 4:
                                    continue
                                x0 = float(bb[0]) * sx
                                y0 = float(bb[1]) * sy
                                bw = float(bb[2]) * sx
                                bh = float(bb[3]) * sy
                                x1, y1 = min(639.0, x0 + bw), min(639.0, y0 + bh)
                                if int(cat) == 0:  # cable category in TTPLA COCO mapping
                                    draw.line([(x0, y0), (x1, y1)], fill=255, width=2)
                                    mx = int(np.clip(0.5 * (x0 + x1) / 4.0, 0, 159))
                                    my = int(np.clip(0.5 * (y0 + y1) / 4.0, 0, 159))
                                    junc_map[0, my, mx] = 1.0
                                elif box_idx < 300:
                                    boxes[box_idx] = [x0 / 640.0, y0 / 640.0, x1 / 640.0, y1 / 640.0]
                                    cls_ids[box_idx] = 2
                                    box_idx += 1
                            wire_mask = (np.asarray(mask_pil, dtype=np.float32) / 255.0)[None, :, :]
                            if wire_mask.max() > 0.0:
                                records.append((img_chw, wire_mask, junc_map, boxes, cls_ids))
                        if records:
                            info["origin"] = f"hf_datasets://{repo_id}"
                            info["frames_ingested"] = len(records)
                            return records, info
                    except Exception:
                        continue
            except Exception as exc:
                info["fallback_reason"] = f"datasets_unavailable:{type(exc).__name__}"

    for root in search_roots:
        for jp in sorted(root.rglob("*.json")):
            if len(records) >= max_samples:
                break
            for ext in (".jpg", ".jpeg", ".png", ".JPG"):
                ip = jp.with_suffix(ext)
                if ip.is_file():
                    rec = _rasterize_ttpla_labelme_pair(ip, jp)
                    if rec is not None:
                        records.append(rec)
                    break

    info["frames_ingested"] = len(records)
    if len(records) == 0 and info["fallback_reason"] is None:
        info["fallback_reason"] = "no_ttpla_pairs_found_using_procedural_apache2_catenary"
    return records, info


def synthesize_catenary_split_batch(
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy Apache-2.0 procedural catenary wire + crane/UAV/bird 640x640 split-stack generator."""
    imgs = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)
    wire_masks = np.zeros((batch_size, 1, 640, 640), dtype=np.float32)
    junc_maps = np.zeros((batch_size, 5, 160, 160), dtype=np.float32)
    target_boxes = np.zeros((batch_size, 300, 4), dtype=np.float32)
    target_cls = np.zeros((batch_size, 300), dtype=np.int64)

    xs = np.linspace(0.0, 1.0, 640, dtype=np.float32)
    for b in range(batch_size):
        sky_grad = np.linspace(0.85, 0.35, 320, dtype=np.float32)[:, None]
        tile_bg = np.stack([sky_grad * 0.72, sky_grad * 0.80, sky_grad * 0.92], axis=0)
        tile_bg = np.broadcast_to(tile_bg, (3, 320, 640)).copy()
        noise = rng.normal(0.0, 0.03, size=(3, 640, 640)).astype(np.float32)
        imgs[b, :, 0:320, :] = tile_bg
        imgs[b, :, 320:640, :] = tile_bg * 0.96
        imgs[b] = np.clip(imgs[b] + noise, 0.0, 1.0)

        for half_idx in (0, 1):
            y_base = float(rng.uniform(70.0, 210.0)) + 320.0 * half_idx
            sag = float(rng.uniform(12.0, 48.0))
            slope = float(rng.uniform(-28.0, 28.0))
            ys = y_base + slope * (xs - 0.5) + sag * (4.0 * (xs - 0.5) ** 2 - 1.0)
            yi = np.clip(np.rint(ys).astype(np.int32), 320 * half_idx + 1, 320 * (half_idx + 1) - 2)
            xi = np.arange(640, dtype=np.int32)
            imgs[b, :, yi, xi] *= 0.25
            wire_masks[b, 0, yi, xi] = 1.0
            wire_masks[b, 0, np.clip(yi + 1, 0, 639), xi] = 0.6

            mx = 80
            my = int(np.clip(yi[320] // 4, 0, 159))
            junc_maps[b, 0, my, mx] = 1.0
            junc_maps[b, 1, my, mx] = -0.5
            junc_maps[b, 2, my, mx] = (float(yi[0] - yi[320])) / 640.0
            junc_maps[b, 3, my, mx] = 0.5
            junc_maps[b, 4, my, mx] = (float(yi[639] - yi[320])) / 640.0

        for obj_i in range(3):
            cx = float(rng.uniform(0.15, 0.85))
            cy = float(rng.uniform(0.15, 0.85))
            bw = float(rng.uniform(0.025, 0.07))
            bh = float(rng.uniform(0.025, 0.08))
            x0, y0 = max(0.0, cx - 0.5 * bw), max(0.0, cy - 0.5 * bh)
            x1, y1 = min(1.0, cx + 0.5 * bw), min(1.0, cy + 0.5 * bh)
            px0, px1 = int(x0 * 639), max(int(x0 * 639) + 2, int(x1 * 639))
            py0, py1 = int(y0 * 639), max(int(y0 * 639) + 2, int(y1 * 639))
            imgs[b, :, py0:py1, px0:px1] = 0.12 + 0.1 * obj_i
            target_boxes[b, obj_i] = [x0, y0, x1, y1]
            target_cls[b, obj_i] = obj_i

    return imgs, wire_masks, junc_maps, target_boxes, target_cls


def _build_torch_models() -> tuple[Any, Any, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SoftClDiceLoss(nn.Module):
        """Differentiable centerline Dice loss (DR-10 §1.3, 3 soft-skeleton iterations)."""

        def __init__(self, iters: int = 3, smooth: float = 1e-4, lambda_cl: float = 0.5) -> None:
            super().__init__()
            self.iters = iters
            self.smooth = smooth
            self.lambda_cl = lambda_cl

        @staticmethod
        def _soft_erode(x: torch.Tensor) -> torch.Tensor:
            p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
            p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
            return torch.minimum(p1, p2)

        @staticmethod
        def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
            return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

        def _soft_open(self, x: torch.Tensor) -> torch.Tensor:
            return self._soft_dilate(self._soft_erode(x))

        def _soft_skel(self, x: torch.Tensor) -> torch.Tensor:
            img1 = self._soft_open(x)
            skel = F.relu(x - img1)
            cur = x
            for _ in range(self.iters):
                cur = self._soft_erode(cur)
                img1 = self._soft_open(cur)
                delta = F.relu(cur - img1)
                skel = skel + F.relu(delta - skel * delta)
            return skel

        def cldice_score(self, prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            skel_p = self._soft_skel(prob)
            skel_t = self._soft_skel(target)
            tprec = ((skel_p * target).sum(dim=(1, 2, 3)) + self.smooth) / (skel_p.sum(dim=(1, 2, 3)) + self.smooth)
            tsens = ((skel_t * prob).sum(dim=(1, 2, 3)) + self.smooth) / (skel_t.sum(dim=(1, 2, 3)) + self.smooth)
            return ((2.0 * tprec * tsens) / (tprec + tsens + self.smooth)).mean()

        def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy_with_logits(logits, target)
            prob = torch.sigmoid(logits)
            inter = (prob * target).sum(dim=(1, 2, 3))
            dice = 1.0 - (
                (2.0 * inter + self.smooth)
                / (prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.smooth)
            ).mean()
            cl_dice = 1.0 - self.cldice_score(prob, target)
            return bce + dice + self.lambda_cl * cl_dice

    class ConvBnAct(nn.Module):
        def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1, p: int = 1) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class MLsdWireSegNet(nn.Module):
        """Compact Apache-2.0 thin-wire + segment head (<3.5M params, baked static uv coords)."""

        def __init__(self) -> None:
            super().__init__()
            u = torch.linspace(-1.0, 1.0, 640).view(1, 1, 1, 640).expand(1, 1, 640, 640)
            v = torch.linspace(-1.0, 1.0, 640).view(1, 1, 640, 1).expand(1, 1, 640, 640)
            self.register_buffer("uv_coords", torch.cat([u, v], dim=1), persistent=True)

            self.enc1 = nn.Sequential(ConvBnAct(5, 32, 3, 2, 1), ConvBnAct(32, 32, 3, 1, 1))
            self.enc2 = nn.Sequential(ConvBnAct(32, 64, 3, 2, 1), ConvBnAct(64, 64, 3, 1, 1))
            self.enc3 = nn.Sequential(ConvBnAct(64, 128, 3, 2, 1), ConvBnAct(128, 128, 3, 1, 1))
            self.enc4 = nn.Sequential(ConvBnAct(128, 192, 3, 2, 1), ConvBnAct(192, 192, 3, 1, 1))
            self.dec3 = ConvBnAct(192 + 128, 96, 3, 1, 1)
            self.dec2 = ConvBnAct(96 + 64, 64, 3, 1, 1)
            self.dec1 = ConvBnAct(64 + 32, 32, 3, 1, 1)
            self.wire_head = nn.Conv2d(32, 1, kernel_size=1)
            self.junc_head = nn.Sequential(ConvBnAct(64, 32, 3, 1, 1), nn.Conv2d(32, 5, kernel_size=1))

        def forward(self, rgb_split_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            coords = self.uv_coords.expand(rgb_split_nchw.shape[0], -1, -1, -1)
            x = torch.cat([rgb_split_nchw, coords], dim=1)
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            e4 = self.enc4(e3)

            d3 = self.dec3(torch.cat([F.interpolate(e4, scale_factor=2.0, mode="bilinear", align_corners=False), e3], dim=1))
            d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2.0, mode="bilinear", align_corners=False), e2], dim=1))
            d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2.0, mode="bilinear", align_corners=False), e1], dim=1))

            wire_mask_logits = F.interpolate(self.wire_head(d1), scale_factor=2.0, mode="bilinear", align_corners=False)
            line_junction_map = self.junc_head(d2)
            return wire_mask_logits, line_junction_map

    class DFineNanoDetector(nn.Module):
        """Compact Apache-2.0 small-object detector (<4.0M params, static 300 queries, zero AdaptiveAvgPool)."""

        def __init__(self, num_queries: int = 300, num_classes: int = 3) -> None:
            super().__init__()
            self.num_queries = num_queries
            self.stem = nn.Sequential(
                ConvBnAct(3, 32, 3, 2, 1),
                ConvBnAct(32, 64, 3, 2, 1),
                ConvBnAct(64, 128, 3, 2, 1),
                ConvBnAct(128, 192, 3, 2, 1),
                ConvBnAct(192, 256, 3, 2, 1),
            )
            self.pool_10 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.neck = ConvBnAct(256, 192, 3, 1, 1)
            self.box_head = nn.Conv2d(192, 3 * 4, kernel_size=1)
            self.cls_head = nn.Conv2d(192, 3 * num_classes, kernel_size=1)

        def forward(self, rgb_split_nchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            feat = self.neck(self.pool_10(self.stem(rgb_split_nchw)))
            bsz = feat.shape[0]
            raw_boxes = torch.sigmoid(self.box_head(feat))
            raw_boxes = raw_boxes.view(bsz, 3, 4, 100).permute(0, 3, 1, 2).reshape(bsz, self.num_queries, 4)
            x0 = torch.minimum(raw_boxes[..., 0], raw_boxes[..., 2])
            y0 = torch.minimum(raw_boxes[..., 1], raw_boxes[..., 3])
            x1 = torch.maximum(raw_boxes[..., 0], raw_boxes[..., 2])
            y1 = torch.maximum(raw_boxes[..., 1], raw_boxes[..., 3])
            boxes_xyxy = torch.stack([x0, y0, x1, y1], dim=-1)
            raw_cls = self.cls_head(feat).view(bsz, 3, 3, 100).permute(0, 3, 1, 2).reshape(bsz, self.num_queries, 3)
            return boxes_xyxy, raw_cls

    return SoftClDiceLoss, MLsdWireSegNet, DFineNanoDetector


def _evaluate_heldout_forward_val(
    wire_net: Any,
    det_net: Any,
    cldice_loss_fn: Any,
    val_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    device: Any,
    *,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Compute real held-out val_wire_iou, val_cldice_score, val_det_box_l1, val_det_cls_acc."""
    import torch
    import torch.nn.functional as F

    wire_net.eval()
    det_net.eval()
    inter_sum = 0.0
    union_sum = 0.0
    cldice_vals: list[float] = []
    box_l1_vals: list[float] = []
    cls_correct = 0
    cls_total = 0

    cur_bsz = max(1, int(batch_size))
    start = 0
    with torch.no_grad():
        while start < len(val_records):
            chunk = val_records[start : start + cur_bsz]
            imgs_t = mask_t = box_t = cls_t = None
            pred_mask_logits = prob = pred_bin = gt_bin = None
            pred_boxes = pred_cls = pred_labels = None
            try:
                imgs_t = torch.from_numpy(np.stack([r[0] for r in chunk], axis=0)).to(device)
                mask_t = torch.from_numpy(np.stack([r[1] for r in chunk], axis=0)).to(device)
                box_t = torch.from_numpy(np.stack([r[3] for r in chunk], axis=0)).to(device)
                cls_t = torch.from_numpy(np.stack([r[4] for r in chunk], axis=0)).to(device)

                pred_mask_logits, _ = wire_net(imgs_t)
                prob = torch.sigmoid(pred_mask_logits.float())
                pred_bin = (prob >= 0.5).float()
                gt_bin = (mask_t >= 0.5).float()
                batch_inter = float((pred_bin * gt_bin).sum().item())
                batch_union = float(((pred_bin + gt_bin) >= 1.0).float().sum().item())
                batch_cldice = float(cldice_loss_fn.cldice_score(prob, gt_bin).item())

                pred_boxes, pred_cls = det_net(imgs_t)
                batch_box_l1 = float(F.l1_loss(pred_boxes[:, :3, :].float(), box_t[:, :3, :]).item())
                pred_labels = pred_cls[:, :3, :].argmax(dim=-1)
                batch_cls_correct = int((pred_labels == cls_t[:, :3]).sum().item())
                batch_cls_total = int(cls_t[:, :3].numel())

                inter_sum += batch_inter
                union_sum += batch_union
                cldice_vals.append(batch_cldice)
                box_l1_vals.append(batch_box_l1)
                cls_correct += batch_cls_correct
                cls_total += batch_cls_total
                del (
                    imgs_t,
                    mask_t,
                    box_t,
                    cls_t,
                    pred_mask_logits,
                    prob,
                    pred_bin,
                    gt_bin,
                    pred_boxes,
                    pred_cls,
                    pred_labels,
                )
                start += len(chunk)
                val_oom_hit = False
            except torch.cuda.OutOfMemoryError as oom_exc:
                if oom_exc.__traceback__ is not None:
                    traceback.clear_frames(oom_exc.__traceback__)
                del (
                    imgs_t,
                    mask_t,
                    box_t,
                    cls_t,
                    pred_mask_logits,
                    prob,
                    pred_bin,
                    gt_bin,
                    pred_boxes,
                    pred_cls,
                    pred_labels,
                    oom_exc,
                )
                if cur_bsz <= 8:
                    raise
                val_oom_hit = True
            if val_oom_hit:
                gc.collect()
                torch.cuda.empty_cache()
                cur_bsz = max(8, cur_bsz // 2)
                continue

    return {
        "val_samples": len(val_records),
        "val_wire_iou": round(inter_sum / max(1.0, union_sum), 6),
        "val_cldice_score": round(float(np.mean(cldice_vals)) if cldice_vals else 0.0, 6),
        "val_det_box_l1": round(float(np.mean(box_l1_vals)) if box_l1_vals else 0.0, 6),
        "val_det_cls_acc": round(float(cls_correct) / float(max(1, cls_total)), 6),
    }


def _export_and_check_onnx(
    model: Any,
    out_path: Path,
    input_names: list[str],
    output_names: list[str],
    sample_shape: tuple[int, ...],
) -> dict[str, Any]:
    import onnx
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.eval().cpu()
    dummy = torch.zeros(sample_shape, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            (dummy,),
            str(out_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamo=False,
        )
    loaded = onnx.load(str(out_path))
    onnx.checker.check_model(loaded)
    op_types = sorted({n.op_type for n in loaded.graph.node})
    forbidden_ops = [op for op in op_types if "Adaptive" in op]
    if forbidden_ops:
        raise RuntimeError(f"Forbidden dynamic ONNX ops found in {out_path.name}: {forbidden_ops}")
    n_params = int(sum(p.numel() for p in model_cpu.parameters()))
    return {
        "filename": out_path.name,
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
        "opset_version": 17,
        "deploy_parameters": n_params,
        "input_names": input_names,
        "output_names": output_names,
        "input_shape": list(sample_shape),
        "onnx_checker_passed": True,
    }


def run_job(args: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed_everything(args.seed)

    rng = np.random.default_rng(args.seed)
    probe = rng.random((720, 1280, 3), dtype=np.float32)
    stacked, split_meta = horizon_split_stack_1280x720_to_640x640(probe, roll_rad=0.0, horizon_row=360)
    assert stacked.shape == (640, 640, 3)
    u_rt, v_rt = split_stack_to_native_xy(123.5, 420.5, horizon_y0=split_meta["horizon_y0"])
    rt_err = abs(u_rt - (640.0 + 123.5)) + abs(v_rt - (split_meta["horizon_y0"] + 100.5))
    assert rt_err < 1e-6

    if args.dry_run:
        ts, es, ms = build_forward_license_manifest(include_planned_dry_run=True)
        license_audit = validate_job_license_manifest("job_forward_wire", ts, es, ms)
        payload = {
            "schema_version": "vertirta.g4_forward_wire_metrics/v4.0.0",
            "mode": "DRY_RUN",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "vram_fraction": float(args.vram_fraction),
            "license_audit": license_audit,
            "split_stack_contract": {
                **split_meta,
                "round_trip_err_px": rt_err,
                "canonical_input_shape": [1, 3, 640, 640],
            },
            "planned_exports": [
                "mlsd_wireseg_640_opset17.onnx",
                "dfine_n_640_opset17.onnx",
            ],
            "wall_clock_s": round(time.perf_counter() - t0, 4),
        }
        write_canonical_json(args.out_json, payload)
        return payload

    import torch
    import torch.nn.functional as F

    vram_cfg = configure_torch_vram_fraction(args.vram_fraction)
    use_cuda = (not args.smoke_cpu) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Data ingestion: real TTPLA (Apache-2.0) in live G4 mode + procedural catenary/crane frames
    real_ttpla_records: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    ttpla_info: dict[str, Any] = {"source": "TTPLA", "frames_ingested": 0, "origin": "skipped_smoke_cpu"}
    if not args.smoke_cpu:
        work_dir = Path(args.blob_dir).parent / "_work_forward_wire"
        real_ttpla_records, ttpla_info = ingest_real_ttpla_apache2(
            work_dir, data_dir=args.data_dir, max_samples=int(args.max_ttpla)
        )

    # Build strictly honest license manifest covering ONLY actually ingested datasets (#329 fix!)
    ts, es, ms = build_forward_license_manifest(
        ingested_ttpla=(len(real_ttpla_records) > 0),
        ingested_teias=False,
        ingested_purdue=False,
    )
    license_audit = validate_job_license_manifest("job_forward_wire", ts, es, ms)

    # Assemble 80/20 train & held-out validation splits
    n_proc_train = 4 if args.smoke_cpu else max(160, int(args.proc_train_samples))
    n_proc_val = 4 if args.smoke_cpu else max(48, int(args.proc_val_samples))
    p_imgs, p_masks, p_juncs, p_boxes, p_cls = synthesize_catenary_split_batch(n_proc_train + n_proc_val, rng)
    proc_all = [
        (p_imgs[i], p_masks[i], p_juncs[i], p_boxes[i], p_cls[i]) for i in range(n_proc_train + n_proc_val)
    ]

    n_ttpla_val = max(1, len(real_ttpla_records) // 5) if len(real_ttpla_records) >= 5 else 0
    train_records = real_ttpla_records[n_ttpla_val:] + proc_all[:n_proc_train]
    val_records = real_ttpla_records[:n_ttpla_val] + proc_all[n_proc_train:]

    SoftClDiceLoss, MLsdWireSegNet, DFineNanoDetector = _build_torch_models()
    wire_net = MLsdWireSegNet().to(device)
    det_net = DFineNanoDetector().to(device)
    cldice_loss_fn = SoftClDiceLoss(iters=3, lambda_cl=0.5)

    opt_wire = torch.optim.AdamW(wire_net.parameters(), lr=3e-4, weight_decay=1e-4)
    opt_det = torch.optim.AdamW(det_net.parameters(), lr=3e-4, weight_decay=1e-4)

    requested_batch_size = 2 if args.smoke_cpu else min(int(args.batch_size), max(2, len(train_records)))
    batch_size = requested_batch_size
    steps = 2 if args.smoke_cpu else int(args.train_steps)
    amp_enabled = use_cuda
    sched_wire = torch.optim.lr_scheduler.CosineAnnealingLR(opt_wire, T_max=max(1, steps), eta_min=1e-5)
    sched_det = torch.optim.lr_scheduler.CosineAnnealingLR(opt_det, T_max=max(1, steps), eta_min=1e-5)

    wire_losses: list[float] = []
    det_losses: list[float] = []
    oom_fallback_triggered = False
    wire_net.train()
    det_net.train()

    step_idx = 0
    while step_idx < steps:
        idxs = rng.integers(0, len(train_records), size=batch_size)
        imgs_t = mask_t = junc_t = box_t = cls_t = None
        pred_mask = pred_junc = loss_w = pred_boxes = pred_cls = loss_d = None
        try:
            imgs_t = torch.from_numpy(np.stack([train_records[i][0] for i in idxs], axis=0)).to(device, non_blocking=use_cuda)
            mask_t = torch.from_numpy(np.stack([train_records[i][1] for i in idxs], axis=0)).to(device, non_blocking=use_cuda)
            junc_t = torch.from_numpy(np.stack([train_records[i][2] for i in idxs], axis=0)).to(device, non_blocking=use_cuda)
            box_t = torch.from_numpy(np.stack([train_records[i][3] for i in idxs], axis=0)).to(device, non_blocking=use_cuda)
            cls_t = torch.from_numpy(np.stack([train_records[i][4] for i in idxs], axis=0)).to(device, non_blocking=use_cuda)

            opt_wire.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                pred_mask, pred_junc = wire_net(imgs_t)
                loss_w = cldice_loss_fn(pred_mask.float(), mask_t) + 0.25 * F.mse_loss(pred_junc.float(), junc_t)
            loss_w.backward()
            w_val = float(loss_w.detach().cpu().item())

            opt_det.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                pred_boxes, pred_cls = det_net(imgs_t)
                loss_d = F.smooth_l1_loss(pred_boxes[:, :3, :].float(), box_t[:, :3, :]) + F.cross_entropy(
                    pred_cls[:, :3, :].float().reshape(-1, 3), cls_t[:, :3].reshape(-1)
                )
            loss_d.backward()
            d_val = float(loss_d.detach().cpu().item())

            opt_wire.step()
            opt_det.step()
            sched_wire.step()
            sched_det.step()
            wire_losses.append(w_val)
            det_losses.append(d_val)
            del imgs_t, mask_t, junc_t, box_t, cls_t, pred_mask, pred_junc, loss_w, pred_boxes, pred_cls, loss_d
            step_idx += 1
            train_oom_hit = False
        except torch.cuda.OutOfMemoryError as oom_exc:
            if oom_exc.__traceback__ is not None:
                traceback.clear_frames(oom_exc.__traceback__)
            del imgs_t, mask_t, junc_t, box_t, cls_t, pred_mask, pred_junc, loss_w, pred_boxes, pred_cls, loss_d, oom_exc
            opt_wire.zero_grad(set_to_none=True)
            opt_det.zero_grad(set_to_none=True)
            if batch_size <= 32:
                raise
            train_oom_hit = True
        if train_oom_hit:
            gc.collect()
            torch.cuda.empty_cache()
            batch_size = 32
            oom_fallback_triggered = True
            continue

    val_metrics = _evaluate_heldout_forward_val(
        wire_net, det_net, cldice_loss_fn, val_records, device, batch_size=max(2, min(32, batch_size))
    )

    peak_cuda_mib = (
        round(float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0), 2)
        if use_cuda
        else 0.0
    )

    blob_dir = Path(args.blob_dir)
    mlsd_meta = _export_and_check_onnx(
        wire_net,
        blob_dir / "mlsd_wireseg_640_opset17.onnx",
        input_names=["rgb_split_nchw"],
        output_names=["wire_mask_logits", "line_junction_map"],
        sample_shape=(1, 3, 640, 640),
    )
    dfine_meta = _export_and_check_onnx(
        det_net,
        blob_dir / "dfine_n_640_opset17.onnx",
        input_names=["rgb_split_nchw"],
        output_names=["boxes_xyxy", "class_logits"],
        sample_shape=(1, 3, 640, 640),
    )

    payload = {
        "schema_version": "vertirta.g4_forward_wire_metrics/v4.0.0",
        "mode": "SMOKE_CPU_ONLY" if args.smoke_cpu else "LIVE_G4_CUDA_BF16",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "device": str(device),
            "vram_config": vram_cfg,
            "peak_cuda_allocated_mib": peak_cuda_mib,
            "nvidia_smi": query_nvidia_smi_snapshot(),
        },
        "license_audit": license_audit,
        "dataset_ingestion": {
            "real_ttpla_frames_ingested": len(real_ttpla_records),
            "procedural_frames_ingested": len(proc_all),
            "train_split_size": len(train_records),
            "val_split_size": len(val_records),
            "ttpla_details": ttpla_info,
        },
        "split_stack_contract": {
            **split_meta,
            "round_trip_err_px": rt_err,
            "canonical_input_shape": [1, 3, 640, 640],
        },
        "training": {
            "requested_batch_size": requested_batch_size,
            "batch_size": batch_size,
            "oom_fallback_triggered": oom_fallback_triggered,
            "lr_scheduler": "CosineAnnealingLR",
            "steps": steps,
            "cldice_iters": 3,
            "wire_loss_start": round(wire_losses[0], 6),
            "wire_loss_end": round(wire_losses[-1], 6),
            "det_loss_start": round(det_losses[0], 6),
            "det_loss_end": round(det_losses[-1], 6),
            "all_losses_finite": bool(all(math.isfinite(v) for v in wire_losses + det_losses)),
        },
        "heldout_validation": val_metrics,
        "onnx_exports": {
            "mlsd_wireseg": mlsd_meta,
            "dfine_n": dfine_meta,
        },
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    write_canonical_json(args.out_json, payload)
    return payload


def main() -> int:
    parser = build_standard_argparser(
        "job_forward_wire",
        default_vram_fraction=WAVE_BUDGETS["wave1"]["forward_wire"],
        description="Wave-1 G4 Job: M-LSD-WireSeg (clDice) + D-FINE-N (640x640 split-stack).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="G4 BF16 batch size (default: 64, OOM fallback 32).")
    parser.add_argument("--train-steps", type=int, default=400, help="G4 optimization steps (default: 400).")
    parser.add_argument("--max-ttpla", type=int, default=800, help="Max real TTPLA Apache-2.0 frames to ingest.")
    parser.add_argument("--proc-train-samples", type=int, default=640)
    parser.add_argument("--proc-val-samples", type=int, default=160)
    args = parser.parse_args()
    run_job(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
