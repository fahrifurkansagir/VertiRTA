# SPDX-License-Identifier: Apache-2.0
"""Colab G4 (RTX PRO 6000 Blackwell 96 GB, BF16 AMP) 2-stage distillation trainer for PIDNet-S SLZ.

Strict locks enforced at startup:
  1) Calls `verify_training_manifest` unconditionally before loading any data or teachers.
  2) Teachers are strictly Apache-2.0: `DINOv2-ViT-L/14` (518x518 -> 37x37x1024) and
     `Depth-Anything-V2-Small` (ViT-S; ViT-B/L/G CC-BY-NC strictly banned).
  3) Stage 1 (80 epochs, bs=32, AdamW lr 3e-4 -> 1e-6, 5-ep warmup + cosine):
       60% real permissive (FLAIR-1 25% / OpenEarthMap-41 20% / FloodNet 15%)
     + 40% synthetic (SkyScenes 20% / Gazebo_Istanbul_8K 20%).
  4) Stage 2 (20 epochs, bs=32, AdamW lr 1e-5 -> 1e-6, poly 0.9):
       55% real (+10% AnomalyMix OE) + 45% synthetic (Gazebo-8K OOD 25% + SkyScenes 20%),
       adding label=254 Free-Energy margin loss (m_in=-23.0, m_out=-5.0).
  5) Post-training: freezes weights, calibrates `ood_calibration.json` (T_ood=1.0, kappa_b=1.5),
     and exports static 1x3x512x512 ONNX Opset 17+ dropping `dino_proj`.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.slz_perception.data.license_gate import (
    DatasetEntry,
    ManifestReport,
    parse_oem_permissive_whitelist,
    verify_training_manifest,
)
from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain
from ml.slz_perception.train.calibrate_ood import OodCalibration, calibrate_temperature_and_tau
from ml.slz_perception.train.distill_losses import SlzDistillCriterion
from ml.slz_perception.train.export_onnx_trt import export_static_onnx


@dataclass(frozen=True)
class Stage1Config:
    """Stage-1 foundation distillation & segmentation hyperparameters (DR-7 Table B.2)."""

    epochs: int = 80
    warmup_epochs: int = 5
    batch_size: int = 32
    lr_start: float = 3e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-2
    amp_bf16: bool = True
    lambda_bdry: float = 0.4
    lambda_dino: float = 0.5
    lambda_geom: float = 0.3
    lambda_ood: float = 0.0
    source_sampling_weights: dict[str, float] = field(
        default_factory=lambda: {
            "FLAIR-1": 0.25,
            "OpenEarthMap-41": 0.20,
            "FloodNet-Supervised-v1.0": 0.15,
            "SkyScenes": 0.20,
            "Gazebo_Istanbul_8K": 0.20,
        }
    )


@dataclass(frozen=True)
class Stage2Config:
    """Stage-2 Free-Energy OOD outlier exposure fine-tuning hyperparameters (DR-7 Table B.2)."""

    epochs: int = 20
    batch_size: int = 32
    lr_start: float = 1e-5
    lr_min: float = 1e-6
    poly_power: float = 0.9
    weight_decay: float = 1e-2
    amp_bf16: bool = True
    lambda_bdry: float = 0.4
    lambda_dino: float = 0.2
    lambda_geom: float = 0.2
    lambda_ood: float = 0.1
    m_in: float = -23.0
    m_out: float = -5.0
    anomaly_mix_prob: float = 0.10
    source_sampling_weights: dict[str, float] = field(
        default_factory=lambda: {
            "FLAIR-1": 0.22,
            "OpenEarthMap-41": 0.18,
            "FloodNet-Supervised-v1.0": 0.15,
            "SkyScenes": 0.20,
            "Gazebo_Istanbul_8K": 0.25,
        }
    )


def build_locked_training_manifest(
    oem_regions: tuple[str, ...],
) -> list[DatasetEntry]:
    """Construct the canonical zero-ShareAlike 5-source + Apache-2.0 teacher manifest."""
    return [
        DatasetEntry(name="FLAIR-1", split="train", license="etalab-2.0", uses_osm_odbl=False),
        DatasetEntry(
            name="OpenEarthMap",
            split="train",
            license="CC BY 4.0 + DL-DE-BY-2.0",
            uses_osm_odbl=False,
            regions=oem_regions,
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
        DatasetEntry(
            name="DINOv2-ViT-L/14",
            split="teacher",
            license="Apache-2.0",
            uses_osm_odbl=False,
        ),
        DatasetEntry(
            name="Depth-Anything-V2-Small",
            split="teacher",
            license="Apache-2.0",
            uses_osm_odbl=False,
        ),
        DatasetEntry(name="PIDNet-S", split="student", license="MIT", uses_osm_odbl=False),
    ]


def validate_training_startup_license(
    entries: Iterable[DatasetEntry],
    oem_attribution_html: Path | str | None,
    *,
    expect_oem_total: int | None = 100,
) -> ManifestReport:
    """Mandatory startup gate invoked before loading any dataset or starting training."""
    return verify_training_manifest(
        entries,
        oem_attribution_html=oem_attribution_html,
        expect_oem_total=expect_oem_total,
    )


def apply_anomaly_mix(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    prob: float = 0.10,
    ood_index: int = 254,
    min_frac: float = 0.08,
    max_frac: float = 0.25,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paste synthetic foreign-texture OOD patches (label=254) for Stage-2 Outlier Exposure."""
    if prob <= 0.0 or images.shape[0] == 0:
        return images, labels

    out_img = images.clone()
    out_lbl = labels.clone()
    bsz, _, h, w = out_img.shape

    for b in range(bsz):
        u = float(torch.rand((), generator=generator).item())
        if u >= prob:
            continue
        ph = max(4, int(h * (min_frac + (max_frac - min_frac) * float(torch.rand((), generator=generator).item()))))
        pw = max(4, int(w * (min_frac + (max_frac - min_frac) * float(torch.rand((), generator=generator).item()))))
        y0 = int(torch.randint(0, max(1, h - ph + 1), (1,), generator=generator).item())
        x0 = int(torch.randint(0, max(1, w - pw + 1), (1,), generator=generator).item())

        # Cross-image random color/contrast-shifted patch simulates unknown rooftop clutter/tarp
        src_idx = (b + 1) % bsz
        patch = images[src_idx, :, y0 : y0 + ph, x0 : x0 + pw]
        color_shift = torch.empty((3, 1, 1), device=images.device, dtype=images.dtype).uniform_(
            -0.5, 0.5, generator=generator
        )
        out_img[b, :, y0 : y0 + ph, x0 : x0 + pw] = (patch.flip(-1) + color_shift).clamp(-3.0, 3.0)
        out_lbl[b, y0 : y0 + ph, x0 : x0 + pw] = int(ood_index)

    return out_img, out_lbl


def compute_stage1_lr(epoch: int, cfg: Stage1Config) -> float:
    """5-epoch linear warmup followed by cosine decay from lr_start to lr_min."""
    if epoch < cfg.warmup_epochs:
        return cfg.lr_start * float(epoch + 1) / float(max(1, cfg.warmup_epochs))
    progress = float(epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
    progress = min(max(progress, 0.0), 1.0)
    return cfg.lr_min + 0.5 * (cfg.lr_start - cfg.lr_min) * (1.0 + math.cos(math.pi * progress))


def compute_stage2_lr(epoch: int, cfg: Stage2Config) -> float:
    """Polynomial decay (power=0.9) from lr_start to lr_min."""
    progress = min(max(float(epoch) / float(max(1, cfg.epochs)), 0.0), 1.0)
    return cfg.lr_min + (cfg.lr_start - cfg.lr_min) * ((1.0 - progress) ** cfg.poly_power)


def pseudo_normals_from_inv_depth(inv_depth: torch.Tensor) -> torch.Tensor:
    """Derive unit pseudo-normals (B, 3, H, W) via finite differences on DAv2-Small output."""
    d = inv_depth.float()
    if d.ndim == 3:
        d = d.unsqueeze(1)
    dz_dx = F.pad(d[:, :, :, 2:] - d[:, :, :, :-2], (1, 1, 0, 0), mode="replicate") * 0.5
    dz_dy = F.pad(d[:, :, 2:, :] - d[:, :, :-2, :], (0, 0, 1, 1), mode="replicate") * 0.5
    ones = torch.ones_like(d)
    normals = torch.cat([-dz_dx, -dz_dy, ones], dim=1)
    return F.normalize(normals, p=2.0, dim=1, eps=1e-6)


def train_one_step(
    model: PIDNetSlzTrain,
    criterion: SlzDistillCriterion,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_normals: torch.Tensor | None = None,
    target_bdry: torch.Tensor | None = None,
    teacher_dino: torch.Tensor | None = None,
    use_bf16_amp: bool = False,
) -> float:
    """Execute one forward + backward + optimizer step under optional BF16 AMP."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device_type = "cuda" if images.is_cuda else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_bf16_amp):
        out = model(images)
        loss = criterion(
            out.seg_logits,
            out.surf_normals,
            out.bdry_prob,
            out.dino_feats,
            labels,
            target_normals=target_normals,
            target_bdry=target_bdry,
            teacher_dino=teacher_dino,
        )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().item())


def finalize_and_export(
    model: PIDNetSlzTrain,
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    val_bdry_prob: torch.Tensor,
    out_dir: Path | str,
    *,
    opset: int = 17,
) -> tuple[OodCalibration, Path]:
    """Run post-hoc OOD calibration and export static 1x3x512x512 ONNX dropping dino_proj."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cal = calibrate_temperature_and_tau(val_logits, val_labels, val_bdry_prob)
    cal.save_json(out_root / "ood_calibration.json")
    deploy = PIDNetSlzDeployWrapper(model).eval()
    onnx_path = export_static_onnx(deploy, out_root / "pidnet_slz_512.onnx", opset=opset)
    return cal, onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VertiRTA 2-Stage Colab G4 PIDNet-S Distillation Entrypoint (License-Gated)."
    )
    parser.add_argument(
        "--oem-attribution-html",
        type=Path,
        required=True,
        help="Path to official OpenEarthMap attribution.html (100 entries -> 41 permissive).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/slz_distill"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    oem_regions = tuple(sorted(parse_oem_permissive_whitelist(args.oem_attribution_html, expect_total=100)))
    manifest = build_locked_training_manifest(oem_regions)
    report = validate_training_startup_license(manifest, args.oem_attribution_html, expect_oem_total=100)
    print(
        f"[VertiRTA-Distill] License gate PASSED: {len(report.approved_train_entries)} train sources, "
        f"{len(report.oem_whitelisted_regions)}/100 OEM regions verified zero-ShareAlike."
    )
    if args.verify_only:
        return

    s1 = Stage1Config()
    s2 = Stage2Config()
    print(
        f"[VertiRTA-Distill] Ready for Stage 1 ({s1.epochs} ep, bs={s1.batch_size}, BF16={s1.amp_bf16}) "
        f"-> Stage 2 ({s2.epochs} ep, OE m_in={s2.m_in}, m_out={s2.m_out}). Output: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
