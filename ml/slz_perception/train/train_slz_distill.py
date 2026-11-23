# SPDX-License-Identifier: Apache-2.0
"""Colab L4 (24 GB) / G4 (96 GB) BF16 AMP 2-stage distillation trainer for PIDNet-S SLZ.

Strict locks enforced at startup:
  1) Calls `verify_training_manifest` unconditionally before loading any data or teachers.
  2) Teachers are strictly Apache-2.0: `DINOv2-ViT-L/14` (518x518 -> 37x37x1024) and
     `Depth-Anything-V2-Small` (ViT-S; ViT-B/L/G CC-BY-NC strictly banned).
  3) Stage 1 (G4 80 ep / L4 configurable ep, effective bs=32, AdamW warmup + cosine):
       60% real permissive (FLAIR-1 / OpenEarthMap-41 / FloodNet-Supervised-v1.0)
     + 40% synthetic permissive (SkyScenes / Gazebo_Istanbul_8K).
  4) Stage 2 (G4 20 ep / L4 configurable ep, effective bs=32, AdamW poly 0.9):
       adds AnomalyMix Outlier Exposure (label=254) + Free-Energy margin loss
       (m_in=-23.0, m_out=-5.0, T_ood=1.0).
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
    lambda_fsr: float = 0.5
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
    """Stage-2 Free-Energy OOD outlier exposure fine-tuning hyperparameters (v3.5.1 5-class calibrated)."""

    epochs: int = 20
    batch_size: int = 32
    lr_start: float = 2e-5
    lr_min: float = 5e-6
    poly_power: float = 0.9
    weight_decay: float = 1e-2
    amp_bf16: bool = True
    lambda_bdry: float = 0.4
    lambda_dino: float = 0.2
    lambda_geom: float = 0.2
    lambda_fsr: float = 0.8
    lambda_ood: float = 0.02
    m_in: float = -8.0
    m_out: float = -2.5
    anomaly_mix_prob: float = 0.15
    max_val_fsr_gate: float = 0.08
    source_sampling_weights: dict[str, float] = field(
        default_factory=lambda: {
            "FLAIR-1": 0.22,
            "OpenEarthMap-41": 0.18,
            "FloodNet-Supervised-v1.0": 0.15,
            "SkyScenes": 0.20,
            "Gazebo_Istanbul_8K": 0.25,
        }
    )


@dataclass(frozen=True)
class L4TrainProfile:
    """Colab L4 (24 GB Ada Lovelace, BF16 AMP) training profile (v3.5.1 FSR-gated)."""

    device: str = "cuda"
    batch_size: int = 16
    grad_accum: int = 2
    stage1_epochs: int = 24
    stage1_warmup_epochs: int = 3
    stage1_lr_start: float = 8e-4
    stage1_lr_min: float = 1e-5
    stage1_lambda_fsr: float = 0.5
    stage2_epochs: int = 10
    stage2_lr_start: float = 2e-5
    stage2_lr_min: float = 5e-6
    stage2_lambda_ood: float = 0.02
    stage2_lambda_fsr: float = 0.8
    max_val_fsr_gate: float = 0.08
    anomaly_mix_prob: float = 0.15
    amp_bf16: bool = True
    seed: int = 20261123


def build_l4_training_manifest(
    oem_regions: tuple[str, ...],
    active_sources: Iterable[str],
    *,
    include_aeroscapes_eval: bool = True,
) -> list[DatasetEntry]:
    """Build a strict zero-ShareAlike manifest covering only active L4 sources + teachers."""
    source_set = {s.strip() for s in active_sources}
    entries: list[DatasetEntry] = []

    if "FLAIR-1" in source_set:
        entries.append(
            DatasetEntry(name="FLAIR-1", split="train", license="etalab-2.0", uses_osm_odbl=False)
        )
    if "OpenEarthMap" in source_set or "OpenEarthMap-41" in source_set:
        entries.append(
            DatasetEntry(
                name="OpenEarthMap",
                split="train",
                license="CC BY 4.0 + DL-DE-BY-2.0",
                uses_osm_odbl=False,
                regions=oem_regions,
            )
        )
    if "FloodNet-Supervised-v1.0" in source_set:
        entries.append(
            DatasetEntry(
                name="FloodNet-Supervised-v1.0",
                split="train",
                license="CDLA-Permissive-1.0",
                uses_osm_odbl=False,
            )
        )
    if "SkyScenes" in source_set:
        entries.append(
            DatasetEntry(name="SkyScenes", split="train", license="MIT", uses_osm_odbl=False)
        )
    if "Gazebo_Istanbul_8K" in source_set:
        entries.append(
            DatasetEntry(
                name="Gazebo_Istanbul_8K",
                split="train",
                license="Apache-2.0",
                uses_osm_odbl=False,
            )
        )

    entries.extend(
        [
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
    )
    if include_aeroscapes_eval:
        entries.append(
            DatasetEntry(
                name="AeroScapes",
                split="zero_shot_eval",
                license="CC BY-SA 4.0",
                uses_osm_odbl=False,
                notes="Quarantined zero-shot evaluation only; never trained or calibrated.",
            )
        )
    return entries


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
        gen_device = generator.device if generator is not None else images.device
        color_shift = (
            torch.empty((3, 1, 1), device=gen_device, dtype=torch.float32)
            .uniform_(-0.5, 0.5, generator=generator)
            .to(device=images.device, dtype=images.dtype)
        )
        out_img[b, :, y0 : y0 + ph, x0 : x0 + pw] = (patch.flip(-1) + color_shift).clamp(-3.0, 3.0)
        out_lbl[b, y0 : y0 + ph, x0 : x0 + pw] = int(ood_index)

    return out_img, out_lbl


def compute_boundary_target(
    labels: torch.Tensor,
    *,
    out_hw: tuple[int, int] = (64, 64),
    ignore_index: int = 255,
    ood_index: int = 254,
) -> torch.Tensor:
    """Derive binary semantic boundary map (B, 1, H_out, W_out) from 4-neighbor class transitions."""
    lbl = labels.long()
    if lbl.ndim == 2:
        lbl = lbl.unsqueeze(0)
    valid = (lbl != ignore_index) & (lbl != ood_index)

    edge = torch.zeros_like(valid)
    edge[:, :, 1:] |= (lbl[:, :, 1:] != lbl[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, :-1] |= (lbl[:, :, :-1] != lbl[:, :, 1:]) & valid[:, :, :-1] & valid[:, :, 1:]
    edge[:, 1:, :] |= (lbl[:, 1:, :] != lbl[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    edge[:, :-1, :] |= (lbl[:, :-1, :] != lbl[:, 1:, :]) & valid[:, :-1, :] & valid[:, 1:, :]

    edge_f = edge.float().unsqueeze(1)
    if edge_f.shape[2:] != out_hw:
        edge_f = F.adaptive_max_pool2d(edge_f, output_size=out_hw)
    return edge_f.clamp(0.0, 1.0)


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
    valid_geom_mask: torch.Tensor | None = None,
    use_bf16_amp: bool = False,
    grad_accum_steps: int = 1,
    zero_grad: bool = True,
    step_optimizer: bool = True,
) -> float:
    """Execute one forward + backward (+ optional optimizer step) under optional BF16 AMP."""
    model.train()
    if zero_grad:
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
            valid_geom_mask=valid_geom_mask,
        )
    scaled_loss = loss / float(max(1, grad_accum_steps))
    scaled_loss.backward()
    if step_optimizer:
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
    simplify: bool = False,
    onnx_name: str = "pidnet_slz_512.onnx",
) -> tuple[OodCalibration, Path]:
    """Run post-hoc OOD calibration and export static 1x3x512x512 ONNX dropping dino_proj."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cal = calibrate_temperature_and_tau(val_logits, val_labels, val_bdry_prob)
    cal.save_json(out_root / "ood_calibration.json")
    deploy = PIDNetSlzDeployWrapper(model).eval()
    onnx_path = export_static_onnx(deploy, out_root / onnx_name, opset=opset, simplify=simplify)
    return cal, onnx_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser supporting both Colab L4 (24 GB) and G4 (96 GB) BF16 AMP profiles."""
    parser = argparse.ArgumentParser(
        description="VertiRTA 2-Stage Colab L4/G4 PIDNet-S Distillation Entrypoint (License-Gated)."
    )
    parser.add_argument(
        "--oem-attribution-html",
        type=Path,
        required=True,
        help="Path to official OpenEarthMap attribution.html (CC BY 4.0 + DL-DE-BY-2.0 whitelist).",
    )
    parser.add_argument(
        "--expect-oem-total",
        type=int,
        default=100,
        help="Expected row count in attribution.html (pass 0 to allow live site row updates).",
    )
    parser.add_argument("--profile", choices=("l4", "g4"), default="l4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--bf16", dest="bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--out-dir", type=Path, default=Path("results/slz_distill"))
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    expect_total: int | None = None if args.expect_oem_total <= 0 else int(args.expect_oem_total)
    oem_regions = tuple(
        sorted(parse_oem_permissive_whitelist(args.oem_attribution_html, expect_total=expect_total))
    )
    manifest = build_locked_training_manifest(oem_regions)
    report = validate_training_startup_license(
        manifest, args.oem_attribution_html, expect_oem_total=expect_total
    )
    print(
        f"[VertiRTA-Distill] License gate PASSED: {len(report.approved_train_entries)} train sources, "
        f"{len(report.oem_whitelisted_regions)} OEM permissive regions verified zero-ShareAlike."
    )
    if args.verify_only:
        return 0

    if args.profile == "g4":
        s1 = Stage1Config()
        s2 = Stage2Config()
        print(
            f"[VertiRTA-Distill] G4 Profile ready: Stage 1 ({s1.epochs} ep, bs={s1.batch_size}, "
            f"BF16={s1.amp_bf16}) -> Stage 2 ({s2.epochs} ep, OE m_in={s2.m_in}, m_out={s2.m_out}). "
            f"Output: {args.out_dir}"
        )
    else:
        prof = L4TrainProfile(
            device=args.device,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            stage1_epochs=args.stage1_epochs,
            stage2_epochs=args.stage2_epochs,
            amp_bf16=args.bf16,
        )
        eff_bs = prof.batch_size * prof.grad_accum
        print(
            f"[VertiRTA-Distill] L4 Profile ready ({prof.device}, BF16={prof.amp_bf16}, "
            f"bs={prof.batch_size}x{prof.grad_accum}={eff_bs}): Stage 1 ({prof.stage1_epochs} ep) "
            f"-> Stage 2 ({prof.stage2_epochs} ep, AnomalyMix p={prof.anomaly_mix_prob}). "
            f"Output: {args.out_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
