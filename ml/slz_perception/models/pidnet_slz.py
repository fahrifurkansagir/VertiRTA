# SPDX-License-Identifier: Apache-2.0
"""Static-PAPPM PIDNet-S (MIT architecture) for 512x512 VertiRTA Active Visual SLZ.

Key design locks (DR-7 Section B.1):
  - ZERO torch.nn.AdaptiveAvgPool2d anywhere in the module graph (eliminates TensorRT
    data-dependent shape / DDS subgraph splits).
  - StaticPAPPM replaces dynamic adaptive pooling with fixed AvgPool2d kernels plus
    global mean reduction (`mean(dim=(2, 3), keepdim=True)`).
  - Three branches: Proportional (P, 64x64 detail), Integral (I, 8x8 global context),
    Derivative (D, 64x64 boundary gate) merged via Bag Fusion.
  - Deploy heads:
      * seg_logits:   (B, 5, 512, 512)
      * surf_normals: (B, 3, 512, 512), unit L2-normalized along dim=1
      * bdry_prob:    (B, 1, 64, 64), sigmoid probability in [0, 1]
  - Train-only head (`dino_proj`): projects Bag features aligned to 37x37 into 1024-dim
    DINOv2-ViT-L/14 dense patch space; completely omitted by `PIDNetSlzDeployWrapper`.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PIDNetTrainOutput(NamedTuple):
    """Structured forward output for `PIDNetSlzTrain`."""

    seg_logits: torch.Tensor
    surf_normals: torch.Tensor
    bdry_prob: torch.Tensor
    dino_feats: torch.Tensor


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with optional projection shortcut."""

    expansion: int = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.no_relu = no_relu

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + identity
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    """PIDNet expansion=2 Bottleneck block."""

    expansion: int = 2

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.no_relu = no_relu

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) + identity
        return out if self.no_relu else self.relu(out)


class PagFM(nn.Module):
    """Pixel-attention-guided fusion module connecting I-branch context into P-branch."""

    def __init__(self, in_ch: int, mid_ch: int, scale_factor: float) -> None:
        super().__init__()
        self.scale_factor = float(scale_factor)
        self.f_x = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
        )

    def forward(self, x_p: torch.Tensor, y_i: torch.Tensor) -> torch.Tensor:
        y_up = F.interpolate(
            y_i, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        )
        q_p = self.f_x(x_p)
        q_i = self.f_y(y_up)
        sim_map = torch.sigmoid(torch.sum(q_p * q_i, dim=1, keepdim=True))
        return (1.0 - sim_map) * x_p + sim_map * y_up


class BagFusion(nn.Module):
    """Boundary-attention-guided fusion combining P, I, and D branches."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1.0 - edge_att) * i)


class StaticPAPPM(nn.Module):
    """Parallel Aggregation Pyramid Pooling Module with ZERO AdaptiveAvgPool2d.

    Locked for static 8x8 stride-64 feature maps (from 512x512 input):
      - scale0: 1x1 identity scale (8x8)
      - scale1: AvgPool2d(k=2, s=2) -> 4x4
      - scale2: AvgPool2d(k=4, s=4) -> 2x2
      - scale3: AvgPool2d(k=3, s=2, p=1) -> 4x4 overlapping context
      - scale4: global mean(dim=(2, 3), keepdim=True) -> 1x1
    """

    def __init__(self, in_ch: int, branch_ch: int, out_ch: int) -> None:
        super().__init__()
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
        )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
        )
        self.scale4_proj = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, branch_ch, kernel_size=1, bias=False),
        )

        self.process1 = nn.Sequential(
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=False),
        )

        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_ch * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_ch * 5, out_ch, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.scale0(x)
        x1 = self.process1(
            F.interpolate(self.scale1(x), scale_factor=2.0, mode="bilinear", align_corners=False)
            + x0
        )
        x2 = self.process2(
            F.interpolate(self.scale2(x), scale_factor=4.0, mode="bilinear", align_corners=False)
            + x0
        )
        x3 = self.process3(
            F.interpolate(self.scale3(x), scale_factor=2.0, mode="bilinear", align_corners=False)
            + x0
        )
        x_global = self.scale4_proj(x.mean(dim=(2, 3), keepdim=True))
        x4 = self.process4(x_global + x0)

        cat = torch.cat([x0, x1, x2, x3, x4], dim=1)
        return self.compression(cat) + self.shortcut(x)


class SegmentHead(nn.Module):
    """PIDNet prediction head (BN -> ReLU -> 3x3 -> BN -> ReLU -> 1x1)."""

    def __init__(self, in_ch: int, inter_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, inter_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _make_basic_layer(
    in_ch: int,
    out_ch: int,
    blocks: int,
    stride: int = 1,
    no_relu_last: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        BasicBlock(in_ch, out_ch, stride=stride, no_relu=(blocks == 1 and no_relu_last))
    ]
    for idx in range(1, blocks):
        layers.append(
            BasicBlock(out_ch, out_ch, stride=1, no_relu=(idx == blocks - 1 and no_relu_last))
        )
    return nn.Sequential(*layers)


def _make_bottleneck_layer(
    in_ch: int,
    mid_ch: int,
    blocks: int,
    stride: int = 1,
    no_relu_last: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        Bottleneck(in_ch, mid_ch, stride=stride, no_relu=(blocks == 1 and no_relu_last))
    ]
    out_ch = mid_ch * Bottleneck.expansion
    for idx in range(1, blocks):
        layers.append(
            Bottleneck(out_ch, mid_ch, stride=1, no_relu=(idx == blocks - 1 and no_relu_last))
        )
    return nn.Sequential(*layers)


class PIDNetSlzBackbone(nn.Module):
    """Three-branch PIDNet-S backbone (~7.6M params) with StaticPAPPM and BagFusion."""

    def __init__(
        self,
        planes: int = 32,
        ppm_planes: int = 96,
        head_planes: int = 128,
    ) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        # Stem: /4 (512 -> 128)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_basic_layer(planes, planes, blocks=2, stride=1)
        self.layer2 = _make_basic_layer(planes, planes * 2, blocks=2, stride=2)  # /8 (64x64)

        # Integral (I) branch: /16 -> /32 -> /64 (8x8) -> StaticPAPPM
        self.layer3 = _make_basic_layer(planes * 2, planes * 4, blocks=3, stride=2)
        self.layer4 = _make_basic_layer(planes * 4, planes * 8, blocks=3, stride=2)
        self.layer5 = _make_bottleneck_layer(planes * 8, planes * 8, blocks=2, stride=2, no_relu_last=True)
        self.spp = StaticPAPPM(planes * 16, ppm_planes, head_planes)

        # Proportional (P) branch at /8 (64x64)
        self.layer3_p = _make_basic_layer(planes * 2, planes * 2, blocks=2, stride=1, no_relu_last=True)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )
        self.pag3 = PagFM(planes * 2, planes, scale_factor=2.0)

        self.layer4_p = _make_basic_layer(planes * 2, planes * 2, blocks=2, stride=1, no_relu_last=True)
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )
        self.pag4 = PagFM(planes * 2, planes, scale_factor=4.0)
        self.layer5_p = _make_bottleneck_layer(planes * 2, planes * 2, blocks=1, stride=1, no_relu_last=True)

        # Derivative (D) boundary branch at /8 (64x64)
        self.layer3_d = _make_basic_layer(planes * 2, planes, blocks=1, stride=1, no_relu_last=True)
        self.diff3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        self.layer4_d = _make_basic_layer(planes, planes * 2, blocks=1, stride=1, no_relu_last=True)
        self.diff4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes * 2),
        )
        self.layer5_d = _make_bottleneck_layer(planes * 2, planes * 2, blocks=1, stride=1, no_relu_last=True)

        self.dfm = BagFusion(head_planes, head_planes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)  # (B, 64, 64, 64)

        # Stage 3 (/16 -> /8 via static 2x upsample)
        x_p = self.layer3_p(x)
        x_d = self.layer3_d(x)
        x_i = self.layer3(x)
        x_p = self.pag3(x_p, self.compression3(x_i))
        x_d = x_d + F.interpolate(
            self.diff3(x_i), scale_factor=2.0, mode="bilinear", align_corners=False
        )

        # Stage 4 (/32 -> /8 via static 4x upsample)
        x_p = self.layer4_p(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))
        x_i = self.layer4(x_i)
        x_p = self.pag4(x_p, self.compression4(x_i))
        x_d = x_d + F.interpolate(
            self.diff4(x_i), scale_factor=4.0, mode="bilinear", align_corners=False
        )

        # Stage 5 (/64 -> /8 via static 8x upsample) + StaticPAPPM + BagFusion
        x_p = self.layer5_p(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))
        x_i = self.spp(self.layer5(x_i))
        x_i_up = F.interpolate(x_i, scale_factor=8.0, mode="bilinear", align_corners=False)

        fused = self.dfm(x_p, x_i_up, x_d)
        return fused, x_d


class PIDNetSlzTrain(nn.Module):
    """Training wrapper for Static-PAPPM PIDNet-S including train-only DINOv2 projector."""

    def __init__(
        self,
        num_classes: int = 5,
        dino_dim: int = 1024,
        planes: int = 32,
        ppm_planes: int = 96,
        head_planes: int = 128,
        dino_grid_hw: tuple[int, int] = (37, 37),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dino_dim = dino_dim
        self.dino_grid_hw = dino_grid_hw

        self.backbone = PIDNetSlzBackbone(
            planes=planes, ppm_planes=ppm_planes, head_planes=head_planes
        )
        self.seg_head = SegmentHead(head_planes, head_planes, num_classes)
        self.normal_head = SegmentHead(head_planes, head_planes, 3)
        self.bdry_head = SegmentHead(head_planes, planes * 2, 1)

        # Train-only projector matching DINOv2-ViT-L/14 dense 37x37x1024 patch tokens
        self.dino_proj = nn.Sequential(
            nn.Conv2d(head_planes, 512, kernel_size=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv2d(512, dino_dim, kernel_size=1, bias=True),
        )

    def forward(self, rgb_nchw: torch.Tensor) -> PIDNetTrainOutput:
        fused, x_d = self.backbone(rgb_nchw)

        seg_logits = F.interpolate(
            self.seg_head(fused), scale_factor=8.0, mode="bilinear", align_corners=False
        )
        raw_normals = F.interpolate(
            self.normal_head(fused), scale_factor=8.0, mode="bilinear", align_corners=False
        )
        normal_den = torch.sqrt(torch.sum(raw_normals * raw_normals, dim=1, keepdim=True) + 1e-12)
        surf_normals = raw_normals / normal_den
        bdry_prob = torch.sigmoid(self.bdry_head(x_d))

        dino_aligned = F.interpolate(
            fused, size=self.dino_grid_hw, mode="bilinear", align_corners=False
        )
        dino_feats = self.dino_proj(dino_aligned)
        return PIDNetTrainOutput(
            seg_logits=seg_logits,
            surf_normals=surf_normals,
            bdry_prob=bdry_prob,
            dino_feats=dino_feats,
        )


class PIDNetSlzDeployWrapper(nn.Module):
    """Deployment wrapper sharing trained weights while completely dropping `dino_proj`.

    Output contract (static 1x3x512x512 input):
      - seg_logits:   (1, 5, 512, 512)
      - surf_normals: (1, 3, 512, 512), L2 unit-normalized
      - bdry_prob:    (1, 1, 64, 64), sigmoid in [0, 1]
    """

    def __init__(self, trained_model: PIDNetSlzTrain) -> None:
        super().__init__()
        self.backbone = trained_model.backbone
        self.seg_head = trained_model.seg_head
        self.normal_head = trained_model.normal_head
        self.bdry_head = trained_model.bdry_head

    def forward(
        self, rgb_nchw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, x_d = self.backbone(rgb_nchw)

        seg_logits = F.interpolate(
            self.seg_head(fused), scale_factor=8.0, mode="bilinear", align_corners=False
        )
        raw_normals = F.interpolate(
            self.normal_head(fused), scale_factor=8.0, mode="bilinear", align_corners=False
        )
        normal_den = torch.sqrt(torch.sum(raw_normals * raw_normals, dim=1, keepdim=True) + 1e-12)
        surf_normals = raw_normals / normal_den
        bdry_prob = torch.sigmoid(self.bdry_head(x_d))
        return seg_logits, surf_normals, bdry_prob
