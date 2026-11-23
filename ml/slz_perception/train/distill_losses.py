# SPDX-License-Identifier: Apache-2.0
"""Foundation-to-student distillation & Free-Energy OOD margin losses for VertiRTA SLZ.

Implements DR-7 Section B.2 & C.1:
  L_total = L_ce_lovasz
          + lambda_bdry * L_bdry
          + lambda_dino * L_feat_dino
          + lambda_geom * L_geom_dav2
          + lambda_ood  * L_energy_margin
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t. sorted errors."""
    p = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1.0)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def _lovasz_softmax_flat(
    probs: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax over flattened valid ID pixels."""
    if probs.numel() == 0:
        return probs.sum() * 0.0
    num_classes = probs.shape[1]
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for c in range(num_classes):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        loss_c = torch.dot(errors_sorted, _lovasz_grad(fg_sorted))
        losses.append(loss_c * class_weights[c])
        weights.append(class_weights[c])
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).sum() / torch.stack(weights).sum()


class SlzDistillCriterion(nn.Module):
    """Composite loss for Static-PAPPM PIDNet-S distillation + Stage-2 Free-Energy OOD."""

    def __init__(
        self,
        lambda_bdry: float = 0.4,
        lambda_dino: float = 0.5,
        lambda_geom: float = 0.3,
        lambda_ood: float = 0.0,
        *,
        lambda_fsr: float = 0.5,
        m_in: float = -8.0,
        m_out: float = -2.5,
        ood_temp: float = 1.0,
        ignore_index: int = 255,
        ood_index: int = 254,
        geom_drop_top_frac: float = 0.10,
        bdry_pos_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_bdry = float(lambda_bdry)
        self.lambda_dino = float(lambda_dino)
        self.lambda_geom = float(lambda_geom)
        self.lambda_ood = float(lambda_ood)
        self.lambda_fsr = float(lambda_fsr)
        self.m_in = float(m_in)
        self.m_out = float(m_out)
        self.ood_temp = float(ood_temp)
        self.ignore_index = int(ignore_index)
        self.ood_index = int(ood_index)
        self.geom_drop_top_frac = float(geom_drop_top_frac)
        self.bdry_pos_weight = float(bdry_pos_weight)

        # 1.5x weight across ALL hazard classes {2: Obstacle, 3: Dynamic, 4: Water/Vegetation} to curb FSR
        self.register_buffer(
            "class_weights",
            torch.tensor([1.0, 1.0, 1.5, 1.5, 1.5], dtype=torch.float32),
        )

    def _seg_ce_lovasz_loss(
        self, seg_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        logits_f = seg_logits.float()
        num_classes = logits_f.shape[1]
        weights = self.class_weights.to(device=logits_f.device, dtype=torch.float32)

        ce_targets = labels.long().clone()
        ce_targets[ce_targets == self.ood_index] = self.ignore_index
        id_mask = (ce_targets >= 0) & (ce_targets < num_classes)
        if not bool(id_mask.any()):
            return logits_f.sum() * 0.0

        ce_loss = F.cross_entropy(
            logits_f,
            ce_targets,
            weight=weights,
            ignore_index=self.ignore_index,
        )
        probs = F.softmax(logits_f, dim=1)
        flat_probs = probs.permute(0, 2, 3, 1)[id_mask]
        flat_labels = ce_targets[id_mask]
        lovasz_loss = _lovasz_softmax_flat(flat_probs, flat_labels, weights)
        return ce_loss + lovasz_loss

    def _boundary_loss(
        self, bdry_prob: torch.Tensor, target_bdry: torch.Tensor | None
    ) -> torch.Tensor:
        if target_bdry is None or self.lambda_bdry <= 0.0:
            return bdry_prob.float().sum() * 0.0
        p = bdry_prob.float().clamp(1e-6, 1.0 - 1e-6)
        tgt = target_bdry.float()
        if tgt.ndim == 3:
            tgt = tgt.unsqueeze(1)
        if tgt.shape[2:] != p.shape[2:]:
            tgt = F.interpolate(tgt, size=p.shape[2:], mode="nearest")
        tgt = tgt.clamp(0.0, 1.0)
        return -(
            self.bdry_pos_weight * tgt * torch.log(p) + (1.0 - tgt) * torch.log(1.0 - p)
        ).mean()

    def _dino_cosine_loss(
        self, student_dino: torch.Tensor | None, teacher_dino: torch.Tensor | None
    ) -> torch.Tensor:
        if student_dino is None or teacher_dino is None or self.lambda_dino <= 0.0:
            ref = student_dino if student_dino is not None else teacher_dino
            return (ref.float().sum() * 0.0) if ref is not None else torch.tensor(0.0)
        s = student_dino.float()
        t = teacher_dino.float()
        if s.shape[2:] != t.shape[2:]:
            s = F.interpolate(s, size=t.shape[2:], mode="bilinear", align_corners=False)
        cos_sim = F.cosine_similarity(s, t, dim=1, eps=1e-6)
        return (1.0 - cos_sim).mean()

    def _geom_normal_loss(
        self,
        surf_normals: torch.Tensor,
        target_normals: torch.Tensor | None,
        labels: torch.Tensor,
        valid_geom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_normals is None or self.lambda_geom <= 0.0:
            return surf_normals.float().sum() * 0.0
        pred = F.normalize(surf_normals.float(), p=2.0, dim=1, eps=1e-6)
        tgt_raw = target_normals.float()
        if tgt_raw.shape[2:] != pred.shape[2:]:
            tgt_raw = F.interpolate(tgt_raw, size=pred.shape[2:], mode="bilinear", align_corners=False)
        tgt_norm = torch.linalg.norm(tgt_raw, dim=1)
        tgt = tgt_raw / tgt_norm.unsqueeze(1).clamp(min=1e-6)

        valid = (labels != self.ignore_index) & (tgt_norm > 1e-6) & torch.isfinite(tgt_norm)
        if valid_geom_mask is not None:
            valid = valid & valid_geom_mask.bool()
        if not bool(valid.any()):
            return pred.sum() * 0.0

        cos_term = 1.0 - (pred * tgt).sum(dim=1)
        l1_term = 0.5 * (pred - tgt).abs().sum(dim=1)
        pix_loss = (cos_term + l1_term)[valid]

        # Depth Anything V2 protocol: mask highest-loss top 10% noisy pseudo-normal pixels
        n_valid = pix_loss.numel()
        if n_valid > 1 and self.geom_drop_top_frac > 0.0:
            keep_k = max(1, int(n_valid * (1.0 - self.geom_drop_top_frac)))
            pix_loss, _ = torch.topk(pix_loss, k=keep_k, largest=False)
        return pix_loss.mean()

    def _false_safe_penalty(
        self, seg_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Asymmetric false-safe penalty on all true hazard pixels {2, 3, 4}."""
        if self.lambda_fsr <= 0.0:
            return seg_logits.float().sum() * 0.0
        logits_f = seg_logits.float()
        hazard_mask = (labels == 2) | (labels == 3) | (labels == 4)
        if not bool(hazard_mask.any()):
            return logits_f.sum() * 0.0
        probs = F.softmax(logits_f, dim=1)
        p_safe = (probs[:, 0] + probs[:, 1]).clamp(1e-6, 1.0 - 1e-6)
        p_safe_haz = p_safe[hazard_mask]
        return (p_safe_haz + 0.25 * (-torch.log((1.0 - p_safe_haz).clamp(min=1e-4)))).mean()

    def _energy_margin_loss(
        self, seg_logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.lambda_ood <= 0.0:
            return seg_logits.float().sum() * 0.0
        logits_f = seg_logits.float()
        num_classes = logits_f.shape[1]
        energy = -self.ood_temp * torch.logsumexp(logits_f / self.ood_temp, dim=1)

        id_mask = (labels >= 0) & (labels < num_classes)
        ood_mask = labels == self.ood_index

        loss = logits_f.sum() * 0.0
        if bool(id_mask.any()):
            loss = loss + F.relu(energy[id_mask] - self.m_in).pow(2).mean()
        if bool(ood_mask.any()):
            loss = loss + F.relu(self.m_out - energy[ood_mask]).pow(2).mean()
        return loss

    def forward(
        self,
        seg_logits: torch.Tensor,
        surf_normals: torch.Tensor,
        bdry_prob: torch.Tensor,
        student_dino: torch.Tensor | None,
        labels: torch.Tensor,
        target_normals: torch.Tensor | None = None,
        target_bdry: torch.Tensor | None = None,
        teacher_dino: torch.Tensor | None = None,
        valid_geom_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        l_seg = self._seg_ce_lovasz_loss(seg_logits, labels)
        l_fsr = self._false_safe_penalty(seg_logits, labels)
        l_bdry = self._boundary_loss(bdry_prob, target_bdry)
        l_dino = self._dino_cosine_loss(student_dino, teacher_dino)
        l_geom = self._geom_normal_loss(surf_normals, target_normals, labels, valid_geom_mask)
        l_ood = self._energy_margin_loss(seg_logits, labels)
        return (
            l_seg
            + self.lambda_fsr * l_fsr
            + self.lambda_bdry * l_bdry
            + self.lambda_dino * l_dino
            + self.lambda_geom * l_geom
            + self.lambda_ood * l_ood
        )
