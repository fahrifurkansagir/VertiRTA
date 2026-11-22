# SPDX-License-Identifier: Apache-2.0
"""Zero-shot real-world SLZ evaluation metrics & strict eval-only quarantine enforcement.

Evaluates trained Apache-2.0 PIDNet-S weights on strictly quarantined eval-only datasets:
  - AeroScapes (CC BY-SA 4.0): 4-class mIoU >= 62%, SLZ F1 >= 74%, FSR <= 4.0%
  - ICG Semantic Drone (NC):   5-class mIoU >= 60%, SLZ F1 >= 75%, FSR <= 3.5%
  - OOD Separation (Free Energy vs MaxLogit): AUROC >= 0.88, FPR@95 <= 25%
  - UAVScenes (CC BY-NC-SA 4.0, RGB + Livox Avia LiDAR): Mod B slope MAE & AbsRel
    (strictly guarded by `assert_uavscenes_cmap_ids`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from ml.slz_perception.data.license_gate import (
    DatasetEntry,
    LicenseViolationError,
    ManifestReport,
    verify_training_manifest,
)
from ml.slz_perception.data.remappers import (
    NUM_CANONICAL_CLASSES,
    assert_uavscenes_cmap_ids,
)

EVAL_ONLY_LICENSES: dict[str, str] = {
    "AeroScapes": "CC BY-SA 4.0",
    "ICG_Semantic_Drone": "Non-Commercial",
    "UAVScenes": "CC BY-NC-SA 4.0",
}

_SAFE_CLASSES = (0, 1)
_HAZARD_CLASSES = (2, 3, 4)


def verify_eval_only_dataset(
    dataset: str,
    *,
    split: str = "eval",
    uavscenes_cmap: dict[int, str] | None = None,
) -> ManifestReport:
    """Enforce eval-only quarantine and UAVScenes sparse cmap.py integrity before evaluation.

    Raises:
        LicenseViolationError: If `split` is not an eval-only split (e.g., `split="train"`),
            or if `dataset` is banned / unknown.
        ValueError / AssertionError: If `dataset == "UAVScenes"` and `uavscenes_cmap` is
            missing or does not match the sparse 0..25 `cmap.py` ID layout.
    """
    if dataset not in EVAL_ONLY_LICENSES:
        raise LicenseViolationError(
            f"Dataset '{dataset}' is not in the locked eval-only quarantine list "
            f"{sorted(EVAL_ONLY_LICENSES)}."
        )

    report = verify_training_manifest(
        [
            DatasetEntry(
                name=dataset,
                split=split,
                license=EVAL_ONLY_LICENSES[dataset],
                uses_osm_odbl=False,
            )
        ]
    )

    if dataset == "UAVScenes":
        if uavscenes_cmap is None:
            raise ValueError(
                "UAVScenes evaluation requires passing the loaded cmap.py dict to "
                "assert_uavscenes_cmap_ids() before remapping."
            )
        assert_uavscenes_cmap_ids(uavscenes_cmap)

    return report


def _exact_auroc_and_fpr95(
    scores: np.ndarray,
    ood_mask: np.ndarray,
) -> tuple[float, float]:
    """Compute tie-aware rank-based AUROC and FPR@95%-ID-TPR (higher score => more OOD)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(ood_mask, dtype=bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores shape {s.shape} != ood_mask shape {y.shape}")

    finite = np.isfinite(s)
    s = s[finite]
    y = y[finite]

    s_ood = s[y]
    s_id = s[~y]
    n_ood = int(s_ood.size)
    n_id = int(s_id.size)
    if n_ood == 0 or n_id == 0:
        return float("nan"), float("nan")

    combined = np.concatenate([s_ood, s_id])
    ranks = rankdata(combined, method="average")
    u_stat = float(np.sum(ranks[:n_ood]) - n_ood * (n_ood + 1) / 2.0)
    auroc = u_stat / float(n_ood * n_id)

    # DR-7 Section C.1: threshold tau_95 at 95th percentile of clean ID energy (TPR_ID = 95%).
    # FPR@95 = fraction of OOD pixels erroneously accepted as ID (score <= tau_95).
    tau_95_id = float(np.quantile(s_id, 0.95))
    fpr95 = float(np.mean(s_ood <= tau_95_id))
    return float(auroc), float(fpr95)


def compute_slz_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    energy_map: np.ndarray,
    ood_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute canonical SLZ segmentation, False-Safe Rate, and Free-Energy OOD metrics.

    Args:
        pred_mask: Predicted canonical class IDs in {0..4}, shape (H, W) or (N, H, W).
        gt_mask: Ground-truth canonical class IDs in {0..4, 254, 255}, matching shape.
        energy_map: Boundary-compensated Free Energy E_tilde (higher => more OOD).
        ood_mask: Boolean mask (True where pixel is an OOD obstacle/outlier).

    Returns:
        Dictionary containing:
          - per_class_iou: {class_id: iou} for classes with union > 0
          - active_class_count: number of present canonical classes (4 for AeroScapes, 5 for ICG)
          - miou: mean IoU across active canonical classes
          - slz_f1: binary Safe ({0,1}) vs Hazard ({2,3,4}) F1 score
          - false_safe_rate: P(pred in {0,1} | gt in {2,3,4}) over valid canonical pixels
          - ood_auroc: exact rank-based AUROC of energy_map against ood_mask
          - ood_fpr95: OOD false acceptance rate at 95% ID acceptance threshold
    """
    pred = np.asarray(pred_mask)
    gt = np.asarray(gt_mask)
    if pred.shape != gt.shape:
        raise ValueError(f"pred_mask shape {pred.shape} != gt_mask shape {gt.shape}")

    valid = (gt >= 0) & (gt < NUM_CANONICAL_CLASSES)
    p_valid = pred[valid]
    g_valid = gt[valid]

    per_class_iou: dict[int, float] = {}
    for cls_id in range(NUM_CANONICAL_CLASSES):
        p_c = p_valid == cls_id
        g_c = g_valid == cls_id
        inter = int(np.sum(p_c & g_c))
        union = int(np.sum(p_c | g_c))
        if union > 0:
            per_class_iou[cls_id] = float(inter) / float(union)

    miou = (
        float(np.mean(list(per_class_iou.values()))) if per_class_iou else float("nan")
    )

    pred_safe = np.isin(p_valid, _SAFE_CLASSES)
    gt_safe = np.isin(g_valid, _SAFE_CLASSES)
    gt_hazard = np.isin(g_valid, _HAZARD_CLASSES)

    tp_safe = int(np.sum(pred_safe & gt_safe))
    fp_safe = int(np.sum(pred_safe & gt_hazard))
    fn_safe = int(np.sum((~pred_safe) & gt_safe))
    denom_f1 = 2 * tp_safe + fp_safe + fn_safe
    slz_f1 = float(2 * tp_safe) / float(denom_f1) if denom_f1 > 0 else 0.0

    n_hazard = int(np.sum(gt_hazard))
    false_safe_rate = float(fp_safe) / float(n_hazard) if n_hazard > 0 else 0.0

    ood_auroc, ood_fpr95 = _exact_auroc_and_fpr95(energy_map, ood_mask)

    return {
        "per_class_iou": per_class_iou,
        "active_class_count": len(per_class_iou),
        "miou": miou,
        "slz_f1": slz_f1,
        "false_safe_rate": false_safe_rate,
        "ood_auroc": ood_auroc,
        "ood_fpr95": ood_fpr95,
    }


def compute_free_energy_vs_maxlogit_ood(
    logits: np.ndarray,
    bdry_prob: np.ndarray,
    ood_mask: np.ndarray,
    *,
    t_ood: float = 1.0,
    kappa_b: float = 1.5,
) -> dict[str, float]:
    """Compare boundary-compensated Helmholtz Free Energy against raw MaxLogit on OOD pixels."""
    arr = np.asarray(logits, dtype=np.float64)
    if arr.ndim < 2 or arr.shape[-1] != NUM_CANONICAL_CLASSES:
        raise ValueError(
            f"logits must have last dim == {NUM_CANONICAL_CLASSES}, got shape {arr.shape}"
        )
    scaled = arr / float(t_ood)
    max_s = np.max(scaled, axis=-1, keepdims=True)
    lse = max_s[..., 0] + np.log(np.sum(np.exp(scaled - max_s), axis=-1))
    e_raw = -float(t_ood) * lse
    e_tilde = e_raw - float(kappa_b) * np.clip(np.asarray(bdry_prob, dtype=np.float64), 0.0, 1.0)
    maxlogit_ood_score = -np.max(arr, axis=-1)

    fe_auroc, fe_fpr95 = _exact_auroc_and_fpr95(e_tilde, ood_mask)
    ml_auroc, ml_fpr95 = _exact_auroc_and_fpr95(maxlogit_ood_score, ood_mask)
    return {
        "free_energy_auroc": fe_auroc,
        "free_energy_fpr95": fe_fpr95,
        "maxlogit_auroc": ml_auroc,
        "maxlogit_fpr95": ml_fpr95,
    }


def compute_monocular_slope_metrics(
    pred_slope_deg: np.ndarray,
    gt_lidar_slope_deg: np.ndarray,
    pred_depth_m: np.ndarray,
    gt_lidar_depth_m: np.ndarray,
) -> dict[str, float]:
    """Compute Mod B monocular slope MAE / P95 error & depth AbsRel against Livox Avia LiDAR GT."""
    ps = np.asarray(pred_slope_deg, dtype=np.float64).ravel()
    gs = np.asarray(gt_lidar_slope_deg, dtype=np.float64).ravel()
    pd = np.asarray(pred_depth_m, dtype=np.float64).ravel()
    gd = np.asarray(gt_lidar_depth_m, dtype=np.float64).ravel()
    if ps.shape != gs.shape or pd.shape != gd.shape:
        raise ValueError("Predicted and ground-truth arrays must have matching shapes.")

    valid_s = np.isfinite(ps) & np.isfinite(gs)
    valid_d = np.isfinite(pd) & np.isfinite(gd) & (gd > 1e-3)
    if not bool(valid_s.any()) or not bool(valid_d.any()):
        raise ValueError("No valid finite samples provided for monocular slope/depth metrics.")

    slope_err = np.abs(ps[valid_s] - gs[valid_s])
    abs_rel = np.abs(pd[valid_d] - gd[valid_d]) / gd[valid_d]
    return {
        "slope_mae_deg": float(np.mean(slope_err)),
        "slope_p95_deg": float(np.percentile(slope_err, 95.0)),
        "depth_abs_rel": float(np.mean(abs_rel)),
        "n_slope_samples": int(valid_s.sum()),
        "n_depth_samples": int(valid_d.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify eval-only quarantine and inspect zero-shot SLZ evaluation protocol."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(EVAL_ONLY_LICENSES),
        default="AeroScapes",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset == "UAVScenes":
        print(
            "[VertiRTA-Eval] UAVScenes requires --cmap verification at dataset load time via "
            "assert_uavscenes_cmap_ids()."
        )
        return

    report = verify_eval_only_dataset(args.dataset, split="eval")
    summary = {
        "dataset": args.dataset,
        "split": "eval",
        "license": EVAL_ONLY_LICENSES[args.dataset],
        "quarantined_eval_entries": list(report.quarantined_eval_entries),
        "zero_sharealike_verified": report.zero_sharealike_verified,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
