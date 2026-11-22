# SPDX-License-Identifier: Apache-2.0
"""Post-hoc temperature scaling (T_prob*) and 95%-ID-TPR Free-Energy threshold (tau_ood)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class OodCalibration:
    """Frozen OOD & probability temperature calibration written to `ood_calibration.json`."""

    T_ood: float
    T_prob: float
    kappa_b: float
    tau_ood: float
    id_tpr_target: float
    n_id_samples: int
    nll_before: float
    nll_after: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def save_json(self, out_path: Path | str) -> Path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path


def compute_free_energy(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Compute Helmholtz Free Energy E(u) = -T * logsumexp(f(u) / T) along class dim -1."""
    if T <= 0.0:
        raise ValueError(f"Temperature T must be positive, got {T}")
    return -float(T) * torch.logsumexp(logits.float() / float(T), dim=-1)


def calibrate_temperature_and_tau(
    id_logits: torch.Tensor,
    id_labels: torch.Tensor,
    id_bdry_prob: torch.Tensor,
    *,
    T_ood: float = 1.0,
    kappa_b: float = 1.5,
    id_tpr_target: float = 0.95,
    t_prob_bounds: tuple[float, float] = (0.5, 3.0),
) -> OodCalibration:
    """Fit NLL-minimizing `T_prob` in [0.5, 3.0] and 95%-ID-TPR `tau_ood` on clean ID pixels.

    Args:
        id_logits: Tensor of shape (N, C) or (..., C) raw canonical class logits.
        id_labels: Tensor of shape (N,) or (...) integer class labels in [0, C-1].
        id_bdry_prob: Tensor of shape (N,) or (...) boundary probabilities in [0, 1].
        T_ood: Fixed OOD energy temperature (DR-7 lock: 1.0).
        kappa_b: Boundary compensation gain for E_tilde = E - kappa_b * b (DR-7 lock: 1.5).
        id_tpr_target: Target clean ID acceptance rate (DR-7 lock: 0.95).
        t_prob_bounds: Bounded search interval for probability temperature T_prob.
    """
    if id_logits.ndim < 2:
        raise ValueError(f"id_logits must have at least 2 dims (..., C), got {id_logits.shape}")
    num_classes = id_logits.shape[-1]
    logits_flat = id_logits.detach().float().reshape(-1, num_classes)
    labels_flat = id_labels.detach().long().reshape(-1)
    bdry_flat = id_bdry_prob.detach().float().reshape(-1)

    if logits_flat.shape[0] != labels_flat.shape[0] or logits_flat.shape[0] != bdry_flat.shape[0]:
        raise ValueError(
            f"Mismatched sample counts: logits={logits_flat.shape[0]}, "
            f"labels={labels_flat.shape[0]}, bdry={bdry_flat.shape[0]}"
        )

    valid = (labels_flat >= 0) & (labels_flat < num_classes)
    if not bool(valid.any()):
        raise ValueError("No valid ID samples found for OOD / temperature calibration.")

    v_logits = logits_flat[valid]
    v_labels = labels_flat[valid]
    v_bdry = bdry_flat[valid].clamp(0.0, 1.0)

    nll_before = float(F.cross_entropy(v_logits, v_labels).item())

    def _nll_for_temp(t_val: float) -> float:
        return float(F.cross_entropy(v_logits / float(t_val), v_labels).item())

    res = minimize_scalar(_nll_for_temp, bounds=t_prob_bounds, method="bounded")
    t_prob_opt = float(min(max(res.x, t_prob_bounds[0]), t_prob_bounds[1]))
    nll_after = _nll_for_temp(t_prob_opt)

    e_raw = compute_free_energy(v_logits, T=T_ood)
    e_tilde = e_raw - float(kappa_b) * v_bdry
    tau_ood = float(torch.quantile(e_tilde, float(id_tpr_target)).item())

    return OodCalibration(
        T_ood=float(T_ood),
        T_prob=t_prob_opt,
        kappa_b=float(kappa_b),
        tau_ood=tau_ood,
        id_tpr_target=float(id_tpr_target),
        n_id_samples=int(v_logits.shape[0]),
        nll_before=nll_before,
        nll_after=nll_after,
    )
