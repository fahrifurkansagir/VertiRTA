# SPDX-License-Identifier: Apache-2.0
"""Balanced 60-seed 3-way ablation manifest, Gate 5 pass checker, and pure manifest merger."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta as beta_dist

from ml.slz_perception.data.generate_istanbul_8k import (
    LOCKED_DATUM_ALT_AMSL_M,
    LOCKED_DATUM_LAT,
    LOCKED_DATUM_LON,
    PROCEDURAL_COASTAL_BLOCKS_ENU,
)

ABLATION_BUCKETS: tuple[str, str, str] = (
    "clean_roof",
    "dynamic_hazard",
    "ood_glare_fog",
)

# Locked Plan 3 Task 7 / DR-7 Gate 5 quantitative acceptance criteria
GATE5_THRESHOLDS: dict[str, tuple[str, float]] = {
    "aeroscape_miou": (">=", 0.62),
    "aeroscape_fsr": ("<=", 0.04),
    "icg_miou": (">=", 0.60),
    "ood_auroc": (">=", 0.88),
    "active_safe_touchdown": (">=", 0.90),
    "active_false_commit": ("<=", 0.02),
    "active_max_soc_pct": ("<=", 1.00),
    "wsl2_peak_vram_mb": ("<", 300.0),
}

GATE5_OPTIONAL_THRESHOLDS: dict[str, tuple[str, float]] = {
    "icg_fsr": ("<=", 0.035),
    "ood_fpr95": ("<=", 0.25),
    "trt_fp16_p99_ms": ("<", 2.0),
    "wsl2_network_p99_ms": ("<", 12.0),
}


def build_ablation_seed_manifest(
    n_total: int = 60,
    *,
    base_seed: int = 1000,
) -> list[dict[str, Any]]:
    """Build deterministic balanced N-seed ablation manifest (20/20/20 for N=60).

    Every seed enforces the locked Kartal datum (40.8860 N, 29.1880 E, 12.0 m AMSL),
    22 kWh HV pack regime, and `uses_osm_odbl=False`.
    """
    if n_total < 3 or (n_total % 3) != 0:
        raise ValueError(f"n_total must be a positive multiple of 3, got {n_total}")

    regimes = ("NOMINAL", "SINGLE_LOSS", "CROSS_DUAL")
    rng = np.random.default_rng(base_seed)
    n_blocks = len(PROCEDURAL_COASTAL_BLOCKS_ENU)

    seeds: list[dict[str, Any]] = []
    for idx in range(n_total):
        bucket = ABLATION_BUCKETS[idx % 3]
        seed_val = int(base_seed + idx)
        block_id = int((idx % n_blocks) + 1)
        regime = regimes[idx % len(regimes)]
        wind_mps = round(float(rng.uniform(0.0, 10.0)), 2)
        initial_soc_pct = round(float(rng.uniform(45.0, 80.0)), 2)

        seeds.append(
            {
                "index": idx,
                "seed": seed_val,
                "bucket": bucket,
                "coastal_block_id": block_id,
                "fault_regime": regime,
                "wind_mps": wind_mps,
                "initial_agl_m": 80.0,
                "initial_soc_pct": initial_soc_pct,
                "datum_lat": LOCKED_DATUM_LAT,
                "datum_lon": LOCKED_DATUM_LON,
                "datum_alt_amsl_m": LOCKED_DATUM_ALT_AMSL_M,
                "uses_osm_odbl": False,
            }
        )

    return seeds


def _passes_op(val: float, op: str, limit: float) -> bool:
    if not math.isfinite(val):
        return False
    if op == ">=":
        return val >= limit
    if op == "<=":
        return val <= limit
    if op == "<":
        return val < limit
    if op == ">":
        return val > limit
    raise ValueError(f"Unsupported comparison operator: {op}")


def check_gate5_pass(metrics: dict[str, Any]) -> list[str]:
    """Return list of failed Gate 5 metric keys (empty list == full Gate 5 PASS)."""
    failures: list[str] = []

    for key, (op, limit) in GATE5_THRESHOLDS.items():
        if key not in metrics:
            failures.append(key)
            continue
        raw_val = metrics[key]
        try:
            val = float(raw_val)
        except (TypeError, ValueError):
            failures.append(key)
            continue
        if not _passes_op(val, op, limit):
            failures.append(key)

    for key, (op, limit) in GATE5_OPTIONAL_THRESHOLDS.items():
        if key not in metrics or metrics[key] is None:
            continue
        try:
            val = float(metrics[key])
        except (TypeError, ValueError):
            failures.append(key)
            continue
        if not _passes_op(val, op, limit):
            failures.append(key)

    return failures


def clopper_pearson_ci(
    successes: int,
    trials: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Exact two-sided Clopper-Pearson (1 - alpha) binomial confidence interval."""
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if successes < 0 or successes > trials:
        raise ValueError(f"successes ({successes}) must lie in [0, {trials}]")

    lo = 0.0 if successes == 0 else float(beta_dist.ppf(alpha / 2.0, successes, trials - successes + 1))
    hi = (
        1.0
        if successes == trials
        else float(beta_dist.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    )
    return lo, hi


def merge_metrics_manifest(
    v1_atlas: dict[str, Any],
    v2a_copilot: dict[str, Any],
    v2b_slz: dict[str, Any],
    *,
    wsl2_smoke: dict[str, Any] | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Purely merge Pillar 1–3 metric dictionaries into a canonical manifest payload.

    Does NOT write to disk unless `out_path` is explicitly provided by the caller.
    """
    payload: dict[str, Any] = {
        "schema_version": "vertirta_metrics_manifest_v2.0",
        "v1_atlas": dict(v1_atlas),
        "v2a_copilot": dict(v2a_copilot),
        "v2b_slz": dict(v2b_slz),
    }
    if wsl2_smoke is not None:
        payload["wsl2_gtx1650s_smoke"] = dict(wsl2_smoke)

    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return payload
