# SPDX-License-Identifier: Apache-2.0
"""Zero-shot real evaluation benchmarks (eval-only quarantine) & Gate 5 ablation aggregation."""

from ml.slz_perception.eval.aggregate_ablation_tables import (
    GATE5_THRESHOLDS,
    build_ablation_seed_manifest,
    check_gate5_pass,
    clopper_pearson_ci,
    merge_metrics_manifest,
)
from ml.slz_perception.eval.eval_zeroshot_real import (
    EVAL_ONLY_LICENSES,
    compute_free_energy_vs_maxlogit_ood,
    compute_monocular_slope_metrics,
    compute_slz_metrics,
    verify_eval_only_dataset,
)

__all__ = [
    "EVAL_ONLY_LICENSES",
    "GATE5_THRESHOLDS",
    "build_ablation_seed_manifest",
    "check_gate5_pass",
    "clopper_pearson_ci",
    "compute_free_energy_vs_maxlogit_ood",
    "compute_monocular_slope_metrics",
    "compute_slz_metrics",
    "merge_metrics_manifest",
    "verify_eval_only_dataset",
]
