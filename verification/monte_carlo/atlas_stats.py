"""Exact Clopper-Pearson & Wilson 95% CI statistics with strict CSFL/CEL pool isolation."""

from __future__ import annotations

import math
from typing import Any
from scipy.stats import beta as beta_dist
from scipy.stats import norm as norm_dist


CSFL_POOL_STRATA = ("S2", "S3", "S5")
CEL_POOL_STRATA = ("S4",)


def min_zero_failure_trials(p_target: float, alpha: float = 0.05) -> int:
    """Exact minimum independent zero-failure trials to bound p <= p_target at (1 - alpha) CI.

    Uses math.log1p(-p_target) to preserve full IEEE-754 precision at p_target = 1e-9.
    """
    if not (0.0 < p_target < 1.0):
        raise ValueError(f"p_target must be in (0, 1), got {p_target}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil(math.log(alpha) / math.log1p(-p_target))


def clopper_pearson_and_wilson(k: int, n: int, alpha: float = 0.05) -> dict[str, Any]:
    """Compute exact 1-sided/2-sided Clopper-Pearson and 2-sided Wilson intervals."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0 <= k <= n):
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    p_hat = k / n

    # Exact 1-sided Clopper-Pearson upper bound (uses -expm1 for cancellation-free k=0)
    if k == 0:
        cp_one_sided_upper = -math.expm1(math.log(alpha) / n)
    elif k == n:
        cp_one_sided_upper = 1.0
    else:
        cp_one_sided_upper = float(beta_dist.ppf(1.0 - alpha, k + 1, n - k))

    # Exact 2-sided Clopper-Pearson interval
    if k == 0:
        cp_two_lower = 0.0
        cp_two_upper = -math.expm1(math.log(alpha / 2.0) / n)
    elif k == n:
        cp_two_lower = float(beta_dist.ppf(alpha / 2.0, n, 1))
        cp_two_upper = 1.0
    else:
        cp_two_lower = float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
        cp_two_upper = float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))

    # 2-sided Wilson score interval
    z = float(norm_dist.ppf(1.0 - alpha / 2.0))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half_width = (z * math.sqrt((p_hat * (1.0 - p_hat)) / n + z2 / (4.0 * n * n))) / denom
    wilson_lower = max(0.0, center - half_width)
    wilson_upper = min(1.0, center + half_width)

    return {
        "k": k,
        "n": n,
        "alpha": alpha,
        "p_hat": p_hat,
        "cp_one_sided_upper": cp_one_sided_upper,
        "p_surv_one_sided_lower": 1.0 - cp_one_sided_upper,
        "cp_two_sided": (cp_two_lower, cp_two_upper),
        "wilson_two_sided": (wilson_lower, wilson_upper),
    }


def _aggregate_subset(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if not bool(r["passed"]))
    stats = clopper_pearson_and_wilson(k=k, n=n, alpha=alpha)
    return stats


def summarize_atlas_pools(
    results: list[dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Summarize Atlas outcomes while strictly isolating CSFL (S2+S3+S5) from CEL (S4)."""
    if not results:
        raise ValueError("Cannot summarize empty Atlas results")

    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        stratum = str(row["stratum"])
        outcome = str(row["outcome_class"])

        if stratum == "S4" and outcome != "CEL":
            raise ValueError(
                f"CSFL/CEL denominator contamination: stratum S4 row {row.get('run_id')} tagged '{outcome}', must be 'CEL'"
            )
        if stratum in CSFL_POOL_STRATA and outcome != "CSFL":
            raise ValueError(
                f"CSFL/CEL denominator contamination: stratum {stratum} row {row.get('run_id')} tagged '{outcome}', must be 'CSFL'"
            )
        by_stratum.setdefault(stratum, []).append(row)

    per_stratum = {
        s: _aggregate_subset(rows, alpha) for s, rows in sorted(by_stratum.items())
    }

    csfl_rows = [r for s in CSFL_POOL_STRATA for r in by_stratum.get(s, [])]
    cel_rows = [r for s in CEL_POOL_STRATA for r in by_stratum.get(s, [])]
    if not csfl_rows or not cel_rows:
        raise ValueError("Atlas summary requires non-empty isolated CSFL (S2+S3+S5) and CEL (S4) pools")

    csfl_summary = _aggregate_subset(csfl_rows, alpha)
    csfl_summary["strata"] = list(CSFL_POOL_STRATA)
    cel_summary = _aggregate_subset(cel_rows, alpha)
    cel_summary["strata"] = list(CEL_POOL_STRATA)

    return {
        "schema_version": "vertirta.atlas_stats/v1.0.0",
        "alpha": alpha,
        "total_runs": len(results),
        "per_stratum": per_stratum,
        "csfl_pool": csfl_summary,
        "cel_pool": cel_summary,
        "full_atlas_footnote_only": _aggregate_subset(results, alpha),
        "zero_failure_1e9_required_fh": min_zero_failure_trials(1e-9, alpha=alpha),
    }
