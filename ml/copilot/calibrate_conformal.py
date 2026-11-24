"""Split Conformal Prediction calibration (KnowNo-style) + Post-Hoc NLL Temperature Scaling.

Computes:
1. Post-hoc NLL temperature scaling (tau* > 0 via log-spaced grid + Brent/golden-section
   optimization strictly on the calibration split — zero test leakage!).
2. Finite-sample Split Conformal quantile:
     q_hat = Quantile({s_i}; ceil((n_cal + 1)*(1 - alpha)) / n_cal)
   over non-conformity scores s_i = 1 - p_tau(y_i* | x_i), dramatically tightening prediction
   set sizes while preserving finite-sample >= (1 - alpha) marginal coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


def softmax_with_temperature(scores: Sequence[float], tau: float = 1.0) -> list[float]:
    """Numerically stable softmax(scores / tau) for tau > 0."""
    if not scores:
        raise ValueError("scores must be non-empty")
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError(f"tau must be finite and > 0, got {tau}")
    scaled = [float(s) / tau for s in scores]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    denom = sum(exps)
    return [e / denom for e in exps]


def compute_nll(
    score_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    tau: float = 1.0,
) -> float:
    """Mean negative log-likelihood -mean(log softmax(z_i / tau)[gold_i])."""
    n = len(score_rows)
    if n == 0 or n != len(gold_indices):
        raise ValueError("score_rows and gold_indices must be non-empty and equal length")
    total = 0.0
    for row, g_idx in zip(score_rows, gold_indices):
        if not (0 <= g_idx < len(row)):
            raise IndexError(f"gold index {g_idx} out of range for row of length {len(row)}")
        probs = softmax_with_temperature(row, tau=tau)
        total -= math.log(max(probs[g_idx], 1e-12))
    return total / float(n)


def compute_ece(
    prob_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Top-1 Expected Calibration Error (ECE) over n_bins uniform confidence intervals."""
    n = len(prob_rows)
    if n == 0 or n != len(gold_indices):
        raise ValueError("prob_rows and gold_indices must be non-empty and equal length")
    bin_counts = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    bin_acc_sum = [0.0] * n_bins
    for probs, g_idx in zip(prob_rows, gold_indices):
        best_i = max(range(len(probs)), key=lambda idx: probs[idx])
        conf = float(probs[best_i])
        acc = 1.0 if best_i == g_idx else 0.0
        b = min(n_bins - 1, max(0, int(conf * n_bins)))
        bin_counts[b] += 1
        bin_conf_sum[b] += conf
        bin_acc_sum[b] += acc
    ece = 0.0
    for b in range(n_bins):
        if bin_counts[b] > 0:
            ece += abs(bin_acc_sum[b] - bin_conf_sum[b]) / float(n)
    return ece


def _minimize_log_tau_golden(
    fn: Any,
    log_lo: float,
    log_hi: float,
    iters: int = 56,
) -> float:
    """Pure-Python deterministic golden-section minimizer on log(tau) (no external solver needed)."""
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = log_lo, log_hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc = fn(math.exp(c))
    fd = fn(math.exp(d))
    for _ in range(iters):
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = fn(math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = fn(math.exp(d))
    return math.exp(0.5 * (a + b))


def fit_temperature_scaling_nll(
    score_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    *,
    tau_min: float = 0.05,
    tau_max: float = 5.0,
    n_grid: int = 160,
) -> dict[str, Any]:
    """Fit scalar temperature tau* minimizing calibration NLL (strictly calibration split ONLY)."""
    n = len(score_rows)
    if n == 0 or n != len(gold_indices):
        raise ValueError("score_rows and gold_indices must be non-empty and equal length")
    if not (0.0 < tau_min < tau_max):
        raise ValueError(f"Require 0 < tau_min < tau_max, got [{tau_min}, {tau_max}]")

    log_min = math.log(tau_min)
    log_max = math.log(tau_max)
    step = (log_max - log_min) / float(max(1, n_grid - 1))
    grid_log_taus = [log_min + i * step for i in range(n_grid)]
    grid_nlls = [compute_nll(score_rows, gold_indices, math.exp(lt)) for lt in grid_log_taus]
    best_gi = min(range(len(grid_nlls)), key=lambda idx: grid_nlls[idx])

    bracket_lo = grid_log_taus[max(0, best_gi - 1)]
    bracket_hi = grid_log_taus[min(len(grid_log_taus) - 1, best_gi + 1)]
    tau_opt = _minimize_log_tau_golden(
        lambda t: compute_nll(score_rows, gold_indices, t),
        bracket_lo,
        bracket_hi,
    )
    nll_opt = compute_nll(score_rows, gold_indices, tau_opt)
    if grid_nlls[best_gi] < nll_opt:
        tau_opt = math.exp(grid_log_taus[best_gi])
        nll_opt = grid_nlls[best_gi]

    nll_before = compute_nll(score_rows, gold_indices, 1.0)
    probs_before = [softmax_with_temperature(r, tau=1.0) for r in score_rows]
    probs_after = [softmax_with_temperature(r, tau=tau_opt) for r in score_rows]
    ece_before = compute_ece(probs_before, gold_indices)
    ece_after = compute_ece(probs_after, gold_indices)
    top1_hits = sum(
        1
        for r, g_idx in zip(score_rows, gold_indices)
        if max(range(len(r)), key=lambda idx: r[idx]) == g_idx
    )

    return {
        "tau_opt": round(tau_opt, 6),
        "nll_before": round(nll_before, 6),
        "nll_after": round(nll_opt, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "cal_top1_accuracy_pct": round(100.0 * top1_hits / float(n), 2),
        "n_cal_fit": n,
        "tau_bounds": [tau_min, tau_max],
        "optimizer": "log_grid_160_plus_golden_section_nll",
    }


def evaluate_conformal_sets(
    score_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    *,
    tau: float,
    q_hat: float,
) -> dict[str, Any]:
    """Evaluate empirical coverage, prediction set size, singleton/clarify/abstain rates."""
    n = len(score_rows)
    if n == 0 or n != len(gold_indices):
        raise ValueError("score_rows and gold_indices must be non-empty and equal length")
    p_min = 1.0 - float(q_hat)
    prob_rows = [softmax_with_temperature(r, tau=tau) for r in score_rows]
    covered = 0
    set_sizes: list[int] = []
    for probs, g_idx in zip(prob_rows, gold_indices):
        if probs[g_idx] >= p_min:
            covered += 1
        set_sizes.append(sum(1 for p in probs if p >= p_min))

    nll = compute_nll(score_rows, gold_indices, tau=tau)
    ece = compute_ece(prob_rows, gold_indices)
    return {
        "tau": round(float(tau), 6),
        "q_hat": round(float(q_hat), 6),
        "p_min_threshold": round(p_min, 6),
        "n_eval": n,
        "covered": covered,
        "empirical_coverage_pct": round(100.0 * covered / float(n), 2),
        "mean_prediction_set_size": round(sum(set_sizes) / float(n), 3),
        "singleton_exec_pct": round(100.0 * sum(1 for s in set_sizes if s == 1) / float(n), 2),
        "clarify_multi_pct": round(100.0 * sum(1 for s in set_sizes if s > 1) / float(n), 2),
        "abstain_empty_pct": round(100.0 * sum(1 for s in set_sizes if s == 0) / float(n), 2),
        "nll": round(nll, 6),
        "ece": round(ece, 6),
    }


def fit_and_calibrate_conformal(
    cal_score_rows: Sequence[Sequence[float]],
    cal_gold_indices: Sequence[int],
    *,
    alpha: float = 0.05,
    tau_min: float = 0.05,
    tau_max: float = 5.0,
) -> dict[str, Any]:
    """Fit post-hoc NLL temperature tau* on cal split and compute Split Conformal q_hat."""
    ts_fit = fit_temperature_scaling_nll(
        cal_score_rows, cal_gold_indices, tau_min=tau_min, tau_max=tau_max
    )
    tau_opt = float(ts_fit["tau_opt"])

    # Baseline (uncalibrated tau = 1.0)
    uncal_scores = [
        1.0 - softmax_with_temperature(r, tau=1.0)[g]
        for r, g in zip(cal_score_rows, cal_gold_indices)
    ]
    q_hat_uncal, rank_k, q_level = compute_split_conformal_qhat(uncal_scores, alpha=alpha)

    # Temperature-calibrated (tau = tau_opt)
    cal_scores_t = [
        1.0 - softmax_with_temperature(r, tau=tau_opt)[g]
        for r, g in zip(cal_score_rows, cal_gold_indices)
    ]
    q_hat_cal, rank_k_cal, q_level_cal = compute_split_conformal_qhat(cal_scores_t, alpha=alpha)

    # Exchangeability-strict 50/50 stratified split-calibration audit
    # (even indices fit tau; disjoint odd indices calibrate q_hat across all buckets).
    split_audit: dict[str, Any] | None = None
    if len(cal_score_rows) >= 20:
        fit_scores = [cal_score_rows[i] for i in range(0, len(cal_score_rows), 2)]
        fit_golds = [cal_gold_indices[i] for i in range(0, len(cal_gold_indices), 2)]
        q_scores = [cal_score_rows[i] for i in range(1, len(cal_score_rows), 2)]
        q_golds = [cal_gold_indices[i] for i in range(1, len(cal_gold_indices), 2)]
        half_fit = fit_temperature_scaling_nll(
            fit_scores, fit_golds, tau_min=tau_min, tau_max=tau_max
        )
        half_tau = float(half_fit["tau_opt"])
        half_nc = [
            1.0 - softmax_with_temperature(r, tau=half_tau)[g]
            for r, g in zip(q_scores, q_golds)
        ]
        q_half, k_half, ql_half = compute_split_conformal_qhat(half_nc, alpha=alpha)
        split_audit = {
            "split_policy": "stratified_even_odd_50_50",
            "n_tau_fit": len(fit_scores),
            "n_qhat_cal": len(q_scores),
            "tau_opt": round(half_tau, 6),
            "q_hat": round(q_half, 6),
            "p_min_threshold": round(1.0 - q_half, 6),
            "order_statistic_k": k_half,
            "q_level": round(ql_half, 6),
        }

    return {
        "alpha": alpha,
        "target_coverage": round(1.0 - alpha, 6),
        "n_cal": len(cal_score_rows),
        "order_statistic_k": rank_k_cal,
        "q_level": round(q_level_cal, 6),
        "temperature_scaling": ts_fit,
        "uncalibrated": {
            "tau": 1.0,
            "q_hat": round(q_hat_uncal, 6),
            "p_min_threshold": round(1.0 - q_hat_uncal, 6),
            "order_statistic_k": rank_k,
            "q_level": round(q_level, 6),
        },
        "calibrated": {
            "tau": round(tau_opt, 6),
            "q_hat": round(q_hat_cal, 6),
            "p_min_threshold": round(1.0 - q_hat_cal, 6),
            "order_statistic_k": rank_k_cal,
            "q_level": round(q_level_cal, 6),
        },
        "strict_split_cal_audit": split_audit,
    }


def compute_split_conformal_qhat(
    nonconformity_scores: Sequence[float],
    alpha: float = 0.05,
) -> tuple[float, int, float]:
    """Return (q_hat, rank_index_1based, adjusted_quantile_level)."""
    n = len(nonconformity_scores)
    if n == 0:
        raise ValueError("nonconformity_scores must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    ordered = sorted(float(s) for s in nonconformity_scores)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = max(1, min(n, k))
    q_level = k / float(n)
    return ordered[k - 1], k, q_level


def calibrate_from_jsonl(
    calibration_path: Path,
    out_path: Path,
    alpha: float = 0.05,
    expected_n_cal: int | None = None,
) -> dict[str, Any]:
    """Load calibration.jsonl records with gold_prob or nonconformity_score and emit JSON."""
    scores: list[float] = []
    with calibration_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if "nonconformity_score" in row:
                score = float(row["nonconformity_score"])
            elif "gold_prob" in row:
                score = 1.0 - float(row["gold_prob"])
            else:
                raise KeyError(
                    f"Line {line_no} missing 'nonconformity_score' or 'gold_prob'"
                )
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"Line {line_no} score {score} outside [0, 1]")
            scores.append(score)

    if expected_n_cal is not None and len(scores) != expected_n_cal:
        raise ValueError(f"Expected n_cal={expected_n_cal}, got {len(scores)}")

    q_hat, rank_k, q_level = compute_split_conformal_qhat(scores, alpha=alpha)
    payload: dict[str, Any] = {
        "schema_version": "vertirta.conformal_calibration.v1",
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "n_cal": len(scores),
        "order_statistic_k": rank_k,
        "q_level": round(q_level, 6),
        "q_hat": round(q_hat, 6),
        "p_min_threshold": round(1.0 - q_hat, 6),
        "loiter_liveness_min_soc_margin_pct": 5.0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return payload


def length_normalized_scores(
    logprob_rows: Sequence[Sequence[float]],
    token_count_rows: Sequence[Sequence[int]],
    gamma: float = 0.5,
) -> list[list[float]]:
    """Compute length-normalized branch scores sum_logprob / max(1, n_tok)^gamma."""
    if len(logprob_rows) != len(token_count_rows) or not logprob_rows:
        raise ValueError("logprob_rows and token_count_rows must be non-empty and equal length")
    out: list[list[float]] = []
    for lps, cnts in zip(logprob_rows, token_count_rows):
        if len(lps) != len(cnts) or not lps:
            raise ValueError("Each row of logprobs and token_counts must be non-empty and equal length")
        out.append([float(lp) / (float(max(1, int(c))) ** float(gamma)) for lp, c in zip(lps, cnts)])
    return out


def cv_select_gamma_tau_calibration_only(
    logprob_rows: Sequence[Sequence[float]],
    token_count_rows: Sequence[Sequence[int]],
    gold_indices: Sequence[int],
    *,
    gammas: Sequence[float] = (0.5, 0.75, 1.0),
    n_folds: int = 5,
    alpha: float = 0.05,
    tau_min: float = 0.05,
    tau_max: float = 5.0,
) -> dict[str, Any]:
    """Calibration-split-only K-fold CV over (gamma, tau) minimizing held-out NLL & set size.

    Zero test-split leakage: strictly inspects calibration rows only.
    """
    n = len(logprob_rows)
    if n < 10 or n != len(token_count_rows) or n != len(gold_indices):
        raise ValueError("Need >=10 equal-length calibration rows for CV selection")
    k_folds = max(2, min(int(n_folds), n // 2))

    candidates: list[dict[str, Any]] = []
    for gamma in gammas:
        norm_rows = length_normalized_scores(logprob_rows, token_count_rows, gamma=float(gamma))
        val_nlls: list[float] = []
        val_set_sizes: list[float] = []
        for fold in range(k_folds):
            tr_idx = [i for i in range(n) if (i % k_folds) != fold]
            va_idx = [i for i in range(n) if (i % k_folds) == fold]
            tr_rows = [norm_rows[i] for i in tr_idx]
            tr_golds = [gold_indices[i] for i in tr_idx]
            va_rows = [norm_rows[i] for i in va_idx]
            va_golds = [gold_indices[i] for i in va_idx]

            ts = fit_temperature_scaling_nll(tr_rows, tr_golds, tau_min=tau_min, tau_max=tau_max)
            tau_f = float(ts["tau_opt"])
            tr_nc = [1.0 - softmax_with_temperature(r, tau=tau_f)[g] for r, g in zip(tr_rows, tr_golds)]
            q_hat_f, _, _ = compute_split_conformal_qhat(tr_nc, alpha=alpha)
            ev = evaluate_conformal_sets(va_rows, va_golds, tau=tau_f, q_hat=q_hat_f)
            val_nlls.append(float(ev["nll"]))
            val_set_sizes.append(float(ev["mean_prediction_set_size"]))

        full_fit = fit_temperature_scaling_nll(norm_rows, gold_indices, tau_min=tau_min, tau_max=tau_max)
        candidates.append(
            {
                "gamma": float(gamma),
                "tau_opt": float(full_fit["tau_opt"]),
                "cv_mean_nll": round(sum(val_nlls) / float(len(val_nlls)), 6),
                "cv_mean_set_size": round(sum(val_set_sizes) / float(len(val_set_sizes)), 4),
                "full_cal_ece": float(full_fit["ece_after"]),
            }
        )

    best = min(candidates, key=lambda c: (c["cv_mean_nll"], c["cv_mean_set_size"]))
    return {
        "best_gamma": best["gamma"],
        "best_tau": best["tau_opt"],
        "cv_folds": k_folds,
        "alpha": float(alpha),
        "candidates": candidates,
        "zero_test_leakage": True,
    }


def fit_mondrian_conformal_groups(
    prob_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    group_labels: Sequence[str],
    *,
    alpha: float = 0.05,
    min_group_size_warn: int = 250,
) -> dict[str, Any]:
    """Fit exact finite-sample Mondrian Split Conformal quantiles q_hat[g] at fixed alpha."""
    n = len(prob_rows)
    if n == 0 or n != len(gold_indices) or n != len(group_labels):
        raise ValueError("prob_rows, gold_indices, and group_labels must be non-empty and equal length")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    grouped_scores: dict[str, list[float]] = {}
    all_scores: list[float] = []
    for probs, g_idx, grp in zip(prob_rows, gold_indices, group_labels):
        if not (0 <= g_idx < len(probs)):
            raise IndexError(f"gold index {g_idx} out of range for row of length {len(probs)}")
        s = max(0.0, min(1.0, 1.0 - float(probs[g_idx])))
        grouped_scores.setdefault(str(grp), []).append(s)
        all_scores.append(s)

    global_q, global_k, global_ql = compute_split_conformal_qhat(all_scores, alpha=alpha)
    group_q_hat: dict[str, float] = {}
    group_stats: dict[str, dict[str, Any]] = {}
    for grp, scores in sorted(grouped_scores.items()):
        q_g, k_g, ql_g = compute_split_conformal_qhat(scores, alpha=alpha)
        group_q_hat[grp] = round(q_g, 6)
        group_stats[grp] = {
            "n_cal": len(scores),
            "meets_min_n": len(scores) >= int(min_group_size_warn),
            "order_statistic_k": k_g,
            "q_level": round(ql_g, 6),
            "q_hat": round(q_g, 6),
            "p_min_threshold": round(1.0 - q_g, 6),
        }

    return {
        "alpha": float(alpha),
        "target_coverage": round(1.0 - float(alpha), 6),
        "n_cal_total": n,
        "global_q_hat": round(global_q, 6),
        "global_order_statistic_k": global_k,
        "global_q_level": round(global_ql, 6),
        "group_q_hat": group_q_hat,
        "groups": group_stats,
    }


def evaluate_mondrian_conformal(
    prob_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    group_labels: Sequence[str],
    group_q_hat: Mapping[str, float],
    *,
    fallback_q_hat: float | None = None,
) -> dict[str, Any]:
    """Evaluate group-conditional coverage, mean set size, singleton rate, and ECE."""
    n = len(prob_rows)
    if n == 0 or n != len(gold_indices) or n != len(group_labels):
        raise ValueError("prob_rows, gold_indices, and group_labels must be non-empty and equal length")

    fb_q = float(fallback_q_hat) if fallback_q_hat is not None else max(float(v) for v in group_q_hat.values())
    per_group_rows: dict[str, list[tuple[Sequence[float], int, float]]] = {}
    all_covered = 0
    all_sizes: list[int] = []

    for probs, g_idx, grp in zip(prob_rows, gold_indices, group_labels):
        q_g = float(group_q_hat.get(str(grp), fb_q))
        p_min = 1.0 - q_g
        hit = 1 if float(probs[g_idx]) >= p_min else 0
        sz = sum(1 for p in probs if float(p) >= p_min)
        all_covered += hit
        all_sizes.append(sz)
        per_group_rows.setdefault(str(grp), []).append((probs, g_idx, p_min))

    group_metrics: dict[str, dict[str, Any]] = {}
    for grp, items in sorted(per_group_rows.items()):
        m = len(items)
        g_cov = sum(1 for probs, g_idx, p_min in items if float(probs[g_idx]) >= p_min)
        g_sizes = [sum(1 for p in probs if float(p) >= p_min) for probs, _, p_min in items]
        g_ece = compute_ece([probs for probs, _, _ in items], [g_idx for _, g_idx, _ in items])
        group_metrics[grp] = {
            "n_eval": m,
            "q_hat": round(float(group_q_hat.get(grp, fb_q)), 6),
            "empirical_coverage_pct": round(100.0 * g_cov / float(m), 2),
            "mean_prediction_set_size": round(sum(g_sizes) / float(m), 3),
            "singleton_exec_pct": round(100.0 * sum(1 for s in g_sizes if s == 1) / float(m), 2),
            "clarify_multi_pct": round(100.0 * sum(1 for s in g_sizes if s > 1) / float(m), 2),
            "abstain_empty_pct": round(100.0 * sum(1 for s in g_sizes if s == 0) / float(m), 2),
            "ece": round(g_ece, 6),
        }

    return {
        "n_eval": n,
        "overall_coverage_pct": round(100.0 * all_covered / float(n), 2),
        "overall_mean_set_size": round(sum(all_sizes) / float(n), 3),
        "overall_singleton_exec_pct": round(100.0 * sum(1 for s in all_sizes if s == 1) / float(n), 2),
        "overall_ece": round(compute_ece(prob_rows, gold_indices), 6),
        "group_metrics": group_metrics,
    }


def bootstrap_resplit_mondrian_coverage(
    prob_rows: Sequence[Sequence[float]],
    gold_indices: Sequence[int],
    group_labels: Sequence[str],
    *,
    alpha: float = 0.05,
    n_resplits: int = 100,
    cal_fraction: float = 0.625,
    seed: int = 42,
) -> dict[str, Any]:
    """Run stratified Monte Carlo calibration/test resplits (default 100) for Paper 2 CIs."""
    n = len(prob_rows)
    if n < 20 or n != len(gold_indices) or n != len(group_labels):
        raise ValueError("Need >=20 equal-length rows for Monte Carlo resplit audit")
    if n_resplits < 1:
        raise ValueError("n_resplits must be >= 1")

    rng = random.Random(seed)
    by_group: dict[str, list[int]] = {}
    for idx, grp in enumerate(group_labels):
        by_group.setdefault(str(grp), []).append(idx)

    grp_cov_series: dict[str, list[float]] = {g: [] for g in by_group}
    grp_sz_series: dict[str, list[float]] = {g: [] for g in by_group}
    grp_sing_series: dict[str, list[float]] = {g: [] for g in by_group}
    overall_cov: list[float] = []
    overall_sz: list[float] = []

    for _ in range(n_resplits):
        cal_idx: list[int] = []
        test_idx: list[int] = []
        for grp, idxs in sorted(by_group.items()):
            perm = list(idxs)
            rng.shuffle(perm)
            split_k = max(2, min(len(perm) - 2, int(round(len(perm) * cal_fraction))))
            cal_idx.extend(perm[:split_k])
            test_idx.extend(perm[split_k:])

        fit = fit_mondrian_conformal_groups(
            [prob_rows[i] for i in cal_idx],
            [gold_indices[i] for i in cal_idx],
            [group_labels[i] for i in cal_idx],
            alpha=alpha,
        )
        ev = evaluate_mondrian_conformal(
            [prob_rows[i] for i in test_idx],
            [gold_indices[i] for i in test_idx],
            [group_labels[i] for i in test_idx],
            fit["group_q_hat"],
            fallback_q_hat=fit["global_q_hat"],
        )
        overall_cov.append(ev["overall_coverage_pct"] / 100.0)
        overall_sz.append(ev["overall_mean_set_size"])
        for grp, gm in ev["group_metrics"].items():
            grp_cov_series[grp].append(gm["empirical_coverage_pct"] / 100.0)
            grp_sz_series[grp].append(gm["mean_prediction_set_size"])
            grp_sing_series[grp].append(gm["singleton_exec_pct"] / 100.0)

    def _mean_sd(vals: Sequence[float]) -> tuple[float, float]:
        mu = sum(vals) / float(len(vals))
        var = sum((v - mu) ** 2 for v in vals) / float(max(1, len(vals) - 1))
        return mu, math.sqrt(max(var, 0.0))

    ov_cov_mu, ov_cov_sd = _mean_sd(overall_cov)
    ov_sz_mu, ov_sz_sd = _mean_sd(overall_sz)
    group_summary: dict[str, dict[str, float]] = {}
    for grp in sorted(by_group):
        c_mu, c_sd = _mean_sd(grp_cov_series[grp])
        s_mu, s_sd = _mean_sd(grp_sz_series[grp])
        sg_mu, sg_sd = _mean_sd(grp_sing_series[grp])
        cov_sorted = sorted(grp_cov_series[grp])
        ci_lo = cov_sorted[max(0, int(0.025 * len(cov_sorted)))]
        ci_hi = cov_sorted[min(len(cov_sorted) - 1, int(0.975 * len(cov_sorted)))]
        group_summary[grp] = {
            "coverage_mean": round(c_mu, 4),
            "coverage_sd": round(c_sd, 4),
            "coverage_ci95_lo": round(ci_lo, 4),
            "coverage_ci95_hi": round(ci_hi, 4),
            "mean_set_size_mean": round(s_mu, 4),
            "mean_set_size_sd": round(s_sd, 4),
            "singleton_rate_mean": round(sg_mu, 4),
            "singleton_rate_sd": round(sg_sd, 4),
        }

    return {
        "alpha": float(alpha),
        "n_resplits": int(n_resplits),
        "cal_fraction": float(cal_fraction),
        "seed": int(seed),
        "overall_coverage_mean": round(ov_cov_mu, 4),
        "overall_coverage_sd": round(ov_cov_sd, 4),
        "overall_mean_set_size_mean": round(ov_sz_mu, 4),
        "overall_mean_set_size_sd": round(ov_sz_sd, 4),
        "group_metrics": group_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate KnowNo Split Conformal quantile q_hat for VertiRTA."
    )
    parser.add_argument("--calibration", type=Path, required=True, help="Path to calibration.jsonl")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("config/conformal_calibration.json"),
        help="Output JSON path",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--expected-n-cal", type=int, default=200)
    args = parser.parse_args()

    res = calibrate_from_jsonl(
        args.calibration,
        args.out,
        alpha=args.alpha,
        expected_n_cal=args.expected_n_cal,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
