# SPDX-License-Identifier: Apache-2.0
"""Pure-Python CI-safe unit tests & synthetic ablation verification for APS/RAPS & Mondrian RAPS.

Zero torch / cv2 dependencies. Strictly separates deterministic mathematical verification
(`SYNTHETIC_ABLATION_VERIFYING_RAPS_MATH`) from live Qwen3-4B v3 G4 results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.copilot.calibrate_conformal import (
    compute_raps_nonconformity_score,
    evaluate_conformal_sets,
    evaluate_mondrian_conformal,
    evaluate_mondrian_raps,
    evaluate_raps_conformal,
    evaluate_temperature_scaled_mondrian_raps,
    fit_and_calibrate_conformal,
    fit_mondrian_conformal_groups,
    fit_mondrian_raps_groups,
    fit_raps_conformal,
    fit_temperature_scaled_mondrian_raps,
    fit_temperature_scaled_raps,
    predict_raps_set,
    softmax_with_temperature,
)


def _generate_synthetic_14branch_logits(
    n_rows: int,
    *,
    seed: int,
    n_classes: int = 14,
) -> tuple[list[list[float]], list[int], list[str]]:
    """SYNTHETIC_ABLATION_VERIFYING_RAPS_MATH: 14-branch boilerplate-diluted logits + hard tail."""
    rng = random.Random(seed)
    strata = ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
    # Models gamma=1.0 boilerplate dilution (25x compressed logit scale) + genuine top-2/top-3 ambiguity tail
    stratum_scale = {"S0_Clean": 0.09, "S1_Cruise": 0.08, "S2_Hover": 0.07, "S3_Garbled": 0.06}
    stratum_hard_rate = {"S0_Clean": 0.04, "S1_Cruise": 0.06, "S2_Hover": 0.08, "S3_Garbled": 0.10}

    score_rows: list[list[float]] = []
    gold_indices: list[int] = []
    groups: list[str] = []

    for i in range(n_rows):
        grp = strata[i % len(strata)]
        gold = rng.randrange(n_classes)
        scale = stratum_scale[grp]
        is_hard = rng.random() < stratum_hard_rate[grp]

        # Compressed logits (boilerplated 14-branch DSL)
        logits = [rng.gauss(-0.6, 0.25) * scale for _ in range(n_classes)]
        if is_hard:
            # Ambiguous top-2 pair (e.g. adjacent SLZs / similar violation codes): gold sits at rank 2
            rival = (gold + 1) % n_classes
            logits[rival] = (3.6 + rng.gauss(0.0, 0.25)) * scale
            logits[gold] = (2.3 + rng.gauss(0.0, 0.25)) * scale
        else:
            # Clear nominal / unambiguous sample before gamma=1.0 compression
            logits[gold] = (4.8 + rng.gauss(0.0, 0.30)) * scale
            logits[(gold + 1) % n_classes] = (0.3 + rng.gauss(0.0, 0.25)) * scale
        score_rows.append(logits)
        gold_indices.append(gold)
        groups.append(grp)
    return score_rows, gold_indices, groups


def test_raps_score_monotonicity_aps_equivalence_and_determinism() -> None:
    probs = [0.05, 0.65, 0.20, 0.08, 0.02]
    # Sorted order: idx 1 (0.65), idx 2 (0.20), idx 3 (0.08), idx 0 (0.05), idx 4 (0.02)
    s_aps = [compute_raps_nonconformity_score(probs, idx, k_reg=1, lam_reg=0.0) for idx in (1, 2, 3, 0, 4)]
    assert s_aps[0] == pytest.approx(0.0)
    assert s_aps[1] == pytest.approx(0.65)
    assert s_aps[2] == pytest.approx(0.85)
    assert s_aps[3] == pytest.approx(0.93)
    assert s_aps[4] == pytest.approx(0.98)
    assert all(s_aps[i] <= s_aps[i + 1] for i in range(len(s_aps) - 1))

    # RAPS adds lam_reg * max(0, rank - k_reg)
    s_raps = [compute_raps_nonconformity_score(probs, idx, k_reg=1, lam_reg=0.10) for idx in (1, 2, 3, 0, 4)]
    assert s_raps[0] == pytest.approx(0.0)
    assert s_raps[1] == pytest.approx(0.65 + 0.10)
    assert s_raps[2] == pytest.approx(0.85 + 0.20)
    assert all(s_raps[i] < s_raps[i + 1] for i in range(len(s_raps) - 1))

    # Prediction set shrinks under RAPS penalty at same q_hat=0.90
    set_aps = predict_raps_set(probs, q_hat=0.90, k_reg=1, lam_reg=0.0)
    set_raps = predict_raps_set(probs, q_hat=0.90, k_reg=1, lam_reg=0.10)
    assert set_aps == [1, 2, 3]
    assert set_raps == [1, 2]

    # Randomized mode exact reproducibility across identical seeds
    cal_scores, cal_golds, _ = _generate_synthetic_14branch_logits(400, seed=101)
    cal_probs = [softmax_with_temperature(r, tau=0.20) for r in cal_scores]
    f1 = fit_raps_conformal(cal_probs, cal_golds, alpha=0.05, k_reg=1, lam_reg=0.05, randomized=True, seed=77)
    f2 = fit_raps_conformal(cal_probs, cal_golds, alpha=0.05, k_reg=1, lam_reg=0.05, randomized=True, seed=77)
    assert f1 == f2


def test_finite_sample_coverage_and_synthetic_ablation_lac_vs_aps_vs_raps() -> None:
    """Verify >= 1-alpha coverage and RAPS set-size / singleton gains on SYNTHETIC_ABLATION_VERIFYING_RAPS_MATH."""
    cal_scores, cal_golds, cal_groups = _generate_synthetic_14branch_logits(2000, seed=20261124)
    test_scores, test_golds, test_groups = _generate_synthetic_14branch_logits(1200, seed=20261125)

    # 1) Uncalibrated naive LAC (tau=1.0) suffers massive miscalibration (ECE > 0.70)
    lac_bundle = fit_and_calibrate_conformal(cal_scores, cal_golds, alpha=0.05)
    ev_lac_uncal = evaluate_conformal_sets(
        test_scores, test_golds, tau=1.0, q_hat=lac_bundle["uncalibrated"]["q_hat"]
    )
    assert ev_lac_uncal["empirical_coverage_pct"] >= 93.0
    assert ev_lac_uncal["ece"] > 0.70

    # 2) Disjoint temperature-scaled APS (lam_reg=0.0) & RAPS (k_reg=1, lam_reg=0.05)
    aps_fit = fit_temperature_scaled_raps(
        cal_scores, cal_golds, alpha=0.05, k_reg=1, lam_reg=0.0, randomized=False, split_disjoint_half=True
    )
    raps_fit = fit_temperature_scaled_raps(
        cal_scores, cal_golds, alpha=0.05, k_reg=1, lam_reg=0.05, randomized=False, split_disjoint_half=True
    )
    raps_rand_fit = fit_temperature_scaled_raps(
        cal_scores, cal_golds, alpha=0.05, k_reg=1, lam_reg=0.05, randomized=True, seed=42, split_disjoint_half=True
    )
    assert aps_fit["zero_test_leakage"] is True
    assert aps_fit["split_policy"] == "disjoint_even_odd_50_50"
    assert raps_fit["n_tau_fit"] == 1000 and raps_fit["n_qhat_cal"] == 1000

    test_probs_aps = [softmax_with_temperature(r, tau=aps_fit["tau_opt"]) for r in test_scores]
    test_probs_raps = [softmax_with_temperature(r, tau=raps_fit["tau_opt"]) for r in test_scores]

    ev_aps = evaluate_raps_conformal(
        test_probs_aps, test_golds, q_hat=aps_fit["q_hat"], k_reg=1, lam_reg=0.0, randomized=False
    )
    ev_raps = evaluate_raps_conformal(
        test_probs_raps, test_golds, q_hat=raps_fit["q_hat"], k_reg=1, lam_reg=0.05, randomized=False
    )
    ev_raps_rand = evaluate_raps_conformal(
        test_probs_raps, test_golds, q_hat=raps_rand_fit["q_hat"], k_reg=1, lam_reg=0.05, randomized=True, seed=99
    )

    # Exact finite-sample coverage >= 1 - alpha (within binomial tolerance on n=1200)
    assert ev_aps["empirical_coverage_pct"] >= 94.0
    assert ev_raps["empirical_coverage_pct"] >= 94.0
    assert ev_raps_rand["empirical_coverage_pct"] >= 93.5

    # Temperature-scaled RAPS slashes ECE > 10x, keeps tight |C| < 2.0, and yields > 90% singletons
    assert ev_raps["ece"] < ev_lac_uncal["ece"] * 0.15
    assert ev_raps["mean_prediction_set_size"] < 2.0
    assert ev_raps["singleton_exec_pct"] > 90.0
    assert ev_raps["mean_prediction_set_size"] <= ev_aps["mean_prediction_set_size"] + 0.05

    # 3) Temperature-scaled Mondrian RAPS across all 4 SNR strata
    m_fit = fit_temperature_scaled_mondrian_raps(
        cal_scores,
        cal_golds,
        cal_groups,
        alpha=0.05,
        k_reg=1,
        lam_reg=0.05,
        randomized=False,
        per_group_temperature=True,
    )
    assert m_fit["zero_test_leakage"] is True
    assert set(m_fit["group_tau"].keys()) == {"S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"}
    m_ev = evaluate_temperature_scaled_mondrian_raps(test_scores, test_golds, test_groups, m_fit)
    assert m_ev["overall_coverage_pct"] >= 94.0
    assert m_ev["overall_singleton_exec_pct"] > 75.0
    for grp, gm in m_ev["group_metrics"].items():
        assert gm["empirical_coverage_pct"] >= 91.5, f"Group {grp} under-covered: {gm}"
        assert gm["mean_prediction_set_size"] < 2.5

    # 4) Direct Mondrian RAPS API backward-compatible comparison vs Mondrian LAC
    cal_probs_t = [softmax_with_temperature(r, tau=raps_fit["tau_opt"]) for r in cal_scores]
    lac_m_fit = fit_mondrian_conformal_groups(cal_probs_t, cal_golds, cal_groups, alpha=0.05)
    lac_m_ev = evaluate_mondrian_conformal(test_probs_raps, test_golds, test_groups, lac_m_fit["group_q_hat"])
    raps_m_fit = fit_mondrian_raps_groups(cal_probs_t, cal_golds, cal_groups, alpha=0.05, k_reg=1, lam_reg=0.05)
    raps_m_ev = evaluate_mondrian_raps(
        test_probs_raps, test_golds, test_groups, raps_m_fit["group_q_hat"], k_reg=1, lam_reg=0.05
    )
    assert lac_m_ev["overall_coverage_pct"] >= 93.0
    assert raps_m_ev["overall_coverage_pct"] >= 94.0
    assert raps_m_ev["overall_ece"] < 0.08


def test_qwen3_v3_conformal_raps_diagnostic_artifact_is_honest_and_sha256_sealed() -> None:
    diag_path = REPO_ROOT / "artifacts/v4.0.0/qwen3_v3_conformal_raps_diagnostic.json"
    if not diag_path.is_file():
        pytest.skip("Diagnostic artifact not yet generated")
    raw = diag_path.read_bytes()
    assert len(raw) < 80 * 1024
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    doc = json.loads(raw.decode("utf-8"))
    assert doc["schema_version"] == "vertirta.qwen3_v3_conformal_raps_diagnostic/v1.0.0"
    assert doc["post_hoc_recalibration_possible_on_saved_artifacts"] is False
    assert doc["synthetic_raps_ablation"]["provenance"] == "SYNTHETIC_ABLATION_VERIFYING_RAPS_MATH"
    assert doc["synthetic_raps_ablation"]["is_live_qwen3_score"] is False
    assert len(doc["root_causes"]) >= 4

    # Verify SHA-256 seal matches canonical body without report_sha256
    recorded_sha = doc["report_sha256"]
    body = {k: v for k, v in doc.items() if k != "report_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded_sha == expected_sha


def test_mondrian_raps_disjoint_split_immune_to_periodic_parity_aliasing() -> None:
    """Regression guard: periodic even/odd difficulty alternation must NOT cause q_hat=0 under-coverage."""
    cal_scores: list[list[float]] = []
    cal_golds: list[int] = []
    cal_groups: list[str] = []
    # Pathological even=hard, odd=trivial ordering inside every group (breaks naive idxs[0::2] / idxs[1::2])
    for i in range(800):
        grp = "S0_Clean" if (i // 2) % 2 == 0 else "S1_Cruise"
        if i % 2 == 0:
            # Hard row: gold at rank 2
            logits = [-4.0] * 14
            logits[0] = 2.0
            logits[1] = 1.6
            cal_golds.append(1)
        else:
            # Trivial row: gold at rank 1 with high confidence
            logits = [-6.0] * 14
            logits[0] = 5.0
            cal_golds.append(0)
        cal_scores.append(logits)
        cal_groups.append(grp)

    fit = fit_temperature_scaled_mondrian_raps(
        cal_scores, cal_golds, cal_groups, alpha=0.05, k_reg=1, lam_reg=0.02, per_group_temperature=True, seed=42
    )
    for grp, q_g in fit["group_q_hat"].items():
        assert q_g > 0.1, f"Parity aliasing collapsed q_hat[{grp}]={q_g} on periodic calibration stream"
    ev = evaluate_temperature_scaled_mondrian_raps(cal_scores, cal_golds, cal_groups, fit, seed=43)
    assert ev["overall_coverage_pct"] >= 94.0
