"""Split Conformal Prediction calibration (KnowNo-style) for VertiRTA Co-Pilot.

Computes the finite-sample quantile q_hat = Quantile({s_i}; ceil((n_cal + 1)*(1 - alpha)) / n_cal)
over human-audited calibration non-conformity scores s_i = 1 - p_hat(y_i* | x_i) and writes
config/conformal_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


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
