"""Tests for VertiBench-TR/EN integrity, <0.15 Jaccard anti-contamination, license lock,
400 V / 22 kWh HV telemetry guard, programmatic builder, and deterministic 4-model eval.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vertibench_tr_en.build_train_split import build_train_split
from benchmarks.vertibench_tr_en.eval_vertibench import evaluate_predictions
from benchmarks.vertibench_tr_en.validate_vertibench import (
    is_model_allowed,
    max_char_5gram_jaccard,
    max_token_3gram_jaccard,
    parse_compact_dsl_py,
    validate_hv_telemetry,
    validate_vertibench,
)


def test_jaccard_gate_blocks_leaked_prompts_above_0_15() -> None:
    train = ["Motor 3 arizali, Kartal Sahil pistine hemen acil inis yap"]
    leaked = "Motor 3 arizali, Kartal Sahil pistine hemen inis yap"
    clean = "Uc numarali pervane durdu; kiyi pistine yonlenebilir miyiz?"
    assert max_char_5gram_jaccard(leaked, train) >= 0.15
    assert max_token_3gram_jaccard(leaked, train) >= 0.15
    assert max_char_5gram_jaccard(clean, train) < 0.15
    assert max_token_3gram_jaccard(clean, train) < 0.15


def test_license_allowlist_and_hv_telemetry_guards() -> None:
    assert is_model_allowed("Qwen/Qwen3-8B", "train")
    assert is_model_allowed("Qwen/Qwen3-4B", "train")
    assert not is_model_allowed("Qwen/Qwen2.5-7B-Instruct", "train")
    assert not is_model_allowed("google/gemma-3-4b-it", "train")
    assert is_model_allowed("Qwen/Qwen2.5-7B-Instruct", "zero_shot_eval")
    assert is_model_allowed("google/gemma-3-4b-it", "zero_shot_eval")
    assert not is_model_allowed("CohereForAI/aya-expanse-8b", "zero_shot_eval")
    assert not is_model_allowed("CohereForAI/aya-expanse-8b", "train")

    assert not validate_hv_telemetry({"ownship": {"v_term_volts": 44.2, "soc_pct": 34.0}})
    assert validate_hv_telemetry(
        {
            "pack_nominal_volts": 400.0,
            "pack_energy_kwh": 22.0,
            "ownship": {"v_term_volts": 368.5, "soc_pct": 34.0},
        }
    )


def test_validate_vertibench_check_schema_passes() -> None:
    report = validate_vertibench(check_schema_only=True)
    assert report.passed, f"Schema errors: {report.errors}"
    assert report.schema_only is True


def test_build_train_split_generates_z3_verified_400v_rows_in_tmp(tmp_path: Path) -> None:
    small_counts = {"NOMINAL": 8, "FAULT": 8, "ADVERSARIAL": 8, "AMBIGUOUS_ASR": 8}
    train_path = build_train_split(tmp_path, seed=2026, bucket_counts=small_counts)
    assert train_path.exists()
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 32
    for row in rows:
        assert validate_hv_telemetry(row["rta_snapshot"])
        assert parse_compact_dsl_py(row["canonical_dsl"]) is not None

    # Deterministic eval harness on generated rows (exact canonical DSL predictions -> 100% grammar/unsafe rejection)
    preds = [{"completion": r["canonical_dsl"], "latency_ms": 180.0, "gold_prob": 0.92} for r in rows]
    metrics = evaluate_predictions(
        "Qwen/Qwen3-8B",
        "sft_z3_grpo_gbnf_z3_conformal",
        rows,
        preds,
        q_hat=0.68,
    )
    assert metrics["grammar_validity_pct"] == 100.0
    assert metrics["unsafe_rejection_pct"] == 100.0
    assert metrics["conformal_coverage_pct"] == 100.0

    # Zero-shot baselines allowed ONLY in zero-shot stages; Aya-Expanse always rejected
    zs = evaluate_predictions("Qwen/Qwen2.5-7B-Instruct", "zero_shot_gbnf_z3", rows, preds)
    assert zs["unsafe_rejection_pct"] == 100.0
    with pytest.raises(ValueError, match="zero-shot"):
        evaluate_predictions("google/gemma-3-4b-it", "sft_z3_grpo", rows, preds)
    with pytest.raises(ValueError, match="banned"):
        evaluate_predictions("CohereForAI/aya-expanse-8b", "zero_shot", rows, preds)
