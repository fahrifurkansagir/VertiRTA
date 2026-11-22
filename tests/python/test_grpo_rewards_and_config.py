"""Tests pinning DR-6 Errata #2 & #3 for Colab G4 Z3-GRPO (RLVR) rewards & policy config."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.copilot.grpo_z3_rewards import (
    PersistentZ3RewardPool,
    VIOLATION_TO_Z3_LABEL,
    score_single_completion,
    violation_matches_expected,
)
from ml.copilot.train_sft_grpo import GrpoPolicyConfig, build_training_plan_summary


def _make_ref_400v_snapshot() -> dict[str, Any]:
    return {
        "schema_version": "vertirta.rta_snapshot.v1",
        "pack_nominal_volts": 400.0,
        "pack_energy_kwh": 22.0,
        "fault_mode": "SINGLE_MOTOR_LOSS",
        "failed_motor_mask": 4,
        "shield_active": True,
        "ownship": {
            "enu_m": [-1150.0, 950.0, 65.0],
            "agl_m": 65.0,
            "soc_pct": 34.0,
            "soc_reserve_floor_pct": 12.0,
            "v_term_volts": 368.5,
            "wind_mps": 5.0,
            "wind_from_deg": 240.0,
        },
        "degraded_ams": {
            "ams_max_accel_mps2": 3.2,
            "ams_max_climb_mps": 2.5,
            "max_cruise_airspeed_mps": 18.0,
            "burn_rate_soc_pct_per_s": 0.140,
        },
        "active_exclusions": ["LTFJ_CTR", "LTBX_CTR"],
        "dynamic_nfzs": [],
        "ranked_slzs": [
            {
                "rank": 1,
                "id": "Kartal_Sahil",
                "enu_m": [0.0, 0.0, 0.0],
                "p_safe": 0.94,
                "slz_ood_variance": 0.03,
                "eta_s": 100.0,
                "soc_margin_pct": 8.0,
            },
            {
                "rank": 2,
                "id": "Cevizli_Helipad",
                "enu_m": [-1348.1, 1943.5, 5.6],
                "p_safe": 0.89,
                "slz_ood_variance": 0.06,
                "eta_s": 75.4,
                "soc_margin_pct": 11.4,
            },
        ],
    }


def _make_b3_pendik_meta() -> dict[str, Any]:
    return {
        "rta_snapshot": _make_ref_400v_snapshot(),
        "bucket": "ADVERSARIAL",
        "expected_lang": "TR",
        "gold_intent": {
            "expected_action": "REJECT_UNSAFE",
            "valid_targets": ["Kartal_Sahil", "Cevizli_Helipad"],
            "expected_unsat_labels": ["c_ltfj_ctr", "c_soc_reserve"],
            "requires_clarification": False,
            "allow_loiter": False,
        },
    }


def _make_b4_garbled_meta() -> dict[str, Any]:
    return {
        "rta_snapshot": _make_ref_400v_snapshot(),
        "bucket": "AMBIGUOUS_ASR",
        "expected_lang": "TR",
        "gold_intent": {
            "expected_action": "HOLD_LOITER",
            "valid_targets": ["Kartal_Sahil", "Cevizli_Helipad"],
            "expected_unsat_labels": [],
            "requires_clarification": False,
            "allow_loiter": True,
        },
    }


def _make_b4_ambig_meta() -> dict[str, Any]:
    return {
        "rta_snapshot": _make_ref_400v_snapshot(),
        "bucket": "AMBIGUOUS_ASR",
        "expected_lang": "TR",
        "gold_intent": {
            "expected_action": "REQUEST_CLARIFICATION",
            "valid_targets": ["Kartal_Sahil", "Cevizli_Helipad"],
            "top2_sat_pads": ["Kartal_Sahil", "Cevizli_Helipad"],
            "expected_unsat_labels": [],
            "requires_clarification": True,
            "allow_loiter": True,
        },
    }


def test_r_z3_maps_uppercase_dsl_violation_to_lowercase_z3_label() -> None:
    assert VIOLATION_TO_Z3_LABEL["LTFJ_CTR"] == "c_ltfj_ctr"
    assert violation_matches_expected("DYN_NFZ", ["c_dyn_nfz_0", "c_dyn_nfz_1"])
    score = score_single_completion(
        "ACT=REJECT_UNSAFE;VIO=LTFJ_CTR;FB=Kartal_Sahil\nLNG=TR;MOD=REJECT",
        _make_b3_pendik_meta(),
    )
    assert score >= 0.85


def test_null_target_on_loiter_and_clarification_is_safe_and_think_penalized() -> None:
    loiter_score = score_single_completion(
        "ACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF",
        _make_b4_garbled_meta(),
    )
    assert 0.80 <= loiter_score <= 1.0

    clarify_score = score_single_completion(
        "ACT=REQUEST_CLARIFICATION;A=Kartal_Sahil;B=Cevizli_Helipad;RSN=AMBIG\nLNG=TR;MOD=CLARIFY",
        _make_b4_ambig_meta(),
    )
    assert 0.80 <= clarify_score <= 1.0

    assert score_single_completion("<think>bad</think>\nACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF", _make_b4_garbled_meta()) == -0.25


def test_grpo_clip_bounds_and_spawn_reward_pool(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="GSPO"):
        GrpoPolicyConfig(
            importance_sampling_level="sequence",
            epsilon=0.20,
            epsilon_high=0.28,
            beta=0.04,
        )
    with pytest.raises(ValueError, match="banned"):
        GrpoPolicyConfig(model_id="Qwen/Qwen2.5-7B-Instruct")
    with pytest.raises(ValueError, match="KL beta"):
        GrpoPolicyConfig(beta=0.0)

    ok_dr = GrpoPolicyConfig(
        model_id="Qwen/Qwen3-8B",
        importance_sampling_level="token",
        loss_type="dr_grpo",
        epsilon=0.20,
        epsilon_high=0.28,
        beta=0.04,
    )
    ok_gspo = GrpoPolicyConfig(
        model_id="Qwen/Qwen3-4B",
        importance_sampling_level="sequence",
        loss_type="dr_grpo",
        epsilon=3e-4,
        epsilon_high=4e-4,
        beta=0.04,
    )
    assert ok_dr.beta == 0.04 and ok_dr.sft_learning_rate == 1.0e-4
    assert ok_gspo.epsilon_high == 4e-4 and ok_gspo.sft_learning_rate == 1.5e-4

    plan = build_training_plan_summary(ok_dr, tmp_path / "train.jsonl", tmp_path / "out")
    assert plan["stage2_grpo"]["z3_reward_pool_start_method"] == "spawn"
    assert plan["stage2_grpo"]["beta"] == 0.04

    with PersistentZ3RewardPool(max_workers=2) as pool:
        assert pool.start_method == "spawn"
        out = pool.score_batch(
            [
                "ACT=HOLD_LOITER;DUR=30;AGL=60\nLNG=TR;MOD=BRIEF",
                "ACT=REJECT_UNSAFE;VIO=LTFJ_CTR;FB=Kartal_Sahil\nLNG=TR;MOD=REJECT",
            ],
            [_make_b4_garbled_meta(), _make_b3_pendik_meta()],
        )
        assert len(out) == 2
        assert out[0] >= 0.80 and out[1] >= 0.85
