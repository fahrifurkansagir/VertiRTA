# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for exact layer-streaming KV-prefix-shared Qwen3-4B v3 RAPS exporter."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs.job_qwen3_v3_vertibench_v2 import (  # noqa: E402
    _load_jsonl,
    enrich_vertibench_v2_row_for_z3,
)
from scripts.ml.run_local_qwen3_v3_conformal_raps import (  # noqa: E402
    NUM_BRANCHES,
    _atomic_save_npz,
    build_leakfree_candidate_branches,
    gold_branch_key_leakfree,
    run_self_test_cpu,
    run_zero_leakage_mondrian_raps_evaluation,
)


def test_leakfree_14_branches_cover_100pct_of_vertibench_v2_without_peeking_at_gold() -> None:
    vb_dir = REPO_ROOT / "benchmarks/vertibench_v2"
    for split_name in ("cal", "test"):
        rows = _load_jsonl(vb_dir / f"{split_name}.jsonl")
        for row in rows[:200]:
            er = enrich_vertibench_v2_row_for_z3(row)
            # Strip ALL gold fields before calling branch builder to prove zero gold leakage!
            runtime_only = {
                "lang": er["lang"],
                "snr_stratum": er["snr_stratum"],
                "rta_snapshot": er["rta_snapshot"],
            }
            branches = build_leakfree_candidate_branches(runtime_only)
            assert len(branches) == NUM_BRANCHES
            keys = [b["key"] for b in branches]
            assert len(set(keys)) == NUM_BRANCHES
            # Every decision_prefix must be mutually prefix-free and a prefix of full_dsl_eos
            prefixes = [b["decision_prefix"] for b in branches]
            for i, p_i in enumerate(prefixes):
                assert branches[i]["full_dsl_eos"].startswith(p_i)
                assert branches[i]["full_dsl_eos"].endswith("<|im_end|>")
                for j, p_j in enumerate(prefixes):
                    if i != j:
                        assert not p_j.startswith(p_i), f"Prefix collision: {p_i!r} vs {p_j!r}"
            g_key = gold_branch_key_leakfree(er)
            assert g_key in keys, f"{split_name} gold key {g_key!r} missing from {keys}"


def test_layer_streaming_kv_prefix_sharing_matches_monolithic_qwen3_under_1e5() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    res = run_self_test_cpu()
    assert res["self_test_passed"] is True
    assert res["max_abs_diff_full_sum_lp"] < 1e-5
    assert res["max_abs_diff_decision_sum_lp"] < 1e-5
    assert res["max_abs_diff_gbnf_tree_lp"] < 1e-5
    assert res["max_abs_diff_raw_split_lp"] < 1e-5
    assert res["max_gbnf_prob_sum_err"] < 1e-6


def test_zero_leakage_mondrian_raps_pipeline_and_atomic_npz_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(2026)
    n_cal, n_test = 240, 120
    langs = np.array(["TR", "EN", "MIX"])
    snrs = np.array(["S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"])
    buckets = np.array(["NOMINAL", "FAULT", "ADVERSARIAL", "AMBIGUOUS_ASR"])

    def _synth_split(n: int) -> dict[str, np.ndarray]:
        golds = rng.integers(0, NUM_BRANCHES, size=n, dtype=np.int32)
        dec_nt = rng.integers(7, 17, size=(n, NUM_BRANCHES), dtype=np.int32)
        full_nt = rng.integers(28, 48, size=(n, NUM_BRANCHES), dtype=np.int32)
        dec_lp = rng.normal(-6.0, 0.8, size=(n, NUM_BRANCHES))
        full_lp = dec_lp - rng.uniform(1.0, 3.0, size=(n, NUM_BRANCHES))
        for i, g in enumerate(golds):
            dec_lp[i, g] = rng.normal(-0.8, 0.3)
            full_lp[i, g] = dec_lp[i, g] - rng.uniform(1.5, 2.5)
        return {
            "dec_lp": dec_lp,
            "dec_ntok": dec_nt,
            "full_lp": full_lp,
            "full_ntok": full_nt,
            "golds": golds,
            "langs": np.array([langs[i % 3] for i in range(n)]),
            "snrs": np.array([snrs[i % 4] for i in range(n)]),
            "buckets": np.array([buckets[i % 4] for i in range(n)]),
        }

    cal_s = _synth_split(n_cal)
    test_s = _synth_split(n_test)
    state = {
        **{f"cal_{k}": v for k, v in cal_s.items()},
        **{f"test_{k}": v for k, v in test_s.items()},
        "cal_done": np.array(n_cal, dtype=np.int32),
        "test_done": np.array(n_test, dtype=np.int32),
        "score_scope": np.array("both"),
        "layer_dtype": np.array("float32"),
    }

    npz_file = tmp_path / "ckpt.npz"
    _atomic_save_npz(npz_file, state)
    assert npz_file.is_file()
    with np.load(npz_file, allow_pickle=False) as loaded:
        reloaded = {k: loaded[k].copy() for k in loaded.files}

    ev = run_zero_leakage_mondrian_raps_evaluation(reloaded, alpha=0.05, seed=42)
    assert ev["zero_test_leakage_verified"] is True
    assert ev["lang_mondrian_raps_test"]["overall_coverage_pct"] >= 90.0
    assert ev["lang_mondrian_raps_test"]["overall_mean_set_size"] < 2.0
    assert ev["lang_mondrian_raps_test"]["overall_singleton_exec_pct"] > 70.0
    assert ev["snr_mondrian_raps_test"]["overall_coverage_pct"] >= 90.0
    assert json.dumps(ev, allow_nan=False)


def test_sealed_local_gtx1650s_qwen3_conformal_raps_artifact_integrity() -> None:
    import hashlib

    art_path = REPO_ROOT / "artifacts/v4.0.0/local_gtx1650s_qwen3_v3_conformal_raps.json"
    assert art_path.is_file(), f"Missing sealed local Qwen3 RAPS artifact: {art_path}"
    raw = art_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    doc = json.loads(raw.decode("utf-8"))

    # Canonical SHA-256 integrity seal
    recorded_sha = doc["report_sha256"]
    body = {k: v for k, v in doc.items() if k != "report_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded_sha == expected_sha

    assert doc["colab_cu_consumed"] == 0.0
    assert doc["quantization_loss_zero"] is True
    assert doc["zero_test_leakage_verified"] is True
    assert doc["split_sizes"] == {"n_cal": 2000, "n_test": 1200}
    assert doc["host_platform"]["peak_vram_mib"] < 950.0
    assert doc["monolithic_spot_check_parity"]["top1_agreement_pct"] == 100.0
    assert doc["summary"]["lang_mondrian_raps_coverage_pct"] >= 94.0
    assert doc["summary"]["snr_mondrian_raps_coverage_pct"] >= 94.0
    assert doc["summary"]["lang_mondrian_raps_ece"] < 0.05
    assert doc["summary"]["snr_mondrian_raps_mean_set_size"] < 2.5
    assert doc["paired_lac_on_fixed_scores"]["snr_mondrian_lac_test"]["overall_mean_set_size"] < 2.2
    assert doc["paired_lac_on_fixed_scores"]["snr_mondrian_lac_test"]["overall_singleton_exec_pct"] > 40.0
