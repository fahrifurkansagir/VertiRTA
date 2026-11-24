# SPDX-License-Identifier: Apache-2.0
"""Verification tests for Live Local Offline Voice ATC Hardware Proof (E2, 0.00 CU)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PROOF_PATH = REPO_ROOT / "artifacts/v4.0.0/local_voice_atc_live_proof.json"
CANONICAL_STRATA = ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")


def _assert_finite_recursive(obj: Any, path: str = "root") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float at {path}: {obj}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_recursive(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite_recursive(v, f"{path}[{i}]")


def test_local_voice_live_proof_artifact_integrity_zero_gpl_ttfa_and_z3() -> None:
    assert PROOF_PATH.is_file(), f"Missing live hardware proof JSON: {PROOF_PATH}"
    raw = json.loads(PROOF_PATH.read_text(encoding="utf-8"))
    _assert_finite_recursive(raw)

    # 1) Verify embedded SHA-256 matches canonical JSON payload
    body = {k: v for k, v in raw.items() if k != "report_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert raw["report_sha256"] == expected_sha
    assert raw["mode"] == "LIVE_LOCAL_HARDWARE_PROOF"
    assert raw["colab_cu_consumed"] == 0.0

    # 2) Verify sherpa-onnx >= 2.0.0 Zero-GPL native binary proof & < 250 ms TTFA
    sherpa = raw["sherpa_onnx_zero_gpl_proof"]
    assert sherpa["semver_ge_2_0_0"] is True
    major = int(str(sherpa["sherpa_onnx_version"]).split(".")[0])
    assert major >= 2
    assert sherpa["zero_gpl_gate_passed"] is True
    assert sherpa["zero_gpl_violations"] == []
    assert len(sherpa["native_libs_scanned"]) >= 1
    assert sherpa["espeak_ng_data_deleted_and_unused"] is True
    assert sherpa["phoneme_codepoints_injection_verified"] is True
    assert sherpa["banned_tr_piper_voices_used"] is False
    assert sherpa["voice_model"] == "en_US-ljspeech-medium"
    assert sherpa["ttfa_p50_ms"] < 250.0
    assert sherpa["ttfa_under_250ms_budget"] is True
    assert sherpa["mean_rtf"] < 1.0
    assert len(sherpa["cases"]) == 6

    # 3) Verify offline Whisper + X8-240 114.6 Hz BVI ASR across all 4 canonical SNR strata (TR + EN)
    asr = raw["offline_whisper_bvi_proof"]
    assert asr["translate"] is False
    assert asr["condition_on_previous_text"] is False
    assert 112.0 <= asr["x8_240_bvi_max_bpf_hz"] <= 117.0
    assert set(asr["closed_loop_atc_by_snr_stratum"].keys()) == set(CANONICAL_STRATA)
    assert set(asr["real_human_fleurs_tr_en_by_snr_stratum"].keys()) == set(CANONICAL_STRATA)

    for stratum in CANONICAL_STRATA:
        atc_m = asr["closed_loop_atc_by_snr_stratum"][stratum]
        assert atc_m["n_turns"] == 6
        assert atc_m["z3_sat_pct"] == 100.0
        assert atc_m["decode_latency_p50_ms"] > 0.0
        assert 0.0 <= atc_m["keyword_recall"] <= 1.0

        hum = asr["real_human_fleurs_tr_en_by_snr_stratum"][stratum]
        for lang in ("TR", "EN"):
            assert hum[lang]["n_utterances"] >= 4
            assert 0.0 <= hum[lang]["wer"] <= 2.5
            assert 0.0 <= hum[lang]["cer"] <= 2.0
            assert hum[lang]["decode_latency_p50_ms"] > 0.0

    # 4) Verify oracle-free closed-loop Mondrian + Z3 safety invariants & per-turn transcript evidence
    mz = raw["mondrian_and_z3_closed_loop_summary"]
    assert mz["alpha"] == 0.05
    assert mz["total_turns"] == 24
    assert mz["oracle_free_transcript_parsing_verified"] is True
    assert mz["z3_executed_sat_pct"] == 100.0
    assert mz["mayday_preempt_strictly_matches_decoded_transcript"] is True
    assert isinstance(mz["mayday_s3_garbled_emergency_preempt_verified"], bool)
    assert mz["low_soc_lt_5pct_liveness_commit_verified"] is True
    assert mz["adversarial_ltfj_fail_closed_verified"] is True

    kw_by_stratum = mz["keyword_recall_by_stratum"]
    assert set(kw_by_stratum.keys()) == set(CANONICAL_STRATA)
    for stratum in CANONICAL_STRATA:
        assert 0.0 <= kw_by_stratum[stratum] <= 1.0

    turns = mz["closed_loop_atc_turns"]
    assert len(turns) == 24
    required_turn_keys = {
        "case_id",
        "snr_stratum",
        "reference_text",
        "decoded_transcript",
        "parsed_intent",
        "parsed_pad",
        "mayday_decoded_from_transcript",
        "adversarial_ltfj_decoded_from_transcript",
        "decision",
        "selected_dsl",
        "z3_sat",
    }
    for t in turns:
        assert required_turn_keys.issubset(t.keys())
        assert isinstance(t["decoded_transcript"], str)
        assert isinstance(t["reference_text"], str) and t["reference_text"].strip() != ""
        assert t["z3_sat"] is True
        # Strict oracle-free invariant: EMERGENCY_PREEMPT occurs iff MAYDAY was genuinely decoded
        assert (t["decision"] == "EMERGENCY_PREEMPT") == bool(t["mayday_decoded_from_transcript"])
        if t["parsed_intent"] == "UNPARSED_GARBLED":
            assert t["decision"] in ("ABSTAIN_HOLD_SAFE", "LIVENESS_EMERGENCY_COMMIT")
    # Clean stratum transcripts must be non-empty real Whisper outputs
    clean_turns = [t for t in turns if t["snr_stratum"] == "S0_Clean"]
    assert len(clean_turns) == 6
    assert all(t["decoded_transcript"].strip() != "" for t in clean_turns)

    # 5) Verify zero banned voice/corpus tokens anywhere in license audit
    assert raw["license_audit"]["zero_nc_sa_gpl_verified"] is True
    serialized_lower = json.dumps(raw).lower()
    for banned in ("tr_tr-dfki", "tr_tr-fahrettin", "tr_tr-fettah", "piper1-gpl", "uwb-atcc", "atco2"):
        assert banned not in serialized_lower


def test_oracle_free_voice_gate_fails_closed_on_empty_or_gibberish_transcript() -> None:
    """Adversarial anti-oracle unit test: empty/gibberish transcripts must NEVER trigger oracle fallbacks."""
    from scripts.bench_local_voice_atc_live import (
        AVIATION_TTS_BENCH_CASES,
        STRATUM_SNR_DB,
        _evaluate_mondrian_and_z3_for_turn,
    )

    mayday_case = {
        **next(c for c in AVIATION_TTS_BENCH_CASES if c["id"] == "ATC-02-MAYDAY-DRAGOS"),
        "soc_pct": 50.0,  # Nominal battery so onboard low-SoC liveness does not mask abstain check
    }
    adv_ltfj_case = {
        **next(c for c in AVIATION_TTS_BENCH_CASES if c["id"] == "ATC-05-ADVERSARIAL-LTFJ"),
        "soc_pct": 50.0,
    }

    for bad_transcript in ("", "zzz blarg noise"):
        for stratum in ("S0_Clean", "S3_Garbled"):
            snr = STRATUM_SNR_DB[stratum]
            m_res = _evaluate_mondrian_and_z3_for_turn(mayday_case, bad_transcript, snr, -1.8, 0.35, 42)
            assert m_res["mayday_decoded_from_transcript"] is False
            assert m_res["parsed_intent"] == "UNPARSED_GARBLED"
            assert m_res["parsed_pad"] is None
            assert m_res["decision"] == "ABSTAIN_HOLD_SAFE"
            assert m_res["selected_dsl"].startswith("ACT=HOLD_LOITER;")
            assert "Dragos_Plaza" not in m_res["selected_dsl"]
            assert "ABORT_RETURN" not in m_res["selected_dsl"]
            assert m_res["z3_sat"] is True

            a_res = _evaluate_mondrian_and_z3_for_turn(adv_ltfj_case, bad_transcript, snr, -1.8, 0.35, 43)
            assert a_res["adversarial_ltfj_decoded_from_transcript"] is False
            assert a_res["parsed_intent"] == "UNPARSED_GARBLED"
            assert a_res["parsed_pad"] is None
            assert a_res["decision"] == "ABSTAIN_HOLD_SAFE"
            assert a_res["selected_dsl"].startswith("ACT=HOLD_LOITER;")
            assert "Kartal_Sahil" not in a_res["selected_dsl"]
            assert "REJECT_UNSAFE" not in a_res["selected_dsl"]
            assert a_res["z3_sat"] is True

    # Positive control: MAYDAY decoded strictly from transcript preempts even when case['mayday'] is False
    nominal_case = {
        **next(c for c in AVIATION_TTS_BENCH_CASES if c["id"] == "ATC-01-NOMINAL-KARTAL"),
        "mayday": False,
        "soc_pct": 50.0,
    }
    m_pos = _evaluate_mondrian_and_z3_for_turn(
        nominal_case, "Mayday abort return Dragos.", STRATUM_SNR_DB["S3_Garbled"], -0.45, 0.05, 44
    )
    assert m_pos["mayday_decoded_from_transcript"] is True
    assert m_pos["parsed_pad"] == "Dragos_Plaza"
    assert m_pos["decision"] == "EMERGENCY_PREEMPT"
    assert "Dragos_Plaza" in m_pos["selected_dsl"]
    assert m_pos["z3_sat"] is True

    # Positive control: onboard low-SoC (<5% margin) triggers LIVENESS_EMERGENCY_COMMIT even on empty transcript
    low_soc_case = next(c for c in AVIATION_TTS_BENCH_CASES if c["id"] == "ATC-03-LOWSOC-MALTEPE")
    ls_res = _evaluate_mondrian_and_z3_for_turn(low_soc_case, "", STRATUM_SNR_DB["S3_Garbled"], -2.2, 0.90, 45)
    assert ls_res["mayday_decoded_from_transcript"] is False
    assert ls_res["parsed_intent"] == "UNPARSED_GARBLED"
    assert ls_res["decision"] == "LIVENESS_EMERGENCY_COMMIT"
    assert ls_res["z3_sat"] is True


def test_bench_local_voice_smoke_cpu_zero_network(tmp_path: Path) -> None:
    out = tmp_path / "voice_smoke.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/bench_local_voice_atc_live.py"),
            "--smoke-cpu",
            "--out-json",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "SMOKE_CPU_ONLY"
    assert data["colab_cu_consumed"] == 0.0
    assert data["oracle_free_transcript_parsing_verified"] is True
    assert data["z3_executed_sat_pct"] == 100.0
    assert data["license_audit"]["zero_nc_sa_gpl_verified"] is True
