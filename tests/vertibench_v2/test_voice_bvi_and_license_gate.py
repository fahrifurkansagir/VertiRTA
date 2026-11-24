# SPDX-License-Identifier: Apache-2.0
"""TDD tests for ml/voice/bvi_cockpit_dsp.py, ml/voice/audio_license_audit.py, and config/voice_atc.yaml."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import yaml

from ml.voice.audio_license_audit import (
    audit_audio_manifest,
    filter_fsd50k_clips,
    is_audio_corpus_allowed,
)
from ml.voice.bvi_cockpit_dsp import (
    CANONICAL_SNR_STRATA,
    X8_240_ACOUSTIC_PARAMS,
    classify_snr_stratum,
    mix_at_snr,
    synthesize_bvi_cockpit_noise,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_bvi_dsp_locks_canonical_x8_240_rpm_and_bpf_overriding_dr12_stale_1850_rpm() -> None:
    assert X8_240_ACOUSTIC_PARAMS.rotor_count == 8
    assert X8_240_ACOUSTIC_PARAMS.blades_per_rotor == 3
    assert X8_240_ACOUSTIC_PARAMS.rotor_radius_m == pytest.approx(0.75, abs=1e-6)
    assert X8_240_ACOUSTIC_PARAMS.rpm_hover == pytest.approx(1449.6, abs=1.0)
    assert X8_240_ACOUSTIC_PARAMS.rpm_max == pytest.approx(2291.8, abs=1.0)
    assert X8_240_ACOUSTIC_PARAMS.bpf_hover_hz == pytest.approx(72.5, abs=0.5)
    assert X8_240_ACOUSTIC_PARAMS.bpf_max_hz == pytest.approx(114.6, abs=0.5)

    noise = synthesize_bvi_cockpit_noise(
        duration_s=2.0, rpm=2291.8, sr=16000, seed=42, rt60_s=0.18, vhf_bandlimit=False
    )
    assert noise.shape == (32000,)
    assert np.all(np.isfinite(noise))

    spec = np.abs(np.fft.rfft(noise.astype(np.float64) * np.hanning(len(noise))))
    freqs = np.fft.rfftfreq(len(noise), d=1.0 / 16000.0)
    bpf_band = spec[(freqs >= 108.0) & (freqs <= 122.0)].max()
    bg_band = np.median(spec[(freqs >= 500.0) & (freqs <= 800.0)])
    assert bpf_band > 4.0 * bg_band


def test_snr_mixer_and_four_stratum_classifier_accuracy() -> None:
    assert CANONICAL_SNR_STRATA == ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    speech = (0.25 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    noise = synthesize_bvi_cockpit_noise(duration_s=1.0, rpm=1650.0, sr=16000, seed=7)
    for target_snr, expected in [
        (18.0, "S0_Clean"),
        (11.0, "S1_Cruise"),
        (5.0, "S2_Hover"),
        (-1.0, "S3_Garbled"),
    ]:
        mixed, actual_snr = mix_at_snr(speech, noise, target_snr)
        assert mixed.shape == speech.shape
        assert abs(actual_snr - target_snr) < 0.25
        assert classify_snr_stratum(actual_snr) == expected


def test_audio_license_firewall_strips_fsd50k_nc_sampling_plus_and_blocks_all_banned_sources() -> None:
    for banned in (
        "UWB-ATCC",
        "ATCO2",
        "ESC-50",
        "DEMAND",
        "piper1-gpl",
        "espeak-ng",
        "piper-phonemize",
        "tr_TR-dfki-medium",
        "tr_TR-fahrettin-medium",
        "tr_TR-fettah-medium",
        "en_US-lessac-medium",
        "en_US-ryan-medium",
        "XTTS-v2",
        "mms-tts-tur",
        "UAVid",
        "CohereForAI/aya-expanse-8b",
    ):
        assert not is_audio_corpus_allowed(banned), f"Failed to block banned asset: {banned}"

    for allowed in (
        "CommonVoice-27.0-CC0",
        "ISSAI-TSC-MIT",
        "FLEURS-CC-BY-4.0",
        "MUSAN-CC-BY-4.0",
        "OpenSLR-28-RIRS",
        "VertiRTA-BVI-DSP-Apache2",
        "en_US-ljspeech-medium",
        "tr_TR-ema-lightning-apache2",
    ):
        assert is_audio_corpus_allowed(allowed)

    # Quarantined assets MUST fail strict allowlist until audited
    assert not is_audio_corpus_allowed("ATCOSIM")
    assert not is_audio_corpus_allowed("pocket-tts-turkish")

    with pytest.raises(ValueError):
        audit_audio_manifest([{"name": "ATCOSIM", "license": "Unknown"}])

    res = audit_audio_manifest(
        [
            {"name": "CommonVoice-27.0-CC0", "license": "CC0-1.0"},
            {"name": "VertiRTA-BVI-DSP-Apache2", "license": "Apache-2.0"},
        ]
    )
    assert res["passed"] is True
    assert res["zero_nc_sa_gpl_verified"] is True

    mock_fsd = {
        "1001": {"license": "http://creativecommons.org/publicdomain/zero/1.0/"},
        "1002": {"license": "http://creativecommons.org/licenses/by/4.0/"},
        "1003": {"license": "http://creativecommons.org/licenses/by-nc/3.0/"},
        "1004": {"license": "http://creativecommons.org/licenses/sampling+/1.0/"},
    }
    kept, stripped = filter_fsd50k_clips(mock_fsd)
    assert kept == ["1001", "1002"]
    assert stripped == {"cc_by_nc": 1, "sampling_plus": 1, "other_rejected": 0}


def test_voice_atc_yaml_locks_sherpa_v2_and_zero_gpl() -> None:
    cfg = yaml.safe_load((REPO_ROOT / "config/voice_atc.yaml").read_text(encoding="utf-8"))
    assert cfg["tts"]["engine"] == "sherpa-onnx"
    assert cfg["tts"]["min_version"] == "2.0.0"
    assert cfg["tts"]["zero_espeak_ng_verified"] is True
    assert cfg["asr"]["translate"] is False
    assert cfg["asr"]["no_context"] is True
    assert cfg["mondrian_conformal_gate"]["alpha"] == pytest.approx(0.05)
    assert cfg["mondrian_conformal_gate"]["loiter_liveness_min_soc_margin_pct"] == pytest.approx(5.0)
    assert cfg["mondrian_conformal_gate"]["mayday_asymmetric_weight_floor"] >= 0.15
