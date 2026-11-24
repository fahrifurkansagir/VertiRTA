# SPDX-License-Identifier: Apache-2.0
"""VertiRTA Apache-2.0 Procedural Coaxial BVI Cockpit DSP & Audio License Firewall."""

from ml.voice.audio_license_audit import (
    CANONICAL_ALLOWED_AUDIO_CORPORA,
    HARD_BANNED_AUDIO_AND_TTS,
    QUARANTINED_AUDIO_ASSETS,
    audit_audio_manifest,
    filter_fsd50k_clips,
    is_audio_corpus_allowed,
)
from ml.voice.bvi_cockpit_dsp import (
    CANONICAL_SNR_STRATA,
    X8_240_ACOUSTIC_PARAMS,
    CoaxialAcousticParams,
    classify_snr_stratum,
    load_x8_240_acoustic_params,
    measure_snr_db,
    mix_at_snr,
    synthesize_bvi_cockpit_noise,
)

__all__ = [
    "CANONICAL_ALLOWED_AUDIO_CORPORA",
    "CANONICAL_SNR_STRATA",
    "CoaxialAcousticParams",
    "HARD_BANNED_AUDIO_AND_TTS",
    "QUARANTINED_AUDIO_ASSETS",
    "X8_240_ACOUSTIC_PARAMS",
    "audit_audio_manifest",
    "classify_snr_stratum",
    "filter_fsd50k_clips",
    "is_audio_corpus_allowed",
    "load_x8_240_acoustic_params",
    "measure_snr_db",
    "mix_at_snr",
    "synthesize_bvi_cockpit_noise",
]
