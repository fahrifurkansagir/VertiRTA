# SPDX-License-Identifier: Apache-2.0
"""Audio Corpus, TTS Voice & Model Commercial License Firewall for VertiRTA M3.

Enforces 01-DR08-13 Errata #7/#8 and DR-12 zero-GPL / zero-NC / zero-SA rules:
  - Hard-bans: UWB-ATCC, ATCO2, ESC-50, DEMAND, piper1-gpl, espeak-ng, piper-phonemize,
    ALL prebuilt tr_TR Piper NC voices (dfki/fahrettin/fettah), en_US-lessac/ryan/hfc/l2arctic,
    XTTS-v2, mms-tts-*, OmniVoice, Supertonic, matcha-icefall, UAVid, aya-expanse-8b.
  - Quarantines: ATCOSIM (pending DOC/license.txt), pocket-tts-turkish (TR-Wiki CC BY-SA audit).
  - Filters FSD50K metadata (`*_clips_info_FSD50K.json`) stripping CC-BY-NC & Sampling+ clips.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

CANONICAL_ALLOWED_AUDIO_CORPORA: dict[str, str] = {
    "commonvoice-27.0-cc0": "CC0-1.0",
    "common-voice-27.0": "CC0-1.0",
    "issai-tsc-mit": "MIT",
    "issai-tsc": "CC-BY-4.0",
    "fleurs-cc-by-4.0": "CC-BY-4.0",
    "mediaspeech-cc-by-4.0": "CC-BY-4.0",
    "musan-cc-by-4.0": "CC-BY-4.0",
    "openslr-28-rirs": "Apache-2.0",
    "fsd50k-filtered-cc0-ccby": "CC0-1.0 / CC-BY-4.0",
    "vertirta-bvi-dsp-apache2": "Apache-2.0",
    "en_us-ljspeech-medium": "Public-Domain",
    "en_us-libritts_r-medium": "CC-BY-4.0",
    "tr_tr-ema-lightning-apache2": "Apache-2.0",
    "tr_tr-okur-tts-apache2": "Apache-2.0 / CC-BY-4.0",
    "vits-piper-tr_tr-vertirta-clean": "Apache-2.0",
    "openai/whisper-small": "MIT",
    "openai/whisper-large-v3-turbo": "MIT",
    "silero-vad-v6.2.0": "MIT",
    "sherpa-onnx>=2.0.0": "Apache-2.0",
}

HARD_BANNED_AUDIO_AND_TTS: frozenset[str] = frozenset(
    {
        "uwb-atcc",
        "atco2",
        "esc-50",
        "demand",
        "piper1-gpl",
        "espeak-ng",
        "espeak",
        "piper-phonemize",
        "piper_phonemize",
        "tr_tr-dfki",
        "tr_tr-fahrettin",
        "tr_tr-fettah",
        "en_us-lessac",
        "en_us-ryan",
        "en_us-hfc",
        "l2arctic",
        "xtts-v2",
        "xtts",
        "mms-tts",
        "omnivoice",
        "supertonic",
        "matcha-icefall",
        "uavid",
        "aya-expanse",
    }
)

QUARANTINED_AUDIO_ASSETS: dict[str, str] = {
    "atcosim": "QUARANTINED_PENDING_DOC_LICENSE_TXT",
    "pocket-tts-turkish": "QUARANTINED_PENDING_TR_WIKI_CC_BY_SA_AUDIT",
}

_BANNED_LICENSE_TOKENS: tuple[str, ...] = (
    "gpl",
    "agpl",
    "lgpl",
    "by-nc",
    "nc-sa",
    "non-commercial",
    "noncommercial",
    "sampling+",
    "cpml",
    "cc-by-sa",
    "by-sa",
)


def is_audio_corpus_allowed(asset_id: str, license_str: str | None = None) -> bool:
    """Return True strictly when asset_id is on the clean allowlist and zero banned tokens appear."""
    low = asset_id.strip().lower()
    if not low:
        return False
    for bad in HARD_BANNED_AUDIO_AND_TTS:
        if bad in low:
            return False
    for q_key in QUARANTINED_AUDIO_ASSETS:
        if q_key in low:
            return False
    if license_str is not None:
        lic_low = license_str.strip().lower()
        for bad_lic in _BANNED_LICENSE_TOKENS:
            if bad_lic in lic_low:
                return False
    return low in CANONICAL_ALLOWED_AUDIO_CORPORA


def filter_fsd50k_clips(
    clips_info: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    """Filter FSD50K `*_clips_info_FSD50K.json`, keeping strictly CC0 / CC-BY (zero NC/Sampling+)."""
    kept: list[str] = []
    stripped = {"cc_by_nc": 0, "sampling_plus": 0, "other_rejected": 0}
    for clip_id, meta in sorted(clips_info.items()):
        lic = str(meta.get("license", "")).strip().lower()
        if "by-nc" in lic or "noncommercial" in lic:
            stripped["cc_by_nc"] += 1
            continue
        if "sampling+" in lic or "sampling%2b" in lic:
            stripped["sampling_plus"] += 1
            continue
        if ("publicdomain/zero" in lic or "cc0" in lic) or (
            "licenses/by/" in lic and "by-nc" not in lic and "by-sa" not in lic
        ):
            kept.append(str(clip_id))
        else:
            stripped["other_rejected"] += 1
    return kept, stripped


def audit_audio_manifest(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit a list of {'name': ..., 'license': ...} entries; raises ValueError on violation."""
    approved: list[dict[str, str]] = []
    violations: list[dict[str, str]] = []
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        lic = str(entry.get("license", "")).strip()
        low = name.lower()
        if any(q in low for q in QUARANTINED_AUDIO_ASSETS):
            reason = next(v for k, v in QUARANTINED_AUDIO_ASSETS.items() if k in low)
            violations.append({"name": name, "license": lic, "reason": reason})
            continue
        if not is_audio_corpus_allowed(name, lic):
            violations.append({"name": name, "license": lic, "reason": "BANNED_OR_UNAPPROVED_AUDIO_ASSET"})
            continue
        approved.append({"name": name, "license": lic})

    if violations:
        raise ValueError(f"Audio/TTS license firewall blocked {len(violations)} asset(s): {violations}")

    return {
        "passed": True,
        "zero_nc_sa_gpl_verified": True,
        "approved_count": len(approved),
        "approved": approved,
        "violations": [],
    }
