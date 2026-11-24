# SPDX-License-Identifier: Apache-2.0
"""Schema constants & frozen canonical locks for VertiBench-v2."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION_V2: str = "vertirta.vertibench_v2/v1.0.0"

# Mandatory honest provenance tag (01-DR08-13 Override #7 — zero fabricated human-native claims).
HONEST_PROVENANCE_TAG_V2: str = (
    "PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2"
)

CANONICAL_SNR_STRATA: tuple[str, ...] = ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")
CANONICAL_LANGS: tuple[str, ...] = ("TR", "EN", "MIX")
CANONICAL_BUCKETS: tuple[str, ...] = ("NOMINAL", "FAULT", "ADVERSARIAL", "AMBIGUOUS_ASR")

EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    "train": 2240,
    "cal": 2000,
    "test": 1200,
    "fleet20": 200,
}

EXPECTED_CAL_LANG_COUNTS: dict[str, int] = {"TR": 800, "EN": 800, "MIX": 400}
EXPECTED_TEST_LANG_COUNTS: dict[str, int] = {"TR": 500, "EN": 500, "MIX": 200}
EXPECTED_CAL_SNR_COUNTS: dict[str, int] = {s: 500 for s in CANONICAL_SNR_STRATA}
EXPECTED_TEST_SNR_COUNTS: dict[str, int] = {s: 300 for s in CANONICAL_SNR_STRATA}

MAX_JACCARD_THRESHOLD: float = 0.15

# Canonical Istanbul Kartal ENU datum & LTFJ NW J2 eastern wall (01-DR08-13 Override #3).
KARTAL_DATUM_LAT_DEG: float = 40.8860
KARTAL_DATUM_LON_DEG: float = 29.1880
KARTAL_DATUM_AMSL_M: float = 12.0
ISTANBUL_EGM2008_N0_M: float = 36.7
EASTERN_GEOFENCE_LON_DEG: float = 29.1895

# Frozen 4 Anatolian coastal vertiports (strictly west of 29.1895 E).
VERTIPORTS: dict[str, tuple[float, float]] = {
    "Kartal_Sahil": (40.88600, 29.18800),
    "Dragos_Plaza": (40.91686, 29.13822),
    "Maltepe_Park": (40.94016, 29.10790),
    "Cevizli_Helipad": (40.90380, 29.15450),
}

# Disjoint seed ranges enforcing 0 seed overlap across splits.
SEED_RANGE_TRAIN: tuple[int, int] = (100_000, 199_999)
SEED_RANGE_CAL: tuple[int, int] = (200_000, 299_999)
SEED_RANGE_TEST: tuple[int, int] = (300_000, 399_999)
SEED_RANGE_FLEET20: tuple[int, int] = (400_000, 499_999)

# Hard-banned models & datasets (01-DR08-13 §3.1) — never referenced in VertiBench-v2.
BANNED_ASSETS: frozenset[str] = frozenset(
    {
        "CohereForAI/aya-expanse-8b",
        "UAVid",
        "UWB-ATCC",
        "ATCO2",
        "ESC-50",
        "DEMAND",
        "piper1-gpl",
        "tr_TR-dfki-medium",
        "tr_TR-fahrettin-medium",
        "tr_TR-fettah-medium",
        "XTTS-v2",
    }
)


@dataclass(frozen=True)
class SnrSpec:
    stratum: str
    lo_db: float
    hi_db: float
    nospeech_max: float


SNR_SPECS: dict[str, SnrSpec] = {
    "S0_Clean": SnrSpec("S0_Clean", 16.0, 25.0, 0.05),
    "S1_Cruise": SnrSpec("S1_Cruise", 8.5, 14.5, 0.14),
    "S2_Hover": SnrSpec("S2_Hover", 2.5, 7.5, 0.30),
    "S3_Garbled": SnrSpec("S3_Garbled", -3.0, 1.8, 0.55),
}
