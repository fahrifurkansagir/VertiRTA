# SPDX-License-Identifier: Apache-2.0
"""VertiBench-v2: Bilingual Voice-ATC (Track A) & Multi-eVTOL Fleet20 (Track B) Benchmark.

Preserves frozen `benchmarks/vertibench_tr_en/` (VertiBench-v1) 100% immutable.
"""

from benchmarks.vertibench_v2.build_vertibench_v2 import build_vertibench_v2_splits
from benchmarks.vertibench_v2.fleet20_scenarios import generate_fleet20_scenarios
from benchmarks.vertibench_v2.schema import (
    CANONICAL_BUCKETS,
    CANONICAL_LANGS,
    CANONICAL_SNR_STRATA,
    HONEST_PROVENANCE_TAG_V2,
    MAX_JACCARD_THRESHOLD,
)
from benchmarks.vertibench_v2.validate_vertibench_v2 import (
    VertiBenchV2Report,
    validate_vertibench_v2,
)

__all__ = [
    "CANONICAL_BUCKETS",
    "CANONICAL_LANGS",
    "CANONICAL_SNR_STRATA",
    "HONEST_PROVENANCE_TAG_V2",
    "MAX_JACCARD_THRESHOLD",
    "VertiBenchV2Report",
    "build_vertibench_v2_splits",
    "generate_fleet20_scenarios",
    "validate_vertibench_v2",
]
