# VertiBench-v2: Bilingual Voice-ATC & Multi-eVTOL Fleet20 Benchmark

- **Schema:** `vertirta.vertibench_v2/v1.0.0`
- **Provenance Lock:** `PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2` *(never claims human-native speech/annotation until genuine human-native recordings and annotator sign-off exist).*
- **Frozen `VertiBench-v1` Preservation:** `benchmarks/vertibench_tr_en/` remains 100% immutable.

## Split Architecture
- **Track A (`VertiBench-Voice-TR/EN`):**
  - `train.jsonl`: `2,240` rows (`960 TR`, `960 EN`, `320 MIX`; balanced `NOMINAL`, `FAULT`, `ADVERSARIAL`, `AMBIGUOUS_ASR`, `< 1 MB`).
  - `cal.jsonl`: **`2,000` rows** (`800 TR`, `800 EN`, `400 MIX`; **`500`** per canonical SNR stratum `S0_Clean`, `S1_Cruise`, `S2_Hover`, `S3_Garbled` satisfying `n_lang >= 750` & `n_snr >= 250`).
  - `test.jsonl`: **`1,200` rows** (`500 TR`, `500 EN`, `200 MIX`; `300` per SNR stratum).
- **Track B (`VertiBench-Fleet20`):**
  - `fleet20.jsonl`: `200` scenarios (`100 cal`, `100 test`, `5–20` aircraft anchored to Kartal `(40.8860 N, 29.1880 E, 12.0 m AMSL, N0=+36.7 m)` strictly west of `29.1895 E`).

## Build & Verify
```bash
uv run --no-project --python 3.11 python benchmarks/vertibench_v2/build_vertibench_v2.py --out-dir benchmarks/vertibench_v2
uv run --no-project --python 3.11 python benchmarks/vertibench_v2/validate_vertibench_v2.py --data-dir benchmarks/vertibench_v2
```
