---
license: apache-2.0
language:
  - tr
  - en
multilinguality:
  - bilingual
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
  - zero-shot-classification
tags:
  - aerospace
  - evtol
  - uam
  - runtime-assurance
  - signal-temporal-logic
  - z3-smt
  - conformal-prediction
  - turkish-asr
  - safety-critical
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: calibration
        path: data/calibration.jsonl
      - split: test
        path: data/test.jsonl
pretty_name: VertiBench-TR/EN
---

# VertiBench-TR/EN — Bilingual Safety-Critical UAM Contingency Benchmark

**License:** Apache-2.0 (`Qwen/Qwen3-8B` and `Qwen/Qwen3-4B` are the ONLY fine-tuned models; `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-3-4b-it` are evaluated strictly zero-shot + GBNF + Z3; `CohereForAI/aya-expanse-8b` is excluded due to `CC-BY-NC-4.0`).

> **Simulation Disclaimer:** All Istanbul `LTFJ` (Sabiha Gokcen) and `LTBX` (Samandira) CTR geometries encoded in VertiBench-TR/EN derive from published DHMI eAIP coordinates strictly for offline software-in-the-loop verification and NEVER replace operational aeronautical charts or live NOTAMs.

## Vehicle & Airspace Envelope Lock
- **Aircraft:** `VertiRTA-X8-240` full-scale 8-rotor coaxial eVTOL (`240 kg`, `T/W = 2.5`, **`400 V / 22 kWh`** HV Li-ion pack, loaded `v_term_volts` ~`350–380 V`, `12%` hard SoC reserve floor, `AGL in [30, 150] m`, `R_SLZ >= 5.0 m`).
- **Canonical ENU Origin:** Kartal Coastal Vertiport (`40.8860° N, 29.1880° E, 12.0 m AMSL`, `N_EGM2008 = +36.7 m`).
- **Valid Coastal SLZs (strictly west of `29.200° E`):** `Kartal_Sahil`, `Cevizli_Helipad`, `Dragos_Plaza`, `Maltepe_Park`.
- **Compact DSL (`config/vertirta_dsl_v1.gbnf`):** `<= 30` tokens under Qwen3 single-digit BPE (`ACT=...;LNG=...;MOD=...`); full `<= 32`-node Bounded-STL formula synthesized deterministically in C++20.

## Stratified Split Provenance & Anti-Contamination Gate

| Bucket | `train` (Programmatic, GBNF+Z3 Filtered) | `calibration` (`human_native_tr_v1`) | `test` (`human_native_tr_v1`) | Focus |
|---|---:|---:|---:|---|
| **B1 (`NOMINAL`)** | 450 | 50 | 140 | Pure TR, pure EN, aviation TR-EN code-switching |
| **B2 (`FAULT`)** | 520 | 55 | 160 | All 5 `FaultMode` states, wind/SoC cutoffs, SLZ OOD spikes |
| **B3 (`ADVERSARIAL`)** | 450 | 50 | 140 | `LTFJ`/`LTBX` CTR intrusions (`Pendik Marina`, `Yakacik`), street landing, shield-disable & `/think` jailbreaks |
| **B4 (`AMBIGUOUS_ASR`)** | 380 | 45 | 120 | Dropped Turkish diacritics (`ı/ş/ğ/ç/ö/ü`), missing target slot, garbled OOD ASR |
| **Total** | **1,800** | **200** | **560** | **760 human-audited zero-leakage eval cases** |

Automated `ubuntu-24.04-arm` CI (`validate_vertibench.py`) enforces:
1. **0** `scenario_seed` overlap between `train` and `calibration U test`.
2. **`< 0.15`** maximum character 5-gram and token 3-gram Jaccard similarity between every eval utterance and all 1,800 `train` utterances.
3. **100%** GBNF + `Z3` `QF_LRA` validity across `train.jsonl` and locked `400 V / 22 kWh` HV telemetry across all splits.
