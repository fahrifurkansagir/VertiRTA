# VertiRTA v4.0.0 — Plan 6B: Offline TR/EN Voice ATC, VertiBench-v2 & Mondrian Conformal, Regulatory Forensics (E5X/ERCS) & Paper 2 (CoRL 2027) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the `M3 (v4.0.0)` Voice ATC, Bilingual Benchmark & Mondrian Conformal Calibration, Deterministic Regulatory Forensics (`CIR (EU) 2026/1821` / `SHGM SHY-OLAY` / `CDR (EU) 2020/2034` ERCS / `ED-155A`), and Double-Anonymous Paper 2 (`CoRL 2027`) subsystems — with **ZERO file overlap** against `Plan 6A` (`core/forward_hazard/`, `sim/gazebo_plugins/mmwave_77ghz/`, `core/gnss_denied/`, root `CMakeLists.txt`, `.github/workflows/m3_perception_voice_forensics.yml`, `apps/vertigcs/`) and `Plan 6C` (`scripts/colab/run_g4_m3_thunder_run.py`, `scripts/colab/g4_jobs/`, `scripts/bench_gtx1650s_m3_timesliced.py`, `artifacts/v4.0.0/`), **ZERO regression** to frozen `v3.5.1` invariants (`28,886 B Flash / 0 B RAM` bare-metal Cortex-M7 `libvertirta_flight_core.a`, `256 B VRTA0001` binary frames, `38,470 B <= 150 KiB` WASM, immutable `VertiBench-v1` `200/560`, `0` ASTM F3269 isolation violations, `0` GPL runtime symbols), and **ZERO `colab new` calls** during Phase 1 or Phase 2 (Absolute Zero-Waste `G4` Lock).

**Architecture:** `Plan 6B` builds strictly above the frozen deterministic RTA core (`libvertirta_flight_core`, `libvertirta_rta_companion`, `libvertirta_copilot`, `libvertirta_fleet_z3`) across five decoupled layers:
1. **`core/voice/` + `ml/voice/` + `ros2_ws/src/vertirta_voice_node/` (Offline Cockpit-Noise TR/EN Voice ATC & Zero-GPL TTS):**
   - `Silero VAD v6.2.0` (`MIT`) frame-gated endpointing feeding dual-tier `whisper.cpp` (`MIT`) ASR: fast tactical path (`whisper-small` Q8_0, `~0.40 s`) escalating on low acoustic confidence or safety-critical slots to verification path (`whisper-large-v3-turbo` Q8_0 with bounded `audio_ctx = round_up_64(ceil(dur_s / 30.0 * 1500) + 128)` clamped to `[256, 384]`, `translate = false`, `no_context = true`, `carry_initial_prompt = true`, `K = 4` N-best beams exposing `whisper_token_data::plog` and `<|nospeech|>`).
   - **Zero-GPL TTS (`< 250 ms` TTFA):** Hard-pinned **`sherpa-onnx >= 2.0.0`** (PR `#4055`, commit `0b36476` eliminating GPL-3.0 `espeak-ng` & `piper-phonemize`) running `en_US-ljspeech-medium` (`PD`) / `libritts_r-medium` (`CC BY 4.0`) + pure Apache-2.0 C++20 `TurkishG2pPhonemizer` feeding `phoneme_codepoints` to clean Turkish acoustic weights (`canberkkkkkk/ema-lightning` `Apache-2.0`, `ahakan/okur-tts` `Apache-2.0 + CC BY 4.0`, or clean-room `vits-piper-tr_TR-vertirta` on `CV 27.0` + `ISSAI TSC`), clause-streamed at `num_threads = 2`.
   - **4-Stratum SNR Mondrian KnowNo Gate (`S0..S3`):** Stratifies cockpit SNR into `S0_Clean (>15 dB)`, `S1_Cruise (8–15 dB)`, `S2_Hover (2–8 dB)`, `S3_Garbled (<2 dB)`, fuses acoustic uncertainty (`H_nbest`, `p_min_slot`, `p_nospeech`), enforces **`alpha = 0.05`** (finite-sample exact Mondrian quantile per stratum `g`), applies **MAYDAY / PAN-PAN asymmetric emergency priority (`w_m >= 0.15`)**, and preserves the **`< 5.0%` SoC margin `LIVENESS_EMERGENCY_COMMIT`** deterministic override.
   - **Apache-2.0 Procedural Coaxial BVI DSP (`ml/voice/bvi_cockpit_dsp.py`) & License Firewall (`ml/voice/audio_license_audit.py`):** Synthesizes 8-rotor coaxial (`B = 3` blades, `R = 0.75 m`) BVI cockpit noise across canonical `vehicle_x8_240.yaml` `1,100–2,292 RPM` (`55.0–114.6 Hz` BPF, `2*B` coaxial crossover modulation, PMSM whine, `RT60 = 0.12–0.25 s`, `300–3,400 Hz` VHF filter, `-3..+25 dB` SNR) and enforces the audio/TTS/dataset license firewall consumed directly by `Plan 6C`'s `job_whisper_bvi_lora.py`.
2. **`benchmarks/vertibench_v2/` + `ml/copilot/calibrate_conformal.py` (VertiBench-v2 & Language/SNR Mondrian Conformal):**
   - Keeps frozen `benchmarks/vertibench_tr_en/` (`VertiBench-v1`) 100% immutable.
   - Constructs **Track A (`VertiBench-Voice-TR/EN`)** (`train = 3,600`, **`N_cal = 2,000`** = `800 TR + 800 EN + 400 MIX` with **`n_g = 500 >= 250`** per SNR stratum `S0..S3`, **`N_test = 1,200`** = `500 TR + 500 EN + 200 MIX`, balanced EN+TR Bucket-3/Bucket-4 closing `v3.5.1`'s EN adversarial gap) and **Track B (`VertiBench-Fleet20`)** (`200` Kartal–LTFJ–LTBX 5–20 aircraft fleet scenarios strictly west of `29.1895° E`).
   - Enforces honest provenance tag **`PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2`** (zero fabricated human-native claims!), 5-layer anti-contamination (`max_char_5gram_jaccard < 0.15`, `max_token_3gram_jaccard < 0.15`, `0` seed overlap, zero BIG-Bench canary strings), and calibration-split-only `(gamma, tau)` CV + per-language & 4-stratum SNR Mondrian quantiles in `ml/copilot/calibrate_conformal.py` (consumed by `Plan 6C`'s `job_qwen3_v3_vertibench_v2.py` to drive `Mean Set Size < 1.8` & `Singleton Exec >= 65%` at `>= 95%` group-conditional coverage).
3. **`core/forensics/` + `config/eccairs2_nodigit.gbnf` (Deterministic Post-Flight Forensics & Zero-Hallucination E5X/ERCS):**
   - Reads frozen `128 B` header + `256 B VRTA0001` binary frames (`vertirta::core::VrtaFileHeader`, `VrtaFrame100Hz`) strictly read-only (`0 B` added to `libvertirta_flight_core.a`!) and joins them via **`(seq, t_sim_us)`** to an append-only **`ED-155A`** power-loss-safe **`.vrta_events.jsonl`** sidecar carrying Voice/ASR transcripts + SHA-256 audio digests + Mondrian `S0..S3`/`|C_g|`, Z3 SMT-LIB2 fingerprints + `unsat_core`, GSN v3 node IDs, chained SHA-256 hashes, and `libsodium >= 1.0.21` Ed25519 signatures.
   - Scores events with a deterministic **`CDR (EU) 2020/2034` ERCS 8-Barrier Scorer (`ID 1109`)** enforcing CICTT **`SCF-NP`** (never `SCF-PP`!) for rotor/control/SW/datalink faults, **Barrier `#8` (Low-Energy) = `0 (N/A)`** on high-energy `MAC`/`LOC-I`, dynamic vehicle capacity scaling (`1 pax` on canonical `X8-240`, configurable `4–6 pax` for `Lift+Cruise`), and a hard compile/runtime invariant emitting **draft/`Incident` (`Entity 24 / ID 431`) ONLY** (never auto-classifying `Accident` or `Serious Incident`).
   - Emits XSD-valid **`.e5x` XML** (`<SET><Occurrence entityId="24">`, strictly ascending `<ATTRIBUTES>` IDs, Article 16 PII-stripped `Headline` `ID 601`, `CIR (EU) 2026/1821` eVTOL triggers, `SHGM SHY-OLAY` 72h ADREP parallel EN+TR `Entity 22` narratives, omission of N/A injury counts).
   - Locks LLM narrative prose behind **`config/eccairs2_nodigit.gbnf`** (banning `[0-9]` digits and permitting only allowlisted `{{SLOT}}` placeholders) + **`verify_numeric_grounding()`** proving **`0.0%` numeric hallucination**.
4. **`docs/paper_corl/` (Paper 2 — Double-Anonymous 8-Page CoRL 2027 Manuscript & Reproducibility Pipeline):**
   - Covers Bilingual Z3-GRPO (`3–5` seeds) + Language/SNR-Conditional Mondrian Conformal (`100` Monte Carlo resplits) + Active Visual SLZ real-imagery transfer (`AeroScapes` / `ICG Drone` / `UAVScenes` — **HARD-BAN `UAVid`**), referencing Paper 1's RTA cage neutrally.
   - Ships `corl_metrics_manifest.json`, `scripts/generate_corl_tables.py`, `scripts/generate_corl_figures.py`, `scripts/verify_corl_numbers.py` (`0` errors), and `docs/paper_corl/tests/test_paper_corl_contracts.py` enforcing strict double-anonymity (zero author/URL/repo/`UAVid` leaks).

**Tech Stack:**
- **WSL2 Ubuntu (verified local):** `gcc/g++ 15.2.0`, `cmake 4.2.3`, `ninja 1.13.2`, `Catch2 v3.7.1`, `arm-none-eabi-gcc 14.2.1`.
- **Host Windows (verified local):** `uv 0.13.5` (`uv run --no-project`), `Python 3.11+` (`pytest`, `pyyaml`, `numpy`, `lxml`/`xmlschema`, `z3-solver`, `matplotlib`).
- **GitHub Actions CI (`ubuntu-24.04` & `ubuntu-24.04-arm`, wired in Plan 6A Wave 2):** `libz3-dev`, immutable pinned **`libsodium >= 1.0.21`** (`libsodium-1.0.22.tar.gz` with pinned SHA-256), optional native `whisper.cpp` (`MIT`) & `sherpa-onnx >= 2.0.0` (`Apache-2.0`), `scripts/ci/check_zero_gpl_symbols.py`, `scripts/ci/check_public_hygiene.py`.

**Spec & Locks:**
- `docs/superpowers/specs/2026-11-22-vertirta-v3-v4-expansion-design.md` (Sec 1.2 Invariants `I1–I7` & Sec 4.2 `M3 v4.0.0`)
- `docs/research/01-DR08-13-FORENSIC-ERRATA-AND-LOCKS.md` (Overrides `#1`, `#3`, `#5`, `#7`, `#8` & License/CVE Firewall)
- `docs/research/DR-12-voice-atc-fleet-z3-eccairs2-forensics.md` (Sec 1, 2, 4, 5 — filtered strictly through Errata overrides)
- `docs/research/DR-13-vertigcs-wasm-nanobind-sdk-opencore-papers.md` (Sec 5 Paper 2 CoRL 2027 — with `UAVid` hard-banned per Erratum `#7`)

---

## Global Constraints & Canonical Errata Overrides

1. **Frozen Bare-Metal Core, Wire Format, WASM & `VertiBench-v1` (ZERO Drift):**
   - NEVER edit `core/flight_core/**`, `core/rta_companion/**`, `core/copilot/**`, `core/fleet_z3/**`, `wasm/**`, `benchmarks/vertibench_tr_en/**`, `config/vehicle_x8_240.yaml`, `config/istanbul_ltfj_airspace.yaml`, or `config/vertirta_dsl_v1.gbnf`.
   - Bare-metal `libvertirta_flight_core.a` MUST stay exact `28,886 B Flash / 0 B RAM` (`0` heap symbols); `VrtaFileHeader` (`128 B`) + `VrtaFrame100Hz` (`256 B`, `pad[80]` untouched) stay **BIT-FROZEN**; `libvertirta_flight_core.wasm` stays `38,470 B <= 150 KiB`; `check_arch_isolation.py` stays `0` violations.
   - **Erratum `#5` Override (DR-12 Sec 4.3):** Reject DR-12's suggestion to pack ASR transcripts, Z3 fingerprints, or Ed25519 signatures into `100 Hz` `.vrta` binary frames. Write all discrete high-level audit events strictly to companion append-only **`.vrta_events.jsonl`** linked by **`(seq, t_sim_us)`**.
2. **Canonical Vehicle Acoustics Override (`01-DR08-13` Erratum `#1`):**
   - Reject DR-12's stale `1,850 RPM`, `90 kt`, `4–6 pax` baseline for `X8-240`.
   - Lock frozen `config/vehicle_x8_240.yaml`: **`8` coaxial rotors (`4` arms, `coax_spacing = 0.35 m`), `B = 3` blades/rotor, `R = 0.75 m` (`D = 1.50 m`), `omega_hover = 151.8 rad/s` (`1,450 RPM`, `BPF_hover = 72.5 Hz`, `2*B` coaxial crossover `145.0 Hz`), `omega_max = 240.0 rad/s` (`2,292 RPM`, `BPF_max = 114.6 Hz`, `2*B` crossover `229.2 Hz`), `v_max = 22.0 m/s` (`42.8 kt`), `240 kg MTOW`, `1` occupant**.
   - Procedural BVI synthesizer spans `1,100–2,292 RPM` (`55.0–114.6 Hz` BPF). ERCS severity scaling reads `occupant_capacity` from vehicle config (`1 pax` + 3rd-party ground-risk band for `X8-240`; `4–6 pax` only when explicitly parameterized for `Lift+Cruise 8+1`).
3. **Canonical Geodesy Override (`01-DR08-13` Erratum `#3`):**
   - Reject DR-12's stale `40.8880° N` Kartal coordinate.
   - Lock frozen `config/istanbul_ltfj_airspace.yaml`: **Kartal datum `(40.8860° N, 29.1880° E, 12.0 m AMSL, N0 = +36.7 m)`**, eastern boundary **`max_lon_near_j2 = 29.1895° E`**, 4 coastal vertiports (`Kartal_Sahil`, `Dragos_Plaza`, `Maltepe_Park`, `Cevizli_Helipad`) + Kartal Hospital helipad (`40.91639° N, 29.17417° E`).
4. **Conformal Alpha, Liveness & Honest Provenance Locks (`01-DR08-13` Erratum `#7`):**
   - Lock **`alpha = 0.05`** (`1 - alpha = 0.95` target). Finite-sample quantile index `ceil((n_g + 1) * (1 - alpha)) / n_g` is computed per stratum `g` (`0.955` is solely the finite-sample index at `n = 200`, never a conflated `alpha`).
   - Preserve deterministic **`< 5.0%` SoC margin `LIVENESS_EMERGENCY_COMMIT`** override (`kLoiterLivenessMinSocMarginPct = 5.0f`) unconditionally in `MondrianVoiceConformalGate`.
   - Tag `VertiBench-v2` strictly as **`PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2`** — NEVER claim human-native evaluation until genuine human-native recordings/audits exist.
   - **HARD BAN `UAVid`** across all code, datasets, manifests, and `docs/paper_corl/` (use `AeroScapes`, `ICG Drone`, and `UAVScenes` exclusively).
5. **Zero-GPL & Commercial License Firewall + CVE Pins:**
   - **HARD BANS (CI-enforced in `ml/voice/audio_license_audit.py`, `core/voice/CMakeLists.txt`, and `scripts/ci/check_zero_gpl_symbols.py`):**
     - TTS/Phonemizer: `OHF-Voice/piper1-gpl` (`>= 1.7.0`, GPL-3.0), `espeak-ng`, `piper-phonemize`, `sherpa-onnx < 2.0.0`, **ALL prebuilt `tr_TR` Piper voices** (`tr_TR-dfki-medium` CC BY-NC-SA; `tr_TR-fahrettin-medium` & `tr_TR-fettah-medium` NC `lessac` lineage), `en_US-lessac/ryan/hfc/l2arctic`, `mms-tts-*` (CC BY-NC), `XTTS-v2` (CPML), `OmniVoice`, `Supertonic`, `matcha-icefall` (`#4265`).
     - Audio Corpora: `UWB-ATCC` (CC BY-NC-SA), `ATCO2` (commercial), `ESC-50` (CC BY-NC), `DEMAND` (CC BY-NC-SA), unfiltered `FSD50K` (must strip `4,616` dev + `1,425` eval CC-BY-NC & `1,777` Sampling+ clips via `*_clips_info_FSD50K.json`), `ATCOSIM` (quarantined pending `DOC/license.txt` check), `pocket-tts-turkish` (flagged for TR-Wiki CC BY-SA audit).
     - LLM / Vision: `CohereForAI/aya-expanse-8b` (CC BY-NC), `UAVid` (academic-only).
   - **Crypto Pin:** `core/forensics/` pins **`libsodium >= 1.0.21`** (rejecting Ubuntu 24.04 apt `1.0.18 <= 1.0.20` Ed25519 subgroup CVE via `#if SODIUM_LIBRARY_VERSION_MAJOR ...` + `VERTIRTA_REQUIRE_LIBSODIUM` CMake option).
6. **Zero-Waste `G4`, Wave Isolation & Hygiene Locks:**
   - **NO `colab new` calls** during Phase 1 or Phase 2.
   - Wave 1 C++ packages (`core/voice/`, `core/forensics/`, `ros2_ws/src/vertirta_voice_node/`) build cleanly standalone in `/tmp/vertirta_m3_*` with **ZERO root `CMakeLists.txt` edits** (`Plan 6A` owns the single Wave 2 root `CMakeLists.txt` and `.github/workflows/m3_perception_voice_forensics.yml` wiring).
   - Pure UTF-8 no BOM, pure LF (`\n`), `uv run --no-project` (never creating `.venv/` or `uv.lock`), `0` `check_public_hygiene.py` hits, `0` compliance/certification claims (research testbed only).

---

## Review Focus (5 Failure Modes for `@devils-advocate`)

1. **Silent GPL-3.0 / NC TTS Contamination (`espeak-ng`, `piper1-gpl`, Prebuilt `tr_TR` Piper Voices & Unfiltered `FSD50K`):** Linking `sherpa-onnx < 2.0.0` re-introduces GPL-3.0 `espeak-ng`/`piper_phonemize`, downloading prebuilt `tr_TR-dfki/fahrettin/fettah` imports CC BY-NC weights, and ingesting raw `FSD50K` leaks `7,818` NC/Sampling+ clips — pinned in Task 1 (`core/voice/CMakeLists.txt` version & symbol guard + `check_zero_gpl_symbols.py`) and Task 2 (`test_voice_bvi_and_license_gate.py`).
2. **`.vrta` Wire-Format Bloat & `libsodium 1.0.18` Subgroup CVE in Forensics Sidecar:** Following DR-12 Sec 4.3 literally would break frozen `256 B VRTA0001` frames / `28,886 B` Cortex-M7 Flash, and using apt `libsodium 1.0.18` allows forged Ed25519 audit chains — pinned in Task 4 (`test_forensics_sidecar_ercs_e5x.cpp` checking `sizeof(VrtaFrame100Hz) == 256`, `(seq, t_sim_us)` sidecar join, SHA-256 chain tamper detection, and `libsodium >= 1.0.21` version lock).
3. **ERCS Regulatory Misclassification (`SCF-PP` vs `SCF-NP`, Barrier `#8` on High-Energy MAC/LOC-I, Auto-`Accident` Emission & LLM Numeric Hallucination):** Classifying inverter/rotor/SW faults as `SCF-PP`, crediting low-energy Barrier `#8` on high-energy collisions, auto-labeling `Accident`/`Serious Incident` (violating SIA statutory prerogative), or letting Qwen3 invent telemetry numbers destroys regulatory validity — pinned in Task 4 (`eccairs2_nodigit.gbnf`, `verify_numeric_grounding()`, `eccairs2_e5x_minimal.xsd`, `test_e5x_xsd_and_grounding.py`).
4. **Conformal Alpha/Liveness Regression, EN Bucket-3 Collapse & Calibration Leakage:** Conflating `q_lvl = 0.955` with `alpha`, omitting `< 5.0%` SoC `LIVENESS_EMERGENCY_COMMIT`, suppressing MAYDAY under noisy `S3_Garbled` audio, or tuning `(gamma, tau)` on `test` instead of `calibration` invalidates coverage guarantees and repeats `v3.5.1`'s `Mean Set Size = 3.125` / `EN B3 = 0%` failure — pinned in Task 1 (`test_mondrian_voice_gate.cpp`) and Task 3 (`test_vertibench_v2_and_mondrian.py`).
5. **CoRL Paper 2 Double-Anonymous Leaks, `UAVid` Regression, or Fabricated Human-Native/G4 Numbers:** Leaking GitHub URLs/author identities, leaving DR-13's banned `UAVid` in tables, fabricating human-native provenance or pre-G4 metrics, or letting LaTeX numbers drift from JSON manifests breaks submission integrity — pinned in Task 6 (`verify_corl_numbers.py`, `test_paper_corl_contracts.py`, `check_public_hygiene.py`).

---

## File Structure, 3-Agent Phase 2 Ownership & Cross-Plan Interfaces (ZERO Overlap)

### External Plan Boundaries (DO NOT TOUCH in `Plan 6B`)
- **Owned strictly by `Plan 6A` (`@vision-slz-eng` + `@flight-stack-auditor` + `@px4-sim-eng`):**
  - `core/forward_hazard/**`, `tests/forward_hazard/**`, `sim/gazebo_plugins/mmwave_77ghz/**`, `core/gnss_denied/**`, `tests/gnss_denied/**`, `ros2_ws/src/vertirta_forward_hazard_node/**`, `ros2_ws/src/vertirta_gnss_denied_node/**`, root `CMakeLists.txt` (Wave 2 registration), `.github/workflows/m3_perception_voice_forensics.yml` (Wave 2 CI), `apps/vertigcs/**`.
- **Owned strictly by `Plan 6C` (`@flight-stack-auditor`):**
  - `scripts/colab/run_g4_m3_thunder_run.py`, `scripts/colab/g4_jobs/**`, `scripts/bench_gtx1650s_m3_timesliced.py`, `tests/colab/**`, `artifacts/v4.0.0/**`.
- **Read-Only Frozen Dependencies Consumed by `Plan 6B`:**
  - `core/flight_core/include/vertirta/core/{vrta_recorder.hpp,geodesy.hpp}`, `core/copilot/` (`vertirta::copilot` via `include/vertirta/copilot/{conformal_gate,dsl_parser,z3_corridor_verifier}.hpp`), `core/fleet_z3/include/vertirta/fleet_z3/fleet_z3_scheduler.hpp`, `config/{vehicle_x8_240,istanbul_ltfj_airspace}.yaml`, `config/vertirta_dsl_v1.gbnf`, `scripts/ci/{check_zero_gpl_symbols,check_public_hygiene,check_arch_isolation}.py`.

### Exported Interfaces Consumed by `Plan 6A` & `Plan 6C`
| Producer (`Plan 6B`) | Consumer | Stable Contract |
|---|---|---|
| `core/voice/CMakeLists.txt` (`vertirta::voice`) & `core/forensics/CMakeLists.txt` (`vertirta::forensics`) | `Plan 6A` Wave 2 root `CMakeLists.txt` & `m3_perception_voice_forensics.yml` | Dual-mode standalone/subdir targets; `VERTIRTA_REQUIRE_LIBSODIUM=ON` respected; zero GPL symbols verified via `check_zero_gpl_symbols.py`. |
| `ml/voice/bvi_cockpit_dsp.py` & `ml/voice/audio_license_audit.py` | `Plan 6C` `scripts/colab/g4_jobs/job_whisper_bvi_lora.py` | `synthesize_bvi_cockpit_noise(duration_s, rpm, sr, seed)`, `mix_at_snr(clean, noise, snr_db)`, `classify_snr_stratum(snr_db)`, `audit_audio_manifest(entries)`. |
| `benchmarks/vertibench_v2/` (`train.jsonl`, `cal.jsonl`, `test.jsonl`, `fleet20.jsonl`) & `ml/copilot/calibrate_conformal.py` | `Plan 6C` `scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py` | `N_cal = 2,000` (`800 TR + 800 EN + 400 MIX`, `500/SNR`; primary TR+EN `1,600 >= 1,500`), `N_test = 1,200`, `cv_select_gamma_tau_calibration_only()`, `fit_mondrian_conformal_groups()`, `evaluate_mondrian_conformal()`, `bootstrap_resplit_mondrian_coverage(n_resplits=100)`. |
| `docs/paper_corl/corl_metrics_manifest.json` & `scripts/verify_corl_numbers.py` | Phase 3 post-`G4` ingest from `Plan 6C` `artifacts/v4.0.0/` | Explicit JSON schema ingesting `g4_qwen3_v3_vertibench_v2_eval.json`, `g4_whisper_bvi_lora_eval.json`, and `g4_pidnet_v21_xfeat_metrics.json` into deterministic `.tex` tables & PDF figures. |

### `Plan 6B` Internal 3-Agent File Ownership Matrix (100% Disjoint)
| Phase 2 Owner | Tasks | Exclusive Files Created / Modified |
|---|---|---|
| **`@ai-data-auditor`** | Tasks 1, 2, 3, 4 | `config/{voice_atc.yaml,forensics_eccairs2.yaml,eccairs2_nodigit.gbnf}`, `core/voice/**`, `tests/voice/**`, `ml/voice/**`, `benchmarks/vertibench_v2/**`, `ml/copilot/calibrate_conformal.py`, `core/forensics/**`, `tests/forensics/**`, `tests/python/{test_voice_bvi_and_license_gate.py,test_vertibench_v2_and_mondrian.py}` |
| **`@px4-sim-eng`** | Task 5 | `ros2_ws/src/vertirta_voice_node/**`, `tests/integration/test_m3_voice_node_contracts.py` |
| **`@career-reg-analyst`** | Task 6 | `docs/paper_corl/**` (`vertirta_corl_manuscript.tex`, `references_corl.bib`, `corl_metrics_manifest.json`, `README.md`, `scripts/**`, `generated/**`, `figures/**`, `tests/**`) |

---

### Task 1: Offline Cockpit-Noise TR/EN Voice ATC Engine, Zero-GPL `sherpa-onnx >= 2.0.0` TTS & 4-Stratum SNR Mondrian KnowNo Gate (`@ai-data-auditor`)

**Files:**
- Create: `config/voice_atc.yaml`
- Create: `core/voice/CMakeLists.txt`
- Create: `core/voice/include/vertirta/voice/{voice_types.hpp,silero_vad.hpp,dual_tier_whisper_asr.hpp,zero_gpl_tts.hpp,mondrian_voice_gate.hpp,voice_pipeline.hpp}`
- Create: `core/voice/src/{silero_vad.cpp,dual_tier_whisper_asr.cpp,zero_gpl_tts.cpp,mondrian_voice_gate.cpp,voice_pipeline.cpp}`
- Create: `tests/voice/{CMakeLists.txt,test_voice_pipeline.cpp,test_mondrian_voice_gate.cpp}`

**Interfaces:**
- Consumes: `vertirta::copilot` (`CandidateScore`, `ConformalDecision`, `ConformalOutcome`, `RtaStateSnapshotV1`, `kLoiterLivenessMinSocMarginPct = 5.0f`, `evaluate_conformal_gate`) read-only; optional native `whisper.cpp` (`MIT`, `VERTIRTA_HAS_WHISPER`) and `sherpa-onnx >= 2.0.0` (`Apache-2.0`, `VERTIRTA_HAS_SHERPA_ONNX` — CMake fails configure if `SHERPA_ONNX_VERSION < 2.0.0` or `espeak`/`piper_phonemize` is detected!). When native model binaries are absent in headless CI, runs deterministic Apache-2.0 software backends (`AsrBackend::DETERMINISTIC_CI_ADAPTER`, `TtsBackend::DETERMINISTIC_PHONEME_SYNTH`) exercising the exact VAD framing, `audio_ctx` sizing, Turkish G2P `phoneme_codepoints`, clause streaming, and Mondrian gate logic without GPL dependencies.
- Produces: Static library `vertirta::voice` exposing `SileroVadGate`, `DualTierWhisperAsr`, `TurkishG2pPhonemizer`, `ZeroGplSherpaTts`, `MondrianVoiceConformalGate`, and end-to-end `VoiceAtcPipeline`.

- [ ] **Step 1: Write failing tests in `tests/voice/test_voice_pipeline.cpp` and `tests/voice/test_mondrian_voice_gate.cpp`**
```cpp
// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vertirta/voice/dual_tier_whisper_asr.hpp"
#include "vertirta/voice/mondrian_voice_gate.hpp"
#include "vertirta/voice/silero_vad.hpp"
#include "vertirta/voice/zero_gpl_tts.hpp"

using namespace vertirta::voice;

TEST_CASE("DualTierWhisperAsr enforces audio_ctx [256, 384] rounding, no_context, translate=false, and K=4 N-best") {
  REQUIRE(compute_bounded_audio_ctx(1.2f) == 256U);
  REQUIRE(compute_bounded_audio_ctx(3.0f) == 320U);
  REQUIRE(compute_bounded_audio_ctx(6.0f) == 384U);
  REQUIRE(compute_bounded_audio_ctx(25.0f) == 384U);  // Hard cap prevents >4s decoder stall

  const WhisperDecodePolicy pol = make_default_whisper_policy();
  REQUIRE_FALSE(pol.translate);
  REQUIRE(pol.no_context);
  REQUIRE(pol.carry_initial_prompt);
  REQUIRE(pol.nbest_k == 4U);
  REQUIRE(pol.fast_model_tier == AsrModelTier::WHISPER_SMALL_Q8_0);
  REQUIRE(pol.verify_model_tier == AsrModelTier::WHISPER_LARGE_V3_TURBO_Q8_0);
}

TEST_CASE("ZeroGplSherpaTts enforces sherpa-onnx >= 2.0.0, bans piper1-gpl & NC TR Piper voices, and maps Turkish G2P") {
  REQUIRE(is_tts_voice_license_allowed("en_US-ljspeech-medium"));
  REQUIRE(is_tts_voice_license_allowed("en_US-libritts_r-medium"));
  REQUIRE(is_tts_voice_license_allowed("tr_TR-ema-lightning-apache2"));
  REQUIRE(is_tts_voice_license_allowed("tr_TR-okur-tts-apache2"));
  REQUIRE(is_tts_voice_license_allowed("vits-piper-tr_TR-vertirta-clean"));

  // Hard license bans (Erratum #8 / DR-12)
  REQUIRE_FALSE(is_tts_voice_license_allowed("tr_TR-dfki-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("tr_TR-fahrettin-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("tr_TR-fettah-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("en_US-lessac-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("en_US-ryan-medium"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("xtts-v2"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("mms-tts-tur"));
  REQUIRE_FALSE(is_tts_voice_license_allowed("piper1-gpl"));

  REQUIRE(check_sherpa_onnx_semver(2, 0, 0));
  REQUIRE_FALSE(check_sherpa_onnx_semver(1, 12, 9));  // Pre-PR #4055 espeak-ng GPL contamination

  const auto cps = TurkishG2pPhonemizer::text_to_phoneme_codepoints("Kartal Sahil'e yönel, irtifa 80 metre.");
  REQUIRE(cps.size() >= 16U);
  ZeroGplSherpaTts tts(make_default_tts_config());
  const auto res = tts.synthesize_clause_stream("Kartal Sahil'e yönel.", VoiceLang::TR);
  REQUIRE(res.ok);
  REQUIRE(res.ttfa_ms < 250.0);
  REQUIRE_FALSE(res.pcm_16khz.empty());
}

TEST_CASE("MondrianVoiceConformalGate enforces S0..S3 strata, alpha=0.05, MAYDAY w_m>=0.15, and <5% SoC liveness") {
  MondrianGateConfig cfg = make_default_mondrian_config();
  REQUIRE_THAT(cfg.alpha, Catch::Matchers::WithinAbs(0.05f, 1e-6f));
  REQUIRE(cfg.mayday_asymmetric_weight_floor >= 0.15f);
  REQUIRE(classify_snr_stratum(18.0f) == SnrStratum::S0_CLEAN);
  REQUIRE(classify_snr_stratum(11.0f) == SnrStratum::S1_CRUISE);
  REQUIRE(classify_snr_stratum(4.5f) == SnrStratum::S2_HOVER);
  REQUIRE(classify_snr_stratum(0.5f) == SnrStratum::S3_GARBLED);

  MondrianVoiceConformalGate gate(cfg);

  // 1) MAYDAY in S3_GARBLED with w_m = 0.22 (>= 0.15) triggers immediate emergency preemption, never silence/abstain
  VoiceTurnSignals mayday_sig{};
  mayday_sig.snr_db = 0.2f;
  mayday_sig.nospeech_prob = 0.18f;
  mayday_sig.mayday_acoustic_weight = 0.22f;
  mayday_sig.min_safety_slot_prob = 0.41f;
  mayday_sig.nbest_shannon_entropy_norm = 0.62f;

  vertirta::copilot::RtaStateSnapshotV1 nominal_rta{};
  nominal_rta.soc_pct = 45.0f;
  nominal_rta.soc_reserve_floor_pct = 12.0f;

  const auto emg = gate.evaluate(mayday_sig, /*candidates=*/{}, nominal_rta);
  REQUIRE(emg.decision == VoiceGateAction::EMERGENCY_PREEMPT);
  REQUIRE(emg.stratum == SnrStratum::S3_GARBLED);

  // 2) <5.0% SoC loiter margin forces LIVENESS_EMERGENCY_COMMIT even when conformal set is multi-element
  vertirta::copilot::RtaStateSnapshotV1 low_soc{};
  low_soc.soc_pct = 15.8f;
  low_soc.soc_reserve_floor_pct = 12.0f;  // margin = 3.8% < 5.0%
  std::vector<vertirta::copilot::CandidateScore> ambig_cands{
      {.id = "DIVERT_SLZ:Kartal_Sahil", .prob = 0.48f, .z3_sat = true},
      {.id = "DIVERT_SLZ:Dragos_Plaza", .prob = 0.44f, .z3_sat = true},
  };
  VoiceTurnSignals cruise_sig{};
  cruise_sig.snr_db = 10.5f;
  cruise_sig.nospeech_prob = 0.05f;
  cruise_sig.min_safety_slot_prob = 0.72f;
  const auto liveness = gate.evaluate(cruise_sig, ambig_cands, low_soc);
  REQUIRE(liveness.decision == VoiceGateAction::LIVENESS_EMERGENCY_COMMIT);
  REQUIRE(liveness.selected_action_id == "DIVERT_SLZ:Kartal_Sahil");
}
```
- [ ] **Step 2: Run standalone configure to confirm red phase** — Configure `/tmp/vertirta_m3_voice` targeting `tests/voice` -> FAIL (headers missing).
- [ ] **Step 3: Implement `config/voice_atc.yaml` and `core/voice/{CMakeLists.txt,include/**,src/**}`**
  - `silero_vad.{hpp,cpp}`: `16 kHz` `512`-sample (`32 ms`) ring-buffered frame endpointing (`Silero VAD v6.2.0` MIT ONNX hook + deterministic energy/spectral-flatness VAD adapter for headless CI), `300 ms` pre-speech lookback, `450 ms` trailing hangover, emitting `SpeechSegment` + real-time RMS SNR estimate (`snr_db`) + SHA-256 PCM digest for `.vrta_events.jsonl`.
  - `dual_tier_whisper_asr.{hpp,cpp}`:
    - Computes `audio_ctx = std::clamp(round_up_64(ceil(dur_s / 30.0 * 1500.0) + 128), 256U, 384U)`.
    - Hard-locks `translate = false`, `no_context = true`, `carry_initial_prompt = true`, bilingual aviation initial prompt (`Kartal Sahil, Dragos Plaza, Maltepe Park, Cevizli Helipad, LTFJ, LTBX, MAYDAY, PAN-PAN, irtifa, divert`).
    - Tier-1 `whisper-small` Q8_0 (`~0.40 s` Ryzen 5 5600X AVX2 target) runs first; escalates to Tier-2 `whisper-large-v3-turbo` Q8_0 (`~1.16–1.26 s`, Vulkan/OpenBLAS/CPU hooks) whenever `mean_token_plog < -0.35`, `snr_db < 8.0 dB`, `min_safety_slot_prob < 0.80`, or emergency keyword detected. Exports `K = 4` N-best hypotheses with normalized Shannon entropy `H_nbest`, min safety-slot token probability `p_min_slot`, and `<|nospeech|>` probability.
  - `zero_gpl_tts.{hpp,cpp}`:
    - Pure C++20 `TurkishG2pPhonemizer` handling UTF-8 Turkish graphemes (`ç, ğ, ı, İ, ö, ş, ü`), soft-g lengthening, aviation callsign/numeral expansion (`0..999`, `LTFJ`, `AGL`, `kt`, `m/s`), outputting `phoneme_codepoints` directly to `sherpa-onnx >= 2.0.0` (ZERO `espeak-ng`, ZERO `piper-phonemize`).
    - Compile-time + runtime allowlist/denylist rejecting `piper1-gpl`, ALL prebuilt `tr_TR` Piper voices, `en_US-lessac/ryan/hfc/l2arctic`, `mms-tts-*`, `XTTS-v2`, `OmniVoice`, `Supertonic`, `matcha-icefall`.
    - Clause-streaming synthesis (`num_threads = 2`, first-clause split on `[,.;:!?]`, target `< 250 ms` TTFA).
  - `mondrian_voice_gate.{hpp,cpp}`:
    - 4 SNR strata `S0_CLEAN (>15 dB)`, `S1_CRUISE (8–15 dB)`, `S2_HOVER (2–8 dB)`, `S3_GARBLED (<2 dB)` with stratum-specific calibrated conformal quantiles `q_hat[4]` (and language-group fallback `max(q_hat_lang, q_hat_snr)` so conditional coverage never under-covers at `alpha = 0.05`).
    - Composite voice non-conformity score `s_voice = 1 - p_cal + lambda_H * H_nbest + lambda_slot * (1 - p_min_slot) + lambda_ns * p_nospeech` clipped to `[0, 1]`.
    - Deterministic 5-branch priority order: (1) MAYDAY/PAN-PAN `w_m >= 0.15` -> `EMERGENCY_PREEMPT`; (2) `< 5.0%` SoC loiter margin -> `LIVENESS_EMERGENCY_COMMIT` (delegating to `vertirta::copilot::evaluate_conformal_gate`); (3) `p_nospeech > 0.70` or `|C_g| == 0` -> `ABSTAIN_HOLD_SAFE`; (4) `|C_g| == 1` -> `EXECUTE_SINGLE`; (5) `|C_g| >= 2` -> `REQUEST_CLARIFICATION` (`RSN=LOW_ASR` in `S2/S3`, `RSN=AMBIG` in `S0/S1`).
  - `voice_pipeline.{hpp,cpp}`: Orchestrates VAD -> Dual-Tier ASR -> Mondrian Gate -> Zero-GPL TTS readback, returning a self-contained `VoiceTurnAuditRecord` ready for `.vrta_events.jsonl`.
- [ ] **Step 4: Build & verify under ASan/UBSan + TSan + Zero-GPL gate**
  - Configure/build standalone `/tmp/vertirta_m3_voice` with `-Wall -Wextra -Wpedantic -Werror -fsanitize=address,undefined` -> `ctest --output-on-failure` PASS.
  - Run `python3 scripts/ci/check_zero_gpl_symbols.py --targets /tmp/vertirta_m3_voice/libvertirta_voice.a /tmp/vertirta_m3_voice/test_voice_pipeline /tmp/vertirta_m3_voice/test_mondrian_voice_gate --source-roots core/voice/` -> `0` GPL/espeak/piper_phonemize hits.
- [ ] **Step 5: Phase 2 Gate Checkpoint** — Pure UTF-8 LF, zero edits outside `Plan 6B` Task 1 files.

---

### Task 2: Apache-2.0 Procedural 8-Rotor Coaxial BVI Cockpit Noise DSP & Audio/TTS License Firewall (`@ai-data-auditor`)

**Files:**
- Create: `ml/voice/__init__.py`
- Create: `ml/voice/bvi_cockpit_dsp.py`
- Create: `ml/voice/audio_license_audit.py`
- Create: `tests/python/test_voice_bvi_and_license_gate.py`

**Interfaces:**
- Consumes: Frozen `config/vehicle_x8_240.yaml` (`rotor_count = 8`, `blades_per_rotor = 3`, `rotor_radius_m = 0.75`, `coax_spacing_m = 0.35`, `omega_hover_rad_s = 151.8`, `omega_max_rad_s = 240.0`).
- Produces: Reusable Apache-2.0 Python package `ml.voice` consumed directly by `Plan 6C`'s `scripts/colab/g4_jobs/job_whisper_bvi_lora.py` and `Plan 6B`'s `benchmarks/vertibench_v2/`.

- [ ] **Step 1: Write failing tests in `tests/python/test_voice_bvi_and_license_gate.py`**
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
import pytest

from ml.voice.bvi_cockpit_dsp import (
    X8_240_ACOUSTIC_PARAMS,
    classify_snr_stratum,
    mix_at_snr,
    synthesize_bvi_cockpit_noise,
)
from ml.voice.audio_license_audit import (
    audit_audio_manifest,
    filter_fsd50k_clips,
    is_audio_corpus_allowed,
)


def test_bvi_dsp_locks_canonical_x8_240_rpm_and_bpf_overriding_dr12_stale_1850_rpm() -> None:
    assert X8_240_ACOUSTIC_PARAMS.blades_per_rotor == 3
    assert X8_240_ACOUSTIC_PARAMS.rotor_radius_m == pytest.approx(0.75, abs=1e-6)
    assert X8_240_ACOUSTIC_PARAMS.rpm_hover == pytest.approx(1449.6, abs=1.0)
    assert X8_240_ACOUSTIC_PARAMS.rpm_max == pytest.approx(2291.8, abs=1.0)
    assert X8_240_ACOUSTIC_PARAMS.bpf_max_hz == pytest.approx(114.6, abs=0.5)

    noise = synthesize_bvi_cockpit_noise(duration_s=2.0, rpm=2291.8, sr=16000, seed=42, rt60_s=0.18, vhf_bandlimit=False)
    assert noise.shape == (32000,)
    assert np.all(np.isfinite(noise))
    # Verify spectral peak around fundamental BPF (~114.6 Hz) & 2*B coaxial crossover (~229.2 Hz)
    spec = np.abs(np.fft.rfft(noise * np.hanning(len(noise))))
    freqs = np.fft.rfftfreq(len(noise), d=1.0 / 16000.0)
    bpf_band = spec[(freqs >= 108.0) & (freqs <= 122.0)].max()
    bg_band = np.median(spec[(freqs >= 500.0) & (freqs <= 800.0)])
    assert bpf_band > 5.0 * bg_band


def test_snr_mixer_and_four_stratum_classifier_accuracy() -> None:
    t = np.linspace(0.0, 1.0, 16000, endpoint=False)
    speech = 0.25 * np.sin(2.0 * np.pi * 440.0 * t)
    noise = synthesize_bvi_cockpit_noise(duration_s=1.0, rpm=1600.0, sr=16000, seed=7)
    for target_snr, expected_stratum in [(18.0, "S0_Clean"), (11.0, "S1_Cruise"), (5.0, "S2_Hover"), (-1.0, "S3_Garbled")]:
        mixed, actual_snr = mix_at_snr(speech, noise, target_snr)
        assert abs(actual_snr - target_snr) < 0.25
        assert classify_snr_stratum(actual_snr) == expected_stratum


def test_audio_license_firewall_strips_fsd50k_nc_sampling_plus_and_blocks_all_banned_sources() -> None:
    for banned in (
        "UWB-ATCC", "ATCO2", "ESC-50", "DEMAND", "piper1-gpl",
        "tr_TR-dfki-medium", "tr_TR-fahrettin-medium", "tr_TR-fettah-medium",
        "en_US-lessac-medium", "XTTS-v2", "mms-tts-tur", "UAVid", "CohereForAI/aya-expanse-8b",
    ):
        assert not is_audio_corpus_allowed(banned), f"Failed to block banned asset: {banned}"

    for allowed in ("CommonVoice-27.0-CC0", "ISSAI-TSC-MIT", "FLEURS-CC-BY-4.0", "MUSAN-CC-BY-4.0", "OpenSLR-28-RIRS", "VertiRTA-BVI-DSP-Apache2"):
        assert is_audio_corpus_allowed(allowed)

    # Quarantined / audit-required assets must fail strict allowlist until cleared
    assert not is_audio_corpus_allowed("ATCOSIM")
    assert not is_audio_corpus_allowed("pocket-tts-turkish")

    # FSD50K NC + Sampling+ clip filter test
    mock_fsd = {
        "1001": {"license": "http://creativecommons.org/publicdomain/zero/1.0/"},
        "1002": {"license": "http://creativecommons.org/licenses/by/4.0/"},
        "1003": {"license": "http://creativecommons.org/licenses/by-nc/3.0/"},
        "1004": {"license": "http://creativecommons.org/licenses/sampling+/1.0/"},
    }
    kept, stripped = filter_fsd50k_clips(mock_fsd)
    assert kept == ["1001", "1002"]
    assert stripped == {"cc_by_nc": 1, "sampling_plus": 1, "other_rejected": 0}
```
- [ ] **Step 2: Run `uv run --no-project pytest tests/python/test_voice_bvi_and_license_gate.py`** -> FAIL (module not found).
- [ ] **Step 3: Implement `ml/voice/bvi_cockpit_dsp.py` and `ml/voice/audio_license_audit.py`**
  - `bvi_cockpit_dsp.py`:
    - Loads `config/vehicle_x8_240.yaml` directly so acoustic constants (`B = 3`, `R = 0.75 m`, `omega_hover = 151.8 rad/s`, `omega_max = 240.0 rad/s`, `1,100–2,292 RPM`, `55.0–114.6 Hz` BPF) stay single-source-of-truth.
    - Combines: (a) 8-rotor harmonic BPF series (`k = 1..8`) with deterministic per-arm phase jitter (`±1.5%` RPM beat), (b) coaxial `2*B` upper-lower blade-crossover impulsive BVI pulse train (amplified during descent/hover regimes), (c) PMSM/BLDC inverter carrier whine (`1.8–4.8 kHz`), (d) turbulent broadband pink/brown aero-noise, (e) synthetic small-cabin reverberation (`RT60` uniform in `[0.12, 0.25] s`), and (f) optional `300–3,400 Hz` VHF radio bandpass + soft clipping.
    - Exposes deterministic SNR calibration curriculum (`S0_Clean: >15 dB`, `S1_Cruise: 8–15 dB`, `S2_Hover: 2–8 dB`, `S3_Garbled: -3..+2 dB`) and `mix_at_snr(speech, noise, target_snr_db)` returning exact measured active-speech SNR.
  - `audio_license_audit.py`:
    - Encodes the complete DR-12 + Errata `#7/#8` license allowlist, hard-ban registry, `ATCOSIM` quarantine flag (`QUARANTINED_PENDING_DOC_LICENSE_TXT`), `pocket-tts-turkish` TR-Wiki CC BY-SA audit flag, `FSD50K` clip-level filter (reproducing exact strip logic for `4,616` dev + `1,425` eval CC-BY-NC & `1,777` Sampling+ clips via `*_clips_info_FSD50K.json`), and JSON report exporter `audit_audio_manifest()` invoked by `Plan 6C` before any `G4` Whisper training step starts!
- [ ] **Step 4: Run `uv run --no-project pytest tests/python/test_voice_bvi_and_license_gate.py`** -> PASS.

---

### Task 3: `VertiBench-v2` (Track A Voice TR/EN + Track B Fleet20) & Language/SNR Mondrian Conformal Calibration (`@ai-data-auditor`)

**Files:**
- Create: `benchmarks/vertibench_v2/__init__.py`, `benchmarks/vertibench_v2/README.md`, `benchmarks/vertibench_v2/schema.py`, `benchmarks/vertibench_v2/fleet20_scenarios.py`, `benchmarks/vertibench_v2/build_vertibench_v2.py`, `benchmarks/vertibench_v2/validate_vertibench_v2.py`
- Modify: `ml/copilot/calibrate_conformal.py` (backward-compatible extension — all existing v3.5.1 tests MUST stay green!)
- Create: `tests/python/test_vertibench_v2_and_mondrian.py`

**Interfaces:**
- Consumes: Frozen `config/vertirta_dsl_v1.gbnf` (DSL grammar parity), `config/istanbul_ltfj_airspace.yaml` (strictly west of `29.1895° E`, Kartal datum `40.8860° N, 29.1880° E, N0 = +36.7 m`), `ml.voice.bvi_cockpit_dsp.classify_snr_stratum`.
- Produces:
  - Deterministic, self-validating **`VertiBench-v2`** benchmark builder + validator (`benchmarks/vertibench_v2/`) leaving `benchmarks/vertibench_tr_en/` (`VertiBench-v1`) 100% untouched:
    - **Track A (`VertiBench-Voice-TR/EN`):**
      - `train`: `3,600` scenarios (`1,600 TR + 1,600 EN + 400 MIX`; `900` each of `NOMINAL`, `FAULT`, `ADVERSARIAL`, `AMBIGUOUS_ASR`; balanced `EN+TR` Bucket-3 adversarial/violation coverage explicitly fixing `v3.5.1`'s `EN B3 = 0%` imbalance!).
      - `calibration`: **`N_cal = 2,000`** (`800 TR + 800 EN + 400 MIX`, **`500`** per SNR stratum `S0..S3` `>= 250`, **`500`** per bucket `NOMINAL/FAULT/ADVERSARIAL/AMBIGUOUS_ASR`). Satisfies DR-13's exact Beta-inversion floor (`n_lang >= 750`, `n_snr >= 250`) AND gives `Plan 6C`'s `job_qwen3_v3_vertibench_v2.py` exact group-conditional calibration!
      - `test`: **`N_test = 1,200`** (`500 TR + 500 EN + 200 MIX`, **`300`** per SNR stratum `S0..S3`, `300` per bucket).
      - **5-Layer Anti-Contamination & Honest Provenance:**
        1. Mandatory provenance field on every split & manifest: **`PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2`** (hard-fails if any file claims `"HUMAN_NATIVE"` without a signed human-annotator audit log!).
        2. Disjoint lexical/template/seed partitions (`seed_overlap == 0` between `train` and `calibration U test`).
        3. Exact character 5-gram and token 3-gram Jaccard gate (`max_char_5gram_jaccard < 0.15`, `max_token_3gram_jaccard < 0.15`).
        4. Disjoint ASR acoustic corruption / phonetic error perturbations (`S0..S3` BVI SNR + Turkish/English aviation homophone confusion pairs held out from `train`).
        5. Zero BIG-Bench canary strings (per DR-13 contamination warning that static canary strings backfire in public repos) + SHA-256 split manifest pinning.
    - **Track B (`VertiBench-Fleet20`):** `200` scenarios (`100 cal` + `100 test`, `N = 5..20` aircraft in Kartal–Dragos–Cevizli–Maltepe corridor west of `29.1895° E`) covering saturated FATO queuing, `< 60 m` dependent-FATO downwash mutex, `120–180 s` LTFJ heavy-wake crossing, Regulation `(EU) 2021/664` Art. 10(8)/(9)/(10) MAYDAY & EMS preemption, and adversarial UNSAT voice commands violating approved U-plans or ATS clearance.
  - **Extended `ml/copilot/calibrate_conformal.py` (for `Plan 6C`'s `job_qwen3_v3_vertibench_v2.py`):**
    - `cv_select_gamma_tau_calibration_only(logprob_rows, token_count_rows, gold_indices, gammas=(0.5, 0.75, 1.0), n_folds=5)`: 5-fold calibration-only CV selecting `(gamma*, tau*)` minimizing NLL & expected conformal set size with **zero test leakage** (solving `v3.5.1`'s length-bias inflation that left `Mean Set Size = 3.125`).
    - `fit_mondrian_conformal_groups(prob_rows, gold_indices, group_labels, alpha=0.05)`: computes exact finite-sample Mondrian quantiles `q_hat[g] = Quantile(s_g; ceil((n_g + 1) * (1 - alpha)) / n_g)` for `lang in {TR, EN, MIX}` and `snr in {S0_Clean, S1_Cruise, S2_Hover, S3_Garbled}`, plus conservative hierarchical intersection `max(q_hat_lang[l], q_hat_snr[s])` when joint `(l, s)` cell size `< 250`.
    - `evaluate_mondrian_conformal(...)` and `bootstrap_resplit_mondrian_coverage(..., n_resplits=100, seed=42)` producing the exact `100`-resplit mean ± SD / 95% CI coverage, ECE, singleton execution rate, and mean set size required by Paper 2 (`docs/paper_corl/`).

- [ ] **Step 1: Write failing tests in `tests/python/test_vertibench_v2_and_mondrian.py`**
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import random
import pytest

from benchmarks.vertibench_v2.build_vertibench_v2 import build_vertibench_v2_splits
from benchmarks.vertibench_v2.schema import HONEST_PROVENANCE_TAG_V2
from benchmarks.vertibench_v2.validate_vertibench_v2 import validate_vertibench_v2
from ml.copilot.calibrate_conformal import (
    bootstrap_resplit_mondrian_coverage,
    cv_select_gamma_tau_calibration_only,
    fit_mondrian_conformal_groups,
)


def test_vertibench_v2_split_counts_honest_provenance_and_anti_contamination(tmp_path: Path) -> None:
    manifest = build_vertibench_v2_splits(out_dir=tmp_path, seed=20261124)
    assert manifest["provenance_tag"] == HONEST_PROVENANCE_TAG_V2
    assert "PENDING_HUMAN_NATIVE" in manifest["provenance_tag"]
    assert manifest["counts"]["train"] == 3600
    assert manifest["counts"]["calibration"] == 2000
    assert manifest["counts"]["test"] == 1200
    assert manifest["counts"]["fleet20"] == 200

    report = validate_vertibench_v2(tmp_path)
    assert report.passed, f"Validation failed: {report.errors}"
    assert report.max_char_5gram_jaccard < 0.15
    assert report.max_token_3gram_jaccard < 0.15
    assert report.seed_overlap == 0
    assert report.cal_lang_counts == {"TR": 800, "EN": 800, "MIX": 400}
    assert all(c == 500 for c in report.cal_snr_counts.values())
    assert report.test_lang_counts == {"TR": 500, "EN": 500, "MIX": 200}
    assert report.all_fleet20_west_of_29_1895 is True
    assert report.vertibench_v1_unmodified is True


def test_mondrian_conformal_and_100_resplit_bootstrap_enforces_alpha_0_05() -> None:
    rng = random.Random(42)
    probs, golds, groups = [], [], []
    for g_idx, g in enumerate(["S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"]):
        for _ in range(300):
            # Noise increases from S0 to S3 -> Mondrian quantiles must widen monotonically
            p_gold = max(0.25, min(0.99, 0.94 - 0.11 * g_idx + rng.uniform(-0.08, 0.05)))
            rem = (1.0 - p_gold) / 3.0
            row = [p_gold, rem, rem, rem]
            probs.append(row)
            golds.append(0)
            groups.append(g)

    fit = fit_mondrian_conformal_groups(probs, golds, groups, alpha=0.05)
    assert fit["alpha"] == pytest.approx(0.05, abs=1e-9)
    assert fit["group_q_hat"]["S3_Garbled"] > fit["group_q_hat"]["S0_Clean"]

    boot = bootstrap_resplit_mondrian_coverage(probs, golds, groups, alpha=0.05, n_resplits=100, seed=7)
    assert boot["n_resplits"] == 100
    for g in ["S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled"]:
        assert boot["group_metrics"][g]["coverage_mean"] >= 0.94  # finite-sample around 0.95
```
- [ ] **Step 2: Run `uv run --no-project pytest tests/python/test_vertibench_v2_and_mondrian.py`** -> FAIL (modules missing).
- [ ] **Step 3: Implement `benchmarks/vertibench_v2/**` and extend `ml/copilot/calibrate_conformal.py`**
  - Ensure `build_vertibench_v2.py` uses genuinely distinct lexical pools, grammatical structures, and scenario seeds between `train` and `calibration U test` so `max_char_5gram_jaccard < 0.15` and `max_token_3gram_jaccard < 0.15` hold empirically across all `3,600 x 3,200` comparisons (not merely mocked!).
  - Ensure `fleet20_scenarios.py` locks Kartal `(40.8860, 29.1880)`, `max_lon < 29.1895`, `N = 5..20`, `< 60 m` dependent FATO mutex, `120–180 s` LTFJ wake, and Art. 10(8)/(9)/(10) preemption flags matching `vertirta::fleet_z3::AircraftSlotRequest`.
  - Keep all existing public function signatures in `ml/copilot/calibrate_conformal.py` 100% backward-compatible.
- [ ] **Step 4: Run pytest across BOTH new & legacy conformal/VertiBench suites**
  - `uv run --no-project pytest tests/python/test_vertibench_v2_and_mondrian.py tests/python/test_copilot_l4_runner.py` -> PASS (`0` regressions to `VertiBench-v1` or `v3.5.1`).

---

### Task 4: Deterministic Post-Flight Regulatory Forensics, `.vrta_events.jsonl` Sidecar, `CDR (EU) 2020/2034` ERCS, XSD-Valid `.e5x` & Zero-Digit GBNF Grounding (`@ai-data-auditor`)

**Files:**
- Create: `config/eccairs2_nodigit.gbnf`, `config/forensics_eccairs2.yaml`
- Create: `core/forensics/CMakeLists.txt`
- Create: `core/forensics/include/vertirta/forensics/{vrta_events_sidecar.hpp,ercs_scorer.hpp,eccairs2_e5x_emitter.hpp,narrative_grounding.hpp,forensics_pipeline.hpp}`
- Create: `core/forensics/src/{vrta_events_sidecar.cpp,ercs_scorer.cpp,eccairs2_e5x_emitter.cpp,narrative_grounding.cpp,forensics_pipeline.cpp}`
- Create: `core/forensics/schemas/eccairs2_e5x_minimal.xsd`
- Create: `tests/forensics/{CMakeLists.txt,test_forensics_sidecar_ercs_e5x.cpp,test_e5x_xsd_and_grounding.py}`

**Interfaces:**
- Consumes: Read-only `vertirta::core` (`VrtaFileHeader` `128 B`, `VrtaFrame100Hz` `256 B`, `verify_vrta_header`, `verify_vrta_frame`, `crc32c`); pinned **`libsodium >= 1.0.21`** controlled by `VERTIRTA_REQUIRE_LIBSODIUM` (same pattern as M2 `core/uspace_fleet/` — rejects `<= 1.0.20` Ed25519 subgroup CVE at compile time, explicit `SIM_SHA256_TEST_SIGNER` only when `VERTIRTA_REQUIRE_LIBSODIUM=OFF` locally, forced `ON` in M3 Cloud CI!).
- Produces: Static library `vertirta::forensics` + CLI emitter `vertirta_forensics_cli` converting `(flight.vrta, flight.vrta_events.jsonl)` -> verified `occurrence_draft.e5x.xml` + `forensics_audit_report.json`.

- [ ] **Step 1: Write failing tests in `tests/forensics/test_forensics_sidecar_ercs_e5x.cpp` and `tests/forensics/test_e5x_xsd_and_grounding.py`**
```cpp
// SPDX-License-Identifier: Apache-2.0
#include <catch2/catch_test_macros.hpp>

#include "vertirta/core/vrta_recorder.hpp"
#include "vertirta/forensics/eccairs2_e5x_emitter.hpp"
#include "vertirta/forensics/ercs_scorer.hpp"
#include "vertirta/forensics/narrative_grounding.hpp"
#include "vertirta/forensics/vrta_events_sidecar.hpp"

using namespace vertirta::forensics;

TEST_CASE("Frozen 256 B VRTA0001 wire format stays untouched while .vrta_events.jsonl chains (seq, t_sim_us) with SHA-256 + Ed25519") {
  static_assert(sizeof(vertirta::core::VrtaFileHeader) == 128U);
  static_assert(sizeof(vertirta::core::VrtaFrame100Hz) == 256U);

  VrtaEventsSidecarWriter writer;
  SidecarEvent ev1{};
  ev1.seq = 1200U;
  ev1.t_sim_us = 12'000'000U;
  ev1.kind = SidecarEventKind::VOICE_CONFORMAL_TURN;
  ev1.snr_stratum = "S2_Hover";
  ev1.conformal_set_size = 1U;
  ev1.asr_transcript_utf8 = "Kartal Sahil'e acil inis";
  ev1.audio_sha256_hex = std::string(64U, 'a');
  ev1.z3_smt2_sha256_hex = std::string(64U, 'b');
  ev1.gsn_node_id = "G3.2-VOICE-CONFORMAL";

  const std::string line1 = writer.append_serialized(ev1);
  SidecarEvent ev2 = ev1;
  ev2.seq = 1250U;
  ev2.t_sim_us = 12'500'000U;
  ev2.kind = SidecarEventKind::Z3_UNSAT_REPAIR;
  ev2.z3_unsat_core = {"LTFJ_CTR_WALL", "SOC_RESERVE_FLOOR"};
  const std::string line2 = writer.append_serialized(ev2);

  const auto verified = VrtaEventsSidecarReader::verify_chain(line1 + "\n" + line2 + "\n", writer.public_key_hex());
  REQUIRE(verified.valid);
  REQUIRE(verified.events.size() == 2U);
  REQUIRE(verified.events[1].prev_event_sha256 == verified.events[0].event_sha256);

  // Tampering a single byte must break chain verification
  std::string tampered = line1 + "\n" + line2 + "\n";
  tampered[tampered.find("1250")] = '9';
  REQUIRE_FALSE(VrtaEventsSidecarReader::verify_chain(tampered, writer.public_key_hex()).valid);
}

TEST_CASE("ERCS CDR (EU) 2020/2034 enforces SCF-NP, Barrier #8 = 0 on high-energy MAC/LOC-I, and draft/Incident ID 431 ONLY") {
  ErcsInput in{};
  in.trigger = Cir2026Trigger::CRITICAL_ENERGY_OR_ENVELOPE_PROT;
  in.fault_domain = FaultDomain::ROTOR_ACTUATOR_OR_INVERTER;
  in.key_risk_area = KeyRiskArea::LOC_I_HIGH_ENERGY;
  in.occupant_capacity = 1U;  // Canonical X8-240 1-pax lock (NOT stale 4-6 pax!)
  in.barriers = {1, 2, 2, 1, 2, 1, 1, /*barrier8_low_energy_requested=*/3};  // Must be clamped to 0!

  const ErcsResult r = score_ercs_cdr_2020_2034(in);
  REQUIRE(r.cictt_category == "SCF-NP");       // NEVER SCF-PP
  REQUIRE(r.cictt_category != "SCF-PP");
  REQUIRE(r.effective_barrier8_score == 0U);   // High-energy LOC-I forces Barrier #8 = 0 (N/A)
  REQUIRE(r.occurrence_class_id_431 == OccurrenceClassDraft::DRAFT_INCIDENT_ONLY);
  REQUIRE_FALSE(r.auto_classified_accident_or_serious_incident);
  REQUIRE(r.ercs_matrix_code.size() >= 2U);
}

TEST_CASE("No-digit GBNF narrative grounding enforces 0.0% numeric hallucination and Article 16 PII stripping") {
  GroundedTelemetrySlots slots{};
  slots.values["EVENT_SEQ"] = "1200";
  slots.values["MIN_LTFJ_MARGIN_M"] = "42.5";
  slots.values["SOC_PCT"] = "14.2";
  slots.values["ERCS_SCORE"] = "C4";

  const std::string clean_template_en =
      "At frame {{EVENT_SEQ}}, lateral boundary margin reached {{MIN_LTFJ_MARGIN_M}} m at {{SOC_PCT}}% SoC (draft ERCS {{ERCS_SCORE}}).";
  const auto ok = render_and_verify_grounded_narrative(clean_template_en, slots);
  REQUIRE(ok.passed);
  REQUIRE(ok.hallucinated_numeric_tokens.empty());
  REQUIRE(ok.rendered_text.find("42.5") != std::string::npos);

  // Any raw digit injected outside {{SLOT}} MUST be rejected fail-closed!
  const std::string hallucinated =
      "At frame {{EVENT_SEQ}}, speed hit 99.9 kt near Kartal.";
  const auto bad = render_and_verify_grounded_narrative(hallucinated, slots);
  REQUIRE_FALSE(bad.passed);
  REQUIRE(bad.hallucinated_numeric_tokens.size() == 1U);

  // Article 16 PII stripping on Headline ID 601
  const std::string headline = sanitize_article16_headline("Pilot Ahmet Yilmaz (ahmet@example.com) diverted X8-240 near Kartal");
  REQUIRE(headline.find("ahmet@example.com") == std::string::npos);
  REQUIRE(headline.find("Ahmet Yilmaz") == std::string::npos);
}
```
- [ ] **Step 2: Run standalone configure in `/tmp/vertirta_m3_forensics` to verify red phase** -> FAIL (headers missing).
- [ ] **Step 3: Implement `config/eccairs2_nodigit.gbnf`, `config/forensics_eccairs2.yaml`, `core/forensics/**`, and `eccairs2_e5x_minimal.xsd`**
  - `config/eccairs2_nodigit.gbnf`: Strict GBNF grammar allowing UTF-8 EN/TR aviation prose characters and closed-allowlist `{{SLOT_NAME}}` tokens (`{{EVENT_SEQ}}`, `{{T_SIM_S}}`, `{{MIN_LTFJ_MARGIN_M}}`, `{{MIN_LTBX_MARGIN_M}}`, `{{SOC_PCT}}`, `{{V_TERM_V}}`, `{{AGL_M}}`, `{{PEAK_CHI2}}`, `{{ROTOR_MASK_HEX}}`, `{{SNR_STRATUM}}`, `{{CONFORMAL_SET_SIZE}}`, `{{ERCS_SCORE}}`, `{{CICTT_CODE}}`) while **syntactically forbidding ASCII digits `[0-9]`** outside slot identifiers!
  - `vrta_events_sidecar.{hpp,cpp}`: Append-only NDJSON writer (`O_APPEND` + explicit flush/fsync hook for `EUROCAE ED-155A` power-loss resilience) and cryptographic verifier linking every event to `(seq, t_sim_us)` in the frozen `.vrta` binary file, maintaining `prev_event_sha256 -> event_sha256` hash chaining + `libsodium >= 1.0.21` Ed25519 signature (`#if defined(VERTIRTA_HAS_LIBSODIUM)` enforces `SODIUM_LIBRARY_VERSION_MAJOR > 26 || (MAJOR == 26 && MINOR >= 2)` or `sodium_version_string()` semver `>= 1.0.21`!).
  - `ercs_scorer.{hpp,cpp}`: Deterministic `CDR (EU) 2020/2034` 8-barrier scorer (`Barrier #1..#8`, weights `0..18` mapped to `0..9` probability index + vehicle-capacity-aware severity `A..X`), hard-locking `SCF-NP` for rotor/inverter/control/SW/datalink faults, forcing `Barrier #8 = 0 (N/A)` on `MAC` and `LOC-I`, and hard-locking `ID 431 = DRAFT_INCIDENT_ONLY` (`reporting_status = "DRAFT_INTERNAL_PRE_SUBMISSION"` — never auto-emitting `Accident` or `Serious Incident`).
  - `eccairs2_e5x_emitter.{hpp,cpp}`: Emits deterministic UTF-8 XML matching `<SET><Occurrence entityId="24"><ATTRIBUTES>...</ATTRIBUTES><ENTITIES><Narrative entityId="22">...</Narrative>...</ENTITIES></Occurrence></SET>`, guaranteeing **strictly ascending `<ATTRIBUTES>` attribute IDs** (`ID 121` eVTOL rotorcraft flight phase, `ID 431` draft/Incident, `ID 453` reporting entity, `ID 601` Article 16 PII-stripped headline, `ID 659` ATA/JASC chapter, `ID 1109` ERCS score, `CIR (EU) 2026/1821` trigger code, `SHGM SHY-OLAY` 72h ADREP deadline metadata), omitting N/A injury count attributes (never zero-filling N/A counts), and emitting parallel EN + TR `Entity 22` narratives.
  - `narrative_grounding.{hpp,cpp}`: Implements `render_and_verify_grounded_narrative()` + `verify_numeric_grounding()` (zero raw digits permitted in templates; post-render AST/regex pass verifies 100% of numeric literals come strictly from grounded `.vrta`/`.vrta_events.jsonl` slots -> **`0.0%` numeric hallucination**) + `sanitize_article16_headline()` PII redactor.
  - `core/forensics/schemas/eccairs2_e5x_minimal.xsd` + `tests/forensics/test_e5x_xsd_and_grounding.py`: Validates emitted `.e5x` XML end-to-end against XSD (`lxml`/`xmlschema` or stdlib ElementTree structural XSD validator fallback), checks strictly ascending `<ATTRIBUTES>` IDs, checks `ID 431` draft/Incident lock, checks `eccairs2_nodigit.gbnf` syntax forbids `[0-9]`, and checks `0.0%` numeric hallucination across 25 randomized synthetic incident traces!
- [ ] **Step 4: Build & verify under ASan/UBSan + TSan + XSD pytest + Zero-GPL gate**
  - Configure/build standalone `/tmp/vertirta_m3_forensics` -> `ctest --output-on-failure` PASS.
  - Run `uv run --no-project pytest tests/forensics/test_e5x_xsd_and_grounding.py` -> PASS.
  - Run `python3 scripts/ci/check_zero_gpl_symbols.py --targets /tmp/vertirta_m3_forensics/libvertirta_forensics.a --source-roots core/forensics/` -> `0` GPL hits.

---

### Task 5: ROS 2 `vertirta_voice_node` — Asynchronous Voice ATC + Z3 Corridor + Forensics Sidecar Bridge (`@px4-sim-eng`)

**Files:**
- Create: `ros2_ws/src/vertirta_voice_node/CMakeLists.txt`
- Create: `ros2_ws/src/vertirta_voice_node/package.xml`
- Create: `ros2_ws/src/vertirta_voice_node/config/voice_node.yaml`
- Create: `ros2_ws/src/vertirta_voice_node/include/vertirta_voice_node/voice_node_core.hpp`
- Create: `ros2_ws/src/vertirta_voice_node/src/voice_node_core.cpp`
- Create: `ros2_ws/src/vertirta_voice_node/src/voice_node.cpp`
- Create: `ros2_ws/src/vertirta_voice_node/test/CMakeLists.txt`
- Create: `ros2_ws/src/vertirta_voice_node/test/test_voice_node_core.cpp`
- Create: `tests/integration/test_m3_voice_node_contracts.py`

**Interfaces:**
- Consumes: `vertirta::voice` (Task 1), `vertirta::copilot` (Z3 corridor verifier & DSL parser, read-only), `vertirta::forensics::VrtaEventsSidecarWriter` (Task 4).
- Produces: Pure C++20 `VoiceNodeCore` (zero ROS 2 required for standalone ASan/UBSan/TSan CTest — exact mirror of `ros2_ws/src/vertirta_uspace_node/include/vertirta_uspace_node/uspace_core.hpp`!) plus thin `rclcpp` wrapper `src/voice_node.cpp` binding topics `/vertirta/voice/audio_16khz`, `/vertirta/rta/state_snapshot`, `/vertirta/voice/dsl_command`, `/vertirta/voice/tts_readback`, and `/vertirta/forensics/sidecar_event`.

- [ ] **Step 1: Write failing tests in `ros2_ws/src/vertirta_voice_node/test/test_voice_node_core.cpp` and `tests/integration/test_m3_voice_node_contracts.py`**
  - Verify end-to-end audio turn -> VAD -> Dual-Tier ASR -> Mondrian Gate -> Z3 corridor verification -> TTS readback (`< 250 ms` TTFA) -> `.vrta_events.jsonl` sidecar line carrying exact `(seq, t_sim_us)` from active `RtaStateSnapshotV1`.
  - Verify MAYDAY in `S3_Garbled` (`w_m >= 0.15`) preempts normal clarification queue and logs `EMERGENCY_PREEMPT` to `.vrta_events.jsonl`.
  - Verify `VoiceNodeCore` runs completely decoupled from the `100 Hz` RTA loop (bounded SPSC queue, non-blocking submit, zero heap allocation inside RTA callback path).
  - Verify `package.xml`, `config/voice_node.yaml`, and `CMakeLists.txt` contain zero GPL / `espeak` / `piper1-gpl` / NC voice references.
- [ ] **Step 2: Run standalone build & pytest to confirm red phase** -> FAIL.
- [ ] **Step 3: Implement `VoiceNodeCore`, thin `rclcpp` node `src/voice_node.cpp`, `package.xml`, and `config/voice_node.yaml`**
- [ ] **Step 4: Build & verify under ASan/UBSan + TSan + pytest**
  - Build standalone `/tmp/vertirta_m3_voice_node` -> `ctest --output-on-failure` PASS.
  - Run `uv run --no-project pytest tests/integration/test_m3_voice_node_contracts.py` -> PASS.

---

### Task 6: Paper 2 (`docs/paper_corl/`) — Double-Anonymous 8-Page CoRL 2027 Manuscript, Reproducible Tables/Figures & Zero-Drift Auditor (`@career-reg-analyst`)

**Files:**
- Create: `docs/paper_corl/README.md`
- Create: `docs/paper_corl/vertirta_corl_manuscript.tex`
- Create: `docs/paper_corl/references_corl.bib`
- Create: `docs/paper_corl/corl_metrics_manifest.json`
- Create: `docs/paper_corl/scripts/__init__.py`
- Create: `docs/paper_corl/scripts/generate_corl_tables.py`
- Create: `docs/paper_corl/scripts/generate_corl_figures.py`
- Create: `docs/paper_corl/scripts/verify_corl_numbers.py`
- Create: `docs/paper_corl/generated/corl_tables.tex` (auto-generated)
- Create: `docs/paper_corl/figures/{fig1_bilingual_mondrian_pareto.pdf,fig2_snr_stratum_coverage.pdf,fig3_slz_real_transfer_pareto.pdf}` (auto-generated)
- Create: `docs/paper_corl/tests/__init__.py`
- Create: `docs/paper_corl/tests/test_paper_corl_contracts.py`

**Interfaces:**
- Consumes: Verified on-disk baseline artifacts (`artifacts/v3.1.0/{l4_pidnet_distill_metrics,l4_qwen3_4b_z3_grpo_eval,local_gtx1650s_pidnet_trained_proof}.json` and `artifacts/v3.5.1/{l4_qwen3_4b_v2_calibrated_eval,l4_pidnet_v2_distill_metrics,local_gtx1650s_pidnet_v2_proof}.json` covering `AeroScapes` zero-shot SLZ transfer and `VertiBench-v1` calibration) plus Phase 3 `G4` artifacts (`artifacts/v4.0.0/g4_{qwen3_v3_vertibench_v2_eval,whisper_bvi_lora_eval,pidnet_v21_xfeat_metrics}.json` produced by `Plan 6C`). `AeroScapes`, `ICG Drone`, and `UAVScenes` are strictly quarantined zero-shot eval-only (banned from training); **`UAVid` is HARD-BANNED everywhere**.
- Produces: Complete CoRL 2027 8-page double-anonymous LaTeX manuscript + reproducible Python table/figure/verification scripts mirroring `docs/paper_ral/`'s exact engineering quality.
- **Phase 2 -> Phase 3 Anti-Fabrication Protocol:**
  - During Phase 2 (before `G4` runs), `corl_metrics_manifest.json` embeds ONLY verified existing `v3.1.0` / `v3.5.1` numbers plus explicitly structured `phase3_g4_slots` marked `"status": "PENDING_PHASE3_G4_THUNDER_RUN"` (`value: null`) and optional zero-shot external SLZ slots (`ICG Drone`, `UAVScenes`) marked `"status": "PENDING_EVAL_ONLY_REAL_IMAGERY"` (`value: null`) if not executed on `G4`, and `generate_corl_tables.py` renders clean `\text{--}` dashes (NEVER fabricated numbers and NEVER `TODO`/`[TBD]` tokens!).
  - `verify_corl_numbers.py --allow-pending-g4` passes `0` errors in Phase 2; in Phase 3 (post-`G4` Thunder Run), `generate_corl_tables.py --ingest-v400 artifacts/v4.0.0/` populates all `phase3_g4_slots` from live `Plan 6C` `G4` JSON outputs (`g4_qwen3_v3_vertibench_v2_eval.json`, `g4_whisper_bvi_lora_eval.json`, `g4_pidnet_v21_xfeat_metrics.json`) — keeping any unrun `ICG`/`UAVScenes` cells honestly tagged `"PENDING_EVAL_ONLY_REAL_IMAGERY"` rather than fabricating — and strict `verify_corl_numbers.py` (without `--allow-pending-g4`) MUST pass with **`0` pending `phase3_g4_slots` and `0` errors**!

- [ ] **Step 1: Write failing tests in `docs/paper_corl/tests/test_paper_corl_contracts.py`**
```python
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import re

from docs.paper_corl.scripts.verify_corl_numbers import verify_corl_manuscript
from scripts.ci.check_public_hygiene import scan_public_hygiene

REPO = Path(__file__).resolve().parents[3]
CORL_DIR = REPO / "docs/paper_corl"


def test_paper_corl_double_anonymous_bans_uavid_and_enforces_honest_provenance() -> None:
    tex_path = CORL_DIR / "vertirta_corl_manuscript.tex"
    tex = tex_path.read_text(encoding="utf-8")
    bib = (CORL_DIR / "references_corl.bib").read_text(encoding="utf-8")

    assert scan_public_hygiene(REPO, tracked_files=["docs/paper_corl/vertirta_corl_manuscript.tex"])["passed"] is True
    assert "Anonymous Author" in tex
    assert "\\input{generated/corl_tables.tex}" in tex

    # Hard double-anonymity + UAVid ban + anti-placeholder firewall
    for forbidden in ("github.com/", "mailto:", "TODO", "[TBD]", "UAVid", "uavid", "piper1-gpl", "UWB-ATCC", "ATCO2"):
        assert forbidden not in tex, f"Forbidden leak or banned corpus in Paper 2 .tex: {forbidden}"
        assert forbidden not in bib, f"Forbidden leak or banned corpus in Paper 2 .bib: {forbidden}"

    # Must cite neutral Paper 1 safety cage, never self-identifying repository or authors
    assert not re.search(r"we (achieve|demonstrate|are) (DO-178C|SC-VTOL|SORA).*certified", tex, re.IGNORECASE)
    # Honest VertiBench-v2 provenance framing (never claims human-native before human audit exists)
    assert not re.search(r"human[- ]native (Turkish|English|bilingual) benchmark", tex, re.IGNORECASE)

    # All real-imagery SLZ datasets strictly from AeroScapes / ICG / UAVScenes
    for ds in ("AeroScapes", "ICG", "UAVScenes"):
        assert ds in tex

    # BibTeX closure check
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))
    cited = {k.strip() for grp in re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex) for k in grp.split(",") if k.strip()}
    assert cited and cited.issubset(bib_keys), f"Unresolved BibTeX keys: {cited - bib_keys}"


def test_verify_corl_numbers_passes_zero_errors() -> None:
    res = verify_corl_manuscript(CORL_DIR, allow_pending_g4=True)
    assert res["passed"] is True
    assert res["errors"] == []
```
- [ ] **Step 2: Run `uv run --no-project pytest docs/paper_corl/tests/test_paper_corl_contracts.py`** -> FAIL (files missing).
- [ ] **Step 3: Author `docs/paper_corl/**` (`vertirta_corl_manuscript.tex`, `references_corl.bib`, `corl_metrics_manifest.json`, table/figure/verifier scripts)**
  - Structure `vertirta_corl_manuscript.tex` as an 8-page CoRL 2027 submission covering:
    1. **Introduction & Neutral Reference to Deterministic RTA Cage** (treating Paper 1's bare-metal HJI-CBVF/INDI-WLS cage strictly as an external deterministic verifier without breaking anonymity).
    2. **Bilingual Z3-GRPO Mission Synthesis (`VertiBench-v2`)**: Token-level GBNF DFA + Z3 `QF_LRA` verifiable reward, multi-seed (`3–5` seeds) stability, balanced EN+TR adversarial refusal, and honest disclosure of `PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2`.
    3. **Language- & SNR-Stratified Mondrian Conformal Prediction (`alpha = 0.05`)**: Exact finite-sample coverage under 4 cockpit BVI SNR strata (`S0..S3`), `100` Monte Carlo calibration/test resplits (mean ± SD coverage, ECE, singleton execution rate, mean set size `< 1.8`), MAYDAY asymmetric priority (`w_m >= 0.15`), and `< 5.0%` SoC liveness override.
    4. **Active Visual SLZ Real-Imagery Transfer (`AeroScapes` / `ICG Drone` / `UAVScenes`, ZERO `UAVid`)**: `PIDNet-S v2.1` oblique-recall focal/FSR trade-off + `XFeat + LighterGlue` visual relocalization on `GTX 1650 SUPER` (`< 12 ms` p99, `< 300 MiB` VRAM).
    5. **Limitations & Reproducibility**: Explicit scope boundaries (research testbed, no certification claim, pending human-native ATC recording campaign).
  - Run `generate_corl_tables.py` and `generate_corl_figures.py` to generate deterministic `generated/corl_tables.tex` and 3 clean vector PDFs under `docs/paper_corl/figures/`.
- [ ] **Step 4: Run `verify_corl_numbers.py` & pytest**
  - `uv run --no-project python docs/paper_corl/scripts/verify_corl_numbers.py --allow-pending-g4` -> `passed: true`, `0` errors.
  - `uv run --no-project pytest docs/paper_corl/tests/test_paper_corl_contracts.py` -> PASS.

---

## End-to-End Verification Command Matrix (Phase 2 Local & Phase 3 Post-`G4` Gates)

1. **Standalone C++20 Build & Sanitizers (WSL2 Ubuntu — ZERO Root `CMakeLists.txt` Edits in Wave 1):**
   ```bash
   cmake -S tests/voice -B /tmp/vertirta_m3_voice -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
   cmake --build /tmp/vertirta_m3_voice && ctest --test-dir /tmp/vertirta_m3_voice --output-on-failure

   cmake -S tests/forensics -B /tmp/vertirta_m3_forensics -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
   cmake --build /tmp/vertirta_m3_forensics && ctest --test-dir /tmp/vertirta_m3_forensics --output-on-failure

   cmake -S ros2_ws/src/vertirta_voice_node/test -B /tmp/vertirta_m3_voice_node -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-fsanitize=thread"
   cmake --build /tmp/vertirta_m3_voice_node && ctest --test-dir /tmp/vertirta_m3_voice_node --output-on-failure
   ```
2. **Zero-GPL Symbol & Header Gate (Mandatory across ALL `Plan 6B` Native Targets):**
   ```bash
   python3 scripts/ci/check_zero_gpl_symbols.py \
     --targets /tmp/vertirta_m3_voice/libvertirta_voice.a /tmp/vertirta_m3_forensics/libvertirta_forensics.a \
     --source-roots core/voice/ core/forensics/ ros2_ws/src/vertirta_voice_node/
   ```
3. **Python Test Suites (Host Windows — Always `--no-project`, Zero `.venv/`):**
   ```powershell
   uv run --no-project pytest tests/python/test_voice_bvi_and_license_gate.py tests/python/test_vertibench_v2_and_mondrian.py tests/forensics/test_e5x_xsd_and_grounding.py tests/integration/test_m3_voice_node_contracts.py docs/paper_corl/tests/test_paper_corl_contracts.py tests/python/test_copilot_l4_runner.py
   ```
4. **Paper 2 Zero-Drift Verifier (Phase 2 `--allow-pending-g4` / Phase 3 strict):**
   ```powershell
   uv run --no-project python docs/paper_corl/scripts/verify_corl_numbers.py --allow-pending-g4
   # Phase 3 post-G4 Thunder Run (strict: 0 pending slots, 0 errors):
   # uv run --no-project python docs/paper_corl/scripts/generate_corl_tables.py --ingest-v400 artifacts/v4.0.0
   # uv run --no-project python docs/paper_corl/scripts/verify_corl_numbers.py
   ```
5. **Global Repo Purity, Architecture Isolation & Bare-Metal Invariance:**
   ```powershell
   python scripts/ci/check_public_hygiene.py
   python scripts/ci/check_arch_isolation.py
   ```
   *(Bare-metal `arm-cortex-m7` `28,886 B Flash / 0 B RAM` invariance is re-confirmed in Wave 2 CI alongside `Plan 6A` & `Plan 6C`.)*
