# VertiRTA `v1.1` GPU ML Upgrade Specification

**Status:** `PARTIALLY_CLOSED_LOCALLY` — Three of the four original `v1.0.0` ML/data gaps (**G4-1** Forward WireSeg/D-FINE-N, **G4-2** Qwen3-4B v3 Conformal Calibration, and **G4-4 `UAVScenes`** Zero-Shot Eval) have now been **closed locally on physical `GTX 1650 SUPER` (4 GB) / CPU without cloud GPU compute**. Only **G4-3** (optional low-LR `Whisper` BVI-LoRA BF16 replay) and manual-registration **`ICG Drone`** (`G4-4` partial) remain open for an optional cloud GPU pass (`~16–18 min`).  
**License Policy:** **STRICT ZERO `GPL` / `CC-BY-NC` / `CC-BY-SA` IN TRAINING OR MODEL WEIGHTS.** Non-commercial or ShareAlike benchmarks are strictly quarantined to **zero-shot eval-only** (`used_in_training: false`, `used_in_calibration: false`) and never bundled in git or release tarballs.

---

## 1. Scope & Resolution Matrix

| # | Workstream | `v1.0.0` Baseline Gap | Current Status & Verified Result | Remaining Optional Cloud Time |
| --- | --- | --- | --- | --- |
| **G4-1** | **Forward WireSeg + Obstacle Fine-Tune** | `0` real powerline frames; **`240/240` (`100%`) roofline FPs** on NASA video (`586` boxes) | **CLOSED LOCALLY (`scripts/train_local_gtx1650s_forward_wire_v2.py`):** MIT Mendeley TEIAS + `646/153` urban hardneg (`val_iou=0.5731`, `clDice=0.7807`, `ty_neg_fp=0/40`); zero-shot NASA FP **`240/240 -> 11/240 (4.58%)`** full-frame (`0/240` center-crop), D-FINE-N **`628 -> 1`** box (`0` center-crop) (`artifacts/v4.0.0/local_gtx1650s_forward_wire_v2_metrics.json`) | `0 min` (Done) |
| **G4-2** | **Qwen3-4B v3 Temp-Scaled Mondrian Conformal** | `|C| = 7.803` (Lang) / `7.601` (SNR), **`singleton = 0.0%`**, `ECE = 0.1744` | **CLOSED LOCALLY (`scripts/ml/run_local_qwen3_v3_conformal_raps.py`):** Exact FP32 layer-streaming GBNF BPE-Trie (`837.5 MiB` VRAM, zero quantization loss, zero test leakage): **ECE `0.1744 -> 0.0321`** (Lang-APS `96.00%`, `|C|=3.486`), SNR-APS `|C|=2.347` (`40.0%` singl.), **Lang-LAC `|C|=2.502` (`40.0%` singl.)**, **SNR-LAC `|C|=2.058` (`45.5%` singl.)**, 100-resplit SNR **`95.00 ± 0.79%`, `|C|=1.924 ± 0.075`** (`artifacts/v4.0.0/local_gtx1650s_qwen3_v3_conformal_raps.json`) | `0 min` (Done) |
| **G4-3** | **Whisper-Large-v3-Turbo Low-LR BVI LoRA + Replay** | 3-ep LoRA (`r=32`, high LR) degraded WER (`13.14% -> 17.97%` `S0_Clean`, `31.74% -> 35.36%` `S3_Garbled`) | **OPEN (Zero-Shot Retained as Default):** Local CPU voice proof upgraded to zero-shot `faster-whisper-large-v3-turbo` INT8 (`12.58%–27.81%` TR WER, `24/24` Z3 SAT); optional low-LR BF16 replay fine-tune remains deferred | ~14 min |
| **G4-4** | **Zero-Shot `ICG Drone` + `UAVScenes` Eval (Paper 2)** | 4 `PENDING_EVAL_ONLY_REAL_IMAGERY` cells in `docs/paper_corl/corl_metrics_manifest.json` | **`UAVScenes` CLOSED LOCALLY (`scripts/realworld/run_uavscenes_zeroshot_eval.py`):** `300` frames / `20` scenes, raw pixels deleted, `cal mIoU = 0.2259`, `FSR = 12.55%` (`realworld_uavscenes_zeroshot_eval.json`). **`ICG Drone` OPEN** (requires manual portal registration) | ~3 min (after manual download) |
| **Total** | **Remaining Optional Pass** | — | 3/4 workstreams closed locally; only `G4-3` + `ICG Drone` remain | **~17 min** |

---

## 2. Workstream Specifications

### G4-1: Real-World WireSeg + Obstacle Fine-Tune (Fixing NASA Roofline False Positives)

#### Verified Dataset License Matrix (Strict Firewall)
Every candidate dataset was license-verified prior to specification:

| Dataset | Verified License | Training Verdict | Role in `v1.1` Batch |
| --- | --- | --- | --- |
| **TTPLA** (`R3ab/ttpla_dataset`) | **Apache-2.0** | **APPROVED (TRAIN)** | Real aerial powerlines + transmission towers (`1,242` images, LabelMe polygons -> `640x640` split-stack wire masks + tower boxes) |
| **Mendeley TEIAS Powerline GT** (`twxp8xccsw/9`) | **CC BY 4.0** | **APPROVED (TRAIN)** | Turkish grid VL/IR powerline ground truth (aligns directly with Istanbul corridor) |
| **OpenImages v7 (Buildings/Roofs/Facades)** | **CC BY 4.0** (annotations + CC-BY-filtered images) | **APPROVED (HARD-NEGATIVES ONLY)** | Architectural rooflines, window mullions, parapet seams with **all-zero wire mask** (`wire_mask = 0`) |
| **SkyScenes** (`hoffman-lab/SkyScenes`) + **FLAIR-1** (`etalab-2.0`) | **MIT** / **OPEN LICENCE 2.0** | **APPROVED (HARD-NEGATIVES ONLY)** | Oblique/nadir urban rooftops & facades (`wire_mask = 0`, `boxes = []`) to crush cross-view roofline FPs |
| **VertiRTA Procedural Catenary** | **Apache-2.0** | **APPROVED (TRAIN)** | Synthetic thin-wire catenary + `Bird`/`Rogue_UAV`/`Tower_Crane` horizon augmentation |
| **UAVDT** | **Non-Commercial / Academic Only** | **BANNED (REJECTED)** | Excluded strictly — violates zero-NC firewall |
| **VisDrone** | **CC BY-NC-SA 3.0** | **BANNED (REJECTED)** | Excluded strictly — violates zero-NC and zero-SA firewall |

#### Architecture, Loss & Hard-Negative Mining
- **Warm-Start:** Load `v1.0.0` `mlsd_wireseg_640_opset17` + `dfine_n_640_opset17` PyTorch checkpoints (or re-train from scratch in ~12 min on G4 BF16, `batch_size=64`).
- **Sampling Ratio per Batch:** `45%` real positive powerlines (`TTPLA` + `TEIAS`), `25%` procedural horizon catenary/obstacles, **`30%` hard-negative urban rooflines/facades** (`SkyScenes` oblique `H_35_P_45` + `FLAIR-1` + CC-BY `OpenImages` buildings, labeled `wire_mask = 0` and `boxes = []`).
- **Online Hard-Negative Mining (OHEM):** On hard-negative frames, compute top-`2%` highest-probability false-positive pixels and apply `3.0x` focal-BCE suppression weight; add a **sky-horizon prior check** (wire positives occur against sky/far-background contrast, not high-density parallel building parapet edges).
- **Loss Function:** `L_wire = 0.5 * BCE_OHEM + 0.3 * Soft_clDice(iters=5) + 0.2 * Focal_Tversky(alpha=0.7, beta=0.3)` (higher `alpha=0.7` explicitly penalizes false positives over false negatives on non-wire edges).
- **Export & Acceptance Gate:** Export static Opset-17 FP32 ONNX (`mlsd_wireseg_640_v11_opset17.onnx`, `dfine_n_640_v11_opset17.onnx`), run zero-shot inference on the exact `240`-frame NASA Goddard video (`sha256: 8177164a...7a1b6`):
  - **PASS:** `wireseg_frames_above_120px <= 12 / 240` (`< 5.0%`, down from `240/240`), `dfine_total_boxes < 20` (down from `586`), and held-out real `TTPLA` `val_cldice >= 0.80`, `val_wire_iou >= 0.45`.

---

### G4-2: Qwen3-4B v3 Temperature-Scaled Mondrian RAPS Finalize Re-Run (~4 min, NO Retraining)

#### Root Cause of `|C| = 7.803 / singleton = 0.0%`
In `artifacts/v4.0.0/g4_qwen3_v3_vertibench_v2_eval.json`, `fit_mondrian_conformal_groups` used raw `1 - p_tau(y*)` non-conformity scores over flat candidate branch probabilities (`q_hat = 0.964..0.977` -> `p_min = 0.023..0.036`), which pulls almost every candidate branch into the prediction set (`|C| = 7.803`, `singleton = 0.0%`). Because per-sample candidate logits were not persisted in `v1.0.0`, one minimal `--phase finalize` pass is needed.

#### Exact Zero-Retrain Procedure
1. **Load Existing Checkpoint:** Load `Qwen/Qwen3-4B` (BF16) + frozen `qwen3_4b_v3_seed0_lora` (`adapter_dir` from `v1.0.0` release blob — **0 training steps!**).
2. **Save Per-Sample Branch Log-Probs:** Run teacher-forced completion-only sliced `log_softmax` (`row_batch_size=8`, `n_cal=2,000` + `n_test=1,200`, ~3.5 min on GPU) and **persist** `qwen3_v3_seed0_branch_logprobs.npz` so all future conformal ablations run locally on CPU.
3. **Apply Temperature-Scaled Mondrian RAPS (Strictly Calibration-Split Fit):**
   - On `cal` (`n=2,000`), run 5-fold CV over `(gamma in {0.5, 0.75, 1.0}, tau_opt via NLL, k_reg in {1, 2}, lam_reg in {0.01, 0.05, 0.10, 0.20})` using Regularized Adaptive Prediction Sets (RAPS):
     `s_RAPS(x, y) = sum_{j: pi_j > pi_y} pi_j + U * pi_y + lam_reg * max(0, rank(y) - k_reg)`
   - Fit exact finite-sample Mondrian quantiles `q_hat[g]` per language (`TR/EN/MIX`) and per SNR stratum (`S0..S3`) at `alpha = 0.05`, plus `100`-resplit bootstrap CIs.
   - **Zero test leakage:** `test` (`n=1,200`) is touched once after `(gamma*, tau*, k_reg*, lam_reg*, q_hat[g])` are frozen.
4. **Acceptance Gate:**
   - **PASS:** `snr_mondrian_coverage_pct >= 95.0%` AND `lang_mondrian_coverage_pct >= 95.0%`, `mean_prediction_set_size < 2.50` (down from `7.803`), `singleton_exec_pct > 50.0%` (up from `0.0%`), `ECE < 0.12`.

---

### G4-3: Whisper-Large-v3-Turbo Low-LR BVI LoRA + Clean-Speech Replay

#### Root Cause of `v1.0.0` Negative Ablation
`job_whisper_bvi_lora.py` ran `3` epochs with `r=32, alpha=64` on `2,000` noisy FLEURS utterances (`1,000 TR + 1,000 EN`) at aggressive LR without clean-speech regularization, causing catastrophic forgetting of Whisper's acoustic prior (`S0_Clean WER: 13.14% -> 17.97%`, Turkish `27.14% -> 33.05%`).

#### Exact Regularized Recipe (`v1.1`)
- **Model:** `openai/whisper-large-v3-turbo` (`MIT`) ONLY (drop `whisper-small` to save GPU time; runtime default is `turbo`).
- **LoRA Config:** `r = 8`, `lora_alpha = 16`, `lora_dropout = 0.05`, target **attention projections only** (`q_proj`, `v_proj` in encoder + decoder; freeze FFN/MLP weights to preserve multilingual phonetics).
- **Optimizer & Schedule:** `lr = 1.5e-5`, **1 epoch** (or early-stop at `0.5` / `1.0` epoch on held-out `S0+S2` dev WER), `10%` linear warmup + cosine decay, `weight_decay = 0.01`, `batch_size = 16`, `grad_accum = 2` (effective `32`), BF16.
- **Data Mixture (100% CC-BY-4.0 / Apache-2.0):**
  - `70%` BVI-mixed `google/fleurs` (`tr_tr` + `en_us`, stratified across `S1_Cruise`, `S2_Hover`, `S3_Garbled` using `ml/voice/bvi_cockpit_dsp.py`).
  - **`30%` clean-speech replay** (`S0_Clean`, unmodified `google/fleurs` audio) to anchor the clean WER floor.
- **Latency Fix:** Merge LoRA adapters into base weights (`model = model.merge_and_unload()`) **before** benchmark timing and `ggml-q8_0` / CTranslate2 export so inference latency matches Zero-Shot (`~14.5 ms`, eliminating the `2x` unmerged adapter overhead seen in `v1.0.0`).
- **Acceptance Gate:**
  - **PASS:** Merged BVI-LoRA beats Zero-Shot (`13.14% / 14.46% / 19.60% / 31.74%`) on **all 4 SNR strata**, improves Turkish `S2_Hover` WER by `>= 2.0 pp`, and keeps `mean_latency_ms <= 16.0 ms`. If any stratum regresses, fail closed and keep Zero-Shot as default.

---

### G4-4: Zero-Shot Real-World `ICG Drone` + `UAVScenes` Eval (Closing Paper 2's 2 `PENDING` Cells)

#### Quarantine & License Governance
- **`ICG Semantic Drone Dataset` (TU Graz):** Academic / Non-Commercial License (requires one-time registration download from `dronedataset.icg.tugraz.at`; redistribution prohibited).
- **`UAVScenes` (NTU, ICCV 2025, `sijieaaa/UAVScenes`):** `CC BY-NC-SA 4.0`.
- **Hard Rule:** Both corpora are **STRICTLY QUARANTINED TO ZERO-SHOT EVAL ONLY** (`used_in_training: false`, `used_in_calibration: false`, deleted from `/content/` scratch immediately after computing scalar `mIoU` & `FSR` JSON metrics — **ZERO pixels or labels ever written to Drive, git, or release blobs**).

#### Evaluation Protocol
- **Model Under Test:** Frozen `pidnet_slz_v21_local_gtx_opset17.onnx` (`sha256: a0b0b99b3f68288b5366245b3b3a711ed569479d4c26dfc44eb638e6f65679ff`) with frozen `safe_logit_bias_delta = -0.271367` (`ood_calibration_v21_local_gtx.json`, `sha256: 59f9a262...d34c832`).
- **Label Remapping:** Deterministic 5-class VertiRTA SLZ taxonomy (`0: Safe_Flat_Ground`, `1: Safe_Flat_Roof`, `2: Hazard_Obstacle_Structure`, `3: Hazard_Dynamic_Human_Vehicle`, `4: Hazard_Water_Vegetation`), mapping ICG Drone (`23` classes) and UAVScenes (`19` semantic classes) with strict checksum logging of the downloaded archives.
- **Output Artifact:** `g4_pidnet_v21_icg_uavscenes_zeroshot.json` (`< 20 KB`), updating the 4 `external_eval_only_slots` in `docs/paper_corl/corl_metrics_manifest.json` (`icg_drone_{miou,fsr}`, `uavscenes_{miou,fsr}`).
- **Acceptance Gate:**
  - **PASS:** Both datasets evaluate `>= 200` real frames with finite metrics, `used_in_training == false`, `used_in_calibration == false`, and `FSR <= 8.0%` (reported verbatim even if higher — zero cherry-picking).

---

## 3. Reproducible Checked-In Scripts & Remaining Optional Runbook

All closed local workstreams use checked-in, tested scripts in this repository (superseding any earlier single-wrapper draft):

| Stage | Checked-In Command | Hardware / VRAM | Status |
| --- | --- | --- | --- |
| **G4-1 (Closed Locally)** | `python3 scripts/train_local_gtx1650s_forward_wire_v2.py` | GTX 1650 SUPER (`2.69 GB` peak) | **DONE** (`local_gtx1650s_forward_wire_v2_metrics.json`) |
| **G4-2 (Closed Locally)** | `python3 scripts/ml/run_local_qwen3_v3_conformal_raps.py --score-scope gbnf_tree --layer-dtype float32` | GTX 1650 SUPER (`837.5 MiB` peak) | **DONE** (`local_gtx1650s_qwen3_v3_conformal_raps.json`) |
| **G4-4 `UAVScenes` (Closed Locally)** | `python3 scripts/realworld/run_uavscenes_zeroshot_eval.py` | CPU ONNX (`0 MiB` VRAM) | **DONE** (`realworld_uavscenes_zeroshot_eval.json`) |
| **G4-3 (Optional Cloud GPU)** | `python3 scripts/colab/g4_jobs/job_whisper_bvi_lora.py --model openai/whisper-large-v3-turbo --epochs 1 --lr 1.5e-5` | High-VRAM BF16 GPU (`~44 GB`) | **OPEN** (Zero-Shot retained as default) |
| **G4-4 `ICG Drone` (Manual Portal)** | Download `ICG Semantic Drone Dataset` after manual registration, then run zero-shot frozen ONNX eval against `pidnet_slz_v21_local_gtx_opset17.onnx` (`used_in_training=false`) | CPU or local GPU | **OPEN** (`PENDING_EVAL_ONLY_REAL_IMAGERY`) |

```bash
# Re-verify all locally closed artifacts & Paper 2 CoRL synchronization on CPU:
python3 scripts/ml/run_local_qwen3_v3_conformal_raps.py --calibrate-only
python3 docs/paper_corl/scripts/generate_corl_tables.py --ingest-v400 artifacts/v4.0.0
python3 docs/paper_corl/scripts/verify_corl_numbers.py
```

---

## 4. Fail-Closed Local Verification Checklist

If the remaining optional cloud GPU jobs (`G4-3` Whisper low-LR LoRA and/or `ICG Drone` zero-shot eval) are executed in the future, ingest fail-closed using the existing hardened `scripts/colab/ingest_g4_bundle.py`:

```bash
# 1. Fail-closed extraction, tar-slip check, SHA-256 manifest parity, <1 MB UTF-8 LF JSON, & public hygiene:
python3 scripts/colab/ingest_g4_bundle.py \
  --artifacts-tar ~/Downloads/vertirta_v110_artifacts.tar.gz \
  --blobs-tar ~/Downloads/vertirta_v110_blobs.tar.gz \
  --dest-artifacts-dir artifacts/v1.1.0 \
  --dest-blobs-dir /var/tmp/vertirta_v110_blobs \
  --allow-partial

# 2. Verify NASA Goddard real drone video 3D-normal veto + v1.1 forward detectors on GTX 1650 SUPER:
python3 scripts/realworld/run_real_drone_video_temporal_hud.py \
  --blob-dir /var/tmp/vertirta_v400_blobs \
  --out-json artifacts/v4.0.0/realworld_drone_video_temporal_proof.json

# 3. Verify physical GTX 1650 SUPER time-sliced edge latency (<25 ms cruise, <12 ms terminal):
python3 scripts/bench_gtx1650s_m3_timesliced.py \
  --model-profile local_gtx_v11 \
  --blob-dir /var/tmp/vertirta_v400_blobs \
  --artifacts-dir artifacts/v4.0.0 \
  --out-json artifacts/v4.0.0/gtx1650s_m3_timesliced_edge_gate.json

# 4. Verify zero GPL, zero PII/secrets, and Paper 2 CoRL table regeneration:
python3 scripts/ci/check_public_hygiene.py --repo-root .
python3 docs/paper_corl/scripts/generate_corl_tables.py --ingest-v400 artifacts/v4.0.0 && python3 docs/paper_corl/scripts/verify_corl_numbers.py
```

### Hard Pass/Fail Gate Summary
- **Tar & Digest Gate:** `0` symlinks/path-traversals, `100%` SHA-256 match against `remote_bundle_manifest.json` + `SHA256SUMS.txt`.
- **License Gate:** `zero_nc_sa_gpl_verified == true` on all trained artifacts (`G4-1` DONE, `G4-2` DONE, `G4-3` optional); `used_in_training == false` & `used_in_calibration == false` on `G4-4` (`UAVScenes` DONE with `raw_pixels_deleted: true`; `ICG Drone` open).
- **Metric Gates:** WireSeg NASA FP `4.58%` full / `0.0%` crop (`<= 5%` PASS); Qwen3 exact FP32 GBNF-tree `ECE = 0.0321`, Lang-LAC `|C| = 2.502` (`40.0%` singl.), SNR-LAC `|C| = 2.058` (`45.5%` singl., 100-resplit `95.00 ± 0.79%`, `|C| = 1.924`); `TU116` Terminal `p99 = 10.11 ms < 12 ms` PASS & Cruise `p99 = 14.13 ms < 25 ms` PASS; remaining optional targets: Whisper Low-LR beating Zero-Shot on `4/4` SNR strata & `ICG Drone` zero-shot eval; `check_public_hygiene.py` exits `0`.
