# VertiRTA Optional `v1.1` Colab G4 Batch Specification (`0.00 CU` Blueprint)

**Status:** `BLUEPRINT_ONLY_ZERO_CU_SPENT` — Prepared locally at `0.00 CU` so that if/when Colab G4 (`NVIDIA RTX PRO 6000 Blackwell`, `96 GB` VRAM) capacity opens later, a single, zero-waste, license-clean batch closes all four genuine `v1.0.0` ML/data gaps in **< 45 minutes (~8–12 CU)**.  
**License Policy:** **STRICT ZERO `GPL` / `CC-BY-NC` / `CC-BY-SA` IN TRAINING OR MODEL WEIGHTS.** Non-commercial or ShareAlike benchmarks are strictly quarantined to **zero-shot eval-only** (`used_in_training: false`, `used_in_calibration: false`) and never bundled in git or release tarballs.

---

## 1. Scope: Only the 4 Genuine GPU-Justified Gaps

| # | Workstream | `v1.0.0` Baseline Gap | `v1.1` Target Gate | Est. G4 Time |
| --- | --- | --- | --- | --- |
| **G4-1** | **Forward WireSeg + Obstacle Fine-Tune** | `0` real `TTPLA` frames ingested; **`240/240` (`100%`) roofline FPs** on NASA Goddard video (`586` spurious boxes) | NASA video WireSeg FP **`< 5%` (`<= 12/240` frames)**; held-out real `TTPLA` `clDice >= 0.80`; `D-FINE-N` spurious boxes `< 20` | ~12 min |
| **G4-2** | **Qwen3-4B v3 Temp-Scaled Mondrian RAPS Finalize** | `|C| = 7.803` (Lang) / `7.601` (SNR), **`singleton_exec_pct = 0.0%`** (100% clarify) | **`|C| < 2.50`**, **`singleton_exec_pct > 50.0%`**, preserve **Mondrian coverage `>= 95.0%`** across all Lang & SNR strata | ~4 min |
| **G4-3** | **Whisper-Large-v3-Turbo Low-LR BVI LoRA + Replay** | 3-ep LoRA (`r=32`, high LR) degraded WER (`13.14% -> 17.97%` `S0_Clean`, `31.74% -> 35.36%` `S3_Garbled`) | Beat Zero-Shot on **all 4 SNR strata** (`S0_Clean < 13.14%`, `S2_Hover < 18.0%`, `S3_Garbled < 29.0%`) | ~14 min |
| **G4-4** | **Zero-Shot `ICG Drone` + `UAVScenes` Eval (Paper 2)** | 4 `PENDING_EVAL_ONLY_REAL_IMAGERY` cells in `docs/paper_corl/corl_metrics_manifest.json` | Populate all 4 cells (`mIoU` + `FSR`) using frozen `pidnet_slz_v21_local_gtx_opset17.onnx` (`0` retraining) | ~5 min |
| **Total** | **Single Consolidated Wave (`v1.1` G4 Batch)** | — | All 4 gates PASS + fail-closed `ingest_g4_bundle.py` | **~35–42 min (~8–12 CU)** |

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
2. **Save Per-Sample Branch Log-Probs:** Run teacher-forced completion-only sliced `log_softmax` (`row_batch_size=8`, `n_cal=2,000` + `n_test=1,200`, ~3.5 min on G4) and **persist** `qwen3_v3_seed0_branch_logprobs.npz` to blobs so all future conformal ablations run locally at **`0.00 CU`**.
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

## 3. Single-Command G4 Runbook & Compute Budget

| Stage | Command (Inside Colab G4 Runtime) | VRAM Cap | Wall-Clock | Est. CU |
| --- | --- | --- | --- | --- |
| **Preflight** | `python3 scripts/colab/run_g4_v11_batch.py --dry-run --check-licenses` | `0 GB` | `10 s` | `0.00` |
| **Wave 1A (Parallel)** | `G4-1` (WireSeg+DFINE, `0.35` VRAM) + `G4-2` (Qwen3 RAPS finalize, `0.30` VRAM) + `G4-4` (ICG+UAVScenes ONNX eval, `0.15` VRAM) | `76 GB` (`80%`) | `~14 min` | `~4.0` |
| **Wave 1B (Serial)** | `G4-3` (Whisper-v3-turbo low-LR LoRA + clean replay + merge + 4-stratum eval, `0.45` VRAM) | `44 GB` (`45%`) | `~14 min` | `~4.0` |
| **Pack & Hash** | Tar verified `< 1 MB` JSONs -> `vertirta_v110_artifacts.tar.gz` + ONNX/NPZ/LoRA -> `vertirta_v110_blobs.tar.gz` + `remote_bundle_manifest.json` | `0 GB` | `~2 min` | `~0.5` |
| **Total** | **1 Single Notebook Invocation** | `<= 80%` | **`~30–40 min`** | **`~8.5–12 CU`** |

```bash
# Single-command Colab G4 invocation (execute ONLY when user authorizes v1.1 G4 run):
python3 scripts/colab/run_g4_v11_batch.py \
  --work-dir /content/vertirta_v11_run \
  --drive-out-dir /content/drive/MyDrive/VertiRTA_v110 \
  --v10-blobs-dir /content/drive/MyDrive/VertiRTA_v400/blobs \
  --icg-archive /content/drive/MyDrive/VertiRTA_EvalOnly/icg_semantic_drone.zip \
  --jobs g4_1_wireseg_ohem,g4_2_qwen3_raps,g4_3_whisper_lowlr,g4_4_icg_uavscenes_eval
```

---

## 4. Fail-Closed Local Ingestion & Verification Checklist (`0.00 CU`)

After downloading `vertirta_v110_artifacts.tar.gz` and `vertirta_v110_blobs.tar.gz` locally, ingest fail-closed using the existing hardened `scripts/colab/ingest_g4_bundle.py`:

```bash
# 1. Fail-closed extraction, tar-slip check, SHA-256 manifest parity, <1 MB UTF-8 LF JSON, & public hygiene:
python3 scripts/colab/ingest_g4_bundle.py \
  --artifacts-tar ~/Downloads/vertirta_v110_artifacts.tar.gz \
  --blobs-tar ~/Downloads/vertirta_v110_blobs.tar.gz \
  --dest-artifacts-dir artifacts/v1.1.0 \
  --dest-blobs-dir /var/tmp/vertirta_v110_blobs \
  --allow-partial

# 2. Verify NASA Goddard real drone video roofline FP drop locally on physical GTX 1650 SUPER:
python3 scripts/realworld/run_real_drone_video_temporal_hud.py \
  --blob-dir /var/tmp/vertirta_v110_blobs \
  --out-json artifacts/v1.1.0/realworld_drone_video_temporal_proof_v11.json

# 3. Verify physical GTX 1650 SUPER time-sliced edge latency (<25 ms cruise, <12 ms terminal):
python3 scripts/bench_gtx1650s_m3_timesliced.py \
  --blob-dir /var/tmp/vertirta_v110_blobs \
  --artifacts-dir artifacts/v1.1.0 \
  --out-json artifacts/v1.1.0/gtx1650s_m3_timesliced_edge_gate_v11.json

# 4. Verify zero GPL, zero PII/secrets, and Paper 2 CoRL table regeneration:
python3 scripts/ci/check_public_hygiene.py --repo-root .
python3 docs/paper_corl/scripts/generate_corl_tables.py && python3 docs/paper_corl/scripts/verify_corl_numbers.py
```

### Hard Pass/Fail Gate Summary
- **Tar & Digest Gate:** `0` symlinks/path-traversals, `100%` SHA-256 match against `remote_bundle_manifest.json` + `SHA256SUMS.txt`.
- **License Gate:** `zero_nc_sa_gpl_verified == true` on all trained artifacts (`G4-1`, `G4-2`, `G4-3`); `used_in_training == false` & `used_in_calibration == false` on `G4-4` (`ICG Drone`, `UAVScenes`).
- **Metric Gates:** WireSeg NASA FP `< 5%` (`<= 12/240`); Qwen3 RAPS `|C| < 2.5`, `singleton > 50%`, Mondrian coverage `>= 95%`; Whisper Low-LR beats Zero-Shot on `4/4` SNR strata; Paper 2 `ICG` + `UAVScenes` `4/4` cells populated; `check_public_hygiene.py` exits `0`.
