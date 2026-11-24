# VertiRTA v4.0.0 Single-Shot 80–88 GB Colab G4 Batch Thunder Run Orchestrator & Physical TU116 Time-Sliced Edge Gate (Plan 6C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the **Approach C-G4** single-shot `80–88 GB` `NVIDIA RTX PRO 6000 Blackwell (G4, 96 GB GDDR7)` batch training, PDE solving, calibration, and physical `GTX 1650 SUPER (TU116, 4 GB)` time-sliced edge verification pipeline for **`M3 (v4.0.0)`** with **zero wasted compute units** (`0 CU` during Phase 1 & Phase 2; one single `G4` opened once and closed once in Phase 3), **zero dummy VRAM padding**, **zero license/GPL/NC/ShareAlike contamination**, **zero edits** to frozen `core/flight_core/` (`28,886 B Flash / 0 B RAM`) or `core/rta_companion/` (`11,313,864 B VRTALUT1`), and **zero file overlap** with Plan 6A (`core/forward_hazard/`, `core/gnss_denied/`, `sim/gazebo_plugins/mmwave_77ghz/`, `ros2_ws/src/vertirta_{forward_hazard,gnss_denied}_node/`, `apps/vertigcs/`, `.github/workflows/m3_perception_voice_forensics.yml`) or Plan 6B (`core/voice/`, `core/forensics/`, `benchmarks/vertibench_v2/`, `docs/paper_corl/`, `ros2_ws/src/vertirta_voice_node/`).

**Architecture:** Plan 6C splits into four clean layers:
1. **Host Lifecycle Orchestrator (`scripts/colab/run_g4_m3_thunder_run.py`):** Enforces a mandatory `0 CU` local pre-flight smoke gate (`pytest tests/colab/` + `--dry-run` on all 5 jobs) strictly before touching `colab_cli`, snapshots `pre_existing_endpoints` via `colab_cli.client.Client.list_assignments()` so external agents' sessions are NEVER disturbed, opens ONE `G4` (`colab new --gpu G4 -s vertirta-m3-g4-thunder`), uploads a minimal source tarball, supervises remote Wave 1 -> hard GPU-drain barrier -> Wave 2 execution with WebSocket-drop-resilient heartbeat polling, downloads large binaries strictly to `/var/tmp/vertirta_v400_blobs/` (never into git) and compact manifests (`< 100 KB`) to `artifacts/v4.0.0/` with SHA-256 verification, and guarantees teardown in `try / finally` via BOTH `colab stop` and explicit `colab_cli.client.Client.unassign(owned_endpoint)` followed by `list_assignments()` confirmation (`owned_endpoint_released: true`).
2. **On-VM Two-Wave GPU Multiplexer & `1 Hz` Telemetry Logger (`scripts/colab/remote_g4_driver.py`):** Launches **Wave 1** (`3` parallel PyTorch CUDA processes capped at `0.46 + 0.22 + 0.20 = 0.88` allocator fraction -> **`82–86 GiB` genuine peak VRAM** out of `95.59 GiB`, leaving `~9.3 GiB` headroom), enforces a hard Wave-1 -> Wave-2 GPU-drain barrier (`0` compute PIDs and `memory.used < 1,500 MiB` via `nvidia-smi`), then launches **Wave 2** (`1` JAX/XLA process pinned to `XLA_PYTHON_CLIENT_MEM_FRACTION=0.58` + `1` PyTorch process capped at `0.30` -> `0.58 + 0.30 = 0.88`, **`78–86 GiB` genuine peak VRAM**, zero XLA/PyTorch allocator collision). Samples `nvidia-smi` at `1 Hz` throughout to emit verifiable telemetry (`artifacts/v4.0.0/g4_thunder_telemetry.json`).
3. **Five Modular Self-Testable Jobs (`scripts/colab/g4_jobs/`):** Every job inherits a uniform CLI contract (`--dry-run`, `--smoke-cpu`, `--vram-fraction`, `--out-json`, `--blob-dir`, `--seed`), runs an upfront zero-GPL/zero-NC/zero-ShareAlike license gate BEFORE any download, and passes 100% locally on CPU (`0 CU`) in `tests/colab/` before `G4` ever boots:
   - `job_qwen3_v3_vertibench_v2.py` (Wave 1, fraction `0.46` / `~42–46 GB`)
   - `job_whisper_bvi_lora.py` (Wave 1, fraction `0.22` / `~20–22 GB`)
   - `job_forward_wire.py` (Wave 1, fraction `0.20` / `~18–20 GB`)
   - `job_jax_4d_hires_and_lc.py` (Wave 2, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.58` / `~50–56 GB`)
   - `job_pidnet_v21_xfeat.py` (Wave 2, fraction `0.30` / `~24–28 GB`)
4. **Physical Local `GTX 1650 SUPER` (`TU116`, `4 GB`) Time-Sliced Edge Gate (`scripts/bench_gtx1650s_m3_timesliced.py`):** Verifies SHA-256 parity of downloaded Opset-17 ONNX binaries (`pidnet_slz_v21_opset17.onnx`, `mlsd_wireseg_640_opset17.onnx`, `dfine_n_640_opset17.onnx`, `xfeat_512_opset17.onnx`), executes single-process flight-phase time-slicing (Cruise: `M-LSD-WireSeg` + `D-FINE-N` + `1 Hz XFeat`; Terminal: `PIDNet-S v2.1`), runs the mandatory empirical **FP32-vs-FP16** TU116 latency/VRAM comparison (`01-DR08-13` Override `#6` — NEVER assuming `2×` FP16 speedup), and enforces `< 300 MiB` time-sliced VRAM + `< 12 ms` p99 SLZ gate (`artifacts/v4.0.0/local_gtx1650s_m3_timesliced_proof.json`).

**Tech Stack:** Python `3.11+` (`uv run --no-project --python 3.11`, zero `.venv` or `uv.lock` created), PyTorch `2.x` (`BF16` AMP, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `torch.cuda.set_per_process_memory_fraction`), `transformers`, `trl` (`dr_grpo`), `peft`, `z3-solver` (`spawn` pool), `jax` + `StanfordASL/hj_reachability v0.7.0` (`FP32`, WENO5 + TVD-RK3), `onnx` (Opset-17) + `onnxruntime-gpu` (`CUDAExecutionProvider` + `IoBinding` + CUDA Graphs), `colab_cli` (`colab_cli.client.Client`), `pytest`, `numpy`, `scipy`, `pyyaml`.

**Spec & Canonical Overrides:** `docs/superpowers/specs/2026-11-22-vertirta-v3-v4-expansion-design.md` (§1.2, §2.2, §5, §6.1, §7), `docs/research/01-DR08-13-FORENSIC-ERRATA-AND-LOCKS.md` (Overrides `#1`, `#3`, `#5`, `#6`, `#7`, §3.1–§3.2), `docs/research/DR-10-forward-radar-vision-gnss-denied-vio.md`, `docs/research/DR-12-voice-atc-fleet-z3-eccairs2-forensics.md`, `config/vehicle_x8_240.yaml`, `core/rta_companion/include/vertirta/rta/mmap_cbvf_lut.hpp`, `core/airframes/lift_cruise/include/vertirta/airframes/lift_cruise_dynamics.hpp`, and `v3.1.0`/`v3.5.1` baselines (`artifacts/v3.1.0/jax_4d_cbvf_l4_manifest.json`, `artifacts/v3.5.1/{l4_pidnet_v2_distill_metrics,l4_qwen3_4b_v2_calibrated_eval,local_gtx1650s_pidnet_v2_proof}.json`).

---

## Global Constraints, Hard Locks & Zero-Overlap Ownership Boundary

### 1. Absolute Zero-Waste `G4` Lock & Pre-Existing Session Safety
- **Phase 1 & Phase 2 (`0.00 CU`):** **NOBODY** invokes `colab new` during Phase 1 (planning) or Phase 2 (implementation & local testing). Every job in `scripts/colab/g4_jobs/`, `remote_g4_driver.py`, `run_g4_m3_thunder_run.py`, and `scripts/bench_gtx1650s_m3_timesliced.py` MUST pass `pytest tests/colab/ -v` (`--dry-run` AND `--smoke-cpu`) locally BEFORE Phase 3 begins.
- **Single-Open / Single-Close Invariant (Phase 3):** Open `G4` (`~10.05 CU/hr`, `96 GB GDDR7`) **ONCE**, run Wave 1 + Wave 2 back-to-back, download verified artifacts, and close **ONCE** inside `try / finally`.
- **Pre-Existing Endpoint Preservation:** Prior swarm runs (`#288`) confirmed external sessions (e.g., an independent `G4`) may coexist on the user's Colab account. The orchestrator MUST record `pre_existing_endpoints = {a.endpoint for a in client.list_assignments()}` BEFORE `colab new` and in `finally:` unassign **STRICTLY** `owned_endpoint` (never external endpoints!). Verify `owned_endpoint not in [a.endpoint for a in client.list_assignments()]` and log exact `remaining_active_assignments` (which equals `0` when no external sessions exist).

### 2. Real-Work VRAM Partitioning & Zero Dummy-Padding Lock
- **Hard Ban on Synthetic Memory Padding:** NEVER allocate dummy tensors (`torch.empty(...)`, `jnp.zeros(...)`, `cudaMalloc` fillers) to inflate `nvidia-smi` memory readouts. `tests/colab/test_g4_thunder_contracts.py` statically audits ASTs across `scripts/colab/g4_jobs/*.py` to forbid filler allocations.
- **Exact `G4` (`95.59 GiB` / `97,887 MiB` GDDR7) Allocator Budget:**

| Wave | Process Script | Runtime / Allocator Cap | Cap (`GiB`) | Target Real `nvidia-smi` Peak | Workload Sizing Driving Real Peak |
| :---: | --- | --- | ---: | ---: | --- |
| **Wave 1** | `job_qwen3_v3_vertibench_v2.py` | PyTorch `fraction=0.46` + `expandable_segments:True` | `43.97` | `38–43 GB` | `Qwen3-4B` BF16 policy + BF16 ref + LoRA `r=64` + **`G=16` Z3-GRPO (`prompts_per_microbatch=1..2`, `per_device_train_batch_size=16..32`, `grad_accum=4..8`, `gradient_checkpointing=True`, completion-only sliced `log_softmax`)** |
| **Wave 1** | `job_whisper_bvi_lora.py` | PyTorch `fraction=0.22` + `expandable_segments:True` | `21.03` | `20–22 GB` | `openai/whisper-large-v3-turbo` (`809M`, pruned from `1.55B` `large-v3`) + `whisper-small` (`244M`) BF16 LoRA (`r=32`, batch `64–96` mel chunks) |
| **Wave 1** | `job_forward_wire.py` | PyTorch `fraction=0.20` + `expandable_segments:True` | `19.12` | `18–20 GB` | `M-LSD-WireSeg` (`clDice`, `640×640`, batch `64` BF16) + `D-FINE-N` (`640×640`, batch `64` BF16) |
| **Wave 1 Sum** | *3 parallel CUDA processes* | **`0.46 + 0.22 + 0.20 = 0.88`** + `~1.8 GiB` CUDA ctx | **`85.92`** | **`78–86 GB`** | **`~9.7 GiB` OOM safety margin** (genuine peak logged by `1 Hz` `nvidia-smi`, zero `[128, L, 151936]` FP32 logit OOM) |
| *Barrier* | `remote_g4_driver.py` | Wait all Wave-1 PIDs exit + poll `nvidia-smi` | `0.00` | `< 1.5 GB` | Mandatory hard drain (`0` active compute PIDs) before launching XLA! |
| **Wave 2** | `job_jax_4d_hires_and_lc.py` | **`XLA_PYTHON_CLIENT_PREALLOCATE=true`**, **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.58`** | `55.44` | `38–50 GB` (`55.44 GiB` XLA BFC pool) | **`vmap_batch_size=2` safe default** (`2 × 2` regimes = `102.8M` simultaneous `101×71×101×71` FP32 WENO5+TVD-RK3 nodes for X8-240, then Lift+Cruise `8+1`; `vmap=4` strictly opt-in behind `--allow-vmap4` + `jax.checkpoint` remat + preflight memory check) |
| **Wave 2** | `job_pidnet_v21_xfeat.py` | PyTorch `fraction=0.30` + `expandable_segments:True` | `28.68` | `24–28 GB` | `PIDNet-S v2.1` (`batch_size=32`, frozen `DINOv2-Large` + `DAv2-Small` BF16 teachers) + `XFeat+LighterGlue` + `MegaLoc` Kartal `0.5 m` reloc eval |
| **Wave 2 Sum** | *1 JAX/XLA + 1 PyTorch* | **`0.58 + 0.30 = 0.88`** + `~1.5 GiB` CUDA ctx | **`85.62`** | **`78–86 GB`** | **`~10.0 GiB` OOM safety margin**; XLA BFC (`58%`) and PyTorch (`30%`) strictly disjoint! |

### 3. Frozen Core, Binary Format, License Firewall & Canonical Overrides
- **Frozen Runtime Guarantees (`01-DR08-13` Override `#5`, Spec `I1`/`I2`):**
  - **ZERO edits** to `core/flight_core/**`, `core/rta_companion/**`, `config/**`, `CMakeLists.txt`, `CMakePresets.json`, or historical `scripts/colab/run_l4_*.py`.
  - `libvertirta_flight_core.a` remains **`28,886 B` Flash / `0 B` static RAM / `0` heap**.
  - **Critical `.vrta_lut` Binary Compatibility Lock:** Frozen `MmapCbvf4DLut` (`core/rta_companion/include/vertirta/rta/mmap_cbvf_lut.hpp`) expects exact `VRTALUT1` (`11,313,864 B`, `dims = {41, 29, 41, 29}`). Because `(101-1)/(41-1) = 100/40 = 2.5` and `(71-1)/(29-1) = 70/28 = 2.5` are half-integer (not stride-2), `job_jax_4d_hires_and_lc.py` exports:
    1. Full lossless `101×71×101×71` high-resolution archives tagged with magic **`"VRTALUT2"`** (`411,389,928 B`, stored strictly in `/var/tmp/vertirta_v400_blobs/`) so a `101×71` file can NEVER be mistakenly loaded by frozen `VRTALUT1` `MmapCbvf4DLut`; **AND**
    2. Exact `41×29×41×29` **`11,313,864 B` `VRTALUT1`** drop-in runtime tables projected via quadrilinear evaluation at the `41×29×41×29` lattice points and verified by a **chunked exact `51,423,241`-node optimistic-overestimation audit** (`eps_grid_f = max_{|V_hires| <= 10 m} max(0, V_hat_41x29(x_hires) - clip(V_hires(x_hires))) + 0.5*q`), mathematically proving `V_hat_41x29(x) - eps_grid_f <= V_hires(x)` everywhere inside `|V_hires| <= 10 m`!
- **Canonical Physics & Provenance Overrides (`01-DR08-13` Overrides `#1`, `#6`, `#7`):**
  - Coaxial BVI cockpit noise synthesis in `job_whisper_bvi_lora.py` uses canonical `config/vehicle_x8_240.yaml` **`omega_hover = 1,450 RPM (72.5 Hz BPF)` up to `omega_max = 2,292 RPM (114.6 Hz ~ 115 Hz BPF)`**, overriding `DR-12`'s drifted `1,850 RPM` ceiling.
  - `Lift+Cruise 8+1` 4D Transition HJI-CBVF PDE uses canonical `core/airframes/lift_cruise/include/vertirta/airframes/lift_cruise_dynamics.hpp` constants (`m = 240.0 kg`, `S = 5.20 m^2`, `V_stall_1g = 21.83 m/s`, `V_blend = [14.0, 28.0] m/s`, `V_cruise = 38.0 m/s`, `V_NE = 46.3 m/s`, `T_max_lift = 8 × 544.3 N`, hover-OEI `<= 79.2% T_max`, `T_max_pusher = 950.0 N`, `L/D_max = 13.46`).
  - `job_qwen3_v3_vertibench_v2.py` stamps the honest provenance tag **`PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2`** (never claiming human native labels), enforces `max_char_5gram_jaccard < 0.15` + `0` seed overlap, locks `alpha = 0.05`, MAYDAY/PAN-PAN `w_m >= 0.15`, and `< 5.0%` SoC `LIVENESS_EMERGENCY_COMMIT`.
  - `scripts/bench_gtx1650s_m3_timesliced.py` enforces **empirical FP32-vs-FP16 comparison** on physical `GTX 1650 SUPER` (`TU116`), acknowledging `v3.5.1` measured FP16 `0.4×` (`2.5×` slower) than FP32+CUDA-Graph (`8.887 ms` p50 / `11.695 ms` p99 / `155.0 MiB` VRAM).
- **Zero-Tolerance License Firewall Across All 5 Jobs (`scripts/colab/g4_jobs/common.py`):**
  - **Approved ONLY:** `Qwen/Qwen3-4B` (`Apache-2.0`), `openai/whisper-large-v3-turbo` & `whisper-small` (`MIT`), `Mozilla Common Voice 27.0` (`CC0`), `ISSAI TSC` (`MIT/CC BY 4.0`), `FLEURS`/`MediaSpeech` (`CC BY 4.0`), `MUSAN` (`CC BY 4.0`), `OpenSLR-28 RIRS` (`Apache-2.0`), filtered `FSD50K` (zero NC/Sampling+), `TTPLA` (`Apache-2.0`, **forward `45°` wire head ONLY**), `Mendeley TEİAŞ` (`CC BY 4.0`), `Purdue UAV` (`BSD-3`), `FLAIR-1` (`etalab-2.0`), `OpenEarthMap` live permissive whitelist (`CC BY 4.0` + `DL-DE-BY-2.0`), `FloodNet` (`CDLA-P`), `SkyScenes` (`MIT`, `<= 30 MB` PNG/JPEG filter), `facebook/dinov2-large` & `DAv2-Small` (`Apache-2.0`), `XFeat + LighterGlue` (`Apache-2.0`), `MegaLoc` (`MIT`).
  - **Quarantined Zero-Shot Eval-Only (STRICTLY Banned from `train_sources` & Calibration, Permitted ONLY in `eval_only_sources` per `01-DR08-13` Override `#7` & `Spec §6.1`):** `AeroScapes` (`CC BY-SA 4.0`), `ICG`, and `UAVScenes`.
  - **Globally HARD-BANNED Everywhere (`HARD_BANNED_IDENTIFIERS` — Aborts Even in `eval_only_sources`):** `UAVid`, `CohereForAI/aya-expanse-8b`, `UWB-ATCC`, `ATCO2`, `ESC-50`, `DEMAND`, `ATCOSIM` (quarantined pending ISO license), `piper1-gpl`, all prebuilt `tr_TR` Piper voices, `XTTS-v2`, `mms-tts-*`, `OmniVoice`, `BlenderProc`/`bpy`, `InsPLAD`, `DDOS`, `Drone-vs-Bird`, `STN PLAD`, `PAMPC`, `Deep Hough Transform`, `Ultralytics`, `SegFormer`, `DINOv3`, `DINOv2-SALAD`, `Sample4Geo`, `SuperPoint`/`SuperGlue`, `EfficientLoFTR`, `PLDU/PLDM`, `SynthWires`, `WireSegHR`, `SPL`, `SOD4SB`, `Seraphim`.

### 4. Strict File Ownership Matrix (Zero Overlap with Plan 6A & Plan 6B)

| Plan / Owner | Exclusive Owned Paths (Create / Modify) | Cross-Plan Interface Contract |
| --- | --- | --- |
| **Plan 6C (`@flight-stack-auditor`)** | `scripts/colab/run_g4_m3_thunder_run.py`, `scripts/colab/remote_g4_driver.py`, `scripts/colab/g4_jobs/{__init__,common,job_qwen3_v3_vertibench_v2,job_whisper_bvi_lora,job_forward_wire,job_jax_4d_hires_and_lc,job_pidnet_v21_xfeat}.py`, `scripts/bench_gtx1650s_m3_timesliced.py`, `tests/colab/{__init__,test_g4_thunder_contracts,test_g4_jobs_smoke_cpu}.py` *(plus Phase 3 JSON outputs in `artifacts/v4.0.0/` and out-of-git blobs in `/var/tmp/vertirta_v400_blobs/`)* | **Produces for 6A:** `mlsd_wireseg_640_opset17.onnx`, `dfine_n_640_opset17.onnx`, `pidnet_slz_v21_opset17.onnx`, `xfeat_512_opset17.onnx`, `local_gtx1650s_m3_timesliced_proof.json`. **Consumes from 6B / Produces for 6B:** Reads `benchmarks/vertibench_v2/` splits (or CLI builder) -> emits `g4_qwen3_v3_vertibench_v2_eval.json`, `g4_whisper_bvi_lora_eval.json`, LoRA adapters, GGML Q8_0 conversion recipes. **Produces for Frozen RTA & LC:** `11,313,864 B` `VRTALUT1` + `101×71` `VRTALUT2` archives. |
| **Plan 6A (`@vision-slz-eng`)** *(Read-Only to 6C)* | `core/forward_hazard/**`, `core/gnss_denied/**`, `sim/gazebo_plugins/mmwave_77ghz/**`, `ros2_ws/src/vertirta_{forward_hazard,gnss_denied}_node/**`, `apps/vertigcs/**`, `.github/workflows/m3_perception_voice_forensics.yml`, `tests/{forward_hazard,gnss_denied}/**` | Plan 6C NEVER edits any Plan 6A file. Locked ONNX tensor names/shapes documented in Tasks 3 & 5 below so Plan 6A's C++ ONNX adapters match bit-for-bit! |
| **Plan 6B (`@ai-data-auditor`, `@career-reg-analyst`, `@px4-sim-eng`)** *(Read-Only to 6C)* | `core/voice/**`, `core/forensics/**`, `benchmarks/vertibench_v2/**`, `docs/paper_corl/**`, `ros2_ws/src/vertirta_voice_node/**`, `tests/{voice,forensics,vertibench_v2}/**` | Plan 6C NEVER edits any Plan 6B file. `job_qwen3_v3_vertibench_v2.py` reads `benchmarks/vertibench_v2/` JSONLs read-only (with `--data-dir` override and self-contained `--smoke-cpu` generator). |

---

## Review Focus (Six Critical Failure Modes to Audit)

1. **Wave 2 XLA + PyTorch Co-Residency Collision (Tasks 4, 5, 6):** Default JAX preallocates `75%` of GPU memory and immediately OOM-kills a co-resident PyTorch process. Verify `remote_g4_driver.py` enforces (a) a hard Wave-1 -> Wave-2 GPU drain barrier (`0` active compute PIDs, `memory.used < 1,500 MiB`), (b) explicit `XLA_PYTHON_CLIENT_PREALLOCATE=true` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.58` ONLY in the JAX child environment, (c) `torch.cuda.set_per_process_memory_fraction(0.30, 0)` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the PyTorch child, and (d) `0.58 + 0.30 = 0.88 <= 0.90` asserted in `test_g4_thunder_contracts.py` (`~10 GiB` safety headroom on `95.59 GiB` G4).
2. **`101×71×101×71` vs. Frozen `41×29×41×29` `.vrta_lut` Half-Integer Ratio Soundness (Task 4):** `100/40 = 2.5` and `70/28 = 2.5` CANNOT use `v_fine[::2, ::2, ::2, ::2]` (which worked only for `81×57`). Verify `job_jax_4d_hires_and_lc.py` projects `101×71×101×71` onto `41×29×41×29` via exact quadrilinear sampling, audits optimistic overestimation `eps_grid_f` across ALL `51,423,241` fine nodes via chunked slab interpolation (zero RAM OOM), emits distinct `"VRTALUT2"` magic for full `101×71` archives and exact `11,313,864 B` `"VRTALUT1"` for frozen `MmapCbvf4DLut`, and verifies `DR-2` Errata `#1–#3` + `Lift+Cruise 8+1` hover-OEI `<= 79.2% T_max` constants.
3. **Zero Dummy-VRAM Padding & OOM-Proof Wave Sizing (Tasks 1–6):** Verify `tests/colab/test_g4_thunder_contracts.py` statically checks all `g4_jobs/*.py` for absence of dummy VRAM reservation tensors, while real OOM-safe workloads (`G=16` `Qwen3-4B` Z3-GRPO with `prompts_per_microbatch=1..2`, `grad_accum=4..8`, `gradient_checkpointing=True`, completion-only sliced `log_softmax` + `Whisper-large-v3-turbo`/`small` batch-64/96 + `M-LSD-WireSeg`/`D-FINE-N` batch-64 in Wave 1; `vmap_batch_size=2` safe-default `101×71` JAX PDE in `55.44 GiB` XLA BFC pool + `batch_size=32` `PIDNet-S v2.1` w/ `DINOv2-Large`+`DAv2-Small` in Wave 2) naturally saturate `78–86 GB` logged by `1 Hz` `nvidia-smi` without `RESOURCE_EXHAUSTED` or FP32 full-logit OOMs.
4. **Strict Zero-GPL / Zero-NC / Zero-ShareAlike License Firewall & `SkyScenes` Nested-Tar Guard (Tasks 1, 2, 3, 5):** Verify every job invokes `validate_job_license_manifest(...)` BEFORE any network fetch, globally rejecting `UAVid`, `UWB-ATCC`, `ATCO2`, `ESC-50`, `DEMAND`, `ATCOSIM`, `piper1-gpl`, all prebuilt `tr_TR` Piper voices, `BlenderProc`, `InsPLAD`, `DDOS`, `Drone-vs-Bird`, `DINOv3`, `DINOv2-SALAD`, `Sample4Geo`, `SuperPoint`, `Ultralytics`, rejecting `UAVScenes`/`AeroScapes`/`ICG` in `train_sources` (allowing them strictly in quarantined zero-shot `eval_only_sources` per `01-DR08-13` Override `#7`), and rejecting `TTPLA` inside nadir SLZ. Verify `job_pidnet_v21_xfeat.py` retains `v3.5.1`'s `<= 30 MB` PNG/JPEG magic filter on `SkyScenes` tarballs.
5. **Single-Open / Single-Close Teardown & Pre-Existing Endpoint Safety (Task 6):** Verify `run_g4_m3_thunder_run.py` refuses to open `G4` if local pre-flight fails, snapshots `pre_existing_endpoints` before `colab new`, persists `owned_endpoint` to `/var/tmp/vertirta_v400_blobs/.active_g4_endpoint.json`, runs BOTH `colab stop` and `colab_cli.client.Client.unassign(owned_endpoint)` inside `try / finally` (verified via mocked exception injection in `test_g4_thunder_contracts.py`), never unassigns pre-existing external endpoints, and verifies `owned_endpoint_released: true`.
6. **Empirical TU116 FP32-vs-FP16 Gate & Single-Context `< 300 MiB` Time-Slicing (Task 7):** Verify `scripts/bench_gtx1650s_m3_timesliced.py` never assumes `2×` FP16 speedup on `GTX 1650 SUPER`, profiles EVERY ONNX model (`PIDNet-S v2.1`, `M-LSD-WireSeg`, `D-FINE-N`, `XFeat`) in both FP32+CUDA-Graph and FP16 inside a single process, verifies SHA-256 parity against `artifacts/v4.0.0/*.json`, and enforces `< 300.0 MiB` peak time-sliced VRAM + `< 12.0 ms` p99 SLZ latency.

---

## File Structure

- `scripts/colab/g4_jobs/__init__.py` — Package marker exporting `G4_JOB_REGISTRY` and pinned Wave 1 / Wave 2 memory fractions.
- `scripts/colab/g4_jobs/common.py` — Shared memory-fraction guard (`configure_torch_vram_fraction`), `PYTORCH_CUDA_ALLOC_CONF` validator, zero-GPL/NC/ShareAlike `validate_job_license_manifest`, SHA-256 helper, deterministic seeding, atomic JSON manifest writer, and standardized `--dry-run` / `--smoke-cpu` CLI parser.
- `scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py` — Wave 1 (`0.46`) `Qwen3-4B v3` on `VertiBench-v2` (`N_cal=2,000` [`800 TR + 800 EN + 400 MIX`, `n_g=500` per SNR stratum], `N_test=1,200`, 1 shared SFT + 3 independent `G=16` Z3-GRPO seeds with `prompts_per_microbatch=1..2`, `grad_accum=4..8`, `gradient_checkpointing=True`, completion-only sliced `log_softmax`, live GBNF+Z3, language-conditional + 4-stratum SNR Mondrian calibration targeting `Mean Set Size < 1.8` at `>= 95%` coverage).
- `scripts/colab/g4_jobs/job_whisper_bvi_lora.py` — Wave 1 (`0.22`) `Whisper-large-v3-turbo` (`809M`) + `Whisper-small` (`244M`) BF16 LoRA on clean permissive audio + `ml.voice.{bvi_cockpit_dsp,audio_license_audit}` (`X8-240` `114.6 Hz` coaxial BVI DSP) + 4-stratum SNR (`S0_Clean`, `S1_Cruise`, `S2_Hover`, `S3_Garbled`) WER & N-best entropy export.
- `scripts/colab/g4_jobs/job_forward_wire.py` — Wave 1 (`0.20`) `M-LSD-WireSeg` (`clDice` loss) + `D-FINE-N` (`640×640` IMU-stabilized horizon split-stack) on `TTPLA` + `Mendeley TEİAŞ` + `Purdue UAV` + Apache-2.0 synthetic catenary wires + Opset-17 ONNX export.
- `scripts/colab/g4_jobs/job_jax_4d_hires_and_lc.py` — Wave 2 (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.58`) `101×71×101×71` X8-240 + `Lift+Cruise 8+1` 4D HJI-CBVF PDE (`vmap_batch_size=2` safe default WENO5+TVD-RK3) exporting lossless `VRTALUT2` + sound `51.4M`-audited `11,313,864 B` `VRTALUT1` + convergence proof JSON.
- `scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py` — Wave 2 (`0.30`) `PIDNet-S v2.1` (`batch_size=32`, oblique-recall balanced focal + asymmetric FSR loss recovering `AeroScapes mIoU` at `FSR <= 3%`) + `XFeat + LighterGlue` + `MegaLoc` Kartal `0.5 m` visual relocalization benchmark + Opset-17 ONNX export.
- `scripts/colab/remote_g4_driver.py` — On-VM Wave 1 -> hard `nvidia-smi` drain barrier -> Wave 2 process supervisor with `1 Hz` GPU telemetry daemon, heartbeat files, and zero-zombie cleanup.
- `scripts/colab/run_g4_m3_thunder_run.py` — Host-side zero-waste orchestrator: `0 CU` local smoke gate -> pre-existing endpoint snapshot -> single `G4` open -> upload -> WS-resilient monitor -> SHA-256 verified blob/JSON download -> guaranteed `try/finally` `colab stop` + `colab_cli.client.Client.unassign()` verification.
- `scripts/bench_gtx1650s_m3_timesliced.py` — Physical `GTX 1650 SUPER` (`TU116`) single-context flight-phase time-sliced ONNX Runtime CUDA FP32-vs-FP16 benchmark (`< 300 MiB` VRAM, `< 12 ms` p99 SLZ) + `--smoke-cpu` test mode.
- `tests/colab/__init__.py` — Test package marker.
- `tests/colab/test_g4_thunder_contracts.py` — Static license firewall, VRAM budget invariants, AST anti-dummy-padding audit, mocked `try/finally` teardown & pre-existing-endpoint safety tests.
- `tests/colab/test_g4_jobs_smoke_cpu.py` — End-to-end `0 CU` local `--dry-run` + `--smoke-cpu` tests across all 5 jobs, `remote_g4_driver.py`, `run_g4_m3_thunder_run.py`, and `bench_gtx1650s_m3_timesliced.py`.

---

### Task 1: Shared G4 Job Runtime Guard, Zero-GPL/NC/SA License Firewall & Static Contract Tests (`scripts/colab/g4_jobs/common.py`)

**Files:** Create `scripts/colab/g4_jobs/__init__.py`, `scripts/colab/g4_jobs/common.py` | Test `tests/colab/__init__.py`, `tests/colab/test_g4_thunder_contracts.py`
**Interfaces:**
- Consumes: Standard library (`argparse`, `ast`, `hashlib`, `json`, `os`, `pathlib`, `subprocess`, `time`) + optional `torch` (guarded import so `--dry-run` / `--smoke-cpu` works without CUDA).
- Produces:
  - `WAVE_BUDGETS`: frozen immutable mapping `{ "wave1": {"qwen3_v3": 0.46, "whisper_bvi": 0.22, "forward_wire": 0.20}, "wave2": {"jax_4d_hires_lc": 0.58, "pidnet_v21_xfeat": 0.30} }` asserting `sum(wave) <= 0.90` at import time.
  - `HARD_BANNED_IDENTIFIERS`: globally banned frozenset (rejected in `train_sources`, `eval_only_sources`, AND `models`) covering `01-DR08-13` §3.1 (`uavid`, `insplad`, `ddos`, `drone-vs-bird`, `uwb-atcc`, `atco2`, `esc-50`, `demand`, `atcosim`, `piper1-gpl`, `tr_tr-dfki`, `tr_tr-fahrettin`, `tr_tr-fettah`, `xtts-v2`, `mms-tts`, `omnivoice`, `blenderproc`, `dinov3`, `dinov2-salad`, `sample4geo`, `superpoint`, `superglue`, `efficientloftr`, `ultralytics`, `segformer`, `deep-hough`, `aya-expanse`, `openvins`, `vins-fusion`, `orb-slam3`, `stn-plad`, `pampc`, `pldu`, `pldm`, `synthwires`, `wireseghr`, `sod4sb`, `seraphim`).
  - `TRAIN_ONLY_BANNED_IDENTIFIERS`: frozenset `{"aeroscapes", "icg", "uavscenes"}` — strictly rejected in `train_sources` and calibration splits, permitted ONLY inside `eval_only_sources` with explicit quarantine tag (matching `01-DR08-13` Override `#7`, `Spec §6.1`, Plan 6A, and Plan 6B Paper 2).
  - `CANONICAL_SNR_STRATA`: `("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")` — exact Plan 6B canonical stratum keys shared across `job_qwen3_v3_vertibench_v2.py` and `job_whisper_bvi_lora.py`.
  - `validate_job_license_manifest(job_name: str, train_sources: list[dict], eval_only_sources: list[dict], models: list[dict]) -> dict`: fails closed (`ValueError`) on ANY `HARD_BANNED_IDENTIFIERS` token anywhere, ANY `TRAIN_ONLY_BANNED_IDENTIFIERS` or `GPL`/`LGPL`/`AGPL`/`NC`/`SA` license in `train_sources` or `models`, or `TTPLA` appearing outside `job_forward_wire`.
  - `configure_torch_vram_fraction(fraction: float, *, device_index: int = 0) -> dict`: sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`, validates `0.05 <= fraction <= 0.65`, calls `torch.cuda.set_per_process_memory_fraction(fraction, device_index)` when CUDA is active, and records GPU identity + cap in metadata.
  - `build_standard_argparser(job_name: str, default_vram_fraction: float)` supporting `--dry-run`, `--smoke-cpu`, `--vram-fraction`, `--out-json`, `--blob-dir`, `--seed`, `--data-dir`.
  - `write_canonical_json(path: Path, payload: dict) -> str`: writes pure UTF-8 LF JSON with deterministic key ordering and embedded `report_sha256` (computed over payload excluding `report_sha256`).

- [ ] **Step 1: Write the failing contract test (`tests/colab/test_g4_thunder_contracts.py`)**
```python
from pathlib import Path
import pytest
from scripts.colab.g4_jobs.common import (
    WAVE_BUDGETS,
    validate_job_license_manifest,
)

def test_wave_vram_fractions_stay_within_safe_g4_ceiling() -> None:
    assert sum(WAVE_BUDGETS["wave1"].values()) == pytest.approx(0.88, abs=1e-6)
    assert sum(WAVE_BUDGETS["wave2"].values()) == pytest.approx(0.88, abs=1e-6)
    assert sum(WAVE_BUDGETS["wave1"].values()) <= 0.90
    assert sum(WAVE_BUDGETS["wave2"].values()) <= 0.90

@pytest.mark.parametrize("banned_name,banned_lic", [
    ("UWB-ATCC", "CC BY-NC-SA 4.0"),
    ("UAVid", "CC BY-NC-SA 4.0"),
    ("InsPLAD", "CC BY-NC 3.0"),
    ("DINOv2-SALAD", "GPL-3.0"),
    ("tr_TR-fahrettin-medium", "CC0-lessac-NC"),
    ("BlenderProc", "GPL-3.0"),
])
def test_license_gate_blocks_all_hard_banned_assets(banned_name: str, banned_lic: str) -> None:
    with pytest.raises(ValueError):
        validate_job_license_manifest(
            job_name="job_test",
            train_sources=[{"name": banned_name, "license": banned_lic}],
            eval_only_sources=[],
            models=[],
        )

def test_license_gate_blocks_ttpla_inside_nadir_slz() -> None:
    with pytest.raises(ValueError, match="TTPLA"):
        validate_job_license_manifest(
            job_name="job_pidnet_v21_xfeat",
            train_sources=[{"name": "TTPLA", "license": "Apache-2.0"}],
            eval_only_sources=[],
            models=[],
        )
```
- [ ] **Step 2: Run test to verify failure** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest python -m pytest tests/colab/test_g4_thunder_contracts.py -v` -> FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement `scripts/colab/g4_jobs/__init__.py` and `scripts/colab/g4_jobs/common.py`** — Include `sha256_file`, `query_nvidia_smi_snapshot`, `validate_job_license_manifest`, `configure_torch_vram_fraction`, `write_canonical_json`, and `build_standard_argparser`.
- [ ] **Step 4: Run test to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest python -m pytest tests/colab/test_g4_thunder_contracts.py -v` -> PASS.

---

### Task 2: Wave 1 Jobs — `job_qwen3_v3_vertibench_v2.py` (`0.46`) & `job_whisper_bvi_lora.py` (`0.22`)

**Files:** Create `scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py`, `scripts/colab/g4_jobs/job_whisper_bvi_lora.py` | Test `tests/colab/test_g4_jobs_smoke_cpu.py`
**Interfaces:**
- **`job_qwen3_v3_vertibench_v2.py`:**
  - *Consumes:* Read-only `benchmarks/vertibench_v2/{train,cal,test}.jsonl` (from Plan 6B when present, or `--data-dir`; self-contained synthetic split in `--smoke-cpu`), read-only `ml/copilot/calibrate_conformal.py` (`cv_select_gamma_tau_calibration_only`, `fit_mondrian_conformal_groups`, `bootstrap_resplit_mondrian_coverage` from Plan 6B — with graceful backward-compatible fallback if run standalone before Plan 6B merges), `ml/copilot/{grpo_z3_rewards,train_sft_grpo}.py`, and `benchmarks/vertibench_tr_en/{audit_cli,validate_vertibench}.py`.
  - *G4 Training/Calibration Execution (`--vram-fraction 0.46`, `~38–43 GB` inside `43.97 GiB` cap):*
    - Verifies `Qwen/Qwen3-4B (Apache-2.0)` via `validate_job_license_manifest(...)` and bans `aya-expanse-8b`.
    - Verifies anti-contamination gate (`seed_overlap == 0`, `max_char_5gram_jaccard < 0.15`) and stamps `eval_split_provenance = "PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2"`.
    - **OOM-Proof Realistic Wave-1 Training Schedule:** Runs **1 shared Stage-1 BF16 LoRA SFT checkpoint** (`r=64, alpha=128`, 7 linear projections, B3 adversarial + B4 clarification/noise oversampling) -> **3 independent Stage-2 `G = 16` Z3-GRPO seeds** (`seeds = [20261124, 20261125, 20261126]`, `80–120` steps/seed) with **`prompts_per_microbatch = 1..2`** (`per_device_train_batch_size = 16..32` completions), **`gradient_accumulation_steps = 4..8`** (effective `128` completions/step), **`gradient_checkpointing = True`**, and **completion-only sliced `log_softmax`** (preventing `[128, L, 151936]` FP32 full-logit `~60+ GiB` OOM!), `dr_grpo`, `beta = 0.04`, `eps = [0.20, 0.28]`, `z3_pool_start_method = "spawn"` strictly, live GBNF-valid grammar constraint + Z3 corridor verifier reward.
    - Runs **Language-Group-Conditional (`TR`, `EN`, `MIX`; `N_cal = 2,000` [`800 TR + 800 EN + 400 MIX`], `N_test = 1,200`) AND 4-Stratum SNR Mondrian (`S0_Clean`, `S1_Cruise`, `S2_Hover`, `S3_Garbled`, `n_g = 500 >= 250/stratum`)** split-conformal calibration via Plan 6B's `cv_select_gamma_tau_calibration_only`, `fit_mondrian_conformal_groups`, and `bootstrap_resplit_mondrian_coverage(..., n_resplits=100)` (`alpha = 0.05`, Paper 2 CoRL table input!), auditing:
      - `Mean Set Size < 1.8` target (down from `v3.5.1` `3.125`),
      - Per-language (`TR`/`EN`/`MIX`) & per-SNR-stratum (`S0_Clean..S3_Garbled`) coverage `>= 95.0%` with exact Clopper-Pearson, Beta-Binomial 95% CIs, and `100`-resplit bootstrap distribution,
      - MAYDAY/PAN-PAN asymmetric priority (`w_m >= 0.15` never silently abstained),
      - `< 5.0%` SoC `LIVENESS_EMERGENCY_COMMIT` deterministic override (`0` stalls under low SoC).
    - Exports shared SFT + 3 GRPO LoRA adapters (`qwen3_4b_v3_seed{k}_lora/`) to `--blob-dir` and writes `g4_qwen3_v3_vertibench_v2_eval.json` (with 3-seed mean ± std + `100`-resplit Mondrian summary + exact SHA-256s).
  - *`--smoke-cpu` Mode (`0 CU`):* Runs full dataset anti-contamination audit, Z3 reward verifier check, synthetic teacher-forced logit `TR`/`EN`/`MIX` + 4-stratum SNR (`S0_Clean..S3_Garbled`) Mondrian calibration (`N_cal=2,000` numeric pass + `100`-resplit check in `< 1.0 s`!), MAYDAY & `< 5%` SoC override assertions, and emits a schema-complete `SMOKE_CPU_ONLY` JSON report.
- **`job_whisper_bvi_lora.py`:**
  - *Consumes:* Strictly clean permissive audio only (`Common Voice 27.0` CC0, `ISSAI TSC` MIT/CC BY 4.0, `FLEURS`/`MediaSpeech` CC BY 4.0, `MUSAN` CC BY 4.0, `OpenSLR-28 RIRS` Apache-2.0, license-filtered `FSD50K`), **directly importing Plan 6B's `ml.voice.audio_license_audit` and `ml.voice.bvi_cockpit_dsp`** (with an identical built-in fallback implementation only if `ml.voice` is absent during standalone Wave-1 testing).
  - *Canonical `X8-240` Coaxial BVI DSP Engine (`01-DR08-13` Override `#1`, synced to `ml.voice.bvi_cockpit_dsp`):*
    - Synthesizes time-varying rotor RPM across **`1,100–2,292 RPM`** (3-blade `BPF = 55.0–114.6 Hz`, `~115 Hz` ceiling matching `omega_max = 240 rad/s` in `config/vehicle_x8_240.yaml` — NEVER capped at `DR-12`'s drifted `1,850 RPM`!), 4-crossover coaxial blade-interaction modulation (`4 × f_shaft`), 10-pole-pair PMSM inverter tone (`200–3,500 Hz`), pink turbulence floor, short-cabin RIR (`RT60 = 0.12–0.25 s`), and VHF `300–3,400 Hz` Butterworth bandpass + PTT squelch click, calibrated to exact target SNR across the **4 canonical Plan 6B strata**: `S0_Clean (> 15 dB)`, `S1_Cruise (8–15 dB)`, `S2_Hover (2–8 dB)`, `S3_Garbled (< 2 dB)`.
  - *G4 Training/Eval Execution (`--vram-fraction 0.22`, `~20–22 GB`):*
    - Fine-tunes BOTH **`openai/whisper-large-v3-turbo`** (`809M`, MIT — explicitly pinning `task="transcribe"`, `translate=False`, `no_context=True`, `audio_ctx in [256, 384]`) AND **`openai/whisper-small`** (`244M`, MIT) with BF16 LoRA (`r=32, alpha=64`, batch `64–96`).
    - Evaluates 4-stratum SNR (`S0_Clean`, `S1_Cruise`, `S2_Hover`, `S3_Garbled`) TR/EN WER, CER, aviation slot error rate (`Kartal Sahil`, `LTFJ`, `LTBX`, `FATO-1..4`, `TC-VRT01..20`, `MAYDAY`/`PAN-PAN`), `audio_ctx=256` vs `1500` latency speedup, and N-best (`K=4`) normalized Shannon entropy + `p_nospeech` exported to `g4_whisper_bvi_lora_eval.json` (feeding Plan 6B's `core/voice/` Mondrian conformal gate!).
    - Saves merged BF16 LoRA checkpoints + deterministic `whisper.cpp` GGML `Q8_0` export manifest in `--blob-dir`.
  - *`--smoke-cpu` Mode (`0 CU`):* Runs startup license gate (`validate_job_license_manifest` + `ml.voice.audio_license_audit`), generates real `16 kHz` coaxial `115 Hz` BVI + VHF-filtered waveforms across all 4 canonical SNR strata (`S0_Clean..S3_Garbled`), verifies spectral BPF peak at `114.6 Hz ± 2 Hz` and measured SNR error `< 0.5 dB` via FFT/RMS, validates `translate=False` & `audio_ctx >= 256` config locks, and emits schema-complete `SMOKE_CPU_ONLY` JSON.

- [ ] **Step 1: Write failing `--dry-run` & `--smoke-cpu` tests in `tests/colab/test_g4_jobs_smoke_cpu.py`**
```python
import json, subprocess, sys
from pathlib import Path

CANONICAL_STRATA = ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled")

def test_qwen3_v3_and_whisper_bvi_smoke_cpu(tmp_path: Path) -> None:
    q_json = tmp_path / "qwen3_smoke.json"
    subprocess.run(
        [sys.executable, "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py",
         "--smoke-cpu", "--out-json", str(q_json), "--blob-dir", str(tmp_path / "blobs")],
        check=True,
    )
    qd = json.loads(q_json.read_text(encoding="utf-8"))
    assert qd["mode"] == "SMOKE_CPU_ONLY"
    assert qd["eval_split_provenance"] == "PROGRAMMATIC_PARAPHRASE_HELDOUT_LOW_JACCARD_PENDING_HUMAN_NATIVE_V2"
    assert qd["conformal_language_conditional"]["TR"]["n_cal"] >= 750
    assert qd["conformal_language_conditional"]["EN"]["n_cal"] >= 750
    assert "MIX" in qd["conformal_language_conditional"]
    assert all(qd["conformal_snr_mondrian"][s]["n_cal"] >= 250 for s in CANONICAL_STRATA)
    assert qd["bootstrap_resplit_audit"]["n_resplits"] == 100
    assert qd["liveness_soc_lt_5pct_override_verified"] is True
    assert qd["mayday_priority_wm_015_verified"] is True

    w_json = tmp_path / "whisper_smoke.json"
    subprocess.run(
        [sys.executable, "scripts/colab/g4_jobs/job_whisper_bvi_lora.py",
         "--smoke-cpu", "--out-json", str(w_json), "--blob-dir", str(tmp_path / "blobs")],
        check=True,
    )
    wd = json.loads(w_json.read_text(encoding="utf-8"))
    assert wd["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert wd["whisper_inference_locks"]["translate"] is False
    assert wd["whisper_inference_locks"]["min_audio_ctx"] == 256
    assert 112.0 <= wd["coaxial_bvi_dsp_audit"]["max_rpm_bpf_hz"] <= 117.0
    assert set(wd["snr_strata_metrics"].keys()) == set(CANONICAL_STRATA)
```
- [ ] **Step 2: Run test to verify failure** -> FAIL (scripts do not exist yet).
- [ ] **Step 3: Implement `scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py` and `scripts/colab/g4_jobs/job_whisper_bvi_lora.py`** — Ensure zero dummy VRAM padding, strict startup license gate, `spawn` Z3 pool, `X8-240` `2,292 RPM / 114.6 Hz` BPF synthesizer, and full `--dry-run` / `--smoke-cpu` paths.
- [ ] **Step 4: Run test to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with scipy --with pyyaml --with z3-solver python -m pytest tests/colab/test_g4_jobs_smoke_cpu.py -k "qwen3_v3_and_whisper" -v` -> PASS.

---

### Task 3: Wave 1 Job — `job_forward_wire.py` (`0.20` — `M-LSD-WireSeg` `clDice` + `D-FINE-N` `640×640` Split-Stack)

**Files:** Create `scripts/colab/g4_jobs/job_forward_wire.py` | Modify `tests/colab/test_g4_jobs_smoke_cpu.py`
**Interfaces:**
- *Consumes:* Approved permissive forward datasets ONLY — `TTPLA` (`Apache-2.0`, strictly forward `45°` wire head ONLY), `Mendeley TEİAŞ Powerline GT` (`CC BY 4.0`), `Purdue UAV` (`BSD-3-Clause`), and built-in Apache-2.0 procedural catenary wire + tower-crane horizon generator (zero `BlenderProc`/`bpy` GPL imports).
- *IMU-Stabilized Split-Stack Preprocessing (`DR-10` §1.1):* Derotates `1280×720` (`45°` HFOV, `f_net = 1545.1 px`) by roll/pitch, crops `1280×320` horizon strip, splits into Left `640×320` + Right `640×320` (with `16 px` overlap metadata), and stacks vertically into canonical **`1×3×640×640`** NCHW input tensor. Normalized `(u, v)` coordinate channels are baked as static constant buffers INSIDE the ONNX graph so C++ `core/forward_hazard/` passes plain `1×3×640×640` RGB frames with zero host copy overhead.
- *Differentiable `clDice` Topology Loss (`DR-10` §1.3):* Implements `SoftClDiceLoss(iters=3)` using differentiable min/max-pooling soft-skeletonization (`L = L_BCE + L_Dice + 0.5 * L_clDice`) preserving `0.25–0.60 px` thin-wire homotopy without broken dotted segments.
- *Locked Opset-17 ONNX Interface for Plan 6A (`core/forward_hazard/`) & Task 7 (`bench_gtx1650s_m3_timesliced.py`):*
  1. `mlsd_wireseg_640_opset17.onnx` (compact Apache-2.0 Mobile-style wire+segment encoder-decoder, `< 3.5M` params, zero `AdaptiveAvgPool2d`, static batch `1`):
     - Input: `rgb_split_nchw` `float32[1, 3, 640, 640]`
     - Outputs: `wire_mask_logits` `float32[1, 1, 640, 640]`, `line_junction_map` `float32[1, 5, 160, 160]` (center + `(dx0, dy0, dx1, dy1)` segment endpoints)
  2. `dfine_n_640_opset17.onnx` (compact Apache-2.0 DETR/D-FINE-Nano style single-scale/FPN detector, `< 4.0M` params, zero `AdaptiveAvgPool2d`, static batch `1`):
     - Input: `rgb_split_nchw` `float32[1, 3, 640, 640]`
     - Outputs: `boxes_xyxy` `float32[1, 300, 4]` (normalized `[0, 1]`), `class_logits` `float32[1, 300, 3]` (`0: Bird, 1: Rogue_UAV, 2: Tower_Crane`)
- *`--smoke-cpu` Mode (`0 CU`):* Runs startup license gate (verifying `TTPLA` allowed here, `InsPLAD`/`DDOS`/`Drone-vs-Bird`/`BlenderProc` rejected), executes 2 CPU forward+backward steps of `M-LSD-WireSeg` with `SoftClDiceLoss` and `D-FINE-N` on synthetic catenary split-stack frames (`1280×720 -> 640×640`), exports both Opset-17 ONNX graphs into `--blob-dir`, verifies `onnx.checker.check_model` + exact input/output tensor names & shapes, and writes `g4_forward_wire_metrics.json`.

- [ ] **Step 1: Add failing test `test_forward_wire_smoke_cpu_and_onnx_contracts` to `tests/colab/test_g4_jobs_smoke_cpu.py`** — Asserts `SoftClDiceLoss` finite gradient, `1280×720 -> 640×640` split-stack geometry, both ONNX files pass `onnx.checker.check_model`, zero `AdaptiveAvgPool` / `GlobalAveragePool` dynamic ops, exact tensor shapes, and clean license manifest.
- [ ] **Step 2: Run test to verify failure** -> FAIL.
- [ ] **Step 3: Implement `scripts/colab/g4_jobs/job_forward_wire.py`** — Clean Apache-2.0 PyTorch `M-LSD-WireSeg` + `D-FINE-N` architectures, `SoftClDiceLoss`, procedural catenary + crane horizon dataset generator, `TTPLA` + `Mendeley TEİAŞ` + `Purdue UAV` loaders, BF16 training loop (`batch_size=64` on G4 under `fraction=0.20`), and Opset-17 static ONNX exporters.
- [ ] **Step 4: Run test to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with torch --with onnx --with pillow python -m pytest tests/colab/test_g4_jobs_smoke_cpu.py -k "forward_wire" -v` -> PASS.

---

### Task 4: Wave 2 Job — `job_jax_4d_hires_and_lc.py` (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.58` — `101×71` X8-240 & `Lift+Cruise 8+1` 4D HJI-CBVF PDE)

**Files:** Create `scripts/colab/g4_jobs/job_jax_4d_hires_and_lc.py` | Modify `tests/colab/test_g4_jobs_smoke_cpu.py`
**Interfaces:**
- *Consumes:* Read-only `reachability/{cbvf_4d_dynamics,solve_cbvf_luts,solve_cel_2d_lut,verify_lut_soundness}.py`, `config/vehicle_x8_240.yaml`, and canonical `Lift+Cruise 8+1` constants matching `core/airframes/lift_cruise/include/vertirta/airframes/lift_cruise_dynamics.hpp` (`m = 240.0 kg`, `S = 5.20 m^2`, `V_stall_1g = 21.83 m/s`, `V_blend = [14.0, 28.0] m/s`, `V_cruise = 38.0 m/s`, `V_NE = 46.3 m/s`, `T_max_lift = 8 × 544.3 N`, hover-OEI `<= 79.2% T_max`, `T_max_pusher = 950.0 N`, `L/D_max = 13.46`). Zero edits to `reachability/` or `core/`!
- *XLA Memory Partitioning Lock (`0.58` = `55.44 GiB` cap):* Verifies `os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]` is set to `<= 0.58` BEFORE importing `jax` (sets it if invoked standalone). Uses **`vmap_batch_size = 2` as the OOM-safe default** (`2 × 2` regimes = `102,846,482` simultaneous FP32 WENO5 + TVD-RK3 nodes on **`HIRES_SHAPE = (101, 71, 101, 71)`**, keeping peak working memory cleanly at **`~34–48 GB`** inside the `55.44 GiB` XLA BFC pool and avoiding `4`-regime `~57.6 GB` `RESOURCE_EXHAUSTED`!). Optional `vmap_batch_size = 4` is strictly opt-in behind `--allow-vmap4` with explicit `jax.checkpoint` remat + preflight memory check; exact `vmap_batch_size` and `peak_bytes_in_use` are always logged.
- *Exact Half-Integer `101×71 -> 41×29` Sound Projection & `51.4M`-Node Slab Audit:*
  - Projects `V_hires (101, 71, 101, 71)` onto frozen `LUT_EXPORT_SHAPE = (41, 29, 41, 29)` by exact quadrilinear evaluation at the `41×29×41×29` coordinates (`idx_hires = 2.5 * idx_coarse`, exact nodes on even indices and exact midpoints on odd indices).
  - Audits optimistic overestimation `eps_grid_f` across **ALL `51,423,241` nodes** of `V_hires` using a **slab-by-slab CPU chunked quadrilinear evaluator** (`<= 120 MB` host RAM peak):
    `eps_grid_f = max_{|V_hires| <= 10 m} max(0, V_hat_41x29(x_hires) - clip(V_hires(x_hires))) + 0.5 * q`
  - Computes exact grid-convergence delta vs `v3.1.0` `81×57` (`||V_101x71 - V_81x57||_mean/p95/max` inside `|V| <= 10 m`).
- *Dual-Airframe PDE Exports (strictly into `--blob-dir`, never git):*
  1. **Coaxial `X8-240` (4 CSFL regimes: `Nominal, 1M, 2M-Cross, Low-Vbat`):**
     - `cbvf_4d_x8_240_hires_101x71.vrta_lut_hires` (`VRTALUT2` header `4096 B` + `4 × 51,423,241 × 2 B = 411,389,928 B`)
     - `cbvf_4d_x8_240_v4_from_101x71.vrta_lut` (exact **`11,313,864 B` `VRTALUT1`**, verified by `verify_all_soundness(...)` + DR-2 Errata `#1–#3` + conservative regime hierarchy `V_0 >= V_1 >= V_3 >= V_2`!)
  2. **`Lift+Cruise 8+1` Transition Envelope (4 regimes: `0: Hover-Nominal T/W=1.85`, `1: Hover-OEI-8p1 T/W_eff=1.46 <= 79.2% trim`, `2: Transition-Corridor [14-28 m/s] wing+lift+pusher`, `3: Cruise-FW-Degraded [38 m/s] pusher+wing`, extended velocity domain `lo_vr = -48.0 m/s` for `46.3 m/s V_NE`!):**
     - `cbvf_4d_lc_8p1_hires_101x71.vrta_lut_hires` (`VRTALUT2`)
     - `cbvf_4d_lc_8p1_41x29.vrta_lut` (exact **`11,313,864 B` `VRTALUT1`** with `Lift+Cruise` `lo/hi` domain bounds in header, drop-in readable by `MmapCbvf4DLut::load()`!)
- *`--smoke-cpu` Mode (`0 CU`):* Runs pure-NumPy kinematic 4D solver on scaled nested grids (`21×15×21×15 -> 9×7×9×7` for fast 2.5:1 ratio verification + exact `41×29×41×29` `11,313,864 B` `VRTALUT1` export for both X8-240 and Lift+Cruise 8+1), verifies `VRTALUT1` header magic/size/SHA-256 + `VRTALUT2` header magic + `eps_grid_f` soundness inequality `V_hat - eps_grid <= V_fine`, and writes `g4_jax_4d_hires_and_lc_manifest.json`.

- [ ] **Step 1: Add failing test `test_jax_4d_hires_and_lc_smoke_cpu` to `tests/colab/test_g4_jobs_smoke_cpu.py`** — Verifies `XLA_PYTHON_CLIENT_MEM_FRACTION <= 0.58` contract, exact `2.5:1` quadrilinear projection & soundness inequality `max(V_hat - eps_grid - V_fine) <= 1e-5` on the safety band, `11,313,864 B` `VRTALUT1` output for BOTH X8-240 and Lift+Cruise 8+1, and distinct `VRTALUT2` magic for high-res archive.
- [ ] **Step 2: Run test to verify failure** -> FAIL.
- [ ] **Step 3: Implement `scripts/colab/g4_jobs/job_jax_4d_hires_and_lc.py`** — Include `_LiftCruise4DTransitionDynamics`, exact 2.5:1 chunked quadrilinear projector + slab-by-slab `51.4M`-node `eps_grid` auditor, `VRTALUT2` serializer, `vmap_batch_size=2` safe-default solver (with opt-in `--allow-vmap4` + `jax.checkpoint` remat), and `v3.1.0` convergence comparator.
- [ ] **Step 4: Run test to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with pyyaml python -m pytest tests/colab/test_g4_jobs_smoke_cpu.py -k "jax_4d" -v` -> PASS.

---

### Task 5: Wave 2 Job — `job_pidnet_v21_xfeat.py` (`0.30` — `PIDNet-S v2.1` Oblique-Recall Fix + `XFeat + LighterGlue` + `MegaLoc` Kartal `0.5 m` Reloc)

**Files:** Create `scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py` | Modify `tests/colab/test_g4_jobs_smoke_cpu.py`
**Interfaces:**
- *Consumes:* Read-only `ml/slz_perception/{data/{license_gate,remappers},models/pidnet_slz,train/train_slz_distill}.py` and `scripts/colab/run_l4_pidnet_distill.py` helper functions (read-only import, NEVER modifying `run_l4_pidnet_distill.py`).
- *Part A — `PIDNet-S v2.1` Oblique-Recall Recovery (`~24–28 GB` on G4 via `batch_size = 32`, `fraction = 0.30`):*
  - Enforces live `validate_training_startup_license()` (`FLAIR-1` etalab-2.0, `OpenEarthMap` live permissive CC BY 4.0 + DL-DE-BY-2.0 whitelist, `FloodNet-Supervised v1.0` CDLA-P, `SkyScenes` MIT with `<= 30 MB` PNG/JPEG nested-tar filter from `v3.5.1`; `AeroScapes` CC BY-SA 4.0 strictly quarantined zero-shot eval-only; **ZERO `UAVid`, ZERO `TTPLA`**).
  - **Root-Cause Fix for `v3.5.1`'s Oblique `mIoU` Regression (`0.3395 -> 0.2387`):**
    - Replaces uniform safe-class suppression with **Oblique-Recall Balanced Focal (`gamma=2.0`, inverse-class-frequency weights clamped `[0.5, 3.0]`) + Asymmetric False-Safe Hinge Loss** (`L_fsr = mean(1[y in {2,3,4}] * ReLU(P_safe - margin)^2)` with cosine ramp `0.0 -> lambda_fsr_max`) + **Synthetic Oblique Homography/Pitch Augmentation** (`15°–45°` perspective warps applied to `35%` of nadir FLAIR-1/OEM/FloodNet frames alongside native oblique SkyScenes frames) + **Pareto Checkpoint Selector** picking `argmax val_mIoU` subject to `val_FSR <= 0.030` (`<= 3.0%`).
    - Targets recovering zero-shot `AeroScapes mIoU >= 0.34` while maintaining `AeroScapes FSR <= 3.0%` and permissive `val_mIoU >= 0.50`, `val_FSR <= 3.0%`.
    - Exports static `1×3×512×512` Opset-17 FP32 ONNX (`pidnet_slz_v21_opset17.onnx`, exact `8,003,817` deploy params, zero `AdaptiveAvgPool2d`, zero `dino_proj`) + post-hoc `ood_calibration_v21.json` fitted strictly on permissive ID val split.
- *Part B — `XFeat + LighterGlue` (`Apache-2.0`) + `MegaLoc` (`MIT`) Kartal `0.5 m` Relocalization Benchmark:*
  - Verifies strict license manifest (`XFeat` Apache-2.0, `LighterGlue` Apache-2.0, `MegaLoc` MIT; zero `DINOv3`/`SALAD`/`Sample4Geo`/`SuperPoint`/`EfficientLoFTR`).
  - Synthesizes deterministic `0.5 m/px` GSD Kartal coastal urban/shoreline orthophoto benchmark patches (and loads optional licensed GCP MBTiles if provided via `--ortho-dir`), simulating `50, 80, 100, 120 m` AGL nadir queries with yaw jitter, shadow/haze contrast shift, and `10% / 35% / 65%` Marmara water fractions.
  - Measures exact `CEP50`, `CE95`, RANSAC inlier pass rate (`>= 25` inliers, `< 2.5 px` RMSE, `> 25%` convex-hull coverage per `DR-10` §3.2), `MegaLoc` lost-in-space `R@1`/`R@5` retrieval, and exports compact `xfeat_512_opset17.onnx` (`input: gray_nchw [1,1,512,512]`, `outputs: descriptors [1,64,64,64], reliability [1,1,64,64], keypoints_logits [1,65,64,64]`) into `--blob-dir`.
- *`--smoke-cpu` Mode (`0 CU`):* Runs startup license gate on synthetic OEM HTML fixture, runs 2 CPU training steps of `PIDNet-S v2.1` with oblique focal+asymmetric-FSR loss, runs synthetic `0.5 m` Kartal keypoint-matching & homography RANSAC relocalization smoke test, exports both `pidnet_slz_v21_opset17.onnx` and `xfeat_512_opset17.onnx` (verified via `onnx.checker.check_model`), and emits `g4_pidnet_v21_xfeat_metrics.json`.

- [ ] **Step 1: Add failing test `test_pidnet_v21_xfeat_smoke_cpu_and_onnx` to `tests/colab/test_g4_jobs_smoke_cpu.py`** — Asserts `zero_sharealike_verified=True`, `pidnet_slz_v21_opset17.onnx` (`8,003,817` deploy params, exact `rgb_nchw -> seg_logits/surf_normals/bdry_prob`) and `xfeat_512_opset17.onnx` (`gray_nchw [1,1,512,512]`) pass `onnx.checker.check_model`, and `kartal_reloc_eval` reports finite `cep50_m`, `ce95_m`, and `ransac_gate_pass_rate`.
- [ ] **Step 2: Run test to verify failure** -> FAIL.
- [ ] **Step 3: Implement `scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py`** — Oblique-recall balanced focal + asymmetric FSR loss, Pareto checkpoint selector, `SkyScenes` `<= 30 MB` nested-tar filter, `XFeat+LighterGlue` + `MegaLoc` benchmark, and static Opset-17 ONNX exporters.
- [ ] **Step 4: Run test to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with torch --with onnx --with pillow python -m pytest tests/colab/test_g4_jobs_smoke_cpu.py -k "pidnet_v21" -v` -> PASS.

---

### Task 6: On-VM Two-Wave Supervisor (`remote_g4_driver.py`) & Host Single-Shot Orchestrator (`run_g4_m3_thunder_run.py`)

**Files:** Create `scripts/colab/remote_g4_driver.py`, `scripts/colab/run_g4_m3_thunder_run.py` | Modify `tests/colab/test_g4_thunder_contracts.py`, `tests/colab/test_g4_jobs_smoke_cpu.py`
**Interfaces:**
- **`scripts/colab/remote_g4_driver.py`:**
  - Runs on the Colab `G4` VM (and locally in `--smoke-cpu` / `--dry-run` mode!).
  - Starts a `1 Hz` `nvidia-smi` telemetry daemon thread (mocked cleanly in `--smoke-cpu`) logging `timestamp_utc`, `memory_used_mib`, `memory_total_mib`, `gpu_util_pct`, `mem_util_pct`, `power_w`, `temp_c`, and per-PID compute memory into `g4_thunder_telemetry.json`.
  - **Wave 1 Launch:** Spawns 3 parallel subprocesses (`job_qwen3_v3_vertibench_v2.py --vram-fraction 0.46`, `job_whisper_bvi_lora.py --vram-fraction 0.22`, `job_forward_wire.py --vram-fraction 0.20`) with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`, streaming per-job logs + atomic `heartbeat.json` every `15 s`.
  - **Hard Wave-1 -> Wave-2 Drain Barrier:** Joins all 3 Wave-1 processes, asserts return codes (recording exact per-job status; never silently swallowing failures), and polls `nvidia-smi --query-compute-apps=pid,used_gpu_memory` up to `60 s` until `0` compute PIDs remain and `memory.used < 1,500 MiB`.
  - **Wave 2 Launch:** Spawns 2 parallel subprocesses with disjoint allocator environments:
    - `job_jax_4d_hires_and_lc.py` with `XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.58`, `JAX_ENABLE_X64=False`
    - `job_pidnet_v21_xfeat.py --vram-fraction 0.30` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512`
  - Joins Wave 2, stops telemetry thread, writes `remote_bundle_manifest.json` listing every generated JSON + binary blob path, byte size, and SHA-256 digest.
- **`scripts/colab/run_g4_m3_thunder_run.py`:**
  - CLI modes: `--dry-run` (validates config, preflight command, tarball exclusion rules, wave commands, teardown hooks with `0 CU`), `--smoke-cpu` (runs `remote_g4_driver.py --smoke-cpu` locally end-to-end into `tmp_path` with `0 CU`), and `--live-g4 --confirm-single-shot-g4` (**explicit double-flag safety lock** so no accidental invocation can ever open a `G4`!).
  - **Live Single-Shot Lifecycle (`--live-g4 --confirm-single-shot-g4`):**
    1. Runs mandatory local pre-flight (`pytest tests/colab/ -v`); aborts with exit `2` if anything fails (`0 CU` spent!).
    2. Dynamically resolves the `colab_cli` Python interpreter (parsing the `#!` shebang of `shutil.which("colab")` or falling back to `Path.home() / ".local/share/uv/tools/google-colab-cli/bin/python"` — zero hardcoded usernames or personal home paths!) to invoke `colab_cli.client.Client.list_assignments()` and snapshot `pre_existing_endpoints`.
    3. Opens ONE `G4`: `colab new --gpu G4 -s vertirta-m3-g4-thunder`, captures `owned_endpoint`, writes `/var/tmp/vertirta_v400_blobs/.active_g4_endpoint.json`.
    4. Inside `try:`: uploads minimal repo archive (excluding `.git`, `build/`, `artifacts/`, `.pytest_cache/`, `node_modules/`, `/var/tmp`), installs pinned dependencies, launches `remote_g4_driver.py` detached with `nohup` so Colab WebSocket drops CANNOT kill training, polls `heartbeat.json` every `60 s` until `DONE`/`FAILED`, downloads all JSONs to `artifacts/v4.0.0/` and all blobs to `/var/tmp/vertirta_v400_blobs/`, and verifies every SHA-256 against `remote_bundle_manifest.json`.
    5. Inside `finally:`: ALWAYS executes `colab stop -s vertirta-m3-g4-thunder`, then explicitly invokes `colab_cli.client.Client.unassign(owned_endpoint)` if still present, polls `client.list_assignments()` confirming `owned_endpoint not in remaining`, removes `.active_g4_endpoint.json`, and writes `artifacts/v4.0.0/g4_thunder_run_manifest.json` with teardown proof + CU estimate (`elapsed_hr * 10.05`).

- [ ] **Step 1: Add failing orchestrator, AST anti-dummy-padding & mocked `try/finally` teardown tests to `tests/colab/test_g4_thunder_contracts.py` and `tests/colab/test_g4_jobs_smoke_cpu.py`**
  - Verify AST audit across all 5 `scripts/colab/g4_jobs/*.py` finds `0` dummy VRAM padding calls (`dummy_vram`, `pad_vram`, `reserve_vram`, unreferenced `torch.empty` > 100 MB).
  - Mock `colab_cli` client with 1 pre-existing external endpoint `gpu-g4-external-001` + our newly allocated `gpu-g4-ours-002`; inject a simulated remote crash inside `try:`; verify `finally:` calls `unassign("gpu-g4-ours-002")`, NEVER calls `unassign("gpu-g4-external-001")`, and records `owned_endpoint_released == True`!
  - Execute `run_g4_m3_thunder_run.py --smoke-cpu --out-dir <tmp_path>` locally end-to-end (`0 CU`) -> verify Wave 1 (3 jobs) -> Drain Barrier -> Wave 2 (2 jobs) -> `remote_bundle_manifest.json` SHA-256 verification ALL pass!
- [ ] **Step 2: Run tests to verify failure** -> FAIL.
- [ ] **Step 3: Implement `scripts/colab/remote_g4_driver.py` and `scripts/colab/run_g4_m3_thunder_run.py`**.
- [ ] **Step 4: Run tests to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with scipy --with pyyaml --with torch --with onnx --with pillow --with z3-solver python -m pytest tests/colab/ -v` -> PASS (`0 CU`).

---

### Task 7: Physical `GTX 1650 SUPER` (`TU116`) Single-Context Time-Sliced Edge Gate (`scripts/bench_gtx1650s_m3_timesliced.py`)

**Files:** Create `scripts/bench_gtx1650s_m3_timesliced.py` | Modify `tests/colab/test_g4_jobs_smoke_cpu.py`
**Interfaces:**
- *Consumes:* Read-only helper patterns from `scripts/run_local_gtx1650s_onnx_bench.py` (Win10/WSL2 `cudart` `cudaMemGetInfo`, `nvidia-smi` per-PID VRAM, ORT FP16 conversion + toposort, `CUDAExecutionProvider` + `IoBinding` + CUDA Graphs) + downloaded Phase 3 ONNX binaries in `/var/tmp/vertirta_v400_blobs/` (`pidnet_slz_v21_opset17.onnx`, `mlsd_wireseg_640_opset17.onnx`, `dfine_n_640_opset17.onnx`, `xfeat_512_opset17.onnx`) + SHA-256 manifests in `artifacts/v4.0.0/`.
- *Single-Process Flight-Phase Time-Slicing Architecture (`DR-10` §1.3 + `01-DR08-13` Override `#6`):*
  - **Phase A (Per-Model Empirical FP32-vs-FP16 Gate):** Runs isolated subprocesses for each of the 4 ONNX models in `FP32`, `FP32 + CUDA Graph`, and `FP16` (`200` warmup + `500` timed `IoBinding` iterations), computing exact `mean/p50/p95/p99` ms, `fp16_vs_fp32_speedup_p50`, `cudaMemGetInfo` VRAM delta, and finite non-degenerate output sanity checks (never assuming `2×` FP16 speedup on Turing `TU116`!).
  - **Phase B (Single-Process Time-Sliced Flight Schedule):** Runs ONE single-process CUDA context (`arena_extend_strategy=kSameAsRequested`, **platform-aware `AUTO` cuDNN tuning** — `cudnn_conv_algo_search=EXHAUSTIVE`, `cudnn_conv_use_max_workspace=1`, `enable_cuda_graph=1` on Windows WDDM [matching `v3.5.1`'s `11.695 ms < 12 ms` vs `HEURISTIC`'s `17.53 ms` regression!]; `cudnn_conv_algo_search=HEURISTIC`, `enable_cuda_graph=1` on Linux/WSL2 — plus explicit session eviction & `gc.collect()` at phase transition) stepping:
    1. **Cruise Regime (`30 Hz`, `33.3 ms` frame):** `M-LSD-WireSeg (640×640)` + `D-FINE-N (640×640)` every frame + `XFeat (512×512)` every 30th frame (`1 Hz`), measuring combined frame latency `p50/p95/p99` and peak single-process VRAM.
    2. **Terminal Approach Regime (`10–30 Hz`):** Releases cruise sessions and loads `PIDNet-S v2.1 (512×512)` in the same CUDA context, measuring `p50/p95/p99` and peak single-process VRAM.
  - **Hard Verification Gates Emitted to `artifacts/v4.0.0/local_gtx1650s_m3_timesliced_proof.json`:**
    - `onnx_sha256_all_verified == true`
    - `timesliced_peak_vram_mib < 300.0` (both cruise and terminal phases)
    - `pidnet_v21_p99_ms < 12.0` (Win10 physical `GTX 1650 SUPER`)
    - `cruise_combined_p99_ms < 25.0` (fits inside `33.3 ms` `30 Hz` budget with `>= 8.3 ms` headroom)
    - `empirical_fp32_vs_fp16_logged == true` + per-model `recommended_tu116_precision`
    - `all_trained_outputs_finite_and_nondegenerate == true`
- *`--smoke-cpu` Mode (`0 CU`):* Runs against the 4 ONNX graphs exported in Task 3 & Task 5 `--smoke-cpu` runs using `CPUExecutionProvider`, verifying SHA-256 parity check, time-sliced regime transitions, FP32-vs-FP16 comparison schema (gracefully marking FP16 skipped on CPU), and JSON schema output (`SMOKE_CPU_ONLY`).

- [ ] **Step 1: Add failing test `test_gtx1650s_timesliced_bench_smoke_cpu` to `tests/colab/test_g4_jobs_smoke_cpu.py`** — Chains Task 3 + Task 5 `--smoke-cpu` ONNX outputs into `scripts/bench_gtx1650s_m3_timesliced.py --smoke-cpu`, asserting SHA-256 mismatch fails closed (exit != 0) and matching SHA-256 emits valid `local_gtx1650s_m3_timesliced_proof.json`.
- [ ] **Step 2: Run test to verify failure** -> FAIL.
- [ ] **Step 3: Implement `scripts/bench_gtx1650s_m3_timesliced.py`** — Full Win10 + WSL2 CUDA `IoBinding` + CUDA Graph + `cudaMemGetInfo` time-sliced profiler + `--smoke-cpu` path.
- [ ] **Step 4: Run test to verify pass** — `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with onnx --with onnxruntime --with torch --with pillow python -m pytest tests/colab/ -v` -> PASS.

---

### Task 8: Final Pre-G4 Local Readiness Audit (`0 CU`) & Phase 3 Execution Runbook

**Files:** Verify all Plan 6C files (`0` edits outside owned paths)
- [ ] **Step 1: Run complete `0 CU` local pre-G4 verification suite (ALL MUST PASS before Phase 3):**
  1. `PYTHONPATH=. uv run --no-project --python 3.11 --with pytest --with numpy --with scipy --with pyyaml --with torch --with onnx --with onnxruntime --with pillow --with z3-solver python -m pytest tests/colab/ -v` -> **`100% PASS` (`0 CU`)**.
  2. `python3 scripts/colab/run_g4_m3_thunder_run.py --dry-run` -> exit `0` (`0 CU`, prints exact Wave 1 & Wave 2 commands, VRAM fraction sums `0.88` / `0.88`, license audit PASS, and teardown guard summary).
  3. `python3 scripts/ci/check_arch_isolation.py` -> **`37` files / `0` violations**.
  4. `python3 scripts/ci/check_public_hygiene.py` -> **`0` violations**.
  5. `python3 scripts/ci/check_bare_metal_size.py --target build/arm-cortex-m7/core/flight_core/libvertirta_flight_core.a --size-bin arm-none-eabi-size` -> **`28,886 B Flash / 0 B static RAM`** (byte-frozen).
  6. Pure UTF-8 LF (`CR=0, BOM=False`) check across all 13 Plan 6C files + `HAS_VENV=False`, `HAS_UV_LOCK=False`, `git diff --name-only core/flight_core/ core/rta_companion/ == ""`.
- [ ] **Step 2: Phase 3 Execution Protocol (ONLY after Coordinator approves Phase 2 exit across Plans 6A + 6B + 6C):**
  1. Execute ONCE on WSL2:
     `python3 scripts/colab/run_g4_m3_thunder_run.py --live-g4 --confirm-single-shot-g4 --vertibench-v2-dir benchmarks/vertibench_v2 --out-artifacts artifacts/v4.0.0 --blob-dir /var/tmp/vertirta_v400_blobs`
  2. Verify `artifacts/v4.0.0/g4_thunder_run_manifest.json` proves `owned_endpoint_released: true`, `g4_thunder_telemetry.json` proves real Wave 1 / Wave 2 VRAM utilization (`>= 80 GB` peak without dummy padding), and all 5 job JSONs + SHA-256 blobs exist cleanly (with zero multi-MB blobs inside git!).
  3. Immediately run physical Windows 10 `GTX 1650 SUPER` time-sliced edge proof:
     `uv run --no-project --python 3.11 --with onnx --with onnxruntime-gpu --with numpy --with pillow python scripts/bench_gtx1650s_m3_timesliced.py --blob-dir <staged_blobs> --artifacts-dir artifacts/v4.0.0 --out-json artifacts/v4.0.0/local_gtx1650s_m3_timesliced_proof.json`
  4. Verify `artifacts/v4.0.0/local_gtx1650s_m3_timesliced_proof.json` confirms `timesliced_peak_vram_mib < 300.0`, `pidnet_v21_p99_ms < 12.0`, `cruise_combined_p99_ms < 25.0`, `onnx_sha256_all_verified: true`, and empirical FP32-vs-FP16 telemetry before handing off to Plan 6A/6B Paper 2 & Release `v4.0.0` packaging.
