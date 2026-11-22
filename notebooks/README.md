# VertiRTA — Turnkey Google Colab GPU Notebooks (Cost-Disciplined Edition)

All four `VertiRTA v1.0 / v2.0` cloud GPU jobs are engineered to run cleanly on **Google Colab `L4 (24 GB)`** (with `T4 16 GB` fallback for the 4D HJI-CBVF solver and an optional `A100 40/80 GB` toggle exclusively for full-BF16 `Qwen/Qwen3-8B`).

> **GPU Cost-Discipline Policy:** **NO `VertiRTA v1.0 / v2.0` workload requires a `G4 (RTX PRO 6000 Blackwell, 96 GB)` runtime.** Every notebook includes a startup `nvidia-smi` preflight guard that prints a loud `[COST WARNING]` if a `>= 60 GB` GPU is accidentally attached, saving `3x–5x` in Colab Compute Units with zero loss of numerical fidelity or model accuracy.

---

## GPU-Tier & Compute-Budget Selector Matrix

| Notebook | Deliverable | Primary GPU (Recommended) | Fallback / Optional Tier | Peak VRAM | Est. Wall Time | Est. Budget Tier | Why `G4 (96 GB)` Is NOT Needed |
|---|---|---|---|---:|---:|---|---|
| [`01_colab_jax_4d_cbvf_luts.ipynb`](./01_colab_jax_4d_cbvf_luts.ipynb) | Exact 4D `81x57x81x57` JAX HJI-CBVF `.vrta_lut` (`11,313,864 B`) + 2D CEL `.vrta_cel2d` (`11,230 B`) + DR-2 soundness audit | **`L4 (24 GB)`** | **`T4 (16 GB)`** fallback | `< 6 GB` | `~5 min` (L4) / `~14 min` (T4) | Very Low (`~0.4–0.5 CU`) | Each 4D FP32 regime grid (`21.3M` nodes) is `~85 MB`; WENO5 + TVD-RK3 uses `< 6 GB` peak VRAM. 96 GB sits `> 93%` idle. |
| [`02_colab_istanbul_4k_cinema.ipynb`](./02_colab_istanbul_4k_cinema.ipynb) | ToS-clean 15x15 km Kartal 4K30 Blender 4.5/5.2 LTS Cycles OptiX cinema + RGBA telemetry HUD (`hevc_nvenc -preset p5`) | **`L4 (24 GB)`** | `A100` not needed (no RT cores); avoid `T4 (16 GB)` due to `< 18 GB` budget | `< 18 GB` (hard preflight gate) | `~45–75 min` (12x 150-frame Drive slices) | Low–Medium (`~3.5–6.0 CU`) | Ada Lovelace `L4` has dedicated RT cores + hardware `hevc_nvenc` + 24 GB VRAM (`6+ GB` headroom above the `18 GB` preflight ceiling). |
| [`03_colab_copilot_qwen3_z3_grpo.ipynb`](./03_colab_copilot_qwen3_z3_grpo.ipynb) | Apache-2.0 `Qwen/Qwen3-4B` BF16 LoRA SFT -> Z3-GRPO (`G=8`) + `Q4_K_M` GGUF + KnowNo Conformal + 4-model `VertiBench-TR/EN` eval | **`L4 (24 GB)`** (`Qwen3-4B`) | **`A100 (40/80 GB)`** optional toggle ONLY for `Qwen/Qwen3-8B` BF16 | `~18–21 GB` (`4B`) / `~32–36 GB` (`8B`) | `~50–80 min` (`4B` on L4) | Medium (`~4–7 CU`) | `Qwen/Qwen3-4B` BF16 LoRA (`r=32`) + 64-token compact DSL rollouts fit inside `24 GB` L4; even `8B` BF16 fits comfortably inside `A100 40 GB`. |
| [`04_colab_slz_pidnet_distill_and_egl60.ipynb`](./04_colab_slz_pidnet_distill_and_egl60.ipynb) | Headless Gazebo Harmonic OGRE2 EGL (`llvmpipe`-blocked), 8K schedule, `PIDNet-S` (`DINOv2-ViT-L/14` + `DAv2-Small`) BF16 distillation (`bs=16` on L4 / `bs=32` on `>= 40 GB`), Opset-17 ONNX, zero-shot real eval & 60-seed 3-way EGL ablation (`results/dr7_eval_manifest.json`) | **`L4 (24 GB)`** (`bs=16`, grad-accum `2`) | **`A100 (40/80 GB)`** optional (`bs=32`, ~1.7x faster) | `~15–18 GB` (`bs=16`) / `~26–30 GB` (`bs=32`) | `~90–140 min` (L4) | Medium (`~7–11 CU`) | Frozen teachers (`~0.7 GB` total) + `7.6M`-param `PIDNet-S` at `512x512` BF16 (`bs=16`, effective batch `32` via 2x grad-accum) stay safely under `18 GB` on L4. |

---

## Tier Quick-Reference Summary

- **`T4 (16 GB)`**: Lowest cost (and available on free Colab). Supported as fallback for **Notebook 01 (4D JAX HJI-CBVF)** only. Do **not** use T4 for Blender 4K (`< 18 GB` scene budget) or BF16 LLM/PIDNet training (Turing `sm_75` lacks native hardware `bfloat16`).
- **`L4 (24 GB)`**: **Default sweet spot for all 4 notebooks.** Native Ada Lovelace (`sm_89`) `bfloat16` Tensor Cores, dedicated OptiX ray-tracing cores, hardware `hevc_nvenc`, and 24 GB GDDR6 at a fraction of A100/G4 compute-unit burn.
- **`A100 (40/80 GB)`**: Optional upgrade strictly when enabling `MODEL_ID = "Qwen/Qwen3-8B"` in Notebook 03 or wanting single-step `bs=32` throughput in Notebook 04.
- **`G4 (96 GB Blackwell)`**: Reserved strictly for future `v3.0+` research (e.g., multi-camera 3D Gaussian/NeRF perception or `>= 14B` models). Never required for `v1.0` or `v2.0` reproduction.

---

## Mandatory Safety & License Firewalls (Enforced Inside Every Notebook)

1. **Deterministic Flight-Core Isolation:** Zero notebook artifacts ever alter the zero-heap `libvertirta_flight_core` (`28,886 B` Flash / `0 B` RAM on Cortex-M7). All AI/perception/co-pilot outputs remain advisory (`EASA AI Level 1/2A`) outside the 100 Hz ASTM F3269 safety boundary.
2. **Apache-2.0 / Zero-ShareAlike Weight Hygiene:**
   - **Notebook 02 (Geospatial/Cinema):** Raw AWS Copernicus GLO-30 (`N40_00_E029_00`, `-dstnodata -9999`) + raw AWS `sentinel-cogs` (`35TPF`) warped into Kartal local `tmerc` (`+lat_0=40.886 +lon_0=29.188`). Hard-bans `EPSG:32635`, EOX Sentinel-2 Cloudless, FABDEM, Bing, Esri, and Google tile scraping.
   - **Notebook 03 (Co-Pilot):** Fine-tuning restricted strictly to Apache-2.0 `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`. `Qwen/Qwen2.5-7B-Instruct` and `google/gemma-3-4b-it` are zero-shot baselines only. `CohereForAI/aya-expanse-8b` (`CC-BY-NC-4.0`) is hard-banned.
   - **Notebook 04 (SLZ Perception):** Teachers restricted strictly to Apache-2.0 `DINOv2-ViT-L/14` and `Depth-Anything-V2-Small` (`ViT-S`). Hard-bans `DINOv3`, `Depth-Anything-V2 ViT-B/L/G` (`CC-BY-NC-4.0`), and OSM ODbL footprints in training data (`uses_osm_odbl=False`). `AeroScapes`, `ICG_Semantic_Drone`, and `UAVScenes` are strictly quarantined to zero-shot evaluation.
3. **Resumable Google Drive Persistence:** Every notebook mounts `/content/drive/MyDrive/VertiRTA_Artifacts/` and prints SHA-256 checksums for all produced binaries, weights, and manifests so Colab preemptions never discard completed work.
