# VertiRTA v2.0.0 Active Visual SLZ Perception & Technical Whitepaper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a 100% Apache-2.0, zero-ShareAlike foundation-distilled 3D Safe Landing Zone (SLZ) perception stack (`PIDNet-S` + Free-Energy OOD + cell-local 2.5D Bayesian Grid + 2-phase Active Inspection Orbit) plugged into `ISlzPriorProvider`, verified via zero-shot real benchmarks, a Colab-G4 `N = 60` 3-way Gazebo EGL ablation, a local WSL2 GTX 1650 SUPER smoke test, and a reproducible 6–8 page Zenodo/arXiv LaTeX technical report.

**Architecture:** Offline Colab G4 (`RTX PRO 6000 Blackwell`, 96 GB) generates 8,000 sensor-only Gazebo Harmonic EGL Istanbul frames (strictly isolating OSM `ODbL`) and distills frozen Apache-2.0 `DINOv2-ViT-L/14` + `Depth Anything V2 Small` teachers into static-PAPPM `PIDNet-S` (`MIT`, `512×512`, `~189 MB` FP16 VRAM) exported to ONNX Opset 17+ / TRT FP16. At runtime (`20–30+ Hz`), RAII C++20 `SlzOrtRunner` feeds a `1.5625 MiB` cell-local 2.5D Bayesian SLZ belief grid (`160×160 @ 0.5 m`) with an explicit OOD variance floor and a deterministic 5-state `ActiveInspectionPolicy` implementing Plan 1A's `ISlzPriorProvider` port plus a decoupled V2.1 `IForwardHazardProvider` stub.

**Tech Stack:** Python 3.11, PyTorch 2.x (BF16 AMP), `onnx`/`onnxsim`, ONNX Runtime C++20 `1.21–1.27+` (`CUDA`/`TensorRT` V2 EPs), Gazebo Harmonic (`gz-sim8`, `gz-transport13`, `gz-msgs10`, OGRE2 NVIDIA EGL), C++20 (`vertirta::rta_companion`, `vertirta::slz_bayes`, GoogleTest), ROS 2 Jazzy (`vertirta_perception_node`, `vertirta_msgs`, `px4_msgs`), LaTeX (`latexmk`).

**Spec:** `docs/superpowers/specs/2026-11-22-vertirta-tier-s-design.md` (v1.2 FROZEN) and `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-7` errata overrides).

## Global Constraints

- **License Lock (Zero-ShareAlike, Zero-NC, Defense-Safe):** All published code & weights are `Apache-2.0`. Train exclusively on **`FLAIR-1` (`etalab-2.0`) + `OpenEarthMap` 41/100 programmatic whitelist (`CC BY 4.0` + `DL-DE-BY-2.0` ONLY) + `FloodNet-Supervised v1.0` (`CDLA-Permissive-1.0`) + `SkyScenes` (`MIT`) + Author 8K Gazebo-Istanbul (Copernicus GLO-30 procedural blocks; OSM `ODbL` strictly isolated)**. Hard-ban `RescueNet` (`CC BY-NC-ND`), `LoveDA` (`CC BY-NC-SA`/GE-ToS), `DroneDeploy`, `FlyAwareV2` (`GPL-3.0`), `SegFormer` (NVIDIA NC), `DINOv3` (ITAR/military ban), `Depth Anything V2 ViT-B/L/G` (`CC-BY-NC`), `Mid-Air`, and `UAVid`. Quarantine `AeroScapes` (`CC BY-SA 4.0`), `ICG` (`NC`), and `UAVScenes` (`CC BY-NC-SA 4.0`) strictly to **EVAL-ONLY**.
- **Vehicle & SLZ Contract (`config/vehicle_x8_240.yaml`):** `VertiRTA-X8-240` (`m = 240.0 kg`, `g = 9.80665 m/s²`, `W = 2,353.6 N`, `R = 0.75 m` → `A_disk = 7.069 m²`, `400 V / 22 kWh` pack — NEVER `13.2 kWh` or `R = 0.80 m`!). Hover electrical power with corrected Leishman coaxial factor `~40.5 kW` (`kBaseSoc12s = 0.614%` SoC per 12 s). SLZ footprint `R_SLZ >= 5.0 m` (`317` cells @ `0.5 m`), slope `theta <= 8.0°`, roughness `sigma_z <= 0.15 m`, step `step_max <= 0.25 m`.
- **Geodetic & Port Locks:** Kartal Coastal Vertiport ENU datum `40.8860° N, 29.1880° E, 12.0 m AMSL` (NOT `40.8880° N`), `EGM2008 N = +36.7 m`, `LTFJ` ARP `40.89833° N, 29.30917° E` (west of `LTFJ` `J2` / `LTBX` CTR). Foxglove WS `:8891`, CesiumJS `:5180`, `MicroXRCEAgent` `:8888`; never bind `:4173, :4174, :8765, :9222, :9223`.
- **Hard Bans & Hardware Split:** Zero MC-Dropout, zero Dense EDL, zero online POMDP/RL control, zero `TTPLA` wires crammed into `512×512` nadir SLZ, zero PX4/ROS 2 in Colab G4. Local WSL2 GPU is **`GTX 1650 SUPER` (`TU116`, 4 GB VRAM, `<300 MB` budget, 2× FP16 ALUs, no Tensor Cores)** — benchmark `ORT-CUDA FP32` vs `FP16`.
- **Regulatory & Metrics Honesty:** Whitepaper NEVER claims DO-178C/SC-VTOL/SORA compliance, NEVER extrapolates `10^-9/FH` from Monte Carlo, keeps CSFL vs Same-Arm CEL pools separate, and auto-generates every reported number from frozen JSON manifests.

## Review Focus

1. **Silent License / Mask Contamination (Task 1):** Unlisted `OpenEarthMap` regions (auto `CC BY-NC-SA`), eval-only sets passed with `split="train"`, OSM `ODbL` in training, or oblique `SkyScenes` frames lacking depth must raise explicit errors rather than tainting weights or mislabeling facades as `Safe_Flat_Roof`.
2. **Global-ENU `float32` Cancellation & Sparse Cells (Task 4):** World-frame `Σ = M/N - p̄p̄ᵀ` in `float32` at Maltepe (`±3,500 m`) yields negative/NaN eigenvalues, and `<12` points can report `0°` flat; accumulate cell-local `δp = p_W - c_ij` and mark sparse cells invalid.
3. **OOD Precision Collapse on Newly Appearing Tarps (Task 4):** Purely additive `Λ_k` cannot raise `σ²_cell` when an OOD tarp appears on a previously clean roof and still collapses `<0.08` over repeated OOD frames; enforce an explicit OOD variance floor + `Λ_k` cap/reset when `E_tilde > τ_ood`.
4. **Cruise-to-Orbit Timeout, Clock Underflow, Silent Rejects & Pad Clipping (Tasks 2 & 5):** Arming the `12.0 s` orbit timer at `80 m` AGL guarantees timeout during the `15–19 s` descent, `now_ns < orbit_start_ns_` underflows `uint64_t`, omitted `step_max_m`/`clearance_r_m` bits yield `REASON_NONE (0u)`, and a fixed nadir `80°` HFOV camera clips the far edge of a `5 m` pad (`42.3° > 40°`); require `APPROACH_ORBIT`, guarded timestamps, complete reason bits, and a tilted 2-axis gimbal.
5. **Silent `llvmpipe` Fallback, Static-Rig Stale Poses & Paper Drift (Tasks 2, 7 & 8):** Colab EGL silently degrades to CPU `llvmpipe` (`gz-sim#1272`), `set_pose/blocking` on `<static>true</static>` rigs without physics leaves frames frozen, and hand-typed paper tables drift; gate all three with automated checks.

---

## File Structure

- `ml/slz_perception/data/` — `taxonomy.json`, `license_gate.py` (ban/quarantine + OEM 41/100 parser), `remappers.py` (5-class + SkyScenes wall filter + UAVScenes `cmap.py` guard), `generate_istanbul_8k.py` (stratified 8K generator).
- `ml/slz_perception/models/` & `ml/slz_perception/train/` — `pidnet_slz.py` (static-PAPPM `PIDNet-S` + normal head + train-only DINOv2 projector), `distill_losses.py`, `train_slz_distill.py` (2-stage Colab G4 BF16), `calibrate_ood.py`, `export_onnx_trt.py` (static `1×3×512×512` Opset 17+).
- `ml/slz_perception/eval/` — `eval_zeroshot_real.py` (`AeroScapes`/`ICG`/`UAVScenes`), `aggregate_ablation_tables.py` (Gate 5 checker + `docs/paper/metrics_manifest.json` merger).
- `sim/colab_egl/` — `colab_g4_gz_egl_setup.sh`, `istanbul_slz_egl.sdf` (kinematic gimbal rig), `include/vertirta/sim/gz_egl_bridge.hpp`, `src/gz_egl_bridge.cpp`, `src/active_vision_ablation_runner.cpp`, `CMakeLists.txt`.
- `core/slz_bayes/` — `slz_ort_engine.{hpp,cpp}` (RAII CUDA stream + 8× `bdry_prob` upsample), `slz_geometry.{hpp,cpp}` (Mod A `gz-sensors#545` + Mod B affine), `slz_belief_grid.{hpp,cpp}` (`1.5625 MiB` cell-local grid + OOD floor), `visual_slz_prior_provider.hpp` (implements Plan 1A's `vertirta::rta::ISlzPriorProvider`), `active_inspection_policy.{hpp,cpp}` (5-state orbit + `IForwardHazardProvider`), `CMakeLists.txt` (`vertirta_slz_bayes` / `ALIAS vertirta::slz_bayes`, auto-discovered by Plan 1A root `CMakeLists.txt`).
- `ros2_ws/src/vertirta_perception_node/` & `scripts/benchmark_ort_gtx1650s.sh` — 20–30+ Hz ROS 2 Jazzy node + local WSL2 GTX 1650 SUPER FP32/FP16 smoke harness.
- `docs/paper/` & `.zenodo.json` — `vertirta_tech_report.tex`, `references.bib`, `scripts/generate_latex_tables.py`, `scripts/verify_paper_numbers.py`.

---

### Task 1: Zero-ShareAlike 5-Source License Gate & Canonical 5-Class Taxonomy

**Files:** Create `ml/slz_perception/data/{taxonomy.json,license_gate.py,remappers.py}`; Test `ml/slz_perception/tests/test_license_and_taxonomy.py`.
**Interfaces:**
- Consumes: `docs/research/00-DR-FORENSIC-ERRATA-AND-LOCKS.md` (`DR-7` license/class locks).
- Produces: `verify_training_manifest(entries: list[DatasetEntry], oem_attribution_html: Path) -> ManifestReport` (raises `LicenseViolationError`), `parse_oem_permissive_whitelist(attribution_html: Path, expect_total: int = 100) -> set[str]`, `remap_mask(dataset: str, mask: np.ndarray, *, depth_m=None, cam_rot_wc=None, intrinsics=None) -> np.ndarray`, `assert_uavscenes_cmap_ids(cmap_dict: dict[int, str]) -> None`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np, pytest
from ml.slz_perception.data.license_gate import DatasetEntry, LicenseViolationError, parse_oem_permissive_whitelist, verify_training_manifest
from ml.slz_perception.data.remappers import remap_mask, assert_uavscenes_cmap_ids

@pytest.mark.parametrize("name", ["RescueNet", "LoveDA", "DroneDeploy", "FlyAwareV2", "DINOv3", "SegFormer-B0", "Mid-Air", "UAVid", "Depth-Anything-V2-Large", "AeroScapes", "ICG_Semantic_Drone", "UAVScenes"])
def test_banned_and_eval_only_blocked_from_training(name, tmp_path):
    with pytest.raises(LicenseViolationError):
        verify_training_manifest([DatasetEntry(name=name, split="train", license="CC-BY-SA-4.0", uses_osm_odbl=False)], tmp_path / "attr.html")

def test_oem_whitelist_excludes_public_domain_blank_and_osm_odbl(tmp_path):
    html = tmp_path / "attr.html"
    html.write_text("<tr><td>christchurch</td><td>CC BY 4.0</td></tr><tr><td>aachen</td><td>DL-DE-BY-2.0</td></tr>"
                    "<tr><td>austin</td><td>Public Domain</td></tr><tr><td>tokyo</td><td></td></tr><tr><td>paris</td><td>CC BY-SA 4.0</td></tr>")
    assert parse_oem_permissive_whitelist(html, expect_total=5) == {"christchurch", "aachen"}
    with pytest.raises(LicenseViolationError):
        verify_training_manifest([DatasetEntry(name="Gazebo_Istanbul_8K", split="train", license="Apache-2.0", uses_osm_odbl=True)], html)

def test_skyscenes_wall_filter_canonical_maps_and_uavscenes_guard():
    mask = np.ones((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        remap_mask("SkyScenes", mask)
    rot_wc = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)  # vertical wall => |n_z| == 0 < 0.85
    assert np.all(remap_mask("SkyScenes", mask, depth_m=np.full((4, 4), 10.0, np.float32), cam_rot_wc=rot_wc, intrinsics=(305.09, 305.09, 2.0, 2.0))[1:3, 1:3] == 2)
    assert remap_mask("FLAIR-1", np.array([[1, 2, 5, 14, 18, 19]], np.uint8)).tolist() == [[1, 0, 4, 255, 2, 255]]
    assert remap_mask("FloodNet-Supervised-v1.0", np.array([[0, 1, 2, 4, 7, 9]], np.uint8)).tolist() == [[255, 2, 1, 0, 3, 4]]
    with pytest.raises(AssertionError):
        assert_uavscenes_cmap_ids({1: "road"})
```

- [ ] **Step 2: Run test to verify it fails** — `pytest ml/slz_perception/tests/test_license_and_taxonomy.py -v` (Expected: FAIL `ModuleNotFoundError`).
- [ ] **Step 3: Implement `taxonomy.json`, `license_gate.py`, and `remappers.py`** — Lock the 5 train + 3 eval-only maps (`{0..4, 254, 255}`), parse `attribution.html` keeping ONLY `CC BY 4.0` + `DL-DE-BY-2.0` (asserting 41/100 on official HTML), block OSM `ODbL` in training, and compute finite-difference world normals on `SkyScenes` `building` pixels (`|n_z| < 0.85 → 2`).
- [ ] **Step 4: Run test to verify it passes** — `pytest ml/slz_perception/tests/test_license_and_taxonomy.py -v` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add ml/slz_perception/data/ ml/slz_perception/tests/test_license_and_taxonomy.py && git commit -m "feat(slz): add zero-ShareAlike license gate, OEM whitelist parser, and 5-class taxonomy"`

---

### Task 2: Sensor-Only Colab G4 Gazebo Harmonic EGL Rig (2-Axis Gimbal) & 8K Istanbul Generator

**Files:** Create `sim/colab_egl/{colab_g4_gz_egl_setup.sh,istanbul_slz_egl.sdf,CMakeLists.txt,include/vertirta/sim/gz_egl_bridge.hpp,src/gz_egl_bridge.cpp}`, `ml/slz_perception/data/generate_istanbul_8k.py`; Test `sim/colab_egl/tests/test_gz_egl_bridge.cpp`, `ml/slz_perception/tests/test_istanbul_8k_manifest.py`.
**Interfaces:**
- Consumes: `config/istanbul_ltfj_airspace.yaml` (Plan 1A: Kartal ENU `40.8860° N, 29.1880° E, 12.0 m AMSL`, Copernicus GLO-30 DEM), `license_gate.py` (Task 1).
- Produces: `vertirta::sim::ComputeGimbalQuaternionEnu(double pitch_down_rad, double yaw_rad) -> QuatWxyz`, `vertirta::sim::GzEglSensorBridge` (`Init`, `StepSensorRigPose`, `CopyLatestRgbd` requiring `|t_rgb - t_depth| <= 2 ms`), `generate_sampling_schedule(n_frames=8000, seed=42) -> list[FrameSpec]`, and `colab_g4_gz_egl_setup.sh` (`<2 min` `.tar.zst` Drive cache + `gz-sim#1272` `llvmpipe` guard).

- [ ] **Step 1: Write the failing tests**

```python
# ml/slz_perception/tests/test_istanbul_8k_manifest.py
import math, subprocess
from pathlib import Path
from ml.slz_perception.data.generate_istanbul_8k import generate_sampling_schedule

def test_8k_schedule_stratification_and_egl_kinematic_gimbal():
    sched = generate_sampling_schedule(8000, seed=42)
    assert len(sched) == 8000 and all(not f.uses_osm_odbl and f.datum_lat == 40.8860 and f.datum_lon == 29.1880 for f in sched)
    assert abs(sum(15.0 <= f.agl_m < 30.0 for f in sched) / 8000 - 0.35) < 0.01
    assert abs(sum(30.0 <= f.agl_m < 65.0 for f in sched) / 8000 - 0.40) < 0.01
    assert abs(sum(f.inject_ood_254 for f in sched) / 8000 - 0.15) < 0.01
    sh = Path("sim/colab_egl/colab_g4_gz_egl_setup.sh").read_text()
    assert "](" not in sh and "llvmpipe" in sh and "libEGL_nvidia.so.0" in sh
    assert subprocess.run(["bash", "-n", "sim/colab_egl/colab_g4_gz_egl_setup.sh"]).returncode == 0
    sdf = Path("sim/colab_egl/istanbul_slz_egl.sdf").read_text()
    assert "<static>false</static>" in sdf and "<kinematic>true</kinematic>" in sdf and "gz-sim-label-system" in sdf
    assert math.degrees(math.atan2(20.0, 22.0) - math.atan2(15.0, 22.0)) < math.degrees(1.396263 / 2.0) - 25.0
```

```cpp
// sim/colab_egl/tests/test_gz_egl_bridge.cpp
TEST(GzEglBridgeTest, GimbalQuaternionNormalizedAndBridgeRejectsUnpairedFrames) {
  auto q = vertirta::sim::ComputeGimbalQuaternionEnu(/*pitch_down_rad=*/0.9721, /*yaw_rad=*/0.0);
  EXPECT_NEAR(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z, 1.0, 1e-9);
  vertirta::sim::GzEglSensorBridge bridge;
  std::vector<uint8_t> rgb; std::vector<float> depth; uint64_t ts = 0;
  EXPECT_FALSE(bridge.CopyLatestRgbd(rgb, depth, ts));
}
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest ml/slz_perception/tests/test_istanbul_8k_manifest.py -v` (Expected: FAIL).
- [ ] **Step 3: Implement `colab_g4_gz_egl_setup.sh`, `istanbul_slz_egl.sdf`, `GzEglSensorBridge`, and `generate_istanbul_8k.py`** — Clean bash (zero mangled `||`/markdown URLs), zero PX4/ROS 2, `ldd libEGL_nvidia.so.0` + `eglinfo -B` + `ogre2.log` `llvmpipe` check. Configure `vertirta_sensor_rig` as `<static>false</static><gravity>false</gravity><kinematic>true</kinematic>` updated via `/world/istanbul_coast/set_pose/blocking`, subscribing to `/slz_rgbd/image` + `/slz_rgbd/depth_image` (bypassing `gz-sensors#545` PointCloud2), with `gz-sim-label-system` (`10/20/30/40/50/254`) on 12 Copernicus-GLO-30 procedural coastal blocks (zero OSM `ODbL`), pitch `0–35°`, quadratic depth noise `σ_Z = 0.002 Z²`.
- [ ] **Step 4: Run tests to verify they pass** — `pytest ml/slz_perception/tests/test_istanbul_8k_manifest.py -v && ctest --test-dir build/sim/colab_egl --output-on-failure` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add sim/colab_egl/ ml/slz_perception/data/generate_istanbul_8k.py ml/slz_perception/tests/test_istanbul_8k_manifest.py && git commit -m "feat(slz-sim): add sensor-only Colab-G4 EGL setup, kinematic gimbal bridge, and 8K generator"`

---

### Task 3: `PIDNet-S` (MIT) Static-PAPPM Student, Colab G4 2-Stage Distillation & ONNX/TRT FP16 Export

**Files:** Create `ml/slz_perception/models/pidnet_slz.py`, `ml/slz_perception/train/{distill_losses.py,train_slz_distill.py,calibrate_ood.py,export_onnx_trt.py}`; Test `ml/slz_perception/tests/test_pidnet_export_and_losses.py`.
**Interfaces:**
- Consumes: `license_gate.py` & `remappers.py` (Task 1), 8K Gazebo-Istanbul dataset (Task 2).
- Produces: `PIDNetSlzTrain(num_classes=5, dino_dim=1024)`, `PIDNetSlzDeployWrapper` returning `(seg_logits[1,5,512,512], surf_normals[1,3,512,512], bdry_prob[1,1,64,64])`, `SlzDistillCriterion`, `calibrate_temperature_and_tau(...) -> OodCalibration` (`ood_calibration.json`), `export_static_onnx(model, out_path, opset=17) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
import onnx, torch
from ml.slz_perception.models.pidnet_slz import PIDNetSlzTrain, PIDNetSlzDeployWrapper
from ml.slz_perception.train.distill_losses import SlzDistillCriterion
from ml.slz_perception.train.calibrate_ood import calibrate_temperature_and_tau
from ml.slz_perception.train.export_onnx_trt import export_static_onnx

def test_pidnet_static_pappm_onnx_export_and_losses(tmp_path):
    net = PIDNetSlzTrain(num_classes=5, dino_dim=1024).eval()
    assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in net.modules())
    deploy = PIDNetSlzDeployWrapper(net).eval()
    logits, normals, bdry = deploy(torch.randn(1, 3, 512, 512))
    assert logits.shape == (1, 5, 512, 512) and normals.shape == (1, 3, 512, 512) and bdry.shape == (1, 1, 64, 64)
    assert torch.allclose(torch.linalg.norm(normals, dim=1), torch.ones(1, 512, 512), atol=1e-4)
    m = onnx.load(str(export_static_onnx(deploy, tmp_path / "pidnet_slz_512.onnx", opset=17)))
    onnx.checker.check_model(m)
    assert [i.name for i in m.graph.input] == ["rgb_nchw"] and [o.name for o in m.graph.output] == ["seg_logits", "surf_normals", "bdry_prob"]
    assert not any("dino_proj" in n.name for n in m.graph.node)

    crit = SlzDistillCriterion(0.4, 0.2, 0.2, 0.1, m_in=-23.0, m_out=-5.0)
    lbl = torch.randint(0, 5, (2, 64, 64)); lbl[:, :8, :8] = 254; lbl[:, 8:12, :8] = 255
    loss = crit(torch.randn(2, 5, 64, 64, requires_grad=True), torch.nn.functional.normalize(torch.randn(2, 3, 64, 64), dim=1),
                torch.sigmoid(torch.randn(2, 1, 64, 64)), torch.randn(2, 1024, 37, 37), lbl, torch.ones(2, 3, 64, 64), torch.zeros(2, 1, 64, 64), torch.randn(2, 1024, 37, 37))
    assert torch.isfinite(loss) and loss.item() > 0.0

    cal = calibrate_temperature_and_tau(torch.randn(1000, 5, generator=torch.Generator().manual_seed(0)) * 3.0, torch.randint(0, 5, (1000,)), torch.zeros(1000))
    assert cal.T_ood == 1.0 and 0.5 <= cal.T_prob <= 3.0 and cal.kappa_b == 1.5
```

- [ ] **Step 2: Run test to verify it fails** — `pytest ml/slz_perception/tests/test_pidnet_export_and_losses.py -v` (Expected: FAIL).
- [ ] **Step 3: Implement `pidnet_slz.py`, `distill_losses.py`, `train_slz_distill.py`, `calibrate_ood.py`, and `export_onnx_trt.py`** — Replace PAPPM `AdaptiveAvgPool2d` with fixed `AvgPool2d` + `mean(dim=(2,3), keepdim=True)` (~189 MB FP16 VRAM). Run `verify_training_manifest` at startup; Stage-1 (80 ep, BF16, `bs=32`, AdamW `3e-4 → 1e-6`, 60% real `FLAIR-1 25% / OEM-41 20% / FloodNet 15%` + 40% synth `SkyScenes 20% / Gazebo-8K 20%`) distills frozen Apache-2.0 `DINOv2-ViT-L/14` (`518×518 → 37×37`) + `DAv2-Small` pseudo-normals (masking sky + top-10% loss) + 1.5× hazard-weighted CE/Lovasz; Stage-2 (20 ep, `lr=1e-5`) adds `label=254` + 10% AnomalyMix OE (`m_in=-23.0, m_out=-5.0`), post-hoc NLL `T_prob*` + 95%-ID-TPR `τ_ood`, and `onnxsim` Opset 17+ export dropping `dino_proj`.
- [ ] **Step 4: Run test to verify it passes** — `pytest ml/slz_perception/tests/test_pidnet_export_and_losses.py -v` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add ml/slz_perception/models/ ml/slz_perception/train/ ml/slz_perception/tests/test_pidnet_export_and_losses.py && git commit -m "feat(slz-train): add static-PAPPM PIDNet-S, DINOv2+DAv2-S distillation, and ONNX Opset 17 export"`

---

### Task 4: C++20 `SlzOrtRunner` (RAII Stream + 8× Boundary Upsample), Dual-Mode Geometry & Cell-Local 2.5D Bayesian Grid

**Files:** Create `core/slz_bayes/{CMakeLists.txt,include/vertirta/perception/{slz_ort_engine.hpp,slz_geometry.hpp,slz_belief_grid.hpp},src/{slz_ort_engine.cpp,slz_geometry.cpp,slz_belief_grid.cpp}}`; Test `core/slz_bayes/tests/test_slz_bayes_grid.cpp`.
**Interfaces:**
- Consumes: Plan 1A `vertirta::flight_core` (`vertirta::Vec3Enu`) & `vertirta::rta_companion` (`vertirta::rta::SlzPrior`), `pidnet_slz_512.onnx` & `ood_calibration.json` (Task 3).
- Produces:
  - CMake target `vertirta_slz_bayes` (`ALIAS vertirta::slz_bayes`), `SlzOrtRunner` (non-copyable RAII destroying `session_` before `cudaStreamDestroy(stream_)`; `Run(...) noexcept -> bool`) and `ApplyUpsampledBoundaryCompensation(span<const float> e_raw_512, span<const float> bdry_64, float kappa_b, span<float> e_out_512) noexcept`.
  - `BackprojectDepthRgbd(...)` (Mod A raw `depth_image` bypassing `gz-sensors#545`) and `AnchorMonocularDepthAffine(span<const float> disp, const Vec3Enu& opt_axis_enu, float agl_m, std::optional<float> rng_m, std::optional<float> far_m, span<float> depth_out) noexcept -> float` (roll/pitch slant `z_AGL / (-d_Wz)`, zero 90%-FOV Metric3D crop).
  - `SlzCell` (`alignas(64)`, `64 B`), `CountCircularDiskCells(float r_m, float res_m) -> uint32_t`, `SlzBeliefGrid` (`160×160 @ 0.5 m` = `1.5625 MiB` static ring buffer, `UpdateFrame(...) noexcept`, `EvaluateDisk(const Vec3Enu&, float r_m = 5.0f) const noexcept -> vertirta::rta::SlzPrior`).

- [ ] **Step 1: Write the failing test**

```cpp
// core/slz_bayes/tests/test_slz_bayes_grid.cpp
TEST(SlzBeliefGridTest, BudgetCellLocalCovarianceOodFloorSparseAndUpsample) {
  static_assert(sizeof(vertirta::perception::SlzCell) == 64);
  EXPECT_EQ(sizeof(vertirta::perception::SlzCell) * 160u * 160u, 1638400u);  // 1.5625 MiB
  EXPECT_EQ(vertirta::perception::CountCircularDiskCells(5.0f, 0.5f), 317u);

  // 1) Cell-local covariance avoids float32 cancellation at Maltepe (-3500 m, +3200 m)
  vertirta::perception::SlzBeliefGrid grid;
  FeedSyntheticSlopedPlane(grid, {-3500.0, 3200.0, 15.0}, /*slope_deg=*/6.0f, /*roughness=*/0.02f);
  auto c = grid.EvaluateDisk({-3500.0, 3200.0, 15.0}, 5.0f);
  EXPECT_TRUE(std::isfinite(c.theta_max_deg) && std::isfinite(c.sigma_z_m));
  EXPECT_NEAR(c.theta_max_deg, 6.0f, 0.35f); EXPECT_LT(c.sigma_z_m, 0.04f);

  // 2) Newly appearing OOD tarp must raise slz_ood_variance >= 0.08 even after 40 clean frames & 25 OOD frames
  vertirta::perception::SlzBeliefGrid ood_grid;
  for (int k = 0; k < 40; ++k) FeedCleanRoofFrame(ood_grid, k * 0.05, -12.0f);
  ASSERT_LT(ood_grid.EvaluateDisk({0, 0, 12}, 5.0f).slz_ood_variance, 0.06f);
  for (int k = 0; k < 25; ++k) FeedOodTarpFrame(ood_grid, 2.0 + k * 0.05, +4.0f);
  EXPECT_GE(ood_grid.EvaluateDisk({0, 0, 12}, 5.0f).slz_ood_variance, 0.08f);

  // 3) Sparse <12 pts stays conservative; Mod B affine uses z_AGL/(-d_Wz); 64x64->512x512 upsample
  vertirta::perception::SlzBeliefGrid sparse; FeedSparsePoints(sparse, 6);
  EXPECT_GT(sparse.EvaluateDisk({0, 0, 0}, 5.0f).theta_max_deg, 80.0f);
  std::array<float, 16> disp, depth{}; disp.fill(1.0f);
  vertirta::Vec3Enu opt{0.45, 0.20, -std::sqrt(1.0 - 0.45*0.45 - 0.20*0.20)};
  EXPECT_GT(vertirta::perception::AnchorMonocularDepthAffine(disp, opt, 22.0f, std::nullopt, std::nullopt, depth), 0.04f);
  EXPECT_NEAR(depth[0], 22.0f / static_cast<float>(-opt.z), 0.1f);
  std::vector<float> e_raw(512*512, -4.0f), b64(64*64, 1.0f), e_out(512*512, 0.0f);
  vertirta::perception::ApplyUpsampledBoundaryCompensation(e_raw, b64, 1.5f, e_out);
  EXPECT_NEAR(e_out[512*511 + 511], -5.5f, 1e-5f);
}
```

- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build/core/slz_bayes -R test_slz_bayes_grid --output-on-failure` (Expected: FAIL).
- [ ] **Step 3: Implement `SlzOrtRunner`, `SlzGeometry`, and `SlzBeliefGrid`** — Store cell-local shifted moments `δp = p_W - c_ij` inside `64 B` `SlzCell` (`s[3], M[6], N, L, h_dyn, lambda_prec, ood_ema, last_ts_ms`) with 3×3 analytic Jacobi eigendecomposition (`>=12` pts), 2× hazard log-odds (`τ_static=30 s`), `τ_dyn=5 s` decay, and **OOD variance floor + `Λ_k` cap/reset** (`σ²_cell = max(1/max(Λ_k, Λ_min), σ²_ood_floor(ood_ema))` when `E_tilde > τ_ood`). Aggregate `317` disk cells via `Q05(P_safe)`, `Q95(θ)`, `Q90(σ²)`, `max(H_dyn)`, `step_max_m`. Configure ORT TRT/CUDA V2 EPs via `Update*ProviderOptionsWithValue("user_compute_stream", stream_)` with zero-leak destructor and bilinear 8× `bdry_prob` upsampling.
- [ ] **Step 4: Run test to verify it passes** — `cmake --build build --target test_slz_bayes_grid && ctest --test-dir build/core/slz_bayes -R test_slz_bayes_grid --output-on-failure` (Expected: PASS under ASan/UBSan, 0 heap allocs in hot loop).
- [ ] **Step 5: Commit** — `git add core/slz_bayes/ && git commit -m "feat(slz-core): add RAII SlzOrtRunner, cell-local 2.5D Bayesian grid, and OOD variance floor"`

---

### Task 5: Deterministic 2-Phase Active Inspection Orbit (`ISlzPriorProvider`) & V2.1 `IForwardHazardProvider` Stub

**Files:** Modify `core/slz_bayes/CMakeLists.txt`; Create `core/slz_bayes/include/vertirta/perception/{visual_slz_prior_provider.hpp,active_inspection_policy.hpp}`, `core/slz_bayes/src/active_inspection_policy.cpp`; Test `core/slz_bayes/tests/test_active_inspection_policy.cpp`.
**Interfaces:**
- Consumes: Plan 1A canonical `vertirta::rta::ISlzPriorProvider`, `SlzPrior`, and `SlzProviderSnapshot` (`core/rta_companion/include/vertirta/rta/slz_prior_provider.hpp`), `vertirta::core::FlightRegime`, and `SlzBeliefGrid` (Task 4).
- Produces: `SlzDecisionState::{CANDIDATE_EVAL, APPROACH_ORBIT, ACTIVE_INSPECT_ORBIT, COMMIT_TOUCHDOWN, REJECT_DIVERT_NEXT}`, `SlzReasonBits` (including `REASON_STEP_EXCEEDED = 1u<<9`, `REASON_CLEARANCE_INSUFF = 1u<<10`), `ActiveInspectSetpoint` (with `gimbal_pitch_rad`, `gimbal_yaw_rad`), `ActiveInspectionPolicy::{Step, EstimateOrbitDeltaSocPct, Reset}`, `VisualSlzPriorProvider : public vertirta::rta::ISlzPriorProvider` (returning `vertirta::rta::SlzProviderSnapshot`), and decoupled V2.1 `IForwardHazardProvider::QueryForwardHazards(uint64_t now_ns) const noexcept -> std::span<const ForwardWireHazard>`.

- [ ] **Step 1: Write the failing test**

```cpp
// core/slz_bayes/tests/test_active_inspection_policy.cpp
TEST(ActiveInspectionPolicyTest, Energy22kWhApproachOrbitTimerReasonsMonoVetoAndUnderflow) {
  using vertirta::core::FlightRegime;
  vertirta::perception::ActiveInspectionPolicy pol;
  // 1) Locked 22 kWh, R=0.75 m => ~40.5 kW => 0.614% * 1.25 = 0.768% (NEVER 13.2 kWh / 1.36%)
  float nom = pol.EstimateOrbitDeltaSocPct({50.f, 20.f, 0.f, FlightRegime::NOMINAL});
  EXPECT_NEAR(nom, 0.614f * 1.25f, 0.03f);
  EXPECT_LT(pol.EstimateOrbitDeltaSocPct({50.f, 20.f, 10.f, FlightRegime::CROSS_DUAL}), 1.35f);

  // 2) APPROACH_ORBIT absorbs 16 s descent from 80->22 m BEFORE starting 12 s orbit timer
  auto unc = MakeUncertainRoofPrior(/*slz_ood_variance=*/0.30f);
  EXPECT_EQ(pol.Step(unc, {40,0,92}, 80.f, DepthMode::RGBD_PRIMARY, {50,20,5,FlightRegime::NOMINAL}, 1'000'000'000ULL).state, SlzDecisionState::APPROACH_ORBIT);
  auto s1 = pol.Step(unc, {16,0,36}, 24.f, DepthMode::RGBD_PRIMARY, {49,20,5,FlightRegime::NOMINAL}, 17'000'000'000ULL);
  EXPECT_EQ(s1.state, SlzDecisionState::ACTIVE_INSPECT_ORBIT);
  EXPECT_NEAR(s1.inspect_setpoint->elapsed_s, 0.0f, 1e-3f);
  EXPECT_NEAR(s1.inspect_setpoint->gimbal_pitch_rad, std::atan2(22.0f, 15.0f), 0.05f);
  // Backwards timestamp inside ACTIVE_INSPECT_ORBIT must NOT underflow into timeout
  EXPECT_EQ(pol.Step(unc, {15,0,34}, 22.f, DepthMode::RGBD_PRIMARY, {49,20,5,FlightRegime::NOMINAL}, 16'000'000'000ULL).state, SlzDecisionState::ACTIVE_INSPECT_ORBIT);
  EXPECT_EQ(pol.Step(unc, {0,15,34}, 22.f, DepthMode::RGBD_PRIMARY, {48,20,5,FlightRegime::NOMINAL}, 28'500'000'000ULL).state, SlzDecisionState::ACTIVE_INSPECT_ORBIT);
  auto tout = pol.Step(unc, {-15,0,34}, 22.f, DepthMode::RGBD_PRIMARY, {48,20,5,FlightRegime::NOMINAL}, 29'100'000'000ULL);
  EXPECT_EQ(tout.state, SlzDecisionState::REJECT_DIVERT_NEXT); EXPECT_NE(tout.reason_mask & REASON_ORBIT_TIMEOUT, 0u);

  // 3) Step/clearance reject bits never return REASON_NONE; Mod B > 30 m vetoes commit
  auto parapet = MakeCleanRoofPrior(); parapet.step_max_m = 0.32f;
  EXPECT_NE(pol.Step(parapet, {0,0,34}, 22.f, DepthMode::RGBD_PRIMARY, {50,20,0,FlightRegime::NOMINAL}, 30'000'000'000ULL).reason_mask & REASON_STEP_EXCEEDED, 0u);
  auto tight = MakeCleanRoofPrior(); tight.clearance_r_m = 4.2f;
  EXPECT_NE(pol.Step(tight, {0,0,34}, 22.f, DepthMode::RGBD_PRIMARY, {50,20,0,FlightRegime::NOMINAL}, 31'000'000'000ULL).reason_mask & REASON_CLEARANCE_INSUFF, 0u);
  auto mono = pol.Step(MakeCleanRoofPrior(), {0,0,50}, 38.f, DepthMode::MONO_FALLBACK, {50,20,0,FlightRegime::NOMINAL}, 32'000'000'000ULL);
  EXPECT_NE(mono.state, SlzDecisionState::COMMIT_TOUCHDOWN); EXPECT_NE(mono.reason_mask & REASON_MONO_ALT_VETO, 0u);
}
```

- [ ] **Step 2: Run test to verify it fails** — `ctest --test-dir build/core/slz_bayes -R test_active_inspection_policy --output-on-failure` (Expected: FAIL).
- [ ] **Step 3: Implement `visual_slz_prior_provider.hpp` and `active_inspection_policy.{hpp,cpp}`** — Deterministic `noexcept` ~170-LOC 5-state machine arming `orbit_start_ns_` strictly on arrival (`|agl - 22.0| <= 2.5 m`, `|r - 15.0| <= 3.0 m`, `approach_timeout_s = 25.0 s`), guarded `now_ns > start_ns` elapsed diff, `gimbal_pitch_rad = atan2(z_inspect, r_inspect)` + yaw to `tlof`, complete 6-criterion commit & reject bitmask (`REASON_STEP_EXCEEDED`/`REASON_CLEARANCE_INSUFF`), Mod B `> 30 m` veto, `kBaseSoc12s = 0.614f`, and V2.1 `IForwardHazardProvider` stub.
- [ ] **Step 4: Run test to verify it passes** — `cmake --build build --target test_active_inspection_policy && ctest --test-dir build/core/slz_bayes -R test_active_inspection_policy --output-on-failure` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add core/slz_bayes/ && git commit -m "feat(slz-policy): add 2-phase APPROACH->ACTIVE_ORBIT state machine, 22kWh energy gate, and wire stub"`

---

### Task 6: ROS 2 Jazzy `vertirta_perception_node` & Local WSL2 GTX 1650 SUPER (`TU116`) Smoke Verification

**Files:** Create `ros2_ws/src/vertirta_perception_node/{package.xml,CMakeLists.txt,include/vertirta_perception_node/perception_node.hpp,src/perception_node.cpp,config/slz_perception.yaml}`, `scripts/benchmark_ort_gtx1650s.sh`; Test `ros2_ws/src/vertirta_perception_node/test/test_perception_node.cpp`.
**Interfaces:**
- Consumes: Plan 1A `/fmu/out/vertirta_capability_envelope` (`px4_msgs/msg/VertirtaCapabilityEnvelope`, `72 B`) & `vertirta_rta_node` (`vertirta::rta::ISlzPriorProvider`), Tasks 4–5 `vertirta::slz_bayes` (`SlzOrtRunner`/`SlzBeliefGrid`/`ActiveInspectionPolicy`/`VisualSlzPriorProvider`).
- Produces: ROS 2 node `vertirta_perception_node` publishing canonical `/vertirta/slz_priors` (`vertirta_msgs/msg/SlzPriorArray` @ `20–30+ Hz` with `age > 250 ms` staleness flag), `/vertirta/slz/gimbal_cmd`, Foxglove `:8891` markers, and `results/wsl2_gtx1650s_smoke.json` (`ORT-CUDA FP32` vs `FP16` latency/VRAM + `3/3` WSL2 smoke check).

- [ ] **Step 1: Write the failing test**

```cpp
// ros2_ws/src/vertirta_perception_node/test/test_perception_node.cpp
TEST(PerceptionNodeTest, PublishesSnapshotWithinBudgetAndFlagsStaleFramesOver250ms) {
  auto node = std::make_shared<vertirta::ros::SlzPerceptionNode>(MakeMockRunnerOptions());
  node->InjectMockFrame(/*stamp_ns=*/1'000'000'000ULL, /*now_ns=*/1'020'000'000ULL);
  EXPECT_FALSE(node->LatestSnapshot().stale);
  node->CheckStaleness(/*now_ns=*/1'300'000'000ULL);  // 300 ms > 250 ms contract
  EXPECT_TRUE(node->LatestSnapshot().stale);
}
```

- [ ] **Step 2: Run test to verify it fails** — `colcon test --packages-select vertirta_perception_node` (Expected: FAIL).
- [ ] **Step 3: Implement `vertirta_perception_node` and `scripts/benchmark_ort_gtx1650s.sh`** — Lock-free SPSC queues isolating inference from 100 Hz RTA (`0 µs` jitter), flagging `age > 250 ms` staleness. In `benchmark_ort_gtx1650s.sh`, profile `ORT-CUDA FP32` vs `FP16` on **`GTX 1650 SUPER (TU116)`** (`peak_vram_mb < 300`, network p99 `< 12 ms`, loop `>= 20 Hz`) and run `3/3` WSL2 PX4 v1.16 + ROS 2 Jazzy + ORT smoke touchdowns (WSLg `QT_QPA_PLATFORM=xcb`, Foxglove `:8891`).
- [ ] **Step 4: Run test & smoke benchmark to verify they pass** — `colcon build --packages-select vertirta_perception_node && colcon test --packages-select vertirta_perception_node --event-handlers console_direct+` (Expected: PASS; `benchmark_ort_gtx1650s.sh` writes `results/wsl2_gtx1650s_smoke.json` with `peak_vram_mb < 300` & `smoke_pass == "3/3"`).
- [ ] **Step 5: Commit** — `git add ros2_ws/src/vertirta_perception_node/ scripts/benchmark_ort_gtx1650s.sh && git commit -m "feat(slz-ros2): add vertirta_perception_node and WSL2 GTX 1650 SUPER FP32/FP16 smoke harness"`

---

### Task 7: Zero-Shot Real Benchmarks (`AeroScapes`/`ICG`/`UAVScenes`) & Colab G4 `N = 60` 3-Way Closed-Loop Ablation

**Files:** Create `ml/slz_perception/eval/{eval_zeroshot_real.py,aggregate_ablation_tables.py}`, `sim/colab_egl/src/active_vision_ablation_runner.cpp`; Modify `sim/colab_egl/CMakeLists.txt`; Test `ml/slz_perception/tests/test_eval_metrics_and_ablation.py`.
**Interfaces:**
- Consumes: Plan 1A standalone 6-DOF coaxial+Thevenin C++20 plant, `GzEglSensorBridge` (Task 2), `pidnet_slz_512.onnx` (Task 3), Tasks 4–5 C++20 stack, `results/wsl2_gtx1650s_smoke.json` (Task 6).
- Produces: `compute_slz_metrics(...) -> dict`, `build_ablation_seed_manifest(n_total=60) -> list[dict]`, `check_gate5_pass(metrics: dict) -> list[str]`, `active_vision_ablation_runner` binary, `results/dr7_eval_manifest.json`, and merged `docs/paper/metrics_manifest.json`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from ml.slz_perception.eval.eval_zeroshot_real import compute_slz_metrics
from ml.slz_perception.eval.aggregate_ablation_tables import build_ablation_seed_manifest, check_gate5_pass

def test_slz_metrics_balanced_60_seeds_and_gate5_checker():
    m = compute_slz_metrics(np.array([[0, 1], [1, 3]], np.uint8), np.array([[0, 1], [2, 3]], np.uint8),
                            np.array([[-10., -9.], [-2., +3.]], np.float32), np.array([[False, False], [True, True]]))
    assert abs(m["false_safe_rate"] - 0.5) < 1e-6 and m["ood_auroc"] == 1.0
    seeds = build_ablation_seed_manifest(60)
    assert [sum(s["bucket"] == b for s in seeds) for b in ("clean_roof", "dynamic_hazard", "ood_glare_fog")] == [20, 20, 20]
    ok = {"aeroscape_miou": 0.65, "aeroscape_fsr": 0.03, "icg_miou": 0.63, "ood_auroc": 0.91,
          "active_safe_touchdown": 0.933, "active_false_commit": 0.017, "active_max_soc_pct": 0.95, "wsl2_peak_vram_mb": 238.0}
    assert check_gate5_pass(ok) == [] and "active_false_commit" in check_gate5_pass(dict(ok, active_false_commit=0.05))
```

- [ ] **Step 2: Run test to verify it fails** — `pytest ml/slz_perception/tests/test_eval_metrics_and_ablation.py -v` (Expected: FAIL).
- [ ] **Step 3: Implement `eval_zeroshot_real.py`, `active_vision_ablation_runner.cpp`, and `aggregate_ablation_tables.py`** — Run eval-only `AeroScapes` (4-class mIoU `>=62%`, FSR `<=4%`), `ICG` (5-class mIoU `>=60%`, FSR `<=3.5%`), OOD AUROC `>=0.88` / FPR@95 `<=25%` (`Free Energy` vs `MaxLogit`), and `UAVScenes` Mod B slope MAE (`assert_uavscenes_cmap_ids`). Run `active_vision_ablation_runner` on Colab G4 (`N = 60` seeds × 3 policies: Blind DEM vs Passive Vision vs Active Orbit; Gate 5: Safe Touchdown `>=90%`, False-Commit `<=2.0%`, Orbit SoC `<=1.0%` on 22 kWh, TRT FP16 p99 `<2 ms`), freezing `results/dr7_eval_manifest.json` and merging v1/v2a/v2b into `docs/paper/metrics_manifest.json`.
- [ ] **Step 4: Run test to verify it passes** — `pytest ml/slz_perception/tests/test_eval_metrics_and_ablation.py -v && cmake --build build --target active_vision_ablation_runner` (Expected: PASS).
- [ ] **Step 5: Commit** — `git add ml/slz_perception/eval/ sim/colab_egl/src/active_vision_ablation_runner.cpp sim/colab_egl/CMakeLists.txt ml/slz_perception/tests/test_eval_metrics_and_ablation.py && git commit -m "feat(slz-eval): add zero-shot real eval suite and Colab-G4 60-seed 3-way active vision ablation"`

---

### Task 8: Reproducible 6–8 Page LaTeX Technical Report & Zenodo DOI / arXiv `cs.RO` Release Pack

**Files:** Create `docs/paper/{vertirta_tech_report.tex,references.bib,scripts/{generate_latex_tables.py,verify_paper_numbers.py}}`, `.zenodo.json`; Test `docs/paper/tests/test_paper_reproducibility.py`.
**Interfaces:**
- Consumes: Frozen `docs/paper/metrics_manifest.json` (merging Plan 1A/1B `v1_atlas`/`v1_determinism`, Plan 2 `v2a_vertibench`, Task 7 `dr7_eval_manifest` & Task 6 `wsl2_gtx1650s_smoke`).
- Produces: `render_tables_from_manifest(manifest_path: Path, out_tex: Path) -> Path` (`docs/paper/generated/tables.tex`), `verify_paper_honesty_and_numbers(tex_dir: Path, manifest_path: Path) -> list[str]`, compiled 6–8p `docs/paper/vertirta_tech_report.pdf`, and Apache-2.0 `.zenodo.json`.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
from docs.paper.scripts.generate_latex_tables import render_tables_from_manifest
from docs.paper.scripts.verify_paper_numbers import verify_paper_honesty_and_numbers

def test_latex_table_generation_honesty_linter_and_zenodo(tmp_path):
    mf = tmp_path / "metrics_manifest.json"
    mf.write_text(json.dumps({"v1_atlas": {"csfl_n": 650, "cel_n": 150, "cbvf_violations_in_brt": 0},
                              "v2a_copilot": {"unsafe_rejection_pct": 100.0, "conformal_cov_pct": 95.5},
                              "v2b_slz": {"active_safe_touchdown_pct": 93.3, "active_false_commit_pct": 1.7, "wsl2_vram_mb": 238.0}}))
    tex = render_tables_from_manifest(mf, tmp_path / "tables.tex").read_text()
    assert "93.3" in tex and "650" in tex and "150" in tex and "TBD" not in tex
    (tmp_path / "bad.tex").write_text("Monte Carlo proves 10^{-9}/FH EASA SC-VTOL certified compliance.")
    assert any("10^{-9}" in e or "certified" in e for e in verify_paper_honesty_and_numbers(tmp_path, mf))
    meta = json.loads(Path(".zenodo.json").read_text())
    assert meta["license"] == "Apache-2.0" and "VertiRTA" in meta["title"]
```

- [ ] **Step 2: Run test to verify it fails** — `pytest docs/paper/tests/test_paper_reproducibility.py -v` (Expected: FAIL).
- [ ] **Step 3: Implement `generate_latex_tables.py`, `verify_paper_numbers.py`, `vertirta_tech_report.tex`, `references.bib`, and `.zenodo.json`** — Author the 6–8 page IEEEtran/arXiv paper covering Pillars 1–3 with all metrics injected via `\input{generated/tables.tex}`, separate CSFL/CEL Clopper-Pearson 95% CIs, explicit non-compliance disclaimer, and linter banning `10^{-9}/FH` MC claims, fake certification phrasing, or `[TBD]`/`[Ölçülecek]` placeholders.
- [ ] **Step 4: Run test & LaTeX build to verify they pass** — `pytest docs/paper/tests/test_paper_reproducibility.py -v && (cd docs/paper && latexmk -pdf -interaction=nonstopmode -halt-on-error vertirta_tech_report.tex)` (Expected: PASS, 6–8 pages verified by `pdfinfo`).
- [ ] **Step 5: Commit** — `git add docs/paper/ .zenodo.json && git commit -m "docs(paper): add reproducible 6-8p VertiRTA LaTeX technical report and Zenodo metadata"`
