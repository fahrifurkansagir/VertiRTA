#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Strictly Quarantined Zero-Shot UAVScenes Real-Imagery SLZ Evaluation (CPU-Only, 0.00 CU).

Evaluates the frozen Apache-2.0 PIDNet-S v2.1 Opset-17 ONNX model
(`/var/tmp/vertirta_v400_blobs/pidnet_slz_v21_local_gtx_opset17.onnx`) on a stratified
300-frame subset (15 frames x all 20 UAV scenes) streamed selectively via HTTP Range
requests (`remotezip`) from ungated `sijieaaa/UAVScenes` (`interval5_CAM_LIDAR.zip` +
`interval5_CAM_label.zip`), with:
  1. Live `cmap.py` fetch + `verify_eval_only_dataset("UAVScenes", split="eval", uavscenes_cmap=...)`
     + `assert_uavscenes_cmap_ids()` sparse 0..25 integrity verification.
  2. Strict `CPUExecutionProvider` ONLY (`CUDA_VISIBLE_DEVICES=""`, zero GPU VRAM touch).
  3. Exact canonical remapping via `ml.slz_perception.data.remappers.remap_mask("uavscenes", ...)`.
  4. Both raw and bias-calibrated (`safe_logit_bias_delta = -0.271367`, frozen from permissive
     ID val — NEVER fitted on UAVScenes) 5-class SLZ mIoU / FSR / safe precision / SLZ F1 metrics.
  5. Immediate deletion of `/tmp/vertirta_uavscenes_eval/` raw pixels in `finally:` so ZERO
     ShareAlike/NC imagery is persisted on disk (`used_in_training=False`, `used_in_calibration=False`).
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any
import urllib.request

# Enforce strict CPU-only execution before importing onnxruntime
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.slz_perception.data.remappers import (  # noqa: E402
    NUM_CANONICAL_CLASSES,
    remap_mask,
)
from ml.slz_perception.eval.eval_zeroshot_real import (  # noqa: E402
    verify_eval_only_dataset,
)
from scripts.colab.g4_jobs.common import (  # noqa: E402
    sha256_file,
    write_canonical_json,
)


def _sanitize_path(p: Path) -> str:
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()

SCHEMA_VERSION = "vertirta.realworld_uavscenes_zeroshot_eval/v1.0.0"
DEFAULT_ONNX = Path("/var/tmp/vertirta_v400_blobs/pidnet_slz_v21_local_gtx_opset17.onnx")
DEFAULT_CAL = Path("/var/tmp/vertirta_v400_blobs/ood_calibration_v21_local_gtx.json")
DEFAULT_TMP_DIR = Path("/tmp/vertirta_uavscenes_eval")
DEFAULT_OUT_JSON = REPO_ROOT / "artifacts/v4.0.0/realworld_uavscenes_zeroshot_eval.json"

UAVSCENES_RGB_ZIP_URL = (
    "https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_LIDAR.zip"
)
UAVSCENES_LABEL_ZIP_URL = (
    "https://huggingface.co/datasets/sijieaaa/UAVScenes/resolve/main/interval5_CAM_label.zip"
)
UAVSCENES_CMAP_URL = "https://raw.githubusercontent.com/sijieaaa/UAVScenes/main/cmap.py"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CANONICAL_CLASS_NAMES: tuple[str, ...] = (
    "Safe_Flat_Ground",
    "Safe_Flat_Roof",
    "Hazard_Obstacle_Structure",
    "Hazard_Dynamic_Human_Vehicle",
    "Hazard_Water_Vegetation",
)


def fetch_and_parse_uavscenes_cmap(url: str = UAVSCENES_CMAP_URL) -> tuple[dict[int, str], str]:
    """Fetch official UAVScenes cmap.py and extract `{id: name}` safely via AST literal_eval."""
    req = urllib.request.Request(url, headers={"User-Agent": "VertiRTA-Eval/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw_bytes = resp.read()
    text = raw_bytes.decode("utf-8")
    tree = ast.parse(text, filename="cmap.py")
    cmap_dict_raw: dict[Any, Any] | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "cmap":
                    cmap_dict_raw = ast.literal_eval(node.value)
                    break
    if not isinstance(cmap_dict_raw, dict):
        raise ValueError("Failed to parse `cmap = {...}` dictionary from UAVScenes cmap.py")
    cmap_names = {int(k): str(v["name"]) for k, v in cmap_dict_raw.items()}
    import hashlib

    cmap_sha = hashlib.sha256(raw_bytes).hexdigest()
    return cmap_names, cmap_sha


def _summarize_confusion(conf: np.ndarray) -> dict[str, Any]:
    """Compute canonical 5-class SLZ metrics matching local_gtx1650s_pidnet_v21_recovery_metrics.json."""
    per_class_iou: dict[str, float | None] = {}
    present_ious: list[float] = []
    for c in range(NUM_CANONICAL_CLASSES):
        gt_sum = int(conf[c, :].sum())
        if gt_sum == 0:
            per_class_iou[CANONICAL_CLASS_NAMES[c]] = None
            continue
        tp = int(conf[c, c])
        union = gt_sum + int(conf[:, c].sum()) - tp
        iou = float(tp) / float(max(1, union))
        per_class_iou[CANONICAL_CLASS_NAMES[c]] = round(iou, 6)
        present_ious.append(iou)

    total_px = int(conf.sum())
    correct_px = int(np.trace(conf))
    true_safe_px = int(conf[:2, :].sum())
    true_haz_px = int(conf[2:, :].sum())
    pred_safe_px = int(conf[:, :2].sum())
    tp_safe_px = int(conf[:2, :2].sum())
    false_safe_px = int(conf[2:, :2].sum())
    fn_safe_px = true_safe_px - tp_safe_px

    miou = float(np.mean(present_ious)) if present_ious else 0.0
    pixel_acc = float(correct_px) / float(max(1, total_px))
    fsr = float(false_safe_px) / float(max(1, true_haz_px))
    safe_prec = float(tp_safe_px) / float(max(1, pred_safe_px))
    f1_denom = 2 * tp_safe_px + false_safe_px + fn_safe_px
    slz_f1 = float(2 * tp_safe_px) / float(max(1, f1_denom))

    return {
        "per_class_iou": per_class_iou,
        "miou_present_classes": round(miou, 6),
        "num_present_classes": len(present_ious),
        "pixel_accuracy": round(pixel_acc, 6),
        "false_safe_rate_fsr": round(fsr, 6),
        "safe_precision": round(safe_prec, 6),
        "slz_f1": round(slz_f1, 6),
        "total_eval_pixels": total_px,
        "true_hazard_pixels": true_haz_px,
        "false_safe_pixels": false_safe_px,
        "confusion_matrix_5x5": [[int(x) for x in row] for row in conf.tolist()],
    }


def _select_stratified_pairs(
    rgb_members: list[str],
    label_members: list[str],
    *,
    target_total: int,
) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Pair RGB `.jpg` and label-ID `.png` members across all scenes with uniform temporal stride."""
    label_by_key: dict[tuple[str, str], str] = {}
    for m in label_members:
        parts = m.split("/")
        if len(parts) == 4 and parts[2] == "interval5_CAM_label_id" and parts[3].endswith(".png"):
            scene = parts[1]
            stem = parts[3][:-4]
            label_by_key[(scene, stem)] = m

    by_scene: dict[str, list[tuple[str, str]]] = {}
    for m in rgb_members:
        parts = m.split("/")
        if len(parts) == 4 and parts[2] == "interval5_CAM" and parts[3].endswith(".jpg"):
            scene = parts[1]
            stem = parts[3][:-4]
            lbl_m = label_by_key.get((scene, stem))
            if lbl_m is not None:
                by_scene.setdefault(scene, []).append((m, lbl_m))

    scenes = sorted(by_scene.keys())
    if not scenes:
        raise RuntimeError("No matched RGB + interval5_CAM_label_id pairs found in UAVScenes archives.")

    for sc in scenes:
        by_scene[sc].sort(key=lambda pair: pair[0])

    base_per_scene = max(1, target_total // len(scenes))
    rem = max(0, target_total - base_per_scene * len(scenes))

    selected: list[tuple[str, str, str]] = []
    counts: dict[str, int] = {}
    for idx, sc in enumerate(scenes):
        pool = by_scene[sc]
        k = min(len(pool), base_per_scene + (1 if idx < rem else 0))
        if k == 1:
            pick_idxs = [len(pool) // 2]
        else:
            pick_idxs = sorted(
                {int(round(i * (len(pool) - 1) / float(k - 1))) for i in range(k)}
            )
        for pi in pick_idxs:
            rgb_m, lbl_m = pool[pi]
            selected.append((sc, rgb_m, lbl_m))
        counts[sc] = len(pick_idxs)

    return selected, counts


def run_uavscenes_zeroshot_eval(
    *,
    onnx_path: Path,
    cal_json: Path,
    tmp_dir: Path,
    out_json: Path,
    n_samples: int = 300,
    cpu_threads: int = 4,
) -> dict[str, Any]:
    """Stream stratified UAVScenes frames via remotezip, run CPU-only ONNX eval, delete raw pixels."""
    import onnxruntime as ort
    from PIL import Image
    from remotezip import RemoteZip

    t0 = time.perf_counter()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Missing frozen PIDNet-S v2.1 ONNX model: {onnx_path}")
    if not cal_json.is_file():
        raise FileNotFoundError(f"Missing frozen OOD/FSR calibration JSON: {cal_json}")

    onnx_sha = sha256_file(onnx_path)
    cal_sha = sha256_file(cal_json)
    cal_data = json.loads(cal_json.read_text(encoding="utf-8"))
    safe_bias_delta = float(cal_data["safe_logit_bias_delta"])

    # 1) Strict quarantine + official cmap.py sparse 0..25 assertion BEFORE touching any pixels
    cmap_names, cmap_sha = fetch_and_parse_uavscenes_cmap()
    quarantine_report = verify_eval_only_dataset(
        "UAVScenes", split="eval", uavscenes_cmap=cmap_names
    )
    print(
        f"[UAVScenes] Verified cmap.py (sha256={cmap_sha[:12]}..., {len(cmap_names)} IDs) "
        f"& eval-only quarantine ({quarantine_report.quarantined_eval_entries}).",
        flush=True,
    )

    # 2) Strict CPU-only ONNX Runtime session (ZERO CUDA / GPU VRAM touch)
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = int(max(1, cpu_threads))
    sess_opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    active_providers = list(sess.get_providers())
    assert active_providers == ["CPUExecutionProvider"], (
        f"Expected strictly CPUExecutionProvider, got {active_providers}"
    )
    inp_name = sess.get_inputs()[0].name

    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    # Warm up CPU session once
    _ = sess.run(None, {inp_name: np.zeros((1, 3, 512, 512), dtype=np.float32)})

    conf_raw = np.zeros((NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES), dtype=np.int64)
    conf_cal = np.zeros((NUM_CANONICAL_CLASSES, NUM_CANONICAL_CLASSES), dtype=np.int64)
    per_scene_conf_cal: dict[str, np.ndarray] = {}
    latencies_ms: list[float] = []
    raw_min_seen = 255
    raw_max_seen = 0
    bytes_streamed_rgb = 0
    bytes_streamed_lbl = 0
    total_available_pairs = 0
    scene_sample_counts: dict[str, int] = {}

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        with RemoteZip(UAVSCENES_RGB_ZIP_URL) as z_rgb, RemoteZip(UAVSCENES_LABEL_ZIP_URL) as z_lbl:
            rgb_names = [i.filename for i in z_rgb.infolist() if not i.is_dir()]
            lbl_names = [i.filename for i in z_lbl.infolist() if not i.is_dir()]
            total_available_pairs = sum(
                1 for n in lbl_names if "/interval5_CAM_label_id/" in n and n.endswith(".png")
            )
            selected, scene_sample_counts = _select_stratified_pairs(
                rgb_names, lbl_names, target_total=int(n_samples)
            )
            print(
                f"[UAVScenes] Streaming {len(selected)}/{total_available_pairs} stratified pairs "
                f"across {len(scene_sample_counts)} scenes into {tmp_dir} (CPU-only ONNX)...",
                flush=True,
            )

            t_eval0 = time.perf_counter()
            for idx, (scene, rgb_member, lbl_member) in enumerate(selected, start=1):
                rgb_bytes = z_rgb.read(rgb_member)
                lbl_bytes = z_lbl.read(lbl_member)
                bytes_streamed_rgb += len(rgb_bytes)
                bytes_streamed_lbl += len(lbl_bytes)

                rgb_local = tmp_dir / f"{idx:04d}.jpg"
                lbl_local = tmp_dir / f"{idx:04d}.png"
                rgb_local.write_bytes(rgb_bytes)
                lbl_local.write_bytes(lbl_bytes)

                with Image.open(rgb_local) as im_rgb:
                    rgb_512 = np.asarray(
                        im_rgb.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                with Image.open(lbl_local) as im_lbl:
                    raw_id_native = np.asarray(im_lbl)
                    if raw_id_native.ndim != 2:
                        raise ValueError(
                            f"Expected 2D grayscale ID mask for {lbl_member}, got shape {raw_id_native.shape}"
                        )
                    raw_min_seen = min(raw_min_seen, int(raw_id_native.min()))
                    raw_max_seen = max(raw_max_seen, int(raw_id_native.max()))
                    raw_id_512 = np.asarray(
                        im_lbl.resize((512, 512), Image.Resampling.NEAREST),
                        dtype=np.uint8,
                    )

                # Immediately unlink per-frame temp files to keep disk footprint near zero
                rgb_local.unlink(missing_ok=True)
                lbl_local.unlink(missing_ok=True)

                gt_512 = remap_mask("uavscenes", raw_id_512)
                valid = (gt_512 >= 0) & (gt_512 < NUM_CANONICAL_CLASSES)
                if not bool(valid.any()):
                    continue

                x = (rgb_512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
                x = np.ascontiguousarray((x - mean) / std, dtype=np.float32)

                t_inf0 = time.perf_counter_ns()
                seg_logits, _surf_normals, _bdry_prob = sess.run(None, {inp_name: x})
                latencies_ms.append((time.perf_counter_ns() - t_inf0) * 1e-6)

                logits_hw = seg_logits[0].astype(np.float32)  # (5, 512, 512)
                pred_raw = np.argmax(logits_hw, axis=0).astype(np.uint8)
                shifted = logits_hw.copy()
                shifted[0:2] = shifted[0:2] + safe_bias_delta
                pred_cal = np.argmax(shifted, axis=0).astype(np.uint8)

                gt_v = gt_512[valid].astype(np.int64)
                c_raw = np.bincount(
                    gt_v * NUM_CANONICAL_CLASSES + pred_raw[valid].astype(np.int64),
                    minlength=25,
                ).reshape(5, 5)
                c_cal = np.bincount(
                    gt_v * NUM_CANONICAL_CLASSES + pred_cal[valid].astype(np.int64),
                    minlength=25,
                ).reshape(5, 5)
                conf_raw += c_raw
                conf_cal += c_cal
                per_scene_conf_cal[scene] = (
                    per_scene_conf_cal.get(scene, np.zeros((5, 5), dtype=np.int64)) + c_cal
                )

                if idx % 50 == 0 or idx == len(selected):
                    cur_cal = _summarize_confusion(conf_cal)
                    print(
                        f"  [{idx:3d}/{len(selected)}] cal_mIoU={cur_cal['miou_present_classes']:.4f} "
                        f"cal_FSR={cur_cal['false_safe_rate_fsr']:.4f} "
                        f"mean_cpu_ms={float(np.mean(latencies_ms)):.1f} "
                        f"elapsed={time.perf_counter() - t_eval0:.1f}s",
                        flush=True,
                    )
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    raw_pixels_deleted = not tmp_dir.exists()
    assert raw_pixels_deleted, f"Quarantine violation: {tmp_dir} was not deleted!"

    raw_summary = _summarize_confusion(conf_raw)
    cal_summary = _summarize_confusion(conf_cal)
    per_scene_summary: dict[str, dict[str, Any]] = {}
    for sc in sorted(per_scene_conf_cal.keys()):
        sm = _summarize_confusion(per_scene_conf_cal[sc])
        per_scene_summary[sc] = {
            "n_samples": scene_sample_counts.get(sc, 0),
            "miou_present_classes": sm["miou_present_classes"],
            "false_safe_rate_fsr": sm["false_safe_rate_fsr"],
            "safe_precision": sm["safe_precision"],
            "num_present_classes": sm["num_present_classes"],
        }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": "LIVE_LOCAL_CPU_ONNX_ZERO_SHOT_EVAL",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "colab_cu_consumed": 0.0,
        "gpu_vram_touched_mib": 0.0,
        "ort_providers_active": active_providers,
        "host_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_threads_used": int(cpu_threads),
        },
        "quarantine_policy": {
            "dataset": "UAVScenes",
            "hf_repo": "sijieaaa/UAVScenes",
            "license": "CC-BY-NC-SA-4.0 / CC-BY-SA-4.0 (QUARANTINED_ZERO_SHOT_EVAL_ONLY)",
            "used_in_training": False,
            "used_in_calibration": False,
            "raw_pixels_deleted": raw_pixels_deleted,
            "cmap_verified_via_assert_uavscenes_cmap_ids": True,
            "cmap_url": UAVSCENES_CMAP_URL,
            "cmap_sha256": cmap_sha,
            "cmap_id_count": len(cmap_names),
            "raw_mask_id_min_observed": int(raw_min_seen),
            "raw_mask_id_max_observed": int(raw_max_seen),
            "quarantined_eval_entries": list(quarantine_report.quarantined_eval_entries),
            "zero_sharealike_in_training_verified": bool(quarantine_report.zero_sharealike_verified),
        },
        "archive_metadata": {
            "rgb_archive_url": UAVSCENES_RGB_ZIP_URL,
            "label_archive_url": UAVSCENES_LABEL_ZIP_URL,
            "ingestion_method": "selective_http_range_remotezip_ephemeral_zero_persistence",
            "total_available_labeled_pairs": int(total_available_pairs),
            "evaluated_images": int(len(latencies_ms)),
            "scenes_covered": len(scene_sample_counts),
            "scene_sample_counts": scene_sample_counts,
            "bytes_streamed_rgb_mb": round(bytes_streamed_rgb / (1024.0 * 1024.0), 2),
            "bytes_streamed_label_mb": round(bytes_streamed_lbl / (1024.0 * 1024.0), 2),
            "eval_resolution_hw": [512, 512],
        },
        "model_frozen": {
            "filename": onnx_path.name,
            "path": _sanitize_path(onnx_path),
            "sha256": onnx_sha,
            "calibration_file": cal_json.name,
            "calibration_sha256": cal_sha,
            "safe_logit_bias_delta_applied_from_permissive_id_val": safe_bias_delta,
        },
        "cpu_latency_ms": {
            "mean_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "p50_ms": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else 0.0,
            "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        },
        "quarantined_zero_shot_uavscenes_raw": {
            **raw_summary,
            "dataset": "UAVScenes",
            "used_in_training": False,
            "used_in_calibration": False,
            "evaluated_images": int(len(latencies_ms)),
        },
        "quarantined_zero_shot_uavscenes_bias_calibrated": {
            **cal_summary,
            "dataset": "UAVScenes",
            "used_in_training": False,
            "used_in_calibration": False,
            "evaluated_images": int(len(latencies_ms)),
            "safe_logit_bias_delta_applied_from_permissive_id_val": safe_bias_delta,
        },
        "per_scene_bias_calibrated": per_scene_summary,
        "wall_clock_s": round(time.perf_counter() - t0, 2),
    }
    write_canonical_json(out_json, payload)
    print(
        f"[DONE] UAVScenes zero-shot eval -> {_sanitize_path(out_json)} | "
        f"RAW mIoU={raw_summary['miou_present_classes']:.4f} FSR={raw_summary['false_safe_rate_fsr']:.4f} | "
        f"CAL mIoU={cal_summary['miou_present_classes']:.4f} FSR={cal_summary['false_safe_rate_fsr']:.4f} | "
        f"raw_pixels_deleted={raw_pixels_deleted}",
        flush=True,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--cal-json", type=Path, default=DEFAULT_CAL)
    parser.add_argument("--tmp-dir", type=Path, default=DEFAULT_TMP_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()

    run_uavscenes_zeroshot_eval(
        onnx_path=args.onnx_path,
        cal_json=args.cal_json,
        tmp_dir=args.tmp_dir,
        out_json=args.out_json,
        n_samples=args.n_samples,
        cpu_threads=args.cpu_threads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
