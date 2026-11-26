# SPDX-License-Identifier: Apache-2.0
"""Contract, unit, smoke, and committed-artifact tests for local GTX 1650 SUPER Forward WireSeg + D-FINE-N v1.1."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.bench_gtx1650s_m3_timesliced import get_model_specs, verify_onnx_sha256_parity
from scripts.colab.g4_jobs.common import write_canonical_json
from scripts.colab.g4_jobs.job_forward_wire import (
    _build_torch_models,
    compute_dfine_sigmoid_focal_loss,
    decode_dfine_sigmoid_boxes,
    encode_dfine_grid_targets,
)
from scripts.train_local_gtx1650s_forward_wire_v2 import compute_wireseg_hardneg_loss

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "v4.0.0"
LOCAL_FW_METRICS = ARTIFACTS_DIR / "local_gtx1650s_forward_wire_v2_metrics.json"
LOCAL_PIDNET_METRICS = ARTIFACTS_DIR / "local_gtx1650s_pidnet_v21_recovery_metrics.json"
G4_PIDNET_XFEAT_METRICS = ARTIFACTS_DIR / "g4_pidnet_v21_xfeat_metrics.json"
TIMESLICED_GATE_JSON = ARTIFACTS_DIR / "gtx1650s_m3_timesliced_edge_gate.json"


def _verify_canonical_report_sha256(path: Path) -> dict:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), f"BOM found in {path}"
    assert b"\r" not in raw, f"CR found in {path}"
    doc = json.loads(raw.decode("utf-8"))
    expected = doc["report_sha256"]
    body = {k: v for k, v in doc.items() if k != "report_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    assert actual == expected, f"Canonical report_sha256 mismatch in {path.name}: {actual} != {expected}"
    return doc


def test_local_forward_wire_v2_hygiene_no_bom_no_cr_no_personal_paths() -> None:
    targets = [
        REPO_ROOT / "scripts" / "train_local_gtx1650s_forward_wire_v2.py",
        REPO_ROOT / "scripts" / "bench_gtx1650s_m3_timesliced.py",
        REPO_ROOT / "scripts" / "colab" / "g4_jobs" / "job_forward_wire.py",
    ]
    forbidden = ("dummy_vram", "pad_vram", "reserve_vram", "/home/dias", "C:\\Users\\", "C:\\Temp\\")
    for p in targets:
        raw = p.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf"), f"BOM found in {p}"
        assert b"\r" not in raw, f"CR found in {p}"
        text = raw.decode("utf-8")
        for tok in forbidden:
            assert tok not in text, f"Forbidden token {tok!r} in {p}"
        tree = ast.parse(text, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr != "empty", f"Forbidden .empty() call in {p}"


def test_dfine_sigmoid_decode_emits_zero_boxes_on_empty_frame_and_applies_nms() -> None:
    # 1) Empty scene with RetinaNet/D-FINE prior logits (-4.595) MUST emit 0 detections!
    boxes_empty = np.tile(np.array([[0.1, 0.1, 0.4, 0.4]], dtype=np.float32), (300, 1))[None, ...]
    logits_empty = np.full((1, 300, 3), -4.59512, dtype=np.float32)
    assert decode_dfine_sigmoid_boxes(boxes_empty, logits_empty, img_size=512, conf_thresh=0.55) == []

    # 2) Encode two distinct targets + duplicate overlapping query -> NMS keeps 2 clean detections
    tb, tc = encode_dfine_grid_targets(
        [
            (0.10, 0.20, 0.25, 0.35, 0),  # Bird
            (0.60, 0.50, 0.80, 0.85, 2),  # Tower_Crane
        ]
    )
    assert int((tc > 0.5).sum()) == 2
    logits = np.full((300, 3), -6.0, dtype=np.float32)
    logits[tc > 0.5] = 4.0  # sigmoid(4.0) ~= 0.982
    # Inject lower-confidence near-duplicate Bird box at query 0 to verify per-class NMS
    tb[0] = np.array([0.11, 0.20, 0.25, 0.35], dtype=np.float32)
    logits[0, 0] = 2.0  # sigmoid(2.0) ~= 0.881
    dets = decode_dfine_sigmoid_boxes(tb[None, ...], logits[None, ...], img_size=640, conf_thresh=0.55)
    assert len(dets) == 2
    labels = {d[4] for d in dets}
    assert labels == {"Bird", "Tower_Crane"}


def test_dfine_sigmoid_focal_and_wireseg_hardneg_losses_finite_on_empty_and_mixed() -> None:
    torch = pytest.importorskip("torch")
    SoftClDiceLoss, _, _ = _build_torch_models()

    # Empty batch (all hard negatives) must still backprop finite non-zero focal gradients
    pred_boxes = torch.rand((2, 300, 4), requires_grad=True)
    pred_cls = torch.zeros((2, 300, 3), requires_grad=True)
    tgt_boxes = torch.zeros((2, 300, 4))
    tgt_cls = torch.zeros((2, 300, 3))
    loss_empty = compute_dfine_sigmoid_focal_loss(pred_boxes, pred_cls, tgt_boxes, tgt_cls)
    assert torch.isfinite(loss_empty) and float(loss_empty.item()) > 0.0
    loss_empty.backward()
    assert pred_cls.grad is not None and torch.all(torch.isfinite(pred_cls.grad))
    assert float(pred_cls.grad.abs().sum().item()) > 0.0

    # Mixed positive + empty rooftop batch for WireSeg OHEM hard-negative loss
    cldice_fn = SoftClDiceLoss(iters=2, lambda_cl=0.5)
    w_logits = torch.zeros((2, 1, 64, 64), requires_grad=True)
    j_pred = torch.zeros((2, 5, 16, 16), requires_grad=True)
    w_tgt = torch.zeros((2, 1, 64, 64))
    w_tgt[0, 0, 30:32, 10:54] = 1.0  # sample 0 positive wire, sample 1 empty rooftop
    j_tgt = torch.zeros((2, 5, 16, 16))
    w_loss = compute_wireseg_hardneg_loss(w_logits, j_pred, w_tgt, j_tgt, cldice_fn)
    assert torch.isfinite(w_loss) and float(w_loss.item()) > 0.0
    w_loss.backward()
    assert w_logits.grad is not None and torch.all(torch.isfinite(w_logits.grad))


def test_train_local_gtx1650s_forward_wire_v2_smoke_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    out_json = tmp_path / "local_fw_v2_smoke.json"
    blob_dir = tmp_path / "blobs"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_local_gtx1650s_forward_wire_v2.py"),
            "--smoke-cpu",
            "--blob-dir",
            str(blob_dir),
            "--out-json",
            str(out_json),
        ],
        check=True,
    )
    doc = _verify_canonical_report_sha256(out_json)
    assert doc["mode"] == "SMOKE_CPU_ONLY"
    assert doc["colab_compute_units_consumed"] == 0.0
    assert doc["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert doc["onnx_exports"]["mlsd_wireseg"]["onnx_checker_passed"] is True
    assert doc["onnx_exports"]["dfine_n"]["onnx_checker_passed"] is True
    assert doc["onnx_exports"]["mlsd_wireseg"]["input_shape"] == [1, 3, 640, 640]
    assert doc["onnx_exports"]["dfine_n"]["output_names"] == ["boxes_xyxy", "class_logits"]


def test_bench_timesliced_local_gtx_v11_profile_parity_and_sanitized_paths(tmp_path: Path) -> None:
    specs = get_model_specs("local_gtx_v11")
    assert [s["key"] for s in specs] == ["pidnet_slz_v21", "mlsd_wireseg", "dfine_n", "xfeat_512"]
    blob_dir = tmp_path / "blobs"
    art_dir = tmp_path / "artifacts"
    blob_dir.mkdir()
    art_dir.mkdir()

    shas: dict[str, str] = {}
    for s in specs:
        p = blob_dir / s["filename"]
        data = f"dummy-{s['filename']}".encode("utf-8")
        p.write_bytes(data)
        shas[s["key"]] = hashlib.sha256(data).hexdigest()

    write_canonical_json(
        art_dir / "local_gtx1650s_pidnet_v21_recovery_metrics.json",
        {"onnx_export": {"sha256": shas["pidnet_slz_v21"]}},
    )
    write_canonical_json(
        art_dir / "local_gtx1650s_forward_wire_v2_metrics.json",
        {
            "onnx_exports": {
                "mlsd_wireseg": {"sha256": shas["mlsd_wireseg"]},
                "dfine_n": {"sha256": shas["dfine_n"]},
            }
        },
    )
    write_canonical_json(
        art_dir / "g4_pidnet_v21_xfeat_metrics.json",
        {"onnx_exports": {"xfeat_512": {"sha256": shas["xfeat_512"]}}},
    )

    verified = verify_onnx_sha256_parity(blob_dir, art_dir, model_profile="local_gtx_v11")
    for k, entry in verified.items():
        assert entry["sha256_verified"] is True
        assert entry["sha256"] == shas[k]
        assert "C:\\" not in entry["path"] and "/home/" not in entry["path"]


def test_committed_local_forward_wire_v2_and_timesliced_artifacts() -> None:
    if not LOCAL_FW_METRICS.is_file():
        pytest.skip("local_gtx1650s_forward_wire_v2_metrics.json not yet generated")
    fw = _verify_canonical_report_sha256(LOCAL_FW_METRICS)
    assert fw["colab_compute_units_consumed"] == 0.0
    assert fw["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert fw["license_audit"]["zero_sharealike_verified"] is True
    assert fw["dataset_ingestion"]["teias_license"] == "MIT"
    assert fw["dataset_ingestion"]["teias_vl_pos_train"] > 0
    assert fw["dataset_ingestion"]["urban_hardneg_train"] > 0
    assert fw["nasa_goddard_zeroshot_240f"]["used_in_training"] is False
    assert fw["nasa_goddard_zeroshot_240f"]["used_in_calibration"] is False
    assert fw["nasa_goddard_zeroshot_240f"]["gates"]["wireseg_nasa_fp_frames_lt_10pct"] is True
    assert fw["nasa_goddard_zeroshot_240f"]["gates"]["dfine_nasa_spurious_boxes_lt_15"] is True

    # Verify refreshed timesliced edge gate references active recovered local GTX v1.1 models
    ts_doc = _verify_canonical_report_sha256(TIMESLICED_GATE_JSON)
    assert ts_doc["model_profile"] == "local_gtx_v11"
    assert ts_doc["gates"]["onnx_sha256_all_verified"] is True
    assert ts_doc["gates"]["all_trained_outputs_finite_and_nondegenerate"] is True

    pid_doc = json.loads(LOCAL_PIDNET_METRICS.read_text(encoding="utf-8"))
    xfeat_doc = json.loads(G4_PIDNET_XFEAT_METRICS.read_text(encoding="utf-8"))
    vm = ts_doc["verified_models"]
    assert vm["pidnet_slz_v21"]["filename"] == "pidnet_slz_v21_local_gtx_opset17.onnx"
    assert vm["pidnet_slz_v21"]["sha256"] == pid_doc["onnx_export"]["sha256"]
    assert vm["mlsd_wireseg"]["filename"] == "mlsd_wireseg_640_v11_local_gtx_opset17.onnx"
    assert vm["mlsd_wireseg"]["sha256"] == fw["onnx_exports"]["mlsd_wireseg"]["sha256"]
    assert vm["dfine_n"]["filename"] == "dfine_n_640_v11_local_gtx_opset17.onnx"
    assert vm["dfine_n"]["sha256"] == fw["onnx_exports"]["dfine_n"]["sha256"]
    assert vm["xfeat_512"]["sha256"] == xfeat_doc["onnx_exports"]["xfeat_512"]["sha256"]
    raw_ts_text = TIMESLICED_GATE_JSON.read_text(encoding="utf-8")
    assert "C:\\Temp\\" not in raw_ts_text
    assert "C:/Temp/" not in raw_ts_text
