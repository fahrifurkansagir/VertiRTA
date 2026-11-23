# SPDX-License-Identifier: Apache-2.0
"""Unit and artifact-contract tests for v3.1.0 Colab L4 PIDNet-S distillation & GTX 1650 SUPER proof."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from ml.slz_perception.data.license_gate import DatasetEntry, LicenseViolationError
from ml.slz_perception.train.train_slz_distill import (
    L4TrainProfile,
    apply_anomaly_mix,
    build_arg_parser as build_train_arg_parser,
    build_l4_training_manifest,
    compute_boundary_target,
    validate_training_startup_license,
)
from scripts.colab.run_l4_pidnet_distill import (
    calibrate_dynamic_energy_margins,
    compute_binary_auroc_and_fpr95,
    compute_confusion_metrics,
    decode_carla_depth_meters,
    extract_skyscenes_frame_id,
    filter_oem_region,
    resolve_tu116_cuda_provider_options,
    run_startup_license_gate,
    select_stage2_checkpoint,
    skyscenes_cam_rot_wc_enu,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
L4_METRICS_PATH = REPO_ROOT / "artifacts" / "v3.1.0" / "l4_pidnet_distill_metrics.json"
GTX_PROOF_PATH = REPO_ROOT / "artifacts" / "v3.1.0" / "local_gtx1650s_pidnet_trained_proof.json"
L4_V2_METRICS_PATH = REPO_ROOT / "artifacts" / "v3.5.1" / "l4_pidnet_v2_distill_metrics.json"
GTX_V2_PROOF_PATH = REPO_ROOT / "artifacts" / "v3.5.1" / "local_gtx1650s_pidnet_v2_proof.json"


def _write_synthetic_oem_html(path: Path) -> Path:
    rows = [
        "<tr><td>Region</td><td>Country</td><td>Source</td><td>Provider</td><td>License</td></tr>",
        "<tr><td>Christchurch</td><td>NZ</td><td>AIRS</td><td>LINZ</td><td>CC BY 4.0</td></tr>",
        "<tr><td>Aachen</td><td>DE</td><td>GeoNRW</td><td>NRW</td><td>DL-DE-BY-2.0</td></tr>",
        "<tr><td>Beijing</td><td>CN</td><td>Map</td><td>Prov</td><td>CC BY-NC-SA 4.0</td></tr>",
        "<tr><td>Austin</td><td>US</td><td>NAIP</td><td>USDA</td><td>Public domain</td></tr>",
        "<tr><td>MysteryCity</td><td>XX</td><td>Unknown</td><td>None</td><td></td></tr>",
        "<tr><td>Zurich</td><td>CH</td><td>Swiss</td><td>Swisstopo</td><td>CC BY-SA 4.0</td></tr>",
    ]
    html = "<html><body><table>" + "".join(rows) + "</table></body></html>\n"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path


def test_l4_train_cli_and_license_gate(tmp_path: Path) -> None:
    html_path = _write_synthetic_oem_html(tmp_path / "attribution.html")
    parser = build_train_arg_parser()
    args = parser.parse_args(
        [
            "--oem-attribution-html",
            str(html_path),
            "--expect-oem-total",
            "0",
            "--profile",
            "l4",
            "--batch-size",
            "16",
            "--grad-accum",
            "2",
            "--stage1-epochs",
            "24",
            "--stage2-epochs",
            "10",
            "--verify-only",
        ]
    )
    assert args.profile == "l4"
    assert args.batch_size == 16 and args.grad_accum == 2 and args.bf16 is True

    prof = L4TrainProfile()
    assert prof.batch_size * prof.grad_accum == 32

    report, oem_regions = run_startup_license_gate(
        html_path,
        ("FLAIR-1", "OpenEarthMap", "FloodNet-Supervised-v1.0", "SkyScenes"),
        expect_oem_total=None,
        include_aeroscapes_eval=True,
    )
    assert report.zero_sharealike_verified is True
    assert set(oem_regions) == {"aachen", "christchurch"}
    assert "AeroScapes" in report.quarantined_eval_entries

    # Region filter strictly admits only CC BY 4.0 / DL-DE-BY-2.0 regions
    wl = frozenset(oem_regions)
    assert filter_oem_region("aachen_12.tif", wl) == "aachen"
    assert filter_oem_region("Christchurch_4.tif", wl) == "christchurch"
    assert filter_oem_region("beijing_1.tif", wl) is None
    assert filter_oem_region("austin_9.tif", wl) is None
    assert filter_oem_region("zurich_2.tif", wl) is None

    # Tainted training manifests must still hard-fail before any download/training
    tainted_manifest = build_l4_training_manifest(oem_regions, ("OpenEarthMap",)) + [
        DatasetEntry(name="Depth-Anything-V2-Large", split="teacher", license="CC BY-NC 4.0")
    ]
    with pytest.raises(LicenseViolationError):
        validate_training_startup_license(tainted_manifest, html_path, expect_oem_total=None)

    # Quarantined AeroScapes in train split must hard-fail
    with pytest.raises(LicenseViolationError):
        validate_training_startup_license(
            [DatasetEntry(name="AeroScapes", split="train", license="CC BY-SA 4.0")],
            html_path,
            expect_oem_total=None,
        )


def test_geometry_boundary_metrics_and_ood_helpers() -> None:
    # SkyScenes rotation matrices must be right-handed orthonormal (det=+1)
    for pitch in (0.0, 45.0, 60.0, 90.0):
        rot = skyscenes_cam_rot_wc_enu(pitch)
        assert rot.shape == (3, 3)
        assert np.allclose(rot.T @ rot, np.eye(3, dtype=np.float32), atol=1e-5)
        assert abs(float(np.linalg.det(rot)) - 1.0) < 1e-5
    # Nadir camera (+Z_c) points straight down (-Z_world)
    rot_nadir = skyscenes_cam_rot_wc_enu(90.0)
    assert np.allclose(rot_nadir[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float32), atol=1e-5)

    # CARLA 24-bit depth decode: RGB=(255,255,255) -> 1000m, (0,0,0) -> 0m
    rgb_depth = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb_depth[0, 0] = [255, 255, 255]
    d_m = decode_carla_depth_meters(rgb_depth, far_plane_m=1000.0)
    assert abs(float(d_m[0, 0]) - 1000.0) < 1e-3
    assert float(d_m[1, 1]) == 0.0

    # Boundary target detects class transitions and ignores 254/255 boundaries
    lbl = torch.zeros((2, 128, 128), dtype=torch.long)
    lbl[:, :, 64:] = 2
    lbl[:, :16, :16] = 255
    bdry = compute_boundary_target(lbl, out_hw=(64, 64))
    assert bdry.shape == (2, 1, 64, 64)
    assert float(bdry.max().item()) == 1.0
    assert float(bdry[:, :, 2, 2].max().item()) == 0.0

    # AnomalyMix works cleanly with CPU generator
    img = torch.randn(2, 3, 64, 64)
    mix_img, mix_lbl = apply_anomaly_mix(
        img, torch.zeros((2, 64, 64), dtype=torch.long), prob=1.0, generator=torch.Generator().manual_seed(7)
    )
    assert mix_img.shape == img.shape
    assert int((mix_lbl == 254).sum().item()) > 0

    # Exact AUROC & FPR@95TPR
    id_scores = np.linspace(-25.0, -15.0, 500)
    ood_scores = np.linspace(-6.0, -1.0, 200)
    scores = np.concatenate([id_scores, ood_scores])
    flags = np.concatenate([np.zeros_like(id_scores, dtype=bool), np.ones_like(ood_scores, dtype=bool)])
    auroc, fpr95 = compute_binary_auroc_and_fpr95(scores, flags)
    assert abs(auroc - 1.0) < 1e-6 and abs(fpr95 - 0.0) < 1e-6

    # Exact confusion metrics & False-Safe Rate (FSR)
    conf = np.array(
        [
            [80, 10, 5, 0, 5],
            [5, 90, 5, 0, 0],
            [10, 0, 80, 5, 5],   # 10 false-safe out of 100 true hazard
            [0, 5, 5, 90, 0],    # 5 false-safe out of 100 true hazard
            [0, 0, 10, 0, 90],   # 0 false-safe out of 100 true hazard
        ],
        dtype=np.int64,
    )
    m = compute_confusion_metrics(conf)
    assert abs(m["false_safe_rate_fsr"] - (15.0 / 300.0)) < 1e-6
    assert m["num_present_classes"] == 5 and 0.7 < m["miou_present_classes"] < 0.9


def test_v310_l4_distill_and_gtx1650s_trained_artifacts_contract() -> None:
    assert L4_METRICS_PATH.is_file(), f"Missing Colab L4 metrics artifact: {L4_METRICS_PATH}"
    assert GTX_PROOF_PATH.is_file(), f"Missing GTX 1650 SUPER trained proof artifact: {GTX_PROOF_PATH}"

    l4_raw = L4_METRICS_PATH.read_bytes()
    gtx_raw = GTX_PROOF_PATH.read_bytes()
    assert b"\r" not in l4_raw and b"\r" not in gtx_raw, "Artifacts must use pure LF line endings"
    assert not l4_raw.startswith(b"\xef\xbb\xbf") and not gtx_raw.startswith(b"\xef\xbb\xbf")

    l4_text = l4_raw.decode("utf-8")
    gtx_text = gtx_raw.decode("utf-8")
    for forbidden in ("UNTRAINED_WEIGHTS", "PENDING_COLAB", "Obsidian" + "Vault"):
        assert forbidden not in l4_text, f"Forbidden placeholder '{forbidden}' in L4 metrics"
        assert forbidden not in gtx_text, f"Forbidden placeholder '{forbidden}' in GTX proof"

    l4 = json.loads(l4_text)
    gtx = json.loads(gtx_text)

    # Colab L4 distillation provenance checks
    assert l4["weights_status"] == "TRAINED_COLAB_L4_DISTILLED_WEIGHTS"
    assert "L4" in l4["hardware"]["gpu"]["name"]
    assert l4["license_audit"]["zero_sharealike_verified"] is True
    assert l4["teachers"]["dinov2_license"] == "Apache-2.0"
    assert l4["teachers"]["dav2_license"] == "Apache-2.0"
    assert "Small" in l4["teachers"]["dav2_repo"]

    cal = l4["ood_calibration"]
    assert cal["T_ood"] == 1.0
    assert cal["kappa_b"] == 1.5
    assert 0.5 <= float(cal["T_prob"]) <= 3.0
    assert math.isfinite(float(cal["tau_ood"]))

    val_m = l4["evaluation"]["permissive_validation"]
    assert 0.0 < float(val_m["miou_present_classes"]) < 1.0
    assert 0.0 <= float(val_m["false_safe_rate_fsr"]) < 0.5
    assert int(val_m["total_eval_pixels"]) > 100_000

    ood_m = l4["evaluation"]["ood_detection"]
    assert 0.5 < float(ood_m["boundary_compensated_auroc"]) <= 1.0

    exp = l4["export"]
    assert int(exp["opset_version"]) >= 17
    assert exp["zero_adaptive_avg_pool2d"] is True
    assert exp["zero_dino_proj_nodes"] is True
    assert exp["zero_dynamic_shape_ops"] is True
    assert exp["onnx_checker_passed"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", exp["onnx_sha256"])

    # Physical GTX 1650 SUPER TU116 trained proof checks + SHA-256 parity
    assert gtx["weights_status"] == "TRAINED_COLAB_L4_DISTILLED_WEIGHTS"
    assert "GTX 1650 SUPER" in gtx["gpu"]["name"]
    assert gtx["model"]["onnx_sha256"] == exp["onnx_sha256"]
    assert gtx["model"]["onnx_sha256_verified_against_l4_metrics"] is True

    cg = gtx["benchmarks"]["ort_cuda_fp32_cuda_graph"]
    assert cg["gates"]["network_p50_lt_12ms"] is True
    assert float(cg["network_iobinding"]["p95_ms"]) < 12.0
    assert float(cg["network_iobinding"]["p99_ms"]) < 20.0
    assert cg["gates"]["e2e_ge_20hz"] is True
    assert cg["gates"]["vram_lt_300mb"] is True
    assert cg["outputs_finite"] is True
    assert gtx["all_local_gates_passed"] == (
        cg["gates"]["network_p50_lt_12ms"]
        and cg["gates"]["network_p99_lt_12ms"]
        and cg["gates"]["e2e_ge_20hz"]
        and cg["gates"]["vram_lt_300mb"]
        and cg["outputs_finite"]
    )
    assert cg["trained_output_sanity"]["all_outputs_finite"] is True
    assert int(cg["trained_output_sanity"]["distinct_predicted_classes"]) >= 3
    assert float(cg["trained_output_sanity"]["normal_l2_norm_max_err"]) < 1e-3


def test_v351_skyscenes_frame_id_dynamic_margins_fsr_gate_and_tu116_options() -> None:
    # 1) SkyScenes frame-ID stem matching across mismatched modality suffixes
    assert extract_skyscenes_frame_id("Images/Town01/008301_clrnoon.png") == "008301"
    assert extract_skyscenes_frame_id("Segment/Town01/008301_semsegCarla_clrnoon.png") == "008301"
    assert extract_skyscenes_frame_id("Depth/Town01/008301_depth_clrnoon.png") == "008301"
    assert extract_skyscenes_frame_id("readme_no_digits.png") is None

    # 2) Dynamic 5-class Stage-2 free-energy margins eliminate 19-class m_in=-23 gradient shock
    m_in, m_out = calibrate_dynamic_energy_margins(-5.654)
    assert -12.0 <= m_in <= -6.0
    assert abs(m_in - (-7.654)) < 1e-3
    assert abs(m_out - (-2.5)) < 1e-3
    assert m_in < m_out

    # 3) FSR-gated Stage-2 checkpoint selection rejects high-FSR regressions (> 0.08)
    cands = [
        {"epoch": 0, "val_miou": 0.42, "val_fsr": 0.053, "ood_auroc": 0.65},
        {"epoch": 1, "val_miou": 0.45, "val_fsr": 0.211, "ood_auroc": 0.85},  # high FSR must be rejected!
        {"epoch": 2, "val_miou": 0.43, "val_fsr": 0.060, "ood_auroc": 0.78},  # eligible & best combo
    ]
    idx, ok = select_stage2_checkpoint(cands, max_val_fsr=0.08)
    assert ok is True and idx == 2

    # Fallback picks minimum FSR when all exceed gate
    bad_cands = [
        {"epoch": 0, "val_miou": 0.45, "val_fsr": 0.14, "ood_auroc": 0.80},
        {"epoch": 1, "val_miou": 0.40, "val_fsr": 0.09, "ood_auroc": 0.70},
    ]
    idx_fb, ok_fb = select_stage2_checkpoint(bad_cands, max_val_fsr=0.08)
    assert ok_fb is False and idx_fb == 1

    # 4) Platform-tuned TU116 ORT CUDA options
    win_opts = resolve_tu116_cuda_provider_options(cuda_graph=True, algo_search="AUTO", platform_system="Windows")
    assert win_opts["arena_extend_strategy"] == "kSameAsRequested"
    assert win_opts["cudnn_conv_algo_search"] == "EXHAUSTIVE"
    assert win_opts["cudnn_conv_use_max_workspace"] == "1"
    assert win_opts["enable_cuda_graph"] == "1"

    lin_opts = resolve_tu116_cuda_provider_options(cuda_graph=True, algo_search="AUTO", platform_system="Linux")
    assert lin_opts["arena_extend_strategy"] == "kSameAsRequested"
    assert lin_opts["cudnn_conv_algo_search"] == "HEURISTIC"
    assert lin_opts["cudnn_conv_use_max_workspace"] == "0"
    assert lin_opts["enable_cuda_graph"] == "1"


def test_v351_l4_pidnet_v2_and_gtx1650s_proof_artifacts_contract() -> None:
    assert L4_V2_METRICS_PATH.is_file(), f"Missing v3.5.1 Colab L4 v2 metrics: {L4_V2_METRICS_PATH}"
    assert GTX_V2_PROOF_PATH.is_file(), f"Missing v3.5.1 GTX 1650 SUPER v2 proof: {GTX_V2_PROOF_PATH}"

    l4_raw = L4_V2_METRICS_PATH.read_bytes()
    gtx_raw = GTX_V2_PROOF_PATH.read_bytes()
    assert b"\r" not in l4_raw and b"\r" not in gtx_raw, "v3.5.1 artifacts must use pure LF"
    assert not l4_raw.startswith(b"\xef\xbb\xbf") and not gtx_raw.startswith(b"\xef\xbb\xbf")

    l4 = json.loads(l4_raw.decode("utf-8"))
    gtx = json.loads(gtx_raw.decode("utf-8"))

    # 1) Colab L4 v2 training provenance, fixed SkyScenes ingestion & FSR <= 0.08 gate
    assert l4["schema_version"] == "vertirta.slz_l4_distill_metrics/v3.5.1"
    assert l4["weights_status"] == "TRAINED_COLAB_L4_DISTILLED_WEIGHTS"
    assert "L4" in l4["hardware"]["gpu"]["name"]
    assert l4["license_audit"]["zero_sharealike_verified"] is True
    assert l4["datasets"]["train_samples_by_source"]["SkyScenes"] >= 150
    assert l4["datasets"]["val_samples_by_source"]["SkyScenes"] >= 50

    sky_src = next(s for s in l4["datasets"]["sources"] if s["source"] == "SkyScenes")
    assert float(sky_src["facade_remap_fraction_of_building"]) > 0.25

    val_m = l4["evaluation"]["permissive_validation"]
    assert float(val_m["miou_present_classes"]) > 0.42
    assert float(val_m["false_safe_rate_fsr"]) <= 0.08
    assert float(val_m["safe_precision"]) >= 0.85

    aero_m = l4["evaluation"]["quarantined_zero_shot_aeroscapes"]
    assert aero_m["used_in_training"] is False and aero_m["used_in_calibration"] is False
    assert float(aero_m["false_safe_rate_fsr"]) <= 0.05
    assert float(aero_m["safe_precision"]) >= 0.75

    s2 = l4["training"]["stage2_ood_oe"]
    assert s2["fsr_gate_satisfied"] is True
    assert -12.0 <= float(s2["m_in"]) <= -6.0
    assert float(s2["m_in"]) < float(s2["m_out"])

    cal = l4["ood_calibration"]
    assert cal["T_ood"] == 1.0 and cal["kappa_b"] == 1.5
    assert 0.5 <= float(cal["T_prob"]) <= 3.0
    assert math.isfinite(float(cal["tau_ood"]))

    exp = l4["export"]
    assert int(exp["opset_version"]) >= 17
    assert exp["zero_adaptive_avg_pool2d"] is True
    assert exp["zero_dino_proj_nodes"] is True
    assert exp["zero_dynamic_shape_ops"] is True
    assert exp["onnx_checker_passed"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", exp["onnx_sha256"])

    # 2) Physical GTX 1650 SUPER v2 proof: SHA-256 parity + p99 < 12.0 ms + all_local_gates_passed
    assert gtx["schema_version"] == "vertirta.local_gpu_trained_proof/v3.5.1"
    assert "GTX 1650 SUPER" in gtx["gpu"]["name"]
    assert gtx["model"]["onnx_sha256"] == exp["onnx_sha256"]
    assert gtx["model"]["onnx_sha256_verified_against_l4_metrics"] is True
    assert "ort_cuda_fp32_cuda_graph" in gtx["benchmarks"]
    assert "ort_cuda_wsl2_headless_fp32_cuda_graph" in gtx["benchmarks"]

    rec_key = gtx["recommended_local_tu116_mode"]
    rec = gtx["benchmarks"][rec_key]
    assert float(rec["network_iobinding"]["p50_ms"]) < 12.0
    assert float(rec["network_iobinding"]["p95_ms"]) < 12.0
    assert float(rec["network_iobinding"]["p99_ms"]) < 12.0
    assert rec["gates"]["e2e_ge_20hz"] is True
    assert rec["gates"]["vram_lt_300mb"] is True
    assert rec["outputs_finite"] is True
    assert rec["trained_output_sanity"]["all_outputs_finite"] is True
    assert int(rec["trained_output_sanity"]["distinct_predicted_classes"]) >= 3
    assert float(rec["trained_output_sanity"]["normal_l2_norm_max_err"]) < 1e-3
    assert gtx["all_local_gates_passed"] is True
