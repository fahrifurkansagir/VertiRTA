# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Task 3: Static-PAPPM PIDNet-S, DINOv2+DAv2-S losses, OOD calibration, ONNX export."""

from __future__ import annotations

import json
from pathlib import Path

import onnx
import pytest
import torch

from ml.slz_perception.data.license_gate import DatasetEntry, LicenseViolationError
from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain
from ml.slz_perception.train.calibrate_ood import calibrate_temperature_and_tau
from ml.slz_perception.train.distill_losses import SlzDistillCriterion
from ml.slz_perception.train.export_onnx_trt import export_static_onnx
from ml.slz_perception.train.train_slz_distill import (
    Stage1Config,
    Stage2Config,
    apply_anomaly_mix,
    validate_training_startup_license,
)


def test_pidnet_static_pappm_onnx_export_and_losses(tmp_path: Path) -> None:
    torch.manual_seed(0)
    net = PIDNetSlzTrain(num_classes=5, dino_dim=1024).eval()
    assert not any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in net.modules())

    train_out = net(torch.randn(1, 3, 512, 512))
    assert train_out.seg_logits.shape == (1, 5, 512, 512)
    assert train_out.surf_normals.shape == (1, 3, 512, 512)
    assert train_out.bdry_prob.shape == (1, 1, 64, 64)
    assert train_out.dino_feats.shape == (1, 1024, 37, 37)

    deploy = PIDNetSlzDeployWrapper(net).eval()
    deploy_params = sum(p.numel() for p in deploy.parameters())
    assert 5_000_000 <= deploy_params <= 9_500_000, f"Unexpected PIDNet-S param count: {deploy_params}"

    with torch.no_grad():
        logits, normals, bdry = deploy(torch.randn(1, 3, 512, 512))
    assert logits.shape == (1, 5, 512, 512)
    assert normals.shape == (1, 3, 512, 512)
    assert bdry.shape == (1, 1, 64, 64)
    assert torch.allclose(torch.linalg.norm(normals, dim=1), torch.ones(1, 512, 512), atol=1e-4)
    assert float(bdry.min().item()) >= 0.0 and float(bdry.max().item()) <= 1.0

    onnx_path = export_static_onnx(deploy, tmp_path / "pidnet_slz_512.onnx", opset=17)
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    assert [i.name for i in m.graph.input] == ["rgb_nchw"]
    assert [o.name for o in m.graph.output] == ["seg_logits", "surf_normals", "bdry_prob"]
    assert not any("dino_proj" in n.name for n in m.graph.node)
    op_types = {n.op_type for n in m.graph.node}
    assert "Shape" not in op_types and "Gather" not in op_types, (
        f"Exported ONNX contains dynamic shape ops: {op_types & {'Shape', 'Gather'}}"
    )
    assert any(op.version >= 17 for op in m.opset_import if op.domain in ("", "ai.onnx"))

    in_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    assert in_dims == [1, 3, 512, 512]
    out_dims = [[d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output]
    assert out_dims == [[1, 5, 512, 512], [1, 3, 512, 512], [1, 1, 64, 64]]

    crit = SlzDistillCriterion(0.4, 0.2, 0.2, 0.1, m_in=-23.0, m_out=-5.0)
    lbl = torch.randint(0, 5, (2, 64, 64))
    lbl[:, :8, :8] = 254
    lbl[:, 8:12, :8] = 255
    seg_in = torch.randn(2, 5, 64, 64, requires_grad=True)
    norm_in = torch.nn.functional.normalize(torch.randn(2, 3, 64, 64), dim=1).requires_grad_(True)
    bdry_in = torch.sigmoid(torch.randn(2, 1, 64, 64)).requires_grad_(True)
    dino_in = torch.randn(2, 1024, 37, 37, requires_grad=True)
    loss = crit(
        seg_in,
        norm_in,
        bdry_in,
        dino_in,
        lbl,
        torch.ones(2, 3, 64, 64),
        torch.zeros(2, 1, 64, 64),
        torch.randn(2, 1024, 37, 37),
    )
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert seg_in.grad is not None and torch.isfinite(seg_in.grad).all()
    assert dino_in.grad is not None and torch.isfinite(dino_in.grad).all()

    cal = calibrate_temperature_and_tau(
        torch.randn(1000, 5, generator=torch.Generator().manual_seed(0)) * 3.0,
        torch.randint(0, 5, (1000,), generator=torch.Generator().manual_seed(1)),
        torch.zeros(1000),
    )
    assert cal.T_ood == 1.0 and 0.5 <= cal.T_prob <= 3.0 and cal.kappa_b == 1.5
    assert cal.nll_after <= cal.nll_before + 1e-6

    cal_json = cal.save_json(tmp_path / "ood_calibration.json")
    loaded = json.loads(cal_json.read_text(encoding="utf-8"))
    assert loaded["T_ood"] == 1.0 and loaded["kappa_b"] == 1.5 and "tau_ood" in loaded


def test_training_startup_license_gate_and_stage_ratios(tmp_path: Path) -> None:
    s1 = Stage1Config()
    s2 = Stage2Config()
    assert abs(sum(s1.source_sampling_weights.values()) - 1.0) < 1e-9
    assert abs(sum(s2.source_sampling_weights.values()) - 1.0) < 1e-9
    real_s1 = (
        s1.source_sampling_weights["FLAIR-1"]
        + s1.source_sampling_weights["OpenEarthMap-41"]
        + s1.source_sampling_weights["FloodNet-Supervised-v1.0"]
    )
    synth_s1 = (
        s1.source_sampling_weights["SkyScenes"]
        + s1.source_sampling_weights["Gazebo_Istanbul_8K"]
    )
    assert abs(real_s1 - 0.60) < 1e-9 and abs(synth_s1 - 0.40) < 1e-9

    # Tainted training manifest MUST be rejected before any training starts
    with pytest.raises(LicenseViolationError):
        validate_training_startup_license(
            [DatasetEntry(name="RescueNet", split="train", license="CC BY-NC-ND 4.0")],
            tmp_path / "missing.html",
        )

    # AnomalyMix injects label=254 patches when prob=1.0
    img = torch.zeros(2, 3, 64, 64)
    lbl = torch.zeros(2, 64, 64, dtype=torch.long)
    mixed_img, mixed_lbl = apply_anomaly_mix(img, lbl, prob=1.0, generator=torch.Generator().manual_seed(0))
    assert mixed_img.shape == img.shape
    assert int((mixed_lbl == 254).sum().item()) > 0
