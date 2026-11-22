# SPDX-License-Identifier: Apache-2.0
"""Static 1x3x512x512 ONNX Opset 17+ & TensorRT FP16 export pipeline for PIDNet-S SLZ."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import onnx
import torch
import torch.nn as nn

from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain


def export_static_onnx(
    model: nn.Module,
    out_path: Path | str,
    opset: int = 17,
    *,
    simplify: bool = True,
    input_hw: tuple[int, int] = (512, 512),
) -> Path:
    """Export `PIDNetSlzDeployWrapper` to static (1, 3, H, W) ONNX Opset >= 17.

    Verifies:
      - Zero AdaptiveAvgPool2d modules (no TensorRT DDS subgraph splits)
      - Exact input name ["rgb_nchw"] & outputs ["seg_logits", "surf_normals", "bdry_prob"]
      - Zero train-only `dino_proj` nodes in the exported graph
      - Full `onnx.checker.check_model` validation (plus optional `onnxsim` pass)
    """
    if opset < 17:
        raise ValueError(f"VertiRTA requires ONNX opset >= 17, got {opset}")
    if any(isinstance(m, nn.AdaptiveAvgPool2d) for m in model.modules()):
        raise ValueError(
            "Model contains AdaptiveAvgPool2d; static PAPPM is mandatory for single-subgraph TRT FP16."
        )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        dummy = torch.randn(1, 3, int(input_hw[0]), int(input_hw[1]), device=device, dtype=torch.float32)

        export_kwargs: dict[str, object] = {
            "opset_version": int(opset),
            "do_constant_folding": True,
            "input_names": ["rgb_nchw"],
            "output_names": ["seg_logits", "surf_normals", "bdry_prob"],
            "dynamic_axes": None,
        }
        # Pin legacy TorchScript ONNX exporter when PyTorch 2.x exposes `dynamo` kwarg so
        # static shapes & exact IO names remain deterministic across torch 2.x versions.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        with torch.no_grad():
            torch.onnx.export(model, dummy, str(out_file), **export_kwargs)  # type: ignore[arg-type]
    finally:
        if was_training:
            model.train()

    onnx_model = onnx.load(str(out_file))
    if simplify:
        try:
            import onnxsim  # type: ignore[import-not-found]

            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx_model = simplified
                onnx.save(onnx_model, str(out_file))
        except ImportError:
            pass

    onnx.checker.check_model(onnx_model)
    if any("dino_proj" in node.name for node in onnx_model.graph.node):
        raise RuntimeError("Exported ONNX graph unexpectedly contains train-only dino_proj nodes.")

    return out_file


def build_trt_fp16_command(
    onnx_path: Path | str,
    engine_path: Path | str,
    *,
    workspace_mb: int = 1024,
) -> list[str]:
    """Return canonical `trtexec` command for static 1x3x512x512 FP16 TensorRT engine build."""
    return [
        "trtexec",
        f"--onnx={Path(onnx_path).as_posix()}",
        f"--saveEngine={Path(engine_path).as_posix()}",
        "--fp16",
        f"--memPoolSize=workspace:{int(workspace_mb)}M",
        "--builderOptimizationLevel=3",
        "--useCudaGraph",
        "--noDataTransfers",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PIDNet-S SLZ deploy model to ONNX Opset 17.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("results/pidnet_slz_512.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    net = PIDNetSlzTrain(num_classes=5, dino_dim=1024).eval()
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    deploy = PIDNetSlzDeployWrapper(net).eval()
    export_static_onnx(deploy, args.out, opset=args.opset, simplify=not args.no_simplify)


if __name__ == "__main__":
    main()
