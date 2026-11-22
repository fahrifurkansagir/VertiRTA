# SPDX-License-Identifier: Apache-2.0
"""Distillation losses, 2-stage Colab G4 trainer, OOD calibration, and ONNX/TRT export."""

from ml.slz_perception.train.calibrate_ood import (
    OodCalibration,
    calibrate_temperature_and_tau,
    compute_free_energy,
)
from ml.slz_perception.train.distill_losses import SlzDistillCriterion
from ml.slz_perception.train.export_onnx_trt import build_trt_fp16_command, export_static_onnx

__all__ = [
    "OodCalibration",
    "SlzDistillCriterion",
    "build_trt_fp16_command",
    "calibrate_temperature_and_tau",
    "compute_free_energy",
    "export_static_onnx",
]
