# SPDX-License-Identifier: Apache-2.0
"""Tests for 0.00 CU Local GTX 1650 SUPER PIDNet-S v2.1 Recovery Pipeline & Sealed Artifact Integrity."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs.common import sha256_file  # noqa: E402
from scripts.train_local_gtx1650s_pidnet_v21_recovery import (  # noqa: E402
    CAL_FILENAME,
    ONNX_FILENAME,
    sanitize_artifact_path,
)
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_local_gtx1650s_pidnet_v21_recovery.py"
SEALED_METRICS_PATH = (
    REPO_ROOT / "artifacts" / "v4.0.0" / "local_gtx1650s_pidnet_v21_recovery_metrics.json"
)
G4_HISTORICAL_METRICS_PATH = (
    REPO_ROOT / "artifacts" / "v4.0.0" / "g4_pidnet_v21_xfeat_metrics.json"
)

PERSONAL_PATH_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"[A-Za-z]:[\\/]" + r"Users[\\/]", re.IGNORECASE),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"Obsidian" + r"Vault", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])one" + r"ra(?![A-Za-z0-9_])", re.IGNORECASE),
)


def _assert_finite_tree(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float {obj!r} at {path}"
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_tree(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite_tree(v, f"{path}[{i}]")


def _assert_no_personal_paths(text: str, label: str) -> None:
    for pat in PERSONAL_PATH_RES:
        m = pat.search(text)
        assert m is None, f"Personal path leak {m.group(0)!r} found in {label}"


def test_recovery_script_pure_utf8_lf_and_zero_personal_paths() -> None:
    raw = SCRIPT_PATH.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "UTF-8 BOM forbidden"
    assert b"\r" not in raw, "CRLF forbidden in recovery script"
    text = raw.decode("utf-8")
    _assert_no_personal_paths(text, SCRIPT_PATH.name)
    assert sanitize_artifact_path(Path("/var/tmp/vertirta_v400_blobs/test.onnx")) == (
        "/var/tmp/vertirta_v400_blobs/test.onnx"
    )
    assert sanitize_artifact_path(SEALED_METRICS_PATH) == (
        "artifacts/v4.0.0/local_gtx1650s_pidnet_v21_recovery_metrics.json"
    )


def test_recovery_smoke_cpu_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    blob_dir = tmp_path / "blobs"
    out_json = tmp_path / "smoke_recovery_metrics.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--smoke-cpu",
            "--blob-dir",
            str(blob_dir),
            "--out-json",
            str(out_json),
        ],
        check=True,
    )

    raw_bytes = out_json.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw_bytes
    raw_text = raw_bytes.decode("utf-8")
    _assert_no_personal_paths(raw_text, out_json.name)

    payload = json.loads(raw_text)
    _assert_finite_tree(payload)

    # Verify canonical SHA-256 seal (exact match with write_canonical_json)
    reported_sha = payload["report_sha256"]
    body = {k: v for k, v in payload.items() if k != "report_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert reported_sha == expected_sha

    assert payload["mode"] == "SMOKE_CPU_ONLY"
    assert payload["cu_consumed"] == 0.0
    assert payload["license_audit"]["zero_sharealike_verified"] is True
    assert payload["license_audit"]["zero_nc_sa_gpl_verified"] is True
    assert payload["skyscenes_nested_tar_filter_active"] is True
    assert payload["training"]["all_losses_finite"] is True
    assert payload["training"]["pareto_selection"]["selected"]["epoch"] == 18.0
    assert payload["training"]["pareto_selection"]["degenerate_rejected_count"] == 1
    assert payload["ood_calibration"]["safe_logit_bias_calibration"]["fsr_gate_met"] is True

    onnx_path = blob_dir / ONNX_FILENAME
    cal_path = blob_dir / CAL_FILENAME
    assert onnx_path.is_file()
    assert cal_path.is_file()
    assert payload["onnx_export"]["deploy_parameters"] == 8003817
    assert payload["onnx_export"]["onnx_checker_passed"] is True
    assert payload["onnx_export"]["sha256"] == sha256_file(onnx_path)
    assert payload["ood_calibration"]["sha256"] == sha256_file(cal_path)


def test_sealed_live_recovery_artifact_integrity_when_present() -> None:
    if not SEALED_METRICS_PATH.is_file():
        pytest.skip("Live GTX 1650 SUPER recovery metrics JSON not yet generated.")

    raw_bytes = SEALED_METRICS_PATH.read_bytes()
    assert len(raw_bytes) < 1024 * 1024, "Metrics artifact must be strictly < 1 MB"
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw_bytes
    raw_text = raw_bytes.decode("utf-8")
    _assert_no_personal_paths(raw_text, SEALED_METRICS_PATH.name)

    payload = json.loads(raw_text)
    _assert_finite_tree(payload)

    # Verify canonical report_sha256 seal (exact match with write_canonical_json)
    reported_sha = payload["report_sha256"]
    body = {k: v for k, v in payload.items() if k != "report_sha256"}
    expected_sha = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert reported_sha == expected_sha

    assert payload["mode"] == "LIVE_LOCAL_GTX1650S_CUDA_FP16"
    assert payload["cu_consumed"] == 0.0
    assert payload["license_audit"]["zero_sharealike_verified"] is True
    assert payload["onnx_export"]["deploy_parameters"] == 8003817
    assert payload["onnx_export"]["onnx_checker_passed"] is True

    # Non-degenerate Pareto selection + low-FSR bias calibration verified
    sel = payload["training"]["pareto_selection"]["selected"]
    assert float(sel["val_miou"]) >= 0.25
    assert float(payload["permissive_val_metrics_raw"]["miou_present_classes"]) >= 0.25
    assert float(payload["permissive_val_metrics_bias_calibrated"]["false_safe_rate_fsr"]) <= 0.035

    # Quarantined AeroScapes zero-shot strictly untouched in training/calibration
    for key in (
        "quarantined_zero_shot_aeroscapes_raw",
        "quarantined_zero_shot_aeroscapes_bias_calibrated",
    ):
        assert payload[key]["used_in_training"] is False
        assert payload[key]["used_in_calibration"] is False

    # Historical G4 ablation artifact remains untouched
    if G4_HISTORICAL_METRICS_PATH.is_file():
        g4 = json.loads(G4_HISTORICAL_METRICS_PATH.read_text(encoding="utf-8"))
        assert g4["pidnet_v21_training"]["pareto_selection"]["selected"]["epoch"] == 2.0

    # Head-to-head comparison present and consistent
    h2h = payload["head_to_head_comparison"]
    assert "l4_v2_distilled" in h2h
    assert "g4_v21_collapsed_epoch2" in h2h
    assert "local_gtx1650s_v21_recovery_raw" in h2h
    assert "local_gtx1650s_v21_recovery_bias_calibrated" in h2h
