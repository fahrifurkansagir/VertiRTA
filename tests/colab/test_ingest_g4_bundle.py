# SPDX-License-Identifier: Apache-2.0
"""Contract & adversarial unit tests for scripts/colab/ingest_g4_bundle.py (0.00 CU)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.ingest_g4_bundle import (  # noqa: E402
    REQUIRED_G4_JOB_JSONS,
    build_argparser,
    ingest_g4_bundles,
    main as ingest_main,
    safe_extract_tar_gz,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _add_bytes_to_tar(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    tf.addfile(ti, io.BytesIO(data))


def _build_synthetic_g4_archives(
    root: Path,
    *,
    include_whisper_in_manifest: bool = False,
    include_whisper_file: bool = True,
    tamper_onnx_blob: bool = False,
    tamper_blob_relpath: str | None = None,
    omit_blob_relpaths: tuple[str, ...] = (),
    tamper_manifest_json: bool = False,
    inject_nan_json: bool = False,
    inject_secret_leak: bool = False,
    oversized_json: bool = False,
    include_realworld_tar: bool = True,
) -> tuple[Path, Path, Path | None]:
    blobs_payloads: dict[str, bytes] = {
        "pidnet_slz_v21_opset17.onnx": b"ONNX_PIDNET_V21_BYTES_0123456789",
        "mlsd_wireseg_640_opset17.onnx": b"ONNX_MLSD_WIRESEG_BYTES_0123456789",
        "dfine_n_640_opset17.onnx": b"ONNX_DFINE_N_BYTES_0123456789",
        "xfeat_512_opset17.onnx": b"ONNX_XFEAT_512_BYTES_0123456789",
        "ood_calibration_v21.json": b'{"schema_version":"v21","temp":1.1}\n',
        "cbvf_4d_x8_240_v4_from_101x71.vrta_lut": b"VRTALUT1_X8_240_BYTES",
        "cbvf_4d_x8_240_hires_101x71.vrta_lut_hires": b"VRTALUT2_X8_HIRES_BYTES",
        "cel_2d_x8_240.vrta_cel2d": b"VRTACEL2D_X8_BYTES",
        "qwen3_4b_v3_lora/adapter_config.json": b'{"r":32}\n',
        "whisper_small_bvi_lora/adapter_config.json": b'{"r":32,"task":"asr"}\n',
        "whisper_small_bvi_lora/adapter_model.safetensors": b"WHISPER_LORA_WEIGHTS_012345",
    }

    onnx_sha = {k: _sha256_bytes(v) for k, v in blobs_payloads.items()}

    json_payloads: dict[str, bytes] = {
        "g4_qwen3_v3_vertibench_v2_eval.json": (
            json.dumps({"job": "qwen3_v3", "ast_pct": 99.1, "z3_sat_pct": 100.0}) + "\n"
        ).encode("utf-8"),
        "g4_forward_wire_metrics.json": (
            json.dumps(
                {
                    "job": "forward_wire",
                    "onnx_exports": {
                        "mlsd_wireseg": {"sha256": onnx_sha["mlsd_wireseg_640_opset17.onnx"]},
                        "dfine_n": {"sha256": onnx_sha["dfine_n_640_opset17.onnx"]},
                    },
                }
            )
            + "\n"
        ).encode("utf-8"),
        "g4_jax_4d_hires_and_lc_manifest.json": (
            json.dumps({"job": "jax_4d_hires_lc", "soundness_violations": 0}) + "\n"
        ).encode("utf-8"),
        "g4_pidnet_v21_xfeat_metrics.json": (
            json.dumps(
                {
                    "job": "pidnet_v21_xfeat",
                    "onnx_exports": {
                        "pidnet_slz_v21": {"sha256": onnx_sha["pidnet_slz_v21_opset17.onnx"]},
                        "xfeat_512": {"sha256": onnx_sha["xfeat_512_opset17.onnx"]},
                    },
                }
            )
            + "\n"
        ).encode("utf-8"),
        "g4_thunder_telemetry.json": (
            json.dumps(
                {
                    "actual_gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                    "actual_gpu_total_mib": 97887.0,
                    "by_wave": {
                        "wave1b_parallel_grpo": {"peak_memory_used_mib": 68981.0, "mean_gpu_util_pct": 96.0},
                        "wave2": {"peak_memory_used_mib": 80120.0, "mean_gpu_util_pct": 94.0},
                    },
                }
            )
            + "\n"
        ).encode("utf-8"),
    }

    whisper_bytes = (json.dumps({"job": "whisper_bvi", "wer_tr": 0.08, "wer_en": 0.05}) + "\n").encode("utf-8")
    if inject_nan_json:
        whisper_bytes = b'{"job": "whisper_bvi", "wer_tr": NaN}\n'
    elif inject_secret_leak:
        fake_tok = "ghp" + "_" + ("A" * 24)
        whisper_bytes = (json.dumps({"job": "whisper_bvi", "leaked": fake_tok}) + "\n").encode("utf-8")
    elif oversized_json:
        whisper_bytes = (json.dumps({"job": "whisper_bvi", "pad": "x" * 1_050_000}) + "\n").encode("utf-8")

    manifest_json_entries: list[dict[str, Any]] = []
    for fname, data in sorted(json_payloads.items()):
        manifest_json_entries.append(
            {"filename": fname, "size_bytes": len(data), "sha256": _sha256_bytes(data)}
        )
    if include_whisper_file and include_whisper_in_manifest:
        manifest_json_entries.append(
            {
                "filename": "g4_whisper_bvi_lora_eval.json",
                "size_bytes": len(whisper_bytes),
                "sha256": _sha256_bytes(whisper_bytes),
            }
        )

    manifest_blob_entries = [
        {"relpath": rel, "size_bytes": len(data), "sha256": _sha256_bytes(data)}
        for rel, data in sorted(blobs_payloads.items())
    ]

    manifest_bytes = (
        json.dumps(
            {
                "schema_version": "vertirta.remote_g4_bundle_manifest/v4.0.0",
                "state": "DONE",
                "actual_gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "actual_gpu_total_mib": 97887.0,
                "json_artifacts": manifest_json_entries,
                "blob_artifacts": manifest_blob_entries,
            }
        )
        + "\n"
    ).encode("utf-8")

    artifacts_tar = root / "vertirta_v400_artifacts.tar.gz"
    with tarfile.open(artifacts_tar, "w:gz") as tf:
        for fname, data in sorted(json_payloads.items()):
            out_data = data + b" " if (tamper_manifest_json and fname == "g4_forward_wire_metrics.json") else data
            _add_bytes_to_tar(tf, f"artifacts/{fname}", out_data)
        if include_whisper_file:
            _add_bytes_to_tar(tf, "artifacts/g4_whisper_bvi_lora_eval.json", whisper_bytes)
        _add_bytes_to_tar(tf, "artifacts/remote_bundle_manifest.json", manifest_bytes)
        # Transient files that MUST NOT enter git artifacts/v4.0.0/
        _add_bytes_to_tar(tf, "artifacts/heartbeat.json", b'{"state":"DONE"}\n')
        _add_bytes_to_tar(tf, "artifacts/logs/qwen3_v3_sft.log", b"raw log line\n")

    omit_set = set(omit_blob_relpaths)
    blobs_tar = root / "vertirta_v400_blobs.tar.gz"
    with tarfile.open(blobs_tar, "w:gz") as tf:
        for rel, data in sorted(blobs_payloads.items()):
            if rel in omit_set:
                continue
            out_data = data
            if tamper_onnx_blob and rel == "pidnet_slz_v21_opset17.onnx":
                out_data = b"CORRUPTED_ONNX_BYTES"
            elif tamper_blob_relpath is not None and rel == tamper_blob_relpath:
                out_data = data + b"_TAMPERED"
            _add_bytes_to_tar(tf, f"blobs/{rel}", out_data)
        # Scratch _work dir MUST be skipped
        _add_bytes_to_tar(tf, "blobs/_work_sft_tmp/temp.pt", b"scratch")

    realworld_tar: Path | None = None
    if include_realworld_tar:
        realworld_tar = root / "vertirta_v400_realworld_eval.tar.gz"
        with tarfile.open(realworld_tar, "w:gz") as tf:
            _add_bytes_to_tar(
                tf,
                "realworld_eval/cell5_summary.json",
                (json.dumps({"qwen3_bf16_sat_pct": 100.0, "whisper_fleurs_wer": 0.06}) + "\n").encode("utf-8"),
            )
            _add_bytes_to_tar(tf, "realworld_eval/vertirta_hud_slz_real_aerial.mp4", b"\x00\x00\x00\x18ftypmp42")

    return artifacts_tar, blobs_tar, realworld_tar


def test_ingest_g4_bundles_end_to_end_including_cell45_whisper_and_cell5_realworld(tmp_path: Path) -> None:
    """Happy path: Cell 4 manifest + post-Cell-4 Cell 4.5 whisper JSON + Cell 5 real-world tar."""
    src_dir = tmp_path / "downloads"
    src_dir.mkdir()
    art_tar, blobs_tar, rw_tar = _build_synthetic_g4_archives(
        src_dir,
        include_whisper_in_manifest=False,  # Matches real Cell 4 -> Cell 4.5 sequence!
        include_whisper_file=True,
        include_realworld_tar=True,
    )
    dest_art = tmp_path / "repo_artifacts"
    dest_blobs = tmp_path / "var_tmp_blobs"

    receipt = ingest_g4_bundles(
        artifacts_tar=art_tar,
        blobs_tar=blobs_tar,
        realworld_tar=rw_tar,
        dest_artifacts_dir=dest_art,
        dest_blobs_dir=dest_blobs,
    )

    assert receipt["all_gates_passed"] is True
    assert receipt["colab_cu_spent"] == 0.0
    assert receipt["onnx_parity_verified"] is True
    assert receipt["public_hygiene_passed"] is True
    assert receipt["missing_required_jsons"] == []

    # All 5 job JSONs + telemetry + manifest + Cell 5 summary installed
    for req in REQUIRED_G4_JOB_JSONS:
        assert (dest_art / req).is_file(), f"Missing installed job JSON: {req}"
    assert (dest_art / "g4_realworld_cell5_summary.json").is_file()

    # Transient heartbeat, raw logs, MP4s, and _work scratch MUST stay out of git artifacts dir
    assert not (dest_art / "heartbeat.json").exists()
    assert not (dest_art / "logs").exists()
    assert list(dest_art.rglob("*.mp4")) == []
    assert list(dest_art.rglob("*.onnx")) == []

    # Blobs, Cell 5 MP4, and SHA256SUMS.txt live outside git in dest_blobs
    assert (dest_blobs / "pidnet_slz_v21_opset17.onnx").is_file()
    assert not (dest_blobs / "_work_sft_tmp").exists()
    assert (dest_blobs / "realworld_demos" / "vertirta_hud_slz_real_aerial.mp4").is_file()

    sums_path = dest_blobs / "SHA256SUMS.txt"
    assert sums_path.is_file()
    sums_raw = sums_path.read_bytes()
    assert b"\r" not in sums_raw and not sums_raw.startswith(b"\xef\xbb\xbf")
    lines = [ln for ln in sums_raw.decode("utf-8").splitlines() if ln.strip()]
    assert any(ln.endswith("  pidnet_slz_v21_opset17.onnx") for ln in lines)
    assert any(ln.endswith("  realworld_demos/vertirta_hud_slz_real_aerial.mp4") for ln in lines)
    for ln in lines:
        digest, rel = ln.split("  ", 1)
        assert len(digest) == 64
        assert _sha256_bytes((dest_blobs / rel).read_bytes()) == digest


def test_ingest_rejects_tar_slip_and_symlinks_fail_closed(tmp_path: Path) -> None:
    evil_tar = tmp_path / "evil_slip.tar.gz"
    with tarfile.open(evil_tar, "w:gz") as tf:
        _add_bytes_to_tar(tf, "../escape.json", b"{}\n")
    with pytest.raises(ValueError, match="Path traversal"):
        safe_extract_tar_gz(evil_tar, tmp_path / "out1")

    symlink_tar = tmp_path / "evil_sym.tar.gz"
    with tarfile.open(symlink_tar, "w:gz") as tf:
        ti = tarfile.TarInfo(name="artifacts/link.json")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)
    with pytest.raises(ValueError, match="Unsafe tar member type"):
        safe_extract_tar_gz(symlink_tar, tmp_path / "out2")


def test_ingest_rejects_tampered_onnx_or_manifest_json_and_leaves_dest_untouched(tmp_path: Path) -> None:
    d1 = tmp_path / "case_onnx"
    d1.mkdir()
    art1, blob1, _ = _build_synthetic_g4_archives(d1, tamper_onnx_blob=True, include_realworld_tar=False)
    dest_art1 = tmp_path / "out_art1"
    dest_blob1 = tmp_path / "out_blob1"
    with pytest.raises(ValueError, match="SHA-256"):
        ingest_g4_bundles(
            artifacts_tar=art1,
            blobs_tar=blob1,
            dest_artifacts_dir=dest_art1,
            dest_blobs_dir=dest_blob1,
        )
    assert not dest_art1.exists()
    assert not dest_blob1.exists()

    d2 = tmp_path / "case_json"
    d2.mkdir()
    art2, blob2, _ = _build_synthetic_g4_archives(d2, tamper_manifest_json=True, include_realworld_tar=False)
    with pytest.raises(ValueError, match="SHA-256"):
        ingest_g4_bundles(
            artifacts_tar=art2,
            blobs_tar=blob2,
            dest_artifacts_dir=tmp_path / "out_art2",
            dest_blobs_dir=tmp_path / "out_blob2",
        )


def test_ingest_rejects_nan_secret_leak_oversized_json_and_missing_required_job(tmp_path: Path) -> None:
    for flag_name, pattern in (
        ("inject_nan_json", "Non-finite"),
        ("inject_secret_leak", "hygiene"),
        ("oversized_json", "1 MB"),
    ):
        case_dir = tmp_path / flag_name
        case_dir.mkdir()
        art, blob, _ = _build_synthetic_g4_archives(
            case_dir,
            include_realworld_tar=False,
            **{flag_name: True},
        )
        with pytest.raises(ValueError, match=pattern):
            ingest_g4_bundles(
                artifacts_tar=art,
                blobs_tar=blob,
                dest_artifacts_dir=tmp_path / f"art_{flag_name}",
                dest_blobs_dir=tmp_path / f"blob_{flag_name}",
            )

    missing_dir = tmp_path / "missing_whisper"
    missing_dir.mkdir()
    art_m, blob_m, _ = _build_synthetic_g4_archives(
        missing_dir,
        include_whisper_file=False,
        include_realworld_tar=False,
    )
    with pytest.raises(FileNotFoundError, match="Incomplete G4 bundle"):
        ingest_g4_bundles(
            artifacts_tar=art_m,
            blobs_tar=blob_m,
            dest_artifacts_dir=tmp_path / "art_missing",
            dest_blobs_dir=tmp_path / "blob_missing",
            allow_partial=False,
        )


def test_ingest_missing_non_onnx_blobs_fail_closed_by_default_and_opt_in_guards_core(tmp_path: Path) -> None:
    """Bug 3: Fail-closed by default; --allow-missing-non-onnx-blobs permits ONLY missing optional blobs."""
    # Verify CLI flag defaults to False (fail-closed)
    default_args = build_argparser().parse_args(["--artifacts-tar", "a.tar.gz", "--blobs-tar", "b.tar.gz"])
    assert default_args.allow_missing_non_onnx_blobs is False

    missing_lora = (
        "whisper_small_bvi_lora/adapter_config.json",
        "whisper_small_bvi_lora/adapter_model.safetensors",
    )
    src_lora = tmp_path / "missing_whisper_lora"
    src_lora.mkdir()
    art_tar, blobs_tar, _ = _build_synthetic_g4_archives(
        src_lora,
        omit_blob_relpaths=missing_lora,
        include_realworld_tar=False,
    )

    # 1) Default (fail-closed) MUST reject missing optional LoRA blobs
    dest_fail_art = tmp_path / "fail_art"
    dest_fail_blobs = tmp_path / "fail_blobs"
    with pytest.raises(FileNotFoundError, match="whisper_small_bvi_lora"):
        ingest_g4_bundles(
            artifacts_tar=art_tar,
            blobs_tar=blobs_tar,
            dest_artifacts_dir=dest_fail_art,
            dest_blobs_dir=dest_fail_blobs,
        )
    assert not dest_fail_art.exists()
    assert not dest_fail_blobs.exists()

    # 2) Explicit CLI --allow-missing-non-onnx-blobs succeeds & transparently records missing optional blobs
    dest_ok_art = tmp_path / "ok_art"
    dest_ok_blobs = tmp_path / "ok_blobs"
    rc = ingest_main(
        [
            "--artifacts-tar",
            str(art_tar),
            "--blobs-tar",
            str(blobs_tar),
            "--dest-artifacts-dir",
            str(dest_ok_art),
            "--dest-blobs-dir",
            str(dest_ok_blobs),
            "--allow-missing-non-onnx-blobs",
        ]
    )
    assert rc == 0
    receipt = json.loads((dest_ok_blobs / "g4_ingestion_receipt.json").read_text(encoding="utf-8"))
    assert receipt["all_gates_passed"] is True
    assert receipt["allow_missing_non_onnx_blobs"] is True
    assert receipt["partial_blob_sync_detected"] is True
    assert receipt["missing_optional_blobs"] == sorted(missing_lora)
    assert receipt["onnx_parity_verified"] is True

    # 3) Even WITH --allow-missing-non-onnx-blobs, missing ANY core ONNX/calibration/LUT/CEL2D MUST fail closed
    for core_rel in (
        "xfeat_512_opset17.onnx",
        "ood_calibration_v21.json",
        "cbvf_4d_x8_240_v4_from_101x71.vrta_lut",
        "cbvf_4d_x8_240_hires_101x71.vrta_lut_hires",
        "cel_2d_x8_240.vrta_cel2d",
    ):
        case_dir = tmp_path / f"missing_core_{core_rel.replace('.', '_')}"
        case_dir.mkdir()
        art_c, blob_c, _ = _build_synthetic_g4_archives(
            case_dir,
            omit_blob_relpaths=(core_rel,),
            include_realworld_tar=False,
        )
        with pytest.raises(FileNotFoundError, match="core required blob"):
            ingest_g4_bundles(
                artifacts_tar=art_c,
                blobs_tar=blob_c,
                dest_artifacts_dir=tmp_path / f"out_art_{core_rel}",
                dest_blobs_dir=tmp_path / f"out_blob_{core_rel}",
                allow_missing_non_onnx_blobs=True,
            )

    # 4) Even WITH --allow-missing-non-onnx-blobs, SHA-256 mismatch on any present optional blob MUST fail closed
    tamper_dir = tmp_path / "tamper_present_lora"
    tamper_dir.mkdir()
    art_t, blob_t, _ = _build_synthetic_g4_archives(
        tamper_dir,
        omit_blob_relpaths=missing_lora,
        tamper_blob_relpath="qwen3_4b_v3_lora/adapter_config.json",
        include_realworld_tar=False,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        ingest_g4_bundles(
            artifacts_tar=art_t,
            blobs_tar=blob_t,
            dest_artifacts_dir=tmp_path / "out_art_tamper",
            dest_blobs_dir=tmp_path / "out_blob_tamper",
            allow_missing_non_onnx_blobs=True,
        )
