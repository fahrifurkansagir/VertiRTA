#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail-Closed Colab G4 Drive Bundle Ingestor, Verifier & Release SHA256SUMS Builder.

Unpacks browser-downloaded Colab G4 tarballs (`vertirta_v400_artifacts.tar.gz`,
`vertirta_v400_blobs.tar.gz`, and optional `vertirta_v400_realworld_eval.tar.gz`) into:
  - Git-tracked small canonical JSON evidence (`artifacts/v4.0.0/`, strictly < 1 MB,
    pure UTF-8 LF, zero NaN/Inf, zero leaked personal paths/secrets, zero raw logs/MP4s)
  - Out-of-git binary release cache (`/var/tmp/vertirta_v400_blobs/`, containing the 4
    Opset-17 ONNX models, calibration JSONs, VRTALUT1/2 binaries, LoRA adapters, optional
    Cell 5 MP4 demos, and a reproducible `SHA256SUMS.txt` manifest for GitHub Releases)

Enforces strict security and integrity gates before writing to final destinations:
  1. Tar-slip / symlink / hardlink / device rejection on every archive member.
  2. Fail-closed SHA-256 verification against `remote_bundle_manifest.json` for every
     manifest-recorded JSON and blob artifact (while cleanly validating post-Cell-4
     additions such as Cell 4.5 `g4_whisper_bvi_lora_eval.json` and Cell 5 real-world JSONs).
  3. Fail-closed SHA-256 parity check of all 4 ONNX models via
     `scripts.bench_gtx1650s_m3_timesliced.verify_onnx_sha256_parity`.
  4. Strict JSON finiteness (`allow_nan=False`), `< 1 MB` git artifact cap, and
     `scripts.ci.check_public_hygiene.scan_public_hygiene` zero-violation gate.
  5. Zero Colab compute units (`0.00 CU`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bench_gtx1650s_m3_timesliced import verify_onnx_sha256_parity  # noqa: E402
from scripts.ci.check_public_hygiene import scan_public_hygiene  # noqa: E402
from scripts.colab.g4_jobs import G4_JOB_REGISTRY  # noqa: E402
from scripts.colab.g4_jobs.common import sha256_file, write_canonical_json  # noqa: E402

SCHEMA_VERSION = "vertirta.g4_bundle_ingestion_receipt/v1.0.0"
MAX_GIT_JSON_BYTES = 1_000_000

REQUIRED_G4_JOB_JSONS: tuple[str, ...] = tuple(
    sorted(str(spec["out_json_name"]) for spec in G4_JOB_REGISTRY.values())
)
REQUIRED_ALL_JSONS: tuple[str, ...] = tuple(
    sorted((*REQUIRED_G4_JOB_JSONS, "g4_thunder_telemetry.json", "remote_bundle_manifest.json"))
)

EXCLUDED_ARTIFACT_JSONS: frozenset[str] = frozenset(
    {
        "heartbeat.json",
    }
)

CORE_ONNX_FILENAMES: frozenset[str] = frozenset(
    {
        "pidnet_slz_v21_opset17.onnx",
        "mlsd_wireseg_640_opset17.onnx",
        "dfine_n_640_opset17.onnx",
        "xfeat_512_opset17.onnx",
    }
)

CORE_CALIBRATION_FILENAMES: frozenset[str] = frozenset(
    {
        "ood_calibration_v21.json",
    }
)


def _is_core_required_blob(relpath: str) -> bool:
    """Return True if a manifest blob is a core edge/safety binary that must never be skipped."""
    name = PurePosixPath(relpath.replace("\\", "/")).name
    lower = name.lower()
    if name in CORE_ONNX_FILENAMES or lower.endswith(".onnx"):
        return True
    if name in CORE_CALIBRATION_FILENAMES:
        return True
    if ".vrta_lut" in lower or lower.endswith(".vrta_cel2d"):
        return True
    return False


def _normalize_tar_member_path(member: tarfile.TarInfo) -> PurePosixPath | None:
    """Validate TarInfo against path traversal, absolute paths, symlinks, hardlinks, and devices."""
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise ValueError(f"Unsafe tar member type rejected (link/device/fifo): {member.name!r}")

    raw_name = member.name.replace("\\", "/")
    if raw_name.startswith("/") or raw_name.startswith("~"):
        raise ValueError(f"Absolute or home-relative tar member path rejected: {member.name!r}")

    posix = PurePosixPath(raw_name)
    parts: list[str] = []
    for part in posix.parts:
        if part in ("", "."):
            continue
        if part == ".." or ":" in part:
            raise ValueError(f"Path traversal or drive-letter tar member rejected: {member.name!r}")
        parts.append(part)

    if not parts:
        return None
    return PurePosixPath(*parts)


def _strip_optional_root_prefix(rel: PurePosixPath, known_prefixes: tuple[str, ...]) -> PurePosixPath:
    """Strip a single wrapper folder like `artifacts/`, `blobs/`, or `realworld_eval/` when present."""
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in known_prefixes:
        return PurePosixPath(*parts[1:])
    return rel


def safe_extract_tar_gz(
    tar_path: Path,
    dest_dir: Path,
    *,
    strip_prefixes: tuple[str, ...] = (),
    skip_work_dirs: bool = False,
) -> list[Path]:
    """Safely extract regular files from a `.tar.gz` archive with strict tar-slip protection."""
    if not tar_path.is_file():
        raise FileNotFoundError(f"Missing archive: {tar_path}")

    dest_root = dest_dir.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            rel = _normalize_tar_member_path(member)
            if rel is None:
                continue
            if strip_prefixes:
                rel = _strip_optional_root_prefix(rel, strip_prefixes)
            if skip_work_dirs and any(p.startswith("_work") or p.startswith(".") for p in rel.parts):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported non-regular tar member rejected: {member.name!r}")

            out_path = (dest_root / Path(*rel.parts)).resolve()
            try:
                out_path.relative_to(dest_root)
            except ValueError as exc:
                raise ValueError(f"Tar member escapes destination root: {member.name!r}") from exc

            out_path.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            if src is None:
                raise ValueError(f"Unable to read tar member stream: {member.name!r}")
            with src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(out_path)

    return extracted


def _assert_all_numbers_finite(obj: Any, *, path_label: str) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Non-finite float ({obj}) found in {path_label}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_numbers_finite(v, path_label=f"{path_label}.{k}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _assert_all_numbers_finite(item, path_label=f"{path_label}[{idx}]")


def validate_json_artifact_strict(path: Path) -> dict[str, Any]:
    """Validate pure UTF-8 LF JSON, < 1 MB size cap, zero NaN/Inf, and valid JSON object."""
    raw = path.read_bytes()
    if len(raw) >= MAX_GIT_JSON_BYTES:
        raise ValueError(
            f"JSON artifact exceeds 1 MB git safety cap: {path.name} ({len(raw)} bytes >= {MAX_GIT_JSON_BYTES})"
        )
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"UTF-8 BOM detected in JSON artifact: {path.name}")
    if b"\r" in raw:
        raise ValueError(f"CR/CRLF line ending detected in JSON artifact: {path.name}")

    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid UTF-8 JSON in {path.name}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level JSON object in {path.name}, got {type(parsed).__name__}")
    _assert_all_numbers_finite(parsed, path_label=path.name)
    # Double-check canonical serialization rejects NaN/Inf
    json.dumps(parsed, allow_nan=False)
    return parsed


def _verify_manifest_digests(
    *,
    manifest: dict[str, Any],
    staged_artifacts_dir: Path,
    staged_blobs_dir: Path,
    allow_missing_non_onnx_blobs: bool = False,
) -> dict[str, Any]:
    """Verify every SHA-256 entry recorded in remote_bundle_manifest.json fail-closed.

    When `allow_missing_non_onnx_blobs=True`, missing non-core optional blobs (such as LoRA
    adapter directories missed during Colab Drive FUSE sync) are recorded transparently in
    `missing_optional_blobs` rather than aborting. Core binaries (all ONNX models,
    `ood_calibration_v21.json`, `*.vrta_lut*`, `*.vrta_cel2d`), all manifest JSONs, and
    SHA-256/size parity for every present file remain strictly fail-closed.
    """
    verified_jsons: dict[str, str] = {}
    for entry in manifest.get("json_artifacts", []):
        fname = str(entry["filename"])
        expected_sha = str(entry["sha256"])
        expected_size = int(entry["size_bytes"])
        target = staged_artifacts_dir / fname
        if not target.is_file():
            raise FileNotFoundError(f"Manifest-listed JSON artifact missing from archive: {fname}")
        actual_size = target.stat().st_size
        actual_sha = sha256_file(target)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise ValueError(
                f"Manifest SHA-256/size mismatch for JSON {fname}: "
                f"actual=({actual_sha}, {actual_size}B) != manifest=({expected_sha}, {expected_size}B)"
            )
        verified_jsons[fname] = actual_sha

    verified_blobs: dict[str, str] = {}
    missing_optional_blobs: list[str] = []
    for entry in manifest.get("blob_artifacts", []):
        rel = str(entry["relpath"]).replace("\\", "/")
        expected_sha = str(entry["sha256"])
        expected_size = int(entry["size_bytes"])
        target = staged_blobs_dir / Path(*PurePosixPath(rel).parts)
        if not target.is_file():
            if allow_missing_non_onnx_blobs and not _is_core_required_blob(rel):
                missing_optional_blobs.append(rel)
                continue
            if allow_missing_non_onnx_blobs:
                raise FileNotFoundError(
                    f"Manifest-listed core required blob missing from archive "
                    f"(forbidden even with --allow-missing-non-onnx-blobs): {rel}"
                )
            raise FileNotFoundError(f"Manifest-listed blob missing from archive: {rel}")
        actual_size = target.stat().st_size
        actual_sha = sha256_file(target)
        if actual_size != expected_size or actual_sha != expected_sha:
            raise ValueError(
                f"Manifest SHA-256/size mismatch for blob {rel}: "
                f"actual=({actual_sha}, {actual_size}B) != manifest=({expected_sha}, {expected_size}B)"
            )
        verified_blobs[rel] = actual_sha

    missing_optional_sorted = sorted(missing_optional_blobs)
    return {
        "manifest_json_count": len(verified_jsons),
        "manifest_blob_count": len(verified_blobs),
        "verified_jsons": verified_jsons,
        "verified_blobs": verified_blobs,
        "missing_optional_blobs": missing_optional_sorted,
        "partial_blob_sync_detected": bool(missing_optional_sorted),
    }


def write_sha256sums(blob_dir: Path, out_file: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Generate deterministic GNU coreutils-compatible SHA256SUMS.txt (pure UTF-8 LF)."""
    entries: list[dict[str, Any]] = []
    lines: list[str] = []
    for p in sorted(blob_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() == out_file.resolve():
            continue
        rel_parts = p.relative_to(blob_dir).parts
        if any(part.startswith("_work") or part.startswith(".") for part in rel_parts):
            continue
        if p.name in ("SHA256SUMS.txt", "g4_ingestion_receipt.json"):
            continue
        rel_posix = "/".join(rel_parts)
        digest = sha256_file(p)
        entries.append(
            {
                "relpath": rel_posix,
                "size_bytes": p.stat().st_size,
                "sha256": digest,
            }
        )
        lines.append(f"{digest}  {rel_posix}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    return out_file, entries


def ingest_g4_bundles(
    *,
    artifacts_tar: Path,
    blobs_tar: Path,
    realworld_tar: Path | None = None,
    dest_artifacts_dir: Path = REPO_ROOT / "artifacts" / "v4.0.0",
    dest_blobs_dir: Path = Path("/var/tmp/vertirta_v400_blobs"),
    out_report: Path | None = None,
    allow_partial: bool = False,
    allow_missing_non_onnx_blobs: bool = False,
) -> dict[str, Any]:
    """Verify and stage downloaded Colab G4 artifacts and blobs with atomic fail-closed checks."""
    with tempfile.TemporaryDirectory(prefix="vertirta_g4_ingest_") as tmp_str:
        tmp_root = Path(tmp_str)
        stage_art = tmp_root / "artifacts"
        stage_blobs = tmp_root / "blobs"
        stage_rw = tmp_root / "realworld"

        # 1) Safe extract archives into staging directories first (dest untouched until ALL checks pass)
        safe_extract_tar_gz(
            artifacts_tar,
            stage_art,
            strip_prefixes=("artifacts", "vertirta_v400_artifacts"),
            skip_work_dirs=True,
        )
        safe_extract_tar_gz(
            blobs_tar,
            stage_blobs,
            strip_prefixes=("blobs", "vertirta_v400_blobs"),
            skip_work_dirs=True,
        )
        if realworld_tar is not None:
            safe_extract_tar_gz(
                realworld_tar,
                stage_rw,
                strip_prefixes=("realworld_eval", "vertirta_v400_realworld_eval"),
                skip_work_dirs=True,
            )

        # 2) Identify top-level canonical JSON files to promote to git artifacts dir
        #    (explicitly excludes logs/*.log, heartbeat.json, MP4s, and binaries)
        candidate_jsons: dict[str, Path] = {}
        for p in sorted(stage_art.glob("*.json")):
            if p.name in EXCLUDED_ARTIFACT_JSONS:
                continue
            candidate_jsons[p.name] = p

        # Optional Cell 5 real-world evaluation JSONs (normalized to g4_realworld_* when bare)
        realworld_demo_files: list[Path] = []
        if realworld_tar is not None and stage_rw.is_dir():
            for p in sorted(stage_rw.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix.lower() == ".json":
                    target_name = p.name if p.name.startswith("g4_") else f"g4_realworld_{p.name}"
                    if target_name in EXCLUDED_ARTIFACT_JSONS:
                        continue
                    candidate_jsons[target_name] = p
                else:
                    realworld_demo_files.append(p)

        # 3) Validate EVERY candidate JSON strictly (UTF-8 LF, < 1 MB, zero NaN/Inf)
        parsed_jsons: dict[str, dict[str, Any]] = {}
        for name, src_path in sorted(candidate_jsons.items()):
            parsed_jsons[name] = validate_json_artifact_strict(src_path)

        # 4) Verify manifest SHA-256 digests when remote_bundle_manifest.json is present
        manifest_path = stage_art / "remote_bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Missing remote_bundle_manifest.json inside artifacts archive.")
        manifest = parsed_jsons["remote_bundle_manifest.json"]
        manifest_verification = _verify_manifest_digests(
            manifest=manifest,
            staged_artifacts_dir=stage_art,
            staged_blobs_dir=stage_blobs,
            allow_missing_non_onnx_blobs=allow_missing_non_onnx_blobs,
        )

        # 5) Check required G4 job JSON completeness (unless --allow-partial)
        missing_required = [name for name in REQUIRED_ALL_JSONS if name not in candidate_jsons]
        if missing_required and not allow_partial:
            raise FileNotFoundError(
                f"Incomplete G4 bundle (missing required JSONs): {missing_required}. "
                "Re-run missing Cell 4.5 job or pass --allow-partial explicitly."
            )

        # 6) Verify all 4 ONNX binaries against G4 manifest SHA-256 digests
        onnx_parity: dict[str, dict[str, Any]] = {}
        if not allow_partial or (
            "g4_pidnet_v21_xfeat_metrics.json" in candidate_jsons
            and "g4_forward_wire_metrics.json" in candidate_jsons
        ):
            onnx_parity = verify_onnx_sha256_parity(stage_blobs, stage_art)

        # 7) Run public-hygiene gate on all staged JSON files BEFORE copying into git repo
        hygiene_stage = tmp_root / "hygiene_check"
        hygiene_relpaths: list[str] = []
        for name, src_path in sorted(candidate_jsons.items()):
            dst = hygiene_stage / "artifacts" / "v4.0.0" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            hygiene_relpaths.append(f"artifacts/v4.0.0/{name}")

        hygiene_res = scan_public_hygiene(hygiene_stage, tracked_files=hygiene_relpaths)
        if not hygiene_res["passed"]:
            raise ValueError(f"Public hygiene gate failed on G4 JSONs: {hygiene_res['violations']}")

        # 8) All gates passed! Commit staged files to final destinations.
        dest_artifacts_dir.mkdir(parents=True, exist_ok=True)
        dest_blobs_dir.mkdir(parents=True, exist_ok=True)

        installed_jsons: list[dict[str, Any]] = []
        for name, src_path in sorted(candidate_jsons.items()):
            dst = dest_artifacts_dir / name
            shutil.copy2(src_path, dst)
            installed_jsons.append(
                {
                    "filename": name,
                    "size_bytes": dst.stat().st_size,
                    "sha256": sha256_file(dst),
                    "in_remote_manifest": name in manifest_verification["verified_jsons"],
                }
            )

        for src_blob in sorted(stage_blobs.rglob("*")):
            if not src_blob.is_file():
                continue
            rel = src_blob.relative_to(stage_blobs)
            dst = dest_blobs_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_blob, dst)

        installed_realworld_demos: list[dict[str, Any]] = []
        if realworld_demo_files:
            rw_out_dir = dest_blobs_dir / "realworld_demos"
            rw_out_dir.mkdir(parents=True, exist_ok=True)
            for src_demo in realworld_demo_files:
                rel = src_demo.relative_to(stage_rw)
                dst = rw_out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_demo, dst)
                installed_realworld_demos.append(
                    {
                        "relpath": f"realworld_demos/{rel.as_posix()}",
                        "size_bytes": dst.stat().st_size,
                        "sha256": sha256_file(dst),
                    }
                )

        # 9) Write deterministic SHA256SUMS.txt in dest_blobs_dir for GitHub Release upload
        sums_path, blob_entries = write_sha256sums(dest_blobs_dir, dest_blobs_dir / "SHA256SUMS.txt")

        # 10) Summarize telemetry & GPU peak saturation if available
        telemetry = parsed_jsons.get("g4_thunder_telemetry.json", {})
        telemetry_summary = {
            "actual_gpu_name": telemetry.get("actual_gpu_name", manifest.get("actual_gpu_name")),
            "actual_gpu_total_mib": telemetry.get("actual_gpu_total_mib", manifest.get("actual_gpu_total_mib")),
            "by_wave": telemetry.get("by_wave", {}),
        }

        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "colab_cu_spent": 0.0,
            "allow_partial": bool(allow_partial),
            "allow_missing_non_onnx_blobs": bool(allow_missing_non_onnx_blobs),
            "partial_blob_sync_detected": bool(manifest_verification["partial_blob_sync_detected"]),
            "missing_optional_blobs": list(manifest_verification["missing_optional_blobs"]),
            "missing_required_jsons": missing_required,
            "manifest_state": manifest.get("state"),
            "manifest_verified_json_count": manifest_verification["manifest_json_count"],
            "manifest_verified_blob_count": manifest_verification["manifest_blob_count"],
            "installed_json_count": len(installed_jsons),
            "installed_jsons": installed_jsons,
            "onnx_parity_verified": bool(onnx_parity) and all(v["sha256_verified"] for v in onnx_parity.values()),
            "onnx_models": {
                k: {
                    "filename": v["filename"],
                    "sha256": v["sha256"],
                    "size_bytes": v["size_bytes"],
                    "sha256_verified": v["sha256_verified"],
                }
                for k, v in sorted(onnx_parity.items())
            },
            "blob_count": len(blob_entries),
            "realworld_demo_count": len(installed_realworld_demos),
            "realworld_demos": installed_realworld_demos,
            "sha256sums_file": sums_path.name,
            "sha256sums_sha256": sha256_file(sums_path),
            "public_hygiene_passed": bool(hygiene_res["passed"]),
            "telemetry_summary": telemetry_summary,
            "all_gates_passed": bool(
                not missing_required
                and hygiene_res["passed"]
                and bool(onnx_parity)
                and all(v["sha256_verified"] for v in onnx_parity.values())
            ),
            "next_edge_gate_command": (
                "python scripts/bench_gtx1650s_m3_timesliced.py "
                "--blob-dir /var/tmp/vertirta_v400_blobs "
                "--artifacts-dir artifacts/v4.0.0 "
                "--out-json artifacts/v4.0.0/gtx1650s_m3_timesliced_edge_gate.json"
            ),
        }

        report_path = out_report if out_report is not None else (dest_blobs_dir / "g4_ingestion_receipt.json")
        write_canonical_json(report_path, receipt)
        return receipt


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and stage downloaded Colab G4 artifacts & blobs (fail-closed, 0 CU)."
    )
    parser.add_argument(
        "--artifacts-tar",
        type=Path,
        required=True,
        help="Path to vertirta_v400_artifacts.tar.gz downloaded from Google Drive.",
    )
    parser.add_argument(
        "--blobs-tar",
        type=Path,
        required=True,
        help="Path to vertirta_v400_blobs.tar.gz downloaded from Google Drive.",
    )
    parser.add_argument(
        "--realworld-tar",
        type=Path,
        default=None,
        help="Optional path to vertirta_v400_realworld_eval.tar.gz (Cell 5 BF16 + HUD MP4s).",
    )
    parser.add_argument(
        "--dest-artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v4.0.0",
        help="Destination directory for small (<1 MB) verified canonical JSON artifacts.",
    )
    parser.add_argument(
        "--dest-blobs-dir",
        type=Path,
        default=Path("/var/tmp/vertirta_v400_blobs"),
        help="Out-of-git destination directory for ONNX/LoRA/LUT blobs and SHA256SUMS.txt.",
    )
    parser.add_argument(
        "--out-report",
        type=Path,
        default=None,
        help="Optional path for ingestion receipt JSON (defaults to <dest-blobs-dir>/g4_ingestion_receipt.json).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow ingestion when one of the 5 G4 job JSONs is not yet present.",
    )
    parser.add_argument(
        "--allow-missing-non-onnx-blobs",
        action="store_true",
        default=False,
        help=(
            "Opt-in: allow missing non-core optional blob paths (e.g. LoRA adapter folders missed "
            "during Colab Drive FUSE sync) while still failing closed on any SHA-256 mismatch or "
            "any missing ONNX model, ood_calibration_v21.json, *.vrta_lut*, *.vrta_cel2d, or manifest JSON."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    receipt = ingest_g4_bundles(
        artifacts_tar=args.artifacts_tar,
        blobs_tar=args.blobs_tar,
        realworld_tar=args.realworld_tar,
        dest_artifacts_dir=args.dest_artifacts_dir,
        dest_blobs_dir=args.dest_blobs_dir,
        out_report=args.out_report,
        allow_partial=args.allow_partial,
        allow_missing_non_onnx_blobs=args.allow_missing_non_onnx_blobs,
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
