#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared runtime guards, license firewall, VRAM fractioning, and JSON I/O for VertiRTA M3 G4 jobs.

Enforces Plan 6C (#316) contracts:
  - Disjoint Wave-1 (0.46 + 0.22 + 0.20 = 0.88) and Wave-2 (0.58 XLA + 0.30 PyTorch = 0.88)
    VRAM partitioning on Colab G4 (NVIDIA RTX PRO 6000 Blackwell, 96 GB GDDR7).
  - Zero-GPL / zero-NC / zero-ShareAlike startup license firewall separating globally banned
    identifiers (UAVid, OpenVINS, UWB-ATCC, ATCO2, piper1-gpl, etc.) from train-only banned
    eval-quarantined corpora (AeroScapes, ICG, UAVScenes) and forward-only TTPLA.
  - Canonical Plan 6B SNR stratum keys: ("S0_Clean", "S1_Cruise", "S2_Hover", "S3_Garbled").
  - Standardized --dry-run / --smoke-cpu CLI parser and deterministic SHA-256 JSON writer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
from typing import Any

WAVE_BUDGETS: dict[str, dict[str, float]] = {
    "wave1": {
        "qwen3_v3": 0.46,
        "whisper_bvi": 0.22,
        "forward_wire": 0.20,
    },
    "wave2": {
        "jax_4d_hires_lc": 0.58,
        "pidnet_v21_xfeat": 0.30,
    },
}

assert sum(WAVE_BUDGETS["wave1"].values()) <= 0.90 + 1e-9, "Wave 1 VRAM fraction sum exceeds 0.90"
assert sum(WAVE_BUDGETS["wave2"].values()) <= 0.90 + 1e-9, "Wave 2 VRAM fraction sum exceeds 0.90"

CANONICAL_SNR_STRATA: tuple[str, str, str, str] = (
    "S0_Clean",
    "S1_Cruise",
    "S2_Hover",
    "S3_Garbled",
)

# Globally hard-banned across train_sources, eval_only_sources, AND models (01-DR08-13 §3.1).
HARD_BANNED_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "uavid",
        "insplad",
        "ddos",
        "drone-vs-bird",
        "uwb-atcc",
        "atco2",
        "esc-50",
        "demand",
        "atcosim",
        "piper1-gpl",
        "espeak-ng",
        "piper-phonemize",
        "tr_tr-dfki",
        "tr_tr-fahrettin",
        "tr_tr-fettah",
        "xtts-v2",
        "mms-tts",
        "omnivoice",
        "blenderproc",
        "bpy",
        "dinov3",
        "dinov2-salad",
        "sample4geo",
        "superpoint",
        "superglue",
        "efficientloftr",
        "ultralytics",
        "segformer",
        "deep-hough",
        "aya-expanse",
        "openvins",
        "vins-fusion",
        "hybvio",
        "orb-slam3",
        "sqrtvins",
        "stn-plad",
        "pampc",
        "pldu",
        "pldm",
        "synthwires",
        "wireseghr",
        "sod4sb",
        "seraphim",
    }
)

# Banned from training/calibration, permitted strictly in quarantined zero-shot eval_only_sources
# (01-DR08-13 Override #7 & Spec §6.1).
TRAIN_ONLY_BANNED_IDENTIFIERS: frozenset[str] = frozenset(
    {
        "aeroscapes",
        "icg",
        "uavscenes",
    }
)

_FORBIDDEN_LICENSE_RE = re.compile(
    r"(?i)(\bgpl\b|\blgpl\b|\bagpl\b|by-nc|nc-sa|by-sa|sharealike|non-commercial|noncommercial|\bnc\b|odbl|openrail|cpml|unlicensed|proprietary)"
)


def _norm_token(value: str) -> str:
    return re.sub(r"[\s_]+", "-", value.strip().lower())


def _matches_identifier(text: str, banned_set: frozenset[str]) -> str | None:
    raw = text.strip().lower()
    norm = _norm_token(text)
    for tok in banned_set:
        tok_norm = _norm_token(tok)
        if tok in raw or tok_norm in norm:
            # Avoid false positives on short tokens like "icg" or "bpy" inside unrelated words.
            if len(tok_norm) <= 4:
                pattern = r"(?<![a-z0-9])" + re.escape(tok_norm) + r"(?![a-z0-9])"
                if re.search(pattern, norm) or re.search(
                    r"(?<![a-z0-9])" + re.escape(tok) + r"(?![a-z0-9])", raw
                ):
                    return tok
            else:
                return tok
    return None


def validate_job_license_manifest(
    job_name: str,
    train_sources: list[dict[str, Any]],
    eval_only_sources: list[dict[str, Any]],
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail-closed license firewall invoked strictly BEFORE any download or training step."""
    job_norm = job_name.strip().lower()

    # 1. Check all entries against globally HARD_BANNED_IDENTIFIERS.
    for section_name, entries in (
        ("train_sources", train_sources),
        ("eval_only_sources", eval_only_sources),
        ("models", models),
    ):
        for entry in entries:
            name = str(entry.get("name", ""))
            repo = str(entry.get("repo", ""))
            lic = str(entry.get("license", ""))
            combined = f"{name} {repo}"
            hit = _matches_identifier(combined, HARD_BANNED_IDENTIFIERS)
            if hit is not None:
                raise ValueError(
                    f"[{job_name}] Hard-banned identifier '{hit}' detected in {section_name}: {entry!r}"
                )
            if not lic.strip():
                raise ValueError(
                    f"[{job_name}] Missing explicit license tag in {section_name}: {entry!r}"
                )

    # 2. Enforce strictly permissive licenses on train_sources and models.
    for section_name, entries in (("train_sources", train_sources), ("models", models)):
        for entry in entries:
            name = str(entry.get("name", ""))
            repo = str(entry.get("repo", ""))
            lic = str(entry.get("license", ""))
            train_hit = _matches_identifier(f"{name} {repo}", TRAIN_ONLY_BANNED_IDENTIFIERS)
            if train_hit is not None:
                raise ValueError(
                    f"[{job_name}] Eval-only dataset '{train_hit}' forbidden in {section_name}: {entry!r}"
                )
            if _FORBIDDEN_LICENSE_RE.search(lic):
                raise ValueError(
                    f"[{job_name}] Non-permissive license '{lic}' forbidden in {section_name}: {name}"
                )
            # TTPLA is strictly restricted to decoupled 45 deg forward wire head ONLY.
            if "ttpla" in name.lower() or "ttpla" in repo.lower():
                if "forward_wire" not in job_norm:
                    raise ValueError(
                        f"[{job_name}] TTPLA is strictly restricted to job_forward_wire "
                        f"(decoupled 45 deg forward wire head ONLY — forbidden in {job_name})."
                    )

    # 3. Verify eval_only_sources carry explicit quarantine flags if non-permissive.
    for entry in eval_only_sources:
        name = str(entry.get("name", ""))
        lic = str(entry.get("license", ""))
        if _FORBIDDEN_LICENSE_RE.search(lic):
            if not bool(entry.get("quarantined_zero_shot_eval_only", False)):
                raise ValueError(
                    f"[{job_name}] Non-permissive eval source '{name}' ({lic}) MUST set "
                    f"quarantined_zero_shot_eval_only=True."
                )

    return {
        "job_name": job_name,
        "zero_nc_sa_gpl_verified": True,
        "zero_sharealike_verified": True,
        "train_source_names": [str(e.get("name", "")) for e in train_sources],
        "eval_only_source_names": [str(e.get("name", "")) for e in eval_only_sources],
        "model_names": [str(e.get("name", "")) for e in models],
    }


def configure_torch_vram_fraction(
    fraction: float,
    *,
    device_index: int = 0,
) -> dict[str, Any]:
    """Pin PyTorch per-process VRAM fraction and expandable_segments allocator config."""
    if not (0.05 <= float(fraction) <= 0.65):
        raise ValueError(f"vram_fraction must be within [0.05, 0.65], got {fraction}")

    alloc_conf = "expandable_segments:True,max_split_size_mb:512"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf

    info: dict[str, Any] = {
        "requested_vram_fraction": round(float(fraction), 4),
        "pytorch_cuda_alloc_conf": alloc_conf,
        "cuda_available": False,
        "applied_on_device": None,
        "gpu_name": None,
        "vram_total_mib": 0,
        "vram_cap_mib": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(device_index)
            torch.cuda.set_per_process_memory_fraction(float(fraction), device=device_index)
            props = torch.cuda.get_device_properties(device_index)
            total_mib = float(props.total_memory) / (1024.0 * 1024.0)
            info.update(
                {
                    "cuda_available": True,
                    "applied_on_device": device_index,
                    "gpu_name": props.name,
                    "vram_total_mib": round(total_mib, 2),
                    "vram_cap_mib": round(total_mib * float(fraction), 2),
                }
            )
    except ImportError:
        pass
    return info


def sha256_file(path: Path) -> str:
    """Compute lowercase hex FIPS 180-4 SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def query_nvidia_smi_snapshot() -> dict[str, Any]:
    """Query GPU stats + active compute PIDs via nvidia-smi (graceful fallback on CPU hosts)."""
    ts = datetime.now(timezone.utc).isoformat()
    gpu_info: dict[str, Any] = {
        "timestamp_utc": ts,
        "available": False,
        "name": "UNAVAILABLE",
        "driver_version": "UNKNOWN",
        "memory_used_mib": 0.0,
        "memory_total_mib": 0.0,
        "gpu_util_pct": 0.0,
        "mem_util_pct": 0.0,
        "power_w": 0.0,
        "temp_c": 0.0,
        "compute_apps": [],
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.used,memory.total,"
                "utilization.gpu,utilization.memory,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 8:
                def _flt(s: str) -> float:
                    try:
                        return float(s)
                    except ValueError:
                        return 0.0

                gpu_info.update(
                    {
                        "available": True,
                        "name": parts[0],
                        "driver_version": parts[1],
                        "memory_used_mib": _flt(parts[2]),
                        "memory_total_mib": _flt(parts[3]),
                        "gpu_util_pct": _flt(parts[4]),
                        "mem_util_pct": _flt(parts[5]),
                        "power_w": _flt(parts[6]),
                        "temp_c": _flt(parts[7]),
                    }
                )
        apps_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
        apps: list[dict[str, Any]] = []
        for line in apps_out.splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) >= 3 and cols[0].isdigit():
                try:
                    mem = float(cols[2])
                except ValueError:
                    mem = 0.0
                apps.append(
                    {
                        "pid": int(cols[0]),
                        "process_name": cols[1],
                        "used_gpu_memory_mib": mem,
                    }
                )
        gpu_info["compute_apps"] = apps
    except Exception:
        pass
    return gpu_info


def seed_everything(seed: int) -> None:
    """Deterministically seed stdlib random, optional NumPy, and optional PyTorch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed & 0xFFFFFFFF)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_standard_argparser(
    job_name: str,
    default_vram_fraction: float,
    description: str = "",
) -> argparse.ArgumentParser:
    """Create standardized Plan 6C CLI parser supporting --dry-run and --smoke-cpu."""
    parser = argparse.ArgumentParser(
        prog=job_name,
        description=description or f"VertiRTA M3 G4 Job: {job_name}",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate license firewall, allocator caps, and job plan with 0 compute.",
    )
    mode_group.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Run full CPU end-to-end smoke path (0 CU, emits schema-complete SMOKE_CPU_ONLY JSON).",
    )
    parser.add_argument(
        "--vram-fraction",
        type=float,
        default=default_vram_fraction,
        help=f"Per-process GPU memory fraction (default: {default_vram_fraction}).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Output path for canonical SHA-256 metrics JSON manifest.",
    )
    parser.add_argument(
        "--blob-dir",
        type=Path,
        default=Path("/var/tmp/vertirta_v400_blobs"),
        help="Out-of-git directory for binary weights, ONNX graphs, and .vrta_lut artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional local dataset directory override.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261124,
        help="Deterministic master RNG seed.",
    )
    return parser


def write_canonical_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomically write pure UTF-8 LF JSON embedding deterministic report_sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {k: v for k, v in payload.items() if k != "report_sha256"}
    canonical_bytes = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    out_obj = dict(body)
    out_obj["report_sha256"] = digest
    text = json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n"
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8", newline="\n")
    tmp_path.replace(path)
    return digest
