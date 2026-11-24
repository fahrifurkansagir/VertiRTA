# SPDX-License-Identifier: Apache-2.0
"""VertiRTA v4.0.0 Single-Shot Colab G4 Batch Thunder Run modular job package."""

from __future__ import annotations

from scripts.colab.g4_jobs.common import (
    CANONICAL_SNR_STRATA,
    HARD_BANNED_IDENTIFIERS,
    TRAIN_ONLY_BANNED_IDENTIFIERS,
    WAVE_BUDGETS,
    build_standard_argparser,
    configure_torch_vram_fraction,
    query_nvidia_smi_snapshot,
    seed_everything,
    sha256_file,
    validate_job_license_manifest,
    write_canonical_json,
)

G4_JOB_REGISTRY: dict[str, dict[str, object]] = {
    "qwen3_v3": {
        "wave": "wave1",
        "script": "scripts/colab/g4_jobs/job_qwen3_v3_vertibench_v2.py",
        "runtime": "pytorch",
        "vram_fraction": WAVE_BUDGETS["wave1"]["qwen3_v3"],
        "out_json_name": "g4_qwen3_v3_vertibench_v2_eval.json",
    },
    "whisper_bvi": {
        "wave": "wave1",
        "script": "scripts/colab/g4_jobs/job_whisper_bvi_lora.py",
        "runtime": "pytorch",
        "vram_fraction": WAVE_BUDGETS["wave1"]["whisper_bvi"],
        "out_json_name": "g4_whisper_bvi_lora_eval.json",
    },
    "forward_wire": {
        "wave": "wave1",
        "script": "scripts/colab/g4_jobs/job_forward_wire.py",
        "runtime": "pytorch",
        "vram_fraction": WAVE_BUDGETS["wave1"]["forward_wire"],
        "out_json_name": "g4_forward_wire_metrics.json",
    },
    "jax_4d_hires_lc": {
        "wave": "wave2",
        "script": "scripts/colab/g4_jobs/job_jax_4d_hires_and_lc.py",
        "runtime": "jax_xla",
        "vram_fraction": WAVE_BUDGETS["wave2"]["jax_4d_hires_lc"],
        "out_json_name": "g4_jax_4d_hires_and_lc_manifest.json",
    },
    "pidnet_v21_xfeat": {
        "wave": "wave2",
        "script": "scripts/colab/g4_jobs/job_pidnet_v21_xfeat.py",
        "runtime": "pytorch",
        "vram_fraction": WAVE_BUDGETS["wave2"]["pidnet_v21_xfeat"],
        "out_json_name": "g4_pidnet_v21_xfeat_metrics.json",
    },
}

__all__ = [
    "CANONICAL_SNR_STRATA",
    "G4_JOB_REGISTRY",
    "HARD_BANNED_IDENTIFIERS",
    "TRAIN_ONLY_BANNED_IDENTIFIERS",
    "WAVE_BUDGETS",
    "build_standard_argparser",
    "configure_torch_vram_fraction",
    "query_nvidia_smi_snapshot",
    "seed_everything",
    "sha256_file",
    "validate_job_license_manifest",
    "write_canonical_json",
]
