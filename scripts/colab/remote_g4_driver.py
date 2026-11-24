#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""On-VM Two-Wave G4 Supervisor & 1 Hz nvidia-smi Telemetry Daemon (Plan 6C Task 6 + Wave-1 Turbo).

Executes on the Colab G4 VM (or locally in --dry-run / --smoke-cpu modes with 0 CU):
  - Writes initial `heartbeat.json` (`state="STARTING"`, `pid=os.getpid()`) BEFORE Wave 1.
  - Wave 1 (in live_g4 mode when `qwen3_v3` is selected):
      * Wave 1A (`wave1a`, 0.46 + 0.22 + 0.20 = 0.88):
          1. job_qwen3_v3_vertibench_v2.py --phase sft (0.46)
          2. job_whisper_bvi_lora.py                   (0.22)
          3. job_forward_wire.py                       (0.20)
      * Wave 1B (`wave1b_parallel_grpo`, 3 x 0.29 = 0.87):
          3 parallel independent `job_qwen3_v3_vertibench_v2.py --phase grpo-seed --seed-index {0,1,2}`
      * Wave 1C (`wave1c_conformal_finalize`, 0.65):
          `job_qwen3_v3_vertibench_v2.py --phase finalize` -> canonical `g4_qwen3_v3_vertibench_v2_eval.json`
  - Hard Wave-1 -> Wave-2 GPU Drain Barrier:
      Waits for ALL Wave-1 PIDs to exit AND polls nvidia-smi until 0 active compute PIDs
      and memory.used < 1,500 MiB (prevents XLA vs PyTorch allocator collision!).
  - Wave 2 (1 JAX/XLA + 1 PyTorch, 0.58 + 0.30 = 0.88):
      1. job_jax_4d_hires_and_lc.py (XLA_PYTHON_CLIENT_MEM_FRACTION=0.58)
      2. job_pidnet_v21_xfeat.py    (PyTorch fraction=0.30)
  - Wraps Wave 1 -> Drain -> Wave 2 in try/except/finally so even drain-barrier timeouts or
    early job crashes flush `heartbeat.json` (`state="FAILED"`), `g4_thunder_telemetry.json`,
    and partial `remote_bundle_manifest.json` cleanly before exit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs import G4_JOB_REGISTRY, WAVE_BUDGETS  # noqa: E402
from scripts.colab.g4_jobs.common import (  # noqa: E402
    query_nvidia_smi_snapshot,
    sha256_file,
    write_canonical_json,
)

DRAIN_MEMORY_THRESHOLD_MIB = 1500.0
WAVE1B_GRPO_SEED_FRACTION = 0.29
WAVE1B_GRPO_NUM_SEEDS = 3
WAVE1C_FINALIZE_FRACTION = 0.65


class TelemetrySampler:
    """1 Hz nvidia-smi telemetry collector with clean stop event."""

    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        snapshot_fn: Callable[[], dict[str, Any]] = query_nvidia_smi_snapshot,
    ) -> None:
        self._interval_s = max(0.05, float(interval_s))
        self._snapshot_fn = snapshot_fn
        self._stop = threading.Event()
        self._samples: list[dict[str, Any]] = []
        self._wave_label = "init"
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def set_wave(self, wave_label: str) -> None:
        with self._lock:
            self._wave_label = wave_label

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        self._thread.join(timeout=5.0)
        with self._lock:
            return list(self._samples)

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = self._snapshot_fn()
            with self._lock:
                snap["wave"] = self._wave_label
                self._samples.append(snap)
            self._stop.wait(self._interval_s)


def wait_for_gpu_drain_barrier(
    *,
    timeout_s: float = 60.0,
    poll_s: float = 1.0,
    max_mem_mib: float = DRAIN_MEMORY_THRESHOLD_MIB,
    snapshot_fn: Callable[[], dict[str, Any]] = query_nvidia_smi_snapshot,
    smoke_mode: bool = False,
) -> dict[str, Any]:
    """Poll nvidia-smi until 0 compute PIDs remain and memory.used < max_mem_mib."""
    t0 = time.perf_counter()
    if smoke_mode:
        return {
            "drained": True,
            "mode": "SMOKE_CPU_BYPASS",
            "elapsed_s": 0.0,
            "remaining_pids": 0,
            "memory_used_mib": 0.0,
        }

    last_snap: dict[str, Any] = {}
    while (time.perf_counter() - t0) <= timeout_s:
        snap = snapshot_fn()
        last_snap = snap
        if not snap.get("available", False):
            return {
                "drained": True,
                "mode": "NO_GPU_DETECTED",
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "remaining_pids": 0,
                "memory_used_mib": 0.0,
            }
        pids = snap.get("compute_apps", [])
        mem = float(snap.get("memory_used_mib", 0.0))
        if len(pids) == 0 and mem < max_mem_mib:
            return {
                "drained": True,
                "mode": "NVIDIA_SMI_VERIFIED",
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "remaining_pids": 0,
                "memory_used_mib": mem,
            }
        time.sleep(poll_s)

    raise RuntimeError(
        f"GPU drain barrier timed out after {timeout_s}s: "
        f"pids={last_snap.get('compute_apps', [])}, mem={last_snap.get('memory_used_mib')} MiB"
    )


def build_wave_commands(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    blob_dir: Path,
    mode: str,
    jobs_filter: set[str] | None = None,
    vertibench_v2_dir: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Construct exact Wave-1 and Wave-2 subprocess specifications with disjoint env locks."""
    waves: dict[str, list[dict[str, Any]]] = {"wave1": [], "wave2": []}
    for key, spec in G4_JOB_REGISTRY.items():
        if jobs_filter is not None and key not in jobs_filter:
            continue
        script_path = repo_root / str(spec["script"])
        if not script_path.is_file():
            continue
        wave = str(spec["wave"])
        frac = float(spec["vram_fraction"])  # type: ignore[arg-type]
        out_json = artifacts_dir / str(spec["out_json_name"])
        cmd = [
            sys.executable,
            str(script_path),
            "--vram-fraction",
            f"{frac:.2f}",
            "--out-json",
            str(out_json),
            "--blob-dir",
            str(blob_dir),
        ]
        if mode == "dry_run":
            cmd.append("--dry-run")
        elif mode == "smoke_cpu":
            cmd.append("--smoke-cpu")
        if key == "qwen3_v3" and vertibench_v2_dir is not None:
            cmd.extend(["--data-dir", str(vertibench_v2_dir)])

        env_overrides: dict[str, str] = {
            "PYTHONPATH": str(repo_root),
            "PYTHONUNBUFFERED": "1",
        }
        if str(spec["runtime"]) == "jax_xla":
            env_overrides["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            env_overrides["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{frac:.2f}"
            env_overrides["JAX_ENABLE_X64"] = "False"
        else:
            env_overrides["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

        waves[wave].append(
            {
                "job_key": key,
                "wave": wave,
                "runtime": str(spec["runtime"]),
                "vram_fraction": frac,
                "cmd": cmd,
                "out_json": str(out_json),
                "env_overrides": env_overrides,
            }
        )
    return waves


def build_live_wave1_stages(
    *,
    repo_root: Path,
    artifacts_dir: Path,
    blob_dir: Path,
    jobs_filter: set[str] | None = None,
    vertibench_v2_dir: Path | None = None,
    mode: str = "live_g4",
) -> dict[str, list[dict[str, Any]]]:
    """Build staged Wave-1A (0.88) + Wave-1B Parallel 3-Seed GRPO (3x0.29=0.87) + Wave-1C Finalize (0.65)."""
    base_waves = build_wave_commands(
        repo_root=repo_root,
        artifacts_dir=artifacts_dir,
        blob_dir=blob_dir,
        mode=mode,
        jobs_filter=jobs_filter,
        vertibench_v2_dir=vertibench_v2_dir,
    )
    stages: dict[str, list[dict[str, Any]]] = {
        "wave1a": [],
        "wave1b_parallel_grpo": [],
        "wave1c_conformal_finalize": [],
    }

    qwen_base: dict[str, Any] | None = None
    for sp in base_waves["wave1"]:
        if sp["job_key"] == "qwen3_v3":
            qwen_base = sp
        else:
            stages["wave1a"].append({**sp, "wave": "wave1a"})

    if qwen_base is None:
        return stages

    qwen_spec = G4_JOB_REGISTRY["qwen3_v3"]
    qwen_script = repo_root / str(qwen_spec["script"])
    qwen_sft_frac = float(qwen_spec["vram_fraction"])  # type: ignore[arg-type]
    qwen_env = dict(qwen_base["env_overrides"])

    def _qwen_stage_cmd(frac: float, out_json: Path, phase: str, seed_index: int | None = None) -> list[str]:
        cmd = [
            sys.executable,
            str(qwen_script),
            "--vram-fraction",
            f"{frac:.2f}",
            "--out-json",
            str(out_json),
            "--blob-dir",
            str(blob_dir),
            "--phase",
            phase,
        ]
        if seed_index is not None:
            cmd.extend(["--seed-index", str(seed_index)])
        if mode == "dry_run":
            cmd.append("--dry-run")
        elif mode == "smoke_cpu":
            cmd.append("--smoke-cpu")
        if vertibench_v2_dir is not None:
            cmd.extend(["--data-dir", str(vertibench_v2_dir)])
        return cmd

    sft_out = artifacts_dir / "g4_qwen3_v3_sft_stage.json"
    stages["wave1a"].insert(
        0,
        {
            "job_key": "qwen3_v3_sft",
            "wave": "wave1a",
            "runtime": "pytorch",
            "vram_fraction": qwen_sft_frac,
            "cmd": _qwen_stage_cmd(qwen_sft_frac, sft_out, "sft"),
            "out_json": str(sft_out),
            "env_overrides": dict(qwen_env),
        },
    )

    for seed_idx in range(WAVE1B_GRPO_NUM_SEEDS):
        seed_out = artifacts_dir / f"g4_qwen3_v3_grpo_seed{seed_idx}.json"
        stages["wave1b_parallel_grpo"].append(
            {
                "job_key": f"qwen3_v3_grpo_seed{seed_idx}",
                "wave": "wave1b_parallel_grpo",
                "runtime": "pytorch",
                "vram_fraction": WAVE1B_GRPO_SEED_FRACTION,
                "cmd": _qwen_stage_cmd(WAVE1B_GRPO_SEED_FRACTION, seed_out, "grpo-seed", seed_index=seed_idx),
                "out_json": str(seed_out),
                "env_overrides": dict(qwen_env),
            }
        )

    final_out = artifacts_dir / str(qwen_spec["out_json_name"])
    stages["wave1c_conformal_finalize"].append(
        {
            "job_key": "qwen3_v3_finalize",
            "wave": "wave1c_conformal_finalize",
            "runtime": "pytorch",
            "vram_fraction": WAVE1C_FINALIZE_FRACTION,
            "cmd": _qwen_stage_cmd(WAVE1C_FINALIZE_FRACTION, final_out, "finalize"),
            "out_json": str(final_out),
            "env_overrides": dict(qwen_env),
        }
    )
    return stages


def _write_heartbeat(path: Path, state: str, details: dict[str, Any]) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "state": state,
        **details,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _run_wave_parallel(
    wave_name: str,
    specs: list[dict[str, Any]],
    logs_dir: Path,
    heartbeat_path: Path,
) -> list[dict[str, Any]]:
    logs_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[dict[str, Any], subprocess.Popen[bytes], Path, Any, float]] = []
    for sp in specs:
        env = os.environ.copy()
        env.update(sp["env_overrides"])
        log_path = logs_dir / f"{sp['job_key']}.log"
        log_fh = log_path.open("wb")
        proc = subprocess.Popen(
            sp["cmd"],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        procs.append((sp, proc, log_path, log_fh, time.perf_counter()))

    results: list[dict[str, Any]] = []
    pending = list(procs)
    while pending:
        still_running = []
        for item in pending:
            sp, proc, log_path, log_fh, t_start = item
            rc = proc.poll()
            if rc is None:
                still_running.append(item)
            else:
                log_fh.close()
                results.append(
                    {
                        "job_key": sp["job_key"],
                        "wave": wave_name,
                        "returncode": int(rc),
                        "wall_s": round(time.perf_counter() - t_start, 3),
                        "out_json": sp["out_json"],
                        "log_path": str(log_path),
                    }
                )
        _write_heartbeat(
            heartbeat_path,
            f"RUNNING_{wave_name.upper()}",
            {
                "active_jobs": [x[0]["job_key"] for x in still_running],
                "completed_jobs": [r["job_key"] for r in results],
            },
        )
        if still_running:
            time.sleep(0.25)
        pending = still_running

    return results


def summarize_telemetry(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_wave: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        by_wave.setdefault(str(s.get("wave", "unknown")), []).append(s)

    summary: dict[str, Any] = {}
    for w, rows in by_wave.items():
        mems = [float(r.get("memory_used_mib", 0.0)) for r in rows]
        utils = [float(r.get("gpu_util_pct", 0.0)) for r in rows]
        summary[w] = {
            "sample_count": len(rows),
            "peak_memory_used_mib": round(max(mems) if mems else 0.0, 2),
            "peak_memory_used_gib": round((max(mems) / 1024.0) if mems else 0.0, 3),
            "mean_gpu_util_pct": round((sum(utils) / len(utils)) if utils else 0.0, 2),
        }
    return {
        "schema_version": "vertirta.g4_thunder_telemetry/v4.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(samples),
        "by_wave": summary,
        "samples": samples[-120:],
    }


def _collect_artifact_digests(artifacts_dir: Path, blob_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    json_files: list[dict[str, Any]] = []
    for p in sorted(artifacts_dir.glob("*.json")):
        if p.name in ("remote_bundle_manifest.json", "heartbeat.json"):
            continue
        json_files.append({"filename": p.name, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})

    blob_files: list[dict[str, Any]] = []
    for p in sorted(blob_dir.rglob("*")):
        if not p.is_file():
            continue
        rel_path = p.relative_to(blob_dir)
        if any(part.startswith("_work") or part.startswith(".") for part in rel_path.parts):
            continue
        rel = str(rel_path).replace("\\", "/")
        blob_files.append({"relpath": rel, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return json_files, blob_files


def run_driver(
    args: argparse.Namespace,
    *,
    drain_snapshot_fn: Callable[[], dict[str, Any]] = query_nvidia_smi_snapshot,
    drain_timeout_s: float = 60.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    mode = "dry_run" if args.dry_run else ("smoke_cpu" if args.smoke_cpu else "live_g4")
    artifacts_dir = Path(args.artifacts_dir)
    blob_dir = Path(args.blob_dir)
    logs_dir = artifacts_dir / "logs"
    heartbeat_path = artifacts_dir / "heartbeat.json"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    blob_dir.mkdir(parents=True, exist_ok=True)

    jobs_filter = set(args.jobs.split(",")) if args.jobs else None
    waves = build_wave_commands(
        repo_root=REPO_ROOT,
        artifacts_dir=artifacts_dir,
        blob_dir=blob_dir,
        mode=mode,
        jobs_filter=jobs_filter,
        vertibench_v2_dir=args.vertibench_v2_dir,
    )

    live_wave1_stages = build_live_wave1_stages(
        repo_root=REPO_ROOT,
        artifacts_dir=artifacts_dir,
        blob_dir=blob_dir,
        jobs_filter=jobs_filter,
        vertibench_v2_dir=args.vertibench_v2_dir,
        mode="live_g4" if mode == "live_g4" else mode,
    )
    use_staged_live_wave1 = (mode == "live_g4") and any(j["job_key"] == "qwen3_v3" for j in waves["wave1"])

    if mode == "dry_run":
        manifest = {
            "schema_version": "vertirta.remote_g4_bundle_manifest/v4.0.0",
            "mode": "DRY_RUN",
            "wave_budgets": WAVE_BUDGETS,
            "waves": waves,
            "live_wave1_stages": live_wave1_stages,
            "wall_clock_s": round(time.perf_counter() - t0, 4),
        }
        write_canonical_json(artifacts_dir / "remote_bundle_manifest.json", manifest)
        return manifest

    # Detect actual GPU hardware honestly from nvidia-smi (strict G4-only wave execution)
    gpu_init = query_nvidia_smi_snapshot() if mode == "live_g4" else {"available": False, "name": "CPU_SMOKE", "memory_total_mib": 0.0}
    actual_gpu_name = str(gpu_init.get("name", "UNAVAILABLE"))
    actual_gpu_total_mib = float(gpu_init.get("memory_total_mib", 0.0))

    # Emit initial STARTING heartbeat with driver PID & exact GPU identity BEFORE Wave 1 launches
    _write_heartbeat(
        heartbeat_path,
        "STARTING",
        {
            "actual_gpu_name": actual_gpu_name,
            "actual_gpu_total_mib": actual_gpu_total_mib,
            "wave1_schedule": "staged_wave1a_1b_1c" if use_staged_live_wave1 else "single_wave1",
            "wave1_jobs": [j["job_key"] for j in waves["wave1"]],
            "wave2_jobs": [j["job_key"] for j in waves["wave2"]],
        },
    )

    sampler = TelemetrySampler(interval_s=0.2 if mode == "smoke_cpu" else 1.0)
    sampler.start()
    all_results: list[dict[str, Any]] = []
    barrier_info: dict[str, Any] = {}
    driver_error: str | None = None

    try:
        if use_staged_live_wave1:
            sampler.set_wave("wave1a")
            w1a_res = _run_wave_parallel("wave1a", live_wave1_stages["wave1a"], logs_dir, heartbeat_path)
            all_results.extend(w1a_res)

            sft_ok = any(r["job_key"] == "qwen3_v3_sft" and r["returncode"] == 0 for r in w1a_res)
            if sft_ok:
                sampler.set_wave("wave1b_parallel_grpo")
                w1b_res = _run_wave_parallel(
                    "wave1b_parallel_grpo",
                    live_wave1_stages["wave1b_parallel_grpo"],
                    logs_dir,
                    heartbeat_path,
                )
                all_results.extend(w1b_res)

                all_grpo_ok = len(w1b_res) == WAVE1B_GRPO_NUM_SEEDS and all(r["returncode"] == 0 for r in w1b_res)
                if all_grpo_ok:
                    sampler.set_wave("wave1c_conformal_finalize")
                    w1c_res = _run_wave_parallel(
                        "wave1c_conformal_finalize",
                        live_wave1_stages["wave1c_conformal_finalize"],
                        logs_dir,
                        heartbeat_path,
                    )
                    all_results.extend(w1c_res)
        else:
            sampler.set_wave("wave1")
            w1_res = _run_wave_parallel("wave1", waves["wave1"], logs_dir, heartbeat_path)
            all_results.extend(w1_res)

        sampler.set_wave("drain_barrier")
        barrier_info = wait_for_gpu_drain_barrier(
            timeout_s=drain_timeout_s,
            snapshot_fn=drain_snapshot_fn,
            smoke_mode=(mode == "smoke_cpu"),
        )

        sampler.set_wave("wave2")
        w2_res = _run_wave_parallel("wave2", waves["wave2"], logs_dir, heartbeat_path)
        all_results.extend(w2_res)
    except Exception as exc:
        driver_error = f"{type(exc).__name__}: {exc}"
    finally:
        samples = sampler.stop()
        telemetry = summarize_telemetry(samples)
        telemetry["actual_gpu_name"] = actual_gpu_name
        telemetry["actual_gpu_total_mib"] = actual_gpu_total_mib
        write_canonical_json(artifacts_dir / "g4_thunder_telemetry.json", telemetry)
        json_files, blob_files = _collect_artifact_digests(artifacts_dir, blob_dir)
        failed = [r for r in all_results if r["returncode"] != 0]
        state = "DONE" if (driver_error is None and not failed) else "FAILED"
        _write_heartbeat(
            heartbeat_path,
            state,
            {
                "actual_gpu_name": actual_gpu_name,
                "actual_gpu_total_mib": actual_gpu_total_mib,
                "failed_jobs": [f["job_key"] for f in failed],
                "error": driver_error,
            },
        )
        manifest = {
            "schema_version": "vertirta.remote_g4_bundle_manifest/v4.0.0",
            "mode": "SMOKE_CPU_ONLY" if mode == "smoke_cpu" else "LIVE_G4_THUNDER",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "error": driver_error,
            "actual_gpu_name": actual_gpu_name,
            "actual_gpu_total_mib": actual_gpu_total_mib,
            "wave_budgets": WAVE_BUDGETS,
            "drain_barrier": barrier_info,
            "job_results": all_results,
            "json_artifacts": json_files,
            "blob_artifacts": blob_files,
            "wall_clock_s": round(time.perf_counter() - t0, 3),
        }
        write_canonical_json(artifacts_dir / "remote_bundle_manifest.json", manifest)

    if driver_error is not None:
        raise RuntimeError(f"G4 driver failed: {driver_error}")
    if failed:
        raise RuntimeError(f"One or more G4 wave jobs failed: {failed}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="VertiRTA M3 On-VM Two-Wave G4 Driver")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--smoke-cpu", action="store_true")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--blob-dir", type=Path, required=True)
    parser.add_argument("--vertibench-v2-dir", type=Path, default=None)
    parser.add_argument("--jobs", type=str, default=None, help="Optional comma-separated subset of job keys.")
    args = parser.parse_args()
    run_driver(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
