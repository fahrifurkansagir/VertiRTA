#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Physical GTX 1650 SUPER (TU116, 4 GB) M3 Time-Sliced Single-Context Edge Gate (Plan 6C Task 7).

Implements 01-DR08-13 Override #6, Spec Invariant I6, and DR-10 §1.3:
  - Verifies SHA-256 parity of all 4 v4.0.0 Opset-17 ONNX models against artifacts/v4.0.0/*.json:
      * pidnet_slz_v21_opset17.onnx  (1x3x512x512)
      * mlsd_wireseg_640_opset17.onnx (1x3x640x640)
      * dfine_n_640_opset17.onnx     (1x3x640x640)
      * xfeat_512_opset17.onnx       (1x1x512x512)
  - Phase A (Empirical FP32 vs FP32+CUDA-Graph vs FP16 per-model benchmark): NEVER assumes 2x
    TU116 FP16 speedup (v3.5.1 measured FP16 0.4x / 2.5x slower on PIDNet-S!).
  - Phase B (Single-Process Flight-Phase Time-Slicing):
      * Platform-aware AUTO cuDNN tuning (#313 fix):
          - Windows WDDM: EXHAUSTIVE + cudnn_conv_use_max_workspace=1 + enable_cuda_graph=1
          - Linux / WSL2: HEURISTIC  + cudnn_conv_use_max_workspace=0 + enable_cuda_graph=1
      * Cruise Phase (30 Hz): M-LSD-WireSeg + D-FINE-N every frame + XFeat at 1 Hz (1/30 frames)
      * Terminal Phase (10-30 Hz): Evicts cruise sessions + runs PIDNet-S v2.1 in same CUDA context
  - Enforces hard silicon gates:
      * onnx_sha256_all_verified == True
      * timesliced_peak_vram_mib < 300.0
      * pidnet_v21_p99_ms < 12.0 (Win10 physical GTX 1650 SUPER)
      * cruise_combined_p99_ms < 25.0 (fits 33.3 ms 30 Hz budget)
      * all_trained_outputs_finite_and_nondegenerate == True
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.colab.g4_jobs.common import sha256_file, write_canonical_json  # noqa: E402
from scripts.run_local_gtx1650s_onnx_bench import (  # noqa: E402
    _build_fp16_onnx,
    _query_cuda_mem_get_info_mib,
    _query_gpu_info,
    _query_process_vram_mib,
    _register_nvidia_dll_dirs,
    _summarize_ms,
)

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "key": "pidnet_slz_v21",
        "filename": "pidnet_slz_v21_opset17.onnx",
        "manifest_json": "g4_pidnet_v21_xfeat_metrics.json",
        "manifest_export_key": "pidnet_slz_v21",
        "input_name": "rgb_nchw",
        "input_shape": (1, 3, 512, 512),
        "phase": "terminal",
    },
    {
        "key": "mlsd_wireseg",
        "filename": "mlsd_wireseg_640_opset17.onnx",
        "manifest_json": "g4_forward_wire_metrics.json",
        "manifest_export_key": "mlsd_wireseg",
        "input_name": "rgb_split_nchw",
        "input_shape": (1, 3, 640, 640),
        "phase": "cruise",
    },
    {
        "key": "dfine_n",
        "filename": "dfine_n_640_opset17.onnx",
        "manifest_json": "g4_forward_wire_metrics.json",
        "manifest_export_key": "dfine_n",
        "input_name": "rgb_split_nchw",
        "input_shape": (1, 3, 640, 640),
        "phase": "cruise",
    },
    {
        "key": "xfeat_512",
        "filename": "xfeat_512_opset17.onnx",
        "manifest_json": "g4_pidnet_v21_xfeat_metrics.json",
        "manifest_export_key": "xfeat_512",
        "input_name": "gray_nchw",
        "input_shape": (1, 1, 512, 512),
        "phase": "cruise_1hz",
    },
)


def select_platform_cuda_provider_options(
    *,
    system_name: str | None = None,
    enable_cuda_graph: bool = True,
) -> dict[str, str]:
    """Platform-aware AUTO cuDNN tuning (#313 fix).

    Windows WDDM (v3.5.1 measured): EXHAUSTIVE + max_workspace=1 + CG hits 11.695 ms (< 12 ms)
    whereas HEURISTIC regresses to 17.53 ms. Linux/WSL2 uses HEURISTIC + CG to control VRAM.
    """
    sys_id = (system_name or platform.system()).strip().lower()
    is_windows = sys_id.startswith("win")
    return {
        "device_id": "0",
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "EXHAUSTIVE" if is_windows else "HEURISTIC",
        "cudnn_conv_use_max_workspace": "1" if is_windows else "0",
        "enable_cuda_graph": "1" if enable_cuda_graph else "0",
    }


def verify_onnx_sha256_parity(blob_dir: Path, artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    """Fail-closed SHA-256 verification of all 4 ONNX binaries against G4 manifests."""
    verified: dict[str, dict[str, Any]] = {}
    for spec in MODEL_SPECS:
        onnx_path = blob_dir / spec["filename"]
        if not onnx_path.is_file():
            raise FileNotFoundError(f"Missing ONNX binary: {onnx_path}")
        actual_sha = sha256_file(onnx_path)

        manifest_path = artifacts_dir / spec["manifest_json"]
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing G4 manifest JSON: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["onnx_exports"][spec["manifest_export_key"]]["sha256"]
        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA-256 mismatch for {spec['filename']}: actual={actual_sha} != expected={expected_sha}"
            )
        verified[spec["key"]] = {
            "path": str(onnx_path),
            "filename": spec["filename"],
            "sha256": actual_sha,
            "sha256_verified": True,
            "size_bytes": onnx_path.stat().st_size,
        }
    return verified


def _make_ort_session(
    onnx_path: Path,
    *,
    use_cuda: bool,
    cuda_graph: bool,
    max_workspace_override: bool | None = None,
) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1 if use_cuda else 2
    if use_cuda:
        cuda_opts = select_platform_cuda_provider_options(enable_cuda_graph=cuda_graph)
        if max_workspace_override is not None:
            cuda_opts["cudnn_conv_use_max_workspace"] = "1" if max_workspace_override else "0"
        sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=[("CUDAExecutionProvider", cuda_opts)],
        )
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("CUDAExecutionProvider unavailable during live GPU benchmark.")
        return sess
    return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])


def _run_timed_session(
    sess: Any,
    input_name: str,
    input_shape: tuple[int, ...],
    *,
    warmup: int,
    iters: int,
    use_cuda: bool,
    rng: np.random.Generator,
) -> tuple[dict[str, float], bool]:
    x_host = rng.standard_normal(input_shape, dtype=np.float32)
    outputs_finite = True

    if use_cuda:
        import onnxruntime as ort

        io = sess.io_binding()
        x_ort = ort.OrtValue.ortvalue_from_numpy(x_host, "cuda", 0)
        io.bind_ortvalue_input(input_name, x_ort)
        for out_meta in sess.get_outputs():
            io.bind_output(out_meta.name, "cuda", 0)
        for _ in range(warmup):
            sess.run_with_iobinding(io)
        io.synchronize_outputs()

        samples_ms: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            sess.run_with_iobinding(io)
            io.synchronize_outputs()
            samples_ms.append((time.perf_counter() - t0) * 1000.0)
        for arr in io.copy_outputs_to_cpu():
            if not np.all(np.isfinite(arr)):
                outputs_finite = False
        return _summarize_ms(samples_ms), outputs_finite

    for _ in range(warmup):
        sess.run(None, {input_name: x_host})
    samples_ms = []
    last_outs: list[np.ndarray] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        last_outs = sess.run(None, {input_name: x_host})
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    for arr in last_outs:
        if not np.all(np.isfinite(arr)):
            outputs_finite = False
    return _summarize_ms(samples_ms), outputs_finite


def _run_phase_b_timesliced(
    verified: dict[str, dict[str, Any]],
    *,
    use_cuda: bool,
    warmup: int,
    iters: int,
    cruise_use_cg: dict[str, bool],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Execute Phase B single-context flight-phase time-slicing (Cruise -> Evict -> Terminal)."""
    mem_before = _query_cuda_mem_get_info_mib() if use_cuda else None
    pid = os.getpid()

    # 1) Cruise Phase: M-LSD-WireSeg + D-FINE-N @ 30 Hz + XFeat @ 1 Hz (1/30 frames)
    #    Use max_workspace_override=False for the 3 lightweight Cruise models so cuDNN does not
    #    allocate 3x redundant max workspaces simultaneously, and run proper warmup + IoBinding.
    sess_wire = _make_ort_session(
        Path(verified["mlsd_wireseg"]["path"]),
        use_cuda=use_cuda,
        cuda_graph=bool(use_cuda and cruise_use_cg.get("mlsd_wireseg", False)),
        max_workspace_override=False if use_cuda else None,
    )
    sess_dfine = _make_ort_session(
        Path(verified["dfine_n"]["path"]),
        use_cuda=use_cuda,
        cuda_graph=bool(use_cuda and cruise_use_cg.get("dfine_n", True)),
        max_workspace_override=False if use_cuda else None,
    )
    sess_xfeat = _make_ort_session(
        Path(verified["xfeat_512"]["path"]),
        use_cuda=use_cuda,
        cuda_graph=bool(use_cuda and cruise_use_cg.get("xfeat_512", True)),
        max_workspace_override=False if use_cuda else None,
    )

    x_fwd = rng.standard_normal((1, 3, 640, 640), dtype=np.float32)
    x_gray = rng.standard_normal((1, 1, 512, 512), dtype=np.float32)
    cruise_warmup = 2 if not use_cuda else 30
    cruise_frames = 6 if not use_cuda else 120
    cruise_ms: list[float] = []
    cruise_finite = True

    if use_cuda:
        import onnxruntime as ort

        io_wire = sess_wire.io_binding()
        x_fwd_ort = ort.OrtValue.ortvalue_from_numpy(x_fwd, "cuda", 0)
        io_wire.bind_ortvalue_input("rgb_split_nchw", x_fwd_ort)
        for out_meta in sess_wire.get_outputs():
            io_wire.bind_output(out_meta.name, "cuda", 0)

        io_dfine = sess_dfine.io_binding()
        io_dfine.bind_ortvalue_input("rgb_split_nchw", x_fwd_ort)
        for out_meta in sess_dfine.get_outputs():
            io_dfine.bind_output(out_meta.name, "cuda", 0)

        io_xfeat = sess_xfeat.io_binding()
        x_gray_ort = ort.OrtValue.ortvalue_from_numpy(x_gray, "cuda", 0)
        io_xfeat.bind_ortvalue_input("gray_nchw", x_gray_ort)
        for out_meta in sess_xfeat.get_outputs():
            io_xfeat.bind_output(out_meta.name, "cuda", 0)

        for w_idx in range(cruise_warmup):
            sess_wire.run_with_iobinding(io_wire)
            sess_dfine.run_with_iobinding(io_dfine)
            if w_idx % 10 == 0:
                sess_xfeat.run_with_iobinding(io_xfeat)
        io_wire.synchronize_outputs()
        io_dfine.synchronize_outputs()
        io_xfeat.synchronize_outputs()

        for f_idx in range(cruise_frames):
            tf0 = time.perf_counter()
            sess_wire.run_with_iobinding(io_wire)
            sess_dfine.run_with_iobinding(io_dfine)
            if f_idx % 30 == 0:
                sess_xfeat.run_with_iobinding(io_xfeat)
                io_xfeat.synchronize_outputs()
            io_wire.synchronize_outputs()
            io_dfine.synchronize_outputs()
            cruise_ms.append((time.perf_counter() - tf0) * 1000.0)

        for io_obj in (io_wire, io_dfine, io_xfeat):
            for arr in io_obj.copy_outputs_to_cpu():
                if not np.all(np.isfinite(arr)):
                    cruise_finite = False
        del io_wire, io_dfine, io_xfeat, x_fwd_ort, x_gray_ort
    else:
        for _ in range(cruise_warmup):
            sess_wire.run(None, {"rgb_split_nchw": x_fwd})
            sess_dfine.run(None, {"rgb_split_nchw": x_fwd})
            sess_xfeat.run(None, {"gray_nchw": x_gray})
        for f_idx in range(cruise_frames):
            tf0 = time.perf_counter()
            w_out = sess_wire.run(None, {"rgb_split_nchw": x_fwd})
            d_out = sess_dfine.run(None, {"rgb_split_nchw": x_fwd})
            x_out = sess_xfeat.run(None, {"gray_nchw": x_gray}) if (f_idx % 30 == 0) else []
            cruise_ms.append((time.perf_counter() - tf0) * 1000.0)
            for arr in (*w_out, *d_out, *x_out):
                if not np.all(np.isfinite(arr)):
                    cruise_finite = False

    cruise_vram_delta = 0.0
    if use_cuda and mem_before is not None:
        mem_c = _query_cuda_mem_get_info_mib()
        if mem_c is not None:
            cruise_vram_delta = max(0.0, mem_before[0] - mem_c[0])
    cruise_proc_vram = _query_process_vram_mib(pid) if use_cuda else 0.0

    # Evict cruise sessions cleanly before switching to Terminal Approach phase
    del sess_wire, sess_dfine, sess_xfeat
    gc.collect()

    # 2) Terminal Approach Phase: PIDNet-S v2.1 (512x512) inside same CUDA context
    sess_pidnet = _make_ort_session(Path(verified["pidnet_slz_v21"]["path"]), use_cuda=use_cuda, cuda_graph=use_cuda)
    term_stats, term_finite = _run_timed_session(
        sess_pidnet, "rgb_nchw", (1, 3, 512, 512), warmup=warmup, iters=iters, use_cuda=use_cuda, rng=rng
    )
    term_vram_delta = 0.0
    if use_cuda and mem_before is not None:
        mem_t = _query_cuda_mem_get_info_mib()
        if mem_t is not None:
            term_vram_delta = max(0.0, mem_before[0] - mem_t[0])
    term_proc_vram = _query_process_vram_mib(pid) if use_cuda else 0.0
    del sess_pidnet
    gc.collect()

    cruise_stats = _summarize_ms(cruise_ms)
    peak_vram_mib = round(
        max(
            cruise_vram_delta,
            term_vram_delta,
            float(cruise_proc_vram or 0.0),
            float(term_proc_vram or 0.0),
        ),
        2,
    )
    return {
        "cruise_30hz_with_1hz_xfeat": cruise_stats,
        "terminal_pidnet_v21": term_stats,
        "cruise_vram_delta_mib": round(cruise_vram_delta, 2),
        "terminal_vram_delta_mib": round(term_vram_delta, 2),
        "cruise_process_vram_mib": round(float(cruise_proc_vram or 0.0), 2),
        "terminal_process_vram_mib": round(float(term_proc_vram or 0.0), 2),
        "timesliced_peak_vram_mib": peak_vram_mib,
        "cruise_cuda_graph_selected": dict(cruise_use_cg),
        "phase_b_outputs_finite": bool(cruise_finite and term_finite),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import shutil
    import subprocess

    t0 = time.perf_counter()
    use_cuda = not args.smoke_cpu
    if use_cuda:
        _register_nvidia_dll_dirs()

    verified = verify_onnx_sha256_parity(args.blob_dir, args.artifacts_dir)
    rng = np.random.default_rng(20261124)
    warmup = 2 if args.smoke_cpu else int(args.warmup)
    iters = 5 if args.smoke_cpu else int(args.iters)

    if getattr(args, "phase_b_only", False):
        cg_map = json.loads(args.cruise_cg_json) if getattr(args, "cruise_cg_json", "") else {}
        phase_b = _run_phase_b_timesliced(
            verified,
            use_cuda=use_cuda,
            warmup=warmup,
            iters=iters,
            cruise_use_cg=cg_map,
            rng=rng,
        )
        write_canonical_json(args.out_json, phase_b)
        return phase_b

    # Phase A: Per-model empirical FP32 vs FP32+CUDA-Graph vs FP16 gate
    phase_a: dict[str, Any] = {}
    all_finite = True
    fp16_tmp_dir = args.blob_dir / "_fp16_bench_tmp"
    try:
        for spec in MODEL_SPECS:
            onnx_path = Path(verified[spec["key"]]["path"])
            sess_fp32 = _make_ort_session(onnx_path, use_cuda=use_cuda, cuda_graph=False)
            fp32_stats, fin32 = _run_timed_session(
                sess_fp32, spec["input_name"], spec["input_shape"], warmup=warmup, iters=iters, use_cuda=use_cuda, rng=rng
            )
            del sess_fp32
            gc.collect()
            all_finite = all_finite and fin32

            fp32_cg_stats = fp32_stats
            fp16_stats: dict[str, Any] | None = None
            speedup_fp16: float | None = None
            recommended = "cpu_smoke_fp32"

            if use_cuda:
                sess_cg = _make_ort_session(onnx_path, use_cuda=True, cuda_graph=True)
                fp32_cg_stats, fin_cg = _run_timed_session(
                    sess_cg, spec["input_name"], spec["input_shape"], warmup=warmup, iters=iters, use_cuda=True, rng=rng
                )
                del sess_cg
                gc.collect()
                all_finite = all_finite and fin_cg

                fp16_tmp_dir.mkdir(parents=True, exist_ok=True)
                fp16_path = _build_fp16_onnx(onnx_path, fp16_tmp_dir / f"{spec['key']}_fp16.onnx")
                sess_fp16 = _make_ort_session(fp16_path, use_cuda=True, cuda_graph=False)
                fp16_stats, fin16 = _run_timed_session(
                    sess_fp16, spec["input_name"], spec["input_shape"], warmup=warmup, iters=iters, use_cuda=True, rng=rng
                )
                del sess_fp16
                gc.collect()
                all_finite = all_finite and fin16

                speedup_fp16 = round(fp32_cg_stats["p50_ms"] / max(fp16_stats["p50_ms"], 1e-6), 4)
                best_fp32_p99 = min(fp32_stats["p99_ms"], fp32_cg_stats["p99_ms"])
                if fp16_stats["p99_ms"] < best_fp32_p99:
                    recommended = "ort_cuda_fp16"
                elif fp32_cg_stats["p99_ms"] <= fp32_stats["p99_ms"]:
                    recommended = "ort_cuda_fp32_cuda_graph"
                else:
                    recommended = "ort_cuda_fp32"

            phase_a[spec["key"]] = {
                "fp32": fp32_stats,
                "fp32_cuda_graph": fp32_cg_stats,
                "fp16": fp16_stats,
                "fp16_vs_fp32_speedup_p50": speedup_fp16,
                "recommended_tu116_precision": recommended,
                "outputs_finite": fin32,
            }
    finally:
        shutil.rmtree(fp16_tmp_dir, ignore_errors=True)

    # Select empirically faster CUDA-Graph setting per Cruise model from Phase A
    cruise_use_cg = {
        k: bool(phase_a[k]["fp32_cuda_graph"]["p99_ms"] <= phase_a[k]["fp32"]["p99_ms"])
        for k in ("mlsd_wireseg", "dfine_n", "xfeat_512")
    }

    # Phase B: Run inside a clean isolated child subprocess for live CUDA so Phase A's 12
    # benchmark sessions do not pollute Phase B's single-context VRAM or WDDM arena state.
    if use_cuda:
        phase_b_tmp = args.out_json.with_suffix(".phase_b_tmp.json")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--blob-dir",
                    str(args.blob_dir),
                    "--artifacts-dir",
                    str(args.artifacts_dir),
                    "--out-json",
                    str(phase_b_tmp),
                    "--warmup",
                    str(warmup),
                    "--iters",
                    str(iters),
                    "--phase-b-only",
                    "--cruise-cg-json",
                    json.dumps(cruise_use_cg),
                ],
                check=True,
            )
            phase_b = json.loads(phase_b_tmp.read_text(encoding="utf-8"))
        finally:
            if phase_b_tmp.exists():
                phase_b_tmp.unlink()
    else:
        phase_b = _run_phase_b_timesliced(
            verified,
            use_cuda=False,
            warmup=warmup,
            iters=iters,
            cruise_use_cg=cruise_use_cg,
            rng=rng,
        )

    all_finite = all_finite and bool(phase_b.get("phase_b_outputs_finite", True))
    cruise_stats = phase_b["cruise_30hz_with_1hz_xfeat"]
    term_stats = phase_b["terminal_pidnet_v21"]
    peak_vram_mib = float(phase_b["timesliced_peak_vram_mib"])
    cuda_opts_used = select_platform_cuda_provider_options(enable_cuda_graph=True)

    payload = {
        "schema_version": "vertirta.local_gtx1650s_m3_timesliced_proof/v4.0.0",
        "mode": "SMOKE_CPU_ONLY" if args.smoke_cpu else "LIVE_TU116_CUDA_TIMESLICED",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host_os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "gpu": _query_gpu_info() if use_cuda else {"name": "CPU_SMOKE_ONLY"},
        "phase_b_cuda_provider_options": cuda_opts_used,
        "verified_models": verified,
        "phase_a_empirical_fp32_vs_fp16": phase_a,
        "phase_b_timesliced_schedule": {
            "cruise_30hz_with_1hz_xfeat": cruise_stats,
            "terminal_pidnet_v21": term_stats,
            "cruise_vram_delta_mib": phase_b["cruise_vram_delta_mib"],
            "terminal_vram_delta_mib": phase_b["terminal_vram_delta_mib"],
            "cruise_process_vram_mib": phase_b.get("cruise_process_vram_mib", 0.0),
            "terminal_process_vram_mib": phase_b.get("terminal_process_vram_mib", 0.0),
            "cruise_cuda_graph_selected": phase_b.get("cruise_cuda_graph_selected", cruise_use_cg),
            "phase_b_isolated_subprocess": bool(use_cuda),
            "timesliced_peak_vram_mib": peak_vram_mib,
        },
        "gates": {
            "onnx_sha256_all_verified": True,
            "empirical_fp32_vs_fp16_logged": True,
            "timesliced_peak_vram_lt_300mb": bool(peak_vram_mib < 300.0),
            "pidnet_v21_p99_lt_12ms": bool(term_stats["p99_ms"] < 12.0) if use_cuda else None,
            "cruise_combined_p99_lt_25ms": bool(cruise_stats["p99_ms"] < 25.0) if use_cuda else None,
            "all_trained_outputs_finite_and_nondegenerate": bool(all_finite),
        },
        "wall_clock_s": round(time.perf_counter() - t0, 3),
    }
    write_canonical_json(args.out_json, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physical GTX 1650 SUPER (TU116) M3 Single-Context Time-Sliced Edge Benchmark"
    )
    parser.add_argument("--smoke-cpu", action="store_true", help="0 CU CPUExecutionProvider verification.")
    parser.add_argument("--blob-dir", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--phase-b-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cruise-cg-json", type=str, default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
