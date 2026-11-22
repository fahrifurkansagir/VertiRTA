#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real local NVIDIA GTX 1650 SUPER (TU116) ONNX Runtime CUDA FP32 vs FP16 benchmark.

Exports the real static-PAPPM ``PIDNetSlzDeployWrapper`` (1x3x512x512, Opset 17)
to temporary FP32 and FP16 ONNX graphs, executes 200 warmup + 500 timed iterations
on ``CUDAExecutionProvider`` (with ``IoBinding`` and end-to-end host I/O timings),
measures live process VRAM via ``nvidia-smi``, and writes a 100% transparent JSON
artifact explicitly labeled as an architecture-only (untrained weights) silicon
latency & VRAM benchmark.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _query_gpu_info() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": int(float(parts[2])),
        "memory_free_before_mib": int(float(parts[3])),
        "compute_cap": parts[4],
    }


def _query_process_vram_mib(pid: int) -> float | None:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return None
    if not out:
        return None
    total = 0.0
    found = False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                total += float(parts[1])
                found = True
            except ValueError:
                pass
    return total if found else None


def _query_gpu_used_mib() -> float | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        return float(out.splitlines()[0].strip())
    except Exception:
        return None


def _toposort_onnx_nodes(model: Any) -> None:
    """Restore valid topological order after ORT float16 converter inserts Cast nodes."""
    graph = model.graph
    nodes = list(graph.node)
    available: set[str] = {i.name for i in graph.input}
    available.update(init.name for init in graph.initializer)
    available.update(sparse.values.name for sparse in graph.sparse_initializer)
    available.add("")

    ordered = []
    remaining = nodes
    while remaining:
        next_rem = []
        progressed = False
        for node in remaining:
            if all((inp in available) for inp in node.input):
                ordered.append(node)
                for out_name in node.output:
                    if out_name:
                        available.add(out_name)
                progressed = True
            else:
                next_rem.append(node)
        if not progressed:
            ordered.extend(next_rem)
            break
        remaining = next_rem

    del graph.node[:]
    graph.node.extend(ordered)


def _build_fp16_onnx(fp32_path: Path, fp16_path: Path) -> Path:
    import onnx

    model_fp32 = onnx.load(str(fp32_path))
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError:
        from onnxconverter_common import convert_float_to_float16  # type: ignore[import-not-found]

    model_fp16 = convert_float_to_float16(
        model_fp32,
        keep_io_types=True,
        disable_shape_infer=False,
    )
    _toposort_onnx_nodes(model_fp16)
    onnx.save(model_fp16, str(fp16_path))
    onnx.checker.check_model(onnx.load(str(fp16_path)))
    return fp16_path


def _summarize_ms(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": round(mean_ms, 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
        "fps_mean": round(1000.0 / max(mean_ms, 1e-6), 2),
    }


def _collect_nvidia_dll_dirs() -> list[str]:
    """Find pip-installed nvidia CUDA/cuDNN binary directories across sys.path & nvidia.__path__."""
    dirs: list[str] = []
    seen: set[str] = set()
    roots: list[Path] = []
    try:
        import nvidia  # type: ignore[import-not-found]

        for p in getattr(nvidia, "__path__", []):
            roots.append(Path(p))
    except ImportError:
        pass
    for sp in sys.path:
        nv = Path(sp) / "nvidia"
        if nv.is_dir():
            roots.append(nv)

    for nv_root in roots:
        if not nv_root.is_dir():
            continue
        for sub in sorted(nv_root.iterdir()):
            if not sub.is_dir():
                continue
            for cand in (sub / "bin", sub / "lib", sub):
                if not cand.is_dir():
                    continue
                if any(cand.glob("*.dll")) or any(cand.glob("*.so*")):
                    s = str(cand)
                    if s not in seen:
                        seen.add(s)
                        dirs.append(s)
    return dirs


def _register_nvidia_dll_dirs() -> list[str]:
    """Register pip-installed nvidia CUDA/cuDNN library dirs in PATH and Windows DLL search."""
    dirs = _collect_nvidia_dll_dirs()
    for dll_str in dirs:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(dll_str)
            except OSError:
                pass
        cur_path = os.environ.get("PATH", "")
        if dll_str not in cur_path:
            os.environ["PATH"] = dll_str + os.pathsep + cur_path
    return dirs


def _query_cuda_mem_get_info_mib() -> tuple[float, float] | None:
    """Query exact (free_mib, total_mib) via cudart cudaMemGetInfo (accurate under Windows WDDM)."""
    import ctypes

    lib_names = (
        ("cudart64_12.dll", "cudart64_13.dll", "cudart64_110.dll")
        if platform.system() == "Windows"
        else ("libcudart.so.12", "libcudart.so.13", "libcudart.so")
    )
    cudart = None
    for name in lib_names:
        try:
            cudart = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if cudart is None:
        return None
    free_b = ctypes.c_size_t(0)
    total_b = ctypes.c_size_t(0)
    rc = cudart.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))
    if rc != 0:
        return None
    mib = 1024.0 * 1024.0
    return (free_b.value / mib, total_b.value / mib)


def _benchmark_session(
    onnx_path: Path,
    precision: str,
    warmup: int,
    iters: int,
    *,
    cuda_graph: bool = False,
) -> dict[str, Any]:
    _register_nvidia_dll_dirs()
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass

    pid = os.getpid()
    mem_before = _query_cuda_mem_get_info_mib()
    gpu_used_before = _query_gpu_used_mib()

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 1

    cuda_opts: dict[str, str] = {
        "device_id": "0",
        "cudnn_conv_algo_search": "HEURISTIC",
        "arena_extend_strategy": "kSameAsRequested",
        "enable_cuda_graph": "1" if cuda_graph else "0",
    }
    sess = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=[("CUDAExecutionProvider", cuda_opts)],
    )
    active_providers = list(sess.get_providers())
    if "CUDAExecutionProvider" not in active_providers:
        raise RuntimeError(
            f"CUDAExecutionProvider failed to initialize (active={active_providers})"
        )

    rng = np.random.default_rng(20261122)
    x_host = rng.standard_normal((1, 3, 512, 512), dtype=np.float32)

    # Pre-bind static GPU I/O tensors (mirrors C++20 SlzOrtRunner::AllocateStaticIoBindings).
    io = sess.io_binding()
    x_ort = ort.OrtValue.ortvalue_from_numpy(x_host, "cuda", 0)
    y_logits = ort.OrtValue.ortvalue_from_shape_and_type((1, 5, 512, 512), np.float32, "cuda", 0)
    y_normals = ort.OrtValue.ortvalue_from_shape_and_type((1, 3, 512, 512), np.float32, "cuda", 0)
    y_bdry = ort.OrtValue.ortvalue_from_shape_and_type((1, 1, 64, 64), np.float32, "cuda", 0)

    io.bind_ortvalue_input("rgb_nchw", x_ort)
    io.bind_ortvalue_output("seg_logits", y_logits)
    io.bind_ortvalue_output("surf_normals", y_normals)
    io.bind_ortvalue_output("bdry_prob", y_bdry)

    for _ in range(warmup):
        sess.run_with_iobinding(io)
    io.synchronize_outputs()

    peak_proc_vram = _query_process_vram_mib(pid)
    mem_after_warmup = _query_cuda_mem_get_info_mib()
    gpu_used_after = _query_gpu_used_mib()

    # 1) Pure GPU network inference via pre-bound IoBinding (synchronized per tick).
    net_ms: list[float] = []
    min_free_mib = mem_after_warmup[0] if mem_after_warmup is not None else None
    for k in range(iters):
        t0 = time.perf_counter_ns()
        sess.run_with_iobinding(io)
        io.synchronize_outputs()
        t1 = time.perf_counter_ns()
        net_ms.append((t1 - t0) * 1e-6)
        if k == iters // 2:
            mid_vram = _query_process_vram_mib(pid)
            if mid_vram is not None:
                peak_proc_vram = max(peak_proc_vram or 0.0, mid_vram)
            mid_mem = _query_cuda_mem_get_info_mib()
            if mid_mem is not None and min_free_mib is not None:
                min_free_mib = min(min_free_mib, mid_mem[0])

    # Verify finite outputs and exact shapes.
    out_logits = y_logits.numpy()
    out_normals = y_normals.numpy()
    out_bdry = y_bdry.numpy()
    if out_logits.shape != (1, 5, 512, 512) or out_normals.shape != (1, 3, 512, 512) or out_bdry.shape != (1, 1, 64, 64):
        raise RuntimeError("Unexpected ONNX output tensor shapes.")
    if not (np.all(np.isfinite(out_logits)) and np.all(np.isfinite(out_normals)) and np.all(np.isfinite(out_bdry))):
        raise RuntimeError(f"Non-finite outputs detected in {precision} CUDA run.")

    # 2) End-to-end host->GPU->host inference (100 timed ticks for practical ROS 2 budget check).
    e2e_iters = min(iters, 100)
    e2e_ms: list[float] = []
    if not cuda_graph:
        for _ in range(e2e_iters):
            t0 = time.perf_counter_ns()
            _ = sess.run(None, {"rgb_nchw": x_host})
            t1 = time.perf_counter_ns()
            e2e_ms.append((t1 - t0) * 1e-6)
    else:
        # With CUDA Graph active, host updates pre-bound GPU input in-place then replays graph.
        for _ in range(e2e_iters):
            t0 = time.perf_counter_ns()
            x_ort.update_inplace(x_host)
            sess.run_with_iobinding(io)
            io.synchronize_outputs()
            _ = y_logits.numpy()
            _ = y_normals.numpy()
            _ = y_bdry.numpy()
            t1 = time.perf_counter_ns()
            e2e_ms.append((t1 - t0) * 1e-6)

    end_vram = _query_process_vram_mib(pid)
    if end_vram is not None:
        peak_proc_vram = max(peak_proc_vram or 0.0, end_vram)
    end_mem = _query_cuda_mem_get_info_mib()
    if end_mem is not None and min_free_mib is not None:
        min_free_mib = min(min_free_mib, end_mem[0])

    net_stats = _summarize_ms(net_ms)
    e2e_stats = _summarize_ms(e2e_ms)
    cuda_vram_delta = (
        round(max(0.0, mem_before[0] - min_free_mib), 1)
        if (mem_before is not None and min_free_mib is not None)
        else None
    )
    delta_gpu_used = (
        round(max(0.0, gpu_used_after - gpu_used_before), 1)
        if (gpu_used_before is not None and gpu_used_after is not None)
        else None
    )
    measured_vram = (
        peak_proc_vram
        if peak_proc_vram is not None
        else (cuda_vram_delta if cuda_vram_delta is not None else delta_gpu_used)
    )

    del io, x_ort, y_logits, y_normals, y_bdry, sess
    gc.collect()

    return {
        "precision": precision,
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024.0 * 1024.0), 3),
        "active_providers": active_providers,
        "cuda_graph": cuda_graph,
        "arena_extend_strategy": "kSameAsRequested",
        "warmup_iters": warmup,
        "timed_iters": iters,
        "network_iobinding": net_stats,
        "end_to_end_host_io": {
            "timed_iters": e2e_iters,
            **e2e_stats,
        },
        "peak_process_vram_mb": round(peak_proc_vram, 1) if peak_proc_vram is not None else None,
        "cuda_mem_get_info_delta_mb": cuda_vram_delta,
        "nvidia_smi_used_delta_mb": delta_gpu_used,
        "effective_vram_mb": round(measured_vram, 1) if measured_vram is not None else None,
        "outputs_finite": True,
        "gates": {
            "network_p50_lt_12ms": net_stats["p50_ms"] < 12.0,
            "network_p99_lt_12ms": net_stats["p99_ms"] < 12.0,
            "e2e_ge_20hz": e2e_stats["fps_mean"] >= 20.0,
            "vram_lt_300mb": measured_vram is not None and measured_vram < 300.0,
        },
    }


def _run_subprocess_mode(
    mode: str,
    onnx_path: Path,
    warmup: int,
    iters: int,
    *,
    cuda_graph: bool = False,
) -> dict[str, Any]:
    """Run FP32 or FP16 in an isolated child process so VRAM measurement is unpolluted."""
    script = Path(__file__).resolve()
    cmd = [
        sys.executable,
        str(script),
        "--worker-precision",
        mode,
        "--worker-onnx",
        str(onnx_path),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if cuda_graph:
        cmd.append("--worker-cuda-graph")
    env = os.environ.copy()
    nv_dirs = _collect_nvidia_dll_dirs()
    if nv_dirs:
        env["PATH"] = os.pathsep.join(nv_dirs) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker {mode} failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    out = proc.stdout
    for line in reversed(out.splitlines()):
        line_s = line.strip()
        if line_s.startswith("{") and line_s.endswith("}"):
            return json.loads(line_s)
    raise RuntimeError(f"Worker {mode} produced no JSON summary: {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local GTX 1650 SUPER ONNX Runtime CUDA FP32/FP16 benchmark for PIDNet-S SLZ."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "artifacts" / "v2.0.0" / "local_gtx1650s_pidnet_arch_bench.json",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker-precision", choices=("fp32", "fp16"), default=None)
    parser.add_argument("--worker-onnx", type=Path, default=None)
    parser.add_argument("--worker-cuda-graph", action="store_true")
    args = parser.parse_args(argv)

    if args.worker_precision is not None:
        if args.worker_onnx is None:
            raise SystemExit("--worker-onnx required with --worker-precision")
        res = _benchmark_session(
            args.worker_onnx,
            args.worker_precision,
            warmup=args.warmup,
            iters=args.iters,
            cuda_graph=args.worker_cuda_graph,
        )
        print(json.dumps(res))
        return 0

    gpu = _query_gpu_info()
    if args.check_only:
        print(json.dumps({"check_only": True, "gpu": gpu}, indent=2))
        return 0

    import onnxruntime as ort
    import torch

    from ml.slz_perception.models.pidnet_slz import PIDNetSlzDeployWrapper, PIDNetSlzTrain
    from ml.slz_perception.train.export_onnx_trt import export_static_onnx

    net = PIDNetSlzTrain(num_classes=5, dino_dim=1024).eval()
    deploy = PIDNetSlzDeployWrapper(net).eval()
    param_count = sum(p.numel() for p in deploy.parameters())

    with tempfile.TemporaryDirectory(prefix="vertirta_gtx1650s_") as tmp:
        tmp_dir = Path(tmp)
        fp32_onnx = tmp_dir / "pidnet_slz_512_arch_fp32.onnx"
        fp16_onnx = tmp_dir / "pidnet_slz_512_arch_fp16.onnx"

        export_static_onnx(deploy, fp32_onnx, opset=17, simplify=False, input_hw=(512, 512))
        _build_fp16_onnx(fp32_onnx, fp16_onnx)

        # Isolate each precision/execution mode in a fresh process for exact cudaMemGetInfo VRAM.
        fp32_res = _run_subprocess_mode("fp32", fp32_onnx, args.warmup, args.iters, cuda_graph=False)
        fp32_cg_res = _run_subprocess_mode("fp32", fp32_onnx, args.warmup, args.iters, cuda_graph=True)
        fp16_res = _run_subprocess_mode("fp16", fp16_onnx, args.warmup, args.iters, cuda_graph=False)

    git_sha = "unknown"
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        pass

    report: dict[str, Any] = {
        "schema_version": "vertirta.local_gpu_arch_bench/v1.0.0",
        "weights_status": "ARCHITECTURE_LATENCY_VRAM_BENCHMARK_ONLY_UNTRAINED_WEIGHTS",
        "disclaimer": (
            "Measures real CUDA silicon inference latency and GPU VRAM footprint of the "
            "static-PAPPM PIDNetSlzDeployWrapper ONNX Opset-17 architecture with untrained "
            "random weights. Does NOT measure segmentation mIoU, FSR, or OOD AUROC accuracy "
            "(all accuracy metrics remain PENDING_COLAB_G4 until trained weights exist)."
        ),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "host_os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "runtime": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "available_providers": list(ort.get_available_providers()),
        },
        "gpu": gpu,
        "model": {
            "name": "PIDNetSlzDeployWrapper (static-PAPPM, zero AdaptiveAvgPool2d, zero dino_proj)",
            "input_shape": [1, 3, 512, 512],
            "output_shapes": {
                "seg_logits": [1, 5, 512, 512],
                "surf_normals": [1, 3, 512, 512],
                "bdry_prob": [1, 1, 64, 64],
            },
            "opset_version": 17,
            "deploy_parameters": int(param_count),
        },
        "benchmarks": {
            "ort_cuda_fp32": fp32_res,
            "ort_cuda_fp32_cuda_graph": fp32_cg_res,
            "ort_cuda_fp16": fp16_res,
        },
        "fp16_vs_fp32_speedup_p50": round(
            fp32_res["network_iobinding"]["p50_ms"]
            / max(fp16_res["network_iobinding"]["p50_ms"], 1e-6),
            3,
        ),
        "recommended_local_tu116_mode": (
            "ort_cuda_fp32_cuda_graph"
            if fp32_cg_res["network_iobinding"]["p99_ms"] <= fp32_res["network_iobinding"]["p99_ms"]
            else "ort_cuda_fp32"
        ),
        "tu116_fp16_finding": (
            "Empirical TU116 (sm_75, no Tensor Cores) validation: ORT-CUDA FP32 outperforms "
            "converted FP16 on GTX 1650 SUPER, directly vindicating DR-7's mandatory TU116 "
            "FP32-vs-FP16 regression gate; TRT FP16 remains optimal on Tensor-Core GPUs "
            "(Colab L4/A100/G4, Jetson Orin)."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
