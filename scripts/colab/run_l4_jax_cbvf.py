#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Colab L4 (24 GB) 4D JAX HJI-CBVF PDE + 2D CEL LUT Solver & Soundness Manifest Generator.

Runs the exact 2:1 nested 81x57x81x57 FP32 WENO5 + TVD-RK3 4D Hamilton-Jacobi-Isaacs
Variational Inequality (StanfordASL/hj_reachability v0.7.0 + JAX CUDA 12) on an NVIDIA L4
(24 GB) or T4 (16 GB), exports exact 11,313,864 B `.vrta_lut` and 11,230 B `.vrta_cel2d`
binaries, runs DR-2 Errata Locks & soundness audits across all 21,316,689 fine-grid nodes,
and emits a lightweight SHA-256 provenance manifest JSON suitable for version control.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reachability.cbvf_4d_dynamics import (  # noqa: E402
    HAS_HJ_REACHABILITY,
    HAS_JAX,
    RegimeSpec,
    VertiRta4DDynamics,
    build_regimes_from_config,
    compute_all_regime_v_caps,
    conservative_smax_4,
    hj,
    jax,
    jnp,
    load_vehicle_params,
)
from reachability.solve_cbvf_luts import (  # noqa: E402
    FLAG_CONSERVATIVE_SMAX,
    FLAG_HAS_V_CAP,
    LUT_BETA_SMAX,
    LUT_EXPORT_SHAPE,
    LUT_FINE_SHAPE,
    LUT_HEADER_BYTES,
    LUT_HI,
    LUT_HORIZON_S,
    LUT_LO,
    LUT_MAGIC,
    LUT_NUM_REGIMES,
    LUT_OFFSET,
    LUT_PAYLOAD_BYTES,
    LUT_SCALE,
    LUT_TOTAL_BYTES,
    LUT_V_HI,
    LUT_V_LO,
    LUT_VERSION,
    VrtaLutHeader,
    _quantize_int16,
    _solve_regime_quick_cpu,
    _upsample_3d_2x,
    pack_lut_header,
)
from reachability.solve_cel_2d_lut import (  # noqa: E402
    CEL2D_TOTAL_BYTES,
    solve_and_export_cel_2d,
)
import reachability.verify_lut_soundness as _vls_mod  # noqa: E402
from reachability.verify_lut_soundness import verify_all_soundness  # noqa: E402


def query_nvidia_gpu_info() -> dict[str, Any]:
    """Query GPU model, driver, and total/used VRAM via nvidia-smi."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        parts = [p.strip() for p in out.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "driver_version": parts[1],
            "vram_total_mib": int(float(parts[2])),
            "vram_used_mib": int(float(parts[3])),
        }
    except Exception as exc:
        return {
            "name": "UNAVAILABLE",
            "driver_version": "UNKNOWN",
            "vram_total_mib": 0,
            "vram_used_mib": 0,
            "error": str(exc),
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_exact_21m_eps_grid_cpu_chunked(
    v_fine_fp32: np.ndarray,
    v_deq_coarse_fp32: np.ndarray,
) -> float:
    """Exact 21,316,689-node 2:1 nested optimistic-overestimation audit without GPU OOM.

    For every node of the 81x57x81x57 fine grid inside the safety band |V_fine| <= 10 m,
    computes exact quadrilinear interpolation V_hat(x) of the quantized 41x29x41x29 table
    via slab-by-slab 2:1 nested upsampling (_upsample_3d_2x) and returns:
        eps_grid_f = max_{|V_fine| <= 10 m} max(0, V_hat(x) - clip(V_fine(x))) + 0.5 * q
    """
    if v_fine_fp32.shape != LUT_FINE_SHAPE:
        raise ValueError(f"Expected fine shape {LUT_FINE_SHAPE}, got {v_fine_fp32.shape}")
    if v_deq_coarse_fp32.shape != LUT_EXPORT_SHAPE:
        raise ValueError(f"Expected coarse shape {LUT_EXPORT_SHAPE}, got {v_deq_coarse_fp32.shape}")

    v_lo = np.float32(LUT_V_LO)
    v_hi = np.float32(LUT_V_HI)
    band_lim = np.float32(10.0)

    def _slice_max_over(ir: int, v_hat_slice: np.ndarray) -> float:
        vf = np.clip(v_fine_fp32[ir], v_lo, v_hi)
        over = np.where(
            np.abs(vf) <= band_lim,
            np.maximum(np.float32(0.0), v_hat_slice - vf),
            np.float32(0.0),
        )
        return float(np.max(over))

    eps_max = 0.0
    slab_lo = _upsample_3d_2x(v_deq_coarse_fp32[0])
    for cr in range(LUT_EXPORT_SHAPE[0] - 1):
        eps_max = max(eps_max, _slice_max_over(2 * cr, slab_lo))
        slab_hi = _upsample_3d_2x(v_deq_coarse_fp32[cr + 1])
        slab_mid = np.float32(0.5) * (slab_lo + slab_hi)
        eps_max = max(eps_max, _slice_max_over(2 * cr + 1, slab_mid))
        slab_lo = slab_hi
    eps_max = max(eps_max, _slice_max_over(LUT_FINE_SHAPE[0] - 1, slab_lo))
    return float(eps_max + 0.5 * float(LUT_SCALE))


class _ConcreteVertiRta4DDynamics(VertiRta4DDynamics):
    """Concrete hj_reachability.Dynamics subclass completing abstract ODE & optimal-input hooks.

    Preserves VertiRta4DDynamics.hamiltonian() and partial_max_magnitudes() verbatim
    (all DR-2 Errata Locks intact) while satisfying hj.Dynamics ABC instantiation.
    """

    def __call__(self, state: Any, control: Any, disturbance: Any, time: Any) -> Any:
        xp = jnp if (jnp is not None and not isinstance(state, np.ndarray)) else np
        _r, vr, _z, vz = state[0], state[1], state[2], state[3]
        u_r, u_z = control[0], control[1]
        d_vr, d_ar, d_vz, d_az = disturbance[0], disturbance[1], disturbance[2], disturbance[3]
        return xp.array(
            [
                vr + d_vr,
                u_r - self.cd_r * vr * xp.abs(vr) + d_ar,
                vz + d_vz,
                u_z - self.cd_z * vz * xp.abs(vz) + d_az,
            ],
            dtype=xp.float32,
        )

    def optimal_control_and_disturbance(self, state: Any, time: Any, grad_value: Any) -> tuple[Any, Any]:
        xp = jnp if (jnp is not None and not isinstance(grad_value, np.ndarray)) else np
        p_r, p_vr, p_z, p_vz = grad_value[0], grad_value[1], grad_value[2], grad_value[3]

        nrm = xp.sqrt(p_vr * p_vr + p_vz * p_vz + 1e-12)
        fr_arc = self.a_T * p_vr / nrm
        fz_arc = self.a_T * p_vz / nrm
        valid_arc = (fz_arc >= self.fz_cap - 1e-6) & (xp.abs(fr_arc) <= fz_arc * self.tan_th + 1e-6)
        uz_cap = self.fz_cap - self.g
        uz_lo = self.fz_lo - self.g

        cand_ur = xp.array([fr_arc, 0.0, self.fr_cap, -self.fr_cap, self.fr_bot, -self.fr_bot], dtype=xp.float32)
        cand_uz = xp.array([fz_arc - self.g, self.a_T - self.g, uz_cap, uz_cap, uz_lo, uz_lo], dtype=xp.float32)
        cand_val = p_vr * cand_ur + p_vz * cand_uz
        cand_val = cand_val.at[0].set(xp.where(valid_arc, cand_val[0], -1e9)) if xp is jnp else np.where(
            np.arange(6) == 0, np.where(valid_arc, cand_val[0], -1e9), cand_val
        )
        best = xp.argmax(cand_val)
        u_opt = xp.array([cand_ur[best], cand_uz[best]], dtype=xp.float32)
        d_opt = xp.array(
            [
                -xp.sign(p_r) * self.d_vr,
                -xp.sign(p_vr) * self.d_ar,
                -xp.sign(p_z) * self.d_vz,
                -xp.sign(p_vz) * self.d_az,
            ],
            dtype=xp.float32,
        )
        return u_opt, d_opt


def solve_regime_jax_l4_safe(
    spec: RegimeSpec,
    *,
    horizon_s: float = LUT_HORIZON_S,
    beta: float = LUT_BETA_SMAX,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Run exact 81x57x81x57 FP32 WENO5 + TVD-RK3 HJI-VI solve on L4/T4 GPU."""
    if not (HAS_JAX and HAS_HJ_REACHABILITY and jax is not None and jnp is not None and hj is not None):
        raise RuntimeError(
            "JAX CUDA + StanfordASL/hj_reachability v0.7.0 are required for live L4 solve."
        )

    t0 = time.perf_counter()
    lo = np.array(LUT_LO, dtype=np.float32)
    hi = np.array(LUT_HI, dtype=np.float32)
    grid_fine = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain=hj.sets.Box(lo=lo, hi=hi),
        shape=LUT_FINE_SHAPE,
    )
    sr = grid_fine.states[..., 0] - np.float32(spec.r_safe)
    sz = grid_fine.states[..., 2] - np.float32(spec.z_safe)
    l_init = conservative_smax_4(sr, sz, beta=beta)

    settings = hj.SolverSettings.with_accuracy(
        "very_high",  # WENO5 + TVD-RK3
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )
    dyn = _ConcreteVertiRta4DDynamics.from_regime_spec(spec)
    v_fine_jax = hj.step(
        settings, dyn, grid_fine, 0.0, l_init, -float(horizon_s), progress_bar=False
    )
    v_fine_jax.block_until_ready()
    solve_s = time.perf_counter() - t0

    # Capture peak device memory before transferring to host.
    peak_vram_bytes = 0
    try:
        mem_stats = jax.local_devices()[0].memory_stats() or {}
        peak_vram_bytes = int(mem_stats.get("peak_bytes_in_use", 0))
    except Exception:
        peak_vram_bytes = 0

    v_fine_np = np.asarray(v_fine_jax, dtype=np.float32)
    del v_fine_jax, l_init, sr, sz, grid_fine
    gc.collect()

    t_audit0 = time.perf_counter()
    v_exp_np = np.ascontiguousarray(v_fine_np[::2, ::2, ::2, ::2], dtype=np.float32)
    vq = _quantize_int16(v_exp_np)
    v_deq = vq.astype(np.float32) * LUT_SCALE + LUT_OFFSET
    eps_grid_f = audit_exact_21m_eps_grid_cpu_chunked(v_fine_np, v_deq)
    audit_s = time.perf_counter() - t_audit0

    # Quantify difference against deterministic kinematic CPU proxy on exact coarse lattice.
    vq_proxy, eps_proxy = _solve_regime_quick_cpu(spec, horizon_s=horizon_s, beta=beta)
    v_proxy_deq = vq_proxy.astype(np.float32) * LUT_SCALE + LUT_OFFSET
    diff = v_deq - v_proxy_deq
    abs_diff = np.abs(diff)
    safe_frac_jax = float(np.mean((v_deq - np.float32(eps_grid_f)) >= 0.0))
    safe_frac_proxy = float(np.mean((v_proxy_deq - np.float32(eps_proxy)) >= 0.0))

    stats = {
        "regime_index": spec.index,
        "regime_name": spec.name,
        "tw_ratio": spec.tw_ratio,
        "tilt_max_deg": spec.tilt_max_deg,
        "solve_wall_s": round(solve_s, 3),
        "audit_21m_wall_s": round(audit_s, 3),
        "eps_grid_m": round(eps_grid_f, 6),
        "proxy_eps_grid_m": round(eps_proxy, 6),
        "jax_peak_vram_mib": round(peak_vram_bytes / (1024.0 * 1024.0), 2),
        "safe_volume_frac_jax": round(safe_frac_jax, 6),
        "safe_volume_frac_proxy": round(safe_frac_proxy, 6),
        "pde_vs_proxy_abs_diff_m": {
            "mean": round(float(np.mean(abs_diff)), 5),
            "p95": round(float(np.quantile(abs_diff, 0.95)), 5),
            "p99": round(float(np.quantile(abs_diff, 0.99)), 5),
            "max": round(float(np.max(abs_diff)), 5),
            "signed_mean_jax_minus_proxy": round(float(np.mean(diff)), 5),
        },
    }
    del v_fine_np
    gc.collect()
    return vq, eps_grid_f, stats


def enforce_conservative_regime_hierarchy(
    vq_by_regime: list[np.ndarray],
) -> dict[str, Any]:
    """Conservatively enforce V_0 >= V_1 >= V_3 >= V_2 via lower-envelope min().

    Never increases any value node (strictly conservative), eliminating minor WENO5
    ghost-cell non-monotonicity deep inside the already-unsafe interior.
    """
    hierarchy_pairs = ((0, 1), (1, 3), (3, 2))
    summary: dict[str, Any] = {}
    for r_hi, r_lo in hierarchy_pairs:
        hi_arr = vq_by_regime[r_hi]
        lo_arr = vq_by_regime[r_lo]
        mask = lo_arr > hi_arr
        n_proj = int(np.count_nonzero(mask))
        max_v_proj = float(
            np.max(lo_arr[mask].astype(np.float64) * float(LUT_SCALE) + float(LUT_OFFSET))
        ) if n_proj > 0 else None
        if n_proj > 0:
            vq_by_regime[r_lo] = np.minimum(lo_arr, hi_arr).astype("<i2")
        summary[f"r{r_hi}_ge_r{r_lo}"] = {
            "projected_nodes": n_proj,
            "projected_frac": round(n_proj / float(lo_arr.size), 8),
            "max_v_projected_m": round(max_v_proj, 4) if max_v_proj is not None else None,
        }
    return summary


def run_l4_solve_and_manifest(
    *,
    out_lut: Path,
    out_cel2d: Path,
    out_manifest: Path,
    config_path: Path | None = None,
    horizon_s: float = LUT_HORIZON_S,
    allow_large_gpu: bool = False,
) -> dict[str, Any]:
    """Execute full 4-regime JAX 4D HJI-CBVF + 2D CEL solve and emit verified manifest JSON."""
    t_total0 = time.perf_counter()
    gpu_info = query_nvidia_gpu_info()
    vram_gib = gpu_info.get("vram_total_mib", 0) / 1024.0
    if vram_gib >= 60.0 and not allow_large_gpu:
        raise RuntimeError(
            f"Cost-lock guard: detected {gpu_info['name']} ({vram_gib:.1f} GiB >= 60 GiB). "
            "Use Colab --gpu L4 (24 GB) or --gpu T4 (16 GB) for <6 GB 4D HJI-CBVF PDE solve."
        )

    out_lut.parent.mkdir(parents=True, exist_ok=True)
    out_cel2d.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    params = load_vehicle_params(config_path)
    regimes = build_regimes_from_config(config_path)
    v_cap_f = compute_all_regime_v_caps(
        regimes,
        r_max=LUT_HI[0],
        horizon_s=horizon_s,
        v_max=params["v_max_mps"],
    )

    vq_by_regime: list[np.ndarray] = []
    eps_grid_list: list[float] = []
    regime_stats: list[dict[str, Any]] = []
    max_peak_vram_mib = 0.0

    for spec in regimes:
        print(
            f"[run_l4_jax_cbvf] Solving regime {spec.index} ({spec.name}, "
            f"T/W={spec.tw_ratio:.2f}, tilt={spec.tilt_max_deg:.1f} deg) on 81x57x81x57 FP32...",
            flush=True,
        )
        vq, eps_f, st = solve_regime_jax_l4_safe(spec, horizon_s=horizon_s)
        vq_by_regime.append(np.ascontiguousarray(vq, dtype="<i2"))
        eps_grid_list.append(eps_f)
        regime_stats.append(st)
        max_peak_vram_mib = max(max_peak_vram_mib, float(st["jax_peak_vram_mib"]))
        print(
            f"  -> done in {st['solve_wall_s']:.2f}s (audit {st['audit_21m_wall_s']:.2f}s), "
            f"eps_grid={eps_f:.5f} m, peak_vram={st['jax_peak_vram_mib']:.1f} MiB, "
            f"PDE-vs-proxy mean|dV|={st['pde_vs_proxy_abs_diff_m']['mean']:.4f} m",
            flush=True,
        )

    # Conservative lower-envelope monotonicity projection (V_lo <- min(V_lo, V_hi)) to remove
    # minor WENO5 ghost-cell non-monotonicity strictly in deep-unsafe corners without ever
    # overestimating safety (preserves exact 21.3M-node eps_grid_f soundness).
    proj_stats = enforce_conservative_regime_hierarchy(vq_by_regime)
    payload = bytearray()
    for vq in vq_by_regime:
        payload.extend(np.ascontiguousarray(vq, dtype="<i2").tobytes(order="C"))

    if len(payload) != LUT_PAYLOAD_BYTES:
        raise RuntimeError(f"Payload mismatch: {len(payload)} B != {LUT_PAYLOAD_BYTES} B")

    payload_sha256 = hashlib.sha256(payload).digest()
    flags = FLAG_HAS_V_CAP | FLAG_CONSERVATIVE_SMAX  # bit0=0 => authentic JAX PDE solve!
    hdr = VrtaLutHeader(
        magic=LUT_MAGIC,
        version=LUT_VERSION,
        num_regimes=LUT_NUM_REGIMES,
        dims=LUT_EXPORT_SHAPE,
        lo=LUT_LO,
        hi=LUT_HI,
        scale=float(LUT_SCALE),
        offset=float(LUT_OFFSET),
        eps_grid=tuple(eps_grid_list),  # type: ignore[arg-type]
        payload_sha256=payload_sha256,
        v_cap_f=v_cap_f,
        flags=flags,
        horizon_s=float(horizon_s),
        d_ar_max=float(regimes[0].d_ar),
        d_az_max=float(regimes[0].d_az),
    )
    hdr_bytes = pack_lut_header(hdr)
    assert len(hdr_bytes) == LUT_HEADER_BYTES
    with out_lut.open("wb") as f:
        f.write(hdr_bytes)
        f.write(payload)
    assert out_lut.stat().st_size == LUT_TOTAL_BYTES

    t_cel0 = time.perf_counter()
    solve_and_export_cel_2d(out_cel2d, config_path=config_path)
    cel_s = time.perf_counter() - t_cel0
    assert out_cel2d.stat().st_size == CEL2D_TOTAL_BYTES

    # Bind concrete Dynamics subclass when hj_reachability is installed so Erratum #1 CFL check runs.
    prev_cls = _vls_mod.VertiRta4DDynamics
    _vls_mod.VertiRta4DDynamics = _ConcreteVertiRta4DDynamics  # type: ignore[assignment]
    try:
        soundness = verify_all_soundness(out_lut, out_cel2d)
    finally:
        _vls_mod.VertiRta4DDynamics = prev_cls
    if not soundness["passed"]:
        raise RuntimeError(f"Soundness verification FAILED: {soundness['failures']}")

    total_s = time.perf_counter() - t_total0
    gpu_after = query_nvidia_gpu_info()
    jax_ver = getattr(jax, "__version__", "unknown") if jax is not None else "unavailable"
    hj_ver = getattr(hj, "__version__", "unknown") if hj is not None else "unavailable"

    manifest: dict[str, Any] = {
        "schema_version": "vertirta.jax_4d_cbvf_l4_manifest/v1.0.0",
        "solver_mode": "jax_hj_reachability_weno5_tvdrk3_fp32",
        "is_quick_cpu_fixture": False,
        "fine_grid_shape": list(LUT_FINE_SHAPE),
        "fine_grid_nodes_per_regime": int(np.prod(LUT_FINE_SHAPE)),
        "export_grid_shape": list(LUT_EXPORT_SHAPE),
        "horizon_s": float(horizon_s),
        "hardware": {
            "gpu_name": gpu_info["name"],
            "driver_version": gpu_info["driver_version"],
            "vram_total_mib": gpu_info["vram_total_mib"],
            "jax_peak_vram_mib": round(max_peak_vram_mib, 2),
            "nvidia_smi_used_after_mib": gpu_after.get("vram_used_mib", 0),
            "python_version": platform.python_version(),
            "jax_version": jax_ver,
            "hj_reachability_version": hj_ver,
        },
        "timing_s": {
            "total_wall_s": round(total_s, 3),
            "cel_2d_wall_s": round(cel_s, 3),
            "regimes_solve_sum_s": round(sum(r["solve_wall_s"] for r in regime_stats), 3),
        },
        "artifacts": {
            "cbvf_4d_lut": {
                "filename": out_lut.name,
                "size_bytes": LUT_TOTAL_BYTES,
                "file_sha256": sha256_file(out_lut),
                "payload_sha256": payload_sha256.hex(),
                "flags": flags,
                "eps_grid_m": [round(e, 6) for e in eps_grid_list],
                "v_cap_f_mps": [round(v, 4) for v in v_cap_f],
            },
            "cel_2d_lut": {
                "filename": out_cel2d.name,
                "size_bytes": CEL2D_TOTAL_BYTES,
                "file_sha256": sha256_file(out_cel2d),
                "payload_sha256": soundness["cel2d_payload_sha256"],
                "eps_grid_m": [round(float(e), 6) for e in soundness["cel2d_eps_grid"]],
                "max_curve_err_m": round(float(soundness["cel2d_max_curve_err_m"]), 5),
                "tier_errors_m": {
                    k: round(float(v), 5) for k, v in soundness["cel2d_tier_errors_m"].items()
                },
            },
        },
        "errata_locks": {
            "erratum1_nom_max_ur_mps2": round(float(soundness["erratum1_nom_max_ur"]), 4),
            "erratum1_max_cfl_ratio": round(float(soundness["erratum1_max_cfl_ratio"]), 6),
            "erratum2_smax_zero_m": round(float(soundness["erratum2_smax_zero"]), 8),
            "erratum3_v_cap_2m_cross_mps": round(float(v_cap_f[2]), 4),
            "conservative_hierarchy_projection": proj_stats,
            "soundness_passed": True,
        },
        "regimes": regime_stats,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()

    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[run_l4_jax_cbvf] ALL SOUNDNESS CHECKS PASSED -> {out_manifest}", flush=True)
    return manifest


def build_colab_cli_plan(
    *,
    session: str = "vertirta-jax-l4",
    gpu: str = "L4",
    local_blob_dir: str = "/tmp/vertirta_v31_blobs",
) -> list[list[str]]:
    """Return the exact cost-guarded google-colab-cli command sequence (L4/T4 ONLY)."""
    if gpu.upper() not in ("L4", "T4"):
        raise ValueError(f"GPU '{gpu}' violates cost lock; allowed values are L4 or T4 only.")
    return [
        ["colab", "new", "-s", session, "--gpu", gpu.upper()],
        [
            "colab",
            "exec",
            "-s",
            session,
            "--timeout",
            "0",
            "-f",
            "scripts/colab/run_l4_jax_cbvf.py",
        ],
        [
            "colab",
            "download",
            "-s",
            session,
            "/content/out/cbvf_4d_x8_240.vrta_lut",
            f"{local_blob_dir}/cbvf_4d_x8_240.vrta_lut",
        ],
        [
            "colab",
            "download",
            "-s",
            session,
            "/content/out/cel_2d_x8_240.vrta_cel2d",
            f"{local_blob_dir}/cel_2d_x8_240.vrta_cel2d",
        ],
        [
            "colab",
            "download",
            "-s",
            session,
            "/content/out/jax_4d_cbvf_l4_manifest.json",
            "artifacts/v3.1.0/jax_4d_cbvf_l4_manifest.json",
        ],
        ["colab", "stop", "-s", session],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run live 4D JAX HJI-CBVF PDE + 2D CEL solver on Colab L4 (24 GB)."
    )
    parser.add_argument(
        "--out-lut",
        type=Path,
        default=Path("/content/out/cbvf_4d_x8_240.vrta_lut"),
    )
    parser.add_argument(
        "--out-cel2d",
        type=Path,
        default=Path("/content/out/cel_2d_x8_240.vrta_cel2d"),
    )
    parser.add_argument(
        "--out-manifest",
        type=Path,
        default=Path("/content/out/jax_4d_cbvf_l4_manifest.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "vehicle_x8_240.yaml",
    )
    parser.add_argument("--horizon-s", type=float, default=LUT_HORIZON_S)
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the L4 Colab CLI orchestration plan without executing.",
    )
    args = parser.parse_args(argv)

    if args.print_plan:
        for cmd in build_colab_cli_plan():
            print(" ".join(cmd))
        return 0

    run_l4_solve_and_manifest(
        out_lut=args.out_lut,
        out_cel2d=args.out_cel2d,
        out_manifest=args.out_manifest,
        config_path=args.config if args.config.is_file() else None,
        horizon_s=args.horizon_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
