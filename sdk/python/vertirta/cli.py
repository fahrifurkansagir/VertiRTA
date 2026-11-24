"""VertiRTA Flat Typer CLI (sim, inspect, replay, bench, atlas, verify-stl, verify, gcs)."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import time
from typing import Optional

import numpy as np
import typer

from . import __version__, get_backend_name
from .env import (
    FAULT_MODE_MAP,
    KARTAL_DATUM,
    VRTA_FRAME_SIZE,
    VRTA_HEADER_SIZE,
    VertiRtaCoaxialX8Env,
    crc32c_py,
    make_vrta_header_bytes,
    verify_vrta_frame_bytes,
)
from .m3 import (
    VertiForensicsEngine,
    VertiForwardHazard,
    VertiGnssDenied,
    VertiMondrianVoiceGate,
)
from .vec_env import VertiVecEnv

FORBIDDEN_PORTS: frozenset[int] = frozenset({4173, 4174, 8765, 9222, 9223})
_HEADER_UNPACK = struct.Struct("<8sII4d")
_FRAME_SUMMARY_UNPACK = struct.Struct("<2Q3d3f4f3f3f3f8f7f4BI")

app = typer.Typer(
    name="vertirta",
    no_args_is_help=True,
    help=(
        "VertiRTA Coaxial-X8 eVTOL Run-Time Assurance CLI — "
        "Kartal Coastal Vertiport (40.8860 N, 29.1880 E) & LTFJ/LTBX corridor."
    ),
)


def _validate_port(port: int, label: str) -> int:
    if port in FORBIDDEN_PORTS or not (1024 <= port <= 65535):
        typer.echo(
            f"ERROR: Forbidden or out-of-range {label} port {port} "
            f"(hard-banned: {sorted(FORBIDDEN_PORTS)}; recommended: 5180 / 8891).",
            err=True,
        )
        raise typer.Exit(code=2)
    return port


def _parse_vrta_summary(vrta_path: Path) -> dict[str, object]:
    raw = vrta_path.read_bytes()
    if len(raw) < VRTA_HEADER_SIZE:
        raise ValueError(f"Truncated .vrta file ({len(raw)} B < {VRTA_HEADER_SIZE} B)")
    header_crc_stored = struct.unpack_from("<I", raw, 124)[0]
    header_crc_calc = crc32c_py(raw[:124])
    if header_crc_stored != header_crc_calc:
        raise ValueError("VRTA0001 header Castagnoli CRC-32C mismatch")

    magic, schema_ver, _, lat, lon, amsl, n_egm = _HEADER_UNPACK.unpack_from(raw, 0)
    if magic != b"VRTA0001":
        raise ValueError(f"Unexpected .vrta magic {magic!r}")

    payload_len = len(raw) - VRTA_HEADER_SIZE
    if payload_len % VRTA_FRAME_SIZE != 0:
        raise ValueError(f"Misaligned .vrta payload: {payload_len} B not divisible by 256 B")
    n_frames = payload_len // VRTA_FRAME_SIZE

    v_min = float("inf")
    v_max = float("-inf")
    agl_min = float("inf")
    agl_max = float("-inf")
    ltfj_min = float("inf")
    ltbx_min = float("inf")
    cel_frames = 0
    bad_crc = 0

    for idx in range(n_frames):
        off = VRTA_HEADER_SIZE + idx * VRTA_FRAME_SIZE
        chunk = raw[off : off + VRTA_FRAME_SIZE]
        if not verify_vrta_frame_bytes(chunk):
            bad_crc += 1
        fields = _FRAME_SUMMARY_UNPACK.unpack_from(chunk, 0)
        v_term = float(fields[29])
        ltfj_m = float(fields[32])
        ltbx_m = float(fields[33])
        agl_m = float(fields[35])
        cel = int(fields[39])
        v_min = min(v_min, v_term)
        v_max = max(v_max, v_term)
        agl_min = min(agl_min, agl_m)
        agl_max = max(agl_max, agl_m)
        ltfj_min = min(ltfj_min, ltfj_m)
        ltbx_min = min(ltbx_min, ltbx_m)
        if cel != 0:
            cel_frames += 1

    return {
        "magic": magic.decode("ascii"),
        "schema_ver": int(schema_ver),
        "frames": n_frames,
        "duration_s": round(n_frames * 0.01, 2),
        "bad_crc_frames": bad_crc,
        "datum": {
            "lat_deg": round(lat, 6),
            "lon_deg": round(lon, 6),
            "amsl_m": round(amsl, 2),
            "egm2008_n_m": round(n_egm, 2),
        },
        "v_term_range_v": [round(v_min, 2), round(v_max, 2)] if n_frames else [0.0, 0.0],
        "agl_range_m": [round(agl_min, 2), round(agl_max, 2)] if n_frames else [0.0, 0.0],
        "min_ltfj_margin_m": round(ltfj_min, 2) if n_frames else 0.0,
        "min_ltbx_margin_m": round(ltbx_min, 2) if n_frames else 0.0,
        "cel_frames": cel_frames,
    }


@app.command("sim")
def cmd_sim(
    airframe: str = typer.Option("x8", help="Airframe identifier (locked: x8 = VertiRTA-X8-240)."),
    duration: float = typer.Option(1.0, min=0.05, max=60.0, help="Simulation duration in seconds."),
    fault: str = typer.Option("none", help="Fault mode: none|single_motor|same_arm_dual|imu_bias|wind_gust."),
    seed: int = typer.Option(42, help="Deterministic RNG seed."),
    out: Path = typer.Option(Path("vertirta_sim.vrta"), help="Output VRTA0001 binary log path."),
) -> None:
    """Run deterministic 400 Hz Coaxial-X8 simulation over Kartal Coastal Vertiport and emit .vrta."""
    if airframe.lower() != "x8":
        typer.echo(f"ERROR: Unsupported airframe {airframe!r}; v3.0.0 supports 'x8' (VertiRTA-X8-240).", err=True)
        raise typer.Exit(code=2)
    if fault not in FAULT_MODE_MAP:
        typer.echo(f"ERROR: Invalid fault {fault!r}; expected one of {sorted(FAULT_MODE_MAP)}.", err=True)
        raise typer.Exit(code=2)

    n_frames = max(1, int(round(duration * 100.0)))
    env = VertiRtaCoaxialX8Env(fault_mode=fault, substeps_400hz=4, max_steps=n_frames + 10)
    env.reset(seed=seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    zero_act = np.zeros((4,), dtype=np.float32)
    with out.open("wb") as fh:
        fh.write(make_vrta_header_bytes(*KARTAL_DATUM))
        for _ in range(n_frames):
            env.step(zero_act)
            fh.write(env.latest_vrta_frame_bytes())

    typer.echo(
        json.dumps(
            {
                "command": "sim",
                "version": __version__,
                "backend": env.backend,
                "airframe": "VertiRTA-X8-240",
                "fault": fault,
                "frames": n_frames,
                "out": str(out),
                "bytes": out.stat().st_size,
            }
        )
    )


@app.command("inspect")
def cmd_inspect(
    vrta_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to VRTA0001 .vrta file."),
    summary: bool = typer.Option(True, "--summary/--no-summary", help="Print JSON summary of .vrta log."),
) -> None:
    """Inspect a VRTA0001 binary flight log and verify Castagnoli CRC-32C checksums."""
    info = _parse_vrta_summary(vrta_file)
    if summary:
        typer.echo(json.dumps(info, indent=2))
    else:
        typer.echo(f"{info['magic']} frames={info['frames']} bad_crc={info['bad_crc_frames']}")
    if int(info["bad_crc_frames"]) > 0:  # type: ignore[arg-type]
        raise typer.Exit(code=1)


@app.command("replay")
def cmd_replay(
    vrta_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to VRTA0001 .vrta file."),
) -> None:
    """Replay and CRC-32C-verify a VRTA0001 Kartal corridor flight log."""
    info = _parse_vrta_summary(vrta_file)
    info["command"] = "replay"
    typer.echo(json.dumps(info, indent=2))
    if int(info["bad_crc_frames"]) > 0:  # type: ignore[arg-type]
        raise typer.Exit(code=1)


@app.command("bench")
def cmd_bench(
    num_envs: int = typer.Option(8, min=1, max=256, help="Number of parallel environments."),
    steps: int = typer.Option(100, min=1, max=5000, help="Number of 100 Hz vectorized steps."),
    seed: int = typer.Option(7, help="Deterministic seed."),
) -> None:
    """Benchmark VertiVecEnv parallel throughput (reporting cpp_nanobind vs python_fallback)."""
    vec = VertiVecEnv(num_envs=num_envs)
    vec.reset(seed=seed)
    act = np.zeros((num_envs, 4), dtype=np.float32)
    t0 = time.perf_counter()
    for _ in range(steps):
        vec.step(act)
    elapsed_s = max(1e-6, time.perf_counter() - t0)
    total_env_steps = num_envs * steps
    typer.echo(
        json.dumps(
            {
                "command": "bench",
                "backend": vec.backend,
                "num_envs": num_envs,
                "steps": steps,
                "elapsed_s": round(elapsed_s, 4),
                "env_steps_per_sec": round(total_env_steps / elapsed_s, 1),
            }
        )
    )


@app.command("atlas")
def cmd_atlas(
    seeds_per_regime: int = typer.Option(2, min=1, max=50, help="Quick seeds per fault regime."),
) -> None:
    """Run deterministic mini-atlas smoke across nominal/single-motor/same-arm-dual Kartal faults."""
    results: dict[str, dict[str, float]] = {}
    zero_act = np.zeros((4,), dtype=np.float32)
    for fault in ("none", "single_motor", "same_arm_dual"):
        cel_hits = 0
        min_v = 999.0
        for s in range(seeds_per_regime):
            env = VertiRtaCoaxialX8Env(fault_mode=fault, max_steps=60)
            env.reset(seed=100 + s)
            for _ in range(40):
                _, _, term, _, info = env.step(zero_act)
                min_v = min(min_v, float(info["v_term_v"]))
                if info["cel_latched"]:
                    cel_hits += 1
                    break
                if term:
                    break
        results[fault] = {"seeds": float(seeds_per_regime), "cel_latches": float(cel_hits), "min_v_term_v": round(min_v, 2)}
    typer.echo(json.dumps({"command": "atlas", "backend": get_backend_name(), "regimes": results}, indent=2))


@app.command("verify-stl")
def cmd_verify_stl(
    vrta_file: Optional[Path] = typer.Argument(
        None,
        help="Optional .vrta file for Kartal Coastal Vertiport / LTFJ-LTBX bounded STL check.",
    ),
    agl_min: float = typer.Option(10.0, help="Minimum corridor AGL bound [m] over Kartal."),
    agl_max: float = typer.Option(120.0, help="Maximum corridor AGL bound [m] over Kartal."),
) -> None:
    """Verify bounded Signal Temporal Logic (STL) robustness on Kartal Coastal Vertiport corridor."""
    if vrta_file is not None:
        info = _parse_vrta_summary(vrta_file)
        agl_lo, agl_hi = info["agl_range_m"]  # type: ignore[misc]
        ltfj_min = float(info["min_ltfj_margin_m"])  # type: ignore[arg-type]
        ltbx_min = float(info["min_ltbx_margin_m"])  # type: ignore[arg-type]
        rho_agl = min(float(agl_lo) - agl_min, agl_max - float(agl_hi))
        rho_ctr = min(ltfj_min, ltbx_min)
        rho_min = min(rho_agl, rho_ctr)
        out = {
            "command": "verify-stl",
            "corridor": "Maltepe-Dragos-Kartal",
            "rho_agl_m": round(rho_agl, 3),
            "rho_ctr_m": round(rho_ctr, 3),
            "rho_min_m": round(rho_min, 3),
            "satisfied": rho_min >= 0.0 and int(info["bad_crc_frames"]) == 0,  # type: ignore[arg-type]
        }
        typer.echo(json.dumps(out, indent=2))
        if not out["satisfied"]:
            raise typer.Exit(code=1)
        return

    typer.echo(
        json.dumps(
            {
                "command": "verify-stl",
                "corridor": "Maltepe-Dragos-Kartal",
                "datum": {"lat_deg": KARTAL_DATUM[0], "lon_deg": KARTAL_DATUM[1], "amsl_m": KARTAL_DATUM[2]},
                "bounds": {"agl_min_m": agl_min, "agl_max_m": agl_max, "ctr_exclusions": ["LTFJ", "LTBX"]},
                "status": "READY",
            },
            indent=2,
        )
    )


@app.command("verify")
def cmd_verify(
    vrta_file: Path = typer.Argument(..., exists=True, readable=True, help="VRTA0001 file to verify."),
) -> None:
    """Verify VRTA0001 CRC-32C, 400 V / 22 kWh telemetry bounds, and Kartal/LTFJ margins."""
    info = _parse_vrta_summary(vrta_file)
    v_lo, v_hi = info["v_term_range_v"]  # type: ignore[misc]
    hv_ok = (300.0 <= float(v_lo)) and (float(v_hi) <= 475.0)
    ctr_ok = float(info["min_ltfj_margin_m"]) >= 0.0 and float(info["min_ltbx_margin_m"]) >= 0.0  # type: ignore[arg-type]
    passed = int(info["bad_crc_frames"]) == 0 and hv_ok and ctr_ok  # type: ignore[arg-type]
    typer.echo(
        json.dumps(
            {
                "command": "verify",
                "passed": passed,
                "hv_400v_ok": hv_ok,
                "ctr_clear_ok": ctr_ok,
                "summary": info,
            },
            indent=2,
        )
    )
    if not passed:
        raise typer.Exit(code=1)


@app.command("gcs")
def cmd_gcs(
    port: int = typer.Option(5180, help="VertiGCS HTTP port (default 5180; forbidden: 4173/4174/8765/9222/9223)."),
    ws_port: int = typer.Option(8891, help="Live WebSocket telemetry port (default 8891)."),
    mode: str = typer.Option("sandbox", help="Launch mode: sandbox | live."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate ports and print launch URL without binding."),
) -> None:
    """Validate VertiGCS ports and print/prepare console launch configuration."""
    _validate_port(port, "HTTP")
    _validate_port(ws_port, "WebSocket")
    if mode not in ("sandbox", "live"):
        typer.echo(f"ERROR: Invalid mode {mode!r}; expected 'sandbox' or 'live'.", err=True)
        raise typer.Exit(code=2)
    url = (
        f"http://127.0.0.1:{port}/?mode=live&ws=ws://127.0.0.1:{ws_port}"
        if mode == "live"
        else f"http://127.0.0.1:{port}/"
    )
    typer.echo(
        json.dumps(
            {
                "command": "gcs",
                "mode": mode,
                "http_port": port,
                "ws_port": ws_port,
                "url": url,
                "dry_run": dry_run,
            }
        )
    )


@app.command("forensics-export")
def cmd_forensics_export(
    vrta_file: Path = typer.Argument(..., exists=True, readable=True, help="Input VRTA0001 .vrta flight log."),
    out_e5x: Path = typer.Option(Path("occurrence_draft.e5x.xml"), help="Output draft ECCAIRS2 .e5x XML path."),
    out_audit: Optional[Path] = typer.Option(None, help="Output forensics audit JSON path."),
    sidecar: Optional[Path] = typer.Option(None, help="Optional .vrta_events.jsonl sidecar path."),
    capacity: int = typer.Option(1, min=1, max=6, help="Occupant capacity (1=X8-240, 4..6=Lift+Cruise)."),
) -> None:
    """Export deterministic CIR (EU) 2026/1821 + CDR 2020/2034 draft/Incident ID 431 .e5x XML."""
    audit_path = out_audit if out_audit is not None else out_e5x.with_suffix(".audit.json")
    eng = VertiForensicsEngine()
    res = eng.export_e5x_from_vrta(
        vrta_path=vrta_file,
        out_e5x_xml_path=out_e5x,
        out_audit_json_path=audit_path,
        sidecar_path=sidecar or "",
        occupant_capacity=capacity,
    )
    payload = {
        "command": "forensics-export",
        "backend": eng.backend,
        "out_e5x": str(out_e5x),
        "out_audit": str(audit_path),
        **res,
    }
    typer.echo(json.dumps(payload, indent=2))
    if not res.get("ok"):
        raise typer.Exit(code=1)


@app.command("gnss-spoof-sim")
def cmd_gnss_spoof_sim(
    ramp_mps: float = typer.Option(1.5, min=0.2, max=10.0, help="Slow-ramp GNSS spoof velocity [m/s]."),
    water_fraction: float = typer.Option(0.05, min=0.0, max=1.0, help="Nadir water fraction [0..1]."),
    duration_s: float = typer.Option(22.0, min=16.0, max=120.0, help="Total 10 Hz simulation horizon [s]."),
) -> None:
    """Run deterministic H0..H3 MCUSUM GNSS spoof guard + 3-sigma dh/dt CTR inflation audit."""
    suite = VertiGnssDenied()
    suite.reset(2.0)
    dt_s = 0.1
    n_steps = int(round(duration_s / dt_s))
    warmup_steps = 160
    pos_n, pos_e = 200.0, -500.0
    spoof_e = 0.0
    ttd_s: float | None = None
    last_guard: dict[str, object] = {}

    for step_idx in range(n_steps):
        stamp_ns = (step_idx + 1) * 100_000_000
        pos_e += 12.0 * dt_s
        v_spoof = ramp_mps if step_idx >= warmup_steps else 0.0
        spoof_e += v_spoof * dt_s
        last_guard = suite.step_spoof_guard_10hz(
            stamp_ns=stamp_ns,
            gnss_pos_ned=(pos_n, pos_e + spoof_e, -45.0),
            gnss_vel_ned=(0.0, 12.0 + v_spoof, 0.0),
            vio_pos_ned=(pos_n, pos_e, -45.0),
            vio_vel_ned=(0.0, 12.0, 0.0),
            water_fraction=water_fraction,
        )
        if last_guard["latched"] and ttd_s is None and step_idx >= warmup_steps:
            ttd_s = round((step_idx - warmup_steps + 1) * dt_s, 3)

    ctr = suite.propagate_and_evaluate_ctr(10.0, 12.0, (pos_n, pos_e, -45.0), 57.0)
    typer.echo(
        json.dumps(
            {
                "command": "gnss-spoof-sim",
                "backend": suite.backend,
                "ramp_mps": ramp_mps,
                "water_fraction": water_fraction,
                "ttd_s": ttd_s,
                "guard": last_guard,
                "ctr_inflation": ctr,
            },
            indent=2,
        )
    )


@app.command("m3-demo")
def cmd_m3_demo(
    out_dir: Path = typer.Option(Path("m3_demo_out"), help="Directory for demo .vrta and .e5x outputs."),
) -> None:
    """Run end-to-end M3 demo: forward grazing wire, GNSS spoof guard, S3 MAYDAY voice & .e5x export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vrta_path = out_dir / "m3_demo.vrta"
    e5x_path = out_dir / "m3_demo.e5x.xml"
    audit_path = out_dir / "m3_demo.audit.json"

    # 1. Generate valid 25-frame VRTA0001 flight clip over Kartal
    env = VertiRtaCoaxialX8Env(fault_mode="same_arm_dual", substeps_400hz=4, max_steps=60)
    env.reset(seed=42, options={"inject_immediately": True})
    zero_act = np.zeros((4,), dtype=np.float32)
    with vrta_path.open("wb") as fh:
        fh.write(make_vrta_header_bytes(*KARTAL_DATUM))
        for _ in range(25):
            env.step(zero_act)
            fh.write(env.latest_vrta_frame_bytes())

    # 2. Forward hazard grazing wire fallback (<= 15 m/s)
    fh_eng = VertiForwardHazard()
    fh_eng.push_pose_200hz(1_000_000_000, (0.0, 0.0, -45.0), (12.0, 0.0, 0.0))
    fh_out = fh_eng.step(
        1_050_000_000,
        radar_healthy=False,
        wire_lines=[(200.0, 180.0, 1080.0, 185.0, 0.92)],
    )

    # 3. Voice Mondrian S3_Garbled MAYDAY preemption
    voice = VertiMondrianVoiceGate()
    voice_out = voice.evaluate(
        snr_db=-3.5,
        lang="TR",
        candidates=[
            ("DIVERT_SLZ:Kartal_Sahil", 0.45, True),
            ("HOLD_LOITER", 0.35, True),
        ],
        mayday_acoustic_weight=0.22,
    )

    # 4. Regulatory forensics .e5x export
    forensics = VertiForensicsEngine()
    e5x_out = forensics.export_e5x_from_vrta(vrta_path, e5x_path, audit_path)

    typer.echo(
        json.dumps(
            {
                "command": "m3-demo",
                "backend": forensics.backend,
                "forward_hazard": fh_out,
                "voice_mayday": voice_out,
                "forensics_e5x": e5x_out,
                "files": {"vrta": str(vrta_path), "e5x": str(e5x_path), "audit": str(audit_path)},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
