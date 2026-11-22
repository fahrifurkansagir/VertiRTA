"""Tiers 2–4 CI Verifier: Same-ISA 0-ULP SHA-256, Cross-ISA <1e-12, Work-Units & Steal Taint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any


STEAL_TAINT_THRESHOLD_PCT = 1.0
VRTA_MAGIC = b"VRTA0001"
VRTA_HEADER_FMT = "<8sII4d32s40s4sI"
VRTA_FRAME_FMT = "<2Q3d31f4BI80sI"
VRTA_HEADER_BYTES = 128
VRTA_FRAME_BYTES = 256


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_same_isa_sha256(
    run1_path: Path,
    run2_path: Path,
    golden_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify Same-ISA 0-ULP bit-identity across repeated .vrta runs and optional golden SHA."""
    sha1 = _sha256_file(run1_path)
    sha2 = _sha256_file(run2_path)
    matches_pair = sha1 == sha2
    matches_golden = True if golden_sha256 is None else (sha1 == golden_sha256)
    report: dict[str, Any] = {
        "schema_version": "vertirta.tier2_same_isa/v1.0.0",
        "sha256_run1": sha1,
        "sha256_run2": sha2,
        "golden_sha256": golden_sha256,
        "passed": matches_pair and matches_golden,
    }
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    return report


def compare_cross_isa_records(
    arm64_frames: list[dict[str, Any]],
    x86_frames: list[dict[str, Any]],
    max_abs_tol: float = 1e-12,
) -> dict[str, Any]:
    """Compare ARM64 vs x86_64 replay frames enforcing 0 discrete mode divergence & <1e-12 float tol."""
    if len(arm64_frames) != len(x86_frames):
        return {
            "schema_version": "vertirta.tier2_cross_isa/v1.0.0",
            "frame_count_arm64": len(arm64_frames),
            "frame_count_x86": len(x86_frames),
            "mode_divergences": abs(len(arm64_frames) - len(x86_frames)),
            "max_abs_diff": math.inf,
            "passed": False,
        }

    mode_divergences = 0
    max_abs_diff = 0.0
    discrete_keys = (
        "mode",
        "rta_mode",
        "flight_regime",
        "tmr_state",
        "active_motor_mask",
        "cel_latched",
        "flags",
        "health_mask",
    )

    for fa, fx in zip(arm64_frames, x86_frames):
        for dk in discrete_keys:
            if dk in fa or dk in fx:
                if fa.get(dk) != fx.get(dk):
                    mode_divergences += 1
        u_a = fa.get("u", [])
        u_x = fx.get("u", [])
        if len(u_a) != len(u_x):
            max_abs_diff = math.inf
        else:
            for va, vx in zip(u_a, u_x):
                diff = abs(float(va) - float(vx))
                if not math.isfinite(diff) or diff > max_abs_diff:
                    max_abs_diff = diff

    passed = (mode_divergences == 0) and (max_abs_diff < max_abs_tol)
    report: dict[str, Any] = {
        "schema_version": "vertirta.tier2_cross_isa/v1.0.0",
        "frames_compared": len(arm64_frames),
        "mode_divergences": mode_divergences,
        "max_abs_diff": max_abs_diff,
        "max_abs_tol": max_abs_tol,
        "passed": passed,
    }
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    return report


def parse_work_units_log(
    mode: str,
    log_text: str,
    ticks: int,
    max_ir_per_tick: int = 1_200_000,
) -> dict[str, Any]:
    """Parse Tier-3 instruction count from `perf stat` or `valgrind --tool=callgrind` fallback."""
    if ticks < 1:
        raise ValueError(f"ticks must be >= 1, got {ticks}")

    total_ir: int | None = None
    if mode == "perf_stat":
        m = re.search(r"([\d,]+)\s+instructions(?::u)?\b", log_text)
        if m:
            total_ir = int(m.group(1).replace(",", ""))
    elif mode == "valgrind_callgrind_fallback":
        m = re.search(r"([\d,]+)\s+(?:\([\d.]+%\)\s+)?PROGRAM TOTALS", log_text)
        if not m:
            m = re.search(r"(?:Collected\s*:\s*|I\s+refs:\s*)([\d,]+)", log_text)
        if m:
            total_ir = int(m.group(1).replace(",", ""))
    else:
        raise ValueError(f"Unsupported PMU mode '{mode}'")

    if total_ir is None:
        raise ValueError(f"Could not extract instruction count from {mode} log")

    ir_per_tick = total_ir // ticks
    report: dict[str, Any] = {
        "schema_version": "vertirta.tier3_work_units/v1.0.0",
        "pmu_mode": mode,
        "ticks": ticks,
        "total_ir": total_ir,
        "ir_per_tick": ir_per_tick,
        "max_ir_per_tick": max_ir_per_tick,
        "passed": ir_per_tick <= max_ir_per_tick,
    }
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    return report


def classify_steal_taint(steal_pct: float) -> str:
    """Flag cloud VM runs exceeding 1.0% hypervisor steal as NOISY_NEIGHBOR_TAINTED."""
    return "NOISY_NEIGHBOR_TAINTED" if steal_pct > STEAL_TAINT_THRESHOLD_PCT else "CLEAN"


def _read_proc_stat_cpu() -> tuple[int, int]:
    """Return (steal_jiffies, total_jiffies) from Linux /proc/stat (or (0, 1) on non-Linux)."""
    proc_stat = Path("/proc/stat")
    if not proc_stat.exists():
        return (0, 1)
    first_line = proc_stat.read_text(encoding="utf-8").splitlines()[0]
    cols = [int(x) for x in first_line.split()[1:]]
    steal = cols[7] if len(cols) > 7 else 0
    return (steal, max(sum(cols), 1))


def measure_monotonic_with_steal(
    bench_bin: Path,
    ticks: int,
    out_json: Path,
) -> dict[str, Any]:
    """Execute Tier-4 CLOCK_MONOTONIC_RAW tail benchmark while recording hypervisor steal %."""
    raw_bench_out = out_json.with_suffix(".raw.json")
    s0, t0 = _read_proc_stat_cpu()
    subprocess.run(
        [str(bench_bin), "--ticks", str(ticks), "--clock", "monotonic_raw", "--out", str(raw_bench_out)],
        check=True,
    )
    s1, t1 = _read_proc_stat_cpu()
    dt = max(t1 - t0, 1)
    steal_pct = 100.0 * max(s1 - s0, 0) / dt

    metrics = json.loads(raw_bench_out.read_text(encoding="utf-8")) if raw_bench_out.exists() else {}
    report: dict[str, Any] = {
        "schema_version": "vertirta.tier4_wallclock/v1.0.0",
        "ticks": ticks,
        "steal_pct": round(steal_pct, 4),
        "taint": classify_steal_taint(steal_pct),
        "metrics": metrics,
        "passed": True,
    }
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def load_vrta_frames(path: Path) -> list[dict[str, Any]]:
    """Load frames from canonical .vrta binary (128 B VRTA0001 header + 256 B POD frames) or JSON."""
    raw = path.read_bytes()
    if raw.startswith(b"[") or raw.startswith(b"{"):
        parsed = json.loads(raw.decode("utf-8"))
        return parsed if isinstance(parsed, list) else parsed.get("frames", [])

    if len(raw) < VRTA_HEADER_BYTES or (len(raw) - VRTA_HEADER_BYTES) % VRTA_FRAME_BYTES != 0:
        raise ValueError(f"Invalid .vrta binary size {len(raw)} in {path}")

    import struct

    hdr = struct.unpack(VRTA_HEADER_FMT, raw[:VRTA_HEADER_BYTES])
    if hdr[0] != VRTA_MAGIC:
        raise ValueError(f"Invalid .vrta header magic {hdr[0]!r} in {path}; expected {VRTA_MAGIC!r}")

    frames: list[dict[str, Any]] = []
    offset = VRTA_HEADER_BYTES
    while offset + VRTA_FRAME_BYTES <= len(raw):
        fields = struct.unpack(VRTA_FRAME_FMT, raw[offset : offset + VRTA_FRAME_BYTES])
        seq, t_sim_us = fields[0], fields[1]
        pos_enu = list(fields[2:5])          # 3x float64 (bytes 16..39)
        floats_31 = list(fields[5:36])       # 31x float32 (bytes 40..163)
        active_motor_mask = fields[36]       # byte 164
        flight_regime = fields[37]           # byte 165
        tmr_state = fields[38]               # byte 166
        cel_latched = fields[39]             # byte 167
        flags = fields[40]                   # uint32 (bytes 168..171)
        frames.append(
            {
                "seq": seq,
                "t_sim_us": t_sim_us,
                "active_motor_mask": active_motor_mask,
                "flight_regime": flight_regime,
                "mode": flight_regime,
                "tmr_state": tmr_state,
                "cel_latched": cel_latched,
                "flags": flags,
                "u": pos_enu + floats_31,
            }
        )
        offset += VRTA_FRAME_BYTES
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="VertiRTA Tiers 2-4 Determinism Verifier.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_same = sub.add_parser("same-isa")
    p_same.add_argument("--run1", type=Path, required=True)
    p_same.add_argument("--run2", type=Path, required=True)
    p_same.add_argument("--golden-sha256", default=None)
    p_same.add_argument("--out", type=Path, required=True)

    p_cross = sub.add_parser("cross-isa")
    p_cross.add_argument("--arm64", type=Path, required=True)
    p_cross.add_argument("--x86", type=Path, required=True)
    p_cross.add_argument("--max-abs-tol", type=float, default=1e-12)
    p_cross.add_argument("--out", type=Path, required=True)

    p_work = sub.add_parser("work-units")
    p_work.add_argument("--mode", required=True)
    p_work.add_argument("--log", type=Path, required=True)
    p_work.add_argument("--ticks", type=int, required=True)
    p_work.add_argument("--max-ir-per-tick", type=int, default=1_200_000)
    p_work.add_argument("--out", type=Path, required=True)

    p_steal = sub.add_parser("monotonic-steal")
    p_steal.add_argument("--bin", dest="bench_bin", type=Path, required=True)
    p_steal.add_argument("--ticks", type=int, default=20000)
    p_steal.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.cmd == "same-isa":
        report = verify_same_isa_sha256(args.run1, args.run2, args.golden_sha256)
    elif args.cmd == "cross-isa":
        report = compare_cross_isa_records(
            load_vrta_frames(args.arm64),
            load_vrta_frames(args.x86),
            max_abs_tol=args.max_abs_tol,
        )
    elif args.cmd == "work-units":
        report = parse_work_units_log(
            args.mode,
            args.log.read_text(encoding="utf-8", errors="replace"),
            args.ticks,
            args.max_ir_per_tick,
        )
    else:
        report = measure_monotonic_with_steal(args.bench_bin, args.ticks, args.out)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
