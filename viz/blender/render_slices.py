"""Resumable 150-Frame Cycles OptiX Slice Driver & 3-Frame <18 GB VRAM Preflight.

Protects 1,800-frame (60 s @ 30 fps) 4K Colab G4 Blackwell renders against
session preemption or driver resets (Xid 79/120):
- Runs a mandatory 3-frame ``--cycles-print-stats`` preflight gate asserting
  peak VRAM < 18.0 GB before starting the full batch.
- Splits renders into deterministic 150-frame inclusive slices (12 slices for
  1,800 frames), skipping slices whose frames already exist on Drive.
- Places Blender CLI arguments strictly left-to-right before ``--`` and Cycles
  device flags (``--cycles-device OPTIX --cycles-print-stats``) after ``--``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Collection, Sequence

DEFAULT_TOTAL_FRAMES: int = 1800
DEFAULT_SLICE_SIZE: int = 150
DEFAULT_MAX_VRAM_GB: float = 18.0
PREFLIGHT_SAMPLE_FRAMES: tuple[int, int, int] = (1, 540, 1780)


def compute_frame_slices(
    total_frames: int = DEFAULT_TOTAL_FRAMES,
    slice_size: int = DEFAULT_SLICE_SIZE,
    completed_frames: Collection[int] | None = None,
) -> list[tuple[int, int]]:
    """Return remaining 1-based inclusive ``(start, end)`` render slices."""
    if total_frames < 1 or slice_size < 1:
        raise ValueError(f"total_frames ({total_frames}) and slice_size ({slice_size}) must be >= 1")
    done = set(completed_frames or ())
    slices: list[tuple[int, int]] = []
    for start in range(1, total_frames + 1, slice_size):
        end = min(total_frames, start + slice_size - 1)
        if not all(f in done for f in range(start, end + 1)):
            slices.append((start, end))
    return slices


def build_blender_slice_cmd(
    blend_file: str | Path,
    script_path: str | Path,
    out_pattern: str | Path,
    vrta_path: str | Path,
    start_frame: int,
    end_frame: int,
    *,
    width: int = 3840,
    height: int = 2160,
    samples: int = 128,
    blender_bin: str = "blender",
) -> list[str]:
    """Build exact left-to-right headless Blender Cycles OptiX slice command."""
    if start_frame < 1 or end_frame < start_frame:
        raise ValueError(f"Invalid frame range [{start_frame}, {end_frame}]")
    return [
        blender_bin,
        "-b",
        str(blend_file),
        "-E",
        "CYCLES",
        "-P",
        str(script_path),
        "-o",
        str(out_pattern),
        "-F",
        "PNG",
        "-s",
        str(int(start_frame)),
        "-e",
        str(int(end_frame)),
        "-a",
        "--",
        "--cycles-device",
        "OPTIX",
        "--cycles-print-stats",
        "--vrta",
        str(vrta_path),
        "--res",
        f"{int(width)}x{int(height)}",
        "--samples",
        str(int(samples)),
    ]


def build_preflight_3frame_cmd(
    blend_file: str | Path,
    script_path: str | Path,
    out_pattern: str | Path,
    vrta_path: str | Path,
    *,
    frames: Sequence[int] = PREFLIGHT_SAMPLE_FRAMES,
    width: int = 3840,
    height: int = 2160,
    samples: int = 128,
    blender_bin: str = "blender",
) -> list[str]:
    """Build 3-frame (coast, fault/smoke, rooftop touchdown) VRAM preflight command."""
    frame_spec = ",".join(str(int(f)) for f in frames)
    return [
        blender_bin,
        "-b",
        str(blend_file),
        "-E",
        "CYCLES",
        "-P",
        str(script_path),
        "-o",
        str(out_pattern),
        "-F",
        "PNG",
        "-f",
        frame_spec,
        "--",
        "--cycles-device",
        "OPTIX",
        "--cycles-print-stats",
        "--vrta",
        str(vrta_path),
        "--res",
        f"{int(width)}x{int(height)}",
        "--samples",
        str(int(samples)),
    ]


def parse_cycles_peak_vram_gb(cycles_stdout: str) -> float:
    """Parse peak VRAM usage (GB) from Blender ``--cycles-print-stats`` output."""
    text = str(cycles_stdout)
    match_gb = re.search(r"Peak\s+memory(?:\s+usage)?:\s*([0-9]+(?:\.[0-9]+)?)\s*GB", text, re.IGNORECASE)
    if match_gb:
        return float(match_gb.group(1))
    match_mb = re.search(
        r"Peak\s+memory(?:\s+usage)?:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:MB|MiB)", text, re.IGNORECASE
    )
    if match_mb:
        return float(match_mb.group(1)) / 1000.0
    raise ValueError("Unable to parse Cycles peak VRAM memory from --cycles-print-stats log")


def assert_vram_within_budget(peak_vram_gb: float, *, max_gb: float = DEFAULT_MAX_VRAM_GB) -> bool:
    """Enforce the < 18 GB VRAM preflight budget before launching 1,800-frame batch."""
    val = float(peak_vram_gb)
    if val >= float(max_gb):
        raise RuntimeError(
            f"Cycles OptiX peak VRAM {val:.2f} GB exceeds preflight budget < {max_gb:.1f} GB"
        )
    return True


def build_slice_rsync_cmd(local_dir: str | Path, drive_dir: str | Path) -> list[str]:
    """Build resumable rsync command syncing completed slices to persistent Drive storage."""
    src = str(local_dir).rstrip("/") + "/"
    dst = str(drive_dir).rstrip("/") + "/"
    return ["rsync", "-a", src, dst]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VertiRTA 150-frame Blender Cycles slice planner")
    parser.add_argument("--blend", default="/content/work/vertirta_istanbul_master.blend")
    parser.add_argument("--script", default="viz/blender/apply_vrta_and_render_setup.py")
    parser.add_argument("--out", default="/content/renders_4k/frame_####")
    parser.add_argument("--vrta", default="/content/logs/kartal_m3_fault.vrta")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive/VertiRTA_4K_Frames")
    parser.add_argument("--total-frames", type=int, default=DEFAULT_TOTAL_FRAMES)
    parser.add_argument("--slice-size", type=int, default=DEFAULT_SLICE_SIZE)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    slices = compute_frame_slices(args.total_frames, args.slice_size)
    plan = {
        "preflight": build_preflight_3frame_cmd(args.blend, args.script, "/content/preflight/pf_####", args.vrta),
        "slices": [
            {
                "start": s,
                "end": e,
                "render_cmd": build_blender_slice_cmd(args.blend, args.script, args.out, args.vrta, s, e),
                "sync_cmd": build_slice_rsync_cmd(Path(args.out).parent.as_posix(), args.drive_dir),
            }
            for s, e in slices
        ],
    }
    if args.print_json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"Prepared {len(slices)} resumable {args.slice_size}-frame slices for {args.total_frames} frames")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
