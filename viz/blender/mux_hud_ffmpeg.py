"""Synced RGBA Telemetry HUD Generator & ``hevc_nvenc -preset p5`` 4K30 Cinema Muxer.

Enforces Plan 1B Task 8 / Global Constraints locks:
- Hardware NVENC encoding strictly uses ``-c:v hevc_nvenc -preset p5`` (hard-banning
  deprecated/legacy ``-preset slow``!), ``-rc vbr -cq 18 -b:v 0 -tag:v hvc1
  -movflags +faststart``.
- Composites frame-exact RGBA telemetry HUD frames (``V``, ``AGL``, ``SoC``,
  ``V_term``, ``TMR``, ``||u_safe - u_nom||``, 8-rotor mask, ``LTFJ``/``LTBX``
  margins) over rendered Blender Cycles PNGs via ``overlay=0:0:shortest=1``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from viz.cesium.vrta_reader import VrtaFrame100Hz

ALLOWED_NVENC_PRESETS: frozenset[str] = frozenset({"p1", "p2", "p3", "p4", "p5", "p6", "p7"})

_TMR_LABELS: dict[int, str] = {
    0: "3/3 NOMINAL",
    1: "2/2 DUPLEX",
    2: "1/3 SIMPLEX",
    3: "0/3 FAILSAFE",
}


def format_hud_telemetry_payload(
    frame: VrtaFrame100Hz, frame_idx: int, total_frames: int
) -> dict[str, Any]:
    """Extract frame-synchronized HUD fields from a ``VrtaFrame100Hz`` record."""
    mask = int(frame.active_motor_mask) & 0xFF
    failed_motors = [i for i in range(8) if not (mask & (1 << i))]
    airspeed_mps = math.hypot(*frame.vel_enu)
    cbf_delta_u = math.hypot(
        frame.a_safe_enu[0] - frame.a_nom_enu[0],
        frame.a_safe_enu[1] - frame.a_nom_enu[1],
        frame.a_safe_enu[2] - frame.a_nom_enu[2],
    )
    return {
        "frame_idx": int(frame_idx),
        "total_frames": int(total_frames),
        "sim_time_s": frame.t_sim_us * 1e-6,
        "airspeed_mps": airspeed_mps,
        "agl_m": float(frame.agl_m),
        "soc_pct": float(frame.soc_pct),
        "v_term_v": float(frame.v_term_v),
        "tmr_status": _TMR_LABELS.get(frame.tmr_state, f"TMR_{frame.tmr_state}"),
        "cbf_delta_u_norm": cbf_delta_u,
        "active_motor_mask": mask,
        "failed_motors": failed_motors,
        "cel_latched": bool(frame.cel_latched),
        "ltfj_margin_m": float(frame.ltfj_margin_m),
        "ltbx_margin_m": float(frame.ltbx_margin_m),
        "attribution": "Copernicus DEM/Sentinel-2 | © OpenStreetMap contributors | DHMI eAIP (SIM ONLY)",
    }


def render_hud_frame_rgba(
    payload: dict[str, Any],
    out_png: str | Path,
    *,
    width: int = 3840,
    height: int = 2160,
) -> Path:
    """Render one transparent RGBA telemetry HUD PNG using Pillow (when installed)."""
    from PIL import Image, ImageDraw  # type: ignore[import-not-found]

    target = Path(out_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGBA", (int(width), int(height)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    panel_w = int(width * 0.24)
    panel_h = int(height * 0.26)
    draw.rectangle([(36, 36), (36 + panel_w, 36 + panel_h)], fill=(11, 16, 23, 200), outline=(18, 181, 203, 220), width=3)

    lines = [
        f"VERTIRTA-X8-240  T={payload['sim_time_s']:05.2f}s  F={payload['frame_idx']}/{payload['total_frames']}",
        f"V={payload['airspeed_mps']:.1f} m/s   AGL={payload['agl_m']:.2f} m",
        f"SoC={payload['soc_pct']:.1f}%   V_term={payload['v_term_v']:.1f} V",
        f"TMR={payload['tmr_status']}   ||u_safe-u_nom||={payload['cbf_delta_u_norm']:.3f}",
        f"LTFJ={payload['ltfj_margin_m']:.0f} m   LTBX={payload['ltbx_margin_m']:.0f} m",
        f"MOTORS=0x{payload['active_motor_mask']:02X}  FAILED={payload['failed_motors']}",
    ]
    y = 56
    step = max(24, int(height * 0.032))
    for line in lines:
        draw.text((56, y), line, fill=(230, 238, 248, 255))
        y += step

    draw.text((36, height - 48), payload["attribution"], fill=(210, 220, 235, 220))
    img.save(target, format="PNG")
    return target


def build_ffmpeg_nvenc_cmd(
    frames_pattern: str | Path,
    hud_pattern: str | Path,
    out_mp4: str | Path,
    *,
    fps: int = 30,
    preset: str = "p5",
    cq: int = 18,
    maxrate: str = "65M",
    bufsize: str = "130M",
) -> list[str]:
    """Build hardware ``hevc_nvenc -preset p5`` 4K30 overlay mux command.

    Hard-rejects legacy ``-preset slow`` (replaced by NVENC ``p1..p7`` presets).
    """
    preset_norm = str(preset).strip().lower()
    if preset_norm == "slow" or preset_norm not in ALLOWED_NVENC_PRESETS:
        raise ValueError(
            f"Forbidden/invalid hevc_nvenc preset {preset!r} (legacy 'slow' is banned; "
            "use NVENC 'p5' preset from p1..p7)"
        )
    return [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_pattern),
        "-framerate",
        str(int(fps)),
        "-i",
        str(hud_pattern),
        "-filter_complex",
        "[0:v][1:v]overlay=0:0:shortest=1,format=yuv420p",
        "-c:v",
        "hevc_nvenc",
        "-preset",
        preset_norm,
        "-rc",
        "vbr",
        "-cq",
        str(int(cq)),
        "-b:v",
        "0",
        "-maxrate",
        str(maxrate),
        "-bufsize",
        str(bufsize),
        "-tag:v",
        "hvc1",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VertiRTA NVENC p5 HUD muxer")
    parser.add_argument("--frames", default="/content/renders_4k/frame_%04d.png")
    parser.add_argument("--hud", default="/content/hud_4k/hud_%04d.png")
    parser.add_argument("--out", default="/content/out/VertiRTA_X8_240_Istanbul_4K30_FlightReplay.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--preset", default="p5")
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args(argv)

    cmd = build_ffmpeg_nvenc_cmd(args.frames, args.hud, args.out, fps=args.fps, preset=args.preset)
    if args.print_json:
        print(json.dumps({"cmd": cmd}, indent=2))
    else:
        print(" ".join(cmd))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
