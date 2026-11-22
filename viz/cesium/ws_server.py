"""VertiRTA CesiumJS Live Tactical Earth Console HTTP + WebSocket Server (:5180).

Streams ``vrta.ws.v1.2`` JSON telemetry packets at 30 Hz pose / 5 Hz CBVF ring
cadence over ``ws://127.0.0.1:5180/telemetry`` and serves the static CesiumJS
console (``index.html``, ``app.js``, ``/api/airspace.geojson``) on the same port.

Mandatory locks (Plan 1B Task 7 / DR-5):
- Dedicated port ``5180`` (hard-bans ``4173, 4174, 8765, 9222, 9223``).
- Clean ``footprint_lat_lon_hae`` ``[lat, lon, hae]`` ordering (NEVER swapped
  ``[lon, lat]`` or legacy ``footprint_llh_hae``).
- Gaussian touchdown height blend ``h_wgs84 + delta_z_surf * exp(-(agl / 15.0)^2)``.
- Pure stdlib ``asyncio`` + RFC 6455 framing so tests and local replay run
  without requiring external WebSocket server libraries.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Sequence

from viz.cesium.vrta_reader import VrtaFileHeader, VrtaFrame100Hz, read_vrta_file
from viz.geo.airspace_polygons import build_ltfj_ltbx_geometries
from viz.geo.vertirta_geodesy import VertiRTAGeodesyBridge, assert_valid_istanbul_geoid

DEFAULT_CESIUM_HOST: str = "127.0.0.1"
DEFAULT_CESIUM_PORT: int = 5180
FORBIDDEN_PORTS: frozenset[int] = frozenset({4173, 4174, 8765, 9222, 9223})
WS_SCHEMA_VER: str = "vrta.ws.v1.2"
TOUCHDOWN_BLEND_SIGMA_M: float = 15.0

_TMR_STATE_LABELS: dict[int, str] = {
    0: "3/3_NOMINAL",
    1: "2/2_DUPLEX",
    2: "1/3_SIMPLEX_ANALYTICAL",
    3: "0/3_FAILSAFE_DIVERT",
}

_FLIGHT_REGIME_LABELS: dict[int, str] = {
    0: "NOMINAL_8_ROTOR",
    1: "DEGRADED_7_ROTOR",
    2: "DEGRADED_6_ROTOR_CROSS",
    3: "CEL_SAME_ARM_ACTIVE_SPIN",
    4: "CEL_CRITICAL_DESCENT",
}

_WS_MAGIC_GUID: str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def validate_port(port: int) -> int:
    """Enforce workspace port isolation lock (rejects 4173/4174/8765/9222/9223)."""
    p = int(port)
    if p in FORBIDDEN_PORTS:
        raise ValueError(
            f"Port {p} is forbidden by AEGIS-VTOL port isolation lock "
            f"(forbidden={sorted(FORBIDDEN_PORTS)}; use {DEFAULT_CESIUM_PORT})"
        )
    if not (1024 <= p <= 65535):
        raise ValueError(f"Port {p} out of valid unprivileged range [1024, 65535]")
    return p


def compute_touchdown_blend_hae(
    h_wgs84_m: float,
    delta_z_surf_m: float,
    agl_m: float,
    sigma_m: float = TOUCHDOWN_BLEND_SIGMA_M,
) -> float:
    """Gaussian surface blend ensuring 0.00 m AGL contact without distorting cruise HAE.

    Formula: ``h_render = h_wgs84 + delta_z_surf * exp(-(max(0, agl) / 15.0)^2)``.
    At ``AGL = 0.0 m``, weight = 1.0 (zero visual gap on Cesium terrain/OSM roof).
    At ``AGL >= 60.0 m``, weight < 1.2e-7 (absolute geodetic accuracy vs LTFJ/LTBX).
    """
    if sigma_m <= 0.0 or not math.isfinite(sigma_m):
        raise ValueError(f"sigma_m must be finite and positive, got {sigma_m!r}")
    agl_clamped = max(0.0, float(agl_m))
    weight = math.exp(-((agl_clamped / float(sigma_m)) ** 2))
    return float(h_wgs84_m) + float(delta_z_surf_m) * weight


def _validate_lat_lon_hae_ring(
    ring: Sequence[Sequence[float]],
) -> list[list[float]]:
    validated: list[list[float]] = []
    for idx, pt in enumerate(ring):
        if len(pt) != 3:
            raise ValueError(f"cbvf_ring_lat_lon_hae[{idx}] must have 3 elements [lat, lon, hae]")
        lat, lon, hae = float(pt[0]), float(pt[1]), float(pt[2])
        if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(hae)):
            raise ValueError(f"Non-finite CBVF vertex at index {idx}: {pt!r}")
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"Out-of-range lat/lon in CBVF vertex {idx}: ({lat}, {lon})")
        validated.append([lat, lon, hae])
    return validated


def encode_ws_telemetry_frame(
    *,
    seq: int,
    sim_time_s: float,
    lat_deg: float,
    lon_deg: float,
    alt_amsl_m: float,
    geoid_n_m: float,
    agl_m: float,
    q_enu_flu: Sequence[float],
    active_motor_mask: int,
    cbvf_ring_lat_lon_hae: Sequence[Sequence[float]],
    vel_enu_mps: Sequence[float] = (0.0, 0.0, 0.0),
    rotor_rpm: Sequence[float] | None = None,
    rotor_thrust_n: Sequence[float] | None = None,
    soc_pct: float = 100.0,
    v_term_v: float = 400.0,
    tmr_status: str = "3/3_NOMINAL",
    flight_regime: str = "NOMINAL_8_ROTOR",
    cel_latched: bool = False,
    cbf_delta_u_norm: float = 0.0,
    active_slz_id: str = "SLZ_KARTAL_VERTIPORT",
    dist_ltfj_ctr_m: float = 1042.0,
    dist_ltbx_ctr_m: float = 2995.0,
    cbvf_update_seq: int | None = None,
    playback_mode: str = "LIVE_REPLAY",
) -> dict[str, Any]:
    """Encode a canonical ``vrta.ws.v1.2`` JSON telemetry packet.

    Uses strictly ``footprint_lat_lon_hae`` (``[lat, lon, hae]`` order) and
    validates Istanbul EGM2008 geoid undulation to guard against 0.0 m fallback.
    """
    n_valid = assert_valid_istanbul_geoid(geoid_n_m)
    alt_hae = float(alt_amsl_m) + n_valid
    mask = int(active_motor_mask) & 0xFF
    failed_indices = [i for i in range(8) if not (mask & (1 << i))]

    if len(q_enu_flu) != 4:
        raise ValueError(f"q_enu_flu must have 4 elements [w, x, y, z], got {len(q_enu_flu)}")
    if len(vel_enu_mps) != 3:
        raise ValueError(f"vel_enu_mps must have 3 elements, got {len(vel_enu_mps)}")

    rpms = (
        [float(v) for v in rotor_rpm]
        if rotor_rpm is not None
        else [2450.0 if (mask & (1 << i)) else 0.0 for i in range(8)]
    )
    thrusts = (
        [float(v) for v in rotor_thrust_n]
        if rotor_thrust_n is not None
        else [295.0 if (mask & (1 << i)) else 0.0 for i in range(8)]
    )
    if len(rpms) != 8 or len(thrusts) != 8:
        raise ValueError("rotor_rpm and rotor_thrust_n must have length 8")

    ring_clean = _validate_lat_lon_hae_ring(cbvf_ring_lat_lon_hae)
    airspeed = math.hypot(float(vel_enu_mps[0]), float(vel_enu_mps[1]), float(vel_enu_mps[2]))
    cbvf_status = (
        "CEL_EMERGENCY_DESCENT"
        if cel_latched
        else ("NOMINAL_8_ROTOR" if not failed_indices else f"DEGRADED_{8 - len(failed_indices)}_ROTOR")
    )

    return {
        "schema_ver": WS_SCHEMA_VER,
        "seq": int(seq),
        "sim_time_s": float(sim_time_s),
        "playback_mode": str(playback_mode),
        "geodetic": {
            "lat_deg": float(lat_deg),
            "lon_deg": float(lon_deg),
            "alt_amsl_m": float(alt_amsl_m),
            "geoid_n_egm2008_m": n_valid,
            "alt_hae_wgs84_m": alt_hae,
            "agl_m": float(agl_m),
            "q_enu_flu": [float(c) for c in q_enu_flu],
            "vel_enu_mps": [float(v) for v in vel_enu_mps],
        },
        "propulsion": {
            "active_motor_mask": mask,
            "failed_motor_indices": failed_indices,
            "rotor_rpm": rpms,
            "rotor_thrust_n": thrusts,
            "smoke_active": bool(failed_indices),
        },
        "safety_avionics": {
            "airspeed_mps": airspeed,
            "soc_pct": float(soc_pct),
            "v_term_v": float(v_term_v),
            "tmr_status": str(tmr_status),
            "flight_regime": str(flight_regime),
            "cel_latched": bool(cel_latched),
            "cbf_delta_u_norm": float(cbf_delta_u_norm),
            "active_slz_id": str(active_slz_id),
            "dist_ltfj_ctr_m": float(dist_ltfj_ctr_m),
            "dist_ltbx_ctr_m": float(dist_ltbx_ctr_m),
        },
        "cbvf_overlay": {
            "update_seq": int(cbvf_update_seq if cbvf_update_seq is not None else seq),
            "status": cbvf_status,
            "cone_apex_hae_m": alt_hae,
            "footprint_lat_lon_hae": ring_clean,
        },
    }


def vrta_frame_to_ws_packet(
    frame: VrtaFrame100Hz,
    header: VrtaFileHeader,
    *,
    bridge: VertiRTAGeodesyBridge | None = None,
    cbvf_ring_lat_lon_hae: Sequence[Sequence[float]] | None = None,
    cbvf_segments: int = 24,
) -> dict[str, Any]:
    """Convert a binary ``VrtaFrame100Hz`` into a ``vrta.ws.v1.2`` WebSocket packet."""
    active_bridge = bridge or VertiRTAGeodesyBridge(
        anchor_lat_deg=header.datum_lat_deg,
        anchor_lon_deg=header.datum_lon_deg,
        anchor_h_amsl_m=header.datum_amsl_m,
        fixed_geoid_n_m=header.egm2008_n_m,
    )
    geo = active_bridge.flat_enu_to_cesium_wgs84(*frame.pos_enu)
    q_enu = active_bridge.px4_quat_to_enu_flu(*frame.q_frd_ned)

    delta_u = math.hypot(
        frame.a_safe_enu[0] - frame.a_nom_enu[0],
        frame.a_safe_enu[1] - frame.a_nom_enu[1],
        frame.a_safe_enu[2] - frame.a_nom_enu[2],
    )

    if cbvf_ring_lat_lon_hae is None:
        # Fault-scaled 24-vertex reachability footprint in strict [lat, lon, hae] order
        n_active = bin(frame.active_motor_mask & 0xFF).count("1")
        reach_scale = 0.35 if frame.cel_latched else (0.65 if n_active < 8 else 1.0)
        radius_m = max(8.0, float(frame.agl_m) * 1.45 * reach_scale)
        ground_z_flat = frame.pos_enu[2] - max(0.0, float(frame.agl_m))
        ring_pts: list[list[float]] = []
        for k in range(cbvf_segments):
            theta = (2.0 * math.pi * k) / cbvf_segments
            pt_geo = active_bridge.flat_enu_to_cesium_wgs84(
                frame.pos_enu[0] + radius_m * math.cos(theta),
                frame.pos_enu[1] + radius_m * math.sin(theta),
                ground_z_flat,
            )
            ring_pts.append([pt_geo["lat_deg"], pt_geo["lon_deg"], pt_geo["alt_hae_m"]])
        cbvf_ring = ring_pts
    else:
        cbvf_ring = [list(pt) for pt in cbvf_ring_lat_lon_hae]

    return encode_ws_telemetry_frame(
        seq=frame.seq,
        sim_time_s=frame.t_sim_us * 1e-6,
        lat_deg=geo["lat_deg"],
        lon_deg=geo["lon_deg"],
        alt_amsl_m=geo["alt_amsl_m"],
        geoid_n_m=geo["geoid_n_m"],
        agl_m=frame.agl_m,
        q_enu_flu=q_enu,
        active_motor_mask=frame.active_motor_mask,
        cbvf_ring_lat_lon_hae=cbvf_ring,
        vel_enu_mps=frame.vel_enu,
        rotor_rpm=frame.rotor_rpm,
        soc_pct=frame.soc_pct,
        v_term_v=frame.v_term_v,
        tmr_status=_TMR_STATE_LABELS.get(frame.tmr_state, f"STATE_{frame.tmr_state}"),
        flight_regime=_FLIGHT_REGIME_LABELS.get(frame.flight_regime, f"REGIME_{frame.flight_regime}"),
        cel_latched=bool(frame.cel_latched),
        cbf_delta_u_norm=delta_u,
        dist_ltfj_ctr_m=frame.ltfj_margin_m,
        dist_ltbx_ctr_m=frame.ltbx_margin_m,
        cbvf_update_seq=frame.seq // 20,
    )


def _encode_ws_text_frame(payload_text: str) -> bytes:
    """Encode an unmasked server-to-client RFC 6455 text frame."""
    data = payload_text.encode("utf-8")
    n = len(data)
    if n < 126:
        header = struct.pack("!BB", 0x81, n)
    elif n < 65536:
        header = struct.pack("!BBH", 0x81, 126, n)
    else:
        header = struct.pack("!BBQ", 0x81, 127, n)
    return header + data


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    *,
    static_dir: Path,
    vrta_path: Path | None,
    pose_hz: float,
) -> None:
    try:
        request_line = await reader.readline()
        if not request_line:
            return
        parts = request_line.decode("latin-1").strip().split()
        if len(parts) < 2:
            return
        path = parts[1].split("?", 1)[0]

        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if not line or line in (b"\r\n", b"\n"):
                break
            if b":" in line:
                k, v = line.decode("latin-1").split(":", 1)
                headers[k.strip().lower()] = v.strip()

        if path == "/telemetry" and headers.get("upgrade", "").lower() == "websocket":
            ws_key = headers.get("sec-websocket-key", "")
            accept = base64.b64encode(
                hashlib.sha1((ws_key + _WS_MAGIC_GUID).encode("ascii")).digest()
            ).decode("ascii")
            handshake = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
            )
            writer.write(handshake.encode("ascii"))
            await writer.drain()

            if vrta_path is not None and vrta_path.exists():
                vrta_hdr, frames = read_vrta_file(vrta_path)
                stride = max(1, int(round(100.0 / max(1.0, pose_hz))))
                period_s = 1.0 / max(1.0, pose_hz)
                bridge = VertiRTAGeodesyBridge(
                    anchor_lat_deg=vrta_hdr.datum_lat_deg,
                    anchor_lon_deg=vrta_hdr.datum_lon_deg,
                    anchor_h_amsl_m=vrta_hdr.datum_amsl_m,
                    fixed_geoid_n_m=vrta_hdr.egm2008_n_m,
                )
                for frame in frames[::stride]:
                    pkt = vrta_frame_to_ws_packet(frame, vrta_hdr, bridge=bridge)
                    writer.write(_encode_ws_text_frame(json.dumps(pkt)))
                    await writer.drain()
                    await asyncio.sleep(period_s)
            return

        # Static HTTP routes for :5180 live tactical console & container /healthz
        if path == "/healthz":
            body = b'{"status":"ok"}\n'
            ctype = "application/json; charset=utf-8"
        elif path == "/api/airspace.geojson":
            body = json.dumps(build_ltfj_ltbx_geometries()["geojson_feature_collection"]).encode("utf-8")
            ctype = "application/geo+json; charset=utf-8"
        elif path in ("/", "/index.html"):
            body = (static_dir / "index.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        elif path == "/app.js":
            body = (static_dir / "app.js").read_bytes()
            ctype = "application/javascript; charset=utf-8"
        else:
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return

        resp = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: {ctype}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Cache-Control: no-store\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("ascii")
        writer.write(resp + body)
        await writer.drain()
    except (ConnectionResetError, BrokenPipeError):
        pass
    finally:
        writer.close()
        await writer.wait_closed()


async def serve_console(
    *,
    host: str = DEFAULT_CESIUM_HOST,
    port: int = DEFAULT_CESIUM_PORT,
    vrta_path: str | Path | None = None,
    pose_hz: float = 30.0,
) -> None:
    """Start the combined HTTP + WebSocket CesiumJS tactical console server."""
    checked_port = validate_port(port)
    static_dir = Path(__file__).resolve().parent
    src_vrta = Path(vrta_path) if vrta_path is not None else None
    server = await asyncio.start_server(
        lambda r, w: _handle_client(r, w, static_dir=static_dir, vrta_path=src_vrta, pose_hz=pose_hz),
        host,
        checked_port,
    )
    async with server:
        await server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VertiRTA CesiumJS :5180 live ops server")
    parser.add_argument("--host", default=DEFAULT_CESIUM_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_CESIUM_PORT)
    parser.add_argument("--vrta", default=None, help="Optional .vrta flight log to replay over WS")
    parser.add_argument("--hz", type=float, default=30.0, help="WebSocket pose stream rate (default 30 Hz)")
    args = parser.parse_args(argv)
    validate_port(args.port)
    asyncio.run(serve_console(host=args.host, port=args.port, vrta_path=args.vrta, pose_hz=args.hz))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
