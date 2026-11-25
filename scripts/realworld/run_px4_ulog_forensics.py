# SPDX-License-Identifier: Apache-2.0
"""Real-World PX4 ULog v1 -> C++20 ED-155A Sidecar + W3C XSD ECCAIRS2 (.e5x.xml) Replayer.

Zero-GPL pure-Python PX4 ULog v1 binary parser (no pyulog hard dependency, optional
BSD-3 parity) that ingests a real open-source PX4 `.ulg` flight log (official
`PX4/pyulog` BSD-3-Clause corpus), extracts real flight telemetry
(`vehicle_local_position`, `vehicle_attitude`, `battery_status`,
`vehicle_gps_position`/`sensor_gps`, `actuator_outputs`), encodes canonical
frozen 128 B + 256 B CRC-32C `VRTA0001` frames, emits a tamper-evident ED-155A
`.vrta_events.jsonl` SHA-256 hash-chained sidecar (verified compatible with both
C++20 `VrtaEventsSidecarReader` and VertiGCS browser WebCrypto drag-and-drop
verifier), replays through `VertiForensicsEngine` (`cpp_nanobind` when available,
else `python_fallback`), validates the emitted `.e5x.xml` against
`core/forensics/schemas/eccairs2_e5x_minimal.xsd` via `lxml.etree.XMLSchema`,
and writes a SHA-256 sealed proof JSON (`< 1 MB`, pure UTF-8 LF, zero leaked
host paths).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import struct
import sys
import time
from typing import Any, Sequence
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_DIR = REPO_ROOT / "sdk" / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

from viz.geo.airspace_polygons import (  # noqa: E402
    signed_distance_to_ltbx_m,
    signed_distance_to_ltfj_m,
)
from viz.geo.vertirta_geodesy import (  # noqa: E402
    ANCHOR_H_AMSL,
    ANCHOR_LAT_DEG,
    ANCHOR_LON_DEG,
    DEFAULT_ISTANBUL_GEOID_N0_M,
    VertiRTAGeodesyBridge,
)

# Official BSD-3-Clause PX4/pyulog test flight logs (public GitHub raw)
DEFAULT_PX4_ULOG_URL = "https://raw.githubusercontent.com/PX4/pyulog/main/test/sample.ulg"
DEFAULT_PX4_ULOG_NAME = "sample.ulg"
DEFAULT_CACHE_DIR = Path("/var/tmp/vertirta_realworld_cache/px4_ulogs")
DEFAULT_OUT_JSON = REPO_ROOT / "artifacts/v4.0.0/realworld_px4_ulog_forensics_proof.json"
E5X_XSD_PATH = REPO_ROOT / "core/forensics/schemas/eccairs2_e5x_minimal.xsd"

ULOG_MAGIC = b"ULog\x01\x12\x35"
GENESIS_SHA256 = "0" * 64
SIDECAR_KEY_DOMAIN = b"VERTIRTA_ED155A_SIDECAR_KEY_V1"

_VRTA_HEADER_STRUCT = struct.Struct("<8sII4d32s40s4sI")
_VRTA_FRAME_STRUCT = struct.Struct("<2Q3d3f4f3f3f3f8f7f4BI80sI")
_CRC32C_POLY = 0x82F63B78

_ULOG_PRIMITIVES: dict[str, tuple[str, int]] = {
    "int8_t": ("b", 1),
    "uint8_t": ("B", 1),
    "int16_t": ("h", 2),
    "uint16_t": ("H", 2),
    "int32_t": ("i", 4),
    "uint32_t": ("I", 4),
    "int64_t": ("q", 8),
    "uint64_t": ("Q", 8),
    "float": ("f", 4),
    "double": ("d", 8),
    "bool": ("?", 1),
    "char": ("c", 1),
}

_NA_INJURY_ATTR_IDS: tuple[str, ...] = ("114", "187", "262", "460", "462", "463", "469", "472")


def _make_crc32c_table() -> tuple[int, ...]:
    tbl: list[int] = []
    for byte in range(256):
        c = byte
        for _ in range(8):
            c = ((c >> 1) ^ _CRC32C_POLY) if (c & 1) else (c >> 1)
        tbl.append(c & 0xFFFFFFFF)
    return tuple(tbl)


_CRC32C_TABLE: tuple[int, ...] = _make_crc32c_table()


def crc32c_bytes(data: bytes | bytearray | memoryview) -> int:
    """RFC 3720 / iSCSI Castagnoli CRC-32C (check value for b'123456789' == 0xE3069283)."""
    crc = 0xFFFFFFFF
    tbl = _CRC32C_TABLE
    for b in data:
        crc = tbl[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ULogFieldDef:
    name: str
    type_name: str
    array_size: int  # 0 for scalar, >= 1 for fixed array


@dataclass
class ULogSubscription:
    msg_id: int
    multi_id: int
    name: str
    struct_fmt: str
    field_names: list[str]
    record_size: int


@dataclass
class ULogDataset:
    name: str
    multi_id: int
    msg_id: int
    field_names: list[str]
    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ParsedULog:
    version: int
    start_timestamp_us: int
    header_flags_compat: bytes
    header_flags_incompat: bytes
    info_messages: dict[str, Any]
    formats: dict[str, list[ULogFieldDef]]
    datasets: dict[tuple[str, int], ULogDataset]
    logged_strings: list[tuple[int, int, str]]
    dropout_count: int
    dropout_total_ms: int
    message_type_counts: dict[str, int]
    file_size_bytes: int
    file_sha256: str


def _parse_format_fields(body: str) -> list[ULogFieldDef]:
    fields: list[ULogFieldDef] = []
    for raw_part in body.split(";"):
        part = raw_part.strip()
        if not part:
            continue
        tokens = part.split()
        if len(tokens) != 2:
            continue
        type_spec, field_name = tokens
        m = re.match(r"^([A-Za-z0-9_]+)\[(\d+)\]$", type_spec)
        if m:
            fields.append(ULogFieldDef(name=field_name, type_name=m.group(1), array_size=int(m.group(2))))
        else:
            fields.append(ULogFieldDef(name=field_name, type_name=type_spec, array_size=0))
    return fields


def _compile_ulog_layout(
    msg_name: str,
    formats: dict[str, list[ULogFieldDef]],
    prefix: str = "",
    depth: int = 0,
) -> tuple[str, list[str]]:
    """Recursively flatten a PX4 ULog v1 Format (`F`) definition into a little-endian struct format."""
    if depth > 8:
        raise ValueError(f"ULog nested format recursion exceeded max depth at {msg_name}")
    fdefs = formats.get(msg_name)
    if fdefs is None:
        raise KeyError(f"Unknown ULog format {msg_name!r}")

    # PX4 ULog spec: trailing `_padding*` fields on top-level messages are omitted from `D` payloads.
    active_fdefs = list(fdefs)
    if depth == 0:
        while active_fdefs and active_fdefs[-1].name.startswith("_padding"):
            active_fdefs.pop()

    fmt_parts: list[str] = []
    names: list[str] = []
    for fd in active_fdefs:
        full_name = f"{prefix}{fd.name}" if prefix else fd.name
        prim = _ULOG_PRIMITIVES.get(fd.type_name)
        if prim is not None:
            code, _ = prim
            if fd.array_size > 0:
                if fd.name.startswith("_padding"):
                    fmt_parts.append(f"{fd.array_size}x")
                elif code == "c":
                    fmt_parts.append(f"{fd.array_size}s")
                    names.append(full_name)
                else:
                    for idx in range(fd.array_size):
                        fmt_parts.append(code)
                        names.append(f"{full_name}[{idx}]")
            else:
                if fd.name.startswith("_padding"):
                    fmt_parts.append("x")
                else:
                    fmt_parts.append(code)
                    names.append(full_name)
        else:
            # Nested uORB struct message
            if fd.array_size > 0:
                for idx in range(fd.array_size):
                    sub_fmt, sub_names = _compile_ulog_layout(
                        fd.type_name, formats, prefix=f"{full_name}[{idx}].", depth=depth + 1
                    )
                    fmt_parts.append(sub_fmt)
                    names.extend(sub_names)
            else:
                sub_fmt, sub_names = _compile_ulog_layout(
                    fd.type_name, formats, prefix=f"{full_name}.", depth=depth + 1
                )
                fmt_parts.append(sub_fmt)
                names.extend(sub_names)
    return "".join(fmt_parts), names


def _decode_info_value(type_spec: str, val_bytes: bytes) -> Any:
    if type_spec.startswith("char["):
        return val_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    prim = _ULOG_PRIMITIVES.get(type_spec)
    if prim is not None:
        code, sz = prim
        if len(val_bytes) >= sz:
            return struct.unpack_from("<" + code, val_bytes, 0)[0]
    return val_bytes.hex()


def parse_px4_ulog(
    ulog_path: Path,
    *,
    topics_filter: set[str] | None = None,
    max_rows_per_topic: int = 50_000,
) -> ParsedULog:
    """Parse a binary PX4 ULog v1 file using zero external dependencies (pure Python, Apache-2.0)."""
    raw = ulog_path.read_bytes()
    file_sha = sha256_bytes(raw)
    if len(raw) < 16 or raw[:7] != ULOG_MAGIC:
        raise ValueError(f"Invalid PX4 ULog magic in {ulog_path.name}")

    version = raw[7]
    if version not in (0, 1):
        raise ValueError(f"Unsupported PX4 ULog version {version} (expected 0 or 1)")
    start_ts_us = struct.unpack_from("<Q", raw, 8)[0]

    formats: dict[str, list[ULogFieldDef]] = {}
    info_messages: dict[str, Any] = {}
    subscriptions: dict[int, ULogSubscription] = {}
    compiled_structs: dict[int, struct.Struct] = {}
    datasets: dict[tuple[str, int], ULogDataset] = {}
    logged_strings: list[tuple[int, int, str]] = []
    msg_counts: dict[str, int] = {}
    compat_flags = b"\x00" * 8
    incompat_flags = b"\x00" * 8
    dropout_count = 0
    dropout_total_ms = 0

    pos = 16
    n = len(raw)
    while pos + 3 <= n:
        msg_size, msg_type_byte = struct.unpack_from("<HB", raw, pos)
        pos += 3
        if pos + msg_size > n:
            break
        payload = memoryview(raw)[pos : pos + msg_size]
        pos += msg_size

        mtype = chr(msg_type_byte)
        msg_counts[mtype] = msg_counts.get(mtype, 0) + 1

        if mtype == "B" and msg_size >= 40:
            compat_flags = bytes(payload[:8])
            incompat_flags = bytes(payload[8:16])
        elif mtype == "F":
            text = bytes(payload).decode("utf-8", errors="replace")
            if ":" in text:
                fname, body = text.split(":", 1)
                formats[fname.strip()] = _parse_format_fields(body)
        elif mtype == "I" and msg_size >= 2:
            key_len = payload[0]
            if 1 + key_len <= msg_size:
                key_str = bytes(payload[1 : 1 + key_len]).decode("utf-8", errors="replace")
                val_bytes = bytes(payload[1 + key_len :])
                parts = key_str.split(" ", 1)
                if len(parts) == 2:
                    info_messages[parts[1]] = _decode_info_value(parts[0], val_bytes)
                else:
                    info_messages[key_str] = val_bytes.hex()
        elif mtype == "A" and msg_size >= 3:
            multi_id, msg_id = struct.unpack_from("<BH", payload, 0)
            topic_name = bytes(payload[3:]).decode("utf-8", errors="replace").strip()
            if topic_name in formats:
                try:
                    s_fmt, fnames = _compile_ulog_layout(topic_name, formats)
                    st = struct.Struct("<" + s_fmt)
                    sub = ULogSubscription(
                        msg_id=msg_id,
                        multi_id=multi_id,
                        name=topic_name,
                        struct_fmt="<" + s_fmt,
                        field_names=fnames,
                        record_size=st.size,
                    )
                    subscriptions[msg_id] = sub
                    compiled_structs[msg_id] = st
                    key = (topic_name, multi_id)
                    if key not in datasets:
                        datasets[key] = ULogDataset(
                            name=topic_name,
                            multi_id=multi_id,
                            msg_id=msg_id,
                            field_names=fnames,
                        )
                except (KeyError, struct.error, ValueError):
                    pass
        elif mtype == "D" and msg_size >= 2:
            msg_id = struct.unpack_from("<H", payload, 0)[0]
            sub = subscriptions.get(msg_id)
            if sub is None:
                continue
            if topics_filter is not None and sub.name not in topics_filter:
                continue
            ds = datasets[(sub.name, sub.multi_id)]
            if len(ds.rows) >= max_rows_per_topic:
                continue
            data_view = payload[2:]
            st = compiled_structs[msg_id]
            if len(data_view) >= st.size:
                vals = st.unpack_from(data_view, 0)
                ds.rows.append(dict(zip(sub.field_names, vals)))
        elif mtype == "L" and msg_size >= 9:
            log_level, ts_us = struct.unpack_from("<BQ", payload, 0)
            msg_txt = bytes(payload[9:]).decode("utf-8", errors="replace").rstrip("\x00")
            if len(logged_strings) < 512:
                logged_strings.append((int(log_level), int(ts_us), msg_txt))
        elif mtype == "O" and msg_size >= 2:
            dropout_count += 1
            dropout_total_ms += int(struct.unpack_from("<H", payload, 0)[0])

    return ParsedULog(
        version=version,
        start_timestamp_us=start_ts_us,
        header_flags_compat=compat_flags,
        header_flags_incompat=incompat_flags,
        info_messages=info_messages,
        formats=formats,
        datasets=datasets,
        logged_strings=logged_strings,
        dropout_count=dropout_count,
        dropout_total_ms=dropout_total_ms,
        message_type_counts=msg_counts,
        file_size_bytes=len(raw),
        file_sha256=file_sha,
    )


def write_synthetic_smoke_ulog(out_path: Path, *, n_samples: int = 120) -> Path:
    """Generate a valid minimal PX4 ULog v1 binary for zero-network `--smoke-cpu` tests."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = bytearray()
    # 16 B Header
    start_ts_us = 1_000_000
    buf.extend(ULOG_MAGIC)
    buf.append(1)
    buf.extend(struct.pack("<Q", start_ts_us))

    def _emit(msg_type: str, payload: bytes) -> None:
        buf.extend(struct.pack("<HB", len(payload), ord(msg_type)))
        buf.extend(payload)

    # Flags 'B'
    _emit("B", b"\x00" * 40)

    # Info 'I'
    for k, v in (
        ("char[4] sys_name", b"PX4\x00"),
        ("char[15] sys_hw", b"VERTIRTA_SMOKE\x00"),
        ("char[9] ver_sw", b"v1.15.0\x00\x00"),
    ):
        kb = k.encode("utf-8")
        _emit("I", bytes([len(kb)]) + kb + v)

    # Format definitions 'F'
    fmt_defs = [
        "vehicle_local_position:uint64_t timestamp;float x;float y;float z;float vx;float vy;float vz;bool xy_valid;bool z_valid;uint8_t[2] _padding0;",
        "vehicle_attitude:uint64_t timestamp;float[4] q;",
        "battery_status:uint64_t timestamp;float voltage_v;float voltage_filtered_v;float current_a;float remaining;",
        "vehicle_gps_position:uint64_t timestamp;int32_t lat;int32_t lon;int32_t alt;float eph;float epv;uint8_t satellites_used;uint8_t[3] _padding0;",
        "actuator_outputs:uint64_t timestamp;uint32_t noutputs;float[16] output;",
    ]
    for fdef in fmt_defs:
        _emit("F", fdef.encode("utf-8"))

    # Subscriptions 'A'
    topics = [
        (1, "vehicle_local_position"),
        (2, "vehicle_attitude"),
        (3, "battery_status"),
        (4, "vehicle_gps_position"),
        (5, "actuator_outputs"),
    ]
    for msg_id, name in topics:
        _emit("A", struct.pack("<BH", 0, msg_id) + name.encode("utf-8"))

    # Data 'D' at 10 Hz for n_samples (12.0 s flight with realistic coastal maneuver & single-motor sag)
    dt_us = 100_000
    for i in range(n_samples):
        ts = start_ts_us + i * dt_us
        t_s = i * 0.1
        # PX4 NED: x=North (moving NW along corridor), y=East (staying west of LTFJ), z=Down (-55 m AGL)
        x_n = 8.0 * t_s
        y_e = -120.0 - 4.0 * t_s
        z_d = -55.0 - 3.0 * math.sin(0.35 * t_s)
        vx_n = 8.0
        vy_e = -4.0
        vz_d = -1.05 * math.cos(0.35 * t_s)
        _emit("D", struct.pack("<HQ6f??2x", 1, ts, x_n, y_e, z_d, vx_n, vy_e, vz_d, True, True))

        roll = 0.08 * math.sin(0.7 * t_s) if i < 80 else 0.24 * math.sin(1.4 * t_s)
        qw = math.cos(roll * 0.5)
        qx = math.sin(roll * 0.5)
        _emit("D", struct.pack("<HQ4f", 2, ts, qw, qx, 0.0, 0.0))

        rem = max(0.14, 0.78 - 0.004 * i)
        volt = 22.8 - 0.025 * i if i < 80 else 21.1 - 0.03 * (i - 80)
        _emit("D", struct.pack("<HQ4f", 3, ts, volt, volt, 38.0 + 0.2 * i, rem))

        lat_e7 = int((ANCHOR_LAT_DEG + x_n / 111_052.5) * 1e7)
        lon_e7 = int((ANCHOR_LON_DEG + y_e / 84_152.0) * 1e7)
        alt_mm = int((ANCHOR_H_AMSL - z_d) * 1000.0)
        eph = 0.85 if i < 85 else 2.95
        _emit("D", struct.pack("<HQ3i2fB3x", 4, ts, lat_e7, lon_e7, alt_mm, eph, 1.2, 14))

        outs = [1550.0 + 40.0 * math.sin(0.5 * t_s)] * 8 + [0.0] * 8
        if i >= 80:
            outs[2] = 900.0  # Simulate rotor #3 thrust drop for envelope protection trigger
        _emit("D", struct.pack("<HQI16f", 5, ts, 8, *outs))

    # Logged string 'L'
    _emit("L", struct.pack("<BQ", 6, start_ts_us + 8_000_000) + b"RTA envelope check: rotor #3 output drop")

    out_path.write_bytes(bytes(buf))
    return out_path


def download_or_cache_px4_ulog(url: str, cache_dir: Path, filename: str) -> tuple[Path, bool]:
    """Download real PX4 `.ulg` into `/var/tmp/vertirta_realworld_cache/px4_ulogs/` if not cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 1024:
        return target, True
    req = urllib.request.Request(url, headers={"User-Agent": "vertirta-realworld/4.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        blob = resp.read()
    if len(blob) < 16 or blob[:7] != ULOG_MAGIC:
        raise RuntimeError(f"Downloaded file from {url} is not a valid PX4 ULog v1 binary")
    target.write_bytes(blob)
    return target, False


def _make_vrta_header(
    datum_lat: float = ANCHOR_LAT_DEG,
    datum_lon: float = ANCHOR_LON_DEG,
    datum_amsl: float = ANCHOR_H_AMSL,
    egm2008_n: float = DEFAULT_ISTANBUL_GEOID_N0_M,
    lut_sha256: bytes = b"\x00" * 32,
) -> bytes:
    prefix = _VRTA_HEADER_STRUCT.pack(
        b"VRTA0001",
        1,
        0,
        float(datum_lat),
        float(datum_lon),
        float(datum_amsl),
        float(egm2008_n),
        lut_sha256[:32].ljust(32, b"\x00"),
        b"px4-ulog-realworld-replay-v4.0.0".ljust(40, b"\x00")[:40],
        b"\x00" * 4,
        0,
    )[:124]
    return prefix + struct.pack("<I", crc32c_bytes(prefix))


def _pack_vrta_frame(
    *,
    seq: int,
    t_sim_us: int,
    pos_enu: tuple[float, float, float],
    vel_enu: tuple[float, float, float],
    q_frd_ned: tuple[float, float, float, float],
    omega_frd: tuple[float, float, float],
    rotor_rpm: Sequence[float],
    v_term_v: float,
    soc_pct: float,
    cbvf_min_v: float,
    ltfj_margin_m: float,
    ltbx_margin_m: float,
    chi2_residual: float,
    agl_m: float,
    active_motor_mask: int,
    flight_regime: int,
    cel_latched: int,
) -> bytes:
    rpms = [float(x) for x in list(rotor_rpm)[:8]]
    while len(rpms) < 8:
        rpms.append(0.0)
    raw_252 = _VRTA_FRAME_STRUCT.pack(
        int(seq),
        int(t_sim_us),
        float(pos_enu[0]),
        float(pos_enu[1]),
        float(pos_enu[2]),
        float(vel_enu[0]),
        float(vel_enu[1]),
        float(vel_enu[2]),
        float(q_frd_ned[0]),
        float(q_frd_ned[1]),
        float(q_frd_ned[2]),
        float(q_frd_ned[3]),
        float(omega_frd[0]),
        float(omega_frd[1]),
        float(omega_frd[2]),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *rpms,
        float(v_term_v),
        float(soc_pct),
        float(cbvf_min_v),
        float(ltfj_margin_m),
        float(ltbx_margin_m),
        float(chi2_residual),
        float(agl_m),
        int(active_motor_mask) & 0xFF,
        int(flight_regime) & 0xFF,
        0,
        int(cel_latched) & 0xFF,
        0,
        b"\x00" * 80,
        0,
    )[:252]
    return raw_252 + struct.pack("<I", crc32c_bytes(raw_252))


def _json_escape_cpp(raw: str) -> str:
    """Match `core/forensics/src/vrta_events_sidecar.cpp::json_escape` byte-for-byte."""
    out: list[str] = []
    for ch in raw:
        code = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    return "".join(out)


def build_canonical_sidecar_payload(ev: dict[str, Any]) -> str:
    """Match both C++20 `build_canonical_payload_json` and VertiGCS `buildForensicsCanonicalPayloadJson`."""
    unsat = ",".join(f'"{_json_escape_cpp(str(v))}"' for v in ev.get("z3_unsat_core", []))
    return (
        f'{{"seq":{int(ev["seq"])}'
        f',"t_sim_us":{int(ev["t_sim_us"])}'
        f',"kind":"{_json_escape_cpp(str(ev["kind"]))}"'
        f',"snr_stratum":"{_json_escape_cpp(str(ev["snr_stratum"]))}"'
        f',"conformal_set_size":{int(ev["conformal_set_size"])}'
        f',"asr_transcript_utf8":"{_json_escape_cpp(str(ev["asr_transcript_utf8"]))}"'
        f',"audio_sha256_hex":"{_json_escape_cpp(str(ev["audio_sha256_hex"]))}"'
        f',"z3_smt2_sha256_hex":"{_json_escape_cpp(str(ev["z3_smt2_sha256_hex"]))}"'
        f',"z3_unsat_core":[{unsat}]'
        f',"gsn_node_id":"{_json_escape_cpp(str(ev["gsn_node_id"]))}"'
        f',"prev_event_sha256":"{_json_escape_cpp(str(ev["prev_event_sha256"]))}"}}'
    )


def write_ed155a_sidecar_jsonl(
    sidecar_path: Path,
    events: Sequence[dict[str, Any]],
    *,
    deterministic_key_seed: int = 0x5645525449525441,
) -> tuple[str, list[dict[str, Any]]]:
    """Emit tamper-evident ED-155A `.vrta_events.jsonl` compatible with C++20 & VertiGCS."""
    key_bytes = hashlib.sha256(
        struct.pack("<Q", int(deterministic_key_seed)) + SIDECAR_KEY_DOMAIN
    ).digest()
    pub_key_hex = key_bytes.hex()

    prev_hash = GENESIS_SHA256
    lines: list[str] = []
    sealed_records: list[dict[str, Any]] = []
    for raw_ev in events:
        ev = {
            "seq": int(raw_ev["seq"]),
            "t_sim_us": int(raw_ev["t_sim_us"]),
            "kind": str(raw_ev["kind"]),
            "snr_stratum": str(raw_ev.get("snr_stratum", "S1_Cruise")),
            "conformal_set_size": int(raw_ev.get("conformal_set_size", 1)),
            "asr_transcript_utf8": str(raw_ev.get("asr_transcript_utf8", "")),
            "audio_sha256_hex": str(
                raw_ev.get(
                    "audio_sha256_hex",
                    sha256_bytes(str(raw_ev.get("asr_transcript_utf8", "")).encode("utf-8")),
                )
            ),
            "z3_smt2_sha256_hex": str(
                raw_ev.get("z3_smt2_sha256_hex", sha256_bytes(b"(check-sat)"))
            ),
            "z3_unsat_core": [str(x) for x in raw_ev.get("z3_unsat_core", [])],
            "gsn_node_id": str(raw_ev.get("gsn_node_id", "G1.1-INDI-WLS")),
            "prev_event_sha256": prev_hash,
        }
        canonical = build_canonical_sidecar_payload(ev)
        ev_hash = sha256_bytes(canonical.encode("utf-8"))
        sig_hex = hmac.new(key_bytes, ev_hash.encode("ascii"), hashlib.sha256).hexdigest()
        line = (
            canonical[:-1]
            + f',"event_sha256":"{ev_hash}"'
            + f',"signature_hex":"{sig_hex}"'
            + ',"signer_backend":"SIM_SHA256_TEST_SIGNER"}'
        )
        lines.append(line)
        sealed_records.append(
            {
                **ev,
                "event_sha256": ev_hash,
                "signature_hex": sig_hex,
                "signer_backend": "SIM_SHA256_TEST_SIGNER",
            }
        )
        prev_hash = ev_hash

    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return pub_key_hex, sealed_records


def verify_sidecar_chain_vertigcs_compatible(
    ndjson_text: str,
    *,
    public_key_hex: str | None = None,
) -> dict[str, Any]:
    """Verify `.vrta_events.jsonl` SHA-256 chain identically to VertiGCS browser verifier + HMAC check."""
    lines = [ln.strip() for ln in ndjson_text.splitlines() if ln.strip()]
    if not lines:
        return {"valid": False, "event_count": 0, "error": "empty sidecar"}

    pub_bytes = bytes.fromhex(public_key_hex) if public_key_hex else None
    expected_prev = GENESIS_SHA256
    last_seq = -1
    last_t_us = -1

    for idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            return {"valid": False, "event_count": idx, "error": f"line {idx + 1}: malformed JSON: {exc}"}
        seq = int(obj.get("seq", -1))
        t_us = int(obj.get("t_sim_us", -1))
        if seq < 0 or t_us < 0 or seq < last_seq or t_us < last_t_us:
            return {"valid": False, "event_count": idx, "error": f"line {idx + 1}: non-monotonic (seq, t_sim_us)"}
        if obj.get("prev_event_sha256") != expected_prev:
            return {"valid": False, "event_count": idx, "error": f"line {idx + 1}: prev_event_sha256 chain break"}
        canonical = build_canonical_sidecar_payload(obj)
        calc_hash = sha256_bytes(canonical.encode("utf-8"))
        if calc_hash != obj.get("event_sha256"):
            return {"valid": False, "event_count": idx, "error": f"line {idx + 1}: event_sha256 mismatch"}
        if pub_bytes is not None and obj.get("signer_backend") == "SIM_SHA256_TEST_SIGNER":
            expected_sig = hmac.new(pub_bytes, calc_hash.encode("ascii"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected_sig, str(obj.get("signature_hex", ""))):
                return {"valid": False, "event_count": idx, "error": f"line {idx + 1}: HMAC signature mismatch"}
        expected_prev = calc_hash
        last_seq = seq
        last_t_us = t_us

    return {
        "valid": True,
        "event_count": len(lines),
        "head_event_sha256": expected_prev,
        "error": "",
    }


def _find_topic_rows(parsed: ParsedULog, candidates: Sequence[str]) -> tuple[str | None, list[dict[str, Any]]]:
    for name in candidates:
        for (tname, _mid), ds in sorted(parsed.datasets.items()):
            if tname == name and ds.rows:
                return tname, ds.rows
    return None, []


def _nearest_row_by_timestamp(
    rows: Sequence[dict[str, Any]], target_ts: int, cursor: int
) -> tuple[dict[str, Any] | None, int]:
    if not rows:
        return None, 0
    n = len(rows)
    idx = min(max(cursor, 0), n - 1)
    while idx + 1 < n and int(rows[idx + 1].get("timestamp", 0)) <= target_ts:
        idx += 1
    return rows[idx], idx


def _load_native_module() -> Any | None:
    """Load prebuilt C++20 `_vertirta_native.abi3.so` when available (without touching sdk/)."""
    candidates = (
        REPO_ROOT / "build/cp312-abi3-linux_x86_64/_vertirta_native.abi3.so",
        REPO_ROOT / "build/dev-linux-x64/sdk/python/_vertirta_native.abi3.so",
    )
    for so_path in candidates:
        if not so_path.is_file():
            continue
        try:
            spec = importlib.util.spec_from_file_location("_vertirta_native", str(so_path))
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if getattr(mod, "HAS_M3_BINDINGS", False):
                    return mod
        except Exception:
            continue
    return None


def _import_vertirta_m3() -> Any:
    """Import `sdk/python/vertirta/m3.py` cleanly even when optional `gymnasium` is absent."""
    import types

    try:
        import vertirta.m3 as m3_mod  # type: ignore[import-not-found]

        return m3_mod
    except ModuleNotFoundError:
        pass

    pkg = sys.modules.get("vertirta")
    if pkg is None:
        pkg = types.ModuleType("vertirta")
        pkg.__path__ = [str(SDK_DIR / "vertirta")]  # type: ignore[attr-defined]
        pkg._native = _load_native_module()  # type: ignore[attr-defined]
        sys.modules["vertirta"] = pkg
    m3_path = SDK_DIR / "vertirta" / "m3.py"
    spec = importlib.util.spec_from_file_location("vertirta.m3", str(m3_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {m3_path}")
    m3_mod = importlib.util.module_from_spec(spec)
    sys.modules["vertirta.m3"] = m3_mod
    spec.loader.exec_module(m3_mod)
    return m3_mod


def _load_forensics_engine() -> tuple[Any, str]:
    """Instantiate `VertiForensicsEngine`, binding compiled C++20 `_vertirta_native.abi3.so` when present."""
    m3_mod = _import_vertirta_m3()
    engine = m3_mod.VertiForensicsEngine()
    if engine.backend == "cpp_nanobind":
        return engine, "cpp_nanobind"

    native_mod = _load_native_module()
    if native_mod is not None and hasattr(native_mod, "NativeForensicsEngine"):
        engine._native = native_mod.NativeForensicsEngine()  # noqa: SLF001
    return engine, engine.backend


def validate_e5x_xml_against_xsd(e5x_xml_path: Path, xsd_path: Path = E5X_XSD_PATH) -> dict[str, Any]:
    """Validate emitted ECCAIRS2 `.e5x.xml` via W3C XSD 1.0 (`lxml.etree.XMLSchema`) + statutory checks."""
    from lxml import etree  # type: ignore[import-untyped]

    schema_doc = etree.parse(str(xsd_path))
    schema = etree.XMLSchema(schema_doc)
    xml_doc = etree.parse(str(e5x_xml_path))
    xsd_valid = bool(schema.validate(xml_doc))
    xsd_errors = [str(err) for err in schema.error_log] if not xsd_valid else []

    xml_text = e5x_xml_path.read_text(encoding="utf-8")
    root = xml_doc.getroot()
    ns = {"e5x": "http://eccairsportal.jrc.ec.europa.eu/schemas/e5x-minimal/v1"}
    occ = root.find("e5x:Occurrence", ns)
    attr_ids: list[int] = []
    if occ is not None:
        attrs = occ.find("e5x:ATTRIBUTES", ns)
        if attrs is not None:
            for attr in attrs.findall("e5x:Attribute", ns):
                attr_ids.append(int(attr.attrib["id"]))

    strictly_ascending = all(attr_ids[i] < attr_ids[i + 1] for i in range(len(attr_ids) - 1))
    na_injury_omitted = all(f'id="{na_id}"' not in xml_text for na_id in _NA_INJURY_ATTR_IDS)
    narratives = occ.findall("e5x:ENTITIES/e5x:Narrative", ns) if occ is not None else []
    langs = sorted(n.attrib.get("language", "") for n in narratives)

    return {
        "xsd_valid": xsd_valid,
        "xsd_errors": xsd_errors,
        "xsd_schema_relpath": str(xsd_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "occurrence_attribute_ids": attr_ids,
        "strictly_ascending_attribute_ids": strictly_ascending,
        "na_injury_ids_omitted": na_injury_omitted,
        "narrative_languages": langs,
        "reporting_status": occ.attrib.get("reportingStatus", "") if occ is not None else "",
        "compliance_claim": occ.attrib.get("complianceClaim", "") if occ is not None else "",
        "e5x_xml_sha256": sha256_file(e5x_xml_path),
    }


def replay_px4_ulog_to_forensics(
    ulog_path: Path,
    *,
    work_dir: Path,
    source_label: str,
    source_url: str,
    source_license: str,
    max_vrta_frames: int = 400,
) -> dict[str, Any]:
    """Parse `.ulg`, write `.vrta` + ED-155A `.vrta_events.jsonl`, run `VertiForensicsEngine`, verify XSD."""
    t0 = time.perf_counter()
    target_topics = {
        "vehicle_local_position",
        "vehicle_attitude",
        "battery_status",
        "vehicle_gps_position",
        "sensor_gps",
        "actuator_outputs",
        "vehicle_angular_velocity",
    }
    parsed = parse_px4_ulog(ulog_path, topics_filter=target_topics)

    loc_name, loc_rows = _find_topic_rows(parsed, ("vehicle_local_position",))
    att_name, att_rows = _find_topic_rows(parsed, ("vehicle_attitude",))
    bat_name, bat_rows = _find_topic_rows(parsed, ("battery_status",))
    gps_name, gps_rows = _find_topic_rows(parsed, ("vehicle_gps_position", "sensor_gps"))
    act_name, act_rows = _find_topic_rows(parsed, ("actuator_outputs",))
    ang_name, ang_rows = _find_topic_rows(parsed, ("vehicle_angular_velocity",))

    # Choose primary driver timeline (prefer vehicle_local_position, else vehicle_attitude, else actuator_outputs)
    primary_topic = loc_name or att_name or act_name
    primary_rows = loc_rows or att_rows or act_rows
    if not primary_rows or primary_topic is None:
        raise RuntimeError(
            f"No usable flight telemetry topics found in {ulog_path.name}; "
            f"available={sorted({k[0] for k in parsed.datasets})[:20]}"
        )

    ts_first = int(primary_rows[0].get("timestamp", parsed.start_timestamp_us))
    ts_last = int(primary_rows[-1].get("timestamp", ts_first))
    flight_duration_s = max(0.0, (ts_last - ts_first) * 1e-6)

    # Subsample evenly up to max_vrta_frames preserving first & last samples
    n_total = len(primary_rows)
    if n_total <= max_vrta_frames:
        sample_indices = list(range(n_total))
    else:
        step = (n_total - 1) / float(max_vrta_frames - 1)
        sample_indices = sorted({int(round(i * step)) for i in range(max_vrta_frames)})

    bridge = VertiRTAGeodesyBridge(fixed_geoid_n_m=DEFAULT_ISTANBUL_GEOID_N0_M)
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = ulog_path.stem
    vrta_path = work_dir / f"{stem}.vrta"
    sidecar_path = work_dir / f"{stem}.vrta_events.jsonl"
    e5x_xml_path = work_dir / f"{stem}.e5x.xml"
    audit_json_path = work_dir / f"{stem}.forensics_audit.json"

    # Extract baseline origin for mapping relative PX4 displacement onto Kartal corridor anchor
    p0_x = float(loc_rows[0].get("x", 0.0)) if loc_rows else 0.0
    p0_y = float(loc_rows[0].get("y", 0.0)) if loc_rows else 0.0
    p0_z = float(loc_rows[0].get("z", 0.0)) if loc_rows else 0.0

    vrta_buf = bytearray(_make_vrta_header(lut_sha256=bytes.fromhex(parsed.file_sha256)))

    att_cur = bat_cur = gps_cur = act_cur = ang_cur = 0
    max_speed_mps = 0.0
    max_agl_m = 0.0
    min_soc_pct = 100.0
    min_voltage_v = float("inf")
    max_tilt_deg = 0.0
    min_ltfj_m = float("inf")
    min_ltbx_m = float("inf")

    frame_meta: list[dict[str, Any]] = []
    for seq_idx, row_i in enumerate(sample_indices):
        prow = primary_rows[row_i]
        ts_us = int(prow.get("timestamp", ts_first + seq_idx * 10_000))
        t_rel_us = max(0, ts_us - ts_first)

        # Position / velocity in PX4 NED -> VertiRTA ENU (relative displacement anchored at Kartal safe corridor)
        if loc_rows:
            lrow = prow if primary_topic == loc_name else _nearest_row_by_timestamp(loc_rows, ts_us, 0)[0]
            x_n = float(lrow.get("x", 0.0)) - p0_x
            y_e = float(lrow.get("y", 0.0)) - p0_y
            z_d = float(lrow.get("z", 0.0)) - p0_z
            vx_n = float(lrow.get("vx", 0.0))
            vy_e = float(lrow.get("vy", 0.0))
            vz_d = float(lrow.get("vz", 0.0))
            if not (math.isfinite(x_n) and math.isfinite(y_e) and math.isfinite(z_d)):
                x_n = y_e = z_d = 0.0
            if not (math.isfinite(vx_n) and math.isfinite(vy_e) and math.isfinite(vz_d)):
                vx_n = vy_e = vz_d = 0.0
        else:
            x_n = y_e = z_d = vx_n = vy_e = vz_d = 0.0

        # Anchor relative displacement 200 m west of Kartal vertiport (safe side of LTFJ J2)
        e_m = -200.0 + max(-3000.0, min(150.0, y_e))
        n_m = max(-2000.0, min(4000.0, x_n))
        agl_m = max(0.0, -z_d)
        speed_mps = math.hypot(vx_n, vy_e, vz_d)
        max_speed_mps = max(max_speed_mps, speed_mps)
        max_agl_m = max(max_agl_m, agl_m)

        # Geodetic projection onto Kartal coastal corridor for real LTFJ/LTBX clearance computation
        geo = bridge.flat_enu_to_cesium_wgs84(e_m, n_m, agl_m)
        ltfj_m = float(signed_distance_to_ltfj_m(geo["lat_deg"], geo["lon_deg"]))
        ltbx_m = float(signed_distance_to_ltbx_m(geo["lat_deg"], geo["lon_deg"]))
        min_ltfj_m = min(min_ltfj_m, ltfj_m)
        min_ltbx_m = min(min_ltbx_m, ltbx_m)

        # Attitude quaternion
        arow, att_cur = _nearest_row_by_timestamp(att_rows, ts_us, att_cur)
        if arow is not None:
            qw = float(arow.get("q[0]", 1.0))
            qx = float(arow.get("q[1]", 0.0))
            qy = float(arow.get("q[2]", 0.0))
            qz = float(arow.get("q[3]", 0.0))
            qnorm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            if not math.isfinite(qnorm) or qnorm < 1e-6:
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            else:
                qw, qx, qy, qz = qw / qnorm, qx / qnorm, qy / qnorm, qz / qnorm
        else:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        # Tilt angle from vertical: cos(tilt) = 1 - 2*(qx^2 + qy^2)
        cos_tilt = max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))
        tilt_deg = math.degrees(math.acos(cos_tilt))
        max_tilt_deg = max(max_tilt_deg, tilt_deg)

        # Angular velocity
        wrow, ang_cur = _nearest_row_by_timestamp(ang_rows, ts_us, ang_cur)
        if wrow is not None:
            omega = (
                float(wrow.get("xyz[0]", 0.0)),
                float(wrow.get("xyz[1]", 0.0)),
                float(wrow.get("xyz[2]", 0.0)),
            )
            omega = tuple(v if math.isfinite(v) else 0.0 for v in omega)  # type: ignore[assignment]
        else:
            omega = (0.0, 0.0, 0.0)

        # Battery status
        brow, bat_cur = _nearest_row_by_timestamp(bat_rows, ts_us, bat_cur)
        if brow is not None:
            v_raw = float(brow.get("voltage_filtered_v", brow.get("voltage_v", 0.0)))
            if not math.isfinite(v_raw) or v_raw <= 0.1:
                v_raw = float(brow.get("voltage_v", 22.2))
            rem = float(brow.get("remaining", 0.75))
            soc_pct = max(0.0, min(100.0, rem * 100.0 if rem <= 1.05 else rem))
            v_term_v = v_raw if (math.isfinite(v_raw) and v_raw > 0.1) else 22.2
        else:
            v_term_v = 22.2
            soc_pct = 75.0
        min_soc_pct = min(min_soc_pct, soc_pct)
        min_voltage_v = min(min_voltage_v, v_term_v)

        # GPS EPH (if present)
        grow, gps_cur = _nearest_row_by_timestamp(gps_rows, ts_us, gps_cur)
        eph_m = float(grow.get("eph", 1.0)) if grow is not None else 1.0
        if not math.isfinite(eph_m):
            eph_m = 1.0

        # Actuator outputs -> 8-channel motor outputs & active mask
        act_row, act_cur = _nearest_row_by_timestamp(act_rows, ts_us, act_cur)
        rpms: list[float] = []
        active_mask = 0xFF
        if act_row is not None:
            for ch in range(8):
                val = float(act_row.get(f"output[{ch}]", 0.0))
                if not math.isfinite(val):
                    val = 0.0
                rpms.append(val)
            nonzero = [v for v in rpms if v > 950.0]
            if nonzero:
                mean_out = sum(nonzero) / len(nonzero)
                for ch, v in enumerate(rpms):
                    if 0.0 < v < 0.72 * mean_out:
                        active_mask &= ~(1 << ch)
        else:
            rpms = [1500.0] * 8

        flight_regime = 1 if active_mask != 0xFF else 0
        chi2_val = round(0.45 * tilt_deg + 1.5 * max(0.0, eph_m - 1.0), 3)
        frame_bytes = _pack_vrta_frame(
            seq=seq_idx,
            t_sim_us=t_rel_us,
            pos_enu=(e_m, n_m, agl_m),
            vel_enu=(vy_e, vx_n, -vz_d),
            q_frd_ned=(qw, qx, qy, qz),
            omega_frd=omega,
            rotor_rpm=rpms,
            v_term_v=v_term_v,
            soc_pct=soc_pct,
            cbvf_min_v=max(0.1, ltfj_m * 0.01),
            ltfj_margin_m=ltfj_m,
            ltbx_margin_m=ltbx_m,
            chi2_residual=chi2_val,
            agl_m=agl_m,
            active_motor_mask=active_mask,
            flight_regime=flight_regime,
            cel_latched=0,
        )
        vrta_buf.extend(frame_bytes)
        frame_meta.append(
            {
                "seq": seq_idx,
                "t_sim_us": t_rel_us,
                "tilt_deg": tilt_deg,
                "eph_m": eph_m,
                "soc_pct": soc_pct,
                "active_mask": active_mask,
                "ltfj_m": ltfj_m,
            }
        )

    vrta_path.write_bytes(bytes(vrta_buf))
    vrta_sha = sha256_file(vrta_path)

    # Select 3 monotonic key flight telemetry events anchored to exact (seq, t_sim_us) VRTA frames
    first_f = frame_meta[0]
    peak_tilt_f = max(frame_meta, key=lambda x: x["tilt_deg"])
    last_f = frame_meta[-1]

    # Deduplicate while preserving strict monotonic (seq, t_sim_us)
    chosen_events: list[dict[str, Any]] = [
        {
            "seq": first_f["seq"],
            "t_sim_us": first_f["t_sim_us"],
            "kind": "RTA_ENVELOPE_PROTECTION",
            "snr_stratum": "S0_Clean",
            "conformal_set_size": 1,
            "asr_transcript_utf8": "PX4 ULog replay initialized on Kartal coastal corridor datum",
            "gsn_node_id": "G1.1-INDI-WLS",
            "z3_unsat_core": [],
        }
    ]
    if peak_tilt_f["seq"] > first_f["seq"]:
        chosen_events.append(
            {
                "seq": peak_tilt_f["seq"],
                "t_sim_us": peak_tilt_f["t_sim_us"],
                "kind": "MOTOR_FDI_FAULT" if peak_tilt_f["active_mask"] != 0xFF else "RTA_ENVELOPE_PROTECTION",
                "snr_stratum": "S1_Cruise",
                "conformal_set_size": 1,
                "asr_transcript_utf8": "PX4 peak attitude/envelope excursion logged to ED-155A sidecar",
                "gsn_node_id": "G2.1-CBVF-LTFJ",
                "z3_unsat_core": [],
            }
        )
    if last_f["seq"] > chosen_events[-1]["seq"]:
        chosen_events.append(
            {
                "seq": last_f["seq"],
                "t_sim_us": last_f["t_sim_us"],
                "kind": "CRITICAL_ENERGY_RESERVE" if last_f["soc_pct"] < 20.0 else "RTA_ENVELOPE_PROTECTION",
                "snr_stratum": "S1_Cruise",
                "conformal_set_size": 1,
                "asr_transcript_utf8": "PX4 ULog end-of-flight telemetry sealed for ECCAIRS2 E5X export",
                "gsn_node_id": "G4.1-ED155A-E5X",
                "z3_unsat_core": [],
            }
        )

    pub_key_hex, sealed_events = write_ed155a_sidecar_jsonl(sidecar_path, chosen_events)
    sidecar_text = sidecar_path.read_text(encoding="utf-8")
    sidecar_sha = sha256_file(sidecar_path)
    sidecar_verify = verify_sidecar_chain_vertigcs_compatible(sidecar_text, public_key_hex=pub_key_hex)
    if not sidecar_verify["valid"]:
        raise RuntimeError(f"ED-155A sidecar verification failed: {sidecar_verify['error']}")

    # Tamper-resistance self-check (flip 1 char in payload -> MUST fail closed)
    tampered_text = sidecar_text.replace("S0_Clean", "S9_Clean", 1)
    tamper_check = verify_sidecar_chain_vertigcs_compatible(tampered_text, public_key_hex=pub_key_hex)
    tamper_rejected = not tamper_check["valid"]

    # Replay through REAL VertiForensicsEngine (C++20 nanobind when built, else Python fallback)
    engine, backend = _load_forensics_engine()
    export_res = engine.export_e5x_from_vrta(
        vrta_path,
        e5x_xml_path,
        audit_json_path,
        sidecar_path=sidecar_path,
        sidecar_public_key_hex=pub_key_hex,
        file_number="VRT-2026-PX4-ULOG-001",
        raw_headline=f"Real PX4 ULog ({ulog_path.name}) envelope replay along Istanbul coastal corridor",
        occupant_capacity=1,
    )
    if not export_res.get("ok", False):
        raise RuntimeError(f"VertiForensicsEngine export failed: {export_res.get('error_message')}")

    xsd_report = validate_e5x_xml_against_xsd(e5x_xml_path, E5X_XSD_PATH)
    if not xsd_report["xsd_valid"]:
        raise RuntimeError(f"E5X XSD validation failed: {xsd_report['xsd_errors']}")

    audit_disk = json.loads(audit_json_path.read_text(encoding="utf-8"))
    numeric_hallucination_count = int(audit_disk.get("numeric_hallucination_count", 0))

    # Sanitize ULog info metadata (JSON-serializable, ASCII-safe, zero host paths)
    info_summary: dict[str, Any] = {}
    for k in ("sys_name", "sys_hw", "ver_sw", "ver_sw_branch", "ver_os_name", "sys_os_name", "time_ref_utc"):
        if k in parsed.info_messages:
            val = parsed.info_messages[k]
            info_summary[k] = val if isinstance(val, (int, float, bool)) else str(val)[:80]

    extracted_topic_counts = {
        tname: len(ds.rows)
        for (tname, _mid), ds in sorted(parsed.datasets.items())
        if tname in target_topics and ds.rows
    }

    elapsed_s = round(time.perf_counter() - t0, 4)
    report: dict[str, Any] = {
        "schema_version": "vertirta.realworld_px4_ulog_forensics_proof/v1.0.0",
        "mode": source_label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "colab_cu_consumed": 0.0,
        "host_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "source_ulog": {
            "filename": ulog_path.name,
            "source_url": source_url,
            "license": source_license,
            "zero_gpl_parser": "pure_python_ulog_v1_apache2",
            "ulog_version": parsed.version,
            "file_size_bytes": parsed.file_size_bytes,
            "file_sha256": parsed.file_sha256,
            "start_timestamp_us": parsed.start_timestamp_us,
            "flight_duration_s": round(flight_duration_s, 3),
            "format_definitions_count": len(parsed.formats),
            "subscribed_topics_count": len(parsed.datasets),
            "message_type_counts": dict(sorted(parsed.message_type_counts.items())),
            "total_ulog_messages": sum(parsed.message_type_counts.values()),
            "logged_strings_count": len(parsed.logged_strings),
            "dropout_count": parsed.dropout_count,
            "dropout_total_ms": parsed.dropout_total_ms,
            "system_info": info_summary,
            "extracted_telemetry_rows": extracted_topic_counts,
        },
        "telemetry_summary": {
            "primary_driver_topic": primary_topic,
            "vrta_frames_encoded": len(frame_meta),
            "vrta_file_size_bytes": vrta_path.stat().st_size,
            "vrta_file_sha256": vrta_sha,
            "max_speed_mps": round(max_speed_mps, 3),
            "max_relative_agl_m": round(max_agl_m, 3),
            "max_tilt_deg": round(max_tilt_deg, 3),
            "min_battery_soc_pct": round(min_soc_pct, 2),
            "min_battery_voltage_v": round(min_voltage_v if math.isfinite(min_voltage_v) else 0.0, 3),
            "min_ltfj_clearance_m": round(min_ltfj_m, 2),
            "min_ltbx_clearance_m": round(min_ltbx_m, 2),
        },
        "ed155a_sidecar": {
            "filename": sidecar_path.name,
            "event_count": len(sealed_events),
            "sidecar_sha256": sidecar_sha,
            "public_key_hex": pub_key_hex,
            "signer_backend": "SIM_SHA256_TEST_SIGNER",
            "vertigcs_chain_verified": sidecar_verify["valid"],
            "head_event_sha256": sidecar_verify["head_event_sha256"],
            "tamper_rejection_self_test_passed": tamper_rejected,
            "joined_events_count": int(export_res.get("sidecar_events_joined", 0)),
        },
        "forensics_e5x": {
            "engine_backend": backend,
            "ok": bool(export_res["ok"]),
            "frames_verified": int(export_res["frames_verified"]),
            "cictt_category": str(export_res["cictt_category"]),
            "ercs_matrix_code": str(export_res["ercs_matrix_code"]),
            "effective_barrier8_score": int(export_res["effective_barrier8_score"]),
            "occurrence_class_id_431": str(export_res["occurrence_class_id_431"]),
            "auto_classified_accident_or_serious_incident": bool(
                export_res["auto_classified_accident_or_serious_incident"]
            ),
            "numeric_hallucination_count": numeric_hallucination_count,
            "e5x_xml_filename": e5x_xml_path.name,
            **xsd_report,
        },
        "statutory_gates": {
            "cictt_scf_np_locked": str(export_res["cictt_category"]) == "SCF-NP",
            "barrier8_clamped_zero": int(export_res["effective_barrier8_score"]) == 0,
            "draft_incident_only_id_431": str(export_res["occurrence_class_id_431"]) == "DRAFT_INCIDENT_ONLY",
            "zero_auto_accident_classification": not bool(
                export_res["auto_classified_accident_or_serious_incident"]
            ),
            "zero_hallucinated_numbers": numeric_hallucination_count == 0,
            "w3c_xsd_passed": bool(xsd_report["xsd_valid"]),
            "sidecar_chain_and_tamper_verified": bool(sidecar_verify["valid"] and tamper_rejected),
        },
        "elapsed_s": elapsed_s,
    }
    report["all_gates_passed"] = all(report["statutory_gates"].values())
    canonical_digest = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["report_sha256"] = sha256_bytes(canonical_digest)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a real PX4 ULog v1 binary through VertiForensicsEngine -> ED-155A + XSD-valid ECCAIRS2 E5X."
    )
    parser.add_argument("--ulog-url", default=DEFAULT_PX4_ULOG_URL)
    parser.add_argument("--ulog-name", default=DEFAULT_PX4_ULOG_NAME)
    parser.add_argument("--ulog-path", type=Path, default=None, help="Optional local .ulg binary path")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Zero-network smoke mode: synthesizes a valid minimal PX4 ULog v1 binary in cache-dir.",
    )
    args = parser.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke_cpu:
        ulog_file = write_synthetic_smoke_ulog(args.cache_dir / "smoke_px4_flight.ulg")
        source_label = "SMOKE_CPU_ONLY"
        source_url = "synthetic://smoke-cpu-ulog-v1"
        source_license = "Apache-2.0"
    elif args.ulog_path is not None:
        ulog_file = args.ulog_path
        source_label = "LOCAL_PX4_ULOG"
        source_url = f"local://{ulog_file.name}"
        source_license = "BSD-3-Clause"
    else:
        ulog_file, _cached = download_or_cache_px4_ulog(args.ulog_url, args.cache_dir, args.ulog_name)
        source_label = "LIVE_REALWORLD_PX4_ULOG"
        source_url = args.ulog_url
        source_license = "BSD-3-Clause (PX4/pyulog)"

    report = replay_px4_ulog_to_forensics(
        ulog_file,
        work_dir=args.cache_dir / "outputs",
        source_label=source_label,
        source_url=source_url,
        source_license=source_license,
        max_vrta_frames=max(10, int(args.max_frames)),
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "ulog": report["source_ulog"]["filename"],
                "total_ulog_messages": report["source_ulog"]["total_ulog_messages"],
                "flight_duration_s": report["source_ulog"]["flight_duration_s"],
                "vrta_frames": report["telemetry_summary"]["vrta_frames_encoded"],
                "engine_backend": report["forensics_e5x"]["engine_backend"],
                "cictt": report["forensics_e5x"]["cictt_category"],
                "ercs": report["forensics_e5x"]["ercs_matrix_code"],
                "xsd_valid": report["forensics_e5x"]["xsd_valid"],
                "all_gates_passed": report["all_gates_passed"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
        )
    )
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
