"""Canonical .vrta Binary Flight Recorder Reader/Writer (128 B Header + 256 B Frames).

Byte-for-byte little-endian POD contract matching Plan 1A Task 5
(``core/flight_core/include/vertirta/core/vrta_recorder.hpp``):

- ``VrtaFileHeader`` (128 B, naturally aligned, zero implicit padding):
  ``<8sII4d32s40s4sI``
  * ``magic[8]`` = ``b"VRTA0001"`` (0..7)
  * ``schema_ver`` = ``1`` (8..11), ``reserved0`` = ``0`` (12..15)
  * ``datum_lat_deg, datum_lon_deg, datum_amsl_m, egm2008_n_m`` (16..47)
  * ``lut_sha256[32]`` (48..79), ``git_sha[40]`` (80..119), ``reserved1[4]`` (120..123)
  * ``header_crc32c`` (124..127, Castagnoli CRC-32C over bytes 0..123)

- ``VrtaFrame100Hz`` (256 B, naturally aligned, zero implicit padding):
  ``<2Q3d3f4f3f3f3f8f7f4BI80sI``
  * ``seq, t_sim_us`` (0..15)
  * ``pos_enu[3]`` double (16..39)
  * ``vel_enu[3], q_frd_ned[4], omega_frd[3], a_nom_enu[3], a_safe_enu[3], rotor_rpm[8]`` float (40..135)
  * ``v_term_v, soc_pct, cbvf_min_v, ltfj_margin_m, ltbx_margin_m, chi2_residual, agl_m`` float (136..163)
  * ``active_motor_mask, flight_regime, tmr_state, cel_latched`` uint8 (164..167)
  * ``flags`` uint32 (168..171), ``pad[80]`` uint8 (172..251)
  * ``crc32c`` uint32 (252..255, Castagnoli CRC-32C over bytes 0..251)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

VRTA_MAGIC: bytes = b"VRTA0001"
VRTA_SCHEMA_VER: int = 1

VRTA_HEADER_FMT: str = "<8sII4d32s40s4sI"
VRTA_HEADER_STRUCT = struct.Struct(VRTA_HEADER_FMT)
VRTA_HEADER_SIZE: int = VRTA_HEADER_STRUCT.size  # 128 bytes exact

VRTA_FRAME_FMT: str = "<2Q3d3f4f3f3f3f8f7f4BI80sI"
VRTA_FRAME_STRUCT = struct.Struct(VRTA_FRAME_FMT)
VRTA_FRAME_SIZE: int = VRTA_FRAME_STRUCT.size  # 256 bytes exact

assert VRTA_HEADER_SIZE == 128, f"Expected 128 B VrtaFileHeader, got {VRTA_HEADER_SIZE}"
assert VRTA_FRAME_SIZE == 256, f"Expected 256 B VrtaFrame100Hz, got {VRTA_FRAME_SIZE}"

# Reflected Castagnoli CRC-32C polynomial (iSCSI / SSE4.2 / ARMv8 crc32cb)
_CRC32C_POLY: int = 0x82F63B78


def _build_crc32c_table() -> tuple[int, ...]:
    table: list[int] = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ _CRC32C_POLY
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return tuple(table)


_CRC32C_TABLE: tuple[int, ...] = _build_crc32c_table()


def crc32c(data: bytes | bytearray | memoryview, init: int = 0) -> int:
    """Compute Castagnoli CRC-32C (check value: crc32c(b'123456789') == 0xE3069283)."""
    crc = (init ^ 0xFFFFFFFF) & 0xFFFFFFFF
    table = _CRC32C_TABLE
    for b in data:
        crc = table[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (crc ^ 0xFFFFFFFF) & 0xFFFFFFFF


@dataclass(frozen=True)
class VrtaFileHeader:
    """Naturally aligned 128-byte little-endian .vrta file header."""

    magic: bytes = VRTA_MAGIC
    schema_ver: int = VRTA_SCHEMA_VER
    reserved0: int = 0
    datum_lat_deg: float = 40.8860
    datum_lon_deg: float = 29.1880
    datum_amsl_m: float = 12.0
    egm2008_n_m: float = 36.68
    lut_sha256: bytes = b"\x00" * 32
    git_sha: bytes = b"0" * 40
    reserved1: bytes = b"\x00" * 4
    header_crc32c: int = 0


@dataclass(frozen=True)
class VrtaFrame100Hz:
    """Naturally aligned 256-byte little-endian 100 Hz .vrta telemetry frame."""

    seq: int = 0
    t_sim_us: int = 0
    pos_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    q_frd_ned: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    omega_frd: tuple[float, float, float] = (0.0, 0.0, 0.0)
    a_nom_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    a_safe_enu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotor_rpm: tuple[float, float, float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    v_term_v: float = 400.0
    soc_pct: float = 100.0
    cbvf_min_v: float = 0.0
    ltfj_margin_m: float = 1042.0
    ltbx_margin_m: float = 2995.0
    chi2_residual: float = 0.0
    agl_m: float = 0.0
    active_motor_mask: int = 0xFF
    flight_regime: int = 0
    tmr_state: int = 0
    cel_latched: int = 0
    flags: int = 0
    pad: bytes = b"\x00" * 80
    crc32c: int = 0


def _pad_or_check_bytes(val: bytes, size: int, field_name: str) -> bytes:
    if not isinstance(val, (bytes, bytearray)):
        raise TypeError(f"{field_name} must be bytes, got {type(val).__name__}")
    raw = bytes(val)
    if len(raw) > size:
        raise ValueError(f"{field_name} length {len(raw)} exceeds fixed {size} B")
    return raw.ljust(size, b"\x00")


def pack_vrta_header(header: VrtaFileHeader) -> bytes:
    """Serialize VrtaFileHeader to 128 little-endian bytes with fresh Castagnoli CRC32C."""
    magic = _pad_or_check_bytes(header.magic, 8, "magic")
    if magic != VRTA_MAGIC:
        raise ValueError(f"Invalid VRTA magic {magic!r}; expected {VRTA_MAGIC!r}")
    lut_sha = _pad_or_check_bytes(header.lut_sha256, 32, "lut_sha256")
    git_sha = _pad_or_check_bytes(header.git_sha, 40, "git_sha")
    res1 = _pad_or_check_bytes(header.reserved1, 4, "reserved1")

    prefix = struct.pack(
        "<8sII4d32s40s4s",
        magic,
        int(header.schema_ver),
        int(header.reserved0),
        float(header.datum_lat_deg),
        float(header.datum_lon_deg),
        float(header.datum_amsl_m),
        float(header.egm2008_n_m),
        lut_sha,
        git_sha,
        res1,
    )
    checksum = crc32c(prefix)
    return prefix + struct.pack("<I", checksum)


def unpack_vrta_header(raw: bytes | bytearray | memoryview, *, verify_crc: bool = True) -> VrtaFileHeader:
    """Deserialize and validate a 128-byte little-endian VrtaFileHeader."""
    if len(raw) != VRTA_HEADER_SIZE:
        raise ValueError(f"VrtaFileHeader requires exactly {VRTA_HEADER_SIZE} B, got {len(raw)} B")
    fields = VRTA_HEADER_STRUCT.unpack(raw)
    magic = fields[0]
    if magic != VRTA_MAGIC:
        raise ValueError(f"Invalid VRTA file magic {magic!r}; expected {VRTA_MAGIC!r}")
    schema_ver = fields[1]
    if schema_ver != VRTA_SCHEMA_VER:
        raise ValueError(f"Unsupported VRTA schema_ver {schema_ver}; expected {VRTA_SCHEMA_VER}")
    stored_crc = fields[10]
    if verify_crc:
        computed_crc = crc32c(raw[:124])
        if stored_crc != computed_crc:
            raise ValueError(
                f"VrtaFileHeader CRC32C mismatch: stored=0x{stored_crc:08x}, computed=0x{computed_crc:08x}"
            )
    return VrtaFileHeader(
        magic=magic,
        schema_ver=schema_ver,
        reserved0=fields[2],
        datum_lat_deg=fields[3],
        datum_lon_deg=fields[4],
        datum_amsl_m=fields[5],
        egm2008_n_m=fields[6],
        lut_sha256=fields[7],
        git_sha=fields[8],
        reserved1=fields[9],
        header_crc32c=stored_crc,
    )


def pack_vrta_frame(frame: VrtaFrame100Hz) -> bytes:
    """Serialize VrtaFrame100Hz to 256 little-endian bytes with fresh Castagnoli CRC32C."""
    if len(frame.pos_enu) != 3 or len(frame.vel_enu) != 3:
        raise ValueError("pos_enu and vel_enu must have length 3")
    if len(frame.q_frd_ned) != 4:
        raise ValueError("q_frd_ned must have length 4")
    if len(frame.omega_frd) != 3 or len(frame.a_nom_enu) != 3 or len(frame.a_safe_enu) != 3:
        raise ValueError("omega_frd, a_nom_enu, and a_safe_enu must have length 3")
    if len(frame.rotor_rpm) != 8:
        raise ValueError("rotor_rpm must have length 8")
    pad = _pad_or_check_bytes(frame.pad, 80, "pad")

    prefix = struct.pack(
        "<2Q3d3f4f3f3f3f8f7f4BI80s",
        int(frame.seq),
        int(frame.t_sim_us),
        *(float(v) for v in frame.pos_enu),
        *(float(v) for v in frame.vel_enu),
        *(float(v) for v in frame.q_frd_ned),
        *(float(v) for v in frame.omega_frd),
        *(float(v) for v in frame.a_nom_enu),
        *(float(v) for v in frame.a_safe_enu),
        *(float(v) for v in frame.rotor_rpm),
        float(frame.v_term_v),
        float(frame.soc_pct),
        float(frame.cbvf_min_v),
        float(frame.ltfj_margin_m),
        float(frame.ltbx_margin_m),
        float(frame.chi2_residual),
        float(frame.agl_m),
        int(frame.active_motor_mask) & 0xFF,
        int(frame.flight_regime) & 0xFF,
        int(frame.tmr_state) & 0xFF,
        int(frame.cel_latched) & 0xFF,
        int(frame.flags) & 0xFFFFFFFF,
        pad,
    )
    checksum = crc32c(prefix)
    return prefix + struct.pack("<I", checksum)


def unpack_vrta_frame(raw: bytes | bytearray | memoryview, *, verify_crc: bool = True) -> VrtaFrame100Hz:
    """Deserialize and validate a 256-byte little-endian VrtaFrame100Hz."""
    if len(raw) != VRTA_FRAME_SIZE:
        raise ValueError(f"VrtaFrame100Hz requires exactly {VRTA_FRAME_SIZE} B, got {len(raw)} B")
    f = VRTA_FRAME_STRUCT.unpack(raw)
    stored_crc = f[42]
    if verify_crc:
        computed_crc = crc32c(raw[:252])
        if stored_crc != computed_crc:
            raise ValueError(
                f"VrtaFrame100Hz (seq={f[0]}) CRC32C mismatch: "
                f"stored=0x{stored_crc:08x}, computed=0x{computed_crc:08x}"
            )
    return VrtaFrame100Hz(
        seq=f[0],
        t_sim_us=f[1],
        pos_enu=(f[2], f[3], f[4]),
        vel_enu=(f[5], f[6], f[7]),
        q_frd_ned=(f[8], f[9], f[10], f[11]),
        omega_frd=(f[12], f[13], f[14]),
        a_nom_enu=(f[15], f[16], f[17]),
        a_safe_enu=(f[18], f[19], f[20]),
        rotor_rpm=(f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28]),
        v_term_v=f[29],
        soc_pct=f[30],
        cbvf_min_v=f[31],
        ltfj_margin_m=f[32],
        ltbx_margin_m=f[33],
        chi2_residual=f[34],
        agl_m=f[35],
        active_motor_mask=f[36],
        flight_regime=f[37],
        tmr_state=f[38],
        cel_latched=f[39],
        flags=f[40],
        pad=f[41],
        crc32c=stored_crc,
    )


def write_vrta_file(
    path: str | Path,
    header: VrtaFileHeader,
    frames: Sequence[VrtaFrame100Hz],
) -> Path:
    """Write a complete .vrta binary flight log (128 B header + N * 256 B frames)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fp:
        fp.write(pack_vrta_header(header))
        for frame in frames:
            fp.write(pack_vrta_frame(frame))
    return out


def iter_vrta_frames(
    path: str | Path, *, verify_crc: bool = True
) -> tuple[VrtaFileHeader, Iterator[VrtaFrame100Hz]]:
    """Validate file framing and return (header, streaming frame iterator)."""
    src = Path(path)
    total_size = src.stat().st_size
    if total_size < VRTA_HEADER_SIZE:
        raise ValueError(
            f"Truncated .vrta file {src}: {total_size} B < {VRTA_HEADER_SIZE} B header"
        )
    payload_bytes = total_size - VRTA_HEADER_SIZE
    if payload_bytes % VRTA_FRAME_SIZE != 0:
        raise ValueError(
            f"Misaligned/truncated .vrta payload in {src}: {payload_bytes} B is not a "
            f"multiple of {VRTA_FRAME_SIZE} B"
        )
    raw = src.read_bytes()
    header = unpack_vrta_header(raw[:VRTA_HEADER_SIZE], verify_crc=verify_crc)
    n_frames = payload_bytes // VRTA_FRAME_SIZE

    def _gen() -> Iterator[VrtaFrame100Hz]:
        for idx in range(n_frames):
            off = VRTA_HEADER_SIZE + idx * VRTA_FRAME_SIZE
            yield unpack_vrta_frame(raw[off : off + VRTA_FRAME_SIZE], verify_crc=verify_crc)

    return header, _gen()


def read_vrta_file(
    path: str | Path, *, verify_crc: bool = True
) -> tuple[VrtaFileHeader, list[VrtaFrame100Hz]]:
    """Read and CRC32C-verify an entire .vrta binary file."""
    header, frame_iter = iter_vrta_frames(path, verify_crc=verify_crc)
    return header, list(frame_iter)
