"""Strict TDD test suite for Plan 1B Task 7 (viz/cesium/ live Earth console :5180).

Pins all DR-5 / Plan 1A Task 5 / Plan 1B Task 7 locks:
- Exact little-endian POD binary layout of VrtaFileHeader (128 B, magic="VRTA0001")
  and VrtaFrame100Hz (256 B) matching Plan 1A Task 5 byte-for-byte with Castagnoli
  CRC32C verification (check vector "123456789" -> 0xE3069283).
- WebSocket schema vrta.ws.v1.2 using clean footprint_lat_lon_hae ([lat, lon, hae],
  NEVER swapped [lon, lat]!), port :5180 (rejecting forbidden 4173/4174/8765/9222/9223),
  and Gaussian touchdown surface blend (AGL <= 15 m).
- GTX 1650 SUPER (< 1.8 GB VRAM) CesiumJS locks in viz/cesium/app.js & index.html:
  cacheBytes=256MB, overflow=128MB, maximumScreenSpaceError=24, msaaSamples=2,
  shadows=false, requestRenderMode=true + viewer.scene.requestRender() per WS tick,
  depthTestAgainstTerrain=true, tileCacheSize=64, sampleHeightMostDetailed,
  swappable basemap with ZERO hardcoded keyless arcgisonline.com URLs, and #cesium-credits.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from viz.cesium.vrta_reader import (
    VRTA_FRAME_FMT,
    VRTA_FRAME_SIZE,
    VRTA_HEADER_FMT,
    VRTA_HEADER_SIZE,
    VRTA_MAGIC,
    VrtaFileHeader,
    VrtaFrame100Hz,
    crc32c,
    pack_vrta_frame,
    pack_vrta_header,
    read_vrta_file,
    unpack_vrta_frame,
    unpack_vrta_header,
    write_vrta_file,
)
from viz.cesium.ws_server import (
    DEFAULT_CESIUM_PORT,
    FORBIDDEN_PORTS,
    WS_SCHEMA_VER,
    compute_touchdown_blend_hae,
    encode_ws_telemetry_frame,
    validate_port,
    vrta_frame_to_ws_packet,
)


def _sample_header() -> VrtaFileHeader:
    return VrtaFileHeader(
        magic=VRTA_MAGIC,
        schema_ver=1,
        reserved0=0,
        datum_lat_deg=40.8860,
        datum_lon_deg=29.1880,
        datum_amsl_m=12.0,
        egm2008_n_m=36.68,
        lut_sha256=bytes(range(32)),
        git_sha=b"a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
        reserved1=b"\x00\x00\x00\x00",
    )


def _sample_frame(seq: int = 100, active_mask: int = 0b11111011) -> VrtaFrame100Hz:
    return VrtaFrame100Hz(
        seq=seq,
        t_sim_us=seq * 10_000,
        pos_enu=(-3540.0, 2940.0, 120.0),
        vel_enu=(-12.5, 6.2, -1.5),
        q_frd_ned=(0.945380, 0.098027, 0.121511, 0.286149),
        omega_frd=(0.01, -0.02, 0.05),
        a_nom_enu=(0.2, -0.1, 0.0),
        a_safe_enu=(0.6, -0.4, 0.0),
        rotor_rpm=(2480.0, 2510.0, 0.0, 2940.0, 2610.0, 2590.0, 2495.0, 2505.0),
        v_term_v=382.5,
        soc_pct=64.2,
        cbvf_min_v=1.85,
        ltfj_margin_m=3885.8,
        ltbx_margin_m=2004.9,
        chi2_residual=4.12,
        agl_m=45.0,
        active_motor_mask=active_mask,
        flight_regime=1,
        tmr_state=0,
        cel_latched=0,
        flags=0x00000001,
    )


def test_vrta_binary_128b_256b_layout_and_crc32c(tmp_path: Path) -> None:
    """Plan 1A Task 5 byte-for-byte POD layout & Castagnoli CRC32C verification."""
    # Standard Castagnoli CRC-32C check value (RFC 3720 / iSCSI)
    assert crc32c(b"123456789") == 0xE3069283
    assert VRTA_HEADER_SIZE == 128 and struct.calcsize(VRTA_HEADER_FMT) == 128
    assert VRTA_FRAME_SIZE == 256 and struct.calcsize(VRTA_FRAME_FMT) == 256

    hdr = _sample_header()
    raw_hdr = pack_vrta_header(hdr)
    assert len(raw_hdr) == 128
    assert raw_hdr[0:8] == b"VRTA0001"
    assert struct.unpack_from("<II", raw_hdr, 8) == (1, 0)
    assert struct.unpack_from("<4d", raw_hdr, 16) == pytest.approx((40.8860, 29.1880, 12.0, 36.68))
    assert struct.unpack_from("<I", raw_hdr, 124)[0] == crc32c(raw_hdr[:124])
    hdr_rt = unpack_vrta_header(raw_hdr)
    assert hdr_rt.datum_lat_deg == 40.8860 and hdr_rt.egm2008_n_m == 36.68
    assert hdr_rt.lut_sha256 == hdr.lut_sha256 and hdr_rt.git_sha == hdr.git_sha

    frm = _sample_frame()
    raw_frm = pack_vrta_frame(frm)
    assert len(raw_frm) == 256
    assert struct.unpack_from("<2Q", raw_frm, 0) == (100, 1_000_000)
    assert struct.unpack_from("<3d", raw_frm, 16) == pytest.approx((-3540.0, 2940.0, 120.0))
    assert struct.unpack_from("<4BI", raw_frm, 164) == (0b11111011, 1, 0, 0, 1)
    assert struct.unpack_from("<I", raw_frm, 252)[0] == crc32c(raw_frm[:252])
    frm_rt = unpack_vrta_frame(raw_frm)
    assert frm_rt.seq == 100 and frm_rt.active_motor_mask == 0b11111011
    assert frm_rt.rotor_rpm[2] == 0.0 and frm_rt.v_term_v == pytest.approx(382.5, abs=1e-4)

    # Full file round-trip + corruption & truncation rejection
    vrta_file = tmp_path / "flight.vrta"
    write_vrta_file(vrta_file, hdr, [frm, _sample_frame(seq=101)])
    assert vrta_file.stat().st_size == 128 + 2 * 256
    loaded_hdr, loaded_frames = read_vrta_file(vrta_file)
    assert loaded_hdr.magic == VRTA_MAGIC and len(loaded_frames) == 2 and loaded_frames[1].seq == 101

    bad_hdr = bytearray(raw_hdr)
    bad_hdr[20] ^= 0x01
    with pytest.raises(ValueError, match="CRC32C"):
        unpack_vrta_header(bytes(bad_hdr))

    bad_frm = bytearray(raw_frm)
    bad_frm[100] ^= 0xFF
    with pytest.raises(ValueError, match="CRC32C"):
        unpack_vrta_frame(bytes(bad_frm))

    trunc_file = tmp_path / "trunc.vrta"
    trunc_file.write_bytes(vrta_file.read_bytes()[:-5])
    with pytest.raises(ValueError, match="aligned|truncated|size"):
        read_vrta_file(trunc_file)


def test_ws_schema_uses_clean_lat_lon_hae_touchdown_blend_and_port_lock() -> None:
    """Plan 1B Task 7: vrta.ws.v1.2 footprint_lat_lon_hae, Gaussian blend, port :5180."""
    assert DEFAULT_CESIUM_PORT == 5180
    assert validate_port(5180) == 5180
    for forbidden in (4173, 4174, 8765, 9222, 9223):
        assert forbidden in FORBIDDEN_PORTS
        with pytest.raises(ValueError, match="forbidden|port"):
            validate_port(forbidden)

    pkt = encode_ws_telemetry_frame(
        seq=10,
        sim_time_s=1.0,
        lat_deg=40.886,
        lon_deg=29.188,
        alt_amsl_m=12.0,
        geoid_n_m=36.68,
        agl_m=0.0,
        q_enu_flu=(1.0, 0.0, 0.0, 0.0),
        active_motor_mask=251,
        cbvf_ring_lat_lon_hae=[(40.886, 29.188, 48.68)],
    )
    assert pkt["schema_ver"] == WS_SCHEMA_VER == "vrta.ws.v1.2"
    assert "footprint_lat_lon_hae" in pkt["cbvf_overlay"]
    assert "footprint_llh_hae" not in pkt["cbvf_overlay"]
    lat, lon, hae = pkt["cbvf_overlay"]["footprint_lat_lon_hae"][0]
    assert 40.0 < lat < 42.0 and 28.0 < lon < 30.0 and hae > 45.0
    assert pkt["propulsion"]["failed_motor_indices"] == [2]
    assert pkt["propulsion"]["smoke_active"] is True

    # Gaussian touchdown blend: exact contact at AGL=0, vanishes by AGL=60 m
    assert compute_touchdown_blend_hae(48.68, 2.4, 0.0) == pytest.approx(51.08, abs=1e-9)
    assert abs(compute_touchdown_blend_hae(100.0, 2.4, 60.0) - 100.0) < 1e-4

    # Direct conversion from binary VrtaFrame100Hz using viz.geo geodesy bridge
    ws_from_bin = vrta_frame_to_ws_packet(_sample_frame(), _sample_header())
    assert ws_from_bin["schema_ver"] == "vrta.ws.v1.2"
    assert 40.85 < ws_from_bin["geodetic"]["lat_deg"] < 40.96
    assert 29.10 < ws_from_bin["geodetic"]["lon_deg"] < 29.20
    assert ws_from_bin["geodetic"]["alt_hae_wgs84_m"] == pytest.approx(12.0 + 120.0 + 36.68, abs=0.2)
    ring = ws_from_bin["cbvf_overlay"]["footprint_lat_lon_hae"]
    assert len(ring) == 24 and all(40.0 < p[0] < 42.0 and 28.0 < p[1] < 30.0 for p in ring)


def test_cesium_app_js_and_html_enforce_gtx1650_vram_and_swappable_basemap() -> None:
    """Review Focus #5 & VRAM/ToS locks on viz/cesium/{app.js,index.html,README.md}."""
    js = Path("viz/cesium/app.js").read_text(encoding="utf-8")
    html = Path("viz/cesium/index.html").read_text(encoding="utf-8")
    readme = Path("viz/cesium/README.md").read_text(encoding="utf-8")

    # GTX 1650 SUPER (< 1.8 GB VRAM) + active requestRenderMode wakeup + Gaussian touchdown blend
    for token in (
        "256 * 1024 * 1024",
        "128 * 1024 * 1024",
        "maximumScreenSpaceError = 24",
        "msaaSamples: 2",
        "shadows: false",
        "requestRenderMode: true",
        "maximumRenderTimeChange: Infinity",
        "viewer.scene.requestRender()",
        "depthTestAgainstTerrain = true",
        "tileCacheSize = 64",
        "sampleHeightMostDetailed",
        "footprint_lat_lon_hae",
        "basemapProvider",
        "cesium-credits",
        "5180",
        "15.0",
    ):
        assert token in js, f"Missing mandatory lock token in app.js: {token}"

    # Zero hardcoded keyless arcgisonline.com streaming endpoint or forbidden ports
    assert "services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer" not in js
    assert "8765" not in js and "4173" not in js

    assert 'id="cesium-credits"' in html and 'id="cesiumContainer"' in html
    assert "app.js" in html
    assert "5180" in readme and "GTX 1650" in readme
