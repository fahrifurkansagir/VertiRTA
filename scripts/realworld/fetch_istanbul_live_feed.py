# SPDX-License-Identifier: Apache-2.0
"""Live Real-Time Istanbul LTFJ METAR + Open-Meteo Coastal Wind + Marmara ADS-B Corridor Audit.

Queries real-time public aviation and meteorological APIs (zero API keys, zero
fabricated data, honest per-feed status logging):
  1. Real `LTFJ` (Sabiha Gokcen) & `LTFM` METAR:
     `https://aviationweather.gov/api/data/metar?ids=LTFJ,LTFM&format=json`
  2. Real Kartal-Maltepe Coastal Wind/Gust (`40.892 N, 29.188 E`):
     `https://api.open-meteo.com/v1/forecast?latitude=40.892&longitude=29.188&current=wind_speed_10m,wind_gusts_10m,wind_direction_10m,temperature_2m,surface_pressure`
  3. Real Live Marmara / `LTFJ` / `LTBX` ADS-B Manned Air Traffic:
     Primary `https://api.adsb.lol/v2/lat/40.898/lon/29.309/dist/25`
     Fallback `https://opensky-network.org/api/states/all?lamin=40.75&lomin=28.95&lamax=41.05&lomax=29.45`

Feeds the live measured Istanbul coastal wind/gust and live manned transponders
into a real VertiRTA Kartal-Dragos-Maltepe corridor safety audit verifying:
  - Frozen `LTFJ` NW J2 (`40.912786 N, 29.192122 E`) and official DHMI AIP
    `LTBX` (`40d59'26"N 029d12'58"E` = `40.990556 N, 29.216111 E`) CTR boundaries
  - Minimum horizontal, vertical, and 3D slant separation to live manned traffic
  - Wind-adjusted CBVF speed & energy envelope for `VertiRTA-X8-240`
  - `VertiGnssDenied` CTR inflation check under live CBVF speed cap

Writes SHA-256 sealed `artifacts/v4.0.0/realworld_istanbul_live_feed_audit.json`
(`< 1 MB`, pure UTF-8 LF, zero leaked host paths). Supports `--smoke-cpu` (zero
network, realistic recorded fixture).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence
import urllib.error
import urllib.request

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_DIR = REPO_ROOT / "sdk" / "python"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

from viz.geo.airspace_polygons import (  # noqa: E402
    LTFJ_J2_LAT_DEG,
    LTFJ_J2_LON_DEG,
    LTBX_ARP_LAT_DEG,
    LTBX_ARP_LON_DEG,
    is_inside_ltbx,
    is_inside_ltfj,
    signed_distance_to_ltbx_m,
    signed_distance_to_ltfj_m,
)
from viz.geo.vertirta_geodesy import (  # noqa: E402
    ANCHOR_H_AMSL,
    DEFAULT_ISTANBUL_GEOID_N0_M,
    VertiRTAGeodesyBridge,
)

METAR_URL = "https://aviationweather.gov/api/data/metar?ids=LTFJ,LTFM&format=json"
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=40.892&longitude=29.188"
    "&current=wind_speed_10m,wind_gusts_10m,wind_direction_10m,temperature_2m,surface_pressure"
)
ADSB_LOL_URL = "https://api.adsb.lol/v2/lat/40.898/lon/29.309/dist/25"
OPENSKY_URL = (
    "https://opensky-network.org/api/states/all"
    "?lamin=40.75&lomin=28.95&lamax=41.05&lomax=29.45"
)

AIRSPACE_YAML_PATH = REPO_ROOT / "config/istanbul_ltfj_airspace.yaml"
VEHICLE_YAML_PATH = REPO_ROOT / "config/vehicle_x8_240.yaml"
DEFAULT_OUT_JSON = REPO_ROOT / "artifacts/v4.0.0/realworld_istanbul_live_feed_audit.json"

KT_TO_MPS = 0.514444
KMH_TO_MPS = 1.0 / 3.6
FT_TO_M = 0.3048
CORRIDOR_CRUISE_AGL_M = 60.0
WELL_CLEAR_HORIZ_M = 1000.0
WELL_CLEAR_VERT_M = 150.0
MAX_OPERATIONAL_GUST_MPS = 18.0
CAUTION_GUST_MPS = 12.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _http_get_json(url: str, timeout_s: float = 20.0) -> tuple[Any, int]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "VertiRTA-LiveAudit/4.0 (research-testbed; non-operational)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        status = int(getattr(resp, "status", 200))
        body = resp.read().decode("utf-8")
        return json.loads(body), status


def build_smoke_fixtures() -> dict[str, Any]:
    """Deterministic realistic recorded fixture for zero-network `--smoke-cpu` testing."""
    return {
        "metar": [
            {
                "icaoId": "LTFJ",
                "reportTime": "2026-11-25T10:00:00.000Z",
                "temp": 8,
                "dewp": -3,
                "wdir": 200,
                "wspd": 6,
                "wgst": None,
                "altim": 1029,
                "rawOb": "METAR LTFJ 250950Z 20006KT CAVOK 08/M03 Q1029 NOSIG",
                "fltCat": "VFR",
                "lat": 40.899,
                "lon": 29.309,
            },
            {
                "icaoId": "LTFM",
                "reportTime": "2026-11-25T10:00:00.000Z",
                "temp": 9,
                "dewp": -5,
                "wdir": 210,
                "wspd": 9,
                "wgst": None,
                "altim": 1029,
                "rawOb": "METAR LTFM 250950Z 21009KT 180V240 CAVOK 09/M05 Q1029 NOSIG",
                "fltCat": "VFR",
                "lat": 41.262,
                "lon": 28.740,
            },
        ],
        "open_meteo": {
            "latitude": 40.892,
            "longitude": 29.188,
            "elevation": 36.0,
            "current": {
                "time": "2026-11-25T10:00",
                "interval": 900,
                "wind_speed_10m": 8.4,
                "wind_gusts_10m": 14.8,
                "wind_direction_10m": 200,
                "temperature_2m": 8.6,
                "surface_pressure": 1024.7,
            },
        },
        "adsb": {
            "provider": "smoke_fixture_adsb_lol_format",
            "ac": [
                {
                    "hex": "4ba9a9",
                    "flight": "THY5NP  ",
                    "t": "A321",
                    "r": "TC-JMI",
                    "lat": 40.8772,
                    "lon": 29.0931,
                    "alt_baro": 7375,
                    "alt_geom": 7600,
                    "gs": 242.9,
                    "track": 298.2,
                },
                {
                    "hex": "4bb101",
                    "flight": "PGT214  ",
                    "t": "A20N",
                    "r": "TC-NBK",
                    "lat": 40.8980,
                    "lon": 29.3100,
                    "alt_baro": 1800,
                    "alt_geom": 1950,
                    "gs": 162.0,
                    "track": 240.0,
                },
                {
                    "hex": "4ba222",
                    "flight": "SXS401  ",
                    "t": "B738",
                    "r": "TC-SEY",
                    "lat": 40.8420,
                    "lon": 29.2450,
                    "alt_baro": 3400,
                    "alt_geom": 3600,
                    "gs": 195.0,
                    "track": 58.0,
                },
                {
                    "hex": "4bb888",
                    "flight": "THY91K  ",
                    "t": "A320",
                    "r": "TC-JPF",
                    "lat": 40.8990,
                    "lon": 29.3085,
                    "alt_baro": "ground",
                    "gs": 12.0,
                    "track": 60.0,
                },
            ],
        },
    }


def fetch_live_feeds(timeout_s: float = 20.0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Query live public METAR, Open-Meteo, and ADS-B APIs with honest status tracking."""
    payloads: dict[str, Any] = {"metar": [], "open_meteo": {}, "adsb": {"provider": "none", "ac": []}}
    statuses: dict[str, Any] = {}

    # 1) AviationWeather METAR (LTFJ + LTFM)
    try:
        metar_data, http_code = _http_get_json(METAR_URL, timeout_s=timeout_s)
        payloads["metar"] = metar_data if isinstance(metar_data, list) else []
        statuses["metar"] = {
            "url": METAR_URL,
            "ok": len(payloads["metar"]) > 0,
            "http_status": http_code,
            "records": len(payloads["metar"]),
            "error": "",
        }
    except Exception as exc:
        statuses["metar"] = {"url": METAR_URL, "ok": False, "http_status": 0, "records": 0, "error": str(exc)}

    # 2) Open-Meteo Coastal Wind / Gust
    try:
        om_data, http_code = _http_get_json(OPEN_METEO_URL, timeout_s=timeout_s)
        payloads["open_meteo"] = om_data if isinstance(om_data, dict) else {}
        statuses["open_meteo"] = {
            "url": OPEN_METEO_URL,
            "ok": "current" in payloads["open_meteo"],
            "http_status": http_code,
            "error": "",
        }
    except Exception as exc:
        statuses["open_meteo"] = {"url": OPEN_METEO_URL, "ok": False, "http_status": 0, "error": str(exc)}

    # 3) Live Marmara ADS-B: primary api.adsb.lol, fallback OpenSky Network
    adsb_ok = False
    adsb_err = ""
    try:
        adsb_data, http_code = _http_get_json(ADSB_LOL_URL, timeout_s=timeout_s)
        if isinstance(adsb_data, dict) and isinstance(adsb_data.get("ac"), list):
            payloads["adsb"] = {"provider": "adsb.lol", "ac": adsb_data["ac"]}
            statuses["adsb"] = {
                "provider": "adsb.lol",
                "url": ADSB_LOL_URL,
                "fallback_used": False,
                "ok": True,
                "http_status": http_code,
                "raw_aircraft_count": len(adsb_data["ac"]),
                "error": "",
            }
            adsb_ok = True
    except Exception as exc:
        adsb_err = str(exc)

    if not adsb_ok:
        try:
            osky_data, http_code = _http_get_json(OPENSKY_URL, timeout_s=timeout_s)
            states = osky_data.get("states") if isinstance(osky_data, dict) else None
            ac_list: list[dict[str, Any]] = []
            if isinstance(states, list):
                for st in states:
                    if not isinstance(st, list) or len(st) < 14:
                        continue
                    lon_v, lat_v = st[5], st[6]
                    if lat_v is None or lon_v is None:
                        continue
                    alt_m = st[13] if st[13] is not None else st[7]
                    ac_list.append(
                        {
                            "hex": str(st[0] or "").strip(),
                            "flight": str(st[1] or "").strip(),
                            "lat": float(lat_v),
                            "lon": float(lon_v),
                            "alt_baro": "ground" if bool(st[8]) else (float(alt_m) / FT_TO_M if alt_m is not None else None),
                            "alt_geom": float(alt_m) / FT_TO_M if alt_m is not None else None,
                            "gs": float(st[9]) / KT_TO_MPS if st[9] is not None else None,
                            "track": float(st[10]) if st[10] is not None else None,
                        }
                    )
            payloads["adsb"] = {"provider": "opensky-network.org", "ac": ac_list}
            statuses["adsb"] = {
                "provider": "opensky-network.org",
                "url": OPENSKY_URL,
                "fallback_used": True,
                "primary_error": adsb_err,
                "ok": True,
                "http_status": http_code,
                "raw_aircraft_count": len(ac_list),
                "error": "",
            }
        except Exception as exc:
            statuses["adsb"] = {
                "provider": "none",
                "url": OPENSKY_URL,
                "fallback_used": True,
                "primary_error": adsb_err,
                "ok": False,
                "http_status": 0,
                "raw_aircraft_count": 0,
                "error": str(exc),
            }

    return payloads, statuses


def _parse_float(val: Any, default: float | None = None) -> float | None:
    if val is None or isinstance(val, bool):
        return default
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(float(val)) else default
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "ground", "vrb", "none", "null"):
            return default
        try:
            f = float(s)
            return f if math.isfinite(f) else default
        except ValueError:
            return default
    return default


def _load_native_module() -> Any | None:
    """Load prebuilt C++20 `_vertirta_native.abi3.so` when available."""
    for so_path in (
        REPO_ROOT / "build/cp312-abi3-linux_x86_64/_vertirta_native.abi3.so",
        REPO_ROOT / "build/dev-linux-x64/sdk/python/_vertirta_native.abi3.so",
    ):
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


def _load_gnss_ctr_evaluator() -> tuple[Any, str]:
    """Instantiate `VertiGnssDenied` with native C++20 nanobind when compiled."""
    m3_mod = _import_vertirta_m3()
    suite = m3_mod.VertiGnssDenied()
    if suite.backend == "cpp_nanobind":
        return suite, "cpp_nanobind"

    native_mod = _load_native_module()
    if native_mod is not None and hasattr(native_mod, "NativeGnssDeniedSuite"):
        suite._native = native_mod.NativeGnssDeniedSuite()  # noqa: SLF001
    return suite, suite.backend


def _point_to_polyline_min_dist_m(
    pt: tuple[float, float], polyline: Sequence[tuple[float, float]]
) -> float:
    px, py = pt
    best = float("inf")
    for i in range(len(polyline) - 1):
        ax, ay = polyline[i]
        bx, by = polyline[i + 1]
        vx, vy = bx - ax, by - ay
        seg_sq = vx * vx + vy * vy
        t = 0.0 if seg_sq <= 1e-12 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / seg_sq))
        cx, cy = ax + t * vx, ay + t * vy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


def run_istanbul_corridor_audit(
    feeds: dict[str, Any],
    feed_statuses: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    """Run live Kartal-Maltepe airspace, ADS-B separation, and wind-adjusted CBVF safety audit."""
    t0 = time.perf_counter()
    airspace_cfg = yaml.safe_load(AIRSPACE_YAML_PATH.read_text(encoding="utf-8"))
    vehicle_cfg = yaml.safe_load(VEHICLE_YAML_PATH.read_text(encoding="utf-8"))

    bridge = VertiRTAGeodesyBridge(fixed_geoid_n_m=DEFAULT_ISTANBUL_GEOID_N0_M)

    # 1) Corridor waypoints & sampled centerline check against LTFJ J2 + official DHMI AIP LTBX
    slz_priors: list[dict[str, Any]] = airspace_cfg["slz_priors"]
    odd = airspace_cfg["odd_bounds"]
    corridor_alt_amsl_m = float(ANCHOR_H_AMSL + CORRIDOR_CRUISE_AGL_M)  # 72.0 m AMSL

    waypoint_checks: list[dict[str, Any]] = []
    wp_enu: list[tuple[float, float]] = []
    for slz in slz_priors:
        lat_d = float(slz["lat_deg"])
        lon_d = float(slz["lon_deg"])
        e_m, n_m, _ = bridge.cesium_wgs84_to_flat_enu(lat_d, lon_d, bridge.h0)
        wp_enu.append((e_m, n_m))
        d_ltfj = float(signed_distance_to_ltfj_m(lat_d, lon_d))
        d_ltbx = float(signed_distance_to_ltbx_m(lat_d, lon_d))
        waypoint_checks.append(
            {
                "name": str(slz["name"]),
                "lat_deg": round(lat_d, 6),
                "lon_deg": round(lon_d, 6),
                "west_of_j2_limit": lon_d <= float(odd["max_lon_near_j2_deg"]),
                "ltfj_clearance_m": round(d_ltfj, 2),
                "ltbx_clearance_m": round(d_ltbx, 2),
                "outside_both_ctrs": (d_ltfj > 0.0 and d_ltbx > 0.0),
            }
        )

    # Sample 25 centerline points between Kartal -> Dragos -> Maltepe
    sample_ltfj: list[float] = []
    sample_ltbx: list[float] = []
    max_corridor_lon = -180.0
    corridor_length_m = 0.0
    for seg_i in range(len(slz_priors) - 1):
        a, b = slz_priors[seg_i], slz_priors[seg_i + 1]
        ae, an = wp_enu[seg_i]
        be, bn = wp_enu[seg_i + 1]
        corridor_length_m += math.hypot(be - ae, bn - an)
        for k in range(13):
            u = k / 12.0
            lat_s = float(a["lat_deg"]) * (1.0 - u) + float(b["lat_deg"]) * u
            lon_s = float(a["lon_deg"]) * (1.0 - u) + float(b["lon_deg"]) * u
            max_corridor_lon = max(max_corridor_lon, lon_s)
            sample_ltfj.append(float(signed_distance_to_ltfj_m(lat_s, lon_s)))
            sample_ltbx.append(float(signed_distance_to_ltbx_m(lat_s, lon_s)))

    min_corridor_ltfj_m = min(sample_ltfj)
    min_corridor_ltbx_m = min(sample_ltbx)

    # Corridor heading bearing (Kartal -> Maltepe, degrees true clockwise from North)
    d_e_leg = wp_enu[-1][0] - wp_enu[0][0]
    d_n_leg = wp_enu[-1][1] - wp_enu[0][1]
    corridor_bearing_deg = (math.degrees(math.atan2(d_e_leg, d_n_leg)) + 360.0) % 360.0

    # Verify official DHMI AIP LTBX ARP = 40d59'26"N 029d12'58"E = 40.990556 N, 29.216111 E
    yaml_ltbx_lat = float(airspace_cfg["airspace"]["ltbx_ctr"]["arp_lat_deg"])
    yaml_ltbx_lon = float(airspace_cfg["airspace"]["ltbx_ctr"]["arp_lon_deg"])
    ltbx_arp_parity_m = math.hypot(
        (yaml_ltbx_lat - LTBX_ARP_LAT_DEG) * 111_052.5,
        (yaml_ltbx_lon - LTBX_ARP_LON_DEG) * 84_152.0,
    )

    # 2) Parse live METAR (LTFJ + LTFM) & Open-Meteo coastal weather
    metar_by_icao: dict[str, dict[str, Any]] = {}
    for rec in feeds.get("metar", []):
        if not isinstance(rec, dict):
            continue
        icao = str(rec.get("icaoId", "")).upper().strip()
        if not icao:
            continue
        wspd_kt = _parse_float(rec.get("wspd"), 0.0) or 0.0
        wgst_kt = _parse_float(rec.get("wgst"), None)
        metar_by_icao[icao] = {
            "icao": icao,
            "raw_metar": str(rec.get("rawOb", "")),
            "report_time_utc": str(rec.get("reportTime", "")),
            "flight_category": str(rec.get("fltCat", "")),
            "wind_dir_deg": _parse_float(rec.get("wdir"), None),
            "wind_speed_kt": round(wspd_kt, 2),
            "wind_speed_mps": round(wspd_kt * KT_TO_MPS, 3),
            "wind_gust_kt": round(wgst_kt, 2) if wgst_kt is not None else None,
            "wind_gust_mps": round(wgst_kt * KT_TO_MPS, 3) if wgst_kt is not None else None,
            "temp_c": _parse_float(rec.get("temp"), None),
            "dewpoint_c": _parse_float(rec.get("dewp"), None),
            "qnh_hpa": _parse_float(rec.get("altim"), None),
        }

    om_cur = feeds.get("open_meteo", {}).get("current", {}) if isinstance(feeds.get("open_meteo"), dict) else {}
    om_wind_kmh = _parse_float(om_cur.get("wind_speed_10m"), 0.0) or 0.0
    om_gust_kmh = _parse_float(om_cur.get("wind_gusts_10m"), om_wind_kmh) or om_wind_kmh
    om_wdir_deg = _parse_float(om_cur.get("wind_direction_10m"), 0.0) or 0.0
    om_temp_c = _parse_float(om_cur.get("temperature_2m"), 15.0)
    if om_temp_c is None:
        om_temp_c = 15.0
    om_press_hpa = _parse_float(om_cur.get("surface_pressure"), 1013.25)
    if om_press_hpa is None:
        om_press_hpa = 1013.25

    om_wind_mps = om_wind_kmh * KMH_TO_MPS
    om_gust_mps = om_gust_kmh * KMH_TO_MPS

    ltfj_metar = metar_by_icao.get("LTFJ", {})
    ltfj_wspd_mps = float(ltfj_metar.get("wind_speed_mps", 0.0) or 0.0)
    ltfj_wgst_mps = float(ltfj_metar.get("wind_gust_mps") or ltfj_wspd_mps)

    # Conservative governing coastal wind/gust = max(Open-Meteo Kartal, LTFJ METAR)
    eff_wind_mps = max(om_wind_mps, ltfj_wspd_mps)
    eff_gust_mps = max(om_gust_mps, ltfj_wgst_mps, eff_wind_mps)
    eff_wdir_deg = om_wdir_deg if om_cur else float(ltfj_metar.get("wind_dir_deg") or 0.0)

    # 3) Wind-adjusted CBVF speed & X8-240 energy envelope
    rho_kg_m3 = (om_press_hpa * 100.0) / (287.05 * (om_temp_c + 273.15))
    mass_kg = float(vehicle_cfg["mass_kg"])
    g_ms2 = float(vehicle_cfg["gravity_m_s2"])
    weight_n = mass_kg * g_ms2
    a_disk_m2 = float(vehicle_cfg["geometry"]["disk_area_4proj_m2"])
    f_drag_m2 = float(vehicle_cfg["propulsion"]["flat_plate_drag_area_m2"])
    eta_coax = float(vehicle_cfg["propulsion"]["eta_coax_hover"])
    eta_drive = float(vehicle_cfg["propulsion"]["drivetrain_efficiency"])
    kappa_int = float(vehicle_cfg["propulsion"]["kappa_int_hover"])
    p_avionics_w = float(vehicle_cfg["battery"]["p_avionics_w"])
    e_pack_kwh = float(vehicle_cfg["battery"]["energy_kwh"])
    soc_reserve_pct = float(vehicle_cfg["battery"]["soc_reserve_min"]) * 100.0
    v_max_mps = float(vehicle_cfg["limits"]["v_max_mps"])

    # Wind direction is meteorological FROM direction; headwind along corridor bearing (and return)
    rel_rad = math.radians(eff_wdir_deg - corridor_bearing_deg)
    headwind_nw_mps = eff_wind_mps * math.cos(rel_rad)
    headwind_gust_worst_mps = abs(eff_gust_mps * math.cos(rel_rad))
    crosswind_gust_mps = abs(eff_gust_mps * math.sin(rel_rad))

    # CBVF gust-adjusted speed cap: reserve 0.5 * V_gust control authority margin below V_max
    v_cbvf_air_cap_mps = max(10.0, min(v_max_mps, v_max_mps - 0.5 * eff_gust_mps))
    v_cruise_nom_mps = min(15.0, v_cbvf_air_cap_mps)
    v_ground_worst_mps = max(4.0, v_cruise_nom_mps - headwind_gust_worst_mps)

    # Coaxial momentum + profile + parasite power at cruise airspeed under live air density
    v_h = math.sqrt(weight_n / (2.0 * rho_kg_m3 * a_disk_m2))
    v_air = v_cruise_nom_mps
    # Glauert forward-flight induced velocity approximation: v_i = v_h^2 / sqrt(v_air^2 + v_h^2)
    v_i = (v_h * v_h) / math.sqrt(v_air * v_air + v_h * v_h)
    p_induced_w = kappa_int * weight_n * v_i / eta_coax
    p_parasite_w = 0.5 * rho_kg_m3 * f_drag_m2 * (v_air**3)
    p_profile_w = 0.12 * (weight_n * v_h)  # ~12% profile power fraction for X8-240 3-blade rotors
    p_elec_w = (p_induced_w + p_parasite_w + p_profile_w) / eta_drive + p_avionics_w
    p_elec_kw = p_elec_w / 1000.0

    one_way_time_s = corridor_length_m / v_ground_worst_mps
    one_way_energy_kwh = p_elec_kw * (one_way_time_s / 3600.0)
    one_way_soc_pct = (one_way_energy_kwh / e_pack_kwh) * 100.0
    roundtrip_soc_pct = 2.0 * one_way_soc_pct
    soc_remaining_after_one_way_pct = 80.0 - one_way_soc_pct
    soc_margin_above_reserve_pct = soc_remaining_after_one_way_pct - soc_reserve_pct

    if eff_gust_mps > MAX_OPERATIONAL_GUST_MPS:
        wind_verdict = "NO_GO_WIND_EXCEEDANCE"
    elif eff_gust_mps > CAUTION_GUST_MPS:
        wind_verdict = "CAUTION_GUST_CAPPED"
    else:
        wind_verdict = "GO_NOMINAL"

    # Run VertiGnssDenied CTR inflation audit with live CBVF speed cap
    gnss_suite, gnss_backend = _load_gnss_ctr_evaluator()
    gnss_suite.reset(2.0)
    ctr_eval = gnss_suite.propagate_and_evaluate_ctr(
        dt_s=10.0,
        horiz_speed_mps=v_cruise_nom_mps,
        own_pos_ned=(0.0, -500.0, -CORRIDOR_CRUISE_AGL_M),
        alt_amsl_m=corridor_alt_amsl_m,
    )

    # 4) Live ADS-B Manned Air Traffic Separation Audit
    raw_ac = feeds.get("adsb", {}).get("ac", [])
    evaluated_ac: list[dict[str, Any]] = []
    airborne_count = 0
    ground_count = 0
    inside_ltfj_count = 0
    inside_ltbx_count = 0
    well_clear_violations: list[dict[str, Any]] = []

    for ac in raw_ac:
        if not isinstance(ac, dict):
            continue
        lat_v = _parse_float(ac.get("lat"), None)
        lon_v = _parse_float(ac.get("lon"), None)
        if lat_v is None or lon_v is None:
            continue

        raw_alt_baro = ac.get("alt_baro")
        is_ground = isinstance(raw_alt_baro, str) and raw_alt_baro.strip().lower() == "ground"
        alt_geom_ft = _parse_float(ac.get("alt_geom"), None)
        alt_baro_ft = None if is_ground else _parse_float(raw_alt_baro, None)
        alt_ft = alt_geom_ft if alt_geom_ft is not None else alt_baro_ft
        alt_amsl_m = 95.1 if is_ground else (alt_ft * FT_TO_M if alt_ft is not None else None)

        if is_ground:
            ground_count += 1
        else:
            airborne_count += 1

        e_ac, n_ac, _ = bridge.cesium_wgs84_to_flat_enu(lat_v, lon_v, bridge.h0)
        horiz_to_corridor_m = _point_to_polyline_min_dist_m((e_ac, n_ac), wp_enu)
        vert_sep_m = abs(alt_amsl_m - corridor_alt_amsl_m) if alt_amsl_m is not None else None
        slant_3d_m = (
            math.hypot(horiz_to_corridor_m, vert_sep_m) if vert_sep_m is not None else horiz_to_corridor_m
        )
        in_ltfj = bool(is_inside_ltfj(lat_v, lon_v, alt_amsl_m))
        in_ltbx = bool(is_inside_ltbx(lat_v, lon_v, alt_amsl_m))
        if in_ltfj:
            inside_ltfj_count += 1
        if in_ltbx:
            inside_ltbx_count += 1

        gs_kt = _parse_float(ac.get("gs"), None)
        well_clear_ok = True
        if (not is_ground) and (horiz_to_corridor_m < WELL_CLEAR_HORIZ_M):
            if vert_sep_m is None or vert_sep_m < WELL_CLEAR_VERT_M:
                well_clear_ok = False

        entry = {
            "hex": str(ac.get("hex", "")).strip(),
            "callsign": str(ac.get("flight", "")).strip() or "UNKNOWN",
            "aircraft_type": str(ac.get("t", "")).strip() or "UNKNOWN",
            "on_ground": is_ground,
            "lat_deg": round(lat_v, 5),
            "lon_deg": round(lon_v, 5),
            "alt_ft": round(alt_ft, 1) if alt_ft is not None else None,
            "alt_amsl_m": round(alt_amsl_m, 1) if alt_amsl_m is not None else None,
            "groundspeed_kt": round(gs_kt, 1) if gs_kt is not None else None,
            "horiz_dist_to_corridor_m": round(horiz_to_corridor_m, 1),
            "vert_sep_to_corridor_m": round(vert_sep_m, 1) if vert_sep_m is not None else None,
            "slant_3d_dist_m": round(slant_3d_m, 1),
            "inside_ltfj_ctr": in_ltfj,
            "inside_ltbx_ctr": in_ltbx,
            "well_clear_satisfied": well_clear_ok,
        }
        evaluated_ac.append(entry)
        if not well_clear_ok:
            well_clear_violations.append(entry)

    evaluated_ac.sort(key=lambda x: x["horiz_dist_to_corridor_m"])
    airborne_ac = [a for a in evaluated_ac if not a["on_ground"]]
    closest_any = evaluated_ac[0] if evaluated_ac else None
    closest_airborne = airborne_ac[0] if airborne_ac else None
    min_horiz_airborne_m = closest_airborne["horiz_dist_to_corridor_m"] if closest_airborne else None
    vert_seps = [a["vert_sep_to_corridor_m"] for a in airborne_ac if a["vert_sep_to_corridor_m"] is not None]
    min_vert_airborne_m = min(vert_seps) if vert_seps else None

    elapsed_s = round(time.perf_counter() - t0, 4)
    report: dict[str, Any] = {
        "schema_version": "vertirta.realworld_istanbul_live_feed_audit/v1.0.0",
        "mode": mode,
        "disclaimer": "RESEARCH TESTBED AUDIT ONLY — NOT FOR OPERATIONAL AERONAUTICAL NAVIGATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "colab_cu_consumed": 0.0,
        "host_platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "feed_statuses": feed_statuses,
        "live_weather": {
            "metar": metar_by_icao,
            "open_meteo_kartal_coastal": {
                "query_lat_deg": 40.892,
                "query_lon_deg": 29.188,
                "observation_time_gmt": str(om_cur.get("time", "")),
                "wind_speed_10m_kmh": round(om_wind_kmh, 2),
                "wind_speed_10m_mps": round(om_wind_mps, 3),
                "wind_gusts_10m_kmh": round(om_gust_kmh, 2),
                "wind_gusts_10m_mps": round(om_gust_mps, 3),
                "wind_direction_10m_deg": round(om_wdir_deg, 1),
                "temperature_2m_c": round(om_temp_c, 2),
                "surface_pressure_hpa": round(om_press_hpa, 2),
                "air_density_kg_m3": round(rho_kg_m3, 4),
            },
            "governing_wind": {
                "steady_wind_mps": round(eff_wind_mps, 3),
                "gust_wind_mps": round(eff_gust_mps, 3),
                "wind_from_deg": round(eff_wdir_deg, 1),
                "corridor_bearing_deg": round(corridor_bearing_deg, 2),
                "headwind_nw_steady_mps": round(headwind_nw_mps, 3),
                "headwind_gust_worst_mps": round(headwind_gust_worst_mps, 3),
                "crosswind_gust_mps": round(crosswind_gust_mps, 3),
            },
        },
        "airspace_boundary_audit": {
            "corridor_name": str(airspace_cfg["corridor_name"]),
            "airac_cycle": str(airspace_cfg["airac_cycle"]),
            "corridor_length_m": round(corridor_length_m, 1),
            "corridor_cruise_agl_m": CORRIDOR_CRUISE_AGL_M,
            "corridor_cruise_amsl_m": corridor_alt_amsl_m,
            "ltfj_j2_pinch_lat_lon": [round(LTFJ_J2_LAT_DEG, 6), round(LTFJ_J2_LON_DEG, 6)],
            "ltbx_official_dhmi_arp_lat_lon": [round(LTBX_ARP_LAT_DEG, 6), round(LTBX_ARP_LON_DEG, 6)],
            "ltbx_yaml_to_code_arp_parity_m": round(ltbx_arp_parity_m, 4),
            "odd_max_lon_near_j2_deg": float(odd["max_lon_near_j2_deg"]),
            "corridor_max_lon_deg": round(max_corridor_lon, 6),
            "min_corridor_ltfj_clearance_m": round(min_corridor_ltfj_m, 2),
            "min_corridor_ltbx_clearance_m": round(min_corridor_ltbx_m, 2),
            "all_waypoints_clear_of_ltfj_and_ltbx": all(w["outside_both_ctrs"] for w in waypoint_checks),
            "waypoints": waypoint_checks,
            "gnss_ctr_inflation_audit": {
                "backend": gnss_backend,
                "nav_state": str(ctr_eval["nav_state"]),
                "three_sigma_pos_m": round(float(ctr_eval["three_sigma_pos_m"]), 3),
                "ltfj_inflated_margin_m": round(float(ctr_eval["ltfj_inflated_margin_m"]), 2),
                "ltbx_inflated_margin_m": round(float(ctr_eval["ltbx_inflated_margin_m"]), 2),
                "v_max_cap_mps": round(float(ctr_eval["v_max_cap_mps"]), 2),
            },
        },
        "live_adsb_traffic_audit": {
            "provider": str(feeds.get("adsb", {}).get("provider", "none")),
            "surveillance_center_lat_lon": [40.898, 29.309],
            "surveillance_radius_nm": 25.0,
            "total_aircraft_with_position": len(evaluated_ac),
            "airborne_aircraft_count": airborne_count,
            "ground_aircraft_count": ground_count,
            "aircraft_inside_ltfj_ctr": inside_ltfj_count,
            "aircraft_inside_ltbx_ctr": inside_ltbx_count,
            "well_clear_thresholds_m": {
                "horizontal_m": WELL_CLEAR_HORIZ_M,
                "vertical_m": WELL_CLEAR_VERT_M,
            },
            "well_clear_violation_count": len(well_clear_violations),
            "min_airborne_horiz_sep_m": min_horiz_airborne_m,
            "min_airborne_vert_sep_m": min_vert_airborne_m,
            "closest_aircraft": closest_any,
            "closest_airborne_aircraft": closest_airborne,
            "top_10_closest_aircraft": evaluated_ac[:10],
        },
        "cbvf_wind_energy_envelope": {
            "vehicle_model": str(vehicle_cfg["model_name"]),
            "mass_kg": mass_kg,
            "nominal_v_max_mps": v_max_mps,
            "cbvf_gust_capped_airspeed_mps": round(v_cbvf_air_cap_mps, 3),
            "cruise_airspeed_mps": round(v_cruise_nom_mps, 3),
            "worst_case_groundspeed_mps": round(v_ground_worst_mps, 3),
            "cruise_electrical_power_kw": round(p_elec_kw, 3),
            "one_way_leg_time_s": round(one_way_time_s, 1),
            "one_way_energy_kwh": round(one_way_energy_kwh, 3),
            "one_way_soc_consumption_pct": round(one_way_soc_pct, 2),
            "roundtrip_soc_consumption_pct": round(roundtrip_soc_pct, 2),
            "soc_remaining_from_80pct_after_one_way_pct": round(soc_remaining_after_one_way_pct, 2),
            "soc_reserve_floor_pct": soc_reserve_pct,
            "soc_margin_above_reserve_pct": round(soc_margin_above_reserve_pct, 2),
            "wind_envelope_verdict": wind_verdict,
        },
        "safety_gates": {
            "ltfj_j2_boundary_clear": min_corridor_ltfj_m > 0.0 and max_corridor_lon <= float(odd["max_lon_near_j2_deg"]),
            "ltbx_official_dhmi_arp_verified": ltbx_arp_parity_m < 1.0 and min_corridor_ltbx_m > 0.0,
            "adsb_well_clear_zero_violations": len(well_clear_violations) == 0,
            "cbvf_soc_reserve_positive": soc_margin_above_reserve_pct > 0.0,
            "weather_and_traffic_feeds_ingested": bool(
                feed_statuses.get("metar", {}).get("ok")
                and feed_statuses.get("open_meteo", {}).get("ok")
                and feed_statuses.get("adsb", {}).get("ok")
            ),
        },
        "elapsed_s": elapsed_s,
    }
    report["all_gates_passed"] = all(report["safety_gates"].values())
    canonical_digest = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    report["report_sha256"] = _sha256_bytes(canonical_digest)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch live Istanbul LTFJ METAR, Open-Meteo wind, and Marmara ADS-B traffic for VertiRTA corridor audit."
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument(
        "--smoke-cpu",
        action="store_true",
        help="Zero-network smoke mode using realistic recorded METAR + Open-Meteo + ADS-B fixtures.",
    )
    args = parser.parse_args(argv)

    if args.smoke_cpu:
        feeds = build_smoke_fixtures()
        feed_statuses = {
            "metar": {"url": "fixture://metar", "ok": True, "http_status": 200, "records": len(feeds["metar"]), "error": ""},
            "open_meteo": {"url": "fixture://open_meteo", "ok": True, "http_status": 200, "error": ""},
            "adsb": {
                "provider": feeds["adsb"]["provider"],
                "url": "fixture://adsb",
                "fallback_used": False,
                "ok": True,
                "http_status": 200,
                "raw_aircraft_count": len(feeds["adsb"]["ac"]),
                "error": "",
            },
        }
        mode = "SMOKE_CPU_ONLY"
    else:
        feeds, feed_statuses = fetch_live_feeds(timeout_s=float(args.timeout_s))
        mode = "LIVE_REALWORLD_ISTANBUL_FEED"

    report = run_istanbul_corridor_audit(feeds, feed_statuses, mode=mode)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    ltfj_raw = report["live_weather"]["metar"].get("LTFJ", {}).get("raw_metar", "")
    closest = report["live_adsb_traffic_audit"]["closest_airborne_aircraft"]
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "timestamp_utc": report["timestamp_utc"],
                "ltfj_raw_metar": ltfj_raw,
                "kartal_wind_mps": report["live_weather"]["open_meteo_kartal_coastal"]["wind_speed_10m_mps"],
                "kartal_gust_mps": report["live_weather"]["open_meteo_kartal_coastal"]["wind_gusts_10m_mps"],
                "adsb_provider": report["live_adsb_traffic_audit"]["provider"],
                "adsb_total_aircraft": report["live_adsb_traffic_audit"]["total_aircraft_with_position"],
                "adsb_airborne_count": report["live_adsb_traffic_audit"]["airborne_aircraft_count"],
                "closest_airborne": (
                    {
                        "callsign": closest["callsign"],
                        "type": closest["aircraft_type"],
                        "horiz_m": closest["horiz_dist_to_corridor_m"],
                        "alt_ft": closest["alt_ft"],
                    }
                    if closest
                    else None
                ),
                "min_corridor_ltfj_m": report["airspace_boundary_audit"]["min_corridor_ltfj_clearance_m"],
                "min_corridor_ltbx_m": report["airspace_boundary_audit"]["min_corridor_ltbx_clearance_m"],
                "wind_verdict": report["cbvf_wind_energy_envelope"]["wind_envelope_verdict"],
                "all_gates_passed": report["all_gates_passed"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
        )
    )
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
