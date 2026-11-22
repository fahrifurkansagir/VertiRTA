"""Exact DHMI eAIP Turkey LTFJ & LTBX CTR Polygon+Arc Airspace Geometries.

SIMULATION REFERENCE ONLY — NOT FOR OPERATIONAL NAVIGATION.

Encodes the official interlocking Control Zones bounding the Maltepe-Dragos-
Kartal UAM corridor per DR-4 / DR-5 / Plan 1B Task 6:
- Sabiha Gokcen CTR (LTFJ): ARP/CTR center 40d53'54"N 029d18'33"E
  (40.89833 N, 29.30917 E), 5 NM (9,260 m) arc joined at the NW straight chord
  J1 (40d59'15"N 029d19'30"E) -> J2 (40d54'46.03"N 029d11'31.64"E =
  40.91279 N, 29.19212 E). Vertical limits: SFC - 2,500 ft AMSL (762.0 m) ->
  798.58 m WGS84 HAE (N_LTFJ = 36.58 m / 120 ft).
- Samandira CTR (LTBX): CTR center 40d59'56"N 029d12'58"E (40.99889 N,
  29.21611 E; ARP 40d59'26"N 029d12'58"E), 5 NM (9,260 m) arc sharing the SE
  straight chord J2 -> J1 with LTFJ. Vertical limits: SFC - 2,000 ft AMSL
  (609.6 m) -> 646.35 m WGS84 HAE (N_LTBX = 36.75 m), plus 1,500 ft (457.2 m)
  Istanbul TMA floor marker.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from viz.geo.vertirta_geodesy import (
    ANCHOR_H_AMSL,
    ANCHOR_LAT_DEG,
    ANCHOR_LON_DEG,
    DEFAULT_ISTANBUL_GEOID_N0_M,
    VertiRTAGeodesyBridge,
)

AIRSPACE_DISCLAIMER: str = (
    "SIMULATION REFERENCE ONLY — NOT FOR OPERATIONAL NAVIGATION "
    "(Derived from DHMI eAIP Turkey LT_AD_2_LTFJ / LT_AD_2_LTBX)"
)

NM_TO_METERS: float = 1852.0
FT_TO_METERS: float = 0.3048
CTR_RADIUS_NM: float = 5.0
CTR_RADIUS_M: float = CTR_RADIUS_NM * NM_TO_METERS  # 9260.0 m

# Sabiha Gokcen CTR (LTFJ): 40d53'54"N, 029d18'33"E
LTFJ_ARP_LAT_DEG: float = 40.0 + 53.0 / 60.0 + 54.0 / 3600.0
LTFJ_ARP_LON_DEG: float = 29.0 + 18.0 / 60.0 + 33.0 / 3600.0
LTFJ_CEILING_FT_AMSL: float = 2500.0
LTFJ_CEILING_AMSL_M: float = LTFJ_CEILING_FT_AMSL * FT_TO_METERS  # 762.0 m
LTFJ_GEOID_N_M: float = 36.58  # Official DHMI eAIP LTFJ geoid undulation (120 ft)
LTFJ_CEILING_HAE_M: float = LTFJ_CEILING_AMSL_M + LTFJ_GEOID_N_M  # 798.58 m

# Shared LTFJ / LTBX straight NW/SE join line:
# J1 (NE corner): 40d59'15"N, 029d19'30"E
LTFJ_J1_LAT_DEG: float = 40.0 + 59.0 / 60.0 + 15.0 / 3600.0
LTFJ_J1_LON_DEG: float = 29.0 + 19.0 / 60.0 + 30.0 / 3600.0
# J2 (SW/NW corner): 40d54'46.03"N, 029d11'31.64"E -> 40.91279 N, 29.19212 E
LTFJ_J2_LAT_DEG: float = 40.0 + 54.0 / 60.0 + 46.03 / 3600.0
LTFJ_J2_LON_DEG: float = 29.0 + 11.0 / 60.0 + 31.64 / 3600.0

# Samandira CTR (LTBX): Center 40d59'56"N, 029d12'58"E; ARP 40d59'26"N, 029d12'58"E
LTBX_CTR_LAT_DEG: float = 40.0 + 59.0 / 60.0 + 56.0 / 3600.0
LTBX_CTR_LON_DEG: float = 29.0 + 12.0 / 60.0 + 58.0 / 3600.0
LTBX_ARP_LAT_DEG: float = 40.0 + 59.0 / 60.0 + 26.0 / 3600.0
LTBX_ARP_LON_DEG: float = 29.0 + 12.0 / 60.0 + 58.0 / 3600.0
LTBX_CEILING_FT_AMSL: float = 2000.0
LTBX_CEILING_AMSL_M: float = LTBX_CEILING_FT_AMSL * FT_TO_METERS  # 609.6 m
LTBX_TMA_FLOOR_AMSL_M: float = 1500.0 * FT_TO_METERS  # 457.2 m
LTBX_GEOID_N_M: float = 36.75
LTBX_CEILING_HAE_M: float = LTBX_CEILING_AMSL_M + LTBX_GEOID_N_M  # 646.35 m

_DEFAULT_BRIDGE = VertiRTAGeodesyBridge(fixed_geoid_n_m=DEFAULT_ISTANBUL_GEOID_N0_M)
_DEFAULT_CACHE: dict[str, Any] | None = None


def _ll_to_enu_2d(
    lat_deg: float, lon_deg: float, bridge: VertiRTAGeodesyBridge
) -> tuple[float, float]:
    e_m, n_m, _ = bridge.cesium_wgs84_to_flat_enu(lat_deg, lon_deg, bridge.h0)
    return (e_m, n_m)


def _point_in_ring(point: tuple[float, float], ring: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test for a closed 2D ring."""
    px, py = point
    inside = False
    for i in range(len(ring) - 1):
        x1, y1 = ring[i]
        x2, y2 = ring[i + 1]
        if (y1 > py) != (y2 > py):
            x_intersect = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < x_intersect:
                inside = not inside
    return inside


def _signed_distance_to_ring(
    point: tuple[float, float], ring: Sequence[tuple[float, float]]
) -> float:
    """Signed Euclidean distance (m): positive outside, negative strictly inside."""
    px, py = point
    min_dist = float("inf")
    for i in range(len(ring) - 1):
        ax, ay = ring[i]
        bx, by = ring[i + 1]
        vx, vy = bx - ax, by - ay
        seg_len_sq = vx * vx + vy * vy
        if seg_len_sq == 0.0:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / seg_len_sq))
        cx, cy = ax + t * vx, ay + t * vy
        dist = math.hypot(px - cx, py - cy)
        if dist < min_dist:
            min_dist = dist
    return -min_dist if _point_in_ring(point, ring) else min_dist


def _circle_line_intersections(
    center: tuple[float, float],
    radius: float,
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Intersect circle (center, radius) with infinite line through p0 -> p1."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    fx, fy = p0[0] - center[0], p0[1] - center[1]
    a_coef = dx * dx + dy * dy
    b_coef = 2.0 * (fx * dx + fy * dy)
    c_coef = fx * fx + fy * fy - radius * radius
    disc = max(0.0, b_coef * b_coef - 4.0 * a_coef * c_coef)
    sqrt_disc = math.sqrt(disc)
    t1 = (-b_coef - sqrt_disc) / (2.0 * a_coef)
    t2 = (-b_coef + sqrt_disc) / (2.0 * a_coef)
    return (
        (p0[0] + t1 * dx, p0[1] + t1 * dy),
        (p0[0] + t2 * dx, p0[1] + t2 * dy),
    )


def _build_ltfj_ring_enu(
    center_enu: tuple[float, float],
    j1_enu: tuple[float, float],
    j2_enu: tuple[float, float],
    arc_segments: int,
) -> list[tuple[float, float]]:
    """Construct closed CCW LTFJ polygon+arc ring on the SE side of chord J1-J2.

    Vertex sequence: J1 -> J2 -> radial join to 5 NM arc at azimuth(J2) ->
    CCW 5 NM arc around South/East side of LTFJ -> radial join at azimuth(J1) -> J1.
    Preserves the exact NW J2 corner (40.91279 N, 29.19212 E) pulling the western
    CTR wall ~603 m west of 29.200 E while remaining strictly simple and disjoint
    from LTBX along J1-J2.
    """
    a1 = math.atan2(j1_enu[1] - center_enu[1], j1_enu[0] - center_enu[0])
    a2 = math.atan2(j2_enu[1] - center_enu[1], j2_enu[0] - center_enu[0])
    end_a = a1 + 2.0 * math.pi if a1 <= a2 else a1

    ring: list[tuple[float, float]] = [j1_enu, j2_enu]
    for i in range(arc_segments + 1):
        ang = a2 + (end_a - a2) * (i / arc_segments)
        ring.append(
            (
                center_enu[0] + CTR_RADIUS_M * math.cos(ang),
                center_enu[1] + CTR_RADIUS_M * math.sin(ang),
            )
        )
    ring.append(j1_enu)
    return ring


def _build_ltbx_ring_enu(
    center_enu: tuple[float, float],
    j1_enu: tuple[float, float],
    j2_enu: tuple[float, float],
    arc_segments: int,
) -> list[tuple[float, float]]:
    """Construct closed CCW LTBX polygon+arc ring strictly on the NW side of J1-J2.

    Because J1 lies ~10.8 m inside the rounded 5 NM circle from LTBX center, the
    arc starts at the exact NE circle-line intersection (~15.1 m past J1 along
    J2->J1) so the closed polygon has zero self-intersection and zero overlap
    across the shared J1-J2 border with LTFJ.
    """
    p_ne, _ = _circle_line_intersections(center_enu, CTR_RADIUS_M, j1_enu, j2_enu)
    start_a = math.atan2(p_ne[1] - center_enu[1], p_ne[0] - center_enu[0])
    end_a = math.atan2(j2_enu[1] - center_enu[1], j2_enu[0] - center_enu[0])
    if end_a <= start_a:
        end_a += 2.0 * math.pi

    ring: list[tuple[float, float]] = [j2_enu, j1_enu]
    for i in range(arc_segments + 1):
        ang = start_a + (end_a - start_a) * (i / arc_segments)
        ring.append(
            (
                center_enu[0] + CTR_RADIUS_M * math.cos(ang),
                center_enu[1] + CTR_RADIUS_M * math.sin(ang),
            )
        )
    ring.append(j2_enu)
    return ring


def _enu_ring_to_geodetic_rings(
    ring_enu: Sequence[tuple[float, float]],
    ceiling_hae_m: float,
    bridge: VertiRTAGeodesyBridge,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    ring_lon_lat: list[tuple[float, float]] = []
    ring_lat_lon_hae: list[tuple[float, float, float]] = []
    for e_m, n_m in ring_enu:
        p = bridge.flat_enu_to_cesium_wgs84(e_m, n_m, 0.0)
        lat_d, lon_d = p["lat_deg"], p["lon_deg"]
        ring_lon_lat.append((lon_d, lat_d))
        ring_lat_lon_hae.append((lat_d, lon_d, ceiling_hae_m))
    # Guarantee bit-exact GeoJSON closure
    if ring_lon_lat:
        ring_lon_lat[-1] = ring_lon_lat[0]
        ring_lat_lon_hae[-1] = ring_lat_lon_hae[0]
    return ring_lon_lat, ring_lat_lon_hae


def build_ltfj_ltbx_geometries(
    arc_segments: int = 360,
    bridge: VertiRTAGeodesyBridge | None = None,
) -> dict[str, Any]:
    """Build exact DHMI LTFJ and LTBX polygon+arc CTR geometries and GeoJSON."""
    if arc_segments < 16:
        raise ValueError(f"arc_segments must be >= 16, got {arc_segments}")
    active_bridge = bridge if bridge is not None else _DEFAULT_BRIDGE

    ltfj_center_enu = _ll_to_enu_2d(LTFJ_ARP_LAT_DEG, LTFJ_ARP_LON_DEG, active_bridge)
    ltbx_center_enu = _ll_to_enu_2d(LTBX_CTR_LAT_DEG, LTBX_CTR_LON_DEG, active_bridge)
    j1_enu = _ll_to_enu_2d(LTFJ_J1_LAT_DEG, LTFJ_J1_LON_DEG, active_bridge)
    j2_enu = _ll_to_enu_2d(LTFJ_J2_LAT_DEG, LTFJ_J2_LON_DEG, active_bridge)

    ltfj_ring_enu = _build_ltfj_ring_enu(ltfj_center_enu, j1_enu, j2_enu, arc_segments)
    ltbx_ring_enu = _build_ltbx_ring_enu(ltbx_center_enu, j1_enu, j2_enu, arc_segments)

    ltfj_lon_lat, ltfj_lat_lon_hae = _enu_ring_to_geodetic_rings(
        ltfj_ring_enu, LTFJ_CEILING_HAE_M, active_bridge
    )
    ltbx_lon_lat, ltbx_lat_lon_hae = _enu_ring_to_geodetic_rings(
        ltbx_ring_enu, LTBX_CEILING_HAE_M, active_bridge
    )

    ltfj_bands_amsl = [500.0 * k * FT_TO_METERS for k in range(1, 6)]
    ltbx_bands_amsl = [500.0 * k * FT_TO_METERS for k in range(1, 5)]

    ltfj_feature = {
        "type": "Feature",
        "properties": {
            "icao": "LTFJ",
            "name": "Sabiha Gokcen CTR",
            "disclaimer": AIRSPACE_DISCLAIMER,
            "floor_amsl_m": 0.0,
            "ceiling_ft_amsl": LTFJ_CEILING_FT_AMSL,
            "ceiling_amsl_m": LTFJ_CEILING_AMSL_M,
            "geoid_n_m": LTFJ_GEOID_N_M,
            "ceiling_hae_m": LTFJ_CEILING_HAE_M,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lon, lat in ltfj_lon_lat]],
        },
    }
    ltbx_feature = {
        "type": "Feature",
        "properties": {
            "icao": "LTBX",
            "name": "Samandira CTR",
            "disclaimer": AIRSPACE_DISCLAIMER,
            "floor_amsl_m": 0.0,
            "ceiling_ft_amsl": LTBX_CEILING_FT_AMSL,
            "ceiling_amsl_m": LTBX_CEILING_AMSL_M,
            "tma_floor_amsl_m": LTBX_TMA_FLOOR_AMSL_M,
            "geoid_n_m": LTBX_GEOID_N_M,
            "ceiling_hae_m": LTBX_CEILING_HAE_M,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[lon, lat] for lon, lat in ltbx_lon_lat]],
        },
    }

    return {
        "datum": {
            "name": "Kartal Coastal Vertiport",
            "lat_deg": ANCHOR_LAT_DEG,
            "lon_deg": ANCHOR_LON_DEG,
            "alt_amsl_m": ANCHOR_H_AMSL,
            "egm2008_n_m": active_bridge.N0,
        },
        "shared_join": {
            "j1_lat_lon": (LTFJ_J1_LAT_DEG, LTFJ_J1_LON_DEG),
            "j2_lat_lon": (LTFJ_J2_LAT_DEG, LTFJ_J2_LON_DEG),
            "j1_enu_m": j1_enu,
            "j2_enu_m": j2_enu,
        },
        "ltfj": {
            "icao": "LTFJ",
            "name": "Sabiha Gokcen CTR",
            "disclaimer": AIRSPACE_DISCLAIMER,
            "center_lat_lon": (LTFJ_ARP_LAT_DEG, LTFJ_ARP_LON_DEG),
            "center_enu_m": ltfj_center_enu,
            "radius_nm": CTR_RADIUS_NM,
            "radius_m": CTR_RADIUS_M,
            "floor_amsl_m": 0.0,
            "ceiling_ft_amsl": LTFJ_CEILING_FT_AMSL,
            "ceiling_amsl_m": LTFJ_CEILING_AMSL_M,
            "geoid_n_m": LTFJ_GEOID_N_M,
            "ceiling_hae_m": LTFJ_CEILING_HAE_M,
            "altitude_bands_amsl_m": ltfj_bands_amsl,
            "altitude_bands_hae_m": [z + LTFJ_GEOID_N_M for z in ltfj_bands_amsl],
            "nw_join_j1_lat_lon": (LTFJ_J1_LAT_DEG, LTFJ_J1_LON_DEG),
            "nw_join_j2_lat_lon": (LTFJ_J2_LAT_DEG, LTFJ_J2_LON_DEG),
            "ring_enu_m": ltfj_ring_enu,
            "ring_lon_lat": ltfj_lon_lat,
            "ring_lat_lon_hae": ltfj_lat_lon_hae,
            "geojson_feature": ltfj_feature,
        },
        "ltbx": {
            "icao": "LTBX",
            "name": "Samandira CTR",
            "disclaimer": AIRSPACE_DISCLAIMER,
            "center_lat_lon": (LTBX_CTR_LAT_DEG, LTBX_CTR_LON_DEG),
            "arp_lat_lon": (LTBX_ARP_LAT_DEG, LTBX_ARP_LON_DEG),
            "center_enu_m": ltbx_center_enu,
            "radius_nm": CTR_RADIUS_NM,
            "radius_m": CTR_RADIUS_M,
            "floor_amsl_m": 0.0,
            "ceiling_ft_amsl": LTBX_CEILING_FT_AMSL,
            "ceiling_amsl_m": LTBX_CEILING_AMSL_M,
            "tma_floor_amsl_m": LTBX_TMA_FLOOR_AMSL_M,
            "geoid_n_m": LTBX_GEOID_N_M,
            "ceiling_hae_m": LTBX_CEILING_HAE_M,
            "altitude_bands_amsl_m": ltbx_bands_amsl,
            "altitude_bands_hae_m": [z + LTBX_GEOID_N_M for z in ltbx_bands_amsl],
            "sw_join_j2_lat_lon": (LTFJ_J2_LAT_DEG, LTFJ_J2_LON_DEG),
            "ne_join_j1_lat_lon": (LTFJ_J1_LAT_DEG, LTFJ_J1_LON_DEG),
            "ring_enu_m": ltbx_ring_enu,
            "ring_lon_lat": ltbx_lon_lat,
            "ring_lat_lon_hae": ltbx_lat_lon_hae,
            "geojson_feature": ltbx_feature,
        },
        "geojson_feature_collection": {
            "type": "FeatureCollection",
            "features": [ltfj_feature, ltbx_feature],
        },
    }


def _get_default_geometries() -> dict[str, Any]:
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = build_ltfj_ltbx_geometries()
    return _DEFAULT_CACHE


def signed_distance_to_ltfj_m(lat_deg: float, lon_deg: float) -> float:
    """Horizontal signed distance (m) to exact LTFJ CTR polygon+arc.

    Positive values indicate clearance outside LTFJ CTR; negative values
    indicate penetration strictly inside LTFJ CTR.
    """
    geom = _get_default_geometries()
    pt_enu = _ll_to_enu_2d(lat_deg, lon_deg, _DEFAULT_BRIDGE)
    return _signed_distance_to_ring(pt_enu, geom["ltfj"]["ring_enu_m"])


def signed_distance_to_ltbx_m(lat_deg: float, lon_deg: float) -> float:
    """Horizontal signed distance (m) to exact LTBX CTR polygon+arc.

    Positive values indicate clearance outside LTBX CTR; negative values
    indicate penetration strictly inside LTBX CTR.
    """
    geom = _get_default_geometries()
    pt_enu = _ll_to_enu_2d(lat_deg, lon_deg, _DEFAULT_BRIDGE)
    return _signed_distance_to_ring(pt_enu, geom["ltbx"]["ring_enu_m"])


def is_inside_ltfj(
    lat_deg: float, lon_deg: float, alt_amsl_m: float | None = None
) -> bool:
    """Return True if (lat, lon[, alt_amsl_m]) violates LTFJ CTR (SFC-2,500 ft)."""
    if alt_amsl_m is not None and float(alt_amsl_m) > LTFJ_CEILING_AMSL_M:
        return False
    return signed_distance_to_ltfj_m(lat_deg, lon_deg) < 0.0


def is_inside_ltbx(
    lat_deg: float, lon_deg: float, alt_amsl_m: float | None = None
) -> bool:
    """Return True if (lat, lon[, alt_amsl_m]) violates LTBX CTR (SFC-2,000 ft)."""
    if alt_amsl_m is not None and float(alt_amsl_m) > LTBX_CEILING_AMSL_M:
        return False
    return signed_distance_to_ltbx_m(lat_deg, lon_deg) < 0.0


def build_enu_curtain_mesh_data(
    zone: str,
    *,
    base_z_enu_m: float = -ANCHOR_H_AMSL,
    arc_segments: int = 360,
) -> dict[str, Any]:
    """Generate 3D ENU curtain vertices & quad indices for Blender/Gazebo CTR walls.

    In local Kartal ENU, sea level (0.0 m AMSL) sits at ``Z_enu = -12.0 m`` and
    CTR ceilings sit at ``ceiling_amsl_m - 12.0 m``.
    """
    key = zone.strip().lower()
    if key not in {"ltfj", "ltbx"}:
        raise ValueError(f"Unsupported CTR zone {zone!r}; expected 'ltfj' or 'ltbx'")

    geom = build_ltfj_ltbx_geometries(arc_segments=arc_segments)
    spec = geom[key]
    ring_open = spec["ring_enu_m"][:-1]
    top_z_enu_m = float(spec["ceiling_amsl_m"]) - ANCHOR_H_AMSL
    n_pts = len(ring_open)

    vertices: list[tuple[float, float, float]] = []
    for e_m, n_m in ring_open:
        vertices.append((e_m, n_m, float(base_z_enu_m)))
    for e_m, n_m in ring_open:
        vertices.append((e_m, n_m, top_z_enu_m))

    quad_faces: list[tuple[int, int, int, int]] = []
    for i in range(n_pts):
        j = (i + 1) % n_pts
        quad_faces.append((i, j, n_pts + j, n_pts + i))

    return {
        "icao": spec["icao"],
        "base_z_enu_m": float(base_z_enu_m),
        "top_z_enu_m": top_z_enu_m,
        "altitude_bands_enu_m": [z - ANCHOR_H_AMSL for z in spec["altitude_bands_amsl_m"]],
        "vertices_enu_m": vertices,
        "quad_faces": quad_faces,
    }


def export_airspace_geojson(
    out_path: str | Path, arc_segments: int = 360
) -> dict[str, Any]:
    """Write the LTFJ/LTBX GeoJSON FeatureCollection to disk (UTF-8, no BOM)."""
    geom = build_ltfj_ltbx_geometries(arc_segments=arc_segments)
    fc = geom["geojson_feature_collection"]
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(fc, indent=2) + "\n", encoding="utf-8", newline="\n")
    return fc
