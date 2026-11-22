"""Strict TDD test suite for VertiRTA Plan 1B Task 6 (viz/geo/).

Pins all DR-4 and DR-5 forensic errata locks:
- Kartal Coastal Vertiport datum (40.8860 N, 29.1880 E, 12.0 m AMSL) and
  Istanbul EGM2008 geoid undulation guard (20.0 < N0 < 50.0 m, ~36.7 m),
  hard-rejecting silent 0.0 m PROJ offline fallback.
- Curvature-compensated FlatEnuToCesium (R_gauss = 6375036.971 m, < 9 mm
  error across the 15x15 km Kartal corridor, eliminating +8.82 m corner float).
- Exact PX4 NED/FRD <-> Gazebo/Blender ENU/FLU quaternion round-trip (< 1e-12).
- Exact DHMI LTFJ (5 NM + NW polygon/arc join J2) and LTBX CTR geometry
  (ceilings 798.58 m / 646.35 m HAE, SLZ-07 ~1.635 km to J2, Pendik Marina
  strictly inside LTFJ).
- ToS-clean Kartal local tmerc gdalwarp builder (-dstnodata -9999, NEVER
  EPSG:32635 or -dstnodata 0), AWS sentinel-cogs 35TPF + GLO-30 N40_00_E029_00,
  commercial license firewall rejecting EOX/FABDEM/Bing/Esri/Google/ion, and
  viz/geo/data/.gitignore ODbL isolation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import viz.geo as viz_geo
from viz.geo.airspace_polygons import (
    build_enu_curtain_mesh_data,
    build_ltfj_ltbx_geometries,
    export_airspace_geojson,
    is_inside_ltbx,
    is_inside_ltfj,
    signed_distance_to_ltbx_m,
    signed_distance_to_ltfj_m,
)
from viz.geo.build_istanbul_corridor import (
    COPERNICUS_GLO30_S3_URL,
    COPERNICUS_GLO30_TILE,
    EARTH_SEARCH_STAC_URL,
    GEOFABRIK_TURKEY_PBF_URL,
    KARTAL_LOCAL_TMERC_PROJ,
    OVERTURE_STAC_URL,
    SENTINEL2_COGS_S3_PREFIX,
    SENTINEL2_MGRS_TILE,
    build_corridor_pipeline_plan,
    build_gdalwarp_cmd,
    build_overture_duckdb_sql,
    validate_asset_url_tos,
)
from viz.geo.vertirta_geodesy import (
    ANCHOR_H_AMSL,
    ANCHOR_LAT_DEG,
    ANCHOR_LON_DEG,
    R_GAUSS_KARTAL,
    VertiRTAGeodesyBridge,
    assert_valid_istanbul_geoid,
    ecef_to_geodetic,
    enu_flu_quat_to_ecef_rotmat,
    enu_flu_to_px4_quat,
    geodetic_to_ecef,
    px4_quat_to_enu_flu,
    quat_to_rotmat,
)


def _matmul3(a: tuple[tuple[float, ...], ...], b: tuple[tuple[float, ...], ...]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def test_geodesy_rejects_missing_egm2008_grid() -> None:
    """DR-5 Review Focus #1: reject silent 0.0 m PROJ EGM2008 grid fallback."""
    assert assert_valid_istanbul_geoid(36.68) == pytest.approx(36.68)

    for bad_n in (0.0, -0.0, -36.7, 19.99, 20.0, 50.0, 50.01, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RuntimeError, match="EGM2008"):
            assert_valid_istanbul_geoid(bad_n)

    with pytest.raises(RuntimeError, match="EGM2008"):
        VertiRTAGeodesyBridge(fixed_geoid_n_m=0.0)

    # Simulated silent PROJ offline fallback returning 0.0 m must hard-fail
    with pytest.raises(RuntimeError, match="EGM2008"):
        VertiRTAGeodesyBridge(geoid_provider=lambda _lon, _lat: 0.0)

    # Default offline-deterministic bridge resolves Kartal datum & ~36.7 m EGM2008
    bridge = VertiRTAGeodesyBridge()
    assert (ANCHOR_LAT_DEG, ANCHOR_LON_DEG, ANCHOR_H_AMSL) == (40.8860, 29.1880, 12.0)
    assert R_GAUSS_KARTAL == pytest.approx(6375036.971, abs=1e-3)
    assert 20.0 < bridge.N0 < 50.0
    assert bridge.N0 == pytest.approx(36.68, abs=0.15)
    assert bridge.h0 == pytest.approx(48.68, abs=0.15)


def test_curvature_compensated_enu_eliminates_8p82m_corner_float() -> None:
    """DR-5: curvature-compensated FlatEnuToCesium stays < 9 mm over 15x15 km."""
    bridge = VertiRTAGeodesyBridge(fixed_geoid_n_m=36.68)

    # Canonical 15x15 km corner check from Plan 1B Task 6
    corner = bridge.flat_enu_to_cesium_wgs84(7500.0, 7500.0, 0.0)
    assert math.isclose(corner["alt_hae_m"], 48.68, abs_tol=0.02)
    assert 8.80 < corner["curvature_drop_m"] < 8.85
    assert corner["u_tan_m"] == pytest.approx(-corner["curvature_drop_m"], abs=1e-9)
    assert abs(corner["tangent_residual_m"]) < 0.009

    # Dragos (~4.60 km -> +1.66 m) and Maltepe (~6.75 km -> +3.56 m) curvature drops
    dragos = bridge.flat_enu_to_cesium_wgs84(-3540.0, 2940.0, 0.0)
    maltepe = bridge.flat_enu_to_cesium_wgs84(-5390.0, 4050.0, 0.0)
    assert 1.60 < dragos["curvature_drop_m"] < 1.72
    assert 3.50 < maltepe["curvature_drop_m"] < 3.63

    # Full 15x15 km grid check: < 9 mm tangent altitude residual AND round-trip accuracy
    for e_m in (-7500.0, -3540.0, 0.0, 3540.0, 7500.0):
        for n_m in (-7500.0, -3540.0, 0.0, 3540.0, 7500.0):
            for z_flat in (0.0, 60.0, 250.0):
                res = bridge.flat_enu_to_cesium_wgs84(e_m, n_m, z_flat)
                assert abs(res["tangent_residual_m"]) < 0.009
                assert res["alt_amsl_m"] == pytest.approx(12.0 + z_flat, abs=1e-9)
                assert res["alt_hae_m"] == pytest.approx(12.0 + z_flat + 36.68, abs=1e-9)

                e_rt, n_rt, z_rt = bridge.cesium_wgs84_to_flat_enu(
                    res["lat_deg"], res["lon_deg"], res["alt_hae_m"]
                )
                assert abs(e_rt - e_m) < 0.009
                assert abs(n_rt - n_m) < 0.009
                assert abs(z_rt - z_flat) < 0.009


def test_px4_quat_to_enu_flu_exact_round_trip() -> None:
    """Exact Hamilton [w, x, y, z] PX4 NED/FRD <-> ENU/FLU transform (< 1e-12)."""
    s = math.sqrt(0.5)

    # Level North in PX4 NED/FRD (1, 0, 0, 0) -> +90 deg yaw in ENU/FLU
    assert px4_quat_to_enu_flu(1.0, 0.0, 0.0, 0.0) == pytest.approx((s, 0.0, 0.0, s), abs=1e-12)
    # Level East in PX4 NED/FRD (s, 0, 0, s) -> identity in ENU/FLU
    assert px4_quat_to_enu_flu(s, 0.0, 0.0, s) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-12)

    # Plan 1A Task 2 heading-035 reference quaternion
    q_ref = (0.945380, 0.098027, 0.121511, 0.286149)
    nrm = math.sqrt(sum(c * c for c in q_ref))
    q_ref_u = tuple(c / nrm for c in q_ref)
    q_enu = VertiRTAGeodesyBridge.px4_quat_to_enu_flu(*q_ref_u)
    assert q_enu[0] == pytest.approx(0.870823, abs=1e-5)
    assert q_enu[1] == pytest.approx(0.155237, abs=1e-5)

    r_enu_ned = ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0))
    r_frd_flu = ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0))

    raw_samples = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        q_ref_u,
        (0.5, -0.5, 0.5, -0.5),
        (0.71, -0.23, 0.41, -0.52),
        (-0.33, 0.64, -0.19, 0.67),
    ]
    for raw in raw_samples:
        norm = math.sqrt(sum(c * c for c in raw))
        q_px4 = tuple(c / norm for c in raw)
        q_e = px4_quat_to_enu_flu(*q_px4)
        q_rt = enu_flu_to_px4_quat(*q_e)

        # Component-wise exact round-trip < 1e-12
        for a_c, b_c in zip(q_px4, q_rt):
            assert abs(a_c - b_c) < 1e-12

        # Rotation matrix equivalence R(q_enu_flu) == R_enu_ned @ R(q_px4) @ R_frd_flu
        r_from_q = quat_to_rotmat(q_e)
        r_expected = _matmul3(_matmul3(r_enu_ned, quat_to_rotmat(q_px4)), r_frd_flu)
        for i in range(3):
            for j in range(3):
                assert abs(r_from_q[i][j] - r_expected[i][j]) < 1e-12


def test_ltfj_ltbx_polygon_arc_geometry_and_pinch_distances() -> None:
    """DR-4 & DR-5: exact LTFJ/LTBX polygon+arc CTRs, J2 pinch, and Pendik Marina inside."""
    geom = build_ltfj_ltbx_geometries()

    # Exact HAE ceilings & LTFJ ARP/J2 metadata
    assert geom["ltfj"]["ceiling_hae_m"] == pytest.approx(798.58, abs=0.2)
    assert geom["ltbx"]["ceiling_hae_m"] == pytest.approx(646.35, abs=0.2)
    assert geom["ltfj"]["center_lat_lon"] == pytest.approx((40.89833, 29.30917), abs=1e-4)
    assert geom["ltfj"]["radius_m"] == pytest.approx(9260.0, abs=1e-6)
    assert geom["ltfj"]["nw_join_j2_lat_lon"] == pytest.approx((40.91279, 29.19212), abs=1e-4)
    assert geom["ltbx"]["center_lat_lon"] == pytest.approx((40.99889, 29.21611), abs=1e-4)

    # DR-4 Erratum #4: SLZ-07 Kartal City Hospital is ~1.635 km from LTFJ NW J2 (NOT 2.34 km!)
    assert signed_distance_to_ltfj_m(40.91546, 29.17299) == pytest.approx(1635.0, abs=40.0)
    assert signed_distance_to_ltbx_m(40.91546, 29.17299) == pytest.approx(699.0, abs=30.0)
    assert is_inside_ltfj(40.91546, 29.17299) is False
    assert is_inside_ltbx(40.91546, 29.17299) is False

    # Pendik Marina & Kurtkoy strictly inside LTFJ CTR (negative signed distance)
    d_pendik = signed_distance_to_ltfj_m(40.86778, 29.23972)
    assert d_pendik < -2000.0
    assert is_inside_ltfj(40.86778, 29.23972) is True
    assert signed_distance_to_ltfj_m(40.91406, 29.29922) < -5000.0

    # J2 NW pinch pulls CTR wall west of 29.200 E: (40.9128, 29.1930) MUST be inside LTFJ!
    assert signed_distance_to_ltfj_m(40.9128, 29.1930) < 0.0
    assert is_inside_ltfj(40.9128, 29.1930) is True

    # Kartal Coastal Vertiport datum is outside both CTRs (~1.04 km W of LTFJ, ~3.00 km S of LTBX)
    assert signed_distance_to_ltfj_m(40.8860, 29.1880) == pytest.approx(1042.0, abs=30.0)
    assert signed_distance_to_ltbx_m(40.8860, 29.1880) == pytest.approx(2995.0, abs=30.0)

    # LTBX ARP strictly inside LTBX CTR
    assert signed_distance_to_ltbx_m(40.99056, 29.21611) < -5000.0
    assert is_inside_ltbx(40.99056, 29.21611) is True

    # GeoJSON FeatureCollection & ENU curtain sanity
    fc = geom["geojson_feature_collection"]
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 2
    for feat in fc["features"]:
        ring = feat["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1] and len(ring) > 100
        assert "SIMULATION REFERENCE ONLY" in feat["properties"]["disclaimer"]

    curtain = build_enu_curtain_mesh_data("ltfj")
    assert curtain["top_z_enu_m"] == pytest.approx(762.0 - 12.0, abs=1e-6)
    assert len(curtain["vertices_enu_m"]) >= 200 and len(curtain["quad_faces"]) >= 100


def test_corridor_builder_local_tmerc_nodata_tos_firewall_and_gitignore() -> None:
    """DR-5 Erratum #1 & Legal Firewall: local tmerc, -dstnodata -9999, AWS open assets, ODbL .gitignore."""
    cmd_list = build_gdalwarp_cmd("in.tif", "out.tif")
    cmd = " ".join(cmd_list)
    assert "+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84" in cmd
    assert "EPSG:32635" not in cmd
    assert "-dstnodata -9999" in cmd
    assert "-dstnodata 0" not in cmd
    assert "-te -7500 -7500 7500 7500" in cmd

    # Hard-reject UTM 35N (115-265 m convergence drift) and -dstnodata 0 (deletes Marmara Sea)
    with pytest.raises(ValueError, match="EPSG:32635"):
        build_gdalwarp_cmd("in.tif", "out.tif", t_srs="EPSG:32635")
    with pytest.raises(ValueError, match="nodata"):
        build_gdalwarp_cmd("in.tif", "out.tif", dstnodata=0.0)
    with pytest.raises(ValueError, match="nodata"):
        build_gdalwarp_cmd("in.tif", "out.tif", dstnodata=0)

    # Hard-reject CC BY-NC-SA sources (EOX Sentinel-2 Cloudless & FABDEM)
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        validate_asset_url_tos("https://tiles.maps.eox.at/s2cloudless-2024")
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        validate_asset_url_tos("https://s2maps.eu/tiles/s2cloudless-2025")
    with pytest.raises(ValueError, match="CC BY-NC-SA"):
        validate_asset_url_tos("https://data.bris.ac.uk/datasets/fabdem-v1-2/N40E029_FABDEM_V1-2.tif")

    # Hard-reject offline scraping of commercial streaming tiles (Bing/Esri/Google/Cesium ion)
    for forbidden_url in (
        "https://ecn.t0.tiles.virtualearth.net/tiles/a120312.jpeg",
        "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer",
        "https://tile.googleapis.com/v1/3dtiles/root.json",
        "https://api.cesium.com/v1/assets/1/endpoint",
        "https://tile.openstreetmap.org/14/9520/6154.png",
    ):
        with pytest.raises(ValueError, match="ToS"):
            validate_asset_url_tos(forbidden_url)

    # Allowed ToS-clean open sources
    for allowed_url in (
        COPERNICUS_GLO30_S3_URL,
        f"{SENTINEL2_COGS_S3_PREFIX}2025/7/S2B_35TPF_20250715_0_L2A/B04.tif",
        EARTH_SEARCH_STAC_URL,
        GEOFABRIK_TURKEY_PBF_URL,
        OVERTURE_STAC_URL,
    ):
        assert validate_asset_url_tos(allowed_url) == allowed_url

    # Full pipeline dry-run plan locks raw AWS sentinel-cogs 35TPF + GLO-30 N40_00_E029_00
    assert COPERNICUS_GLO30_TILE == "N40_00_E029_00"
    assert SENTINEL2_MGRS_TILE == "35TPF"
    plan = build_corridor_pipeline_plan("viz/geo/data")
    plan_text = str(plan)
    assert "sentinel-cogs" in plan_text and "35TPF" in plan_text
    assert "N40_00_E029_00" in plan_text
    assert KARTAL_LOCAL_TMERC_PROJ in plan_text
    assert "EPSG:32635" not in plan_text
    assert "eox.at" not in plan_text.lower() and "fabdem" not in plan_text.lower()

    # DuckDB Overture SQL must have valid '||' concatenation (no broken DR-5 markdown pipes)
    sql = build_overture_duckdb_sql("viz/geo/data/kartal_15km_buildings.parquet")
    assert "|| getvariable('latest') ||" in sql
    assert "29.099011" in sql and "40.953536" in sql

    # viz/geo/data/.gitignore must exclude all ODbL & raster artifacts while tracking itself
    gi_path = Path("viz/geo/data/.gitignore")
    assert gi_path.exists()
    gi_lines = {line.strip() for line in gi_path.read_text(encoding="utf-8").splitlines()}
    for required_pattern in ("*", "!.gitignore", "*.osm", "*.pbf", "*.parquet", "*.tif"):
        assert required_pattern in gi_lines

    # Non-finite nodata must also be rejected
    with pytest.raises(ValueError, match="nodata"):
        build_gdalwarp_cmd("in.tif", "out.tif", dstnodata=float("nan"))

    # Lazy package re-exports resolve identically to direct submodule imports
    assert viz_geo.VertiRTAGeodesyBridge is VertiRTAGeodesyBridge
    assert viz_geo.build_ltfj_ltbx_geometries is build_ltfj_ltbx_geometries
    assert viz_geo.build_gdalwarp_cmd is build_gdalwarp_cmd


def test_ecef_roundtrip_so3_rotation_and_geojson_export(tmp_path: Path) -> None:
    """Verify < 1e-8 m ECEF round-trip, SO(3) ECEF attitude, vertical CTR cutoffs, and GeoJSON export."""
    for lat, lon, h in (
        (40.8860, 29.1880, 48.68),
        (40.912786, 29.192122, 798.58),
        (40.998889, 29.216111, 646.35),
    ):
        xyz = geodetic_to_ecef(lat, lon, h)
        lat_rt, lon_rt, h_rt = ecef_to_geodetic(xyz)
        assert abs(lat_rt - lat) < 1e-11
        assert abs(lon_rt - lon) < 1e-11
        assert abs(h_rt - h) < 1e-8  # ~10 nm (~10 ULPs of 6.387e6 m WGS84 float64 radius)

    q_enu = px4_quat_to_enu_flu(0.945380, 0.098027, 0.121511, 0.286149)
    nrm = math.sqrt(sum(c * c for c in q_enu))
    q_enu_u = tuple(c / nrm for c in q_enu)
    r_ecef = enu_flu_quat_to_ecef_rotmat(40.8860, 29.1880, q_enu_u)
    r_t = tuple(tuple(r_ecef[j][i] for j in range(3)) for i in range(3))
    rtr = _matmul3(r_t, r_ecef)
    for i in range(3):
        for j in range(3):
            assert abs(rtr[i][j] - (1.0 if i == j else 0.0)) < 1e-12
    det = (
        r_ecef[0][0] * (r_ecef[1][1] * r_ecef[2][2] - r_ecef[1][2] * r_ecef[2][1])
        - r_ecef[0][1] * (r_ecef[1][0] * r_ecef[2][2] - r_ecef[1][2] * r_ecef[2][0])
        + r_ecef[0][2] * (r_ecef[1][0] * r_ecef[2][1] - r_ecef[1][1] * r_ecef[2][0])
    )
    assert abs(det - 1.0) < 1e-12

    # Altitude ceiling cutoffs & CTR mutual disjointness
    assert is_inside_ltfj(40.86778, 29.23972, alt_amsl_m=500.0) is True
    assert is_inside_ltfj(40.86778, 29.23972, alt_amsl_m=800.0) is False
    assert is_inside_ltbx(40.99056, 29.21611, alt_amsl_m=500.0) is True
    assert is_inside_ltbx(40.99056, 29.21611, alt_amsl_m=650.0) is False
    assert signed_distance_to_ltbx_m(40.89833, 29.30917) > 5000.0  # LTFJ ARP outside LTBX
    assert signed_distance_to_ltfj_m(40.99889, 29.21611) > 5000.0  # LTBX center outside LTFJ

    # GeoJSON export writes clean UTF-8 without BOM
    out_geojson = tmp_path / "ctr.geojson"
    fc = export_airspace_geojson(out_geojson)
    raw = out_geojson.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    loaded = json.loads(raw.decode("utf-8"))
    assert loaded == fc and len(loaded["features"]) == 2

