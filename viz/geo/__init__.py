"""VertiRTA ToS-clean Istanbul geospatial builder, airspace polygons, and geodesy bridge.

Uses PEP 562 lazy attribute loading so running ``python -m viz.geo.build_istanbul_corridor``
emits zero ``runpy`` double-import warnings while preserving ``from viz.geo import ...``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # vertirta_geodesy
    "ANCHOR_H_AMSL": ("viz.geo.vertirta_geodesy", "ANCHOR_H_AMSL"),
    "ANCHOR_LAT_DEG": ("viz.geo.vertirta_geodesy", "ANCHOR_LAT_DEG"),
    "ANCHOR_LON_DEG": ("viz.geo.vertirta_geodesy", "ANCHOR_LON_DEG"),
    "DEFAULT_ISTANBUL_GEOID_N0_M": ("viz.geo.vertirta_geodesy", "DEFAULT_ISTANBUL_GEOID_N0_M"),
    "R_GAUSS_KARTAL": ("viz.geo.vertirta_geodesy", "R_GAUSS_KARTAL"),
    "VertiRTAGeodesyBridge": ("viz.geo.vertirta_geodesy", "VertiRTAGeodesyBridge"),
    "assert_valid_istanbul_geoid": ("viz.geo.vertirta_geodesy", "assert_valid_istanbul_geoid"),
    "ecef_to_geodetic": ("viz.geo.vertirta_geodesy", "ecef_to_geodetic"),
    "enu_flu_quat_to_ecef_rotmat": ("viz.geo.vertirta_geodesy", "enu_flu_quat_to_ecef_rotmat"),
    "enu_flu_to_px4_quat": ("viz.geo.vertirta_geodesy", "enu_flu_to_px4_quat"),
    "geodetic_to_ecef": ("viz.geo.vertirta_geodesy", "geodetic_to_ecef"),
    "geoid_undulation_egm2008": ("viz.geo.vertirta_geodesy", "geoid_undulation_egm2008"),
    "istanbul_egm2008_patch_m": ("viz.geo.vertirta_geodesy", "istanbul_egm2008_patch_m"),
    "px4_quat_to_enu_flu": ("viz.geo.vertirta_geodesy", "px4_quat_to_enu_flu"),
    "quat_to_rotmat": ("viz.geo.vertirta_geodesy", "quat_to_rotmat"),
    # airspace_polygons
    "AIRSPACE_DISCLAIMER": ("viz.geo.airspace_polygons", "AIRSPACE_DISCLAIMER"),
    "CTR_RADIUS_M": ("viz.geo.airspace_polygons", "CTR_RADIUS_M"),
    "LTBX_CEILING_AMSL_M": ("viz.geo.airspace_polygons", "LTBX_CEILING_AMSL_M"),
    "LTBX_CEILING_HAE_M": ("viz.geo.airspace_polygons", "LTBX_CEILING_HAE_M"),
    "LTFJ_CEILING_AMSL_M": ("viz.geo.airspace_polygons", "LTFJ_CEILING_AMSL_M"),
    "LTFJ_CEILING_HAE_M": ("viz.geo.airspace_polygons", "LTFJ_CEILING_HAE_M"),
    "build_enu_curtain_mesh_data": ("viz.geo.airspace_polygons", "build_enu_curtain_mesh_data"),
    "build_ltfj_ltbx_geometries": ("viz.geo.airspace_polygons", "build_ltfj_ltbx_geometries"),
    "export_airspace_geojson": ("viz.geo.airspace_polygons", "export_airspace_geojson"),
    "is_inside_ltbx": ("viz.geo.airspace_polygons", "is_inside_ltbx"),
    "is_inside_ltfj": ("viz.geo.airspace_polygons", "is_inside_ltfj"),
    "signed_distance_to_ltbx_m": ("viz.geo.airspace_polygons", "signed_distance_to_ltbx_m"),
    "signed_distance_to_ltfj_m": ("viz.geo.airspace_polygons", "signed_distance_to_ltfj_m"),
    # build_istanbul_corridor
    "COPERNICUS_GLO30_S3_URL": ("viz.geo.build_istanbul_corridor", "COPERNICUS_GLO30_S3_URL"),
    "COPERNICUS_GLO30_TILE": ("viz.geo.build_istanbul_corridor", "COPERNICUS_GLO30_TILE"),
    "KARTAL_LOCAL_TMERC_PROJ": ("viz.geo.build_istanbul_corridor", "KARTAL_LOCAL_TMERC_PROJ"),
    "SENTINEL2_COGS_S3_PREFIX": ("viz.geo.build_istanbul_corridor", "SENTINEL2_COGS_S3_PREFIX"),
    "SENTINEL2_MGRS_TILE": ("viz.geo.build_istanbul_corridor", "SENTINEL2_MGRS_TILE"),
    "build_copernicus_glo30_download_cmd": (
        "viz.geo.build_istanbul_corridor",
        "build_copernicus_glo30_download_cmd",
    ),
    "build_corridor_pipeline_plan": (
        "viz.geo.build_istanbul_corridor",
        "build_corridor_pipeline_plan",
    ),
    "build_gdalwarp_cmd": ("viz.geo.build_istanbul_corridor", "build_gdalwarp_cmd"),
    "build_osmium_extract_cmd": ("viz.geo.build_istanbul_corridor", "build_osmium_extract_cmd"),
    "build_overture_duckdb_sql": ("viz.geo.build_istanbul_corridor", "build_overture_duckdb_sql"),
    "build_sentinel2_stac_query": ("viz.geo.build_istanbul_corridor", "build_sentinel2_stac_query"),
    "validate_asset_url_tos": ("viz.geo.build_istanbul_corridor", "validate_asset_url_tos"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val
