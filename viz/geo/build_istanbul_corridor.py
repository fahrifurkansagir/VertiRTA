"""ToS-Clean Offline Istanbul 15x15 km Corridor Builder (Kartal Local tmerc).

Enforces the VertiRTA DR-5 / Plan 1B Task 6 legal & geometric firewall:
- Warps rasters strictly into the Kartal local topocentric Transverse Mercator
  CRS (``+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84 +units=m
  +no_defs``), hard-rejecting UTM Zone 35N (``EPSG:32635``) which induces
  1.432 deg meridian convergence (115-265 m horizontal drift) and clips the
  15x15 km bounding box.
- Enforces ``-dstnodata -9999`` (hard-rejecting ``-dstnodata 0`` / ``0.0``
  which deletes the 0.0 m AMSL Sea of Marmara!).
- Pulls only commercial-safe open offline sources:
  * Raw AWS Copernicus DEM GLO-30 (``N40_00_E029_00``, EGM2008).
  * Raw AWS ``sentinel-cogs`` Sentinel-2 L2A (``35TPF`` MGRS tile, 10 m RGB+NIR)
    via Element 84 Earth Search STAC API.
  * Geofabrik Turkey OSM extract & Overture Maps S3 GeoParquet (downloaded
    locally into git-ignored ``viz/geo/data/`` only; zero ODbL committed).
- Hard-bans ``CC BY-NC-SA 4.0`` sources (EOX Sentinel-2 Cloudless, FABDEM) and
  all Layer-1 commercial live-streaming tile scraping (Bing, Esri, Google,
  Cesium ion offline export, OSM raster tiles).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

KARTAL_LOCAL_TMERC_PROJ: str = (
    "+proj=tmerc +lat_0=40.886 +lon_0=29.188 +k=1 +datum=WGS84 +units=m +no_defs"
)

# Canonical 15x15 km symmetric Kartal Coastal Vertiport corridor
KARTAL_EXTENT_TMERC_M: tuple[int, int, int, int] = (-7500, -7500, 7500, 7500)
KARTAL_BBOX_WGS84: tuple[float, float, float, float] = (
    29.099011,  # west lon
    40.818464,  # south lat
    29.276989,  # east lon
    40.953536,  # north lat
)

# Verified ToS-clean Layer-2 offline sources
COPERNICUS_GLO30_TILE: str = "N40_00_E029_00"
COPERNICUS_GLO30_S3_URL: str = (
    "s3://copernicus-dem-30m/"
    "Copernicus_DSM_COG_10_N40_00_E029_00_DEM/"
    "Copernicus_DSM_COG_10_N40_00_E029_00_DEM.tif"
)
SENTINEL2_MGRS_TILE: str = "35TPF"
SENTINEL2_COGS_S3_PREFIX: str = "s3://sentinel-cogs/sentinel-s2-l2a-cogs/35/T/PF/"
EARTH_SEARCH_STAC_URL: str = "https://earth-search.aws.element84.com/v1"
GEOFABRIK_TURKEY_PBF_URL: str = "https://download.geofabrik.de/europe/turkey-latest.osm.pbf"
OVERTURE_STAC_URL: str = "https://stac.overturemaps.org/catalog.json"
OVERTURE_S3_PREFIX: str = "s3://overturemaps-us-west-2/release/"
IBB_OPEN_DATA_URL: str = "https://data.ibb.gov.tr/dataset/mahalle-bazli-bina-analiz-verisi"
PROJ_EGM2008_GRID_URL: str = "https://cdn.proj.org/us_nga_egm08_25.tif"

_CC_BY_NC_SA_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(^|[./])eox\.at\b", "EOX Sentinel-2 Cloudless"),
    (r"(^|[./])s2maps\.eu\b", "EOX s2maps Sentinel-2 Cloudless"),
    (r"s2cloudless", "EOX Sentinel-2 Cloudless"),
    (r"fabdem", "FABDEM bare-earth DEM"),
)

_TOS_SCRAPING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(^|[./])virtualearth\.net\b", "Microsoft Bing Maps"),
    (r"(^|[./])bing\.com\b", "Microsoft Bing Maps"),
    (r"(^|[./])arcgisonline\.com\b", "Esri World Imagery"),
    (r"(^|[./])arcgis\.com\b", "Esri ArcGIS Imagery"),
    (r"tile\.googleapis\.com", "Google Map Tiles / Photorealistic 3D Tiles"),
    (r"maps\.googleapis\.com", "Google Maps Platform"),
    (r"(mt[0-9]|khms[0-9]?)\.google\.com", "Google Satellite Tiles"),
    (r"earth\.google\.com", "Google Earth Imagery"),
    (r"(api|assets)(\.ion)?\.cesium\.com", "Cesium ion Asset Depot offline export"),
    (r"tile\.openstreetmap\.org", "OSM raster tile server bulk scraping"),
)

_ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "copernicus-dem-30m",
        "sentinel-cogs",
        "earth-search.aws.element84.com",
        "download.geofabrik.de",
        "overturemaps-us-west-2",
        "stac.overturemaps.org",
        "data.ibb.gov.tr",
        "cdn.proj.org",
    }
)


def validate_asset_url_tos(url: str, *, strict_allowlist: bool = True) -> str:
    """Validate that an offline geospatial asset URL is 100% ToS & license clean.

    Raises:
        ValueError: On CC BY-NC-SA sources (EOX, FABDEM), commercial tile
            scraping (Bing, Esri, Google, Cesium ion, OSM raster tiles),
            invalid schemes, or non-allowlisted hosts when ``strict_allowlist``
            is True.
    """
    cleaned = str(url).strip()
    if not cleaned:
        raise ValueError("Empty geospatial asset URL is forbidden by ToS firewall")

    lowered = cleaned.lower()
    for pattern, label in _CC_BY_NC_SA_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError(
                f"Forbidden CC BY-NC-SA 4.0 non-commercial asset ({label}): {cleaned}. "
                "Use raw AWS sentinel-cogs (35TPF) or Copernicus GLO-30 (N40_00_E029_00)."
            )

    for pattern, label in _TOS_SCRAPING_PATTERNS:
        if re.search(pattern, lowered):
            raise ValueError(
                f"ToS violation — offline download/scraping of {label} is forbidden: "
                f"{cleaned}. Restrict commercial tiles strictly to Layer-1 :5180 live streaming."
            )

    parsed = urlparse(cleaned)
    if parsed.scheme not in {"s3", "https"}:
        raise ValueError(
            f"Unsupported asset URL scheme {parsed.scheme!r} in {cleaned!r}; "
            "expected 's3://' or 'https://'"
        )

    host = (parsed.netloc or "").lower()
    if strict_allowlist and host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Unverified geospatial asset host {host!r} in {cleaned!r} "
            "(not in VertiRTA ToS-clean Layer-2 allowlist)"
        )
    return cleaned


def _format_num_arg(value: int | float) -> str:
    if isinstance(value, int) or float(value).is_integer():
        return str(int(round(float(value))))
    return f"{float(value):g}"


def build_gdalwarp_cmd(
    src: str | Path,
    dst: str | Path,
    *,
    t_srs: str = KARTAL_LOCAL_TMERC_PROJ,
    extent_m: Sequence[int | float] = KARTAL_EXTENT_TMERC_M,
    resolution_m: Sequence[int | float] = (10, 10),
    resampling: str = "cubicspline",
    dstnodata: int | float = -9999,
    overwrite: bool = True,
) -> list[str]:
    """Build a DR-5 compliant ``gdalwarp`` command in Kartal local topocentric CRS.

    Hard-rejects ``EPSG:32635`` / UTM projections (115-265 m convergence drift)
    and ``-dstnodata 0`` / ``0.0`` (which deletes the Sea of Marmara).
    """
    srs_str = str(t_srs).strip()
    if "32635" in srs_str or "+proj=utm" in srs_str.lower():
        raise ValueError(
            f"Forbidden target CRS {srs_str!r}: EPSG:32635 / UTM introduces 1.432 deg "
            "meridian convergence (115-265 m drift) across Kartal. Use Kartal local tmerc."
        )
    required_tmerc_tokens = (
        "+proj=tmerc",
        "+lat_0=40.886",
        "+lon_0=29.188",
        "+k=1",
        "+datum=WGS84",
    )
    if not all(tok in srs_str for tok in required_tmerc_tokens):
        raise ValueError(
            f"Target CRS {srs_str!r} must specify Kartal local topocentric tmerc: "
            f"{KARTAL_LOCAL_TMERC_PROJ!r}"
        )

    nodata_val = float(dstnodata)
    if not math.isfinite(nodata_val) or nodata_val == 0.0:
        raise ValueError(
            f"Forbidden -dstnodata {dstnodata!r}: must be finite and non-zero "
            "(0 / 0.0 deletes 0.0 m AMSL Sea of Marmara pixels; use -dstnodata -9999)."
        )

    if len(extent_m) != 4:
        raise ValueError(f"extent_m must have 4 values (xmin, ymin, xmax, ymax), got {extent_m!r}")
    if len(resolution_m) != 2 or any(float(r) <= 0.0 for r in resolution_m):
        raise ValueError(f"resolution_m must contain 2 positive values, got {resolution_m!r}")

    cmd: list[str] = ["gdalwarp"]
    if overwrite:
        cmd.append("-overwrite")
    cmd.extend(
        [
            "-t_srs",
            srs_str,
            "-te",
            *(_format_num_arg(v) for v in extent_m),
            "-tr",
            *(_format_num_arg(v) for v in resolution_m),
            "-r",
            str(resampling),
            "-dstnodata",
            _format_num_arg(dstnodata),
            str(src),
            str(dst),
        ]
    )
    return cmd


def build_copernicus_glo30_download_cmd(
    out_tif: str | Path,
    source_url: str = COPERNICUS_GLO30_S3_URL,
) -> list[str]:
    """Build anonymous AWS S3 download command for Copernicus GLO-30 N40_00_E029_00."""
    verified_url = validate_asset_url_tos(source_url)
    if COPERNICUS_GLO30_TILE not in verified_url:
        raise ValueError(
            f"Expected Copernicus GLO-30 tile {COPERNICUS_GLO30_TILE} in {verified_url!r}"
        )
    return ["aws", "s3", "cp", "--no-sign-request", verified_url, str(out_tif)]


def build_sentinel2_stac_query(
    *,
    cloud_cover_max: float = 2.0,
    datetime_range: str = "2025-06-01/2026-09-30",
    stac_url: str = EARTH_SEARCH_STAC_URL,
) -> dict[str, Any]:
    """Build an Element 84 Earth Search STAC query for AWS sentinel-cogs L2A 35TPF."""
    verified_stac = validate_asset_url_tos(stac_url)
    verified_prefix = validate_asset_url_tos(SENTINEL2_COGS_S3_PREFIX)
    return {
        "stac_api_url": verified_stac,
        "s3_bucket_prefix": verified_prefix,
        "collections": ["sentinel-2-l2a"],
        "bbox": list(KARTAL_BBOX_WGS84),
        "datetime": datetime_range,
        "query": {
            "eo:cloud_cover": {"lt": float(cloud_cover_max)},
            "mgrs:utm_zone": {"eq": 35},
            "mgrs:latitude_band": {"eq": "T"},
            "mgrs:grid_square": {"eq": "PF"},
        },
        "mgrs_tile": SENTINEL2_MGRS_TILE,
        "assets": ["red", "green", "blue", "nir"],  # B04, B03, B02, B08 (10 m GSD)
    }


def build_osmium_extract_cmd(
    turkey_pbf: str | Path,
    out_pbf: str | Path,
    bbox_wgs84: Sequence[float] = KARTAL_BBOX_WGS84,
) -> list[str]:
    """Build ``osmium extract`` command clipping Geofabrik Turkey PBF to Kartal 15x15 km."""
    validate_asset_url_tos(GEOFABRIK_TURKEY_PBF_URL)
    if len(bbox_wgs84) != 4:
        raise ValueError(f"bbox_wgs84 must have 4 values (W, S, E, N), got {bbox_wgs84!r}")
    bbox_str = ",".join(f"{float(v):.6f}" for v in bbox_wgs84)
    return ["osmium", "extract", "-b", bbox_str, str(turkey_pbf), "-o", str(out_pbf)]


def build_overture_duckdb_sql(
    out_parquet: str | Path,
    bbox_wgs84: Sequence[float] = KARTAL_BBOX_WGS84,
    stac_url: str = OVERTURE_STAC_URL,
) -> str:
    """Build valid DuckDB S3 GeoParquet SQL extracting Overture buildings for Kartal."""
    verified_stac = validate_asset_url_tos(stac_url)
    verified_prefix = validate_asset_url_tos(OVERTURE_S3_PREFIX)
    if len(bbox_wgs84) != 4:
        raise ValueError(f"bbox_wgs84 must have 4 values (W, S, E, N), got {bbox_wgs84!r}")
    west, south, east, north = (float(v) for v in bbox_wgs84)
    return (
        "INSTALL spatial; LOAD spatial;\n"
        "INSTALL httpfs; LOAD httpfs;\n"
        "SET s3_region='us-west-2';\n"
        f"SET VARIABLE latest = (SELECT latest FROM '{verified_stac}');\n"
        "COPY (\n"
        "  SELECT id, height, num_floors, roof_shape, roof_material, roof_color,\n"
        "         facade_material, subtype, class, geometry\n"
        f"  FROM read_parquet('{verified_prefix}' || getvariable('latest') || "
        "'/theme=buildings/type=building/*.parquet', hive_partitioning=1)\n"
        f"  WHERE bbox.xmin <= {east:.6f} AND bbox.xmax >= {west:.6f}\n"
        f"    AND bbox.ymin <= {north:.6f} AND bbox.ymax >= {south:.6f}\n"
        f") TO '{Path(out_parquet).as_posix()}' (FORMAT PARQUET);\n"
    )


def build_corridor_pipeline_plan(data_dir: str | Path = "viz/geo/data") -> dict[str, Any]:
    """Assemble the complete ToS-clean dry-run pipeline manifest for Istanbul 15x15 km."""
    root = Path(data_dir)
    raw_dir = root / "raw"

    raw_glo30 = raw_dir / f"cop30_{COPERNICUS_GLO30_TILE}.tif"
    warped_dem = root / "kartal_15km_egm2008.tif"
    raw_s2_vrt = raw_dir / f"s2_l2a_{SENTINEL2_MGRS_TILE}_rgbnir.vrt"
    warped_s2 = root / "kartal_15km_s2_l2a.tif"
    raw_osm = raw_dir / "turkey-latest.osm.pbf"
    clipped_osm = root / "kartal_15km.osm.pbf"
    buildings_parquet = root / "kartal_15km_buildings.parquet"
    airspace_geojson = root / "istanbul_ltfj_ltbx_ctr.geojson"

    # Re-verify every upstream source through the ToS firewall
    sources = {
        "copernicus_glo30": validate_asset_url_tos(COPERNICUS_GLO30_S3_URL),
        "sentinel2_cogs_prefix": validate_asset_url_tos(SENTINEL2_COGS_S3_PREFIX),
        "earth_search_stac": validate_asset_url_tos(EARTH_SEARCH_STAC_URL),
        "geofabrik_turkey_osm": validate_asset_url_tos(GEOFABRIK_TURKEY_PBF_URL),
        "overture_stac": validate_asset_url_tos(OVERTURE_STAC_URL),
        "ibb_building_priors": validate_asset_url_tos(IBB_OPEN_DATA_URL),
        "proj_egm2008_grid": validate_asset_url_tos(PROJ_EGM2008_GRID_URL),
    }

    return {
        "corridor": "Maltepe-Dragos-Kartal 15x15 km (VertiRTA-X8-240)",
        "local_tmerc_crs": KARTAL_LOCAL_TMERC_PROJ,
        "extent_tmerc_m": list(KARTAL_EXTENT_TMERC_M),
        "bbox_wgs84": list(KARTAL_BBOX_WGS84),
        "tiles": {
            "copernicus_glo30": COPERNICUS_GLO30_TILE,
            "sentinel2_mgrs": SENTINEL2_MGRS_TILE,
        },
        "verified_sources": sources,
        "outputs": {
            "dem_tif": warped_dem.as_posix(),
            "sentinel2_tif": warped_s2.as_posix(),
            "osm_pbf": clipped_osm.as_posix(),
            "buildings_parquet": buildings_parquet.as_posix(),
            "airspace_geojson": airspace_geojson.as_posix(),
        },
        "commands": {
            "download_glo30": build_copernicus_glo30_download_cmd(raw_glo30.as_posix()),
            "warp_glo30_dem": build_gdalwarp_cmd(
                raw_glo30.as_posix(), warped_dem.as_posix(), resampling="cubicspline"
            ),
            "sentinel2_stac_query": build_sentinel2_stac_query(),
            "warp_sentinel2_l2a": build_gdalwarp_cmd(
                raw_s2_vrt.as_posix(), warped_s2.as_posix(), resampling="bilinear"
            ),
            "extract_osm": build_osmium_extract_cmd(raw_osm.as_posix(), clipped_osm.as_posix()),
            "overture_duckdb_sql": build_overture_duckdb_sql(buildings_parquet.as_posix()),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="VertiRTA ToS-clean Istanbul 15x15 km geospatial pipeline planner"
    )
    parser.add_argument(
        "--data-dir",
        default="viz/geo/data",
        help="Git-ignored output directory for local offline geospatial assets",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print verified dry-run pipeline plan as JSON",
    )
    args = parser.parse_args(argv)
    plan = build_corridor_pipeline_plan(args.data_dir)
    if args.print_json:
        print(json.dumps(plan, indent=2))
    else:
        print(
            f"Verified ToS-clean Kartal corridor plan ({COPERNICUS_GLO30_TILE}, "
            f"{SENTINEL2_MGRS_TILE}) -> {args.data_dir}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
