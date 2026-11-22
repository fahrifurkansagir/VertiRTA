"""Canonical Geodetic, Vertical Datum (EGM2008 <-> WGS84 HAE) & Frame Bridge.

Implements the VertiRTA DR-5 / Plan 1B Task 6 geodesy contract for the
Maltepe-Dragos-Kartal Istanbul corridor anchored at the Kartal Coastal Vertiport
(40.8860 N, 29.1880 E, 12.0 m AMSL EGM2008):

- Hard-guards against silent PROJ 0.0 m EGM2008 geoid fallback (missing
  ``us_nga_egm08_25.tif``), enforcing ``20.0 < N < 50.0 m`` across Istanbul.
- Applies Earth curvature drop ``u_tan = Z_flat - (E^2 + N^2) / (2 * R_gauss)``
  with ``R_GAUSS_KARTAL = 6375036.971 m``, keeping vertical tangent residual
  under 9 mm over the 15x15 km domain (eliminating the +8.82 m corner float).
- Provides exact Hamilton scalar-first ``[w, x, y, z]`` PX4 (NED/FRD) <->
  Gazebo/Blender/ROS (ENU/FLU) quaternion conversion (< 1e-12 round-trip).
- Pure Python 64-bit IEEE-754 implementation with optional ``numpy`` array
  interop and optional ``pyproj`` (EPSG:9518 -> EPSG:4979) fast-path.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

try:  # Optional numpy interop without making numpy a hard runtime dependency.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in lean WSL environments
    _np = None  # type: ignore[assignment]

try:  # Optional pyproj fast-path for full PROJ EGM2008 TIFF grid lookup.
    from pyproj import Transformer as _ProjTransformer
except ImportError:  # pragma: no cover - exercised when pyproj is absent
    _ProjTransformer = None  # type: ignore[assignment]

# WGS84 Reference Ellipsoid
WGS84_A: float = 6378137.0
WGS84_F: float = 1.0 / 298.257223563
WGS84_E2: float = 2.0 * WGS84_F - WGS84_F * WGS84_F

# Kartal Coastal Vertiport Anchor (40.8860 N, 29.1880 E, 12.0 m AMSL EGM2008)
ANCHOR_LAT_DEG: float = 40.8860
ANCHOR_LON_DEG: float = 29.1880
ANCHOR_H_AMSL: float = 12.0
R_GAUSS_KARTAL: float = 6375036.971  # sqrt(M_curv * N_curv) at 40.8860 N

# Istanbul EGM2008 geoid reference & sanity guard bounds (DR-5)
# Official DHMI eAIP benchmarks: LTFM 37.19 m, LTBA 36.88 m, LTFJ 36.58 m
# Nominal Kartal reference: +36.68 m (~+36.7 m), gradient +0.85 cm/km N, -0.54 cm/km E
DEFAULT_ISTANBUL_GEOID_N0_M: float = 36.68
ISTANBUL_GEOID_MIN_M: float = 20.0
ISTANBUL_GEOID_MAX_M: float = 50.0
_GEOID_GRAD_NORTH_M_PER_KM: float = 0.0085
_GEOID_GRAD_EAST_M_PER_KM: float = -0.0054
_METERS_PER_DEG_LAT_KARTAL: float = 111052.5
_METERS_PER_DEG_LON_KARTAL: float = 84152.0

_SQRT_HALF: float = math.sqrt(0.5)
_CACHED_PROJ_TRANSFORMER: Any = None
_PROJ_INIT_ATTEMPTED: bool = False


def assert_valid_istanbul_geoid(n_m: float) -> float:
    """Reject missing/silent 0.0 m EGM2008 PROJ fallback or non-finite values.

    Raises:
        RuntimeError: If ``n_m`` is non-finite or not strictly inside
            ``(20.0, 50.0)`` meters (Istanbul EGM2008 range is ~36.5-37.3 m).
    """
    try:
        val = float(n_m)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid Istanbul EGM2008 geoid undulation {n_m!r}: non-numeric value"
        ) from exc
    if not math.isfinite(val) or not (ISTANBUL_GEOID_MIN_M < val < ISTANBUL_GEOID_MAX_M):
        raise RuntimeError(
            f"Invalid Istanbul EGM2008 geoid undulation N={val!r} m "
            f"(expected {ISTANBUL_GEOID_MIN_M} < N < {ISTANBUL_GEOID_MAX_M} m; "
            "missing local PROJ us_nga_egm08_25.tif grid causes silent 0.0 m fallback)"
        )
    return val


def istanbul_egm2008_patch_m(lat_deg: float, lon_deg: float) -> float:
    """Deterministic offline Istanbul EGM2008 geoid undulation model (meters).

    Anchored to Kartal Coastal Vertiport (+36.68 m) with the DHMI eAIP
    Istanbul planar gradient (+0.85 cm/km North, -0.54 cm/km East across
    LTFM/LTBA/LTFJ). Always verified through ``assert_valid_istanbul_geoid``.
    """
    dn_km = (float(lat_deg) - ANCHOR_LAT_DEG) * (_METERS_PER_DEG_LAT_KARTAL / 1000.0)
    de_km = (float(lon_deg) - ANCHOR_LON_DEG) * (_METERS_PER_DEG_LON_KARTAL / 1000.0)
    n_est = (
        DEFAULT_ISTANBUL_GEOID_N0_M
        + _GEOID_GRAD_NORTH_M_PER_KM * dn_km
        + _GEOID_GRAD_EAST_M_PER_KM * de_km
    )
    return assert_valid_istanbul_geoid(n_est)


def _get_proj_transformer() -> Any:
    global _CACHED_PROJ_TRANSFORMER, _PROJ_INIT_ATTEMPTED
    if _PROJ_INIT_ATTEMPTED:
        return _CACHED_PROJ_TRANSFORMER
    _PROJ_INIT_ATTEMPTED = True
    if _ProjTransformer is None:
        return None
    try:
        _CACHED_PROJ_TRANSFORMER = _ProjTransformer.from_crs(
            "EPSG:9518", "EPSG:4979", always_xy=True
        )
    except Exception:  # pragma: no cover - defensive when PROJ data dir is broken
        _CACHED_PROJ_TRANSFORMER = None
    return _CACHED_PROJ_TRANSFORMER


def query_pyproj_egm2008(lon_deg: float, lat_deg: float) -> float:
    """Query EPSG:9518 -> EPSG:4979 via pyproj and reject silent 0.0 m fallback."""
    transformer = _get_proj_transformer()
    if transformer is None:
        raise RuntimeError(
            "pyproj EPSG:9518 -> EPSG:4979 transformer unavailable; cannot query "
            "EGM2008 grid (us_nga_egm08_25.tif)"
        )
    _, _, h_hae_at_zero_amsl = transformer.transform(float(lon_deg), float(lat_deg), 0.0)
    return assert_valid_istanbul_geoid(float(h_hae_at_zero_amsl))


def geoid_undulation_egm2008(
    lon_deg: float,
    lat_deg: float,
    *,
    require_proj_grid: bool = False,
) -> float:
    """Return EGM2008 geoid undulation N(lat, lon) in meters for Istanbul.

    Uses pyproj ``EPSG:9518 -> EPSG:4979`` when installed with a valid
    ``us_nga_egm08_25.tif`` grid; otherwise raises when ``require_proj_grid``
    is True or uses the audited offline Istanbul EGM2008 planar patch.
    """
    if require_proj_grid:
        return query_pyproj_egm2008(lon_deg, lat_deg)
    try:
        return query_pyproj_egm2008(lon_deg, lat_deg)
    except RuntimeError:
        return istanbul_egm2008_patch_m(lat_deg, lon_deg)


def _as_vec3(x: float, y: float, z: float) -> Any:
    if _np is not None:
        return _np.array([x, y, z], dtype=_np.float64)
    return (x, y, z)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, h_hae: float) -> Any:
    """Convert WGS84 geodetic (lat_deg, lon_deg, h_hae_m) to ECEF [X, Y, Z] (m)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    s_lat = math.sin(lat)
    c_lat = math.cos(lat)
    n_nu = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s_lat * s_lat)
    x = (n_nu + h_hae) * c_lat * math.cos(lon)
    y = (n_nu + h_hae) * c_lat * math.sin(lon)
    z = (n_nu * (1.0 - WGS84_E2) + h_hae) * s_lat
    return _as_vec3(x, y, z)


def ecef_to_geodetic(xyz: Sequence[float]) -> tuple[float, float, float]:
    """Iterative ECEF -> WGS84 geodetic (lat_deg, lon_deg, h_hae_m), < 1e-9 m."""
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    lat = math.atan2(z, p * (1.0 - WGS84_E2)) if p > 0.0 else math.copysign(math.pi / 2.0, z)
    h = 0.0
    for _ in range(8):
        s_lat = math.sin(lat)
        c_lat = math.cos(lat)
        n_nu = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s_lat * s_lat)
        h = p * c_lat + z * s_lat - n_nu * (1.0 - WGS84_E2 * s_lat * s_lat)
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n_nu / (n_nu + h)))
    s_lat = math.sin(lat)
    c_lat = math.cos(lat)
    n_nu = WGS84_A / math.sqrt(1.0 - WGS84_E2 * s_lat * s_lat)
    h = p * c_lat + z * s_lat - n_nu * (1.0 - WGS84_E2 * s_lat * s_lat)
    return math.degrees(lat), math.degrees(lon), h


def R_ecef_from_enu(lat_deg: float, lon_deg: float) -> tuple[tuple[float, float, float], ...]:
    """3x3 orthogonal rotation matrix mapping local ENU vectors into ECEF."""
    s_lat = math.sin(math.radians(lat_deg))
    c_lat = math.cos(math.radians(lat_deg))
    s_lon = math.sin(math.radians(lon_deg))
    c_lon = math.cos(math.radians(lon_deg))
    return (
        (-s_lon, -s_lat * c_lon, c_lat * c_lon),
        (c_lon, -s_lat * s_lon, c_lat * s_lon),
        (0.0, c_lat, s_lat),
    )


def px4_quat_to_enu_flu(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Exact closed-form q_ENU_FLU from PX4 q_NED_FRD (Hamilton [w, x, y, z]).

    Implements q_ENU_NED (*) q_NED_FRD (*) q_FRD_FLU with sign canonicalized to
    s * (w + z, x + y, x - y, w - z), s = sqrt(1/2). This orthogonal map is an
    exact involution (M^2 = I).
    """
    return (
        _SQRT_HALF * (w + z),
        _SQRT_HALF * (x + y),
        _SQRT_HALF * (x - y),
        _SQRT_HALF * (w - z),
    )


def enu_flu_to_px4_quat(w: float, x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Exact inverse: q_NED_FRD from Gazebo/Blender q_ENU_FLU (Hamilton [w, x, y, z])."""
    return (
        _SQRT_HALF * (w + z),
        _SQRT_HALF * (x + y),
        _SQRT_HALF * (x - y),
        _SQRT_HALF * (w - z),
    )


def quat_to_rotmat(q_wxyz: Sequence[float]) -> tuple[tuple[float, float, float], ...]:
    """Convert unit Hamilton quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = (float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3]))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def enu_flu_quat_to_ecef_rotmat(
    lat_deg: float, lon_deg: float, q_enu_flu: Sequence[float]
) -> tuple[tuple[float, float, float], ...]:
    """Return R_ECEF_from_FLU = R_ECEF_from_ENU(lat, lon) @ R(q_enu_flu)."""
    r_en = R_ecef_from_enu(lat_deg, lon_deg)
    r_q = quat_to_rotmat(q_enu_flu)
    return tuple(
        tuple(sum(r_en[i][k] * r_q[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


class VertiRTAGeodesyBridge:
    """Curvature-compensated flat ENU <-> CesiumJS WGS84 HAE geodesy bridge."""

    def __init__(
        self,
        *,
        anchor_lat_deg: float = ANCHOR_LAT_DEG,
        anchor_lon_deg: float = ANCHOR_LON_DEG,
        anchor_h_amsl_m: float = ANCHOR_H_AMSL,
        fixed_geoid_n_m: float | None = None,
        geoid_provider: Callable[[float, float], float] | None = None,
        require_proj_grid: bool = False,
        r_gauss_m: float = R_GAUSS_KARTAL,
    ) -> None:
        self.anchor_lat_deg = float(anchor_lat_deg)
        self.anchor_lon_deg = float(anchor_lon_deg)
        self.anchor_h_amsl_m = float(anchor_h_amsl_m)
        self.r_gauss_m = float(r_gauss_m)
        self._fixed_geoid_n_m = (
            assert_valid_istanbul_geoid(fixed_geoid_n_m)
            if fixed_geoid_n_m is not None
            else None
        )
        self._geoid_provider = geoid_provider
        self._require_proj_grid = bool(require_proj_grid)

        self.N0: float = self.geoid_undulation(self.anchor_lon_deg, self.anchor_lat_deg)
        self.h0: float = self.anchor_h_amsl_m + self.N0
        self.p0_ecef = geodetic_to_ecef(self.anchor_lat_deg, self.anchor_lon_deg, self.h0)
        self.R0 = R_ecef_from_enu(self.anchor_lat_deg, self.anchor_lon_deg)

    def geoid_undulation(self, lon_deg: float, lat_deg: float) -> float:
        """Resolve and validate Istanbul EGM2008 geoid undulation at (lon, lat)."""
        if self._fixed_geoid_n_m is not None:
            return self._fixed_geoid_n_m
        if self._geoid_provider is not None:
            return assert_valid_istanbul_geoid(
                self._geoid_provider(float(lon_deg), float(lat_deg))
            )
        return assert_valid_istanbul_geoid(
            geoid_undulation_egm2008(
                float(lon_deg),
                float(lat_deg),
                require_proj_grid=self._require_proj_grid,
            )
        )

    def flat_enu_to_cesium_wgs84(
        self, east_m: float, north_m: float, z_flat_m: float
    ) -> dict[str, Any]:
        """Map flat Gazebo/Blender ENU (E, N, Z_flat) to curvature-compensated WGS84 HAE.

        Eliminates both the +36.7 m EGM2008 geoid offset and up to +8.82 m
        Earth-curvature corner float across the 15x15 km Kartal corridor
        (< 9 mm vertical tangent residual).
        """
        e = float(east_m)
        n = float(north_m)
        z_f = float(z_flat_m)

        curvature_drop_m = (e * e + n * n) / (2.0 * self.r_gauss_m)
        u_tan_m = z_f - curvature_drop_m

        p0x, p0y, p0z = float(self.p0_ecef[0]), float(self.p0_ecef[1]), float(self.p0_ecef[2])
        r0 = self.R0
        p_corr = (
            p0x + r0[0][0] * e + r0[0][1] * n + r0[0][2] * u_tan_m,
            p0y + r0[1][0] * e + r0[1][1] * n + r0[1][2] * u_tan_m,
            p0z + r0[2][0] * e + r0[2][1] * n + r0[2][2] * u_tan_m,
        )
        lat_deg, lon_deg, h_tangent_hae = ecef_to_geodetic(p_corr)

        alt_amsl_m = self.anchor_h_amsl_m + z_f
        geoid_n_m = self.geoid_undulation(lon_deg, lat_deg)
        alt_hae_m = alt_amsl_m + geoid_n_m

        return {
            "lat_deg": lat_deg,
            "lon_deg": lon_deg,
            "alt_amsl_m": alt_amsl_m,
            "geoid_n_m": geoid_n_m,
            "alt_hae_m": alt_hae_m,
            "curvature_drop_m": curvature_drop_m,
            "u_tan_m": u_tan_m,
            "tangent_residual_m": h_tangent_hae - (self.h0 + z_f),
            "ecef_m": geodetic_to_ecef(lat_deg, lon_deg, alt_hae_m),
        }

    def cesium_wgs84_to_flat_enu(
        self, lat_deg: float, lon_deg: float, alt_hae_m: float
    ) -> tuple[float, float, float]:
        """Inverse map from WGS84 (lat_deg, lon_deg, alt_hae_m) to flat ENU (E, N, Z_flat)."""
        geoid_n_m = self.geoid_undulation(float(lon_deg), float(lat_deg))
        alt_amsl_m = float(alt_hae_m) - geoid_n_m
        z_flat_m = alt_amsl_m - self.anchor_h_amsl_m

        p_target = geodetic_to_ecef(float(lat_deg), float(lon_deg), self.h0 + z_flat_m)
        dx = float(p_target[0]) - float(self.p0_ecef[0])
        dy = float(p_target[1]) - float(self.p0_ecef[1])
        dz = float(p_target[2]) - float(self.p0_ecef[2])

        r0 = self.R0
        east_m = r0[0][0] * dx + r0[1][0] * dy + r0[2][0] * dz
        north_m = r0[0][1] * dx + r0[1][1] * dy + r0[2][1] * dz
        return (east_m, north_m, z_flat_m)

    @staticmethod
    def px4_quat_to_enu_flu(
        w: float, x: float, y: float, z: float
    ) -> tuple[float, float, float, float]:
        """Exact closed-form q_ENU_FLU from PX4 q_NED_FRD (Hamilton [w, x, y, z])."""
        return px4_quat_to_enu_flu(w, x, y, z)

    @staticmethod
    def enu_flu_to_px4_quat(
        w: float, x: float, y: float, z: float
    ) -> tuple[float, float, float, float]:
        """Exact inverse q_NED_FRD from ENU/FLU quaternion (Hamilton [w, x, y, z])."""
        return enu_flu_to_px4_quat(w, x, y, z)
