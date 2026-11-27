// SPDX-License-Identifier: Apache-2.0
//! Exact WGS-84 ECEF <-> local ENU/NED geodesy matching `core/flight_core/geodesy.hpp`.

use crate::finite::Vec3Ned;

/// Canonical Kartal Coastal Vertiport ENU datum origin (40.8860 N, 29.1880 E, 48.7 m HAE).
pub const KARTAL_DATUM_LAT_DEG: f64 = 40.8860;
pub const KARTAL_DATUM_LON_DEG: f64 = 29.1880;
pub const KARTAL_DATUM_HAE_M: f64 = 48.7;

/// Verified DHMI AIP Turkey LTBX Samandira ARP (40d59m26sN 029d12m58sE).
pub const LTBX_ARP_LAT_DEG: f64 = 40.990556;
pub const LTBX_ARP_LON_DEG: f64 = 29.216111;

/// Kartal Dr. Lutfi Kirdar City Hospital active helipad.
pub const KARTAL_HOSPITAL_LAT_DEG: f64 = 40.91639;
pub const KARTAL_HOSPITAL_LON_DEG: f64 = 29.17417;

const WGS84_A: f64 = 6_378_137.0;
const WGS84_F: f64 = 1.0 / 298.257_223_563;
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

#[derive(Debug, Clone, Copy)]
pub struct GeodeticWgs84 {
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub h_hae_m: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Vec3Ecef {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[inline]
pub fn geodetic_to_ecef(geo: GeodeticWgs84) -> Vec3Ecef {
    let lat = geo.lat_deg.to_radians();
    let lon = geo.lon_deg.to_radians();
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let sin_lon = lon.sin();
    let cos_lon = lon.cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    Vec3Ecef {
        x: (n + geo.h_hae_m) * cos_lat * cos_lon,
        y: (n + geo.h_hae_m) * cos_lat * sin_lon,
        z: (n * (1.0 - WGS84_E2) + geo.h_hae_m) * sin_lat,
    }
}

#[inline]
pub fn ecef_to_geodetic(ecef: Vec3Ecef) -> GeodeticWgs84 {
    let lon = ecef.y.atan2(ecef.x);
    let p = ecef.x.hypot(ecef.y);
    let mut lat = (ecef.z / p.max(1e-9)).atan();
    for _ in 0..8 {
        let sin_lat = lat.sin();
        let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
        lat = ((ecef.z + WGS84_E2 * n * sin_lat) / p.max(1e-9)).atan();
    }
    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat * sin_lat).sqrt();
    let h = if cos_lat.abs() > 1e-6 {
        p / cos_lat - n
    } else {
        ecef.z / sin_lat.max(1e-6) - n * (1.0 - WGS84_E2)
    };
    GeodeticWgs84 {
        lat_deg: lat.to_degrees(),
        lon_deg: lon.to_degrees(),
        h_hae_m: h,
    }
}

#[inline]
pub fn geodetic_to_kartal_ned(geo: GeodeticWgs84) -> Option<Vec3Ned> {
    let datum = GeodeticWgs84 {
        lat_deg: KARTAL_DATUM_LAT_DEG,
        lon_deg: KARTAL_DATUM_LON_DEG,
        h_hae_m: KARTAL_DATUM_HAE_M,
    };
    let p = geodetic_to_ecef(geo);
    let o = geodetic_to_ecef(datum);
    let dx = p.x - o.x;
    let dy = p.y - o.y;
    let dz = p.z - o.z;

    let lat0 = datum.lat_deg.to_radians();
    let lon0 = datum.lon_deg.to_radians();
    let sl = lat0.sin();
    let cl = lat0.cos();
    let sn = lon0.sin();
    let cn = lon0.cos();

    let e = -sn * dx + cn * dy;
    let n = -sl * cn * dx - sl * sn * dy + cl * dz;
    let u = cl * cn * dx + cl * sn * dy + sl * dz;
    Vec3Ned::new(n, e, -u)
}

#[inline]
pub fn kartal_ned_to_geodetic(ned: Vec3Ned) -> GeodeticWgs84 {
    let datum = GeodeticWgs84 {
        lat_deg: KARTAL_DATUM_LAT_DEG,
        lon_deg: KARTAL_DATUM_LON_DEG,
        h_hae_m: KARTAL_DATUM_HAE_M,
    };
    let o = geodetic_to_ecef(datum);
    let lat0 = datum.lat_deg.to_radians();
    let lon0 = datum.lon_deg.to_radians();
    let sl = lat0.sin();
    let cl = lat0.cos();
    let sn = lon0.sin();
    let cn = lon0.cos();

    let e = ned.e.get();
    let n = ned.n.get();
    let u = -ned.d.get();

    let dx = -sn * e - sl * cn * n + cl * cn * u;
    let dy = cn * e - sl * sn * n + cl * sn * u;
    let dz = cl * n + sl * u;

    ecef_to_geodetic(Vec3Ecef {
        x: o.x + dx,
        y: o.y + dy,
        z: o.z + dz,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kartal_geodesy_round_trip_submillimeter() {
        let src = GeodeticWgs84 {
            lat_deg: KARTAL_HOSPITAL_LAT_DEG,
            lon_deg: KARTAL_HOSPITAL_LON_DEG,
            h_hae_m: KARTAL_DATUM_HAE_M + 30.0,
        };
        let ned = geodetic_to_kartal_ned(src).expect("valid NED");
        let back = kartal_ned_to_geodetic(ned);
        assert!((back.lat_deg - src.lat_deg).abs() < 1e-8);
        assert!((back.lon_deg - src.lon_deg).abs() < 1e-8);
        assert!((back.h_hae_m - src.h_hae_m).abs() < 1e-3);
        // Kartal Hospital is ~3375 m north and ~1165 m west of Kartal datum
        assert!((ned.n.get() - 3375.0).abs() < 20.0);
        assert!((ned.e.get() + 1165.0).abs() < 20.0);
    }
}
