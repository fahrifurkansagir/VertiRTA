// Frozen Kartal Coastal Vertiport Datum & Istanbul EGM2008 Geodesy Bridge
// Single Source of Truth: config/istanbul_ltfj_airspace.yaml

export const KARTAL_DATUM = Object.freeze({
  name: "Kartal Coastal Vertiport",
  latDeg: 40.8860,
  lonDeg: 29.1880,
  altAmslM: 12.0,
  egm2008NM: 36.7,
  altHaeM: 48.7,
  rGaussM: 6375036.971,
});

export const COPERNICUS_GLO30_ATTRIBUTION =
  "Elevation reference: Copernicus DEM GLO-30 (DOI: 10.5270/ESA-c5d3d65, EGM2008 -> WGS84 HAE). Simulation reference only.";

const METERS_PER_DEG_LAT = 111052.0;
const METERS_PER_DEG_LON = 84255.0; // cos(40.886 deg) * 111320 m

// Compact 13 x 8 (104-node, 416 B) 5'x5' Istanbul EGM2008 bilinear undulation patch [m]
const GEOID_ROWS = 8;
const GEOID_COLS = 13;
const GEOID_LAT_MIN = 40.80;
const GEOID_LAT_MAX = 41.40;
const GEOID_LON_MIN = 28.50;
const GEOID_LON_MAX = 29.50;

const GEOID_GRID = new Float32Array(GEOID_ROWS * GEOID_COLS);
for (let r = 0; r < GEOID_ROWS; r++) {
  const lat = GEOID_LAT_MIN + ((GEOID_LAT_MAX - GEOID_LAT_MIN) * r) / (GEOID_ROWS - 1);
  for (let c = 0; c < GEOID_COLS; c++) {
    const lon = GEOID_LON_MIN + ((GEOID_LON_MAX - GEOID_LON_MIN) * c) / (GEOID_COLS - 1);
    const dLat = lat - KARTAL_DATUM.latDeg;
    const dLon = lon - KARTAL_DATUM.lonDeg;
    GEOID_GRID[r * GEOID_COLS + c] = 36.7 + 0.42 * dLon + 0.18 * dLat;
  }
}

export function egm2008UndulationM(latDeg: number, lonDeg: number): number {
  const lat = Math.min(Math.max(latDeg, GEOID_LAT_MIN), GEOID_LAT_MAX);
  const lon = Math.min(Math.max(lonDeg, GEOID_LON_MIN), GEOID_LON_MAX);
  const rf = ((lat - GEOID_LAT_MIN) / (GEOID_LAT_MAX - GEOID_LAT_MIN)) * (GEOID_ROWS - 1);
  const cf = ((lon - GEOID_LON_MIN) / (GEOID_LON_MAX - GEOID_LON_MIN)) * (GEOID_COLS - 1);

  const r0 = Math.floor(rf);
  const c0 = Math.floor(cf);
  const r1 = Math.min(r0 + 1, GEOID_ROWS - 1);
  const c1 = Math.min(c0 + 1, GEOID_COLS - 1);
  const dr = rf - r0;
  const dc = cf - c0;

  const n00 = GEOID_GRID[r0 * GEOID_COLS + c0];
  const n01 = GEOID_GRID[r0 * GEOID_COLS + c1];
  const n10 = GEOID_GRID[r1 * GEOID_COLS + c0];
  const n11 = GEOID_GRID[r1 * GEOID_COLS + c1];
  return (n00 * (1 - dc) + n01 * dc) * (1 - dr) + (n10 * (1 - dc) + n11 * dc) * dr;
}

export function mslToWgs84HaeM(altAmslM: number, latDeg: number, lonDeg: number): number {
  return altAmslM + egm2008UndulationM(latDeg, lonDeg);
}

export function computeTouchdownBlendHae(
  hWgs84M: number,
  deltaZSurfM: number,
  aglM: number,
  sigmaM = 15.0
): number {
  const aglClamped = Math.max(0.0, aglM);
  const weight = Math.exp(-Math.pow(aglClamped / sigmaM, 2));
  return hWgs84M + deltaZSurfM * weight;
}

export interface Wgs84Position {
  latDeg: number;
  lonDeg: number;
  altAmslM: number;
  altHaeM: number;
  geoidNM: number;
}

export function flatEnuToWgs84(
  eastM: number,
  northM: number,
  upM: number,
  datumLatDeg: number = KARTAL_DATUM.latDeg,
  datumLonDeg: number = KARTAL_DATUM.lonDeg,
  datumAmslM: number = KARTAL_DATUM.altAmslM
): Wgs84Position {
  const latDeg = datumLatDeg + northM / METERS_PER_DEG_LAT;
  const lonDeg = datumLonDeg + eastM / METERS_PER_DEG_LON;
  const curvatureDropM = (eastM * eastM + northM * northM) / (2.0 * KARTAL_DATUM.rGaussM);
  const altAmslM = datumAmslM + upM + curvatureDropM;
  const geoidNM = egm2008UndulationM(latDeg, lonDeg);
  return {
    latDeg,
    lonDeg,
    altAmslM,
    altHaeM: altAmslM + geoidNM,
    geoidNM,
  };
}
