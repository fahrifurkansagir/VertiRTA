import { describe, expect, it } from "vitest";
import {
  COPERNICUS_GLO30_ATTRIBUTION,
  KARTAL_DATUM,
  computeTouchdownBlendHae,
  egm2008UndulationM,
  flatEnuToWgs84,
  mslToWgs84HaeM,
} from "../src/geo/istanbul_datum";
import {
  clampDevicePixelRatio,
  selectProjectionMode,
  validateLivePort,
} from "../src/adapters/maplibre_deck_adapter";
import { CesiumBridgeAdapter } from "../src/adapters/cesium_bridge_adapter";

describe("Istanbul geodesy, GlobeView culling guard, DPR cap, and port isolation", () => {
  it("anchors Kartal datum to 40.8860 N, 29.1880 E, +36.7 m EGM2008 and Gaussian touchdown blend", () => {
    expect(KARTAL_DATUM.latDeg).toBe(40.8860);
    expect(KARTAL_DATUM.lonDeg).toBe(29.1880);
    expect(KARTAL_DATUM.altAmslM).toBe(12.0);
    expect(COPERNICUS_GLO30_ATTRIBUTION).toContain("10.5270/ESA-c5d3d65");
    expect(egm2008UndulationM(40.8860, 29.1880)).toBeCloseTo(36.7, 2);
    expect(mslToWgs84HaeM(12.0, 40.8860, 29.1880)).toBeCloseTo(48.7, 2);
    expect(computeTouchdownBlendHae(48.7, -1.2, 0.0)).toBeCloseTo(47.5, 4);
    expect(computeTouchdownBlendHae(48.7, -1.2, 60.0)).toBeCloseTo(48.7, 4);

    const wgs = flatEnuToWgs84(0.0, 0.0, 35.0);
    expect(wgs.latDeg).toBeCloseTo(40.8860, 5);
    expect(wgs.lonDeg).toBeCloseTo(29.1880, 5);
    expect(wgs.altHaeM).toBeCloseTo(83.7, 1);
  });

  it("enforces MapView for z >= 8 (neutralizing @deck.gl/maplibre GlobeView culling), DPR <= 1.5, and port isolation", () => {
    expect(selectProjectionMode(12.5)).toBe("mercator");
    expect(selectProjectionMode(8.0)).toBe("mercator");
    expect(selectProjectionMode(5.5)).toBe("globe");
    expect(clampDevicePixelRatio(3.0)).toBe(1.5);
    expect(clampDevicePixelRatio(1.0)).toBe(1.0);
    expect(validateLivePort(5180)).toBe(5180);
    expect(validateLivePort(8891)).toBe(8891);
    for (const bad of [4173, 4174, 8765, 9222, 9223]) {
      expect(() => validateLivePort(bad)).toThrow(/forbidden/i);
    }

    const cesiumAdapter = new CesiumBridgeAdapter("http://127.0.0.1:5180");
    expect(cesiumAdapter.adapterId).toBe("cesium-enterprise-bridge");
  });
});
