// SPDX-License-Identifier: Apache-2.0
// Zero-token offline Istanbul Anatolian Marmara Coastline & Princes' Islands MapLibre GL v6 Style.
// Guarantees crisp geographic context (Maltepe - Dragos - Cevizli - Kartal - Pendik - Adalar)
// with zero external tile-server 503s and zero paid API keys.

import type { StyleSpecification } from "maplibre-gl";

const ANATOLIAN_LAND_GEOJSON = {
  type: "FeatureCollection" as const,
  features: [
    {
      type: "Feature" as const,
      properties: { name: "Anatolian Peninsula (Kadikoy-Maltepe-Kartal-Pendik-Tuzla)" },
      geometry: {
        type: "Polygon" as const,
        coordinates: [
          [
            [29.010, 41.040],
            [29.400, 41.040],
            [29.400, 40.815],
            [29.312, 40.818],
            [29.295, 40.838],
            [29.272, 40.858],
            [29.245, 40.870],
            [29.218, 40.876],
            [29.195, 40.882],
            [29.188, 40.886], // Kartal Coastal Vertiport
            [29.170, 40.893],
            [29.154, 40.903], // Cevizli Coast
            [29.138, 40.916], // Dragos Coast
            [29.124, 40.922],
            [29.108, 40.939], // Maltepe Orhangazi Park (reclaimed fill)
            [29.092, 40.950],
            [29.076, 40.961], // Bostanci
            [29.058, 40.969],
            [29.035, 40.980],
            [29.020, 40.992], // Kadikoy
            [29.010, 41.040],
          ],
        ],
      },
    },
    {
      type: "Feature" as const,
      properties: { name: "Buyukada" },
      geometry: {
        type: "Polygon" as const,
        coordinates: [
          [
            [29.115, 40.877],
            [29.136, 40.874],
            [29.139, 40.856],
            [29.126, 40.842],
            [29.113, 40.849],
            [29.112, 40.866],
            [29.115, 40.877],
          ],
        ],
      },
    },
    {
      type: "Feature" as const,
      properties: { name: "Heybeliada" },
      geometry: {
        type: "Polygon" as const,
        coordinates: [
          [
            [29.085, 40.883],
            [29.103, 40.881],
            [29.105, 40.867],
            [29.091, 40.864],
            [29.082, 40.872],
            [29.085, 40.883],
          ],
        ],
      },
    },
    {
      type: "Feature" as const,
      properties: { name: "Burgazada" },
      geometry: {
        type: "Polygon" as const,
        coordinates: [
          [
            [29.058, 40.888],
            [29.073, 40.885],
            [29.072, 40.873],
            [29.057, 40.876],
            [29.058, 40.888],
          ],
        ],
      },
    },
    {
      type: "Feature" as const,
      properties: { name: "Kinaliada" },
      geometry: {
        type: "Polygon" as const,
        coordinates: [
          [
            [29.042, 40.918],
            [29.056, 40.916],
            [29.055, 40.903],
            [29.043, 40.905],
            [29.042, 40.918],
          ],
        ],
      },
    },
  ],
};

const COASTAL_ARTERIALS_GEOJSON = {
  type: "FeatureCollection" as const,
  features: [
    {
      type: "Feature" as const,
      properties: { kind: "sahil_yolu", name: "Turgut Ozal Sahil Bulvari" },
      geometry: {
        type: "LineString" as const,
        coordinates: [
          [29.078, 40.960],
          [29.108, 40.940],
          [29.138, 40.917],
          [29.155, 40.904],
          [29.188, 40.886],
          [29.232, 40.874],
        ],
      },
    },
    {
      type: "Feature" as const,
      properties: { kind: "d100", name: "D-100 Highway" },
      geometry: {
        type: "LineString" as const,
        coordinates: [
          [29.085, 40.972],
          [29.126, 40.948],
          [29.157, 40.928],
          [29.176, 40.916],
          [29.205, 40.901],
          [29.255, 40.885],
        ],
      },
    },
    {
      type: "Feature" as const,
      properties: { kind: "runway", name: "LTFJ RWY 06L/24R" },
      geometry: {
        type: "LineString" as const,
        coordinates: [
          [29.293, 40.892],
          [29.325, 40.905],
        ],
      },
    },
  ],
};

function buildTacticalGraticuleGeoJson() {
  const features: Array<{
    type: "Feature";
    properties: { axis: "lat" | "lon" };
    geometry: { type: "LineString"; coordinates: [number, number][] };
  }> = [];

  for (let lon = 29.0; lon <= 29.381; lon += 0.02) {
    const x = Number(lon.toFixed(4));
    features.push({
      type: "Feature",
      properties: { axis: "lon" },
      geometry: {
        type: "LineString",
        coordinates: [
          [x, 40.82],
          [x, 41.02],
        ],
      },
    });
  }
  for (let lat = 40.82; lat <= 41.021; lat += 0.02) {
    const y = Number(lat.toFixed(4));
    features.push({
      type: "Feature",
      properties: { axis: "lat" },
      geometry: {
        type: "LineString",
        coordinates: [
          [29.0, y],
          [29.38, y],
        ],
      },
    });
  }
  return {
    type: "FeatureCollection" as const,
    features,
  };
}

const TACTICAL_GRATICULE_GEOJSON = buildTacticalGraticuleGeoJson();

export function buildIstanbulTacticalStyle(): StyleSpecification {
  return {
    version: 8,
    name: "VertiGCS Zero-Token Istanbul Anatolian Tactical Dark",
    sources: {
      "istanbul-graticule": {
        type: "geojson",
        data: TACTICAL_GRATICULE_GEOJSON,
      },
      "istanbul-land": {
        type: "geojson",
        data: ANATOLIAN_LAND_GEOJSON,
      },
      "istanbul-arterials": {
        type: "geojson",
        data: COASTAL_ARTERIALS_GEOJSON,
      },
    },
    layers: [
      {
        id: "bg-marmara-sea",
        type: "background",
        paint: { "background-color": "#071424" },
      },
      {
        id: "tactical-graticule-sea",
        type: "line",
        source: "istanbul-graticule",
        paint: {
          "line-color": "#16314c",
          "line-width": 1,
          "line-opacity": 0.55,
        },
      },
      {
        id: "anatolian-land-fill",
        type: "fill",
        source: "istanbul-land",
        paint: {
          "fill-color": "#172c43",
          "fill-opacity": 0.96,
        },
      },
      {
        id: "tactical-graticule-land",
        type: "line",
        source: "istanbul-graticule",
        paint: {
          "line-color": "#213f5d",
          "line-width": 0.8,
          "line-opacity": 0.32,
        },
      },
      {
        id: "anatolian-shoreline-glow",
        type: "line",
        source: "istanbul-land",
        paint: {
          "line-color": "#297cb3",
          "line-width": 6,
          "line-opacity": 0.28,
          "line-blur": 2.5,
        },
      },
      {
        id: "anatolian-shoreline-stroke",
        type: "line",
        source: "istanbul-land",
        paint: {
          "line-color": "#5cb6ec",
          "line-width": 2.0,
        },
      },
      {
        id: "coastal-arterials-line",
        type: "line",
        source: "istanbul-arterials",
        paint: {
          "line-color": [
            "match",
            ["get", "kind"],
            "runway",
            "#c3def7",
            "sahil_yolu",
            "#4e88b5",
            "#375f83",
          ],
          "line-width": [
            "match",
            ["get", "kind"],
            "runway",
            3.5,
            "sahil_yolu",
            2.1,
            1.6,
          ],
          "line-dasharray": [3, 2],
        },
      },
    ],
  };
}
