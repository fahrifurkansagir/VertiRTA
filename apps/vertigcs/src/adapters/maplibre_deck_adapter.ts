// Unified MapLibre GL JS v6 + @deck.gl/maplibre v9.4 Interleaved WebGL2 Adapter
// Locks:
//   - MapView ("mercator") mandatory for z >= 8 to neutralize @deck.gl/maplibre GlobeView culling defect
//   - devicePixelRatio clamped <= 1.5 for 4 GB VRAM GPUs (GTX 1650 SUPER)
//   - Zero paid API tokens (pmtiles:// + offline vector grid fallback; Copernicus GLO-30 attribution)
//   - Dedicated port isolation: hard-bans 4173, 4174, 8765, 9222, 9223

import { MapLibreOverlay } from "@deck.gl/maplibre";
import { ColumnLayer, GeoJsonLayer, PathLayer, ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import * as maplibregl from "maplibre-gl";
import maplibreWorkerAssetUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { Protocol } from "pmtiles";
import { buildFleetDeckLayers } from "../fleet/fleet_deck_layer";
import type { FleetSnapshot } from "../fleet/fleet_types";
import { buildIstanbulTacticalStyle } from "../geo/istanbul_coastline_style";
import { COPERNICUS_GLO30_ATTRIBUTION, KARTAL_DATUM } from "../geo/istanbul_datum";
import { buildM3DeckLayers } from "../m3/m3_deck_layer";
import type { M3Snapshot } from "../m3/m3_types";
import type { TelemetrySnapshot } from "../types/telemetry";
import type { GeoViewportOptions, IGeoViewportAdapter } from "./geo_viewport_adapter";

export const FORBIDDEN_PORTS = new Set<number>([4173, 4174, 8765, 9222, 9223]);
export const MAX_VERTIGCS_DPR = 1.5;

const COASTAL_CORRIDOR_RIBBONS: Array<{
  id: string;
  lane: "SE_70M" | "NW_105M";
  path: [number, number, number][];
}> = [
  {
    id: "corridor-se-70m",
    lane: "SE_70M",
    path: [
      [29.1066, 40.9401, 75],
      [29.1369, 40.9168, 76],
      [29.1532, 40.9038, 78],
      [29.1866, 40.8860, 82],
    ],
  },
  {
    id: "corridor-nw-105m",
    lane: "NW_105M",
    path: [
      [29.1056, 40.9394, 110],
      [29.1359, 40.9160, 111],
      [29.1522, 40.9030, 113],
      [29.1856, 40.8852, 117],
    ],
  },
];

const HOSPITAL_KEEPOUT_CYLINDER = [
  {
    id: "HELIPAD_KARTAL_HOSPITAL_CYL",
    position: [29.17417, 40.91639] as [number, number],
    elevationM: 145,
  },
];

interface TacticalLabelDatum {
  text: string;
  position: [number, number, number];
  color: [number, number, number, number];
  borderColor: [number, number, number, number];
  size: number;
}

const TACTICAL_LABELS: TacticalLabelDatum[] = [
  {
    text: "LTFJ CTR · 2500 ft",
    position: [29.208, 40.884, 815],
    color: [255, 198, 198, 255],
    borderColor: [255, 82, 82, 235],
    size: 11,
  },
  {
    text: "LTBX CTR · 2000 ft",
    position: [29.172, 40.936, 650],
    color: [255, 234, 168, 255],
    borderColor: [255, 193, 7, 235],
    size: 11,
  },
  {
    text: "HOSPITAL · 150m KEEPOUT",
    position: [29.1675, 40.9135, 190],
    color: [255, 208, 224, 255],
    borderColor: [255, 64, 129, 245],
    size: 10.5,
  },
  {
    text: "KARTAL VERTIPORT",
    position: [29.188, 40.8825, 32],
    color: [186, 248, 255, 255],
    borderColor: [0, 229, 255, 225],
    size: 10.5,
  },
  {
    text: "CEVIZLI",
    position: [29.1545, 40.9038, 48],
    color: [204, 242, 252, 245],
    borderColor: [56, 189, 248, 180],
    size: 10,
  },
  {
    text: "DRAGOS",
    position: [29.13822, 40.91686, 48],
    color: [204, 242, 252, 245],
    borderColor: [56, 189, 248, 180],
    size: 10,
  },
  {
    text: "MALTEPE",
    position: [29.1079, 40.94016, 48],
    color: [186, 248, 255, 255],
    borderColor: [0, 229, 255, 210],
    size: 10.5,
  },
  {
    text: "BUYUKADA",
    position: [29.124, 40.861, 18],
    color: [148, 180, 208, 220],
    borderColor: [56, 98, 135, 160],
    size: 9.5,
  },
];

export function resolveMapLibreWorkerUrl(
  assetUrl: string = maplibreWorkerAssetUrl,
  pageHref?: string
): string {
  const base =
    pageHref ??
    (typeof window !== "undefined" && window.location?.href
      ? window.location.href
      : "http://127.0.0.1:5180/");
  return new URL(assetUrl, base).href;
}

export function selectProjectionMode(zoom: number): "mercator" | "globe" {
  // GlobeView restricted strictly to continental overview (z < 6); z >= 8 locked to MapView (mercator)
  return zoom < 6.0 ? "globe" : "mercator";
}

export function clampDevicePixelRatio(rawDpr: number): number {
  const safe = Number.isFinite(rawDpr) && rawDpr > 0 ? rawDpr : 1.0;
  return Math.min(safe, MAX_VERTIGCS_DPR);
}

export function validateLivePort(port: number): number {
  const p = Math.trunc(port);
  if (FORBIDDEN_PORTS.has(p)) {
    throw new Error(
      `Port ${p} is forbidden by VertiRTA port isolation policy (use 5180 or 8891)`
    );
  }
  if (p < 1024 || p > 65535) {
    throw new Error(`Port ${p} outside unprivileged TCP/UDP range [1024, 65535]`);
  }
  return p;
}

let pmtilesRegistered = false;

export class MapLibreDeckAdapter implements IGeoViewportAdapter {
  readonly adapterId = "maplibre-deck-v6-v9.4";

  private map: maplibregl.Map | null = null;
  private overlay: MapLibreOverlay | null = null;
  private airspaceUrl: string | null = null;
  private lastTelemetry: TelemetrySnapshot | null = null;
  private fleetSnapshot: FleetSnapshot | null = null;
  private m3Snapshot: M3Snapshot | null = null;
  private readonly trailCoords: [number, number, number][] = [];
  private readonly maxTrailPts = 240;

  async mount(opts: GeoViewportOptions): Promise<void> {
    if (typeof maplibregl.setWorkerUrl === "function") {
      maplibregl.setWorkerUrl(resolveMapLibreWorkerUrl());
    }
    if (!pmtilesRegistered) {
      const protocol = new Protocol();
      maplibregl.addProtocol("pmtiles", protocol.tile);
      pmtilesRegistered = true;
    }

    this.airspaceUrl = opts.airspaceGeoJsonUrl;

    const dpr = clampDevicePixelRatio(typeof window !== "undefined" ? window.devicePixelRatio : 1.0);
    const baseStyle: maplibregl.StyleSpecification = buildIstanbulTacticalStyle();

    this.map = new maplibregl.Map({
      container: opts.container,
      style: baseStyle,
      center: [KARTAL_DATUM.lonDeg - 0.030, KARTAL_DATUM.latDeg + 0.020],
      zoom: 12.05,
      pitch: 46,
      bearing: -16,
      pixelRatio: dpr,
      attributionControl: false,
    });

    this.map.addControl(
      new maplibregl.AttributionControl({
        compact: true,
        customAttribution: COPERNICUS_GLO30_ATTRIBUTION,
      })
    );

    this.overlay = new MapLibreOverlay({
      interleaved: true,
      layers: this.buildLayers(null),
    });
    this.map.addControl(this.overlay as unknown as maplibregl.IControl);

    this.map.on("zoomend", () => {
      if (!this.map) return;
      const mode = selectProjectionMode(this.map.getZoom());
      this.map.setProjection({ type: mode });
    });
  }

  updateTelemetry(snapshot: TelemetrySnapshot): void {
    this.lastTelemetry = snapshot;
    this.trailCoords.push([snapshot.lonDeg, snapshot.latDeg, snapshot.altAmslM]);
    if (this.trailCoords.length > this.maxTrailPts) {
      this.trailCoords.shift();
    }
    this.overlay?.setProps({
      layers: this.buildLayers(snapshot),
    });
  }

  updateFleetSnapshot(snapshot: FleetSnapshot | null): void {
    this.fleetSnapshot = snapshot;
    this.overlay?.setProps({
      layers: this.buildLayers(this.lastTelemetry),
    });
  }

  updateM3Snapshot(snapshot: M3Snapshot | null): void {
    this.m3Snapshot = snapshot;
    this.overlay?.setProps({
      layers: this.buildLayers(this.lastTelemetry),
    });
  }

  private buildLayers(snapshot: TelemetrySnapshot | null) {
    const layers = [];

    if (this.airspaceUrl) {
      layers.push(
        new GeoJsonLayer({
          id: "istanbul-corridor-airspace",
          data: this.airspaceUrl,
          stroked: true,
          filled: true,
          extruded: true,
          wireframe: true,
          getElevation: (f: { properties?: { id?: string; ceiling_amsl_m?: number } }) => {
            const id = f.properties?.id ?? "";
            if (id === "LTFJ_CTR") return 762;
            if (id === "LTBX_CTR") return 610;
            return 28;
          },
          getFillColor: (f: { properties?: { id?: string } }) => {
            const id = f.properties?.id ?? "";
            if (id === "LTFJ_CTR") return [239, 68, 68, 76];
            if (id === "LTBX_CTR") return [245, 158, 11, 56];
            if (id === "HELIPAD_KARTAL_HOSPITAL") return [255, 56, 96, 220];
            return [0, 229, 255, 210];
          },
          getLineColor: (f: { properties?: { id?: string } }) => {
            const id = f.properties?.id ?? "";
            if (id === "LTFJ_CTR") return [255, 92, 92, 255];
            if (id === "LTBX_CTR") return [255, 213, 79, 248];
            if (id === "HELIPAD_KARTAL_HOSPITAL") return [255, 138, 168, 255];
            return [165, 243, 252, 255];
          },
          lineWidthMinPixels: 2.6,
          pointRadiusMinPixels: 7,
        })
      );
    }

    layers.push(
      new PathLayer({
        id: "coastal-corridor-ribbon-glow",
        data: COASTAL_CORRIDOR_RIBBONS,
        getPath: (d: { path: [number, number, number][] }) => d.path,
        getColor: [0, 229, 255, 48],
        getWidth: 165,
        widthMinPixels: 10,
        capRounded: true,
        jointRounded: true,
      }),
      new PathLayer({
        id: "coastal-corridor-ribbon-core",
        data: COASTAL_CORRIDOR_RIBBONS,
        getPath: (d: { path: [number, number, number][] }) => d.path,
        getColor: (d: { lane: "SE_70M" | "NW_105M" }) =>
          d.lane === "SE_70M" ? [0, 230, 145, 215] : [0, 229, 255, 225],
        getWidth: 26,
        widthMinPixels: 2.5,
        capRounded: true,
        jointRounded: true,
      }),
      new ColumnLayer({
        id: "kartal-hospital-keepout-cylinder",
        data: HOSPITAL_KEEPOUT_CYLINDER,
        diskResolution: 32,
        radius: 150,
        extruded: true,
        stroked: true,
        getPosition: (d: { position: [number, number] }) => d.position,
        getElevation: (d: { elevationM: number }) => d.elevationM,
        getFillColor: [255, 56, 96, 108],
        getLineColor: [255, 122, 152, 255],
        lineWidthMinPixels: 2,
      })
    );

    if (this.trailCoords.length >= 2) {
      layers.push(
        new PathLayer({
          id: "evtol-trajectory-trail",
          data: [{ path: this.trailCoords }],
          getPath: (d: { path: [number, number, number][] }) => d.path,
          getColor: snapshot?.celLatched ? [255, 82, 82, 245] : [0, 229, 255, 240],
          getWidth: 10,
          widthMinPixels: 3,
          capRounded: true,
          updateTriggers: { getPath: [this.trailCoords.length], getColor: [snapshot?.celLatched] },
        })
      );
    }

    if (snapshot) {
      const posBuf = new Float32Array([snapshot.lonDeg, snapshot.latDeg, snapshot.altAmslM]);
      const colBuf = new Uint8Array(
        snapshot.celLatched ? [255, 61, 0, 255] : [0, 230, 118, 255]
      );
      layers.push(
        new ScatterplotLayer({
          id: "evtol-live-aircraft-halo",
          data: [{ position: [snapshot.lonDeg, snapshot.latDeg, snapshot.altAmslM] }],
          getPosition: (d: { position: [number, number, number] }) => d.position,
          getFillColor: snapshot.celLatched ? [255, 61, 0, 72] : [0, 230, 118, 68],
          getLineColor: snapshot.celLatched ? [255, 138, 101, 235] : [105, 240, 174, 210],
          getRadius: 110,
          radiusMinPixels: 15,
          stroked: true,
          lineWidthMinPixels: 1.5,
          updateTriggers: { getPosition: [snapshot.seq], getFillColor: [snapshot.celLatched] },
        }),
        new ScatterplotLayer({
          id: "evtol-live-aircraft",
          data: {
            length: 1,
            attributes: {
              getPosition: { value: posBuf, size: 3 },
              getFillColor: { value: colBuf, size: 4 },
            },
          },
          getRadius: Math.max(22, snapshot.aglM * 0.8),
          radiusMinPixels: 9,
          stroked: true,
          getLineColor: [255, 255, 255, 255],
          lineWidthMinPixels: 2.5,
          updateTriggers: { getPosition: [snapshot.seq], getFillColor: [snapshot.celLatched] },
        })
      );
    }

    if (this.fleetSnapshot) {
      layers.push(...buildFleetDeckLayers(this.fleetSnapshot));
    }

    if (this.m3Snapshot) {
      layers.push(...buildM3DeckLayers(this.m3Snapshot));
    }

    const labelsData: TacticalLabelDatum[] = snapshot
      ? [
          ...TACTICAL_LABELS,
          {
            text: snapshot.celLatched ? "EGO X8-240 · CEL SPIN" : "EGO X8-240",
            position: [snapshot.lonDeg, snapshot.latDeg, snapshot.altAmslM + 30],
            color: snapshot.celLatched ? [255, 214, 204, 255] : [185, 246, 202, 255],
            borderColor: snapshot.celLatched ? [255, 61, 0, 250] : [0, 230, 118, 240],
            size: 10.5,
          },
        ]
      : TACTICAL_LABELS;

    layers.push(
      new TextLayer<TacticalLabelDatum>({
        id: "istanbul-tactical-labels",
        data: labelsData,
        getPosition: (d) => d.position,
        getText: (d) => d.text,
        getColor: (d) => d.color,
        getSize: (d) => d.size,
        sizeUnits: "pixels",
        billboard: true,
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        fontWeight: 700,
        characterSet: "auto",
        background: true,
        getBackgroundColor: [7, 16, 28, 224],
        getBorderColor: (d) => d.borderColor,
        getBorderWidth: 1.2,
        backgroundPadding: [6, 3],
        getPixelOffset: [0, -14],
        parameters: { depthCompare: "always", depthWriteEnabled: false },
        updateTriggers: {
          getPosition: [snapshot?.seq ?? 0],
          getText: [snapshot?.celLatched ?? false],
        },
      })
    );

    return layers;
  }

  dispose(): void {
    if (this.map && this.overlay) {
      this.map.removeControl(this.overlay as unknown as maplibregl.IControl);
    }
    this.overlay = null;
    this.map?.remove();
    this.map = null;
  }
}
