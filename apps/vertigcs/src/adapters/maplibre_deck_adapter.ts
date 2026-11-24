// Unified MapLibre GL JS v6 + @deck.gl/maplibre v9.4 Interleaved WebGL2 Adapter
// Locks:
//   - MapView ("mercator") mandatory for z >= 8 to neutralize @deck.gl/maplibre GlobeView culling defect
//   - devicePixelRatio clamped <= 1.5 for 4 GB VRAM GPUs (GTX 1650 SUPER)
//   - Zero paid API tokens (pmtiles:// + offline vector grid fallback; Copernicus GLO-30 attribution)
//   - Dedicated port isolation: hard-bans 4173, 4174, 8765, 9222, 9223

import { MapLibreOverlay } from "@deck.gl/maplibre";
import { GeoJsonLayer, PathLayer, ScatterplotLayer } from "@deck.gl/layers";
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
      `Port ${p} is forbidden by AEGIS-VTOL port isolation lock (use 5180 or 8891)`
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
      center: [KARTAL_DATUM.lonDeg - 0.025, KARTAL_DATUM.latDeg + 0.018],
      zoom: 12.2,
      pitch: 48,
      bearing: -18,
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
          getElevation: (f: { properties?: { id?: string; ceiling_amsl_m?: number } }) => {
            const id = f.properties?.id ?? "";
            if (id === "LTFJ_CTR") return 762;
            if (id === "LTBX_CTR") return 610;
            return 25;
          },
          getFillColor: (f: { properties?: { id?: string } }) => {
            const id = f.properties?.id ?? "";
            if (id === "LTFJ_CTR") return [229, 57, 53, 45];
            if (id === "LTBX_CTR") return [255, 179, 0, 40];
            if (id === "HELIPAD_KARTAL_HOSPITAL") return [255, 82, 82, 140];
            return [38, 198, 218, 160];
          },
          getLineColor: (f: { properties?: { id?: string } }) => {
            const id = f.properties?.id ?? "";
            if (id === "LTFJ_CTR") return [255, 82, 82, 220];
            if (id === "LTBX_CTR") return [255, 213, 79, 220];
            return [128, 222, 234, 240];
          },
          lineWidthMinPixels: 2,
          pointRadiusMinPixels: 6,
        })
      );
    }

    if (this.trailCoords.length >= 2) {
      layers.push(
        new PathLayer({
          id: "evtol-trajectory-trail",
          data: [{ path: this.trailCoords }],
          getPath: (d: { path: [number, number, number][] }) => d.path,
          getColor: snapshot?.celLatched ? [255, 112, 67, 230] : [0, 229, 255, 220],
          getWidth: 4,
          widthMinPixels: 2,
          updateTriggers: { getPath: [this.trailCoords.length] },
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
          id: "evtol-live-aircraft",
          data: {
            length: 1,
            attributes: {
              getPosition: { value: posBuf, size: 3 },
              getFillColor: { value: colBuf, size: 4 },
            },
          },
          getRadius: Math.max(12, snapshot.aglM * 0.6),
          radiusMinPixels: 8,
          stroked: true,
          getLineColor: [255, 255, 255, 255],
          lineWidthMinPixels: 2,
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
