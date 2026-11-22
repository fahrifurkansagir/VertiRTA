import type { TelemetrySnapshot } from "../types/telemetry";

export interface GeoViewportOptions {
  container: HTMLElement;
  airspaceGeoJsonUrl: string;
  pmtilesBaseUrl?: string;
}

export interface IGeoViewportAdapter {
  readonly adapterId: string;
  mount(opts: GeoViewportOptions): Promise<void>;
  updateTelemetry(snapshot: TelemetrySnapshot): void;
  dispose(): void;
}
