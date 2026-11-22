// Enterprise BVLOS CesiumJS Bridge Adapter
// Preserves existing viz/cesium/ (:5180) intact behind the engine-neutral IGeoViewportAdapter contract.

import { validateLivePort } from "./maplibre_deck_adapter";
import type { TelemetrySnapshot } from "../types/telemetry";
import type { GeoViewportOptions, IGeoViewportAdapter } from "./geo_viewport_adapter";

export class CesiumBridgeAdapter implements IGeoViewportAdapter {
  readonly adapterId = "cesium-enterprise-bridge";
  private iframe: HTMLIFrameElement | null = null;
  private readonly cesiumUrl: string;

  constructor(cesiumBaseUrl = "http://127.0.0.1:5180") {
    const parsed = new URL(cesiumBaseUrl);
    if (parsed.port) {
      validateLivePort(Number(parsed.port));
    }
    this.cesiumUrl = cesiumBaseUrl;
  }

  async mount(opts: GeoViewportOptions): Promise<void> {
    const frame = document.createElement("iframe");
    frame.src = this.cesiumUrl;
    frame.title = "VertiRTA CesiumJS Enterprise 3D Globe";
    frame.style.width = "100%";
    frame.style.height = "100%";
    frame.style.border = "0";
    opts.container.replaceChildren(frame);
    this.iframe = frame;
  }

  updateTelemetry(snapshot: TelemetrySnapshot): void {
    this.iframe?.contentWindow?.postMessage(
      { type: "vertirta.telemetry", snapshot },
      "*"
    );
  }

  dispose(): void {
    this.iframe?.remove();
    this.iframe = null;
  }
}
