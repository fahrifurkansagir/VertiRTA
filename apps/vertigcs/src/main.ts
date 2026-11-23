import "maplibre-gl/dist/maplibre-gl.css";

import { CesiumBridgeAdapter } from "./adapters/cesium_bridge_adapter";
import type { IGeoViewportAdapter } from "./adapters/geo_viewport_adapter";
import { MapLibreDeckAdapter, clampDevicePixelRatio } from "./adapters/maplibre_deck_adapter";
import { bindFleetPanel } from "./fleet/fleet_panel";
import {
  LiveWsTransport,
  SandboxWorkerTransport,
  createReplayFaultCoordinator,
  parseLaunchMode,
  resolvePublicAssetUrl,
  type FlightTransport,
} from "./transport/flight_transport";
import type { TelemetrySnapshot } from "./types/telemetry";
import { bindFaultPanel } from "./ui/fault_panel";
import { HudRingStore } from "./ui/hud_store";
import { VrtaReplayScrubber } from "./ui/replay_scrubber";
import { TelemetryHud } from "./ui/telemetry_hud";

export { parseLaunchMode };

async function bootstrapVertiGcs(): Promise<void> {
  const container = document.getElementById("map-viewport");
  if (!container) return;

  const cfg = parseLaunchMode(window.location.search);
  const badge = document.getElementById("mode-badge");
  if (badge) {
    badge.textContent = cfg.mode === "live" ? "LIVE WS OPS" : "SANDBOX WASM";
  }

  const auditEl = document.getElementById("z3-audit-log");
  const setStatus = (msg: string) => {
    if (auditEl) {
      const dpr = clampDevicePixelRatio(window.devicePixelRatio || 1);
      auditEl.textContent = `Adapter: ${cfg.adapter} (MapView z>=8, DPR=${dpr.toFixed(1)})\n${msg}`;
    }
  };

  const viewport: IGeoViewportAdapter =
    cfg.adapter === "cesium-bridge"
      ? new CesiumBridgeAdapter("http://127.0.0.1:5180")
      : new MapLibreDeckAdapter();

  await viewport.mount({
    container,
    airspaceGeoJsonUrl: resolvePublicAssetUrl("data/istanbul_corridor_airspace.geojson"),
  });

  const hud = new TelemetryHud();
  const ring = new HudRingStore(300);

  let pendingSnap: TelemetrySnapshot | null = null;
  let rafScheduled = false;

  const scheduleRender = (snap: TelemetrySnapshot) => {
    pendingSnap = snap;
    ring.push({
      seq: snap.seq,
      simTimeS: snap.simTimeS,
      vTermV: snap.vTermV,
      socPct: snap.socPct,
      cbvfMinV: snap.cbvfMinV,
      aglM: snap.aglM,
    });
    if (!rafScheduled) {
      rafScheduled = true;
      window.requestAnimationFrame(() => {
        rafScheduled = false;
        if (pendingSnap) {
          hud.render(pendingSnap);
          viewport.updateTelemetry(pendingSnap);
        }
      });
    }
  };

  const transport: FlightTransport =
    cfg.mode === "live" ? new LiveWsTransport(cfg.wsUrl) : new SandboxWorkerTransport();

  transport.start(scheduleRender, setStatus);

  let stopReplayFn: () => void = () => {};
  const replayFaultCoord = createReplayFaultCoordinator(
    transport,
    () => stopReplayFn(),
    scheduleRender,
    setStatus
  );

  const scrubber = new VrtaReplayScrubber((snap) => {
    replayFaultCoord.enterReplay();
    scheduleRender(snap);
  }, setStatus);
  stopReplayFn = () => scrubber.stop();

  bindFaultPanel((cmd) => {
    replayFaultCoord.dispatchFault(cmd);
  });

  bindFleetPanel((fleetSnap) => {
    if (viewport instanceof MapLibreDeckAdapter) {
      viewport.updateFleetSnapshot(fleetSnap);
    }
  }, setStatus);
}

if (typeof document !== "undefined") {
  void bootstrapVertiGcs();
}
