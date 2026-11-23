import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { resolveMapLibreWorkerUrl } from "../src/adapters/maplibre_deck_adapter";
import { buildIstanbulTacticalStyle } from "../src/geo/istanbul_coastline_style";
import { HudRingStore } from "../src/ui/hud_store";
import {
  createReplayFaultCoordinator,
  parseLaunchMode,
  resolvePublicAssetUrl,
  resolveWasmModuleUrl,
  type FlightTransport,
} from "../src/transport/flight_transport";
import type { FaultCommand } from "../src/types/telemetry";

describe("VertiGCS dual-mode controller, ring-buffer HUD, and frozen Istanbul corridor GeoJSON", () => {
  it("resolves sandbox mode by default with /telemetry WS path and enforces port isolation", () => {
    expect(parseLaunchMode("")).toEqual({
      mode: "sandbox",
      wsUrl: "ws://127.0.0.1:8891/telemetry",
      adapter: "maplibre-deck",
    });
    expect(parseLaunchMode("?mode=live&ws=ws://127.0.0.1:5180/telemetry").mode).toBe("live");
    expect(() => parseLaunchMode("?mode=live&ws=ws://127.0.0.1:4173/telemetry")).toThrow(/forbidden/i);
    expect(() => parseLaunchMode("?mode=live&ws=ws://127.0.0.1:9222/telemetry")).toThrow(/forbidden/i);
  });

  it("resolves MapLibre GL v6 worker URL for Pages /VertiRTA/ subpath and uses zero-token offline coastal style", () => {
    expect(
      resolveMapLibreWorkerUrl(
        "/VertiRTA/assets/maplibre-gl-worker-abc.mjs",
        "https://fahrifurkansagir.github.io/VertiRTA/"
      )
    ).toBe("https://fahrifurkansagir.github.io/VertiRTA/assets/maplibre-gl-worker-abc.mjs");
    expect(
      resolveMapLibreWorkerUrl("./assets/maplibre-gl-worker-abc.mjs", "http://127.0.0.1:5180/")
    ).toBe("http://127.0.0.1:5180/assets/maplibre-gl-worker-abc.mjs");

    const style = buildIstanbulTacticalStyle();
    expect(style.sources).toHaveProperty("istanbul-land");
    expect(style.sources).toHaveProperty("istanbul-arterials");
    expect(JSON.stringify(style)).not.toMatch(/https?:\/\//i); // 100% offline zero-token GeoJSON
  });

  it("exits .vrta replay and delivers the injected fault on the very first click", () => {
    const events: string[] = [];
    const faults: FaultCommand[] = [];
    const fakeTransport: FlightTransport = {
      start() {
        events.push("start");
      },
      stop() {
        events.push("stop");
      },
      sendFault(cmd) {
        events.push(`fault:${cmd.kind}:${cmd.targetIdx}`);
        faults.push(cmd);
      },
    };
    let replayStopped = 0;
    const coord = createReplayFaultCoordinator(
      fakeTransport,
      () => {
        replayStopped++;
        events.push("stopReplay");
      },
      () => {},
      () => {}
    );

    coord.enterReplay();
    expect(coord.isTransportActive()).toBe(false);
    coord.dispatchFault({ kind: "same_arm", targetIdx: 0 });
    expect(replayStopped).toBe(1);
    expect(coord.isTransportActive()).toBe(true);
    expect(faults).toEqual([{ kind: "same_arm", targetIdx: 0 }]);
    expect(events).toEqual(["stop", "stopReplay", "start", "fault:same_arm:0"]);
  });

  it("resolves GitHub Pages /VertiRTA/ subpath and local ./ asset & WASM URLs cleanly", () => {
    expect(resolvePublicAssetUrl("data/corridor_faults.vrta", "/VertiRTA/")).toBe(
      "/VertiRTA/data/corridor_faults.vrta"
    );
    expect(resolvePublicAssetUrl("data/istanbul_corridor_airspace.geojson", "./")).toBe(
      "./data/istanbul_corridor_airspace.geojson"
    );
    expect(
      resolveWasmModuleUrl(
        "/VertiRTA/",
        false,
        "https://fahrifurkansagir.github.io/VertiRTA/assets/flight_sim.worker-abc.js"
      )
    ).toBe("https://fahrifurkansagir.github.io/VertiRTA/wasm/libvertirta_flight_core.mjs");
    expect(
      resolveWasmModuleUrl("./", false, "http://127.0.0.1:5180/assets/flight_sim.worker-abc.js")
    ).toBe("http://127.0.0.1:5180/wasm/libvertirta_flight_core.mjs");
    expect(
      resolveWasmModuleUrl("./", true, "http://127.0.0.1:5180/src/worker/flight_sim.worker.ts")
    ).toBe("http://127.0.0.1:5180/wasm/libvertirta_flight_core.mjs");

    const html = readFileSync(resolve(__dirname, "../index.html"), "utf-8");
    expect(html).toContain('id="btn-load-sample-vrta"');
    expect(html).toContain("Load Sample Fault Flight");
  });

  it("bounds ring-buffer telemetry history to fixed capacity without unbounded allocations", () => {
    const store = new HudRingStore(64);
    for (let i = 0; i < 200; i++) {
      store.push({
        seq: i,
        simTimeS: i * 0.01,
        vTermV: 400 - i * 0.05,
        socPct: 80 - i * 0.01,
        cbvfMinV: 2.5,
        aglM: 35,
      });
    }
    expect(store.length).toBe(64);
    expect(store.latest()?.seq).toBe(199);
    expect(store.oldest()?.seq).toBe(136);
  });

  it("bundles exact LTFJ NW J2 pinch, LTBX, Kartal/Dragos/Maltepe/Cevizli pads, and Hospital helipad", () => {
    const raw = readFileSync(
      resolve(__dirname, "../public/data/istanbul_corridor_airspace.geojson"),
      "utf-8"
    );
    const fc = JSON.parse(raw);
    const ids = new Set(fc.features.map((f: { properties: { id: string } }) => f.properties.id));
    for (const req of [
      "LTFJ_CTR",
      "LTBX_CTR",
      "SLZ_KARTAL",
      "SLZ_DRAGOS",
      "SLZ_MALTEPE",
      "SLZ_CEVIZLI",
      "HELIPAD_KARTAL_HOSPITAL",
    ]) {
      expect(ids.has(req)).toBe(true);
    }
    expect(raw).not.toContain("29.2057"); // ISTMarina inside LTFJ CTR hard-banned
  });
});
