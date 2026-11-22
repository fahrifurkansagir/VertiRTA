import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { HudRingStore } from "../src/ui/hud_store";
import { parseLaunchMode } from "../src/transport/flight_transport";

describe("VertiGCS dual-mode controller, ring-buffer HUD, and frozen Istanbul corridor GeoJSON", () => {
  it("resolves sandbox mode by default and enforces port isolation in live mode", () => {
    expect(parseLaunchMode("")).toEqual({
      mode: "sandbox",
      wsUrl: "ws://127.0.0.1:8891",
      adapter: "maplibre-deck",
    });
    expect(parseLaunchMode("?mode=live&ws=ws://127.0.0.1:5180/telemetry").mode).toBe("live");
    expect(() => parseLaunchMode("?mode=live&ws=ws://127.0.0.1:4173/telemetry")).toThrow(/forbidden/i);
    expect(() => parseLaunchMode("?mode=live&ws=ws://127.0.0.1:9222/telemetry")).toThrow(/forbidden/i);
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
