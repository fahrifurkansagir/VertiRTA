// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { buildFleetDeckLayers } from "../src/fleet/fleet_deck_layer";
import { FleetCorridorController } from "../src/fleet/fleet_sim_replay";

describe("VertiGCS 20-eVTOL U-Space Fleet & Part-IS Cyber-Attack Simulator", () => {
  it("steps 20 simultaneous eVTOLs strictly west of 29.1895 E with 15 C chilled vertiport stands", () => {
    const fleet = new FleetCorridorController(20);
    for (let i = 0; i < 60; i++) fleet.step(0.1);
    const snap = fleet.snapshot();
    expect(snap.aircraft.length).toBe(20);
    expect(snap.vertiports.length).toBe(4);
    for (const ac of snap.aircraft) {
      expect(ac.lonDeg).toBeLessThan(29.1895);
      expect(ac.aglM).toBeLessThanOrEqual(120.5);
    }
    for (const vp of snap.vertiports) {
      expect(vp.chillerCoolantC).toBe(15);
      expect(vp.standsTotal).toBe(4);
    }

    const layers = buildFleetDeckLayers(snap);
    expect(layers.length).toBe(2);
  });

  it("demonstrates <500 ms CEL preemption, V2V ReplayWindow128 drop, and LTFJ USSP quarantine RTB <= 10 s", () => {
    const fleet = new FleetCorridorController(20);
    const p0 = fleet.triggerCelPreemption(0);
    expect(p0.preemptedPeers).toBeGreaterThanOrEqual(1);
    expect(p0.v2vLatencyMs).toBeLessThan(500);
    expect(buildFleetDeckLayers(fleet.snapshot()).length).toBe(3);

    const replayEvt = fleet.injectPartIsReplayAttack(3);
    expect(replayEvt.droppedByReplayWindow128).toBe(true);
    expect(replayEvt.cyberState).toBe("SEC_SUSPECT_PEER");

    const spoofEvt = fleet.injectSpoofedUsspIntoLtfj(1);
    expect(spoofEvt.cyberState).toBe("SEC_USSP_QUARANTINE");
    expect(spoofEvt.rtoSec).toBeLessThanOrEqual(10.0);
    expect(spoofEvt.isOr230DraftTriggered).toBe(true);

    // Advance 10.5 s -> transitions SEC_USSP_QUARANTINE -> SEC_AUTONOMOUS_RTB
    for (let i = 0; i < 105; i++) fleet.step(0.1);
    expect(fleet.snapshot().aircraft[1].cyberState).toBe("SEC_AUTONOMOUS_RTB");
  });
});
