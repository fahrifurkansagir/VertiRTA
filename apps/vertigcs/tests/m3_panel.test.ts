// SPDX-License-Identifier: Apache-2.0
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import { buildM3CtrInflationData, buildM3DeckLayers } from "../src/m3/m3_deck_layer";
import {
  buildE5xDownloadDescriptor,
  buildSampleSignedSidecarJsonl,
  verifyVrtaEventsJsonlChain,
} from "../src/m3/m3_forensics_inspector";
import { renderM3PanelDom, type M3PanelElements } from "../src/m3/m3_panel";
import { M3SimReplay } from "../src/m3/m3_sim_replay";
import { VERIFIED_LTBX_ARP } from "../src/m3/m3_types";

describe("VertiGCS M3 Mission Control panel, Deck.gl layers & .e5x draft contracts", () => {
  it("drives wire grazing (<=15 m/s), H3 slow-ramp spoof (<=1.4 s + rollback), and water drift -> VISUAL_SLZ_ABORT", () => {
    const replay = new M3SimReplay();
    const nom = replay.reset();
    expect(nom.forwardHazard.wireCap).toBe(64);
    expect(nom.forwardHazard.intruderCap).toBe(16);
    expect(nom.forwardHazard.vMaxCapMps).toBe(22.0);
    expect(nom.forwardHazard.enableFp16).toBe(false);

    const grazing = replay.activateScenario("wire_grazing");
    expect(grazing.forwardHazard.grazingFallback).toBe(true);
    expect(grazing.forwardHazard.vMaxCapMps).toBeLessThanOrEqual(15.0);

    const spoof = replay.activateScenario("slow_ramp_spoof");
    expect(spoof.gnssDenied.hypothesis).toBe("H3_SLOW_RAMP_SPOOF");
    expect(spoof.gnssDenied.rollbackActive).toBe(true);
    expect(spoof.gnssDenied.ttdSec).not.toBeNull();
    expect(spoof.gnssDenied.ttdSec!).toBeLessThanOrEqual(1.4);

    let water = replay.activateScenario("water_drift");
    expect(water.gnssDenied.navFsmState).toBe("DEGRADED_CAUTION");
    expect(water.gnssDenied.vMaxCapMps).toBeLessThanOrEqual(15.0);
    expect(water.gnssDenied.dhDtMps).toBeGreaterThan(0);

    for (let i = 0; i < 30; ++i) {
      water = replay.step(1.0);
    }
    expect(water.gnssDenied.sigmaPosM).toBeGreaterThanOrEqual(65.0);
    expect(water.gnssDenied.navFsmState).toBe("DETERMINISTIC_VISUAL_SLZ_ABORT");
    expect(water.gnssDenied.relocBackend).toBe("MegaLoc");
  });

  it("locks Voice MAYDAY S3_Garbled preemption (<250 ms TTFA) and draft/Incident ID 431 .e5x preview", () => {
    const replay = new M3SimReplay();
    const mayday = replay.activateScenario("voice_mayday");
    expect(mayday.voiceAtc.snrStratum).toBe("S3_Garbled");
    expect(mayday.voiceAtc.maydayPreempted).toBe(true);
    expect(mayday.voiceAtc.maydayPosterior).toBeGreaterThanOrEqual(0.15);
    expect(mayday.voiceAtc.ttsTtfaMs).toBeLessThan(250.0);

    expect(mayday.e5xDraft.occurrenceClassId431).toBe("DRAFT_INCIDENT_ONLY");
    expect(mayday.e5xDraft.cicttCategory).toBe("SCF-NP");
    expect(mayday.e5xDraft.barrier8Score).toBe(0);
    expect(mayday.e5xDraft.autoClassifiedAccident).toBe(false);
    expect(mayday.e5xDraft.numericHallucinationRate).toBe(0.0);
    expect(mayday.e5xDraft.xmlSnippet).toContain('attributeId="431">DRAFT_INCIDENT_ONLY');
    expect(mayday.e5xDraft.xmlSnippet).toContain('attributeId="659">SCF-NP');
  });

  it("builds Deck.gl wire & inflated CTR layers using verified DHMI AIP LTBX ARP (40.990556 N, 29.216111 E)", () => {
    expect(VERIFIED_LTBX_ARP.latDeg).toBeCloseTo(40.990556, 6);
    expect(VERIFIED_LTBX_ARP.lonDeg).toBeCloseTo(29.216111, 6);

    const replay = new M3SimReplay();
    const water = replay.activateScenario("water_drift");
    const ctrData = buildM3CtrInflationData(water);
    expect(ctrData.map((d) => d.id)).toEqual(["LTFJ_CTR_INFLATED", "LTBX_CTR_INFLATED"]);
    expect(ctrData[1].centerLatDeg).toBeCloseTo(40.990556, 6);
    expect(ctrData[1].inflatedRadiusM).toBeGreaterThan(9260.0 + 80.0);

    const layers = buildM3DeckLayers(water);
    expect(layers.map((l) => l.id)).toEqual(["m3-ctr-inflated-walls", "m3-forward-wire-capsules"]);
  });

  it("renders M3 DOM slots safely and preserves 360px zero-overflow CSS locks", () => {
    const makeFakeEl = () => ({ textContent: "" } as unknown as HTMLElement);
    const els: M3PanelElements = {
      summaryEl: makeFakeEl(),
      forwardEl: makeFakeEl(),
      gnssEl: makeFakeEl(),
      relocEl: makeFakeEl(),
      fsmEl: makeFakeEl(),
      voiceEl: makeFakeEl(),
      e5xPreEl: makeFakeEl(),
    };
    const snap = new M3SimReplay().activateScenario("voice_mayday");
    renderM3PanelDom(snap, els);
    expect(els.forwardEl?.textContent).toContain("Wires");
    expect(els.gnssEl?.textContent).toContain("H0_NOMINAL");
    expect(els.voiceEl?.textContent).toContain("MAYDAY PREEMPT");
    expect(els.e5xPreEl?.textContent).toContain("DRAFT_INCIDENT_ONLY");

    const html = readFileSync(resolve(__dirname, "../index.html"), "utf-8");
    expect(html).toContain('id="m3-panel"');
    expect(html).toContain('id="m3-e5x-details"');
    expect(html).toContain('id="fleet-table-details"');

    const css = readFileSync(resolve(__dirname, "../src/styles/vertigcs.css"), "utf-8");
    expect(css).toContain("#m3-e5x-xml-preview");
    expect(css).toContain("overflow-wrap: anywhere");
    expect(css).toContain(".sidecar-timeline");
  });

  it("drives 6-fault compound_apex scenario, builds XSD-compliant .e5x download, and verifies WebCrypto SHA-256 .vrta_events.jsonl chain", async () => {
    const replay = new M3SimReplay();
    const cmp = replay.activateScenario("compound_apex");
    expect(cmp.compoundFaults.active).toBe(true);
    expect(cmp.compoundFaults.rotor3Csfl).toBe(true);
    expect(cmp.compoundFaults.canyonWind12Mps).toBe(true);
    expect(cmp.compoundFaults.gnssSlowRampSpoof).toBe(true);
    expect(cmp.compoundFaults.grazingWire15MpsCap).toBe(true);
    expect(cmp.compoundFaults.partIsUsspQuarantine).toBe(true);
    expect(cmp.compoundFaults.voiceHoverMaydayTr).toBe(true);
    expect(cmp.forwardHazard.vMaxCapMps).toBeLessThanOrEqual(15.0);
    expect(cmp.gnssDenied.hypothesis).toBe("H3_SLOW_RAMP_SPOOF");
    expect(cmp.voiceAtc.snrStratum).toBe("S2_Hover");
    expect(cmp.voiceAtc.maydayPreempted).toBe(true);

    // 1-click .e5x.xml download payload matches eccairs2_e5x_minimal.xsd structure
    const dl = buildE5xDownloadDescriptor(cmp);
    expect(dl.filename).toMatch(/^vertirta_draft_seq\d+\.e5x\.xml$/);
    expect(dl.xml).toContain('schemaVersion="1.0"');
    expect(dl.xml).toContain('reportingStatus="DRAFT_INTERNAL_PRE_SUBMISSION"');
    expect(dl.xml).toContain('<Attribute id="431" name="OccurrenceClass">DRAFT_INCIDENT_ONLY</Attribute>');
    expect(dl.xml).toContain('<Attribute id="659" name="CicttCategory">SCF-NP</Attribute>');
    expect(dl.xml).toContain('<Narrative entityId="22" language="en">');
    expect(dl.xml).toContain('<Narrative entityId="22" language="tr">');

    // WebCrypto SHA-256 sidecar chain verifier passes signed chain & catches 1-byte tamper
    const signedJsonl = await buildSampleSignedSidecarJsonl();
    const ok = await verifyVrtaEventsJsonlChain(signedJsonl);
    expect(ok.valid).toBe(true);
    expect(ok.events).toHaveLength(3);
    expect(ok.events[0].kind).toBe("MOTOR_FDI_FAULT");
    expect(ok.events[1].kind).toBe("GNSS_SPOOF_ALERT");
    expect(ok.events[2].kind).toBe("VOICE_CONFORMAL_TURN");

    const tampered = signedJsonl.replace("1340", "1349");
    const bad = await verifyVrtaEventsJsonlChain(tampered);
    expect(bad.valid).toBe(false);
    expect(bad.failedLineIndex).toBe(1);
    expect(bad.errorMessage).toContain("SHA-256 digest mismatch");
  });
});
