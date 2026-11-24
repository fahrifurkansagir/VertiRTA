// SPDX-License-Identifier: Apache-2.0
// Deterministic browser-side M3 Mission Control scenario engine (zero WASM/flight_core bloat)

import type {
  E5xDraftIncidentPreview,
  M3ScenarioId,
  M3Snapshot,
  WireCapsule3D,
} from "./m3_types";

const NOMINAL_WIRES: WireCapsule3D[] = [
  {
    id: 101,
    wireClass: "distribution",
    p0: [29.1815, 40.8888, 42],
    p1: [29.1848, 40.8894, 42],
    radiusM: 3.55,
    skewDeg: 7.5,
    radarConfirmed: true,
  },
  {
    id: 102,
    wireClass: "transmission",
    p0: [29.1742, 40.8960, 62],
    p1: [29.1786, 40.8972, 62],
    radiusM: 3.85,
    skewDeg: 11.2,
    radarConfirmed: true,
  },
  {
    id: 103,
    wireClass: "crane_cable",
    p0: [29.1675, 40.9035, 58],
    p1: [29.1682, 40.9041, 88],
    radiusM: 3.60,
    skewDeg: 9.0,
    radarConfirmed: true,
  },
];

function buildE5xDraft(
  seq: number,
  ercsCode: string,
  headline: string,
  ltfjMarginM: number,
  snrStratum: string
): E5xDraftIncidentPreview {
  const xmlSnippet = [
    `<SET>`,
    `  <Occurrence entityId="24" reportingStatus="DRAFT_INTERNAL_PRE_SUBMISSION">`,
    `    <ATTRIBUTES>`,
    `      <FlightPhase attributeId="121">EN_ROUTE_EVTOL</FlightPhase>`,
    `      <OccurrenceClass attributeId="431">DRAFT_INCIDENT_ONLY</OccurrenceClass>`,
    `      <ReportingEntity attributeId="453">SHGM_SHY_OLAY_DRAFT</ReportingEntity>`,
    `      <Headline attributeId="601">${headline}</Headline>`,
    `      <CicttCategory attributeId="659">SCF-NP</CicttCategory>`,
    `      <ErcsScore attributeId="1109">${ercsCode}</ErcsScore>`,
    `    </ATTRIBUTES>`,
    `    <ENTITIES>`,
    `      <Narrative entityId="22" lang="EN">Frame ${seq}: CIR (EU) 2026/1821 draft; LTFJ margin ${ltfjMarginM.toFixed(1)} m; SNR ${snrStratum}; Barrier #8=0.</Narrative>`,
    `    </ENTITIES>`,
    `  </Occurrence>`,
    `</SET>`,
  ].join("\n");

  return {
    regulation: "CIR (EU) 2026/1821 + SHGM SHY-OLAY + CDR (EU) 2020/2034",
    occurrenceClassId431: "DRAFT_INCIDENT_ONLY",
    cicttCategory: "SCF-NP",
    barrier8Score: 0,
    ercsMatrixCode: ercsCode,
    autoClassifiedAccident: false,
    numericHallucinationRate: 0.0,
    sidecarSeq: seq,
    xmlSnippet,
  };
}

export class M3SimReplay {
  private scenario: M3ScenarioId = "nominal";
  private simTimeS = 0.0;
  private scenarioElapsedS = 0.0;
  private sigmaPosM = 1.5;
  private prevCtrInflationM = 6.5;
  private seq = 1200;

  reset(): M3Snapshot {
    this.scenario = "nominal";
    this.simTimeS = 0.0;
    this.scenarioElapsedS = 0.0;
    this.sigmaPosM = 1.5;
    this.prevCtrInflationM = 6.5;
    this.seq = 1200;
    return this.snapshot();
  }

  activateScenario(scenario: M3ScenarioId): M3Snapshot {
    this.scenario = scenario;
    this.scenarioElapsedS = 0.0;
    if (scenario === "nominal") {
      return this.reset();
    }
    if (scenario === "slow_ramp_spoof") {
      this.sigmaPosM = 3.5;
      this.prevCtrInflationM = 12.5;
    } else if (scenario === "compound_apex") {
      this.sigmaPosM = 14.5;
      this.prevCtrInflationM = 45.5;
    } else if (scenario === "water_drift") {
      this.sigmaPosM = 28.0;
      this.prevCtrInflationM = 86.0;
    }
    return this.snapshot();
  }

  step(dtS: number): M3Snapshot {
    const dt = Math.max(0.05, Math.min(dtS, 5.0));
    this.simTimeS += dt;
    this.scenarioElapsedS += dt;
    this.seq += Math.round(dt * 100);

    if (this.scenario === "slow_ramp_spoof") {
      this.sigmaPosM = Math.min(18.0, this.sigmaPosM + 0.14 * dt);
    } else if (this.scenario === "compound_apex") {
      this.sigmaPosM = Math.min(34.0, this.sigmaPosM + 0.45 * dt);
    } else if (this.scenario === "water_drift") {
      this.sigmaPosM = Math.min(72.0, this.sigmaPosM + 1.85 * dt);
    } else {
      this.sigmaPosM = 1.5;
    }

    return this.snapshot(dt);
  }

  snapshot(dtS = 0.2): M3Snapshot {
    const compound = this.scenario === "compound_apex";
    const grazing = this.scenario === "wire_grazing" || compound;
    const spoof = this.scenario === "slow_ramp_spoof" || compound;
    const water = this.scenario === "water_drift";
    const mayday = this.scenario === "voice_mayday" || compound;

    const wires: WireCapsule3D[] = NOMINAL_WIRES.map((w) => ({ ...w }));
    if (grazing) {
      wires.push({
        id: 104,
        wireClass: "crane_cable",
        p0: [29.1835, 40.8872, 46],
        p1: [29.1862, 40.8910, 74],
        radiusM: 6.05,
        skewDeg: 28.4,
        radarConfirmed: false,
      });
    }

    const forwardCap = grazing ? 15.0 : 22.0;

    // GNSS-denied state & dynamic 3-sigma + 2.0 m CTR inflation
    const threeSigmaM = 3.0 * this.sigmaPosM;
    const ctrInflationM = threeSigmaM + 2.0;
    const dhDtMps = water
      ? Math.max(0.25, (ctrInflationM - this.prevCtrInflationM) / Math.max(dtS, 0.05))
      : spoof
        ? 0.42
        : 0.0;
    this.prevCtrInflationM = ctrInflationM;

    const navFsmState =
      this.sigmaPosM >= 65.0
        ? "DETERMINISTIC_VISUAL_SLZ_ABORT"
        : this.sigmaPosM >= 25.0
          ? "DEGRADED_CAUTION"
          : "BOUNDED_GNSS_DENIED_CRUISE";

    const gnssCap = navFsmState === "BOUNDED_GNSS_DENIED_CRUISE" ? 22.0 : 13.5;
    const baseLtfjM = 1042.0;
    const baseLtbxM = 2995.0;
    const ltfjInflatedMarginM = Math.max(80.0, baseLtfjM - ctrInflationM);
    const ltbxInflatedMarginM = Math.max(120.0, baseLtbxM - ctrInflationM);

    const hypothesis = spoof
      ? "H3_SLOW_RAMP_SPOOF"
      : water
        ? "H1_RF_JAMMING"
        : "H0_NOMINAL";
    const mcusumNorm = spoof ? 3.85 : water ? 1.42 : 0.18;
    const ttdSec = spoof ? 0.90 : null;
    const relocBackend = this.sigmaPosM > 50.0 ? "MegaLoc" : "XFeat+LighterGlue";
    const relocInliers = water ? 6 : 64;

    const voiceAtc = compound
      ? {
          snrStratum: "S2_Hover" as const,
          snrDb: 4.5,
          asrTier: "whisper-large-v3-turbo-Q8" as const,
          conformalSetSize: 2,
          maydayPreempted: true,
          maydayPosterior: 0.24,
          livenessEmergencyCommit: false,
          ttsEngine: "sherpa-onnx >= 2.0.0",
          ttsTtfaMs: 142.0,
          lastTranscript: "MAYDAY rotor uc arizasi Kartal Sahil acil inis",
          lastReadback: "ACIL ONCELIK ONAYLANDI: Kartal_Sahil guvenli inis sahasina yonlendiriliyor.",
        }
      : mayday
        ? {
            snrStratum: "S3_Garbled" as const,
            snrDb: -3.5,
            asrTier: "whisper-large-v3-turbo-Q8" as const,
            conformalSetSize: 3,
            maydayPreempted: true,
            maydayPosterior: 0.21,
            livenessEmergencyCommit: false,
            ttsEngine: "sherpa-onnx >= 2.0.0",
            ttsTtfaMs: 142.0,
            lastTranscript: "MAYDAY rotor vibration Kartal immediate landing",
            lastReadback: "EMERGENCY PREEMPT CONFIRMED: Diverting to Kartal_Sahil.",
          }
        : {
            snrStratum: "S1_Cruise" as const,
            snrDb: 14.2,
            asrTier: "whisper-small-Q8" as const,
            conformalSetSize: 1,
            maydayPreempted: false,
            maydayPosterior: 0.01,
            livenessEmergencyCommit: false,
            ttsEngine: "sherpa-onnx >= 2.0.0",
            ttsTtfaMs: 154.0,
            lastTranscript: "Kartal Sahil inis onayi",
            lastReadback: "Z3 koridoru dogrulandi: Kartal inis rotasi aktif.",
          };

    const ercsCode = compound || mayday || navFsmState === "DETERMINISTIC_VISUAL_SLZ_ABORT"
      ? "E4"
      : spoof || grazing
        ? "C4"
        : "A1";
    const headline = compound
      ? "6-Fault Compound Apex: Rotor #3 CSFL, 12 m/s wind, H3 spoof, grazing wire, Part-IS quarantine & TR MAYDAY"
      : mayday
        ? "Voice MAYDAY emergency preemption along Kartal coastal corridor"
        : spoof
          ? "Slow-ramp GNSS spoof isolated via H3 MCUSUM with 2.0 s VIO rollback"
          : water
            ? "Open-water VIO uncertainty triggered CTR wall inflation and SLZ abort"
            : grazing
              ? "Grazing-angle unconfirmed cable engaged 15.0 m/s CBVF speed cap"
              : "Nominal M3 Kartal coastal corridor surveillance";

    const e5xDraft = buildE5xDraft(
      this.seq,
      ercsCode,
      headline,
      ltfjInflatedMarginM,
      voiceAtc.snrStratum
    );

    const summaryLine =
      `M3 ${this.scenario.toUpperCase()} • Wires ${wires.length}/64 (${forwardCap.toFixed(1)} m/s) • ` +
      `${hypothesis} / ${navFsmState} (3σ=${threeSigmaM.toFixed(1)} m) • ` +
      `Voice ${voiceAtc.snrStratum} (|C|=${voiceAtc.conformalSetSize})`;

    return {
      scenario: this.scenario,
      simTimeS: this.simTimeS,
      compoundFaults: {
        active: compound,
        rotor3Csfl: compound,
        canyonWind12Mps: compound,
        gnssSlowRampSpoof: compound,
        grazingWire15MpsCap: compound,
        partIsUsspQuarantine: compound,
        voiceHoverMaydayTr: compound,
      },
      forwardHazard: {
        wireCount: wires.length,
        wireCap: 64,
        intruderCount: grazing ? 2 : 1,
        intruderCap: 16,
        grazingFallback: grazing,
        vMaxCapMps: forwardCap,
        rVehM: 2.1,
        enableFp16: false,
        wires,
      },
      gnssDenied: {
        hypothesis,
        mcusumNorm,
        surfaceMode: water ? "OPEN_WATER" : "LAND",
        rollbackActive: spoof,
        ttdSec,
        relocBackend,
        relocInliers,
        navFsmState,
        sigmaPosM: this.sigmaPosM,
        threeSigmaM,
        ctrInflationM,
        dhDtMps,
        vMaxCapMps: gnssCap,
        ltfjInflatedMarginM,
        ltbxInflatedMarginM,
      },
      voiceAtc,
      e5xDraft,
      summaryLine,
    };
  }
}
