// SPDX-License-Identifier: Apache-2.0
// Deterministic 20-eVTOL Anatolian Coastal Corridor Fleet & Part-IS Cyber-Attack Simulator
// Anchored strictly to frozen Kartal datum (40.8860 N, 29.1880 E, N0 = 36.7 m) & < 29.1895 E LTFJ margin.

import { KARTAL_DATUM } from "../geo/istanbul_datum";
import type {
  CelPreemptionEvent,
  FleetAircraftState,
  FleetCyberState,
  FleetSnapshot,
  PartIsReplayEvent,
  PreemptionWaveState,
  SpoofedUsspEvent,
  VertiportStandState,
} from "./fleet_types";

export const EASTERN_GEOFENCE_LON_DEG = 29.1895;
export const CORRIDOR_CEILING_AGL_M = 120.0;
export const HOSPITAL_HELIPAD = Object.freeze({
  name: "Kartal Dr. Lutfi Kirdar Hospital Helipad",
  latDeg: 40.91639,
  lonDeg: 29.17417,
  radiusM: 150.0,
});

interface CorridorVertiportDef {
  code: string;
  name: string;
  latDeg: number;
  lonDeg: number;
  altAmslM: number;
}

const CORRIDOR_VERTIPORTS: readonly CorridorVertiportDef[] = Object.freeze([
  { code: "VPT-MLT", name: "Maltepe Orhangazi Park", latDeg: 40.94016, lonDeg: 29.10790, altAmslM: 5.0 },
  { code: "VPT-DRG", name: "Dragos Sahil Park", latDeg: 40.91686, lonDeg: 29.13822, altAmslM: 6.0 },
  { code: "VPT-CVZ", name: "Cevizli Coastal Hub", latDeg: 40.90380, lonDeg: 29.15450, altAmslM: 8.0 },
  { code: "VPT-KRT", name: "Kartal Coastal Vertiport", latDeg: KARTAL_DATUM.latDeg, lonDeg: KARTAL_DATUM.lonDeg, altAmslM: KARTAL_DATUM.altAmslM },
]);

const METERS_PER_DEG_LAT = 111052.0;
const METERS_PER_DEG_LON = 84255.0;

interface AgentInternal {
  id: number;
  callsign: string;
  progress: number; // [0..1] along Maltepe -> Kartal coastal spine
  dir: 1 | -1; // +1 SE (70 m), -1 NW (105 m)
  speedMps: number;
  lateralOffsetM: number;
  priorityClass: 0 | 1 | 2;
  celActive: boolean;
  preempted: boolean;
  cyberState: FleetCyberState;
  quarantineElapsedS: number;
  packTempC: number;
  socPct: number;
}

export class FleetCorridorController {
  private readonly count: number;
  private simTimeS = 0;
  private agents: AgentInternal[] = [];
  private vertiports: VertiportStandState[] = [];
  private preemptionWave: PreemptionWaveState | null = null;
  private cyberSummary = "Part-IS Mode B nominal: 40 B header + ReplayWindow128 active (0 drops).";
  private activeCyberState: FleetCyberState = "SEC_NOMINAL";
  private nonceCounter = 1000;

  constructor(aircraftCount = 20) {
    this.count = Math.max(5, Math.min(20, Math.trunc(aircraftCount)));
    this.reset();
  }

  reset(): void {
    this.simTimeS = 0;
    this.preemptionWave = null;
    this.activeCyberState = "SEC_NOMINAL";
    this.cyberSummary = "Part-IS Mode B nominal: 40 B header + ReplayWindow128 active (0 drops).";

    this.vertiports = CORRIDOR_VERTIPORTS.map((vp, idx) => ({
      code: vp.code,
      name: vp.name,
      latDeg: vp.latDeg,
      lonDeg: vp.lonDeg,
      altAmslM: vp.altAmslM,
      chillerCoolantC: 15,
      standsTotal: 4,
      standsOccupied: idx === 0 || idx === 3 ? 3 : 2,
      peakStandTempC: 37.2,
      fatoEmgReserved: false,
    }));

    this.agents = [];
    for (let i = 0; i < this.count; i++) {
      const dir: 1 | -1 = i % 2 === 0 ? 1 : -1;
      const frac = (i + 0.5) / this.count;
      this.agents.push({
        id: i + 1,
        callsign: `VRTA-X8-${String(i + 1).padStart(2, "0")}`,
        progress: 0.04 + 0.90 * frac,
        dir,
        speedMps: 14.5 + (i % 4) * 0.8,
        lateralOffsetM: dir === 1 ? -95.0 : -165.0, // Coastal corridor offset west of 29.1895 E
        priorityClass: i % 5 === 4 ? 2 : 1,
        celActive: false,
        preempted: false,
        cyberState: "SEC_NOMINAL",
        quarantineElapsedS: 0,
        packTempC: 35.8 + (i % 5) * 0.4,
        socPct: 78.0 - (i % 6) * 4.0,
      });
    }
  }

  step(dtSec: number): void {
    const dt = Math.max(0.001, Math.min(0.5, dtSec));
    this.simTimeS += dt;

    // Total spine length from Maltepe (0) to Kartal (3) ~ 9,150 m
    const spineLengthM = 9150.0;
    for (const a of this.agents) {
      const effectiveSpeed = a.celActive ? 4.5 : a.preempted ? 7.0 : a.speedMps;
      const dProg = (effectiveSpeed * dt) / spineLengthM;
      a.progress += a.dir * dProg;

      if (a.progress >= 0.96) {
        a.progress = 0.96;
        a.dir = -1;
        a.socPct = 85.0; // Recharged on 15 C chilled stand
        a.packTempC = 36.8;
      } else if (a.progress <= 0.04) {
        a.progress = 0.04;
        a.dir = 1;
        a.socPct = 85.0;
        a.packTempC = 36.8;
      }

      a.socPct = Math.max(25.0, a.socPct - 0.028 * dt);
      a.packTempC = Math.min(39.8, a.packTempC + 0.005 * dt);

      if (a.cyberState === "SEC_USSP_QUARANTINE") {
        a.quarantineElapsedS += dt;
        if (a.quarantineElapsedS >= 10.0) {
          a.cyberState = "SEC_AUTONOMOUS_RTB";
          this.activeCyberState = "SEC_AUTONOMOUS_RTB";
        }
      }
    }

    if (this.preemptionWave?.active) {
      this.preemptionWave.radiusM = Math.min(650.0, this.preemptionWave.radiusM + 85.0 * dt);
    }
  }

  triggerCelPreemption(agentIndex: number): CelPreemptionEvent {
    const idx = Math.max(0, Math.min(this.agents.length - 1, Math.trunc(agentIndex)));
    const ego = this.agents[idx];
    ego.celActive = true;
    ego.priorityClass = 0;

    const egoPos = this.computeAgentGeo(ego);
    let preemptedPeers = 0;
    for (let i = 0; i < this.agents.length; i++) {
      if (i === idx) continue;
      const peer = this.agents[i];
      const pPos = this.computeAgentGeo(peer);
      const distM = this.approxDistM(egoPos.latDeg, egoPos.lonDeg, pPos.latDeg, pPos.lonDeg);
      if (distM <= 2600.0 || preemptedPeers === 0) {
        peer.preempted = true;
        peer.lateralOffsetM -= 45.0; // Shift seaward (west) to open >= 25 m CEL cylinder
        preemptedPeers++;
      }
    }

    const nearestVp = this.nearestVertiport(egoPos.latDeg, egoPos.lonDeg);
    nearestVp.fatoEmgReserved = true;

    const v2vLatencyMs = 97.0; // BEST_EFFORT 3x UDP burst (82 ms comm + 15 ms FDI/RTA)
    const usspVacateMs = 210.0; // F3548-21 Contingent OVN-omitted DSS write

    this.preemptionWave = {
      active: true,
      sourceAgentId: ego.id,
      centerLatDeg: egoPos.latDeg,
      centerLonDeg: egoPos.lonDeg,
      radiusM: 120.0,
      v2vLatencyMs,
      usspVacateMs,
      preemptedPeers,
      targetFatoEmg: nearestVp.code,
    };

    this.cyberSummary =
      `P0 CEL Preemption (${ego.callsign}): V2V ${v2vLatencyMs} ms (<500 ms), ` +
      `${preemptedPeers} peers yielded, ${nearestVp.code} FATO-EMG cleared (${usspVacateMs} ms).`;

    return {
      agentIndex: idx,
      callsign: ego.callsign,
      preemptedPeers,
      v2vLatencyMs,
      usspVacateMs,
      targetFatoEmg: nearestVp.code,
    };
  }

  injectPartIsReplayAttack(peerIndex: number): PartIsReplayEvent {
    const idx = Math.max(0, Math.min(this.agents.length - 1, Math.trunc(peerIndex)));
    const peer = this.agents[idx];
    const dupNonce = this.nonceCounter;
    this.nonceCounter += 1;

    peer.cyberState = "SEC_SUSPECT_PEER";
    peer.priorityClass = 2; // Strip Priority-0/1 rights; beta_ij demoted to 1.0 (non-cooperative)
    this.activeCyberState = "SEC_SUSPECT_PEER";
    this.cyberSummary =
      `Part-IS ReplayWindow128 DROP: duplicate nonce #${dupNonce} from ${peer.callsign} ` +
      `-> SEC_SUSPECT_PEER (beta_ij=1.0, P0 stripped).`;

    return {
      peerIndex: idx,
      callsign: peer.callsign,
      duplicateNonce: dupNonce,
      droppedByReplayWindow128: true,
      cyberState: "SEC_SUSPECT_PEER",
      demotedBetaIj: 1.0,
    };
  }

  injectSpoofedUsspIntoLtfj(agentIndex: number): SpoofedUsspEvent {
    const idx = Math.max(0, Math.min(this.agents.length - 1, Math.trunc(agentIndex)));
    const ego = this.agents[idx];
    const egoPos = this.computeAgentGeo(ego);
    const nearestVp = this.nearestVertiport(egoPos.latDeg, egoPos.lonDeg);
    nearestVp.fatoEmgReserved = true;

    const spoofedLon = 29.2100; // East of 29.1895 E into LTFJ CTR!
    ego.cyberState = "SEC_USSP_QUARANTINE";
    ego.quarantineElapsedS = 0;
    this.activeCyberState = "SEC_USSP_QUARANTINE";
    const rtoSec = 8.5;

    this.cyberSummary =
      `Part-IS IS.I.OR.230 QUARANTINE: spoofed USSP cmd (${spoofedLon.toFixed(4)} E >= 29.1895 E LTFJ) ` +
      `dropped for ${ego.callsign} -> SEC_USSP_QUARANTINE (RTB ${nearestVp.code} RTO=${rtoSec.toFixed(1)} s <= 10 s).`;

    return {
      agentIndex: idx,
      callsign: ego.callsign,
      spoofedLonDeg: spoofedLon,
      easternFenceLonDeg: EASTERN_GEOFENCE_LON_DEG,
      cyberState: "SEC_USSP_QUARANTINE",
      rtoSec,
      isOr230DraftTriggered: true,
      divertVertiportCode: nearestVp.code,
    };
  }

  snapshot(): FleetSnapshot {
    const aircraft: FleetAircraftState[] = this.agents.map((a) => {
      const geo = this.computeAgentGeo(a);
      const aglM = a.celActive ? 45.0 : a.dir === 1 ? 70.0 : 105.0;
      const baseAmsl = 5.0 + 7.0 * a.progress;
      return {
        id: a.id,
        callsign: a.callsign,
        latDeg: geo.latDeg,
        lonDeg: geo.lonDeg,
        aglM: Math.min(CORRIDOR_CEILING_AGL_M, aglM),
        altAmslM: baseAmsl + Math.min(CORRIDOR_CEILING_AGL_M, aglM),
        headingDeg: a.dir === 1 ? 132.0 : 312.0,
        speedMps: a.celActive ? 4.5 : a.speedMps,
        lane: a.dir === 1 ? "SE_70M" : "NW_105M",
        priorityClass: a.priorityClass,
        celActive: a.celActive,
        preempted: a.preempted,
        cyberState: a.cyberState,
        packTempC: a.packTempC,
        socPct: a.socPct,
      };
    });

    return {
      simTimeS: this.simTimeS,
      aircraft,
      vertiports: this.vertiports.map((v) => ({ ...v })),
      preemptionWave: this.preemptionWave ? { ...this.preemptionWave } : null,
      cyberSummary: this.cyberSummary,
      activeCyberState: this.activeCyberState,
    };
  }

  private computeAgentGeo(a: AgentInternal): { latDeg: number; lonDeg: number } {
    const p0 = CORRIDOR_VERTIPORTS[0];
    const p3 = CORRIDOR_VERTIPORTS[3];
    const latDeg = p0.latDeg + a.progress * (p3.latDeg - p0.latDeg);
    let lonDeg = p0.lonDeg + a.progress * (p3.lonDeg - p0.lonDeg) + a.lateralOffsetM / METERS_PER_DEG_LON;

    // Hard geofence clamp strictly west of 29.1895 E LTFJ NW J2 margin
    lonDeg = Math.min(lonDeg, EASTERN_GEOFENCE_LON_DEG - 0.0008);

    // Keep strictly outside Kartal Hospital helipad 150 m cylinder
    const dHosp = this.approxDistM(latDeg, lonDeg, HOSPITAL_HELIPAD.latDeg, HOSPITAL_HELIPAD.lonDeg);
    if (dHosp < HOSPITAL_HELIPAD.radiusM + 40.0) {
      lonDeg -= 220.0 / METERS_PER_DEG_LON;
    }
    return { latDeg, lonDeg };
  }

  private approxDistM(lat1: number, lon1: number, lat2: number, lon2: number): number {
    const dN = (lat1 - lat2) * METERS_PER_DEG_LAT;
    const dE = (lon1 - lon2) * METERS_PER_DEG_LON;
    return Math.hypot(dN, dE);
  }

  private nearestVertiport(latDeg: number, lonDeg: number): VertiportStandState {
    let best = this.vertiports[0];
    let bestD = Number.POSITIVE_INFINITY;
    for (const vp of this.vertiports) {
      const d = this.approxDistM(latDeg, lonDeg, vp.latDeg, vp.lonDeg);
      if (d < bestD) {
        bestD = d;
        best = vp;
      }
    }
    return best;
  }
}
