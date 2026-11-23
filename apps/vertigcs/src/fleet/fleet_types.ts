// SPDX-License-Identifier: Apache-2.0
// VertiRTA v3.5.0 VertiGCS 20-eVTOL U-Space Fleet & Part-IS Cyber Simulator Types

export type FleetCyberState =
  | "SEC_NOMINAL"
  | "SEC_SUSPECT_PEER"
  | "SEC_USSP_QUARANTINE"
  | "SEC_AUTONOMOUS_RTB";

export interface FleetAircraftState {
  id: number;
  callsign: string;
  latDeg: number;
  lonDeg: number;
  aglM: number;
  altAmslM: number;
  headingDeg: number;
  speedMps: number;
  lane: "SE_70M" | "NW_105M";
  priorityClass: 0 | 1 | 2;
  celActive: boolean;
  preempted: boolean;
  cyberState: FleetCyberState;
  packTempC: number;
  socPct: number;
}

export interface VertiportStandState {
  code: string;
  name: string;
  latDeg: number;
  lonDeg: number;
  altAmslM: number;
  chillerCoolantC: number;
  standsTotal: number;
  standsOccupied: number;
  peakStandTempC: number;
  fatoEmgReserved: boolean;
}

export interface PreemptionWaveState {
  active: boolean;
  sourceAgentId: number;
  centerLatDeg: number;
  centerLonDeg: number;
  radiusM: number;
  v2vLatencyMs: number;
  usspVacateMs: number;
  preemptedPeers: number;
  targetFatoEmg: string;
}

export interface FleetSnapshot {
  simTimeS: number;
  aircraft: FleetAircraftState[];
  vertiports: VertiportStandState[];
  preemptionWave: PreemptionWaveState | null;
  cyberSummary: string;
  activeCyberState: FleetCyberState;
}

export interface CelPreemptionEvent {
  agentIndex: number;
  callsign: string;
  preemptedPeers: number;
  v2vLatencyMs: number;
  usspVacateMs: number;
  targetFatoEmg: string;
}

export interface PartIsReplayEvent {
  peerIndex: number;
  callsign: string;
  duplicateNonce: number;
  droppedByReplayWindow128: boolean;
  cyberState: FleetCyberState;
  demotedBetaIj: number;
}

export interface SpoofedUsspEvent {
  agentIndex: number;
  callsign: string;
  spoofedLonDeg: number;
  easternFenceLonDeg: number;
  cyberState: FleetCyberState;
  rtoSec: number;
  isOr230DraftTriggered: boolean;
  divertVertiportCode: string;
}
