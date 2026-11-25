// SPDX-License-Identifier: Apache-2.0
// VertiRTA v1.0.0 Browser-Side M3 Telemetry Contracts (0 B added to flight_core or WASM)

export const VERIFIED_LTFJ_ARP = {
  latDeg: 40.898333,
  lonDeg: 29.309167,
  radiusM: 9260.0,
} as const;

// Confirmed DHMI AIP Turkey LTBX ARP: 40 deg 59 min 26 sec N, 029 deg 12 min 58 sec E
export const VERIFIED_LTBX_ARP = {
  latDeg: 40.990556,
  lonDeg: 29.216111,
  radiusM: 9260.0,
} as const;

export const LTFJ_NW_J2_PINCH = {
  latDeg: 40.912786,
  lonDeg: 29.192122,
} as const;

export type WireHazardClass = "distribution" | "transmission" | "crane_cable";
export type SpoofHypothesisCode =
  | "H0_NOMINAL"
  | "H1_RF_JAMMING"
  | "H2_STEP_MEACONING"
  | "H3_SLOW_RAMP_SPOOF";
export type SurfaceMode = "LAND" | "TRANSITION" | "OPEN_WATER";
export type GnssNavFsmState =
  | "BOUNDED_GNSS_DENIED_CRUISE"
  | "DEGRADED_CAUTION"
  | "DETERMINISTIC_VISUAL_SLZ_ABORT";
export type SnrStratumCode = "S0_Clean" | "S1_Cruise" | "S2_Hover" | "S3_Garbled";

export interface WireCapsule3D {
  id: number;
  wireClass: WireHazardClass;
  p0: [number, number, number]; // [lon, lat, altAmslM]
  p1: [number, number, number];
  radiusM: number;
  skewDeg: number;
  radarConfirmed: boolean;
}

export interface ForwardHazardTelemetry {
  wireCount: number;
  wireCap: number;
  intruderCount: number;
  intruderCap: number;
  grazingFallback: boolean;
  vMaxCapMps: number;
  rVehM: number;
  enableFp16: boolean;
  wires: WireCapsule3D[];
}

export interface GnssDeniedTelemetry {
  hypothesis: SpoofHypothesisCode;
  mcusumNorm: number;
  surfaceMode: SurfaceMode;
  rollbackActive: boolean;
  ttdSec: number | null;
  relocBackend: "XFeat+LighterGlue" | "MegaLoc";
  relocInliers: number;
  navFsmState: GnssNavFsmState;
  sigmaPosM: number;
  threeSigmaM: number;
  ctrInflationM: number;
  dhDtMps: number;
  vMaxCapMps: number;
  ltfjInflatedMarginM: number;
  ltbxInflatedMarginM: number;
}

export interface VoiceAtcTelemetry {
  snrStratum: SnrStratumCode;
  snrDb: number;
  asrTier: "whisper-small-Q8" | "whisper-large-v3-turbo-Q8";
  conformalSetSize: number;
  maydayPreempted: boolean;
  maydayPosterior: number;
  livenessEmergencyCommit: boolean;
  ttsEngine: string;
  ttsTtfaMs: number;
  lastTranscript: string;
  lastReadback: string;
}

export interface E5xDraftIncidentPreview {
  regulation: string;
  occurrenceClassId431: "DRAFT_INCIDENT_ONLY";
  cicttCategory: "SCF-NP";
  barrier8Score: 0;
  ercsMatrixCode: string;
  autoClassifiedAccident: false;
  numericHallucinationRate: 0.0;
  sidecarSeq: number;
  xmlSnippet: string;
}

export type M3ScenarioId =
  | "nominal"
  | "wire_grazing"
  | "slow_ramp_spoof"
  | "water_drift"
  | "voice_mayday"
  | "compound_apex";

export interface CompoundFaultFlags {
  active: boolean;
  rotor3Csfl: boolean;
  canyonWind12Mps: boolean;
  gnssSlowRampSpoof: boolean;
  grazingWire15MpsCap: boolean;
  partIsUsspQuarantine: boolean;
  voiceHoverMaydayTr: boolean;
}

export interface M3Snapshot {
  scenario: M3ScenarioId;
  simTimeS: number;
  compoundFaults: CompoundFaultFlags;
  forwardHazard: ForwardHazardTelemetry;
  gnssDenied: GnssDeniedTelemetry;
  voiceAtc: VoiceAtcTelemetry;
  e5xDraft: E5xDraftIncidentPreview;
  summaryLine: string;
}
