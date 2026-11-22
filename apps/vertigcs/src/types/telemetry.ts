export interface VrtaFileHeader {
  magic: string;
  schemaVer: number;
  datumLatDeg: number;
  datumLonDeg: number;
  datumAmslM: number;
  egm2008NM: number;
  gitSha: string;
  headerCrc32c: number;
}

export interface VrtaFrame100Hz {
  seq: number;
  tSimUs: number;
  posEnu: [number, number, number];
  velEnu: [number, number, number];
  qFrdNed: [number, number, number, number];
  omegaFrd: [number, number, number];
  aNomEnu: [number, number, number];
  aSafeEnu: [number, number, number];
  rotorRpm: [number, number, number, number, number, number, number, number];
  vTermV: number;
  socPct: number;
  cbvfMinV: number;
  ltfjMarginM: number;
  ltbxMarginM: number;
  chi2Residual: number;
  aglM: number;
  activeMotorMask: number;
  flightRegime: number;
  tmrState: number;
  celLatched: boolean;
  flags: number;
  crc32c: number;
}

export type FaultCommandKind = "single" | "same_arm" | "imu" | "wind" | "reset";

export interface FaultCommand {
  kind: FaultCommandKind;
  targetIdx: number;
}

export interface TelemetrySnapshot {
  seq: number;
  simTimeS: number;
  latDeg: number;
  lonDeg: number;
  altAmslM: number;
  altHaeM: number;
  aglM: number;
  velEnu: [number, number, number];
  rotorRpm: number[];
  vTermV: number;
  socPct: number;
  cbvfMinV: number;
  ltfjMarginM: number;
  ltbxMarginM: number;
  chi2Residual: number;
  activeMotorMask: number;
  flightRegimeLabel: string;
  tmrStatusLabel: string;
  celLatched: boolean;
}

export type LaunchMode = "sandbox" | "live";
export type ViewportAdapterKind = "maplibre-deck" | "cesium-bridge";

export interface LaunchConfig {
  mode: LaunchMode;
  wsUrl: string;
  adapter: ViewportAdapterKind;
}
