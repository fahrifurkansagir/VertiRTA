import { flatEnuToWgs84, KARTAL_DATUM } from "../geo/istanbul_datum";
import { validateLivePort } from "../adapters/maplibre_deck_adapter";
import { parseVrtaFrame } from "../vrta/vrta_parser";
export { resolvePublicAssetUrl, resolveWasmModuleUrl } from "./asset_urls";
import type {
  FaultCommand,
  LaunchConfig,
  LaunchMode,
  TelemetrySnapshot,
  ViewportAdapterKind,
  VrtaFileHeader,
  VrtaFrame100Hz,
} from "../types/telemetry";

const REGIME_LABELS: Record<number, string> = {
  0: "NOMINAL_8_ROTOR",
  1: "DEGRADED_7_ROTOR",
  2: "DEGRADED_6_ROTOR_CROSS",
  3: "CEL_SAME_ARM_ACTIVE_SPIN",
  4: "CEL_CRITICAL_DESCENT",
};

const TMR_LABELS: Record<number, string> = {
  0: "3/3_NOMINAL",
  1: "2/2_DUPLEX",
  2: "1/3_SIMPLEX_ANALYTICAL",
  3: "0/3_FAILSAFE_DIVERT",
};

export function parseLaunchMode(search: string): LaunchConfig {
  const params = new URLSearchParams(search.startsWith("?") ? search : `?${search}`);
  const rawMode = params.get("mode");
  const mode: LaunchMode = rawMode === "live" ? "live" : "sandbox";
  const wsUrl = params.get("ws") || "ws://127.0.0.1:8891/telemetry";
  const rawAdapter = params.get("adapter");
  const adapter: ViewportAdapterKind =
    rawAdapter === "cesium-bridge" ? "cesium-bridge" : "maplibre-deck";

  // Always enforce VertiRTA port isolation policy on WebSocket URLs
  try {
    const parsed = new URL(wsUrl);
    if (parsed.port) {
      validateLivePort(Number(parsed.port));
    }
  } catch (err) {
    if (err instanceof Error && /forbidden/i.test(err.message)) {
      throw err;
    }
    throw new Error(`Invalid WebSocket URL: ${wsUrl}`);
  }

  return { mode, wsUrl, adapter };
}

export function vrtaFrameToSnapshot(
  frame: VrtaFrame100Hz,
  header?: VrtaFileHeader
): TelemetrySnapshot {
  const geo = flatEnuToWgs84(
    frame.posEnu[0],
    frame.posEnu[1],
    frame.posEnu[2],
    header?.datumLatDeg ?? KARTAL_DATUM.latDeg,
    header?.datumLonDeg ?? KARTAL_DATUM.lonDeg,
    header?.datumAmslM ?? KARTAL_DATUM.altAmslM
  );
  return {
    seq: frame.seq,
    simTimeS: frame.tSimUs * 1e-6,
    latDeg: geo.latDeg,
    lonDeg: geo.lonDeg,
    altAmslM: geo.altAmslM,
    altHaeM: geo.altHaeM,
    aglM: frame.aglM,
    velEnu: frame.velEnu,
    rotorRpm: [...frame.rotorRpm],
    vTermV: frame.vTermV,
    socPct: frame.socPct,
    cbvfMinV: frame.cbvfMinV,
    ltfjMarginM: frame.ltfjMarginM,
    ltbxMarginM: frame.ltbxMarginM,
    chi2Residual: frame.chi2Residual,
    activeMotorMask: frame.activeMotorMask,
    flightRegimeLabel: REGIME_LABELS[frame.flightRegime] ?? `REGIME_${frame.flightRegime}`,
    tmrStatusLabel: TMR_LABELS[frame.tmrState] ?? `TMR_${frame.tmrState}`,
    celLatched: frame.celLatched,
  };
}

export interface FlightTransport {
  start(onSnapshot: (snap: TelemetrySnapshot) => void, onStatus?: (msg: string) => void): void;
  sendFault(cmd: FaultCommand): void;
  stop(): void;
}

export class SandboxWorkerTransport implements FlightTransport {
  private worker: Worker | null = null;
  private timerId: number | null = null;
  private recycleBuf: ArrayBuffer = new ArrayBuffer(256);
  private workerReady = false;
  private pendingFaults: FaultCommand[] = [];

  start(onSnapshot: (snap: TelemetrySnapshot) => void, onStatus?: (msg: string) => void): void {
    this.stop();
    this.workerReady = false;
    this.pendingFaults = [];
    this.recycleBuf = new ArrayBuffer(256);
    this.worker = new Worker(new URL("../worker/flight_sim.worker.ts", import.meta.url), {
      type: "module",
    });
    const flushPendingFaults = () => {
      if (!this.worker || !this.workerReady || this.pendingFaults.length === 0) {
        return;
      }
      const queued = this.pendingFaults.splice(0, this.pendingFaults.length);
      for (const cmd of queued) {
        this.worker.postMessage({ type: "fault", cmd });
      }
    };
    this.worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data as { type: string; buffer?: ArrayBuffer; detail?: string };
      if (!this.workerReady && (msg.type === "status" || msg.type === "frame")) {
        this.workerReady = true;
        flushPendingFaults();
      }
      if (msg.type === "frame" && msg.buffer) {
        try {
          const frame = parseVrtaFrame(msg.buffer);
          onSnapshot(vrtaFrameToSnapshot(frame));
        } finally {
          // Ping-pong pre-allocated ArrayBuffer back to worker with zero copy
          this.recycleBuf = msg.buffer;
        }
      } else if (msg.type === "status" && msg.detail && onStatus) {
        onStatus(msg.detail);
      }
    };
    this.worker.postMessage({ type: "init" });
    this.timerId = window.setInterval(() => {
      if (!this.worker || !this.workerReady || this.recycleBuf.byteLength !== 256) {
        return;
      }
      const out = this.recycleBuf;
      this.recycleBuf = new ArrayBuffer(0);
      this.worker.postMessage({ type: "step", ticks: 8, buffer: out }, [out]);
    }, 33); // ~30 Hz UI update cadence
  }

  sendFault(cmd: FaultCommand): void {
    if (!this.worker) {
      this.pendingFaults.push(cmd);
      return;
    }
    if (!this.workerReady) {
      this.pendingFaults.push(cmd);
      return;
    }
    this.worker.postMessage({ type: "fault", cmd });
  }

  stop(): void {
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }
    this.worker?.terminate();
    this.worker = null;
    this.workerReady = false;
    this.pendingFaults = [];
  }
}

export interface ReplayFaultCoordinator {
  enterReplay(): void;
  dispatchFault(cmd: FaultCommand): void;
  isTransportActive(): boolean;
}

// Guarantees 1st fault click during .vrta replay immediately exits replay AND applies the fault
// on the exact same click (no dropped first click during worker cold-init).
export function createReplayFaultCoordinator(
  transport: FlightTransport,
  stopReplay: () => void,
  onSnapshot: (snap: TelemetrySnapshot) => void,
  onStatus: (msg: string) => void
): ReplayFaultCoordinator {
  let transportActive = true;

  return {
    enterReplay(): void {
      if (transportActive) {
        transport.stop();
        transportActive = false;
      }
    },
    dispatchFault(cmd: FaultCommand): void {
      stopReplay();
      if (!transportActive) {
        transport.start(onSnapshot, onStatus);
        transportActive = true;
      }
      transport.sendFault(cmd);
      onStatus(`Fault command dispatched: ${cmd.kind} (target=${cmd.targetIdx})`);
    },
    isTransportActive(): boolean {
      return transportActive;
    },
  };
}

export class LiveWsTransport implements FlightTransport {
  private ws: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private reconnectDelayMs = 500;
  private manuallyStopped = false;
  private onSnapshotCb: ((snap: TelemetrySnapshot) => void) | null = null;
  private onStatusCb: ((msg: string) => void) | null = null;

  constructor(private readonly wsUrl: string) {
    const parsed = new URL(wsUrl);
    if (parsed.port) {
      validateLivePort(Number(parsed.port));
    }
  }

  start(onSnapshot: (snap: TelemetrySnapshot) => void, onStatus?: (msg: string) => void): void {
    this.stop();
    this.manuallyStopped = false;
    this.reconnectDelayMs = 500;
    this.onSnapshotCb = onSnapshot;
    this.onStatusCb = onStatus ?? null;
    this.connect();
  }

  private connect(): void {
    if (this.manuallyStopped || !this.onSnapshotCb) {
      return;
    }
    const onSnapshot = this.onSnapshotCb;
    const onStatus = this.onStatusCb;

    const ws = new WebSocket(this.wsUrl);
    this.ws = ws;
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.reconnectDelayMs = 500;
      onStatus?.(`Live WS connected: ${this.wsUrl}`);
    };
    ws.onerror = () => onStatus?.(`Live WS error on ${this.wsUrl}`);
    ws.onclose = () => {
      if (this.manuallyStopped) return;
      const delay = this.reconnectDelayMs;
      this.reconnectDelayMs = Math.min(10000, this.reconnectDelayMs * 2);
      onStatus?.(`Live WS disconnected — reconnecting in ${(delay / 1000).toFixed(1)} s`);
      this.reconnectTimer = window.setTimeout(() => {
        this.reconnectTimer = null;
        this.connect();
      }, delay);
    };
    ws.onmessage = (ev: MessageEvent) => {
      if (ev.data instanceof ArrayBuffer && ev.data.byteLength === 256) {
        try {
          const frame = parseVrtaFrame(ev.data);
          onSnapshot(vrtaFrameToSnapshot(frame));
        } catch (err) {
          onStatus?.(`Live WS binary frame rejected: ${err instanceof Error ? err.message : String(err)}`);
        }
        return;
      }
      if (typeof ev.data === "string") {
        let pkt: Record<string, unknown>;
        try {
          pkt = JSON.parse(ev.data) as Record<string, unknown>;
        } catch {
          onStatus?.("Live WS ignored malformed JSON payload");
          return;
        }
        const geo = pkt.geodetic as Record<string, unknown> | undefined;
        const prop = pkt.propulsion as Record<string, unknown> | undefined;
        const safety = pkt.safety_avionics as Record<string, unknown> | undefined;
        if (pkt.schema_ver === "vrta.ws.v1.2" && geo && prop && safety) {
          const rawCbvf = safety.cbvf_min_v ?? safety.cbvf_min_margin_m;
          onSnapshot({
            seq: Number(pkt.seq ?? 0),
            simTimeS: Number(pkt.sim_time_s ?? 0),
            latDeg: Number(geo.lat_deg),
            lonDeg: Number(geo.lon_deg),
            altAmslM: Number(geo.alt_amsl_m),
            altHaeM: Number(geo.alt_hae_wgs84_m),
            aglM: Number(geo.agl_m),
            velEnu: (geo.vel_enu_mps as [number, number, number]) ?? [0, 0, 0],
            rotorRpm: (prop.rotor_rpm as number[]) ?? new Array(8).fill(0),
            vTermV: Number(safety.v_term_v),
            socPct: Number(safety.soc_pct),
            cbvfMinV: rawCbvf !== undefined ? Number(rawCbvf) : 0,
            ltfjMarginM: Number(safety.dist_ltfj_ctr_m),
            ltbxMarginM: Number(safety.dist_ltbx_ctr_m),
            chi2Residual: Number(safety.chi2_residual ?? 0),
            activeMotorMask: Number(prop.active_motor_mask),
            flightRegimeLabel: String(safety.flight_regime),
            tmrStatusLabel: String(safety.tmr_status),
            celLatched: Boolean(safety.cel_latched),
          });
        }
      }
    };
  }

  sendFault(cmd: FaultCommand): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "fault_inject", ...cmd }));
    }
  }

  stop(): void {
    this.manuallyStopped = true;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.ws?.close();
    this.ws = null;
  }
}
