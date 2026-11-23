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
  const wsUrl = params.get("ws") || "ws://127.0.0.1:8891";
  const rawAdapter = params.get("adapter");
  const adapter: ViewportAdapterKind =
    rawAdapter === "cesium-bridge" ? "cesium-bridge" : "maplibre-deck";

  // Always enforce AEGIS-VTOL port isolation on WebSocket URLs
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

  start(onSnapshot: (snap: TelemetrySnapshot) => void, onStatus?: (msg: string) => void): void {
    this.stop();
    this.worker = new Worker(new URL("../worker/flight_sim.worker.ts", import.meta.url), {
      type: "module",
    });
    this.worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data as { type: string; buffer?: ArrayBuffer; detail?: string };
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
      if (!this.worker || this.recycleBuf.byteLength !== 256) {
        return;
      }
      const out = this.recycleBuf;
      this.recycleBuf = new ArrayBuffer(0);
      this.worker.postMessage({ type: "step", ticks: 8, buffer: out }, [out]);
    }, 33); // ~30 Hz UI update cadence
  }

  sendFault(cmd: FaultCommand): void {
    this.worker?.postMessage({ type: "fault", cmd });
  }

  stop(): void {
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }
    this.worker?.terminate();
    this.worker = null;
  }
}

export class LiveWsTransport implements FlightTransport {
  private ws: WebSocket | null = null;

  constructor(private readonly wsUrl: string) {
    const parsed = new URL(wsUrl);
    if (parsed.port) {
      validateLivePort(Number(parsed.port));
    }
  }

  start(onSnapshot: (snap: TelemetrySnapshot) => void, onStatus?: (msg: string) => void): void {
    this.stop();
    this.ws = new WebSocket(this.wsUrl);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => onStatus?.(`Live WS connected: ${this.wsUrl}`);
    this.ws.onerror = () => onStatus?.(`Live WS error on ${this.wsUrl}`);
    this.ws.onmessage = (ev: MessageEvent) => {
      if (ev.data instanceof ArrayBuffer && ev.data.byteLength === 256) {
        const frame = parseVrtaFrame(ev.data);
        onSnapshot(vrtaFrameToSnapshot(frame));
        return;
      }
      if (typeof ev.data === "string") {
        const pkt = JSON.parse(ev.data);
        if (pkt.schema_ver === "vrta.ws.v1.2" && pkt.geodetic && pkt.propulsion && pkt.safety_avionics) {
          onSnapshot({
            seq: Number(pkt.seq ?? 0),
            simTimeS: Number(pkt.sim_time_s ?? 0),
            latDeg: Number(pkt.geodetic.lat_deg),
            lonDeg: Number(pkt.geodetic.lon_deg),
            altAmslM: Number(pkt.geodetic.alt_amsl_m),
            altHaeM: Number(pkt.geodetic.alt_hae_wgs84_m),
            aglM: Number(pkt.geodetic.agl_m),
            velEnu: pkt.geodetic.vel_enu_mps ?? [0, 0, 0],
            rotorRpm: pkt.propulsion.rotor_rpm ?? new Array(8).fill(0),
            vTermV: Number(pkt.safety_avionics.v_term_v),
            socPct: Number(pkt.safety_avionics.soc_pct),
            cbvfMinV: Number(pkt.safety_avionics.cbf_delta_u_norm ?? 0),
            ltfjMarginM: Number(pkt.safety_avionics.dist_ltfj_ctr_m),
            ltbxMarginM: Number(pkt.safety_avionics.dist_ltbx_ctr_m),
            chi2Residual: 0,
            activeMotorMask: Number(pkt.propulsion.active_motor_mask),
            flightRegimeLabel: String(pkt.safety_avionics.flight_regime),
            tmrStatusLabel: String(pkt.safety_avionics.tmr_status),
            celLatched: Boolean(pkt.safety_avionics.cel_latched),
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
    this.ws?.close();
    this.ws = null;
  }
}
