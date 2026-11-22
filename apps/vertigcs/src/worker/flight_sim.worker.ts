// Single-threaded Dedicated Web Worker for libvertirta_flight_core.wasm
// Strictly ZERO pthreads, ZERO SharedArrayBuffer; transfers 256 B VRTA0001 ArrayBuffers via postMessage(buf, [buf]).

import { crc32c, VRTA_FRAME_SIZE } from "../vrta/vrta_parser";
import type { FaultCommand } from "../types/telemetry";

interface WasmExports {
  HEAPU8: Uint8Array;
  _vertirta_init(lat: number, lon: number, amsl: number, n0: number): number;
  _vertirta_reset(): number;
  _vertirta_step_batch(ticks: number, ae: number, an: number, au: number): number;
  _vertirta_inject_fault(kind: number, idx: number): number;
  _vertirta_get_vrta_ptr(): number;
}

const FAULT_CODES: Record<FaultCommand["kind"], number> = {
  reset: 0,
  single: 1,
  same_arm: 2,
  imu: 3,
  wind: 4,
};

let wasmMod: WasmExports | null = null;

// Deterministic TypeScript fallback kinematics (used cleanly when static .wasm binary is not bundled locally)
const fallback = {
  seq: 0,
  tUs: 0,
  posE: -180.0,
  posN: 120.0,
  aglM: 35.0,
  velE: 0.0,
  velN: 0.0,
  velU: 0.0,
  socPct: 80.0,
  vTermV: 403.2,
  activeMask: 0xff,
  regime: 0,
  tmrState: 0,
  chi2: 1.2,
  celLatched: false,
  windGust: false,
};

function resetFallback(): void {
  fallback.seq = 0;
  fallback.tUs = 0;
  fallback.posE = -180.0;
  fallback.posN = 120.0;
  fallback.aglM = 35.0;
  fallback.velE = 0.0;
  fallback.velN = 0.0;
  fallback.velU = 0.0;
  fallback.socPct = 80.0;
  fallback.vTermV = 403.2;
  fallback.activeMask = 0xff;
  fallback.regime = 0;
  fallback.tmrState = 0;
  fallback.chi2 = 1.2;
  fallback.celLatched = false;
  fallback.windGust = false;
}

function stepFallbackIntoBuffer(ticks400Hz: number, buf: ArrayBuffer): void {
  const dt = ticks400Hz * 0.0025;
  fallback.tUs += Math.round(ticks400Hz * 2500);
  fallback.seq += Math.max(1, Math.floor(ticks400Hz / 4));
  const tS = fallback.tUs * 1e-6;

  if (fallback.celLatched) {
    const vz = fallback.aglM > 10 ? -2.0 : fallback.aglM > 5 ? -1.2 : -0.6;
    fallback.aglM = Math.max(0.0, fallback.aglM + vz * dt);
    fallback.velU = fallback.aglM > 0 ? vz : 0.0;
    fallback.velE = fallback.windGust ? 1.4 : 0.2;
    fallback.velN = -0.3;
  } else {
    // Smooth coastal corridor loiter strictly west of LTFJ NW J2 pinch
    const targetE = -220.0 + 90.0 * Math.cos(tS * 0.12);
    const targetN = 150.0 + 110.0 * Math.sin(tS * 0.12);
    fallback.velE = (targetE - fallback.posE) * 0.6;
    fallback.velN = (targetN - fallback.posN) * 0.6;
    fallback.velU = (35.0 - fallback.aglM) * 0.5;
    fallback.aglM = Math.max(12.0, fallback.aglM + fallback.velU * dt);
  }
  fallback.posE += fallback.velE * dt;
  fallback.posN += fallback.velN * dt;
  const burn = fallback.celLatched ? 0.22 : 0.14;
  fallback.socPct = Math.max(12.0, fallback.socPct - burn * dt);
  fallback.vTermV = 342.0 + 0.76 * fallback.socPct - (fallback.celLatched ? 12.0 : 0.0);

  const view = new DataView(buf);
  new Uint8Array(buf).fill(0);
  view.setBigUint64(0, BigInt(fallback.seq), true);
  view.setBigUint64(8, BigInt(fallback.tUs), true);
  view.setFloat64(16, fallback.posE, true);
  view.setFloat64(24, fallback.posN, true);
  view.setFloat64(32, fallback.aglM, true);
  view.setFloat32(40, fallback.velE, true);
  view.setFloat32(44, fallback.velN, true);
  view.setFloat32(48, fallback.velU, true);
  view.setFloat32(52, 1.0, true);
  view.setFloat32(76, fallback.celLatched ? 2.2 : 0.0, true); // Option B1b flat spin 2.2 rad/s
  for (let i = 0; i < 8; i++) {
    const alive = (fallback.activeMask & (1 << i)) !== 0;
    const rpm = alive ? (fallback.celLatched ? 1980.0 : 1450.0) : 0.0;
    view.setFloat32(104 + i * 4, rpm, true);
  }
  view.setFloat32(136, fallback.vTermV, true);
  view.setFloat32(140, fallback.socPct, true);
  view.setFloat32(144, fallback.celLatched ? 6.5 : 24.0, true);
  view.setFloat32(148, Math.max(120.0, 1050.0 - fallback.posE), true);
  view.setFloat32(152, 2980.0, true);
  view.setFloat32(156, fallback.chi2, true);
  view.setFloat32(160, fallback.aglM, true);
  view.setUint8(164, fallback.activeMask);
  view.setUint8(165, fallback.regime);
  view.setUint8(166, fallback.tmrState);
  view.setUint8(167, fallback.celLatched ? 1 : 0);
  view.setUint32(252, crc32c(new Uint8Array(buf, 0, 252)), true);
}

async function initWasmIfPresent(): Promise<void> {
  try {
    const baseUrl = import.meta.env.BASE_URL || "./";
    const wasmRel = baseUrl.startsWith("/")
      ? `${baseUrl.replace(/\/+$/, "")}/wasm/libvertirta_flight_core.mjs`
      : import.meta.env.DEV
        ? "/wasm/libvertirta_flight_core.mjs"
        : "../wasm/libvertirta_flight_core.mjs";
    const modUrl = new URL(wasmRel, self.location.href).href;
    const loaded = await import(/* @vite-ignore */ modUrl);
    const factory = loaded.default;
    if (typeof factory === "function") {
      const instance = (await factory()) as WasmExports;
      if (instance._vertirta_init(40.8860, 29.1880, 12.0, 36.7) === 0) {
        wasmMod = instance;
        self.postMessage({ type: "status", detail: "WASM active: libvertirta_flight_core.wasm (single-threaded Worker)" });
        return;
      }
    }
  } catch {
    // Fallback mode stays active when optional Emscripten artifact is absent
  }
  resetFallback();
  self.postMessage({ type: "status", detail: "Sandbox Worker active (deterministic TS kinematic fallback; build wasm/build_wasm.sh for native C++ WASM)" });
}

self.onmessage = async (ev: MessageEvent) => {
  const msg = ev.data as { type: string; ticks?: number; buffer?: ArrayBuffer; cmd?: FaultCommand };
  if (msg.type === "init") {
    await initWasmIfPresent();
    return;
  }
  if (msg.type === "fault" && msg.cmd) {
    if (wasmMod) {
      if (msg.cmd.kind === "reset") {
        wasmMod._vertirta_reset();
      } else {
        wasmMod._vertirta_inject_fault(FAULT_CODES[msg.cmd.kind], msg.cmd.targetIdx);
      }
    } else {
      if (msg.cmd.kind === "reset") {
        resetFallback();
      } else if (msg.cmd.kind === "single") {
        fallback.activeMask &= ~(1 << (msg.cmd.targetIdx & 7));
        fallback.regime = 1;
      } else if (msg.cmd.kind === "same_arm") {
        const arm = msg.cmd.targetIdx & 3;
        fallback.activeMask &= ~((1 << arm) | (1 << (arm + 4)));
        fallback.regime = 3;
        fallback.celLatched = true;
      } else if (msg.cmd.kind === "imu") {
        fallback.tmrState = 1;
        fallback.chi2 = 96.4;
      } else if (msg.cmd.kind === "wind") {
        fallback.windGust = true;
      }
    }
    return;
  }
  if (msg.type === "step") {
    const out = msg.buffer && msg.buffer.byteLength === VRTA_FRAME_SIZE ? msg.buffer : new ArrayBuffer(VRTA_FRAME_SIZE);
    const ticks = msg.ticks ?? 8;
    if (wasmMod) {
      wasmMod._vertirta_step_batch(ticks, 0.0, 0.0, 0.0);
      const ptr = wasmMod._vertirta_get_vrta_ptr();
      new Uint8Array(out).set(wasmMod.HEAPU8.subarray(ptr, ptr + VRTA_FRAME_SIZE));
    } else {
      stepFallbackIntoBuffer(ticks, out);
    }
    self.postMessage({ type: "frame", buffer: out }, [out]);
  }
};
