// Canonical VRTA0001 Binary Flight Recorder Decoder (128 B Header + 256 B Frames)
// Matches core/flight_core/include/vertirta/core/vrta_recorder.hpp byte-for-byte.

import type { VrtaFileHeader, VrtaFrame100Hz } from "../types/telemetry";

export const VRTA_MAGIC = "VRTA0001";
export const VRTA_HEADER_SIZE = 128;
export const VRTA_FRAME_SIZE = 256;

const CRC32C_POLY = 0x82f63b78;
const CRC32C_TABLE = new Uint32Array(256);

for (let i = 0; i < 256; i++) {
  let c = i;
  for (let bit = 0; bit < 8; bit++) {
    c = (c & 1) !== 0 ? (c >>> 1) ^ CRC32C_POLY : c >>> 1;
  }
  CRC32C_TABLE[i] = c >>> 0;
}

export function crc32c(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC32C_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function parseVrtaHeader(buffer: ArrayBuffer, byteOffset = 0): VrtaFileHeader {
  if (buffer.byteLength - byteOffset < VRTA_HEADER_SIZE) {
    throw new Error(`VrtaFileHeader truncated: need ${VRTA_HEADER_SIZE} B`);
  }
  const view = new DataView(buffer, byteOffset, VRTA_HEADER_SIZE);
  const magic = new TextDecoder("ascii").decode(new Uint8Array(buffer, byteOffset, 8));
  if (magic !== VRTA_MAGIC) {
    throw new Error(`Invalid VRTA magic: expected ${VRTA_MAGIC}, got ${magic}`);
  }
  const storedCrc = view.getUint32(124, true);
  const computedCrc = crc32c(new Uint8Array(buffer, byteOffset, 124));
  if (storedCrc !== computedCrc) {
    throw new Error(`VrtaFileHeader CRC32C mismatch: stored=0x${storedCrc.toString(16)} computed=0x${computedCrc.toString(16)}`);
  }
  const gitSha = new TextDecoder("ascii")
    .decode(new Uint8Array(buffer, byteOffset + 80, 40))
    .replace(/\0+$/, "");

  return {
    magic,
    schemaVer: view.getUint32(8, true),
    datumLatDeg: view.getFloat64(16, true),
    datumLonDeg: view.getFloat64(24, true),
    datumAmslM: view.getFloat64(32, true),
    egm2008NM: view.getFloat64(40, true),
    gitSha,
    headerCrc32c: storedCrc,
  };
}

export function parseVrtaFrame(buffer: ArrayBuffer, byteOffset = 0): VrtaFrame100Hz {
  if (buffer.byteLength - byteOffset < VRTA_FRAME_SIZE) {
    throw new Error(`VrtaFrame100Hz truncated: need ${VRTA_FRAME_SIZE} B`);
  }
  const view = new DataView(buffer, byteOffset, VRTA_FRAME_SIZE);
  const storedCrc = view.getUint32(252, true);
  const computedCrc = crc32c(new Uint8Array(buffer, byteOffset, 252));
  if (storedCrc !== computedCrc) {
    throw new Error(`VrtaFrame100Hz CRC32C mismatch: stored=0x${storedCrc.toString(16)} computed=0x${computedCrc.toString(16)}`);
  }

  const rotorRpm: [number, number, number, number, number, number, number, number] = [
    view.getFloat32(104, true),
    view.getFloat32(108, true),
    view.getFloat32(112, true),
    view.getFloat32(116, true),
    view.getFloat32(120, true),
    view.getFloat32(124, true),
    view.getFloat32(128, true),
    view.getFloat32(132, true),
  ];

  return {
    seq: Number(view.getBigUint64(0, true)),
    tSimUs: Number(view.getBigUint64(8, true)),
    posEnu: [view.getFloat64(16, true), view.getFloat64(24, true), view.getFloat64(32, true)],
    velEnu: [view.getFloat32(40, true), view.getFloat32(44, true), view.getFloat32(48, true)],
    qFrdNed: [
      view.getFloat32(52, true),
      view.getFloat32(56, true),
      view.getFloat32(60, true),
      view.getFloat32(64, true),
    ],
    omegaFrd: [view.getFloat32(68, true), view.getFloat32(72, true), view.getFloat32(76, true)],
    aNomEnu: [view.getFloat32(80, true), view.getFloat32(84, true), view.getFloat32(88, true)],
    aSafeEnu: [view.getFloat32(92, true), view.getFloat32(96, true), view.getFloat32(100, true)],
    rotorRpm,
    vTermV: view.getFloat32(136, true),
    socPct: view.getFloat32(140, true),
    cbvfMinV: view.getFloat32(144, true),
    ltfjMarginM: view.getFloat32(148, true),
    ltbxMarginM: view.getFloat32(152, true),
    chi2Residual: view.getFloat32(156, true),
    aglM: view.getFloat32(160, true),
    activeMotorMask: view.getUint8(164),
    flightRegime: view.getUint8(165),
    tmrState: view.getUint8(166),
    celLatched: view.getUint8(167) !== 0,
    flags: view.getUint32(168, true),
    crc32c: storedCrc,
  };
}

export function parseVrtaFile(buffer: ArrayBuffer): {
  header: VrtaFileHeader;
  frames: VrtaFrame100Hz[];
} {
  const header = parseVrtaHeader(buffer, 0);
  const payloadBytes = buffer.byteLength - VRTA_HEADER_SIZE;
  if (payloadBytes < 0 || payloadBytes % VRTA_FRAME_SIZE !== 0) {
    throw new Error(`VRTA payload length ${payloadBytes} is not a multiple of ${VRTA_FRAME_SIZE} B`);
  }
  const nFrames = payloadBytes / VRTA_FRAME_SIZE;
  const frames: VrtaFrame100Hz[] = [];
  for (let i = 0; i < nFrames; i++) {
    frames.push(parseVrtaFrame(buffer, VRTA_HEADER_SIZE + i * VRTA_FRAME_SIZE));
  }
  return { header, frames };
}
