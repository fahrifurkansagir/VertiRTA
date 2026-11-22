import { describe, expect, it } from "vitest";
import {
  VRTA_FRAME_SIZE,
  VRTA_HEADER_SIZE,
  crc32c,
  parseVrtaFile,
  parseVrtaFrame,
  parseVrtaHeader,
} from "../src/vrta/vrta_parser";

describe("VRTA0001 binary parser & Castagnoli CRC-32C", () => {
  it("matches RFC 3720 CRC-32C check value 0xE3069283", () => {
    const bytes = new TextEncoder().encode("123456789");
    expect(crc32c(bytes)).toBe(0xe3069283);
  });

  it("decodes canonical 128 B header and 256 B frame and rejects corrupted CRC", () => {
    expect(VRTA_HEADER_SIZE).toBe(128);
    expect(VRTA_FRAME_SIZE).toBe(256);

    const hdrBuf = new ArrayBuffer(128);
    const hv = new DataView(hdrBuf);
    const magic = new TextEncoder().encode("VRTA0001");
    new Uint8Array(hdrBuf, 0, 8).set(magic);
    hv.setUint32(8, 1, true);
    hv.setFloat64(16, 40.8860, true);
    hv.setFloat64(24, 29.1880, true);
    hv.setFloat64(32, 12.0, true);
    hv.setFloat64(40, 36.7, true);
    hv.setUint32(124, crc32c(new Uint8Array(hdrBuf, 0, 124)), true);

    const hdr = parseVrtaHeader(hdrBuf);
    expect(hdr.magic).toBe("VRTA0001");
    expect(hdr.datumLatDeg).toBeCloseTo(40.8860, 6);
    expect(hdr.datumLonDeg).toBeCloseTo(29.1880, 6);
    expect(hdr.egm2008NM).toBeCloseTo(36.7, 4);

    const fBuf = new ArrayBuffer(256);
    const fv = new DataView(fBuf);
    fv.setBigUint64(0, 42n, true);
    fv.setBigUint64(8, 420000n, true);
    fv.setFloat64(16, -150.0, true);
    fv.setFloat64(24, 80.0, true);
    fv.setFloat64(32, 35.0, true);
    fv.setFloat32(136, 398.5, true);
    fv.setFloat32(140, 79.2, true);
    fv.setFloat32(144, 18.4, true);
    fv.setFloat32(148, 1105.0, true);
    fv.setFloat32(152, 2890.0, true);
    fv.setFloat32(160, 35.0, true);
    fv.setUint8(164, 0xee);
    fv.setUint8(165, 3);
    fv.setUint8(166, 1);
    fv.setUint8(167, 1);
    fv.setUint32(252, crc32c(new Uint8Array(fBuf, 0, 252)), true);

    const frame = parseVrtaFrame(fBuf);
    expect(frame.seq).toBe(42);
    expect(frame.posEnu[0]).toBeCloseTo(-150.0, 4);
    expect(frame.vTermV).toBeCloseTo(398.5, 2);
    expect(frame.socPct).toBeCloseTo(79.2, 2);
    expect(frame.activeMotorMask).toBe(0xee);
    expect(frame.celLatched).toBe(true);

    // Combine header + 1 frame into a full .vrta buffer
    const fileBuf = new ArrayBuffer(128 + 256);
    new Uint8Array(fileBuf, 0, 128).set(new Uint8Array(hdrBuf));
    new Uint8Array(fileBuf, 128, 256).set(new Uint8Array(fBuf));
    const parsed = parseVrtaFile(fileBuf);
    expect(parsed.frames.length).toBe(1);
    expect(parsed.frames[0].seq).toBe(42);

    new Uint8Array(fBuf)[16] ^= 0xff;
    expect(() => parseVrtaFrame(fBuf)).toThrow(/CRC32C/i);
  });
});
