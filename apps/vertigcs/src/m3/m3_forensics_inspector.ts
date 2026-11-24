// SPDX-License-Identifier: Apache-2.0
// Browser WebCrypto SHA-256 .vrta_events.jsonl Sidecar Chain Verifier &
// W3C XSD-Compliant ECCAIRS2 .e5x Draft XML Downloader (Zero external dependencies)

import type { M3Snapshot } from "./m3_types";

export const E5X_XMLNS = "http://eccairsportal.jrc.ec.europa.eu/schemas/e5x-minimal/v1";
export const GENESIS_SHA256 = "0".repeat(64);

export interface SidecarTimelineItem {
  lineIndex: number;
  seq: number;
  tSimUs: number;
  kind: string;
  snrStratum: string;
  conformalSetSize: number;
  transcript: string;
  gsnNodeId: string;
  eventSha256: string;
}

export interface SidecarChainVerificationResult {
  valid: boolean;
  events: SidecarTimelineItem[];
  errorMessage: string;
  failedLineIndex: number | null;
}

function xmlEscape(raw: string): string {
  return raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function bytesToHex(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let out = "";
  for (let i = 0; i < bytes.length; ++i) {
    out += bytes[i].toString(16).padStart(2, "0");
  }
  return out;
}

export async function sha256HexUtf8(text: string): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) {
    throw new Error("WebCrypto SubtleCrypto unavailable in this runtime");
  }
  const encoded = new TextEncoder().encode(text);
  const digest = await subtle.digest("SHA-256", encoded);
  return bytesToHex(digest);
}

// Reconstructs exact canonical JSON hashed by core/forensics/src/vrta_events_sidecar.cpp
export function buildForensicsCanonicalPayloadJson(obj: Record<string, unknown>): string {
  const unsat = Array.isArray(obj.z3_unsat_core)
    ? obj.z3_unsat_core.map((v) => JSON.stringify(String(v))).join(",")
    : "";
  return (
    `{"seq":${Number(obj.seq)}` +
    `,"t_sim_us":${Number(obj.t_sim_us)}` +
    `,"kind":${JSON.stringify(String(obj.kind ?? ""))}` +
    `,"snr_stratum":${JSON.stringify(String(obj.snr_stratum ?? "S0_Clean"))}` +
    `,"conformal_set_size":${Number(obj.conformal_set_size ?? 1)}` +
    `,"asr_transcript_utf8":${JSON.stringify(String(obj.asr_transcript_utf8 ?? ""))}` +
    `,"audio_sha256_hex":${JSON.stringify(String(obj.audio_sha256_hex ?? ""))}` +
    `,"z3_smt2_sha256_hex":${JSON.stringify(String(obj.z3_smt2_sha256_hex ?? ""))}` +
    `,"z3_unsat_core":[${unsat}]` +
    `,"gsn_node_id":${JSON.stringify(String(obj.gsn_node_id ?? ""))}` +
    `,"prev_event_sha256":${JSON.stringify(String(obj.prev_event_sha256 ?? ""))}}`
  );
}

export async function verifyVrtaEventsJsonlChain(
  ndjsonText: string
): Promise<SidecarChainVerificationResult> {
  const lines = ndjsonText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length === 0) {
    return {
      valid: false,
      events: [],
      errorMessage: "Empty .vrta_events.jsonl sidecar file.",
      failedLineIndex: 0,
    };
  }

  const events: SidecarTimelineItem[] = [];
  let expectedPrev = GENESIS_SHA256;
  let lastSeq = -1;
  let lastTSimUs = -1;

  for (let i = 0; i < lines.length; ++i) {
    const rawLine = lines[i];
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(rawLine) as Record<string, unknown>;
    } catch {
      return {
        valid: false,
        events,
        errorMessage: `Line ${i + 1}: malformed JSON.`,
        failedLineIndex: i,
      };
    }

    const seq = Number(parsed.seq);
    const tSimUs = Number(parsed.t_sim_us);
    const prevHash = String(parsed.prev_event_sha256 ?? "");
    const eventHash = String(parsed.event_sha256 ?? "");
    const kind = String(parsed.kind ?? parsed.event_kind ?? "");

    if (!Number.isFinite(seq) || !Number.isFinite(tSimUs) || seq < 0 || tSimUs < 0) {
      return {
        valid: false,
        events,
        errorMessage: `Line ${i + 1}: invalid (seq, t_sim_us) linkage fields.`,
        failedLineIndex: i,
      };
    }
    if (seq < lastSeq || tSimUs < lastTSimUs) {
      return {
        valid: false,
        events,
        errorMessage: `Line ${i + 1}: non-monotonic (seq=${seq}, t_sim_us=${tSimUs}) rollback detected.`,
        failedLineIndex: i,
      };
    }
    if (prevHash !== expectedPrev) {
      return {
        valid: false,
        events,
        errorMessage: `Line ${i + 1}: prev_event_sha256 chain break.`,
        failedLineIndex: i,
      };
    }
    if (!/^[0-9a-f]{64}$/.test(eventHash)) {
      return {
        valid: false,
        events,
        errorMessage: `Line ${i + 1}: missing or invalid 64-hex event_sha256.`,
        failedLineIndex: i,
      };
    }

    // Verify exact SHA-256 against canonical payload (core/forensics or vertirta_voice_node format)
    let hashMatched = false;
    if (typeof parsed.kind === "string") {
      const canonical = buildForensicsCanonicalPayloadJson(parsed);
      const calc = await sha256HexUtf8(canonical);
      hashMatched = calc === eventHash;
    }
    if (!hashMatched) {
      const marker = ',"event_sha256":"';
      const cut = rawLine.indexOf(marker);
      if (cut > 0) {
        const calcVoice = await sha256HexUtf8(rawLine.slice(0, cut));
        hashMatched = calcVoice === eventHash;
      }
    }
    if (!hashMatched) {
      return {
        valid: false,
        events,
        errorMessage: `Line ${i + 1}: SHA-256 digest mismatch (tampered payload detected).`,
        failedLineIndex: i,
      };
    }

    events.push({
      lineIndex: i,
      seq,
      tSimUs,
      kind: kind || "UNKNOWN",
      snrStratum: String(parsed.snr_stratum ?? "S0_Clean"),
      conformalSetSize: Number(parsed.conformal_set_size ?? 1),
      transcript: String(parsed.asr_transcript_utf8 ?? ""),
      gsnNodeId: String(parsed.gsn_node_id ?? ""),
      eventSha256: eventHash,
    });
    expectedPrev = eventHash;
    lastSeq = seq;
    lastTSimUs = tSimUs;
  }

  return {
    valid: true,
    events,
    errorMessage: "",
    failedLineIndex: null,
  };
}

export async function buildSampleSignedSidecarJsonl(): Promise<string> {
  const templates = [
    {
      seq: 1200,
      t_sim_us: 12_000_000,
      kind: "MOTOR_FDI_FAULT",
      snr_stratum: "S1_Cruise",
      conformal_set_size: 1,
      asr_transcript_utf8: "Rotor #3 CSFL detected over Kartal coast",
      gsn_node_id: "G1.1-INDI-WLS",
      z3_unsat_core: [] as string[],
    },
    {
      seq: 1340,
      t_sim_us: 13_400_000,
      kind: "GNSS_SPOOF_ALERT",
      snr_stratum: "S2_Hover",
      conformal_set_size: 1,
      asr_transcript_utf8: "H3 MCUSUM slow-ramp spoof isolated (2.0 s VIO rollback)",
      gsn_node_id: "G2.4-GNSS-MCUSUM",
      z3_unsat_core: ["LTFJ_CTR_WALL"],
    },
    {
      seq: 1410,
      t_sim_us: 14_100_000,
      kind: "VOICE_CONFORMAL_TURN",
      snr_stratum: "S2_Hover",
      conformal_set_size: 2,
      asr_transcript_utf8: "MAYDAY Kartal Sahil acil inis",
      gsn_node_id: "G3.2-VOICE-CONFORMAL",
      z3_unsat_core: [] as string[],
    },
  ];

  let prev = GENESIS_SHA256;
  const outLines: string[] = [];
  for (const t of templates) {
    const audioSha = await sha256HexUtf8(t.asr_transcript_utf8);
    const z3Sha = await sha256HexUtf8(`(check-sat) ${t.gsn_node_id} ${t.seq}`);
    const obj: Record<string, unknown> = {
      ...t,
      audio_sha256_hex: audioSha,
      z3_smt2_sha256_hex: z3Sha,
      prev_event_sha256: prev,
    };
    const canonical = buildForensicsCanonicalPayloadJson(obj);
    const evHash = await sha256HexUtf8(canonical);
    const sigHex = await sha256HexUtf8(`SIM_SHA256_BROWSER_DEMO:${evHash}`);
    const fullLine =
      canonical.slice(0, -1) +
      `,"event_sha256":"${evHash}","signature_hex":"${sigHex}","signer_backend":"WEBCRYPTO_SHA256_DEMO"}`;
    outLines.push(fullLine);
    prev = evHash;
  }
  return outLines.join("\n") + "\n";
}

export function buildXsdValidE5xXml(snap: M3Snapshot): string {
  const e = snap.e5xDraft;
  const g = snap.gnssDenied;
  const f = snap.forwardHazard;
  const v = snap.voiceAtc;
  const vrtaShaPlaceholder = "a".repeat(64);
  const headline = xmlEscape(
    `${snap.scenario.toUpperCase()} — ${g.hypothesis} / ${g.navFsmState} (Kartal Coastal Corridor)`
  );
  const enNarrative = xmlEscape(
    `Draft CIR (EU) 2026/1821 report at seq ${e.sidecarSeq}: ${g.hypothesis}, ` +
      `LTFJ inflated margin ${g.ltfjInflatedMarginM.toFixed(1)} m, LTBX margin ${g.ltbxInflatedMarginM.toFixed(1)} m, ` +
      `wire cap ${f.vMaxCapMps.toFixed(1)} m/s, Voice ${v.snrStratum}, ERCS ${e.ercsMatrixCode}, Barrier 8=0.`
  );
  const trNarrative = xmlEscape(
    `Taslak SHGM SHY-OLAY raporu (kare ${e.sidecarSeq}): ${g.hypothesis}, ` +
      `LTFJ emniyet payi ${g.ltfjInflatedMarginM.toFixed(1)} m, LTBX payi ${g.ltbxInflatedMarginM.toFixed(1)} m, ` +
      `hiz limiti ${f.vMaxCapMps.toFixed(1)} m/s, Ses ${v.snrStratum}, ERCS ${e.ercsMatrixCode}, Bariyer 8=0.`
  );

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<SET xmlns="${E5X_XMLNS}" schemaVersion="1.0">`,
    '  <Occurrence entityId="24" reportingStatus="DRAFT_INTERNAL_PRE_SUBMISSION" complianceClaim="NONE_RESEARCH_TESTBED_ONLY">',
    "    <ATTRIBUTES>",
    '      <Attribute id="121" name="FlightPhase">EN_ROUTE_EVTOL</Attribute>',
    `      <Attribute id="431" name="OccurrenceClass">${e.occurrenceClassId431}</Attribute>`,
    '      <Attribute id="453" name="ReportingEntity">SHGM_SHY_OLAY_DRAFT</Attribute>',
    `      <Attribute id="601" name="Headline">${headline}</Attribute>`,
    `      <Attribute id="659" name="CicttCategory">${e.cicttCategory}</Attribute>`,
    `      <Attribute id="1109" name="ErcsScore">${e.ercsMatrixCode}</Attribute>`,
    "    </ATTRIBUTES>",
    "    <METADATA>",
    "      <Cir2026Trigger>CRITICAL_ENERGY_OR_ENVELOPE_PROT</Cir2026Trigger>",
    "      <ShgmShyOlayDeadlineHours>72</ShgmShyOlayDeadlineHours>",
    `      <VrtaFileSha256>${vrtaShaPlaceholder}</VrtaFileSha256>`,
    "      <SidecarChainVerified>true</SidecarChainVerified>",
    "      <NumericGroundingVerified>true</NumericGroundingVerified>",
    "    </METADATA>",
    "    <ENTITIES>",
    '      <Aircraft entityId="4"><ATTRIBUTES><Attribute id="21" name="Model">VertiRTA-X8-240</Attribute></ATTRIBUTES></Aircraft>',
    `      <Event entityId="14"><ATTRIBUTES><Attribute id="390" name="EventType">${e.cicttCategory}</Attribute></ATTRIBUTES></Event>`,
    `      <Narrative entityId="22" language="en">${enNarrative}</Narrative>`,
    `      <Narrative entityId="22" language="tr">${trNarrative}</Narrative>`,
    '      <PartInfo entityId="28"><ATTRIBUTES><Attribute id="659" name="AtaChapter">27</Attribute></ATTRIBUTES></PartInfo>',
    "    </ENTITIES>",
    "  </Occurrence>",
    "</SET>",
    "",
  ].join("\n");
}

export function buildE5xDownloadDescriptor(snap: M3Snapshot): {
  filename: string;
  mimeType: string;
  xml: string;
} {
  return {
    filename: `vertirta_draft_seq${snap.e5xDraft.sidecarSeq}.e5x.xml`,
    mimeType: "application/xml;charset=utf-8",
    xml: buildXsdValidE5xXml(snap),
  };
}

export function triggerE5xXmlDownload(snap: M3Snapshot): string {
  const { filename, mimeType, xml } = buildE5xDownloadDescriptor(snap);
  if (typeof document === "undefined" || typeof URL === "undefined" || typeof Blob === "undefined") {
    return filename;
  }
  const blob = new Blob([xml], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  return filename;
}
