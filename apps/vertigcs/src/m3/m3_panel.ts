// SPDX-License-Identifier: Apache-2.0
// Interactive M3 Mission Control Panel: Forward Wire/mmWave, GNSS Spoof MCUSUM + 3-sigma CTR,
// Voice ATC SNR Mondrian Card & 1-Click .e5x Draft Incident Preview

import {
  buildSampleSignedSidecarJsonl,
  buildXsdValidE5xXml,
  triggerE5xXmlDownload,
  verifyVrtaEventsJsonlChain,
  type SidecarChainVerificationResult,
} from "./m3_forensics_inspector";
import { M3SimReplay } from "./m3_sim_replay";
import type { M3Snapshot } from "./m3_types";

export interface M3PanelHandle {
  stop(): void;
  getSnapshot(): M3Snapshot;
}

export interface M3PanelElements {
  summaryEl: HTMLElement | null;
  forwardEl: HTMLElement | null;
  gnssEl: HTMLElement | null;
  relocEl: HTMLElement | null;
  fsmEl: HTMLElement | null;
  voiceEl: HTMLElement | null;
  e5xPreEl: HTMLElement | null;
}

export function renderSidecarVerificationDom(
  statusEl: HTMLElement | null,
  timelineEl: HTMLElement | null,
  result: SidecarChainVerificationResult
): void {
  if (statusEl) {
    statusEl.textContent = result.valid
      ? `CHAIN VALID (${result.events.length} events verified via WebCrypto SHA-256)`
      : `CHAIN INVALID: ${result.errorMessage}`;
    statusEl.setAttribute("data-valid", result.valid ? "1" : "0");
  }
  if (!timelineEl) return;
  timelineEl.replaceChildren();
  const doc = timelineEl.ownerDocument;
  for (const ev of result.events) {
    const li = doc.createElement("li");
    li.textContent =
      `#${ev.seq} (${(ev.tSimUs / 1e6).toFixed(2)}s) ${ev.kind} [${ev.snrStratum}, |C|=${ev.conformalSetSize}] ` +
      `${ev.eventSha256.slice(0, 10)}…`;
    timelineEl.appendChild(li);
  }
}

export function renderM3PanelDom(snap: M3Snapshot, els: M3PanelElements): void {
  if (els.summaryEl) {
    els.summaryEl.textContent = snap.summaryLine;
  }
  if (els.forwardEl) {
    const f = snap.forwardHazard;
    els.forwardEl.textContent =
      `Wires ${f.wireCount}/${f.wireCap} • Intruders ${f.intruderCount}/${f.intruderCap} • ` +
      `${f.grazingFallback ? "GRAZING FALLBACK" : "RADAR LOCK"} • Vmax ${f.vMaxCapMps.toFixed(1)} m/s`;
  }
  if (els.gnssEl) {
    const g = snap.gnssDenied;
    els.gnssEl.textContent =
      `${g.hypothesis} • MCUSUM ${g.mcusumNorm.toFixed(2)} (${g.surfaceMode}) • ` +
      `${g.rollbackActive ? "2.0s ROLLBACK" : "NO ROLLBACK"}`;
  }
  if (els.relocEl) {
    const g = snap.gnssDenied;
    els.relocEl.textContent = `${g.relocBackend} (${g.relocInliers} inliers)`;
  }
  if (els.fsmEl) {
    const g = snap.gnssDenied;
    els.fsmEl.textContent =
      `${g.navFsmState} • 1σ=${g.sigmaPosM.toFixed(1)} m / 3σ=${g.threeSigmaM.toFixed(1)} m • ` +
      `dh/dt=${g.dhDtMps.toFixed(2)} m/s • LTFJ/LTBX ${g.ltfjInflatedMarginM.toFixed(0)}/${g.ltbxInflatedMarginM.toFixed(0)} m`;
  }
  if (els.voiceEl) {
    const v = snap.voiceAtc;
    els.voiceEl.textContent =
      `${v.snrStratum} (${v.snrDb.toFixed(1)} dB) • ${v.asrTier} • |C|=${v.conformalSetSize} • ` +
      `${v.maydayPreempted ? `MAYDAY PREEMPT (w=${v.maydayPosterior.toFixed(2)})` : "NOMINAL"} • ` +
      `${v.ttsEngine} ${v.ttsTtfaMs.toFixed(0)} ms`;
  }
  if (els.e5xPreEl) {
    const e = snap.e5xDraft;
    els.e5xPreEl.textContent =
      `[${e.occurrenceClassId431} | ${e.cicttCategory} | Barrier#8=${e.barrier8Score} | ERCS ${e.ercsMatrixCode} | 0.0% num halluc]\n` +
      buildXsdValidE5xXml(snap);
  }
}

export function bindM3Panel(
  onSnapshot: (snapshot: M3Snapshot | null) => void,
  onStatus: (message: string) => void
): M3PanelHandle {
  const btnCompound = document.getElementById("btn-m3-compound") as HTMLButtonElement | null;
  const btnWire = document.getElementById("btn-m3-wire") as HTMLButtonElement | null;
  const btnSpoof = document.getElementById("btn-m3-spoof") as HTMLButtonElement | null;
  const btnWater = document.getElementById("btn-m3-water") as HTMLButtonElement | null;
  const btnVoice = document.getElementById("btn-m3-voice-mayday") as HTMLButtonElement | null;
  const btnE5x = document.getElementById("btn-m3-e5x-preview") as HTMLButtonElement | null;
  const btnE5xDownload = document.getElementById("btn-m3-e5x-download") as HTMLButtonElement | null;
  const btnSampleSidecar = document.getElementById("btn-m3-sample-sidecar") as HTMLButtonElement | null;
  const btnReset = document.getElementById("btn-m3-reset") as HTMLButtonElement | null;
  const e5xDetails = document.getElementById("m3-e5x-details") as HTMLDetailsElement | null;
  const sidecarDetails = document.getElementById("m3-sidecar-details") as HTMLDetailsElement | null;
  const sidecarDropzone = document.getElementById("m3-sidecar-dropzone");
  const sidecarInput = document.getElementById("m3-sidecar-file-input") as HTMLInputElement | null;
  const sidecarStatusEl = document.getElementById("m3-sidecar-status");
  const sidecarTimelineEl = document.getElementById("m3-sidecar-timeline");

  const els: M3PanelElements = {
    summaryEl: document.getElementById("m3-hud-summary"),
    forwardEl: document.getElementById("m3-forward-status"),
    gnssEl: document.getElementById("m3-gnss-status"),
    relocEl: document.getElementById("m3-reloc-status"),
    fsmEl: document.getElementById("m3-fsm-status"),
    voiceEl: document.getElementById("m3-voice-status"),
    e5xPreEl: document.getElementById("m3-e5x-xml-preview"),
  };

  const replay = new M3SimReplay();
  let current = replay.reset();
  let timerId: number | null = null;

  const emit = (snap: M3Snapshot) => {
    current = snap;
    renderM3PanelDom(snap, els);
    onSnapshot(snap);
  };

  const ensureTicker = () => {
    if (timerId !== null) return;
    timerId = window.setInterval(() => {
      emit(replay.step(0.5));
    }, 250);
  };

  const stop = () => {
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
  };

  const inspectSidecarText = async (text: string, sourceLabel: string) => {
    const res = await verifyVrtaEventsJsonlChain(text);
    if (sidecarDetails) {
      sidecarDetails.open = true;
    }
    renderSidecarVerificationDom(sidecarStatusEl, sidecarTimelineEl, res);
    onStatus(
      res.valid
        ? `Sidecar ${sourceLabel}: WebCrypto SHA-256 chain VERIFIED (${res.events.length} events).`
        : `Sidecar ${sourceLabel}: TAMPER / CHAIN FAILURE — ${res.errorMessage}`
    );
  };

  // Initial nominal render
  renderM3PanelDom(current, els);
  onSnapshot(current);

  btnCompound?.addEventListener("click", () => {
    const snap = replay.activateScenario("compound_apex");
    emit(snap);
    ensureTicker();
    onStatus(
      "M3 6-Fault Compound Apex (E1): Rotor #3 CSFL + 12 m/s wind + H3 GNSS spoof + grazing wire 15.0 m/s + Part-IS quarantine + S2 TR MAYDAY."
    );
  });

  btnWire?.addEventListener("click", () => {
    stop();
    const snap = replay.activateScenario("wire_grazing");
    emit(snap);
    onStatus(
      `M3 Forward Wire Grazing: skew 28.4° > 20° cliff (-45 dBsm) -> CBVF cap ${snap.forwardHazard.vMaxCapMps.toFixed(1)} m/s.`
    );
  });

  btnSpoof?.addEventListener("click", () => {
    const snap = replay.activateScenario("slow_ramp_spoof");
    emit(snap);
    ensureTicker();
    onStatus(
      `M3 H3 Slow-Ramp Spoof latched in ${snap.gnssDenied.ttdSec?.toFixed(2)} s (MCUSUM ${snap.gnssDenied.mcusumNorm.toFixed(2)}, 2.0 s VIO rollback).`
    );
  });

  btnWater?.addEventListener("click", () => {
    const snap = replay.activateScenario("water_drift");
    emit(snap);
    ensureTicker();
    onStatus(
      `M3 Marmara Open-Water Drift: ${snap.gnssDenied.navFsmState} (3σ=${snap.gnssDenied.threeSigmaM.toFixed(1)} m, dh/dt=${snap.gnssDenied.dhDtMps.toFixed(2)} m/s).`
    );
  });

  btnVoice?.addEventListener("click", () => {
    stop();
    const snap = replay.activateScenario("voice_mayday");
    emit(snap);
    onStatus(
      `M3 Voice ATC ${snap.voiceAtc.snrStratum}: MAYDAY w=${snap.voiceAtc.maydayPosterior.toFixed(2)} preempted clarification (${snap.voiceAtc.ttsTtfaMs.toFixed(0)} ms TTFA).`
    );
  });

  btnE5x?.addEventListener("click", () => {
    if (e5xDetails) {
      e5xDetails.open = !e5xDetails.open;
    }
    renderM3PanelDom(current, els);
    onStatus(
      `M3 Forensics .e5x Draft: ${current.e5xDraft.occurrenceClassId431} (${current.e5xDraft.cicttCategory}, ERCS ${current.e5xDraft.ercsMatrixCode}, Barrier #8=0).`
    );
  });

  btnE5xDownload?.addEventListener("click", () => {
    const fname = triggerE5xXmlDownload(current);
    onStatus(
      `Downloaded XSD-valid draft ${fname} (${current.e5xDraft.occurrenceClassId431}, ${current.e5xDraft.cicttCategory}, Barrier #8=0).`
    );
  });

  btnSampleSidecar?.addEventListener("click", () => {
    void buildSampleSignedSidecarJsonl().then((jsonl) =>
      inspectSidecarText(jsonl, "sample_compound.vrta_events.jsonl")
    );
  });

  sidecarInput?.addEventListener("change", () => {
    const file = sidecarInput.files?.[0];
    if (!file) return;
    void file.text().then((txt) => inspectSidecarText(txt, file.name));
  });

  sidecarDropzone?.addEventListener("dragover", (evt) => {
    evt.preventDefault();
  });

  sidecarDropzone?.addEventListener("drop", (evt) => {
    evt.preventDefault();
    const file = (evt as DragEvent).dataTransfer?.files?.[0];
    if (!file) return;
    void file.text().then((txt) => inspectSidecarText(txt, file.name));
  });

  btnReset?.addEventListener("click", () => {
    stop();
    const snap = replay.reset();
    emit(snap);
    onStatus("M3 Mission Control reset to nominal Kartal coastal corridor.");
  });

  return {
    stop,
    getSnapshot: () => current,
  };
}
