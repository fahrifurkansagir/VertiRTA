import { parseVrtaFile } from "../vrta/vrta_parser";
import { vrtaFrameToSnapshot } from "../transport/flight_transport";
import type { TelemetrySnapshot, VrtaFileHeader, VrtaFrame100Hz } from "../types/telemetry";

export class VrtaReplayScrubber {
  private header: VrtaFileHeader | null = null;
  private frames: VrtaFrame100Hz[] = [];
  private cursor = 0;
  private playing = false;
  private timerId: number | null = null;

  constructor(
    private readonly onFrame: (snap: TelemetrySnapshot) => void,
    private readonly onStatus: (msg: string) => void
  ) {
    this.bindDom();
  }

  private bindDom(): void {
    const input = document.getElementById("vrta-file-input") as HTMLInputElement | null;
    const dropzone = document.getElementById("vrta-dropzone");
    const playBtn = document.getElementById("btn-play-pause") as HTMLButtonElement | null;
    const slider = document.getElementById("replay-slider") as HTMLInputElement | null;

    const handleFile = async (file: File) => {
      try {
        const buf = await file.arrayBuffer();
        const parsed = parseVrtaFile(buf);
        this.header = parsed.header;
        this.frames = parsed.frames;
        this.cursor = 0;
        if (slider) {
          slider.max = String(Math.max(0, this.frames.length - 1));
          slider.value = "0";
        }
        this.onStatus(
          `Loaded .vrta (${file.name}): ${this.frames.length} frames, datum=(${this.header.datumLatDeg.toFixed(4)}, ${this.header.datumLonDeg.toFixed(4)}), N0=${this.header.egm2008NM.toFixed(1)} m`
        );
        this.emitCurrent();
        this.setPlaying(true);
      } catch (err) {
        this.onStatus(`VRTA parse error: ${err instanceof Error ? err.message : String(err)}`);
      }
    };

    input?.addEventListener("change", () => {
      const f = input.files?.[0];
      if (f) void handleFile(f);
    });

    dropzone?.addEventListener("dragover", (ev) => {
      ev.preventDefault();
    });
    dropzone?.addEventListener("drop", (ev) => {
      ev.preventDefault();
      const f = ev.dataTransfer?.files?.[0];
      if (f) void handleFile(f);
    });

    playBtn?.addEventListener("click", () => {
      this.setPlaying(!this.playing);
      if (playBtn) playBtn.textContent = this.playing ? "Pause" : "Play";
    });

    slider?.addEventListener("input", () => {
      this.cursor = Math.min(this.frames.length - 1, Math.max(0, Number(slider.value)));
      this.emitCurrent();
    });
  }

  private setPlaying(active: boolean): void {
    this.playing = active;
    if (this.timerId !== null) {
      window.clearInterval(this.timerId);
      this.timerId = null;
    }
    if (active && this.frames.length > 0) {
      this.timerId = window.setInterval(() => {
        if (this.cursor >= this.frames.length - 1) {
          this.cursor = 0;
        } else {
          this.cursor++;
        }
        this.emitCurrent();
      }, 33);
    }
  }

  private emitCurrent(): void {
    if (!this.header || this.frames.length === 0) return;
    const frame = this.frames[this.cursor];
    const snap = vrtaFrameToSnapshot(frame, this.header);
    const slider = document.getElementById("replay-slider") as HTMLInputElement | null;
    const timeEl = document.getElementById("replay-time");
    if (slider) slider.value = String(this.cursor);
    if (timeEl) timeEl.textContent = `${snap.simTimeS.toFixed(2)} s`;
    this.onFrame(snap);
  }
}
