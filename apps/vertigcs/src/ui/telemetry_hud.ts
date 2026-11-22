import type { TelemetrySnapshot } from "../types/telemetry";

export class TelemetryHud {
  private readonly rotorBarEls: HTMLDivElement[] = [];

  constructor() {
    const container = document.getElementById("hud-rotors");
    if (container) {
      container.replaceChildren();
      for (let i = 0; i < 8; i++) {
        const wrap = document.createElement("div");
        wrap.className = "rotor-cell";
        const bar = document.createElement("div");
        bar.className = "rotor-fill";
        const lbl = document.createElement("span");
        lbl.textContent = `M${i + 1}`;
        wrap.append(bar, lbl);
        container.append(wrap);
        this.rotorBarEls.push(bar);
      }
    }
  }

  render(snap: TelemetrySnapshot): void {
    this.setText("hud-regime", snap.flightRegimeLabel);
    this.setText("hud-alt", `${snap.aglM.toFixed(1)} / ${snap.altHaeM.toFixed(1)} m`);
    this.setText("hud-batt", `${snap.socPct.toFixed(1)}% / ${snap.vTermV.toFixed(1)} V`);
    this.setText("hud-tmr", `${snap.tmrStatusLabel} / ${snap.chi2Residual.toFixed(2)}`);
    this.setText("hud-cbvf", `${snap.cbvfMinV >= 0 ? "+" : ""}${snap.cbvfMinV.toFixed(1)} m`);
    this.setText("hud-ctr", `${snap.ltfjMarginM.toFixed(0)} / ${snap.ltbxMarginM.toFixed(0)} m`);

    const activeCount = countBits8(snap.activeMotorMask);
    const hexMask = `0x${(snap.activeMotorMask & 0xff).toString(16).toUpperCase().padStart(2, "0")}`;
    this.setText("hud-mask", `${hexMask} (${activeCount}/8)`);
    this.setText("hud-cel", snap.celLatched ? "LATCHED (2.2 rad/s SPIN)" : "DISARMED");

    for (let i = 0; i < this.rotorBarEls.length; i++) {
      const rpm = snap.rotorRpm[i] ?? 0;
      const pct = Math.min(100, Math.max(0, (rpm / 2292.0) * 100));
      const alive = (snap.activeMotorMask & (1 << i)) !== 0;
      const el = this.rotorBarEls[i];
      el.style.height = `${pct.toFixed(0)}%`;
      el.dataset.fault = alive ? "0" : "1";
    }
  }

  private setText(id: string, value: string): void {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }
}

function countBits8(mask: number): number {
  let n = 0;
  for (let i = 0; i < 8; i++) {
    if ((mask & (1 << i)) !== 0) n++;
  }
  return n;
}
