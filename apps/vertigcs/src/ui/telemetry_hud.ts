import type { TelemetrySnapshot } from "../types/telemetry";

type SeverityState = "nominal" | "caution" | "danger";

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
    this.setText("hud-tick", `t=${snap.simTimeS.toFixed(1)}s • #${snap.seq}`);

    const activeCount = countBits8(snap.activeMotorMask);
    const hexMask = `0x${(snap.activeMotorMask & 0xff).toString(16).toUpperCase().padStart(2, "0")}`;
    this.setText("hud-mask", `${hexMask} (${activeCount}/8)`);
    this.setText("hud-cel", snap.celLatched ? "LATCHED 2.2 rad/s" : "DISARMED");

    const regimeState: SeverityState = snap.celLatched
      ? "danger"
      : activeCount < 8 || snap.chi2Residual > 20.0
        ? "caution"
        : "nominal";
    this.setState("hud-regime-card", regimeState);
    this.setState("hud-cel-card", snap.celLatched ? "danger" : "nominal");
    this.setState(
      "hud-tmr-card",
      snap.chi2Residual > 20.0 || snap.tmrStatusLabel !== "3/3_NOMINAL" ? "caution" : "nominal"
    );
    this.setState(
      "hud-cbvf-card",
      snap.cbvfMinV < 5.0 ? "danger" : snap.cbvfMinV < 12.0 ? "caution" : "nominal"
    );
    this.setState(
      "hud-ctr-card",
      snap.ltfjMarginM < 250 || snap.ltbxMarginM < 250
        ? "danger"
        : snap.ltfjMarginM < 600
          ? "caution"
          : "nominal"
    );

    const mapAlertEl = document.getElementById("map-alert-banner");
    const mapAlertText = document.getElementById("map-alert-text");
    if (mapAlertEl && mapAlertText) {
      mapAlertEl.dataset.state = regimeState;
      if (snap.celLatched) {
        mapAlertText.textContent = `CEL SPIN LATCH ACTIVE (2.2 rad/s) • AGL ${snap.aglM.toFixed(1)} m • LTFJ ${snap.ltfjMarginM.toFixed(0)} m`;
      } else if (activeCount < 8) {
        mapAlertText.textContent = `DEGRADED CSFL (${activeCount}/8 ROTORS) • CBVF +${snap.cbvfMinV.toFixed(1)} m`;
      } else if (snap.chi2Residual > 20.0) {
        mapAlertText.textContent = `TMR IMU ISOLATION (${snap.tmrStatusLabel}) • χ²=${snap.chi2Residual.toFixed(1)}`;
      } else {
        mapAlertText.textContent = `RTA ENVELOPE NOMINAL • LTFJ ${snap.ltfjMarginM.toFixed(0)} m / LTBX ${snap.ltbxMarginM.toFixed(0)} m`;
      }
    }

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

  private setState(id: string, state: SeverityState): void {
    const el = document.getElementById(id);
    if (el) el.dataset.state = state;
  }
}

function countBits8(mask: number): number {
  let n = 0;
  for (let i = 0; i < 8; i++) {
    if ((mask & (1 << i)) !== 0) n++;
  }
  return n;
}
