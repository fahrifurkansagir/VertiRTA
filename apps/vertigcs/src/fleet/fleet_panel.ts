// SPDX-License-Identifier: Apache-2.0
// Interactive 20-eVTOL U-Space Fleet & 1-Click Part-IS Cyber-Attack Simulator Panel

import { FleetCorridorController } from "./fleet_sim_replay";
import type { FleetSnapshot } from "./fleet_types";

export interface FleetPanelHandle {
  stop(): void;
  isEnabled(): boolean;
}

export function bindFleetPanel(
  onSnapshot: (snapshot: FleetSnapshot | null) => void,
  onStatus: (message: string) => void
): FleetPanelHandle {
  const btnToggle = document.getElementById("btn-fleet-toggle") as HTMLButtonElement | null;
  const btnCel = document.getElementById("btn-fleet-cel") as HTMLButtonElement | null;
  const btnReplay = document.getElementById("btn-fleet-replay") as HTMLButtonElement | null;
  const btnSpoof = document.getElementById("btn-fleet-spoof") as HTMLButtonElement | null;
  const btnReset = document.getElementById("btn-fleet-reset") as HTMLButtonElement | null;
  const summaryEl = document.getElementById("fleet-hud-summary");
  const logEl = document.getElementById("fleet-cyber-log");

  const controller = new FleetCorridorController(20);
  let enabled = false;
  let timerId: number | null = null;

  const renderHud = (snap: FleetSnapshot | null) => {
    if (!snap) {
      if (summaryEl) {
        summaryEl.textContent = "Standby (0/20 active) • 15°C Chilled Stands Ready (4×4)";
      }
      if (logEl) {
        logEl.textContent = "Fleet overlay paused — click Enable to launch 20-eVTOL Anatolian U-Space ops.";
      }
      return;
    }

    const totalOccupied = snap.vertiports.reduce((acc, v) => acc + v.standsOccupied, 0);
    const totalStands = snap.vertiports.reduce((acc, v) => acc + v.standsTotal, 0);
    const celCount = snap.aircraft.filter((a) => a.celActive).length;
    if (summaryEl) {
      summaryEl.textContent =
        `${snap.aircraft.length} eVTOLs (<29.1895°E) • 15°C Stands ${totalOccupied}/${totalStands} • ` +
        `CEL P0: ${celCount} • ${snap.activeCyberState}`;
    }
    if (logEl) {
      logEl.textContent = snap.cyberSummary;
    }
  };

  const ensureEnabled = () => {
    if (enabled) return;
    enabled = true;
    if (btnToggle) {
      btnToggle.textContent = "Disable 20-eVTOL U-Space Fleet";
    }
    const snap = controller.snapshot();
    renderHud(snap);
    onSnapshot(snap);
    timerId = window.setInterval(() => {
      controller.step(0.1);
      const next = controller.snapshot();
      renderHud(next);
      onSnapshot(next);
    }, 100);
  };

  const stop = () => {
    enabled = false;
    if (timerId !== null) {
      window.clearInterval(timerId);
      timerId = null;
    }
    if (btnToggle) {
      btnToggle.textContent = "Enable 20-eVTOL U-Space Fleet";
    }
    renderHud(null);
    onSnapshot(null);
  };

  btnToggle?.addEventListener("click", () => {
    if (enabled) {
      stop();
      onStatus("20-eVTOL U-Space fleet overlay disabled.");
    } else {
      ensureEnabled();
      onStatus("20-eVTOL U-Space fleet active (4 vertiports, 15°C chilled stands, <29.1895°E).");
    }
  });

  btnCel?.addEventListener("click", () => {
    ensureEnabled();
    const evt = controller.triggerCelPreemption(0);
    const snap = controller.snapshot();
    renderHud(snap);
    onSnapshot(snap);
    onStatus(
      `CEL P0 Preemption: ${evt.callsign} -> ${evt.preemptedPeers} peers yielded in ${evt.v2vLatencyMs} ms (${evt.targetFatoEmg} FATO-EMG).`
    );
  });

  btnReplay?.addEventListener("click", () => {
    ensureEnabled();
    const evt = controller.injectPartIsReplayAttack(3);
    const snap = controller.snapshot();
    renderHud(snap);
    onSnapshot(snap);
    onStatus(
      `Part-IS Replay Attack: nonce #${evt.duplicateNonce} dropped by ReplayWindow128 (${evt.callsign} -> ${evt.cyberState}).`
    );
  });

  btnSpoof?.addEventListener("click", () => {
    ensureEnabled();
    const evt = controller.injectSpoofedUsspIntoLtfj(1);
    const snap = controller.snapshot();
    renderHud(snap);
    onSnapshot(snap);
    onStatus(
      `Part-IS Spoofed USSP blocked (${evt.spoofedLonDeg.toFixed(4)}°E -> LTFJ): ${evt.callsign} ${evt.cyberState}, RTB ${evt.divertVertiportCode} (RTO ${evt.rtoSec.toFixed(1)} s).`
    );
  });

  btnReset?.addEventListener("click", () => {
    controller.reset();
    if (enabled) {
      const snap = controller.snapshot();
      renderHud(snap);
      onSnapshot(snap);
    } else {
      renderHud(null);
    }
    onStatus("20-eVTOL U-Space fleet & Part-IS cyber state reset to SEC_NOMINAL.");
  });

  renderHud(null);
  return {
    stop,
    isEnabled: () => enabled,
  };
}
