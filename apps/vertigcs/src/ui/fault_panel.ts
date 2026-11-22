import type { FaultCommand, FaultCommandKind } from "../types/telemetry";

export function bindFaultPanel(onFault: (cmd: FaultCommand) => void): void {
  const panel = document.getElementById("fault-panel");
  if (!panel) return;

  panel.addEventListener("click", (ev) => {
    const btn = (ev.target as HTMLElement | null)?.closest("button[data-fault]") as HTMLButtonElement | null;
    if (!btn) return;
    const kind = btn.dataset.fault as FaultCommandKind | undefined;
    const targetIdx = Number(btn.dataset.target ?? "0");
    if (!kind) return;
    onFault({ kind, targetIdx: Number.isFinite(targetIdx) ? targetIdx : 0 });
  });
}
