// SPDX-License-Identifier: Apache-2.0
// Accessible WAI-ARIA tab controller for VertiGCS right cockpit sidebar.

export type HudTabId = "tab-flight" | "tab-fleet" | "tab-m3";

export interface HudTabsHandle {
  selectTab(tabId: HudTabId): void;
  setFleetBadge(activeCount: number): void;
  setM3AlertBadge(active: boolean): void;
}

export function initHudTabs(): HudTabsHandle {
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('.hud-tab[role="tab"]'));
  const panes = new Map<string, HTMLElement>();
  for (const tab of tabs) {
    const targetId = tab.dataset.tabTarget;
    if (targetId) {
      const pane = document.getElementById(targetId);
      if (pane) panes.set(tab.id, pane);
    }
  }

  const selectTab = (tabId: HudTabId, moveFocus = false): void => {
    for (const tab of tabs) {
      const active = tab.id === tabId;
      tab.setAttribute("aria-selected", active ? "true" : "false");
      tab.tabIndex = active ? 0 : -1;
      const pane = panes.get(tab.id);
      if (pane) {
        pane.hidden = !active;
      }
      if (active && moveFocus) {
        tab.focus();
      }
    }
  };

  for (let i = 0; i < tabs.length; i++) {
    const tab = tabs[i];
    tab.addEventListener("click", () => {
      selectTab(tab.id as HudTabId, false);
    });
    tab.addEventListener("keydown", (ev) => {
      if (tabs.length === 0) return;
      let nextIdx = -1;
      if (ev.key === "ArrowRight") nextIdx = (i + 1) % tabs.length;
      else if (ev.key === "ArrowLeft") nextIdx = (i - 1 + tabs.length) % tabs.length;
      else if (ev.key === "Home") nextIdx = 0;
      else if (ev.key === "End") nextIdx = tabs.length - 1;
      if (nextIdx >= 0) {
        ev.preventDefault();
        selectTab(tabs[nextIdx].id as HudTabId, true);
      }
    });
  }

  const fleetBadge = document.getElementById("tab-badge-fleet");
  const m3Badge = document.getElementById("tab-badge-m3");

  return {
    selectTab: (tabId: HudTabId) => selectTab(tabId, false),
    setFleetBadge: (activeCount: number) => {
      if (!fleetBadge) return;
      fleetBadge.hidden = activeCount <= 0;
      if (activeCount > 0) fleetBadge.textContent = String(activeCount);
    },
    setM3AlertBadge: (active: boolean) => {
      if (!m3Badge) return;
      m3Badge.hidden = !active;
    },
  };
}
