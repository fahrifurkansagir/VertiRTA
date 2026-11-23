// SPDX-License-Identifier: Apache-2.0
// Zero-copy TypedArray Deck.gl layers for 20-eVTOL Anatolian U-Space Fleet & Part-IS Cyber Overlay

import type { Layer } from "@deck.gl/core";
import { ScatterplotLayer } from "@deck.gl/layers";
import type { FleetSnapshot, PreemptionWaveState, VertiportStandState } from "./fleet_types";

export function buildFleetDeckLayers(snapshot: FleetSnapshot | null): Layer[] {
  if (!snapshot || snapshot.aircraft.length === 0) {
    return [];
  }

  const n = snapshot.aircraft.length;
  const posBuf = new Float32Array(n * 3);
  const colBuf = new Uint8Array(n * 4);

  for (let i = 0; i < n; i++) {
    const ac = snapshot.aircraft[i];
    posBuf[i * 3 + 0] = ac.lonDeg;
    posBuf[i * 3 + 1] = ac.latDeg;
    posBuf[i * 3 + 2] = ac.altAmslM;

    if (ac.celActive) {
      // Priority-0 CEL: vivid coral-red
      colBuf[i * 4 + 0] = 255;
      colBuf[i * 4 + 1] = 61;
      colBuf[i * 4 + 2] = 0;
      colBuf[i * 4 + 3] = 255;
    } else if (ac.cyberState === "SEC_USSP_QUARANTINE" || ac.cyberState === "SEC_AUTONOMOUS_RTB") {
      // Part-IS USSP quarantine / autonomous RTB: magenta-amber alert
      colBuf[i * 4 + 0] = 234;
      colBuf[i * 4 + 1] = 67;
      colBuf[i * 4 + 2] = 255;
      colBuf[i * 4 + 3] = 255;
    } else if (ac.cyberState === "SEC_SUSPECT_PEER") {
      // ReplayWindow128 demoted peer: amber warning
      colBuf[i * 4 + 0] = 255;
      colBuf[i * 4 + 1] = 196;
      colBuf[i * 4 + 2] = 0;
      colBuf[i * 4 + 3] = 255;
    } else if (ac.preempted) {
      // Yielding peer opening >= 25 m CEL corridor: bright cyan
      colBuf[i * 4 + 0] = 64;
      colBuf[i * 4 + 1] = 196;
      colBuf[i * 4 + 2] = 255;
      colBuf[i * 4 + 3] = 235;
    } else if (ac.lane === "SE_70M") {
      // Nominal 70 m SE lane: emerald green
      colBuf[i * 4 + 0] = 0;
      colBuf[i * 4 + 1] = 230;
      colBuf[i * 4 + 2] = 118;
      colBuf[i * 4 + 3] = 230;
    } else {
      // Nominal 105 m NW lane: sky teal
      colBuf[i * 4 + 0] = 38;
      colBuf[i * 4 + 1] = 198;
      colBuf[i * 4 + 2] = 218;
      colBuf[i * 4 + 3] = 230;
    }
  }

  const layers: Layer[] = [
    new ScatterplotLayer<VertiportStandState>({
      id: "uspace-vertiport-chilled-stands",
      data: snapshot.vertiports,
      getPosition: (v: VertiportStandState): [number, number, number] => [
        v.lonDeg,
        v.latDeg,
        v.altAmslM + 4,
      ],
      getRadius: 85,
      radiusMinPixels: 7,
      stroked: true,
      filled: true,
      getFillColor: (v: VertiportStandState): [number, number, number, number] =>
        v.fatoEmgReserved ? [255, 82, 82, 160] : [0, 188, 212, 120],
      getLineColor: [255, 255, 255, 220],
      lineWidthMinPixels: 2,
      updateTriggers: {
        getFillColor: [snapshot.vertiports.map((v) => v.fatoEmgReserved).join(",")],
      },
    }),
    new ScatterplotLayer({
      id: "uspace-fleet-20-aircraft",
      data: {
        length: n,
        attributes: {
          getPosition: { value: posBuf, size: 3 },
          getFillColor: { value: colBuf, size: 4 },
        },
      },
      getRadius: 42,
      radiusMinPixels: 6,
      stroked: true,
      getLineColor: [255, 255, 255, 230],
      lineWidthMinPixels: 1.5,
      updateTriggers: {
        getPosition: [snapshot.simTimeS],
        getFillColor: [snapshot.activeCyberState, snapshot.preemptionWave?.active ?? false],
      },
    }),
  ];

  if (snapshot.preemptionWave?.active) {
    layers.push(
      new ScatterplotLayer<PreemptionWaveState>({
        id: "uspace-cel-preemption-wave",
        data: [snapshot.preemptionWave],
        getPosition: (w: PreemptionWaveState): [number, number, number] => [
          w.centerLonDeg,
          w.centerLatDeg,
          75,
        ],
        getRadius: (w: PreemptionWaveState) => w.radiusM,
        radiusMinPixels: 18,
        stroked: true,
        filled: true,
        getFillColor: [255, 61, 0, 45],
        getLineColor: [255, 110, 64, 240],
        lineWidthMinPixels: 2.5,
        updateTriggers: {
          getRadius: [snapshot.preemptionWave.radiusM],
        },
      })
    );
  }

  return layers;
}
