// SPDX-License-Identifier: Apache-2.0
// Deck.gl 3D Overlay for M3 Forward Wire Capsules & Dynamic 3-sigma + b_map CTR Walls

import { PathLayer, PolygonLayer } from "@deck.gl/layers";
import {
  VERIFIED_LTBX_ARP,
  VERIFIED_LTFJ_ARP,
  type M3Snapshot,
  type WireCapsule3D,
} from "./m3_types";

const METERS_PER_DEG_LAT = 111052.0;
const METERS_PER_DEG_LON = 84255.0;

export interface CtrInflationPolygonDatum {
  id: "LTFJ_CTR_INFLATED" | "LTBX_CTR_INFLATED";
  centerLatDeg: number;
  centerLonDeg: number;
  inflatedRadiusM: number;
  ceilingAmslM: number;
  polygon: [number, number][];
}

export function buildInflatedCtrRing(
  centerLatDeg: number,
  centerLonDeg: number,
  radiusM: number,
  segments = 48
): [number, number][] {
  const ring: [number, number][] = [];
  for (let i = 0; i <= segments; ++i) {
    const theta = (2.0 * Math.PI * i) / segments;
    const dLat = (radiusM * Math.sin(theta)) / METERS_PER_DEG_LAT;
    const dLon = (radiusM * Math.cos(theta)) / METERS_PER_DEG_LON;
    ring.push([centerLonDeg + dLon, centerLatDeg + dLat]);
  }
  return ring;
}

export function buildM3CtrInflationData(snapshot: M3Snapshot): CtrInflationPolygonDatum[] {
  const deltaR = Math.max(0.0, snapshot.gnssDenied.ctrInflationM);
  return [
    {
      id: "LTFJ_CTR_INFLATED",
      centerLatDeg: VERIFIED_LTFJ_ARP.latDeg,
      centerLonDeg: VERIFIED_LTFJ_ARP.lonDeg,
      inflatedRadiusM: VERIFIED_LTFJ_ARP.radiusM + deltaR,
      ceilingAmslM: 762,
      polygon: buildInflatedCtrRing(
        VERIFIED_LTFJ_ARP.latDeg,
        VERIFIED_LTFJ_ARP.lonDeg,
        VERIFIED_LTFJ_ARP.radiusM + deltaR
      ),
    },
    {
      id: "LTBX_CTR_INFLATED",
      centerLatDeg: VERIFIED_LTBX_ARP.latDeg,
      centerLonDeg: VERIFIED_LTBX_ARP.lonDeg,
      inflatedRadiusM: VERIFIED_LTBX_ARP.radiusM + deltaR,
      ceilingAmslM: 610,
      polygon: buildInflatedCtrRing(
        VERIFIED_LTBX_ARP.latDeg,
        VERIFIED_LTBX_ARP.lonDeg,
        VERIFIED_LTBX_ARP.radiusM + deltaR
      ),
    },
  ];
}

function wireClassColor(wire: WireCapsule3D): [number, number, number, number] {
  if (!wire.radarConfirmed) {
    return [255, 64, 129, 250];
  }
  if (wire.wireClass === "transmission") {
    return [255, 112, 67, 240];
  }
  if (wire.wireClass === "crane_cable") {
    return [255, 82, 82, 245];
  }
  return [255, 202, 40, 235];
}

export function buildM3DeckLayers(snapshot: M3Snapshot) {
  const wireLayer = new PathLayer<WireCapsule3D>({
    id: "m3-forward-wire-capsules",
    data: snapshot.forwardHazard.wires,
    getPath: (w) => [w.p0, w.p1],
    getColor: (w) => wireClassColor(w),
    getWidth: (w) => Math.max(2.5, w.radiusM * 0.75),
    widthMinPixels: 3,
    updateTriggers: {
      getColor: [snapshot.scenario, snapshot.forwardHazard.wireCount],
      getWidth: [snapshot.forwardHazard.grazingFallback],
    },
  });

  const ctrData = buildM3CtrInflationData(snapshot);
  const inflated = snapshot.gnssDenied.ctrInflationM > 10.0;
  const ctrLayer = new PolygonLayer<CtrInflationPolygonDatum>({
    id: "m3-ctr-inflated-walls",
    data: ctrData,
    stroked: true,
    filled: true,
    extruded: false,
    getPolygon: (d) => d.polygon,
    getFillColor: (d) =>
      d.id === "LTFJ_CTR_INFLATED"
        ? [255, 82, 82, inflated ? 48 : 22]
        : [255, 193, 7, inflated ? 42 : 18],
    getLineColor: (d) =>
      d.id === "LTFJ_CTR_INFLATED" ? [255, 82, 82, 235] : [255, 213, 79, 225],
    lineWidthMinPixels: inflated ? 3 : 1.5,
    updateTriggers: {
      getPolygon: [Math.round(snapshot.gnssDenied.ctrInflationM)],
      getFillColor: [inflated],
    },
  });

  return [ctrLayer, wireLayer];
}
