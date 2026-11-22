/**
 * VertiRTA-X8-240 Live Tactical Earth Console (CesiumJS, Port :5180)
 *
 * Hardened for local NVIDIA GTX 1650 SUPER (4 GB VRAM, < 1.8 GB target):
 * - shadows: false, msaaSamples: 2, requestRenderMode: true,
 *   maximumRenderTimeChange: Infinity, viewer.scene.requestRender() per WS tick.
 * - globe.depthTestAgainstTerrain = true, globe.tileCacheSize = 64.
 * - osmBuildings.cacheBytes = 256 * 1024 * 1024,
 *   maximumCacheOverflowBytes = 128 * 1024 * 1024, maximumScreenSpaceError = 24.
 * - Swappable Layer-1 live-streaming basemapProvider (NO hardcoded keyless
 *   arcgisonline.com endpoint; API tokens injected at runtime via window.VERTIRTA_CONFIG).
 * - Consumes vrta.ws.v1.2 footprint_lat_lon_hae ([lat, lon, hae]) and blends
 *   landing contact using scene.sampleHeightMostDetailed with AGL <= 15.0 m Gaussian.
 */

const DEFAULT_CONFIG = Object.freeze({
  wsUrl: "ws://127.0.0.1:5180/telemetry",
  airspaceGeoJsonUrl: "/api/airspace.geojson",
  // Swappable providers: "open_grid_offline" | "ion_bing" | "esri_key" | "azure_maps"
  basemapProvider: "open_grid_offline",
  cesiumIonToken: "",
  esriApiKey: "",
  esriMapServerUrl: "",
  azureMapsKey: "",
  aircraftModelUri: "",
});

const KARTAL_DATUM = Object.freeze({
  latDeg: 40.8860,
  lonDeg: 29.1880,
  altAmslM: 12.0,
  geoidNM: 36.68,
});

const TOUCHDOWN_BLEND_SIGMA_M = 15.0;

export function computeTouchdownBlendHae(hWgs84M, deltaZSurfM, aglM) {
  const aglClamped = Math.max(0.0, Number(aglM) || 0.0);
  const weight = Math.exp(-Math.pow(aglClamped / 15.0, 2));
  return Number(hWgs84M) + Number(deltaZSurfM) * weight;
}

function createSwappableBaseLayer(cfg) {
  if (cfg.basemapProvider === "ion_bing" && cfg.cesiumIonToken) {
    Cesium.Ion.defaultAccessToken = cfg.cesiumIonToken;
    return Cesium.ImageryLayer.fromProviderAsync(
      Cesium.IonImageryProvider.fromAssetId(2)
    );
  }
  if (cfg.basemapProvider === "esri_key" && cfg.esriApiKey && cfg.esriMapServerUrl) {
    const resource = new Cesium.Resource({
      url: cfg.esriMapServerUrl,
      queryParameters: { token: cfg.esriApiKey },
    });
    return Cesium.ImageryLayer.fromProviderAsync(
      Cesium.ArcGisMapServerImageryProvider.fromUrl(resource)
    );
  }
  if (cfg.basemapProvider === "azure_maps" && cfg.azureMapsKey) {
    return new Cesium.ImageryLayer(
      new Cesium.UrlTemplateImageryProvider({
        url: `https://atlas.microsoft.com/map/imagery/png?api-version=2.0&style=satellite&zoom={z}&x={x}&y={y}&subscription-key=${encodeURIComponent(cfg.azureMapsKey)}`,
        credit: "Microsoft Azure Maps",
      })
    );
  }
  // Safe zero-token fallback grid (never scrapes keyless commercial tiles)
  return new Cesium.ImageryLayer(
    new Cesium.GridImageryProvider({
      color: Cesium.Color.fromCssColorString("#1c2b3a"),
      backgroundColor: Cesium.Color.fromCssColorString("#0d1520"),
    })
  );
}

function updateHudDom(pkt) {
  const setTxt = (id, text) => {
    const el = typeof document !== "undefined" ? document.getElementById(id) : null;
    if (el) el.textContent = text;
  };
  const g = pkt.geodetic;
  const p = pkt.propulsion;
  const s = pkt.safety_avionics;
  setTxt("hud-regime", s.flight_regime);
  setTxt("hud-speed", `${s.airspeed_mps.toFixed(1)} m/s`);
  setTxt("hud-alt", `${g.agl_m.toFixed(1)} / ${g.alt_hae_wgs84_m.toFixed(1)} m`);
  setTxt("hud-batt", `${s.soc_pct.toFixed(1)}% / ${s.v_term_v.toFixed(1)} V`);
  setTxt("hud-tmr", `${s.tmr_status} / ${s.cbf_delta_u_norm.toFixed(2)}`);
  setTxt("hud-ctr", `${s.dist_ltfj_ctr_m.toFixed(0)} / ${s.dist_ltbx_ctr_m.toFixed(0)} m`);
  const activeCount = 8 - p.failed_motor_indices.length;
  setTxt("hud-mask", `0x${p.active_motor_mask.toString(16).toUpperCase().padStart(2, "0")} (${activeCount}/8)`);
}

async function addAirspaceCtrWalls(viewer, geoJsonUrl) {
  const resp = await fetch(geoJsonUrl);
  if (!resp.ok) return;
  const fc = await resp.json();
  for (const feat of fc.features || []) {
    const props = feat.properties || {};
    const ring = feat.geometry?.coordinates?.[0] || [];
    if (ring.length < 4) continue;
    const isLtfj = props.icao === "LTFJ";
    const ceilingHae = Number(props.ceiling_hae_m || (isLtfj ? 798.58 : 646.35));
    const wallColor = isLtfj
      ? Cesium.Color.fromCssColorString("#e53935").withAlpha(0.18)
      : Cesium.Color.fromCssColorString("#ffb300").withAlpha(0.16);
    const edgeColor = isLtfj
      ? Cesium.Color.fromCssColorString("#ff5252")
      : Cesium.Color.fromCssColorString("#ffd54f");

    const positions = ring.map(([lon, lat]) => Cesium.Cartesian3.fromDegrees(lon, lat, ceilingHae));
    const minHeights = ring.map(() => 0.0);
    const maxHeights = ring.map(() => ceilingHae);

    viewer.entities.add({
      name: `${props.icao || "CTR"} Wall`,
      wall: {
        positions,
        minimumHeights: minHeights,
        maximumHeights: maxHeights,
        material: wallColor,
        outline: true,
        outlineColor: edgeColor,
      },
    });
  }
}

export async function initVertiRtaCesiumConsole(userConfig = {}) {
  const cfg = { ...DEFAULT_CONFIG, ...(window.VERTIRTA_CONFIG || {}), ...userConfig };
  const useWorldTerrain = Boolean(cfg.cesiumIonToken);
  if (cfg.cesiumIonToken) {
    Cesium.Ion.defaultAccessToken = cfg.cesiumIonToken;
  }

  const viewer = new Cesium.Viewer("cesiumContainer", {
    terrain: useWorldTerrain
      ? Cesium.Terrain.fromWorldTerrain({ requestVertexNormals: true })
      : undefined,
    baseLayer: createSwappableBaseLayer(cfg),
    shadows: false,                    // Avoids heavy shadow maps on GTX 1650 4 GB VRAM
    msaaSamples: 2,                    // Crisp CTR/HUD edges at half 4x MSAA footprint
    requestRenderMode: true,           // Render on 30 Hz WS telemetry tick or camera input
    maximumRenderTimeChange: Infinity,
    creditContainer: "cesium-credits", // Mandatory attribution bar
  });

  // GTX 1650 SUPER (< 1.8 GB VRAM) globe & 3D Tiles locks
  viewer.scene.globe.depthTestAgainstTerrain = true;
  viewer.scene.globe.tileCacheSize = 64;
  viewer.scene.globe.maximumScreenSpaceError = 2.5;

  if (useWorldTerrain) {
    const osmBuildings = await Cesium.createOsmBuildingsAsync();
    osmBuildings.cacheBytes = 256 * 1024 * 1024;
    osmBuildings.maximumCacheOverflowBytes = 128 * 1024 * 1024;
    osmBuildings.maximumScreenSpaceError = 24;
    osmBuildings.dynamicScreenSpaceError = true;
    osmBuildings.dynamicScreenSpaceErrorDensity = 0.00278;
    osmBuildings.dynamicScreenSpaceErrorFactor = 4.0;
    osmBuildings.skipLevelOfDetail = true;
    osmBuildings.foveatedScreenSpaceError = true;
    viewer.scene.primitives.add(osmBuildings);
  }

  await addAirspaceCtrWalls(viewer, cfg.airspaceGeoJsonUrl);

  // Aircraft marker + coaxial X8 glTF (with NodeTransformationProperty rotors when model URI provided)
  let currentPosition = Cesium.Cartesian3.fromDegrees(
    KARTAL_DATUM.lonDeg,
    KARTAL_DATUM.latDeg,
    KARTAL_DATUM.altAmslM + KARTAL_DATUM.geoidNM
  );
  let currentOrientation = Cesium.Quaternion.IDENTITY;
  let currentCbvfPositions = [];
  let currentCbvfColor = Cesium.Color.fromCssColorString("#12b5cb").withAlpha(0.28);
  let cachedDeltaZSurfM = 0.0;
  let lastSurfaceSampleMs = 0;

  const rotorTransformations = {};
  for (let i = 0; i < 8; i += 1) {
    rotorTransformations[`Rotor_${i}`] = new Cesium.NodeTransformationProperty({
      rotation: Cesium.Quaternion.IDENTITY,
    });
  }

  const aircraftEntity = viewer.entities.add({
    name: "VertiRTA-X8-240",
    position: new Cesium.CallbackProperty(() => currentPosition, false),
    orientation: new Cesium.CallbackProperty(() => currentOrientation, false),
    point: cfg.aircraftModelUri
      ? undefined
      : {
          pixelSize: 12,
          color: Cesium.Color.CYAN,
          outlineColor: Cesium.Color.WHITE,
          outlineWidth: 2,
        },
    model: cfg.aircraftModelUri
      ? {
          uri: cfg.aircraftModelUri,
          minimumPixelSize: 48,
          nodeTransformations: rotorTransformations,
        }
      : undefined,
  });

  viewer.entities.add({
    name: "CBVF Safe Landing Footprint",
    polygon: {
      hierarchy: new Cesium.CallbackProperty(
        () => new Cesium.PolygonHierarchy(currentCbvfPositions),
        false
      ),
      material: new Cesium.ColorMaterialProperty(
        new Cesium.CallbackProperty(() => currentCbvfColor, false)
      ),
      classificationType: Cesium.ClassificationType.BOTH,
    },
  });

  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(KARTAL_DATUM.lonDeg - 0.025, KARTAL_DATUM.latDeg - 0.015, 1600.0),
    orientation: {
      heading: Cesium.Math.toRadians(35.0),
      pitch: Cesium.Math.toRadians(-24.0),
      roll: 0.0,
    },
    duration: 0.0,
  });
  viewer.scene.requestRender();

  const ws = new WebSocket(cfg.wsUrl);
  ws.onopen = () => {
    const el = document.getElementById("hud-link");
    if (el) el.textContent = "LIVE :5180";
  };
  ws.onmessage = async (event) => {
    const pkt = JSON.parse(event.data);
    if (pkt.schema_ver !== "vrta.ws.v1.2") return;
    const g = pkt.geodetic;

    // Sample rendered terrain + OSM rooftop height near landing to keep AGL = 0.00 m exact
    const nowMs = performance.now();
    if (g.agl_m <= TOUCHDOWN_BLEND_SIGMA_M * 2.5 && nowMs - lastSurfaceSampleMs > 250) {
      lastSurfaceSampleMs = nowMs;
      try {
        const carto = Cesium.Cartographic.fromDegrees(g.lon_deg, g.lat_deg);
        const sampled = await viewer.scene.sampleHeightMostDetailed([carto]);
        if (sampled && sampled[0] && Number.isFinite(sampled[0].height)) {
          const simSurfaceHae = g.alt_hae_wgs84_m - Math.max(0.0, g.agl_m);
          cachedDeltaZSurfM = sampled[0].height - simSurfaceHae;
        }
      } catch (_err) {
        // Offline grid mode without terrain provider keeps cachedDeltaZSurfM = 0.0
      }
    }

    const renderHaeM = computeTouchdownBlendHae(g.alt_hae_wgs84_m, cachedDeltaZSurfM, g.agl_m);
    currentPosition = Cesium.Cartesian3.fromDegrees(g.lon_deg, g.lat_deg, renderHaeM);

    // Transform ENU/FLU attitude into WGS84 ECEF quaternion
    const enuFrame = Cesium.Transforms.eastNorthUpToFixedFrame(currentPosition);
    const enuRot = Cesium.Matrix4.getMatrix3(enuFrame, new Cesium.Matrix3());
    const [qw, qx, qy, qz] = g.q_enu_flu;
    const qBody = new Cesium.Quaternion(qx, qy, qz, qw);
    const bodyRot = Cesium.Matrix3.fromQuaternion(qBody, new Cesium.Matrix3());
    const ecefRot = Cesium.Matrix3.multiply(enuRot, bodyRot, new Cesium.Matrix3());
    currentOrientation = Cesium.Quaternion.fromRotationMatrix(ecefRot);

    // Update 24-vertex CBVF footprint using strict [lat, lon, hae] ordering
    const ring = pkt.cbvf_overlay?.footprint_lat_lon_hae || [];
    if (ring.length >= 3) {
      currentCbvfPositions = ring.map(([lat, lon, hae]) =>
        Cesium.Cartesian3.fromDegrees(lon, lat, hae)
      );
      currentCbvfColor = pkt.safety_avionics?.cel_latched
        ? Cesium.Color.fromCssColorString("#ff3d00").withAlpha(0.35)
        : pkt.propulsion?.smoke_active
          ? Cesium.Color.fromCssColorString("#ffb300").withAlpha(0.32)
          : Cesium.Color.fromCssColorString("#12b5cb").withAlpha(0.28);
    }

    updateHudDom(pkt);
    // CRITICAL (Review Focus #5): wake CesiumJS requestRenderMode on every WS telemetry frame!
    viewer.scene.requestRender();
  };

  return { viewer, aircraftEntity, ws };
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    if (typeof Cesium !== "undefined") {
      initVertiRtaCesiumConsole();
    }
  });
}
