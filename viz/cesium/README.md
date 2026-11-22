# VertiRTA-X8-240 Live Tactical Earth Console (`viz/cesium/`, Port `:5180`)

Layer-1 **live streaming only** CesiumJS tactical console for real-time and `.vrta` flight-log replay across the Maltepe–Dragos–Kartal coastal corridor bounded by DHMİ `LTFJ` (`798.58 m HAE`) and `LTBX` (`646.35 m HAE`) CTR walls.

## Hardware & Legal Locks

- **Dedicated Port:** `http://127.0.0.1:5180` (`ws://127.0.0.1:5180/telemetry`). Forbidden ports (`4173, 4174, 8765, 9222, 9223`) are hard-rejected by `ws_server.py`.
- **GTX 1650 SUPER (`4 GB VRAM`, `< 1.8 GB` Target):** `shadows: false`, `msaaSamples: 2`, `requestRenderMode: true` with explicit `viewer.scene.requestRender()` on every WebSocket frame, `globe.tileCacheSize = 64`, `osmBuildings.cacheBytes = 256 MB`, `maximumCacheOverflowBytes = 128 MB`, and `maximumScreenSpaceError = 24`.
- **Swappable Basemap & ToS Firewall:** No commercial tile endpoint or API key is hardcoded. Configure `window.VERTIRTA_CONFIG` at runtime with your own `basemapProvider` (`"ion_bing"`, `"esri_key"`, `"azure_maps"`, or zero-token `"open_grid_offline"`). Commercial streaming tiles are strictly confined to this browser client and are never cached or exported to Blender.
- **Zero-Gap Touchdown Alignment:** `ws_server.py` streams `vrta.ws.v1.2` packets with strict `footprint_lat_lon_hae` (`[lat, lon, hae]`) ordering and `app.js` applies `scene.sampleHeightMostDetailed` Gaussian blending (`sigma = 15.0 m`) so `0.00 m AGL` contacts the rendered rooftop without distorting cruise HAE against `LTFJ`/`LTBX` walls.

## Running Locally

```bash
python -m viz.cesium.ws_server --host 127.0.0.1 --port 5180 --vrta path/to/flight.vrta
```

Open `http://127.0.0.1:5180/` in Chrome/Edge.
