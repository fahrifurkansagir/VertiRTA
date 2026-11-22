# VertiRTA Hardened Docker Deployment (`:5180` / `:8891`)

Multi-stage, non-root, read-only container stack serving:
- **`vertirta-sim` (`127.0.0.1:8891`):** Stripped C++20 `vertirta_replay_cli` / `vertirta_6dof_batch_cli` + Python `vrta.ws.v1.2` WebSocket bridge and `/healthz` probe running as `USER 10001:10001`.
- **`vertigcs-web` (`127.0.0.1:5180`):** Unprivileged Nginx (`USER 101`) serving the static `VertiGCS` MapLibre GL JS v6 + `@deck.gl/maplibre` v9.4 bundle and single-threaded `<150 KiB` Emscripten WASM Worker (with `gzip off` on `.pmtiles` HTTP 206 byte ranges).

## Quick Start

```bash
docker compose -f docker/docker-compose.yml up --build
```

Open `http://127.0.0.1:5180/?mode=live&ws=ws://127.0.0.1:8891/telemetry` (or omit query parameters for zero-backend Sandbox WASM mode).

## Security & Port Isolation Locks
- Both runtime containers run with `read_only: true`, `cap_drop: [ALL]`, and `no-new-privileges:true`.
- Dedicated ports `:5180` (HTTP) and `:8891` (WS); optional local PX4 SITL MicroXRCEAgent uses `:8888/udp`.
- Forbidden workspace ports (`4173`, `4174`, `8765`, `9222`, `9223`) are strictly blocked.
