"""Contract & live /healthz verification tests for hardened Docker stack (Plan 4B Task 5)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
FORBIDDEN_PORTS = {"4173", "4174", "8765", "9222", "9223"}


def test_docker_compose_enforces_nonroot_readonly_healthchecks_and_port_isolation() -> None:
    compose_path = REPO / "docker/docker-compose.yml"
    raw = compose_path.read_bytes()
    assert b"\r" not in raw, "docker-compose.yml must use pure LF"
    doc = yaml.safe_load(raw.decode("utf-8"))
    services = doc["services"]
    assert {"vertirta-sim", "vertigcs-web"}.issubset(services.keys())

    for name, svc in services.items():
        assert svc.get("read_only") is True, f"{name} must set read_only: true"
        assert "ALL" in svc.get("cap_drop", []), f"{name} must drop ALL capabilities"
        assert "no-new-privileges:true" in svc.get("security_opt", []), (
            f"{name} must set security_opt no-new-privileges:true"
        )
        assert svc.get("healthcheck"), f"{name} must define a healthcheck"
        for p in svc.get("ports", []):
            p_str = str(p)
            for bad in FORBIDDEN_PORTS:
                assert f"{bad}:" not in p_str and f":{bad}" not in p_str, (
                    f"Forbidden port {bad} found in {name}: {p_str}"
                )

    assert services["vertigcs-web"]["depends_on"]["vertirta-sim"]["condition"] == "service_healthy"


def test_dockerfiles_are_multistage_nonroot_copy_full_viz_and_expose_healthz() -> None:
    for df in ("docker/Dockerfile.sim", "docker/Dockerfile.vertigcs"):
        raw = (REPO / df).read_bytes()
        assert b"\r" not in raw, f"{df} must use pure LF"
        text = raw.decode("utf-8")
        assert text.count("FROM ") >= 2, f"{df} must be multi-stage"
        assert "USER 10001:10001" in text or "USER 101" in text, (
            f"{df} must switch to explicit non-root USER"
        )

    sim_df = (REPO / "docker/Dockerfile.sim").read_text(encoding="utf-8")
    for req in ("viz/geo", "viz/cesium", "viz/__init__.py", "config/istanbul_ltfj_airspace.yaml"):
        assert req in sim_df, f"Dockerfile.sim missing runtime import dependency: {req}"

    nginx_conf = (REPO / "docker/nginx.vertigcs.conf").read_bytes()
    assert b"\r" not in nginx_conf
    nginx_txt = nginx_conf.decode("utf-8")
    assert "listen 5180" in nginx_txt
    assert "/healthz" in nginx_txt
    assert "gzip off" in nginx_txt  # Prevent Fastly/Nginx gzip corruption on .pmtiles HTTP 206 Range

    ws_src = (REPO / "viz/cesium/ws_server.py").read_text(encoding="utf-8")
    assert '"/healthz"' in ws_src, "ws_server.py must serve 200 OK on /healthz for Docker healthcheck"


def test_ws_server_live_healthz_returns_200_json() -> None:
    from viz.cesium.ws_server import _handle_client

    async def _probe() -> tuple[str, dict]:
        static_dir = REPO / "viz/cesium"
        server = await asyncio.start_server(
            lambda r, w: _handle_client(r, w, static_dir=static_dir, vrta_path=None, pose_hz=30.0),
            "127.0.0.1",
            0,
        )
        port = server.sockets[0].getsockname()[1]
        async with server:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            await writer.drain()
            raw = await reader.read()
            writer.close()
            await writer.wait_closed()
        head, _, body = raw.partition(b"\r\n\r\n")
        return head.decode("ascii"), json.loads(body.decode("utf-8"))

    status_head, payload = asyncio.run(_probe())
    assert "200 OK" in status_head
    assert payload.get("status") == "ok"
