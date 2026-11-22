"""VertiRTA Live Tactical Earth Console (:5180 CesiumJS + .vrta binary reader/WS bridge)."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "VRTA_FRAME_FMT": ("viz.cesium.vrta_reader", "VRTA_FRAME_FMT"),
    "VRTA_FRAME_SIZE": ("viz.cesium.vrta_reader", "VRTA_FRAME_SIZE"),
    "VRTA_HEADER_FMT": ("viz.cesium.vrta_reader", "VRTA_HEADER_FMT"),
    "VRTA_HEADER_SIZE": ("viz.cesium.vrta_reader", "VRTA_HEADER_SIZE"),
    "VRTA_MAGIC": ("viz.cesium.vrta_reader", "VRTA_MAGIC"),
    "VrtaFileHeader": ("viz.cesium.vrta_reader", "VrtaFileHeader"),
    "VrtaFrame100Hz": ("viz.cesium.vrta_reader", "VrtaFrame100Hz"),
    "crc32c": ("viz.cesium.vrta_reader", "crc32c"),
    "iter_vrta_frames": ("viz.cesium.vrta_reader", "iter_vrta_frames"),
    "pack_vrta_frame": ("viz.cesium.vrta_reader", "pack_vrta_frame"),
    "pack_vrta_header": ("viz.cesium.vrta_reader", "pack_vrta_header"),
    "read_vrta_file": ("viz.cesium.vrta_reader", "read_vrta_file"),
    "unpack_vrta_frame": ("viz.cesium.vrta_reader", "unpack_vrta_frame"),
    "unpack_vrta_header": ("viz.cesium.vrta_reader", "unpack_vrta_header"),
    "write_vrta_file": ("viz.cesium.vrta_reader", "write_vrta_file"),
    "DEFAULT_CESIUM_PORT": ("viz.cesium.ws_server", "DEFAULT_CESIUM_PORT"),
    "FORBIDDEN_PORTS": ("viz.cesium.ws_server", "FORBIDDEN_PORTS"),
    "WS_SCHEMA_VER": ("viz.cesium.ws_server", "WS_SCHEMA_VER"),
    "compute_touchdown_blend_hae": ("viz.cesium.ws_server", "compute_touchdown_blend_hae"),
    "encode_ws_telemetry_frame": ("viz.cesium.ws_server", "encode_ws_telemetry_frame"),
    "validate_port": ("viz.cesium.ws_server", "validate_port"),
    "vrta_frame_to_ws_packet": ("viz.cesium.ws_server", "vrta_frame_to_ws_packet"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr_name = target
    mod = importlib.import_module(mod_name)
    val = getattr(mod, attr_name)
    globals()[name] = val
    return val
