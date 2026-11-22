"""VertiRTA Python SDK — Deterministic Coaxial-X8 eVTOL RTA & Gymnasium Interface."""

from __future__ import annotations

from ._version import __version__

try:
    from . import _vertirta_native as _native

    HAS_NATIVE_BACKEND: bool = True
except ImportError:
    _native = None
    HAS_NATIVE_BACKEND = False


def get_backend_name() -> str:
    """Return active physics backend: 'cpp_nanobind' or 'python_fallback'."""
    return "cpp_nanobind" if HAS_NATIVE_BACKEND else "python_fallback"


from .env import VertiRtaCoaxialX8Env, register_vertirta_envs  # noqa: E402
from .vec_env import VertiVecEnv  # noqa: E402

register_vertirta_envs()

__all__ = [
    "HAS_NATIVE_BACKEND",
    "VertiRtaCoaxialX8Env",
    "VertiVecEnv",
    "__version__",
    "get_backend_name",
    "register_vertirta_envs",
]
