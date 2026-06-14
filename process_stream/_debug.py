"""Opt-in debug logging for best-effort errors that are otherwise swallowed.

Most Win32 / psutil lookups in this package degrade to ``None`` on failure so the
stream never crashes. That hides genuine programming errors (e.g. a wrong ctypes
signature) behind the same path as ordinary access-denied conditions. Enable this
(``--debug``) to surface those exceptions on stderr without changing behavior.
"""

from __future__ import annotations

import sys

_enabled = False


def set_enabled(value: bool) -> None:
    global _enabled
    _enabled = bool(value)


def is_enabled() -> bool:
    return _enabled


def log(where: str, exc: BaseException) -> None:
    """Print a swallowed exception to stderr when debug logging is enabled."""
    if _enabled:
        print(
            f"[process-stream] DEBUG {where}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
