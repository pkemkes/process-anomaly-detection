"""Snapshot of currently-running processes (backfill at startup)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Iterator, Optional

import psutil

from .enrich import enrich, finalize
from .record import ProcessRecord, utc_now_iso

_kernel32 = ctypes.windll.kernel32
_kernel32.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
_kernel32.ProcessIdToSessionId.restype = wintypes.BOOL


def _session_id(pid: int) -> Optional[int]:
    """Return the Windows terminal session ID for ``pid``, or ``None`` if unavailable."""
    session = wintypes.DWORD()
    if _kernel32.ProcessIdToSessionId(wintypes.DWORD(pid), ctypes.byref(session)):
        return session.value
    return None


def snapshot() -> Iterator[ProcessRecord]:
    """Yield a ProcessRecord for every process currently running.

    Each record is marked ``source="existing"``. Processes that vanish mid-scan
    are skipped silently. ``process_seq`` is an ETW-only counter and stays
    ``None`` here; integrity level and elevation are filled from the process
    token via ``finalize`` for parity with live records.
    """
    for proc in psutil.process_iter(["pid"]):
        try:
            pid = proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
            continue

        info = enrich(pid)

        record = ProcessRecord(
            pid=pid,
            source="existing",
            timestamp=utc_now_iso(),
            ppid=info.ppid,
            image=info.image,
            command_line=info.command_line,
            user=info.user,
            cwd=info.cwd,
            session_id=_session_id(pid),
            parent_image=info.parent_image,
            parent_command_line=info.parent_command_line,
            parent_user=info.parent_user,
            create_time=info.create_time,
            enriched=info.ok,
        )
        finalize(record)
        yield record
