"""ETW real-time source for process-start events.

Wraps `pywintrace` to subscribe to the ``Microsoft-Windows-Kernel-Process``
provider and push normalized ProcessStart (event id 1) dicts onto a bounded queue.
"""

from __future__ import annotations

import ctypes
import json
import queue
import re
import string
import sys
from functools import lru_cache
from typing import Any, Dict, Optional

from etw import ETW, ProviderInfo
from etw.GUID import GUID

# Microsoft-Windows-Kernel-Process
KERNEL_PROCESS_GUID = "{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}"
PROCESS_START_EVENT_ID = 1
PROCESS_STOP_EVENT_ID = 2


def _as_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(str(value), 0) if isinstance(value, str) else int(value)
    except (ValueError, TypeError):
        return None


def _as_int32(value: Any) -> Optional[int]:
    """Interpret a value as a signed 32-bit integer.

    Exit codes are reported as an unsigned DWORD, so e.g. ``-1`` arrives as
    ``4294967295``. Reinterpreting the low 32 bits as signed yields the
    human-readable, NTSTATUS-style value (``-1``) used in the stream.
    """
    raw = _as_int(value)
    if raw is None:
        return None
    raw &= 0xFFFFFFFF
    return raw - 0x100000000 if raw >= 0x80000000 else raw


def _clean_iso(value: Optional[str]) -> Optional[str]:
    """Normalize the event's CreateTime to millisecond ISO-8601 UTC.

    The provider emits values like ``\u200e2026\u200e-\u200e06...T18:02:06.931455900Z``
    -- littered with Unicode left-to-right marks and carrying nanosecond precision.
    We strip the marks and truncate the fraction to milliseconds.
    """
    if not value:
        return None
    s = value.replace("\u200e", "").replace("\u200f", "")
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z?", s)
    if not m:
        return s
    base, frac = m.group(1), m.group(2)
    ms = (frac or "000")[:3].ljust(3, "0")
    return f"{base}.{ms}Z"


@lru_cache(maxsize=1)
def _device_drive_map() -> tuple:
    """Map NT device prefixes to DOS drives, e.g. ``\\Device\\HarddiskVolume3`` -> ``C:``.

    Cached for the process lifetime: a volume mounted or removed mid-run is not
    reflected, so an image on a newly mounted drive may keep its NT device path.
    Acceptable for the short-lived runs this tool targets.
    """
    mapping = []
    buf = ctypes.create_unicode_buffer(1024)
    for letter in string.ascii_uppercase:
        drive = f"{letter}:"
        if ctypes.windll.kernel32.QueryDosDeviceW(drive, buf, 1024):
            mapping.append((buf.value, drive))
    return tuple(mapping)


def _dos_path(device_path: Optional[str]) -> Optional[str]:
    """Convert an NT device image path to a drive-letter path when possible.

    ETW reports images as ``\\Device\\HarddiskVolumeN\\...``; we rewrite the device
    prefix to its drive letter. Returns the input unchanged if no mapping is found.
    """
    if not device_path:
        return None
    for dev, drive in _device_drive_map():
        if device_path == dev or device_path.startswith(dev + "\\"):
            return drive + device_path[len(dev):]
    return device_path


def normalize(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ProcessStart fields we use from a pywintrace event dict.

    Key names are taken verbatim from the Microsoft-Windows-Kernel-Process manifest
    (ProcessStart, event id 1, version 3). They are fixed by the manifest, not guessed.
    """
    return {
        "kind": "start",
        "pid": _as_int(event.get("ProcessID")),
        "ppid": _as_int(event.get("ParentProcessID")),
        "process_seq": _as_int(event.get("ProcessSequenceNumber")),
        "session_id": _as_int(event.get("SessionID")),
        "image": _dos_path(event.get("ImageName")),
        "create_time": _clean_iso(event.get("CreateTime")),
        "mandatory_label": event.get("MandatoryLabel") or None,
        "token_is_elevated": _as_int(event.get("ProcessTokenIsElevated")),
    }


def normalize_stop(event: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ProcessStop (event id 2) fields we use.

    Field names are taken verbatim from the Microsoft-Windows-Kernel-Process
    manifest (ProcessStop). ``CreateTime`` is carried by the stop event itself,
    so process lifetime can be computed without correlating against the start.
    """
    return {
        "kind": "stop",
        "pid": _as_int(event.get("ProcessID")),
        "process_seq": _as_int(event.get("ProcessSequenceNumber")),
        "exit_code": _as_int32(event.get("ExitCode")),
        "create_time": _clean_iso(event.get("CreateTime")),
        "exit_time": _clean_iso(event.get("ExitTime")),
        "image": _dos_path(event.get("ImageName")),
    }


class EtwProcessSource:
    """Real-time ETW session emitting normalized ProcessStart events onto a queue."""

    def __init__(
        self,
        queue_size: int = 10000,
        session_name: str = "ProcessStreamSession",
        debug_raw: bool = False,
    ) -> None:
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=queue_size)
        self.dropped = 0
        self._debug_raw = debug_raw
        self._session_name = session_name
        provider = ProviderInfo("Microsoft-Windows-Kernel-Process", GUID(KERNEL_PROCESS_GUID))
        self._etw = ETW(
            session_name=session_name,
            providers=[provider],
            event_callback=self._on_event,
            event_id_filters=[PROCESS_START_EVENT_ID, PROCESS_STOP_EVENT_ID],
        )

    def _on_event(self, event_tufo) -> None:
        event_id, event = event_tufo
        if event_id == PROCESS_START_EVENT_ID:
            normalized = normalize(event)
        elif event_id == PROCESS_STOP_EVENT_ID:
            normalized = normalize_stop(event)
        else:
            return
        if self._debug_raw:
            try:
                dump = json.dumps(event, default=str, sort_keys=True)
            except (TypeError, ValueError):
                dump = repr(event)
            print(f"[process-stream] RAW {event_id} {dump}", file=sys.stderr, flush=True)
        try:
            self.events.put_nowait(normalized)
        except queue.Full:
            self.dropped += 1
            if self.dropped % 100 == 1:
                print(
                    f"[process-stream] WARNING: event queue full, dropped {self.dropped} events",
                    file=sys.stderr,
                    flush=True,
                )

    def start(self) -> None:
        self._etw.start()

    def stop(self) -> None:
        try:
            self._etw.stop()
        except Exception as exc:  # noqa: BLE001 - teardown must never raise
            print(f"[process-stream] WARNING: ETW stop failed: {exc}", file=sys.stderr, flush=True)
