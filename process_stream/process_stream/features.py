"""Derived convenience features computed from raw record fields (pure functions)."""

from __future__ import annotations

import os
from typing import Optional

from .record import parse_iso

# Pseudo / non-launch image basenames (lower-cased, as reported by ETW / psutil).
_PSEUDO_NAMES = {
    "registry",
    "memory compression",
    "memorycompression",
    "system",
    "secure system",
    "system idle process",
}


def image_name(image: Optional[str]) -> Optional[str]:
    """Basename of the image path, lower-cased."""
    if not image:
        return None
    return os.path.basename(image).lower() or None


def _norm(path: str) -> str:
    return path.replace("/", "\\").lower()


def path_bucket(image: Optional[str]) -> Optional[str]:
    """Classify the image path into a coarse, model-friendly bucket."""
    if not image:
        return None
    p = _norm(image)
    if "\\system32\\" in p or p.endswith("\\system32"):
        return "System32"
    if "\\syswow64\\" in p or p.endswith("\\syswow64"):
        return "SysWOW64"
    if "\\appdata\\local\\temp\\" in p or "\\windows\\temp\\" in p or "\\temp\\" in p:
        return "Temp"
    if "\\downloads\\" in p:
        return "Downloads"
    if "\\appdata\\" in p:
        return "AppData"
    if "\\program files (x86)\\" in p or "\\program files\\" in p:
        return "ProgramFiles"
    if "\\users\\" in p:
        return "User"
    return "Other"


def is_user_writable_path(image: Optional[str]) -> Optional[bool]:
    """Whether the image lives in a typically user-writable location."""
    bucket = path_bucket(image)
    if bucket is None:
        return None
    return bucket in ("Temp", "Downloads", "AppData", "User")


def is_pseudo_process(
    pid: Optional[int], image: Optional[str], create_time: Optional[str]
) -> bool:
    """Whether this record is a non-launch pseudo-process rather than a real start."""
    if pid in (0, 4):
        return True
    name = image_name(image)
    if name:
        if name in _PSEUDO_NAMES:
            return True
        stem = name[:-4] if name.endswith(".exe") else name
        if stem in _PSEUDO_NAMES:
            return True
    if create_time and create_time.startswith("1970"):
        return True
    return False


def hour_of_day(timestamp: Optional[str]) -> Optional[int]:
    """UTC hour (0-23) parsed from an ISO-8601 timestamp."""
    dt = parse_iso(timestamp)
    if dt is None or dt.year <= 1970:
        # 1970 is the create-time sentinel for kernel pseudo-processes (pid 0/4);
        # deriving an hour from it would be meaningless noise.
        return None
    return dt.hour


def day_of_week(timestamp: Optional[str]) -> Optional[int]:
    """UTC day of week (Monday=0 .. Sunday=6) parsed from an ISO-8601 timestamp."""
    dt = parse_iso(timestamp)
    if dt is None or dt.year <= 1970:
        return None
    return dt.weekday()
