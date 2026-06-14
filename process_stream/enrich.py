"""psutil-based enrichment for process records (best-effort)."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Optional, Tuple

import psutil

from . import _debug
from .features import (
    day_of_week,
    hour_of_day,
    image_name,
    is_pseudo_process,
    path_bucket,
)
from .filecache import file_facts
from .normalize import normalize_command_line
from .record import ProcessRecord, epoch_to_iso
from .tokeninfo import token_info


class Enrichment:
    """Best-effort metadata gathered from psutil for a given PID."""

    __slots__ = (
        "command_line",
        "image",
        "user",
        "cwd",
        "create_time",
        "ppid",
        "parent_image",
        "parent_command_line",
        "parent_user",
        "ok",
    )

    def __init__(self) -> None:
        self.command_line: Optional[str] = None
        self.image: Optional[str] = None
        self.user: Optional[str] = None
        self.cwd: Optional[str] = None
        self.create_time: Optional[str] = None
        self.ppid: Optional[int] = None
        self.parent_image: Optional[str] = None
        self.parent_command_line: Optional[str] = None
        self.parent_user: Optional[str] = None
        self.ok: bool = False


_PARENT_CACHE_MAX = 2048
_parent_cache: "OrderedDict[Tuple[int, float], Tuple[Optional[str], Optional[str], Optional[str]]]" = (
    OrderedDict()
)


def _parent_facts(
    parent: "psutil.Process", create_time: float
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Resolve a parent's (exe, cmdline, user), reusing ``parent`` for the lookups.

    Cached by ``(pid, create_time)`` so a reused PID with a different start time
    is not served stale data.
    """
    key = (parent.pid, create_time)
    cached = _parent_cache.get(key)
    if cached is not None:
        _parent_cache.move_to_end(key)
        return cached

    exe: Optional[str] = None
    cmdline: Optional[str] = None
    user: Optional[str] = None
    with parent.oneshot():
        try:
            exe = parent.exe() or None
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            pass
        try:
            cmd = parent.cmdline()
            cmdline = " ".join(cmd) if cmd else None
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            pass
        try:
            user = parent.username() or None
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            pass

    facts = (exe, cmdline, user)
    _parent_cache[key] = facts
    if len(_parent_cache) > _PARENT_CACHE_MAX:
        _parent_cache.popitem(last=False)
    return facts


def enrich(pid: int) -> Enrichment:
    """Gather available metadata for ``pid``. Never raises; partial data on failure."""
    result = Enrichment()
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            try:
                cmd = proc.cmdline()
                result.command_line = " ".join(cmd) if cmd else None
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                result.image = proc.exe() or None
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                result.user = proc.username() or None
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                result.cwd = proc.cwd() or None
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                result.create_time = epoch_to_iso(proc.create_time())
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
            try:
                result.ppid = proc.ppid()
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                pass
        result.ok = True
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError) as exc:
        _debug.log("enrich", exc)
        return result

    if result.ppid is not None:
        try:
            parent = psutil.Process(result.ppid)
            exe, cmdline, user = _parent_facts(parent, parent.create_time())
            result.parent_image = exe
            result.parent_command_line = cmdline
            result.parent_user = user
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError) as exc:
            _debug.log("enrich.parent", exc)

    return result


def finalize(record: ProcessRecord) -> None:
    """Populate file-identity, token, normalization, and derived fields in place.

    Shared by the backfill and live paths so a given process serializes identically
    regardless of source. Best-effort: every lookup degrades to ``None`` on failure.
    """
    record.is_pseudo = is_pseudo_process(record.pid, record.image, record.create_time)

    if record.command_line:
        record.command_line_normalized = normalize_command_line(record.command_line)

    if record.image:
        facts = file_facts(record.image)
        record.image_hash = facts.image_hash
        record.image_hash_truncated = facts.image_hash_truncated
        record.image_size = facts.image_size
        record.is_signed = facts.is_signed
        record.signature_status = facts.signature_status
        record.signer = facts.signer
        record.signer_is_microsoft = facts.signer_is_microsoft
        record.original_file_name = facts.original_file_name
        record.company_name = facts.company_name
        record.product_name = facts.product_name
        record.file_description = facts.file_description
        record.file_version = facts.file_version
        record.name_mismatch = _name_mismatch(record.image, facts.original_file_name)

    token = token_info(record.pid)
    record.user_sid = token.user_sid
    record.logon_type = token.logon_type
    # ETW already provides these for live records; fill the backfill gap only.
    if record.integrity_level is None:
        record.integrity_level = token.integrity_level
    if record.is_elevated is None:
        record.is_elevated = token.is_elevated

    record.path_bucket = path_bucket(record.image)
    record.image_name = image_name(record.image)
    record.hour_of_day = hour_of_day(record.create_time or record.timestamp)
    record.day_of_week = day_of_week(record.create_time or record.timestamp)


def _name_mismatch(image: Optional[str], original_file_name: Optional[str]) -> Optional[bool]:
    """True when the on-disk basename differs from the PE OriginalFilename."""
    if not image or not original_file_name:
        return None

    def stem(name: str) -> str:
        base = os.path.basename(name).lower()
        return base[:-4] if base.endswith(".exe") else base

    return stem(image) != stem(original_file_name)

