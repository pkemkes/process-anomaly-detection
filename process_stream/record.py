"""Process record schema and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

# Stream schema version. Bump (semver) whenever fields are added or changed so
# downstream training pipelines can bucket / migrate datasets.
SCHEMA_VERSION = "1.0.0"

# Windows integrity-level SIDs (RID -> human label).
# See https://learn.microsoft.com/windows/win32/secauthz/well-known-sids
_INTEGRITY_LEVELS = {
    0x0000: "Untrusted",
    0x1000: "Low",
    0x2000: "Medium",
    0x2100: "Medium Plus",
    0x3000: "High",
    0x4000: "System",
    0x5000: "Protected Process",
}


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with millisecond precision."""
    return _iso(datetime.now(timezone.utc))


def epoch_to_iso(epoch_seconds: float) -> str:
    """Convert a POSIX timestamp (seconds, UTC) to an ISO-8601 string."""
    return _iso(datetime.fromtimestamp(epoch_seconds, tz=timezone.utc))


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 UTC string produced by this module back to a datetime."""
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
    return None


def integrity_from_sid(sid) -> Optional[str]:
    """Map a mandatory-label SID (e.g. ``S-1-16-8192``) to a readable level.

    Accepts either a SID string or a bare integer RID (some provider/manifest
    builds report the mandatory label as an integer rather than a SID string).
    """
    if sid is None or sid == "":
        return None
    if isinstance(sid, int):
        rid = sid
    else:
        try:
            rid = int(str(sid).rsplit("-", 1)[-1])
        except (ValueError, AttributeError):
            return None
    return _INTEGRITY_LEVELS.get(rid, f"Unknown ({rid})")


@dataclass
class ProcessRecord:
    """A single process-start observation in the output stream.

    Fields:
        pid: Process ID of the started process.
        source: Origin of the record -- ``"existing"`` for the startup backfill
            snapshot of already-running processes, ``"new"`` for a live ETW
            process-start event.
        timestamp: When this record was observed/emitted, ISO-8601 UTC with
            millisecond precision. This is the stream-emission time and is
            independent of the process create time (see ``create_time``): for a
            live start it is roughly the create time, but for a backfill snapshot
            it is the scan time of an already-running process, and for a
            ``process_stop`` record it is the time the exit was observed.
        event: Record type discriminator. Always ``"process_start"`` for now;
            present so consumers can branch if more event types are added later.
        ppid: Parent process ID. May be ``None`` if it could not be determined.
        process_seq: ETW ProcessSequenceNumber -- a monotonic identifier unique
            for the current boot session, used to disambiguate reused PIDs.
            Only populated for live ETW events (``None`` for backfill).
        image: Full path to the executable image (e.g.
            ``C:\\Windows\\System32\\cmd.exe``).
        command_line: Full command line the process was launched with. ``None``
            if the process exited before it could be read, or access was denied.
        user: Owning user as ``DOMAIN\\User`` (e.g. ``MACHINE\\pkemkes``).
            ``None`` if it could not be resolved.
        cwd: Current working directory of the process at observation time.
            ``None`` if unavailable.
        session_id: Windows terminal session ID (0 = services/system session,
            1+ = interactive logons). Only populated for live ETW events.
        integrity_level: Process integrity level derived from the mandatory-label
            SID -- e.g. ``"Low"``, ``"Medium"``, ``"High"``, ``"System"``.
            From ETW for live events; from the process token for backfill.
        is_elevated: Whether the process token is elevated (UAC). ``True``/``False``
            when known (ETW for live, token query for backfill), ``None`` if the
            process was inaccessible.
        parent_image: Full path to the parent process's executable, best-effort.
            ``None`` if the parent already exited or was inaccessible.
        create_time: Process creation time, ISO-8601 UTC. ``None`` if it could
            not be read.
        enriched: ``True`` if psutil enrichment succeeded for this process;
            ``False`` if the process vanished or access was denied, in which case
            the human-meaningful fields (command_line, user, cwd, ...) may be
            ``None`` while the ETW core metadata is still present.
    """

    pid: int
    source: str  # "existing" (backfill) or "new" (live ETW)
    timestamp: str
    event: str = "process_start"  # or "process_stop"
    schema_version: str = SCHEMA_VERSION
    is_pseudo: bool = False
    ppid: Optional[int] = None
    process_seq: Optional[int] = None
    image: Optional[str] = None
    command_line: Optional[str] = None
    command_line_normalized: Optional[str] = None
    user: Optional[str] = None
    cwd: Optional[str] = None
    session_id: Optional[int] = None
    integrity_level: Optional[str] = None
    is_elevated: Optional[bool] = None
    parent_image: Optional[str] = None
    parent_command_line: Optional[str] = None
    parent_user: Optional[str] = None
    create_time: Optional[str] = None
    enriched: bool = False
    # --- file identity (filecache) ---
    image_hash: Optional[str] = None  # sha256 hex, lowercase
    image_hash_truncated: Optional[bool] = None
    image_size: Optional[int] = None  # bytes
    # --- code signing ---
    is_signed: Optional[bool] = None
    signature_status: Optional[str] = None  # trusted|untrusted|expired|unsigned|error
    signer: Optional[str] = None
    signer_is_microsoft: Optional[bool] = None
    # --- PE / file metadata ---
    original_file_name: Optional[str] = None
    company_name: Optional[str] = None
    product_name: Optional[str] = None
    file_description: Optional[str] = None
    file_version: Optional[str] = None
    name_mismatch: Optional[bool] = None
    # --- account identity ---
    user_sid: Optional[str] = None
    logon_type: Optional[str] = None
    # --- lifetime (process_stop only) ---
    exit_code: Optional[int] = None
    lifetime_ms: Optional[int] = None
    # --- derived convenience features ---
    path_bucket: Optional[str] = None
    image_name: Optional[str] = None
    hour_of_day: Optional[int] = None
    day_of_week: Optional[int] = None

    def as_json(self, pretty: bool = False) -> str:
        data = asdict(self)
        if pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data, separators=(",", ":"))
