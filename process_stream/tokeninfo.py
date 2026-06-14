"""Per-process token facts via pywin32 (integrity, elevation, SID, logon type).

Used to give backfill records the same security fields as live ETW records and
to add account-identity features. All lookups are best-effort: on any failure the
corresponding field is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import win32api
import win32con
import win32security

from . import _debug
from .record import integrity_from_sid

# PROCESS_QUERY_LIMITED_INFORMATION access right (winnt.h), sourced from pywin32.
_PROCESS_QUERY_LIMITED_INFORMATION = win32con.PROCESS_QUERY_LIMITED_INFORMATION

# TOKEN_INFORMATION_CLASS values (stable across pywin32 versions; resolve with
# fallback in case a given build omits the symbol).
_TokenUser = getattr(win32security, "TokenUser", 1)
_TokenElevation = getattr(win32security, "TokenElevation", 20)
_TokenIntegrityLevel = getattr(win32security, "TokenIntegrityLevel", 25)
_TokenStatistics = getattr(win32security, "TokenStatistics", 10)

_LOGON_TYPES = {
    0: "System",
    2: "Interactive",
    3: "Network",
    4: "Batch",
    5: "Service",
    7: "Unlock",
    8: "NetworkCleartext",
    9: "NewCredentials",
    10: "RemoteInteractive",
    11: "CachedInteractive",
}


@dataclass
class TokenInfo:
    integrity_level: Optional[str] = None
    is_elevated: Optional[bool] = None
    user_sid: Optional[str] = None
    logon_type: Optional[str] = None


def _logon_type_for_luid(luid) -> Optional[str]:
    try:
        data = win32security.LsaGetLogonSessionData(luid)
    except Exception:  # noqa: BLE001 - best-effort
        return None
    if not data:
        return None
    code = data.get("LogonType")
    if code is None:
        return None
    return _LOGON_TYPES.get(int(code), f"Unknown ({int(code)})")


def token_info(pid: int) -> TokenInfo:
    """Gather token-derived facts for ``pid``. Never raises."""
    info = TokenInfo()
    if pid is None or pid <= 0:
        return info

    handle = None
    token = None
    try:
        handle = win32api.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        token = win32security.OpenProcessToken(handle, win32security.TOKEN_QUERY)
    except Exception as exc:  # noqa: BLE001 - access denied / gone
        _debug.log("token_info.open", exc)
        if handle is not None:
            try:
                handle.Close()
            except Exception:  # noqa: BLE001
                pass
        return info

    try:
        try:
            sid, _attrs = win32security.GetTokenInformation(token, _TokenIntegrityLevel)
            info.integrity_level = integrity_from_sid(win32security.ConvertSidToStringSid(sid))
        except Exception:  # noqa: BLE001
            pass

        try:
            elevation = win32security.GetTokenInformation(token, _TokenElevation)
            info.is_elevated = bool(elevation)
        except Exception:  # noqa: BLE001
            pass

        try:
            user = win32security.GetTokenInformation(token, _TokenUser)
            sid = user[0] if isinstance(user, (tuple, list)) else user
            info.user_sid = win32security.ConvertSidToStringSid(sid)
        except Exception:  # noqa: BLE001
            pass

        try:
            stats = win32security.GetTokenInformation(token, _TokenStatistics)
            # TOKEN_STATISTICS is returned as a dict; AuthenticationId is the
            # logon session LUID.
            auth_luid = stats.get("AuthenticationId") if isinstance(stats, dict) else None
            if auth_luid is not None:
                info.logon_type = _logon_type_for_luid(auth_luid)
        except Exception:  # noqa: BLE001
            pass
    finally:
        # PyHANDLE.Close() detaches the wrapper so it is not closed again on GC,
        # avoiding a double-close of a possibly reused handle value.
        for h in (token, handle):
            if h is not None:
                try:
                    h.Close()
                except Exception:  # noqa: BLE001
                    pass
    return info
