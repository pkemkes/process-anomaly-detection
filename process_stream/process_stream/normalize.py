"""Command-line normalization so identical processes serialize identically.

Backfill yields unexpanded paths like ``%SystemRoot%\\system32\\csrss.exe`` while
live enrichment yields expanded, space-joined argv. ``normalize_command_line``
produces a single source-independent form: environment variables expanded, the
executable path portion lower-cased, and whitespace collapsed.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

_ENV_VAR = re.compile(r"%([^%]+)%")
_WS = re.compile(r"\s+")
# An unquoted leading executable path ending in a known image extension.
_EXE_HEAD = re.compile(r"^(.*?\.(?:exe|com|bat|cmd|scr|pif))(?:\s+(.*))?$", re.IGNORECASE)


def _expand_env(text: str) -> str:
    """Expand ``%VAR%`` plus the NT-style ``\\SystemRoot\\`` and ``\\??\\`` prefixes."""

    def repl(match: "re.Match[str]") -> str:
        name = match.group(1)
        value = os.environ.get(name)
        if value is None:
            # Case-insensitive lookup (Windows env is case-insensitive).
            lname = name.lower()
            for k, v in os.environ.items():
                if k.lower() == lname:
                    value = v
                    break
        return value if value is not None else match.group(0)

    text = _ENV_VAR.sub(repl, text)

    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    if text[:1] == "\\":
        lower = text.lower()
        if lower.startswith("\\systemroot\\"):
            text = system_root + "\\" + text[len("\\SystemRoot\\"):]
        elif lower.startswith("\\??\\"):
            text = text[len("\\??\\"):]
    return text


def _split_exe(text: str) -> Tuple[str, str]:
    """Split a command line into (executable, remainder)."""
    text = text.strip()
    if not text:
        return text, ""
    if text[0] == '"':
        end = text.find('"', 1)
        if end != -1:
            return text[1:end], text[end + 1:].strip()
        return text, ""
    # Prefer splitting at a known executable extension so unquoted paths that
    # contain spaces (e.g. "C:\\Program Files\\...") stay intact.
    match = _EXE_HEAD.match(text)
    if match:
        return match.group(1), (match.group(2) or "").strip()
    space = text.find(" ")
    if space == -1:
        return text, ""
    return text[:space], text[space + 1:].strip()


def normalize_command_line(raw: Optional[str]) -> Optional[str]:
    """Return a source-independent normalized command line, or ``None``."""
    if not raw:
        return None
    collapsed = _WS.sub(" ", raw.strip())
    if not collapsed:
        return None
    expanded = _expand_env(collapsed)
    exe, rest = _split_exe(expanded)
    exe = exe.lower()
    return f"{exe} {rest}".rstrip() if rest else exe
