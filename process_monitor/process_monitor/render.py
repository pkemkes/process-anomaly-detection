"""Pure formatting of a monitor frame into a list of ANSI-styled lines.

Everything here is a side-effect-free function of the data plus the current
terminal size, which keeps the rendering trivially testable and lets the
monitor loop simply hand the result to :meth:`Screen.render`.

Layout (one line per process, most suspicious at the top)::

    PROCESS ANOMALY MONITOR        123 procs  high 4  medium 9  live  Ctrl+C quit
      SCORE    PID  IMAGE                    USER               DETAIL
      12.84   8421  powershell.exe           CORP\\alice         pair(parent=...)
       9.10   2290  rundll32.exe             CORP\\alice         freq_image_name=...
       ...
     showing 18 of 123   most suspicious at top

The number of process rows grows with the terminal height; columns are clipped
to the terminal width so a line never wraps.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

from .store import Process, SORT_TIME

# --- SGR colour codes --------------------------------------------------------
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"
_REVERSE = "\x1b[7m"
_RED = "\x1b[91m"
_YELLOW = "\x1b[93m"
_GREEN = "\x1b[92m"

_RANK_COLOR = {"high": _RED, "medium": _YELLOW, "low": _GREEN}

# SCORE and PID stay fixed; IMAGE, USER and DETAIL are *elastic* and grow with
# the terminal width (DETAIL, the final column, absorbs whatever is left over).
_SCORE_W = 7
_PID_W = 6
_SEPARATORS = 5  # total spaces between the five columns in ``_row_text``
_FIXED_W = _SCORE_W + _PID_W + _SEPARATORS

# Lower/upper bounds for the elastic columns so they neither collapse nor grow
# wider than their content ever needs.
_IMAGE_MIN, _IMAGE_MAX = 16, 44
_USER_MIN, _USER_MAX = 12, 32
_DETAIL_MIN = 12


def _columns(width: int) -> Tuple[int, int]:
    """Return ``(image_w, user_w)`` for ``width``; both grow with the terminal.

    The space left after the fixed SCORE/PID columns and separators is shared:
    IMAGE and USER each take a bounded proportion and the DETAIL column (drawn
    last) consumes the remainder, so a wider terminal widens every column.
    """
    elastic = max(width - _FIXED_W, _IMAGE_MIN + _USER_MIN + _DETAIL_MIN)
    image_w = max(_IMAGE_MIN, min(_IMAGE_MAX, int(elastic * 0.34)))
    user_w = max(_USER_MIN, min(_USER_MAX, int(elastic * 0.26)))
    return image_w, user_w


def _truncate(text: str, width: int) -> str:
    """Clip ``text`` to ``width`` columns, marking elision with an ellipsis."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "\u2026"
    return text[: width - 1] + "\u2026"


def _pad(text: str, width: int) -> str:
    """Clip or right-pad ``text`` to exactly ``width`` columns."""
    if len(text) >= width:
        return text[:width]
    return text + " " * (width - len(text))


def _bar(text: str, width: int) -> str:
    """A full-width reverse-video status bar."""
    return f"{_REVERSE}{_BOLD}{_pad(_truncate(text, width), width)}{_RESET}"


def _fmt_elapsed(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def _row_text(score: str, pid: str, image: str, user: str, detail: str) -> str:
    """Assemble the column row body (uncoloured, untruncated to width)."""
    return f"{score} {pid}  {image} {user} {detail}"


def _format_row(proc: Process, width: int) -> str:
    image_w, user_w = _columns(width)
    score = f"{proc.score:{_SCORE_W}.2f}"
    pid = f"{proc.pid:>{_PID_W}}"
    image = _pad(_truncate(proc.image_name, image_w), image_w)
    user = _pad(_truncate(proc.user, user_w), user_w)
    detail = proc.top_fields[0] if proc.top_fields else ""
    text = _truncate(_row_text(score, pid, image, user, detail), width)
    color = _RANK_COLOR.get(proc.rank_hint, _GREEN)
    return f"{color}{text}{_RESET}"


def _header_row(width: int) -> str:
    image_w, user_w = _columns(width)
    score = f"{'SCORE':>{_SCORE_W}}"
    pid = f"{'PID':>{_PID_W}}"
    image = _pad("IMAGE", image_w)
    user = _pad("USER", user_w)
    text = _truncate(_row_text(score, pid, image, user, "DETAIL"), width)
    return f"{_DIM}{_BOLD}{text}{_RESET}"


def render_frame(
    processes: Sequence[Process],
    counts: Tuple[int, int, int],
    width: int,
    height: int,
    eof: bool,
    started: float,
    sort_mode: str = "score",
) -> List[str]:
    """Build exactly ``height`` lines describing the current monitor state."""
    width = max(width, 20)
    height = max(height, 5)
    high, medium, total = counts

    # --- title bar ---
    title = " PROCESS ANOMALY MONITOR"
    status = "stream ended" if eof else "live"
    elapsed = _fmt_elapsed(int(time.monotonic() - started))
    stats = (
        f"{total} procs   high {high}  medium {medium}   "
        f"{status}   {elapsed}   Ctrl+C quit "
    )
    bar_text = title + stats.rjust(max(0, width - len(title)))
    lines: List[str] = [_bar(bar_text, width), _header_row(width)]

    # --- process rows (count scales with terminal height) ---
    body_rows = max(height - 3, 1)
    shown = processes[:body_rows]
    lines.extend(_format_row(proc, width) for proc in shown)
    lines.extend("" for _ in range(body_rows - len(shown)))

    # --- footer ---
    if total:
        order = "newest first" if sort_mode == SORT_TIME else "most suspicious at top"
        footer = (
            f" showing {len(shown)} of {total}   sort: {sort_mode} ({order})"
            "   [s]core [t]ime "
        )
    else:
        footer = (
            " waiting for scored processes\u2026  "
            "pipe `process_stream | model score` into this monitor "
        )
    lines.append(f"{_DIM}{_truncate(footer, width)}{_RESET}")
    return lines
