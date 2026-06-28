"""Pure formatting of a monitor frame into a Rich renderable.

``render_frame`` is a side-effect-free function of the data plus the current
terminal size: it builds a :class:`rich.console.Group` (status bar + ranked
:class:`rich.table.Table` + footer) that the monitor loop hands straight to
:class:`rich.live.Live`. Rich owns colours, elastic column widths, truncation
and the flicker-free repaint, so this module only decides *what* to show.

Layout (one line per process, most suspicious at the top)::

    PROCESS ANOMALY MONITOR        123 procs  high 4  medium 9  live  Ctrl+C quit
      SCORE    PID  IMAGE                    USER               DETAIL
      12.84   8421  powershell.exe           CORP\\alice         pair(parent=...)
       9.10   2290  rundll32.exe             CORP\\alice         freq_image_name=...
       ...
     showing 18 of 123   sort: score (most suspicious at top)
"""

from __future__ import annotations

import time
from typing import Sequence, Tuple

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .store import Process, SORT_TIME

# Rank-hint -> row colour (Rich style names).
_RANK_STYLE = {"high": "bright_red", "medium": "bright_yellow", "low": "bright_green"}

# SCORE and PID are fixed-width; IMAGE, USER and DETAIL are elastic and share
# the remaining width via Rich column ratios (DETAIL absorbs the remainder).
_SCORE_W = 7
_PID_W = 6


def _pad(text: str, width: int) -> str:
    """Clip or right-pad ``text`` to exactly ``width`` columns."""
    if len(text) >= width:
        return text[:width]
    return text + " " * (width - len(text))


def _truncate(text: str, width: int) -> str:
    """Clip ``text`` to ``width`` columns, marking elision with an ellipsis."""
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "\u2026"
    return text[: width - 1] + "\u2026"


def _fmt_elapsed(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def _title_bar(
    width: int, counts: Tuple[int, int, int], eof: bool, started: float
) -> Text:
    high, medium, total = counts
    title = " PROCESS ANOMALY MONITOR"
    status = "stream ended" if eof else "live"
    elapsed = _fmt_elapsed(int(time.monotonic() - started))
    stats = (
        f"{total} procs   high {high}  medium {medium}   "
        f"{status}   {elapsed}   Ctrl+C quit "
    )
    bar = title + stats.rjust(max(0, width - len(title)))
    return Text(_pad(bar, width), style="reverse bold")


def _build_table() -> Table:
    table = Table(
        box=None,
        expand=True,
        pad_edge=False,
        padding=(0, 1, 0, 0),
        header_style="dim bold",
    )
    table.add_column("SCORE", justify="right", width=_SCORE_W, no_wrap=True)
    table.add_column("PID", justify="right", width=_PID_W, no_wrap=True)
    table.add_column("IMAGE", ratio=34, min_width=16, no_wrap=True, overflow="ellipsis")
    table.add_column("USER", ratio=26, min_width=12, no_wrap=True, overflow="ellipsis")
    table.add_column("DETAIL", ratio=40, min_width=12, no_wrap=True, overflow="ellipsis")
    return table


def render_frame(
    processes: Sequence[Process],
    counts: Tuple[int, int, int],
    width: int,
    height: int,
    eof: bool,
    started: float,
    sort_mode: str = "score",
) -> Group:
    """Build the renderable describing the current monitor state."""
    width = max(width, 20)
    height = max(height, 5)
    _, _, total = counts

    table = _build_table()
    # Title (1) + table header (1) + footer (1) leaves this many process rows;
    # blank filler rows keep the footer pinned to the bottom of the viewport.
    body_rows = max(height - 3, 1)
    shown = processes[:body_rows]
    for proc in shown:
        detail = proc.top_fields[0] if proc.top_fields else ""
        table.add_row(
            f"{proc.score:.2f}",
            str(proc.pid),
            proc.image_name,
            proc.user,
            detail,
            style=_RANK_STYLE.get(proc.rank_hint, "bright_green"),
        )
    for _ in range(body_rows - len(shown)):
        table.add_row("", "", "", "", "")

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
    footer_text = Text(_truncate(footer, width), style="dim")

    return Group(_title_bar(width, counts, eof, started), table, footer_text)
