"""The monitor run-loop: read scored NDJSON on stdin, repaint on a timer.

A background reader thread consumes stdin line-by-line (so freshly scored
processes appear with minimal latency even though stdin is a pipe) and folds
each record into a shared :class:`ProcessStore`. The main thread owns the
terminal -- via Rich's :class:`~rich.live.Live` in alternate-screen mode -- and
repaints the ranked table at a fixed interval until interrupted.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import TextIO

from rich.console import Console
from rich.live import Live

from .render import render_frame
from .store import ProcessStore, SORT_SCORE, SORT_TIME
from .terminal import KeyReader


def _reconfigure_utf8(stream: TextIO) -> None:
    """Best-effort switch of a text stream to UTF-8 (Python 3.7+)."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


class _Reader(threading.Thread):
    """Reads scored NDJSON from stdin and updates the store under a lock."""

    def __init__(self, store: ProcessStore, lock: threading.Lock, stream: TextIO) -> None:
        super().__init__(daemon=True)
        self._store = store
        self._lock = lock
        self._stream = stream
        self.eof = False

    def run(self) -> None:
        while True:
            line = self._stream.readline()
            if line == "":  # EOF: upstream pipeline closed
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            with self._lock:
                self._store.update(record)
        self.eof = True


def run(refresh: float = 0.5) -> int:
    """Run the monitor until EOF-then-quit or Ctrl+C. Returns an exit code."""
    _reconfigure_utf8(sys.stdout)
    _reconfigure_utf8(sys.stdin)

    store = ProcessStore()
    lock = threading.Lock()
    reader = _Reader(store, lock, sys.stdin)
    reader.start()
    started = time.monotonic()

    console = Console()
    try:
        with Live(
            console=console, screen=True, auto_refresh=False, transient=False
        ) as live, KeyReader() as keys:
            while True:
                for ch in keys.poll():
                    lower = ch.lower()
                    if lower == "s":
                        store.set_sort(SORT_SCORE)
                    elif lower == "t":
                        store.set_sort(SORT_TIME)
                    elif lower in (" ", "\t"):
                        store.toggle_sort()
                size = console.size
                with lock:
                    snapshot = store.snapshot()
                    counts = store.counts()
                    sort_mode = store.sort_mode
                frame = render_frame(
                    snapshot, counts, size.width, size.height, reader.eof, started,
                    sort_mode,
                )
                live.update(frame, refresh=True)
                time.sleep(refresh)
    except KeyboardInterrupt:
        return 0
    return 0

