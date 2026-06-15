"""The monitor run-loop: read scored NDJSON on stdin, repaint on a timer.

A background reader thread consumes stdin line-by-line (so freshly scored
processes appear with minimal latency even though stdin is a pipe) and folds
each record into a shared :class:`ProcessStore`. The main thread owns the
terminal and repaints the ranked table at a fixed interval until interrupted.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
import time
from typing import TextIO

from .render import render_frame
from .store import ProcessStore
from .terminal import Screen


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

    try:
        with Screen() as screen:
            while True:
                size = shutil.get_terminal_size((80, 24))
                with lock:
                    snapshot = store.snapshot()
                    counts = store.counts()
                frame = render_frame(
                    snapshot, counts, size.columns, size.lines, reader.eof, started
                )
                screen.render(frame)
                time.sleep(refresh)
    except KeyboardInterrupt:
        return 0
    return 0
