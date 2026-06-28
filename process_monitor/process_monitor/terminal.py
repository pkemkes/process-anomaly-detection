"""Non-blocking console keyboard input for the monitor's hotkeys.

Rich (via :class:`rich.live.Live`) owns the screen: it switches to the
alternate buffer, enables virtual-terminal processing on Windows and repaints
without flicker. All that remains here is reading single keypresses directly
from the controlling terminal, because the monitor's ``stdin`` is a pipe
carrying scored NDJSON and cannot also be polled for hotkeys.
"""

from __future__ import annotations

import os
from typing import List, Optional, TextIO


class KeyReader:
    """Non-blocking reader of single keypresses from the controlling terminal.

    The monitor's ``stdin`` is a pipe carrying scored NDJSON, so hotkeys must be
    read from the console directly: ``msvcrt`` on Windows, ``/dev/tty`` in raw
    mode elsewhere. When no real terminal is attached (e.g. fully redirected
    I/O) the reader degrades to a no-op and :meth:`poll` always returns ``[]``.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._msvcrt = None
        self._tty: Optional[TextIO] = None
        self._fd: Optional[int] = None
        self._saved = None

    def __enter__(self) -> "KeyReader":
        if os.name == "nt":
            try:
                import msvcrt

                self._msvcrt = msvcrt
                self._enabled = True
            except ImportError:
                self._enabled = False
            return self
        try:
            import termios
            import tty

            self._tty = open("/dev/tty", "rb", buffering=0)  # noqa: SIM115
            self._fd = self._tty.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except (ImportError, OSError):
            self._enabled = False
            if self._tty is not None:
                self._tty.close()
                self._tty = None
        return self

    def __exit__(self, *_exc: object) -> bool:
        if self._tty is not None:
            if self._saved is not None:
                try:
                    import termios

                    termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
                except Exception:  # noqa: BLE001
                    pass
            self._tty.close()
            self._tty = None
        self._enabled = False
        return False

    def poll(self) -> List[str]:
        """Return all keypresses available right now (possibly empty)."""
        if not self._enabled:
            return []
        if self._msvcrt is not None:
            keys: List[str] = []
            while self._msvcrt.kbhit():
                ch = self._msvcrt.getwch()
                # Function/arrow keys arrive as a two-char prefix; drop them.
                if ch in ("\x00", "\xe0"):
                    if self._msvcrt.kbhit():
                        self._msvcrt.getwch()
                    continue
                keys.append(ch)
            return keys
        import select

        keys = []
        while select.select([self._fd], [], [], 0)[0]:
            data = os.read(self._fd, 1)
            if not data:
                break
            keys.append(data.decode("utf-8", "replace"))
        return keys
