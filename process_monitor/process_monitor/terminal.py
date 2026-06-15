"""Low-level terminal control for a non-scrolling, full-screen display.

Rendering uses raw ANSI/VT escape sequences so the monitor has **no
third-party dependencies** (``curses`` is not available on Windows). On Windows
the console's virtual-terminal processing mode must be enabled first; modern
*nix terminals support these sequences natively.

The :class:`Screen` context manager switches to the terminal's *alternate*
screen buffer on entry and restores the original buffer (and cursor) on exit,
so the user's scrollback is left untouched -- the monitor "takes over" the
terminal for its lifetime and cleanly hands it back afterwards.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, TextIO

# --- CSI / private-mode escape sequences -------------------------------------
_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"
_ENTER_ALT_SCREEN = "\x1b[?1049h"
_LEAVE_ALT_SCREEN = "\x1b[?1049l"
_DISABLE_WRAP = "\x1b[?7l"
_ENABLE_WRAP = "\x1b[?7h"
_CURSOR_HOME = "\x1b[H"
_CLEAR_LINE_TAIL = "\x1b[K"
_CLEAR_SCREEN_TAIL = "\x1b[J"
_RESET = "\x1b[0m"


def enable_vt_processing() -> bool:
    """Enable ANSI escape handling on the Windows console; no-op elsewhere.

    Returns ``True`` if VT sequences are expected to render, ``False`` if the
    console mode could not be configured (in which case output will contain
    literal escape codes -- the caller may still proceed but the display will
    be garbled).
    """
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        std_output_handle = -11
        enable_virtual_terminal_processing = 0x0004

        handle = kernel32.GetStdHandle(std_output_handle)
        if handle in (0, -1):
            return False
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        if not kernel32.SetConsoleMode(
            handle, mode.value | enable_virtual_terminal_processing
        ):
            return False
        return True
    except Exception:  # noqa: BLE001 - any failure simply means "no VT"
        return False


class Screen:
    """Alternate-screen context manager with a flicker-free frame renderer."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self.vt_ok = False

    def __enter__(self) -> "Screen":
        self.vt_ok = enable_vt_processing()
        self._stream.write(
            _ENTER_ALT_SCREEN + _DISABLE_WRAP + _HIDE_CURSOR + _CURSOR_HOME
        )
        self._stream.flush()
        return self

    def __exit__(self, *_exc: object) -> bool:
        self._stream.write(
            _RESET + _SHOW_CURSOR + _ENABLE_WRAP + _LEAVE_ALT_SCREEN
        )
        self._stream.flush()
        return False

    def render(self, lines: List[str]) -> None:
        """Repaint the whole screen from the top without scrolling.

        Each line clears to its end so stale characters from a previous,
        longer frame are wiped; the screen tail is cleared after the last line.
        No trailing newline is emitted, which keeps the final row pinned to the
        bottom of the viewport instead of scrolling the alternate buffer.
        """
        body = (_CLEAR_LINE_TAIL + "\n").join(lines)
        self._stream.write(_CURSOR_HOME + body + _CLEAR_LINE_TAIL + _CLEAR_SCREEN_TAIL)
        self._stream.flush()


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
