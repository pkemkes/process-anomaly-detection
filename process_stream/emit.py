"""NDJSON output writer."""

from __future__ import annotations

import sys
from typing import TextIO

from .record import ProcessRecord


class Emitter:
    """Writes one JSON object per line to a stream; call ``flush`` to drain it."""

    def __init__(self, stream: TextIO | None = None, pretty: bool = False) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._pretty = pretty

    def write(self, record: ProcessRecord) -> None:
        self._stream.write(record.as_json(pretty=self._pretty) + "\n")

    def flush(self) -> None:
        self._stream.flush()
