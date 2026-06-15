"""Live table of the most recently scored processes.

The store consumes parsed NDJSON records emitted by ``model score`` and keeps a
single up-to-date row per live process, keyed by a stable identity. Identity,
de-duplication and ranking live here so the render layer can stay a pure
formatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class Process:
    """A single scored process as shown in one row of the display."""

    key: str
    pid: int
    image_name: str
    user: str
    score: float
    rank_hint: str
    top_fields: List[str]
    seq: int  # arrival order, used as a recency tie-breaker


def _basename(path: Optional[str]) -> str:
    """Best-effort executable basename from a Windows or POSIX path."""
    if not path:
        return ""
    return path.replace("/", "\\").rsplit("\\", 1)[-1]


def _identity(record: dict) -> str:
    """Stable key for a process across its start/stop records.

    Prefers the ETW ``process_seq`` (unique per boot, immune to PID reuse) and
    falls back to ``pid`` + ``create_time`` for backfill records that carry no
    sequence number.
    """
    seq = record.get("process_seq")
    if seq is not None:
        return f"seq:{seq}"
    return f"pid:{record.get('pid')}:{record.get('create_time')}"


class ProcessStore:
    """Holds the current set of live, scored processes."""

    def __init__(self) -> None:
        self._procs: Dict[str, Process] = {}
        self._counter = 0

    def update(self, record: dict) -> None:
        """Fold one scored NDJSON record into the table.

        - ``process_stop`` records evict the matching process.
        - Records with a ``null`` ``anomaly_score`` (pseudo / non-eligible) are
          ignored -- they have no suspiciousness to rank.
        - Eligible ``process_start`` records insert or refresh a row.
        """
        key = _identity(record)

        if record.get("event") == "process_stop":
            self._procs.pop(key, None)
            return

        score = record.get("anomaly_score")
        if score is None:
            return

        self._counter += 1
        self._procs[key] = Process(
            key=key,
            pid=int(record.get("pid", -1) or -1),
            image_name=record.get("image_name") or _basename(record.get("image")) or "?",
            user=record.get("user") or "",
            score=float(score),
            rank_hint=record.get("anomaly_rank_hint") or "low",
            top_fields=list(record.get("top_contributing_fields") or []),
            seq=self._counter,
        )

    def snapshot(self) -> List[Process]:
        """Processes ordered most-suspicious first (newest breaks ties)."""
        return sorted(self._procs.values(), key=lambda p: (-p.score, -p.seq))

    def counts(self) -> Tuple[int, int, int]:
        """``(high, medium, total)`` rank-hint tallies for the header."""
        high = sum(1 for p in self._procs.values() if p.rank_hint == "high")
        medium = sum(1 for p in self._procs.values() if p.rank_hint == "medium")
        return high, medium, len(self._procs)

    def __len__(self) -> int:
        return len(self._procs)
