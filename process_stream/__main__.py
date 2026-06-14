"""CLI entry point: backfill existing processes, then stream new starts as NDJSON."""

from __future__ import annotations

import argparse
import ctypes
import queue
import signal
import sys
import threading
from typing import Any, Dict

from . import _debug
from .backfill import snapshot
from .emit import Emitter
from .enrich import enrich, finalize
from .etw_source import EtwProcessSource
from .features import day_of_week, hour_of_day, image_name
from .record import ProcessRecord, integrity_from_sid, parse_iso, utc_now_iso


def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:  # noqa: BLE001
        return False


def _build_live_record(event: Dict[str, Any]) -> ProcessRecord:
    """Merge ETW core metadata with best-effort psutil enrichment."""
    pid = event.get("pid")
    info = enrich(pid) if pid is not None else None

    record = ProcessRecord(
        pid=pid if pid is not None else -1,
        source="new",
        timestamp=utc_now_iso(),
        ppid=event.get("ppid") if event.get("ppid") is not None else (info.ppid if info else None),
        process_seq=event.get("process_seq"),
        image=(info.image if info and info.image else event.get("image")),
        command_line=info.command_line if info else None,
        user=info.user if info else None,
        cwd=info.cwd if info else None,
        session_id=event.get("session_id"),
        integrity_level=integrity_from_sid(event.get("mandatory_label")),
        is_elevated=(bool(event["token_is_elevated"]) if event.get("token_is_elevated") is not None else None),
        parent_image=info.parent_image if info else None,
        parent_command_line=info.parent_command_line if info else None,
        parent_user=info.parent_user if info else None,
        create_time=event.get("create_time") or (info.create_time if info else None),
        enriched=bool(info and info.ok),
    )
    finalize(record)
    return record


def _lifetime_ms(create_time: Any, exit_time: Any) -> Any:
    """Milliseconds between two ISO-8601 timestamps, or ``None``."""
    start = parse_iso(create_time)
    end = parse_iso(exit_time)
    if start is None or end is None:
        return None
    delta = (end - start).total_seconds() * 1000.0
    return int(delta) if delta >= 0 else None


def _build_stop_record(event: Dict[str, Any]) -> ProcessRecord:
    """Build a ``process_stop`` record carrying exit code and lifetime."""
    pid = event.get("pid")
    create_time = event.get("create_time")
    record = ProcessRecord(
        pid=pid if pid is not None else -1,
        source="new",
        timestamp=utc_now_iso(),
        event="process_stop",
        process_seq=event.get("process_seq"),
        image=event.get("image"),
        create_time=create_time,
        exit_code=event.get("exit_code"),
        lifetime_ms=_lifetime_ms(create_time, event.get("exit_time")),
    )
    record.is_pseudo = False
    # Derived features are cheap pure functions of fields the stop event already
    # carries, so populate them here too for shape parity with start records.
    # The ETW stop event only reports the image basename (no path), so
    # ``path_bucket`` is left null rather than misclassifying it as "Other".
    record.image_name = image_name(record.image)
    record.hour_of_day = hour_of_day(record.create_time or record.timestamp)
    record.day_of_week = day_of_week(record.create_time or record.timestamp)
    return record


def _build_record(event: Dict[str, Any]) -> ProcessRecord:
    if event.get("kind") == "stop":
        return _build_stop_record(event)
    return _build_live_record(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="process-stream",
        description="Stream Windows process starts as NDJSON, enriched for anomaly detection.",
    )
    parser.add_argument("--no-backfill", action="store_true", help="Skip the existing-process snapshot.")
    parser.add_argument("--pretty", action="store_true", help="Emit indented JSON (debug aid).")
    parser.add_argument("--queue-size", type=int, default=10000, help="ETW event buffer size.")
    parser.add_argument(
        "--drop-pseudo",
        action="store_true",
        help="Omit pseudo / non-launch records (System Idle, Registry, ...) entirely.",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Dump each raw ETW event dict to stderr (for verifying field names).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log best-effort errors (normally swallowed) to stderr for diagnosis.",
    )
    args = parser.parse_args(argv)

    _debug.set_enabled(args.debug)

    if not _is_admin():
        print(
            "[process-stream] ERROR: must be run as Administrator (ETW real-time session).",
            file=sys.stderr,
            flush=True,
        )
        return 1

    emitter = Emitter(pretty=args.pretty)
    source = EtwProcessSource(queue_size=args.queue_size, debug_raw=args.debug_raw)

    def _emit(record: ProcessRecord) -> None:
        if args.drop_pseudo and record.is_pseudo:
            return
        emitter.write(record)

    stop_event = threading.Event()

    def _handle_signal(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # 1. Start ETW first so live events buffer while we backfill (avoids a gap).
    source.start()
    try:
        # 2. Backfill currently-running processes.
        if not args.no_backfill:
            for record in snapshot():
                _emit(record)
            emitter.flush()

        # 3. Consume live ProcessStart/ProcessStop events until interrupted.
        while not stop_event.is_set():
            try:
                event = source.events.get(timeout=0.5)
            except queue.Empty:
                emitter.flush()  # caught up: flush buffered output
                continue
            _emit(_build_record(event))
            if source.events.empty():
                emitter.flush()  # bound latency once momentarily drained
    finally:
        source.stop()
        # Drain anything still buffered so we don't silently lose tail events.
        while True:
            try:
                event = source.events.get_nowait()
            except queue.Empty:
                break
            _emit(_build_record(event))
        emitter.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
