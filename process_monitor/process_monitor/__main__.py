"""CLI entry point for the live process-anomaly monitor.

Example::

    python -m process_stream | python -m model score --model model.joblib \\
        | python -m process_monitor
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .monitor import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="process-monitor",
        description=(
            "Full-screen, non-scrolling monitor of scored process starts, "
            "ranked with the most suspicious process at the top."
        ),
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=0.5,
        help="Screen refresh interval in seconds (default: 0.5).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(refresh=args.refresh)


if __name__ == "__main__":
    raise SystemExit(main())
