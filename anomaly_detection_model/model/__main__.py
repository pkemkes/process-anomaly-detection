"""CLI entry point for the anomaly model: ``train`` and ``score`` subcommands.

Examples::

    python -m model train --input processes.ndjson --out model.joblib
    python -m model score --model model.joblib --input new.ndjson
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .artifact import Artifact, SchemaMismatchError
from .score import score_stream
from .train import train_from_file


def _cmd_train(args: argparse.Namespace) -> int:
    contamination: object = args.contamination
    if contamination != "auto":
        contamination = float(contamination)
    artifact = train_from_file(
        args.input,
        args.out,
        alpha=args.alpha,
        contamination=contamination,
        seed=args.seed,
    )
    summary = artifact.train_summary
    print(
        f"[model] trained on {summary.get('n_records')} records, "
        f"{summary.get('n_features')} features -> {args.out} "
        f"(schema {artifact.schema_version}, model {artifact.model_version})",
        file=sys.stderr,
        flush=True,
    )
    return 0


def _open_input(path: Optional[str]):
    """Return a line iterator over ``path`` or stdin."""
    if path is None or path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8")


def _cmd_score(args: argparse.Namespace) -> int:
    artifact = Artifact.load(args.model)
    if args.threshold_medium is not None:
        artifact.threshold_medium = args.threshold_medium
    if args.threshold_high is not None:
        artifact.threshold_high = args.threshold_high

    handle = _open_input(args.input)
    try:
        for line in score_stream(
            handle,
            artifact,
            top_k=args.top_k,
            guard_schema=not args.no_schema_guard,
        ):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    except SchemaMismatchError as exc:
        print(f"[model] ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        if handle is not sys.stdin:
            handle.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anomaly-detection-model",
        description="Train and score an unsupervised process-anomaly baseline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Fit a baseline from recorded NDJSON.")
    train_p.add_argument("--input", required=True, help="Training NDJSON file.")
    train_p.add_argument("--out", required=True, help="Output artifact path (joblib).")
    train_p.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing (default 1.0).")
    train_p.add_argument(
        "--contamination",
        default="auto",
        help="IsolationForest contamination ('auto' or a float).",
    )
    train_p.add_argument("--seed", type=int, default=0, help="Random seed (default 0).")
    train_p.set_defaults(func=_cmd_train)

    score_p = sub.add_parser("score", help="Score an NDJSON stream against a model.")
    score_p.add_argument("--model", required=True, help="Artifact path (joblib).")
    score_p.add_argument(
        "--input",
        default=None,
        help="Input NDJSON file (default: stdin).",
    )
    score_p.add_argument("--top-k", type=int, default=5, help="Contributing fields to report.")
    score_p.add_argument(
        "--threshold-medium",
        type=float,
        default=None,
        help="Override the 'medium' rank-hint cutoff.",
    )
    score_p.add_argument(
        "--threshold-high",
        type=float,
        default=None,
        help="Override the 'high' rank-hint cutoff.",
    )
    score_p.add_argument(
        "--no-schema-guard",
        action="store_true",
        help="Score even when the record schema_version differs from the model.",
    )
    score_p.set_defaults(func=_cmd_score)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
