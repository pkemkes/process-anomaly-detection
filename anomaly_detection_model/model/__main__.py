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

# Human-readable labels for the training phases reported via the progress hook.
_PHASE_LABELS = {
    "load": "loading records",
    "vocab": "building vocabulary",
    "featurize": "featurizing",
    "scaler": "fitting scaler",
    "isolation-forest": "fitting isolation forest",
    "normalize": "normalizing heads",
    "save": "saving artifact",
}


def _make_progress(stream):
    """Return a progress callback that renders a single updating stderr line."""
    is_tty = getattr(stream, "isatty", lambda: False)()

    def report(phase: str, done: int, total: int) -> None:
        label = _PHASE_LABELS.get(phase, phase)
        if total > 0:
            pct = (done / total) * 100.0
            text = f"[model] {label}: {done}/{total} ({pct:5.1f}%)"
        else:
            text = f"[model] {label}..."
        if is_tty:
            # Stay on one line while the percentage climbs; finish with newline.
            terminator = "\n" if (total > 0 and done >= total) else ""
            stream.write("\r\033[K" + text + terminator)
        else:
            stream.write(text + "\n")
        stream.flush()

    return report


def _cmd_train(args: argparse.Namespace) -> int:
    contamination: object = args.contamination
    if contamination != "auto":
        contamination = float(contamination)
    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 3:
        print("[model] ERROR: --weights expects three comma-separated numbers", file=sys.stderr)
        return 2
    progress = None if args.quiet else _make_progress(sys.stderr)
    artifact = train_from_file(
        args.input,
        args.out,
        alpha=args.alpha,
        contamination=contamination,
        seed=args.seed,
        window_minutes=args.window_minutes,
        saturation_k=args.saturation_k,
        alpha_t=args.alpha_t,
        temporal_min_samples=args.temporal_min_samples,
        head_weights=weights,  # type: ignore[arg-type]
        medium_quantile=args.medium_quantile,
        high_quantile=args.high_quantile,
        **({"progress": progress} if progress is not None else {}),
    )
    summary = artifact.train_summary
    print(
        f"[model] trained on {summary.get('n_records')} records, "
        f"{summary.get('n_features')} features -> {args.out} "
        f"(schema {artifact.schema_version}, model {artifact.model_version}, "
        f"K_eff {summary.get('saturation_k_eff')} over "
        f"{summary.get('distinct_windows')} windows)",
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
    # A file is scored in large batches for throughput; a live stream (stdin)
    # trickles one record at a time, so batch per record to bound latency --
    # otherwise output is withheld until the batch fills and nothing appears.
    streaming = handle is sys.stdin
    try:
        for line in score_stream(
            handle,
            artifact,
            top_k=args.top_k,
            guard_schema=not args.no_schema_guard,
            batch_size=1 if streaming else 2048,
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
    train_p.add_argument(
        "--window-minutes",
        type=int,
        default=60,
        help="Width of the recurrence window in minutes (default 60).",
    )
    train_p.add_argument(
        "--saturation-k",
        type=int,
        default=5,
        help="Distinct windows after which a value is 'regular' (default 5).",
    )
    train_p.add_argument(
        "--alpha-t",
        type=float,
        default=1.0,
        help="Smoothing for the temporal head (default 1.0).",
    )
    train_p.add_argument(
        "--temporal-min-samples",
        type=int,
        default=20,
        help="Minimum per-image samples before temporal surprise is emitted (default 20).",
    )
    train_p.add_argument(
        "--weights",
        default="0.4,0.3,0.3",
        help="Comma-separated head weights wa,wb,wc (default 0.4,0.3,0.3).",
    )
    train_p.add_argument(
        "--medium-quantile",
        type=float,
        default=0.90,
        help="Percentile level for the 'medium' rank hint (default 0.90).",
    )
    train_p.add_argument(
        "--high-quantile",
        type=float,
        default=0.99,
        help="Percentile level for the 'high' rank hint (default 0.99).",
    )
    train_p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-phase training progress output.",
    )
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
        help="Override the 'medium' rank-hint percentile level (0..1).",
    )
    score_p.add_argument(
        "--threshold-high",
        type=float,
        default=None,
        help="Override the 'high' rank-hint percentile level (0..1).",
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
