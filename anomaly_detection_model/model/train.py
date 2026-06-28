"""Training pipeline: NDJSON baseline -> versioned anomaly-detection artifact.

Two passes over the (filtered) records:

1. Build frozen saturating window-count tables (identity) and per-image temporal
   row counts for every categorical field.
2. Featurize each record into a numeric matrix; fit the ``StandardScaler`` and
   ``IsolationForest``; compute the three scoring heads (identity NLL, forest,
   temporal NLL); fit per-head robust normalization; build the combined-score
   quantile sketch; persist everything as one versioned :class:`Artifact`.
"""

from __future__ import annotations

import json
import math
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import MODEL_VERSION
from .artifact import Artifact, HeadNorm
from .featurize import (
    FEATURE_COLUMNS,
    categorical_values,
    frequency_features,
    head_a_nll,
    head_c_nll,
    is_eligible,
    numeric_features,
    temporal_features,
    temporal_values,
    window_id,
)
from .vocab import Vocabulary

# Constant scaling MAD into a robust standard-deviation estimate.
_MAD_SCALE = 1.4826
# Smallest robust scale used when z-scoring a head, to avoid divide-by-zero on a
# degenerate (constant) training distribution.
_SCALE_FLOOR = 1e-9
# Number of evenly spaced quantiles stored for the percentile mapping sketch.
_N_QUANTILES = 1000
# Sentinel window id for records without a usable timestamp (see featurize).
_NO_WINDOW = -1

# Progress callback signature: (phase, done, total). ``total`` is 0 for
# indeterminate stages that only announce a phase transition.
ProgressFn = Callable[[str, int, int], None]
# Emit a per-record progress tick at most this often to keep output cheap.
_PROGRESS_EVERY = 500


def _noop_progress(phase: str, done: int, total: int) -> None:
    """Default progress sink that ignores every update."""


def iter_records(lines: Iterable[str]) -> Iterator[Dict[str, object]]:
    """Yield parsed JSON objects from an iterable of NDJSON lines.

    Blank lines and lines that fail to parse are skipped silently so a partially
    written stream does not abort training.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def load_training_records(path: str) -> List[Dict[str, object]]:
    """Read ``path`` and return the eligible records to train on."""
    with open(path, "r", encoding="utf-8") as handle:
        return [r for r in iter_records(handle) if is_eligible(r)]


def _detect_schema_version(records: List[Dict[str, object]]) -> str:
    """Schema version shared by the training records (the most common one)."""
    versions: Dict[str, int] = {}
    for record in records:
        version = record.get("schema_version")
        if isinstance(version, str):
            versions[version] = versions.get(version, 0) + 1
    if not versions:
        return "unknown"
    return max(versions, key=lambda v: versions[v])


def _resolve_k_eff(window_ids: Sequence[Optional[int]], saturation_k: int) -> int:
    """Adaptive saturation cap for short baselines.

    A value can only reach ``K`` distinct windows if the baseline spans at least
    ``K`` windows of wall-clock time. On shorter captures the cap is lowered to
    ``ceil(0.5 * distinct_windows)`` (but at least 1) so "regular" remains
    reachable, and the resolved value is stored so train/score agree.
    """
    distinct = len({w for w in window_ids if w is not None})
    if distinct <= 0:
        return 1
    return max(1, min(saturation_k, math.ceil(0.5 * distinct)))


def _build_vocab(
    records: List[Dict[str, object]],
    window_ids: Sequence[Optional[int]],
    *,
    alpha: float,
    window_minutes: int,
    saturation_k: int,
    alpha_t: float,
    hour_buckets: int,
    dow_buckets: int,
    temporal_min_samples: int,
    progress: ProgressFn = _noop_progress,
) -> Vocabulary:
    """First pass: saturating identity windows + per-image temporal row counts."""
    vocab = Vocabulary(
        alpha=alpha,
        window_minutes=window_minutes,
        saturation_k=saturation_k,
        alpha_t=alpha_t,
        hour_buckets=hour_buckets,
        dow_buckets=dow_buckets,
        temporal_min_samples=temporal_min_samples,
    )
    total = len(records)
    for index, (record, wid) in enumerate(zip(records, window_ids), start=1):
        cats = categorical_values(record)
        vocab.observe_row(cats, _NO_WINDOW if wid is None else wid)
        vocab.observe_temporal_context("image_name", cats["image_name"])
        for pair_field, pair_value in temporal_values(record, vocab).items():
            vocab.observe_temporal_pair(pair_field, pair_value)
        if index % _PROGRESS_EVERY == 0:
            progress("vocab", index, total)
    progress("vocab", total, total)
    return vocab


def _build_matrix(
    records: List[Dict[str, object]],
    vocab: Vocabulary,
    progress: ProgressFn = _noop_progress,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Second pass: dense feature matrix and the Head-A / Head-C NLL vectors."""
    rows: List[List[float]] = []
    head_a: List[float] = []
    head_c: List[float] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        freq = frequency_features(record, vocab)
        temporal = temporal_features(record, vocab)
        head_a.append(head_a_nll(freq))
        head_c.append(head_c_nll(temporal))
        full = dict(freq)
        full.update(temporal)
        full.update(numeric_features(record))
        rows.append([full[col] for col in FEATURE_COLUMNS])
        if index % _PROGRESS_EVERY == 0:
            progress("featurize", index, total)
    progress("featurize", total, total)
    matrix = np.asarray(rows, dtype=np.float64)
    return (
        matrix,
        np.asarray(head_a, dtype=np.float64),
        np.asarray(head_c, dtype=np.float64),
    )


def _fit_head_norm(values: np.ndarray) -> HeadNorm:
    """Fit winsor bounds (p0.1/p99.9) and a robust median/MAD scale for a head.

    The scale falls back from the robust ``1.4826 * MAD`` to the winsorized
    standard deviation when more than half the head sits exactly at its floor
    (so MAD collapses to 0 even though an informative upper tail exists, as for
    the gated temporal head). A truly constant head has ``lo == hi`` and is thus
    self-disabling: every value, including novel ones, winsorizes to that
    constant and contributes a zero z-score rather than exploding.
    """
    lo = float(np.quantile(values, 0.001))
    hi = float(np.quantile(values, 0.999))
    clipped = np.clip(values, lo, hi)
    median = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped - median)))
    scale = max(_MAD_SCALE * mad, float(clipped.std()), _SCALE_FLOOR)
    return HeadNorm(lo=lo, hi=hi, median=median, scale=scale)


def _apply_head_norm(values: np.ndarray, norm: HeadNorm) -> np.ndarray:
    """Vectorized winsorize + robust z-score of a head array with stored params."""
    clipped = np.clip(values, norm.lo, norm.hi)
    return (clipped - norm.median) / norm.scale


def train(
    records: List[Dict[str, object]],
    *,
    alpha: float = 1.0,
    contamination: object = "auto",
    n_estimators: int = 200,
    seed: int = 0,
    window_minutes: int = 60,
    saturation_k: int = 5,
    alpha_t: float = 1.0,
    temporal_min_samples: int = 20,
    hour_buckets: int = 8,
    dow_buckets: int = 2,
    head_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    medium_quantile: float = 0.90,
    high_quantile: float = 0.99,
    progress: ProgressFn = _noop_progress,
) -> Artifact:
    """Fit the full pipeline over ``records`` and return a versioned artifact."""
    if not records:
        raise ValueError("no eligible training records (need real process_start events)")

    schema_version = _detect_schema_version(records)

    window_ids = [window_id(r, window_minutes) for r in records]
    k_eff = _resolve_k_eff(window_ids, saturation_k)

    vocab = _build_vocab(
        records,
        window_ids,
        alpha=alpha,
        window_minutes=window_minutes,
        saturation_k=k_eff,
        alpha_t=alpha_t,
        hour_buckets=hour_buckets,
        dow_buckets=dow_buckets,
        temporal_min_samples=temporal_min_samples,
        progress=progress,
    )
    matrix, head_a, head_c = _build_matrix(records, vocab, progress)

    progress("scaler", 0, 0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    progress("isolation-forest", 0, 0)
    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
    )
    iforest.fit(scaled)

    progress("normalize", 0, 0)
    # Higher = more anomalous (score_samples returns higher = more normal).
    head_b = -iforest.score_samples(scaled)

    head_a_norm = _fit_head_norm(head_a)
    head_b_norm = _fit_head_norm(head_b)
    head_c_norm = _fit_head_norm(head_c)

    weight_a, weight_b, weight_c = head_weights
    za = _apply_head_norm(head_a, head_a_norm)
    zb = _apply_head_norm(head_b, head_b_norm)
    zc = _apply_head_norm(head_c, head_c_norm)
    combined = weight_a * za + weight_b * zb + weight_c * zc

    combined_quantiles = np.quantile(
        combined, np.linspace(0.0, 1.0, _N_QUANTILES)
    ).tolist()

    distinct_windows = len({w for w in window_ids if w is not None})
    summary = {
        "n_records": len(records),
        "n_features": len(FEATURE_COLUMNS),
        "distinct_windows": distinct_windows,
        "saturation_k_eff": k_eff,
        "combined_score_min": float(combined.min()),
        "combined_score_max": float(combined.max()),
        "combined_score_mean": float(combined.mean()),
        "head_a_median": head_a_norm.median,
        "head_b_median": head_b_norm.median,
        "head_c_median": head_c_norm.median,
        "head_weights": list(head_weights),
        "medium_quantile": medium_quantile,
        "high_quantile": high_quantile,
    }

    return Artifact(
        vocab=vocab,
        scaler=scaler,
        iforest=iforest,
        feature_columns=list(FEATURE_COLUMNS),
        alpha=alpha,
        seed=seed,
        schema_version=schema_version,
        window_minutes=window_minutes,
        saturation_k=k_eff,
        alpha_t=alpha_t,
        temporal_min_samples=temporal_min_samples,
        hour_buckets=hour_buckets,
        dow_buckets=dow_buckets,
        head_a_norm=head_a_norm,
        head_b_norm=head_b_norm,
        head_c_norm=head_c_norm,
        head_weights=tuple(head_weights),
        combined_quantiles=combined_quantiles,
        threshold_medium=medium_quantile,
        threshold_high=high_quantile,
        model_version=MODEL_VERSION,
        train_summary=summary,
    )


def train_from_file(
    input_path: str,
    output_path: str,
    *,
    alpha: float = 1.0,
    contamination: object = "auto",
    seed: int = 0,
    window_minutes: int = 60,
    saturation_k: int = 5,
    alpha_t: float = 1.0,
    temporal_min_samples: int = 20,
    head_weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    medium_quantile: float = 0.90,
    high_quantile: float = 0.99,
    progress: ProgressFn = _noop_progress,
) -> Artifact:
    """Load ``input_path``, train, write the artifact to ``output_path``."""
    progress("load", 0, 0)
    records = load_training_records(input_path)
    artifact = train(
        records,
        alpha=alpha,
        contamination=contamination,
        seed=seed,
        window_minutes=window_minutes,
        saturation_k=saturation_k,
        alpha_t=alpha_t,
        temporal_min_samples=temporal_min_samples,
        head_weights=head_weights,
        medium_quantile=medium_quantile,
        high_quantile=high_quantile,
        progress=progress,
    )
    progress("save", 0, 0)
    artifact.save(output_path)
    return artifact
