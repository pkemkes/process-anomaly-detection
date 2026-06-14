"""Training pipeline: NDJSON baseline -> versioned anomaly-detection artifact.

Two passes over the (filtered) records:

1. Build frozen frequency tables for every categorical single/pair field.
2. Featurize each record into a numeric matrix; fit the ``StandardScaler`` and
   ``IsolationForest``; fit the per-head score normalizer and the rank-hint
   thresholds; persist everything as one versioned :class:`Artifact`.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from . import MODEL_VERSION
from .artifact import Artifact
from .featurize import (
    FEATURE_COLUMNS,
    categorical_values,
    frequency_features,
    head_a_nll,
    is_eligible,
    numeric_features,
)
from .vocab import Vocabulary

# Smallest standard deviation used when z-scoring a head, to avoid divide-by-zero
# on a degenerate (constant) training distribution.
_STD_FLOOR = 1e-9


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


def _build_vocab(records: List[Dict[str, object]], alpha: float) -> Vocabulary:
    """First pass: accumulate frequency tables over every categorical field."""
    vocab = Vocabulary(alpha=alpha)
    for record in records:
        vocab.observe_row(categorical_values(record))
    return vocab


def _build_matrix(
    records: List[Dict[str, object]], vocab: Vocabulary
) -> Tuple[np.ndarray, np.ndarray]:
    """Second pass: dense feature matrix and the per-record Head-A NLL vector."""
    rows: List[List[float]] = []
    head_a: List[float] = []
    for record in records:
        freq = frequency_features(record, vocab)
        head_a.append(head_a_nll(freq))
        full = dict(freq)
        full.update(numeric_features(record))
        rows.append([full[col] for col in FEATURE_COLUMNS])
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix, np.asarray(head_a, dtype=np.float64)


def train(
    records: List[Dict[str, object]],
    *,
    alpha: float = 1.0,
    contamination: object = "auto",
    n_estimators: int = 200,
    seed: int = 0,
    head_weights: Tuple[float, float] = (0.5, 0.5),
    medium_quantile: float = 0.90,
    high_quantile: float = 0.99,
) -> Artifact:
    """Fit the full pipeline over ``records`` and return a versioned artifact."""
    if not records:
        raise ValueError("no eligible training records (need real process_start events)")

    schema_version = _detect_schema_version(records)
    vocab = _build_vocab(records, alpha)
    matrix, head_a = _build_matrix(records, vocab)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)

    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
    )
    iforest.fit(scaled)

    # Higher = more anomalous (score_samples returns higher = more normal).
    head_b = -iforest.score_samples(scaled)

    head_a_mean = float(head_a.mean())
    head_a_std = float(head_a.std()) or _STD_FLOOR
    head_b_mean = float(head_b.mean())
    head_b_std = float(head_b.std()) or _STD_FLOOR

    weight_a, weight_b = head_weights
    za = (head_a - head_a_mean) / head_a_std
    zb = (head_b - head_b_mean) / head_b_std
    combined = weight_a * za + weight_b * zb

    threshold_medium = float(np.quantile(combined, medium_quantile))
    threshold_high = float(np.quantile(combined, high_quantile))

    summary = {
        "n_records": len(records),
        "n_features": len(FEATURE_COLUMNS),
        "combined_score_min": float(combined.min()),
        "combined_score_max": float(combined.max()),
        "combined_score_mean": float(combined.mean()),
        "head_a_mean": head_a_mean,
        "head_b_mean": head_b_mean,
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
        head_a_mean=head_a_mean,
        head_a_std=head_a_std,
        head_b_mean=head_b_mean,
        head_b_std=head_b_std,
        head_weights=tuple(head_weights),
        threshold_medium=threshold_medium,
        threshold_high=threshold_high,
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
) -> Artifact:
    """Load ``input_path``, train, write the artifact to ``output_path``."""
    records = load_training_records(input_path)
    artifact = train(records, alpha=alpha, contamination=contamination, seed=seed)
    artifact.save(output_path)
    return artifact
