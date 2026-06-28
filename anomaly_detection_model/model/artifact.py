"""Versioned model artifact: container, persistence, and schema guard.

A single :class:`Artifact` bundles everything needed to score a stream: the
frozen saturating :class:`~model.vocab.Vocabulary`, the fitted scaler and
Isolation Forest, the scoring configuration, the per-head robust-normalization
parameters, the combined-score quantile sketch used to map a record to a bounded
``0..1`` percentile, the percentile thresholds for the rank hint, and the
metadata required to reproduce and guard scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib

from . import MODEL_VERSION
from .vocab import Vocabulary


class SchemaMismatchError(RuntimeError):
    """Raised when a record's ``schema_version`` differs from the trained model."""


@dataclass
class HeadNorm:
    """Robust per-head normalization: winsor bounds + median/MAD scale.

    A head value is winsorized to ``[lo, hi]`` (the training ``[p0.1, p99.9]``
    quantiles) then robustly z-scored as ``(x - median) / scale`` where
    ``scale = max(1.4826 * mad, eps)``. Median/MAD resist the heavy tail a single
    essentially-never-seen field would otherwise create.
    """

    lo: float
    hi: float
    median: float
    scale: float

    def normalize(self, value: float) -> float:
        """Winsorize ``value`` to ``[lo, hi]`` and return its robust z-score."""
        clipped = min(max(value, self.lo), self.hi)
        return (clipped - self.median) / self.scale


@dataclass
class Artifact:
    """Everything required to score a process stream against a trained baseline."""

    vocab: Vocabulary
    scaler: Any  # sklearn StandardScaler
    iforest: Any  # sklearn IsolationForest
    feature_columns: List[str]
    alpha: float
    seed: int
    schema_version: str
    # Scoring configuration (mirrors the operational params held on the vocab).
    window_minutes: int
    saturation_k: int
    alpha_t: float
    temporal_min_samples: int
    hour_buckets: int
    dow_buckets: int
    # Per-head robust normalization (Head A identity, Head B forest, Head C time).
    head_a_norm: HeadNorm
    head_b_norm: HeadNorm
    head_c_norm: HeadNorm
    head_weights: Tuple[float, float, float]  # (weight_a, weight_b, weight_c)
    # Sorted quantile sketch of the training combined scores; a record's combined
    # score is mapped through it to an empirical percentile in [0, 1].
    combined_quantiles: List[float]
    # Rank-hint cutoffs expressed as percentile levels in [0, 1].
    threshold_medium: float
    threshold_high: float
    model_version: str = MODEL_VERSION
    train_summary: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Persist the artifact to ``path`` via joblib."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Artifact":
        """Load an artifact previously written by :meth:`save`."""
        artifact = joblib.load(path)
        if not isinstance(artifact, Artifact):
            raise TypeError(f"{path!r} does not contain a model Artifact")
        return artifact

    def guard_schema(self, record_schema_version: Optional[str]) -> None:
        """Raise :class:`SchemaMismatchError` if the record schema differs.

        Feature extraction assumes the training schema; scoring a record emitted
        under a different ``schema_version`` would silently misinterpret fields.
        """
        if record_schema_version != self.schema_version:
            raise SchemaMismatchError(
                f"record schema_version {record_schema_version!r} does not match "
                f"model schema_version {self.schema_version!r}"
            )
