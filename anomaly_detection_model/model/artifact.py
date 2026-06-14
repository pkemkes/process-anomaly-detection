"""Versioned model artifact: container, persistence, and schema guard.

A single :class:`Artifact` bundles everything needed to score a stream: the
frozen frequency :class:`~process_stream.model.vocab.Vocabulary`, the fitted
scaler and Isolation Forest, the per-head normalizer parameters, the score
thresholds, and the metadata (feature column order, ``alpha``, seed, stream
``schema_version``, model version) required to reproduce and guard scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import joblib

from . import MODEL_VERSION
from .vocab import Vocabulary


class SchemaMismatchError(RuntimeError):
    """Raised when a record's ``schema_version`` differs from the trained model."""


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
    # Per-head normalization (z-score) parameters fit on the training scores.
    head_a_mean: float
    head_a_std: float
    head_b_mean: float
    head_b_std: float
    head_weights: tuple  # (weight_a, weight_b)
    # Combined-score thresholds (training quantiles) for the rank hint.
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
