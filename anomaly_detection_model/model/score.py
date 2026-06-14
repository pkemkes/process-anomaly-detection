"""Streaming scorer: load an artifact, score each record, augment the stream.

Scoring is stateless per line and uses the **frozen** training vocabulary --
unseen categories map to the smoothing floor (maximum surprise) and the model is
never refit. Each eligible ``process_start`` record is augmented with an
``anomaly_score``, a coarse ``anomaly_rank_hint``, the ``top_contributing_fields``
that drove the score, and the ``model_version``. Non-eligible records (pseudo
processes, ``process_stop``) pass through with a ``null`` score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

from .artifact import Artifact
from .featurize import (
    BOOLEAN_FLAGS,
    FEATURE_COLUMNS,
    FREQ_PREFIX,
    MISSING,
    categorical_values,
    frequency_features,
    head_a_nll,
    is_eligible,
    numeric_features,
)

# Minimum standardized deviation for a feature to be reported as a contributor.
_MIN_CONTRIB_Z = 1.0
# Default number of contributing fields to surface per record.
_DEFAULT_TOP_K = 5

# Human-readable templates for the lineage pair fields.
_PAIR_LABELS = {
    "pair_parent_image": ("parent", "image"),
    "pair_image_path": ("image", "path"),
    "pair_user_image": ("user", "image"),
}


@dataclass
class ScoreResult:
    """Outcome of scoring a single record."""

    anomaly_score: float
    rank_hint: str
    top_contributing_fields: List[str]


def _rank_hint(score: float, artifact: Artifact) -> str:
    """Map a combined score to ``low`` / ``medium`` / ``high`` via thresholds."""
    if score >= artifact.threshold_high:
        return "high"
    if score >= artifact.threshold_medium:
        return "medium"
    return "low"


def _pair_label(field: str, value: str) -> str:
    """Render a lineage-pair feature as ``pair(left=a,right=b)``."""
    left_name, right_name = _PAIR_LABELS[field]
    parts = value.split("\x1f")
    left = parts[0] if parts else MISSING
    right = parts[1] if len(parts) > 1 else MISSING
    return f"pair({left_name}={left},{right_name}={right})"


def _numeric_label(col: str, value: float) -> str:
    """Render a boolean flag / command-line scalar feature for an analyst."""
    if col in BOOLEAN_FLAGS:
        if value >= 1.0:
            state = "true"
        elif value <= 0.0:
            state = "false"
        else:
            state = "unknown"
        return f"{col}={state}"
    if value == int(value):
        return f"{col}={int(value)}"
    return f"{col}={round(value, 2)}"


def _label(col: str, cats: Dict[str, str], numeric: Dict[str, float]) -> str:
    """Friendly label for a feature column given the record's raw values."""
    if col.startswith(FREQ_PREFIX):
        field = col[len(FREQ_PREFIX):]
        value = cats[field]
        if field in _PAIR_LABELS:
            return _pair_label(field, value)
        return f"{field}={value}"
    return _numeric_label(col, numeric.get(col, 0.0))


class Scorer:
    """Scores records against a loaded :class:`Artifact`."""

    def __init__(self, artifact: Artifact, top_k: int = _DEFAULT_TOP_K) -> None:
        self.artifact = artifact
        self.top_k = top_k

    def score_record(self, record: Dict[str, object]) -> ScoreResult:
        """Compute the anomaly score and explanation for one eligible record."""
        art = self.artifact
        cats = categorical_values(record)
        freq = frequency_features(record, art.vocab)
        numeric = numeric_features(record)
        full = dict(freq)
        full.update(numeric)

        vector = np.asarray([[full[col] for col in FEATURE_COLUMNS]], dtype=np.float64)
        scaled = art.scaler.transform(vector)

        head_a = head_a_nll(freq)
        head_b = float(-art.iforest.score_samples(scaled)[0])

        za = (head_a - art.head_a_mean) / art.head_a_std
        zb = (head_b - art.head_b_mean) / art.head_b_std
        weight_a, weight_b = art.head_weights
        combined = float(weight_a * za + weight_b * zb)

        contributors = self._explain(scaled[0], cats, numeric)
        return ScoreResult(
            anomaly_score=combined,
            rank_hint=_rank_hint(combined, art),
            top_contributing_fields=contributors,
        )

    def _explain(
        self, scaled: np.ndarray, cats: Dict[str, str], numeric: Dict[str, float]
    ) -> List[str]:
        """Top contributing features by standardized deviation (anomalous side)."""
        ranked = sorted(
            zip(FEATURE_COLUMNS, scaled),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        labels: List[str] = []
        for col, z in ranked:
            # Frequency columns are only anomalous when *rarer* than average
            # (positive deviation); a common value is not a contributor.
            if col.startswith(FREQ_PREFIX) and z <= 0:
                continue
            if abs(z) < _MIN_CONTRIB_Z and labels:
                break
            labels.append(_label(col, cats, numeric))
            if len(labels) >= self.top_k:
                break
        return labels


def augment(record: Dict[str, object], result: Optional[ScoreResult], model_version: str) -> Dict[str, object]:
    """Return ``record`` with anomaly fields appended (``null`` if not scored)."""
    augmented = dict(record)
    if result is None:
        augmented["anomaly_score"] = None
        augmented["anomaly_rank_hint"] = None
        augmented["top_contributing_fields"] = None
    else:
        augmented["anomaly_score"] = result.anomaly_score
        augmented["anomaly_rank_hint"] = result.rank_hint
        augmented["top_contributing_fields"] = result.top_contributing_fields
    augmented["model_version"] = model_version
    return augmented


def score_stream(
    lines: Iterable[str],
    artifact: Artifact,
    *,
    top_k: int = _DEFAULT_TOP_K,
    guard_schema: bool = True,
) -> Iterator[str]:
    """Yield augmented NDJSON lines for an iterable of input NDJSON lines.

    Non-eligible records pass through with ``null`` anomaly fields. Eligible
    records are scored; with ``guard_schema`` set, a record whose
    ``schema_version`` differs from the model's raises
    :class:`~model.artifact.SchemaMismatchError`.
    """
    scorer = Scorer(artifact, top_k=top_k)
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if is_eligible(record):
            if guard_schema:
                artifact.guard_schema(record.get("schema_version"))  # type: ignore[arg-type]
            result: Optional[ScoreResult] = scorer.score_record(record)
        else:
            result = None
        augmented = augment(record, result, artifact.model_version)
        yield json.dumps(augmented, separators=(",", ":"))
