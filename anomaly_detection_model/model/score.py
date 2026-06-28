"""Streaming scorer: load an artifact, score each record, augment the stream.

Scoring is stateless per line and uses the **frozen** training vocabulary --
unseen categories map to the smoothing floor (maximum surprise) and the model is
never refit. Each eligible ``process_start`` record is augmented with an
``anomaly_score``, a coarse ``anomaly_rank_hint``, the ``top_contributing_fields``
that drove the score, and the ``model_version``. Each contributing field carries
its ``contribution_pct`` -- the share (in percent) of the record's total
anomalous deviation attributable to that feature -- so an analyst can see at a
glance by how much it drove the score. Non-eligible records (pseudo processes,
``process_stop``) are dropped from the output stream.
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
    head_c_nll,
    is_eligible,
    numeric_features,
    temporal_features,
    temporal_values,
)

# Minimum standardized deviation for a feature to be reported as a contributor.
_MIN_CONTRIB_Z = 1.0
# Default number of contributing fields to surface per record.
_DEFAULT_TOP_K = 5

# Human-readable templates for the lineage / conditioned pair fields.
_PAIR_LABELS = {
    "pair_parent_image": ("parent", "image"),
    "pair_image_path": ("image", "path"),
    "pair_user_image": ("user", "image"),
    "pair_image_integrity": ("image", "integrity"),
    "pair_image_elevated": ("image", "elevated"),
    "pair_image_hour": ("image", "hour"),
    "pair_image_dow": ("image", "dow"),
}


@dataclass
class FieldContribution:
    """A single contributing feature and how much it drove the score.

    ``contribution_pct`` is the share, in percent, of the record's total
    anomalous deviation attributable to this feature -- a value of ``25.0`` means
    this single field accounts for roughly a quarter of how unusual the record
    looks. It is far easier to read than a raw z-score while preserving the same
    ranking.
    """

    field: str
    contribution_pct: float

    def as_dict(self) -> Dict[str, object]:
        """Serialize to a plain JSON-friendly mapping."""
        return {"field": self.field, "contribution_pct": self.contribution_pct}


@dataclass
class ScoreResult:
    """Outcome of scoring a single record."""

    anomaly_score: float
    rank_hint: str
    top_contributing_fields: List[FieldContribution]


def _percentile(value: float, quantiles: List[float]) -> float:
    """Empirical percentile of ``value`` against the training quantile sketch.

    Returns a bounded score in ``[0, 1]`` -- "more anomalous than this fraction of
    the baseline". Uses the mid-rank convention so a value tied with a dense
    cluster of baseline points lands in the middle of that tie rather than at its
    upper edge; this keeps a perfectly typical record near the centre of the
    distribution instead of artificially close to the high threshold.
    """
    arr = np.asarray(quantiles, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    lo = int(np.searchsorted(arr, value, side="left"))
    hi = int(np.searchsorted(arr, value, side="right"))
    return (lo + hi) / (2.0 * arr.size)


def _rank_hint(score: float, artifact: Artifact) -> str:
    """Map a percentile score to ``low`` / ``medium`` / ``high`` via thresholds."""
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
        return self.score_batch([record])[0]

    def score_batch(self, records: List[Dict[str, object]]) -> List[ScoreResult]:
        """Score many eligible records at once.

        The IsolationForest and scaler carry a large fixed per-call overhead, so
        scoring records one at a time is dominated by that overhead. Building a
        single feature matrix and calling ``transform`` / ``score_samples`` once
        per batch makes scoring a large stream orders of magnitude faster while
        producing identical results to :meth:`score_record`.
        """
        art = self.artifact
        n = len(records)
        if n == 0:
            return []

        cats_list: List[Dict[str, str]] = []
        freq_list: List[Dict[str, float]] = []
        temporal_list: List[Dict[str, float]] = []
        numeric_list: List[Dict[str, float]] = []
        matrix = np.empty((n, len(FEATURE_COLUMNS)), dtype=np.float64)
        for i, record in enumerate(records):
            cats = categorical_values(record)
            cats.update(temporal_values(record, art.vocab))
            freq = frequency_features(record, art.vocab)
            temporal = temporal_features(record, art.vocab)
            numeric = numeric_features(record)
            full = dict(freq)
            full.update(temporal)
            full.update(numeric)
            for j, col in enumerate(FEATURE_COLUMNS):
                matrix[i, j] = full[col]
            cats_list.append(cats)
            freq_list.append(freq)
            temporal_list.append(temporal)
            numeric_list.append(numeric)

        scaled = art.scaler.transform(matrix)
        forest = -art.iforest.score_samples(scaled)

        weight_a, weight_b, weight_c = art.head_weights
        results: List[ScoreResult] = []
        for i in range(n):
            head_a = head_a_nll(freq_list[i])
            head_b = float(forest[i])
            head_c = head_c_nll(temporal_list[i])

            za = art.head_a_norm.normalize(head_a)
            zb = art.head_b_norm.normalize(head_b)
            zc = art.head_c_norm.normalize(head_c)
            combined = weight_a * za + weight_b * zb + weight_c * zc

            anomaly_score = _percentile(combined, art.combined_quantiles)
            contributors = self._explain(scaled[i], cats_list[i], numeric_list[i])
            results.append(
                ScoreResult(
                    anomaly_score=anomaly_score,
                    rank_hint=_rank_hint(anomaly_score, art),
                    top_contributing_fields=contributors,
                )
            )
        return results

    def _explain(
        self, scaled: np.ndarray, cats: Dict[str, str], numeric: Dict[str, float]
    ) -> List[FieldContribution]:
        """Top contributing features (anomalous side), ranked by their share.

        Surprise columns (identity + temporal) and raw numeric columns are scored
        on the same standardized scale and against the same denominator, so they
        are ranked together by ``contribution_pct`` -- each field's share of the
        record's total anomalous deviation -- in descending order. Fields whose
        share rounds to ``0%`` are dropped as noise; the list keeps at most
        ``top_k`` genuine contributors.
        """
        # Total anomalous deviation across every feature, used as the denominator
        # for each field's percentage share. Surprise columns only count when
        # rarer/odder than average (positive side); numeric columns count their
        # absolute deviation either way.
        total_anom = float(
            sum(
                (z if z > 0 else 0.0) if col.startswith(FREQ_PREFIX) else abs(z)
                for col, z in zip(FEATURE_COLUMNS, scaled)
            )
        )

        candidates: List[tuple] = []
        for col, z in zip(FEATURE_COLUMNS, scaled):
            if col.startswith(FREQ_PREFIX):
                # Surprise columns are only anomalous when *rarer/odder* than
                # average (positive deviation); a common value is no contributor.
                if z > 0 and abs(z) >= _MIN_CONTRIB_Z:
                    candidates.append((col, z))
            elif abs(z) >= _MIN_CONTRIB_Z:
                candidates.append((col, z))
        candidates.sort(key=lambda item: abs(item[1]), reverse=True)

        contributions: List[FieldContribution] = []
        for col, z in candidates:
            contribution = self._contribution(col, z, cats, numeric, total_anom)
            if contribution.contribution_pct <= 0:
                # Negligible share (rounds to 0%) -- not worth surfacing.
                continue
            contributions.append(contribution)
            if len(contributions) >= self.top_k:
                break

        if not contributions:
            # Nothing crossed the reporting threshold: surface the single most
            # deviant column (anomalous side) so the explanation is never empty.
            ranked = sorted(
                zip(FEATURE_COLUMNS, scaled), key=lambda item: abs(item[1]), reverse=True
            )
            for col, z in ranked:
                if col.startswith(FREQ_PREFIX) and z <= 0:
                    continue
                contributions.append(self._contribution(col, z, cats, numeric, total_anom))
                break
        return contributions

    @staticmethod
    def _contribution(
        col: str,
        z: float,
        cats: Dict[str, str],
        numeric: Dict[str, float],
        total_anom: float,
    ) -> FieldContribution:
        """Build a labelled contribution as a percentage of the total deviation."""
        magnitude = abs(float(z))
        pct = round(magnitude / total_anom * 100.0, 1) if total_anom > 0 else 0.0
        return FieldContribution(field=_label(col, cats, numeric), contribution_pct=pct)


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
        augmented["top_contributing_fields"] = [
            c.as_dict() for c in result.top_contributing_fields
        ]
    augmented["model_version"] = model_version
    return augmented


def score_stream(
    lines: Iterable[str],
    artifact: Artifact,
    *,
    top_k: int = _DEFAULT_TOP_K,
    guard_schema: bool = True,
    batch_size: int = 2048,
) -> Iterator[str]:
    """Yield augmented NDJSON lines for an iterable of input NDJSON lines.

    Non-eligible records (pseudo processes, ``process_stop``) are dropped and
    not emitted. Eligible records are scored; with ``guard_schema`` set, a
    record whose ``schema_version`` differs from the model's raises
    :class:`~model.artifact.SchemaMismatchError`. Eligible records are scored in
    batches of ``batch_size`` so the IsolationForest/scaler per-call overhead is
    amortised across many records; output order matches input order.
    """
    scorer = Scorer(artifact, top_k=top_k)
    buffer: List[Dict[str, object]] = []

    def flush() -> Iterator[str]:
        results = scorer.score_batch(buffer)
        for record, result in zip(buffer, results):
            augmented = augment(record, result, artifact.model_version)
            yield json.dumps(augmented, separators=(",", ":"))
        buffer.clear()

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
        if not is_eligible(record):
            continue
        if guard_schema:
            artifact.guard_schema(record.get("schema_version"))  # type: ignore[arg-type]
        buffer.append(record)
        if len(buffer) >= batch_size:
            yield from flush()
    if buffer:
        yield from flush()
