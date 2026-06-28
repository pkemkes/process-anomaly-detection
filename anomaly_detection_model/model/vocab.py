"""Saturating window-count tables and surprise lookup for categorical fields.

A :class:`Vocabulary` records, per field, the number of **distinct time windows**
in which each categorical value was observed during training (capped at a small
saturation constant ``K``). At score time it converts a (field, value) pair into
a **saturating surprise**: a value that recurred across at least ``K`` windows is
"regular" and scores ~0, regardless of how large the baseline grew; a value that
has essentially never been seen receives a fixed finite floor.

This replaces the older frequency-based negative log-likelihood whose surprise
grew without bound for rare items and scaled with dataset size. See ``PLAN.md``
for the full reasoning.

Two count families are tracked:

* **Identity windows** (``observe`` / ``observe_row``) -- distinct-window counts,
  capped at ``K``, used by the saturating marginal and conditional surprises.
* **Temporal row counts** (``observe_temporal_*``) -- plain row frequencies used
  by the conditional temporal surprise ``-log P(bucket | image)``, which must
  *sharpen* with evidence and therefore deliberately does not saturate.
"""

from __future__ import annotations

import math
from typing import Dict, Set, Union

# Sentinel window id used when a record carries no usable timestamp at all. Such
# records collapse into a single window so they cannot inflate a recurrence count.
_NO_WINDOW = -1

# A capped window entry is a growing ``set`` while training, then an ``int`` once
# the vocabulary has been serialized and reloaded (we only persist the count).
_WindowEntry = Union[Set[int], int]


class Vocabulary:
    """Per-field saturating window-count tables with surprise lookup.

    The saturating marginal surprise for a value ``v`` of field ``f`` is::

        surprise(f, v) = -log( (min(c_win, K) + alpha) / (K + alpha) )

    where ``c_win`` is the number of distinct ``window_minutes``-sized windows in
    which ``v`` was seen. The denominator is the constant ``K + alpha`` (not the
    dataset total), so the surprise is **bounded** and **independent of baseline
    size**: ``c_win >= K`` -> ~0, ``c_win == 0`` -> the floor ``log(1 + K/alpha)``.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        window_minutes: int = 60,
        saturation_k: int = 5,
        alpha_t: float = 1.0,
        hour_buckets: int = 8,
        dow_buckets: int = 2,
        temporal_min_samples: int = 20,
    ) -> None:
        self.alpha = float(alpha)
        self.window_minutes = int(window_minutes)
        self.saturation_k = int(saturation_k)
        self.alpha_t = float(alpha_t)
        self.hour_buckets = int(hour_buckets)
        self.dow_buckets = int(dow_buckets)
        self.temporal_min_samples = int(temporal_min_samples)
        # field -> value -> distinct-window set (capped at K) or persisted count.
        self._windows: Dict[str, Dict[str, _WindowEntry]] = {}
        # Temporal row counts (frequency-based, never saturated).
        self._temporal_pair_counts: Dict[str, Dict[str, int]] = {}
        self._temporal_ctx_counts: Dict[str, Dict[str, int]] = {}

    # --- identity window observation --------------------------------------

    def observe(self, field: str, value: str, window_id: int = _NO_WINDOW) -> None:
        """Record that ``value`` of ``field`` appeared in window ``window_id``.

        The distinct-window set is capped at :attr:`saturation_k` entries: once a
        value has recurred in ``K`` windows it is "regular" and further windows
        add no information, which also bounds memory.
        """
        field_windows = self._windows.get(field)
        if field_windows is None:
            field_windows = {}
            self._windows[field] = field_windows
        entry = field_windows.get(value)
        if entry is None:
            field_windows[value] = {window_id}
        elif isinstance(entry, set):
            if len(entry) < self.saturation_k:
                entry.add(window_id)

    def observe_row(self, values: Dict[str, str], window_id: int = _NO_WINDOW) -> None:
        """Record one window observation for every (field, value) in ``values``."""
        for field, value in values.items():
            self.observe(field, value, window_id)

    # --- temporal row observation -----------------------------------------

    def observe_temporal_context(self, field: str, value: str) -> None:
        """Increment the row count of a temporal conditioning context (e.g. image)."""
        table = self._temporal_ctx_counts.setdefault(field, {})
        table[value] = table.get(value, 0) + 1

    def observe_temporal_pair(self, field: str, value: str) -> None:
        """Increment the row count of a temporal ``(context, bucket)`` pair."""
        table = self._temporal_pair_counts.setdefault(field, {})
        table[value] = table.get(value, 0) + 1

    # --- lookups ----------------------------------------------------------

    def window_count(self, field: str, value: str) -> int:
        """Distinct-window count (capped at ``K``) of ``value`` for ``field``."""
        field_windows = self._windows.get(field)
        if field_windows is None:
            return 0
        entry = field_windows.get(value)
        if entry is None:
            return 0
        return entry if isinstance(entry, int) else len(entry)

    def cardinality(self, field: str) -> int:
        """Number of distinct values seen for ``field``."""
        return len(self._windows.get(field, {}))

    def surprise(self, field: str, value: str) -> float:
        """Saturating marginal surprise of ``value`` for ``field``.

        Returns ``0.0`` for a field never seen during training (no signal). For a
        tracked field the result lies in ``[0, log(1 + K/alpha)]``: a value that
        recurred in at least ``K`` windows scores ~0; an unseen value hits the
        floor.
        """
        if field not in self._windows:
            return 0.0
        k = self.saturation_k
        c_win = min(self.window_count(field, value), k)
        numerator = c_win + self.alpha
        denominator = k + self.alpha
        return -math.log(numerator / denominator)

    def max_surprise(self, field: str) -> float:
        """Surprise an unseen value receives for ``field`` (the saturation floor)."""
        if field not in self._windows:
            return 0.0
        return -math.log(self.alpha / (self.saturation_k + self.alpha))

    def conditional_surprise(
        self,
        pair_field: str,
        pair_value: str,
        context_field: str,
        context_value: str,
        target_field: str,
    ) -> float:
        """Saturating conditional surprise ``-log P(target | context)``.

        Built from distinct-window counts so the "seen in >=K windows = regular"
        rule holds at the conditional level too::

            -log( (min(c_win[pair], K) + alpha)
                  / (min(c_win[context], K) + alpha * (M + 1)) )

        where ``M`` is the cardinality of ``target_field`` (reserving smoothing
        mass for unseen targets). A target value that recurred across enough
        windows for this context (e.g. ``High`` integrity for an admin image) is
        learned as normal. An unseen context falls back to a moderate floor since
        the context's own novelty is scored separately.
        """
        if context_field not in self._windows:
            return 0.0
        k = self.saturation_k
        joint = min(self.window_count(pair_field, pair_value), k)
        context = min(self.window_count(context_field, context_value), k)
        target_card = self.cardinality(target_field)
        numerator = joint + self.alpha
        denominator = context + self.alpha * (target_card + 1)
        return -math.log(numerator / denominator)

    def temporal_surprise(
        self,
        context_field: str,
        context_value: str,
        pair_field: str,
        pair_value: str,
        n_buckets: int,
    ) -> float:
        """Smoothed conditional temporal surprise ``-log P(bucket | context)``.

        Uses **row counts** (concentration is the signal: a tight per-image time
        profile should sharpen, not saturate)::

            -log( (count[context, bucket] + alpha_t)
                  / (count[context] + alpha_t * n_buckets) )

        A minimum-sample gate returns ``0.0`` when the context has fewer than
        :attr:`temporal_min_samples` observations, so thin entities never inject
        temporal noise.
        """
        ctx_count = self._temporal_ctx_counts.get(context_field, {}).get(context_value, 0)
        if ctx_count < self.temporal_min_samples:
            return 0.0
        joint = self._temporal_pair_counts.get(pair_field, {}).get(pair_value, 0)
        numerator = joint + self.alpha_t
        denominator = ctx_count + self.alpha_t * n_buckets
        return -math.log(numerator / denominator)

    # --- (de)serialization ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict representation suitable for joblib/JSON serialization.

        Identity windows are persisted as their capped integer counts (the raw
        window-id sets are not needed once frozen), keeping artifacts small.
        """
        window_counts = {
            field: {value: self.window_count(field, value) for value in values}
            for field, values in self._windows.items()
        }
        return {
            "alpha": self.alpha,
            "window_minutes": self.window_minutes,
            "saturation_k": self.saturation_k,
            "alpha_t": self.alpha_t,
            "hour_buckets": self.hour_buckets,
            "dow_buckets": self.dow_buckets,
            "temporal_min_samples": self.temporal_min_samples,
            "window_counts": window_counts,
            "temporal_pair_counts": self._temporal_pair_counts,
            "temporal_ctx_counts": self._temporal_ctx_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        """Reconstruct a :class:`Vocabulary` from :meth:`to_dict` output."""
        vocab = cls(
            alpha=float(data["alpha"]),  # type: ignore[arg-type]
            window_minutes=int(data.get("window_minutes", 60)),  # type: ignore[arg-type]
            saturation_k=int(data.get("saturation_k", 5)),  # type: ignore[arg-type]
            alpha_t=float(data.get("alpha_t", 1.0)),  # type: ignore[arg-type]
            hour_buckets=int(data.get("hour_buckets", 8)),  # type: ignore[arg-type]
            dow_buckets=int(data.get("dow_buckets", 2)),  # type: ignore[arg-type]
            temporal_min_samples=int(data.get("temporal_min_samples", 20)),  # type: ignore[arg-type]
        )
        vocab._windows = {
            str(field): {str(v): int(c) for v, c in values.items()}  # type: ignore[union-attr]
            for field, values in data["window_counts"].items()  # type: ignore[union-attr]
        }
        vocab._temporal_pair_counts = {
            str(field): {str(v): int(c) for v, c in values.items()}
            for field, values in data.get("temporal_pair_counts", {}).items()  # type: ignore[union-attr]
        }
        vocab._temporal_ctx_counts = {
            str(field): {str(v): int(c) for v, c in values.items()}
            for field, values in data.get("temporal_ctx_counts", {}).items()  # type: ignore[union-attr]
        }
        return vocab
