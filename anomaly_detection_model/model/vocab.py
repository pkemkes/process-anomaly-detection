"""Frequency tables and smoothed-surprise lookup for categorical fields.

A :class:`Vocabulary` records, per field, how often each categorical value was
seen during training. At score time it converts a (field, value) pair into a
**smoothed negative log-likelihood** ("surprise"): higher means rarer means more
anomalous. Laplace smoothing guarantees that values never seen in training (and
the explicit ``__missing__`` sentinel) receive a finite, maximal surprise floor
rather than an infinity.
"""

from __future__ import annotations

import math
from typing import Dict


class Vocabulary:
    """Per-field categorical frequency tables with smoothed-surprise lookup.

    The smoothed surprise for a value ``v`` of field ``f`` is::

        surprise(f, v) = -log( (count[f][v] + alpha)
                               / (total[f] + alpha * (cardinality[f] + 1)) )

    The ``+ 1`` in the denominator reserves probability mass for a single unseen
    value, so an out-of-vocabulary value at score time maps to the maximum
    surprise for that field without ever refitting.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self._counts: Dict[str, Dict[str, int]] = {}
        self._totals: Dict[str, int] = {}

    def observe(self, field: str, value: str) -> None:
        """Record one occurrence of ``value`` for ``field``."""
        field_counts = self._counts.get(field)
        if field_counts is None:
            field_counts = {}
            self._counts[field] = field_counts
            self._totals[field] = 0
        field_counts[value] = field_counts.get(value, 0) + 1
        self._totals[field] += 1

    def observe_row(self, values: Dict[str, str]) -> None:
        """Record one occurrence for every (field, value) pair in ``values``."""
        for field, value in values.items():
            self.observe(field, value)

    def cardinality(self, field: str) -> int:
        """Number of distinct values seen for ``field``."""
        return len(self._counts.get(field, {}))

    def total(self, field: str) -> int:
        """Total observations recorded for ``field``."""
        return self._totals.get(field, 0)

    def count(self, field: str, value: str) -> int:
        """Times ``value`` was observed for ``field`` (0 if never)."""
        return self._counts.get(field, {}).get(value, 0)

    def surprise(self, field: str, value: str) -> float:
        """Smoothed negative log-likelihood of ``value`` for ``field``.

        For a field never seen during training the result is ``0.0`` (no signal),
        keeping the score well-defined for fields added after training.
        """
        total = self._totals.get(field)
        if total is None:
            return 0.0
        card = len(self._counts[field])
        numerator = self.count(field, value) + self.alpha
        denominator = total + self.alpha * (card + 1)
        return -math.log(numerator / denominator)

    def max_surprise(self, field: str) -> float:
        """Surprise an unseen value would receive for ``field`` (the floor)."""
        total = self._totals.get(field)
        if total is None:
            return 0.0
        card = len(self._counts[field])
        denominator = total + self.alpha * (card + 1)
        return -math.log(self.alpha / denominator)

    # --- (de)serialization ------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        """Plain-dict representation suitable for joblib/JSON serialization."""
        return {
            "alpha": self.alpha,
            "counts": self._counts,
            "totals": self._totals,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Vocabulary":
        """Reconstruct a :class:`Vocabulary` from :meth:`to_dict` output."""
        vocab = cls(alpha=float(data["alpha"]))  # type: ignore[arg-type]
        vocab._counts = {str(k): dict(v) for k, v in data["counts"].items()}  # type: ignore[union-attr]
        vocab._totals = {str(k): int(v) for k, v in data["totals"].items()}  # type: ignore[union-attr]
        return vocab
