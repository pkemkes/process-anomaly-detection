"""Unsupervised process-anomaly detection over NDJSON process-start streams.

This package is self-contained: it consumes the NDJSON records produced by a
collector (such as the ``process_stream`` Windows agent) as plain dictionaries
and has **no import dependency** on the collector. It can therefore be trained
and run on a completely different system, needing only ``scikit-learn``,
``numpy`` and ``joblib``.

See :mod:`__main__` for the ``train`` / ``score`` command-line interface.
"""

from __future__ import annotations

MODEL_VERSION = "1.0.0"
