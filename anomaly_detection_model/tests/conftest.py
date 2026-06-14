"""Pytest configuration for the model test suite.

Test data factories live in :mod:`tests.factories`. This file only ensures the
project root is importable so ``model`` and ``tests`` resolve when
pytest is invoked from any working directory.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
