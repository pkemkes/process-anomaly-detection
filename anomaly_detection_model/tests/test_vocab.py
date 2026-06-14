"""Unit tests for the frequency-table vocabulary and smoothed surprise."""

from __future__ import annotations

import math

from model.vocab import Vocabulary


def test_more_common_value_is_less_surprising():
    vocab = Vocabulary(alpha=1.0)
    for _ in range(100):
        vocab.observe("image_name", "cmd.exe")
    for _ in range(2):
        vocab.observe("image_name", "rare.exe")

    common = vocab.surprise("image_name", "cmd.exe")
    rare = vocab.surprise("image_name", "rare.exe")
    assert rare > common
    assert common >= 0.0


def test_unseen_value_hits_the_floor():
    vocab = Vocabulary(alpha=1.0)
    for _ in range(50):
        vocab.observe("signer", "Microsoft Windows")

    unseen = vocab.surprise("signer", "Totally New Signer")
    assert math.isfinite(unseen)
    assert unseen == vocab.max_surprise("signer")
    # The floor is the most surprising outcome for the field.
    assert unseen >= vocab.surprise("signer", "Microsoft Windows")


def test_unknown_field_has_zero_surprise():
    vocab = Vocabulary(alpha=1.0)
    vocab.observe("image_name", "cmd.exe")
    assert vocab.surprise("never_seen_field", "x") == 0.0


def test_smoothing_formula_matches_definition():
    vocab = Vocabulary(alpha=1.0)
    vocab.observe("f", "a")
    vocab.observe("f", "a")
    vocab.observe("f", "b")
    # count[a]=2, total=3, cardinality=2
    expected = -math.log((2 + 1.0) / (3 + 1.0 * (2 + 1)))
    assert vocab.surprise("f", "a") == expected


def test_serialization_round_trip():
    vocab = Vocabulary(alpha=0.5)
    vocab.observe_row({"image_name": "cmd.exe", "signer": "MS"})
    vocab.observe_row({"image_name": "cmd.exe", "signer": "Other"})

    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.alpha == vocab.alpha
    for field in ("image_name", "signer"):
        for value in ("cmd.exe", "MS", "Other", "unseen"):
            assert restored.surprise(field, value) == vocab.surprise(field, value)
