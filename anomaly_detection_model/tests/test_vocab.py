"""Unit tests for the saturating window-count vocabulary and surprise lookup."""

from __future__ import annotations

import math

from model.vocab import Vocabulary


def test_recurrence_across_k_windows_is_unsurprising():
    vocab = Vocabulary(alpha=1.0, saturation_k=5)
    # A value recurring across K distinct windows is "regular" -> ~0 surprise,
    # even though it was only seen a handful of times.
    for window in range(5):
        vocab.observe("image_name", "admin.exe", window)
    assert vocab.surprise("image_name", "admin.exe") == 0.0


def test_unseen_value_hits_the_floor():
    vocab = Vocabulary(alpha=1.0, saturation_k=5)
    for window in range(5):
        vocab.observe("signer", "Microsoft Windows", window)

    unseen = vocab.surprise("signer", "Totally New Signer")
    assert math.isfinite(unseen)
    assert unseen == vocab.max_surprise("signer")
    assert unseen == math.log(1 + 5 / 1.0)
    # The floor is the most surprising outcome for the field.
    assert unseen >= vocab.surprise("signer", "Microsoft Windows")


def test_surprise_is_monotonic_up_to_k():
    # Build values seen in increasing numbers of distinct windows.
    vocab = Vocabulary(alpha=1.0, saturation_k=5)
    for windows in range(1, 5):
        for w in range(windows):
            vocab.observe("f", f"v{windows}", w)
    surprises = [vocab.surprise("f", f"v{windows}") for windows in range(1, 5)]
    assert surprises == sorted(surprises, reverse=True)


def test_surprise_independent_of_dataset_size():
    # Same distinct-window count -> same surprise, regardless of how much *other*
    # traffic the field saw. This is the core "doesn't scale with baseline" goal.
    small = Vocabulary(alpha=1.0, saturation_k=5)
    for w in range(3):
        small.observe("image_name", "tool.exe", w)

    big = Vocabulary(alpha=1.0, saturation_k=5)
    for w in range(3):
        big.observe("image_name", "tool.exe", w)
    # Flood the field with unrelated values across many windows.
    for w in range(100):
        big.observe("image_name", f"other{w}.exe", w)

    assert big.surprise("image_name", "tool.exe") == small.surprise("image_name", "tool.exe")


def test_many_observations_in_one_window_count_once():
    vocab = Vocabulary(alpha=1.0, saturation_k=5)
    # A burst of 100 spawns inside a single window must not look "regular".
    for _ in range(100):
        vocab.observe("image_name", "installer.exe", window_id=42)
    assert vocab.window_count("image_name", "installer.exe") == 1
    # One window is far from the K-window saturation point -> still surprising.
    assert vocab.surprise("image_name", "installer.exe") > 0.5


def test_unknown_field_has_zero_surprise():
    vocab = Vocabulary(alpha=1.0)
    vocab.observe("image_name", "cmd.exe", 0)
    assert vocab.surprise("never_seen_field", "x") == 0.0


def test_conditional_surprise_normal_for_context_is_low():
    # An image whose elevation recurred across enough windows should not be
    # surprised when elevated, even though "elevated" is globally rare.
    vocab = Vocabulary(alpha=1.0, saturation_k=5)
    for w in range(5):
        vocab.observe("image_name", "common.exe", w)
        vocab.observe("is_elevated", "false", w)
        vocab.observe("image_name", "admin.exe", w)
        vocab.observe("is_elevated", "true", w)
        vocab.observe("pair_image_elevated", "admin.exe\x1ftrue", w)

    cond_admin = vocab.conditional_surprise(
        "pair_image_elevated", "admin.exe\x1ftrue",
        "image_name", "admin.exe", "is_elevated",
    )
    # Elevated never recurred for common.exe -> high conditional surprise.
    cond_common = vocab.conditional_surprise(
        "pair_image_elevated", "common.exe\x1ftrue",
        "image_name", "common.exe", "is_elevated",
    )
    assert cond_admin < cond_common
    assert cond_admin < 1.0


def test_conditional_surprise_unknown_context_is_zero():
    vocab = Vocabulary(alpha=1.0)
    vocab.observe("image_name", "cmd.exe", 0)
    assert vocab.conditional_surprise(
        "pair_x", "a\x1fb", "missing_field", "a", "target_field",
    ) == 0.0


def test_temporal_tight_profile_flags_off_hours():
    vocab = Vocabulary(alpha_t=1.0, temporal_min_samples=20, hour_buckets=8)
    for _ in range(30):
        vocab.observe_temporal_context("image_name", "backup.exe")
        vocab.observe_temporal_pair("pair_image_hour", "backup.exe\x1f0")

    usual = vocab.temporal_surprise(
        "image_name", "backup.exe", "pair_image_hour", "backup.exe\x1f0", 8
    )
    odd = vocab.temporal_surprise(
        "image_name", "backup.exe", "pair_image_hour", "backup.exe\x1f4", 8
    )
    assert odd > usual


def test_temporal_thin_entity_is_silent():
    vocab = Vocabulary(alpha_t=1.0, temporal_min_samples=20, hour_buckets=8)
    for _ in range(5):  # below the min-sample gate
        vocab.observe_temporal_context("image_name", "thin.exe")
        vocab.observe_temporal_pair("pair_image_hour", "thin.exe\x1f3")
    # Not enough evidence to call any hour "odd".
    assert vocab.temporal_surprise(
        "image_name", "thin.exe", "pair_image_hour", "thin.exe\x1f9", 8
    ) == 0.0


def test_serialization_round_trip():
    vocab = Vocabulary(alpha=0.5, saturation_k=5, hour_buckets=8, temporal_min_samples=20)
    for w in range(3):
        vocab.observe_row({"image_name": "cmd.exe", "signer": "MS"}, w)
    vocab.observe_row({"image_name": "cmd.exe", "signer": "Other"}, 4)
    for _ in range(25):
        vocab.observe_temporal_context("image_name", "cmd.exe")
        vocab.observe_temporal_pair("pair_image_hour", "cmd.exe\x1f3")

    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.alpha == vocab.alpha
    assert restored.saturation_k == vocab.saturation_k
    assert restored.hour_buckets == vocab.hour_buckets
    for field in ("image_name", "signer"):
        for value in ("cmd.exe", "MS", "Other", "unseen"):
            assert restored.surprise(field, value) == vocab.surprise(field, value)
    assert restored.temporal_surprise(
        "image_name", "cmd.exe", "pair_image_hour", "cmd.exe\x1f3", 8
    ) == vocab.temporal_surprise(
        "image_name", "cmd.exe", "pair_image_hour", "cmd.exe\x1f3", 8
    )

