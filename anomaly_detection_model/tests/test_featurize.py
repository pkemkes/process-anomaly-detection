"""Unit tests for the pure feature-extraction functions."""

from __future__ import annotations

import math

from model import featurize
from tests.factories import make_record, malicious_record


def test_is_eligible_filters_pseudo_and_stops():
    assert featurize.is_eligible(make_record())
    assert not featurize.is_eligible(make_record(is_pseudo=True))
    assert not featurize.is_eligible(make_record(event="process_stop"))


def test_missing_categorical_maps_to_sentinel():
    cats = featurize.categorical_values(make_record(signer=None, company_name=None))
    assert cats["signer"] == featurize.MISSING
    assert cats["company_name"] == featurize.MISSING


def test_categorical_values_are_normalized_and_paired():
    cats = featurize.categorical_values(make_record())
    assert cats["image_name"] == "cmd.exe"
    assert cats["parent_image_name"] == "explorer.exe"
    assert cats["pair_parent_image"] == "explorer.exe\x1fcmd.exe"
    # Feature fields and conditioning context fields are all present.
    assert set(featurize.CATEGORICAL_FIELDS) <= set(cats)
    assert set(featurize.CONTEXT_FIELDS) <= set(cats)


def test_boolean_flags_tri_state_encoding():
    flags = featurize.boolean_flags(make_record(is_signed=True, signer_is_microsoft=None))
    assert flags["is_signed"] == 1.0
    assert flags["signer_is_microsoft"] == 0.5
    assert set(flags) == set(featurize.BOOLEAN_FLAGS)


def test_elevation_is_conditioned_on_image():
    # Elevation is modelled as an image-conditioned pair, not a standalone flag.
    assert "is_elevated" not in featurize.BOOLEAN_FLAGS
    cats = featurize.categorical_values(make_record(image_name="python.exe", is_elevated=True))
    assert cats["pair_image_elevated"] == "python.exe\x1ftrue"
    assert cats["pair_image_integrity"] == "python.exe\x1fmedium"


def test_shannon_entropy_bounds():
    assert featurize.shannon_entropy("") == 0.0
    assert featurize.shannon_entropy("aaaa") == 0.0
    # Four distinct equiprobable symbols -> 2 bits.
    assert math.isclose(featurize.shannon_entropy("abcd"), 2.0)


def test_commandline_features_detect_suspicious_tokens():
    benign = featurize.commandline_features(make_record())
    suspicious = featurize.commandline_features(malicious_record())
    assert suspicious["cmd_suspicious_count"] > benign["cmd_suspicious_count"]
    assert suspicious["cmd_length"] > 0
    assert set(suspicious) == set(featurize.COMMANDLINE_FEATURES)


def test_empty_command_line_is_safe():
    feats = featurize.commandline_features(make_record(command_line_normalized=None))
    assert feats["cmd_length"] == 0.0
    assert feats["cmd_entropy"] == 0.0
    assert feats["cmd_non_alnum_ratio"] == 0.0


def test_feature_row_matches_column_set():
    from model.vocab import Vocabulary

    vocab = Vocabulary()
    vocab.observe_row(featurize.categorical_values(make_record()))
    row = featurize.feature_row(make_record(), vocab)
    assert set(row) == set(featurize.FEATURE_COLUMNS)


def test_window_id_prefers_create_time_and_falls_back_to_timestamp():
    rec = make_record(
        create_time="2026-06-14T10:00:00.000Z",
        timestamp="2026-06-14T09:00:00.000Z",
    )
    # create_time (10:00) drives the window, not timestamp (09:00).
    expected = featurize.window_id(make_record(create_time="2026-06-14T10:30:00.000Z"), 60)
    assert featurize.window_id(rec, 60) == expected

    # An epoch-zero create_time is rejected; timestamp is used instead.
    backfill = make_record(
        create_time="1970-01-01T00:00:00.000Z",
        timestamp="2026-06-14T09:00:00.000Z",
    )
    via_ts = featurize.window_id(make_record(create_time="2026-06-14T09:00:00.000Z"), 60)
    assert featurize.window_id(backfill, 60) == via_ts


def test_window_id_none_without_timestamp():
    rec = make_record(create_time=None, timestamp=None)
    assert featurize.window_id(rec, 60) is None


def test_hour_and_dow_bucketing():
    # 2026-06-01 is a Monday; 10:00 UTC -> hour bucket 10*8//24 = 3.
    rec = make_record(create_time="2026-06-01T10:00:00.000Z")
    assert featurize.hour_bucket(rec, 8) == "3"
    assert featurize.dow_bucket(rec, 2) == "0"  # Monday is a weekday
    # 2026-06-06 is a Saturday -> weekend bucket.
    weekend = make_record(create_time="2026-06-06T10:00:00.000Z")
    assert featurize.dow_bucket(weekend, 2) == "1"


def test_temporal_feature_columns_are_in_feature_columns():
    from model.vocab import Vocabulary

    vocab = Vocabulary()
    temporal = featurize.temporal_features(make_record(), vocab)
    assert set(temporal) == {
        featurize.FREQ_PREFIX + f for f in featurize.TEMPORAL_FIELDS
    }
    assert set(temporal) <= set(featurize.FEATURE_COLUMNS)
    # Thin/empty vocab -> temporal head is silent.
    assert featurize.head_c_nll(temporal) == 0.0
