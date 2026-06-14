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
    assert set(cats) == set(featurize.CATEGORICAL_FIELDS)


def test_boolean_flags_tri_state_encoding():
    flags = featurize.boolean_flags(make_record(is_signed=True, is_elevated=None))
    assert flags["is_signed"] == 1.0
    assert flags["is_elevated"] == 0.5
    assert set(flags) == set(featurize.BOOLEAN_FLAGS)


def test_ran_from_temp_flag():
    assert featurize.boolean_flags(make_record(path_bucket="Temp"))["ran_from_temp"] == 1.0
    assert featurize.boolean_flags(make_record(path_bucket="System32"))["ran_from_temp"] == 0.0


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
