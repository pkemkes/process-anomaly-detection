"""Integration tests: train -> score, synthetic injection, round-trip, schema guard."""

from __future__ import annotations

import json

import pytest

from model.artifact import Artifact, SchemaMismatchError
from model.score import Scorer, score_stream
from model.train import train
from tests.factories import (
    at_hour,
    baseline_records,
    fixed_hour_records,
    make_record,
    malicious_record,
)


def _train_baseline():
    return train(baseline_records(80), seed=0)


def test_synthetic_injection_scores_above_normal():
    artifact = _train_baseline()
    scorer = Scorer(artifact)

    normal = scorer.score_record(make_record())
    attack = scorer.score_record(malicious_record())

    assert attack.anomaly_score > normal.anomaly_score
    assert attack.rank_hint in ("medium", "high")


def test_anomaly_score_is_bounded_unit_interval():
    artifact = _train_baseline()
    scorer = Scorer(artifact)
    for record in (make_record(), malicious_record(), baseline_records(4)[0]):
        score = scorer.score_record(record).anomaly_score
        assert 0.0 <= score <= 1.0


def test_temporal_anomaly_raises_score():
    # backup.exe always runs at 02:00; the same process at 14:00 is "odd for it".
    records = baseline_records(40) + fixed_hour_records("backup.exe", 40, hour=2)
    artifact = train(records, seed=0)
    scorer = Scorer(artifact)

    usual = make_record(
        image="C:\\Windows\\System32\\backup.exe", image_name="backup.exe"
    )
    normal_hour = scorer.score_record(at_hour(usual, 2)).anomaly_score
    odd_hour = scorer.score_record(at_hour(usual, 14)).anomaly_score
    assert odd_hour > normal_hour


def test_recurrence_keeps_rare_but_regular_low():
    # An admin tool that recurs across many windows but is a tiny fraction of the
    # baseline must still score low (the core requirement of the redesign).
    records = baseline_records(80) + fixed_hour_records(
        "admin_tool.exe", 6, hour=3, image="C:\\Windows\\System32\\admin_tool.exe"
    )
    artifact = train(records, seed=0)
    scorer = Scorer(artifact)

    regular = scorer.score_record(
        make_record(
            image="C:\\Windows\\System32\\admin_tool.exe", image_name="admin_tool.exe"
        )
    )
    attack = scorer.score_record(malicious_record())
    assert regular.anomaly_score < attack.anomaly_score
    assert regular.rank_hint == "low"


def test_explanations_name_the_suspicious_features():
    artifact = _train_baseline()
    scorer = Scorer(artifact)
    result = scorer.score_record(malicious_record())

    joined = " ".join(c.field for c in result.top_contributing_fields).lower()
    assert "powershell.exe" in joined or "winword.exe" in joined
    assert result.top_contributing_fields  # non-empty
    # Each contributor reports its share (percent) of the anomalous deviation.
    assert all(0 <= c.contribution_pct <= 100 for c in result.top_contributing_fields)


def test_round_trip_preserves_scores(tmp_path):
    artifact = _train_baseline()
    path = tmp_path / "model.joblib"
    artifact.save(str(path))
    reloaded = Artifact.load(str(path))

    before = Scorer(artifact).score_record(malicious_record())
    after = Scorer(reloaded).score_record(malicious_record())
    assert before.anomaly_score == pytest.approx(after.anomaly_score)
    assert before.top_contributing_fields == after.top_contributing_fields


def test_determinism_same_seed_same_scores():
    a = train(baseline_records(80), seed=0)
    b = train(baseline_records(80), seed=0)
    rec = malicious_record()
    assert Scorer(a).score_record(rec).anomaly_score == pytest.approx(
        Scorer(b).score_record(rec).anomaly_score
    )


def test_score_stream_drops_pseudo_and_stops():
    artifact = _train_baseline()
    lines = [
        json.dumps(make_record()),
        json.dumps(make_record(is_pseudo=True)),
        json.dumps(make_record(event="process_stop")),
    ]
    out = [json.loads(line) for line in score_stream(lines, artifact)]

    assert len(out) == 1
    assert out[0]["anomaly_score"] is not None
    assert out[0]["model_version"] == artifact.model_version


def test_golden_ordering_is_stable():
    artifact = _train_baseline()
    scorer = Scorer(artifact)
    records = [
        make_record(),  # normal
        make_record(image="C:\\Users\\alice\\Downloads\\unknown.exe",
                    image_name="unknown.exe", path_bucket="Downloads",
                    is_signed=False, signature_status="unsigned", signer=None,
                    signer_is_microsoft=False),
        malicious_record(),  # worst
    ]
    scores = [scorer.score_record(r).anomaly_score for r in records]
    assert scores[0] < scores[1] < scores[2]


def test_schema_mismatch_is_refused():
    artifact = _train_baseline()
    bad = json.dumps(make_record(schema_version="9.9.9"))
    with pytest.raises(SchemaMismatchError):
        list(score_stream([bad], artifact, guard_schema=True))


def test_schema_guard_can_be_disabled():
    artifact = _train_baseline()
    bad = json.dumps(make_record(schema_version="9.9.9"))
    out = [json.loads(line) for line in score_stream([bad], artifact, guard_schema=False)]
    assert out[0]["anomaly_score"] is not None


def test_train_requires_records():
    with pytest.raises(ValueError):
        train([], seed=0)
