"""Integration tests: train -> score, synthetic injection, round-trip, schema guard."""

from __future__ import annotations

import json

import pytest

from model.artifact import Artifact, SchemaMismatchError
from model.score import Scorer, score_stream
from model.train import train
from tests.factories import baseline_records, make_record, malicious_record


def _train_baseline():
    return train(baseline_records(80), seed=0)


def test_synthetic_injection_scores_above_normal():
    artifact = _train_baseline()
    scorer = Scorer(artifact)

    normal = scorer.score_record(make_record())
    attack = scorer.score_record(malicious_record())

    assert attack.anomaly_score > normal.anomaly_score
    assert attack.rank_hint in ("medium", "high")


def test_explanations_name_the_suspicious_features():
    artifact = _train_baseline()
    scorer = Scorer(artifact)
    result = scorer.score_record(malicious_record())

    joined = " ".join(result.top_contributing_fields).lower()
    assert "powershell.exe" in joined or "winword.exe" in joined
    assert result.top_contributing_fields  # non-empty


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


def test_score_stream_passes_through_pseudo_and_stops():
    artifact = _train_baseline()
    lines = [
        json.dumps(make_record()),
        json.dumps(make_record(is_pseudo=True)),
        json.dumps(make_record(event="process_stop")),
    ]
    out = [json.loads(line) for line in score_stream(lines, artifact)]

    assert out[0]["anomaly_score"] is not None
    assert out[1]["anomaly_score"] is None
    assert out[2]["anomaly_score"] is None
    for record in out:
        assert record["model_version"] == artifact.model_version


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
