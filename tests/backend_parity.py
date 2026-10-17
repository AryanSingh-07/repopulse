"""Emit real sklearn probabilities and shared feature results for JS parity tests."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/.python-deps"))
sys.path.insert(0, str(ROOT))

import joblib
from backend.core import load_model, predict_forest
from backend.ml.features import extract_features


def parity_data():
    artifact = joblib.load(ROOT / "model/repopulse.joblib")
    forest, _ = load_model()
    with (ROOT / "data/historical_windows.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    features = [{name: float(row[name]) for name in forest["feature_names"]} for row in rows[::7]]
    # Probe float32-sensitive tree boundaries as well as held historical rows.
    for tree in forest["trees"][:8]:
        for index, feature in enumerate(tree["feature"]):
            if feature >= 0:
                for epsilon in [-1e-8, 0, 1e-8]:
                    probe = dict(features[0])
                    probe[forest["feature_names"][feature]] = tree["threshold"][index] + epsilon
                    features.append(probe)
    vectors = [[row[name] for name in forest["feature_names"]] for row in features]
    expected = artifact["model"].predict_proba(vectors)[:, 1].tolist()
    python_predictions = [predict_forest(row, forest) for row in features]
    assert max(abs(a - b) for a, b in zip(expected, python_predictions)) < 1e-12
    now = "2026-10-17T12:00:00Z"
    repository = {"created_at": "2020-01-01T00:00:00Z"}
    commits = [
        {"author": {"login": "first"}, "commit": {"committer": {"date": "2026-10-17T12:00:00Z"}, "author": {"email": "Person@Example.com"}}},
        {"author": {"login": "second"}, "commit": {"committer": {"date": "2026-09-17T12:00:00Z"}, "author": {"email": " person@example.com "}}},
        {"author": {"login": "third"}, "commit": {"committer": {"date": "2026-07-19T12:00:00Z"}, "author": {"email": "other@example.com"}}},
        {"timestamp": 1792152000, "author": "normalized-history-author"},
    ]
    return {"features": features, "expected": expected, "feature_fixture": {"commits": commits, "repository": repository, "now": now}, "feature_expected": extract_features(commits, repository, now)}


if __name__ == "__main__":
    print(json.dumps(parity_data(), allow_nan=False))
