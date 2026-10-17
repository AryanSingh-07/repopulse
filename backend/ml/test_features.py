"""Tests covering leakage boundaries, censoring and deployment compatibility."""
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path
import sys
import unittest

from .features import FEATURE_NAMES, extract_features, feature_vector, inactivity_label


class HistoricalFeatureTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.repo = {"created_at": "2020-01-01T00:00:00Z"}
        self.history = [
            {"timestamp": self.cutoff - timedelta(days=90), "author": "outside"},
            {"timestamp": self.cutoff - timedelta(days=30), "author": "prior"},
            {"timestamp": self.cutoff - timedelta(days=1), "author": "recent"},
            {"timestamp": self.cutoff, "author": "recent"},
        ]

    def test_future_commits_do_not_change_features(self):
        original = extract_features(self.history, self.repo, self.cutoff)
        with_future = extract_features(self.history + [{"timestamp": self.cutoff + timedelta(seconds=1), "author": "future"}], self.repo, self.cutoff)
        self.assertEqual(original, with_future)

    def test_window_boundaries_and_shared_feature_order(self):
        result = extract_features(self.history, self.repo, self.cutoff)
        self.assertEqual(result["commits_30d"], 2)
        self.assertEqual(result["commits_90d"], 3)
        self.assertAlmostEqual(result["commit_trend"], 2 / 1.5 - 1)
        self.assertEqual(result["contributors_90d"], 2)
        self.assertEqual(result["contributor_trend"], -0.5)
        self.assertEqual(list(result), FEATURE_NAMES)
        self.assertEqual(feature_vector(result), list(result.values()))

    def test_censored_outcomes_rejected_and_horizon_boundary_counted(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            inactivity_label(self.history, self.cutoff, self.cutoff + timedelta(days=182))
        self.assertEqual(inactivity_label(self.history, self.cutoff, self.cutoff + timedelta(days=183)), 1)
        future = self.history + [{"timestamp": self.cutoff + timedelta(days=183), "author": "future"}]
        self.assertEqual(inactivity_label(future, self.cutoff, self.cutoff + timedelta(days=183)), 0)

    def test_github_and_historical_inputs_match(self):
        normalized = [{"timestamp": "2023-12-31T00:00:00Z", "author": "single"}]
        github = [{"commit": {"committer": {"date": "2023-12-31T00:00:00Z"}, "author": {"email": "single@example.com"}}}]
        self.assertEqual(extract_features(normalized, self.repo, self.cutoff), extract_features(github, self.repo, self.cutoff))

    def test_non_finite_input_rejected(self):
        with self.assertRaises(ValueError):
            feature_vector({name: float("nan") for name in FEATURE_NAMES})


class PersistedDatasetTests(unittest.TestCase):
    def test_persisted_partitions_and_portable_contract(self):
        root = Path(__file__).resolve().parents[2]
        info_file = root / "model" / "model-info.json"
        if not info_file.exists():
            self.skipTest("Run scripts/train_model.py first")
        info = json.loads(info_file.read_text(encoding="utf-8"))
        evaluation = info["evaluation"]
        self.assertFalse(set(evaluation["train_repositories"]) & set(evaluation["test_repositories"]))
        self.assertLess(evaluation["train_outcome_latest"], evaluation["test_cutoff_earliest"])
        forest = json.loads((root / "model" / "forest.json").read_text(encoding="utf-8"))
        self.assertEqual(forest["feature_names"], FEATURE_NAMES)
        self.assertEqual(len(forest["trees"]), 80)
        self.assertLessEqual(evaluation["portable_export_max_absolute_error"], 1e-10)

    def test_saved_rows_match_real_historical_sources(self):
        root = Path(__file__).resolve().parents[2]
        dataset = root / "data" / "historical_windows.csv"
        if not dataset.exists():
            self.skipTest("Run scripts/train_model.py first")
        with dataset.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        for row in [rows[0], rows[len(rows) // 2], rows[-1]]:
            history = json.loads((root / "data" / "history" / (row["repository"].replace("/", "__") + ".json")).read_text(encoding="utf-8"))
            actual = extract_features(history["commits"], history["metadata"], row["cutoff"])
            for feature in FEATURE_NAMES:
                self.assertAlmostEqual(actual[feature], float(row[feature]))
            self.assertEqual(inactivity_label(history["commits"], row["cutoff"], history["collected_at"]), int(row["inactive_183d"]))

    def test_saved_joblib_and_exported_forest_predict_identically(self):
        root = Path(__file__).resolve().parents[2]
        if not (root / "model" / "repopulse.joblib").exists():
            self.skipTest("Run scripts/train_model.py first")
        deps = root / "data" / ".python-deps"
        if deps.is_dir():
            sys.path.insert(0, str(deps))
        import joblib
        import numpy as np
        fitted = joblib.load(root / "model" / "repopulse.joblib")
        forest = json.loads((root / "model" / "forest.json").read_text(encoding="utf-8"))
        with (root / "data" / "test.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        vectors = np.asarray([feature_vector(row) for row in rows[::19]], dtype=np.float32)
        expected = fitted["model"].predict_proba(vectors)[:, 1]
        for vector, prediction in zip(vectors, expected):
            portable = []
            for tree in forest["trees"]:
                node = 0
                while tree["children_left"][node] >= 0:
                    index = tree["feature"][node]
                    node = tree["children_left"][node] if float(vector[index]) <= tree["threshold"][node] else tree["children_right"][node]
                portable.append(tree["values"][node][1])
            self.assertAlmostEqual(sum(portable) / len(portable), float(prediction), places=12)


if __name__ == "__main__":
    unittest.main()
