"""Train and honestly evaluate RepoPulse's historical inactivity classifier.

Run after collect_history.py. The holdout contains entirely different repositories
AND later prediction dates. The exported forest reproduces sklearn probabilities
without a Python runtime, so the deployed Worker executes the very same model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL_DEPS = ROOT / "data" / ".python-deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "backend"))

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURE_NAMES, extract_features, feature_vector, inactivity_label, utc_datetime

MODEL_VERSION = "exploratory-1.0"
HORIZON_DAYS = 183
FEATURE_LABELS = {
    "commits_30d": "Commits in 30 days",
    "commits_90d": "Commits in 90 days",
    "commit_trend": "Commit momentum",
    "contributors_90d": "Contributors in 90 days",
    "contributor_trend": "Contributor momentum",
    "days_since_last_commit": "Days since last commit",
    "repository_age_days": "Repository age in days",
}
FEATURE_DEFINITIONS = {
    "commits_30d": "Number of default-branch commits with cutoff - 30 days < committer timestamp <= cutoff.",
    "commits_90d": "Number of default-branch commits with cutoff - 90 days < committer timestamp <= cutoff.",
    "commit_trend": "recent 30-day commits / (preceding 60-day commits / 2 + 1) - 1.",
    "contributors_90d": "Distinct commit author email identities in the 90-day window; fall back to author login/name when absent.",
    "contributor_trend": "Distinct recent 30-day authors / (distinct preceding 60-day authors + 1) - 1.",
    "days_since_last_commit": "Elapsed UTC days from most recent observed default-branch committer timestamp to cutoff.",
    "repository_age_days": "Elapsed UTC days from GitHub repository created_at metadata to cutoff.",
}


def holdout_repository(repository: str) -> bool:
    """A label-independent, reproducible repository-group assignment."""
    return hashlib.sha256(repository.lower().encode()).digest()[0] % 4 == 0


def snapshots():
    for year in range(2016, 2027):
        for month in (1, 4, 7, 10):
            yield datetime(year, month, 1, tzinfo=timezone.utc)


def build_dataset() -> tuple[list[dict], dict]:
    rows = []
    sources = []
    excluded = {"before_repository_history": 0, "incomplete_future_window": 0, "outside_partition_period": 0}
    for path in sorted((ROOT / "data" / "history").glob("*.json")):
        history = json.loads(path.read_text(encoding="utf-8"))
        if not history.get("history_complete") or not history.get("commits"):
            continue
        repo = history["repository"]
        created_at = utc_datetime(history["metadata"]["created_at"])
        commits = history["commits"]
        first_commit = min(utc_datetime(c["timestamp"]) for c in commits)
        collected_at = utc_datetime(history["collected_at"])
        split = "test" if holdout_repository(repo) else "train"
        sources.append({"repository": repo, "split": split, "commit_count": len(commits), "head_sha": history["head_sha"], "collected_at": history["collected_at"], "created_at": history["metadata"]["created_at"], "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for cutoff in snapshots():
            if cutoff < max(created_at, first_commit) + timedelta(days=90):
                excluded["before_repository_history"] += 1
                continue
            if cutoff + timedelta(days=HORIZON_DAYS) > collected_at:
                excluded["incomplete_future_window"] += 1
                continue
            if (split == "train" and cutoff > datetime(2022, 12, 31, tzinfo=timezone.utc)) or (split == "test" and cutoff < datetime(2023, 7, 1, tzinfo=timezone.utc)):
                excluded["outside_partition_period"] += 1
                continue
            values = extract_features(commits, history["metadata"], cutoff)
            rows.append({"repository": repo, "cutoff": cutoff.isoformat(), "outcome_end": (cutoff + timedelta(days=HORIZON_DAYS)).isoformat(), "collected_at": collected_at.isoformat(), "split": split, **values, "inactive_183d": inactivity_label(commits, cutoff, collected_at, HORIZON_DAYS)})
    if not rows:
        raise RuntimeError("No historical rows available; run scripts/collect_history.py first")
    return rows, {"sources": sources, "excluded_windows": excluded}


def metrics(y, probability) -> dict:
    predictions = (np.asarray(probability) >= 0.5).astype(int)
    return {
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 6),
        "accuracy": round(float(accuracy_score(y, predictions)), 6),
        "roc_auc": round(float(roc_auc_score(y, probability)), 6) if len(set(y)) > 1 else None,
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
    }


def portable_probability(forest: dict, vector: list[float]) -> float:
    estimates = []
    for tree in forest["trees"]:
        node = 0
        while tree["children_left"][node] != -1:
            node = tree["children_left"][node] if vector[tree["feature"][node]] <= tree["threshold"][node] else tree["children_right"][node]
        estimates.append(tree["values"][node][1])
    return sum(estimates) / len(estimates)


def export_forest(model) -> dict:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        counts = tree.value[:, 0, :]
        probabilities = counts / counts.sum(axis=1, keepdims=True)
        trees.append({"children_left": tree.children_left.tolist(), "children_right": tree.children_right.tolist(), "feature": tree.feature.tolist(), "threshold": tree.threshold.tolist(), "values": probabilities.tolist()})
    return {"model_version": MODEL_VERSION, "feature_names": FEATURE_NAMES, "positive_class": 1, "classes": [0, 1], "feature_importances": model.feature_importances_.tolist(), "trees": trees}


def save_eda(rows: list[dict], model_metrics: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#F5F4F6", "axes.labelcolor": "#F5F4F6", "xtick.color": "#A7ACBD", "ytick.color": "#A7ACBD"})
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#181A2F")
    for axis in axes:
        axis.set_facecolor("#242E49")
        for spine in axis.spines.values():
            spine.set_color("#37415C")
    for i, split in enumerate(("train", "test")):
        selected = [r for r in rows if r["split"] == split]
        axes[0].bar(i - 0.16, sum(r["inactive_183d"] == 0 for r in selected), width=0.32, color="#FDAA81", label="Active outcome" if i == 0 else None)
        axes[0].bar(i + 0.16, sum(r["inactive_183d"] == 1 for r in selected), width=0.32, color="#B4182D", label="Inactive outcome" if i == 0 else None)
    axes[0].set_xticks([0, 1], ["Train", "Later / unseen repositories"])
    axes[0].set_ylabel("Historical observation windows")
    axes[0].legend(facecolor="#242E49", labelcolor="#F5F4F6", frameon=False)
    matrix = np.array(model_metrics["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Oranges", alpha=0.9)
    for (row, col), value in np.ndenumerate(matrix):
        axes[1].text(col, row, str(value), ha="center", va="center", color="#181A2F", fontsize=18)
    axes[1].set_xticks([0, 1], ["Active", "Inactive"])
    axes[1].set_yticks([0, 1], ["Active", "Inactive"])
    axes[1].set_xlabel("Predicted outcome")
    axes[1].set_ylabel("Observed outcome")
    figure.suptitle("RepoPulse · actual held-out evaluation", color="#F5F4F6")
    figure.tight_layout()
    figure.savefig(ROOT / "model" / "evaluation.png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def save_reliability_diagram(y_test, prob_uncal, prob_cal, brier_before: float, brier_after: float) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#F5F4F6", "axes.labelcolor": "#F5F4F6", "xtick.color": "#A7ACBD", "ytick.color": "#A7ACBD"})
    figure, ax = plt.subplots(figsize=(6, 5), facecolor="#181A2F")
    ax.set_facecolor("#242E49")
    for spine in ax.spines.values():
        spine.set_color("#37415C")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#78819B", label="Perfect calibration")
    frac_pos_uncal, mean_pred_uncal = calibration_curve(y_test, prob_uncal, n_bins=5)
    ax.plot(mean_pred_uncal, frac_pos_uncal, marker="o", color="#B4182D", label=f"Uncalibrated (Brier: {brier_before:.4f})")
    frac_pos_cal, mean_pred_cal = calibration_curve(y_test, prob_cal, n_bins=5)
    ax.plot(mean_pred_cal, frac_pos_cal, marker="s", color="#FDAA81", label=f"Isotonic calibrated (Brier: {brier_after:.4f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("RepoPulse · Reliability Diagram", color="#F5F4F6")
    ax.legend(facecolor="#242E49", labelcolor="#F5F4F6", frameon=True, edgecolor="#37415C")
    ax.grid(True, color="#37415C", linestyle=":", alpha=0.6)
    figure.tight_layout()
    figure.savefig(ROOT / "model" / "reliability_diagram.png", dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def compute_cross_validation(rows: list[dict], n_splits: int = 5) -> dict:
    X = np.array([feature_vector(r) for r in rows], dtype=np.float32)
    y = np.array([r["inactive_183d"] for r in rows], dtype=int)
    groups = np.array([r["repository"] for r in rows])

    gkf = GroupKFold(n_splits=n_splits)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    raw_scores = {name: {"roc_auc": [], "brier_score": [], "f1": [], "precision": [], "recall": [], "accuracy": []} for name in models}

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        for name, m in models.items():
            m.fit(X_tr, y_tr)
            prob = m.predict_proba(X_te)[:, 1]
            pred = (prob >= 0.5).astype(int)
            raw_scores[name]["roc_auc"].append(float(roc_auc_score(y_te, prob)))
            raw_scores[name]["brier_score"].append(float(brier_score_loss(y_te, prob)))
            raw_scores[name]["f1"].append(float(f1_score(y_te, pred, zero_division=0)))
            raw_scores[name]["precision"].append(float(precision_score(y_te, pred, zero_division=0)))
            raw_scores[name]["recall"].append(float(recall_score(y_te, pred, zero_division=0)))
            raw_scores[name]["accuracy"].append(float(accuracy_score(y_te, pred)))

    summary = {}
    for name in models:
        summary[name] = {}
        for metric, vals in raw_scores[name].items():
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            ci95 = float(1.96 * std / np.sqrt(len(vals)))
            summary[name][metric] = {
                "mean": round(mean, 6),
                "std": round(std, 6),
                "ci95_half_width": round(ci95, 6),
                "ci95_low": round(mean - ci95, 6),
                "ci95_high": round(mean + ci95, 6),
            }

    return {
        "method": f"{n_splits}-fold GroupKFold by repository",
        "n_splits": n_splits,
        "n_samples": len(rows),
        "n_repositories": len(set(groups)),
        "comparison_table": summary,
    }




def main() -> None:
    rows, provenance = build_dataset()
    frame = pd.DataFrame(rows)
    if frame.duplicated(subset=["repository", "cutoff"]).any():
        raise RuntimeError("Duplicate repository observation windows detected")
    if frame[FEATURE_NAMES].isna().any().any() or not np.isfinite(frame[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise RuntimeError("Missing or non-finite historical model features detected")
    frame.groupby(["split", "inactive_183d"]).agg(windows=("cutoff", "count"), repositories=("repository", "nunique")).to_csv(ROOT / "data" / "dataset_summary.csv")
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    if len(train) < 100 or len(test) < 20:
        raise RuntimeError(f"Insufficient historical dataset: {len(train)} training and {len(test)} test rows")
    training_repositories = sorted({r["repository"] for r in train})
    testing_repositories = sorted({r["repository"] for r in test})
    assert not (set(training_repositories) & set(testing_repositories))
    assert max(r["outcome_end"] for r in train) < min(r["cutoff"] for r in test)
    X_train, y_train = np.array([feature_vector(r) for r in train], dtype=np.float32), np.array([r["inactive_183d"] for r in train])
    X_test, y_test = np.array([feature_vector(r) for r in test], dtype=np.float32), np.array([r["inactive_183d"] for r in test])
    if len(set(y_train)) != 2 or len(set(y_test)) != 2:
        raise RuntimeError("Both classes are required in training and test partitions")
    model = RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    forest = export_forest(model)
    probabilities = model.predict_proba(X_test)[:, 1]
    exported = [portable_probability(forest, vector) for vector in X_test.tolist()]
    parity_error = float(np.max(np.abs(probabilities - exported)))
    if parity_error > 1e-10:
        raise RuntimeError(f"Portable model export differs from sklearn: {parity_error}")
    model_metrics = metrics(y_test, probabilities)
    baseline_metrics = metrics(y_test, baseline.predict_proba(X_test)[:, 1])
    brier_before = float(brier_score_loss(y_test, probabilities))
    prob_train = model.predict_proba(X_train)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(prob_train, y_train)
    probabilities_calibrated = calibrator.predict(probabilities)
    brier_after = float(brier_score_loss(y_test, probabilities_calibrated))
    save_reliability_diagram(y_test, probabilities, probabilities_calibrated, brier_before, brier_after)
    cross_validation = compute_cross_validation(rows)
    trained_at = datetime.now(timezone.utc).isoformat()
    dataset_path = ROOT / "data" / "historical_windows.csv"
    with dataset_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for partition, examples in (("train", train), ("test", test)):
        with (ROOT / "data" / f"{partition}.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(examples)
    info = {
        "version": MODEL_VERSION,
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "cross_validation": cross_validation,
        "algorithm": "Random Forest Classifier",
        "status": "trained",
        "horizon_days": HORIZON_DAYS,
        "target_definition": "No commits reachable from the collected default-branch HEAD in the 183 days after the prediction cutoff. Merged pull requests count through their resulting default-branch commits. Merges solely to other branches are outside the target.",
        "probability_type": "Isotonic-calibrated probability (uncalibrated leaf average evaluated as baseline score).",
        "calibrated": True,
        "calibration": {
            "method": "isotonic",
            "brier_score_before": round(brier_before, 6),
            "brier_score_after": round(brier_after, 6),
            "diagram": "model/reliability_diagram.png",
        },
        "risk_thresholds": {"low_below": 0.30, "high_at_or_above": 0.65, "classification_threshold": 0.50, "note": "Low/medium/high display bands are product conventions, not empirically optimized decision thresholds."},
        "features": [{"name": name, "label": FEATURE_LABELS[name], "definition": FEATURE_DEFINITIONS[name], "importance": float(importance)} for name, importance in zip(FEATURE_NAMES, model.feature_importances_)],
        "feature_names": FEATURE_NAMES,
        "hyperparameters": model.get_params(),
        "metrics": {
            "model": model_metrics,
            "baseline": baseline_metrics,
            "brier_score_uncalibrated": round(brier_before, 6),
            "brier_score_calibrated": round(brier_after, 6),
            "calibration": {
                "method": "isotonic",
                "brier_score_before": round(brier_before, 6),
                "brier_score_after": round(brier_after, 6),
            },
            "baseline_method": "DummyClassifier(strategy='prior')",
            "positive_class": "inactive",
            "classification_threshold": 0.5,
            "confusion_matrix_order": "rows = observed [active, inactive]; columns = predicted [active, inactive]"
        },
        "evaluation": {
            "method": "Chronological holdout AND disjoint repository groups. SHA-256(repo.lower())[0] % 4 == 0 assigns test repositories before outcomes are computed.",
            "train_samples": len(train), "test_samples": len(test),
            "train_repositories": training_repositories, "test_repositories": testing_repositories,
            "train_repository_count": len(training_repositories), "test_repository_count": len(testing_repositories),
            "train_cutoff_earliest": min(r["cutoff"] for r in train), "train_cutoff_latest": max(r["cutoff"] for r in train),
            "train_outcome_latest": max(r["outcome_end"] for r in train),
            "test_cutoff_earliest": min(r["cutoff"] for r in test), "test_cutoff_latest": max(r["cutoff"] for r in test),
            "train_class_counts": {"active": int(sum(y_train == 0)), "inactive": int(sum(y_train == 1))},
            "test_class_counts": {"active": int(sum(y_test == 0)), "inactive": int(sum(y_test == 1))},
            "portable_export_max_absolute_error": parity_error,
        },
        "limitations": [
            "Exploratory mini-project model trained on a small convenience sample of public libraries; this sample does not represent all GitHub repositories.",
            "Historical windows from one repository are correlated and future windows overlap; window-level metrics are descriptive, not independent-trial confidence estimates.",
            "Repository groups are disjoint, but projects can share maintainers or ecosystems. No external, independently collected validation dataset is available.",
            "Probabilities are not calibrated; the percentage is a model score and must not be described as a validated confidence level.",
            "No default-branch commits for 183 days is an operational inactivity outcome, not proof that a project is permanently abandoned. Stable mature projects can need few changes.",
            "The current default-branch ancestry is reconstructed retrospectively. Deleted branches, rewritten commits and historical default-branch changes cannot be recovered; Git committer dates can be edited.",
            "Only commit-derived features and repository creation date enter the model. Issue, pull request and release analytics shown elsewhere are descriptive context, not model inputs.",
            "Merged pull requests count when their merge/squash commits reach the current default branch; merges confined to another branch do not enter the target.",
            "Live GitHub API truncation, missing authors, rate limits and newly created repositories may prevent a comparable prediction; the service must report insufficient data explicitly.",
        ],
        "provenance": {"repository_count": len(training_repositories) + len(testing_repositories), "collected_repository_count": len(provenance["sources"]), "commit_count": sum(s["commit_count"] for s in provenance["sources"]), "dataset": "data/historical_windows.csv", "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(), "collection_manifest": "data/provenance.json", "training_provenance": "data/training_provenance.json", "real_data": True, "synthetic_records": 0, "manually_assigned_outcomes": 0},
        "runtime": {"python": sys.version.split()[0], "scikit_learn": sklearn_version, "numpy": np.__version__, "pandas": pd.__version__},
    }
    (ROOT / "model").mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "feature_names": FEATURE_NAMES, "model_version": MODEL_VERSION, "metadata": info}, ROOT / "model" / "repopulse.joblib", compress=3)
    (ROOT / "model" / "forest.json").write_text(json.dumps(forest, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    (ROOT / "model" / "model-info.json").write_text(json.dumps(info, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT / "data" / "training_provenance.json").write_text(json.dumps({**provenance, "trained_at": trained_at, "dataset_sha256": info["provenance"]["dataset_sha256"], "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "feature_script_sha256": hashlib.sha256((ROOT / "backend" / "ml" / "features.py").read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    save_eda(rows, model_metrics)
    print(json.dumps({"version": MODEL_VERSION, "train_samples": len(train), "test_samples": len(test), "train_repositories": len(training_repositories), "test_repositories": len(testing_repositories), "model": model_metrics, "baseline": baseline_metrics, "brier_uncalibrated": brier_before, "brier_calibrated": brier_after, "export_parity_error": parity_error}, indent=2))



if __name__ == "__main__":
    main()
