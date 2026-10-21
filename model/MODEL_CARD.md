# RepoPulse exploratory-1.0

An actually trained Random Forest estimates an operational six-month inactivity
score from repository commit history. The percentage is **uncalibrated** and
must not be presented as a validated confidence level or proof of abandonment.

## Data and target

Collection on 17 October 2026 obtained 54,079 real commits from 48 small public
GitHub repositories across Python, JavaScript, C, shell, Rust and Ruby. Source
metadata, collected HEADs, hashed author identities and collection timestamps
are retained in `data/history/` and `data/provenance.json`.

The binary positive outcome is **zero commits reachable from the collected
default-branch HEAD during the 183 days after a historical prediction cutoff**.
Merged pull requests count through their resulting default-branch commits;
merges confined to other branches do not enter this target. No records or target
labels are fabricated or manually assigned. Incomplete future outcome windows
are excluded.

Quarterly observation windows produce 864 training rows from 35 repositories
and 144 test rows from 12 different repositories. The remaining collected
repository, minimistjs/minimist, has no eligible window in its assigned training
period because of its GitHub repository creation date. All 47 usable projects
belong to exactly one partition, assigned by a fixed hash of the repository
name before outcomes are examined.

Training prediction dates end in October 2022; their future outcome windows end
before the first test prediction date in July 2023. Test prediction dates extend
through April 2026. Training contains 670 active and 194 inactive outcomes;
test contains 98 active and 46 inactive outcomes.

## Inputs and model

The seven inputs, in exact inference order, are `commits_30d`, `commits_90d`,
`commit_trend`, `contributors_90d`, `contributor_trend`,
`days_since_last_commit`, and `repository_age_days`. Precise definitions and
global feature importances are saved in `model-info.json`. Commit windows use
committer timestamps and include only information available at each prediction
cutoff. Issue, pull request and release analytics are descriptive context and do
not enter the model.

The forest has 80 trees, maximum depth 6, minimum 2 samples per leaf,
`class_weight="balanced"`, and random seed 42. Hyperparameters and software
versions are retained with the model. No test-driven threshold or
hyperparameter optimization was performed. Scores are not calibrated.

## Actual held-out evaluation

At a classification threshold of 0.50:

| Metric | Random Forest | Dummy prior baseline |
| --- | ---: | ---: |
| Precision, inactive class | 0.478873 | 0.000000 |
| Recall, inactive class | 0.739130 | 0.000000 |
| F1, inactive class | 0.581197 | 0.000000 |
| ROC-AUC | 0.800688 | 0.500000 |
| Accuracy | 0.659722 | 0.680556 |
| Brier score (uncalibrated) | 0.227476 | 0.217361 |
| Brier score (calibrated, isotonic) | 0.213802 | — |

Probability calibration using isotonic regression reduces the Brier score loss from 0.227476 (uncalibrated) to 0.213802 (calibrated). The reliability diagram is saved to `model/reliability_diagram.png`.

The confusion matrix is `[[61, 37], [12, 34]]`: rows are observed active /
inactive; columns are predicted active / inactive. The model identifies more
inactive outcomes and improves ranking over the dummy baseline, but its
**accuracy is slightly lower than the majority-class baseline**. These results
do not justify a claim of production-grade predictive accuracy.

`evaluation.png` shows actual partition class counts and the test confusion
matrix. `reliability_diagram.png` shows the calibration curve before and after
isotonic calibration. `repopulse.joblib` persists the fitted sklearn model. `forest.json`
exports the same decision trees for the deployed inference runtime. Every test
vector was evaluated by both implementations; maximum absolute probability
error was 5.551115123125783e-16. Inputs are cast to float32 for sklearn-compatible
threshold comparisons.

### 5-fold repository-grouped cross-validation

To evaluate generalization across repositories without temporal or maintainer leakage, 5-fold `GroupKFold` cross-validation was run across all 47 repositories (1,008 historical observation windows). 95% confidence intervals are computed across folds:

| Model | ROC-AUC (95% CI) | Brier score (95% CI) | F1 score (95% CI) | Accuracy (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| **Random Forest** | 0.9033 ± 0.0457 | 0.1116 ± 0.0292 | 0.6724 ± 0.1014 | 0.8356 ± 0.0448 |
| **Logistic Regression** | 0.9097 ± 0.0488 | 0.1138 ± 0.0261 | 0.6716 ± 0.1179 | 0.8266 ± 0.0517 |
| **Gradient Boosting** | 0.8938 ± 0.0408 | 0.1036 ± 0.0231 | 0.6781 ± 0.0975 | 0.8705 ± 0.0278 |


## Limits and intended use

This is an educational, exploratory mini-project. The convenience sample is
small and not representative of all GitHub. Historical windows from one project
are correlated, and adjacent six-month outcomes overlap. Window-level metrics
are descriptive measurements, not independent-trial confidence bounds.
Disjoint repositories can still share maintainers or ecosystems. An independent
external validation dataset is not available.

Retrospective default-branch ancestry cannot reconstruct deleted commits,
history rewrites or historical default-branch changes. Commit timestamps are
user-editable. Stable mature projects may need few changes without being
abandoned. New repositories, missing author identities or incomplete live API
history can invalidate a comparable prediction; the service must report data
coverage and insufficient-data conditions.

Display bands are Low below 0.30, Medium from 0.30 to below 0.65, and High at or
above 0.65. These bands are product conventions, not empirically optimized
operating thresholds. Global feature importances describe the fitted model and
are not causal or individualized explanations.

## Reproduce

From the project directory, install `backend/ml/requirements.txt`, then run:

```powershell
python scripts/collect_history.py --workers 6
python scripts/train_model.py
python -m unittest backend.ml.test_features -v
```

Cached history files make re-training reproducible without GitHub requests.
Use `--refresh` only to intentionally collect a new source snapshot. Dataset,
source-history and training-script hashes are in `data/training_provenance.json`.
